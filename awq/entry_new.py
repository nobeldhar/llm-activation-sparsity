from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
try:  # only needed for the (unused) AWQ quantization paths; requires CUDA kernels
    from awq.quantize.quantizer import (
        pseudo_quantize_model_weight,
        real_quantize_model_weight,
    )
except ImportError:
    pseudo_quantize_model_weight = real_quantize_model_weight = None
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

from awq.quantize.pre_quant import get_blocks
import torch.nn.functional as F
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import sys
import warnings
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument(
    "--thresholds",
    default="./LLama-3-8B_activation_sample/thresholds_50_percent_sparsity.json",
    type=str,
    help="path to a calibrated per-layer thresholds JSON (see *_activation_sample*/ dirs)",
)
parser.add_argument(
    "--collect-activations",
    default=None,
    type=str,
    metavar="OUTDIR",
    help="calibration mode: instead of evaluating, run WikiText-2 windows through the "
         "UNMODIFIED model and dump absolute FFN activations to OUTDIR as HDF5 chunks "
         "(the input format of threshold_determination.py)",
)
parser.add_argument(
    "--collect-windows",
    default=4,
    type=int,
    help="number of 2048-token WikiText-2 windows to collect activations from",
)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
args = parser.parse_args()

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # not a local dir -> treat as a HF Hub id
        print(f"* {model_path} not found locally; treating it as a Hugging Face Hub id")
    print(f"* Building model {model_path}")

    # all hf model
    if "llava" in model_path.lower() or "vila" in model_path.lower():
        from llava_main.llava.model.builder import load_pretrained_model
        from llava_main.llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            try:
                enc = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False, trust_remote_code=True
                )
            except (ValueError, OSError):  # models with fast-only tokenizers (e.g. Llama-3)
                enc = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(
        #         config=config, torch_dtype=torch.float16, trust_remote_code=True
        #     )
        # real_quantize_model_weight(
        #     model, w_bit=args.w_bit, q_config=q_config, init_only=True
        # )

        # model.tie_weights()

        # # Infer device map
        # kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        # device_map = infer_auto_device_map(
        #     model,
        #     no_split_module_classes=[
        #         "OPTDecoderLayer",
        #         "LlamaDecoderLayer",
        #         "BloomBlock",
        #         "MPTBlock",
        #         "DecoderLayer",
        #     ],
        #     **kwargs,
        # )
        # # Load checkpoint in the model
        # load_checkpoint_in_model(
        #     model,
        #     checkpoint=args.load_quant,
        #     device_map=device_map,
        #     offload_state_dict=True,
        # )
        # # Dispatch model
        # model = simple_dispatch_model(model, device_map=device_map)

        # model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not ("llava" in model_path.lower() or "vila" in model_path.lower()):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        # if args.run_awq:
        #     assert args.dump_awq, "Please save the awq results with --dump_awq"

        #     awq_results = run_awq(
        #         model,
        #         enc,
        #         w_bit=args.w_bit,
        #         q_config=q_config,
        #         n_samples=128,
        #         seqlen=512,
        #     )
        #     if args.dump_awq:
        #         dirpath = os.path.dirname(args.dump_awq)
        #         os.makedirs(dirpath, exist_ok=True)

        #         torch.save(awq_results, args.dump_awq)
        #         print("AWQ results saved at", args.dump_awq)

        #     exit(0)

        # if args.load_awq:
        #     print("Loading pre-computed AWQ results from", args.load_awq)
        #     awq_results = torch.load(args.load_awq, map_location="cpu")
        #     apply_awq(model, awq_results)

        # # weight quantization
        # if args.w_bit is not None:
        #     if args.q_backend == "fake":
        #         assert (
        #             args.dump_quant is None
        #         ), "Need to use real quantization to dump quantized weights"
        #         pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        #     elif args.q_backend == "real":  # real quantization
        #         real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
        #         if args.dump_quant:
        #             if not args.dump_quant.endswith("v2.pt"):
        #                 print("[Info] Auto-change the dump_quant file name to *v2.pt")
        #                 args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
        #             dirpath = os.path.dirname(args.dump_quant)
        #             os.makedirs(dirpath, exist_ok=True)

        #             print(f"Saving the quantized model at {args.dump_quant}...")
        #             torch.save(model.cpu().state_dict(), args.dump_quant)
        #             exit(0)
        #     else:
        #         raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
                "PhiDecoderLayer",
                "MistralDecoderLayer",
                "Phi3Model"
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc

def count_zeros_input(tensors):
    total_zeros = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            total_zeros += (tensor == 0).sum().item()
    return total_zeros

def count_zeros_output(tensor):
    return (tensor == 0).sum().item()

def calculate_percentage(num_zeros, shape):
    total_values = shape[1] * shape[2]
    return (num_zeros / total_values) * 100

def calculate_hist(activations, bins):
    modulated_values = torch.abs(activations.cpu())
    all_activation_values_np = modulated_values.numpy()
    # Apply np.histogram to the numpy array of activation values
    # NumPy will internally treat the data as a flat array
    hist, bin_edges = np.histogram(all_activation_values_np, bins)
    return hist


count = 0
percentage_list = []
hist_list = []
pageCount = 0


# # Global lists to store selected neurons' outputs for each category
# gate_proj_outputs = []
# up_proj_outputs = []
# down_proj_outputs = []
# act_fn_outputs = [] 
# hist_list = {'gate_up_proj': [], 'down_proj': [], 'activation_fn': [], 'act_fn': []}

# # Hook function for all categories with Layer Name
# def hook_fn_with_layer_name(module, input, output, layer_name):
#     global gate_proj_outputs, up_proj_outputs, down_proj_outputs, act_fn_outputs
#     modulated_values = torch.abs(output[0].cpu())
#     hist_list[layer_name].append(modulated_values.numpy().flatten())

#     if len(hist_list[layer_name]) == 32:  # Check if we've collected data for all layers
#         skip_neurons = 100
#         selected_neurons_indices = torch.arange(0, output[0].numel(), skip_neurons)
#         selected_neurons_outputs = [layer[selected_neurons_indices] for layer in hist_list[layer_name]]

#         if layer_name == 'gate_up_proj':
#             gate_proj_outputs.append(selected_neurons_outputs)
#         elif layer_name == 'down_proj':
#             up_proj_outputs.append(selected_neurons_outputs)
#         elif layer_name == 'activation_fn':
#             down_proj_outputs.append(selected_neurons_outputs)
#         elif layer_name == 'act_fn':
#             act_fn_outputs.append(selected_neurons_outputs)

#         hist_list[layer_name] = []  # Reset list for the next batch

#         # Write the data to JSON
#         json_file_name = "selected_neurons_outputs.json"
#         data_to_write = {
#             'gate_proj': gate_proj_outputs,
#             'up_proj': up_proj_outputs,
#             'down_proj': down_proj_outputs,
#             'act_fn': act_fn_outputs
#         }
#         with open(json_file_name, 'w') as json_file:
#             json.dump(data_to_write, json_file, cls=NumpyEncoder)

#         print(f"Data has been written for {layer_name}")

#         if layer_name == 'down_proj':
#             sys.exit()  # Exit after writing down_proj data

# class NumpyEncoder(json.JSONEncoder):
#     """ Custom encoder for numpy data types """
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)












def hook_fn(module, input, output):
    global count
    global percentage_list
    global hist_list
    global pageCount
    count = count + 1
    # ######## for OPT #########
    
    # ## INPUT ###
    num_zeros = count_zeros_input(input)
    print(f"input of {module} Number of zeros: {num_zeros}")
    
    for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                print(f"input of {module}: {inp.shape}")

    # num_zeros = count_zeros_output(output)
    
    # if (module == "ReLU()"):
    #     relu_output = F.relu(output)
    #     num_zeros = count_zeros_output(relu_output)
    
    # percentage_list.append(calculate_percentage(num_zeros, output.shape))
    
    # if count == 32:
    #     csv_file_name = f"zero_ratio_percentages{pageCount}.csv"
    #     pageCount +=1

    #     # Write the zero ratio percentages to the CSV file
    #     with open(csv_file_name, mode='w', newline='') as file:
    #         writer = csv.writer(file)
            
    #         # Write each percentage on a new row
    #         for percentage in percentage_list:
    #             writer.writerow([percentage])
    #     count = 0
    #     percentage_list = []
    

    # print(f"Input of {module} Number of zeros: {num_zeros}")

    # for i, inp in enumerate(input):
    #         if isinstance(inp, torch.Tensor):
    #             print(f"Input of {module}: {inp.shape}")

    # percentage_list.append(calculate_percentage(num_zeros, input[0].shape))
    # if count == 32:
    #     csv_file_name = f"zero_ratio_percentages{pageCount}.csv"
    #     pageCount +=1

    #     # Write the zero ratio percentages to the CSV file
    #     with open(csv_file_name, mode='w', newline='') as file:
    #         writer = csv.writer(file)
            
    #         # Write each percentage on a new row
    #         for percentage in percentage_list:
    #             writer.writerow([percentage])
    #     count = 0
    #     percentage_list = []

    ## OUTPUT ###
    for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"output of {module}: {out.shape}")
    #relu_output = F.silu(output[0])
    num_zeros = count_zeros_output(output[0])
    print(f"output of {module} Number of zeros: {num_zeros}\n\n")

    ############################## GRAPH FOR ACTIVATION DESNSITY AND CDF ######################################

    # modulated_values = torch.abs(output[0].cpu())
    # hist_list.append(modulated_values.numpy().flatten())

    # if count == 32:
        

    #     # # Write the zero ratio percentages to the CSV file
    #     # with open(csv_file_name, mode='w', newline='') as file:
    #     #     writer = csv.writer(file)
            
    #     #     for i, tensor in enumerate(hist_list):
    #     #         # Flatten tensor and convert to list
    #     #         flattened_tensor = tensor.cpu().numpy().flatten().tolist()
                
    #     #         # Optionally prepend an identifier (e.g., tensor index)
    #     #         row = [i] + flattened_tensor
                
    #     #         writer.writerow(row)
    #     # Define a colormap and a normalization instance

    #     # and `hist_list` contains flattened outputs for each layer as PyTorch tensors


    #     skip_neurons = 100

    #     # Calculate indices assuming `output[0]` is representative of each layer's shape
    #     selected_neurons_indices = torch.arange(0, output[0].numel(), skip_neurons)

    #     # Extracting the selected neuron's outputs from each layer using PyTorch indexing
    #     selected_neurons_outputs = [layer[selected_neurons_indices] for layer in hist_list]

    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ############################### DENSITY ############################


        # cmap = plt.get_cmap('magma')
        # norm = Normalize(vmin=0, vmax=len(selected_neurons_outputs) - 1)

        # # Create a figure and a set of subplots
        # # fig, ax = plt.subplots(figsize=(12, 9))

        


        # # Plot the density for each layer
        # for i, data in enumerate(selected_neurons_outputs):
        #     sns.kdeplot(data, color=cmap(norm(i)), lw=0.5, ax=ax1, warn_singular=False)

        # # Add the grid
        # ax1.grid(True)

        # # Set the x-axis limits
        # ax1.set_xlim(0, 0.6)

        # # Create a ScalarMappable with the colormap and the normalization
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        # # Create an axes on the right side of the main axes for the colorbar
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        # # Add a colorbar to the plot using the explicitly created axes for the colorbar
        # cbar = plt.colorbar(sm, cax=cax, ticks=np.linspace(0, len(selected_neurons_outputs) - 1, len(selected_neurons_outputs)))
        # cbar.set_ticklabels(np.arange(1, len(selected_neurons_outputs) + 1))
        # cbar.set_label('Layer')

        # # Customize the plot
        # ax1.set_title('Activation Density Plot of Phi-3')
        # ax1.set_xlabel('Activation Values')
        # ax1.set_ylabel('Density')
        # # csv_file_name = f"activations_density{pageCount}.png"
        # # pageCount +=1
        # # plt.savefig(csv_file_name)

        ###################### CDF #############################



        # cmap = plt.get_cmap('viridis')  # Colormap
        # norm = Normalize(vmin=0, vmax=len(selected_neurons_outputs) - 1)  # Normalization from 0 to number of layers

        # fig, ax = plt.subplots(figsize=(12, 9))

        # ## Plot the CDF for each layer with colors based on the colormap
        # for i, data in enumerate(selected_neurons_outputs):
        #     sorted_data = np.sort(data)
        #     cdf = np.arange(len(sorted_data)) / (len(sorted_data) - 1) * 100  # CDF as a percentage
        #     ax2.plot(sorted_data, cdf, color=cmap(norm(i)), lw=0.5)

        # ## Create colorbar as legend
        # sm = ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # You have to set the array for the ScalarMappable
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(sm, cax=cax)

        # ## Set colorbar tick positions and labels
        # cbar.set_ticks(np.linspace(0, len(selected_neurons_outputs) - 1, num=len(selected_neurons_outputs)))
        # cbar.set_ticklabels(np.arange(1, len(selected_neurons_outputs) + 1))
        # cbar.set_label('Layer')

        # ## Add the grid
        # ax2.grid(True)

        # ## Set the x-axis limits
        # ax2.set_xlim(0,0.6)

        # ax2.set_xlabel('Activation Values')
        # ax2.set_ylabel('CDF (%)')
        # ax2.set_title('Activation Cumulative Distribution of Phi-3')
        # ax2.grid(True)

        # csv_file_name = f"activations_density_CDF{pageCount}.pdf"
        # pageCount +=1
        # plt.tight_layout(pad=3, w_pad=2)
        # plt.savefig(csv_file_name, format='pdf')

        # count = 0
        # hist_list = []

        ############################## CDF for gate_proj, up_proj, down_proj, and act_fn ######################################
    # modulated_values = torch.abs(output[0].cpu())
    # hist_list.append(modulated_values.numpy().flatten())

    # if count == 32:
    #     # Skip neurons
    #     skip_neurons = 100
    #     selected_neurons_indices = torch.arange(0, output[0].numel(), skip_neurons)
    #     selected_neurons_outputs = [layer[selected_neurons_indices] for layer in hist_list]

    #     fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    #     layers = ['gate_proj', 'up_proj', 'down_proj', 'act_fn']
        
    #     cmap = plt.get_cmap('viridis')
    #     norm = Normalize(vmin=0, vmax=len(selected_neurons_outputs) - 1)

    #     for idx, layer_name in enumerate(layers):
    #         ax = axs[idx]
    #         for i, data in enumerate(selected_neurons_outputs):
    #             sorted_data = np.sort(data)
    #             cdf = np.arange(len(sorted_data)) / (len(sorted_data) - 1) * 100
    #             ax.plot(sorted_data, cdf, color=cmap(norm(i)), lw=0.5)

    #         sm = ScalarMappable(cmap=cmap, norm=norm)
    #         sm.set_array([])
    #         divider = make_axes_locatable(ax)
    #         cax = divider.append_axes("right", size="5%", pad=0.05)
    #         cbar = plt.colorbar(sm, cax=cax)
    #         cbar.set_ticks(np.linspace(0, len(selected_neurons_outputs) - 1, num=len(selected_neurons_outputs)))
    #         cbar.set_ticklabels(np.arange(1, len(selected_neurons_outputs) + 1))
    #         cbar.set_label('Layer')
    #         ax.grid(True)
    #         ax.set_xlim(0, 0.6)
    #         ax.set_xlabel('Activation Values')
    #         ax.set_ylabel('CDF (%)')
    #         ax.set_title(f'Activation Cumulative Distribution of {layer_name}')
    #         ax.grid(True)

    #     csv_file_name = f"activations_density_CDF_{pageCount}.pdf"
    #     pageCount += 1
    #     plt.tight_layout(pad=3, w_pad=2)
    #     plt.savefig(csv_file_name, format='pdf')

    #     count = 0
    #     hist_list = []


# # Custom MLP layer with thresholding
# class ThresholdLlamaMLP(nn.Module):
#     def __init__(self, original_mlp, positive_threshold, negative_threshold):
#         super(ThresholdLlamaMLP, self).__init__()
#         self.gate_proj = original_mlp.gate_proj
#         self.up_proj = original_mlp.up_proj
#         self.down_proj = original_mlp.down_proj
#         self.act_fn = original_mlp.act_fn
#         self.positive_threshold = positive_threshold
#         self.negative_threshold = negative_threshold

#     def forward(self, x):
#         gate_x = self.gate_proj(x)
#         up_x = self.up_proj(x)
#         act_x = self.act_fn(gate_x) * up_x
#         condition = (act_x >= self.positive_threshold) | (act_x <= self.negative_threshold)
#         act_x = torch.where(condition, act_x, torch.zeros_like(act_x))
#         down_x = self.down_proj(act_x)
#         return down_x
    



# class ThresholdLlamaMLPAllOneThreshold(nn.Module):
#     def __init__(self, original_mlp, positive_threshold, negative_threshold):
#         super(ThresholdLlamaMLPAllOneThreshold, self).__init__()
#         self.gate_proj = original_mlp.gate_proj
#         self.up_proj = original_mlp.up_proj
#         self.down_proj = original_mlp.down_proj
#         self.act_fn = original_mlp.act_fn
#         self.positive_threshold = positive_threshold
#         self.negative_threshold = negative_threshold

#     def apply_threshold(self, x):
#         condition = (x >= self.positive_threshold) | (x <= self.negative_threshold)
#         return torch.where(condition, x, torch.zeros_like(x))

#     def forward(self, x):
#         # Apply threshold to gate_proj output
#         gate_x = self.gate_proj(x)
#         gate_x = self.apply_threshold(gate_x)
        
#         # Apply threshold to up_proj output
#         up_x = self.up_proj(x)
#         up_x = self.apply_threshold(up_x)
        
#         # Apply activation function
#         act_x = self.act_fn(gate_x) * up_x
        
#         # Apply threshold to activation output
#         act_x = self.apply_threshold(act_x)
        
#         # Apply down_proj and optionally apply threshold to its output
#         down_x = self.down_proj(act_x)
#         down_x = self.apply_threshold(down_x)
        
#         return down_x

# Threshold for each layer of each category


class ThresholdLlamaMLPDownOnlyExperiment(nn.Module):
    # Early experiment variant (down_proj-only thresholding); superseded by
    # ThresholdLlamaMLP below, which thresholds gate/up/down as in the paper.
    def __init__(self, original_mlp, thresholds, layer_index):
        super(ThresholdLlamaMLPDownOnlyExperiment, self).__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.thresholds = thresholds
        self.layer_index = layer_index

    def apply_threshold(self, x, category):
        threshold_value = self.thresholds[category][f'layer_{self.layer_index}']
        condition = (x >= threshold_value) | (x <= -threshold_value)
        print(f"Applying threshold {threshold_value} for {category}, layer {self.layer_index}")
        return torch.where(condition, x, torch.zeros_like(x))

    def forward(self, x):
        # Apply threshold to gate_proj output
        gate_x = self.gate_proj(x)
        print(f"gate_proj output before threshold: {torch.sum(gate_x == 0)} zeros")
        #gate_x = self.apply_threshold(gate_x, 'gate_proj')
        print(f"gate_proj output after threshold: {torch.sum(gate_x == 0)} zeros")
        
        # Apply threshold to up_proj output
        up_x = self.up_proj(x)
        print(f"up_proj output before threshold: {torch.sum(up_x == 0)} zeros")
        #up_x = self.apply_threshold(up_x, 'up_proj')
        print(f"up_proj output after threshold: {torch.sum(up_x == 0)} zeros")
        
        # Apply activation function
        act_x = self.act_fn(gate_x) * up_x
        print(f"act_fn output before threshold: {torch.sum(act_x == 0)} zeros")
        
        # Apply threshold to activation output
        #act_x = self.apply_threshold(act_x, 'act_fn')
        print(f"act_fn output after threshold: {torch.sum(act_x == 0)} zeros")
        
        # Apply down_proj and optionally apply threshold to its output
        down_x = self.down_proj(act_x)
        print(f"down_proj output before threshold: {torch.sum(down_x == 0)} zeros")
        down_x = self.apply_threshold(down_x, 'down_proj')
        print(f"down_proj output after threshold: {torch.sum(down_x == 0)} zeros")
        
        return down_x


class ThresholdPhi3MLP(nn.Module):
    def __init__(self, original_mlp, thresholds, layer_index):
        super(ThresholdPhi3MLP, self).__init__()
        self.gate_up_proj = original_mlp.gate_up_proj
        self.down_proj = original_mlp.down_proj
        self.activation_fn = original_mlp.activation_fn
        self.thresholds = thresholds
        self.layer_index = layer_index

    def apply_threshold(self, x, category):
        if category not in self.thresholds:  # e.g. 'up_states' absent from shipped JSONs
            return x
        threshold_value = self.thresholds[category][f'layer_{self.layer_index}']
        condition = (x >= threshold_value) | (x <= -threshold_value)
        print(f"Applying threshold {threshold_value} for {category}, layer {self.layer_index}")
        return torch.where(condition, x, torch.zeros_like(x))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # Apply threshold to gate_up_proj output
        up_states = self.gate_up_proj(hidden_states)
        print(f"gate_up_proj output before threshold: {torch.sum(up_states == 0)} zeros")

        # Split the output into gate and up states
        gate, up_states = up_states.chunk(2, dim=-1)

        # Apply threshold to gate
        print(f"gate before threshold: {torch.sum(gate == 0)} zeros")
        gate = self.apply_threshold(gate, 'gate_up_proj')
        print(f"gate after threshold: {torch.sum(gate == 0)} zeros")

        # Apply activation function
        up_states = up_states * self.activation_fn(gate)
        print(f"up_states after activation: {torch.sum(up_states == 0)} zeros")

        # Apply threshold to up_states
        up_states = self.apply_threshold(up_states, 'up_states')
        print(f"up_states after threshold: {torch.sum(up_states == 0)} zeros")

        # Apply down_proj and threshold to its output
        down_states = self.down_proj(up_states)
        print(f"down_proj output before threshold: {torch.sum(down_states == 0)} zeros")
        down_states = self.apply_threshold(down_states, 'down_proj')
        print(f"down_proj output after threshold: {torch.sum(down_states == 0)} zeros")

        return down_states


import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import json
import h5py

# Function to save activation samples to HDF5 files
def save_activation_samples_hdf5(activation_samples, output_path, chunk_idx):
    """Save activation samples to an HDF5 file."""
    print(f"Saving activation samples to {output_path}_{chunk_idx}.h5")
    with h5py.File(f"{output_path}_{chunk_idx}.h5", "w") as f:
        for category in activation_samples:
            grp = f.create_group(category)
            for i, layer_activations in enumerate(activation_samples[category]):
                if len(layer_activations) > 0:
                    dset = grp.create_dataset(f"layer_{i}", data=np.concatenate(layer_activations, axis=0))
    print(f"Activation samples saved to {output_path}_{chunk_idx}.h5")

# Function to evaluate model and capture activation samples
def evaluate_wikitext(model, enc, device, output_path=None, memory_limit_gb=32, save_every_n_samples=5):
    activation_samples = {
        'gate_up_proj': [[] for _ in range(32)],
        'activation_fn': [[] for _ in range(32)],
        'gate_proj': [[] for _ in range(32)],
        'up_proj': [[] for _ in range(32)],
        'down_proj': [[] for _ in range(32)],
        'act_fn': [[] for _ in range(32)]
    }
    activation_output_paths = {
        'gate_up_proj': 'activation_samples_gate_up_proj',
        'activation_fn': 'activation_samples_activation_fn',
        'gate_proj': 'activation_samples_gate_proj',
        'up_proj': 'activation_samples_up_proj',
        'down_proj': 'activation_samples_down_proj',
        'act_fn': 'activation_samples_act_fn'
    }

    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(device)
    nsamples = testenc.numel() // model.seqlen
    model.eval()
    nlls = []
    chunk_idx = 0

    def hook_fn(module, input, output, category, layer_idx):
        abs_output = torch.abs(output.detach().cpu()).numpy()
        activation_samples[category][layer_idx].append(abs_output)
        print(f"Captured activations for {category} layer {layer_idx}: {abs_output.shape}")

    # Register hooks
    handles = []
    for idx, layer in enumerate(model.model.layers):
        for name, submodule in layer.named_children():
            if name == 'mlp':
                for sub_name, sub_submodule in submodule.named_children():
                    if sub_name in ['gate_proj', 'up_proj', 'down_proj', 'act_fn', 'gate_up_proj', 'activation_fn']:
                        handles.append(sub_submodule.register_forward_hook(lambda m, i, o, category=sub_name, idx=idx: hook_fn(m, i, o, category, idx)))

    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

        # Periodically save activation samples and clear memory
        if (i + 1) % save_every_n_samples == 0:
            print(f"Reached {(i + 1) / nsamples * 100:.2f}% of the iterations. Writing activation samples")
            for category in activation_samples:
                save_activation_samples_hdf5(activation_samples, activation_output_paths[category], chunk_idx)
                activation_samples[category] = [[] for _ in range(32)]
            chunk_idx += 1

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    results = {"ppl": ppl.item()}
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Remove hooks
    for handle in handles:
        handle.remove()

    print(f"Final activation samples saved")


import torch
import torch.nn as nn

activation_samples = {
        #'gate_up_proj': [[] for _ in range(32)],
        #'activation_fn': [[] for _ in range(32)],
        'gate_proj': [[] for _ in range(32)],
        #'up_proj': [[] for _ in range(32)],
        #'down_proj': [[] for _ in range(32)],
        #'act_fn': [[] for _ in range(32)]
    }

class ThresholdLlamaMLP(nn.Module):
    def __init__(self, original_mlp, thresholds, layer_index):
        super(ThresholdLlamaMLP, self).__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.thresholds = thresholds
        self.layer_index = layer_index

    def apply_threshold(self, x, name):
        threshold_value = self.thresholds[name][f'layer_{self.layer_index}']
        condition = (x >= threshold_value) | (x <= -threshold_value)
        return torch.where(condition, x, torch.zeros_like(x))

    def forward(self, x):
        gate_x = self.gate_proj(x)
        gate_x = self.apply_threshold(gate_x, 'gate_proj')

        up_x = self.up_proj(x)
        up_x = self.apply_threshold(up_x, 'up_proj')

        act_x = self.act_fn(gate_x) * up_x

        down_x = self.down_proj(act_x)
        down_x = self.apply_threshold(down_x, 'down_proj')

        return down_x

# Example hook function
def hook_fn(module, input, output, category, idx):
    abs_output = torch.abs(output.detach().cpu()).numpy()
    print(f"Output after threshold: {np.sum(abs_output == 0)} zeros")
    activation_samples[category][idx].append(abs_output)
    with h5py.File(f"output_path.h5", "w") as f:
        for category in activation_samples:
            grp = f.create_group(category)
            for i, layer_activations in enumerate(activation_samples[category]):
                if len(layer_activations) > 0:
                    dset = grp.create_dataset(f"layer_{i}", data=np.concatenate(layer_activations, axis=0))


# Function to replace the MLP layers with thresholded wrappers.
# (The wrapper's forward() applies the thresholds itself; the activation-capture
# hooks used during calibration double-applied them and are not needed for
# evaluation, so no hooks are registered here.)
def replace_and_hook_mlp(model, thresholds):
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        threshold_mlp = ThresholdLlamaMLP(original_mlp, thresholds, layer_index)
        layer.mlp = threshold_mlp

    return handles

# Calibration stage 1 (paper Section II-D, Eqs. 1-2): run WikiText-2 windows
# through the unmodified model, capturing |activation| of each FFN projection
# per layer via forward hooks, and dump HDF5 chunks that
# threshold_determination.py consumes (one chunk file per window).
def collect_activations(model, enc, outdir, num_windows):
    os.makedirs(outdir, exist_ok=True)
    layers = model.model.layers
    num_layers = len(layers)
    categories = [c for c in ("gate_proj", "up_proj", "down_proj", "act_fn",
                              "gate_up_proj", "fc1", "fc2")
                  if hasattr(layers[0].mlp, c)]
    print(f"* Collecting activations for categories {categories} over {num_windows} windows")

    samples = {c: [[] for _ in range(num_layers)] for c in categories}
    hooks = []

    def make_hook(cat, idx):
        def hook(module, inputs, output):
            samples[cat][idx].append(
                torch.abs(output.detach()).to(torch.float16).cpu().numpy().reshape(-1)
            )
        return hook

    for idx, layer in enumerate(layers):
        for cat in categories:
            hooks.append(getattr(layer.mlp, cat).register_forward_hook(make_hook(cat, idx)))

    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    seqlen = 2048
    input_ids = testenc.input_ids.to(model.device)
    num_windows = min(num_windows, input_ids.numel() // seqlen)
    for i in tqdm.tqdm(range(num_windows), desc="collecting..."):
        batch = input_ids[:, (i * seqlen):((i + 1) * seqlen)]
        with torch.no_grad():
            model(batch)
        save_activation_samples_hdf5(samples, os.path.join(outdir, "activations"), i)
        for cat in categories:  # flush per window to bound memory
            for lst in samples[cat]:
                lst.clear()

    for h in hooks:
        h.remove()
    print(f"* Done. Calibrate with: python threshold_determination.py --input-dir {outdir} "
          f"--sparsity 0.50 --categories {' '.join(categories)}")



def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

    if args.collect_activations is not None:
        # calibration mode: capture natural activations, no thresholding applied
        collect_activations(model, enc, args.collect_activations, args.collect_windows)
        return

    ####################################### APPLYRING THRESHOLD #########################################
    # Load the thresholds from the JSON file

    print(f"* Loading sparsity thresholds from {args.thresholds}")
    with open(args.thresholds, 'r') as file:
        thresholds = json.load(file)
    handles = replace_and_hook_mlp(model, thresholds)



    print(model)

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

        if args.output_path is not None and "config" in results:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
    
    # Remove hooks after usage
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
