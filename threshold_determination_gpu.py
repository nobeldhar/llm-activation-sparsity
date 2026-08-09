import h5py
import torch
import json
import os

# Function to calculate the threshold for 30% sparsity using GPU
def calculate_threshold_for_sparsity(data, sparsity_level=0.675):
    """
    Calculates the threshold for the specified sparsity level using GPU.
    """
    # Move data to GPU and flatten
    flattened_data = torch.cat([torch.tensor(d, dtype=torch.float32, device="cuda:0").flatten() for d in data])
    
    # Calculate the percentile on GPU
    k = int(flattened_data.numel() * sparsity_level)  # Index for the desired percentile
    threshold = torch.kthvalue(flattened_data, k).values.item()  # Torch equivalent of percentile
    
    return threshold

# Function to load activation data from HDF5 files and calculate thresholds
def calculate_thresholds(category, num_layers, input_paths, output_json_file):
    thresholds = {category: {}}
    
    for layer_idx in range(num_layers):
        all_data = []
        
        for path in input_paths:
            with h5py.File(path, 'r') as f:
                dataset_path = f"{category}/layer_{layer_idx}"
                if dataset_path in f:
                    data = f[dataset_path][:]
                    all_data.append(data)
                    print(f"Loaded data for {category} layer {layer_idx} from {path}")

        if all_data:
            threshold = calculate_threshold_for_sparsity(all_data)
            thresholds[category][f"layer_{layer_idx}"] = threshold
            print(f"Threshold for {category} layer {layer_idx}: {threshold}")
        else:
            print(f"No data found for {category} layer {layer_idx}")

    # Save the thresholds to a JSON file
    with open(output_json_file, 'a') as out_file:
        json.dump(thresholds, out_file, indent=2)
        out_file.write('\n')

    print(f"Thresholds for {category} have been saved to {output_json_file}")

# Directory containing the HDF5 files
input_dir = "Phi-3_activation_samples"

# Paths to the HDF5 files containing activation samples for each category
input_paths_gate_up_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'gate_up_proj' in f]
# input_paths_gate_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'gate_proj' in f]
# input_paths_up_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'up_proj' in f]
input_paths_down_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'down_proj' in f]

# Number of layers in the model
num_layers = 32

# Output JSON file to store the thresholds
output_json_file = "./Phi-3_activation_samples/thresholds_675_percent_sparsity.json"

# Calculate and save thresholds for each category
categories = {
    'gate_up_proj': input_paths_gate_up_proj,
    # 'gate_proj': input_paths_gate_proj,
    # 'up_proj': input_paths_up_proj,
    'down_proj': input_paths_down_proj,
}

for category, paths in categories.items():
    calculate_thresholds(category, num_layers, paths, output_json_file)
