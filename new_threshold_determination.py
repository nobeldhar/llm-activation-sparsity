import h5py
import numpy as np
import json
import os

# Function to calculate the threshold for a specific sparsity level
def calculate_threshold_for_sparsity(data, sparsity_level=0.80):
    flattened_data = np.concatenate(data)
    threshold = np.percentile(flattened_data, sparsity_level * 100)
    return threshold

# Function to load activation data and calculate thresholds for each layer, accounting for neuron count differences
def calculate_thresholds(category, num_layers, input_paths, output_json_file, neuron_count):
    thresholds = {category: {}}
    
    for layer_idx in range(num_layers):
        all_data = []
        
        for path in input_paths:
            with h5py.File(path, 'r') as f:
                dataset_path = f"{category}/layer_{layer_idx}"
                if dataset_path in f:
                    data = f[dataset_path][:]
                    
                    # Check if data matches expected shape for neuron count
                    if data.shape[-1] == neuron_count:
                        all_data.append(data)
                        print(f"Loaded data for {category} layer {layer_idx} from {path}")
                    else:
                        print(f"Warning: Mismatch in neuron count for {dataset_path} in {path}")
        
        # Calculate threshold only if data is available for the layer
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
input_dir = "Mistral-7B-Activation_samples"

# Paths to the HDF5 files for each projection type
input_paths_gate_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'gate_proj' in f]
input_paths_up_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'up_proj' in f]
input_paths_down_proj = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if 'down_proj' in f]

# Number of layers in the model
num_layers = 32

# Output JSON file to store the thresholds
output_json_file = "./Mistral-7B-Activation_samples/thresholds_80_percent_sparsity.json"

# Neuron counts per projection type
neuron_counts = {
    'gate_proj': 14336,
    'up_proj': 14336,
    'down_proj': 4096
}

# Categories and their paths
categories = {
    'gate_proj': input_paths_gate_proj,
    'up_proj': input_paths_up_proj,
    'down_proj': input_paths_down_proj
}

# Calculate and save thresholds for each category, accounting for neuron counts
for category, paths in categories.items():
    neuron_count = neuron_counts[category]
    calculate_thresholds(category, num_layers, paths, output_json_file, neuron_count)