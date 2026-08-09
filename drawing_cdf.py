import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Function to process each layer's data
def process_layer(layer_data, skip_neurons):
    flattened_data = layer_data.flatten()
    selected_neurons_indices = np.arange(0, len(flattened_data), skip_neurons)
    selected_neurons_outputs = flattened_data[selected_neurons_indices]
    sorted_data = np.sort(selected_neurons_outputs)
    cdf = np.arange(len(sorted_data)) / (len(sorted_data) - 1) * 100  # CDF as a percentage
    return sorted_data, cdf

# Load the HDF5 files and store the activation data in memory
file_paths = ["./LLama-3-8B_activation_sample/activation_samples_gate_proj_0.h5",
              "./LLama-3-8B_activation_sample/activation_samples_gate_proj_1.h5",
              "./LLama-3-8B_activation_sample/activation_samples_gate_proj_2.h5"]
layer_datasets = []

for layer_idx in range(32):
    combined_layer_data = []
    for path in file_paths:
        with h5py.File(path, "r") as f:
            combined_layer_data.append(f["gate_proj"][f"layer_{layer_idx}"][:])
    combined_layer_data = np.concatenate(combined_layer_data, axis=0)
    layer_datasets.append(combined_layer_data)

# Assume 32 layers distributed across the files
skip_neurons = 100
results = [process_layer(data, skip_neurons) for data in layer_datasets[:32]]

# Plot the CDF
fig, ax = plt.subplots(figsize=(18, 9))
cmap = plt.get_cmap('viridis')  # Colormap
norm = Normalize(vmin=0, vmax=31)  # Normalization for 32 layers

# Plot the CDF for each layer with colors based on the colormap
for i, (sorted_data, cdf) in enumerate(results):
    ax.plot(sorted_data, cdf, color=cmap(norm(i)), lw=1.5)

# Create colorbar as legend
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You have to set the array for the ScalarMappable
cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(0, 32, 4))
# Add the custom tick for layer 31
cbar.set_ticks(np.append(np.arange(0, 32, 4), 31))
# Set custom tick labels
cbar.set_ticklabels(np.append(np.arange(0, 32, 4), 31))

cbar.ax.tick_params(labelsize=20)  # Increase the font size for the layer numbers
cbar.set_label('Layer', fontsize=30)

# Customize the plot
ax.set_xlabel('Activation Magnitude', fontsize=30)
ax.set_ylabel('CDF (%)', fontsize=30)
ax.set_xlim(0, 1.5)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)
# Adjust layout to remove blank spaces
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
# Save the plot as a PDF
plt.savefig("activation_cdf_gate_proj_combined.png", format='png', dpi=300)

plt.show()
