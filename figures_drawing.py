import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the activation data from the JSON file
json_file_name = "LLama-3-8B-selected_neurons_outputs.json"
with open(json_file_name, 'r') as json_file:
    category_data = json.load(json_file)

# Convert lists back to numpy arrays
for category in category_data:
    for i in range(len(category_data[category])):
        category_data[category][i] = [np.array(x) for x in category_data[category][i]]

# Debug: Print out the structure of the loaded data
for category in category_data:
    print(f"Category: {category}")
    for i, layer in enumerate(category_data[category]):
        print(f"  Layer {i}: {len(layer)} activations sets")

# Plot the CDFs
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=0, vmax=31)  # Normalization from 0 to number of layers - 1

categories = ['gate_proj', 'up_proj', 'down_proj']

for idx, category in enumerate(categories):
    ax = axs[idx]
    for i, neurons_list in enumerate(category_data[category]):
        for j, neurons in enumerate(neurons_list):
            sorted_data = np.sort(neurons)
            cdf = np.arange(len(sorted_data)) / (len(sorted_data) - 1) * 100
            ax.plot(sorted_data, cdf, color=cmap(norm(i)), lw=0.5, label=f'Layer {i+1}')  # Apply color based on layer index
        print(f"Plotted CDF for {category}, layer {i}, number of activations sets: {len(neurons_list)}")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks(np.linspace(0, 31, num=32))
    cbar.set_ticklabels(np.arange(1, 33))
    cbar.set_label('Layer')
    
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Activation Values')
    ax.set_ylabel('CDF (%)')
    ax.set_title(f'Activation Cumulative Distribution of {category}')
    ax.grid(True)

plt.tight_layout(pad=3, w_pad=2)
output_file_name = "activation_cdf_plots.png"
plt.savefig(output_file_name, format='png')
plt.show()

print(f"CDF plots have been saved to {output_file_name}")
