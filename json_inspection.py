import json
import pandas as pd

# Load the JSON file
json_file_name = "thresholds_gate_proj_30_percent_sparsity.json"
with open(json_file_name, 'r') as json_file:
    category_data = json.load(json_file)

# Flatten the JSON data and load into a DataFrame
flattened_data = []

for category, layers in category_data.items():
    for layer_index, activation_sets in enumerate(layers):
        for activation_set in activation_sets:
            flattened_data.append({
                'category': category,
                'layer': layer_index,
                'activation_set': activation_set
            })

df = pd.DataFrame(flattened_data)
print(df.head())
