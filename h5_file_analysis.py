import h5py
import numpy as np

def verify_h5_file(file_path):
    """
    Verifies the contents of an HDF5 file, checking for:
    - Dataset existence
    - Data shape and dtype
    - Presence of NaN or Inf values
    - Basic statistics (min, max, mean, std)

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        None
    """
    print(f"Verifying file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            def recursive_inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\nDataset: {name}")
                    data = obj[:]

                    # Check shape and dtype
                    print(f"Shape: {data.shape}")
                    print(f"Dtype: {data.dtype}")

                    # Check for NaN or Inf values
                    has_nan = np.isnan(data).any()
                    has_inf = np.isinf(data).any()
                    print(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")

                    # Calculate basic statistics
                    if np.issubdtype(data.dtype, np.number):
                        min_val = np.min(data)
                        max_val = np.max(data)
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
                    else:
                        print("Non-numeric data; statistics not available.")

            # Recursively inspect all groups and datasets
            f.visititems(recursive_inspect)

    except Exception as e:
        print(f"Error while processing file {file_path}: {e}")

# Example usage
file_path = "Phi-3_activation_samples/activation_samples_activation_fn_0.h5"
verify_h5_file(file_path)
