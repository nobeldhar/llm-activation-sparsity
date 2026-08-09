import h5py
import numpy as np
import json
import os

def safe_abs_percentile(data_list, sparsity_level=0.50, tag=""):
    """
    Computes percentile on |x| but ignores NaN/Inf.
    Returns float threshold.
    """
    flat = np.concatenate([np.abs(x).astype(np.float32, copy=False).reshape(-1) for x in data_list], axis=0)

    finite = np.isfinite(flat)
    total = flat.size
    good = int(finite.sum())
    bad = total - good

    if bad > 0:
        n_nan = int(np.isnan(flat).sum())
        n_inf = int(np.isinf(flat).sum())
        print(f"[warn] {tag}: total={total:,} finite={good:,} bad={bad:,} (nan={n_nan:,}, inf={n_inf:,})")

    flat = flat[finite]
    if flat.size == 0:
        raise RuntimeError(f"[error] {tag}: all values are non-finite, cannot compute threshold.")

    return float(np.percentile(flat, sparsity_level * 100))

def calculate_phi3_gate_up_thresholds_from_gate_up_proj(
    num_layers: int,
    input_paths: list,
    output_json_file: str,
    sparsity_level: float = 0.50,
):
    """
    Uses EXACT hierarchy you showed:
      gate_up_proj/layer_{i} : (5, 2048, 16384) float16
    Splits into:
      gate = first 8192
      up   = last  8192
    Computes thresholds on |values| (matches your runtime mask |x| >= thr).
    """
    thresholds = {"gate_proj": {}, "up_proj": {}}

    for layer_idx in range(num_layers):
        all_gate = []
        all_up = []

        dpath = f"gate_up_proj/layer_{layer_idx}"

        for path in input_paths:
            with h5py.File(path, "r") as f:
                if dpath not in f:
                    continue

                data = f[dpath][:]  # (5, 2048, 16384) float16
                if data.shape[-1] != 16384:
                    print(f"[warn] {os.path.basename(path)} {dpath} lastdim={data.shape[-1]} (expected 16384)")
                    continue

                gate = data[..., :8192]
                up   = data[..., 8192:]

                all_gate.append(gate)
                all_up.append(up)

                print(f"Loaded layer {layer_idx} from {os.path.basename(path)}  gate={gate.shape} up={up.shape}")

        if not all_gate:
            print(f"No data found for {dpath} in any file.")
            continue

        thr_g = safe_abs_percentile(all_gate, sparsity_level, tag=f"layer{layer_idx:02d}/gate")
        thr_u = safe_abs_percentile(all_up,   sparsity_level, tag=f"layer{layer_idx:02d}/up")

        thresholds["gate_proj"][f"layer_{layer_idx}"] = thr_g
        thresholds["up_proj"][f"layer_{layer_idx}"] = thr_u

        print(f"[layer {layer_idx:02d}] gate_thr={thr_g:.6g}  up_thr={thr_u:.6g}")

    with open(output_json_file, "w") as out:
        json.dump(thresholds, out, indent=2)
        out.write("\n")

    print(f"Saved thresholds to: {output_json_file}")


# ---------------------- example usage -------------------------------------

input_dir = "Phi-3_activation_samples"

input_paths_gate_up = [
    os.path.join(input_dir, fn)
    for fn in os.listdir(input_dir)
    if ("gate_up_proj" in fn) and fn.endswith((".h5", ".hdf5"))
]

num_layers = 32
output_json_file = os.path.join(input_dir, "new_phi3_thresholds_50_percent_sparsity.json")

calculate_phi3_gate_up_thresholds_from_gate_up_proj(
    num_layers=num_layers,
    input_paths=input_paths_gate_up,
    output_json_file=output_json_file,
    sparsity_level=0.50,
)