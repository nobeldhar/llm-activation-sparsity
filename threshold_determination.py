"""Per-layer activation-threshold calibration (paper Section II-D, Eqs. 3-6).

Reads HDF5 activation samples collected by awq/entry_new.py, pools each
layer's absolute activations per projection category, and takes the
sparsity-level percentile as that layer's magnitude threshold.

Example (calibrate 50% sparsity thresholds for Mistral-7B):
  python threshold_determination.py --input-dir Mistral-7B-Activation_samples \
      --sparsity 0.50 --output my_thresholds_50.json

For SwiGLU models (Llama-3, Mistral) use the default categories
gate_proj/up_proj/down_proj; for Phi-2 pass --categories fc1 fc2; for Phi-3
pass --categories gate_up_proj.
"""

import argparse
import json
import os

import h5py
import numpy as np


def calculate_threshold_for_sparsity(data, sparsity_level):
    flattened_data = np.concatenate(data)
    # fp16 forward passes yield occasional inf/NaN outliers; exclude them from
    # the percentile (same treatment as the Phi-3 calibration script). Compute
    # in float32: np.percentile on large float16 arrays overflows internally.
    flattened_data = flattened_data[np.isfinite(flattened_data)].astype(np.float32)
    return float(np.percentile(flattened_data, sparsity_level * 100))


def calculate_thresholds(category, num_layers, input_paths, sparsity_level):
    thresholds = {}
    for layer_idx in range(num_layers):
        all_data = []
        for path in input_paths:
            with h5py.File(path, "r") as f:
                dataset_path = f"{category}/layer_{layer_idx}"
                if dataset_path in f:
                    all_data.append(f[dataset_path][:])
                    print(f"Loaded {category} layer {layer_idx} from {path}")
        if all_data:
            threshold = calculate_threshold_for_sparsity(all_data, sparsity_level)
            thresholds[f"layer_{layer_idx}"] = float(threshold)
            print(f"Threshold for {category} layer {layer_idx}: {threshold}")
        else:
            print(f"No data found for {category} layer {layer_idx}")
    return thresholds


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", required=True,
                    help="directory of .h5 activation-sample files")
    ap.add_argument("--sparsity", type=float, default=0.50,
                    help="target sparsity level, e.g. 0.30 ... 0.50 (default 0.50)")
    ap.add_argument("--categories", nargs="+",
                    default=["gate_proj", "up_proj", "down_proj"],
                    help="projection categories to calibrate")
    ap.add_argument("--num-layers", type=int, default=32)
    ap.add_argument("--output", default=None,
                    help="output JSON (default: thresholds_<pct>_percent_sparsity.json)")
    args = ap.parse_args()

    out_file = args.output or f"thresholds_{int(args.sparsity * 100)}_percent_sparsity.json"

    h5_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir) if f.endswith(".h5")
    )
    if not h5_files:
        raise SystemExit(f"No .h5 activation-sample files found in {args.input_dir} "
                         "- run the collection path of awq/entry_new.py first.")

    all_thresholds = {}
    for category in args.categories:
        paths = [p for p in h5_files if category in os.path.basename(p)] or h5_files
        all_thresholds[category] = calculate_thresholds(
            category, args.num_layers, paths, args.sparsity
        )

    with open(out_file, "w") as f:
        json.dump(all_thresholds, f, indent=2)
    print(f"Thresholds saved to {out_file}")


if __name__ == "__main__":
    main()
