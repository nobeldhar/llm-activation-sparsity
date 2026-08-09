# Activation Sparsity Opportunities for Compressing General Large Language Models

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIPCCC59868.2024.10850382-blue)](https://doi.org/10.1109/IPCCC59868.2024.10850382)
[![Venue](https://img.shields.io/badge/IEEE%20IPCCC-2024-orange)](https://doi.org/10.1109/IPCCC59868.2024.10850382)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research code and calibrated artifacts for our **IEEE IPCCC 2024** paper: enforcing
**~50% activation sparsity in the FFN layers of pre-trained LLMs — without retraining
and with negligible perplexity loss** — as a foundation for predict-and-prefetch
inference on memory-constrained devices.

> **Nobel Dhar**, Bobin Deng, Md Romyull Islam, Kazi Fahim Ahmad Nasif, Liang Zhao,
> Kun Suo. *Activation Sparsity Opportunities for Compressing General Large Language
> Models.* IEEE International Performance, Computing, and Communications Conference
> (IPCCC 2024). DOI:
> [10.1109/IPCCC59868.2024.10850382](https://doi.org/10.1109/IPCCC59868.2024.10850382)

## TL;DR

Modern SwiGLU LLMs (Llama-3, Mistral, Phi-3) have **no natural activation
sparsity** (and GELU-based Phi-2 under 6%) — unlike ReLU models (OPT-6.7B is
naturally 92.8%+ sparse in every layer). We show that
sparsity can be **injected**: zero every FFN activation whose magnitude falls below a
per-layer threshold calibrated to a target sparsity level. Perplexity on WikiText-2
stays essentially flat up to **30% enforced sparsity** and remains acceptable at
**50%** across Llama-3-8B, Mistral-7B, and Phi-3:

![Sparsity vs perplexity](paper_figures/sparsity_vs_ppl_score.png)

Because inactive neurons' weights need never be fetched from storage, this enables
predict-and-prefetch inference: activation patterns are highly stable — **9 of 12
input samples retain a 100% layer-1 activation-pattern match even when 30% of the
input tokens are replaced** (Table II of the paper).

## The story: how we found the room for extra sparsity

The paper is a process of elimination — each experiment closes one door until
enforced activation sparsity is the only one left open. Every step below maps
onto a stage of this repository's pipeline.

**1. The FFN is the bottleneck.** In decoder-only transformers, the FFN/MLP
layers (Gate/Up/Down projections + activation function) hold about **2/3 of
all parameters** — they are the storage and compute bottleneck. Attention
layers are deliberately left untouched: modifying them is far more likely to
hurt accuracy. So compression must come from the FFN.

**2. Weight sparsity is a dead end.** Weight magnitudes cluster near zero,
but almost none are *exactly* zero — in all four models examined there is no
exploitable natural weight sparsity (paper Insight: *"we must explore another
sparsity type: activation sparsity"*):

![Weight magnitude distributions](paper_figures/weight_distribution_histogram.png)

**3. Natural activation sparsity is a ReLU-only privilege.** ReLU zeroes every
negative input, so OPT-6.7B gets ≥92.8% sparsity in every layer for free —
but modern LLMs abandoned ReLU. NewGELU-based Phi-2 shows under 6%, and the
SwiGLU models (Llama-3, Mistral, Phi-3) show essentially none:

![OPT natural activation sparsity](paper_figures/OPT_Activation_Sparsity.png)

**4. But the magnitudes reveal the room.** Collecting the FFN activations
(→ pipeline stage ①, `--collect-activations`) and plotting their CDFs shows
that although SwiGLU activations are almost never exactly zero, **their
magnitudes concentrate in tiny ranges** — e.g. ~80% of Mistral-7B's early-layer
gate activations fall below 0.1–0.25. Values that small contribute almost
nothing to the output. That concentration *is* the room: a small per-layer
magnitude threshold can zero huge fractions of neurons at minimal cost (paper
Insight: *"set small thresholds to omit fewer-contribution weights and easily
obtain high sparsity levels"*):

![Mistral-7B activation magnitude CDFs](paper_figures/Mistral-7B_cdf.png)

**5. Price the trade-off.** Calibrate percentile thresholds at each target
sparsity level (→ stage ②, `threshold_determination.py`), enforce them with
`x = where(|x| >= T, x, 0)` on the gate/up/down outputs — no retraining, no
weight changes — and measure WikiText-2 perplexity (→ stage ③,
`awq/entry_new.py`). Result: **30% sparsity is essentially free, and 50% keeps
PPL acceptable** (Llama-3 stays under 11; Mistral barely moves):

![Sparsity vs perplexity](paper_figures/sparsity_vs_ppl_score.png)

**6. And the patterns are predictable — sparsity becomes compression.** Which
neurons survive the threshold is stable across similar inputs: 9 of 12 test
samples keep a **100% layer-1 activation-pattern match even with 30% of input
tokens replaced** (Table II; inputs shipped in
[`perturbation_samples/`](perturbation_samples/)), and best-case samples match
100% at every probed depth:

![Activation pattern matching across layers](paper_figures/activation_matching_heatmap.png)

A cheap early-layer predictor can therefore prefetch only the ~50% of FFN
weights that will actually activate — the paper's closing insight: effective
prediction + prefetching *"can compress 50% of LLMs from the system resources'
perspective"*, the thread our follow-up work
([Euro-Par 2025](https://arxiv.org/abs/2507.14179)) picks up.

## What's in this repository

This is the official [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
codebase at commit `7901983` (our experiment base — its build/eval scaffolding is
reused; AWQ's original README is preserved as [AWQ_README.md](AWQ_README.md), and
the unrelated TinyChat demo is omitted) plus our research additions:

| Addition | What it is |
|---|---|
| [`awq/entry_new.py`](awq/entry_new.py) | **The paper's evaluation script** (June 2024): loads a model, wraps its MLPs with threshold classes (`ThresholdLlamaMLP` for SwiGLU models, `ThresholdPhi3MLP` for Phi-3's fused gate_up), collects activation samples to HDF5, and runs WikiText-2 perplexity / lm-eval tasks under enforced sparsity |
| [`threshold_determination.py`](threshold_determination.py) / [`_gpu`](threshold_determination_gpu.py) | Percentile threshold calibration from HDF5 activation samples (NumPy / CUDA `kthvalue` variants); [`new_threshold_determination.py`](new_threshold_determination.py) and [`phi-3_threshold_determination.py`](phi-3_threshold_determination.py) are per-model variants |
| `LLama-3-8B_activation_sample/`, `Mistral-7B-Activation_samples/`, `Phi-3_activation_samples/` | **Calibrated per-layer thresholds** for 30/35/40/45/50% sparsity — the exact JSONs behind the paper's Fig. 8 sweep (directory names preserved so `entry_new.py`'s relative paths work; the multi-hundred-GB raw HDF5 activation samples are not distributed — regenerate with `entry_new.py`) |
| [`perturbation_samples/`](perturbation_samples/) | The 12 input samples of the paper's Table II experiment, each with its 6 perturbed variants at 95–70% similarity |
| [`results/`](results/) | Weight-histogram counts behind Fig. 2 (`hist_counts_*.csv`), activation-pattern-match heatmaps (Figs. 9–12) |
| [`paper_figures/`](paper_figures/) | Published figures: FFN architecture, per-model activation CDFs (Figs. 5–7), natural-sparsity plots (Figs. 3–4), sparsity-vs-PPL sweep (Fig. 8) |
| [`paper/`](paper/) | The paper (PDF, © IEEE) |
| [`awq/quantize/pre_quant.py`](awq/quantize/pre_quant.py) | Patched `get_blocks()` adding Phi-2/Phi-3/Mistral/Qwen2/Gemma2 support |
| `drawing_cdf.py`, `figures_drawing.py`, `h5_file_analysis.py`, `json_inspection.py` | Figure-generation and HDF5/JSON inspection utilities |

## Quick start

```bash
# Environment (Python >= 3.10)
conda create -n sparsity python=3.10 -y && conda activate sparsity
pip install -e .            # installs the awq package + all evaluation deps

# Evaluate Llama-3-8B at 50% enforced FFN sparsity on WikiText-2
# (run from the repo root — the threshold JSON is loaded by relative path;
#  --model_path takes a local model directory or a Hugging Face Hub id)
python -m awq.entry_new --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext --output_path ./ppl_llama3_50.json
# entry_new.py loads ./LLama-3-8B_activation_sample/thresholds_50_percent_sparsity.json
# (edit the threshold-file path in main() to sweep other sparsity levels or
#  models — Mistral-7B / Phi-3 JSONs are included)
```

To calibrate thresholds for a **new** model or sparsity level (paper Sec. II-D
stages 1–2):

```bash
# Stage 1 - collect absolute FFN activations from the unmodified model (HDF5)
python -m awq.entry_new --model_path <model> \
    --collect-activations calib_samples --collect-windows 4

# Stage 2 - percentile thresholds at the target sparsity level
python threshold_determination.py --input-dir calib_samples --sparsity 0.50
```

Hardware note: the paper's experiments ran on a Lambda server with four NVIDIA
RTX 2080 Ti GPUs (the scripts shard models with `accelerate`); a single 24 GB
GPU suffices for the 7–8B models in fp16.

> Release note: the 2024 research scripts are shipped faithfully, with minimal
> fixes for public use (threshold-JSON key lookup, optional-import guards, a CLI
> for the calibration script, removal of experiment-scaffolding hooks, and a
> `--collect-activations` flag wiring up the previously script-internal
> collection stage). The measurement logic — activation capture, thresholding
> math, and perplexity loop — is unchanged.

## Reproduced results (August 2026)

The artifacts were re-verified end-to-end on an A100 server
(raw outputs in [`results/reproduction_2026/`](results/reproduction_2026/)).

**Evaluation with the shipped 2024 thresholds** (`awq/entry_new.py` exactly as
in the Quick start) reproduces the paper's Fig. 8 within reading precision:

| Model | Enforced sparsity | Reproduced WikiText-2 PPL | Paper (Fig. 8) |
|---|---|---|---|
| Mistral-7B | 30% | **5.418** | ≈ baseline (~5.3–5.5) |
| Mistral-7B | 50% | **6.448** | ~6.5 |
| Llama-3-8B | 50% | **10.675** | ~10.6 (stated "< 11") |

**Full-pipeline reproduction** — all three stages rerun from scratch for
Mistral-7B at 50%: activations freshly collected with
`--collect-activations` (4 WikiText-2 windows), thresholds recalibrated with
`threshold_determination.py`, then evaluated. The regenerated thresholds agree
with the shipped May-2024 calibration to **1.6% mean / 4.5% max relative
difference across all 96 layer-thresholds**
([`regenerated_thresholds_50.json`](results/reproduction_2026/regenerated_thresholds_50.json)),
and yield **PPL 6.439** vs 6.448 with the shipped thresholds — the calibrate →
threshold → evaluate story reproduces from this repository alone.

## Key results (from the paper)

| Model | Activation | Natural sparsity | Enforced sparsity @ ~flat PPL |
|---|---|---|---|
| OPT-6.7B | ReLU | ~93%+ (all layers) | — (already sparse) |
| Phi-2-2.7B | NewGELU | < 6% | — |
| Llama-3-8B | SwiGLU | ~0% | 30% (PPL < 11 even at 50%) |
| Mistral-7B | SwiGLU | ~0% | 30% (most stable through 50%) |
| Phi-3-3.8B | SwiGLU | ~0% | 30% (acceptable at 50%) |

Follow-up work building on this repo's findings:
**activation-pattern clustering for sparsity prediction** — Euro-Par 2025, Springer
LNCS 15900, [arXiv:2507.14179](https://arxiv.org/abs/2507.14179).

## Citation

```bibtex
@inproceedings{dhar2024activation,
  author    = {Dhar, Nobel and Deng, Bobin and Islam, Md Romyull and
               Nasif, Kazi Fahim Ahmad and Zhao, Liang and Suo, Kun},
  title     = {Activation Sparsity Opportunities for Compressing General Large
               Language Models},
  booktitle = {IEEE International Performance, Computing, and Communications
               Conference (IPCCC 2024)},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/IPCCC59868.2024.10850382}
}
```

## License & acknowledgments

MIT — this repository inherits the [MIT license](LICENSE) of
[llm-awq](https://github.com/mit-han-lab/llm-awq) (© 2023 MIT HAN Lab) for the base
code; our additions are © 2024–2026 the paper's authors, same license. The paper PDF
is © IEEE; the Version of Record is at the DOI above.

Research supported in part by U.S. National Science Foundation grants, conducted at
Kennesaw State University.

Contact: Nobel Dhar —
[Google Scholar](https://scholar.google.com/citations?user=OA35VhMAAAAJ) ·
[GitHub](https://github.com/nobeldhar) · nobeldhar807@gmail.com
