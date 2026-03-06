# BiRefNet Baseline Experiments

This folder adds a reproducible experimentation layer around the upstream BiRefNet codebase.

## 1) Repository understanding summary

### Architecture components

BiRefNet uses:

- **Backbone encoder** (`models/backbones/*`) to extract multi-scale features.
- **Context squeeze module** (optional, configurable) on deep features.
- **Lateral + decoder path** (`models/modules/lateral_blocks.py`, `models/modules/decoder_blocks.py`) with multi-stage upsampling.
- **ASPP / ASPPDeformable attention module** in decoding (`models/modules/aspp.py`).
- **Multi-scale supervision + gradient reference branch** for boundary-aware training (`models/birefnet.py`).

### Training pipeline

- Entry point: `train.py`.
- Reads hyperparameters from `config.py`.
- Builds training dataset with `dataset.MyData`.
- Uses `PixLoss` + optional classification loss + gradient supervision loss (`loss.py`, `train.py`).
- Supports single GPU, DDP, or HuggingFace Accelerate.

### Inference pipeline

- Entry point: `inference.py`.
- Loads checkpoint(s), runs model in eval mode.
- Resizes prediction back to GT/image size and writes predicted masks.

### Supported backbones

Configured in `config.py` and built in `models/backbones/build_backbone.py`:

- CNN: `vgg16`, `vgg16bn`, `resnet50`
- Swin V1: `swin_v1_t/s/b/l`
- PVT V2: `pvt_v2_b0/b1/b2/b5`
- DINOv3: `dino_v3_s/s_plus/b/l/h_plus/7b`

### Configuration system

- Centralized through the `Config` class (`config.py`).
- Includes task presets, dataset paths, training settings, losses, and backbone choice.
- Shared globally across train/inference/model construction.

### Pretrained model options

- Official checkpoints are linked in root `README.md` (Google Drive + HuggingFace).
- Backbone weight file mappings are configured in `config.py` (`weights_root_dir`, `weights` dictionary).

### Short architecture diagram

```text
Input image
   │
   ▼
Backbone encoder (x1, x2, x3, x4)
   │
   ├── optional context aggregation + squeeze
   ▼
Lateral fusion + decoder (coarse-to-fine)
   │
   ├── multi-scale heads (training supervision)
   ├── gradient reference branch (training only)
   ▼
Final 1-channel probability map (sigmoid)
```

## 2) Reproducible environment

Files:

- `env/environment.yml`
- `env/requirements.txt`

Recommended baseline stack:

- Python 3.10
- PyTorch 2.5.1 + TorchVision 0.20.1
- CUDA 12.1 compatible wheels

Install:

```bash
cd birefnet_experiments/env
conda env create -f environment.yml
conda activate birefnet-exp
pip install -r requirements.txt
```

## 3) Clean inference wrapper

`inference/birefnet_inference.py` provides:

- Checkpoint loading
- Runtime backbone selection
- Single image or directory inference
- Outputs per sample:
  - `mask.png`
  - `masked_image.png`
  - `raw_probability_map.npy`
- Configurable:
  - `--resolution`
  - `--device` (`cpu`, `cuda`)
  - `--half` (fp16 autocast)

## 4) Baseline evaluation pipeline

`benchmarks/benchmark_birefnet.py` evaluates datasets with structure:

```text
dataset/
  images/
  masks/
```

Metrics:

- IoU
- Dice
- MAE
- Precision
- Recall
- F-measure

Outputs:

- `results.csv`
- `results_summary.json`

## 5) Multi-dataset experiment support

`benchmarks/dataset_profiles.yaml` and `benchmarks/run_benchmark_suite.py` support:

- DIS5K
- DUTS
- product background removal datasets
- automotive segmentation datasets
- generic foreground segmentation
- custom datasets by adding entries to YAML

## 6) Resolution scaling analysis

`benchmark_birefnet.py` supports:

```bash
--resolutions 512 1024 2048
```

For each resolution it records:

- quality metrics
- mean latency
- peak GPU memory (if CUDA)

## 7) Unified comparison interface

`segmentation_models/` exposes a common adapter API.

- Implemented: `BiRefNetAdapter`
- Placeholders ready for extension: U²Net, MODNet, SAM, IS-Net

This keeps benchmarking code unchanged when swapping model backends.

## 8) Visualization utilities

`visualization/visualize_results.py` generates:

- 4-panel plots (input / prediction / GT / overlay)
- `report.html` with embedded image previews

## 9) Practical application notes (from expected baseline outcomes)

### Strengths

- High-quality dichotomous segmentation boundaries
- Strong high-resolution behavior (especially Swin-based checkpoints)
- Good for foreground/background separation tasks

### Weaknesses

- Runtime/memory can increase notably at very high resolution
- Backbone and checkpoint pairing must be consistent
- Fine structures may still need post-processing in hard scenes

### Typical failure cases

- Low contrast foreground/background
- Transparent/reflective boundaries
- Dense clutter with foreground-like distractors

### Latency vs quality

- 512: fastest, lower detail fidelity
- 1024: balanced practical default
- 2048: best fine edges, highest compute and memory

### Ideal production scenarios

- Product cutout and e-commerce background removal
- Automotive vehicle isolation masks
- Large-scale mask generation for dataset curation
- Pre-processing masks for detection/tracking pipelines

## 10) Project structure

```text
birefnet_experiments/
  env/
  inference/
  benchmarks/
  datasets/
  visualization/
  segmentation_models/
  results/
```
