# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Management with uv

Use uv exclusively for Python package management in this project.

### Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Lock dependencies: `uv lock`

### Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools with `uv run <tool>` (e.g. `uv run pytest`, `uv run ruff`, `uv run mypy`, `uv run pre-commit`)
- Launch a Python REPL with `uv run python`

## What This Is

GEM-X (Generalist Estimation of Human Motion) is a commercial-grade monocular video 3D human pose estimation model by NVIDIA. It recovers full-body 77-joint SOMA body parameters (body + hands + face) from monocular video, and can retarget recovered motion to a Unitree G1 humanoid robot.

## Environment Setup

**Prerequisites:** Python 3.12+, CUDA 12.6+ GPU, Git LFS, `uv`

```bash
# Full install (Linux/GPU)
pip install uv && uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e third_party/soma && cd third_party/soma && git lfs pull && cd ../..
bash scripts/install_env.sh

# Optional: robot retargeting
uv pip install -e third_party/soma-retargeter
```

`scripts/install_env.sh` installs the `gem` package in editable mode plus SAM-3D-Body runtime deps. On macOS it skips Detectron2 and installs ONNX Runtime instead.

If submodules are missing: `git submodule update --init --recursive`

## Linting and Formatting

```bash
ruff check gem/ scripts/ tools/     # lint (B, E, F, I, UP, W rules)
ruff format gem/ scripts/ tools/    # auto-format
black gem/ scripts/ tools/          # formatter (100 char line length)
```

No test suite is defined in the repo.

## Running Demos

```bash
# Full 3D pipeline (auto-downloads checkpoint from HuggingFace if --ckpt omitted)
python scripts/demo/demo_soma.py --video path/to/video.mp4 --ckpt inputs/pretrained/gem_soma.ckpt

# ONNX/TensorRT accelerated (also supports Apple Silicon via CoreML)
python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4

# 2D keypoints only (no GEM model, lightweight)
python scripts/demo/demo_2d_keypoints.py --video path/to/video.mp4

# With G1 robot retargeting
python scripts/demo/demo_soma.py --video path/to/video.mp4 --retarget
```

Key flags: `--output_root`, `--verbose` (debug overlays), `-s` (static camera), `--ddim` (higher-quality 50-step sampling in ONNX script).

## Training

```bash
# Single-GPU
python scripts/train.py exp=gem_soma_regression use_wandb=false

# Multi-GPU (DDP)
python scripts/train.py exp=gem_soma_regression pl_trainer.devices=4

# Evaluation
python scripts/train.py exp=gem_soma_regression task=test
```

Config uses [Hydra](https://hydra.cc/). Override anything from CLI: `python scripts/train.py exp=gem_soma_regression pl_trainer.max_steps=100000 optimizer.lr=1e-4`

Training data (`inputs/metrosim_data_A2G_Bones2_DH/`) is NVIDIA-internal and not publicly released.

## Exporting ONNX Models

```bash
python tools/export/export_vitpose_onnx.py
python tools/export/export_sam3db_onnx.py
python tools/export/export_denoiser_onnx.py --ckpt <path>
```

## Code Architecture

### Inference Pipeline

```
Input Video
  → YOLOX + ByteTrack  (person detection & tracking)     gem/utils/yolox_detector.py
  → VitPose             (2D keypoints, 77 SOMA joints)    gem/utils/vitpose_extractor.py
  → SAM-3D-Body         (video feature extraction)        gem/utils/sam3db_extractor.py
  → GEM denoiser        (3D SOMA parameter regression)    gem/network/gem_denoiser.py
  → Postprocessing      (world-space, smoothing)          gem/pipeline/postprocess.py
  → Renderers           (in-cam mesh, global view)        gem/utils/vis/
  → (opt) Retargeter    (SOMA → Unitree G1 BVH/CSV)       third_party/soma-retargeter/
```

Preprocessing caches intermediate results under `<output_root>/<video_name>/preprocess/` (`.pt` files), so reruns skip completed stages.

### Key Modules

- **`gem/gem.py`** — PyTorch Lightning module; wraps `pipeline`, `endecoder`, optimizer, and scheduler. Entry point for training.
- **`gem/pipeline/gem_pipeline.py`** — Forward pass orchestration: calls endecoder, computes losses, handles train/test modes.
- **`gem/pipeline/gem_denoiser.py`** — Core diffusion denoiser logic (regression + optional DDIM).
- **`gem/network/gem_denoiser.py`** — Transformer-based denoiser architecture (ViT-style, 12 layers, 512 latent dim). Takes video features + 2D keypoints + camera intrinsics; outputs SOMA body params.
- **`gem/network/endecoder.py`** — Encoder/decoder wrapping the network; manages observation indices.
- **`gem/utils/soma_utils/soma_layer.py`** — Wraps the SOMA parametric body model (assets at `inputs/soma_assets/`).
- **`gem/utils/vitpose_extractor.py`** — DINOv3-based 2D pose model (77 SOMA keypoints).
- **`gem/utils/geo_transform.py`, `rotation_conversions.py`, `quaternion.py`** — 3D math utilities used throughout.

### Config System

Hydra configs live in `configs/`. Key groups:
- `configs/exp/` — top-level experiment configs (e.g. `gem_soma_regression.yaml`)
- `configs/model/`, `configs/network/` — model architecture
- `configs/pipeline/` — loss weights and feature flags
- `configs/train_datasets/`, `configs/test_datasets/` — data paths

### Third-Party Submodules

All live under `third_party/` and must be installed with `uv pip install -e`:
- `third_party/soma` — SOMA body model (requires Git LFS for assets)
- `third_party/soma-retargeter` — robot motion retargeting (SSH submodule, optional)
- `third_party/sam-3d-body` — video feature extractor backbone

### Directory Layout

```
gem/           # main Python package
  network/     # model architecture (denoiser transformer, endecoder)
  pipeline/    # training pipeline, loss, diffusion utils
  datasets/    # dataset loaders (MetroSim, mocap)
  datamodule/  # Lightning DataModule
  utils/       # 3D math, I/O, detectors, renderers, SOMA utils
scripts/
  demo/        # demo_soma.py, demo_soma_onnx.py, demo_2d_keypoints.py
  train.py     # training entry point
configs/       # Hydra config tree
tools/         # ONNX export scripts
inputs/        # pretrained checkpoints, SOMA assets, training data (gitignored)
outputs/       # demo results (gitignored)
third_party/   # git submodules
docs/          # INSTALL.md, DEMO.md, TRAINING.md, MODEL_OVERVIEW.md
```

## Pretrained Checkpoint

Place at `inputs/pretrained/gem_soma.ckpt` or pass via `--ckpt`. Demo scripts auto-download from `nvidia/GEM-X` on HuggingFace if omitted.

## Common Gotchas

- OpenGL/EGL errors on headless servers: set `PYOPENGL_PLATFORM=egl` and `EGL_PLATFORM=surfaceless`
- `ModuleNotFoundError: gem` means `bash scripts/install_env.sh` was not run with the venv active
- SOMA body model `.pt` files under `third_party/soma` will be Git LFS pointers if `git lfs pull` was not run
