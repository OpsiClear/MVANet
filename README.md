# OC Masker - Image Segmentation Suite

A modular image segmentation application with a plugin architecture. Features a high-performance inference engine with multi-GPU support, multiple segmentation models, and a web-based monitoring dashboard.

## Features

- **Plugin Architecture**: Models auto-register via Python entry points
- **Multi-GPU Support**: Linear scaling with persistent worker architecture
- **High Performance**: Image-level pipelining for optimal throughput
- **Web Dashboard**: Real-time monitoring with live console logs
- **CLI Tools**: Comprehensive command-line interface
- **Multiple Models**: MVANet and PDFNet segmentation models

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/OpsiClear/MVANet.git
cd MVANet

# Install dependencies using uv
uv sync
```

### CLI Usage

```bash
# List available models
uv run oc-masker list

# Show model info
uv run oc-masker info --model mvanet

# Run inference on a folder
uv run oc-masker infer --model mvanet --input-folder /path/to/images --device cuda:0

# Multi-GPU inference
uv run oc-masker infer --model mvanet --input-folder /path/to/images --device cuda:0,cuda:1

# Recursive processing (finds all images/ folders)
uv run oc-masker infer --model mvanet --input-folder /path/to/root --recursive

# With test-time augmentation
uv run oc-masker infer --model mvanet --input-folder /path/to/images --tta-merge-mode mean
```

### Web Application

```bash
# Start the web server
uv run api_app.py

# Access at http://localhost:8001
```

## Package Structure

```
MVANet/
  oc_masker/                 # Core inference engine package
    src/oc_masker/
      base.py               # SegmentationModel abstract base class
      engine.py             # InferenceEngine - folder/recursive processing
      multi_gpu.py          # MultiGPUInferenceEngine - persistent worker pools
      registry.py           # Plugin discovery via Python entry points
      cli.py                # CLI commands (list, info, infer)
      monitor.py            # Folder monitoring for auto-processing
      app/                  # FastAPI web application
    tests/                  # Engine tests
    README.md               # Full engine documentation

  oc_masker_mvanet/         # MVANet model plugin
    src/oc_masker_mvanet/
      model.py              # MVANetModel(SegmentationModel)
      models/               # Network architecture
    checkpoints/            # Bundled model weights (Git LFS)
    tests/                  # Model tests
    README.md               # Model documentation

  oc_masker_pdfnet/         # PDFNet model plugin
    src/oc_masker_pdfnet/
      model.py              # PDFNetModel(SegmentationModel)
      models/               # Network architecture
    checkpoints/            # Model weights (auto-downloaded)
    tests/                  # Model tests
    README.md               # Model documentation

  scripts/                   # Utility scripts
    profile_full_pipeline.py
    profile_pdfnet_pipeline.py
    restore_scan_complete.py

  api_app.py                 # Development entry point for web app
  pyproject.toml             # Root project configuration
```

## Available Models

### MVANet

Multi-View Aggregation Network for image segmentation. Includes bundled model weights.

```bash
uv run oc-masker infer --model mvanet --input-folder /path/to/images
```

See [oc_masker_mvanet/README.md](oc_masker_mvanet/README.md) for full documentation.

### PDFNet

Patch-Depth Fusion Network with MoGe depth estimation. Produces masks, depth maps, and normal maps.

```bash
uv run oc-masker infer --model pdfnet --input-folder /path/to/images
```

See [oc_masker_pdfnet/README.md](oc_masker_pdfnet/README.md) for full documentation.

## Input/Output Structure

**Input folder structure:**
```
your-folder/
  images/           # Must be named 'images'
    subfolder/      # Supports nested structure
      image.jpg
```

**Output structure (created as siblings to images/):**
```
your-folder/
  masks/            # Binary masks (filename: {name}.png)
  overlays/         # RGBA overlays with transparency
  depths/           # Depth maps (if model supports)
  normals/          # Normal maps (if model supports)
  .mask_complete    # Flag when done
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/process` | Submit processing job |
| `GET /api/status/{request_id}` | Check task status |
| `GET /api/logs/{task_id}` | Get task-specific logs |
| `GET /api/models` | List available models |
| `GET /api/gpus` | List detected GPU devices |
| `GET /api/latest-image` | Get most recently processed image |

## Running Tests

```bash
# Run all tests
uv run pytest

# Run engine tests
uv run pytest oc_masker/tests/ -v

# Run PDFNet tests
uv run pytest oc_masker_pdfnet/tests/ -v
```

## Creating a New Model Plugin

See [oc_masker/README.md](oc_masker/README.md) for the plugin development guide.

## Requirements

- Python 3.12+
- PyTorch with CUDA support
- CUDA-capable GPU with 8GB+ VRAM

## License

MIT License
