# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OC Masker is an image segmentation application with a modular plugin architecture. The project consists of:

- **Core Engine** (`oc_masker/`): High-performance inference engine with multi-GPU support
- **Model Plugins** (`oc_masker_mvanet/`, `oc_masker_pdfnet/`): Segmentation models that auto-register via Python entry points
- **Web Application** (`api_app.py`): FastAPI backend with real-time monitoring dashboard
- **Utility Scripts** (`scripts/`): Profiling and maintenance utilities

## Running the Application

Start the FastAPI web server:
```bash
uv run api_app.py
```
The application runs on `http://localhost:8001` and serves both the API and web UI.

## CLI Commands

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

# Fill holes in masks (fix highlights/reflections)
uv run oc-masker infer --model mvanet --input-folder /path/to/images --fill-holes 2000

# Monitor folder for new datasets (auto-process when .scan_complete appears)
uv run oc-masker monitor --parent-folder /path/to/datasets --model mvanet --device cuda:0

# Monitor with reprocessing (backup and redo existing outputs)
uv run oc-masker monitor --parent-folder /path/to/datasets --model mvanet --reprocess
```

## Dependencies

- Use `uv sync` to install dependencies
- Requires Python 3.12+
- PyTorch with CUDA 12.8 support configured via uv sources
- Local packages installed via `[tool.uv.sources]` in pyproject.toml

## Architecture

### Package Structure

```
MVANet/
  oc_masker/                 # Core inference engine package
    src/oc_masker/
      base.py               # SegmentationModel abstract base class
      engine.py             # InferenceEngine - folder/recursive processing
      multi_gpu.py          # MultiGPUInferenceEngine - persistent worker pools
      registry.py           # Plugin discovery via Python entry points
      cli.py                # CLI commands (infer, monitor, list, info)
      monitor.py            # Folder monitoring for auto-processing
      app/                  # FastAPI web application with static files
    tests/                  # Engine and output format tests
    README.md               # Full engine documentation

  oc_masker_mvanet/         # MVANet model plugin
    src/oc_masker_mvanet/
      model.py              # MVANetModel(SegmentationModel)
      models/
        mvanet.py           # MVANet architecture
        swin_transformer.py # Swin Transformer backbone
    checkpoints/            # Bundled model weights (Git LFS)
    tests/                  # Model-specific tests
    README.md               # Model documentation

  oc_masker_pdfnet/         # PDFNet model plugin
    src/oc_masker_pdfnet/
      model.py              # PDFNetModel(SegmentationModel)
      models/               # Network architecture
    checkpoints/            # Model weights (auto-downloaded from HuggingFace)
    tests/                  # TensorRT and GPU preprocessing tests
    README.md               # Model documentation

  scripts/                   # Utility scripts
    profile_full_pipeline.py      # MVANet pipeline profiling
    profile_pdfnet_pipeline.py    # PDFNet pipeline profiling
    restore_scan_complete.py      # Restore completion flags

  api_app.py                # Development entry point for web app
  pyproject.toml            # Root project configuration
```

### Plugin System

Models auto-register via Python entry points in `pyproject.toml`:
```toml
[project.entry-points."oc_masker.models"]
mvanet = "oc_masker_mvanet:MVANetModel"
```

Key registry functions:
- `list_models()` - Returns list of installed model names
- `create_model(name, **kwargs)` - Instantiates model by name
- `get_model_info(name)` - Returns model metadata

### Inference Pipeline

1. **InferenceEngine** creates **MultiGPUInferenceEngine** with persistent worker threads
2. Workers stay alive between batches, avoiding repeated model loading
3. Main thread preprocesses images (CPU) and feeds to GPU worker queues
4. Image-level pipelining: preprocessing overlaps with GPU inference
5. Results collected and saved via ThreadPoolExecutor

### SegmentationModel Interface

All model plugins implement:
```python
class MyModel(SegmentationModel):
    def load(self, model_path: Path | None, device: torch.device): ...  # None = use bundled weights
    def preprocess(self, image, to_device=True) -> tuple[tensor, metadata]: ...
    def forward(self, tensor) -> tensor | dict: ...
    def postprocess(self, output, metadata) -> ndarray | dict: ...
    @property
    def name(self) -> str: ...
    @property
    def supports_tta(self) -> bool: ...
    def get_output_names(self) -> list[str]: ...  # ["mask"] or ["mask", "depth", ...]
```

### Web API (api_app.py)

- FastAPI with ThreadPoolExecutor (max_workers=1) for GPU tasks
- Single-task processing queue with `processing_lock`
- In-memory logging with task-specific log tracking via `MemoryLogHandler`
- Engine caching: engines keyed by `{model}_{devices}_{config}`
- Static files served from `oc_masker/src/oc_masker/app/static/`

Key endpoints:
- `POST /api/process` - Submit processing job
- `GET /api/status/{request_id}` - Check task status
- `GET /api/logs/{task_id}` - Get task-specific logs
- `GET /api/models` - List available models
- `GET /api/gpus` - List detected GPU devices
- `GET /api/latest-image` - Get most recently processed image

## File Organization

Input folder structure:
```
your-folder/
  images/           # Must be named 'images'
    subfolder/      # Supports nested structure
      image.jpg
```

Output structure (created as siblings to `images/`):
```
your-folder/
  masks/            # Binary masks (filename: {name}.png)
  overlays/         # RGBA overlays with transparency
  depths/           # Depth maps (if model supports)
  normals/          # Normal maps (if model supports)
  .mask_processing  # Flag during processing
  .mask_complete    # Flag when done
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run engine tests
uv run pytest oc_masker/tests/ -v

# Run PDFNet tests (requires TensorRT)
uv run pytest oc_masker_pdfnet/tests/ -v
```

## Performance Configuration

Key settings in `InferenceEngine.__init__`:
- `use_fp16=True` - FP16 mixed precision
- `chunk_size=20` - Images per batch (adjust for GPU memory: 10 for 8GB, 30 for 24GB+)
- `fill_holes_max_area=0` - Fill isolated holes in masks (0=disabled)

TTA merge modes:
- `none` - No augmentation (fastest)
- `mean` - Arithmetic average (balanced)
- `max` - Maximum value (high recall)
- `gmean` - Geometric mean (conservative)

## Creating a New Model Plugin

1. Create package with `src/oc_masker_yourmodel/` structure
2. Implement `SegmentationModel` subclass in `model.py`
3. Add entry point in `pyproject.toml`:
   ```toml
   [project.entry-points."oc_masker.models"]
   yourmodel = "oc_masker_yourmodel:YourModel"
   ```
4. Bundle weights in `checkpoints/` directory (use Git LFS)
5. Export `get_checkpoint_path()` function for bundled weights access

See `oc_masker/README.md` for complete plugin development documentation.

## Important Implementation Details

- Device selection: Defaults to `cuda:0`, `auto` detects all GPUs
- Multi-GPU: Persistent worker architecture avoids model reload overhead
- Preprocessing: `to_device=False` keeps tensors in CPU pinned memory for multi-GPU
- File skipping: Already-processed files detected by checking output existence
- Supported formats: .jpg, .jpeg, .png, .bmp
