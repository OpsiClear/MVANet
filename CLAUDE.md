# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MVANet is an image segmentation web application that uses deep learning (MVANet model with Swin Transformer backbone) to process images. It features a FastAPI backend with a modern web dashboard for batch image processing with real-time monitoring.

## Running the Application

Start the FastAPI web server:
```bash
uv run api_app.py
```
The application runs on `http://localhost:8001` and serves both the API and web UI.

## Core Processing Commands

Run inference on a folder (CLI):
```bash
uv run cli.py --input-folder <path> [--use-tta] [--device cuda:0] [--log-level INFO]
```

Note: CLI uses `tyro` for type-safe argument parsing with Python 3.12 type hints.

The application expects folders named `images` (case-insensitive) and will:
- Recursively find all `images/` folders under the input path
- Generate outputs in sibling folders: `overlays/` and `masks/`
- Skip already-processed files automatically

## Dependencies

- Use `uv sync` to install dependencies (preferred)
- Requires Python 3.12+
- PyTorch with CUDA 12.8 support configured via uv sources
- Key packages: FastAPI, PyTorch, timm, mmcv, ttach, einops

## Architecture

### Backend (api_app.py)
- FastAPI application with background task executor (ThreadPoolExecutor)
- Single-task processing queue (one GPU task at a time)
- In-memory logging with task-specific log tracking via MemoryLogHandler
- Real-time log streaming and system status endpoints
- Thread-safe state management using `processing_lock`

### Model & Inference (src/)
- `MVANet.py`: Model architecture (MCLM multi-head attention modules)
- `SwinTransformer.py`: Swin Transformer backbone
- `inference.py`: Optimized inference pipeline with 8 performance optimizations:
  - Channels-last memory format for model and tensors
  - Pinned memory for CPU->GPU transfers
  - Single GPU->CPU transfer in postprocessing
  - Optional FP16 mixed precision (controlled by `USE_FP16` constant)
  - TensorFloat32 for Ampere+ GPUs
  - Chunked processing (20 images per chunk)
  - Fast PNG compression (compress_level=1)
  - Thread-local transforms

### Frontend (static/)
- Bootstrap 5 UI with vanilla JavaScript
- Real-time console log streaming
- Latest image viewer with task persistence across refreshes

## Key API Endpoints

- `POST /api/process` - Submit processing job (requires `input_folder` and optional `use_tta`)
- `GET /api/status/{request_id}` - Check task status
- `GET /api/system/status` - Current processing state
- `GET /api/logs/{task_id}` - Get task-specific logs (supports `since` parameter)
- `GET /api/latest-image` - Retrieve most recently processed image

## File Organization

Input folder structure:
```
your-folder/
├── images/           # Must be named 'images'
│   ├── subfolder/    # Supports nested structure
│   └── image.jpg
```

Output structure (created as siblings to `images/`):
```
your-folder/
├── overlays/         # RGBA overlays with transparency (filename: {stem}.png)
└── masks/            # Binary masks (filename: {name}.png, e.g., image.jpg → image.jpg.png)
```

## Model Files

Required models in `models/` directory (tracked by Git LFS):
- `MVANet.pth` - Main segmentation model
- `swin_base_patch4_window12_384_22kto1k.pth` - Backbone model

## Performance Configuration

Inference optimization flags (in `src/inference.py`):
- `USE_FP16 = True` - FP16 precision (2x faster, <0.1% accuracy loss)
- `NUM_WORKERS` - Thread pool size (default: min(8, cpu_count))
- `chunk_size = 20` - Batch size (adjust for GPU memory: 10 for 8GB, 30 for 24GB+)

## Important Implementation Details

- Device selection: Defaults to `cuda:0`, overridable via `SELECTED_DEVICE` environment variable
- Test-Time Augmentation: Uses HorizontalFlip only (not both H+V flips) for batch processing
- Logging: Task-specific logging via monkey-patched logging module during background processing
- File skipping: Images with "montage" in filename are skipped
- Supported formats: .jpg, .jpeg, .png, .bmp
