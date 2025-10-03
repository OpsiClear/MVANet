# Decoupled Inference Architecture

## Overview

The inference system is now split into two independent packages:

1. **inference-engine** - Core framework (model-agnostic)
2. **inference-engine-mvanet** - MVANet plugin

## Structure

```
mask_processor/
├── inference-engine/              # Core package (pip installable)
│   ├── src/inference_engine/
│   │   ├── __init__.py
│   │   ├── base.py               # SegmentationModel interface
│   │   ├── engine.py             # InferenceEngine
│   │   ├── registry.py           # Plugin system
│   │   └── cli.py                # CLI tool
│   ├── pyproject.toml
│   └── README.md
│
└── plugins/
    └── mvanet/                   # MVANet plugin (pip installable)
        ├── src/inference_engine_mvanet/
        │   ├── __init__.py
        │   ├── model.py          # MVANetModel
        │   ├── MVANet.py         # Original architecture
        │   └── SwinTransformer.py
        ├── pyproject.toml        # Registers entry point
        └── README.md
```

## Plugin System

Plugins register via entry points in `pyproject.toml`:

```toml
[project.entry-points."inference_engine.models"]
mvanet = "inference_engine_mvanet:MVANetModel"
```

The registry auto-discovers plugins at runtime.

## Installation

### Core Package
```bash
cd inference-engine
pip install -e .
```

### MVANet Plugin
```bash
cd plugins/mvanet
pip install -e .
```

## Usage

### List Models
```bash
inference-engine list
```

### Run Inference
```bash
inference-engine infer \
  --model mvanet \
  --model-path models/MVANet.pth \
  --input-folder /path/to/images
```

### Python API
```python
from inference_engine import InferenceEngine, create_model
import torch

model = create_model("mvanet")
model.load("models/MVANet.pth", torch.device("cuda:0"))

engine = InferenceEngine(model, device=torch.device("cuda:0"))
result = engine.process_folder("input", "overlays", "masks")
```

## Creating New Plugins

1. Create package structure:
```
my-model-plugin/
├── src/inference_engine_mymodel/
│   ├── __init__.py
│   └── model.py  # Implements SegmentationModel
└── pyproject.toml
```

2. Implement `SegmentationModel` interface
3. Register entry point in `pyproject.toml`
4. Install with `pip install -e .`

## Benefits

- **Decoupled**: Core engine independent of any specific model
- **Extensible**: Add models without touching core code
- **Installable**: Both packages pip-installable
- **Discoverable**: Auto-discovers plugins via entry points
- **Clean**: Clear separation of concerns
