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

**Single output (mask only)**:
```bash
inference-engine infer \
  --model mvanet \
  --model-path models/MVANet.pth \
  --input-folder /path/to/images
```

**Multi-output (mask, depth, normal, etc.)**:
```bash
inference-engine infer \
  --model mymodel \
  --model-path weights.pth \
  --input-folder /path/to/images \
  --output mask=output/masks \
  --output depth=output/depths \
  --output normal=output/normals \
  --output overlay=output/overlays
```

### Python API

**Single output**:
```python
from inference_engine import InferenceEngine, create_model
from pathlib import Path
import torch

model = create_model("mvanet")
model.load("models/MVANet.pth", torch.device("cuda:0"))

engine = InferenceEngine(model, device=torch.device("cuda:0"))
result = engine.process_folder(
    folder_path=Path("input"),
    output_folders={"mask": Path("masks"), "overlay": Path("overlays")}
)
```

**Multi-output**:
```python
result = engine.process_folder(
    folder_path=Path("input"),
    output_folders={
        "mask": Path("output/masks"),
        "depth": Path("output/depths"),
        "normal": Path("output/normals"),
        "overlay": Path("output/overlays"),
    },
    create_overlays=True,
)
```

## Creating New Plugins

### Quick Start

1. **Copy the template**:
   ```bash
   cp -r plugins/template plugins/your-model
   ```

2. **Customize your plugin**:
   - Rename package: `inference_engine_template` → `inference_engine_yourmodel`
   - Implement model in `model.py`
   - Update `pyproject.toml` with entry point

3. **Install and test**:
   ```bash
   pip install -e plugins/your-model
   inference-engine list
   ```

### Required Implementation

Your model class must inherit from `SegmentationModel` and implement:

```python
class YourModel(SegmentationModel):
    def load(self, model_path: Path, device: torch.device) -> None
    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]
    def forward(self, tensor: torch.Tensor) -> torch.Tensor
    def postprocess(self, output: torch.Tensor, metadata: dict) -> np.ndarray

    @property
    def name(self) -> str

    @property
    def supports_tta(self) -> bool
```

### Documentation

- **Detailed guide**: [inference-engine/PLUGIN_GUIDE.md](inference-engine/PLUGIN_GUIDE.md)
- **Template plugin**: [plugins/template/](plugins/template/)
- **Working example**: [plugins/mvanet/](plugins/mvanet/)

## Benefits

- **Decoupled**: Core engine independent of any specific model
- **Extensible**: Add models without touching core code
- **Installable**: Both packages pip-installable
- **Discoverable**: Auto-discovers plugins via entry points
- **Developer-friendly**: Template + comprehensive guide
- **Clean**: Clear separation of concerns
