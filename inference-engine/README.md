# Inference Engine

Generic inference engine for segmentation models with plugin architecture.

## Features

- **Model-agnostic**: Works with any segmentation model via plugin system
- **Fast**: Optimized with FP16, channels_last, chunked processing
- **Extensible**: Easy plugin system via entry points
- **CLI**: Simple command-line interface

## Installation

```bash
pip install inference-engine
```

## Usage

### CLI

List available models:
```bash
inference-engine list
inference-engine list --verbose
```

Show model info:
```bash
inference-engine info --model mvanet
```

Run inference:
```bash
inference-engine infer \
  --model mvanet \
  --model-path models/MVANet.pth \
  --input-folder /path/to/images \
  --device cuda:0
```

### Python API

```python
from inference_engine import InferenceEngine, create_model
import torch

# Create model
model = create_model("mvanet")
model.load("models/MVANet.pth", torch.device("cuda:0"))

# Create engine
engine = InferenceEngine(model, device=torch.device("cuda:0"))

# Process folder
result = engine.process_folder(
    folder_path="input/images",
    overlay_folder="output/overlays",
    mask_folder="output/masks"
)
```

## Creating Plugins

See plugin documentation in `plugins/` directory.

Example plugin entry point in `pyproject.toml`:

```toml
[project.entry-points."inference_engine.models"]
mvanet = "inference_engine_mvanet:MVANetModel"
```
