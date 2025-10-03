# Inference Engine

Generic inference engine for segmentation models with plugin architecture.

## Features

- **Model-agnostic**: Works with any segmentation model via plugin system
- **Multi-output support**: Models can output masks, depth maps, normal maps, etc.
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
# Single output model (default folders)
inference-engine infer \
  --model mvanet \
  --model-path models/MVANet.pth \
  --input-folder /path/to/images \
  --device cuda:0

# Multi-output model (custom folders)
inference-engine infer \
  --model mymodel \
  --model-path weights.pth \
  --input-folder /path/to/images \
  --output mask=output/masks \
  --output depth=output/depths \
  --output overlay=output/overlays
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

**See [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for detailed instructions.**

### Quick Example

```python
from inference_engine import SegmentationModel

class MyModel(SegmentationModel):
    def load(self, model_path, device): ...
    def preprocess(self, image): ...  # RGB numpy -> tensor
    def forward(self, tensor): ...
    def postprocess(self, output, metadata): ...  # -> uint8 numpy

    @property
    def name(self) -> str: return "MyModel"

    @property
    def supports_tta(self) -> bool: return True
```

Register in `pyproject.toml`:
```toml
[project.entry-points."inference_engine.models"]
mymodel = "inference_engine_mymodel:MyModel"
```

Install and use:
```bash
pip install -e .
inference-engine list  # Shows your model
inference-engine infer --model mymodel --model-path weights.pth --input-folder images/
```
