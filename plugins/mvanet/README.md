# MVANet Plugin for Inference Engine

MVANet (Multi-View Aggregation Network) segmentation model plugin.

## Installation

```bash
pip install inference-engine-mvanet
```

## Usage

```bash
# Download model weights first
# Then run inference
inference-engine infer \
  --model mvanet \
  --model-path models/MVANet.pth \
  --input-folder /path/to/images
```

## Python API

```python
from inference_engine import InferenceEngine
from inference_engine_mvanet import MVANetModel
import torch

model = MVANetModel()
model.load("models/MVANet.pth", torch.device("cuda:0"))

engine = InferenceEngine(model, device=torch.device("cuda:0"))
result = engine.process_folder("input/images", "output/overlays", "output/masks")
```

## Model Weights

Download MVANet weights from the original repository.
