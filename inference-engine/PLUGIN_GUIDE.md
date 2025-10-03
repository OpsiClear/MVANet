# Plugin Development Guide

## Overview

Create custom segmentation model plugins for the inference engine.

## Quick Start

### 1. Create Package Structure

```
my-model-plugin/
├── src/
│   └── inference_engine_mymodel/
│       ├── __init__.py
│       └── model.py
├── pyproject.toml
└── README.md
```

### 2. Implement Model Class

Create `src/inference_engine_mymodel/model.py`:

```python
"""My custom model plugin"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

from inference_engine import SegmentationModel


class MyModel(SegmentationModel):
    """My segmentation model"""

    def __init__(self, image_size: tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.model = None
        self.device = None

    def load(self, model_path: Path, device: torch.device) -> None:
        """Load model weights"""
        self.device = device
        # Load your model architecture
        # self.model = YourModelArchitecture()
        # checkpoint = torch.load(model_path, map_location=device)
        # self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

    def optimize_for_inference(self, device: torch.device) -> None:
        """Apply optimizations"""
        if device.type == "cuda" and self.model:
            self.model = self.model.to(memory_format=torch.channels_last)

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image (RGB numpy array)

        Args:
            image: RGB image from OpenCV (H, W, 3)

        Returns:
            (tensor, metadata dict)
        """
        original_size = (image.shape[1], image.shape[0])  # (W, H)

        # Resize
        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        # Normalize (example: ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = resized.astype(np.float32) / 255.0
        img_normalized = (img_float - mean) / std

        # Convert to tensor: HWC -> CHW
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)

        # GPU optimizations
        if self.device.type == "cuda":
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.pin_memory()

        img_tensor = img_tensor.to(self.device, non_blocking=True)

        return img_tensor, {"original_size": original_size}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, output: torch.Tensor, metadata: dict) -> np.ndarray:
        """
        Convert model output to grayscale mask

        Args:
            output: Model output tensor
            metadata: Contains 'original_size' (W, H)

        Returns:
            Grayscale mask (H, W) as uint8 numpy array
        """
        original_size = metadata["original_size"]

        # Apply activation (if needed)
        mask_tensor = torch.sigmoid(output)  # or F.softmax, etc.

        # Resize to original size
        mask_resized = F.interpolate(
            mask_tensor,
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bilinear",
            align_corners=False
        )

        # Convert to numpy uint8
        mask_np = (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)
        return mask_np

    @property
    def name(self) -> str:
        return "MyModel"

    @property
    def supports_tta(self) -> bool:
        return True  # or False if TTA not supported

    @classmethod
    def get_metadata(cls) -> dict:
        """Return model metadata for display"""
        return {
            "name": "MyModel",
            "description": "My awesome segmentation model",
            "author": "Your Name",
            "version": "1.0.0",
        }
```

### 3. Create Package Init

Create `src/inference_engine_mymodel/__init__.py`:

```python
"""My Model plugin for inference engine"""

from .model import MyModel

__version__ = "1.0.0"
__all__ = ["MyModel"]
```

### 4. Configure pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "inference-engine-mymodel"
version = "1.0.0"
description = "My model plugin for inference-engine"
requires-python = ">=3.12"
dependencies = [
    "inference-engine>=0.1.0",
    "torch>=2.8.0",
    # Add your model dependencies here
]

[project.entry-points."inference_engine.models"]
mymodel = "inference_engine_mymodel:MyModel"

[tool.hatch.build.targets.wheel]
packages = ["src/inference_engine_mymodel"]
```

### 5. Install and Test

```bash
cd my-model-plugin
pip install -e .

# Verify plugin is discovered
inference-engine list

# Test inference
inference-engine infer \
  --model mymodel \
  --model-path /path/to/weights.pth \
  --input-folder /path/to/images
```

## Key Requirements

### SegmentationModel Interface

Your model class MUST implement:

1. **`load(model_path, device)`** - Load weights
2. **`preprocess(image)`** - Convert RGB numpy to tensor + metadata
3. **`forward(tensor)`** - Run inference
4. **`postprocess(output, metadata)`** - Convert output to uint8 numpy mask
5. **`name`** property - Model name string
6. **`supports_tta`** property - Boolean for TTA support

Optional:
- **`optimize_for_inference(device)`** - Apply optimizations
- **`get_metadata()`** - Return model info dict

### Image Format

- **Input**: RGB numpy array (H, W, 3) from OpenCV
- **Output**: Grayscale mask (H, W) uint8 numpy array (0-255)

### Entry Point

Must register in `pyproject.toml`:

```toml
[project.entry-points."inference_engine.models"]
your_model_name = "your_package:YourModelClass"
```

## Complete Example: U2Net Plugin

```python
# src/inference_engine_u2net/model.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

from inference_engine import SegmentationModel
from .u2net_arch import U2NET  # Your architecture


class U2NetModel(SegmentationModel):
    def __init__(self, image_size: tuple[int, int] = (320, 320)):
        self.image_size = image_size
        self.model = None
        self.device = None

    def load(self, model_path: Path, device: torch.device) -> None:
        self.device = device
        self.model = U2NET(3, 1)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        original_size = (image.shape[1], image.shape[0])
        resized = cv2.resize(image, self.image_size)

        # U2Net normalization
        img_float = resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)

        if self.device.type == "cuda":
            img_tensor = img_tensor.pin_memory()

        return img_tensor.to(self.device), {"original_size": original_size}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        d1, d2, d3, d4, d5, d6, d7 = self.model(tensor)
        return d1  # Use first output

    def postprocess(self, output: torch.Tensor, metadata: dict) -> np.ndarray:
        original_size = metadata["original_size"]

        # Normalize to 0-1
        ma = torch.max(output)
        mi = torch.min(output)
        output = (output - mi) / (ma - mi)

        # Resize
        mask_resized = F.interpolate(
            output,
            size=(original_size[1], original_size[0]),
            mode="bilinear"
        )

        return (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)

    @property
    def name(self) -> str:
        return "U2Net"

    @property
    def supports_tta(self) -> bool:
        return True

    @classmethod
    def get_metadata(cls) -> dict:
        return {
            "name": "U2Net",
            "description": "U2Net: Going Deeper with Nested U-Structure",
            "author": "Xuebin Qin et al.",
            "version": "1.0.0",
        }
```

## Multi-Output Models

Models that produce multiple outputs (masks, depth, normals, etc.) should:

### 1. Override get_output_names()

```python
def get_output_names(self) -> list[str]:
    return ["mask", "depth", "normal"]
```

### 2. Return dict from postprocess()

```python
def postprocess(
    self, output: dict[str, torch.Tensor], metadata: dict
) -> dict[str, np.ndarray]:
    original_size = metadata["original_size"]
    results = {}

    # Process mask
    mask_tensor = torch.sigmoid(output["mask"])
    mask_resized = F.interpolate(mask_tensor, size=(original_size[1], original_size[0]), mode="bilinear")
    results["mask"] = (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)

    # Process depth
    depth_resized = F.interpolate(output["depth"], size=(original_size[1], original_size[0]), mode="bilinear")
    depth_np = depth_resized.squeeze().cpu().numpy()
    depth_np = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255)
    results["depth"] = depth_np.astype(np.uint8)

    # Process normal map (RGB)
    normal_resized = F.interpolate(output["normal"], size=(original_size[1], original_size[0]), mode="bilinear")
    normal_np = normal_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    results["normal"] = ((normal_np + 1) / 2 * 255).astype(np.uint8)

    return results
```

### 3. Usage

```bash
# CLI automatically creates folders for each output
inference-engine infer \
  --model mymodel \
  --model-path weights.pth \
  --input-folder images/

# Or specify custom output folders
inference-engine infer \
  --model mymodel \
  --model-path weights.pth \
  --input-folder images/ \
  --output mask=output/masks \
  --output depth=output/depths \
  --output normal=output/normals \
  --output overlay=output/overlays
```

### 4. Python API

```python
from inference_engine import InferenceEngine, create_model
from pathlib import Path

model = create_model("mymodel")
model.load("weights.pth", device)

engine = InferenceEngine(model, device)
result = engine.process_folder(
    folder_path=Path("images"),
    output_folders={
        "mask": Path("output/masks"),
        "depth": Path("output/depths"),
        "normal": Path("output/normals"),
        "overlay": Path("output/overlays"),
    },
    create_overlays=True,  # Creates RGBA overlay from mask
)
```

## Best Practices

1. **Memory Management**: Use `torch.no_grad()`, clear cache when needed
2. **GPU Optimization**: Use `channels_last`, `pin_memory()` for better performance
3. **Error Handling**: Validate inputs, handle edge cases
4. **Metadata**: Provide accurate model information
5. **Documentation**: Document preprocessing/postprocessing details
6. **Testing**: Test with various image sizes and formats
7. **Multi-Output**: Use consistent naming (mask, depth, normal, etc.)

## Publishing

### PyPI Release

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### Installation

```bash
pip install inference-engine-mymodel
```

The plugin will be auto-discovered when users run `inference-engine list`.
