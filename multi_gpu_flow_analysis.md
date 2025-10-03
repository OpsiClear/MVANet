# Multi-GPU Flow Analysis

## Current Implementation (3 GPUs: cuda:0, cuda:1, cuda:2)

### Initialization
```python
# engine.py:68-75
devices = [cuda:0, cuda:1, cuda:2]
multi_gpu_engine = MultiGPUInferenceEngine(model_factory, devices, use_fp16)
# Creates 3 model instances:
#   models[cuda:0].device = cuda:0
#   models[cuda:1].device = cuda:1
#   models[cuda:2].device = cuda:2

self.model = multi_gpu_engine.models[devices[0]]  # GPU 0's model
self.device = devices[0]  # cuda:0
```

### Preprocessing Flow
```python
# engine.py:424
img_tensor, metadata = self.model.preprocess(image_rgb)
# self.model is the cuda:0 model instance
```

```python
# model.py:63-89 (MVANet preprocess, self.device = cuda:0)
img_tensor = torch.from_numpy(...)  # CPU tensor
if self.device.type == "cuda":
    img_tensor = img_tensor.to(memory_format=torch.channels_last)
    img_tensor = img_tensor.pin_memory()  # CPU pinned memory
img_tensor = img_tensor.to(self.device)  # Move to cuda:0 ⚠️
return img_tensor  # Tensor is on cuda:0
```

### Worker Processing
```python
# multi_gpu.py:66-70
# Worker 0 (device = cuda:0)
img_tensor = img_tensor.to(device)  # cuda:0 → cuda:0 (no-op)

# Worker 1 (device = cuda:1)
img_tensor = img_tensor.to(device)  # cuda:0 → cuda:1 ⚠️ GPU-to-GPU transfer

# Worker 2 (device = cuda:2)
img_tensor = img_tensor.to(device)  # cuda:0 → cuda:2 ⚠️ GPU-to-GPU transfer
```

## Problem Identified

**GPU-to-GPU transfers via PCIe:**
- Worker 1 transfers: cuda:0 → cuda:1 (slower, goes through CPU/PCIe)
- Worker 2 transfers: cuda:0 → cuda:2 (slower, goes through CPU/PCIe)

**Better approach:**
- Preprocess to CPU pinned memory
- Workers transfer: CPU → their GPU (direct, optimal)

## Proposed Fix

### Modified Preprocessing
```python
# model.py preprocess with to_device parameter
def preprocess(self, image: np.ndarray, to_device: bool = True):
    img_tensor = torch.from_numpy(...)  # CPU tensor

    if to_device:
        # Single GPU mode
        if self.device.type == "cuda":
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.pin_memory()
        img_tensor = img_tensor.to(self.device, non_blocking=True)
    else:
        # Multi-GPU mode: keep in pinned CPU memory
        img_tensor = img_tensor.pin_memory()

    return img_tensor  # CPU pinned or GPU tensor
```

### Engine calls with to_device flag
```python
# Single GPU mode
img_tensor, metadata = self.model.preprocess(image_rgb, to_device=True)

# Multi-GPU mode
img_tensor, metadata = self.model.preprocess(image_rgb, to_device=False)
```

### Workers transfer from CPU
```python
# All workers transfer from CPU pinned → their GPU
# Worker 0: CPU → cuda:0
# Worker 1: CPU → cuda:1
# Worker 2: CPU → cuda:2
img_tensor = img_tensor.to(device, non_blocking=True)
```

## Benefits
✅ No GPU-to-GPU transfers
✅ Equal PCIe bandwidth usage across GPUs
✅ Simpler data flow
✅ Backward compatible (to_device=True by default)
