"""Test that preprocessing uses correct device mode"""

import torch
import numpy as np
from pathlib import Path
from inference_engine import create_model

# Create model on GPU 0
model = create_model("mvanet")
model.load(Path("models/MVANet.pth"), torch.device("cuda:0"))

# Test image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Test 1: to_device=True (single GPU mode)
print("Test 1: to_device=True (single GPU mode)")
tensor, metadata = model.preprocess(test_image, to_device=True)
print(f"  Tensor device: {tensor.device}")
print(f"  Expected: cuda:0")
assert tensor.device == torch.device("cuda:0"), "Should be on cuda:0"
print("  PASSED")

# Test 2: to_device=False (multi-GPU mode)
print("\nTest 2: to_device=False (multi-GPU mode)")
tensor, metadata = model.preprocess(test_image, to_device=False)
print(f"  Tensor device: {tensor.device}")
print(f"  Tensor is_pinned: {tensor.is_pinned()}")
print(f"  Expected: cpu with pinned memory")
assert tensor.device == torch.device("cpu"), "Should be on CPU"
assert tensor.is_pinned(), "Should be pinned in memory"
print("  PASSED")

# Test 3: Worker can transfer from CPU to any GPU
print("\nTest 3: Transfer from CPU pinned to GPU")
tensor_gpu1 = tensor.to(torch.device("cuda:0"), non_blocking=True)
print(f"  Transferred to: {tensor_gpu1.device}")
print(f"  Expected: cuda:0")
assert tensor_gpu1.device == torch.device("cuda:0"), "Should transfer to cuda:0"
print("  PASSED")

print("\nAll tests passed!")
