"""Compare preprocessing overhead: to_device=True vs False"""

import torch
import time
import cv2
import numpy as np
from pathlib import Path
from inference_engine import create_model

# Create model
model = create_model("mvanet")
model.load(Path("models/MVANet.pth"), torch.device("cuda:0"))

# Get test images
folder = Path("D:/test/scan_20250930_131312_toy_car/images/cam_1")
image_files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]][:40]

# Load all images to memory first
images = []
for img_path in image_files:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)

print(f"Loaded {len(images)} images into memory\n")

# Test 1: to_device=True (single GPU mode)
print("Test 1: to_device=True (move to GPU)")
start = time.time()
for img in images:
    tensor, metadata = model.preprocess(img, to_device=True)
time1 = time.time() - start
print(f"  Time: {time1:.2f}s ({time1/len(images)*1000:.1f}ms/img)\n")

# Test 2: to_device=False (multi-GPU mode)
print("Test 2: to_device=False (CPU pinned)")
start = time.time()
for img in images:
    tensor, metadata = model.preprocess(img, to_device=False)
time2 = time.time() - start
print(f"  Time: {time2:.2f}s ({time2/len(images)*1000:.1f}ms/img)\n")

# Test 3: to_device=False + manual GPU transfer
print("Test 3: to_device=False + manual transfer to GPU")
start = time.time()
for img in images:
    tensor, metadata = model.preprocess(img, to_device=False)
    tensor_gpu = tensor.to(torch.device("cuda:0"), non_blocking=True)
time3 = time.time() - start
print(f"  Time: {time3:.2f}s ({time3/len(images)*1000:.1f}ms/img)\n")

print("Overhead analysis:")
print(f"  to_device=False overhead: +{(time2-time1)*1000:.0f}ms total ({(time2-time1)/len(images)*1000:.1f}ms/img)")
print(f"  Manual transfer overhead: +{(time3-time2)*1000:.0f}ms total ({(time3-time2)/len(images)*1000:.1f}ms/img)")
print(f"  Total multi-GPU overhead: +{(time3-time1)*1000:.0f}ms total ({(time3-time1)/len(images)*1000:.1f}ms/img)")
