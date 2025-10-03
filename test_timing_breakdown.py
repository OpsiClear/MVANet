"""Profile timing breakdown for multi-GPU engine"""

import torch
import time
from pathlib import Path
from inference_engine import InferenceEngine, create_model
import logging

logging.basicConfig(level=logging.WARNING)

# Test with 1 GPU using multi-GPU code path
devices = [torch.device("cuda:0")]

def model_factory(device):
    model = create_model("mvanet")
    model.load(Path("models/MVANet.pth"), device)
    model.optimize_for_inference(device)
    return model

# Create engine in multi-GPU mode with just 1 GPU
engine = InferenceEngine(
    devices=devices,
    model_factory=model_factory,
    chunk_size=20,  # Larger chunks
)

# Get test files
folder = Path("D:/test/scan_20250930_131312_toy_car/images/cam_1")
all_files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
image_files = all_files[:40]

print(f"Testing with {len(image_files)} images")
print(f"Chunk size: {engine.chunk_size}")
print()

# Time each phase
total_start = time.time()

# Phase 1: Preprocessing
preprocess_start = time.time()
preprocessed = []
for img_path in image_files:
    result = engine._load_and_preprocess(img_path)
    preprocessed.append(result)
preprocess_time = time.time() - preprocess_start
print(f"Preprocessing: {preprocess_time:.2f}s ({preprocess_time/len(image_files)*1000:.1f}ms/img)")

# Phase 2: GPU processing via multi-GPU engine
gpu_start = time.time()
image_data = [(img_path, img, tensor, meta) for img_path, (img, tensor, meta) in zip(image_files, preprocessed)]
results = engine.multi_gpu_engine.process_images(image_data, use_tta=False)
gpu_time = time.time() - gpu_start
print(f"GPU processing: {gpu_time:.2f}s ({gpu_time/len(image_files)*1000:.1f}ms/img)")

# Phase 3: Saving (simulated - just measure overhead)
save_start = time.time()
for _ in results:
    pass  # Just iterate
save_time = time.time() - save_start
print(f"Result collection: {save_time:.3f}s ({save_time/len(image_files)*1000:.1f}ms/img)")

total_time = time.time() - total_start
print()
print(f"Total: {total_time:.2f}s ({total_time/len(image_files)*1000:.1f}ms/img)")
print()
print("Breakdown:")
print(f"  Preprocessing: {preprocess_time/total_time*100:.1f}%")
print(f"  GPU processing: {gpu_time/total_time*100:.1f}%")
print(f"  Overhead: {(total_time - preprocess_time - gpu_time)/total_time*100:.1f}%")
