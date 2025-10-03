"""Test persistent workers with large batch"""

import torch
import time
from pathlib import Path
from inference_engine import InferenceEngine, create_model
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Collect all images from all cameras
base_folder = Path("D:/test/scan_20250930_131312_toy_car/images")
all_images = []
for cam_folder in sorted(base_folder.glob("cam_*")):
    images = [f for f in cam_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    all_images.extend(images)

print(f"\nFound {len(all_images)} total images across all cameras")

# Create temporary folder for test output
output_base = Path("D:/test/large_batch_test")
output_base.mkdir(exist_ok=True)

# Test 1: Single GPU mode
print("\n" + "="*60)
print("Test 1: Single GPU Mode")
print("="*60)

model = create_model("mvanet")
model.load(Path("models/MVANet.pth"), torch.device("cuda:0"))
model.optimize_for_inference(torch.device("cuda:0"))

engine_single = InferenceEngine(
    model=model,
    device=torch.device("cuda:0"),
    chunk_size=20,
)

# Create combined input folder
combined_folder = output_base / "input"
combined_folder.mkdir(exist_ok=True)
for i, img_path in enumerate(all_images):
    import shutil
    shutil.copy(img_path, combined_folder / f"img_{i:04d}{img_path.suffix}")

start = time.time()
result_single = engine_single.process_folder(
    folder_path=combined_folder,
    output_folders={
        "mask": output_base / "single_gpu_masks",
        "overlay": output_base / "single_gpu_overlays",
    }
)
time_single = time.time() - start

print(f"\nSingle GPU Results:")
print(f"  Processed: {result_single['processed']} images")
print(f"  Time: {time_single:.2f}s")
print(f"  Speed: {time_single/result_single['processed']*1000:.1f}ms/img")

# Test 2: Multi-GPU mode with 1 GPU
print("\n" + "="*60)
print("Test 2: Multi-GPU Mode (1 GPU, persistent workers)")
print("="*60)

def model_factory(device):
    m = create_model("mvanet")
    m.load(Path("models/MVANet.pth"), device)
    m.optimize_for_inference(device)
    return m

engine_multi = InferenceEngine(
    devices=[torch.device("cuda:0")],
    model_factory=model_factory,
    chunk_size=20,
)

start = time.time()
result_multi = engine_multi.process_folder(
    folder_path=combined_folder,
    output_folders={
        "mask": output_base / "multi_gpu_masks",
        "overlay": output_base / "multi_gpu_overlays",
    }
)
time_multi = time.time() - start

print(f"\nMulti-GPU (1 GPU) Results:")
print(f"  Processed: {result_multi['processed']} images")
print(f"  Time: {time_multi:.2f}s")
print(f"  Speed: {time_multi/result_multi['processed']*1000:.1f}ms/img")

# Comparison
print("\n" + "="*60)
print("Comparison")
print("="*60)
overhead = time_multi - time_single
overhead_pct = (overhead / time_single) * 100
print(f"Single GPU:         {time_single:.2f}s ({time_single/result_single['processed']*1000:.1f}ms/img)")
print(f"Multi-GPU (1 GPU):  {time_multi:.2f}s ({time_multi/result_multi['processed']*1000:.1f}ms/img)")
print(f"Overhead:           {overhead:.2f}s ({overhead_pct:.1f}%)")
print(f"")

# Cleanup
import shutil
shutil.rmtree(output_base)
print("Cleaned up test output")
