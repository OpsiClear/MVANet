"""Test job scheduling logic with 1 GPU"""

import torch
from pathlib import Path
from inference_engine import InferenceEngine, create_model
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Test with 1 GPU using multi-GPU code path
devices = [torch.device("cuda:0")]

def model_factory(device):
    model = create_model("mvanet")
    model.load(Path("models/MVANet.pth"), device)
    model.optimize_for_inference(device)
    return model

# Create engine in multi-GPU mode with just 1 GPU
# This tests the scheduling logic
engine = InferenceEngine(
    devices=devices,
    model_factory=model_factory,
    chunk_size=5,  # Small chunks to test multiple batches
)

# Process folder
result = engine.process_folder(
    folder_path=Path("D:/test/scan_20250930_131312_toy_car/images/cam_1"),
    output_folders={
        "mask": Path("D:/test/scan_20250930_131312_toy_car/images/cam_1/masks"),
        "overlay": Path("D:/test/scan_20250930_131312_toy_car/images/cam_1/overlays"),
    },
)

print(f"\n✅ Test passed!")
print(f"Results: {result}")
print(f"Processed: {result['processed']}")
print(f"Total: {result['total']}")
print(f"Time: {result['time']:.2f}s")
