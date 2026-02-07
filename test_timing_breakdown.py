"""Timing breakdown for unified batch processing."""

import time
import logging
from pathlib import Path
from oc_masker import InferenceEngine, create_model
from oc_masker.multi_gpu import parse_devices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup
input_folder = Path(r"D:\test\scan_20250928_115837_bella_mushroom")
device = "cuda:0"
batch_size = 8

devices = parse_devices(device)


def model_factory(dev):
    m = create_model(
        "pdfnet", trt_batch_size=batch_size, use_tensorrt=True, use_moge_tensorrt=True
    )
    m.load(None, dev)
    m.optimize_for_inference(dev)
    return m


engine = InferenceEngine(devices=devices, model_factory=model_factory, use_fp16=True)

# Time the collection phase
t0 = time.perf_counter()
all_images, skipped = engine._collect_all_images(
    current_folder=input_folder / "images",
    base_output_folders={"mask": input_folder / "masks"},
    relative_path=Path(),
    force_overwrite=True,
)
t_collect = time.perf_counter() - t0
logger.info(f"Collection: {t_collect:.3f}s for {len(all_images)} images")

# Time the processing phase
t0 = time.perf_counter()
processed = engine._process_unified_batch(all_images, use_tta=False, tta_merge_mode="none")
t_process = time.perf_counter() - t0
logger.info(f"Processing: {t_process:.2f}s for {processed} images")
logger.info(f"Throughput: {processed/t_process:.1f} img/s")
logger.info(f"Total: {t_collect + t_process:.2f}s")
