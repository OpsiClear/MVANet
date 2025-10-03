"""CLI entry point for generic segmentation inference"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tyro
import torch
import logging
import time

from src.models.mvanet import MVANetModel
from src.engine import InferenceEngine


@dataclass
class InferenceConfig:
    """Image segmentation inference configuration"""

    input_folder: Path
    """Path to folder containing input images"""

    model_path: Path = Path("models/MVANet.pth")
    """Path to model weights"""

    model_type: Literal["mvanet"] = "mvanet"
    """Model type to use"""

    use_tta: bool = False
    """Use test-time augmentation"""

    use_fp16: bool = True
    """Use FP16 mixed precision"""

    device: str = "cuda:0"
    """Device to use for inference (e.g., cuda:0, cpu)"""

    chunk_size: int = 20
    """Number of images to process per chunk"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Set the logging level"""


def find_images_folders_recursive(root_path: Path) -> list[Path]:
    """Find all folders named 'images' recursively"""
    images_folders = []
    for item in root_path.rglob("*"):
        if item.is_dir() and item.name.lower() == "images":
            images_folders.append(item)
    return images_folders


def get_output_paths_for_images_folder(images_folder: Path) -> tuple[Path, Path]:
    """Get overlay and mask output paths for an images folder"""
    parent = images_folder.parent
    overlays_path = parent / "overlays"
    masks_path = parent / "masks"
    return overlays_path, masks_path


def main():
    start_time = time.time()

    config = tyro.cli(InferenceConfig)
    torch.cuda.empty_cache()
    device = torch.device(config.device)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create model based on type
    if config.model_type == "mvanet":
        model = MVANetModel(use_fp16=config.use_fp16)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Load model
    logging.info(f"Loading {model.name} model from {config.model_path}")
    model.load(config.model_path, device)
    model.optimize_for_inference(device)

    # Create inference engine
    engine = InferenceEngine(
        model=model,
        device=device,
        use_fp16=config.use_fp16,
        chunk_size=config.chunk_size,
    )

    # Find images folders
    images_folders = find_images_folders_recursive(config.input_folder)

    if not images_folders:
        logging.warning(f"No folders named 'images' found under {config.input_folder}")
        return

    logging.info(f"Found {len(images_folders)} 'images' folders to process")

    # Process each images folder
    for i, images_folder in enumerate(images_folders, 1):
        logging.info(
            f"Processing images folder {i}/{len(images_folders)}: {images_folder}"
        )

        overlays_path, masks_path = get_output_paths_for_images_folder(images_folder)

        # Process folder
        engine.process_folder(
            folder_path=images_folder,
            overlay_folder=overlays_path,
            mask_folder=masks_path,
            use_tta=config.use_tta,
        )

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
