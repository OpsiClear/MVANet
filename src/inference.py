import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F
import ttach as tta
from torch.amp import autocast
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging
import time
import os
from functools import lru_cache
import threading
import io
from .MVANet import inf_MVANet

# Configuration Constants
MODEL_PATH = Path(__file__).parent.parent / "models" / "MVANet.pth"
MODEL_IMAGE_SIZE = (1024, 1024)
NUM_WORKERS = min(8, os.cpu_count() or 1)  # Optimize thread count
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file operations
SKIP_KEYWORDS = ["montage"]  # Keywords in filename to skip processing

# Thread-local storage for per-thread transform objects
thread_local = threading.local()

# Device and CUDA Setup
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Core Utility Functions
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_tta_transforms():
    """Setup test-time augmentation transforms"""
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ]
    )


# Image Processing Functions
def rescale_to(target: torch.Tensor, scale_factor: float = 2, interpolation="nearest"):
    """Rescales tensor by a factor"""
    return F.interpolate(target, scale_factor=scale_factor, mode=interpolation)


def resize_as(target: torch.Tensor, source: torch.Tensor, interpolation="bilinear"):
    """Resizes x to match y's dimensions"""
    return F.interpolate(target, size=source.shape[-2:], mode=interpolation)


def rgb_loader_refiner(
    original_image: Image.Image, target_size: tuple[int, int]
) -> tuple[torch.Tensor, int, int, Image.Image]:
    """Handles initial image loading and scaling"""
    h, w = original_image.size
    image = original_image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
    return image.convert("RGB"), h, w, original_image


# Model Related Functions
def load_model(device: torch.device, model_path: Path | None = None) -> torch.nn.Module:
    """Loads and prepares the model for inference

    :param device: Device to load the model on
    :param model_path: Optional path to the model weights file (uses default if None)
    :return: Loaded model ready for inference
    """
    if model_path is None:
        model_path = MODEL_PATH

    # Initialize the inference model architecture
    try:
        model = inf_MVANet()
    except FileNotFoundError as e:
        if "swin_base_patch4_window12_384_22kto1k.pth" in str(e):
            # Handle missing Swin pretrained weights - they're included in the complete checkpoint
            logging.warning(
                "Swin pretrained weights file not found locally. Using weights from complete checkpoint."
            )
            # Temporarily patch the SwinB function to skip pretrained loading
            from . import SwinTransformer as swin_module

            original_swinb = swin_module.SwinB

            def patched_swinb(pretrained=True):
                from .SwinTransformer import SwinTransformer

                model = SwinTransformer(
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    window_size=12,
                )
                # Skip pretrained loading - weights will come from complete checkpoint
                return model

            # Temporarily replace SwinB
            swin_module.SwinB = patched_swinb
            try:
                model = inf_MVANet()
            finally:
                # Restore original SwinB
                swin_module.SwinB = original_swinb
        else:
            raise e

    # Load the state dict from the saved model
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        # Assume checkpoint is the model itself or state_dict
        if hasattr(checkpoint, "load_state_dict"):
            return checkpoint.to(device).eval()
        else:
            state_dict = checkpoint

    # Load weights into the model
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=None)
def get_transforms():
    """Cached transform pipeline"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_thread_transform():
    """Get thread-local transform instance"""
    if not hasattr(thread_local, "transform"):
        thread_local.transform = get_transforms()
    return thread_local.transform


def preprocess_image(
    image_path: Path, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Optimized image preprocessing"""
    with io.open(str(image_path), "rb") as f:
        # Use buffer for faster reading
        img_buffer = io.BytesIO(f.read())
        with Image.open(img_buffer) as image:
            original_size = image.size
            # Use BILINEAR for faster resizing
            resized_image = image.convert("RGB").resize(
                MODEL_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
            )
            # Use thread-local transform
            img_tensor = get_thread_transform()(resized_image).unsqueeze(0)

            # Batch transfer to GPU
            return img_tensor.to(device, non_blocking=True), original_size


# Post-processing Functions
def postprocess_mask(
    mask_tensor: torch.Tensor, original_size: tuple[int, int]
) -> Image.Image:
    """Post-processes the mask with sigmoid activation and normalization"""
    mask_tensor = torch.sigmoid(mask_tensor)
    mask_tensor = mask_tensor.cpu()
    mask_resized = torch.squeeze(
        F.interpolate(
            mask_tensor, size=(original_size[1], original_size[0]), mode="bilinear"
        ),
        0,
    )
    mask_np = (mask_resized.squeeze() * 255).cpu().data.numpy().astype(np.uint8)
    return Image.fromarray(mask_np)


def create_overlay(original_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Creates an overlay of the mask on the original image"""
    if original_image.size != mask.size:
        mask = mask.resize(original_image.size, Image.Resampling.LANCZOS)
    if original_image.mode != "RGBA":
        original_image = original_image.convert("RGBA")
    if mask.mode != "L":
        mask = mask.convert("L")
    r, g, b, _ = original_image.split()
    return Image.merge("RGBA", (r, g, b, mask))


# File Operations
def save_image_files(
    mask: Image.Image,
    original_image: Image.Image,
    overlay_path: Path,
):
    """Save overlay image with buffered writes"""
    overlay = create_overlay(original_image, mask)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    with open(str(overlay_path), "wb") as f:
        f.write(buffer.getvalue())


# Main Processing Functions
def infer_image(
    image_path: Path, model: torch.nn.Module, device: torch.device, use_tta: bool = True
) -> Image.Image:
    """Enhanced inference with optional test-time augmentation"""
    img_tensor, original_size = preprocess_image(image_path, device)

    with torch.no_grad():
        if use_tta:
            tta_transforms = setup_tta_transforms()
            masks = []
            for transformer in tta_transforms:
                aug_img = transformer.augment_image(img_tensor)
                mask = model(aug_img)
                deaug_mask = transformer.deaugment_mask(mask)
                masks.append(deaug_mask)
            mask_tensor = torch.mean(torch.stack(masks, dim=0), dim=0)
        else:
            mask_tensor = model(img_tensor)

    return postprocess_mask(mask_tensor, original_size)


def process_folder_recursive(
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    use_tta: bool = True,
):
    """Process all images in folder and its subfolders recursively"""
    # Create overlay output folder at parent level
    overlay_output_folder = folder_path.parent / (folder_path.name + "_overlay")
    overlay_output_folder.mkdir(exist_ok=True)

    def process_folder_internal(current_folder: Path, relative_path: Path = Path()):
        # Get all subfolders
        subfolders = [f for f in current_folder.iterdir() if f.is_dir()]

        if subfolders:
            # If subfolders exist, process each subfolder
            for subfolder in subfolders:
                new_relative_path = relative_path / subfolder.name
                process_folder_internal(subfolder, new_relative_path)

        # Process current folder's images
        # Create corresponding subfolder in overlay output directory
        current_overlay_folder = overlay_output_folder / relative_path
        current_overlay_folder.mkdir(exist_ok=True, parents=True)

        # Process folder with overlay output location
        process_folder(
            current_folder,
            model,
            device,
            use_tta=use_tta,
            overlay_folder=current_overlay_folder,
        )

    process_folder_internal(folder_path)


def process_folder(
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    use_tta: bool = True,
    overlay_folder: Path | None = None,
):
    """Process images with optimized CPU and I/O operations"""
    start_time = time.time()
    logging.info(f"Starting processing of folder: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist")

    image_files = [
        file_path
        for file_path in folder_path.iterdir()
        if file_path.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not image_files:
        logging.warning(f"No supported images found in {folder_path}")
        return

    # Filter out files with skip keywords in filename
    filtered_image_files = []
    keyword_skipped_files = []

    for file_path in image_files:
        filename_lower = file_path.name.lower()
        should_skip = any(
            keyword.lower() in filename_lower for keyword in SKIP_KEYWORDS
        )

        if should_skip:
            keyword_skipped_files.append(file_path)
            logging.info(f"Skipping {file_path.name} - contains skip keyword")
        else:
            filtered_image_files.append(file_path)

    # Log keyword filtering summary
    if keyword_skipped_files:
        logging.info(
            f"Filtered out {len(keyword_skipped_files)} files containing skip keywords: {SKIP_KEYWORDS}"
        )

    # Update image_files to use filtered list
    image_files = filtered_image_files

    if not image_files:
        logging.warning(
            "No images to process after filtering (all contained skip keywords)"
        )
        return

    logging.info(f"Found {len(image_files)} images to process after filtering")

    # Use provided overlay folder or create default one
    if overlay_folder is None:
        overlay_folder = folder_path.parent / (folder_path.name + "_overlay")
    overlay_folder.mkdir(exist_ok=True)

    # Dry run: Filter out images that already have processed outputs
    files_to_process = []
    skipped_files = []

    for img_path in image_files:
        overlay_path = overlay_folder / (img_path.stem + ".png")
        if overlay_path.exists():
            skipped_files.append(img_path)
            logging.info(
                f"Skipping {img_path.name} - output already exists at {overlay_path}"
            )
        else:
            files_to_process.append(img_path)

    # Log summary of dry run
    total_files = len(image_files)
    files_to_process_count = len(files_to_process)
    skipped_count = len(skipped_files)

    logging.info(f"Dry run complete: {total_files} total images found")
    logging.info(f"  - {files_to_process_count} images to process")
    logging.info(f"  - {skipped_count} images already processed (skipped)")

    if not files_to_process:
        logging.info("All images have already been processed. Nothing to do.")
        return

    # Pre-compile TTA transforms
    if use_tta:
        tta_transforms = setup_tta_transforms()

    # Process images in chunks for better memory management
    chunk_size = 10  # Adjust based on available memory
    for i in range(0, len(files_to_process), chunk_size):
        chunk = files_to_process[i : i + chunk_size]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit preprocessing tasks
            preprocess_futures = {
                executor.submit(preprocess_image, img_path, device): img_path
                for img_path in chunk
            }

            save_futures = []

            for future in as_completed(preprocess_futures):
                img_path = preprocess_futures[future]
                try:
                    img_start_time = time.time()
                    img_tensor, original_size = future.result()

                    # GPU Processing
                    with autocast(enabled=True, device_type="cuda"), torch.no_grad():
                        if use_tta:
                            masks = []
                            for transformer in tta_transforms:
                                aug_img = transformer.augment_image(img_tensor)
                                with autocast(device_type="cuda"):
                                    mask = model(aug_img)
                                deaug_mask = transformer.deaugment_mask(mask)
                                masks.append(deaug_mask)
                            mask_tensor = torch.mean(torch.stack(masks, dim=0), dim=0)
                        else:
                            mask_tensor = model(img_tensor)

                    # CPU Processing
                    mask = postprocess_mask(mask_tensor.float(), original_size)

                    # Load original image for overlay
                    with Image.open(img_path) as original_image:
                        original_image = original_image.convert("RGB")
                        overlay_path = overlay_folder / (img_path.stem + ".png")

                    # Submit save task
                    save_futures.append(
                        executor.submit(
                            save_image_files,
                            mask,
                            original_image,
                            overlay_path,
                        )
                    )

                    img_process_time = time.time() - img_start_time
                    logging.info(
                        f"Processed {img_path.name} in {img_process_time:.2f} seconds"
                    )

                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")
                    continue

            # Wait for all saves to complete
            for future in save_futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error during save operation: {e}")

        # Clear CUDA cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    logging.info(
        f"Completed processing {files_to_process_count} new images in {total_time:.2f} seconds"
    )
    if files_to_process_count > 0:
        logging.info(
            f"Average time per image: {total_time/files_to_process_count:.2f} seconds"
        )


# CLI Setup
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Image segmentation inference script")
    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--use_tta", action="store_true", help="Use test-time augmentation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (e.g., cuda:0, cpu)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


# Main Entry Point
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting inference script")

    args = parse_args()
    torch.cuda.empty_cache()
    device = torch.device(args.device)

    # Configure logging based on argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Loading model from {MODEL_PATH}")
    model = load_model(device)
    logging.info(f"Model (inf_MVANet) loaded successfully on {device}")

    # Use the recursive processing function (always saves overlays)
    process_folder_recursive(
        args.input_folder,
        model,
        device,
        use_tta=args.use_tta,
    )

    total_time = time.time() - start_time
    logging.info(f"Total script execution time: {total_time:.2f} seconds")
