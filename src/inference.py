import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F
import ttach as tta
from torch.amp.autocast_mode import autocast
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
NUM_WORKERS = min(8, os.cpu_count() or 1)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_KEYWORDS = ["montage"]

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


# Model Related Functions
def load_model(device: torch.device, model_path: Path | None = None) -> torch.nn.Module:
    """Loads and prepares the model for inference"""
    if model_path is None:
        model_path = MODEL_PATH

    model = inf_MVANet()
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

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
        img_buffer = io.BytesIO(f.read())
        with Image.open(img_buffer) as image:
            original_size = image.size
            resized_image = image.convert("RGB").resize(
                MODEL_IMAGE_SIZE, resample=Image.Resampling.BILINEAR
            )
            img_tensor = get_thread_transform()(resized_image).unsqueeze(0)
            return img_tensor.to(device, non_blocking=True), original_size


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


def find_images_folders_recursive(root_path: Path) -> list[Path]:
    """Find all folders named 'images' recursively under root_path"""
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


def save_image_files(
    mask: Image.Image,
    original_image: Image.Image,
    overlay_path: Path,
    mask_path: Path | None = None,
):
    """Save overlay image and optionally pure mask with buffered writes"""
    # Create RGBA overlay
    overlay = create_overlay(original_image, mask)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    with open(str(overlay_path), "wb") as f:
        f.write(buffer.getvalue())
    
    # Save pure alpha mask if path provided
    if mask_path is not None:
        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format="PNG", optimize=True)
        with open(str(mask_path), "wb") as f:
            f.write(mask_buffer.getvalue())


def run_inference(
    img_tensor: torch.Tensor, 
    model: torch.nn.Module, 
    tta_model: torch.nn.Module | None = None
) -> torch.Tensor:
    """Consolidated inference function"""
    with torch.no_grad():
        if tta_model is not None:
            return tta_model(img_tensor)
        else:
            return model(img_tensor)


def infer_image(
    image_path: Path, model: torch.nn.Module, device: torch.device, use_tta: bool = True
) -> Image.Image:
    """Enhanced inference with optional test-time augmentation"""
    img_tensor, original_size = preprocess_image(image_path, device)
    
    # Setup TTA model once if needed
    tta_model = None
    if use_tta:
        tta_wrapper = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model, tta_wrapper)
    
    mask_tensor = run_inference(img_tensor, model, tta_model)
    return postprocess_mask(mask_tensor, original_size)


def process_folder_recursive(
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    use_tta: bool = True,
):
    """Process all images in folders named 'images' recursively"""
    images_folders = find_images_folders_recursive(folder_path)
    
    if not images_folders:
        logging.warning(f"No folders named 'images' found under {folder_path}")
        return
    
    logging.info(f"Found {len(images_folders)} 'images' folders to process")
    
    for i, images_folder in enumerate(images_folders, 1):
        logging.info(f"Processing images folder {i}/{len(images_folders)}: {images_folder}")
        
        overlays_path, masks_path = get_output_paths_for_images_folder(images_folder)
        overlays_path.mkdir(exist_ok=True, parents=True)
        masks_path.mkdir(exist_ok=True, parents=True)
        
        process_images_recursively(
            images_folder,
            model,
            device,
            overlays_path,
            masks_path,
            use_tta=use_tta,
        )


def process_images_recursively(
    current_folder: Path,
    model: torch.nn.Module,
    device: torch.device,
    base_overlay_path: Path,
    base_mask_path: Path,
    relative_path: Path = Path(),
    use_tta: bool = True,
):
    """Recursively process images in current folder and all subdirectories"""
    current_overlay_path = base_overlay_path / relative_path
    current_mask_path = base_mask_path / relative_path
    
    current_overlay_path.mkdir(exist_ok=True, parents=True)
    current_mask_path.mkdir(exist_ok=True, parents=True)
    
    process_folder(
        current_folder,
        model,
        device,
        use_tta=use_tta,
        overlay_folder=current_overlay_path,
        mask_folder=current_mask_path,
    )
    
    for item in current_folder.iterdir():
        if item.is_dir():
            new_relative_path = relative_path / item.name
            process_images_recursively(
                item,
                model,
                device,
                base_overlay_path,
                base_mask_path,
                new_relative_path,
                use_tta,
            )


def process_folder(
    folder_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    use_tta: bool = True,
    overlay_folder: Path | None = None,
    mask_folder: Path | None = None,
):
    """Process images with optimized CPU and I/O operations"""
    start_time = time.time()

    if not folder_path.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist")

    # Find image files efficiently
    try:
        image_files = [
            file_path for file_path in folder_path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS
        ]
    except Exception as e:
        logging.error(f"Error listing directory contents: {e}")
        return

    if not image_files:
        return

    # Filter out files with skip keywords
    filtered_image_files = []
    keyword_skipped_count = 0

    for file_path in image_files:
        filename_lower = file_path.name.lower()
        should_skip = any(
            keyword.lower() in filename_lower for keyword in SKIP_KEYWORDS
        )

        if should_skip:
            keyword_skipped_count += 1
        else:
            filtered_image_files.append(file_path)

    image_files = filtered_image_files

    if not image_files:
        if keyword_skipped_count > 0:
            logging.info(f"Skipped {keyword_skipped_count} files with keywords: {SKIP_KEYWORDS}")
        return

    if overlay_folder is None or mask_folder is None:
        raise ValueError("Both overlay_folder and mask_folder must be provided")
    
    overlay_folder.mkdir(exist_ok=True)
    mask_folder.mkdir(exist_ok=True)

    # Check which files need processing
    files_to_process = []
    skipped_count = 0

    for img_path in image_files:
        overlay_path = overlay_folder / (img_path.stem + ".png")
        if overlay_path.exists():
            skipped_count += 1
        else:
            files_to_process.append(img_path)

    # Log processing summary
    total_files = len(image_files)
    files_to_process_count = len(files_to_process)
    
    if files_to_process_count == 0:
        if total_files > 0:
            logging.info(f"All {total_files} images in {folder_path.name} already processed")
        return

    if skipped_count > 0:
        logging.info(f"Processing {files_to_process_count}/{total_files} images in {folder_path.name} ({skipped_count} already done)")
    else:
        logging.info(f"Processing {files_to_process_count} images in {folder_path.name}")

    # Setup TTA model once for the entire batch (MAJOR OPTIMIZATION)
    tta_model = None
    if use_tta:
        tta_wrapper = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        tta_model = tta.SegmentationTTAWrapper(model, tta_wrapper)

    # Process images in chunks for better memory management
    chunk_size = 10
    processed_count = 0
    
    for i in range(0, len(files_to_process), chunk_size):
        chunk = files_to_process[i : i + chunk_size]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            preprocess_futures = {
                executor.submit(preprocess_image, img_path, device): img_path
                for img_path in chunk
            }

            save_futures = []

            for future in as_completed(preprocess_futures):
                img_path = preprocess_futures[future]
                try:
                    img_tensor, original_size = future.result()

                    # GPU Processing - use pre-created TTA model
                    with autocast(enabled=True, device_type="cuda"):
                        mask_tensor = run_inference(img_tensor, model, tta_model)

                    # CPU Processing
                    mask = postprocess_mask(mask_tensor.float(), original_size)

                    # Load original image for overlay
                    with Image.open(img_path) as original_image:
                        original_image = original_image.convert("RGB")
                        overlay_path = overlay_folder / (img_path.stem + ".png")
                        mask_path = mask_folder / (img_path.stem + ".png")

                    # Submit save task
                    save_futures.append(
                        executor.submit(
                            save_image_files,
                            mask,
                            original_image,
                            overlay_path,
                            mask_path,
                        )
                    )

                    processed_count += 1
                    
                    # Progress logging for large batches
                    if files_to_process_count > 20 and processed_count % 10 == 0:
                        logging.info(f"Progress: {processed_count}/{files_to_process_count} images processed")

                except Exception as e:
                    logging.error(f"Error processing {img_path.name}: {e}")
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
    avg_time = total_time / files_to_process_count if files_to_process_count > 0 else 0
    
    if files_to_process_count > 0:
        logging.info(f"Completed {files_to_process_count} images in {total_time:.1f}s (avg: {avg_time:.2f}s/image)")


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
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

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
    logging.info(f"Model loaded successfully on {device}")

    process_folder_recursive(
        args.input_folder,
        model,
        device,
        use_tta=args.use_tta,
    )

    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.1f} seconds")
