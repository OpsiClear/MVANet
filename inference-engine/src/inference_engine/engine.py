"""
Generic Inference Engine for Segmentation Models

Supports any model implementing the SegmentationModel interface.
Uses OpenCV for fast image I/O.
"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from torch.amp.autocast_mode import autocast
from concurrent.futures import ThreadPoolExecutor, as_completed
import ttach as tta
import time
import os

from .base import SegmentationModel

# Configuration
NUM_WORKERS = min(8, os.cpu_count() or 1)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_KEYWORDS = ["montage"]

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Generic inference engine for segmentation models"""

    def __init__(
        self,
        model: SegmentationModel,
        device: torch.device,
        use_fp16: bool = True,
        chunk_size: int = 20,
    ):
        """
        Initialize inference engine

        Args:
            model: Segmentation model implementing SegmentationModel interface
            device: torch device
            use_fp16: Use FP16 mixed precision
            chunk_size: Number of images to process per chunk
        """
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.chunk_size = chunk_size

    def infer_image(self, image: np.ndarray, use_tta: bool = False) -> np.ndarray:
        """
        Run inference on a single image

        Args:
            image: OpenCV image (numpy array, BGR format)
            use_tta: Use test-time augmentation

        Returns:
            Mask as numpy array (grayscale)
        """
        # Convert BGR to RGB for preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor, metadata = self.model.preprocess(image_rgb)

        # Setup TTA if requested
        if use_tta and self.model.supports_tta:
            tta_wrapper = tta.Compose([tta.HorizontalFlip()])
            tta_model = tta.SegmentationTTAWrapper(
                lambda x: self.model.forward(x), tta_wrapper
            )

            with torch.no_grad():
                autocast_dtype = torch.float16 if self.use_fp16 else torch.float32
                with autocast(enabled=True, device_type="cuda", dtype=autocast_dtype):
                    output = tta_model(img_tensor)
        else:
            with torch.no_grad():
                autocast_dtype = torch.float16 if self.use_fp16 else torch.float32
                with autocast(enabled=True, device_type="cuda", dtype=autocast_dtype):
                    output = self.model.forward(img_tensor)

        # Postprocess
        mask = self.model.postprocess(output.float(), metadata)
        return mask

    def process_folder(
        self,
        folder_path: Path,
        overlay_folder: Path,
        mask_folder: Path,
        use_tta: bool = False,
    ) -> dict:
        """
        Process all images in a folder

        Args:
            folder_path: Input folder path
            overlay_folder: Output folder for overlays
            mask_folder: Output folder for masks
            use_tta: Use test-time augmentation

        Returns:
            dict with processing statistics
        """
        start_time = time.time()

        if not folder_path.is_dir():
            raise ValueError(f"Folder {folder_path} does not exist")

        # Find image files
        try:
            image_files = [
                file_path
                for file_path in folder_path.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS
            ]
        except Exception as e:
            logger.error(f"Error listing directory contents: {e}")
            return {"error": str(e)}

        if not image_files:
            return {"processed": 0, "skipped": 0, "total": 0}

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
                logger.info(
                    f"Skipped {keyword_skipped_count} files with keywords: {SKIP_KEYWORDS}"
                )
            return {"processed": 0, "skipped": keyword_skipped_count, "total": 0}

        # Create output folders
        overlay_folder.mkdir(exist_ok=True, parents=True)
        mask_folder.mkdir(exist_ok=True, parents=True)

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
                logger.info(
                    f"All {total_files} images in {folder_path.name} already processed"
                )
            return {"processed": 0, "skipped": skipped_count, "total": total_files}

        if skipped_count > 0:
            logger.info(
                f"Processing {files_to_process_count}/{total_files} images in {folder_path.name} ({skipped_count} already done)"
            )
        else:
            logger.info(f"Processing {files_to_process_count} images in {folder_path.name}")

        # Process images in chunks
        processed_count = 0

        for i in range(0, len(files_to_process), self.chunk_size):
            chunk = files_to_process[i : i + self.chunk_size]

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Preprocess in parallel
                preprocess_futures = {
                    executor.submit(self._load_and_preprocess, img_path): img_path
                    for img_path in chunk
                }

                save_futures = []

                for future in as_completed(preprocess_futures):
                    img_path = preprocess_futures[future]
                    try:
                        image, img_tensor, metadata = future.result()

                        # GPU inference
                        autocast_dtype = torch.float16 if self.use_fp16 else torch.float32
                        with autocast(
                            enabled=True, device_type="cuda", dtype=autocast_dtype
                        ):
                            if use_tta and self.model.supports_tta:
                                tta_wrapper = tta.Compose([tta.HorizontalFlip()])
                                tta_model = tta.SegmentationTTAWrapper(
                                    lambda x: self.model.forward(x), tta_wrapper
                                )
                                output = tta_model(img_tensor)
                            else:
                                output = self.model.forward(img_tensor)

                        # Postprocess
                        mask = self.model.postprocess(output.float(), metadata)

                        # Save paths
                        overlay_path = overlay_folder / (img_path.stem + ".png")
                        mask_path = mask_folder / (img_path.name + ".png")

                        # Submit save task
                        save_futures.append(
                            executor.submit(
                                self._save_outputs, mask, image, overlay_path, mask_path
                            )
                        )

                        processed_count += 1

                        # Progress logging
                        if files_to_process_count > 20 and processed_count % 10 == 0:
                            logger.info(
                                f"Progress: {processed_count}/{files_to_process_count} images processed"
                            )

                    except Exception as e:
                        logger.error(f"Error processing {img_path.name}: {e}")
                        continue

                # Wait for saves
                for future in save_futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error during save operation: {e}")

            # Clear CUDA cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        avg_time = total_time / files_to_process_count if files_to_process_count > 0 else 0

        if files_to_process_count > 0:
            logger.info(
                f"Completed {files_to_process_count} images in {total_time:.1f}s (avg: {avg_time:.2f}s/image)"
            )

        return {
            "processed": processed_count,
            "skipped": skipped_count,
            "total": total_files,
            "time": total_time,
        }

    def _load_and_preprocess(self, image_path: Path) -> tuple[np.ndarray, torch.Tensor, dict]:
        """Load and preprocess image using OpenCV"""
        # Load image with OpenCV (BGR format)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to RGB for model preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor, metadata = self.model.preprocess(image_rgb)

        return image, img_tensor, metadata

    def _save_outputs(
        self, mask: np.ndarray, original_image: np.ndarray, overlay_path: Path, mask_path: Path
    ):
        """Save overlay and mask using OpenCV"""
        # Create RGBA overlay
        overlay = self._create_overlay(original_image, mask)

        # Save with OpenCV (fast PNG compression)
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Save pure mask
        cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    @staticmethod
    def _create_overlay(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create RGBA overlay from image and mask

        Args:
            original_image: BGR image
            mask: Grayscale mask

        Returns:
            BGRA overlay
        """
        # Resize mask if needed
        if original_image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )

        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create BGRA overlay (OpenCV uses BGR, not RGB)
        overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        overlay[:, :, 3] = mask  # Set alpha channel to mask

        return overlay
