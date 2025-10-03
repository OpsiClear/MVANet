"""
Generic Inference Engine for Segmentation Models

Supports any model implementing the SegmentationModel interface.
Uses OpenCV for fast image I/O.
Supports multiple outputs (masks, depth maps, normal maps, etc.)
Supports multi-GPU inference for parallel processing.
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
from typing import Callable

from .base import SegmentationModel
from .multi_gpu import MultiGPUInferenceEngine

# Configuration
NUM_WORKERS = min(8, os.cpu_count() or 1)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}
SKIP_KEYWORDS = ["montage"]

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Generic inference engine for segmentation models"""

    def __init__(
        self,
        model: SegmentationModel | None = None,
        device: torch.device | None = None,
        devices: list[torch.device] | None = None,
        model_factory: Callable[[torch.device], SegmentationModel] | None = None,
        use_fp16: bool = True,
        chunk_size: int = 20,
    ):
        """
        Initialize inference engine

        Single GPU mode:
            model: SegmentationModel instance
            device: torch device

        Multi-GPU mode:
            devices: List of torch devices
            model_factory: Function that creates model for each device

        Args:
            model: Segmentation model (single GPU)
            device: torch device (single GPU)
            devices: List of devices (multi-GPU)
            model_factory: Factory function for multi-GPU
            use_fp16: Use FP16 mixed precision
            chunk_size: Number of images to process per chunk
        """
        self.use_fp16 = use_fp16
        self.chunk_size = chunk_size

        # Multi-GPU mode (or devices + model_factory mode)
        if devices and model_factory:
            self.multi_gpu = len(devices) > 1
            self.devices = devices
            self.device = devices[0]  # Primary device for preprocessing
            self.multi_gpu_engine = MultiGPUInferenceEngine(
                model_factory, devices, use_fp16
            )
            self.model = self.multi_gpu_engine.models[self.device]
            if self.multi_gpu:
                logger.info(f"Multi-GPU mode: Using {len(devices)} GPUs")
            else:
                logger.info(f"Multi-GPU engine with 1 GPU: {devices[0]}")
        # Single GPU mode
        elif model and device:
            self.multi_gpu = False
            self.model = model
            self.device = device
            self.devices = [device]
            self.multi_gpu_engine = None
            logger.info(f"Single GPU mode: Using {device}")
        else:
            raise ValueError(
                "Either provide (devices + model_factory) or (model + device)"
            )

    def infer_image(
        self, image: np.ndarray, use_tta: bool = False
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Run inference on a single image

        Args:
            image: OpenCV image (numpy array, BGR format)
            use_tta: Use test-time augmentation

        Returns:
            Single mask or dict of outputs
        """
        # Convert BGR to RGB for preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess (always to_device=True for single image inference)
        img_tensor, metadata = self.model.preprocess(image_rgb, to_device=True)

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

        # Postprocess (handle dict or single tensor)
        if isinstance(output, dict):
            output_float = {k: v.float() for k, v in output.items()}
        else:
            output_float = output.float()

        result = self.model.postprocess(output_float, metadata)
        return result

    def process_folder(
        self,
        folder_path: Path,
        output_folders: dict[str, Path] | None = None,
        overlay_folder: Path | None = None,
        mask_folder: Path | None = None,
        use_tta: bool = False,
        create_overlays: bool = True,
    ) -> dict:
        """
        Process all images in a folder

        Args:
            folder_path: Input folder path
            output_folders: Dict mapping output names to folder paths
                          e.g., {"mask": Path("masks"), "depth": Path("depths")}
            overlay_folder: (Deprecated) Use output_folders with "overlay" key
            mask_folder: (Deprecated) Use output_folders with "mask" key
            use_tta: Use test-time augmentation
            create_overlays: Create RGBA overlays for mask outputs

        Returns:
            dict with processing statistics
        """
        start_time = time.time()

        # Handle backward compatibility
        if output_folders is None:
            output_folders = {}
            if mask_folder:
                output_folders["mask"] = mask_folder
            if overlay_folder:
                output_folders["overlay"] = overlay_folder

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
        for folder in output_folders.values():
            folder.mkdir(exist_ok=True, parents=True)

        # Check which files need processing (based on first output)
        files_to_process = []
        skipped_count = 0
        first_output_key = list(output_folders.keys())[0] if output_folders else None

        if first_output_key and first_output_key != "overlay":
            check_folder = output_folders[first_output_key]
        elif "mask" in output_folders:
            check_folder = output_folders["mask"]
        else:
            check_folder = list(output_folders.values())[0] if output_folders else None

        for img_path in image_files:
            if check_folder:
                output_path = check_folder / (img_path.stem + ".png")
                if output_path.exists():
                    skipped_count += 1
                    continue
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

            # Multi-GPU mode: Parallel GPU processing
            if self.multi_gpu and self.multi_gpu_engine:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    # Preprocess in parallel
                    preprocess_futures = {
                        executor.submit(self._load_and_preprocess, img_path): img_path
                        for img_path in chunk
                    }

                    # Collect preprocessed data
                    image_data = []
                    for future in as_completed(preprocess_futures):
                        try:
                            image, img_tensor, metadata = future.result()
                            img_path = preprocess_futures[future]
                            image_data.append((img_path, image, img_tensor, metadata))
                        except Exception as e:
                            img_path = preprocess_futures[future]
                            logger.error(f"Error preprocessing {img_path.name}: {e}")

                # Process batch on multiple GPUs in parallel
                gpu_results = self.multi_gpu_engine.process_images(image_data, use_tta)

                # Save results
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    save_futures = []
                    for img_path, image, results_dict, error in gpu_results:
                        if error:
                            logger.error(f"GPU processing error for {img_path.name}: {error}")
                            continue

                        save_futures.append(
                            executor.submit(
                                self._save_outputs,
                                results_dict,
                                image,
                                img_path,
                                output_folders,
                                create_overlays,
                            )
                        )
                        processed_count += 1

                    # Wait for saves
                    for future in save_futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error during save operation: {e}")

                # Progress logging
                if files_to_process_count > 20 and processed_count % (self.chunk_size * len(self.devices)) == 0:
                    logger.info(
                        f"Progress: {processed_count}/{files_to_process_count} images processed"
                    )

            # Single GPU mode: Sequential processing
            else:
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
                            if isinstance(output, dict):
                                output_float = {k: v.float() for k, v in output.items()}
                            else:
                                output_float = output.float()

                            results = self.model.postprocess(output_float, metadata)

                            # Normalize results to dict
                            if isinstance(results, dict):
                                results_dict = results
                            else:
                                # Single output - use first output name from model
                                output_names = self.model.get_output_names()
                                results_dict = {output_names[0]: results}

                            # Submit save task
                            save_futures.append(
                                executor.submit(
                                    self._save_outputs,
                                    results_dict,
                                    image,
                                    img_path,
                                    output_folders,
                                    create_overlays,
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

    def _load_and_preprocess(
        self, image_path: Path
    ) -> tuple[np.ndarray, torch.Tensor, dict]:
        """Load and preprocess image using OpenCV"""
        # Load image with OpenCV (BGR format)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to RGB for model preprocessing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Multi-GPU mode: keep tensor in CPU pinned memory to avoid GPU-to-GPU transfers
        # Single GPU mode: move tensor to device immediately
        to_device = not (self.multi_gpu and self.multi_gpu_engine)
        img_tensor, metadata = self.model.preprocess(image_rgb, to_device=to_device)

        return image, img_tensor, metadata

    def _save_outputs(
        self,
        outputs: dict[str, np.ndarray],
        original_image: np.ndarray,
        img_path: Path,
        output_folders: dict[str, Path],
        create_overlays: bool,
    ):
        """Save all outputs to their respective folders"""

        for output_name, output_array in outputs.items():
            if output_name not in output_folders:
                continue

            folder = output_folders[output_name]
            output_path = folder / (img_path.name + ".png")

            # Save output
            cv2.imwrite(str(output_path), output_array, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Create overlay if requested and mask output exists
        if create_overlays and "overlay" in output_folders and "mask" in outputs:
            overlay = self._create_overlay(original_image, outputs["mask"])
            overlay_path = output_folders["overlay"] / (img_path.stem + ".png")
            cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Create BGRA overlay (OpenCV uses BGR, not RGB)
        overlay = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        overlay[:, :, 3] = mask  # Set alpha channel to mask

        return overlay
