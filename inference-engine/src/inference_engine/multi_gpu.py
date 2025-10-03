"""Multi-GPU inference support"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from typing import Callable

from .base import SegmentationModel

logger = logging.getLogger(__name__)


class MultiGPUInferenceEngine:
    """
    Multi-GPU inference engine that distributes work across multiple GPUs

    Each GPU gets its own model instance and processes batches in parallel
    """

    def __init__(
        self,
        model_factory: Callable[[torch.device], SegmentationModel],
        devices: list[torch.device],
        use_fp16: bool = True,
    ):
        """
        Initialize multi-GPU engine

        Args:
            model_factory: Function that creates a model instance for a given device
            devices: List of torch devices to use
            use_fp16: Use FP16 mixed precision
        """
        self.devices = devices
        self.use_fp16 = use_fp16
        self.num_gpus = len(devices)

        # Create model instance for each GPU
        self.models = {}
        for device in devices:
            logger.info(f"Loading model on {device}")
            self.models[device] = model_factory(device)

    def _worker(
        self,
        device: torch.device,
        worker_id: int,
        work_queue: Queue,
        result_queue: Queue,
        use_tta: bool,
    ):
        """Worker thread that processes images on a specific GPU"""
        model = self.models[device]

        while True:
            item = work_queue.get()
            if item is None:  # Sentinel to stop worker
                work_queue.task_done()
                break

            img_path, image, img_tensor, metadata = item

            try:
                # Move tensor to this GPU
                img_tensor = img_tensor.to(device, non_blocking=True)

                # Inference
                from torch.amp.autocast_mode import autocast

                autocast_dtype = torch.float16 if self.use_fp16 else torch.float32

                with torch.no_grad():
                    with autocast(enabled=True, device_type="cuda", dtype=autocast_dtype):
                        if use_tta and model.supports_tta:
                            import ttach as tta

                            tta_wrapper = tta.Compose([tta.HorizontalFlip()])
                            tta_model = tta.SegmentationTTAWrapper(
                                lambda x: model.forward(x), tta_wrapper
                            )
                            output = tta_model(img_tensor)
                        else:
                            output = model.forward(img_tensor)

                # Postprocess
                if isinstance(output, dict):
                    output_float = {k: v.float() for k, v in output.items()}
                else:
                    output_float = output.float()

                results = model.postprocess(output_float, metadata)

                # Normalize to dict
                if isinstance(results, dict):
                    results_dict = results
                else:
                    output_names = model.get_output_names()
                    results_dict = {output_names[0]: results}

                # Put result in queue
                result_queue.put((img_path, image, results_dict, None))

            except Exception as e:
                logger.error(f"GPU {device} error processing {img_path.name}: {e}")
                result_queue.put((img_path, image, None, str(e)))

            finally:
                work_queue.task_done()

    def process_images(
        self,
        image_data: list[tuple[Path, np.ndarray, torch.Tensor, dict]],
        use_tta: bool = False,
    ) -> list[tuple[Path, np.ndarray, dict[str, np.ndarray] | None, str | None]]:
        """
        Process images in parallel across multiple GPUs

        Args:
            image_data: List of (img_path, image, img_tensor, metadata) tuples
            use_tta: Use test-time augmentation

        Returns:
            List of (img_path, image, results_dict, error) tuples
        """
        if not image_data:
            return []

        # Create fresh queues for this batch
        work_queue = Queue()
        result_queue = Queue()

        # Add all images to work queue
        for img_path, image, img_tensor, metadata in image_data:
            work_queue.put((img_path, image, img_tensor, metadata))

        # Add sentinel values to stop workers (one per GPU)
        for _ in self.devices:
            work_queue.put(None)

        # Start worker threads
        workers = []
        for i, device in enumerate(self.devices):
            worker = threading.Thread(
                target=self._worker,
                args=(device, i, work_queue, result_queue, use_tta),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        # Wait for all work to complete
        work_queue.join()

        # Collect exact number of results (one per input image)
        num_images = len(image_data)
        results = []
        for _ in range(num_images):
            results.append(result_queue.get())

        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=1.0)

        return results

    def get_device_for_preprocessing(self) -> torch.device:
        """Get device for preprocessing (use first GPU)"""
        return self.devices[0]


def detect_gpus() -> list[torch.device]:
    """
    Detect available GPUs

    Returns:
        List of available GPU devices
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return [torch.device("cpu")]

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s)")

    devices = []
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(device)
        logger.info(
            f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)"
        )
        devices.append(device)

    return devices


def parse_devices(device_spec: str | list[str]) -> list[torch.device]:
    """
    Parse device specification

    Args:
        device_spec: String like "cuda:0,cuda:1" or "auto" or list ["cuda:0", "cuda:1"]

    Returns:
        List of torch devices
    """
    if isinstance(device_spec, list):
        return [torch.device(d) for d in device_spec]

    if device_spec == "auto":
        return detect_gpus()

    # Parse comma-separated devices
    device_strs = [d.strip() for d in device_spec.split(",")]
    return [torch.device(d) for d in device_strs]
