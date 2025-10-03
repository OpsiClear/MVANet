"""MVANet model plugin for inference engine"""

import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
import logging

from inference_engine import SegmentationModel
from .MVANet import inf_MVANet

logger = logging.getLogger(__name__)


class MVANetModel(SegmentationModel):
    """MVANet segmentation model"""

    def __init__(
        self,
        image_size: tuple[int, int] = (1024, 1024),
        use_fp16: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.use_fp16 = use_fp16
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.model = None
        self.device = None

    def load(self, model_path: Path, device: torch.device) -> None:
        """Load MVANet weights"""
        self.device = device
        self.model = inf_MVANet()

        checkpoint = torch.load(model_path, weights_only=False, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        logger.info(f"MVANet loaded on {device}")

    def optimize_for_inference(self, device: torch.device) -> None:
        """Apply optimizations"""
        if device.type == "cuda" and self.model:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                logger.info("Using channels_last")
            except Exception as e:
                logger.warning(f"Could not set channels_last: {e}")

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image (RGB numpy array from OpenCV)

        Returns:
            tuple: (tensor, metadata)
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Resize
        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        # Normalize: convert to float, scale to [0,1], normalize with mean/std
        img_float = resized.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std

        # Convert to tensor: HWC -> CHW
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)

        # Optimize for GPU
        if self.device.type == "cuda":
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.pin_memory()

        img_tensor = img_tensor.to(self.device, non_blocking=True)

        return img_tensor, {"original_size": original_size}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, output: torch.Tensor, metadata: dict) -> np.ndarray:
        """
        Postprocess to mask

        Returns:
            numpy array (grayscale mask)
        """
        original_size = metadata["original_size"]

        # Sigmoid + resize
        mask_tensor = torch.sigmoid(output)
        mask_resized = F.interpolate(
            mask_tensor,
            size=(original_size[1], original_size[0]),  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Convert to numpy
        mask_np = (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)
        return mask_np

    @property
    def name(self) -> str:
        return "MVANet"

    @property
    def supports_tta(self) -> bool:
        return True

    def get_output_names(self) -> list[str]:
        """Return list of outputs this model produces"""
        return ["mask"]

    @classmethod
    def get_metadata(cls) -> dict:
        return {
            "name": "MVANet",
            "description": "Multi-View Aggregation Network for salient object detection",
            "author": "Original MVANet authors",
            "version": "1.0",
        }
