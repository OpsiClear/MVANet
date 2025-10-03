"""Template model plugin - Copy this to create your own plugin"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import logging
from pathlib import Path

from inference_engine import SegmentationModel

logger = logging.getLogger(__name__)


class TemplateModel(SegmentationModel):
    """Template segmentation model - Replace with your model"""

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 512),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize model

        Args:
            image_size: Input size for model
            mean: ImageNet mean for normalization
            std: ImageNet std for normalization
        """
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.model = None
        self.device = None

    def load(self, model_path: Path, device: torch.device) -> None:
        """
        Load model weights

        Args:
            model_path: Path to model checkpoint
            device: torch device
        """
        self.device = device

        # TODO: Replace with your model architecture
        # Example:
        # from .your_architecture import YourModel
        # self.model = YourModel()
        # checkpoint = torch.load(model_path, map_location=device)
        # self.model.load_state_dict(checkpoint)

        self.model = torch.nn.Identity()  # Placeholder

        self.model = self.model.to(device)
        self.model.eval()
        logger.info(f"Template model loaded on {device}")

    def optimize_for_inference(self, device: torch.device) -> None:
        """Apply inference optimizations"""
        if device.type == "cuda" and self.model:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                logger.info("Using channels_last memory format")
            except Exception as e:
                logger.warning(f"Could not set channels_last: {e}")

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image

        Args:
            image: RGB numpy array (H, W, 3) from OpenCV

        Returns:
            (tensor, metadata dict)
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Resize image
        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        img_float = resized.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std

        # Convert to tensor: HWC -> CHW
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)

        # GPU optimizations
        if self.device.type == "cuda":
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.pin_memory()

        img_tensor = img_tensor.to(self.device, non_blocking=True)

        return img_tensor, {"original_size": original_size}

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass

        Args:
            tensor: Preprocessed input tensor

        Returns:
            Raw model output
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, output: torch.Tensor, metadata: dict) -> np.ndarray:
        """
        Postprocess model output to mask

        Args:
            output: Raw model output tensor
            metadata: Contains 'original_size' (width, height)

        Returns:
            Grayscale mask (H, W) as uint8 numpy array
        """
        original_size = metadata["original_size"]

        # Apply activation (adjust based on your model)
        mask_tensor = torch.sigmoid(output)  # or torch.softmax, etc.

        # Resize to original dimensions
        mask_resized = F.interpolate(
            mask_tensor,
            size=(original_size[1], original_size[0]),  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Convert to uint8 numpy array
        mask_np = (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)
        return mask_np

    @property
    def name(self) -> str:
        """Model name"""
        return "Template"

    @property
    def supports_tta(self) -> bool:
        """Whether model supports test-time augmentation"""
        return True

    @classmethod
    def get_metadata(cls) -> dict:
        """
        Return model metadata

        Returns:
            Dictionary with model information
        """
        return {
            "name": "Template",
            "description": "Template segmentation model - replace with your model",
            "author": "Your Name",
            "version": "1.0.0",
        }
