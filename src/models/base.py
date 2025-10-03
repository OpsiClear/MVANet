"""Base model interface for segmentation models"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol
import torch
from PIL import Image


class SegmentationModel(ABC):
    """Abstract base class for segmentation models"""

    @abstractmethod
    def load(self, model_path: Path, device: torch.device) -> None:
        """Load model weights from path"""
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image for inference

        Returns:
            tuple: (preprocessed_tensor, metadata dict with 'original_size', etc.)
        """
        pass

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run model forward pass"""
        pass

    @abstractmethod
    def postprocess(self, output: torch.Tensor, metadata: dict) -> Image.Image:
        """
        Postprocess model output to mask image

        Args:
            output: Raw model output tensor
            metadata: Metadata from preprocess (contains original_size, etc.)

        Returns:
            PIL Image mask
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name"""
        pass

    @property
    @abstractmethod
    def supports_tta(self) -> bool:
        """Whether model supports test-time augmentation"""
        pass

    def optimize_for_inference(self, device: torch.device) -> None:
        """Optional: Apply inference optimizations (channels_last, etc.)"""
        pass


class ModelConfig(Protocol):
    """Configuration protocol for models"""

    model_path: Path
    image_size: tuple[int, int]
    use_fp16: bool
    mean: list[float]
    std: list[float]
