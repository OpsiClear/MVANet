"""Base model interface for segmentation models"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
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

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """
        Optional: Return JSON schema for model configuration

        Returns:
            dict: JSON schema describing model configuration options
        """
        return {}

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        """
        Optional: Return model metadata for display

        Returns:
            dict: Model metadata (description, author, version, etc.)
        """
        return {
            "name": cls.__name__,
            "description": cls.__doc__ or "No description available",
        }
