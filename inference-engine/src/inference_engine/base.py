"""Base model interface for segmentation models"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
import numpy as np


class SegmentationModel(ABC):
    """Abstract base class for segmentation models"""

    @abstractmethod
    def load(self, model_path: Path, device: torch.device) -> None:
        """Load model weights from path"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image for inference

        Args:
            image: RGB numpy array (H, W, 3)

        Returns:
            tuple: (preprocessed_tensor, metadata dict with 'original_size', etc.)
        """
        pass

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Run model forward pass

        Returns:
            Single tensor or dict of named tensors for multi-output models
        """
        pass

    @abstractmethod
    def postprocess(
        self, output: torch.Tensor | dict[str, torch.Tensor], metadata: dict
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Postprocess model output to image(s)

        Args:
            output: Raw model output (single tensor or dict of tensors)
            metadata: Metadata from preprocess (contains original_size, etc.)

        Returns:
            Single numpy array or dict of named numpy arrays
            All arrays should be (H, W) grayscale uint8 or (H, W, 3) RGB uint8
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

    def get_output_names(self) -> list[str]:
        """
        Return list of output names this model produces

        Returns:
            List of output names (e.g., ["mask", "depth", "normal"])
            Default: ["mask"] for single-output models
        """
        return ["mask"]

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
