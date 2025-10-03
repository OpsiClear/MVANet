"""Inference Engine - Generic segmentation model inference framework"""

__version__ = "0.1.0"

from .base import SegmentationModel
from .engine import InferenceEngine
from .registry import (
    ModelRegistry,
    register_model,
    get_model,
    list_models,
    get_model_info,
    create_model,
)

__all__ = [
    "SegmentationModel",
    "InferenceEngine",
    "ModelRegistry",
    "register_model",
    "get_model",
    "list_models",
    "get_model_info",
    "create_model",
]
