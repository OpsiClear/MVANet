"""Inference Engine - Generic segmentation model inference framework"""

__version__ = "0.1.0"

from .base import SegmentationModel
from .engine import InferenceEngine
from .multi_gpu import MultiGPUInferenceEngine, detect_gpus, parse_devices
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
    "MultiGPUInferenceEngine",
    "detect_gpus",
    "parse_devices",
    "ModelRegistry",
    "register_model",
    "get_model",
    "list_models",
    "get_model_info",
    "create_model",
]
