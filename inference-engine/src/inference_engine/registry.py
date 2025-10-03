"""Model registry and plugin discovery system"""

import importlib.metadata
import logging
from typing import Type, Any
from pathlib import Path

from .base import SegmentationModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for segmentation model plugins"""

    def __init__(self):
        self._models: dict[str, Type[SegmentationModel]] = {}
        self._loaded = False

    def register(self, name: str, model_class: Type[SegmentationModel]) -> None:
        """
        Register a model class

        Args:
            name: Model identifier (e.g., 'mvanet', 'sam', 'u2net')
            model_class: Model class implementing SegmentationModel
        """
        if not issubclass(model_class, SegmentationModel):
            raise TypeError(
                f"Model class {model_class} must inherit from SegmentationModel"
            )

        if name in self._models:
            logger.warning(f"Overwriting existing model registration: {name}")

        self._models[name] = model_class
        logger.info(f"Registered model: {name} ({model_class.__name__})")

    def discover_plugins(self) -> None:
        """Discover and load model plugins via entry points"""
        if self._loaded:
            return

        entry_points = importlib.metadata.entry_points()

        # Get models from the 'inference_engine.models' group
        if hasattr(entry_points, "select"):
            # Python 3.10+
            model_eps = entry_points.select(group="inference_engine.models")
        else:
            # Python 3.9
            model_eps = entry_points.get("inference_engine.models", [])

        for ep in model_eps:
            try:
                model_class = ep.load()
                self.register(ep.name, model_class)
            except Exception as e:
                logger.error(f"Failed to load model plugin {ep.name}: {e}")

        self._loaded = True
        logger.info(f"Discovered {len(self._models)} model(s)")

    def get(self, name: str) -> Type[SegmentationModel]:
        """
        Get a model class by name

        Args:
            name: Model identifier

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        self.discover_plugins()

        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(
                f"Model '{name}' not found. Available models: {available or 'none'}"
            )

        return self._models[name]

    def list_models(self) -> list[str]:
        """
        List all registered model names

        Returns:
            List of model identifiers
        """
        self.discover_plugins()
        return sorted(self._models.keys())

    def get_model_info(self, name: str) -> dict[str, Any]:
        """
        Get model metadata

        Args:
            name: Model identifier

        Returns:
            Dictionary with model metadata
        """
        model_class = self.get(name)
        metadata = model_class.get_metadata()
        metadata["name"] = name
        metadata["class"] = model_class.__name__
        metadata["supports_tta"] = (
            model_class.__dict__.get("supports_tta") is not None
        )
        return metadata

    def create_model(
        self, name: str, model_path: Path | None = None, **kwargs
    ) -> SegmentationModel:
        """
        Create a model instance

        Args:
            name: Model identifier
            model_path: Optional path to model weights
            **kwargs: Additional arguments for model constructor

        Returns:
            Model instance
        """
        model_class = self.get(name)
        return model_class(**kwargs)


# Global registry instance
_registry = ModelRegistry()


def register_model(name: str, model_class: Type[SegmentationModel]) -> None:
    """Register a model in the global registry"""
    _registry.register(name, model_class)


def get_model(name: str) -> Type[SegmentationModel]:
    """Get a model class from the global registry"""
    return _registry.get(name)


def list_models() -> list[str]:
    """List all available models"""
    return _registry.list_models()


def get_model_info(name: str) -> dict[str, Any]:
    """Get model metadata"""
    return _registry.get_model_info(name)


def create_model(
    name: str, model_path: Path | None = None, **kwargs
) -> SegmentationModel:
    """Create a model instance"""
    return _registry.create_model(name, model_path, **kwargs)
