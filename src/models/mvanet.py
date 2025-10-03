"""MVANet model implementation"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
import logging

from .base import SegmentationModel
from ..MVANet import inf_MVANet


class MVANetModel(SegmentationModel):
    """MVANet segmentation model wrapper"""

    def __init__(
        self,
        image_size: tuple[int, int] = (1024, 1024),
        use_fp16: bool = True,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.image_size = image_size
        self.use_fp16 = use_fp16
        self.mean = mean
        self.std = std
        self.model = None
        self.device = None

        # Create transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def load(self, model_path: Path, device: torch.device) -> None:
        """Load MVANet model weights"""
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

        logging.info(f"MVANet model loaded on {device}")

    def optimize_for_inference(self, device: torch.device) -> None:
        """Apply inference optimizations"""
        if device.type == "cuda" and self.model is not None:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                logging.info("Model using channels_last memory format")
            except Exception as e:
                logging.warning(f"Could not set channels_last: {e}")

    def preprocess(self, image: Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Preprocess image for MVANet

        Returns:
            tuple: (tensor, metadata dict with 'original_size')
        """
        original_size = image.size  # (width, height)

        # Resize and convert
        resized_image = image.convert("RGB").resize(
            self.image_size, resample=Image.Resampling.BILINEAR
        )

        # Transform
        img_tensor = self.transform(resized_image).unsqueeze(0)

        # OPTIMIZATION: Use channels_last for better GPU performance
        if self.device.type == "cuda":
            img_tensor = img_tensor.to(memory_format=torch.channels_last)
            img_tensor = img_tensor.pin_memory()

        img_tensor = img_tensor.to(self.device, non_blocking=True)

        metadata = {"original_size": original_size}
        return img_tensor, metadata

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run MVANet forward pass"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, output: torch.Tensor, metadata: dict) -> Image.Image:
        """
        Postprocess MVANet output to mask

        Args:
            output: Raw model output tensor
            metadata: Must contain 'original_size' (width, height)

        Returns:
            PIL Image mask (grayscale)
        """
        original_size = metadata["original_size"]

        # Apply sigmoid activation
        mask_tensor = torch.sigmoid(output)

        # Resize to original dimensions
        mask_resized = F.interpolate(
            mask_tensor,
            size=(original_size[1], original_size[0]),  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Convert to PIL Image
        mask_np = (mask_resized.squeeze() * 255).cpu().numpy().astype(np.uint8)
        return Image.fromarray(mask_np)

    @property
    def name(self) -> str:
        return "MVANet"

    @property
    def supports_tta(self) -> bool:
        return True
