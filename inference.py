import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F
import ttach as tta
from torch.amp import autocast
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rgb_loader_refiner(
    original_image: Image.Image, target_size: tuple[int, int]
) -> tuple[torch.Tensor, int, int, Image.Image]:
    """Handles initial image loading and scaling"""
    h, w = original_image.size
    image = original_image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
    return image.convert("RGB"), h, w, original_image


def rescale_to(target: torch.Tensor, scale_factor: float = 2, interpolation="nearest"):
    """Rescales tensor by a factor"""
    return F.interpolate(target, scale_factor=scale_factor, mode=interpolation)


def resize_as(target: torch.Tensor, source: torch.Tensor, interpolation="bicubic"):
    """Resizes x to match y's dimensions"""
    return F.interpolate(target, size=source.shape[-2:], mode=interpolation)


def postprocess_image(result: torch.Tensor, im_size: tuple[int, int]) -> np.ndarray:
    """Post-processes and rescales the output mask"""
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bicubic"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).cpu().numpy().astype(np.uint8)
    return im_array


def preprocess_image(
    image_path: Path, target_size: tuple[int, int], device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    # Create transform once and reuse
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use with statement for proper file handling
    with Image.open(image_path) as image:
        original_size = image.size
        # Combine convert and resize operations
        resized_image = image.convert("RGB").resize(
            target_size, resample=Image.Resampling.LANCZOS
        )
        img_tensor: torch.Tensor = img_transform(resized_image).unsqueeze(0).to(device)
        return img_tensor, original_size


def postprocess_mask(
    mask_tensor: torch.Tensor, original_size: tuple[int, int]
) -> Image.Image:
    """Post-processes the mask with sigmoid activation and normalization"""
    # Apply sigmoid activation
    mask_tensor = torch.sigmoid(mask_tensor)

    # Continue with existing post-processing
    mask_tensor = mask_tensor.cpu()
    mask_resized = torch.squeeze(
        F.interpolate(
            mask_tensor, size=(original_size[1], original_size[0]), mode="bicubic"
        ),
        0,
    )
    ma = torch.max(mask_resized)
    mi = torch.min(mask_resized)
    mask_normalized = (mask_resized - mi) / (ma - mi)
    mask_np = (mask_normalized * 255).cpu().data.numpy().astype(np.uint8)
    mask_pil = Image.fromarray(mask_np.squeeze())
    return mask_pil


def setup_tta_transforms():
    """Setup test-time augmentation transforms"""
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ]
    )


def infer_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    target_size: tuple[int, int],
    use_tta: bool = True,
) -> Image.Image:
    """Enhanced inference with optional test-time augmentation"""
    img_tensor, original_size = preprocess_image(image_path, target_size, device)

    with torch.no_grad():
        if use_tta:
            tta_transforms = setup_tta_transforms()
            masks = []
            for transformer in tta_transforms:
                aug_img = transformer.augment_image(img_tensor)
                mask = model(aug_img)
                deaug_mask = transformer.deaugment_mask(mask)
                masks.append(deaug_mask)
            mask_tensor = torch.mean(torch.stack(masks, dim=0), dim=0)
        else:
            mask_tensor = model(img_tensor)

    mask = postprocess_mask(mask_tensor, original_size)
    return mask


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model: torch.nn.Module = torch.load(
        model_path, weights_only=False, map_location=device
    )
    model = model.to(device)
    model.eval()
    return model


def create_overlay(original_image: Image.Image, mask: Image.Image) -> Image.Image:
    """Creates an overlay of the mask on the original image"""
    # Ensure the mask is the same size as the original image
    if original_image.size != mask.size:
        mask = mask.resize(original_image.size, Image.Resampling.LANCZOS)

    # Convert original image to RGBA if it isn't already
    if original_image.mode != "RGBA":
        original_image = original_image.convert("RGBA")

    # Use the mask as the alpha channel
    r, g, b, _ = original_image.split()
    overlay = Image.merge("RGBA", (r, g, b, mask))
    return overlay


def save_image_files(
    mask: Image.Image,
    original_image: Optional[Image.Image],
    output_path: Path,
    overlay_path: Optional[Path] = None,
):
    """Helper function to save mask and overlay images"""
    def save_mask():
        mask.save(str(output_path), optimize=True)
    
    def save_overlay():
        if overlay_path and original_image:
            overlay = create_overlay(original_image, mask)
            overlay.save(str(overlay_path), optimize=True)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        futures.append(executor.submit(save_mask))
        if overlay_path and original_image:
            futures.append(executor.submit(save_overlay))
        
        # Wait for all saving operations to complete
        for future in futures:
            future.result()


def process_folder(
    folder_path: Path,
    model: torch.nn.Module,
    target_size: tuple[int, int],
    device: torch.device,
    save_overlay: bool = False,
    use_tta: bool = True,
):
    if not folder_path.is_dir():
        raise ValueError(f"Folder {folder_path} does not exist")

    # Get all image files
    image_files = [
        file_path
        for file_path in folder_path.iterdir()
        if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_files:
        raise ValueError(f"No supported images found in {folder_path}")

    # Create output folders
    output_folder = folder_path.parent / (folder_path.name + "_mask")
    output_folder.mkdir(exist_ok=True)
    if save_overlay:
        overlay_folder = folder_path.parent / (folder_path.name + "_overlay")
        overlay_folder.mkdir(exist_ok=True)

    # Process images one at a time since the model requires specific batch structure
    with ThreadPoolExecutor() as executor:
        save_futures = []
        
        for img_path in image_files:
            try:
                # Process single image
                img_tensor, original_size = preprocess_image(img_path, target_size, device)

                # Process through model
                with torch.no_grad():
                    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        if use_tta:
                            tta_transforms = setup_tta_transforms()
                            masks = []
                            for transformer in tta_transforms:
                                aug_img = transformer.augment_image(img_tensor)
                                mask = model(aug_img)
                                deaug_mask = transformer.deaugment_mask(mask)
                                masks.append(deaug_mask)
                            mask_tensor = torch.mean(torch.stack(masks, dim=0), dim=0)
                        else:
                            mask_tensor = model(img_tensor)

                # Post-process and save
                mask = postprocess_mask(mask_tensor, original_size)
                output_path = output_folder / (img_path.stem + ".png")

                if save_overlay:
                    with Image.open(img_path) as original_image:
                        original_image = original_image.convert("RGB")
                        overlay_path = overlay_folder / (img_path.stem + ".png")
                else:
                    original_image = None
                    overlay_path = None

                # Submit saving task to thread pool
                save_futures.append(
                    executor.submit(save_image_files, mask, original_image, output_path, overlay_path)
                )

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Wait for all saving operations to complete
        for future in save_futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error during save operation: {e}")


model_image_size = (1024, 1024)

torch.cuda.empty_cache()
device = torch.device("cuda:0")
model = load_model(Path("Mvanet_complete.pth"), device)
folder_path = Path("./test_images")
process_folder(
    folder_path, model, model_image_size, device, save_overlay=True, use_tta=False
)
