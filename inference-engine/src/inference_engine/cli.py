"""CLI for inference engine"""

from pathlib import Path
from typing import Literal
import tyro
import torch
import logging

from . import InferenceEngine, list_models, create_model, get_model_info


def infer(
    input_folder: Path,
    model: str,
    model_path: Path,
    output_overlay: Path | None = None,
    output_mask: Path | None = None,
    device: str = "cuda:0",
    use_fp16: bool = True,
    use_tta: bool = False,
    chunk_size: int = 20,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
):
    """Run inference on images"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device_obj = torch.device(device)

    # Create model
    logging.info(f"Loading model: {model}")
    model_obj = create_model(model)
    model_obj.load(model_path, device_obj)
    model_obj.optimize_for_inference(device_obj)

    # Create engine
    engine = InferenceEngine(
        model=model_obj,
        device=device_obj,
        use_fp16=use_fp16,
        chunk_size=chunk_size,
    )

    # Determine output paths
    overlay_folder = output_overlay or input_folder / "overlays"
    mask_folder = output_mask or input_folder / "masks"

    # Process
    logging.info(f"Processing: {input_folder}")
    result = engine.process_folder(
        folder_path=input_folder,
        overlay_folder=overlay_folder,
        mask_folder=mask_folder,
        use_tta=use_tta,
    )

    logging.info(f"Results: {result}")


def list_models_cmd(verbose: bool = False):
    """List available models"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = list_models()

    if not models:
        logger.info("No models installed")
        return

    logger.info(f"Available models ({len(models)}):")
    for model_name in models:
        if verbose:
            info = get_model_info(model_name)
            logger.info(f"  {model_name}: {info.get('description', 'N/A')}")
        else:
            logger.info(f"  - {model_name}")


def model_info(model: str):
    """Show model information"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        info = get_model_info(model)
        logger.info(f"Model: {model}")
        logger.info(f"  Class: {info.get('class', 'N/A')}")
        logger.info(f"  Description: {info.get('description', 'N/A')}")
        logger.info(f"  TTA: {info.get('supports_tta', False)}")
    except KeyError as e:
        logger.error(f"Error: {e}")


def main():
    """Main CLI"""
    tyro.extras.subcommand_cli_from_dict({
        "infer": infer,
        "list": list_models_cmd,
        "info": model_info,
    })


if __name__ == "__main__":
    main()
