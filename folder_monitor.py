import time
import yaml
from pathlib import Path
import logging
from typing import Set
import argparse
from inference import process_folder_recursive, load_model
import torch
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class MonitorConfiguration:
    root_folders: list[Path]
    model_path: Path
    check_interval_minutes: int
    save_overlay: bool
    use_tta: bool
    device: torch.device

    def __post_init__(self):
        self.root_folders = [Path(folder) for folder in self.root_folders]
        self.model_path = Path(self.model_path)
        self.check_interval = self.check_interval_minutes * 60
        self.device = torch.device(self.device)

class ProcessedFoldersTracker:
    def __init__(self, root_folder: Path):
        self.yaml_path = root_folder / "processed_folders.yaml"

    def load_processed_folders(self) -> Set[str]:
        if self.yaml_path.exists():
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                return set(data.get('processed_folders', []))
        return set()

    def save_processed_folders(self, processed_folders: Set[str]):
        with open(self.yaml_path, 'w') as f:
            yaml.dump({'processed_folders': list(processed_folders)}, f)

class FolderMonitor:
    def __init__(self, config: MonitorConfiguration, model: torch.nn.Module):
        self.config = config
        self.model = model
        logging.info(f"Model loaded successfully on {self.config.device}")

    def process_root_folder(self, root_folder: Path):
        tracker = ProcessedFoldersTracker(root_folder)
        processed_folders = tracker.load_processed_folders()

        for folder in self._get_unprocessed_subfolders(root_folder, processed_folders):
            try:
                self._process_folder(folder, tracker, processed_folders)
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _get_unprocessed_subfolders(self, root_folder: Path, processed_folders: Set[str]) -> list[Path]:
        unprocessed_folders = []
        for folder in root_folder.iterdir():
            if not folder.is_dir() or folder.name.endswith(('_mask', '_overlay')):
                continue
            folder_str = str(folder.relative_to(root_folder))
            if folder_str not in processed_folders or not self._are_targets_exist(folder):
                unprocessed_folders.append(folder)
        return unprocessed_folders

    def _are_targets_exist(self, folder_path: Path) -> bool:
        mask_folder = folder_path.parent / (folder_path.name + "_mask")
        if not mask_folder.exists():
            return False
        if self.config.save_overlay:
            overlay_folder = folder_path.parent / (folder_path.name + "_overlay")
            if not overlay_folder.exists():
                return False
        return True

    def _process_folder(self, folder: Path, tracker: ProcessedFoldersTracker, processed_folders: Set[str]):
        logging.info(f"Processing folder: {folder}")
        folder_str = str(folder.relative_to(folder.parent))
        processed_folders.add(folder_str)
        tracker.save_processed_folders(processed_folders)

        process_folder_recursive(
            folder,
            self.model,
            self.config.device,
            save_overlay=self.config.save_overlay,
            use_tta=self.config.use_tta
        )
        logging.info(f"Successfully processed: {folder}")

    def monitor(self):
        logging.info("Starting folder monitor")
        try:
            while True:
                for root_folder in self.config.root_folders:
                    if not root_folder.exists():
                        logging.warning(f"Root folder does not exist: {root_folder}")
                        continue
                    self.process_root_folder(root_folder)
                logging.info(f"Sleeping for {self.config.check_interval} seconds")
                time.sleep(self.config.check_interval)
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")

def parse_args():
    parser = argparse.ArgumentParser(description="Folder monitoring script for image processing")
    parser.add_argument(
        "--root_folders",
        nargs="+",
        type=str,
        required=True,
        help="List of root folders to monitor"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Mvanet_complete.pth",
        help="Path to the model weights file"
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=5,
        help="Interval between checks in minutes"
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save overlay of mask on original image"
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging level from arguments
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = MonitorConfiguration(
        root_folders=args.root_folders,
        model_path=args.model_path,
        check_interval_minutes=args.check_interval,
        save_overlay=args.save_overlay,
        use_tta=args.use_tta,
        device=args.device,
    )

    logging.info(f"Loading model from {config.model_path}")
    model = load_model(config.model_path, config.device)

    monitor = FolderMonitor(config, model)
    monitor.monitor()

if __name__ == "__main__":
    main()
