import uvicorn
import os
import argparse
import torch
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", BASE_DIR / "Mvanet_complete.pth")
STATIC_DIR = BASE_DIR / "static"

# Server settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

def get_available_devices():
    """Get a list of available GPU devices"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices

def select_device(device_str=None):
    """
    Select the appropriate device for inference.
    
    Args:
        device_str: Device string (e.g., 'cuda:0', 'cpu', 'auto')
        
    Returns:
        torch.device: The selected device
    """
    available_devices = get_available_devices()
    
    # If no device specified or 'auto', choose the best available
    if device_str is None or device_str == "auto":
        if "cuda:0" in available_devices:
            device_str = "cuda:0"
        else:
            device_str = "cpu"
    
    # Validate the selected device
    if device_str not in available_devices:
        print(f"Warning: Device '{device_str}' not available. Available devices: {available_devices}")
        print(f"Falling back to CPU.")
        device_str = "cpu"
    
    device = torch.device(device_str)
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MVANet segmentation API server")
    parser.add_argument("--host", type=str, default=HOST, help=f"Host to bind the server to (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to bind the server to (default: {PORT})")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use for inference (e.g., 'cuda:0', 'cpu', 'auto'). Default is 'auto'.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set the selected device as an environment variable for api_app.py to use
    selected_device = select_device(args.device)
    os.environ["SELECTED_DEVICE"] = str(selected_device)
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Using device: {selected_device}")
    
    uvicorn.run("api_app:app", host=args.host, port=args.port, reload=args.reload)
