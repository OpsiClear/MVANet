# MVANet Image Segmentation API

A FastAPI-based web service for image segmentation using the MVANet model.

## Features

- Process folders of images with the MVANet segmentation model
- Generate segmentation masks and optional overlays
- Track processing tasks with status updates
- View system logs
- Manage task history
- GPU selection for inference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MVANet.git
   cd MVANet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the model file `Mvanet_complete.pth` in the project root directory.

## Usage

### Starting the Server

The application can be started using the `run.py` script, which provides GPU selection functionality:

```bash
python run.py --device auto
```

Command-line options:
- `--device`: Specify which device to use for inference (e.g., 'cuda:0', 'cpu', 'auto'). Default is 'auto'.
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development

Examples:
```bash
# Use the first GPU
python run.py --device cuda:0

# Use CPU only
python run.py --device cpu

# Automatically select the best available device
python run.py --device auto

# Run on a specific port
python run.py --port 8080
```

### API Endpoints

#### Process a Folder
```bash
curl -X POST "http://localhost:8000/api/process" \
    -H "Content-Type: application/json" \
    -d '{
        "input_folder": "/path/to/images",
        "save_overlay": true,
        "use_tta": true,
        "device": "cuda:0",
        "callback_url": "http://your-callback-url/webhook"
    }'
```

#### Check Processing Status
```bash
curl "http://localhost:8000/api/status/{request_id}"
```

#### Get Device Information
```bash
curl "http://localhost:8000/api/device/info"
```

#### Get Queue Length
```bash
curl "http://localhost:8000/api/queue/length"
```

#### Get System Logs
```bash
curl "http://localhost:8000/api/logs?limit=100&level=INFO"
```

## Web Interface

The application also provides a web interface accessible at `http://localhost:8000/` for easy task management and monitoring.

## License

[MIT License](LICENSE) 