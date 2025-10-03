# MVANet Image Segmentation

A modern web application for image segmentation using the MVANet deep learning model. Features a beautiful web dashboard and FastAPI-based backend for batch processing of images with real-time progress monitoring.

## Features

### Web Dashboard
- **Modern UI**: Clean, responsive interface with real-time updates
- **Job Submission**: Easy folder path input for batch image processing
- **Test-Time Augmentation**: Toggle TTA for improved segmentation accuracy
- **Real-time Console**: Live processing logs with timestamps and color-coded messages
- **Latest Image Viewer**: Preview the most recently processed image with overlays
- **Task Persistence**: Processing continues across page refreshes - monitor from anywhere

### Processing Capabilities
- **Recursive Folder Processing**: Automatically finds and processes all 'images' folders
- **Dual Output**: Generates both segmentation masks and overlays
- **GPU Acceleration**: Supports CUDA for fast processing
- **Background Processing**: Non-blocking job execution with task tracking
- **Error Handling**: Comprehensive logging and error reporting

### API Endpoints
- Task submission and status tracking
- Real-time log streaming
- System status monitoring
- Image serving and retrieval

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OpsiClear/MVANet.git
   cd MVANet
   ```

2. Install dependencies using `uv` (recommended) or `pip`:
   ```bash
   # Using uv (faster)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. Ensure the model files are in the `models/` directory:
   - `models/MVANet.pth` - Main segmentation model
   - `models/swin_base_patch4_window12_384_22kto1k.pth` - Backbone model

## Usage

### Starting the Application

Run the FastAPI server directly:

```bash
python api_app.py
```

The application will start on `http://localhost:8001` by default.

### Web Interface

1. Open your browser and navigate to `http://localhost:8001`
2. Enter the full path to your input folder (must contain folders named 'images')
3. Toggle Test-Time Augmentation if desired (improves accuracy but slower)
4. Click "Submit Job" to start processing
5. Monitor progress in real-time via the console output
6. View the latest processed image using the "Latest Image" button

**Input Folder Structure:**
```
your-input-folder/
├── images/           # Must be named 'images'
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
```

**Output Structure:**
```
your-input-folder_overlay/  # Segmentation overlays
your-input-folder_masks/    # Binary masks
```

### API Endpoints

#### Submit Processing Job
```bash
curl -X POST "http://localhost:8001/api/process" \
    -H "Content-Type: application/json" \
    -d '{
        "input_folder": "/path/to/folder",
        "use_tta": true
    }'
```

#### Check Task Status
```bash
curl "http://localhost:8001/api/status/{request_id}"
```

#### Get System Status
```bash
curl "http://localhost:8001/api/system/status"
```

#### Get Task Logs
```bash
curl "http://localhost:8001/api/logs/{task_id}"
```

#### Get Latest Processed Image
```bash
curl "http://localhost:8001/api/latest-image"
```

## Technology Stack

- **Backend**: FastAPI with async/await support
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **Deep Learning**: PyTorch with MVANet architecture
- **Model Backbone**: Swin Transformer
- **Image Processing**: PIL/Pillow, NumPy

## License

[MIT License](LICENSE) 