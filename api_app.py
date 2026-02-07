"""FastAPI web application for image segmentation.

Provides REST API and web UI for running inference with oc_masker models.
"""

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from oc_masker import (
    InferenceEngine,
    create_model,
    list_models,
    get_model_info,
    detect_gpus,
    parse_devices,
)
from oc_masker.engine import SUPPORTED_FORMATS


# Configure logging to capture to memory
class MemoryLogHandler(logging.Handler):
    def __init__(self, max_logs=1000):
        super().__init__()
        self.logs = deque(maxlen=max_logs)
        self.current_task_logs = {}

    def emit(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "task_id": getattr(record, "task_id", None),
        }
        self.logs.append(log_entry)

        # Store logs per task
        task_id = log_entry["task_id"]
        if task_id:
            if task_id not in self.current_task_logs:
                self.current_task_logs[task_id] = deque(maxlen=500)
            self.current_task_logs[task_id].append(log_entry)

    def get_task_logs(self, task_id, since_timestamp=None):
        if task_id not in self.current_task_logs:
            return []

        logs = list(self.current_task_logs[task_id])
        if since_timestamp:
            logs = [log for log in logs if log["timestamp"] > since_timestamp]

        return logs


# Create memory handler and configure logging
memory_handler = MemoryLogHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        memory_handler,  # Memory storage
    ],
)
logger = logging.getLogger(__name__)

# Create main app
app = FastAPI(title="Image Segmentation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the frontend at root URL
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


class RequestStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessRequest(BaseModel):
    input_folder: str
    model: str = "mvanet"  # Model selection
    devices: str = "auto"  # GPU devices: "auto", "cuda:0", "cuda:0,cuda:1", etc.
    tta_merge_mode: str = "none"  # TTA merge strategy: "none", "mean", "max", "gmean"
    force_overwrite: bool = False
    output_folder_postfix: str = ""  # e.g., "_v2" -> "masks_v2"
    # Model-specific options (passed to model constructor)
    image_size: tuple[int, int] | None = None  # e.g., (1024, 1024)
    use_fp16: bool = True


class ProcessResponse(BaseModel):
    request_id: str
    status: RequestStatus
    output_folder: str | None = None
    output_folders: dict[str, str] | None = None  # Map of output types to folders
    output_types: list[str] | None = None  # e.g., ["mask", "depth", "normal"]
    error_message: str | None = None
    input_folder: str | None = None
    tta_merge_mode: str | None = None  # TTA merge strategy
    model: str | None = None
    devices: str | None = None  # GPU devices used
    model_metadata: dict | None = None
    force_overwrite: bool | None = None
    output_folder_postfix: str | None = None
    stats: dict | None = None  # Processing statistics


class SystemStatus(BaseModel):
    """Current system processing status."""

    is_processing: bool
    current_request_id: str | None = None
    current_input_folder: str | None = None


# Global state
tasks: dict[str, ProcessResponse] = {}
engines: dict[str, InferenceEngine] = {}  # Cache engines per model
available_gpus: list[dict] = []  # Detected GPUs at startup

# Processing state tracking
processing_lock = threading.Lock()
current_processing_task: str | None = None
current_input_folder: str | None = None

# Background executor for processing tasks
executor = ThreadPoolExecutor(max_workers=1)


def get_or_create_engine(
    model_name: str,
    devices_spec: str = "auto",
    image_size: tuple[int, int] | None = None,
    use_fp16: bool = True,
) -> InferenceEngine:
    """Get or create inference engine for specified model with configuration"""
    global engines

    # Parse device specification
    devices = parse_devices(devices_spec)
    devices_str = ",".join(str(d) for d in devices)

    # Create cache key based on model name and configuration
    image_size_str = f"{image_size[0]}x{image_size[1]}" if image_size else "default"
    cache_key = f"{model_name}_{devices_str}_{image_size_str}_fp16_{use_fp16}"

    if cache_key not in engines:
        logger.info(f"Creating inference engine for model: {model_name} on devices: {devices_str}")

        # Model factory using bundled weights with specified options
        def model_factory(dev):
            model_kwargs = {}
            if image_size is not None:
                model_kwargs["image_size"] = image_size
            model_kwargs["use_fp16"] = use_fp16

            m = create_model(model_name, **model_kwargs)
            m.load(None, dev)  # None = use bundled weights
            m.optimize_for_inference(dev)
            return m

        engines[cache_key] = InferenceEngine(
            devices=devices,
            model_factory=model_factory,
            use_fp16=use_fp16,
        )
        logger.info(f"Inference engine for {model_name} loaded successfully on {len(devices)} device(s)")

    return engines[cache_key]


def log_with_task_id(message: str, level: str = "INFO", task_id: str = None):
    """Log a message with task ID for filtering"""
    log_record = logging.LogRecord(
        name=logger.name,
        level=getattr(logging, level),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    log_record.task_id = task_id
    log_record.created = time.time()
    memory_handler.emit(log_record)


def background_process_task(
    request_id: str,
    input_folder_str: str,
    model_name: str,
    devices_spec: str = "auto",
    tta_merge_mode: str = "none",
    force_overwrite: bool = False,
    output_folder_postfix: str = "",
    image_size: tuple[int, int] | None = None,
    use_fp16: bool = True,
):
    """Background task for processing images"""
    global current_processing_task, current_input_folder

    try:
        log_with_task_id(
            f"Starting background processing for task {request_id}", "INFO", request_id
        )

        # Set processing state
        with processing_lock:
            current_processing_task = request_id
            current_input_folder = input_folder_str

        # Get model metadata
        try:
            model_metadata = get_model_info(model_name)
            log_with_task_id(f"Model: {model_metadata.get('name', model_name)}", "INFO", request_id)
        except Exception as e:
            log_with_task_id(f"Warning: Could not get model metadata: {e}", "WARNING", request_id)
            model_metadata = {}

        # Parse and log devices
        devices = parse_devices(devices_spec)
        devices_str = ", ".join(str(d) for d in devices)
        log_with_task_id(f"Using {len(devices)} device(s): {devices_str}", "INFO", request_id)

        # Get or create engine for specified model
        log_with_task_id(f"Loading inference engine for model: {model_name}...", "INFO", request_id)
        engine = get_or_create_engine(
            model_name=model_name,
            devices_spec=devices_spec,
            image_size=image_size,
            use_fp16=use_fp16,
        )
        log_with_task_id(f"Inference engine for {model_name} loaded successfully", "INFO", request_id)

        # Get output types from model
        output_types = engine.model.get_output_names()
        log_with_task_id(f"Model outputs: {', '.join(output_types)}", "INFO", request_id)

        # Process folder recursively
        input_folder = Path(input_folder_str)

        log_with_task_id(
            f"Starting recursive processing for: {input_folder}", "INFO", request_id
        )
        log_with_task_id(
            f"TTA Merge Mode: {tta_merge_mode}",
            "INFO",
            request_id,
        )
        log_with_task_id(
            f"Force Overwrite: {'Enabled' if force_overwrite else 'Disabled'}",
            "INFO",
            request_id,
        )
        if output_folder_postfix:
            log_with_task_id(f"Output Folder Postfix: {output_folder_postfix}", "INFO", request_id)

        # Determine output folders for status update
        output_folders_dict = {}
        output_folder = None  # Primary output folder for compatibility

        if (input_folder / "images").exists():
            # Direct images folder - build output paths
            parent = input_folder
            for output_type in output_types:
                folder_name = f"{output_type}s{output_folder_postfix}"
                output_folders_dict[output_type] = str(parent / folder_name)
            output_folder = output_folders_dict.get("mask")
        else:
            # Search for first images folder
            for p in input_folder.rglob("images"):
                if p.is_dir():
                    parent = p.parent
                    for output_type in output_types:
                        folder_name = f"{output_type}s{output_folder_postfix}"
                        output_folders_dict[output_type] = str(parent / folder_name)
                    output_folder = output_folders_dict.get("mask")
                    break

        # Update task with output folder paths
        if request_id in tasks:
            tasks[request_id].output_folder = output_folder
            tasks[request_id].output_folders = output_folders_dict
            tasks[request_id].output_types = output_types
            tasks[request_id].model_metadata = model_metadata
            if output_folders_dict:
                log_with_task_id(f"Outputs will be saved to:", "INFO", request_id)
                for output_type, folder_path in output_folders_dict.items():
                    log_with_task_id(f"  - {output_type}: {folder_path}", "INFO", request_id)

        # Process using engine
        result = engine.process_dataset_recursive(
            root_path=input_folder,
            tta_merge_mode=tta_merge_mode,
            output_folder_postfix=output_folder_postfix,
            force_overwrite=force_overwrite,
        )

        log_with_task_id(
            f"Processing completed! Processed {result['processed']} images in {result['time']:.1f}s",
            "INFO",
            request_id,
        )

        # Update task as completed
        tasks[request_id] = ProcessResponse(
            request_id=request_id,
            status=RequestStatus.COMPLETED,
            output_folder=output_folder or "Multiple folders processed",
            output_folders=output_folders_dict,
            output_types=output_types,
            input_folder=input_folder_str,
            tta_merge_mode=tta_merge_mode,
            model=model_name,
            devices=devices_spec,
            model_metadata=model_metadata,
            force_overwrite=force_overwrite,
            output_folder_postfix=output_folder_postfix,
            stats=result,
        )

        log_with_task_id(
            f"Task {request_id} completed successfully", "INFO", request_id
        )

    except Exception as e:
        log_with_task_id(
            f"Error processing task {request_id}: {e}", "ERROR", request_id
        )
        tasks[request_id] = ProcessResponse(
            request_id=request_id,
            status=RequestStatus.FAILED,
            error_message=str(e),
            input_folder=input_folder_str,
            tta_merge_mode=tta_merge_mode,
            model=model_name,
            devices=devices_spec,
            force_overwrite=force_overwrite,
            output_folder_postfix=output_folder_postfix,
        )

    finally:
        # Clear processing state
        with processing_lock:
            current_processing_task = None
            current_input_folder = None

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_with_task_id(
            f"Background processing for task {request_id} finished", "INFO", request_id
        )


@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get current system processing status.

    Returns:
        SystemStatus with processing state and current task info.
    """
    with processing_lock:
        return SystemStatus(
            is_processing=current_processing_task is not None,
            current_request_id=current_processing_task,
            current_input_folder=current_input_folder,
        )


@app.get("/api/models")
async def get_available_models():
    """List all available model plugins."""
    try:
        models = list_models()
        return {"models": models, "total": len(models)}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}",
        )


@app.get("/api/models/{model_name}")
async def get_model_metadata(model_name: str):
    """Get metadata for a specific model."""
    try:
        metadata = get_model_info(model_name)
        return metadata
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}",
        )


@app.get("/api/gpus")
async def get_available_gpus():
    """Get list of available GPU devices.

    Returns:
        Dictionary with list of GPU info and total count.
    """
    return {"gpus": available_gpus, "total": len(available_gpus)}


@app.get("/api/logs/{request_id}")
async def get_task_logs(request_id: str, since: str | None = None):
    """Get logs for a specific task.

    Args:
        request_id: The task/request ID to get logs for.
        since: Optional ISO timestamp to filter logs after this time.
    """
    try:
        logs = memory_handler.get_task_logs(request_id, since)
        return {"request_id": request_id, "logs": logs, "total_logs": len(logs)}
    except Exception as e:
        logger.error(f"Error fetching logs for task {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching logs: {str(e)}",
        )


@app.post("/api/process", response_model=ProcessResponse)
async def process_folder(request: ProcessRequest) -> ProcessResponse:
    """Submit a folder for image segmentation processing"""
    # Check if already processing
    with processing_lock:
        if current_processing_task is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another task is already being processed. Please wait for it to complete.",
            )

    request_id = str(uuid.uuid4())

    try:
        # Validate input folder
        input_folder = Path(request.input_folder)
        if not input_folder.exists() or not input_folder.is_dir():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        # Store initial task with processing status
        tasks[request_id] = ProcessResponse(
            request_id=request_id,
            status=RequestStatus.PROCESSING,
            input_folder=request.input_folder,
            tta_merge_mode=request.tta_merge_mode,
            model=request.model,
            devices=request.devices,
            force_overwrite=request.force_overwrite,
            output_folder_postfix=request.output_folder_postfix,
        )

        # Submit background task with all options
        executor.submit(
            background_process_task,
            request_id=request_id,
            input_folder_str=request.input_folder,
            model_name=request.model,
            devices_spec=request.devices,
            tta_merge_mode=request.tta_merge_mode,
            force_overwrite=request.force_overwrite,
            output_folder_postfix=request.output_folder_postfix,
            image_size=request.image_size,
            use_fp16=request.use_fp16,
        )

        logger.info(f"Task {request_id} submitted for background processing")
        return tasks[request_id]

    except Exception as e:
        logger.error(f"Error submitting task {request_id}: {e}")
        tasks[request_id] = ProcessResponse(
            request_id=request_id,
            status=RequestStatus.FAILED,
            error_message=str(e),
            input_folder=request.input_folder,
            tta_merge_mode=request.tta_merge_mode,
            model=request.model,
            devices=request.devices,
            force_overwrite=request.force_overwrite,
            output_folder_postfix=request.output_folder_postfix,
        )
        return tasks[request_id]


@app.get("/api/status/{request_id}", response_model=ProcessResponse)
async def get_status(request_id: str) -> ProcessResponse:
    """Get the status of a processing request"""
    if request_id not in tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request with ID {request_id} not found",
        )
    return tasks[request_id]


def find_images_in_folder(folder_path: Path) -> list[Path]:
    """Find all image files in a folder with supported extensions.

    Args:
        folder_path: Directory to search for images.

    Returns:
        List of paths to found image files.
    """
    if not folder_path.exists():
        return []

    images = []
    for ext in SUPPORTED_FORMATS:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))
        images.extend(folder_path.glob(f"**/*{ext}"))
        images.extend(folder_path.glob(f"**/*{ext.upper()}"))

    return images


def get_current_processing_output_folder() -> Path | None:
    """Get the output folder for the currently processing task."""
    global current_input_folder
    if not current_input_folder:
        return None

    try:
        input_path = Path(current_input_folder)

        # Check if input_path has an 'images' subfolder
        if (input_path / "images").is_dir():
            return input_path / "masks"

        # Search for first 'images' folder recursively
        for p in input_path.rglob("images"):
            if p.is_dir():
                return p.parent / "masks"

    except Exception as e:
        logger.error(f"Error getting current processing output folder: {e}")

    return None


def create_image_response(
    image_path: Path,
    output_folder: Path,
    request_id: str,
    input_folder: str | None = None,
    task_status: str = "completed",
) -> dict:
    """Create a standardized image response.

    Args:
        image_path: Path to the image file.
        output_folder: Path to the output folder containing the image.
        request_id: The task/request ID or 'current_processing'.
        input_folder: Original input folder path.
        task_status: Current task status.

    Returns:
        Dictionary with image metadata and URL.
    """
    return {
        "request_id": request_id,
        "input_folder": input_folder,
        "output_folder": str(output_folder),
        "image_name": image_path.name,
        "image_path": str(image_path.relative_to(output_folder)),
        "image_url": f"/api/images/{request_id}/{image_path.name}"
        if request_id != "current_processing"
        else f"/api/file/current_processing/{image_path.name}",
        "modified_time": image_path.stat().st_mtime,
        "task_status": task_status,
    }


@app.get("/api/latest-image")
async def get_latest_processed_image():
    """Get the most recently saved processed image from any output folder."""
    try:
        all_candidate_images = []

        # 1. Check completed/failed tasks with output folders
        tasks_with_output = [
            (request_id, task)
            for request_id, task in tasks.items()
            if task.output_folder
        ]

        for request_id, task in tasks_with_output:
            output_folder = Path(task.output_folder)
            images = find_images_in_folder(output_folder)

            for img_path in images:
                all_candidate_images.append(
                    {
                        "path": img_path,
                        "request_id": request_id,
                        "task": task,
                        "output_folder": output_folder,
                        "mtime": img_path.stat().st_mtime,
                    }
                )

        # 2. Check current processing task
        current_output_folder = get_current_processing_output_folder()
        if current_output_folder and current_output_folder.exists():
            images = find_images_in_folder(current_output_folder)

            for img_path in images:
                all_candidate_images.append(
                    {
                        "path": img_path,
                        "request_id": "current_processing",
                        "task": None,
                        "output_folder": current_output_folder,
                        "mtime": img_path.stat().st_mtime,
                    }
                )

        # 3. Find the most recent image
        if not all_candidate_images:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No processed images found",
            )

        latest = max(all_candidate_images, key=lambda x: x["mtime"])

        # 4. Create response
        if latest["request_id"] == "current_processing":
            return create_image_response(
                latest["path"],
                latest["output_folder"],
                "current_processing",
                input_folder=current_input_folder,
                task_status="processing",
            )
        else:
            return create_image_response(
                latest["path"],
                latest["output_folder"],
                latest["request_id"],
                input_folder=latest["task"].input_folder,
                task_status=latest["task"].status,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting latest image: {str(e)}",
        )


@app.get("/api/images/{request_id}/{filename}")
async def get_image_file(request_id: str, filename: str):
    """Get a specific image file from a task's output folder.

    Args:
        request_id: The task/request ID.
        filename: Name of the image file to retrieve.
    """
    if request_id not in tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {request_id} not found",
        )

    task = tasks[request_id]

    if not task.output_folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No output folder for this task",
        )

    try:
        output_folder = Path(task.output_folder)
        image_path = output_folder / filename

        # Security check - ensure the file is within the output folder
        if not str(image_path.resolve()).startswith(str(output_folder.resolve())):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        if not image_path.exists():
            # Try to find the file recursively
            for found_file in output_folder.rglob(filename):
                image_path = found_file
                break
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image {filename} not found",
                )

        return FileResponse(image_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving image: {str(e)}",
        )


@app.get("/api/file/{request_id}/{filename}")
async def get_file_from_processing(request_id: str, filename: str):
    """Serve files from a processing task's output folder.

    Args:
        request_id: Task ID or 'current_processing' for active task.
        filename: Name of the file to retrieve.
    """
    try:
        if request_id == "current_processing":
            output_folder = get_current_processing_output_folder()
            if not output_folder:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No active processing task",
                )
        elif request_id in tasks:
            task = tasks[request_id]
            if not task.output_folder:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No output folder for this task",
                )
            output_folder = Path(task.output_folder)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {request_id} not found",
            )

        file_path = output_folder / filename

        # Security check - ensure file is within output folder
        if not str(file_path.resolve()).startswith(str(output_folder.resolve())):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )

        if file_path.exists():
            return FileResponse(file_path)

        # Try to find the file recursively
        for found_file in output_folder.rglob(filename):
            return FileResponse(found_file)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {filename} not found",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving file: {str(e)}",
        )


@app.on_event("startup")
async def startup_event():
    """Detect available GPUs on startup"""
    global available_gpus

    try:
        devices = detect_gpus()
        for device in devices:
            if device.type == "cuda":
                props = torch.cuda.get_device_properties(device)
                available_gpus.append({
                    "id": str(device),
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 1),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            else:
                available_gpus.append({
                    "id": str(device),
                    "name": "CPU",
                    "memory_gb": 0,
                    "compute_capability": "N/A"
                })

        logger.info(f"Detected {len(available_gpus)} device(s): {[g['id'] for g in available_gpus]}")
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        # Fallback to single CPU device
        available_gpus = [{"id": "cpu", "name": "CPU", "memory_gb": 0, "compute_capability": "N/A"}]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
