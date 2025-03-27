"""Image Segmentation API Service

This API provides endpoints for processing image folders with a segmentation model.

Usage Examples:
    1. Submit a folder for processing:
        ```bash
        curl -X POST "http://localhost:8000/api/process" \\
            -H "Content-Type: application/json" \\
            -d '{
                "input_folder": "/path/to/images",
                "save_overlay": true,
                "use_tta": true,
                "device": "cuda:0",
                "callback_url": "http://your-callback-url/webhook"
            }'
        ```
        Response:
        ```json
        {
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "pending",
            "input_folder": "/path/to/images",
            "created_at": "2024-01-01T12:00:00.000Z"
        }
        ```

    2. Check processing status:
        ```bash
        curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
        ```
        Response:
        ```json
        {
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "completed",
            "input_folder": "/path/to/images",
            "output_folders": [
                "/path/to/images_mask",
                "/path/to/images_overlay"
            ],
            "created_at": "2024-01-01T12:00:00.000Z",
            "started_at": "2024-01-01T12:00:01.000Z",
            "completed_at": "2024-01-01T12:05:00.000Z"
        }
        ```

    3. Check queue length:
        ```bash
        curl "http://localhost:8000/api/queue/length"
        ```
        Response:
        ```json
        {
            "queue_size": 2
        }
        ```

Output Structure:
    - For each input folder 'X', creates:
        - X_mask/: Contains segmentation masks
        - X_overlay/: Contains overlay visualizations (if save_overlay=true)
    - Maintains original folder hierarchy in output folders
"""

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Deque
import torch
import logging
import threading
from queue import Queue
from dataclasses import dataclass, asdict, field
from contextlib import asynccontextmanager
from functools import lru_cache
from collections import deque
import os
import json
import time
import requests


from fastapi import FastAPI, HTTPException, status, APIRouter
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import load_model, process_folder_recursive


# Custom log handler to capture logs in memory
class MemoryLogHandler(logging.Handler):
    def __init__(self, max_logs=1000):
        super().__init__()
        self.log_records = deque(maxlen=max_logs)
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def emit(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": self.formatter.format(record),
            "source": record.name,
        }
        self.log_records.append(log_entry)

    def get_logs(self, limit=100, level=None):
        """Return the most recent logs up to limit, optionally filtered by level"""
        if level is None:
            return list(self.log_records)[-limit:]
        else:
            filtered = [
                log for log in self.log_records if log["level"] == level.upper()
            ]
            return filtered[-limit:]

    def clear(self):
        """Clear all log records"""
        self.log_records.clear()


# Configure logging only once
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create memory log handler and add it to the root logger
memory_log_handler = MemoryLogHandler()
logging.getLogger().addHandler(memory_log_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No background task for checking stalled tasks
    try:
        # Load task history on startup
        load_task_history()

        # Yield control back to FastAPI
        yield
    finally:
        # Clean up
        # Save task history on shutdown
        save_task_history()

        # Signal worker thread to exit
        task_queue.put(None)
        worker.join(timeout=5)


# Create main app
app = FastAPI(title="Image Segmentation API", lifespan=lifespan)

# Create the API router with /api prefix
api_router = APIRouter(prefix="/api")

# Add CORS middleware to ensure frontend can access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
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
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Get the device from environment variable (set by run.py)
DEFAULT_DEVICE = os.environ.get("SELECTED_DEVICE", "cuda:0")


class ProcessRequest(BaseModel):
    input_folder: str
    save_overlay: bool = True
    use_tta: bool = True
    device: str = DEFAULT_DEVICE
    callback_url: Optional[str] = None


class RequestResponse(BaseModel):
    request_id: str
    status: RequestStatus
    input_folder: str
    output_folders: List[str] = []
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ProcessingTask:
    request_id: str
    input_folder: Path
    save_overlay: bool
    use_tta: bool
    device: str
    callback_url: Optional[str] = None
    status: RequestStatus = RequestStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_folders: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def update(self, **kwargs) -> "ProcessingTask":
        """Create a new task with updated fields"""
        task_dict = asdict(self)
        task_dict.update(kwargs)
        return ProcessingTask(**task_dict)


# Global state
tasks: Dict[str, ProcessingTask] = {}
task_queue = Queue()
processing_lock = threading.RLock()  # Reentrant lock for nested acquire
model = None
model_lock = threading.RLock()

# Path for persistent task storage
TASK_HISTORY_FILE = Path("task_history.json")


# Function to load task history from disk
def load_task_history():
    global tasks
    try:
        if TASK_HISTORY_FILE.exists():
            with open(TASK_HISTORY_FILE, "r") as f:
                task_data = json.load(f)
                loaded_task_count = 0
                fixed_task_count = 0

                for task_id, task_dict in task_data.items():
                    # Convert back to Path object for input_folder
                    task_dict["input_folder"] = Path(task_dict["input_folder"])

                    # Fix inconsistent states - any tasks that were previously processing/pending
                    # should be marked as failed since they were interrupted
                    if task_dict["status"] in [
                        "pending",
                        "processing",
                        RequestStatus.PENDING,
                        RequestStatus.PROCESSING,
                    ]:
                        logger.warning(
                            f"Found interrupted task {task_id} with status {task_dict['status']}, marking as failed"
                        )
                        task_dict["status"] = RequestStatus.FAILED
                        task_dict["completed_at"] = datetime.now().isoformat()
                        task_dict["error_message"] = (
                            "Task was interrupted by server restart"
                        )
                        fixed_task_count += 1

                    # Convert string status to enum if needed
                    if isinstance(task_dict["status"], str):
                        task_dict["status"] = RequestStatus(task_dict["status"])

                    tasks[task_id] = ProcessingTask(**task_dict)
                    loaded_task_count += 1

                logger.info(
                    f"Loaded {loaded_task_count} tasks from history file ({fixed_task_count} fixed)"
                )

                # Save immediately if we fixed any tasks
                if fixed_task_count > 0:
                    save_task_history()
        else:
            logger.info("No task history file found, starting with empty task list")
    except Exception as e:
        logger.error(f"Error loading task history: {e}", exc_info=True)
        # Create a backup of the corrupted file if it exists
        if TASK_HISTORY_FILE.exists():
            backup_file = TASK_HISTORY_FILE.with_suffix(".json.bak")
            try:
                import shutil

                shutil.copy(TASK_HISTORY_FILE, backup_file)
                logger.info(
                    f"Created backup of corrupted task history at {backup_file}"
                )
            except Exception as backup_err:
                logger.error(f"Failed to create backup: {backup_err}")

        # Start with empty tasks dictionary
        tasks = {}


# Function to save task history to disk
def save_task_history():
    try:
        with processing_lock:
            task_data = {}
            for task_id, task in tasks.items():
                task_dict = asdict(task)
                # Convert Path to string for JSON serialization
                task_dict["input_folder"] = str(task_dict["input_folder"])
                task_data[task_id] = task_dict

            with open(TASK_HISTORY_FILE, "w") as f:
                json.dump(task_data, f, indent=2)
        logger.info(f"Saved {len(tasks)} tasks to history file")
    except Exception as e:
        logger.error(f"Error saving task history: {e}")


# Global variables
app_state = {
    "initialized": False,
    "model_loading": False,
    "device": None,
    "model": None,
    "worker_thread": None,
    "task_queue": [],
    "stop_event": threading.Event(),
    "version": "1.0.0",
    "last_history_save": 0,
}

# Create a global task queue
task_queue = Queue()

# Global processing lock to ensure thread safety
processing_lock = threading.RLock()


# Setup device cache
@lru_cache(maxsize=8)
def get_device(device_str: str) -> torch.device:
    """Get cached torch device for inference"""
    return torch.device(device_str)


# Setup callback function
def send_callback(url: str, data: dict) -> None:
    """Send HTTP callback to the provided URL with the task data"""
    try:
        requests.post(url, json=data, timeout=10)
        logger.info(f"Callback sent successfully to {url}")
    except Exception as e:
        logger.error(f"Failed to send callback to {url}: {e}")


def worker_thread() -> None:
    """Background worker that processes the task queue"""
    global model

    logger.info("Worker thread started, waiting for tasks...")

    while True:
        task: ProcessingTask = task_queue.get()

        if task is None:  # Exit signal
            logger.info("Worker thread received exit signal, shutting down...")
            break

        # Update task status
        with processing_lock:
            tasks[task.request_id] = task.update(
                status=RequestStatus.PROCESSING, started_at=datetime.now().isoformat()
            )

        logger.info(
            f"Starting task {task.request_id}: processing folder {task.input_folder}"
        )

        try:
            input_folder = task.input_folder
            if not input_folder.exists() or not input_folder.is_dir():
                raise ValueError(f"Input folder does not exist: {input_folder}")

            # Get cached device
            device = get_device(task.device)
            logger.info(f"Using device: {device}")

            # Lazy-load model
            with model_lock:
                if model is None:
                    model_path = Path("Mvanet_complete.pth")
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    logger.info(f"Loading model from {model_path}")
                    # Use the default device from run.py for model loading
                    default_device = get_device(DEFAULT_DEVICE)
                    model = load_model(model_path, default_device)
                    logger.info(f"Model loaded successfully on {default_device}")

            # Create output paths
            mask_output_folder = input_folder.parent / (input_folder.name + "_mask")
            output_folders = [str(mask_output_folder)]

            if task.save_overlay:
                overlay_output_folder = input_folder.parent / (
                    input_folder.name + "_overlay"
                )
                output_folders.append(str(overlay_output_folder))

            logger.info(
                f"Processing with options: save_overlay={task.save_overlay}, use_tta={task.use_tta}"
            )
            logger.info(f"Output folders: {', '.join(output_folders)}")

            # Process folder
            process_folder_recursive(
                input_folder,
                model,
                device,
                save_overlay=task.save_overlay,
                use_tta=task.use_tta,
            )

            logger.info(f"Task {task.request_id} completed successfully")

            # Update task as completed
            completed_task = task.update(
                status=RequestStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                output_folders=output_folders,
            )

            with processing_lock:
                tasks[task.request_id] = completed_task
                # Double-check the task was stored with the correct status
                logger.info(
                    f"Task {task.request_id} updated with status: {tasks[task.request_id].status}"
                )
                if tasks[task.request_id].status != RequestStatus.COMPLETED:
                    logger.warning(
                        f"Task status mismatch! Expected {RequestStatus.COMPLETED}, got {tasks[task.request_id].status}"
                    )
                    # Force correct enum value if needed
                    task_dict = asdict(tasks[task.request_id])
                    task_dict["status"] = RequestStatus.COMPLETED
                    tasks[task.request_id] = ProcessingTask(**task_dict)

                # Update task history periodically
                current_time = time.time()
                if (
                    current_time - app_state["last_history_save"] > 60
                ):  # Save every minute
                    save_task_history()
                    app_state["last_history_save"] = current_time

            # Send callback if provided (non-blocking)
            if task.callback_url:
                logger.info(f"Sending completion callback to {task.callback_url}")
                threading.Thread(
                    target=send_callback,
                    args=(task.callback_url, asdict(completed_task)),
                    daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Error processing task {task.request_id}: {e}", exc_info=True)

            # Update task as failed
            failed_task = task.update(
                status=RequestStatus.FAILED,
                completed_at=datetime.now().isoformat(),
                error_message=str(e),
            )

            with processing_lock:
                tasks[task.request_id] = failed_task

                # Save history when a task fails
                save_task_history()

            # Send error callback if provided (non-blocking)
            if task.callback_url:
                logger.info(f"Sending error callback to {task.callback_url}")
                threading.Thread(
                    target=send_callback,
                    args=(task.callback_url, asdict(failed_task)),
                    daemon=True,
                ).start()

        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                logger.info("Cleaning up GPU memory")
                torch.cuda.empty_cache()
            task_queue.task_done()
            logger.info(f"Queue size: {task_queue.qsize()}")


# Start worker thread
worker = threading.Thread(target=worker_thread, daemon=True)
worker.start()


# API endpoints with APIRouter
@api_router.post(
    "/process", response_model=RequestResponse, status_code=status.HTTP_202_ACCEPTED
)
async def process_folder(request: ProcessRequest) -> RequestResponse:
    """Submit a folder for image segmentation processing"""
    request_id = str(uuid.uuid4())

    logger.info(
        f"New processing request {request_id} for folder: {request.input_folder}"
    )
    logger.info(
        f"Request options: save_overlay={request.save_overlay}, use_tta={request.use_tta}, device={request.device}"
    )

    # Use the device from the request, or fall back to the default device
    device = request.device if request.device else DEFAULT_DEVICE

    task = ProcessingTask(
        request_id=request_id,
        input_folder=Path(request.input_folder),
        save_overlay=request.save_overlay,
        use_tta=request.use_tta,
        device=device,
        callback_url=request.callback_url,
    )

    with processing_lock:
        tasks[request_id] = task

    task_queue.put(task)

    # Save task history after adding new task
    save_task_history()

    return RequestResponse(
        request_id=request_id,
        status=RequestStatus.PENDING,
        input_folder=request.input_folder,
        created_at=task.created_at,
    )


@api_router.get("/status/{request_id}", response_model=RequestResponse)
async def get_status(request_id: str) -> RequestResponse:
    """Get the status of a processing request by its ID"""
    logger.debug(f"Status check for request: {request_id}")

    with processing_lock:
        if request_id not in tasks:
            logger.warning(f"Request ID not found: {request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request with ID {request_id} not found",
            )
        task = tasks[request_id]

    return RequestResponse(
        request_id=task.request_id,
        status=task.status,
        input_folder=str(task.input_folder),
        output_folders=task.output_folders,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )


@api_router.get("/queue/length")
async def get_queue_length() -> dict:
    """Get the current length of the processing queue."""
    with processing_lock:
        return {
            "queue_size": task_queue.qsize(),
            "processing": sum(
                1 for task in tasks.values() if task.status == RequestStatus.PROCESSING
            ),
            "pending": sum(
                1 for task in tasks.values() if task.status == RequestStatus.PENDING
            ),
            "timestamp": datetime.now().isoformat(),
        }


@api_router.get("/logs")
async def get_logs(limit: int = 100, level: Optional[str] = None) -> dict:
    """Get system logs with optional filtering."""
    logs = memory_log_handler.get_logs(limit=limit, level=level)
    return {"logs": logs}


@api_router.post("/logs/clear")
async def clear_logs() -> dict:
    """Clear all stored logs"""
    logger.info("Clearing all logs")
    memory_log_handler.clear()
    return {"status": "success", "message": "Logs cleared"}


@api_router.get("/images/{request_id}/{image_type}/{index}")
@api_router.head("/images/{request_id}/{image_type}/{index}")
async def get_task_image(request_id: str, image_type: str, index: int):
    """
    Get an image from a task's output folder.
    """
    try:
        # Find the task
        with processing_lock:
            if request_id not in tasks:
                logger.warning(f"Request ID not found for image access: {request_id}")
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Task with request_id {request_id} not found"}
                )
            task = tasks[request_id]
        
        # Check if the task is completed
        is_completed = task.status == RequestStatus.COMPLETED or task.status == "completed"
        if not is_completed:
            logger.warning(f"Attempted to access images for non-completed task: {request_id}, status: {task.status}")
            return JSONResponse(
                status_code=404,
                content={"detail": f"Task with request_id {request_id} is not completed (current status: {task.status})"}
            )
        
        # Get the appropriate output folder
        folder_type = "overlay" if image_type == "overlay" else "mask"
        output_folder = None
        for folder in task.output_folders:
            if folder_type in folder:
                output_folder = folder
                break
        
        if not output_folder:
            logger.warning(f"No {folder_type} folder found for task {request_id}")
            return JSONResponse(
                status_code=404,
                content={"detail": f"No {folder_type} folder found for task {request_id}"}
            )
        
        # Log the output folder for debugging
        logger.info(f"Looking for images in folder: {output_folder}")
        
        # Check if this is a UNC path
        is_unc_path = output_folder.startswith('\\\\')
        if is_unc_path:
            logger.info(f"Detected UNC path: {output_folder}")
        
        # Get all image files in the folder
        image_files = []
        camera_folders = set()
        try:
            output_path = Path(output_folder)
            
            # Check for subfolders (camera folders)
            subfolders = [d for d in output_path.iterdir() if d.is_dir()]
            
            # If we have subfolders and index is 0, return the first image from each subfolder
            if subfolders and index == 0:
                logger.info(f"Found {len(subfolders)} camera subfolders")
                
                # Get the first image from each subfolder
                for subfolder in subfolders:
                    camera_folders.add(subfolder.name)
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]:
                        subfolder_images = list(subfolder.glob(ext)) + list(subfolder.glob(ext.upper()))
                        if subfolder_images:
                            image_files.append(subfolder_images[0])
                            break
            else:
                # Search recursively through all subfolders
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]:
                    # Search in the main folder
                    image_files.extend(list(output_path.glob(ext)))
                    image_files.extend(list(output_path.glob(ext.upper())))
                    
                    # Search recursively in all subfolders
                    image_files.extend(list(output_path.glob(f"**/{ext}")))
                    image_files.extend(list(output_path.glob(f"**/{ext.upper()}")))
                    
                    # Collect camera folder names
                    for img_path in output_path.glob(f"**/{ext}"):
                        if img_path.parent.name.startswith("camera_"):
                            camera_folders.add(img_path.parent.name)
            
            image_files.sort()
            logger.info(f"Found {len(image_files)} images in {output_folder} (including subfolders)")
            logger.info(f"Camera folders: {', '.join(camera_folders)}")
        except Exception as e:
            logger.error(f"Error listing files in {output_folder}: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error listing files in output folder: {str(e)}"}
            )
        
        if not image_files:
            logger.warning(f"No images found in folder: {output_folder}")
            return JSONResponse(
                status_code=404,
                content={"detail": f"No images found in {folder_type} folder"}
            )
        
        if index >= len(image_files):
            logger.warning(f"Image index {index} out of range (max: {len(image_files)-1})")
            return JSONResponse(
                status_code=404,
                content={"detail": f"No image with index {index} found in {folder_type} folder (max: {len(image_files)-1})"}
            )
        
        requested_image = image_files[index]
        logger.info(f"Serving image: {requested_image}")
        
        # Try to access the file
        try:
            # Check if file exists and is accessible
            if not os.path.exists(requested_image):
                logger.error(f"Image file exists in directory listing but not on disk: {requested_image}")
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Image file not accessible: {requested_image}"}
                )
                
            # Create response with cache control headers
            file_response = FileResponse(requested_image)
            file_response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            file_response.headers["Pragma"] = "no-cache"
            file_response.headers["Expires"] = "0"
            
            # Add camera folder information to headers
            file_response.headers["X-Total-Images"] = str(len(image_files))
            file_response.headers["X-Camera-Folders"] = ",".join(camera_folders)
            file_response.headers["X-Camera-Count"] = str(len(camera_folders))
            
            # Add current camera folder to headers if applicable
            current_camera = None
            for folder in camera_folders:
                if folder in str(requested_image):
                    current_camera = folder
                    break
            
            if current_camera:
                file_response.headers["X-Current-Camera"] = current_camera
                
                # Find the index of the current camera in the sorted list of camera folders
                sorted_cameras = sorted(list(camera_folders))
                camera_index = sorted_cameras.index(current_camera) if current_camera in sorted_cameras else -1
                file_response.headers["X-Camera-Index"] = str(camera_index)
            
            return file_response
        except Exception as file_error:
            logger.error(f"Error accessing file {requested_image}: {file_error}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error accessing image file: {str(file_error)}"}
            )
            
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error serving image: {str(e)}"}
        )


# New endpoint to get all tasks
@api_router.get("/tasks", response_model=List[RequestResponse])
async def get_all_tasks(
    limit: int = 50,
    status: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> List[RequestResponse]:
    """Get all processing tasks with optional filtering and sorting"""
    logger.debug(
        f"Task history requested: limit={limit}, status={status}, sort_by={sort_by}, sort_order={sort_order}"
    )

    with processing_lock:
        # Convert tasks to list of dictionaries
        task_list = []
        for task_id, task in tasks.items():
            task_dict = {
                "request_id": task.request_id,
                "status": task.status,
                "input_folder": str(task.input_folder),
                "output_folders": task.output_folders,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error_message": task.error_message,
            }

            # Apply status filter if provided
            if status and task_dict["status"] != status:
                continue

            task_list.append(task_dict)

        # Sort the list
        reverse = sort_order.lower() == "desc"
        try:
            sorted_tasks = sorted(
                task_list, key=lambda x: x.get(sort_by, ""), reverse=reverse
            )
        except (KeyError, TypeError):
            logger.warning(f"Invalid sort key: {sort_by}, falling back to created_at")
            sorted_tasks = sorted(
                task_list, key=lambda x: x.get("created_at", ""), reverse=reverse
            )

        # Apply limit
        return sorted_tasks[:limit]


# New endpoint to delete a task
@api_router.delete("/tasks/{request_id}")
async def delete_task(request_id: str):
    """Delete a task from history by ID"""
    logger.info(f"Request to delete task: {request_id}")

    with processing_lock:
        if request_id not in tasks:
            logger.warning(f"Task not found for deletion: {request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {request_id} not found",
            )

        # Check if task is currently processing
        if tasks[request_id].status == RequestStatus.PROCESSING:
            logger.warning(
                f"Cannot delete task {request_id} that is currently processing"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete a task that is currently processing",
            )

        # Remove the task
        del tasks[request_id]

        # Save changes to disk
        save_task_history()

        return {
            "status": "success",
            "message": f"Task {request_id} deleted successfully",
        }


# New endpoint to clear task history
@api_router.delete("/tasks")
async def clear_task_history(status: Optional[str] = None, keep_recent: int = 0):
    """
    Clear task history with optional filtering

    Args:
        status: Only delete tasks with this status
        keep_recent: Number of recent tasks to keep (sorted by created_at)
    """
    logger.info(
        f"Request to clear task history: status={status}, keep_recent={keep_recent}"
    )

    with processing_lock:
        # Check if any tasks are currently processing
        if status is None and any(
            task.status == RequestStatus.PROCESSING for task in tasks.values()
        ):
            logger.warning("Cannot clear all history while tasks are processing")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot clear history while tasks are processing. Use status filter to delete only completed or failed tasks.",
            )

        # Filter tasks to delete
        if status is not None:
            # Convert status string to enum if needed
            target_status = RequestStatus(status) if isinstance(status, str) else status
            task_ids_to_delete = [
                task_id
                for task_id, task in tasks.items()
                if task.status == target_status
            ]
        else:
            # Get all task IDs except processing ones
            task_ids_to_delete = list(tasks.keys())

        # Keep recent tasks if requested
        if keep_recent > 0:
            # Sort tasks by created_at
            recent_tasks = sorted(
                [
                    (task_id, tasks[task_id].created_at)
                    for task_id in task_ids_to_delete
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            # Keep the most recent ones
            task_ids_to_keep = [task_id for task_id, _ in recent_tasks[:keep_recent]]
            # Remove these from the deletion list
            task_ids_to_delete = [
                task_id
                for task_id in task_ids_to_delete
                if task_id not in task_ids_to_keep
            ]

        # Delete tasks
        for task_id in task_ids_to_delete:
            del tasks[task_id]

        # Save changes to disk
        save_task_history()

        return {
            "status": "success",
            "message": f"Deleted {len(task_ids_to_delete)} tasks from history",
            "deleted_count": len(task_ids_to_delete),
        }


# New endpoint to abort a task
@api_router.post("/tasks/{request_id}/abort")
async def abort_task(request_id: str):
    """Abort a pending or processing task by updating its status to failed"""
    logger.info(f"Request to abort task: {request_id}")

    with processing_lock:
        if request_id not in tasks:
            logger.warning(f"Task not found for aborting: {request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {request_id} not found",
            )

        task = tasks[request_id]

        # Only abort tasks that are pending or processing
        if task.status not in [RequestStatus.PENDING, RequestStatus.PROCESSING]:
            logger.warning(f"Cannot abort task {request_id} with status {task.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot abort a task that is already {task.status}",
            )

        # Update the task status
        aborted_task = task.update(
            status=RequestStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error_message="Task aborted by user",
        )

        tasks[request_id] = aborted_task

        # Save changes to disk immediately
        save_task_history()

        return {
            "status": "success",
            "message": f"Task {request_id} aborted successfully",
        }


@api_router.get("/device/info")
async def get_device_info():
    """Get information about the currently selected device and available devices"""
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.extend(
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )

    # Get current device info
    current_device = DEFAULT_DEVICE
    device_properties = {}

    if current_device.startswith("cuda") and torch.cuda.is_available():
        try:
            device_idx = int(current_device.split(":")[-1])
            if device_idx < torch.cuda.device_count():
                props = torch.cuda.get_device_properties(device_idx)
                device_properties = {
                    "name": props.name,
                    "total_memory": f"{props.total_memory / (1024**3):.2f} GB",
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }

                # Add current memory usage
                if hasattr(torch.cuda, "memory_allocated"):
                    allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                    device_properties["allocated_memory"] = f"{allocated:.2f} GB"
                    device_properties["reserved_memory"] = f"{reserved:.2f} GB"
        except Exception as e:
            logger.error(f"Error getting CUDA device properties: {e}")

    return {
        "current_device": current_device,
        "available_devices": available_devices,
        "device_properties": device_properties,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }


# Include the API router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
