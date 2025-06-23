import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
import torch
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference import load_model, process_folder_recursive


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
    use_tta: bool = True


class ProcessResponse(BaseModel):
    request_id: str
    status: RequestStatus
    output_folder: str | None = None
    error_message: str | None = None
    input_folder: str | None = None
    use_tta: bool | None = None


class SystemStatus(BaseModel):
    is_processing: bool
    current_task_id: str | None = None
    current_input_folder: str | None = None


# Global state
tasks: dict[str, ProcessResponse] = {}
model = None
device = torch.device(os.environ.get("SELECTED_DEVICE", "cuda:0"))

# Processing state tracking
processing_lock = threading.Lock()
current_processing_task: str | None = None
current_input_folder: str | None = None

# Background executor for processing tasks
executor = ThreadPoolExecutor(max_workers=1)


def load_model_once():
    """Load model once on first use"""
    global model
    if model is None:
        model_path = Path("models/MVANet.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.info(f"Loading model from {model_path}")
        model = load_model(device, model_path)
        logger.info(f"Model loaded successfully on {device}")


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


def background_process_task(request_id: str, input_folder_str: str, use_tta: bool):
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

        # Load model if needed
        log_with_task_id("Loading model...", "INFO", request_id)
        load_model_once()
        log_with_task_id("Model loaded successfully", "INFO", request_id)

        # Process folder
        input_folder = Path(input_folder_str)
        overlay_output_folder = input_folder.parent / (input_folder.name + "_overlay")

        # Update task with output folder path immediately
        if request_id in tasks:
            tasks[request_id].output_folder = str(overlay_output_folder)

        log_with_task_id(
            f"Starting image processing for folder: {input_folder}", "INFO", request_id
        )
        log_with_task_id(
            f"Output will be saved to: {overlay_output_folder}", "INFO", request_id
        )
        log_with_task_id(
            f"Test-Time Augmentation: {'Enabled' if use_tta else 'Disabled'}",
            "INFO",
            request_id,
        )

        # Hook into the processing function to capture progress
        original_logger = logging.getLogger("src.inference")

        class TaskLoggerAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add task_id to all log records
                record = kwargs.get("extra", {})
                record["task_id"] = request_id
                kwargs["extra"] = record
                return msg, kwargs

        # Replace the inference logger temporarily
        task_logger = TaskLoggerAdapter(original_logger, {"task_id": request_id})

        # Patch the logging in inference module
        import src.inference as inference_module

        original_logging = inference_module.logging

        class TaskLoggingModule:
            def info(self, msg):
                log_with_task_id(msg, "INFO", request_id)

            def warning(self, msg):
                log_with_task_id(msg, "WARNING", request_id)

            def error(self, msg):
                log_with_task_id(msg, "ERROR", request_id)

        inference_module.logging = TaskLoggingModule()

        try:
            process_folder_recursive(
                input_folder,
                model,
                device,
                use_tta=use_tta,
            )
        finally:
            # Restore original logging
            inference_module.logging = original_logging

        log_with_task_id("Processing completed successfully!", "INFO", request_id)

        # Update task as completed
        tasks[request_id] = ProcessResponse(
            request_id=request_id,
            status=RequestStatus.COMPLETED,
            output_folder=str(overlay_output_folder),
            input_folder=input_folder_str,
            use_tta=use_tta,
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
            use_tta=use_tta,
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
    """Get current system processing status"""
    with processing_lock:
        return SystemStatus(
            is_processing=current_processing_task is not None,
            current_task_id=current_processing_task,
            current_input_folder=current_input_folder,
        )


@app.get("/api/logs/{task_id}")
async def get_task_logs(task_id: str, since: str = None):
    """Get logs for a specific task"""
    try:
        logs = memory_handler.get_task_logs(task_id, since)
        return {"task_id": task_id, "logs": logs, "total_logs": len(logs)}
    except Exception as e:
        logger.error(f"Error fetching logs for task {task_id}: {e}")
        return JSONResponse(
            status_code=500, content={"detail": f"Error fetching logs: {str(e)}"}
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
            use_tta=request.use_tta,
        )

        # Submit background task
        executor.submit(
            background_process_task, request_id, request.input_folder, request.use_tta
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
            use_tta=request.use_tta,
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


@app.get("/api/latest-image")
async def get_latest_processed_image():
    """Get the most recently saved processed image from any output folder"""
    try:
        # Find all tasks that have output folders (including currently processing ones)
        tasks_with_output = [
            (task_id, task) for task_id, task in tasks.items() if task.output_folder
        ]

        if not tasks_with_output:
            # Check if there's a current processing task that might have images
            global current_input_folder
            if current_input_folder:
                try:
                    input_path = Path(current_input_folder)
                    potential_output = input_path.parent / (
                        input_path.name + "_overlay"
                    )

                    if potential_output.exists():
                        # Find images in this folder
                        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
                        all_images = []

                        for ext in image_extensions:
                            all_images.extend(list(potential_output.glob(f"*{ext}")))
                            all_images.extend(
                                list(potential_output.glob(f"*{ext.upper()}"))
                            )
                            all_images.extend(list(potential_output.glob(f"**/*{ext}")))
                            all_images.extend(
                                list(potential_output.glob(f"**/*{ext.upper()}"))
                            )

                        if all_images:
                            latest_image = max(
                                all_images, key=lambda f: f.stat().st_mtime
                            )

                            return {
                                "task_id": "current_processing",
                                "input_folder": current_input_folder,
                                "output_folder": str(potential_output),
                                "image_name": latest_image.name,
                                "image_path": str(latest_image.relative_to(potential_output)),
                                "image_url": f"/api/file/current_processing/{latest_image.name}",
                                "modified_time": latest_image.stat().st_mtime,
                                "task_status": "processing",
                            }

                except Exception as e:
                    logger.error(f"Error checking current processing folder: {e}")

            return JSONResponse(
                status_code=404,
                content={"detail": "No processed images found"},
            )

        # Find all image files from all output folders
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
        all_image_files = []

        for task_id, task in tasks_with_output:
            output_folder = Path(task.output_folder)
            if not output_folder.exists():
                continue

            # Find all image files in this output folder
            for ext in image_extensions:
                all_image_files.extend(
                    [(f, task_id, task) for f in output_folder.glob(f"*{ext}")]
                )
                all_image_files.extend(
                    [(f, task_id, task) for f in output_folder.glob(f"*{ext.upper()}")]
                )
                all_image_files.extend(
                    [(f, task_id, task) for f in output_folder.glob(f"**/*{ext}")]
                )
                all_image_files.extend(
                    [
                        (f, task_id, task)
                        for f in output_folder.glob(f"**/*{ext.upper()}")
                    ]
                )

        if not all_image_files:
            return JSONResponse(
                status_code=404,
                content={"detail": "No images found in any output folders"},
            )

        # Get the most recently modified image across all output folders
        latest_image_file, associated_task_id, associated_task = max(
            all_image_files, key=lambda x: x[0].stat().st_mtime
        )

        output_folder = Path(associated_task.output_folder)

        return {
            "task_id": associated_task_id,
            "input_folder": associated_task.input_folder,
            "output_folder": str(output_folder),
            "image_name": latest_image_file.name,
            "image_path": str(latest_image_file.relative_to(output_folder)),
            "image_url": f"/api/images/{associated_task_id}/{latest_image_file.name}",
            "modified_time": latest_image_file.stat().st_mtime,
            "task_status": associated_task.status,
        }

    except Exception as e:
        logger.error(f"Error getting latest image: {str(e)}")
        return JSONResponse(
            status_code=500, content={"detail": f"Error getting latest image: {str(e)}"}
        )


@app.get("/api/images/{request_id}/{filename}")
async def get_image_file(request_id: str, filename: str):
    """Get a specific image file"""
    if request_id not in tasks:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Task with request_id {request_id} not found"},
        )

    task = tasks[request_id]

    if not task.output_folder:
        return JSONResponse(
            status_code=404,
            content={"detail": "No output folder for this task"},
        )

    try:
        output_folder = Path(task.output_folder)
        image_path = output_folder / filename

        # Security check - ensure the file is within the output folder
        if not str(image_path.resolve()).startswith(str(output_folder.resolve())):
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"},
            )

        if not image_path.exists():
            # Try to find the file recursively
            for found_file in output_folder.rglob(filename):
                image_path = found_file
                break
            else:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Image {filename} not found"},
                )

        return FileResponse(image_path)

    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return JSONResponse(
            status_code=500, content={"detail": f"Error serving image: {str(e)}"}
        )


@app.get("/api/file/{task_id}/{filename}")
async def get_file_from_current_processing(task_id: str, filename: str):
    """Serve files from current processing output folder"""
    try:
        if task_id == "current_processing":
            global current_input_folder
            if current_input_folder:
                input_path = Path(current_input_folder)
                output_folder = input_path.parent / (input_path.name + "_overlay")
                file_path = output_folder / filename

                # Security check
                if not str(file_path.resolve()).startswith(
                    str(output_folder.resolve())
                ):
                    return JSONResponse(
                        status_code=403, content={"detail": "Access denied"}
                    )

                if file_path.exists():
                    return FileResponse(file_path)

                # Try to find the file recursively
                for found_file in output_folder.rglob(filename):
                    return FileResponse(found_file)

        return JSONResponse(status_code=404, content={"detail": "File not found"})

    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return JSONResponse(
            status_code=500, content={"detail": f"Error serving file: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
