from fastapi import FastAPI, HTTPException, BackgroundTasks
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
from PIL import Image
from io import BytesIO
import base64
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import time
import asyncio
from typing import Dict, Any
from threading import Event

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
pipe = None
device = None
task_cancellation_events = {}

# Task storage with timestamps
task_storage: Dict[str, Dict[str, Any]] = {}
image_storage: Dict[str, str] = {}

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code - declare globals at the beginning
    global pipe, device
    
    logger.info("Starting model loading process")
    base_model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Use float32 for CPU, float16 for CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            use_safetensors=True
        )

        unet_base = pipe.unet
        lora_model_id = "AryanMakadiya/pokemon_lora"
        unet = PeftModel.from_pretrained(unet_base, lora_model_id)
        pipe.unet = unet
        pipe = pipe.to(device)
        logger.info("Model loaded successfully")
        
        # Start the background cleanup task
        asyncio.create_task(cleanup_expired_tasks())
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield
    
    # Shutdown code
    logger.info("Shutting down and cleaning up resources")
    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Background task to periodically clean up expired tasks
async def cleanup_expired_tasks():
    while True:
        try:
            current_time = time.time()
            expired_tasks = []
            
            # Find tasks older than 1 hour (3600 seconds)
            for task_id, task_data in task_storage.items():
                if current_time - task_data.get("created_at", current_time) > 3600:
                    expired_tasks.append(task_id)
            
            # Remove expired tasks
            for task_id in expired_tasks:
                if task_id in task_storage:
                    del task_storage[task_id]
                if task_id in image_storage:
                    del image_storage[task_id]
                logger.info(f"Cleaned up expired task {task_id}")
            
            # Run cleanup every 10 minutes
            await asyncio.sleep(600)
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute if an error occurs

# Define request model
class PromptRequest(BaseModel):
    prompt: str

# Initialize FastAPI with the lifespan manager
app = FastAPI(title="Pokemon Image Generator API", lifespan=lifespan)

# Increase timeout for long-running operations
app.router.default_timeout = 1800.0  # 30 minutes in seconds

# Add CORS middleware to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "online", "model": "loaded" if pipe is not None else "not_loaded"}

# Original direct image generation endpoint (kept for compatibility)
@app.post("/generate-image")
def generate_image(request: PromptRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a moment.")
    
    logger.info(f"Generating image for prompt: {request.prompt}")
    try:
        import time
        start_time = time.time()
        
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                image = pipe(
                    request.prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
        else:
            # No autocast for CPU
            image = pipe(
                request.prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Log the total generation time
        end_time = time.time()
        generation_time = end_time - start_time
        logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
        
        return {"image": img_str}
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoints for async workflow
@app.post("/start-generation")
async def start_generation(request: PromptRequest, background_tasks: BackgroundTasks):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a moment.")
    
    task_id = str(uuid.uuid4())
    
    # Create a cancellation event for this task
    cancel_event = Event()
    task_cancellation_events[task_id] = cancel_event
    
    task_storage[task_id] = {
        "status": "processing", 
        "prompt": request.prompt,
        "created_at": time.time()
    }
    
    # Run image generation in the background with cancellation event
    background_tasks.add_task(generate_image_task, task_id, request.prompt, cancel_event)
    
    logger.info(f"Started task {task_id} for prompt: {request.prompt}")
    return {"task_id": task_id}

def generate_image_task(task_id: str, prompt: str, cancel_event: Event):
    try:
        start_time = time.time()
        logger.info(f"Processing task {task_id} for prompt: {prompt}")
        
        # Use existing image generation code with cancellation checks
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                # Create pipeline object with num_inference_steps set
                pipe_output = pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    callback=lambda step, *_: cancel_event.is_set()  # Will stop if True
                )
                # Check for cancellation
                if cancel_event.is_set():
                    logger.info(f"Task {task_id} was cancelled")
                    if task_id in task_storage:
                        task_storage[task_id]["status"] = "cancelled"
                    return
                
                image = pipe_output.images[0]
        else:
            # CPU version with cancellation check
            pipe_output = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                callback=lambda step, *_: cancel_event.is_set()  # Will stop if True
            )
            # Check for cancellation
            if cancel_event.is_set():
                logger.info(f"Task {task_id} was cancelled")
                if task_id in task_storage:
                    task_storage[task_id]["status"] = "cancelled"
                return
                
            image = pipe_output.images[0]
        
        # Convert and store image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Store result
        image_storage[task_id] = img_str
        task_storage[task_id]["status"] = "completed"
        
        # Log the total generation time
        end_time = time.time()
        generation_time = end_time - start_time
        logger.info(f"Task {task_id} completed successfully in {generation_time:.2f} seconds")
    except Exception as e:
        if cancel_event.is_set():
            logger.info(f"Task {task_id} was cancelled")
            if task_id in task_storage:
                task_storage[task_id]["status"] = "cancelled"
        else:
            logger.error(f"Task {task_id} failed: {str(e)}")
            if task_id in task_storage:
                task_storage[task_id]["status"] = "failed"
                task_storage[task_id]["error"] = str(e)
    finally:
        # Clean up the cancellation event
        if task_id in task_cancellation_events:
            del task_cancellation_events[task_id]

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "status": task_storage[task_id]["status"],
        "error": task_storage[task_id].get("error", None)
    }

@app.get("/get-image/{task_id}")
async def get_generated_image(task_id: str):
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_storage[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Image generation not completed")
    
    if task_id not in image_storage:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return {"image": image_storage[task_id], "prompt": task_storage[task_id]["prompt"]}

@app.delete("/cleanup-task/{task_id}")
async def cleanup_task(task_id: str):
    """Manually clean up a task and attempt to cancel it if running"""
    found = False
    
    # Signal cancellation if the task is still running
    if task_id in task_cancellation_events:
        task_cancellation_events[task_id].set()
        found = True
        logger.info(f"Signaled cancellation for task {task_id}")
    
    if task_id in task_storage:
        # Update status if still processing
        if task_storage[task_id]["status"] == "processing":
            task_storage[task_id]["status"] = "cancelled"
        del task_storage[task_id]
        found = True
        
    if task_id in image_storage:
        del image_storage[task_id]
        found = True
        
    if found:
        logger.info(f"Manually cleaned up task {task_id}")
        return {"status": "cleaned"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/active-tasks")
async def get_active_tasks():
    """Get information about currently active tasks"""
    task_count = len(task_storage)
    processing_tasks = sum(1 for t in task_storage.values() if t.get("status") == "processing")
    completed_tasks = sum(1 for t in task_storage.values() if t.get("status") == "completed")
    failed_tasks = sum(1 for t in task_storage.values() if t.get("status") == "failed")
    
    return {
        "total": task_count,
        "processing": processing_tasks,
        "completed": completed_tasks,
        "failed": failed_tasks,
        "image_storage_size": len(image_storage)
    }
