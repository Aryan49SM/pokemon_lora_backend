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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
pipe = None
device = None

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
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
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield
    
    # Shutdown code
    logger.info("Shutting down and cleaning up resources")
    global pipe
    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Define request model
class PromptRequest(BaseModel):
    prompt: str

# Initialize FastAPI with the lifespan manager
app = FastAPI(title="Pokemon Image Generator API", lifespan=lifespan)

# a very long timeout limit (30 minutes)
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

@app.post("/generate-image")
async def generate_image(request: PromptRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a moment.")
    
    logger.info(f"Generating image for prompt: {request.prompt}")
    try:
        # Log the start time for monitoring long operations
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
