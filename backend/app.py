import os
# Force Level Zero backend and apply the Immediate Command Lists workaround 
# This bypasses "Error 45 (UR_RESULT_ERROR_INVALID_ARGUMENT)" on Intel Iris Xe
os.environ["SYCL_DEVICE_FILTER"] = "level_zero"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["IPEX_OPTIMIZE_TRANSFORMERS"] = "1"
import sys
import threading
import json
import asyncio
import psutil
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# Add root folder to sys.path to resolve backend package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.downloader import check_models_status, download_model, MODEL_DEFINITIONS
from backend.orchestrator import AgentOrchestrator

app = FastAPI(title="Local Multi-Agent XPU System API")

# Allow CORS for React frontend (development and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Cancellation flag — set when user clicks Cancel to stop generation (models stay loaded)
generation_cancel = threading.Event()

# Global orchestrator instance — receives the cancel flag for mid-generation stopping
orchestrator = AgentOrchestrator(cancel_event=generation_cancel)

# Lock and progress tracker for downloads
download_lock = threading.Lock()
download_progress = {}

class ChatRequest(BaseModel):
    prompt: str
    mode: str  # reasoning, coding, writing, searching, auto
    context_length: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.7
    selected_models: Optional[List[str]] = None
    image: Optional[str] = None
    device_mode: Optional[str] = None
    gpu_layers: Optional[int] = None
    enable_web_search: bool = False

class SettingsRequest(BaseModel):
    context_length: int
    max_tokens: int
    temperature: float
    device_mode: Optional[str] = "gpu"
    gpu_layers: Optional[int] = -1
    enable_web_search: bool = False

def bg_download_task(model_key: str):
    global download_progress
    with download_lock:
        download_progress[model_key] = {"status": "downloading", "progress": 0}
        
    try:
        # Perform download
        download_model(model_key)
        with download_lock:
            download_progress[model_key] = {"status": "completed", "progress": 100}
    except Exception as e:
        with download_lock:
            download_progress[model_key] = {"status": "failed", "error": str(e)}

@app.get("/api/status")
def get_system_status():
    """Retrieve system health and model statuses."""
    # Check downloaded models
    models_status = check_models_status()
    
    # Merge with active downloads
    with download_lock:
        for key, dl in download_progress.items():
            if key in models_status:
                models_status[key]["download_task"] = dl

    # System Resources
    ram = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    # Check GPU VRAM if Vulkan is available or active
    gpu_info = "N/A"
    try:
        # Check from llama_cpp if any model is loaded
        loaded_keys = list(orchestrator.loaded_models.keys())
        if loaded_keys:
            gpu_info = f"Active ({len(loaded_keys)} models cached)"
        else:
            gpu_info = "Standby (Vulkan device ready)"
    except Exception:
        pass
        
    return {
        "models": models_status,
        "system": {
            "cpu": f"{cpu_percent}%",
            "ram_used": f"{ram.used / (1024**3):.1f} GB",
            "ram_total": f"{ram.total / (1024**3):.1f} GB",
            "ram_percent": f"{ram.percent}%",
            "gpu": gpu_info
        },
        "settings": {
            "context_length": orchestrator.context_length,
            "max_tokens": orchestrator.max_tokens,
            "temperature": orchestrator.temperature,
            "device_mode": orchestrator.device_mode,
            "gpu_layers": orchestrator.gpu_layers,
            "enable_web_search": getattr(orchestrator, "enable_web_search", False)
        }
    }

@app.post("/api/download/{model_key}")
def trigger_download(model_key: str, background_tasks: BackgroundTasks):
    """Trigger background download of a model."""
    if model_key not in MODEL_DEFINITIONS:
        raise HTTPException(status_code=400, detail="Invalid model key")
        
    status = check_models_status()
    if status[model_key]["downloaded"]:
        return {"status": "already_downloaded"}
        
    with download_lock:
        if model_key in download_progress and download_progress[model_key]["status"] == "downloading":
            return {"status": "in_progress"}
            
    background_tasks.add_task(bg_download_task, model_key)
    return {"status": "started"}

@app.post("/api/settings")
def update_settings(settings: SettingsRequest):
    """Update settings on the orchestrator."""
    orchestrator.update_settings(
        context_length=settings.context_length,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        device_mode=settings.device_mode,
        gpu_layers=settings.gpu_layers,
        enable_web_search=settings.enable_web_search
    )
    return {"status": "updated"}

# Audio transcription endpoint removed — no speech model in the Dual-Core pipeline.

@app.post("/api/cancel")
async def cancel_generation():
    """Stop active generation but keep models loaded in RAM for instant reuse."""
    generation_cancel.set()
    return {"status": "success", "message": "Generation cancelled. Models remain loaded."}

@app.post("/api/offload")
async def offload_memory():
    """Fully unload all models from RAM/VRAM."""
    try:
        # Signal any running generation to stop first
        generation_cancel.set()
        import time
        time.sleep(0.5)
        orchestrator.unload_all_models()
        generation_cancel.clear()
        return {"status": "success", "message": "All models offloaded from memory."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main Chat Endpoint. Streams status updates as agents work,
    and then streams the final output.
    """
    # 1. Update request-specific settings
    orchestrator.update_settings(
        context_length=request.context_length,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        device_mode=request.device_mode if request.device_mode else orchestrator.device_mode,
        gpu_layers=request.gpu_layers if request.gpu_layers is not None else orchestrator.gpu_layers,
        enable_web_search=request.enable_web_search
    )
    
    # 2. Check if required models are downloaded
    models_status = check_models_status()
    needed_models = []
    
    if request.image:
        needed_models += ["qwen_vl"]
        
    needed_models += ["router", "deepseek_r1", "vibethinker", "opencode"]
        
    missing_models = [MODEL_DEFINITIONS[m]["name"] for m in needed_models if not models_status.get(m, {}).get("downloaded", False)]
    
    if missing_models:
        missing_list = ", ".join(missing_models)
        raise HTTPException(
            status_code=400, 
            detail=f"Please download the following required models first: {missing_list}"
        )

    # 3. Stream generator
    async def response_generator():
        try:
            yield json.dumps({"type": "status", "message": "Initiating agent core...", "level": "info", "model": "coordinator", "progress": 0}) + "\n"
            
            import queue
            q = queue.Queue()
            
            def thread_cb(msg, lvl="info", model=None, progress=None):
                payload = {"type": "status", "message": msg, "level": lvl}
                if model:
                    payload["model"] = model
                if progress is not None:
                    payload["progress"] = progress
                q.put(payload)

            def run_orchestrator():
                try:
                    # Clear any stale cancel signal before starting
                    generation_cancel.clear()
                    
                    # If an image was uploaded, run Qwen 2.5 VL vision parsing first
                    final_prompt = request.prompt
                    if request.image:
                        if generation_cancel.is_set():
                            q.put({"type": "error", "message": "Generation cancelled."})
                            q.put(None)
                            return
                        try:
                            ocr_text = orchestrator.transcribe_image(request.image, status_callback=thread_cb)
                            final_prompt = (
                                f"[Transcribed Image Content via Qwen 2.5-VL 7B]:\n"
                                f"{ocr_text}\n\n"
                                f"User Prompt: {request.prompt}"
                            )
                        except Exception as ex:
                            thread_cb(f"Vision transcription failed: {str(ex)}. Proceeding with prompt only.", "warning")
                    
                    if generation_cancel.is_set():
                        q.put({"type": "error", "message": "Generation cancelled."})
                        q.put(None)
                        return
                            
                    res = orchestrator.process_query(
                        final_prompt, 
                        request.mode, 
                        selected_models=request.selected_models,
                        status_callback=thread_cb
                    )
                    if not generation_cancel.is_set():
                        q.put({"type": "final_response", "text": res})
                    else:
                        q.put({"type": "error", "message": "Generation cancelled."})
                except Exception as ex:
                    if not generation_cancel.is_set():
                        q.put({"type": "error", "message": str(ex)})
                q.put(None)  # Sentinel

            thread = threading.Thread(target=run_orchestrator)
            thread.start()
            
            while True:
                try:
                    item = q.get(timeout=0.1)
                    if item is None:
                        break
                    yield json.dumps(item) + "\n"
                except queue.Empty:
                    yield json.dumps({"type": "keep_alive"}) + "\n"
                    await asyncio.sleep(0.5)
                    
            thread.join()

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/api/memory/count")
def get_memory_count():
    """Get total number of memories stored."""
    return {"count": orchestrator.memory.count()}

@app.post("/api/memory/clear")
def clear_all_memory():
    """Reset vector database / SQLite store."""
    try:
        # Recreate db tables and client
        orchestrator.memory = Memory()
        # Remove sqlite db file and recreate
        path = orchestrator.memory.sqlite_path
        if os.path.exists(path):
            os.remove(path)
        orchestrator.memory._init_sqlite()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/unload")
def unload_models():
    """Unload all models to free system RAM."""
    orchestrator.unload_all_models()
    return {"status": "unloaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
