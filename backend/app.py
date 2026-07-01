import os
# Force Level Zero backend and apply the Immediate Command Lists workaround 
# This bypasses "Error 45 (UR_RESULT_ERROR_INVALID_ARGUMENT)" on Intel Iris Xe
os.environ["SYCL_DEVICE_FILTER"] = "level_zero"
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["IPEX_OPTIMIZE_TRANSFORMERS"] = "1"
import sys
# Add root folder to sys.path to resolve backend package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import json
import asyncio
try:
    import psutil
except ImportError:
    class _MockVM:
        def __init__(self):
            self.total = 16 * (1024 ** 3)
            self.available = 8 * (1024 ** 3)
            self.used = 8 * (1024 ** 3)
            self.percent = 50.0
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            self.total = int(line.split()[1]) * 1024
                        elif line.startswith('MemAvailable:'):
                            self.available = int(line.split()[1]) * 1024
                self.used = self.total - self.available
                self.percent = round((self.used / self.total) * 100, 1) if self.total else 50.0
            except Exception:
                pass
    class _MockPsutil:
        def virtual_memory(self):
            return _MockVM()
        def cpu_percent(self):
            return 0.0
    psutil = _MockPsutil()
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from fastapi.responses import HTMLResponse
from backend.memory import Memory
from backend.downloader import check_models_status, download_model, MODEL_DEFINITIONS
from backend.orchestrator import AgentOrchestrator
from backend.benchmark_runner import BENCHMARK_STATE, run_benchmark_suite, STATE_LOCK

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

os.makedirs("workspaces", exist_ok=True)
app.mount("/workspaces", StaticFiles(directory="workspaces"), name="workspaces")

# Cancellation flag — set when user clicks Cancel to stop generation (models stay loaded)
generation_cancel = threading.Event()

# Global orchestrator instance — receives the cancel flag for mid-generation stopping
orchestrator = AgentOrchestrator(cancel_event=generation_cancel)

# Lock and progress tracker for downloads
download_lock = threading.Lock()
download_progress = {}

# Global lock to prevent multi-user orchestrator state collisions
chat_lock = threading.Lock()

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
    search_mode: str = "off"  # off, simple, prediction, extreme

class SettingsRequest(BaseModel):
    context_length: int
    max_tokens: int
    temperature: float
    device_mode: Optional[str] = "gpu"
    gpu_layers: Optional[int] = -1
    search_mode: str = "off"  # off, simple, prediction, extreme

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
            "gpu": gpu_info,
            "evm_active": getattr(orchestrator, 'kaggle_hotswap_mode', False)
        },
        "settings": {
            "context_length": orchestrator.context_length,
            "max_tokens": orchestrator.max_tokens,
            "temperature": orchestrator.temperature,
            "device_mode": orchestrator.device_mode,
            "gpu_layers": orchestrator.gpu_layers,
            "search_mode": getattr(orchestrator, "search_mode", "off")
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
        search_mode=settings.search_mode
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

@app.post("/api/load_all")
def load_all_models():
    """Pre-load ALL core models directly into System RAM (CPU mode) at startup.
    
    This keeps VRAM 100% empty, maximizing available space for the active model's KV cache.
    During conversation, EVM hot-swaps models from RAM → VRAM one at a time.
    """
    import gc
    try:
        models_status = check_models_status()
        downloaded = [k for k, v in models_status.items() if v.get("downloaded")]
        
        benchmark_models = ["router", "deepseek_r1", "vibethinker", "opencode"]
        loaded = []
        
        # Suspend EVM hot-swap during this initialization phase
        evm_was_active = getattr(orchestrator, 'kaggle_hotswap_mode', False)
        orchestrator.kaggle_hotswap_mode = False
        
        try:
            for model_key in benchmark_models:
                if model_key in downloaded:
                    try:
                        print(f"📥 EVM Pre-Load: Instantiating '{model_key}' in System RAM (CPU)...")
                        # Enforce force_cpu=True to load it into CPU RAM
                        # Use a uniform, safe context (2048) to minimize RAM footprint while warm
                        cpu_ctx = 2048
                        
                        orchestrator._get_model(model_key, required_ctx=cpu_ctx, force_cpu=True)
                        loaded.append(model_key)
                        gc.collect()
                    except Exception as e:
                        print(f"⚠️ Failed to pre-load '{model_key}' to RAM: {e}")
        finally:
            # Restore EVM hot-swap mode
            orchestrator.kaggle_hotswap_mode = evm_was_active
            
        print(f"🚀 EVM Pre-Load Complete: {len(loaded)} models instantiated in System RAM. VRAM is 100% free!")
        
        return {
            "status": "success", 
            "message": f"Instantiated {len(loaded)} models in System RAM. VRAM is 100% free for KV cache."
        }
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
        search_mode=request.search_mode
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
                # Acquire global lock to prevent state mixing between simultaneous requests
                if not chat_lock.acquire(blocking=False):
                    q.put({"type": "error", "message": "The AI is currently processing another request. Please wait until it finishes."})
                    q.put(None)
                    return
                try:
                    # Clear any stale cancel signal before starting
                    generation_cancel.clear()
                    
                    # If an image was uploaded, run Qwen 2.5 VL vision parsing first
                    final_prompt = request.prompt
                    if request.image:
                        if generation_cancel.is_set():
                            q.put({"type": "error", "message": "Generation cancelled."})
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
                finally:
                    chat_lock.release()
                    q.put(None)  # Sentinel

            thread = threading.Thread(target=run_orchestrator)
            thread.start()
            
            while True:
                try:
                    item = q.get(timeout=0.1)
                    if item is None:
                        break
                    yield f"data: {json.dumps(item)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'keep_alive'})}\n\n"
                    await asyncio.sleep(0.5)
                    
            thread.join()

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/memory/count")
def get_memory_count():
    """Get total number of memories stored."""
    return {"count": orchestrator.memory.count()}

@app.post("/api/memory/clear")
def clear_all_memory():
    """Reset vector database / SQLite store."""
    try:
        import shutil
        db_path = orchestrator.memory.db_path
        # Remove the entire persistent DB directory (ChromaDB + SQLite)
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        orchestrator.memory = Memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/unload")
def unload_models():
    """Unload all models to free system RAM."""
    orchestrator.unload_all_models()
    return {"status": "unloaded"}

# ── Benchmark Endpoints for TPU v5e-8 ─────────────────────────────────────

class BenchmarkStartRequest(BaseModel):
    category: str
    sample_size: int

@app.get("/benchmark", response_class=HTMLResponse)
def get_benchmark_dashboard():
    """Serve the gorgeous glassmorphic TPU v5e-8 Benchmark Dashboard."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.html")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)

@app.get("/api/benchmark/status")
def get_benchmark_status():
    """Retrieve the real-time status of all 8 TPU worker cores and active KPIs."""
    with STATE_LOCK:
        # Return a copy to prevent race conditions during serialization
        return dict(BENCHMARK_STATE)

@app.post("/api/benchmark/start")
def start_benchmark_suite(req: BenchmarkStartRequest, background_tasks: BackgroundTasks):
    """Start the parallel benchmark evaluation in the background."""
    with STATE_LOCK:
        if BENCHMARK_STATE["active"]:
            return {"status": "already_running"}
    
    # run_benchmark_suite is async — we must bridge it into a sync wrapper
    # for FastAPI's BackgroundTasks, which runs tasks in a plain thread.
    def _run_sync():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_benchmark_suite(req.category, req.sample_size, orchestrator))
            loop.close()
        except Exception as e:
            import traceback
            print(f"❌ Benchmark runner crashed: {traceback.format_exc()}")
            with STATE_LOCK:
                BENCHMARK_STATE["active"] = False
                BENCHMARK_STATE["logs"].append(f"[ERROR] Benchmark crashed: {str(e)}")
    
    background_tasks.add_task(_run_sync)
    return {"status": "started"}

@app.post("/api/benchmark/stop")
def stop_benchmark_suite():
    """Stop the active benchmark evaluation and release the workers."""
    with STATE_LOCK:
        BENCHMARK_STATE["active"] = False
    return {"status": "stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
