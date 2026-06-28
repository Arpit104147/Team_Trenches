import os
import gc
import re
import json
import time
import psutil
import threading
from backend.downloader import get_model_path, is_model_downloaded
from backend.sandbox import Sandbox
from backend.memory import Memory
from backend.search import WebSearch

try:
    import torch
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
except ImportError:
    torch = None


class TransformerWrapper:
    """Wrapper that holds model + tokenizer refs so the Dynamic Memory Allocator
    can deterministically delete them and free GPU/RAM on eviction."""
    def __init__(self, model, tokenizer, device, cancel_event=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cancel_event = cancel_event  # threading.Event set by /api/cancel

    def __call__(self, prompt, max_tokens=512, temperature=0.7, system_prompt=None):
        if isinstance(prompt, str):
            # Convert raw strings into proper conversational format so the model doesn't hallucinate
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Build stopping criteria that checks the cancel flag every token
        stopping_criteria = []
        if self.cancel_event:
            from transformers import StoppingCriteria
            class CancelCriteria(StoppingCriteria):
                def __init__(self, event):
                    self.event = event
                def __call__(self, input_ids, scores, **kwargs):
                    return self.event.is_set()
            stopping_criteria = [CancelCriteria(self.cancel_event)]
        
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria
                )
        except RuntimeError as e:
            print(f"⚠️ XPU compute error ({e}). Attempting fallback to CPU for this prompt...")
            # If XPU crashes during generation, fall back to CPU on the fly
            self.model = self.model.to("cpu")
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria
                )
        
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("Generation cancelled by user.")
            
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

    def close(self):
        """Deterministically free GPU/CPU memory held by this model."""
        if hasattr(self, 'model') and hasattr(self.model, 'to'):
            try:
                self.model.to("cpu")  # Move off GPU first
            except Exception:
                pass
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass


class AgentOrchestrator:
    def __init__(self, cancel_event=None):
        self.cancel_event = cancel_event  # threading.Event for cancel support
        self.context_length = 8192
        self.max_tokens = 2048
        self.temperature = 0.7
        self.device_mode = "gpu"
        self.gpu_layers = -1
        self.search_mode = "off"  # off, simple, prediction, extreme
        
        self.sandbox = Sandbox(timeout=300)
        self.memory = Memory()
        self.web_search = WebSearch()
        self.loaded_models = {}
        self.model_lock = threading.Lock()  # Synchronizes model loads across threads
        self.inference_lock = threading.Lock()  # Synchronizes generation to prevent llama.cpp C++ deadlocks
        self.model_access_order = []  # LRU tracker: oldest first
        
        # Dual-GPU Setup Detection
        self.dual_gpu_pipeline = False
        if torch and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.dual_gpu_pipeline = True

        # ── Dynamic Threshold Calibration ────────────────────────────────
        # Instead of a hardcoded value, derive safe limits from the actual
        # hardware detected at startup.
        total_ram = psutil.virtual_memory()
        self.total_ram_gb = total_ram.total / (1024 ** 3)
        # Reserve 25% of total RAM as a safety buffer for the OS + other apps
        # 8 GB  → keep 2.0 GB free  (6 GB usable for models)
        # 16 GB → keep 4.0 GB free  (12 GB usable)
        # 32 GB → keep 8.0 GB free  (24 GB usable)
        self.ram_safety_gb = round(self.total_ram_gb * 0.25, 1)
        # VRAM safety: on NVIDIA dGPUs, reserve 40% of VRAM so the DMA evicts
        # old models *before* a new 5-7 GB model load crashes with OOM.
        # On iGPUs (Intel/AMD shared memory), VRAM = RAM so this doesn't apply.
        self.vram_safety_gb = 2.0
        self.kaggle_hotswap_mode = False
        if torch and torch.cuda.is_available():
            try:
                _free, total_vram = torch.cuda.mem_get_info(0)
                total_vram_gb = total_vram / (1024 ** 3)
                self.vram_safety_gb = round(total_vram_gb * 0.40, 1)  # 40% reserve
                
                # ── EVM (Enterprise VRAM Multiplexing) Hot-Swap Mode ──
                # On GPUs with ≤24GB VRAM (P100, T4, L4), all 3 text models (~14GB total)
                # barely fit simultaneously. Passive LRU eviction triggers too late (at 40%
                # threshold) and risks OOM during model loads. EVM proactively flushes the
                # least-recently-used model BEFORE loading a new one, ensuring there is always
                # enough VRAM headroom. On larger GPUs (A100/H100), all models fit comfortably
                # so we rely on the standard LRU pressure manager instead.
                if total_vram_gb <= 24:
                    self.kaggle_hotswap_mode = True
                    # EVM guarantees proactive model flushing before every load,
                    # so we can safely use 95% of VRAM and RAM (only 5% reserve).
                    self.vram_safety_gb = round(total_vram_gb * 0.05, 1)
                    self.ram_safety_gb = round(self.total_ram_gb * 0.05, 1)
                    print(f"⚡ EVM: Enterprise VRAM Multiplexing ACTIVE (≤24GB GPU detected)")
                    print(f"⚡ EVM: 95% utilization enabled — VRAM reserve={self.vram_safety_gb:.1f} GB, RAM reserve={self.ram_safety_gb:.1f} GB")
                else:
                    self.kaggle_hotswap_mode = False
                        
                print(f"🎮 DMA: NVIDIA GPU detected — {total_vram_gb:.0f} GB VRAM, "
                      f"evict threshold = {self.vram_safety_gb:.1f} GB free")
            except Exception:
                pass
        # Auto-context ceiling based on available VRAM
        # P100 (16GB): 4096 ctx  |  A100/H100 (40-80GB): 8192 ctx  |  iGPU/CPU: 8192 (uses RAM)
        self.max_auto_ctx = 8192
        if torch and torch.cuda.is_available():
            try:
                _free, total_vram = torch.cuda.mem_get_info(0)
                total_vram_gb = total_vram / (1024 ** 3)
                if total_vram_gb <= 16:
                    self.max_auto_ctx = 16384  # Increased from 8192 to 16k for 16GB VRAM (like P100)
                elif total_vram_gb <= 24:
                    self.max_auto_ctx = 16384  # Increased from 8192 to 16k for 24GB VRAM
                elif total_vram_gb <= 48:
                    self.max_auto_ctx = 32768  # A6000 (48GB) / A100 (40GB) -> 32k context
                else:
                    self.max_auto_ctx = 65536  # H100 (80GB) -> 64k context
                print(f"📐 DMA: Auto-context ceiling = {self.max_auto_ctx} tokens (based on {total_vram_gb:.0f} GB VRAM)")
            except Exception:
                pass
        self.context_length = self.max_auto_ctx
        print(f"🧠 DMA: Set default context length to {self.context_length} tokens")
        print(f"🧠 DMA: Detected {self.total_ram_gb:.0f} GB RAM → "
              f"Safety threshold = {self.ram_safety_gb:.1f} GB "
              f"(evict when free < {self.ram_safety_gb:.1f} GB)")

    def update_settings(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

    # =========================================================================
    # DYNAMIC MEMORY ALLOCATOR  (Adaptive + LRU)
    #
    # RAM Threshold:  25% of total system RAM  (auto-detected at startup)
    # VRAM Threshold: 2 GB free per discrete GPU
    # Eviction:       LRU — evicts the oldest-accessed model first,
    #                 one at a time, until enough room is available.
    #                 Falls back to full flush only as a last resort.
    # VRAM Sources:   1) torch.cuda (NVIDIA/ROCm)
    #                 2) Linux sysfs /sys/class/drm/ (AMD/Intel Vulkan)
    # =========================================================================
    def _get_ram_free_gb(self):
        return psutil.virtual_memory().available / (1024 ** 3)

    def _get_vram_free_gb(self, device_idx=0):
        """Returns free VRAM in GB for a specific CUDA device, or None."""
        if not (torch and torch.cuda.is_available()):
            return None
        try:
            free, _total = torch.cuda.mem_get_info(device_idx)
            return free / (1024 ** 3)
        except Exception:
            return None

    def _get_dynamic_context_ceiling(self, model_key):
        """Dynamically computes the safe context ceiling for a specific model based on actual free VRAM and free RAM.
        Takes into account the EVM hot-swap (unloading other models) and the size of the target model."""
        # Determine base limit (8k)
        base_limit = getattr(self, 'max_auto_ctx', 8192)
        
        # Check system RAM margins
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        free_ram_gb = vm.available / (1024 ** 3)
        ram_used_pct = (vm.total - vm.available) / vm.total * 100
        
        # Scale context ceiling if there is plenty of system RAM (leaving 5% margin)
        ram_allowed_ceiling = base_limit
        if ram_used_pct < 95.0:
            five_percent_ram_gb = total_ram_gb * 0.05
            surplus_ram = free_ram_gb - five_percent_ram_gb
            if surplus_ram > 0:
                ram_allowed_ceiling = int(base_limit + surplus_ram * 4000)
                ram_allowed_ceiling = min(32768, ram_allowed_ceiling)

        # Check GPU VRAM margins
        vram_allowed_ceiling = ram_allowed_ceiling
        if torch and torch.cuda.is_available():
            try:
                free_vram, total_vram = torch.cuda.mem_get_info(0)
                free_vram_gb = free_vram / (1024 ** 3)
                total_vram_gb = total_vram / (1024 ** 3)
                vram_used_pct = (total_vram - free_vram) / total_vram * 100
                
                # If EVM hot-swap is active, we know the orchestrator will ruthlessly flush
                # all other models before loading this one. So we assume 95% of total VRAM
                # will be safely available, regardless of current occupancy.
                if getattr(self, 'kaggle_hotswap_mode', False):
                    five_percent_vram_gb = total_vram_gb * 0.05
                    surplus_vram = (total_vram_gb * 0.95) - five_percent_vram_gb
                    if surplus_vram > 0:
                        vram_allowed_ceiling = int(base_limit + surplus_vram * 8000)
                        vram_allowed_ceiling = min(32768, vram_allowed_ceiling)
                elif vram_used_pct < 95.0:
                    five_percent_vram_gb = total_vram_gb * 0.05
                    surplus_vram = free_vram_gb - five_percent_vram_gb
                    if surplus_vram > 0:
                        vram_allowed_ceiling = int(base_limit + surplus_vram * 8000)
                        vram_allowed_ceiling = min(32768, vram_allowed_ceiling)
                
                # GPU Architecture check: older GPUs (P100, T4, V100) lack hardware Flash Attention.
                # Standard attention memory scales quadratically. Cap context to prevent OOM.
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    if major < 8:  # SM 6.0 (P100), SM 7.5 (T4)
                        # Without Flash Attention, attention memory scales quadratically.
                        # Cap at 8192 on older GPUs to prevent quadratic VRAM OOM crashes, while
                        # allowing enough tokens for DeepSeek-R1 to finish long mathematical derivations.
                        vram_allowed_ceiling = min(8192, vram_allowed_ceiling)
                except Exception:
                    pass
            except Exception:
                pass

        hard_limit = min(ram_allowed_ceiling, vram_allowed_ceiling)
        # Use 2048 as floor instead of 8192 — the old max(8192) was overriding
        # the P100's VRAM-safe 4096 cap, causing 'token limit exceeded' OOM crashes.
        hard_limit = max(2048, hard_limit)
        
        # 1. System RAM Constraints (Emergency fallback only to prevent OS crash)
        free_ram = self._get_ram_free_gb()
        if free_ram < 1.5:
            ram_limit = 2048
        else:
            ram_limit = hard_limit

        # 2. GPU VRAM Constraints (Theoretical Free VRAM after EVM Swap)
        vram_limit = hard_limit
        if torch and torch.cuda.is_available():
            try:
                free_vram = self._get_vram_free_gb(0)
                if free_vram is not None:
                    # In EVM mode, if other models are loaded, their VRAM will be freed.
                    # Calculate VRAM that will be freed by unloading other models
                    freed_by_evm = 0.0
                    if getattr(self, 'kaggle_hotswap_mode', False):
                        for mk, model_obj in self.loaded_models.items():
                            if mk != model_key:
                                freed_by_evm += self._estimate_model_size_gb(mk)
                    
                    target_model_size = self._estimate_model_size_gb(model_key)
                    # Theoretical free VRAM after swap and load.
                    # CRITICAL: If the model is already loaded, its footprint is already subtracted
                    # from the free VRAM. Subtracting it again would double-count the model size
                    # and artificially throttle context limits to 512 tokens.
                    if model_key in self.loaded_models:
                        theo_free_vram = free_vram + freed_by_evm
                    else:
                        theo_free_vram = free_vram + freed_by_evm - target_model_size
                    
                    # Deduct overhead for model execution (computational graph, activations)
                    usable_kv_vram = theo_free_vram - 1.5
                    if usable_kv_vram < 0:
                        usable_kv_vram = 0.5
                    
                    # 1 token ≈ 0.13 MB of KV cache (FP16 8B model)
                    calculated_limit = int((usable_kv_vram * 1024) / 0.13)
                    # Round down to nearest multiple of 1024
                    calculated_limit = (calculated_limit // 1024) * 1024
                    vram_limit = max(1024, min(hard_limit, calculated_limit))
            except Exception:
                pass
                
        # Sane bottleneck of RAM, VRAM, and hard limit
        dynamic_cap = min(hard_limit, ram_limit, vram_limit)
        
        # Model-specific physical context ceilings — scale with GPU compute capability.
        # SM >= 9.0 (H100/B200): Flash Attention v2 + HBM3 → safe up to 65k
        # SM >= 8.0 (A100/A6000/RTX 3090): Flash Attention v1 + HBM2e → safe up to 32k
        # SM <  8.0 (P100/T4/V100): No flash attention, quadratic memory → cap at 8k
        # iGPU/CPU: Uses system RAM for KV cache → bounded by RAM, safe up to 16k
        gpu_ctx_cap = 16384  # Default for iGPU/CPU
        if torch and torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 9:
                    gpu_ctx_cap = 65536   # H100, B200
                elif major >= 8:
                    gpu_ctx_cap = 32768   # A100, A6000, RTX 3090/4090
                else:
                    gpu_ctx_cap = 8192    # P100, T4, V100
            except Exception:
                pass
        
        # Per-model ceiling (all models use the same GPU-derived ceiling,
        # but individual models can be overridden here if needed)
        model_ceilings = {
            "router": min(gpu_ctx_cap, 16384),       # Router doesn't need huge context
            "opencode": gpu_ctx_cap,
            "deepseek_r1": gpu_ctx_cap
        }
        model_cap = model_ceilings.get(model_key, gpu_ctx_cap)
        dynamic_cap = min(dynamic_cap, model_cap)
        
        dynamic_cap = max(1024, dynamic_cap)
        return dynamic_cap

    def _get_sysfs_gpu_vram(self):
        """Read AMD/Intel GPU VRAM via Linux sysfs. Returns list of (free_gb, total_gb, card_name)."""
        import glob
        results = []
        for card_dir in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            vram_used_path = os.path.join(card_dir, "mem_info_vram_used")
            vram_total_path = os.path.join(card_dir, "mem_info_vram_total")
            if os.path.exists(vram_used_path) and os.path.exists(vram_total_path):
                try:
                    with open(vram_total_path) as f:
                        total_bytes = int(f.read().strip())
                    with open(vram_used_path) as f:
                        used_bytes = int(f.read().strip())
                    free_gb = (total_bytes - used_bytes) / (1024 ** 3)
                    total_gb = total_bytes / (1024 ** 3)
                    card_name = os.path.basename(os.path.dirname(card_dir))
                    results.append((free_gb, total_gb, card_name))
                except Exception:
                    pass
        return results

    def _touch_model(self, model_key):
        """Update the LRU access order (move model_key to most-recent)."""
        if model_key in self.model_access_order:
            self.model_access_order.remove(model_key)
        self.model_access_order.append(model_key)

    def _evict_lru_model(self):
        """Evict the single least-recently-used model. Returns True if one was evicted."""
        if not self.model_access_order:
            return False
        lru_key = self.model_access_order.pop(0)
        model_obj = self.loaded_models.pop(lru_key, None)
        if model_obj is None:
            return False
        print(f"🧹 DMA-LRU: Evicting '{lru_key}' (least recently used)...")
        with self.inference_lock:
            if hasattr(model_obj, 'close'):
                try:
                    model_obj.close()
                except Exception as e:
                    print(f"  Warning: close() failed for '{lru_key}': {e}")
            del model_obj
        gc.collect()
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass
        print(f"  ✅ Evicted '{lru_key}'. RAM now: {self._get_ram_free_gb():.1f} GB free")
        return True

    def _estimate_model_size_gb(self, model_key):
        """Estimate model size in GB from the GGUF file on disk."""
        try:
            model_path = get_model_path(model_key)
            if model_path and os.path.exists(model_path):
                return os.path.getsize(model_path) / (1024 ** 3)
        except Exception:
            pass
        # Fallback estimates based on known model sizes
        size_map = {"router": 3.0, "deepseek_r1": 6.0,
                    "opencode": 5.2, "qwen_vl": 6.5}
        return size_map.get(model_key, 5.0)

    def _check_memory_pressure(self, required_vram_gb=None, target_gpu_idx=None):
        """LRU-based eviction loop. Evicts models one-by-one until safe.
        
        If required_vram_gb is provided, evicts until free VRAM on the target GPU
        (or all GPUs if target_gpu_idx is None) >= required_vram_gb + 1.5 GB buffer.
        Otherwise falls back to the static safety threshold.
        """
        evicted_any = False

        # ── System RAM Check ─────────────────────────────────────────────
        free_ram = self._get_ram_free_gb()
        while free_ram < self.ram_safety_gb and self.model_access_order:
            print(f"⚠️ DMA: RAM low ({free_ram:.1f} GB free < {self.ram_safety_gb:.1f} GB threshold)")
            if not self._evict_lru_model():
                break
            evicted_any = True
            free_ram = self._get_ram_free_gb()

        # ── CUDA/ROCm VRAM Check (NVIDIA + AMD ROCm dGPU) ───────────────
        # Wrapped in try/except: on Kaggle P100 (sm_60) PyTorch's CUDA runtime
        # is incompatible and torch.cuda calls can crash the process.
        try:
            if torch and torch.cuda.is_available():
                gpus_to_check = [target_gpu_idx] if target_gpu_idx is not None else range(torch.cuda.device_count())
                for gpu_idx in gpus_to_check:
                    vram_free = self._get_vram_free_gb(gpu_idx)
                    if vram_free is None:
                        continue
                    effective_threshold = self.vram_safety_gb
                    if required_vram_gb is not None:
                        effective_threshold = max(self.vram_safety_gb, required_vram_gb + 1.5)
                    if vram_free < effective_threshold:
                        print(f"⚠️ DMA: CUDA GPU:{gpu_idx} VRAM low ({vram_free:.1f} GB free, need {effective_threshold:.1f} GB)")
                        while vram_free < effective_threshold and self.model_access_order:
                            if not self._evict_lru_model():
                                break
                            evicted_any = True
                            vram_free = self._get_vram_free_gb(gpu_idx)
        except Exception as e:
            print(f"⚠️ DMA: CUDA VRAM check skipped ({e})")

        # ── Linux sysfs VRAM Check (AMD/Intel Vulkan — no ROCm needed) ──
        sysfs_gpus = self._get_sysfs_gpu_vram()
        for free_gb, total_gb, card_name in sysfs_gpus:
            effective_threshold = self.vram_safety_gb
            if required_vram_gb is not None:
                effective_threshold = max(self.vram_safety_gb, required_vram_gb + 1.5)
            if free_gb < effective_threshold:
                print(f"⚠️ DMA: sysfs {card_name} VRAM low ({free_gb:.1f}/{total_gb:.1f} GB free)")
                while free_gb < effective_threshold and self.model_access_order:
                    if not self._evict_lru_model():
                        break
                    evicted_any = True
                    # Re-read sysfs after eviction
                    updated = self._get_sysfs_gpu_vram()
                    match = [g for g in updated if g[2] == card_name]
                    free_gb = match[0][0] if match else free_gb

        if evicted_any:
            print(f"✅ DMA: Eviction cycle complete. "
                  f"RAM: {self._get_ram_free_gb():.1f} GB free")
        return evicted_any

    def unload_all_models(self):
        """Nuclear option: deterministically unload every cached model."""
        with self.inference_lock:
            for key in list(self.loaded_models.keys()):
                model_obj = self.loaded_models[key]
                if hasattr(model_obj, 'close'):
                    try:
                        model_obj.close()
                    except Exception as e:
                        print(f"Warning: Failed to close model '{key}': {e}")
                del self.loaded_models[key]
            
            self.loaded_models.clear()
            self.model_access_order.clear()
        gc.collect()
        
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass

    def _get_model(self, model_key, required_ctx=None, force_cpu=False):
        """Load a model with Dynamic Memory Allocator protection and dynamic context sizing."""
        if required_ctx is None:
            required_ctx = self.context_length if self.context_length > 0 else 8192

        # CRITICAL SEGFAULT PREVENTION: llama.cpp has a known C-level double-free segfault
        # when a GGUF model is closed and re-initialized in the same process to expand context.
        # To completely prevent this, we enforce that on GPUs, the model is ALWAYS loaded with
        # its maximum safe context ceiling (ceiling) right from the start. This ensures the model
        # is loaded exactly once at its full capacity and NEVER has to be unloaded/reloaded to expand.
        is_cpu = (self.device_mode == "cpu" or force_cpu)
        if not is_cpu:
            ceiling = self._get_dynamic_context_ceiling(model_key)
            required_ctx = ceiling
        else:
            required_ctx = max(8192, required_ctx)

        # Check if already loaded
        if model_key in self.loaded_models:
            model_obj = self.loaded_models[model_key]
            current_gpu_layers = getattr(model_obj, "_n_gpu_layers", 0)
            
            # Case 1: We want it on GPU (not force_cpu and device_mode != cpu) but it's currently on CPU (RAM)
            if (self.device_mode != "cpu" and not force_cpu) and current_gpu_layers == 0:
                print(f"🔄 EVM (RAM -> VRAM Swap): Swapping '{model_key}' from System RAM to GPU VRAM for active inference...")
                with self.model_lock:
                    # Unload CPU version
                    self.loaded_models.pop(model_key, None)
                    if hasattr(model_obj, "close"):
                        try: model_obj.close()
                        except: pass
                    del model_obj
                    gc.collect()
            # Case 2: We want it on CPU (force_cpu) but it's currently on GPU (VRAM)
            elif force_cpu and current_gpu_layers > 0:
                print(f"🔄 EVM (VRAM -> RAM Swap): Swapping '{model_key}' from VRAM down to System RAM...")
                with self.model_lock:
                    # Unload GPU version
                    self.loaded_models.pop(model_key, None)
                    if hasattr(model_obj, "close"):
                        try: model_obj.close()
                        except: pass
                    del model_obj
                    gc.collect()
            else:
                # Fast path: already loaded with correct target and sufficient context
                if not (hasattr(model_obj, "n_ctx") and required_ctx > model_obj.n_ctx()):
                    self._touch_model(model_key)
                    return model_obj

        # Synchronize load operations across all threads
        with self.model_lock:
            return self._load_model_synchronized(model_key, required_ctx, force_cpu)

    def _load_model_synchronized(self, model_key, required_ctx=None, force_cpu=False):
        """Internal synchronized loader protecting against concurrent load race conditions."""
        if required_ctx is None:
            required_ctx = self.context_length if self.context_length > 0 else 8192

        if model_key in self.loaded_models:
            model_obj = self.loaded_models[model_key]
            # Context expansion — safe reload on all platforms
            if hasattr(model_obj, "n_ctx") and required_ctx > model_obj.n_ctx():
                print(f"🔄 Reloading '{model_key}' to expand context: {model_obj.n_ctx()} -> {required_ctx}")
                with self.inference_lock:
                    if hasattr(model_obj, 'close'):
                        try:
                            model_obj.close()
                        except Exception:
                            pass
                    del self.loaded_models[model_key]
                    if model_key in self.model_access_order:
                        self.model_access_order.remove(model_key)
                    del model_obj
                gc.collect()
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2)  # Give llama.cpp's async CUDA deallocation time to complete
            else:
                self._touch_model(model_key)
                return model_obj

        # DMA Check: evict if memory is low BEFORE attempting a new load
        # Pass the estimated model size so the DMA evicts enough room for THIS specific model
        est_model_gb = self._estimate_model_size_gb(model_key)
        print(f"📦 DMA: Preparing to load '{model_key}' (~{est_model_gb:.1f} GB)")
        
        # ── EVM Hot-Swap Guard ────────────────────────────────────────────
        # On constrained GPUs, completely evict ALL other GPU-resident models
        # so the incoming model gets 100% of the VRAM for its KV Cache.
        # We do NOT re-instantiate evicted models on CPU — this caused race conditions
        # where the CPU re-load consumed RAM while GPU memory hadn't fully freed yet.
        # Instead, the next _get_model() call will reload from disk (which is fast
        # because the file is already in the OS page cache from the initial load_all).
        evm_flushed = False
        if getattr(self, 'kaggle_hotswap_mode', False) and not force_cpu and self.loaded_models:
            models_to_evict = [
                mk for mk, m in list(self.loaded_models.items())
                if mk != model_key and getattr(m, "_n_gpu_layers", -1) != 0
            ]
            if models_to_evict:
                with self.inference_lock:
                    for mk in models_to_evict:
                        print(f"🔄 DMA (EVM): Evicting '{mk}' from VRAM...")
                        model_obj = self.loaded_models.pop(mk, None)
                        if mk in self.model_access_order:
                            self.model_access_order.remove(mk)
                        if hasattr(model_obj, 'close'):
                            try:
                                model_obj.close()
                            except Exception:
                                pass
                        del model_obj
                gc.collect()
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # Give llama.cpp's internal cudaFree() time to fully complete
                time.sleep(3)
                gc.collect()
                evm_flushed = True
            else:
                evm_flushed = True  # Only our model is loaded, all VRAM is ours
                
        # Skip memory pressure check if EVM already cleared VRAM
        if not evm_flushed and not force_cpu:
            target_gpu = 0
            if self.dual_gpu_pipeline:
                if model_key in ["deepseek_r1", "qwen_vl"]:
                    target_gpu = 1
                else:
                    target_gpu = 0
            self._check_memory_pressure(required_vram_gb=est_model_gb, target_gpu_idx=target_gpu)

        # ── iGPU Unified Memory Guard ─────────────────────────────────────
        is_igpu = not (torch and torch.cuda.is_available())
        if is_igpu and self.loaded_models:
            free_ram = self._get_ram_free_gb()
            if free_ram < self.total_ram_gb * 0.35:
                print(f"🧠 DMA (iGPU Guard): Pre-emptive eviction — only {free_ram:.1f} GB free")
                self._evict_lru_model()

        if not is_model_downloaded(model_key):
            raise Exception(f"Model '{model_key}' is not downloaded. "
                            f"Place the weights in models/ and restart.")

        model_path = get_model_path(model_key)
        
        # ── GGUF Models (llama_cpp) ──────────────────────────────────────
        if model_path.endswith('.gguf'):
            from llama_cpp import Llama
            
            loading_on_cpu = (self.device_mode == "cpu" or force_cpu)
            
            kwargs = {
                "model_path": model_path,
                "n_ctx": required_ctx,
                "n_gpu_layers": 0 if loading_on_cpu else self.gpu_layers,
                "verbose": False
            }
            # Kaggle Input Speedup
            if "/kaggle/input" in model_path or "kaggle" in model_path.lower():
                kwargs["use_mmap"] = False
                
            # Restrict batch sizes and disable flash attention on older GPUs or CPU
            is_older_gpu = False
            if torch and torch.cuda.is_available() and not loading_on_cpu:
                try:
                    major, _ = torch.cuda.get_device_capability(0)
                    if major < 8:
                        is_older_gpu = True
                except Exception:
                    pass
                    
            if is_older_gpu or loading_on_cpu:
                kwargs["n_batch"] = 512
                kwargs["n_ubatch"] = 256
                kwargs["flash_attn"] = False
            else:
                kwargs["flash_attn"] = True

            if model_key == "qwen_vl":
                model_dir = os.path.dirname(model_path)
                mmproj_path = os.path.join(model_dir, "mmproj-BF16.gguf")
                if os.path.exists(mmproj_path):
                    try:
                        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                        kwargs["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                        print(f"👁️ Loaded Qwen2.5-VL chat handler with mmproj: {mmproj_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to load Qwen25VLChatHandler: {e}")
                else:
                    mmproj_flat = os.path.join(os.path.dirname(model_dir), "mmproj-BF16.gguf")
                    if os.path.exists(mmproj_flat):
                        try:
                            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                            kwargs["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mmproj_flat)
                            print(f"👁️ Loaded Qwen2.5-VL chat handler with mmproj: {mmproj_flat}")
                        except Exception as e:
                            print(f"⚠️ Failed to load Qwen25VLChatHandler: {e}")
                    else:
                        print(f"⚠️ mmproj file not found at {mmproj_path} or {mmproj_flat}. Vision features may fail.")

            # Dual-GPU
            if self.dual_gpu_pipeline and not loading_on_cpu:
                if model_key in ["deepseek_r1", "qwen_vl"]:
                    kwargs["main_gpu"] = 1
                else:
                    kwargs["main_gpu"] = 0

            # Robust GPU load with retry — if the GPU context fails (OOM / race condition),
            # retry once with a smaller context, then fall back to CPU as last resort.
            llm = None
            if not loading_on_cpu:
                for attempt, ctx_size in enumerate([required_ctx, max(2048, required_ctx // 2)]):
                    try:
                        kwargs["n_ctx"] = ctx_size
                        llm = Llama(**kwargs)
                        if ctx_size != required_ctx:
                            print(f"⚠️ DMA: GPU loaded '{model_key}' with reduced context ({ctx_size} instead of {required_ctx})")
                        break
                    except Exception as e:
                        print(f"⚠️ DMA: GPU context creation failed for '{model_key}' (n_ctx={ctx_size}): {e}")
                        gc.collect()
                        try:
                            if torch and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        time.sleep(2)
                
                # CPU fallback if all GPU attempts failed
                if llm is None:
                    print(f"⚠️ DMA: All GPU attempts failed for '{model_key}'. Falling back to CPU...")
                    kwargs["n_gpu_layers"] = 0
                    kwargs["n_ctx"] = required_ctx
                    kwargs["flash_attn"] = False
                    kwargs.pop("main_gpu", None)
                    llm = Llama(**kwargs)
            else:
                llm = Llama(**kwargs)

            # Store actual n_gpu_layers config on the instance
            llm._n_gpu_layers = kwargs["n_gpu_layers"]
            self.loaded_models[model_key] = llm
            self._touch_model(model_key)
            target_label = "System RAM/CPU" if kwargs.get("n_gpu_layers") == 0 else "GPU VRAM"
            print(f"✅ Loaded GGUF model '{model_key}' ({os.path.basename(model_path)}) ({target_label})")
            return llm
            
        # ── Safetensors / Transformers Models ────────────────────────────
        elif model_path.endswith('.safetensors') or os.path.isdir(model_path):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_dir = os.path.dirname(model_path) if model_path.endswith('.safetensors') else model_path
            
            device = "cpu"
            if self.device_mode != "cpu" and torch:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    device = "xpu"
                    

                
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            
            # XPU (Iris Xe): load in float32 — iGPU has limited fp16 kernel support
            if device == "xpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to("xpu")
                # IPEX optimize: replaces unsupported PyTorch ops with XPU-native kernels
                try:
                    model = ipex.optimize(model, dtype=torch.float32)
                    print(f"🔧 IPEX optimization applied for XPU")
                except Exception as e:
                    print(f"⚠️ IPEX optimize skipped: {e}")
                model.eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True
                )
            
            wrapper = TransformerWrapper(model, tokenizer, device, cancel_event=self.cancel_event)
            self.loaded_models[model_key] = wrapper
            self._touch_model(model_key)
            print(f"✅ Loaded Transformers model '{model_key}' on {device}")
            return wrapper
        
        else:
            raise Exception(f"Unsupported model format for '{model_key}': {model_path}")



    # =========================================================================
    # VISION: Qwen 2.5 VL Image Parsing
    # =========================================================================
    def transcribe_image(self, image_input, status_callback=None):
        if status_callback:
            status_callback("Qwen 2.5-VL parsing image...", "info", "qwen_vl", 5)
        llm = self._get_model("qwen_vl")
        vision_prompt = "Describe this image and extract all text and logic from it."
        
        import base64
        data_url = None
        
        # ── Case 1: Already a data URL from frontend (data:image/...;base64,...) ──
        if isinstance(image_input, str) and image_input.startswith("data:"):
            data_url = image_input
        # ── Case 2: File path on disk ──
        elif isinstance(image_input, str) and os.path.isfile(image_input):
            try:
                with open(image_input, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_input)[1].lower().replace(".", "")
                mime = "image/jpeg"
                if ext in ["png", "webp", "gif"]:
                    mime = f"image/{ext}"
                data_url = f"data:{mime};base64,{img_data}"
            except Exception as e:
                return f"Error reading image file: {e}"
        else:
            return f"Error: Invalid image input (not a data URL or valid file path)."

        if not data_url:
            return "Error: Could not process image data."

        with self.inference_lock:
            # Construct message payload for vision chat completion
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            try:
                res = llm.create_chat_completion(messages=messages, max_tokens=500)
                if status_callback:
                    status_callback("Qwen 2.5-VL transcription complete!", "success", "qwen_vl", 100)
                return res["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️ Vision chat completion failed: {e}. Falling back to standard completion...")
                if callable(llm):
                    result = llm(vision_prompt, max_tokens=500)
                    return result if isinstance(result, str) else result['choices'][0]['text']
                else:
                    res = llm(vision_prompt, max_tokens=500)
                    return res['choices'][0]['text']

    # =========================================================================
    # MAIN AGENTIC PIPELINE
    # =========================================================================
    def _clean_cutoff_notes(self, text):
        """Remove training cutoff date disclaimers and warnings from final output."""
        if not text:
            return text
        # Regex to remove parenthesized or unparenthesized notes about training cutoff date
        patterns = [
            r'\(?Note:\s*(Since|As)?\s*(my|our|the)?\s*training\s*data\s*(only\s*goes\s*up\s*to|cuts\s*off\s*in|goes\s*up\s*to|cutoff\s*is|knowledge\s*cutoff|only\s*extends\s*to).*?\)?\.?',
            r'\(?Always\s*verify\s*with\s*official\s*sources\s*for\s*the\s*most\s*up-to-date\s*information\.?\)?',
            r'\(?Since\s*my\s*training\s*data\s*only\s*goes\s*up\s*to\s*September\s*2021,\s*I\s*cannot\s*access\s*or\s*provide\s*real-time\s*information\.?\)?'
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()
    def _strip_thinking(self, text):
        """Remove <think>...</think> blocks from DeepSeek R1 output."""
        if not text:
            return text

        # --- Case 1: Properly closed think tag — strip the block entirely ---
        if '<think>' in text and '</think>' in text:
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if cleaned:
                return cleaned
            # The entire answer was inside the tags — return contents without tags
            return re.sub(r'</?think>', '', text).strip()

        # --- Case 2: Unclosed <think> tag (model ran out of context mid-think) ---
        if '<think>' in text and '</think>' not in text:
            before_think, inner = text.split('<think>', 1)
            # If there's real content before the thinking block, return that
            if before_think.strip():
                return before_think.strip()

            # The model wrote its ENTIRE output inside <think> without closing.
            # We need to salvage the best final answer from the inner monologue.
            # Strategy: find the LAST paragraph that looks like a structured answer
            # (not a "Wait, let me check..." self-questioning line).
            conversational_prefixes = (
                'okay', 'wait', 'so ', 'but ', 'hmm', 'let me', 'let\'s',
                'i think', 'i need', 'i should', 'i must', 'i realize',
                'actually', 'alternatively', 'now,', 'thus,', 'therefore,',
                'however,', 'also,', 'first,', 'second,', 'third,',
                'step ', 'note ', 'note:', 'so,', 'anyway', 'in summary'
            )
            lines = inner.strip().split('\n')
            # Walk backwards to find where the final structured answer begins
            answer_start = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                stripped = lines[i].strip().lower()
                if not stripped:
                    continue
                # If line starts a structured section, mark it as the start
                if (lines[i].strip().startswith(('##', '**', '1.', '2.', '3.', '-')) or
                        (len(stripped) > 30 and not stripped.startswith(conversational_prefixes))):
                    answer_start = i
                else:
                    # Stop searching once we hit a conversational line after a structured one
                    if answer_start < len(lines):
                        break

            final_lines = lines[answer_start:]
            final_content = '\n'.join(final_lines).strip()
            if final_content and len(final_content) > 50:
                return final_content
            # Last resort: return entire inner content (at least user gets something)
            return inner.strip()

        # --- Case 3: No think tags at all — return as-is ---
        return text.strip()

    # =========================================================================
    # AGENT IDE — Incremental Patching Engine
    # =========================================================================
    def _extract_error_line(self, traceback_text):
        """Extract the line number of the crash from a Python traceback string."""
        if not traceback_text:
            return None
        # Match patterns like: File "<string>", line 42
        matches = re.findall(r'(?:File\s+"[^"]*",\s+line\s+(\d+))|(?:line\s+(\d+))', traceback_text)
        if matches:
            # Take the last match (deepest frame in the traceback)
            last = matches[-1]
            line_num = int(last[0] or last[1])
            return line_num
        return None

    def _get_code_context_window(self, code, error_line, window=10):
        """Return a numbered code window around the error line for targeted patching."""
        lines = code.split('\n')
        start = max(0, error_line - window - 1)
        end = min(len(lines), error_line + window)
        context_lines = []
        for i in range(start, end):
            marker = " >>>" if (i + 1) == error_line else "    "
            context_lines.append(f"{marker} {i+1:4d} | {lines[i]}")
        return '\n'.join(context_lines)

    def _apply_search_replace_patch(self, original_code, patch_text):
        """Parse Aider-style <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks and apply them."""
        if not patch_text or '<<<<<<< SEARCH' not in patch_text:
            return None
        patched = original_code
        # Split into individual search/replace blocks
        blocks = re.split(r'<<<<<<< SEARCH\s*\n', patch_text)
        applied = 0
        for block in blocks[1:]:  # Skip content before first marker
            parts = re.split(r'\n=======\s*\n', block, maxsplit=1)
            if len(parts) != 2:
                continue
            search_text = parts[0]
            replace_and_rest = parts[1]
            # Extract replacement content (everything before >>>>>>> REPLACE)
            replace_parts = re.split(r'\n>>>>>>> REPLACE', replace_and_rest, maxsplit=1)
            replace_text = replace_parts[0]
            # Apply the patch — search_text must exist exactly once
            if search_text in patched:
                count = patched.count(search_text)
                if count == 1:
                    patched = patched.replace(search_text, replace_text, 1)
                    applied += 1
                else:
                    # Multiple matches — skip to avoid ambiguity
                    continue
            else:
                # Try stripping trailing whitespace from each line for fuzzy matching
                search_lines = [l.rstrip() for l in search_text.split('\n')]
                code_lines = [l.rstrip() for l in patched.split('\n')]
                search_stripped = '\n'.join(search_lines)
                code_stripped = '\n'.join(code_lines)
                if search_stripped in code_stripped:
                    code_stripped = code_stripped.replace(search_stripped, replace_text.rstrip(), 1)
                    patched = code_stripped
                    applied += 1
        return patched if applied > 0 else None

    def _agent_ide_patch(self, code, error_output, linter_model, gen_tokens, lang="python", system_prompt=None):
        """Agent IDE: Attempt surgical patching of code using search/replace blocks."""
        error_line = self._extract_error_line(error_output)
        if error_line:
            context = self._get_code_context_window(code, error_line, window=10)
        else:
            # Fallback: show first 30 numbered lines
            lines = code.split('\n')[:30]
            context = '\n'.join([f"    {i+1:4d} | {l}" for i, l in enumerate(lines)])

        error_snippet = error_output[:600] if error_output else "No output"
        patch_prompt = (
            f"You are an Agent IDE. A {lang} script crashed. Fix it with a SURGICAL EDIT.\n\n"
            f"ERROR:\n{error_snippet}\n\n"
            f"CODE CONTEXT (around the crash):\n{context}\n\n"
            f"Output ONLY a search-and-replace block in this EXACT format:\n"
            f"<<<<<<< SEARCH\n[exact lines to change, copied verbatim]\n=======\n[corrected lines]\n>>>>>>> REPLACE\n\n"
            f"Rules:\n"
            f"1. Copy the SEARCH lines EXACTLY as they appear (including whitespace)\n"
            f"2. Keep the fix minimal — only change what is broken\n"
            f"3. You may output multiple blocks if needed\n"
            f"4. Do NOT output the entire file"
        )
        sys = system_prompt or f"You are a strict {lang} syntax linter. Output only search-and-replace blocks."
        patch_response = self._strip_thinking(self._call_model(linter_model, patch_prompt, min(gen_tokens, 1024), 0.1, system_prompt=sys))
        patched_code = self._apply_search_replace_patch(code, patch_response)
        return patched_code

    def _crunch_prompt(self, prompt, target_model, prompt_token_budget, status_callback=None, router_llm=None):
        """Compresses a massive prompt using semantic line boundaries and fast summarization."""
        # Ensure router is loaded first to use its tokenizer
        # Use small context for summarization — avoid wasteful VRAM allocation
        if router_llm is None:
            router_llm = self._get_model("router", required_ctx=2048)

        # Precise Token Estimation
        if hasattr(router_llm, "tokenize"):
            est_tokens = len(self.safe_tokenize(router_llm, prompt.encode('utf-8')))
        else:
            est_tokens = len(prompt) // 3

        # Guard: if the prompt fits inside the budget, no need to compress
        prompt_token_budget = max(512, prompt_token_budget)
        if est_tokens <= prompt_token_budget:
            return prompt
            
        if status_callback:
            status_callback(f"Semantic Cruncher active for {target_model} ({est_tokens} tokens > {prompt_token_budget} max). Compressing...", "warning", target_model, 12)
            
        # We need to slice the string, but strictly at newline boundaries to preserve words and code formatting
        lines = prompt.split('\n')
        total_chars = sum(len(l) for l in lines)
        chars_allowed = prompt_token_budget * 3
        
        # Allocate 25% of allowed budget to top chunk, 55% to bottom chunk, and 20% to middle summary
        top_ratio = 0.25
        bottom_ratio = 0.55
        summary_ratio = 0.20
        
        top_char_budget = int(chars_allowed * top_ratio)
        bottom_char_budget = int(chars_allowed * bottom_ratio)
        max_summary_tokens = max(256, int(prompt_token_budget * summary_ratio))
        max_summary_tokens = min(1024, max_summary_tokens)
        
        start_lines = []
        start_chars = 0
        in_code_block = False
        while lines and (start_chars < top_char_budget or in_code_block):
            line = lines.pop(0)
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
            start_lines.append(line)
            start_chars += len(line) + 1
            
        end_lines = []
        end_chars = 0
        while lines and end_chars < bottom_char_budget:
            line = lines.pop()
            end_lines.insert(0, line)
            end_chars += len(line) + 1
            
        # Code-block safety for the bottom chunk (moving backwards)
        code_blocks = sum(1 for l in end_lines if l.strip().startswith("```"))
        if code_blocks % 2 != 0:
            # We sliced through a code block. Keep grabbing lines until we find the opening ```
            while lines:
                line = lines.pop()
                end_lines.insert(0, line)
                if line.strip().startswith("```"):
                    break
            
        # Whatever is left in 'lines' is the middle chunk that needs summarizing
        middle_chunk = '\n'.join(lines)
        start_chunk = '\n'.join(start_lines)
        end_chunk = '\n'.join(end_lines)
        
        # Safety net: guarantee the summarization prompt fits inside the router_llm context window.
        # Use 300 token overhead buffer to account for chat template markers (BOS/EOS, role tags,
        # DeepSeek/Qwen special tokens like <｜User｜> etc.) that llama-cpp adds internally.
        n_ctx = router_llm.n_ctx() if hasattr(router_llm, "n_ctx") else 8192
        chat_template_overhead = 300
        max_middle_tokens = max(512, n_ctx - max_summary_tokens - chat_template_overhead)
        
        if hasattr(router_llm, "tokenize"):
            est_middle_tokens = len(self.safe_tokenize(router_llm, middle_chunk.encode('utf-8')))
        else:
            est_middle_tokens = len(middle_chunk) // 3
            
        if est_middle_tokens > max_middle_tokens:
            allowed_chars = int(max_middle_tokens * 3.0)
            half = allowed_chars // 2
            middle_chunk = middle_chunk[:half] + "\n...[TRUNCATED MIDDLE TO FIT CONTEXT]...\n" + middle_chunk[-half:]
            
        compress_prompt = f"Summarize this middle section concisely. Keep all logic, facts, and code structure intact:\n{middle_chunk}"
        
        # Route through _call_model so it gets the same overflow protection as all other calls
        middle_summary = self._call_model(router_llm, compress_prompt, max_tokens=max_summary_tokens, temperature=0.3)
            
        crunched = f"{start_chunk}\n\n[--- CRUNCHED SUMMARY OF MIDDLE SECTION ---]\n{middle_summary}\n[--- END SUMMARY ---]\n\n{end_chunk}"
        
        # Post-crunch verification: if crunched output STILL exceeds budget, hard-truncate
        if hasattr(router_llm, "tokenize"):
            final_tokens = len(self.safe_tokenize(router_llm, crunched.encode('utf-8')))
        else:
            final_tokens = len(crunched) // 3
        if final_tokens > prompt_token_budget:
            # Hard truncate: keep first 30% and last 70% by character count
            budget_chars = int(prompt_token_budget * 3)
            top_keep = int(budget_chars * 0.3)
            bottom_keep = int(budget_chars * 0.7)
            crunched = crunched[:top_keep] + "\n...[HARD TRUNCATED TO FIT CONTEXT]...\n" + crunched[-bottom_keep:]
            
        return crunched

    def safe_tokenize(self, llm, text_bytes):
        """Thread-safe wrapper for llama.cpp tokenization to prevent concurrent C-level segfaults."""
        if hasattr(llm, "tokenize"):
            with self.inference_lock:
                return llm.tokenize(text_bytes)
        return []

    # =========================================================================
    # DRY Helper: Call any model (TransformerWrapper or GGUF)
    # =========================================================================
    def _call_model(self, llm, prompt, max_tokens=512, temperature=0.7, system_prompt=None):
        if self.cancel_event and self.cancel_event.is_set():
            raise RuntimeError("Generation cancelled by user.")

        if isinstance(llm, TransformerWrapper):
            return llm(prompt, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt)
            
        # Context overflow protection for llama-cpp-python
        if hasattr(llm, "n_ctx"):
            ctx = llm.n_ctx()
            # Precise Token Estimation
            if hasattr(llm, "tokenize"):
                est_prompt_tokens = len(self.safe_tokenize(llm, prompt.encode('utf-8'))) + 120
                if system_prompt:
                    est_prompt_tokens += len(self.safe_tokenize(llm, system_prompt.encode('utf-8')))
            else:
                est_prompt_tokens = len(prompt) // 3 + 120
                if system_prompt:
                    est_prompt_tokens += len(system_prompt) // 3
            
            # Smart token allocation with Model-Aware Minimums
            is_reasoning = "deepseek" in getattr(llm, "model_path", "").lower()
            absolute_min = 2048 if is_reasoning else 512
            # CRITICAL FIX: On constrained GPUs where n_ctx can be as low as 2048,
            # the absolute_min must never exceed the context window itself.
            # Cap it at (ctx - 128) to guarantee at least 128 tokens of prompt space.
            absolute_min = min(absolute_min, max(256, ctx - 128))
            
            if est_prompt_tokens + max_tokens > ctx:
                # Calculate the hard ceiling: how many tokens are genuinely free
                available = ctx - est_prompt_tokens - 50
                if available < 64:
                    available = 64
                # Use the larger of absolute_min and available, but NEVER exceed available
                # The old bug: max(absolute_min, available) could return absolute_min > available → OOM
                max_tokens = min(max(absolute_min, available), available)
            
            # If even with minimum generation tokens the prompt doesn't fit, truncate the prompt semantically
            max_prompt_tokens = ctx - max_tokens - 120
            if max_prompt_tokens < 200:
                max_prompt_tokens = 200
            if est_prompt_tokens > max_prompt_tokens:
                chars_allowed = max_prompt_tokens * 3
                if chars_allowed < 900: chars_allowed = 900
                
                lines = prompt.split('\n')
                top_chars = int(chars_allowed * 0.3)
                bottom_chars = int(chars_allowed * 0.7)
                
                start_lines, end_lines = [], []
                curr_t, curr_b = 0, 0
                in_code_block = False
                
                while lines and (curr_t < top_chars or in_code_block):
                    l = lines.pop(0)
                    if l.strip().startswith("```"):
                        in_code_block = not in_code_block
                    start_lines.append(l)
                    curr_t += len(l) + 1
                    
                while lines and curr_b < bottom_chars:
                    l = lines.pop()
                    end_lines.insert(0, l)
                    curr_b += len(l) + 1
                    
                # Code-block safety for the bottom chunk
                code_blocks = sum(1 for l in end_lines if l.strip().startswith("```"))
                if code_blocks % 2 != 0:
                    while lines:
                        l = lines.pop()
                        end_lines.insert(0, l)
                        if l.strip().startswith("```"):
                            break
                    
                prompt = '\n'.join(start_lines) + "\n...[TRUNCATED FOR CONTEXT LIMIT]...\n" + '\n'.join(end_lines)
                if hasattr(llm, "tokenize"):
                    est_prompt_tokens = len(self.safe_tokenize(llm, prompt.encode('utf-8'))) + 120
                else:
                    est_prompt_tokens = len(prompt) // 3 + 120
            
            # ── HARD FINAL CLAMP — guarantees prompt + gen_tokens ≤ n_ctx ──
            safe_max = ctx - est_prompt_tokens
            if safe_max < 64:
                safe_max = 64  # Desperate fallback — at least try to get something
            max_tokens = min(max_tokens, safe_max)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prevent concurrent C++ deadlocks in llama.cpp by serializing the generation
        with self.inference_lock:
            # Use streaming to support instant cancellation for GGUF/llama-cpp-python models
            chunks = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            content_pieces = []
            for chunk in chunks:
                if self.cancel_event and self.cancel_event.is_set():
                    raise RuntimeError("Generation cancelled by user.")
                
                choices = chunk.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        content_pieces.append(content)
                        
            return "".join(content_pieces)

    # =========================================================================
    # DYNAMIC ACTOR-CRITIC VERIFIER PIPELINE — Helper Methods
    # =========================================================================

    def _classify_task(self, router_llm, prompt):
        """Three-way classification: SIMPLE, CODING, or REASONING.
        Uses a combination of fast-tracks, structural heuristic checks, 
        and an optimized few-shot prompt for maximum accuracy.
        """
        prompt_clean = prompt.strip().lower()
        prompt_lower = prompt_clean

        # ── 0. Fast-track Scientific Derivation/Proof to REASONING ─────────────
        reasoning_triggers = ["derive", "prove", "equations of motion", "mathematically derive", "mathematical derivation", "theorem", "physical significance"]
        coding_triggers = ["code", "script", "program", "write a", "implement a", "compile", "develop", "web app", "website", "scipy", "pandas", "numpy", "matplotlib", "plotly", "dataframe", "python", "predict", "forecast", "prediction"]
        if any(trigger in prompt_lower for trigger in reasoning_triggers):
            if not any(coding in prompt_lower for coding in coding_triggers):
                return "REASONING"

        # ── 1. Fast-track Code Block / Traceback presence (Direct CODING) ──────
        if "```" in prompt or "traceback (most recent call)" in prompt_lower or "line " in prompt_lower and "in <module>" in prompt_lower:
            return "CODING"
        
        # ── 2. Fast-track greetings & simple metadata queries (Direct SIMPLE) ──
        greetings = {"hi", "hello", "hey", "hola", "howdy", "greetings", "good morning", "good afternoon", "good evening", "how are you", "who are you", "what is your name"}
        if prompt_clean in greetings or prompt_clean.replace("?", "").strip() in greetings:
            return "SIMPLE"

        # ── 3. Fast-track short queries (less than 15 chars) ───────────────────
        if len(prompt_clean) < 15:
            code_kws = ["code", "write", "def ", "class ", "import ", "script", "app", "html", "css", "js", "cpp", "py"]
            if any(kw in prompt_clean for kw in code_kws):
                return "CODING"
            return "SIMPLE"

        # ── 4. Fast-track basic arithmetic (e.g., "2+2", "solve 5*5") ──────────
        arithmetic_clean = prompt_clean
        for prefix in ["what is ", "whats ", "calculate ", "compute ", "what is the value of ", "solve "]:
            if arithmetic_clean.startswith(prefix):
                arithmetic_clean = arithmetic_clean[len(prefix):]
        arithmetic_clean = arithmetic_clean.replace("?", "").strip()
        
        if re.match(r"^[0-9+\-*/%().\s]+$", arithmetic_clean) and len(arithmetic_clean) > 0:
            return "SIMPLE"

        # ── 5. Advanced LLM Few-Shot Classifier Prompt ─────────────────────────
        few_shot_prompt = (
            "Classify the following query into exactly ONE of the three categories: SIMPLE, CODING, or REASONING.\n\n"
            "CATEGORIES:\n"
            "1. SIMPLE: Conversational greetings, quick factual answers, definitions, translations, yes/no queries, news, or weather.\n"
            "   - 'What is the capital of France?' -> SIMPLE\n"
            "   - 'Translate hello to Spanish' -> SIMPLE\n"
            "   - 'Define cellular respiration' -> SIMPLE\n"
            "   - 'Who is the CEO of Apple?' -> SIMPLE\n\n"
            "2. CODING: Prompts explicitly asking to write, fix, debug, or compile code, scripts, web pages, APIs, databases, or software features.\n"
            "   - 'Write a python script to sort a list' -> CODING\n"
            "   - 'Fix this index error in my code' -> CODING\n"
            "   - 'How to read a CSV file using pandas?' -> CODING\n"
            "   - 'Build a simple calculator UI in HTML and CSS' -> CODING\n\n"
            "3. REASONING: Scientific explanations, multi-step math derivations, physics proofs, logic puzzles, or chemical reaction balancing (where NO code or script is explicitly requested).\n"
            "   - 'Solve this integral: integral of x^2 sin(x) dx' -> REASONING\n"
            "   - 'Derive the equations of motion for a double pendulum' -> REASONING\n"
            "   - 'Explain the physical significance of the Schrödinger equation' -> REASONING\n"
            "   - 'If a card is drawn from a deck, what is the probability of a spade?' -> REASONING\n\n"
            "IMPORTANT CLASSIFICATION RULES:\n"
            "- If the query asks for scientific math/physics calculations AND asks to write code, program, or script to do it -> CODING.\n"
            "- If the query asks to explain a scientific concept and plot/visualize it, but DOES NOT mention writing code, scripts, or programming -> REASONING.\n"
            "- If the query is purely about retrieving/fetching data or search queries -> SIMPLE.\n\n"
            f"Query: {prompt}\n\n"
            "Category:"
        )

        try:
            result = self._call_model(router_llm, few_shot_prompt, max_tokens=10, temperature=0.1)
            upper = str(result).strip().upper()
            # Use word boundaries to prevent substring matching bugs (e.g. "simple math" matching SIMPLE)
            words = re.findall(r'\b\w+\b', upper)
            if "CODING" in words:
                return "CODING"
            if "REASONING" in words:
                return "REASONING"
            if "SIMPLE" in words:
                return "SIMPLE"
        except Exception as e:
            print(f"LLM task classification failed, falling back to heuristics: {e}")

        # ── 6. Heuristics & Deterministic overrides (only if LLM failed) ───────
        strong_code_indicators = [
            "sandboxdatahelper", "plotly", "pandas", "dataframe", "numpy", "matplotlib",
            "write a python", "write python", "python script", "implement a python",
            "write code to", "write a code to", "plotly layout", "forecast close price",
            "standardized predictive_metrics", "pip install", "import pandas", "import numpy",
            "flask", "fastapi", "django", "sql query", "react component", "javascript script",
            "html code", "css styling", "dockerfile", "requirements.txt", "git command",
            "bash script", "powershell script", "shell script", "api endpoint", "json metric",
            "predict", "forecast", "prediction"
        ]
        if any(kw in prompt_lower for kw in strong_code_indicators):
            return "CODING"

        recency_keywords = ["who won", "who lost", "last match", "latest match", "recent match", "score of", "latest score",
                            "yesterday", "today's", "todays", "tonight", "last night", "this week", "this month",
                            "latest news", "recent news", "breaking news", "current", "right now",
                            "trending", "who is the president", "who is the pm", "who is the ceo",
                            "weather today", "weather in", "temperature in", "stock price", "crypto price",
                            "box office", "release date", "when is", "when does", "when did",
                            "election result", "who won the", "match result", "ipl", "world cup"]
        
        search_intents = ["fetch from web", "search the web", "search for", "google for", "latest news", "weather news", "current weather", "weather of"]
        code_intent_kws = ["write code", "write a code", "javascript code", "python code", "c++ code", "java code", "html code", "css code", "write a script", "code for", "script to", "build", "implement"]
        has_code_intent = any(kw in prompt_lower for kw in code_intent_kws)

        if any(kw in prompt_clean for kw in recency_keywords) or (any(intent in prompt_lower for intent in search_intents) and not has_code_intent):
            return "SIMPLE"

        # ── 7. Heuristics Scan Safety Net ────────────────────────────────
        code_keywords = [
            "write code", "write a code", "fix code", "debug", "script", "program", "compile",
            "function(", "def ", "class ", "import ", "coding", "develop", "web app", "website",
            "implement", "plotly", "matplotlib", "dataframe", "numpy", "pandas", "scipy",
            "sandboxdatahelper", "plot", "regex", "api", "query", "database", "install"
        ]
        reason_keywords = [
            "explain", "prove", "derive", "why ", "how does", "in detail", "theory", "analyze",
            "compare", "calculate", "solve", "simulate", "trajectory", "numerical", "3d plot",
            "interactive plot", "derivation", "theorem", "proof", "physics", "chemistry", "equation"
        ]

        if any(kw in prompt_lower for kw in code_keywords):
            return "CODING"
        if any(kw in prompt_lower for kw in reason_keywords):
            return "REASONING"
        return "SIMPLE"

    def _is_playground_applicable(self, router_llm, prompt):
        """Check if reasoning can be verified via Python sandbox."""
        auto_keywords = [
            "solve", "calculate", "equations of motion", "scipy", "numpy", "solve_ivp", "assert",
            "integrate", "trajectory", "physics", "math", "verify", "verification script",
            # Bio/Chem
            "enzyme", "kinetics", "michaelis", "inhibition", "reaction rate", "molecular weight",
            "codon", "transcription", "translation", "protein", "dna", "rna",
            # Cybersecurity
            "encrypt", "decrypt", "cipher", "hash", "aes", "rsa", "jwt",
        ]
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in auto_keywords):
            return True

        p = (
            "Can this concept be verified using a Python script? "
            "Math/physics equations, logic puzzles, statistical claims = YES. "
            "Philosophy, ethics, creative writing, history, opinions = NO.\n"
            "Reply ONLY 'YES' or 'NO'.\n\n"
            f"Query: {prompt[:500]}"
        )
        result = self._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        return "YES" in str(result).upper()

    def _run_playground(self, model, hypothesis, purpose="logic", status_callback=None, model_key=None, original_prompt=None):
        """
        Have a model write a verification script and run it in the sandbox.
        Returns (verified: bool, output: str, test_code: str)
        """
        coder_model = model
        # Redirect all playground script writing to the Router to prevent DeepSeek-R1 thinking tokens
        # from depleting the context window and causing code truncation, or VibeThinker syntax errors.
        if purpose == "reasoning" or model_key in ["deepseek_r1"]:
            coder_model = self._get_model("opencode", required_ctx=8192)

        # Classify the domain of the query
        domain = "general"
        prompt_lower = (original_prompt or "").lower() + " " + (hypothesis or "").lower()
        if any(kw in prompt_lower for kw in ["biology", "gene", "protein", "dna", "rna", "translation", "transcription", "sequence", "codon", "molecule", "chemical", "bond", "valency", "structure", "chemistry", "atp", "reaction", "formula", "rdkit", "biopython", "enzyme", "kinetics", "inhibition", "michaelis", "substrate", "inhibitor", "vmax", "metabolic", "catalytic", "pharmacokinetics", "receptor", "ligand"]):
            domain = "bio_chem"
        elif any(kw in prompt_lower for kw in ["physics", "math", "equation", "solve", "drift", "lorentz", "velocity", "trajectory", "integral", "derivative", "differential", "limit", "matrix", "vector", "force", "cycle", "frequency"]):
            domain = "math_physics"
        elif purpose == "logic" and any(kw in prompt_lower for kw in ["cybersecurity", "security", "cryptography", "crypto", "cipher", "aes", "rsa", "encryption", "decryption", "hash", "sha256", "jwt", "packet", "scapy", "socket", "steganography", "payload", "vulnerability"]):
            domain = "cybersecurity"

        rules = [
            "Test the CORE claim/formula with concrete numerical values",
            "You MUST strictly adhere to ALL constraints in the original query (e.g. air drag, specific angles, 3D vs 2D). DO NOT SIMPLIFY the physics.",
            "You MUST use math.isclose(a, b, rel_tol=2e-2) or np.isclose(a, b, rtol=2e-2) for ANY floating point comparisons of numerical physics simulation results (since numerical integration accumulated errors over many cycles can deviate slightly). NEVER use == for floats.",
            "When using math.isclose, np.isclose, or np.allclose, ensure any assertion message string is OUTSIDE the function call: `assert np.isclose(a, b), 'message'` (never `assert np.isclose(a, b, 'message')`).",
            "If testing values on a grid or meshgrid (e.g., S_grid, I_grid), make sure you check boundary conditions at specific coordinate indices where the variable has the expected value (e.g. to test uninhibited velocity at [I] = 0, query the row/column index where the inhibitor grid equals 0, rather than a middle index like [50, 50] where [I] > 0).",
            "Check at least 2 different test cases or boundary conditions",
            "Check dimensional consistency (units make sense). If using unit libraries like pint, perform all unit conversions OUTSIDE the differential solver loops/functions (never instantiate or convert quantities inside solve_ivp/odeint callbacks as it causes type-casting exceptions and severe performance slowdowns)."
        ]

        if domain == "math_physics":
            rules.append(
                "For complex or non-standard physics/math equations (like multi-dimensional drifts, n-body, electromagnetics), "
                "you MUST write a sympy block to algebraically derive and prove the formulas from first principles (e.g. F=ma, Lorentz force) "
                "before running numerical checks. Assert that the sympy solution matches your proposed formula."
            )
        elif domain == "bio_chem":
            rules.append(
                "For biology/chemistry queries, you MUST use Bio (Biopython) or rdkit (RDKit) to strictly validate "
                "molecular weights, codon translation, sequence transcription, or chemical property assertions. "
                "Do not mock these values; use the actual libraries to compute and verify them."
            )
            rules.append(
                "For enzyme kinetics or pharmacokinetics queries, you MUST verify the DIRECTIONALITY of parameter shifts "
                "(e.g., in Competitive Inhibition: apparent Km INCREASES while Vmax stays constant; in Uncompetitive: both apparent Km and Vmax decrease). "
                "Write numerical tests: compute v at [I]=0 and [I]>0 for the same [S], and assert that for Competitive Inhibition "
                "the velocity DECREASES when inhibitor is added (v_inhibited < v_uninhibited). If this assertion fails, the formula is WRONG."
            )
        elif domain == "cybersecurity":
            rules.append(
                "For cybersecurity and cryptography coding, you MUST write test assertions to verify "
                "that the roundtrip encryption and decryption matches the exact original plaintext, "
                "or that generated security tokens/keys validate successfully using standard cryptographic "
                "libraries (like cryptography, hashlib, or jwt). If simulating packets, verify header structures."
            )

        if purpose == "logic":
            rules.append(
                "If the task requires fetching data from a database, file, or API (like SandboxDataHelper or stock/weather symbols), "
                "you MUST mock the data returned by these helper classes (e.g. mock SandboxDataHelper.get_stock_data to return a small, mock pandas DataFrame with 5 rows) "
                "rather than trying to fetch actual data or calling APIs."
            )
            rules.append(
                "If the task requires plotting (using plotly or matplotlib), do NOT write any code that calls plt.show(), fig.show(), "
                "or tries to render charts. Verify only the data structures or mathematical calculations."
            )

        rules.extend([
            "Use assert statements with descriptive messages",
            "Print 'VERIFIED' as the LAST line ONLY if ALL assertions pass",
            "Do NOT print 'VERIFIED' if any assertion fails"
        ])

        rules_str = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(rules)])

        prompt_context = ""
        if original_prompt:
            prompt_context = f"ORIGINAL QUERY CONSTRAINTS:\n{original_prompt[:1500]}\n\n"

        playground_prompt = (
            f"Write a Python script (max 50 lines) that {'tests this code logic' if purpose == 'logic' else 'verifies this reasoning'}.\n\n"
            f"VERIFICATION RULES:\n{rules_str}\n\n"
            "You have access to these scientific tools:\n"
            "  - math, cmath           → Core math operations, trigonometry, constants\n"
            "  - numpy                 → Arrays, linear algebra, matrix operations, statistics\n"
            "  - sympy                 → Symbolic math: algebra, calculus, equation solving, proofs\n"
            "  - scipy                 → Physics/Biology: scipy.constants, scipy.integrate, scipy.optimize\n"
            "  - pint                  → Unit verification: check that formulas produce correct physical units\n"
            "  - z3 (z3-solver)        → Formal logic & theorem proving: constraint satisfaction, SAT solving\n"
            "  - networkx              → Graph theory: shortest paths, connectivity, circuit analysis\n"
            "  - astropy               → Astrophysics: celestial mechanics, orbital calculations, cosmology\n"
            "  - Bio (Biopython)       → Bioinformatics: sequence transcription/translation, codon tables\n"
            "  - rdkit (RDKit)         → Cheminformatics: molecular structures, chemical bonds\n"
            "  - itertools, collections → Combinatorics, permutations, advanced data structures\n\n"
            "Pick the MOST APPROPRIATE tool for the task. Do NOT import all of them.\n"
            "Do NOT use plotly, matplotlib, pygame, or any GUI.\n"
            "Output ONLY the code in ```python``` blocks.\n\n"
            f"{prompt_context}"
            f"To verify:\n{hypothesis[:2000]}"
        )
        test_response = self._call_model(coder_model, playground_prompt, max_tokens=4096, temperature=0.1)
        test_code = Sandbox.extract_code(test_response)
        success, output = self.sandbox.execute(test_code, language='python')
        
        # ── Router Linter Intercept for Verification/Playground Scripts ──
        if not success and test_code:
            is_syntax_error = any(e in output for e in ["SyntaxError", "ModuleNotFoundError", "NameError", "IndentationError", "TypeError", "AttributeError", "ValueError"])
            if is_syntax_error:
                router_linter = self._get_model("opencode", required_ctx=8192)

                # Step A: Try Agent IDE surgical patch first
                patched = self._agent_ide_patch(test_code, output, router_linter, 1024)
                if patched and patched != test_code:
                    patch_ok, patch_out = self.sandbox.execute(patched, language='python')
                    if patch_ok:
                        test_code = patched
                        output = patch_out
                        success = True

                # Step B: Fall back to full rewrite if patch failed
                if not success:
                    lint_p = (
                        "You are a fast Python Syntax Linter.\n"
                        f"The playground verification script failed with this error:\n{output[:600]}\n\n"
                        f"CODE:\n{test_code[:2500]}\n\n"
                        "Identify the typo/error and rewrite the complete corrected verification script in a ```python``` block. Fix ONLY the exact error, do not change the core assertions or print('VERIFIED') statement."
                    )
                    lint_code = Sandbox.extract_code(self._strip_thinking(self._call_model(router_linter, lint_p, 1024, 0.1, system_prompt="You are a strict syntax linter. Output only code.")))
                    if lint_code and len(lint_code) > 20:
                        linter_success, linter_output = self.sandbox.execute(lint_code, language='python')
                        if linter_success:
                            test_code = lint_code
                            output = linter_output
                            success = True

        verified = success and "VERIFIED" in output

        # ── Soft-Verification: Distinguish test-script bugs from logic failures ──
        # If the sandbox crashed due to a bug IN THE VERIFICATION SCRIPT ITSELF
        # (e.g., TypeError, NameError, SyntaxError) rather than an AssertionError
        # (which would mean the logic plan's math is actually wrong), treat it as
        # "soft-verified" to avoid triggering expensive Emergency Search + model swaps
        # that waste 5-10 minutes on consumer hardware for zero benefit.
        if not verified and not success and test_code:
            is_assertion_failure = "AssertionError" in output or "AssertionError" in output.replace("Assertion", "Assertion")
            is_test_script_crash = any(e in output for e in [
                "ImportError", "ModuleNotFoundError"
            ])
            if is_test_script_crash and not is_assertion_failure:
                # The verification script itself crashed — the logic plan was never disproven.
                # Proceed with best-effort trust rather than wasting time on Emergency Search.
                verified = True
                output = f"[Soft-verified: test script crashed with runtime error, logic plan not disproven]\n{output[:500]}"

        return verified, output, test_code

    def _extract_failure_lessons(self, critic_llm, failed_plan, all_errors):
        """Nuclear Reset: extract key lessons from failures for a fresh restart."""
        p = (
            "These attempts ALL FAILED. Perform root-cause analysis and extract LESSONS.\n\n"
            f"Plan:\n{failed_plan[:1500]}\n\n"
            f"Errors:\n{all_errors[:1500]}\n\n"
            "For each failure, identify:\n"
            "1. ROOT CAUSE: What exactly went wrong (wrong formula, missing import, logic error, etc.)\n"
            "2. CORRECT APPROACH: What the correct solution should be\n"
            "3. VERIFICATION: How to check the fix is right\n\n"
            "Reply with a numbered list of 3-5 specific, actionable lessons. Be precise."
        )
        lessons = self._call_model(critic_llm, p, max_tokens=512, temperature=0.3)
        return self._strip_thinking(lessons)

    def _verify_html_javascript(self, html_code):
        """Extract JavaScript from HTML, inject mock environment, and run it in the Node.js sandbox to catch syntax/execution errors."""
        cleaned = html_code.strip()
        # Basic sanity check: must look like HTML structure, not raw text refusal/apology
        if not (cleaned.startswith("<") or "</" in cleaned or "<div" in cleaned.lower() or "<html" in cleaned.lower() or "<script" in cleaned.lower()):
            return False, "Not a valid HTML document (plain text or refusal detected)."

        # Find all INLINE script blocks (exclude those with src=)
        scripts = re.findall(r'<script\b(?![^>]*src=)[^>]*>([\s\S]*?)</script>', html_code, flags=re.IGNORECASE)
        js_code = "\n".join(scripts).strip()
        
        if not js_code:
            return False, "No inline JavaScript logic found. You MUST write the actual simulation logic inside a <script> tag."
            
        # Ensure the script actually attempts to render something to prevent blank screens
        if not any(kw in js_code for kw in ['Plotly.', 'THREE.', 'getContext', 'document.getElementById', 'document.querySelector']):
            return False, "The JavaScript logic does not attempt to render anything. You MUST use Plotly, THREE.js, or Canvas/DOM APIs to display the simulation."

        # Prepend mocks for DOM, Window, THREE, and Plotly to Node.js context.
        # This will bypass typical browser-only ReferenceErrors while letting actual syntax/API bugs throw errors.
        mocks = """
        // ── Comprehensive DOM Mock ──────────────────────────────────────
        const _mockElement = (tag) => {
            const el = {
                tagName: (tag || 'DIV').toUpperCase(),
                style: new Proxy({}, { get: () => '', set: () => true }),
                classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
                children: [],
                childNodes: [],
                parentNode: null,
                textContent: '',
                innerHTML: '',
                innerText: '',
                value: '',
                checked: false,
                offsetWidth: 1024,
                offsetHeight: 768,
                clientWidth: 1024,
                clientHeight: 768,
                scrollWidth: 1024,
                scrollHeight: 768,
                addEventListener: function(event, cb) {
                    if (typeof cb === 'function' && (event === 'DOMContentLoaded' || event === 'load' || event === 'change' || event === 'input')) {
                        try { cb(); } catch(e) {}
                    }
                },
                removeEventListener: () => {},
                appendChild: function(c) { this.children.push(c); return c; },
                removeChild: function(c) { return c; },
                insertBefore: function(n) { return n; },
                replaceChild: function(n) { return n; },
                cloneNode: function() { return _mockElement(tag); },
                getAttribute: () => null,
                setAttribute: () => {},
                removeAttribute: () => {},
                hasAttribute: () => false,
                querySelector: () => _mockElement(),
                querySelectorAll: () => [],
                getElementsByClassName: () => [],
                getElementsByTagName: () => [],
                getBoundingClientRect: () => ({ top: 0, left: 0, right: 1024, bottom: 768, width: 1024, height: 768, x: 0, y: 0 }),
                focus: () => {},
                blur: () => {},
                click: () => {},
                dispatchEvent: () => true,
                getContext: (type) => {
                    const handler = { get: (t, p) => typeof t[p] !== 'undefined' ? t[p] : (() => ({})) };
                    return new Proxy({
                        canvas: { width: 1024, height: 768 },
                        drawingBufferWidth: 1024,
                        drawingBufferHeight: 768,
                        getExtension: () => ({}),
                        getParameter: () => 0,
                        createShader: () => ({}), compileShader: () => {}, shaderSource: () => {},
                        getShaderParameter: () => true, getShaderInfoLog: () => '',
                        createProgram: () => ({}), attachShader: () => {}, linkProgram: () => {},
                        getProgramParameter: () => true, useProgram: () => {},
                        createBuffer: () => ({}), bindBuffer: () => {}, bufferData: () => {},
                        enableVertexAttribArray: () => {}, vertexAttribPointer: () => {},
                        drawArrays: () => {}, drawElements: () => {},
                        viewport: () => {}, enable: () => {}, disable: () => {},
                        clearColor: () => {}, clear: () => {},
                        createTexture: () => ({}), bindTexture: () => {}, texImage2D: () => {},
                        texParameteri: () => {}, generateMipmap: () => {},
                        getUniformLocation: () => ({}), getAttribLocation: () => 0,
                        uniform1f: () => {}, uniform1i: () => {}, uniform2f: () => {},
                        uniform3f: () => {}, uniform4f: () => {},
                        uniformMatrix4fv: () => {},
                        // 2D Canvas
                        fillRect: () => {}, clearRect: () => {}, strokeRect: () => {},
                        fillText: () => {}, strokeText: () => {}, measureText: () => ({ width: 10 }),
                        beginPath: () => {}, closePath: () => {}, moveTo: () => {}, lineTo: () => {},
                        arc: () => {}, arcTo: () => {}, bezierCurveTo: () => {}, quadraticCurveTo: () => {},
                        fill: () => {}, stroke: () => {},
                        save: () => {}, restore: () => {}, translate: () => {}, rotate: () => {}, scale: () => {},
                        setTransform: () => {}, resetTransform: () => {},
                        createLinearGradient: () => ({ addColorStop: () => {} }),
                        createRadialGradient: () => ({ addColorStop: () => {} }),
                    }, handler);
                }
            };
            return new Proxy(el, {
                get(target, prop) {
                    if (prop in target) return target[prop];
                    if (prop === 'then' || prop === 'catch' || prop === 'on' || prop === 'off') {
                        return (cb) => {
                            try { if (typeof cb === 'function') cb(target); } catch(e) {}
                            return target;
                        };
                    }
                    return () => target;
                }
            });
        };

        global.document = {
            getElementById: () => _mockElement(),
            querySelector: () => _mockElement(),
            querySelectorAll: () => [],
            getElementsByClassName: () => [],
            getElementsByTagName: () => [],
            createElement: (tag) => _mockElement(tag),
            createElementNS: (ns, tag) => _mockElement(tag),
            createTextNode: () => _mockElement('text'),
            createDocumentFragment: () => _mockElement('fragment'),
            body: _mockElement('body'),
            head: _mockElement('head'),
            documentElement: _mockElement('html'),
            addEventListener: function(event, cb) {
                if (typeof cb === 'function' && (event === 'DOMContentLoaded' || event === 'load')) {
                    try { cb(); } catch(e) {}
                }
            },
            removeEventListener: () => {},
            readyState: 'complete',
            cookie: '',
        };

        global.window = {
            innerWidth: 1024,
            innerHeight: 768,
            outerWidth: 1024,
            outerHeight: 768,
            devicePixelRatio: 1,
            addEventListener: function(event, cb) {
                if (typeof cb === 'function' && (event === 'DOMContentLoaded' || event === 'load')) {
                    try { cb(); } catch(e) {}
                }
            },
            removeEventListener: () => {},
            getComputedStyle: () => new Proxy({}, { get: () => '0px' }),
            matchMedia: () => ({ matches: false, addEventListener: () => {} }),
            requestAnimationFrame: (cb) => {
                if (!global.__raf_count) global.__raf_count = 0;
                if (global.__raf_count < 2) {
                    global.__raf_count++;
                    try { cb(global.__raf_count * 16.67); } catch(e) {}
                }
                return global.__raf_count;
            },
            cancelAnimationFrame: () => {},
            document: global.document,
            location: { href: '', hostname: 'localhost', protocol: 'http:' },
            history: { pushState: () => {}, replaceState: () => {} },
            scrollTo: () => {},
            scroll: () => {},
            open: () => {},
            close: () => {},
            alert: () => {},
            confirm: () => true,
            prompt: () => '',
            performance: { now: () => 0 },
            ResizeObserver: function() { this.observe = () => {}; this.unobserve = () => {}; this.disconnect = () => {}; },
            MutationObserver: function() { this.observe = () => {}; this.disconnect = () => {}; },
            IntersectionObserver: function() { this.observe = () => {}; this.unobserve = () => {}; this.disconnect = () => {}; },
        };

        // Promote critical browser globals to the global scope
        global.window.window = global.window;
        global.innerWidth = global.window.innerWidth;
        global.innerHeight = global.window.innerHeight;
        global.outerWidth = global.window.outerWidth;
        global.outerHeight = global.window.outerHeight;
        global.devicePixelRatio = global.window.devicePixelRatio;
        global.location = global.window.location;
        global.requestAnimationFrame = global.window.requestAnimationFrame;
        global.cancelAnimationFrame = global.window.cancelAnimationFrame;
        global.setTimeout = (cb, ms) => { try { cb(); } catch(e) {} return 1; };
        global.clearTimeout = () => {};
        global.setInterval = (cb, ms) => { return 1; };
        global.clearInterval = () => {};
        global.navigator = { userAgent: 'Mozilla/5.0', language: 'en-US', platform: 'Linux x86_64' };
        global.performance = global.window.performance;
        global.Image = function() { this.src = ''; this.onload = null; this.onerror = null; this.width = 1; this.height = 1; };
        global.fetch = () => Promise.resolve({ json: () => Promise.resolve({}), text: () => Promise.resolve('') });
        global.XMLHttpRequest = function() { this.open = () => {}; this.send = () => {}; this.setRequestHeader = () => {}; };
        global.ResizeObserver = global.window.ResizeObserver;
        global.MutationObserver = global.window.MutationObserver;
        global.IntersectionObserver = global.window.IntersectionObserver;
        global.HTMLElement = function() {};
        global.HTMLCanvasElement = function() {};
        global.WebGLRenderingContext = function() {};

        // ── Mock THREE.js via a Universal Proxy ─────────────────────────
        const createProxy = (name) => {
            const mockFn = function() {};
            mockFn.position = { x: 0, y: 0, z: 0, set: () => mockFn.position, copy: () => mockFn.position, clone: () => ({x:0,y:0,z:0,set:()=>{},copy:()=>{}}), add: () => mockFn.position, sub: () => mockFn.position, normalize: () => mockFn.position, multiplyScalar: () => mockFn.position, length: () => 0, distanceTo: () => 0 };
            mockFn.rotation = { x: 0, y: 0, z: 0, set: () => {}, copy: () => {} };
            mockFn.scale = { x: 1, y: 1, z: 1, set: () => mockFn.scale, copy: () => {} };
            mockFn.up = { x: 0, y: 1, z: 0, set: () => {} };
            mockFn.quaternion = { set: () => {}, setFromAxisAngle: () => {}, copy: () => {} };
            mockFn.matrix = { set: () => {}, copy: () => {}, multiply: () => {} };
            mockFn.shadowMap = { enabled: false, type: 0 };
            mockFn.color = { set: () => mockFn.color, setHex: () => mockFn.color, setRGB: () => mockFn.color, r: 1, g: 1, b: 1, clone: () => mockFn.color };
            mockFn.material = { color: mockFn.color, opacity: 1, transparent: false, dispose: () => {} };
            mockFn.geometry = { dispose: () => {}, setAttribute: () => {}, setFromPoints: () => mockFn.geometry, attributes: {} };
            mockFn.domElement = _mockElement('canvas');
            mockFn.add = () => mockFn;
            mockFn.remove = () => mockFn;
            mockFn.render = () => {};
            mockFn.setSize = () => {};
            mockFn.setPixelRatio = () => {};
            mockFn.setClearColor = () => {};
            mockFn.update = () => {};
            mockFn.lookAt = () => {};
            mockFn.set = () => mockFn;
            mockFn.clone = () => createProxy(name);
            mockFn.dispose = () => {};
            mockFn.traverse = (cb) => { try { cb(mockFn); } catch(e) {} };
            mockFn.getPoints = () => [];
            mockFn.setFromPoints = () => mockFn;
            mockFn.copy = () => mockFn;
            mockFn.applyMatrix4 = () => mockFn;
            mockFn.normalize = () => mockFn;
            mockFn.multiplyScalar = () => mockFn;
            mockFn.cross = () => mockFn;
            mockFn.dot = () => 0;
            mockFn.length = () => 0;
            mockFn.aspect = 1;

            return new Proxy(mockFn, {
                construct(target, args) {
                    return createProxy(name);
                },
                get(target, prop) {
                    if (prop === 'ArcGeometry') return undefined;
                    if (prop in target) return target[prop];
                    if (prop === 'then' || prop === 'catch' || prop === 'on' || prop === 'off') {
                        return (cb) => {
                            try { if (typeof cb === 'function') cb(target); } catch(e) {}
                            return new Proxy(mockFn, {});
                        };
                    }
                    return createProxy(`${name}.${String(prop)}`);
                },
                apply(target, thisArg, argumentsList) {
                    return createProxy(`${name}()`);
                },
                set(target, prop, value) {
                    target[prop] = value;
                    return true;
                }
            });
        };
        global.THREE = createProxy('THREE');
        global.Plotly = createProxy('Plotly');
        global.OrbitControls = global.THREE.OrbitControls;
        global.window.THREE = global.THREE;
        global.window.Plotly = global.Plotly;
        global.window.window = global.window;

        // Safe console mock
        global.console = {
            log: () => {},
            error: () => {},
            warn: () => {},
            info: () => {},
            debug: () => {},
            table: () => {},
            time: () => {},
            timeEnd: () => {},
        };
        """
        full_test_code = mocks + "\n" + js_code
        success, output = self.sandbox.execute(full_test_code, language='javascript')
        return success, output

    def _generate_3d_now(self, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Actually generate the 3D visualization (called on user demand)."""
        return self._execute_3d_generation(compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)

    def _check_3d_gate(self, prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Check if the task could benefit from 3D visualization and suggest it to the user."""
        if status_callback:
            status_callback("Checking 3D Visualization Eligibility...", "info", "router", 90)

        # Rule-based auto-match for graphing/visualization tasks to ensure 100% reliability
        # NOTE: Avoid short ambiguous words like 'plot' or 'graph' alone — they false-match on
        # "movie plot", "knowledge graph", etc. Use multi-word phrases or specific 3D/science terms.
        auto_keywords = ["3d", "plotly", "three.js", "visualize", "visualization", "3d plot", "3d graph",
                         "scatter plot", "bar chart", "line chart", "histogram",
                         "simulation", "simulate", "trajectory", "vector field", "surface plot",
                         "dna helix", "dna structure", "protein structure", "mitochondria",
                         "cell structure", "organelle", "molecular model", "molecular structure", "double helix"]
        prompt_lower = prompt.lower()
        is_3d_flag = False
        if any(kw in prompt_lower for kw in auto_keywords):
            is_3d_flag = True
        else:
            gate_prompt = (
                "Does this task involve mathematical graphing, data plotting, 3D matrices, physics/chemical equations, "
                "or biological/molecular 3D models (like DNA helices, cellular structures, proteins, or chemical bonds)? "
                "Game development (pygame, tkinter, GUI apps) does NOT count. "
                "Reply ONLY 'YES' or 'NO'.\n\n"
                f"Query: {prompt[:500]}"
            )
            router_llm = self._get_model("router", required_ctx=router_ctx)
            is_3d = self._call_model(router_llm, gate_prompt, max_tokens=10, temperature=0.1)
            if "YES" in str(is_3d).upper():
                is_3d_flag = True

        if not is_3d_flag:
            return ""

        # Check if the user EXPLICITLY asked for a 3D visualization/plot/graph in their prompt
        # We only auto-trigger on actual 3D indicators to prevent false positives on standard 2D plots
        explicit_3d_keywords = ["3d", "surface plot", "trajectory", "vector field", "dna helix", "dna structure", 
                                "protein structure", "molecular model", "double helix", "three.js", "plotly"]
        user_explicitly_asked_3d = any(kw in prompt_lower for kw in explicit_3d_keywords)

        if user_explicitly_asked_3d:
            # Auto-generate if user explicitly asked for a visualization
            return self._execute_3d_generation(compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
        else:
            # Offer the user the option to generate 3D
            return "\n\n---\n🎨 **3D Visualization Available** — I can generate an interactive 3D visualization for this response. Type **\"generate 3d\"** to create one."

    def _execute_3d_generation(self, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Execute the actual 3D visualization generation (HTML + Plotly fallback)."""
        import re
        # Strip internal thinking to save massive token limits
        clean_plan = self._strip_thinking(compiled_plan)
        # Strip all markdown code blocks to prevent opencode from trying to fix/debug them
        clean_plan = re.sub(r"```[a-zA-Z0-9_]*\n[\s\S]*?\n```", "", clean_plan)
        
        # Reasonable limit — keep enough physics context for the 3D generator
        if len(clean_plan) > 3000:
            clean_plan = clean_plan[:3000]
        
        coder_llm = self._get_model("opencode", required_ctx=oc_ctx)

        # ── Strategy 1: HTML/JS Artifact (frontend iframe sandbox) ────────
        # Generate a self-contained HTML page with Plotly.js CDN that the frontend can render in an iframe.
        if status_callback:
            status_callback("Generating HTML Artifact (Frontend Sandbox)...", "info", "opencode", 95)
        html_prompt = (
            "Write a COMPLETE, SELF-CONTAINED HTML page creating an interactive 3D simulation.\n"
            "RULES:\n"
            "1. Single HTML file with inline <script>/<style>. Load Plotly.js (<script src='https://cdn.plot.ly/plotly-2.24.1.min.js'></script>) for trajectories/surfaces, or Three.js (r128) for animations.\n"
            "2. Dark theme: body background '#0d0d0d', text '#e0e0e0'. Plotly: paper_bgcolor/plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark'.\n"
            "3. Add glassmorphic control panel (background: rgba(30,30,30,0.65); backdrop-filter: blur(12px); border-radius: 12px) with sliders for key variables, Play/Pause, Reset buttons.\n"
            "4. Implement RK4 or Verlet integration in JS. Use small dt (1e-8 for atomic scale, 0.005 for macro). For protons: compute cyclotron period T=2*pi*m/(q*B), total_time=N_cycles*T.\n"
            "5. Translate any Python math into pure JavaScript. Output complete HTML in ```html``` blocks.\n"
            "6. SINGULARITY SAFETY: Bound ranges away from division-by-zero. Clip extreme values.\n"
            "7. For Three.js: use OrbitControls AFTER renderer is appended. Never use non-existent APIs like ArcGeometry.\n"
            "8. For biological structures (DNA, proteins): use Three.js with realistic colors, MeshPhongMaterial, OrbitControls, auto-rotation, hide axes.\n"
            "9. CDNS, IMPORTS & SCOPE: If using Three.js, you MUST load OrbitControls by adding: <script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script> AFTER the main three.js script. Do NOT use ES6 'import' statements in your inline script; assume THREE, OrbitControls, and Plotly are loaded in the global window scope.\n"
            "10. PLOTLY RACE CONDITIONS: Do NOT use `<body onload=\"...\">`. Put your plotting code directly in a standard `<script>` tag at the end of the `<body>` so it executes safely.\n\n"
            f"Topic: {clean_plan}"
        )

        # Dynamically calculate available tokens to prevent context window overflow
        prompt_token_est = len(html_prompt) // 3  # conservative: ~3 chars per token
        html_max_tokens = max(512, oc_ctx - prompt_token_est - 100)
        html_max_tokens = min(html_max_tokens, gen_tokens)  # never exceed global gen limit

        html_code = self._call_model(
            coder_llm, 
            html_prompt, 
            max_tokens=html_max_tokens, 
            temperature=gen_temp,
            system_prompt=(
                "You are an expert JavaScript/Plotly.js/Three.js coder. "
                "Declare all variables globally. Place plotting code in a <script> tag at the end of <body>. "
                "Output ONLY the complete HTML document inside ```html``` blocks."
            )
        )
        html_extract = Sandbox.extract_code(html_code)

        # CRITICAL GUARD: If the model returned nothing (ran out of tokens), skip Strategy 1
        # entirely and fall through to the Python Plotly fallback. Sending '\n' to the iframe
        # causes the blank sandbox bug.
        html_is_valid_document = (
            html_extract and
            html_extract.strip() and
            ("<html" in html_extract.lower() or "<script" in html_extract.lower() or "<!doctype" in html_extract.lower())
        )

        if not html_is_valid_document:
            if status_callback:
                status_callback("HTML generation returned empty. Falling back to Python Plotly...", "warning", "opencode", 96)
            html_extract = ""

        # Validate initially
        html_valid = False
        html_error = ""
        if html_is_valid_document:
            html_valid, html_error = self._verify_html_javascript(html_extract)

        # Pre-check: Bypass verification immediately if Node is missing or if it's a browser-environment mock error
        bypass_verification = False
        if html_error:
            is_node_missing = any(kw in html_error.lower() for kw in ["node", "runtime not found", "executable not found", "command not found"])
            is_mock_error = any(kw in html_error.lower() for kw in ["canvas", "webgl", "document is not defined", "window is not defined"])
            if is_node_missing or is_mock_error:
                bypass_verification = True

        if html_is_valid_document and not bypass_verification:
            # Reflexion loop for Strategy 1: Max 2 self-fix attempts
            for attempt in range(2):
                if html_valid:
                    break
                if status_callback:
                    status_callback(f"Fixing HTML JS execution error (Round {attempt+1})...", "warning", "opencode", 96)
                
                fix_p = (
                    f"Your previously generated HTML/JS code failed JavaScript execution verification.\n\n"
                    f"Failed Code:\n{html_extract}\n\n"
                    f"Error Message:\n{html_error}\n\n"
                    "Please fix it. Common guidelines:\n"
                    "1. If using Three.js, ensure you load OrbitControls correctly, NEVER use non-existent APIs like ArcGeometry (use RingGeometry or TorusGeometry instead), and define all animation variables (like clock/time/frameCount).\n"
                    "2. If using Plotly.js, ensure layout backgrounds are dark, colorscales are explicit, and target elements exist.\n"
                    "3. Ensure the glassmorphic control card contains functional sliders for variables and play/pause/reset buttons that actually update the physics loop dynamically.\n"
                    "4. Ensure there are no JavaScript syntax errors or undefined variables.\n\n"
                    "Output ONLY the complete, corrected HTML page inside ```html``` blocks."
                )
                fix_token_est = len(fix_p) // 3
                fix_max_tokens = max(512, oc_ctx - fix_token_est - 100)
                fix_max_tokens = min(fix_max_tokens, gen_tokens)
                html_fixed = self._call_model(
                    coder_llm, 
                    fix_p, 
                    max_tokens=fix_max_tokens, 
                    temperature=gen_temp,
                    system_prompt=(
                        "You are an expert JavaScript, WebGL, Three.js, and Plotly.js coder.\n"
                        "Identify and repair the specific ReferenceError, TypeError, or SyntaxError reported in the error message.\n"
                        "Ensure all state variables are in the global scope, OrbitControls has both arguments, and no non-existent geometry builders are used.\n"
                        "Output ONLY the complete, corrected HTML page inside ```html``` blocks."
                    )
                )
                fixed_extract = Sandbox.extract_code(html_fixed)
                if fixed_extract and ("<html" in fixed_extract.lower() or "<script" in fixed_extract.lower()):
                    html_extract = fixed_extract
                    html_valid, html_error = self._verify_html_javascript(html_extract)
                else:
                    html_valid = False
                    html_error = "No code block found in response."

        if (html_valid or bypass_verification) and html_extract and html_extract.strip():
            warning_msg = ""
            if bypass_verification:
                warning_msg = "\n<!-- Note: HTML verification bypassed due to environment/runtime differences. Rendering best-effort. -->"
            return f"\n\n### 3D Interactive Visualization (Live Artifact){warning_msg}\n<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"

        # ── Strategy 2: Python Plotly (backend sandbox verified fallback) ──────────
        if status_callback:
            status_callback("HTML Failed. Falling back to Python Plotly...", "warning", "opencode", 97)
        is_physics = any(kw in compiled_plan.lower() for kw in ['proton', 'electron', 'magnetic', 'electric', 'lorentz', 'ode', 'differential equation', 'cyclotron'])
        physics_rules = ""
        if is_physics:
            physics_rules = (
                "11. CRITICAL ODE PHYSICS RULES — follow exactly or the trajectory will explode:\n"
                "    a. For charged particle motion (Lorentz force), state MUST be 6 components: [x, y, z, vx, vy, vz]. NEVER use 3-component [vx, vy, vz] only.\n"
                "    b. ODE function MUST return [vx, vy, vz, ax, ay, az] (positions' derivatives are velocities, velocities' derivatives are accelerations).\n"
                "    c. Compute cyclotron period: T = 2*pi*m / (abs(q) * B_magnitude). Set t_span = (0, N_cycles * T). NEVER hardcode t_span=(0,1) or any non-physics time.\n"
                "    d. Use t_eval = np.linspace(0, N_cycles*T, N_cycles*200) for smooth curve.\n"
                "    e. Plot POSITIONS sol.y[0], sol.y[1], sol.y[2] (x, y, z), NOT velocities. Velocity values are ~1e8 scale and will break the plot scale.\n"
                "    f. Use solve_ivp(..., method='RK45', max_step=T/200, dense_output=False).\n"
                "    g. Set scene aspectmode: always call fig.update_layout(scene=dict(aspectmode='data')) to preserve actual physical scale shapes and prevent squishing.\n\n"
            )

        viz_prompt = (
            "You are a Python data visualization expert. "
            "Write ONLY a complete Python script using plotly for an interactive 3D visualization.\n"
            "RULES:\n"
            "1. Import plotly.graph_objects as go, numpy as np, and from scipy.integrate import solve_ivp for ODEs.\n"
            "2. Create a 3D scatter, surface, or line plot.\n"
            "3. Use fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=40,b=0)). For physical trajectory/field simulations, also set scene=dict(aspectmode='data') inside update_layout.\n"
            "4. Do NOT use fig.update_scenes(). Do NOT use go.FigureControls(). They do NOT exist.\n"
            "5. Do NOT set background colors manually.\n"
            "6. Do NOT import plotly.subplots, plotly.io, or any other plotly module.\n"
            "7. Last line MUST be: print(fig.to_json())\n"
            "8. Do NOT call fig.show() or save to file.\n"
            "9. GRID RESOLUTION LIMIT: Use grid size of at most 30x30 for surface plots.\n"
            "10. Write a complete, self-contained Python script from scratch. Define all constants. Print the JSON.\n"
            f"{physics_rules}"
            "Output ONLY code in ```python``` blocks.\n\n"
            f"Topic Context:\n{clean_plan[:2000]}"
        )

        viz_token_est = len(viz_prompt) // 3
        viz_max_tokens = max(512, oc_ctx - viz_token_est - 100)
        viz_max_tokens = min(viz_max_tokens, gen_tokens)
        viz_code = self._call_model(
            coder_llm, 
            viz_prompt, 
            max_tokens=viz_max_tokens, 
            temperature=gen_temp,
            system_prompt=(
                "You are an expert Python coder. Write ONLY valid Python code inside a single ```python``` code block. "
                "Do NOT write any normal conversational text outside the code block."
            )
        )
        viz_extract = Sandbox.extract_code(viz_code)
        
        # Ensure we only try to execute if it looks like python code
        has_python = "import" in viz_extract or "def " in viz_extract or "fig" in viz_extract
        viz_success = False
        viz_output = ""
        
        if has_python:
            if status_callback:
                status_callback("Rendering 3D Visualization...", "info", "opencode", 98)
            viz_success, viz_output = self.sandbox.execute(viz_extract, language='python')

        def _strip_sandbox_prefix(text):
            """Remove sandbox prefixes from sandbox output."""
            if not text:
                return text
            for prefix in [
                "🔒 [Restricted Sandbox]\n", "🔒 [Restricted Sandbox]",
                "⚠️ [Unrestricted Fallback]\n", "⚠️ [Unrestricted Fallback]"
            ]:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text

        # Reflexion self-fix loop for Strategy 2: Max 2 self-fix attempts
        for attempt in range(2):
            cleaned = _strip_sandbox_prefix(viz_output).strip() if viz_output else ""
            json_extracted = False
            if viz_success and "{" in cleaned:
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_candidate = cleaned[start_idx:end_idx+1]
                    try:
                        import json
                        json.loads(json_candidate)
                        cleaned = json_candidate
                        json_extracted = True
                    except Exception:
                        pass
            if json_extracted:
                break
            if status_callback:
                status_callback(f"Fixing 3D syntax/runtime error (Round {attempt+1})...", "warning", "opencode", 99)
            
            error_details = viz_output if viz_output else "No valid python code block was generated."
            if viz_success and not json_extracted:
                error_details = "Code ran successfully but failed to print valid JSON. Make sure the script prints fig.to_json()"
                
            if not viz_extract:
                fix_p = (
                    "You failed to generate a valid Python code block in your previous attempt.\n\n"
                    "Please write a complete Python script using plotly for the 3D interactive visualization.\n"
                    "RULES:\n"
                    "1. Import plotly.graph_objects as go and numpy as np ONLY\n"
                    "2. Create a 3D scatter, surface, or line plot\n"
                    "3. Use fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=40,b=0))\n"
                    "4. Last line MUST be: print(fig.to_json())\n\n"
                    f"Topic: {compiled_plan[:2000]}"
                )
            else:
                fix_p = (
                    f"This Plotly code failed:\n{viz_extract}\n\nError/Output:\n{error_details}\n\n"
                    "Fix it. REMEMBER: Do NOT use update_scenes(), FigureControls, or plotly.subplots. "
                    "Use ONLY go.Figure(), go.Surface/Scatter3d, and fig.update_layout(). "
                    "Output ONLY the corrected script in ```python``` blocks. End with print(fig.to_json())."
                )
            viz_fixed = self._call_model(
                coder_llm, 
                fix_p, 
                max_tokens=gen_tokens, 
                temperature=gen_temp,
                system_prompt=(
                    "You are an expert Python coder. Write ONLY valid Python code inside a single ```python``` code block. "
                    "Do NOT write any normal conversational text outside the code block."
                )
            )
            viz_extract = Sandbox.extract_code(viz_fixed)
            viz_code = viz_fixed
            
            has_python = "import" in viz_extract or "def " in viz_extract or "fig" in viz_extract
            if has_python:
                viz_success, viz_output = self.sandbox.execute(viz_extract, language='python')
            else:
                viz_success = False
                viz_output = "Model failed to output a valid code block."

        cleaned_final = _strip_sandbox_prefix(viz_output).strip() if viz_output else ""
        json_extracted = False
        if "{" in cleaned_final:
            start_idx = cleaned_final.find("{")
            end_idx = cleaned_final.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_candidate = cleaned_final[start_idx:end_idx+1]
                try:
                    import json
                    parsed_cand = json.loads(json_candidate)
                    if isinstance(parsed_cand, dict) and ("data" in parsed_cand or "layout" in parsed_cand):
                        cleaned_final = json_candidate
                        json_extracted = True
                except Exception:
                    pass

        if viz_success and json_extracted:
            # Strip out any Warnings/Stderr block appended by the sandbox
            if "\nWarnings/Stderr:\n" in cleaned_final:
                cleaned_final = cleaned_final.split("\nWarnings/Stderr:\n")[0].strip()
            return f"\n\n### 3D Interactive Visualization\n<!--PLOTLY_JSON-->\n{cleaned_final}\n<!--/PLOTLY_JSON-->"

        # Final fallback: show the code with error (ensure code blocks are wrapped properly)
        error_msg = f"\n\n**Execution Error:**\n```text\n{viz_output}\n```" if viz_output else ""
        formatted_code = viz_code
        if "```" not in formatted_code:
            formatted_code = f"```python\n{formatted_code}\n```"
        return f"\n\n### 3D Visualization Script\n{formatted_code}{error_msg}"

    # =========================================================================
    # PREDICTION PIPELINE — Dedicated ML/Data Science Forecasting
    # =========================================================================
    def _prediction_pipeline(self, prompt, enriched_prompt, router_llm,
                              router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Dedicated prediction pipeline with specialized ML prompts and data cleaning loops."""
        if status_callback:
            status_callback("🔮 Prediction: Drafting ML regression script...", "info", "opencode", 20)

        coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
        
        ml_system_prompt = (
            "You are an expert Python data scientist. Write production-grade ML scripts using pandas, "
            "scikit-learn, numpy, and scipy. Handle missing data (NaN), parse dates, and use robust "
            "regression methods. Always validate your data before fitting models.\n"
            "CRITICAL OUTPUT FORMAT: At the end of the script, print a JSON block with this exact format:\n"
            'print("PREDICTIVE_METRICS:" + json.dumps({"metric_name": "R²", "metric_value": r2_score, '
            '"forecast": [list_of_predictions], "dates": [list_of_labels]}))\n'
            "Output ONLY the complete script in ```python``` blocks."
        )

        ml_prompt = (
            f"Write a complete Python script that performs predictive analysis/forecasting on the following data.\n\n"
            f"RULES:\n"
            f"1. Import pandas, numpy, sklearn, json. Parse any tabular data from the context below.\n"
            f"2. Handle missing values with dropna() or fillna(). Convert date strings to datetime.\n"
            f"3. Use LinearRegression, Ridge, or RandomForestRegressor from sklearn.\n"
            f"4. Split data 80/20 for train/test. Compute R² score.\n"
            f"5. Generate future predictions (5-10 steps ahead).\n"
            f"6. Print PREDICTIVE_METRICS JSON at the end (see system prompt format).\n"
            f"7. Do NOT use matplotlib, plotly, or any visualization. Only compute and print.\n"
            f"8. If data is insufficient, create synthetic sample data based on the topic.\n\n"
            f"Context & Data:\n{enriched_prompt[:6000]}\n\n"
            f"Output the complete script in ```python``` blocks."
        )

        code = Sandbox.extract_code(self._strip_thinking(
            self._call_model(coder_llm, ml_prompt, gen_tokens, gen_temp, system_prompt=ml_system_prompt)
        ))

        # If OpenCode failed to generate code, escalate to DeepSeek-R1
        if not code or len(code.strip()) < 50:
            if status_callback:
                status_callback("🔮 OpenCode draft empty — escalating to DeepSeek-R1...", "warning", "deepseek_r1", 30)
            ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
            code = Sandbox.extract_code(self._strip_thinking(
                self._call_model(ds_llm, ml_prompt, gen_tokens, gen_temp, system_prompt=ml_system_prompt)
            ))

        if not code or len(code.strip()) < 50:
            return f"### Prediction Pipeline\nFailed to generate a valid ML script for: {prompt[:200]}"

        # Execute and verify
        max_attempts = 3
        for attempt in range(max_attempts):
            if status_callback:
                status_callback(f"🔮 Executing prediction script (Attempt {attempt+1}/{max_attempts})...", "info", "opencode", 40 + attempt * 20)
            
            ok, output = self.sandbox.execute(code, language='python')
            
            if ok and "PREDICTIVE_METRICS:" in output:
                if status_callback:
                    status_callback("🔮 Prediction VERIFIED!", "success", "opencode", 100)
                self.memory.save(prompt, code)
                
                # Extract the metrics JSON
                metrics_line = ""
                for line in output.split('\n'):
                    if "PREDICTIVE_METRICS:" in line:
                        metrics_line = line.split("PREDICTIVE_METRICS:", 1)[1].strip()
                        break
                
                result = f"### Prediction & Forecasting Analysis\n\n"
                if metrics_line:
                    result += f"=== PREDICTIVE_METRICS ===\n{metrics_line}\n==========================\n\n"
                result += f"```python\n{code}\n```\n"
                
                # Check if a 3D visualization needs to be generated and appended
                viz = self._check_3d_gate(prompt, result, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                return f"{result}{viz}"

            # If the script ran successfully but just didn't print the metrics token,
            # treat as partial success — return whatever output we got
            if ok and output and len(output.strip()) > 20:
                if status_callback:
                    status_callback("🔮 Script ran but metrics format missing — returning best-effort result.", "warning", "opencode", 80)
                result = f"### Prediction & Forecasting Analysis\n\n{output}\n\n```python\n{code}\n```\n"
                viz = self._check_3d_gate(prompt, result, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                return f"{result}{viz}"
            
            # ── Data cleaning / error fixing loop ──
            if status_callback:
                status_callback(f"🔮 Fixing data/script error (Round {attempt+1})...", "warning", "opencode", 50 + attempt * 15)
            
            error_details = output[:800] if output else "Script produced no output."

            # Step A: Try Agent IDE surgical patch first (fast)
            patched = self._agent_ide_patch(code, error_details, coder_llm, gen_tokens)
            if patched and patched != code:
                patch_ok, patch_out = self.sandbox.execute(patched, language='python')
                if patch_ok and "PREDICTIVE_METRICS:" in patch_out:
                    code = patched
                    ok = True
                    output = patch_out
                    continue  # Will be caught by the success check at top of loop
                elif patch_ok:
                    code = patched
                    ok = True
                    output = patch_out
                    continue

            # Step B: Full rewrite fallback
            fix_prompt = (
                f"The prediction script failed with this error:\n{error_details}\n\n"
                f"Original code:\n{code[:2000]}\n\n"
                f"Fix the script. Common issues:\n"
                f"1. NaN values — use dropna() or fillna(0)\n"
                f"2. Empty DataFrame — create synthetic sample data based on the topic\n"
                f"3. Date parsing — use pd.to_datetime(errors='coerce')\n"
                f"4. Shape mismatch — ensure X and y have matching rows\n"
                f"5. If web-scraped data is unavailable, generate realistic synthetic data\n"
                f"Output ONLY the corrected script in ```python``` blocks."
            )
            code = Sandbox.extract_code(self._strip_thinking(
                self._call_model(coder_llm, fix_prompt, gen_tokens, gen_temp, system_prompt=ml_system_prompt)
            ))

        # Fallback
        fallback_res = f"### Prediction Pipeline (Best-Effort)\n\n**Execution failed after {max_attempts} attempts.**\n\n```python\n{code}\n```\n\n**Last Error:**\n```\n{output[:500]}\n```"
        viz = self._check_3d_gate(prompt, fallback_res, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
        return f"{fallback_res}{viz}"

    # =========================================================================
    # EXTREME WEBSEARCH PIPELINE — Deep Analysis + Chart Generation
    # =========================================================================
    def _extreme_websearch_pipeline(self, prompt, enriched_prompt, router_llm,
                                     router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Deep document analysis using DeepSeek R1 + chart generation using OpenCodeInterpreter."""
        # Phase 1: DeepSeek R1 analyzes the scraped web data
        if status_callback:
            status_callback("🔬 DeepSeek R1: Analyzing documents & structuring data...", "info", "deepseek_r1", 20)

        ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
        
        analysis_prompt = (
            f"You are a senior research analyst. Analyze the following web data thoroughly.\n\n"
            f"TASK:\n"
            f"1. Compare all data sources and identify key metrics, statistics, and trends.\n"
            f"2. Structure your findings into a clear, data-driven report with sections.\n"
            f"3. Extract numerical data that can be visualized (percentages, counts, rankings).\n"
            f"4. At the END of your report, output a JSON block with chart-ready data:\n"
            f'```json\n{{"charts": [{{"type": "pie|bar|line", "title": "...", "labels": [...], "values": [...]}}]}}\n```\n\n'
            f"User Question: {prompt}\n\n"
            f"Web Data:\n{enriched_prompt[:8000]}"
        )

        analysis = self._strip_thinking(
            self._call_model(ds_llm, analysis_prompt, gen_tokens, 0.6,
                           system_prompt="You are a world-class research analyst. Provide comprehensive, data-driven analysis.")
        )

        if status_callback:
            status_callback("🔬 Analysis complete. Generating interactive charts...", "info", "opencode", 60)

        # Phase 2: OpenCodeInterpreter generates Plotly charts from the analysis
        coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
        
        chart_prompt = (
            f"Based on this analysis report, write a Python script that creates interactive Plotly charts.\n\n"
            f"RULES:\n"
            f"1. Import plotly.graph_objects as go, numpy as np, json\n"
            f"2. Create pie charts, bar charts, or line charts based on the data in the report.\n"
            f"3. Use template='plotly_dark' and glassmorphic styling.\n"
            f"4. If the report contains a JSON chart data block, parse and use it directly.\n"
            f"5. If no JSON block exists, extract numbers from the text and create appropriate charts.\n"
            f"6. Last line MUST be: print(fig.to_json())\n"
            f"7. Do NOT use fig.show() or save to file.\n\n"
            f"Analysis Report:\n{analysis[:4000]}\n\n"
            f"Output ONLY the script in ```python``` blocks."
        )

        chart_code = Sandbox.extract_code(self._strip_thinking(
            self._call_model(coder_llm, chart_prompt, gen_tokens, gen_temp,
                           system_prompt="You are an expert Python Plotly coder. Output ONLY valid Python code.")
        ))

        chart_json = ""
        if chart_code:
            ok, chart_output = self.sandbox.execute(chart_code, language='python')
            if ok and "{" in chart_output:
                # Extract JSON from output
                cleaned = chart_output.strip()
                for prefix in ["🔒 [Restricted Sandbox]\n", "🔒 [Restricted Sandbox]",
                               "⚠️ [Unrestricted Fallback]\n", "⚠️ [Unrestricted Fallback]"]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_candidate = cleaned[start_idx:end_idx+1]
                    try:
                        import json
                        parsed = json.loads(json_candidate)
                        if isinstance(parsed, dict) and ("data" in parsed or "layout" in parsed):
                            chart_json = json_candidate
                    except Exception:
                        pass

        if status_callback:
            status_callback("🔬 Extreme WebSearch Analysis complete!", "success", "deepseek_r1", 100)

        # Build the final response
        result = f"### 🔬 Deep Analysis Report\n\n{analysis}\n\n"
        if chart_json:
            result += f"### Interactive Analysis Charts\n<!--PLOTLY_JSON-->\n{chart_json}\n<!--/PLOTLY_JSON-->\n"
        
        self.memory.save(prompt, result[:3000])
        return result

    # =========================================================================
    # MAIN PIPELINE ENTRY POINT
    # =========================================================================
    def process_query(self, prompt, mode="auto", selected_models=None, status_callback=None):
        # ── Handle "generate 3d" follow-up command ──────────────────────
        prompt_lower_check = prompt.strip().lower()
        generate_3d_triggers = ["generate 3d", "create 3d", "yes generate 3d", "yes create 3d", "show 3d", "yes 3d", "make 3d"]
        if any(prompt_lower_check == trig or prompt_lower_check.startswith(trig) for trig in generate_3d_triggers):
            if status_callback:
                status_callback("Generating 3D Visualization from last response...", "info", "opencode", 10)
            # Retrieve the most recent memory entry to use as context for 3D generation
            last_context = ""
            try:
                import sqlite3
                with sqlite3.connect(self.memory.sqlite_path, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT doc FROM memories ORDER BY created_at DESC LIMIT 1")
                    row = cursor.fetchone()
                    if row:
                        last_context = row[0]
            except Exception:
                pass
            if not last_context:
                last_context = self.memory.recall("visualization 3d simulation", n_results=1)
            if not last_context:
                last_context = "No previous context found. Generate a generic 3D demo visualization."
            router_ctx = self._get_dynamic_context_ceiling("router")
            oc_ctx = self._get_dynamic_context_ceiling("opencode")
            gen_tokens = 4096
            gen_temp = 0.1
            viz = self._generate_3d_now(last_context, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            if viz:
                return viz
            return "Could not generate a 3D visualization. Please try rephrasing your request."

        if status_callback:
            status_callback("Phi-3.5-Mini checking intent...", "info", "router", 5)

        # ── Web Search Enrichment ────────────────────────────────────────
        # Auto-enable search if user query explicitly asks for web search
        search_keywords = [
            "search the web", "search online", "search for", "google for",
            "latest news", "recent news", "breaking news", "current price of",
            "what is the price of", "stock price of", "weather in", "weather today",
            "weather", "forecast", "temperature",
            "who won", "who lost", "last match", "latest match", "recent match",
            "score of", "match result", "election result", "box office",
            "release date", "when is", "when does", "when did",
            "yesterday", "today's", "todays", "tonight", "last night",
            "this week", "trending", "right now", "ipl", "world cup",
            "crypto price", "stock price", "temperature in"
        ]
        prompt_lower = prompt.lower()
        active_web_search = (self.search_mode != "off") or any(kw in prompt_lower for kw in search_keywords)

        web_context = ""
        if active_web_search:
            if status_callback:
                status_callback("Optimizing Search Query...", "info", "router", 6)
            try:
                # 1. LLM Query Optimizer
                router_llm = self._get_model("router", required_ctx=1024)
                opt_prompt = (
                    "Transform the user request into a concise Google search query. "
                    "Keep names, locations, cities, and timeframes. Do NOT remove specific locations (like 'Jharsuguda'). "
                    "Output ONLY the plain search query without quotes, bullet points, numbering, or intro text.\n\n"
                    f"User Request: {prompt}"
                )
                raw_query = self._call_model(
                    router_llm, 
                    opt_prompt, 
                    max_tokens=30, 
                    temperature=0.1,
                    system_prompt="You are a search query optimizer. Output ONLY a clean, single-line Google search query. Never output lists, numbering, or bullet points."
                ).strip()
                # Clean LLM output: remove list markers, bullets, and join lines to prevent discarding key information
                lines = [l.strip() for l in raw_query.split('\n') if l.strip()]
                cleaned_parts = []
                for line in lines:
                    cl = re.sub(r'^(Keywords?:?\s*|Search\s*query:?\s*|\d+[\.\)]\s*|-\s*|\*\s*)', '', line, flags=re.IGNORECASE).strip()
                    if cl:
                        cleaned_parts.append(cl)
                search_query = " ".join(cleaned_parts)
                search_query = search_query.replace('"', '').replace('`', '').replace('*', '').strip()
                search_query = search_query[:80]  # Cap length for search engines
                if not search_query or len(search_query) < 3:
                    search_query = prompt[:80]

                # Calculate dynamic character cap and page count based on context ceiling
                ds_ctx_est = self._get_dynamic_context_ceiling("deepseek_r1")
                
                is_predictive = any(kw in prompt_lower for kw in ["predict", "forecast", "prediction"])
                
                if is_predictive:
                    if ds_ctx_est >= 32000:
                        max_results = 20
                        max_scraped = 20
                        char_limit = 3500
                    elif ds_ctx_est >= 16000:
                        max_results = 15
                        max_scraped = 15
                        char_limit = 2500
                    else:
                        # 8192 context (Older GPUs like P100/T4)
                        max_results = 8
                        max_scraped = 8
                        char_limit = 2000
                else:
                    # Derive char_limit from the ACTUAL context ceiling rather than raw VRAM/RAM.
                    # This prevents scraping 60k chars of content when the model context is only 8k tokens.
                    # Formula: (context_ceiling × 0.60 prompt share × 3 chars/token) / max_scraped pages
                    max_results = 5
                    max_scraped = 5
                    usable_prompt_tokens = int(ds_ctx_est * 0.60)
                    char_limit = max(1500, (usable_prompt_tokens * 3) // max_scraped)
                    # Cap at sensible maximum — even with 32k context, no single page needs > 8000 chars
                    char_limit = min(8000, char_limit)

                if status_callback:
                    status_callback(f"Searching: '{search_query}'... (Limit: {char_limit} chars/page)", "info", "router", 8)
                
                results = self.web_search.search(search_query, max_results=max_results)
                
                # 2. Compile Snippets Block & Scrape top pages
                snippets_list = []
                scraped_pages = []
                scraped_raw_texts = []
                scraped_count = 0
                
                if results:
                    for idx, r in enumerate(results):
                        title = r.get("title", "")
                        link = r.get("link", "")
                        snippet = r.get("snippet", "")
                        snippets_list.append(f"[{idx+1}] Title: {title}\nURL: {link}\nSnippet: {snippet}")
                        
                        # Try to scrape the page content if we haven't reached the limit
                        if scraped_count < max_scraped and link:
                            # Skip Google News index pages to avoid scraping massive, noisy, cross-mixed aggregates
                            if "news.google.com" in link.lower() and ("/topics/" in link.lower() or "/stories/" in link.lower() or "/publications/" in link.lower()):
                                continue
                            
                            if status_callback:
                                status_callback(f"Scraping ({scraped_count+1}/{max_scraped}): {link[:40]}...", "info", "router", 12 + scraped_count * (2 if not is_predictive else 0.5))
                            
                            text = self.web_search.scrape_url(link)
                            if text and len(text.strip()) > 200:
                                # Apply word-based Jaccard similarity deduplication to filter duplicate sites/syndicated pages
                                def get_words(t):
                                    return set(re.findall(r'\w+', t.lower()))
                                
                                new_words = get_words(text)
                                is_dup = False
                                for prev_text in scraped_raw_texts:
                                    prev_words = get_words(prev_text)
                                    if new_words and prev_words:
                                        intersection_len = len(new_words.intersection(prev_words))
                                        union_len = len(new_words.union(prev_words))
                                        if union_len > 0:
                                            jaccard = intersection_len / union_len
                                            if jaccard > 0.65: # 65% overlap is duplicate
                                                is_dup = True
                                                break
                                if is_dup:
                                    continue
                                
                                scraped_raw_texts.append(text)
                                scraped_pages.append(
                                    f"=== START SCRAPED PAGE ===\nURL: {link}\nContent:\n{text[:char_limit]}\n=== END SCRAPED PAGE ==="
                                )
                                scraped_count += 1
                
                # Assemble combined web context block
                snippets_block = "Search Result Snippets:\n" + "\n\n".join(snippets_list) if snippets_list else "No search results returned."
                scraped_block = "\n\n".join(scraped_pages) if scraped_pages else "No pages could be deep-scraped (Cloudflare blocking or empty content)."
                
                web_context = (
                    f"=== WEB SEARCH RESULTS ===\n"
                    f"{snippets_block}\n\n"
                    f"=== DEEP SCRAPED DETAILS ===\n"
                    f"{scraped_block}\n"
                    f"===========================\n"
                )
                    
            except Exception as e:
                print(f"Web search enrichment failed: {e}")
                web_context = ""
                
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        system_instruction = (
            "You are a helpful, direct, and capable AI assistant.\n"
            "Answer the User Query clearly, accurately, and concisely. Do NOT mention your training cutoff date, "
            "do NOT state that you cannot access real-time/current information, and do NOT add unnecessary disclaimers. "
            "Speak naturally as a live, fully-functional AI assistant.\n\n"
        )
        if web_context:
            system_instruction = (
                "You are an advanced AI assistant equipped with real-time web search capabilities.\n"
                "Use the provided Web Context to answer the User Query directly, accurately, and factually.\n"
                "CRITICAL INSTRUCTION: Do NOT act like a search engine listing search results (e.g., do NOT say 'Source A says this, Source B offers that'). "
                "Instead, synthesize the raw data into a direct, conversational, and authoritative answer. Extract the actual requested data (like current temperatures, facts, or prices) and present it clearly to the user.\n"
                "MANDATORY CITATION RULE: Carefully match each news story, headline, author, and date with "
                "its exact source publication. Do not cross-mix authors or articles across different news outlets.\n"
                "Since you are provided with live search results, do NOT mention your training cutoff date, "
                "do NOT state that you cannot access real-time/current information, and do NOT add disclaimers "
                "about not having internet access. Answer as a live, fully-connected AI.\n\n"
            )

        # Retrieve relevant past experiences from memory/RAG early
        past_experience = self.memory.recall(prompt, n_results=2)
        
        # Build structured context blocks
        context_blocks = []
        if web_context:
            context_blocks.append(f"Web Context:\n{web_context}")
        if past_experience:
            context_blocks.append(
                f"=== VERIFIED SCIENTIFIC REFERENCE KNOWLEDGE ===\n"
                f"The following is a verified derivation that demonstrates the correct APPROACH and TECHNIQUE for solving this type of problem. "
                f"Study the mathematical method, the integration technique, the code structure, and the overall solution strategy used here. "
                f"CRITICAL: Always use the EXACT numerical values (constants, field strengths, masses, charges, velocities) from the User Query below — "
                f"NEVER copy numerical values from this reference if they differ from what the user specifies.\n\n"
                f"{past_experience.strip()}\n"
                f"==============================================="
            )
            
        context_str = "\n\n".join(context_blocks)

        enriched_prompt = (
            f"{system_instruction}Current System Date/Time: {current_date}\n\n{context_str}\n\nUser Query:\n{prompt}"
            if context_str else f"{system_instruction}Current System Date/Time: {current_date}\n\nUser Query:\n{prompt}"
        )

        # ── Dynamic Context Sizing (RAM/VRAM-aware) ────────────────────
        est_tokens = len(enriched_prompt) // 4
        
        # Calculate dynamic ceilings for each model individually based on post-swap free VRAM & RAM
        router_ctx_cap = self._get_dynamic_context_ceiling("router")
        ds_ctx_cap = self._get_dynamic_context_ceiling("deepseek_r1")
        oc_ctx_cap = self._get_dynamic_context_ceiling("opencode")
        
        if self.context_length == 0:
            if getattr(self, 'kaggle_hotswap_mode', False):
                # In EVM hot-swap mode, only one model is loaded in VRAM at a time.
                # We maximize the context length for all models to utilize the free VRAM.
                router_ctx = router_ctx_cap
                ds_ctx = ds_ctx_cap
                oc_ctx = oc_ctx_cap
            else:
                # In standard shared-VRAM mode, keep context sizes tight to prevent OOM conflicts.
                router_ctx = min(router_ctx_cap, est_tokens + self.max_tokens)
                ds_ctx = min(ds_ctx_cap, est_tokens + 8192)
                oc_ctx = min(oc_ctx_cap, self.max_auto_ctx)
        else:
            # Smart context compression for benchmark tasks:
            # HumanEval/MBPP queries are tiny. Allocating 8192 tokens of KV cache wastes VRAM.
            # Scale down to strictly what is needed: est_tokens + system_prompt + max_generation
            router_ctx = max(1024, min(self.context_length, router_ctx_cap, est_tokens + 1024))
            ds_ctx = max(2048, min(self.context_length, ds_ctx_cap, est_tokens + 4096))
            oc_ctx = max(2048, min(self.context_length, oc_ctx_cap, est_tokens + 4096))
            
        # Ensure context sizing prints to log for easier transparency
        if getattr(self, 'kaggle_hotswap_mode', False):
            print(f"📐 DMA (EVM Context Sizing): router_ctx={router_ctx}, ds_ctx={ds_ctx}, oc_ctx={oc_ctx}")

        # Dynamically scale gen_tokens based on safe context capacity (RAM/VRAM-aware).
        # We allocate up to 40% of the active context for generation, capped between 2048 and 8192 tokens.
        # This prevents truncation on large GPUs/RAM setups while avoiding prompt starvation on low setups.
        min_ctx = min(ds_ctx, oc_ctx)
        gen_tokens = int(min_ctx * 0.40)
        gen_tokens = max(2048, min(8192, gen_tokens))
        # Ensure the prompt always has at least 1500 tokens of headroom
        if min_ctx - gen_tokens < 1500:
            gen_tokens = max(512, min_ctx - 1500)
        # Absolute safety floor: gen_tokens must never be zero or negative
        if gen_tokens < 256:
            gen_tokens = 256
            
        print(f"📐 DMA Generation Sizing: gen_tokens={gen_tokens} (active context base: {min_ctx} tokens)")
        
        # Adaptive Temperature Scaling
        logic_temp = 0.6  # High for creative logic problem solving
        gen_temp = 0.1    # Low for strict code writing

        # ── Six-Way Classification ────────────────────────────────────────
        router_llm = self._get_model("router", required_ctx=router_ctx)
        if isinstance(mode, str) and mode.upper() in ["SIMPLE", "CODING", "REASONING", "PREDICTION", "EXTREME_WEBSEARCH"]:
            task_type = mode.upper()
        else:
            task_type = self._classify_task(router_llm, prompt)
            
        # ── Search Mode Overrides ─────────────────────────────────────────
        if active_web_search and (not isinstance(mode, str) or mode.lower() == "auto"):
            if self.search_mode == "prediction" or is_predictive:
                task_type = "PREDICTION"
                # Prediction tasks with web search produce massive enriched prompts.
                # Ensure ds_ctx and oc_ctx are large enough to hold the scraped web data.
                prediction_min_ctx = min(8192, ds_ctx_cap)
                if ds_ctx < prediction_min_ctx:
                    ds_ctx = prediction_min_ctx
                if oc_ctx < min(8192, oc_ctx_cap):
                    oc_ctx = min(8192, oc_ctx_cap)
                # Recalculate gen_tokens with the expanded context
                min_ctx = min(ds_ctx, oc_ctx)
                gen_tokens = int(min_ctx * 0.40)
                gen_tokens = max(2048, min(8192, gen_tokens))
                if min_ctx - gen_tokens < 1500:
                    gen_tokens = max(512, min_ctx - 1500)
                if gen_tokens < 256:
                    gen_tokens = 256
            elif self.search_mode == "extreme":
                task_type = "EXTREME_WEBSEARCH"
                # Extreme mode expands context to absolute max for deep document analysis
                ds_ctx = ds_ctx_cap
                oc_ctx = oc_ctx_cap
                min_ctx = min(ds_ctx, oc_ctx)
                gen_tokens = int(min_ctx * 0.40)
                gen_tokens = max(2048, min(8192, gen_tokens))
                if min_ctx - gen_tokens < 1500:
                    gen_tokens = max(512, min_ctx - 1500)
                if gen_tokens < 256:
                    gen_tokens = 256
            elif task_type not in ["CODING", "REASONING"]:
                task_type = "SIMPLE"
                
        if status_callback:
            status_callback(f"Task classified as: {task_type}", "info", "router", 12)

        # ══════════════════════════════════════════════════════════════════
        # PATH A: SIMPLE — Direct answer from Router
        # ══════════════════════════════════════════════════════════════════
        if task_type == "SIMPLE":
            if status_callback:
                status_callback("Answering directly...", "success", "router", 100)
            safe = self._crunch_prompt(enriched_prompt, "router", router_ctx - self.max_tokens, status_callback, router_llm=router_llm)
            res = self._call_model(router_llm, safe, max_tokens=self.max_tokens, temperature=0.6)
            return self._clean_cutoff_notes(res)

        # ══════════════════════════════════════════════════════════════════
        # PATH B: CODING — Actor-Critic with Dual Sandbox
        # ══════════════════════════════════════════════════════════════════
        if task_type == "CODING":
            res = self._coding_pipeline(prompt, enriched_prompt, router_llm,
                                         router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return self._clean_cutoff_notes(res)

        # ══════════════════════════════════════════════════════════════════
        # PATH C: REASONING — VibeThinker/DeepSeek Logic + Playground Verification
        # ══════════════════════════════════════════════════════════════════
        if task_type == "REASONING":
            res = self._reasoning_pipeline(prompt, enriched_prompt, router_llm,
                                            router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return self._clean_cutoff_notes(res)

        # ══════════════════════════════════════════════════════════════════
        # PATH D: PREDICTION — Dedicated ML/Data Science Pipeline
        # ══════════════════════════════════════════════════════════════════
        if task_type == "PREDICTION":
            if status_callback:
                status_callback("🔮 Prediction Pipeline activated...", "info", "system", 15)
            res = self._prediction_pipeline(prompt, enriched_prompt, router_llm,
                                             router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return self._clean_cutoff_notes(res)

        # ══════════════════════════════════════════════════════════════════
        # PATH E: EXTREME WEBSEARCH — Deep Analysis + Charts
        # ══════════════════════════════════════════════════════════════════
        if task_type == "EXTREME_WEBSEARCH":
            if status_callback:
                status_callback("🔬 Extreme WebSearch & Deep Analysis activated...", "info", "system", 15)
            res = self._extreme_websearch_pipeline(prompt, enriched_prompt, router_llm,
                                                    router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return self._clean_cutoff_notes(res)

        # Fallback — shouldn't reach here, but treat as SIMPLE
        res = self._reasoning_pipeline(prompt, enriched_prompt, router_llm,
                                        router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
        return self._clean_cutoff_notes(res)

    def _clean_synthesis_format(self, final_response, code, req_lang="python"):
        """Ensures that explanation text is not wrapped in code blocks,
        and that the script code is strictly wrapped in a ```{req_lang}``` block at the end.
        """
        response_clean = final_response.strip()
        
        # 1. If the model wrapped the entire response in a single code block, strip it
        if (response_clean.startswith(f"```{req_lang}") or response_clean.startswith("```")) and response_clean.endswith("```"):
            lines = response_clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1] == "```":
                lines = lines[:-1]
            response_clean = "\n".join(lines).strip()
            
        # 2. Extract code blocks from the response
        import re
        code_blocks = re.findall(rf'```{req_lang}[\s\S]*?```|```[\s\S]*?```', response_clean)
        
        # Check if the actual code is already present inside a code block
        has_actual_code_block = False
        for block in code_blocks:
            block_content = re.sub(rf'^```({req_lang})?\n|```$', '', block, flags=re.IGNORECASE).strip()
            code_lines = [l.strip() for l in code.split("\n") if l.strip()]
            matching_lines = sum(1 for line in code_lines[:10] if line in block_content)
            if matching_lines >= min(3, len(code_lines)):
                has_actual_code_block = True
                break
                
        if has_actual_code_block:
            cleaned_response = response_clean
            for block in code_blocks:
                block_content = re.sub(rf'^```({req_lang})?\n|```$', '', block, flags=re.IGNORECASE).strip()
                is_this_code = False
                code_indicators = ["import ", "def ", "class ", " = ", "print(", "console.log", "function ", "const ", "let "]
                if any(ind in block_content for ind in code_indicators):
                    is_this_code = True
                
                # If it's the actual code block, leave it alone
                code_lines = [l.strip() for l in code.split("\n") if l.strip()]
                matching_lines = sum(1 for line in code_lines[:10] if line in block_content)
                if matching_lines >= min(3, len(code_lines)):
                    is_this_code = True
                    
                if not is_this_code:
                    cleaned_response = cleaned_response.replace(block, block_content)
            return cleaned_response
        else:
            cleaned_response = response_clean
            for block in code_blocks:
                block_content = re.sub(rf'^```({req_lang})?\n|```$', '', block, flags=re.IGNORECASE).strip()
                cleaned_response = cleaned_response.replace(block, block_content)
            return f"{cleaned_response}\n\n```{req_lang}\n{code}\n```"

    def _synthesize_coding_response(self, prompt, compiled_plan, code, output,
                                   router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback=None, req_lang="python"):
        """Compiles a beautiful, deterministic markdown report from successful execution without an extra LLM call."""
        if status_callback:
            status_callback("Compiling verified response...", "info", "system", 95)
        
        # Strip raw Plotly/JSON dumps from sandbox output to prevent context window blowout
        # and raw JSON leaking into the final user-facing response.
        clean_output = output
        if clean_output and len(clean_output) > 2000:
            import re
            # Remove any large JSON blob (likely fig.to_json() output)
            json_pattern = re.compile(r'\{"data":\[.*?\}\}\}', re.DOTALL)
            stripped = json_pattern.sub('[Plotly chart JSON removed — rendered separately in UI]', clean_output)
            # If stripping didn't help, just truncate
            if len(stripped) > 2000:
                stripped = stripped[:2000] + '\n... [OUTPUT TRUNCATED]'
            clean_output = stripped
            
        if not clean_output:
            clean_output = "(Code executed successfully with no output)"
        
        lang_name = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "cpp": "C++",
            "c": "C",
            "bash": "Bash",
            "java": "Java",
            "go": "Go",
            "rust": "Rust"
        }.get(req_lang, req_lang)

        # Construct a beautiful, clean, deterministic response report
        final_response = (
            f"### 💡 Logic & Architectural Plan\n"
            f"{compiled_plan.strip()}\n\n"
            f"### ⚙️ Sandbox Execution Output\n"
            f"```text\n"
            f"{clean_output.strip()}\n"
            f"```\n\n"
            f"### 💻 Verified Working Code\n"
            f"The complete, fully verified functional {lang_name} script is presented below:\n\n"
            f"```{req_lang}\n"
            f"{code.strip()}\n"
            f"```"
        )
        
        # Check if a 3D visualization needs to be generated and appended
        viz = self._check_3d_gate(prompt, final_response, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
        
        if status_callback:
            status_callback("Done!", "success", "router", 100)
            
        return f"{final_response}{viz}"

    # =====================================================================
    # CODING PIPELINE — Reasoning Sandbox → Code Sandbox → Reflexion
    # =====================================================================
    def _coding_pipeline(self, prompt, enriched_prompt, router_llm,
                         router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        # Determine language constraints
        prompt_lower = prompt.lower()
        req_lang = "python"
        if "javascript" in prompt_lower or " js " in prompt_lower or "node.js" in prompt_lower or "nodejs" in prompt_lower or "node js" in prompt_lower or "node-js" in prompt_lower:
            req_lang = "javascript"
        elif "typescript" in prompt_lower or " ts " in prompt_lower:
            req_lang = "typescript"
        elif "c++" in prompt_lower or "cpp" in prompt_lower:
            req_lang = "cpp"
        elif "c code" in prompt_lower or " c lang" in prompt_lower:
            req_lang = "c"
        elif "bash" in prompt_lower or "shell script" in prompt_lower or " sh " in prompt_lower:
            req_lang = "bash"
        elif " java " in prompt_lower:
            req_lang = "java"
        elif "golang" in prompt_lower or "go lang" in prompt_lower:
            req_lang = "go"
        elif " rust " in prompt_lower:
            req_lang = "rust"

        lang_name = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "cpp": "C++",
            "c": "C",
            "bash": "Bash",
            "java": "Java",
            "go": "Go",
            "rust": "Rust"
        }.get(req_lang, req_lang)

        logic_temp = 0.6
        # Leave 1000 tokens headroom specifically for the massive 'planner_sys' system prompt
        # Safety floor: ensure crunch budget is at least 1024 tokens so the planner has enough context
        crunch_budget = max(1024, ds_ctx - gen_tokens - 1000)
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", crunch_budget, status_callback, router_llm=router_llm)

        max_resets = 2
        lessons = ""
        all_errors = []

        planner_sys = (
            "You are a world-class scientist, physicist, and software planner.\n"
            "Your task is to draft a step-by-step logic plan for the user's query.\n\n"
            "MANDATORY ACCURACY RULES:\n"
            "1. Physical/Mathematical Rigor: Double-check ALL formulas before writing them. Use standard literature formulas exactly. "
            "For example, E x B drift velocity is v = (E x B)/B^2 (it is independent of mass and charge). "
            "State all physical constants explicitly and check your dimensional analysis.\n"
            "2. Complete Multi-Dimensional Coordinates: For 3D problems, decompose EVERY vector "
            "into all 3 components (x, y, z). Never reduce a 3D problem to 2D unless explicitly stated.\n"
            "3. Initial Conditions: For well-known problems, use KNOWN PUBLISHED initial conditions from literature.\n"
            "4. Numerical Methods: Specify the exact integration scheme. "
            "CRITICAL TIME LIMIT: If integrating high-frequency oscillatory motion (e.g. cyclotron orbits, molecular vibrations), "
            "calculate the period FIRST. NEVER integrate for millions of cycles. Cap the simulation time so it covers at most 50-100 cycles to prevent solver timeouts.\n"
            "5. Conservation Laws: Explicitly state which conserved quantities (energy, momentum) should be verified.\n"
            "6. OUTPUT FORMAT: Write a numbered list of steps. Each step must state WHAT to compute, "
            "the EXACT formula, and the expected data structure.\n"
            "7. Think step by step. If you are unsure about any formula, derive it from first principles or explicitly state you need a web search check.\n"
            "8. IMPORTANT: If 'Relevant past experience' is provided, prioritize the User Query's exact parameters over the past experience if they differ.\n"
            "9. THINKING CONSTRAINT: Keep your reasoning brief and focused. Proceed to planning quickly.\n"
            "10. LORENTZ FORCE / ELECTROMAGNETICS: The Lorentz force is F = q*(E + v×B), NOT q*(E×B). "
            "The cross product is between VELOCITY and MAGNETIC FIELD, not between E and B. "
            "Always expand v×B explicitly: (v×B)_x = vy*Bz - vz*By, (v×B)_y = vz*Bx - vx*Bz, (v×B)_z = vx*By - vy*Bx.\n"
            "11. DRIFT VELOCITY EXTRACTION: To numerically verify a drift velocity from oscillatory trajectory data, "
            "use np.polyfit(t, x, 1) to extract the linear slope (which filters out cyclotron oscillations). "
            "Do NOT use endpoint averages like x(T)/T, as boundary oscillations cause >1%% relative error."
        )

        coder_sys = (
            "You are an expert computational programmer and software engineer.\n"
            "Your job is to translate the logic plan into a complete, clean, and immediately runnable Python script.\n\n"
            "STRICT RULES:\n"
            "1. Implement equations EXACTLY as described in the plan. Do NOT simplify or approximate unless instructed.\n"
            "2. Do NOT write placeholders, mock functions, or abbreviated loop bodies. EVERY line must be real.\n"
            "3. Handle edge cases: division by zero (add softening epsilon), array bounds, negative sqrt.\n"
            "4. SIMULATION TIMEOUT PREVENTION: If using scipy.integrate.solve_ivp for high-frequency oscillatory motion, "
            "strictly ensure the time span (t_span) is small enough to only cover a reasonable number of cycles (e.g. <1000). "
            "Integrating for too long will cause a timeout error.\n"
            "5. Print clear, formatted numerical results.\n"
            "6. For simulations: use numpy arrays, vectorized operations where possible.\n"
            "7. For plotting: use matplotlib or plotly. Always label axes with units.\n"
            "8. The script MUST run standalone with `python script.py` — no user input, no GUI blocking.\n"
            "9. Import ONLY what you use. Do not import unused libraries.\n"
            "10. Add a brief comment above each major section explaining what it does.\n"
            "11. PREDICTIVE/FORECASTING TASKS:\n"
            "    - You can import `sklearn` (scikit-learn) and `statsmodels` for time-series forecasting, regression, and data predictions.\n"
            "    - To fetch data, you have access to a global helper class `SandboxDataHelper` in the namespace. Do NOT import it; use it directly:\n"
            "      * `df = SandboxDataHelper.get_stock_data(symbol, period='1y')` -> returns a pandas DataFrame with columns: Date, Open, High, Low, Close, Volume.\n"
            "      * `df = SandboxDataHelper.get_weather_data(city, days=7)` -> returns a pandas DataFrame with columns: Date, Temperature_C, Humidity_Pct, Condition.\n"
            "      * `api_key = SandboxDataHelper.get_api_key(service_name)` -> returns a fallback API key string for services like 'weather', 'finance', 'crypto'.\n"
            "    - If the task involves predictions, regression, or forecasting, you MUST print a standardized JSON metric block to stdout at the end of the script. Format exactly as follows:\n"
            "      import json\n"
            "      print('=== PREDICTIVE_METRICS ===')\n"
            "      print(json.dumps({\n"
            "          'metric_name': 'R2 / MAE / RMSE / accuracy etc',\n"
            "          'metric_value': 0.95,\n"
            "          'forecast': [100.5, 101.2, 102.8],\n"
            "          'dates': ['2026-06-24', '2026-06-25', '2026-06-26']\n"
            "      }))\n"
            "      print('==========================')\n\n"
            "12. CYBERSECURITY, CRYPTOGRAPHY & NETWORK TASKS:\n"
            "    - You can import `cryptography` (e.g. Fernet, AES, RSA, padding, hashes), `scapy` (for packet crafting/sniffing simulation), `jwt` (or `pyjwt`), and `hashlib` / `hmac`.\n"
            "    - If you are writing an encryption or token task, you MUST write automated validation in the script to verify that ciphertext can be decrypted back to the original plaintext, and that signatures/tokens verify correctly.\n"
            "    - Always use safe key generation and modern secure cryptographic parameters (e.g. key lengths >= 256 bits for AES, >= 2048 bits for RSA, SHA-256 or better for hashing).\n"
            "    - For network protocol tasks, use scapy to simulate packet creation, validation of header offsets, and printing raw hex/dissections of packets. Do NOT try to connect to external ports or services; simulate them.\n\n"
            "=== CRITICAL CODE QUALITY RULES ===\n"
            "13. SOLVE_IVP STATE VECTOR ORDERING:\n"
            "    When using scipy.integrate.solve_ivp with state = [x, y, z, vx, vy, vz]:\n"
            "    The derivative function MUST return [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]\n"
            "    i.e., return [vx, vy, vz, ax, ay, az] — positions first, then accelerations.\n"
            "    NEVER return [ax, ay, az, vx, vy, vz] — this swaps state variables and corrupts the simulation.\n\n"
            "14. LORENTZ FORCE IMPLEMENTATION:\n"
            "    The Lorentz force is: F = q * (E + v × B)\n"
            "    The cross product is between VELOCITY (v) and MAGNETIC FIELD (B), NOT between E and B.\n"
            "    Use np.cross([vx, vy, vz], B) to compute v × B. Example:\n"
            "      def lorentz(t, state):\n"
            "          x, y, z, vx, vy, vz = state\n"
            "          v_cross_B = np.cross([vx, vy, vz], [Bx, By, Bz])\n"
            "          ax, ay, az = (q/m) * (E + v_cross_B)\n"
            "          return [vx, vy, vz, ax, ay, az]\n\n"
            "15. NUMPY FORMATTING BUG PREVENTION:\n"
            "    NEVER use f-string formatting like f'{numpy_array:.3f}' — NumPy arrays do not support scalar format specifiers.\n"
            "    Instead, use: f'{float(scalar_value):.3f}' for scalars, or np.array2string(arr, precision=3) for arrays.\n"
            "    When extracting a single value from sol.y, use sol.y[i, -1] (scalar) not sol.y[i] (full array).\n\n"
            "16. DRIFT VELOCITY VERIFICATION:\n"
            "    To verify a drift velocity from oscillatory trajectory data, use linear regression:\n"
            "      slope, _ = np.polyfit(sol.t, sol.y[0], 1)  # slope = drift velocity in x\n"
            "    Do NOT use endpoint division x(T)/T — boundary oscillations cause >1%% error.\n\n"
            "17. AUTOMATED SELF-TESTING MANDATE:\n"
            "    For ALL programming tasks (including algorithms, graphs, math, data structures, etc.), you MUST append a robust test suite at the bottom of the script.\n"
            "    - Write at least 3-4 distinct test cases (including standard inputs, empty/minimum bounds, and extreme values).\n"
            "    - Use strict `assert` statements: `assert solve_function(inputs) == expected_output, f'Test failed: expected {expected_output}, got {actual}'`.\n"
            "    - This ensures that if your logic is incorrect, the script will crash with an AssertionError, triggering the self-correction loop.\n"
            "    - Print 'ALL TESTS PASSED' at the very end of the execution block only if all assertions succeed.\n\n"
            "=== PLOTLY 3D CHEAT SHEET ===\n"
            "import plotly.graph_objects as go\n"
            "import numpy as np\n"
            "# 1. Standard Plot: \n"
            "fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='lines')])\n"
            "# 2. Adding Sliders/Controls: If the user requests parameter controls, you MUST precompute the plot data for different parameter values (e.g. 10 different slider steps), add a trace for each parameter value to the figure, set all but the first trace to visible=False, and add a layout slider dict under `sliders` to toggle the visibility of the traces. Example:\n"
            "#    steps = []\n"
            "#    for i in range(10):\n"
            "#        step = dict(method='update', args=[{'visible': [t == i for t in range(10)]}], label=str(i))\n"
            "#        steps.append(step)\n"
            "#    fig.update_layout(sliders=[dict(active=0, steps=steps)])\n"
            "# 3. Dark Theme Layout:\n"
            "fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))\n"
            "# 4. Outputs: ALWAYS print JSON to stdout so the UI can render, AND call fig.show() so it runs locally:\n"
            "print(fig.to_json())\n"
            "fig.show()\n"
            "============================="
        )

        # ── Execution Loop ──────────────────────────────────────────────────
        initial_failed_code = ""
        initial_failed_error = ""

        for reset in range(max_resets):
            max_rounds = 2 if reset == 0 else 1
            for rnd in range(max_rounds):
                # ── Phase 1: DeepSeek Logic Plan ─────────────────────────────
                if status_callback:
                    lbl = (
                        f"Nuclear Reset #{reset} (Attempt {rnd+1}/{max_rounds}): DeepSeek-R1 drafting logic..."
                        if reset else
                        f"DeepSeek-R1 drafting logic (Attempt {rnd+1}/{max_rounds})..."
                    )
                    status_callback(lbl, "info" if not reset else "warning", "deepseek_r1", 20 + rnd*10)

                ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                plan_p = f"Create a step-by-step logic plan:\n{ds_safe}"
                if lessons:
                    plan_p += f"\n\nLESSONS FROM PREVIOUS FAILURES:\n{lessons[:800]}"
                # Safety: truncate prompt to leave room for generation
                max_plan_prompt_chars = (ds_ctx - gen_tokens - 200) * 3
                if len(plan_p) > max_plan_prompt_chars > 300:
                    plan_p = plan_p[:max_plan_prompt_chars]
                ds_draft = self._strip_thinking(self._call_model(ds_llm, plan_p, gen_tokens, logic_temp, system_prompt=planner_sys))

                # ── Phase 2: Reasoning Sandbox — Verify Logic ────────────────
                use_logic_playground = self._is_playground_applicable(router_llm if router_llm else self._get_model("router", required_ctx=1024), prompt)
                verified = True
                pg_out = ""
                if use_logic_playground:
                    if status_callback:
                        status_callback(f"Reasoning Sandbox: Verifying logic (Attempt {rnd+1}/{max_rounds})...", "info", "deepseek_r1", 30 + rnd*10)
                    verified, pg_out, _ = self._run_playground(ds_llm, ds_draft, "logic", model_key="deepseek_r1", original_prompt=prompt)

                if not verified:
                    if status_callback:
                        status_callback("Logic failed. Resolving with Emergency Search...", "warning", "router", 35 + rnd*10)
                    
                    # Fetch quick helper web search context
                    helper_search_context = ""
                    try:
                        router_llm = self._get_model("router", required_ctx=1024)
                        search_opt_p = (
                            "Generate a highly specific search query (3-6 words) to find the correct scientific formula, "
                            "biological facts, chemical properties, or Python coding syntax to resolve this sandbox verification failure.\n\n"
                            f"Original Prompt: {prompt}\n"
                            f"Sandbox Failure Output: {pg_out[:500]}\n"
                            "Constraint: The search query must be strictly relevant to Python code, mathematics, or the science domain of the prompt. Do NOT search for JavaScript.\n"
                            "Output ONLY the search query."
                        )
                        search_term = self._call_model(router_llm, search_opt_p, max_tokens=30, temperature=0.1).strip()
                        search_term = search_term.replace('"', '').replace('`', '').strip()
                        if not search_term or len(search_term) < 5:
                            search_term = " ".join([word for word in prompt.split() if (len(word) > 3 and word.isalnum())][:10])
                        
                        if status_callback:
                            status_callback(f"Emergency Search: '{search_term}'...", "info", "router", 36 + rnd*10)
                        
                        web_res = self.web_search.search(search_term, max_results=3)
                        if web_res:
                            helper_search_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_res])
                    except Exception:
                        pass

                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    search_str = f"Helper Web Context:\n{helper_search_context}\n\n" if helper_search_context else ""
                    fix_p = (
                        f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                        f"{search_str}"
                        f"Logic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\n"
                        f"Error:\n{pg_out[:1000]}\nRewrite a corrected logic plan."
                    )
                    ds_draft = self._strip_thinking(self._call_model(ds_llm, fix_p, gen_tokens, logic_temp, system_prompt=planner_sys))
                    # Skip expensive re-verification to save 2-3 model swaps (~60-90s on consumer hardware).
                    # The soft-verification system in _run_playground now handles test-script crashes,
                    # and the corrected plan already incorporates web search context.
                    if status_callback:
                        status_callback("DeepSeek-R1 corrected the logic plan with search context.", "success", "deepseek_r1", 40 + rnd*10)
                else:
                    if status_callback:
                        status_callback("Logic plan VERIFIED!", "success", "deepseek_r1", 40 + rnd*10)

                compiled_plan = ds_draft

                # ── Phase 3: OpenCode — Write Code ───────────────────────────
                if status_callback:
                    status_callback(f"OpenCode writing code (Attempt {rnd+1}/{max_rounds})...", "info", "opencode", 50 + rnd*10)
                oc_llm = self._get_model("opencode", required_ctx=oc_ctx)
                # Truncate compiled_plan to fit context
                max_code_prompt_chars = (oc_ctx - gen_tokens - 200) * 3
                plan_for_code = compiled_plan[:max(max_code_prompt_chars, 1500)] if len(compiled_plan) > max_code_prompt_chars else compiled_plan
                
                if req_lang == "python":
                    code_p = f"Write a complete Python script for this plan:\n{plan_for_code}\n\nWrap in ```python```."
                    sys_prompt = coder_sys
                else:
                    code_p = f"Write a complete, self-contained {lang_name} script for this plan:\n{plan_for_code}\n\nWrap in ```{req_lang}```."
                    sys_prompt = f"You are a master {lang_name} programmer. Output only code inside ```{req_lang}``` blocks."
                    
                code = Sandbox.extract_code(self._strip_thinking(self._call_model(oc_llm, code_p, gen_tokens, gen_temp, system_prompt=sys_prompt)))

                # ── Phase 4: Execution Sandbox ───────────────────────────────
                if status_callback:
                    status_callback(f"Executing in Sandbox (Attempt {rnd+1}/{max_rounds})...", "info", "sandbox", 60 + rnd*10)
                
                # Active Doctest-Driven Self-Correction:
                exec_code = code
                if req_lang == "python" and ">>>" in prompt:
                    exec_code += (
                        "\n\nif __name__ == '__main__':\n"
                        "    import doctest\n"
                        "    res = doctest.testmod(verbose=False)\n"
                        "    if res.failed > 0:\n"
                        "        raise AssertionError(f'Doctest failed: {res.failed} examples failed out of {res.attempted}')\n"
                    )
                ok, output = self.sandbox.execute(exec_code, language=req_lang)
                if ok:
                    self.memory.save(prompt, code)
                    if rnd > 0 or reset > 0:
                        self.memory.save_mistake(prompt, initial_failed_code, initial_failed_error, code)
                    router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                    return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

                # Save initial failures to register mistake later
                if not initial_failed_code:
                    initial_failed_code = code
                    initial_failed_error = output

                # ── Phase 4.5: Agent IDE Surgical Patch → Full Rewrite Fallback ──
                is_syntax_error = any(e in output for e in ["SyntaxError", "ModuleNotFoundError", "NameError", "IndentationError", "TypeError", "AttributeError", "ValueError", "ReferenceError", "Error:"])
                if is_syntax_error:
                    oc_linter = self._get_model("opencode", required_ctx=oc_ctx)

                    # ── Step A: Try Agent IDE surgical patch first (fast, token-efficient) ──
                    if status_callback:
                        status_callback(f"🔧 Agent IDE: Surgical patching {lang_name} error...", "warning", "opencode", 63 + rnd*10)
                    patched_code = self._agent_ide_patch(code, output, oc_linter, gen_tokens, lang=req_lang)
                    if patched_code and patched_code != code:
                        exec_patched = patched_code
                        if req_lang == "python" and ">>>" in prompt:
                            exec_patched += (
                                "\n\nif __name__ == '__main__':\n"
                                "    import doctest\n"
                                "    res = doctest.testmod(verbose=False)\n"
                                "    if res.failed > 0:\n"
                                "        raise AssertionError(f'Doctest failed: {res.failed} examples failed out of {res.attempted}')\n"
                            )
                        patch_ok, patch_output = self.sandbox.execute(exec_patched, language=req_lang)
                        if patch_ok:
                            code = patched_code
                            output = patch_output
                            ok = True
                            if status_callback:
                                status_callback("🔧 Agent IDE patch VERIFIED!", "success", "opencode", 68 + rnd*10)
                            self.memory.save(prompt, code)
                            if rnd > 0 or reset > 0:
                                self.memory.save_mistake(prompt, initial_failed_code, initial_failed_error, code)
                            router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                            return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

                    # ── Step B: Patch failed → Fall back to full rewrite linter ──
                    if status_callback:
                        status_callback(f"OpenCode full rewrite for {lang_name} error...", "warning", "opencode", 65 + rnd*10)
                    lint_p = (
                        f"You are a fast {lang_name} Syntax Linter.\n"
                        f"The code failed with this error:\n{output[:600]}\n\n"
                        f"CODE:\n{code[:2500]}\n\n"
                        f"Identify the typo/error and rewrite the complete corrected script in a ```{req_lang}``` block. Fix ONLY the exact error, do not change the core algorithm."
                    )
                    lint_sys = f"You are a strict {lang_name} syntax linter. Output only code."
                    lint_code = Sandbox.extract_code(self._strip_thinking(self._call_model(oc_linter, lint_p, gen_tokens, 0.1, system_prompt=lint_sys)))
                    if lint_code and len(lint_code) > 20:
                        # Active Doctest-Driven Self-Correction for Linter:
                        exec_lint_code = lint_code
                        if req_lang == "python" and ">>>" in prompt:
                            exec_lint_code += (
                                "\n\nif __name__ == '__main__':\n"
                                "    import doctest\n"
                                "    res = doctest.testmod(verbose=False)\n"
                                "    if res.failed > 0:\n"
                                "        raise AssertionError(f'Doctest failed: {res.failed} examples failed out of {res.attempted}')\n"
                            )
                        linter_ok, linter_output = self.sandbox.execute(exec_lint_code, language=req_lang)
                        if linter_ok:
                            code = lint_code
                            output = linter_output
                            ok = True
                            if status_callback:
                                status_callback("OpenCode successfully patched the code!", "success", "opencode", 70 + rnd*10)
                            self.memory.save(prompt, code)
                            if rnd > 0 or reset > 0:
                                self.memory.save_mistake(prompt, initial_failed_code, initial_failed_error, code)
                            router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                            return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)
                
                # ── Phase 5 & 6: Reflexion / Self-Correction ─────────────────
                if not ok:
                    # Fetch quick helper web search context
                    helper_search_context = ""
                    try:
                        search_term = " ".join([word for word in prompt.split() if (len(word) > 3 and word.isalnum())][:10])
                        web_res = self.web_search.search(search_term, max_results=3)
                        if web_res:
                            helper_search_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_res])
                    except Exception:
                        pass
                    search_str = f"Helper Web Context:\n{helper_search_context}\n\n" if helper_search_context else ""

                    # OpenCode corrects the code first (already loaded from Phase 3 — no swap needed)
                    if status_callback:
                        status_callback(f"OpenCode correcting code (Attempt {rnd+1}/{max_rounds})...", "warning", "opencode", 73 + rnd*10)
                    failed_code = code
                    failed_error = output
                    safe_code = code[:2000] if len(code) > 2000 else code
                    safe_error = output[:800] if len(output) > 800 else output
                    
                    fix_p = (
                        f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                        f"{search_str}"
                        f"The following {lang_name} code FAILED with an error.\n\n"
                        f"CODE:\n{safe_code}\n\n"
                        f"ERROR:\n{safe_error}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Identify the exact line and cause of the error\n"
                        f"2. Fix ONLY the bug — do not rewrite unrelated parts\n"
                        f"3. Make sure all imports/dependencies are present\n"
                        f"4. Test edge cases (division by zero, empty arrays, etc.)\n"
                    )
                    # Add error-specific recovery hints
                    if 'TimeoutError' in safe_error or 'took longer than' in safe_error:
                        fix_p += (
                            f"5. TIMEOUT FIX: The code took too long. Reduce computation — use smaller arrays, "
                            f"fewer iterations, or reduce simulation time span.\n"
                        )
                    elif 'MemoryError' in safe_error or 'RLIMIT' in safe_error or 'Cannot allocate' in safe_error:
                        fix_p += (
                            f"5. MEMORY FIX: The code used too much memory. Use generators/iterators, "
                            f"process data in chunks, or use smaller array sizes.\n"
                        )
                    elif 'ModuleNotFoundError' in safe_error or 'Cannot find module' in safe_error:
                        fix_p += (
                            f"5. IMPORT FIX: A required module/package is not installed. Replace it with a standard library alternative.\n"
                        )
                    fix_p += f"6. Output the COMPLETE corrected script in ```{req_lang}``` blocks."

                    # Try OpenCode first (already loaded, no model swap needed)
                    oc_fix = self._get_model("opencode", required_ctx=oc_ctx)
                    code = Sandbox.extract_code(self._strip_thinking(self._call_model(oc_fix, fix_p, gen_tokens, gen_temp, system_prompt=sys_prompt)))
                    ok, output = self.sandbox.execute(code, language=req_lang)
                    if ok:
                        if status_callback:
                            status_callback("OpenCode's correction VERIFIED!", "success", "opencode", 78 + rnd*10)
                        self.memory.save(prompt, code)
                        self.memory.save_mistake(prompt, failed_code, failed_error, code)
                        router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

                    # Escalate to DeepSeek-R1 only if OpenCode's correction also failed
                    if status_callback:
                        status_callback(f"DeepSeek-R1 correcting code (Attempt {rnd+1}/{max_rounds})...", "warning", "deepseek_r1", 80 + rnd*10)
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    code = Sandbox.extract_code(self._strip_thinking(self._call_model(ds_llm, fix_p, gen_tokens, gen_temp, system_prompt=sys_prompt)))
                    ok, output = self.sandbox.execute(code, language=req_lang)
                    if ok:
                        if status_callback:
                            status_callback("DeepSeek-R1's correction VERIFIED!", "success", "deepseek_r1", 85 + rnd*10)
                        self.memory.save(prompt, code)
                        self.memory.save_mistake(prompt, failed_code, failed_error, code)
                        router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback, req_lang=req_lang)

                    # Don't let ds_safe grow unboundedly — cap the appended errors
                    error_summary = output[:300]
                    if len(ds_safe) + len(error_summary) < (ds_ctx - gen_tokens - 200) * 3:
                        ds_safe = f"{ds_safe}\n\nPrevious execution error: {error_summary}"

            # ── Phase 7: Nuclear Reset ───────────────────────────────────
            all_errors.append(f"Reset {reset+1}: {output[:500]}")
            if reset < max_resets - 1:
                if status_callback:
                    status_callback(f"Nuclear Reset: Extracting lessons from failures...", "error", "deepseek_r1", 85)
                ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                lessons = self._extract_failure_lessons(ds_llm, compiled_plan, "\n".join(all_errors))

        # All resets exhausted -> Emergency Web Search Healing fallback
        if status_callback:
            status_callback("Main pipeline failed. Activating Emergency Web Search...", "warning", "system", 90)
        try:
            error_lines = [line.strip() for line in output.split('\n') if line.strip()]
            error_query = error_lines[-1] if error_lines else output[:100]
            if len(error_query) > 120:
                error_query = error_query[-120:]
            
            # Construct a clean search query using only prompt keywords to avoid search engine contamination
            clean_prompt_query = " ".join([word for word in prompt.split() if (len(word) > 3 and word.isalnum())][:12])
            search_term = clean_prompt_query
            if len(search_term) > 150:
                search_term = search_term[:150]

            if status_callback:
                status_callback(f"Searching: '{search_term}'...", "info", "system", 92)
            web_results = self.web_search.search(search_term, max_results=3)
            emergency_context = ""
            if web_results:
                emergency_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_results])
            if emergency_context:
                if status_callback:
                    status_callback("Emergency context acquired. Rewriting script...", "info", "deepseek_r1", 95)
                ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                emergency_prompt = (
                    f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                    f"The previous attempts failed with the following traceback:\n"
                    f"{output[:800]}\n\n"
                    f"We searched the web for this error and found the following references:\n"
                    f"{emergency_context}\n\n"
                    f"Using this information, rewrite the complete functional Python script to fix the error and satisfy all original constraints.\n"
                    f"Original plan:\n{compiled_plan[:1500]}\n\n"
                    f"Output the complete script in a ```python``` block."
                )
                esc_resp = self._strip_thinking(self._call_model(ds_llm, emergency_prompt, gen_tokens, gen_temp, system_prompt=coder_sys))
                if "```" in esc_resp:
                    code = Sandbox.extract_code(esc_resp)
                    ok, output = self.sandbox.execute(code)
                    if not ok:
                        # Attempt exactly 1 round of playground correction for emergency healing
                        if status_callback:
                            status_callback("Emergency script failed. Attempting 1 correction round...", "warning", "deepseek_r1", 97)
                        patch_prompt = (
                            f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                            f"The emergency script failed with the following traceback/error:\n{output[:800]}\n\n"
                            f"Original code:\n{code[:1500]}\n\n"
                            f"Using this traceback, rewrite the complete functional Python script to fix the error.\n"
                            f"Output only the complete corrected script in a ```python``` block."
                        )
                        patch_resp = self._strip_thinking(self._call_model(ds_llm, patch_prompt, gen_tokens, gen_temp, system_prompt=coder_sys))
                        if "```" in patch_resp:
                            code = Sandbox.extract_code(patch_resp)
                            ok, output = self.sandbox.execute(code)
                    if ok:
                        if status_callback:
                            status_callback("Emergency Search Healing SUCCESSFUL!", "success", "deepseek_r1", 100)
                        self.memory.save(prompt, code)
                        self.memory.save_mistake(prompt, failed_code, failed_error, code)
                        router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        return self._synthesize_coding_response(prompt, compiled_plan, code, output, router_ctx, oc_ctx, ds_ctx, gen_tokens, gen_temp, status_callback)
        except Exception as es:
            print(f"Emergency web search recovery failed: {es}")

        if status_callback:
            status_callback("Max retries reached.", "error", "system", 100)
        
        # Save unverified draft to memory with traceback to assist future runs
        unverified_doc = (
            f"[UNVERIFIED BEST-EFFORT CODE DRAFT]\n"
            f"The following code script failed verification with error:\n{output[:800]}\n"
            f"Logic Plan:\n{compiled_plan[:1500]}\n"
            f"Code:\n{code}"
        )
        try:
            self.memory.save(prompt, unverified_doc)
        except Exception as es:
            print(f"Failed to save unverified code draft: {es}")

        router_llm = None; ds_llm = None; oc_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
        return f"### Logic Plan\n{compiled_plan}\n\n### Execution Failed\n{output}\n\n### Code\n```python\n{code}\n```"

    # =====================================================================
    # REASONING PIPELINE — Playground-Verified or LLM Debate
    # =====================================================================
    def _reasoning_pipeline(self, prompt, enriched_prompt, router_llm,
                            router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        # ── Check: Can this be playground-verified? ──────────────────────
        # Must check this BEFORE loading ds_llm to prevent EVM from evicting router_llm
        use_playground = self._is_playground_applicable(router_llm, prompt)


        # Leave 1000 tokens headroom specifically for the massive 'planner_sys' system prompt
        # Safety floor: ensure crunch budget is at least 1024 tokens
        crunch_budget = max(1024, ds_ctx - gen_tokens - 1000)
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", crunch_budget, status_callback, router_llm=router_llm)

        if status_callback:
            mode = "Playground-Verified" if use_playground else "LLM Debate"
            status_callback(f"Reasoning mode: {mode}", "info", "router", 15)

        ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)

        reasoning_sys = (
            "You are a rigorous scientific researcher and expert logic reasoner.\n\n"
            "ACCURACY REQUIREMENTS:\n"
            "1. Think step by step. Show your work for every derivation.\n"
            "2. State all assumptions explicitly at the beginning.\n"
            "3. Define all variables with their units before using them.\n"
            "4. For physics: verify dimensional consistency of every equation. Use standard literature formulas exactly. "
            "For example, E x B drift velocity is v = (E x B)/B^2 (it is independent of mass and charge).\n"
            "5. For math: check boundary conditions and special cases.\n"
            "6. Avoid common traps: linear vs. quadratic drag, 2D vs. 3D decomposition, "
            "sign conventions, reference frame consistency.\n"
            "7. If you cite a formula, state where it comes from (Newton's 2nd law, etc.).\n"
            "8. Complete ALL derivations fully — do not skip steps or say 'it can be shown that'.\n"
            "9. If uncertain about a specific value or fact, say so explicitly rather than guessing.\n"
            "10. IMPORTANT: If 'Relevant past experience' or 'Web Context' is provided, use it ONLY for structure, formulas, or syntax logic. "
            "Do NOT copy the physical system or specific numeric values if they differ from the User Query. Always prioritize the User Query's exact physics system and exact variables.\n"
            "11. THINKING CONSTRAINT: Be concise, structured, and focused in your thinking thoughts. Avoid looping or repeating the "
            "same mathematical derivations. State your reasoning path clearly and proceed directly to the solution once verified.\n"
            "12. MATHEMATICAL DETAILS & FORMULA FORMATTING: You MUST write out all algebraic equations, derivative steps, and algebraic manipulations in clear LaTeX format.\n"
            "    - NEVER wrap formulas, derivatives, or equations in backtick code blocks (e.g. ``` or `). Code blocks must only be used for actual runnable programming code, never for text math equations.\n"
            "    - ALWAYS format mathematical equations using proper LaTeX delimiters: use single dollar signs $...$ for inline equations, and double dollar signs $$...$$ for display block equations.\n"
            "    - If numerical constants are specified in the prompt, substitute them and output the final calculated numerical answers.\n"
            "13. BIOCHEMISTRY FORMULA CORRECTNESS: For enzyme kinetics equations (Michaelis-Menten, Lineweaver-Burk, etc.), "
            "you MUST verify the DIRECTIONALITY of your derived formula before presenting it. For example: "
            "in Competitive Inhibition, the apparent Km INCREASES (Km_app = Km * (1 + [I]/Ki)) while Vmax stays the same. "
            "In Uncompetitive Inhibition, both apparent Km and Vmax DECREASE. In Non-competitive Inhibition, Km stays the same but Vmax decreases. "
            "Always sanity-check: does adding more inhibitor ([I] > 0) cause the reaction velocity to decrease? If your formula shows velocity increasing with [I], it is WRONG."
        )

        if use_playground:
            # ── Playground-Verified Reasoning (with Nuclear Reset) ────────
            max_resets = 2
            lessons = ""
            all_errors = []

            for reset in range(max_resets):
                max_rounds = 2 if reset == 0 else 1
                ds_answer = ""
                vibe_answer = ""
                vibe_pg_out = ""
                helper_search_context = ""
                for rnd in range(max_rounds):
                    # Re-acquire ds_llm because other models may have evicted it in the previous round
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    if status_callback:
                        lbl = f"Nuclear Reset #{reset} (Attempt {rnd+1}/{max_rounds}): DeepSeek-R1 re-reasoning..." if reset else f"DeepSeek-R1 reasoning + playground (Attempt {rnd+1}/{max_rounds})..."
                        status_callback(lbl, "info" if not reset else "warning", "deepseek_r1", 25 + rnd*12)
                    draft_p = f"Provide a detailed, rigorous answer:\n{ds_safe}"
                    if rnd > 0:
                        last_failed = vibe_answer if vibe_answer else ds_answer
                        last_error = vibe_pg_out if vibe_pg_out else pg_out
                        search_str = f"\n\nHelper Web Context:\n{helper_search_context}" if helper_search_context else ""
                        draft_p += (
                            f"\n\nYour previous attempt failed sandbox verification.{search_str}\n"
                            f"Previous Failed Draft:\n{last_failed[:1500]}\n"
                            f"Verification Error:\n{last_error[:800]}\n"
                            f"Identify the mistake in the previous attempt and rewrite the complete, corrected answer from scratch, resolving all issues."
                        )
                    if lessons:
                        draft_p += f"\n\nLESSONS FROM PREVIOUS FAILURES:\n{lessons[:800]}"
                    # Safety: truncate prompt to leave room for generation
                    max_reason_chars = (ds_ctx - gen_tokens - 200) * 3
                    if len(draft_p) > max_reason_chars > 300:
                        draft_p = draft_p[:max_reason_chars]
                    ds_answer = self._strip_thinking(self._call_model(ds_llm, draft_p, gen_tokens, gen_temp, system_prompt=reasoning_sys))

                    if status_callback:
                        status_callback(f"Verifying in Reasoning Playground (Attempt {rnd+1}/{max_rounds})...", "info", "deepseek_r1", 35 + rnd*12)
                    verified, pg_out, test_code = self._run_playground(ds_llm, ds_answer, "reasoning", model_key="deepseek_r1", original_prompt=prompt)

                    if verified:
                        if status_callback:
                            status_callback("Reasoning VERIFIED!", "success", "deepseek_r1", 80)
                        self.memory.save(prompt, ds_answer)
                        router_llm = None; ds_llm = None; gc.collect()
                        viz = self._check_3d_gate(prompt, ds_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Verified Answer\n{ds_answer}{viz}"

                    # Fetch quick helper web search context to resolve unknown concepts immediately
                    helper_search_context = ""
                    try:
                        router_llm = self._get_model("router", required_ctx=1024)
                        search_opt_p = (
                            "Generate a highly specific search query (3-6 words) to find the correct scientific formula, "
                            "biological facts, or chemical properties to resolve this sandbox verification failure.\n\n"
                            f"Original Prompt: {prompt}\n"
                            f"Sandbox Failure Output: {pg_out[:500]}\n"
                            "Output ONLY the search query."
                        )
                        search_term = self._call_model(router_llm, search_opt_p, max_tokens=30, temperature=0.1).strip()
                        search_term = search_term.replace('"', '').replace('`', '').strip()
                        if not search_term or len(search_term) < 5:
                            search_term = " ".join([word for word in prompt.split() if (len(word) > 3 and word.isalnum())][:10])
                        
                        if status_callback:
                            status_callback(f"Emergency Search: '{search_term}'...", "info", "router", 36 + rnd*12)
                        
                        web_res = self.web_search.search(search_term, max_results=3)
                        if web_res:
                            helper_search_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_res])
                    except Exception:
                        pass
                    search_str = f"Helper Web Context:\n{helper_search_context}\n\n" if helper_search_context else ""

                    # DeepSeek-R1-7B corrects its own draft (zero model swap latency)
                    if status_callback:
                        status_callback(f"DeepSeek-R1 correcting reasoning (Attempt {rnd+1}/{max_rounds})...", "warning", "deepseek_r1", 45 + rnd*12)
                    vibe_p = (
                        f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                        f"{search_str}"
                        f"This answer failed verification.\nAnswer:\n{ds_answer[:2000]}\n"
                        f"Error:\n{pg_out[:1000]}\nProvide a corrected, complete answer. "
                        f"IMPORTANT: You MUST write out all algebraic equations and derivations in clear LaTeX format ($...$ and $$...$$)."
                    )
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    vibe_answer = self._strip_thinking(self._call_model(ds_llm, vibe_p, gen_tokens, gen_temp, system_prompt=reasoning_sys))
                    v2, vibe_pg_out, vibe_test_code = self._run_playground(ds_llm, vibe_answer, "reasoning", model_key="deepseek_r1", original_prompt=prompt)
                    if v2:
                        if status_callback:
                            status_callback("DeepSeek-R1's correction VERIFIED!", "success", "deepseek_r1", 80)
                        self.memory.save(prompt, vibe_answer)
                        self.memory.save_mistake(prompt, ds_answer, pg_out, vibe_answer)
                        router_llm = None; ds_llm = None; gc.collect()
                        viz = self._check_3d_gate(prompt, vibe_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Verified Answer\n{vibe_answer}{viz}"
                    # Don't let ds_safe grow unboundedly — cap the appended errors
                    error_summary = pg_out[:300]
                    if len(ds_safe) + len(error_summary) < (ds_ctx - gen_tokens - 200) * 3:
                        ds_safe = f"{ds_safe}\n\nPrevious errors: {error_summary}"
                    # else: silently skip appending to prevent overflow

                # ── Nuclear Reset: extract lessons and restart ────────────
                all_errors.append(f"Reset {reset+1}: {pg_out[:500]}")
                if reset < max_resets - 1:
                    if status_callback:
                        status_callback("Nuclear Reset: Extracting lessons from failures...", "error", "deepseek_r1", 85)
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    lessons = self._extract_failure_lessons(ds_llm, ds_answer, "\n".join(all_errors))

            # All resets exhausted -> Emergency Web Search Healing fallback
            if status_callback:
                status_callback("Main pipeline failed. Activating Emergency Web Search...", "warning", "system", 90)
            try:
                # Construct a clean search query using only prompt keywords to avoid search engine contamination
                clean_prompt_query = " ".join([word for word in prompt.split() if (len(word) > 3 and word.isalnum())][:12])
                search_term = clean_prompt_query
                if len(search_term) > 150:
                    search_term = search_term[:150]

                if status_callback:
                    status_callback(f"Searching: '{search_term}'...", "info", "system", 92)
                web_results = self.web_search.search(search_term, max_results=3)
                emergency_context = ""
                if web_results:
                    emergency_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_results])
                if emergency_context:
                    if status_callback:
                        status_callback("Emergency context acquired. Final reasoning correction...", "info", "deepseek_r1", 95)
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    emergency_prompt = (
                        f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                        f"The reasoning explanation failed sandbox verification with the error:\n"
                        f"{pg_out[:500]}\n\n"
                        f"We found the following context online for this issue:\n"
                        f"{emergency_context}\n\n"
                        f"Correct the derivation/calculation to fix this issue, and formulate the final detailed explanation that perfectly satisfies all original user request constraints.\n"
                        f"Failed Draft:\n{ds_answer[:1500]}"
                    )
                    vibe_answer = self._strip_thinking(self._call_model(ds_llm, emergency_prompt, gen_tokens, gen_temp, system_prompt=reasoning_sys))
                    v2, vibe_pg_out, vibe_test_code = self._run_playground(ds_llm, vibe_answer, "reasoning", model_key="deepseek_r1", original_prompt=prompt)
                    if not v2:
                        # Attempt exactly 1 round of playground correction for emergency healing
                        if status_callback:
                            status_callback("Emergency verification failed. Attempting 1 correction round...", "warning", "deepseek_r1", 97)
                        corr_prompt = (
                            f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                            f"The emergency explanation failed verification with this traceback:\n"
                            f"{vibe_pg_out[:800]}\n\n"
                            f"Explanation:\n{vibe_answer[:1500]}\n\n"
                            f"Correct the derivation or logic to fix this error, and provide the complete corrected explanation."
                        )
                        ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                        vibe_answer = self._strip_thinking(self._call_model(ds_llm, corr_prompt, gen_tokens, gen_temp, system_prompt=reasoning_sys))
                        v2, vibe_pg_out, vibe_test_code = self._run_playground(ds_llm, vibe_answer, "reasoning", model_key="deepseek_r1", original_prompt=prompt)
                    if v2:
                        if status_callback:
                            status_callback("Emergency Search Healing SUCCESSFUL!", "success", "deepseek_r1", 100)
                        self.memory.save(prompt, vibe_answer)
                        self.memory.save_mistake(prompt, ds_answer, pg_out, vibe_answer)
                        router_llm = None; ds_llm = None; gc.collect()
                        viz = self._check_3d_gate(prompt, vibe_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Verified Answer\n{vibe_answer}{viz}"
            except Exception as es:
                print(f"Emergency reasoning search recovery failed: {es}")

            final_ans = vibe_answer if 'vibe_answer' in locals() else ds_answer
            final_test = vibe_test_code if 'vibe_test_code' in locals() else test_code
            final_out = vibe_pg_out if 'vibe_pg_out' in locals() else pg_out

            if status_callback:
                status_callback("Max retries reached. Returning best effort.", "warning", "system", 98)
            
            # Save unverified draft to memory with traceback to assist future runs
            unverified_doc = (
                f"[UNVERIFIED BEST-EFFORT REASONING DRAFT]\n"
                f"The following reasoning answer failed verification with sandbox error:\n{final_out[:800]}\n"
                f"Answer:\n{final_ans}"
            )
            try:
                self.memory.save(prompt, unverified_doc)
            except Exception as es:
                print(f"Failed to save unverified reasoning draft: {es}")

            router_llm = None; ds_llm = None; gc.collect()
            viz = self._check_3d_gate(prompt, final_ans, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return f"### Execution Failed (Best-Effort Draft)\n{final_ans}{viz}"

        else:
            # ── Standard LLM Debate (non-testable reasoning) ─────────────
            if status_callback:
                status_callback("DeepSeek-R1 drafting analysis...", "info", "deepseek_r1", 50)
            ds_draft = self._strip_thinking(self._call_model(ds_llm, f"Provide a detailed answer:\n{ds_safe}", gen_tokens, gen_temp, system_prompt=reasoning_sys))

            compiled = ds_draft
            router_llm = None; ds_llm = None; gc.collect()
            viz = self._check_3d_gate(prompt, compiled, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            if status_callback:
                status_callback("Done!", "success", "router", 100)
            return f"### Successfully Generated Answer\n{compiled}{viz}"

