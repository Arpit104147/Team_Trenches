import os
import gc
import re
import json
import psutil
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
        self.enable_web_search = False
        
        self.sandbox = Sandbox(timeout=300)
        self.memory = Memory()
        self.web_search = WebSearch()
        self.loaded_models = {}
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
                
                # ── Kaggle dGPU Hot-Swap Mode Detection ──
                # If System RAM is massive (>24GB) but VRAM is restricted (<=16GB)
                if self.total_ram_gb >= 24 and total_vram_gb <= 16:
                    ram_percent = psutil.virtual_memory().percent
                    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None or os.path.exists('/kaggle')
                    # Activate if we are in Kaggle or have enough free memory (RAM usage < 25%)
                    if is_kaggle or ram_percent < 25.0:
                        self.kaggle_hotswap_mode = True
                        # ── EVM Resource Override ──────────────────────────
                        # In EVM mode, only ONE model is ever in VRAM at a time,
                        # and System RAM is a dedicated holding area.
                        # Safe to use 90-95% of all resources.
                        # RAM:  keep only 5% free (~1.5 GB on 31 GB) for OS kernel
                        # VRAM: keep only 5% free (~0.8 GB on 16 GB) for CUDA runtime
                        self.ram_safety_gb = round(self.total_ram_gb * 0.05, 1)
                        self.vram_safety_gb = round(total_vram_gb * 0.05, 1)
                        print("🚀 DMA: Activated EVM (Enterprise VRAM Multiplexing)!")
                        print(f"   ⚡ EVM Override: RAM threshold = {self.ram_safety_gb:.1f} GB "
                              f"(95% usable of {self.total_ram_gb:.0f} GB)")
                        print(f"   ⚡ EVM Override: VRAM threshold = {self.vram_safety_gb:.1f} GB "
                              f"(95% usable of {total_vram_gb:.0f} GB)")
                    else:
                        print(f"⚠️ DMA: Hot-Swap skipped — RAM usage too high ({ram_percent:.1f}%)")
                        
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
                    self.max_auto_ctx = 8192  # Increased from 4096 to prevent token cutoff
                elif total_vram_gb <= 24:
                    self.max_auto_ctx = 8192  # Increased from 6144
                elif total_vram_gb <= 48:
                    self.max_auto_ctx = 32768  # A6000 (48GB) / A100 (40GB) -> 32k context
                else:
                    self.max_auto_ctx = 65536  # H100 (80GB) -> 64k context
                print(f"📐 DMA: Auto-context ceiling = {self.max_auto_ctx} tokens (based on {total_vram_gb:.0f} GB VRAM)")
            except Exception:
                pass
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
                
                if vram_used_pct < 95.0:
                    five_percent_vram_gb = total_vram_gb * 0.05
                    surplus_vram = free_vram_gb - five_percent_vram_gb
                    if surplus_vram > 0:
                        vram_allowed_ceiling = int(base_limit + surplus_vram * 8000)
                        vram_allowed_ceiling = min(32768, vram_allowed_ceiling)
            except Exception:
                pass

        hard_limit = min(ram_allowed_ceiling, vram_allowed_ceiling)
        hard_limit = max(8192, hard_limit)
        
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
                    # Theoretical free VRAM after swap and load
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
        
        # Model-specific physical context ceilings to prevent VRAM OOM and RoPE overflow
        model_ceilings = {
            "router": 8192,
            "vibethinker": 8192,
            "opencode": 16384,
            "deepseek_r1": 16384
        }
        model_cap = model_ceilings.get(model_key, 16384)
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
        size_map = {"router": 3.0, "deepseek_r1": 6.0, "vibethinker": 1.4,
                    "opencode": 5.2, "qwen_vl": 6.5}
        return size_map.get(model_key, 5.0)

    def _check_memory_pressure(self, required_vram_gb=None):
        """LRU-based eviction loop. Evicts models one-by-one until safe.
        
        If required_vram_gb is provided, evicts until free VRAM >= required_vram_gb + 2 GB buffer.
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
                for gpu_idx in range(torch.cuda.device_count()):
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

    def _get_model(self, model_key, required_ctx=None):
        """Load a model with Dynamic Memory Allocator protection and dynamic context sizing."""
        import time
        if required_ctx is None:
            required_ctx = self.context_length if self.context_length > 0 else 8192

        if model_key in self.loaded_models:
            model_obj = self.loaded_models[model_key]
            # Context expansion — safe reload on all platforms (including Intel iGPU/Vulkan)
            if hasattr(model_obj, "n_ctx") and required_ctx > model_obj.n_ctx():
                print(f"🔄 Reloading '{model_key}' to expand context: {model_obj.n_ctx()} -> {required_ctx}")
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
                # ⚠️ llama-cpp-python manages its own CUDA context, independent of PyTorch.
                # On Kaggle P100 (sm_60), PyTorch's CUDA runtime is INCOMPATIBLE and
                # torch.cuda.synchronize() can segfault the process.
                # We rely on time.sleep() to let llama.cpp's internal cudaFree() complete.
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass  # Safe to ignore — llama.cpp doesn't use PyTorch's CUDA allocator
                if torch and hasattr(torch, "xpu") and torch.xpu.is_available():
                    try:
                        torch.xpu.empty_cache()
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
        # On Kaggle P100 (16GB VRAM, 32GB RAM), aggressively flush ALL other models
        # from VRAM so the incoming model gets 100% of the VRAM KV Cache space.
        # After EVM flush, skip _check_memory_pressure entirely — EVM guarantees
        # all VRAM is free, and the pressure check's torch.cuda calls can crash
        # on P100 (sm_60) where PyTorch's CUDA runtime is incompatible.
        evm_flushed = False
        if getattr(self, 'kaggle_hotswap_mode', False) and self.loaded_models:
            models_to_flush = [mk for mk in list(self.loaded_models.keys()) if mk != model_key]
            if models_to_flush:
                for mk in models_to_flush:
                    print(f"🔄 DMA (EVM Hot-Swap): Unloading '{mk}' from VRAM...")
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
                # Use time.sleep instead of torch.cuda.synchronize which crashes on P100
                try:
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass  # Safe — llama.cpp manages its own CUDA memory
                time.sleep(2)  # Let llama.cpp's internal cudaFree() complete
                # Verify eviction actually freed VRAM
                try:
                    free_vram, total_vram = torch.cuda.mem_get_info(0)
                    free_vram_gb = free_vram / (1024 ** 3)
                    print(f"✅ DMA (EVM): VRAM after eviction: {free_vram_gb:.1f} GB free / {total_vram/(1024**3):.0f} GB total")
                except Exception:
                    pass
                evm_flushed = True
            else:
                evm_flushed = True  # Only our model is loaded, all VRAM is ours
                
        # Skip memory pressure check if EVM already cleared VRAM — the pressure
        # check runs torch.cuda.synchronize() internally which can crash on P100 (sm_60)
        if not evm_flushed:
            self._check_memory_pressure(required_vram_gb=est_model_gb)

        # ── iGPU Unified Memory Guard ─────────────────────────────────────
        # On Intel Iris Xe (and similar iGPUs), RAM IS VRAM. Loading two 7B
        # models simultaneously causes glibc heap corruption ('corrupted size
        # vs. prev_size') because llama.cpp does one giant contiguous malloc.
        # Fix: if we are on an iGPU (no discrete CUDA GPU), proactively evict
        # the least-recently-used model before loading a new heavy one.
        is_igpu = not (torch and torch.cuda.is_available())
        if is_igpu and self.loaded_models:
            # Estimate: 7B Q6_K models need ~6 GB, 1.5B need ~1.4 GB
            # If we already have a 7B loaded, evict it first
            free_ram = self._get_ram_free_gb()
            if free_ram < self.total_ram_gb * 0.35:  # Less than 35% RAM free
                print(f"🧠 DMA (iGPU Guard): Pre-emptive eviction — only {free_ram:.1f} GB free")
                self._evict_lru_model()

        if not is_model_downloaded(model_key):
            raise Exception(f"Model '{model_key}' is not downloaded. "
                            f"Place the weights in models/ and restart.")

        model_path = get_model_path(model_key)
        
        # ── GGUF Models (llama_cpp) ──────────────────────────────────────
        if model_path.endswith('.gguf'):
            from llama_cpp import Llama
            kwargs = {
                "model_path": model_path,
                "n_ctx": required_ctx,
                "n_gpu_layers": self.gpu_layers if self.device_mode != "cpu" else 0,
                "verbose": False
            }
            # Dual-GPU: send heavy 7B models to GPU 1, lighter ones to GPU 0
            if self.dual_gpu_pipeline and model_key in ["deepseek_r1", "opencode"]:
                kwargs["main_gpu"] = 1
            elif self.dual_gpu_pipeline:
                kwargs["main_gpu"] = 0

            try:
                llm = Llama(**kwargs)
            except Exception as e:
                print(f"⚠️ DMA: Failed to create llama_context on GPU for '{model_key}' ({e}). Falling back to CPU...")
                kwargs["n_gpu_layers"] = 0
                kwargs.pop("main_gpu", None)
                llm = Llama(**kwargs)

            self.loaded_models[model_key] = llm
            self._touch_model(model_key)
            print(f"✅ Loaded GGUF model '{model_key}' ({os.path.basename(model_path)})" + 
                  (" (CPU Fallback)" if kwargs.get("n_gpu_layers") == 0 and self.device_mode != "cpu" else ""))
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
                    
            if self.dual_gpu_pipeline and model_key == "vibethinker":
                device = "cuda:1"
                
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
    def transcribe_image(self, image_path, status_callback=None):
        if status_callback:
            status_callback("Qwen 2.5-VL parsing image...", "info", "qwen_vl", 5)
        llm = self._get_model("qwen_vl")
        vision_prompt = "Describe this image and extract all text and logic from it."
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
        """Remove <think>...</think> blocks from DeepSeek R1 / VibeThinker output."""
        if not text:
            return text
            
        # Handle unclosed <think> tag gracefully
        if '<think>' in text and '</think>' not in text:
            before_think, after_think = text.split('<think>', 1)
            if '```' in after_think:
                # Close the think block right before the first code fence
                after_think = after_think.replace('```', '</think>\n```', 1)
                text = before_think + '<think>' + after_think
            else:
                # If there's no code fence and before_think is empty, the model only generated thinking.
                # Do NOT return empty. Return the thinking process (without <think> tag) so the user gets a response.
                if not before_think.strip():
                    return after_think.strip()
                return before_think.strip()
                
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        if cleaned:
            return cleaned
            
        # If stripping removed EVERYTHING, it means the model put its entire answer 
        # inside the think block. We must strip ONLY the tags, keeping the content!
        text_without_tags = re.sub(r'</?think>', '', text)
        return text_without_tags.strip()



    def _crunch_prompt(self, prompt, target_model, prompt_token_budget, status_callback=None, router_llm=None):
        """Compresses a massive prompt using semantic line boundaries and fast summarization."""
        # Ensure router is loaded first to use its tokenizer
        # Use small context for summarization — avoid wasteful VRAM allocation
        if router_llm is None:
            router_llm = self._get_model("router", required_ctx=2048)

        # Precise Token Estimation
        if hasattr(router_llm, "tokenize"):
            est_tokens = len(router_llm.tokenize(prompt.encode('utf-8')))
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
        
        # 30% of allowed budget at the top, 70% at the bottom
        top_char_budget = int(chars_allowed * 0.3)
        bottom_char_budget = int(chars_allowed * 0.7)
        
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
        
        # Safety net: Don't spend hours summarizing a 500MB file
        if len(middle_chunk) > 50000:
            middle_chunk = middle_chunk[:25000] + "\n...[SNIPPED EXTREME LENGTH]...\n" + middle_chunk[-25000:]
            
        compress_prompt = f"Summarize this middle section concisely. Keep all logic, facts, and code structure intact:\n{middle_chunk}"
        
        if isinstance(router_llm, TransformerWrapper):
            middle_summary = router_llm(compress_prompt, max_tokens=1024)
        else:
            middle_summary = router_llm.create_chat_completion(
                messages=[{"role": "user", "content": compress_prompt}], 
                max_tokens=1024
            )['choices'][0]['message']['content']
            
        crunched = f"{start_chunk}\n\n[--- CRUNCHED SUMMARY OF MIDDLE SECTION ---]\n{middle_summary}\n[--- END SUMMARY ---]\n\n{end_chunk}"
        return crunched

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
                est_prompt_tokens = len(llm.tokenize(prompt.encode('utf-8'))) + 120
                if system_prompt:
                    est_prompt_tokens += len(llm.tokenize(system_prompt.encode('utf-8')))
            else:
                est_prompt_tokens = len(prompt) // 3 + 120
                if system_prompt:
                    est_prompt_tokens += len(system_prompt) // 3
            
            # Smart token allocation with Model-Aware Minimums
            is_reasoning = "deepseek" in getattr(llm, "model_path", "").lower()
            absolute_min = 2048 if is_reasoning else 512
            
            if est_prompt_tokens + max_tokens > ctx:
                # Force a larger generation runway for reasoning models
                max_tokens = max(absolute_min, ctx - est_prompt_tokens - 50)
            
            # If even with minimum generation tokens the prompt doesn't fit, truncate the prompt semantically
            max_prompt_tokens = ctx - max_tokens - 120
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
                    est_prompt_tokens = len(llm.tokenize(prompt.encode('utf-8'))) + 120
                else:
                    est_prompt_tokens = len(prompt) // 3 + 120
            
            # Ensure we never request more tokens than the available space
            safe_max = ctx - est_prompt_tokens
            if safe_max < 64:
                safe_max = 64  # Desperate fallback — at least try to get something
            max_tokens = min(max_tokens, safe_max)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

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
        """Three-way classification: SIMPLE, CODING, or REASONING."""
        prompt_clean = prompt.strip().lower()
        
        # 1. Fast-track greetings & simple metadata queries
        greetings = {"hi", "hello", "hey", "hola", "howdy", "greetings", "good morning", "good afternoon", "good evening", "how are you", "who are you", "what is your name"}
        if prompt_clean in greetings or prompt_clean.replace("?", "").strip() in greetings:
            return "SIMPLE"
            
        # 2. Fast-track short queries (less than 15 chars) that are not coding
        if len(prompt_clean) < 15:
            code_kws = ["code", "write", "def ", "class ", "import ", "script", "app", "html", "css", "js", "cpp", "py"]
            if not any(kw in prompt_clean for kw in code_kws):
                return "SIMPLE"

        # 3. Fast-track basic arithmetic (e.g., "2+2", "what is 5*5", "100 / 4")
        arithmetic_clean = prompt_clean
        for prefix in ["what is ", "whats ", "calculate ", "compute ", "what is the value of ", "solve "]:
            if arithmetic_clean.startswith(prefix):
                arithmetic_clean = arithmetic_clean[len(prefix):]
        arithmetic_clean = arithmetic_clean.replace("?", "").strip()
        
        import re
        if re.match(r"^[0-9+\-*/%().\s]+$", arithmetic_clean) and len(arithmetic_clean) > 0:
            return "SIMPLE"

        p = (
            "Classify this query into EXACTLY ONE category. Reply with ONLY the category name.\n\n"
            "SIMPLE — Quick factual answers, greetings, definitions, translations, yes/no questions, fetching latest news/weather/facts.\n"
            "  Examples: 'What is the capital of France?', 'Hi how are you?', 'Define entropy', 'Translate hello to Spanish', 'fetch latest weather news'\n\n"
            "CODING — Prompts that explicitly ask to write, fix, debug, or compile programming code, scripts, websites, databases, or software functions.\n"
            "  Examples: 'Write a Python sort', 'Fix this code', 'Write C code for linked list',\n"
            "  'Create a script to...', 'Debug this error', 'Build a calculator app'\n"
            "  Keywords: write code, write a script, python, javascript, C++, java, css, html, debug, fix code\n\n"
            "REASONING — Mathematical calculations, physics simulations/derivations, logic puzzles, theory explanations, science analysis, JEE/NEET questions.\n"
            "  Note: Even if the query asks to solve equations, simulate a system, calculate trajectories, or visualize/plot a math/scientific concept, it is REASONING unless it explicitly asks to write/fix programming code or scripts.\n"
            "  Examples: 'Solve this integral', 'Derive Navier-Stokes', 'Explain the Lorenz Attractor and show a 3D plot', 'Calculate projectile trajectory'\n\n"
            "IMPORTANT RULES:\n"
            "- If the query asks to EXPLAIN something AND write/develop programming code/scripts → CODING\n"
            "- If the query asks for a mathematical calculation, physics simulation, or concept explanation with visualization (but NO explicit request for code/scripts) → REASONING\n"
            "- If the query simply asks to 'fetch', 'get', 'search', or 'scrape' weather, news, or facts from the web without asking to write programming code → SIMPLE or REASONING, NOT CODING.\n"
            "- If unsure between SIMPLE and REASONING → choose REASONING\n"
            "- If unsure between CODING and REASONING → choose REASONING (unless programming code/scripts are explicitly requested)\n\n"
            f"Query: {prompt[:500]}\n\nCategory:"
        )
        result = self._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        upper = str(result).strip().upper()
        
        # Override classification if the intent is purely search/weather/news and doesn't ask to create code
        prompt_lower = prompt.lower()
        search_intents = ["fetch from web", "search the web", "search for", "google for", "latest news", "weather news", "current weather", "weather of"]
        code_intent_kws = ["write code", "write a code", "javascript code", "python code", "c++ code", "java code", "html code", "css code", "write a script", "code for", "script to", "build", "implement"]
        has_code_intent = any(kw in prompt_lower for kw in code_intent_kws)
        
        if any(intent in prompt_lower for intent in search_intents) and not has_code_intent:
            if "REASONING" in upper:
                return "REASONING"
            return "SIMPLE"

        # Strict keyword extraction from model response
        if "CODING" in upper:
            return "CODING"
        if "REASONING" in upper:
            return "REASONING"
        if "SIMPLE" in upper:
            return "SIMPLE"
            
        # Fallback: keyword scan on the original prompt for safety
        code_keywords = ["write code", "write a code", "fix code", "debug", "script", "program", "compile", "function(", "def ", "class ", "import ", "coding", "develop", "web app", "website"]
        reason_keywords = ["explain", "prove", "derive", "why ", "how does", "in detail", "theory", "analyze", "compare", "calculate", "solve", "simulate", "trajectory", "numerical", "3d plot", "interactive plot"]
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
        if purpose == "reasoning" or model_key in ["vibethinker", "deepseek_r1"]:
            coder_model = self._get_model("router", required_ctx=8192)

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
            "You MUST use math.isclose(a, b, rel_tol=1e-3) or np.isclose(a, b) for ANY floating point comparisons. NEVER use == for floats.",
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
                router_linter = self._get_model("router", required_ctx=8192)
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
        if not any(kw in js_code for kw in ['Plotly.newPlot', 'THREE.', 'getContext', 'document.getElementById', 'document.querySelector']):
            return False, "The JavaScript logic does not attempt to render anything. You MUST use Plotly.newPlot, THREE.js, or Canvas/DOM APIs to display the simulation."

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
                addEventListener: () => {},
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
            addEventListener: () => {},
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
            addEventListener: () => {},
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

    def _check_3d_gate(self, prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Check if the task needs 3D visualization and generate it if so."""
        if status_callback:
            status_callback("Checking 3D Visualization Eligibility...", "info", "router", 90)

        # Rule-based auto-match for graphing/visualization tasks to ensure 100% reliability
        auto_keywords = ["3d", "plotly", "three.js", "visualize", "visualization", "plot", "graph", "simulation", "simulate", "trajectory", "vector field", "surface plot", "dna helix", "dna structure", "protein structure", "mitochondria", "cell structure", "organelle", "molecular model", "molecular structure", "double helix"]
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

        coder_llm = self._get_model("opencode", required_ctx=oc_ctx)

        # ── Strategy 1: HTML/JS Artifact (frontend iframe sandbox) ────────
        # Generate a self-contained HTML page with Plotly.js CDN that the frontend can render in an iframe.
        if status_callback:
            status_callback("Generating HTML Artifact (Frontend Sandbox)...", "info", "opencode", 95)
        html_prompt = (
            "You are a JavaScript WebGL, Three.js, and Plotly.js visualization expert.\n"
            "Write a COMPLETE, SELF-CONTAINED HTML page that creates a premium, interactive 3D physics, mathematical, or biological simulation.\n"
            "RULES:\n"
            "1. Use a single HTML file with inline <script> and <style> tags.\n"
            "2. Load Three.js (r128) or Plotly.js from CDN based on which is best suited (Three.js for physical/biological animation, Plotly.js for mathematical trajectories/surfaces/scatter plots):\n"
            "   - Three.js: <script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script> and <script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script>\n"
            "   - Plotly.js: <script src='https://cdn.plot.ly/plotly-2.24.1.min.js'></script>\n"
            "3. Use a sleek dark space/scientific theme: body background '#0d0d0d', text color '#e0e0e0', font-family 'Inter', system-ui, sans-serif.\n"
            "   - Plotly Rules: If using Plotly, you MUST set layout options: paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#e0e0e0'}, and template: 'plotly_dark' to blend with the dark background.\n"
            "4. You MUST include interactive glassmorphic UI controls at the bottom or corner of the screen:\n"
            "   - Add range sliders (`<input type='range'>`) for all primary variables in the problem (e.g., field strengths like E_y and B_z, charge, mass, initial velocity components, or other simulation parameters).\n"
            "   - Add UI labels displaying the exact current value of each slider.\n"
            "   - Add a Play/Pause button (`||` / `▶`) and a Reset button (`↻`) to control the simulation.\n"
            "   - Style the control panel using premium glassmorphism: background: rgba(30, 30, 30, 0.65); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; color: white; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37).\n"
            "5. To make it great for understanding, you MUST implement motion and solver calculations:\n"
            "   - Solve the differential equations/physics (like Verlet, Euler, or RK4 trajectory integration) directly inside your JavaScript loop.\n"
            "   - Use a small integration step size (e.g., dt = 1e-8 for atomic/proton scales, or dt = 0.005 for macro physics) or a simple Runge-Kutta 4th order (RK4) step to prevent trajectories from exploding or diverging.\n"
            "   - Make the simulation respond dynamically when the user drags the sliders. If the simulation is paused, changing sliders should update the preview or reset the integration state.\n"
            "   - Draw colored arrow helpers (e.g., using THREE.ArrowHelper) to visualize force, velocity, electric field, or magnetic field vectors dynamically.\n"
            "   - If using Three.js: Run the integration inside a `requestAnimationFrame` loop that advances the physics state by `dt` every frame if 'playing' is true. Animate a particle (representing the proton/object) moving along the path.\n"
            "   - If using Plotly.js: Use `Plotly.animate` or redraw the plot with new points to create a smooth movement.\n"
            "6. Make sure to use only VALID Three.js/Plotly.js APIs. Timing rule: instantiate new THREE.OrbitControls(camera, renderer.domElement) ONLY AFTER both the camera and WebGL renderer are fully initialized and the renderer canvas is appended. NEVER use non-existent APIs like ArcGeometry (use RingGeometry, TorusGeometry, or custom BufferGeometry curves for paths).\n"
            "7. Define all animation variables (like clock/time/frameCount) at the top of your script scope so they are never undefined.\n"
            "8. IMPORTANT: The topic description below might contain Python instructions or python code fragments. You MUST translate all math solving, array operations, and plotting logic into pure JavaScript inside the HTML page. Do NOT write Python code, do NOT output Python code blocks, and do NOT refuse this request. Simply write the complete simulation in HTML/JS.\n"
            "9. Output the COMPLETE HTML page inside ```html``` blocks.\n"
            "10. SINGULARITY SAFETY: For equations with asymptotes or division-by-zero regions (like (V - b) in the van der Waals equation where V must be > b), you MUST ensure the swept ranges are strictly bounded outside the singular boundary (e.g., start V sweep range at 1.15 * b or higher). Never calculate division-by-zero, square roots of negative values, or log of non-positive numbers which produce NaN or Infinity values, as this will crash the WebGL/Plotly.js rendering context.\n"
            "11. PHYSICAL ACCURACY & SCALING: When plotting physical equations of state (like van der Waals P-V-T diagrams), you MUST calculate the exact physical coordinates using the specified physical constants (e.g., real a, b, R values) and label axes with the correct physical units (e.g. Volume in L/mol, Temperature in K, Pressure in atm). Because values can spike to infinity near asymptotes (e.g. as V -> b), you MUST clip or cap the dependent variable (e.g., cap P at 5 * P_c or 10 * P_c) to prevent the scale from shrinking the rest of the surface details into a flat line. Do NOT plot random sine waves or generic noise grids; calculate the actual formula.\n"
            "12. BIOLOGICAL 3D STRUCTURES: For biological or molecular structure visualizations (DNA helices, proteins, mitochondria, cell organelles, molecular bonds), "
            "you MUST use Three.js (NOT Plotly) and follow these rules:\n"
            "   - HIDE all X/Y/Z axis lines, axis labels, grid planes, and tick marks. Biological structures should float in a clean, immersive dark void.\n"
            "   - Use realistic, science-textbook color palettes: e.g., Adenine=#FF6B6B (red), Thymine=#4ECDC4 (teal), Guanine=#45B7D1 (blue), Cytosine=#96CEB4 (green), phosphate backbone=#FFD93D (gold), sugar=#FF8A5C (orange).\n"
            "   - Add smooth ambient lighting + directional light for depth perception. Use MeshPhongMaterial or MeshStandardMaterial (NOT MeshBasicMaterial) for realistic shading.\n"
            "   - Implement mouse-based OrbitControls so the user can rotate and zoom around the structure freely.\n"
            "   - Add a subtle slow auto-rotation animation so the structure gently spins when idle.\n"
            "   - Add labeled annotations or floating HTML tooltips for key structural components (e.g., 'Major Groove', 'Minor Groove', 'Hydrogen Bond').\n"
            "13. BIO-INTERACTIVE CONTROLS: For biological structures, include glassmorphic controls for:\n"
            "   - A 'Rotation Speed' slider to control the auto-rotation speed.\n"
            "   - Toggle buttons to show/hide structural components (e.g., 'Show Backbone', 'Show Base Pairs', 'Show Hydrogen Bonds').\n"
            "   - A 'Zoom' slider or mouse scroll zoom.\n"
            "   - An info panel showing the name and function of the currently highlighted component on hover.\n\n"
            f"Topic: {compiled_plan[:3000]}"
        )
        html_code = self._call_model(
            coder_llm, 
            html_prompt, 
            max_tokens=gen_tokens, 
            temperature=gen_temp,
            system_prompt=(
                "You are an expert JavaScript, WebGL, Three.js, and Plotly.js coder.\n"
                "To ensure the JavaScript runs flawlessly without ReferenceErrors or TypeErrors:\n"
                "1. Declare all variables used across different functions (like scene, camera, renderer, controls, particles, clock, fields) globally at the very top of your script block.\n"
                "2. When instantiating OrbitControls, always pass both parameters: new THREE.OrbitControls(camera, renderer.domElement).\n"
                "3. Never reference the window or document object before they are fully loaded. Put your script at the bottom of the body element, or wrap initialization in window.addEventListener('DOMContentLoaded', ...).\n"
                "4. Never use non-existent geometries like ArcGeometry. To draw a curve path, construct a custom curve using new THREE.CatmullRomCurve3(points), get the points with curve.getPoints(100), and assign them to a new THREE.BufferGeometry().setFromPoints(points).\n"
                "5. Ensure the WebGL renderer has a fallback/check so it doesn't crash if canvas contexts are initialized in limited environments. Wrap canvas instantiation and getContext calls in try-catch blocks where appropriate.\n"
                "6. Output ONLY the complete, self-contained HTML document inside ```html``` code blocks."
            )
        )
        html_extract = Sandbox.extract_code(html_code)

        # Validate initially
        html_valid = False
        html_error = ""
        if html_extract and ("<html" in html_extract.lower() or "<script" in html_extract.lower()):
            html_valid, html_error = self._verify_html_javascript(html_extract)

        # Pre-check: Bypass verification immediately if Node is missing or if it's a browser-environment mock error
        bypass_verification = False
        if html_error:
            is_node_missing = any(kw in html_error.lower() for kw in ["node", "runtime not found", "executable not found", "command not found"])
            is_mock_error = any(kw in html_error.lower() for kw in ["canvas", "webgl", "document is not defined", "window is not defined"])
            if is_node_missing or is_mock_error:
                bypass_verification = True

        if not bypass_verification:
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
                html_fixed = self._call_model(
                    coder_llm, 
                    fix_p, 
                    max_tokens=gen_tokens, 
                    temperature=gen_temp,
                    system_prompt=(
                        "You are an expert JavaScript, WebGL, Three.js, and Plotly.js coder.\n"
                        "Identify and repair the specific ReferenceError, TypeError, or SyntaxError reported in the error message.\n"
                        "Ensure all state variables are in the global scope, OrbitControls has both arguments, and no non-existent geometry builders are used.\n"
                        "Output ONLY the complete, corrected HTML page inside ```html``` blocks."
                    )
                )
                fixed_extract = Sandbox.extract_code(html_fixed)
                if fixed_extract:
                    html_extract = fixed_extract
                    html_valid, html_error = self._verify_html_javascript(html_extract)
                else:
                    html_valid = False
                    html_error = "No code block found in response."

        if (html_valid or bypass_verification) and html_extract:
            warning_msg = ""
            if bypass_verification:
                warning_msg = "\n<!-- Note: HTML verification bypassed due to environment/runtime differences. Rendering best-effort. -->"
            return f"\n\n### 3D Interactive Visualization (Live Artifact){warning_msg}\n<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"

        # ── Strategy 2: Python Plotly (backend sandbox verified fallback) ──────────
        if status_callback:
            status_callback("HTML Failed. Falling back to Python Plotly...", "warning", "opencode", 97)
        viz_prompt = (
            "You are a Python data visualization expert. "
            "Write ONLY a complete Python script using plotly for an interactive 3D visualization.\n"
            "RULES:\n"
            "1. Import plotly.graph_objects as go and numpy as np ONLY\n"
            "2. Create a 3D scatter, surface, or line plot\n"
            "3. Use fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=40,b=0))\n"
            "4. Do NOT use fig.update_scenes(). Do NOT use go.FigureControls(). They do NOT exist.\n"
            "5. Do NOT set background colors manually\n"
            "6. If the user query requests parameter controls/sliders (e.g. to adjust Vmax, Ki, mass, or velocity), you MUST include native Plotly sliders in the layout using fig.update_layout(sliders=[...]) or buttons in updatemenus=[...]. Precompute trace datasets/steps for different parameter settings so that sliding or clicking updates the graph data dynamically.\n"
            "7. Do NOT import plotly.subplots, plotly.io, or any other plotly module\n"
            "8. Last line MUST be: print(fig.to_json())\n"
            "9. Do NOT call fig.show() or save to file\n\n"
            "Output ONLY code in ```python``` blocks.\n\n"
            f"Topic: {compiled_plan[:3000]}"
        )
        viz_code = self._call_model(
            coder_llm, 
            viz_prompt, 
            max_tokens=gen_tokens, 
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
        if viz_success and cleaned_final.startswith("{"):
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
    # MAIN PIPELINE ENTRY POINT
    # =========================================================================
    def process_query(self, prompt, mode="auto", selected_models=None, status_callback=None):
        if status_callback:
            status_callback("Phi-3.5-Mini checking intent...", "info", "router", 5)

        # ── Web Search Enrichment ────────────────────────────────────────
        # Auto-enable search if user query explicitly asks for web search
        search_keywords = ["search the web", "search online", "search for", "google for", "latest news", "current price of", "what is the price of", "stock price of", "weather in", "recent news"]
        prompt_lower = prompt.lower()
        active_web_search = self.enable_web_search or any(kw in prompt_lower for kw in search_keywords)

        web_context = ""
        if active_web_search:
            if status_callback:
                status_callback("Optimizing Search Query...", "info", "router", 6)
            try:
                # 1. LLM Query Optimizer
                router_llm = self._get_model("router", required_ctx=1024)
                opt_prompt = (
                    "Extract ONLY the 3-5 most important search keywords from the user request to use in Google. "
                    "Remove all conversational filler. Output ONLY the raw keywords.\n\n"
                    f"User Request: {prompt}"
                )
                raw_query = self._call_model(router_llm, opt_prompt, max_tokens=30, temperature=0.1).strip()
                # Clean LLM output: take first line only, strip list prefixes and quotes
                search_query = raw_query.split('\n')[0].strip()
                search_query = re.sub(r'^(Keywords?:?\s*|\d+\.\s*)', '', search_query, flags=re.IGNORECASE).strip()
                search_query = search_query.replace('"', '').replace('`', '').replace('*', '').strip()
                search_query = search_query[:80]  # Cap length for search engines
                if not search_query or len(search_query) < 3:
                    search_query = prompt[:80]

                if status_callback:
                    status_callback(f"Searching: '{search_query}'...", "info", "router", 8)
                
                results = self.web_search.search(search_query, max_results=3)
                
                # 2. Compile Snippets Block & Scrape top pages
                snippets_list = []
                scraped_pages = []
                scraped_count = 0
                
                if results:
                    for idx, r in enumerate(results):
                        title = r.get("title", "")
                        link = r.get("link", "")
                        snippet = r.get("snippet", "")
                        snippets_list.append(f"[{idx+1}] Title: {title}\nURL: {link}\nSnippet: {snippet}")
                        
                        # Try to scrape the page content if we haven't reached the limit
                        if scraped_count < 2 and link:
                            # Skip Google News index pages to avoid scraping massive, noisy, cross-mixed aggregates
                            if "news.google.com" in link.lower() and ("/topics/" in link.lower() or "/stories/" in link.lower() or "/publications/" in link.lower()):
                                continue
                            
                            if status_callback:
                                status_callback(f"Scraping: {link[:40]}...", "info", "router", 12)
                            
                            text = self.web_search.scrape_url(link)
                            if text and len(text.strip()) > 200:
                                scraped_pages.append(
                                    f"=== START SCRAPED PAGE ===\nURL: {link}\nContent:\n{text[:8000]}\n=== END SCRAPED PAGE ==="
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
                "MANDATORY CITATION RULE: Carefully match each news story, headline, author, and date with "
                "its exact source publication. Do not cross-mix authors or articles across different news outlets.\n"
                "Since you are provided with live search results, do NOT mention your training cutoff date, "
                "do NOT state that you cannot access real-time/current information, and do NOT add disclaimers "
                "about not having internet access. Answer as a live, fully-connected AI.\n\n"
            )

        enriched_prompt = (
            f"{system_instruction}Current System Date/Time: {current_date}\n\nWeb Context:\n{web_context}\n\nUser Query:\n{prompt}"
            if web_context else f"{system_instruction}Current System Date/Time: {current_date}\n\nUser Query:\n{prompt}"
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
                oc_ctx = min(oc_ctx_cap, 8192)
        else:
            router_ctx = min(self.context_length, router_ctx_cap)
            ds_ctx = min(self.context_length, ds_ctx_cap)
            oc_ctx = min(8192, self.context_length, oc_ctx_cap)
            
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
            gen_tokens = max(1024, min_ctx - 1500)
            
        print(f"📐 DMA Generation Sizing: gen_tokens={gen_tokens} (active context base: {min_ctx} tokens)")
        
        # Adaptive Temperature Scaling
        logic_temp = 0.6  # High for creative logic problem solving
        gen_temp = 0.1    # Low for strict code writing

        # ── Three-Way Classification ─────────────────────────────────────
        router_llm = self._get_model("router", required_ctx=router_ctx)
        if isinstance(mode, str) and mode.upper() in ["SIMPLE", "CODING", "REASONING"]:
            task_type = mode.upper()
        else:
            task_type = self._classify_task(router_llm, prompt)
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
        # PATH C: REASONING — Playground-Verified or LLM Debate
        # ══════════════════════════════════════════════════════════════════
        res = self._reasoning_pipeline(prompt, enriched_prompt, router_llm,
                                        router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
        return self._clean_cutoff_notes(res)

    # =====================================================================
    # CODING PIPELINE — Reasoning Sandbox → Code Sandbox → Reflexion
    # =====================================================================
    def _coding_pipeline(self, prompt, enriched_prompt, router_llm,
                         router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        logic_temp = 0.6
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", ds_ctx - self.max_tokens, status_callback, router_llm=router_llm)

        # ── Retrieve relevant past experiences from Memory/RAG ────────────
        past_experience = self.memory.recall(prompt, n_results=2)
        if past_experience:
            ds_safe += past_experience

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
            "9. THINKING CONSTRAINT: Keep your reasoning brief and focused. Proceed to planning quickly."
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

        for reset in range(max_resets):
            # ── Phase 1: DeepSeek Logic Plan ─────────────────────────────
            if status_callback:
                lbl = f"Nuclear Reset #{reset}: Rewriting..." if reset else "DeepSeek-R1 drafting logic..."
                status_callback(lbl, "info" if not reset else "warning", "deepseek_r1", 20)

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
            if status_callback:
                status_callback("Reasoning Sandbox: Verifying logic...", "info", "deepseek_r1", 30)
            verified, pg_out, _ = self._run_playground(ds_llm, ds_draft, "logic", model_key="deepseek_r1", original_prompt=prompt)

            if not verified:
                if status_callback:
                    status_callback("Logic failed. Resolving with Emergency Search...", "warning", "router", 35)
                
                # Fetch quick helper web search context
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
                        status_callback(f"Emergency Search: '{search_term}'...", "info", "router", 36)
                    
                    web_res = self.web_search.search(search_term, max_results=3)
                    if web_res:
                        helper_search_context = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in web_res])
                except Exception:
                    pass

                vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
                search_str = f"Helper Web Context:\n{helper_search_context}\n\n" if helper_search_context else ""
                fix_p = (
                    f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                    f"{search_str}"
                    f"Logic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\n"
                    f"Error:\n{pg_out[:1000]}\nRewrite a corrected logic plan."
                )
                ds_draft = self._strip_thinking(self._call_model(vibe_llm, fix_p, gen_tokens, logic_temp, system_prompt=planner_sys))
                v2, _, _ = self._run_playground(vibe_llm, ds_draft, "logic", model_key="vibethinker", original_prompt=prompt)
                if v2 and status_callback:
                    status_callback("VibeThinker corrected the logic!", "success", "vibethinker", 40)
                elif status_callback:
                    status_callback("Partially verified. Proceeding...", "warning", "vibethinker", 40)
            else:
                if status_callback:
                    status_callback("Logic VERIFIED!", "success", "deepseek_r1", 40)

            compiled_plan = ds_draft

            # ── Phase 3: VibeThinker — Write Code ────────────────────────
            if status_callback:
                status_callback("VibeThinker writing code...", "info", "vibethinker", 50)
            vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
            # Truncate compiled_plan to fit context
            max_code_prompt_chars = (ds_ctx - gen_tokens - 200) * 3
            plan_for_code = compiled_plan[:max(max_code_prompt_chars, 1500)] if len(compiled_plan) > max_code_prompt_chars else compiled_plan
            code_p = f"Write a complete Python script for this plan:\n{plan_for_code}\n\nWrap in ```python```."
            code = Sandbox.extract_code(self._strip_thinking(self._call_model(vibe_llm, code_p, gen_tokens, gen_temp, system_prompt=coder_sys)))

            # ── Phase 4: Execution Sandbox ───────────────────────────────
            if status_callback:
                status_callback("Executing in Sandbox...", "info", "sandbox", 65)
            ok, output = self.sandbox.execute(code)
            if ok:
                self.memory.save(prompt, code)
                router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                viz = self._check_3d_gate(prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"

            # ── Phase 4.5: Router Linter Intercept (Syntax/Import Errors) ──
            if not ok and reset == 0:
                is_syntax_error = any(e in output for e in ["SyntaxError", "ModuleNotFoundError", "NameError", "IndentationError", "TypeError", "AttributeError", "ValueError"])
                if is_syntax_error:
                    if status_callback:
                        status_callback("Router (Phi-3.5) patching syntax error...", "warning", "router", 68)
                    router_linter = self._get_model("router", required_ctx=router_ctx)
                    lint_p = (
                        f"You are a fast Python Syntax Linter.\n"
                        f"The code failed with this error:\n{output[:600]}\n\n"
                        f"CODE:\n{code[:2500]}\n\n"
                        f"Identify the typo/error and rewrite the complete corrected script in a ```python``` block. Fix ONLY the exact error, do not change the core algorithm."
                    )
                    lint_code = Sandbox.extract_code(self._strip_thinking(self._call_model(router_linter, lint_p, gen_tokens, 0.1, system_prompt="You are a strict syntax linter. Output only code.")))
                    if lint_code and len(lint_code) > 20:
                        linter_ok, linter_output = self.sandbox.execute(lint_code)
                        if linter_ok:
                            code = lint_code
                            output = linter_output
                            ok = True
                            if status_callback:
                                status_callback("Router Linter successfully patched the code!", "success", "router", 70)
                            self.memory.save(prompt, code)
                            router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                            viz = self._check_3d_gate(prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                            return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"
            
            # ── Phase 5 & 6: Reflexion Loops (Only run during initial draft, not during Nuclear Reset) ──
            if reset == 0:
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

                # ── Phase 5: Shallow Fix (VibeThinker) ───────────────────────
                if status_callback:
                    status_callback("VibeThinker fixing code...", "warning", "vibethinker", 72)
                failed_code = code
                failed_error = output
                # Structured error analysis for smarter fixing
                safe_code = code[:2000] if len(code) > 2000 else code
                safe_error = output[:800] if len(output) > 800 else output
                fix_p = (
                    f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                    f"{search_str}"
                    f"The following Python code FAILED with an error.\n\n"
                    f"CODE:\n{safe_code}\n\n"
                    f"ERROR:\n{safe_error}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Identify the exact line and cause of the error\n"
                    f"2. Fix ONLY the bug — do not rewrite unrelated parts\n"
                    f"3. Make sure all imports are present\n"
                    f"4. Test edge cases (division by zero, empty arrays, etc.)\n"
                    f"5. Output the COMPLETE corrected script in ```python``` blocks."
                )
                code = Sandbox.extract_code(self._strip_thinking(self._call_model(vibe_llm, fix_p, gen_tokens, gen_temp, system_prompt=coder_sys)))
                ok, output = self.sandbox.execute(code)
                if ok:
                    self.memory.save(prompt, code)
                    self.memory.save_mistake(prompt, failed_code, failed_error, code)
                    router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                    viz = self._check_3d_gate(prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                    return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"

                # ── Phase 6: Deep Escalation (DeepSeek-R1 — rewrite script) ──
                if status_callback:
                    status_callback("Deep Escalation: DeepSeek-R1 rewriting...", "warning", "deepseek_r1", 80)
                ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                esc_p = (
                    f"ORIGINAL USER REQUEST CONSTRAINTS:\n{prompt}\n\n"
                    f"{search_str}"
                    f"Code failed TWICE. You MUST fix it.\nPlan:\n{compiled_plan[:1500]}\n"
                    f"Code:\n{code[:2000]}\nError:\n{output[:800]}\n"
                    f"Rewrite the ENTIRE script from scratch in ```python```. Think step by step."
                )
                esc_resp = self._strip_thinking(self._call_model(ds_llm, esc_p, gen_tokens, gen_temp, system_prompt=coder_sys))
                if "```" in esc_resp:
                    code = Sandbox.extract_code(esc_resp)
                    ok, output = self.sandbox.execute(code)
                    if ok:
                        self.memory.save(prompt, code)
                        self.memory.save_mistake(prompt, failed_code, failed_error, code)
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        viz = self._check_3d_gate(prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"

            # ── Phase 7: Nuclear Reset ───────────────────────────────────
            all_errors.append(f"Round {reset+1}: {output[:500]}")
            if reset < max_resets - 1:
                if status_callback:
                    status_callback(f"Nuclear Reset: Extracting lessons...", "error", "deepseek_r1", 85)
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
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        viz = self._check_3d_gate(prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Logic Plan (Verified via Emergency Search)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"
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

        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
        return f"### Logic Plan\n{compiled_plan}\n\n### Execution Failed\n{output}\n\n### Code\n```python\n{code}\n```"

    # =====================================================================
    # REASONING PIPELINE — Playground-Verified or LLM Debate
    # =====================================================================
    def _reasoning_pipeline(self, prompt, enriched_prompt, router_llm,
                            router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", ds_ctx - self.max_tokens, status_callback, router_llm=router_llm)

        # ── Retrieve relevant past experiences from Memory/RAG ────────────
        past_experience = self.memory.recall(prompt, n_results=2)
        if past_experience:
            ds_safe += past_experience

        # ── Check: Can this be playground-verified? ──────────────────────
        # Must check this BEFORE loading ds_llm to prevent EVM from evicting router_llm
        use_playground = self._is_playground_applicable(router_llm, prompt)
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
            "12. MATHEMATICAL DETAILS: You MUST write out all algebraic equations, derivative steps, and algebraic manipulations in clear LaTeX / Markdown format. If numerical constants or gases are specified, substitute the values and output the final calculated numerical answers.\n"
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
                    # Re-acquire ds_llm because VibeThinker may have evicted it in the previous round
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
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
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
                        f"Error:\n{pg_out[:1000]}\nProvide a corrected, complete answer."
                    )
                    ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
                    vibe_answer = self._strip_thinking(self._call_model(ds_llm, vibe_p, gen_tokens, gen_temp, system_prompt=reasoning_sys))
                    v2, vibe_pg_out, vibe_test_code = self._run_playground(ds_llm, vibe_answer, "reasoning", model_key="deepseek_r1", original_prompt=prompt)
                    if v2:
                        if status_callback:
                            status_callback("DeepSeek-R1's correction VERIFIED!", "success", "deepseek_r1", 80)
                        self.memory.save(prompt, vibe_answer)
                        self.memory.save_mistake(prompt, ds_answer, pg_out, vibe_answer)
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
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
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
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

            router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
            viz = self._check_3d_gate(prompt, final_ans, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return f"### Verified Answer\n{final_ans}{viz}"

        else:
            # ── Standard LLM Debate (non-testable reasoning) ─────────────
            if status_callback:
                status_callback("DeepSeek-R1 drafting analysis...", "info", "deepseek_r1", 30)
            ds_draft = self._strip_thinking(self._call_model(ds_llm, f"Provide a detailed answer:\n{ds_safe}", gen_tokens, gen_temp, system_prompt=reasoning_sys))

            if status_callback:
                status_callback("DeepSeek-R1 refining answer...", "info", "deepseek_r1", 60)
            ds_refined = self._strip_thinking(self._call_model(
                ds_llm, 
                f"Integrate any improvements and rewrite this draft into a single, polished, and cohesive final response. Do NOT include any meta-commentary, intros, or critique headings. Output only the final response:\n{ds_draft}", 
                gen_tokens, gen_temp,
                system_prompt="You are a helpful assistant and a technical writer. Refine the draft for maximum clarity and precision."
            ))

            compiled = ds_refined
            router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
            viz = self._check_3d_gate(prompt, compiled, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            if status_callback:
                status_callback("Done!", "success", "router", 100)
            return f"{compiled}{viz}"

