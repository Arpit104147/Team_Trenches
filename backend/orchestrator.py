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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()


class AgentOrchestrator:
    def __init__(self, cancel_event=None):
        self.cancel_event = cancel_event  # threading.Event for cancel support
        self.context_length = 8192
        self.max_tokens = 2048
        self.temperature = 0.7
        self.device_mode = "gpu"
        self.gpu_layers = -1
        self.enable_web_search = False
        
        self.sandbox = Sandbox(timeout=60)
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
        if torch and torch.cuda.is_available():
            try:
                _free, total_vram = torch.cuda.mem_get_info(0)
                total_vram_gb = total_vram / (1024 ** 3)
                self.vram_safety_gb = round(total_vram_gb * 0.40, 1)  # 40% reserve
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
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
        if torch and torch.cuda.is_available():
            for gpu_idx in range(torch.cuda.device_count()):
                vram_free = self._get_vram_free_gb(gpu_idx)
                if vram_free is None:
                    continue
                # Use the larger of: static safety threshold OR (incoming model + 3GB buffer)
                effective_threshold = self.vram_safety_gb
                if required_vram_gb is not None:
                    effective_threshold = max(self.vram_safety_gb, required_vram_gb + 3.0)
                if vram_free < effective_threshold:
                    print(f"⚠️ DMA: CUDA GPU:{gpu_idx} VRAM low ({vram_free:.1f} GB free, need {effective_threshold:.1f} GB)")
                    while vram_free < effective_threshold and self.model_access_order:
                        if not self._evict_lru_model():
                            break
                        evicted_any = True
                        vram_free = self._get_vram_free_gb(gpu_idx)

        # ── Linux sysfs VRAM Check (AMD/Intel Vulkan — no ROCm needed) ──
        sysfs_gpus = self._get_sysfs_gpu_vram()
        for free_gb, total_gb, card_name in sysfs_gpus:
            effective_threshold = self.vram_safety_gb
            if required_vram_gb is not None:
                effective_threshold = max(self.vram_safety_gb, required_vram_gb + 3.0)
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

    def _get_model(self, model_key, required_ctx=None):
        """Load a model with Dynamic Memory Allocator protection and dynamic context sizing."""
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
                gc.collect()
                import time
                time.sleep(2)
                if torch and hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            else:
                self._touch_model(model_key)
                return model_obj

        # DMA Check: evict if memory is low BEFORE attempting a new load
        # Pass the estimated model size so the DMA evicts enough room for THIS specific model
        est_model_gb = self._estimate_model_size_gb(model_key)
        print(f"📦 DMA: Preparing to load '{model_key}' (~{est_model_gb:.1f} GB)")
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

            llm = Llama(**kwargs)
            self.loaded_models[model_key] = llm
            self._touch_model(model_key)
            print(f"✅ Loaded GGUF model '{model_key}' ({os.path.basename(model_path)})")
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
            
        # Handle unclosed <think> tag gracefully without wiping code
        if '<think>' in text and '</think>' not in text:
            before_think, after_think = text.split('<think>', 1)
            if '```' in after_think:
                # Close the think block right before the first code fence
                after_think = after_think.replace('```', '</think>\n```', 1)
                text = before_think + '<think>' + after_think
            else:
                return (before_think + after_think).strip()
                
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        if cleaned:
            return cleaned
            
        # If stripping removed EVERYTHING, it means the model put its entire answer 
        # inside the think block. We must strip ONLY the tags, keeping the content!
        text_without_tags = re.sub(r'</?think>', '', text)
        return text_without_tags.strip()


    def _crunch_prompt(self, prompt, target_model, max_tokens_limit, status_callback=None, router_llm=None):
        """Compresses a massive prompt safely using the fast Router model."""
        # Guard: if ctx is smaller than generation headroom, don't try to compress
        max_tokens_limit = max(512, max_tokens_limit)
        est_tokens = len(prompt) // 4
        if est_tokens <= max_tokens_limit:
            return prompt
            
        if status_callback:
            status_callback(f"Prompt Cruncher active for {target_model} ({est_tokens} tokens > {max_tokens_limit} max). Compressing...", "warning", target_model, 12)
            
        chars_allowed = max_tokens_limit * 4
        chunk_size = chars_allowed // 3
        
        start_chunk = prompt[:chunk_size]
        middle_chunk = prompt[chunk_size:-chunk_size]
        end_chunk = prompt[-chunk_size:]
        
        # Safety net: Don't spend hours summarizing a 500MB file
        if len(middle_chunk) > 50000:
            middle_chunk = middle_chunk[:25000] + "\n...[SNIPPED EXTREME LENGTH]...\n" + middle_chunk[-25000:]
            
        compress_prompt = f"Summarize this middle section concisely. Keep all logic, facts, and code structure intact:\n{middle_chunk}"
        
        # Reuse pre-loaded router if available, otherwise load it
        if router_llm is None:
            router_llm = self._get_model("router", required_ctx=8192)
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
            # Chat template overhead: role markers, BOS/EOS tokens (~100 tokens)
            template_overhead = 100
            # Estimate prompt tokens: ~3.5 chars per token (conservative)
            est_prompt_tokens = len(prompt) // 3 + template_overhead
            if system_prompt:
                est_prompt_tokens += len(system_prompt) // 3
            
            # If prompt alone would overflow, truncate it to fit
            max_prompt_chars = (ctx - max_tokens - template_overhead) * 3
            if system_prompt:
                max_prompt_chars -= len(system_prompt)
            if max_prompt_chars < 300:
                max_prompt_chars = 300
            if len(prompt) > max_prompt_chars:
                # Keep beginning and end of prompt (most important parts)
                keep_start = int(max_prompt_chars * 0.2)
                keep_end = int(max_prompt_chars * 0.8)
                prompt = prompt[:keep_start] + "\n...[TRUNCATED FOR CONTEXT LIMIT]...\n" + prompt[-keep_end:]
                est_prompt_tokens = len(prompt) // 3 + template_overhead
                if system_prompt:
                    est_prompt_tokens += len(system_prompt) // 3
            
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
        p = (
            "Classify this query into EXACTLY ONE category. Reply with ONLY the category name.\n\n"
            "SIMPLE — Quick factual answers, greetings, definitions, translations, yes/no questions, fetching latest news/weather/facts.\n"
            "  Examples: 'What is the capital of France?', 'Hi how are you?', 'Define entropy', 'Translate hello to Spanish', 'fetch latest weather news'\n\n"
            "CODING — Anything that needs writing, fixing, debugging, or executing code in ANY language.\n"
            "  Examples: 'Write a Python sort', 'Fix this code', 'Write C code for linked list',\n"
            "  'Create a script to...', 'Debug this error', 'Build a calculator', 'Implement binary search'\n"
            "  Keywords: write, code, script, program, implement, debug, fix, compile, function, algorithm, API\n\n"
            "REASONING — Deep explanations, math proofs, physics theory, science analysis, logic puzzles,\n"
            "  comparisons, detailed breakdowns, JEE/NEET level problems, step-by-step derivations.\n"
            "  Examples: 'Explain Newton\'s laws in detail', 'Prove Pythagorean theorem',\n"
            "  'Why is the sky blue?', 'Compare TCP vs UDP in depth', 'Solve this integral'\n"
            "  Keywords: explain, prove, derive, analyze, compare, why, how does, in detail, theory\n\n"
            "IMPORTANT RULES:\n"
            "- If the query asks to EXPLAIN something AND write code → CODING\n"
            "- If the query asks for detailed explanation with visualization → CODING\n"
            "- If the query simply asks to 'fetch', 'get', 'search', or 'scrape' weather, news, or facts from the web without asking to write programming code → SIMPLE or REASONING, NOT CODING.\n"
            "- If unsure between SIMPLE and REASONING → choose REASONING\n"
            "- If unsure between CODING and REASONING → choose CODING\n\n"
            f"Query: {prompt[:500]}\n\nCategory:"
        )
        result = self._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        upper = str(result).strip().upper()
        
        # Override classification if the intent is purely search/weather/news and doesn't ask to create code
        prompt_lower = prompt.lower()
        search_intents = ["fetch from web", "search the web", "search for", "google for", "latest news", "weather news", "current weather", "weather of"]
        code_intent_kws = ["write code", "write a code", "javascript code", "python code", "c++ code", "java code", "html code", "css code", "write a script", "code for", "script to", "simulate", "build", "implement"]
        has_code_intent = any(kw in prompt_lower for kw in code_intent_kws)
        
        if any(intent in prompt_lower for intent in search_intents) and not has_code_intent:
            if "REASONING" in upper:
                return "REASONING"
            return "SIMPLE"

        # Override: computational/simulation prompts that need code execution
        # These are often misclassified as REASONING but require code to produce results
        compute_keywords = [
            "calculate", "simulate", "trajectory", "numerical", "rk4", "runge-kutta",
            "euler method", "integration", "ode", "differential equation", "3d plot",
            "interactive plot", "render", "visualization", "animation", "n-body",
            "pendulum", "projectile", "orbit", "monte carlo", "finite element",
            "neural network", "train a model", "gradient descent", "backpropagation",
            "fourier transform", "fft", "wavelet", "signal processing",
            "3d interactive", "figure-eight", "three-body", "chaotic"
        ]
        
        # Don't override if the user explicitly wants theoretical reasoning
        theory_keywords = ["explain", "derive", "why", "concept", "theory", "theoretically", "describe", "prove"]
        wants_theory = any(kw in prompt_lower for kw in theory_keywords)
        
        # Strict compute override: Bypass theory keywords if an action keyword is present
        action_keywords = ["calculate", "compute", "simulate", "model", "solve", "render", "plot", "show", "verify computationally", "build"]
        has_action = any(kw in prompt_lower for kw in action_keywords)
        
        if any(kw in prompt_lower for kw in compute_keywords) and "CODING" not in upper:
            if not wants_theory or has_action:
                return "CODING"

        # Strict keyword extraction from model response
        if "CODING" in upper:
            return "CODING"
        if "REASONING" in upper:
            return "REASONING"
        if "SIMPLE" in upper:
            return "SIMPLE"
            
        # Fallback: keyword scan on the original prompt for safety
        code_keywords = ["write code", "write a code", "fix code", "debug", "script", "implement", "program", "compile", "function(", "def ", "class ", "import ", "calculate", "simulate", "trajectory", "numerical", "3d plot"]
        reason_keywords = ["explain", "prove", "derive", "why ", "how does", "in detail", "theory", "analyze", "compare"]
        if any(kw in prompt_lower for kw in code_keywords):
            return "CODING"
        if any(kw in prompt_lower for kw in reason_keywords):
            return "REASONING"
        return "SIMPLE"

    def _is_playground_applicable(self, router_llm, prompt):
        """Check if reasoning can be verified via Python sandbox."""
        p = (
            "Can this concept be verified using a Python script? "
            "Math/physics equations, logic puzzles, statistical claims = YES. "
            "Philosophy, ethics, creative writing, history, opinions = NO.\n"
            "Reply ONLY 'YES' or 'NO'.\n\n"
            f"Query: {prompt[:500]}"
        )
        result = self._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        return "YES" in str(result).upper()

    def _run_playground(self, model, hypothesis, purpose="logic", status_callback=None):
        """
        Have a model write a verification script and run it in the sandbox.
        Returns (verified: bool, output: str, test_code: str)
        """
        playground_prompt = (
            f"Write a Python script (max 50 lines) that {'tests this code logic' if purpose == 'logic' else 'verifies this reasoning'}.\n\n"
            "VERIFICATION RULES:\n"
            "1. Test the CORE claim/formula with concrete numerical values\n"
            "2. You MUST use math.isclose(a, b, rel_tol=1e-3) for ANY floating point comparisons. NEVER use == for floats.\n"
            "3. Check at least 2 different test cases or boundary conditions\n"
            "4. Check dimensional consistency (units make sense)\n"
            "5. Use assert statements with descriptive messages\n"
            "6. Print 'VERIFIED' as the LAST line ONLY if ALL assertions pass\n"
            "7. Do NOT print 'VERIFIED' if any assertion fails\n\n"
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
            f"To verify:\n{hypothesis[:2000]}"
        )
        test_response = self._call_model(model, playground_prompt, max_tokens=1024, temperature=0.1)
        test_code = Sandbox.extract_code(test_response)
        success, output = self.sandbox.execute(test_code)
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

        # Find all script blocks in HTML
        scripts = re.findall(r'<script\b[^>]*>([\s\S]*?)</script>', html_code)
        if not scripts:
            return True, ""
            
        js_code = "\n".join(scripts)
        if not js_code.strip():
            return True, ""

        # Prepend mocks for DOM, Window, THREE, and Plotly to Node.js context.
        # This will bypass typical browser-only ReferenceErrors while letting actual syntax/API bugs throw errors.
        mocks = """
        // Mock DOM & Window
        global.document = {
            getElementById: () => ({ 
                addEventListener: () => {},
                appendChild: () => {},
                style: {},
                getContext: () => ({
                    createShader: () => ({}),
                    compileShader: () => ({}),
                    createProgram: () => ({}),
                    attachShader: () => ({}),
                    linkProgram: () => ({}),
                    getProgramParameter: () => true,
                    useProgram: () => ({}),
                    createBuffer: () => ({}),
                    bindBuffer: () => ({}),
                    bufferData: () => ({}),
                    enableVertexAttribArray: () => ({}),
                    vertexAttribPointer: () => ({}),
                    drawArrays: () => ({}),
                })
            }),
            createElement: () => ({ style: {}, getContext: () => ({}) }),
            body: { appendChild: () => {}, style: {} },
            addEventListener: () => {}
        };
        global.window = {
            innerWidth: 1024,
            innerHeight: 768,
            addEventListener: () => {},
            requestAnimationFrame: (cb) => {
                // Run animation loop exactly ONCE to verify there are no runtime ReferenceErrors
                if (!global.__ran_animation_loop) {
                    global.__ran_animation_loop = true;
                    try { cb(0); } catch(e) {}
                }
            },
            document: global.document,
            location: { href: "" }
        };
        global.navigator = { userAgent: "" };
        
        // Mock THREE.js via a Universal Proxy to catch missing API constructors or undefined variables
        const createProxy = (name) => {
            const mockFn = function() {};
            
            // Common nested properties to avoid undefined errors during instantiation/method chains
            mockFn.position = { x: 0, y: 0, z: 0, set: () => {}, copy: () => {} };
            mockFn.rotation = { x: 0, y: 0, z: 0, set: () => {} };
            mockFn.scale = { x: 1, y: 1, z: 1, set: () => {} };
            mockFn.shadowMap = { enabled: false, type: 0 };
            mockFn.color = { set: () => {}, setHex: () => {}, setRGB: () => {} };
            mockFn.domElement = { addEventListener: () => {}, removeEventListener: () => {}, style: {} };
            mockFn.add = () => {};
            mockFn.render = () => {};
            mockFn.setSize = () => {};
            mockFn.setPixelRatio = () => {};
            mockFn.update = () => {};
            mockFn.lookAt = () => {};
            mockFn.set = () => {};
            mockFn.clone = () => mockFn;
            
            return new Proxy(mockFn, {
                construct(target, args) {
                    return createProxy(name);
                },
                get(target, prop) {
                    // Specific mock for ArcGeometry to fail verification
                    if (prop === 'ArcGeometry') {
                        return undefined; // Will throw TypeError
                    }
                    if (prop in target) {
                        return target[prop];
                    }
                    // Return a new proxy for un-mocked nested properties
                    return createProxy(`${name}.${prop}`);
                },
                set(target, prop, value) {
                    target[prop] = value;
                    return true;
                }
            });
        };
        global.THREE = createProxy('THREE');
        global.Plotly = createProxy('Plotly');
        
        // Safe console mock
        global.console = {
            log: () => {},
            error: () => {},
            warn: () => {}
        };
        """
        full_test_code = mocks + "\n" + js_code
        success, output = self.sandbox.execute(full_test_code, language='javascript')
        return success, output

    def _check_3d_gate(self, prompt, compiled_plan, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        """Check if the task needs 3D visualization and generate it if so."""
        if status_callback:
            status_callback("Checking 3D Visualization Eligibility...", "info", "router", 90)
        gate_prompt = (
            "Does this task involve mathematical graphing, data plotting, 3D matrices, physics/chemical equations, "
            "or biological/molecular 3D models (like DNA helices, cellular structures, proteins, or chemical bonds)? "
            "Game development (pygame, tkinter, GUI apps) does NOT count. "
            "Reply ONLY 'YES' or 'NO'.\n\n"
            f"Query: {prompt[:500]}"
        )
        router_llm = self._get_model("router", required_ctx=router_ctx)
        is_3d = self._call_model(router_llm, gate_prompt, max_tokens=10, temperature=0.1)

        if "YES" not in str(is_3d).upper():
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
            "3. Use a sleek dark space/scientific theme: body background '#0d0d0d', text color '#e0e0e0'.\n"
            "4. Include interactive glassmorphic UI controls (like range sliders to change mass ratio, drag coefficient, launch velocity/angle, or biological/chemical parameters) at the bottom or corner with CSS: background: rgba(30, 30, 30, 0.65), backdrop-filter: blur(10px), border: 1px solid rgba(255,255,255,0.1), padding: 15px, border-radius: 10px, color: white.\n"
            "5. To make it great for understanding, you MUST:\n"
            "   - For Physics/Trajectories: Solve the physics equations (like RK4 trajectory calculation) directly in JavaScript and plot/animate them in 3D dynamically.\n"
            "   - For Biological/Molecular systems: Render organic curves (DNA Double Helices via custom THREE.CatmullRomCurve3), membrane channels/lipid bilayers (spheres & cylinders), or cellular transport particles (THREE.Points particle systems).\n"
            "   - Display real-time data labels (e.g. current positions, velocities, concentrations, or pH) updating on-screen.\n"
            "6. Make sure to use only VALID Three.js/Plotly.js APIs: NEVER use non-existent APIs like ArcGeometry (use RingGeometry, TorusGeometry, or custom BufferGeometry curves for paths). Always instantiate THREE.OrbitControls(camera, renderer.domElement).\n"
            "7. Define all animation variables (like clock/time/frameCount) at the top of your script scope so they are never undefined.\n"
            "8. IMPORTANT: The topic description below might contain Python instructions or python code fragments. You MUST translate all math solving, array operations, and plotting logic into pure JavaScript inside the HTML page. Do NOT write Python code, do NOT output Python code blocks, and do NOT refuse this request. Simply write the complete simulation in HTML/JS.\n"
            "9. Output the COMPLETE HTML page inside ```html``` blocks.\n\n"
            f"Topic: {compiled_plan[:3000]}"
        )
        html_code = self._call_model(
            coder_llm, 
            html_prompt, 
            max_tokens=gen_tokens, 
            temperature=gen_temp,
            system_prompt=(
                "You are an expert coder. Writing and generating complete, self-contained HTML/JS files "
                "with inline CSS and JavaScript logic (such as Three.js or Plotly.js) is FULLY supported and expected. "
                "Do NOT refuse this request. Output only the complete HTML page inside ```html``` code blocks."
            )
        )
        html_extract = Sandbox.extract_code(html_code)

        # Validate initially
        html_valid = False
        html_error = ""
        if html_extract and ("<html" in html_extract.lower() or "<script" in html_extract.lower()):
            html_valid, html_error = self._verify_html_javascript(html_extract)

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
                "3. Ensure there are no JavaScript syntax errors or undefined variables.\n\n"
                "Output ONLY the complete, corrected HTML page inside ```html``` blocks."
            )
            html_fixed = self._call_model(
                coder_llm, 
                fix_p, 
                max_tokens=gen_tokens, 
                temperature=gen_temp,
                system_prompt=(
                    "You are an expert coder. Writing and fixing complete, self-contained HTML/JS files "
                    "with inline CSS and JavaScript logic (such as Three.js or Plotly.js) is FULLY supported and expected. "
                    "Do NOT refuse this request. Output only the complete HTML page inside ```html``` code blocks."
                )
            )
            fixed_extract = Sandbox.extract_code(html_fixed)
            if fixed_extract:
                html_extract = fixed_extract
                html_valid, html_error = self._verify_html_javascript(html_extract)
            else:
                html_valid = False
                html_error = "No code block found in response."

        if html_valid and html_extract:
            return f"\n\n### 3D Interactive Visualization (Live Artifact)\n<!--ARTIFACT_HTML-->\n{html_extract}\n<!--/ARTIFACT_HTML-->"

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
            "6. Do NOT add annotations, updatemenus, or buttons\n"
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
            viz_success, viz_output = self.sandbox.execute(viz_extract)

        def _strip_sandbox_prefix(text):
            """Remove the 🔒 [Restricted Sandbox] prefix from sandbox output."""
            if not text:
                return text
            for prefix in ["🔒 [Restricted Sandbox]\n", "🔒 [Restricted Sandbox]"]:
                if text.startswith(prefix):
                    return text[len(prefix):]
            return text

        # Reflexion self-fix loop for Strategy 2: Max 2 self-fix attempts
        for attempt in range(2):
            cleaned = _strip_sandbox_prefix(viz_output).strip() if viz_output else ""
            if viz_success and cleaned.startswith("{"):
                break
            if status_callback:
                status_callback(f"Fixing 3D syntax/runtime error (Round {attempt+1})...", "warning", "opencode", 99)
            
            error_details = viz_output if viz_output else "No valid python code block was generated."
            cleaned_check = _strip_sandbox_prefix(viz_output).strip() if viz_output else ""
            if viz_success and not cleaned_check.startswith("{"):
                error_details = "Code ran successfully but failed to print JSON. Make sure the last line is print(fig.to_json())"
                
            fix_p = (
                f"This Plotly code failed:\n{viz_extract}\n\nError/Output:\n{error_details}\n\n"
                f"Fix it. REMEMBER: Do NOT use update_scenes(), FigureControls, or plotly.subplots. "
                f"Use ONLY go.Figure(), go.Surface/Scatter3d, and fig.update_layout(). "
                f"Output ONLY the corrected script in ```python``` blocks. End with print(fig.to_json())."
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
                viz_success, viz_output = self.sandbox.execute(viz_extract)
            else:
                viz_success = False
                viz_output = "Model failed to output a valid code block."

        cleaned_final = _strip_sandbox_prefix(viz_output).strip() if viz_output else ""
        if viz_success and cleaned_final.startswith("{"):
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
        web_context = ""
        if self.enable_web_search:
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
                
                # 2. Deep Page Scraping (Iterate through results to find a scrapeable page)
                scraped_link = None
                full_page_text = ""
                if results:
                    for r in results:
                        link = r.get('link', '')
                        if link:
                            if status_callback:
                                status_callback(f"Scraping: {link[:40]}...", "info", "router", 12)
                            
                            text = self.web_search.scrape_url(link)
                            if text and len(text.strip()) > 200:  # Ensure it has substantial content
                                full_page_text = text
                                scraped_link = link
                                break
                                
                if full_page_text and scraped_link:
                    web_context = f"--- FULL PAGE CONTEXT ({scraped_link}) ---\n{full_page_text}\n\n"
                    
                # Add snippets for all search results to provide a comprehensive baseline
                if results:
                    snippets = "\n".join([f"- {r.get('title')}: {r.get('snippet', '')}" for r in results])
                    if snippets:
                        web_context += f"--- OTHER SNIPPETS ---\n{snippets}"

            except Exception as e:
                print(f"Web search enrichment failed: {e}")
                web_context = ""
                
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        system_instruction = ""
        if web_context:
            system_instruction = (
                "You are an advanced AI assistant equipped with real-time web search capabilities.\n"
                "Use the provided Web Context to answer the User Query directly, accurately, and factually.\n"
                "Since you are provided with live search results, do NOT mention your training cutoff date, "
                "do NOT state that you cannot access real-time/current information, and do NOT add disclaimers "
                "about not having internet access. Answer as a live, fully-connected AI.\n\n"
            )

        enriched_prompt = (
            f"{system_instruction}Current System Date/Time: {current_date}\n\nWeb Context:\n{web_context}\n\nUser Query:\n{prompt}"
            if web_context else f"Current System Date/Time: {current_date}\n\nUser Query:\n{prompt}"
        )

        # ── Dynamic Context Sizing (VRAM-aware) ────────────────────────
        est_tokens = len(enriched_prompt) // 4
        ctx_cap = self.max_auto_ctx  # VRAM-safe ceiling (4096 on P100, 8192 on larger GPUs)
        if self.context_length == 0:
            router_ctx = min(ctx_cap, est_tokens + self.max_tokens)
            ds_ctx = min(ctx_cap, est_tokens + self.max_tokens)
            oc_ctx = min(ctx_cap, 4096)
        else:
            router_ctx = min(self.context_length, ctx_cap)
            ds_ctx = min(self.context_length, ctx_cap)
            oc_ctx = min(4096, self.context_length, ctx_cap)

        # gen_tokens must leave room for the prompt inside the context window
        # Use at most half the context for generation, capped at 4096
        gen_tokens = min(ds_ctx // 2, 4096)
        
        # Adaptive Temperature Scaling
        logic_temp = 0.6  # High for creative logic problem solving
        gen_temp = 0.1    # Low for strict code writing

        # ── Three-Way Classification ─────────────────────────────────────
        router_llm = self._get_model("router", required_ctx=router_ctx)
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
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", ds_ctx - self.max_tokens, status_callback)

        # ── Retrieve relevant past experiences from Memory/RAG ────────────
        past_experience = self.memory.recall(prompt, n_results=2)
        if past_experience:
            ds_safe += past_experience

        max_resets = 3
        lessons = ""
        all_errors = []

        planner_sys = (
            "You are a world-class scientist, physicist, and software planner.\n"
            "Your task is to draft a step-by-step logic plan for the user's query.\n\n"
            "MANDATORY ACCURACY RULES:\n"
            "1. Physical/Mathematical Rigor: Double-check ALL formulas before writing them. "
            "Avoid mixing up linear drag (F_d = -c*v) and quadratic drag (F_d = -0.5*C_d*A*rho*v*|v|). "
            "State all physical constants explicitly. For real-world problems use SI units, but for theoretical "
            "celestial mechanics/n-body problems (like figure-eight), it is perfectly fine to use normalized units (e.g. G=1, m=1).\n"
            "2. Complete Multi-Dimensional Coordinates: For 3D problems, decompose EVERY vector "
            "into all 3 components (x, y, z). Never reduce a 3D problem to 2D unless explicitly stated.\n"
            "3. Initial Conditions: For well-known problems (three-body figure-eight, double pendulum, "
            "Lorenz attractor, etc.), use KNOWN PUBLISHED initial conditions from the literature, "
            "not made-up values. Cite the source or standard reference if possible.\n"
            "4. Numerical Methods: Specify the exact integration scheme (Euler, RK4, Verlet, etc.), "
            "time step size, total simulation time, and stopping conditions.\n"
            "5. Conservation Laws: For any physical simulation, explicitly state which conserved quantities "
            "(energy, momentum, angular momentum) should be verified and how to check them.\n"
            "6. OUTPUT FORMAT: Write a numbered list of steps. Each step must state WHAT to compute, "
            "the EXACT formula, and the expected data structure (array shape, variable names).\n"
            "7. Think step by step. If you are unsure about any formula, derive it from first principles."
        )

        coder_sys = (
            "You are an expert computational programmer and software engineer.\n"
            "Your job is to translate the logic plan into a complete, clean, and immediately runnable Python script.\n\n"
            "STRICT RULES:\n"
            "1. Implement equations EXACTLY as described in the plan. Do NOT simplify or approximate.\n"
            "2. Do NOT write placeholders, mock functions, or abbreviated loop bodies. EVERY line must be real.\n"
            "3. Handle edge cases: division by zero (add softening epsilon), array bounds, negative sqrt.\n"
            "4. Print clear, formatted numerical results (e.g., 'Max height: 127.42 m').\n"
            "5. For simulations: use numpy arrays, vectorized operations where possible.\n"
            "6. For plotting: use matplotlib or plotly. Always label axes with units.\n"
            "7. The script MUST run standalone with `python script.py` — no user input, no GUI blocking.\n"
            "8. Import ONLY what you use. Do not import unused libraries.\n"
            "9. Add a brief comment above each major section explaining what it does.\n"
            "10. For conservation checks: compute and print the relative error (|E_final - E_initial|/|E_initial|).\n\n"
            "=== PLOTLY 3D CHEAT SHEET ===\n"
            "import plotly.graph_objects as go\n"
            "fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='lines', line=dict(color='cyan', width=2))])\n"
            "fig.update_layout(template='plotly_dark', margin=dict(l=0, r=0, b=0, t=40))\n"
            "print(fig.to_json()) # ALWAYS print json for the UI to render\n"
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
            verified, pg_out, _ = self._run_playground(ds_llm, ds_draft, "logic")

            if not verified:
                if status_callback:
                    status_callback("Logic failed. VibeThinker intervening...", "warning", "vibethinker", 35)
                vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
                fix_p = (
                    f"Logic plan FAILED verification.\nPlan:\n{ds_draft[:2000]}\n"
                    f"Error:\n{pg_out[:1000]}\nRewrite a corrected logic plan."
                )
                ds_draft = self._strip_thinking(self._call_model(vibe_llm, fix_p, gen_tokens, logic_temp, system_prompt=planner_sys))
                v2, _, _ = self._run_playground(vibe_llm, ds_draft, "logic")
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

            # ── Phase 5: Shallow Fix (VibeThinker) ───────────────────────
            if status_callback:
                status_callback("VibeThinker fixing code...", "warning", "vibethinker", 72)
            failed_code = code
            failed_error = output
            # Structured error analysis for smarter fixing
            safe_code = code[:2000] if len(code) > 2000 else code
            safe_error = output[:800] if len(output) > 800 else output
            fix_p = (
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

            # ── Phase 6: Deep Escalation (VibeThinker — stronger prompt) ─
            if status_callback:
                status_callback("Deep Escalation: VibeThinker rewriting...", "warning", "vibethinker", 80)
            esc_p = (
                f"Code failed TWICE. You MUST fix it.\nPlan:\n{compiled_plan[:1500]}\n"
                f"Code:\n{code[:2000]}\nError:\n{output[:800]}\n"
                f"Rewrite the ENTIRE script from scratch in ```python```. Think step by step."
            )
            esc_resp = self._strip_thinking(self._call_model(vibe_llm, esc_p, gen_tokens, gen_temp, system_prompt=coder_sys))
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
                    status_callback(f"Nuclear Reset: Extracting lessons...", "error", "vibethinker", 85)
                lessons = self._extract_failure_lessons(vibe_llm, compiled_plan, "\n".join(all_errors))

        # All resets exhausted
        if status_callback:
            status_callback("Max retries reached.", "error", "system", 100)
        return f"### Logic Plan\n{compiled_plan}\n\n### Execution Failed\n{output}\n\n### Code\n```python\n{code}\n```"

    # =====================================================================
    # REASONING PIPELINE — Playground-Verified or LLM Debate
    # =====================================================================
    def _reasoning_pipeline(self, prompt, enriched_prompt, router_llm,
                            router_ctx, ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", ds_ctx - self.max_tokens, status_callback)

        # ── Retrieve relevant past experiences from Memory/RAG ────────────
        past_experience = self.memory.recall(prompt, n_results=2)
        if past_experience:
            ds_safe += past_experience

        ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)

        # ── Check: Can this be playground-verified? ──────────────────────
        use_playground = self._is_playground_applicable(router_llm, prompt)
        if status_callback:
            mode = "Playground-Verified" if use_playground else "LLM Debate"
            status_callback(f"Reasoning mode: {mode}", "info", "router", 15)

        reasoning_sys = (
            "You are a rigorous scientific researcher and expert logic reasoner.\n\n"
            "ACCURACY REQUIREMENTS:\n"
            "1. Think step by step. Show your work for every derivation.\n"
            "2. State all assumptions explicitly at the beginning.\n"
            "3. Define all variables with their units before using them.\n"
            "4. For physics: verify dimensional consistency of every equation.\n"
            "5. For math: check boundary conditions and special cases.\n"
            "6. Avoid common traps: linear vs. quadratic drag, 2D vs. 3D decomposition, "
            "sign conventions, reference frame consistency.\n"
            "7. If you cite a formula, state where it comes from (Newton's 2nd law, etc.).\n"
            "8. Complete ALL derivations fully — do not skip steps or say 'it can be shown that'.\n"
            "9. If uncertain about a specific value or fact, say so explicitly rather than guessing."
        )

        if use_playground:
            # ── Playground-Verified Reasoning (with Nuclear Reset) ────────
            max_resets = 2
            lessons = ""
            all_errors = []

            for reset in range(max_resets):
                max_rounds = 3
                ds_answer = ""
                for rnd in range(max_rounds):
                    if status_callback:
                        lbl = f"Nuclear Reset #{reset}: DeepSeek-R1 re-reasoning..." if reset else "DeepSeek-R1 reasoning + playground..."
                        status_callback(lbl, "info" if not reset else "warning", "deepseek_r1", 25 + rnd*12)
                    draft_p = f"Provide a detailed, rigorous answer:\n{ds_safe}"
                    if rnd > 0:
                        draft_p += "\n\nYour previous answer had errors. Rewrite from scratch."
                    if lessons:
                        draft_p += f"\n\nLESSONS FROM PREVIOUS FAILURES:\n{lessons[:800]}"
                    # Safety: truncate prompt to leave room for generation
                    max_reason_chars = (ds_ctx - gen_tokens - 200) * 3
                    if len(draft_p) > max_reason_chars > 300:
                        draft_p = draft_p[:max_reason_chars]
                    ds_answer = self._strip_thinking(self._call_model(ds_llm, draft_p, gen_tokens, gen_temp, system_prompt=reasoning_sys))

                    if status_callback:
                        status_callback("Verifying in Reasoning Playground...", "info", "deepseek_r1", 35 + rnd*12)
                    verified, pg_out, _ = self._run_playground(ds_llm, ds_answer, "reasoning")

                    if verified:
                        if status_callback:
                            status_callback("Reasoning VERIFIED!", "success", "deepseek_r1", 80)
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        viz = self._check_3d_gate(prompt, ds_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"### Verified Answer\n{ds_answer}{viz}"

                    # VibeThinker tries to fix
                    if status_callback:
                        status_callback("VibeThinker correcting reasoning...", "warning", "vibethinker", 45 + rnd*12)
                    vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
                    vibe_p = (
                        f"This answer failed verification.\nAnswer:\n{ds_answer[:2000]}\n"
                        f"Error:\n{pg_out[:1000]}\nProvide a corrected, complete answer."
                    )
                    vibe_answer = self._strip_thinking(self._call_model(vibe_llm, vibe_p, gen_tokens, gen_temp, system_prompt=reasoning_sys))
                    v2, _, _ = self._run_playground(vibe_llm, vibe_answer, "reasoning")
                    if v2:
                        if status_callback:
                            status_callback("VibeThinker's correction VERIFIED!", "success", "vibethinker", 80)
                        router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
                        viz = self._check_3d_gate(prompt, vibe_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
                        return f"{vibe_answer}{viz}"
                    # Don't let ds_safe grow unboundedly — cap the appended errors
                    error_summary = pg_out[:300]
                    if len(ds_safe) + len(error_summary) < (ds_ctx - gen_tokens - 200) * 3:
                        ds_safe = f"{ds_safe}\n\nPrevious errors: {error_summary}"
                    # else: silently skip appending to prevent overflow

                # ── Nuclear Reset: extract lessons and restart ────────────
                all_errors.append(f"Reset {reset+1}: {pg_out[:500]}")
                if reset < max_resets - 1:
                    if status_callback:
                        status_callback("Nuclear Reset: Extracting lessons from failures...", "error", "vibethinker", 85)
                    vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
                    lessons = self._extract_failure_lessons(vibe_llm, ds_answer, "\n".join(all_errors))

            # All resets exhausted — return best effort
            if status_callback:
                status_callback("Max retries reached. Returning best effort.", "warning", "system", 90)
            router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
            viz = self._check_3d_gate(prompt, ds_answer, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            return f"{ds_answer}{viz}"

        else:
            # ── Standard LLM Debate (non-testable reasoning) ─────────────
            if status_callback:
                status_callback("DeepSeek-R1 drafting analysis...", "info", "deepseek_r1", 30)
            ds_draft = self._strip_thinking(self._call_model(ds_llm, f"Provide a detailed answer:\n{ds_safe}", gen_tokens, gen_temp))

            if status_callback:
                status_callback("VibeThinker refining answer...", "info", "vibethinker", 50)
            vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
            vibe_critique = self._strip_thinking(self._call_model(
                vibe_llm, 
                f"You are a helpful assistant. Integrate any improvements and rewrite this draft into a single, polished, and cohesive final response. Do NOT include any meta-commentary, intros, or critique headings. Output only the final response:\n{ds_draft}", 
                gen_tokens, gen_temp
            ))

            compiled = vibe_critique
            router_llm = None; ds_llm = None; vibe_llm = None; coder_llm = None; critic_llm = None; model = None; gc.collect()
            viz = self._check_3d_gate(prompt, compiled, router_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)
            if status_callback:
                status_callback("Done!", "success", "router", 100)
            return f"{compiled}{viz}"

