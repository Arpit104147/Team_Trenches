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

    def __call__(self, prompt, max_tokens=512, temperature=0.7):
        if isinstance(prompt, str):
            # Convert raw strings into proper conversational format so the model doesn't hallucinate
            messages = [{"role": "user", "content": prompt}]
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
        # VRAM safety: keep 2 GB free per GPU (only matters for discrete GPUs)
        self.vram_safety_gb = 2.0
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

    def _check_memory_pressure(self):
        """LRU-based eviction loop. Evicts models one-by-one until safe."""
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
                if vram_free is not None and vram_free < self.vram_safety_gb:
                    print(f"⚠️ DMA: CUDA GPU:{gpu_idx} VRAM low ({vram_free:.1f} GB free)")
                    while vram_free < self.vram_safety_gb and self.model_access_order:
                        if not self._evict_lru_model():
                            break
                        evicted_any = True
                        vram_free = self._get_vram_free_gb(gpu_idx)

        # ── Linux sysfs VRAM Check (AMD/Intel Vulkan — no ROCm needed) ──
        sysfs_gpus = self._get_sysfs_gpu_vram()
        for free_gb, total_gb, card_name in sysfs_gpus:
            if free_gb < self.vram_safety_gb:
                print(f"⚠️ DMA: sysfs {card_name} VRAM low ({free_gb:.1f}/{total_gb:.1f} GB free)")
                while free_gb < self.vram_safety_gb and self.model_access_order:
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
        # Floor: never load a model below 8192 ctx — avoids reload-on-expand segfaults on Intel XPU
        required_ctx = max(8192, required_ctx)

        if model_key in self.loaded_models:
            model_obj = self.loaded_models[model_key]
            # Context expansion – safe path for Intel XPU (Level‑Zero)
            if hasattr(model_obj, "n_ctx") and required_ctx > model_obj.n_ctx():
                # Detect Intel XPU environment
                is_xpu = False
                try:
                    if torch and hasattr(torch, "xpu") and torch.xpu.is_available():
                        is_xpu = True
                except Exception:
                    pass
                if is_xpu:
                    print(f"⚠️ Skipping context expansion for '{model_key}' on Intel XPU (current: {model_obj.n_ctx()}, requested: {required_ctx})")
                    # No reload – keep current model; the larger context request will be capped later
                else:
                    # Non‑XPU path – safe reload with delay and cache clear
                    print(f"🔄 Reloading '{model_key}' to expand context: {model_obj.n_ctx()} -> {required_ctx}")
                    if hasattr(model_obj, 'close'):
                        try:
                            model_obj.close()
                        except Exception:
                            pass
                    del self.loaded_models[model_key]
                    gc.collect()
                    import time
                    time.sleep(1.5)
                    if torch and hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.empty_cache()
            else:
                self._touch_model(model_key)
                return model_obj

        # DMA Check: evict if memory is low BEFORE attempting a new load
        self._check_memory_pressure()

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
    def _strip_thinking(self, text):
        """Remove <think>...</think> blocks from DeepSeek R1 / VibeThinker output."""
        if not text:
            return text
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Also handle unclosed <think> tags (model sometimes forgets to close)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()


    def _crunch_prompt(self, prompt, target_model, max_tokens_limit, status_callback=None):
        """Compresses a massive prompt safely using the fast Router model."""
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
        
        # Temporarily get router with fixed small context for the summarization
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
    def _call_model(self, llm, prompt, max_tokens=512, temperature=0.7):
        if isinstance(llm, TransformerWrapper):
            return llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature
        )['choices'][0]['message']['content']

    # =========================================================================
    # DYNAMIC ACTOR-CRITIC VERIFIER PIPELINE — Helper Methods
    # =========================================================================

    def _classify_task(self, router_llm, prompt):
        """Three-way classification: SIMPLE, CODING, or REASONING."""
        p = (
            "Classify this query into EXACTLY ONE category. Reply with ONLY the category name.\n\n"
            "SIMPLE — Quick factual answers, greetings, definitions, translations, yes/no questions.\n"
            "  Examples: 'What is the capital of France?', 'Hi how are you?', 'Define entropy', 'Translate hello to Spanish'\n\n"
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
            "- If the query asks for detailed explanation with visualization → REASONING\n"
            "- If unsure between SIMPLE and REASONING → choose REASONING\n"
            "- If unsure between CODING and REASONING → choose CODING\n\n"
            f"Query: {prompt[:500]}\n\nCategory:"
        )
        result = self._call_model(router_llm, p, max_tokens=10, temperature=0.1)
        upper = str(result).strip().upper()
        # Strict keyword extraction from model response
        if "CODING" in upper:
            return "CODING"
        if "REASONING" in upper:
            return "REASONING"
        if "SIMPLE" in upper:
            return "SIMPLE"
        # Fallback: keyword scan on the original prompt for safety
        prompt_lower = prompt.lower()
        code_keywords = ["write code", "write a code", "fix code", "debug", "script", "implement", "program", "compile", "function(", "def ", "class ", "import "]
        reason_keywords = ["explain", "prove", "derive", "why ", "how does", "in detail", "theory", "analyze", "compare", "solve"]
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
            f"Write a Python script (max 40 lines) that {'tests this code logic' if purpose == 'logic' else 'verifies this reasoning'}.\n"
            "Use assertions and print 'VERIFIED' as the last line if all checks pass.\n\n"
            "You have access to these scientific tools:\n"
            "  - math, cmath           → Core math operations, trigonometry, constants\n"
            "  - numpy                 → Arrays, linear algebra, matrix operations, statistics\n"
            "  - sympy                 → Symbolic math: algebra, calculus, equation solving, proofs\n"
            "  - scipy                 → Physics: scipy.constants (speed of light, gravity, Planck),\n"
            "                            scipy.integrate (kinematics, ODE), scipy.optimize (curve fitting)\n"
            "  - pint                  → Unit verification: check that formulas produce correct physical units\n"
            "                            Example: ureg = pint.UnitRegistry(); v = 10 * ureg.meter / ureg.second\n"
            "  - z3 (z3-solver)        → Formal logic & theorem proving: constraint satisfaction, SAT solving\n"
            "                            Example: from z3 import *; x = Int('x'); solve(x > 2, x < 5)\n"
            "  - networkx              → Graph theory: shortest paths, connectivity, circuit analysis\n"
            "  - astropy               → Astrophysics: celestial mechanics, orbital calculations, cosmology\n"
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
            "These attempts ALL FAILED. Extract 3-5 KEY LESSONS for rewriting:\n\n"
            f"Plan:\n{failed_plan[:1500]}\n\n"
            f"Errors:\n{all_errors[:1500]}\n\n"
            "Reply with a numbered list of specific, actionable lessons ONLY."
        )
        lessons = self._call_model(critic_llm, p, max_tokens=512, temperature=0.3)
        return self._strip_thinking(lessons)

    def _check_3d_gate(self, prompt, compiled_plan, router_llm, coder_llm, gen_tokens, gen_temp, status_callback=None):
        """Check if the task needs 3D visualization and generate it if so."""
        if status_callback:
            status_callback("Checking 3D Visualization Eligibility...", "info", "router", 90)
        gate_prompt = (
            "Does this task involve mathematical graphing, data plotting, 3D matrices, or physics equations? "
            "Game development (pygame, tkinter, GUI apps) does NOT count. "
            "Reply ONLY 'YES' or 'NO'.\n\n"
            f"Query: {prompt[:500]}"
        )
        is_3d = self._call_model(router_llm, gate_prompt, max_tokens=10, temperature=0.1)

        if "YES" not in str(is_3d).upper():
            return ""

        if status_callback:
            status_callback("Generating Interactive 3D Chart...", "info", "opencode", 95)
        viz_prompt = (
            "You are a Python data visualization expert. "
            "Write ONLY a complete Python script using plotly for an interactive 3D visualization.\n"
            "RULES:\n"
            "1. Import plotly.graph_objects and numpy\n"
            "2. Create a 3D scatter, surface, or line plot\n"
            "3. Use fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=40,b=0))\n"
            "4. Do NOT set background colors manually\n"
            "5. Do NOT add annotations, updatemenus, or buttons\n"
            "6. Last line: print(fig.to_json())\n"
            "7. Do NOT call fig.show() or save to file\n\n"
            "Output ONLY code in ```python``` blocks.\n\n"
            f"Topic: {compiled_plan[:3000]}"
        )
        viz_code = self._call_model(coder_llm, viz_prompt, max_tokens=gen_tokens, temperature=gen_temp)
        viz_extract = Sandbox.extract_code(viz_code)

        if status_callback:
            status_callback("Rendering 3D Visualization...", "info", "opencode", 97)
        viz_success, viz_output = self.sandbox.execute(viz_extract)

        # Reflexion self-fix for 3D
        if not viz_success:
            if status_callback:
                status_callback("Fixing 3D syntax error...", "warning", "opencode", 98)
            fix_p = (
                f"This Plotly code failed:\n{viz_extract}\n\nError:\n{viz_output}\n\n"
                f"Fix it. Output ONLY the corrected script in ```python``` blocks. End with print(fig.to_json())."
            )
            viz_fixed = self._call_model(coder_llm, fix_p, max_tokens=gen_tokens, temperature=gen_temp)
            viz_extract = Sandbox.extract_code(viz_fixed)
            viz_success, viz_output = self.sandbox.execute(viz_extract)
            viz_code = viz_fixed

        if viz_success and viz_output and viz_output.strip().startswith("{"):
            return f"\n\n### 3D Interactive Visualization\n<!--PLOTLY_JSON-->\n{viz_output.strip()}\n<!--/PLOTLY_JSON-->"
        error_msg = f"\n\n**Execution Error:**\n```text\n{viz_output}\n```" if not viz_success else ""
        return f"\n\n### 3D Visualization Script\n{viz_code}{error_msg}"

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
                status_callback("Gathering live facts via SearXNG...", "info", "router", 8)
            try:
                results = self.web_search.search(prompt, max_results=3)
                web_context = "\n".join([r.get('snippet', '') for r in results])
            except Exception:
                web_context = ""
        enriched_prompt = (
            f"Web Context:\n{web_context}\n\nUser Query:\n{prompt}"
            if web_context else prompt
        )

        # ── Dynamic Context Sizing ───────────────────────────────────────
        est_tokens = len(enriched_prompt) // 4
        if self.context_length == 0:
            router_ctx = max(8192, min(64000, est_tokens + self.max_tokens))
            ds_ctx = max(8192, min(64000, est_tokens + self.max_tokens))
            oc_ctx = 8192
        else:
            router_ctx = self.context_length
            ds_ctx = self.context_length
            oc_ctx = min(8192, self.context_length)

        gen_tokens = 8192
        gen_temp = 0.1

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
            safe = self._crunch_prompt(enriched_prompt, "router", router_ctx - 4096, status_callback)
            return self._call_model(router_llm, safe, max_tokens=4096, temperature=0.6)

        # ══════════════════════════════════════════════════════════════════
        # PATH B: CODING — Actor-Critic with Dual Sandbox
        # ══════════════════════════════════════════════════════════════════
        if task_type == "CODING":
            return self._coding_pipeline(prompt, enriched_prompt, router_llm,
                                         ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)

        # ══════════════════════════════════════════════════════════════════
        # PATH C: REASONING — Playground-Verified or LLM Debate
        # ══════════════════════════════════════════════════════════════════
        return self._reasoning_pipeline(prompt, enriched_prompt, router_llm,
                                        ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback)

    # =====================================================================
    # CODING PIPELINE — Reasoning Sandbox → Code Sandbox → Reflexion
    # =====================================================================
    def _coding_pipeline(self, prompt, enriched_prompt, router_llm,
                         ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
        ds_safe = self._crunch_prompt(enriched_prompt, "deepseek_r1", ds_ctx - self.max_tokens, status_callback)

        # ── Retrieve relevant past experiences from Memory/RAG ────────────
        past_experience = self.memory.recall(prompt, n_results=2)
        if past_experience:
            ds_safe += past_experience

        max_resets = 3
        lessons = ""
        all_errors = []

        for reset in range(max_resets):
            # ── Phase 1: DeepSeek Logic Plan ─────────────────────────────
            if status_callback:
                lbl = f"Nuclear Reset #{reset}: Rewriting..." if reset else "DeepSeek-R1 drafting logic..."
                status_callback(lbl, "info" if not reset else "warning", "deepseek_r1", 20)

            ds_llm = self._get_model("deepseek_r1", required_ctx=ds_ctx)
            plan_p = f"Create a step-by-step logic plan:\n{ds_safe}"
            if lessons:
                plan_p += f"\n\nLESSONS FROM PREVIOUS FAILURES:\n{lessons}"
            ds_draft = self._strip_thinking(self._call_model(ds_llm, plan_p, gen_tokens, gen_temp))

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
                ds_draft = self._strip_thinking(self._call_model(vibe_llm, fix_p, gen_tokens, gen_temp))
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
            code_p = f"Write a complete Python script for this plan:\n{compiled_plan}\n\nWrap in ```python```."
            code = Sandbox.extract_code(self._strip_thinking(self._call_model(vibe_llm, code_p, gen_tokens, gen_temp)))

            # ── Phase 4: Execution Sandbox ───────────────────────────────
            if status_callback:
                status_callback("Executing in Sandbox...", "info", "sandbox", 65)
            ok, output = self.sandbox.execute(code)
            if ok:
                self.memory.save(prompt, code)
                coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
                viz = self._check_3d_gate(prompt, compiled_plan, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
                return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"

            # ── Phase 5: Shallow Fix (VibeThinker) ───────────────────────
            if status_callback:
                status_callback("VibeThinker fixing code...", "warning", "vibethinker", 72)
            failed_code = code
            failed_error = output
            fix_p = f"Code failed:\n{code}\n\nError:\n{output}\n\nFix it. Output in ```python```."
            code = Sandbox.extract_code(self._strip_thinking(self._call_model(vibe_llm, fix_p, gen_tokens, gen_temp)))
            ok, output = self.sandbox.execute(code)
            if ok:
                self.memory.save(prompt, code)
                self.memory.save_mistake(prompt, failed_code, failed_error, code)
                coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
                viz = self._check_3d_gate(prompt, compiled_plan, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
                return f"### Logic Plan (Verified)\n{compiled_plan}\n\n### Execution Output\n{output}{viz}\n\n### Code\n```python\n{code}\n```"

            # ── Phase 6: Deep Escalation (VibeThinker — stronger prompt) ─
            if status_callback:
                status_callback("Deep Escalation: VibeThinker rewriting...", "warning", "vibethinker", 80)
            esc_p = (
                f"Code failed TWICE. You MUST fix it.\nPlan:\n{compiled_plan[:1500]}\n"
                f"Code:\n{code}\nError:\n{output}\n"
                f"Rewrite the ENTIRE script from scratch in ```python```. Think step by step."
            )
            esc_resp = self._strip_thinking(self._call_model(vibe_llm, esc_p, gen_tokens, gen_temp))
            if "```" in esc_resp:
                code = Sandbox.extract_code(esc_resp)
                ok, output = self.sandbox.execute(code)
                if ok:
                    self.memory.save(prompt, code)
                    self.memory.save_mistake(prompt, failed_code, failed_error, code)
                    coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
                    viz = self._check_3d_gate(prompt, compiled_plan, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
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
                            ds_ctx, oc_ctx, gen_tokens, gen_temp, status_callback=None):
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

        if use_playground:
            # ── Playground-Verified Reasoning ────────────────────────────
            max_rounds = 3
            ds_answer = ""
            for rnd in range(max_rounds):
                if status_callback:
                    status_callback("DeepSeek-R1 reasoning + playground...", "info", "deepseek_r1", 25 + rnd*15)
                draft_p = f"Provide a detailed, rigorous answer:\n{ds_safe}"
                if rnd > 0:
                    draft_p += "\n\nYour previous answer had errors. Rewrite from scratch."
                ds_answer = self._strip_thinking(self._call_model(ds_llm, draft_p, gen_tokens, gen_temp))

                if status_callback:
                    status_callback("Verifying in Reasoning Playground...", "info", "deepseek_r1", 35 + rnd*15)
                verified, pg_out, _ = self._run_playground(ds_llm, ds_answer, "reasoning")

                if verified:
                    if status_callback:
                        status_callback("Reasoning VERIFIED!", "success", "deepseek_r1", 80)
                    coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
                    viz = self._check_3d_gate(prompt, ds_answer, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
                    return f"### Verified Answer\n{ds_answer}{viz}"

                # VibeThinker tries to fix
                if status_callback:
                    status_callback("VibeThinker correcting reasoning...", "warning", "vibethinker", 45 + rnd*15)
                vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
                vibe_p = (
                    f"This answer failed verification.\nAnswer:\n{ds_answer[:2000]}\n"
                    f"Error:\n{pg_out[:1000]}\nProvide a corrected, complete answer."
                )
                vibe_answer = self._strip_thinking(self._call_model(vibe_llm, vibe_p, gen_tokens, gen_temp))
                v2, _, _ = self._run_playground(vibe_llm, vibe_answer, "reasoning")
                if v2:
                    if status_callback:
                        status_callback("VibeThinker's correction VERIFIED!", "success", "vibethinker", 80)
                    coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
                    viz = self._check_3d_gate(prompt, vibe_answer, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
                    return f"### Verified Answer\n{vibe_answer}{viz}"
                ds_safe = f"{ds_safe}\n\nPrevious errors: {pg_out[:500]}"

            # Exhausted playground rounds — return best effort
            coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
            viz = self._check_3d_gate(prompt, ds_answer, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
            return f"### Answer (Best Effort)\n{ds_answer}{viz}"

        else:
            # ── Standard LLM Debate (non-testable reasoning) ─────────────
            if status_callback:
                status_callback("DeepSeek-R1 drafting analysis...", "info", "deepseek_r1", 30)
            ds_draft = self._strip_thinking(self._call_model(ds_llm, f"Provide a detailed answer:\n{ds_safe}", gen_tokens, gen_temp))

            if status_callback:
                status_callback("VibeThinker critiquing...", "info", "vibethinker", 50)
            vibe_llm = self._get_model("vibethinker", required_ctx=ds_ctx)
            vibe_critique = self._strip_thinking(self._call_model(
                vibe_llm, f"Critique and refine:\n{ds_draft}", gen_tokens, gen_temp
            ))

            compiled = f"{ds_draft}\n\nRefinements:\n{vibe_critique}"
            coder_llm = self._get_model("opencode", required_ctx=oc_ctx)
            viz = self._check_3d_gate(prompt, compiled, router_llm, coder_llm, gen_tokens, gen_temp, status_callback)
            if status_callback:
                status_callback("Done!", "success", "router", 100)
            return f"### Analysis\n{compiled}{viz}"

