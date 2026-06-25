<div align="center">
  
# 🧠 DeepThinker Multi-Agent Hub

### A Fully Local Multi-Agent AI System with Dual Sandbox Verification & Dynamic Hardware Scaling

*Running an orchestrated fleet of specialized LLMs on any hardware — from Intel iGPUs to NVIDIA H100s*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Architecture Multi-Agent MoE](https://img.shields.io/badge/Architecture-Multi--Agent%20MoE-purple)
![Hardware Adaptive](https://img.shields.io/badge/Hardware-Adaptive-orange)

</div>

---

DeepThinker is a **production-grade, fully local multi-agent AI system** that runs multiple heavyweight LLMs on consumer hardware — from Intel Iris Xe integrated GPUs to NVIDIA data center cards. It routes every query through a dynamic pipeline of specialized agents that reason, verify, code, fix, search the web, and generate interactive 3D visualizations — **all without a single API call to the cloud.**

The system auto-detects your hardware at startup and dynamically scales **context windows, batch sizes, scraping depth, memory thresholds, and model configurations** to extract maximum performance from whatever GPU/CPU you have.

> [!CAUTION]
> ### ⚠️ Experimental Status
> This project was developed as a submission for the **India Agentic AI Open Hackathon 2026**. The dual-sandbox architecture, Reflexion loops, GPU-aware scaling, and Dynamic Memory Allocator push consumer hardware to its absolute limits.

---

## ✨ What Makes This Different

Most "local AI" projects run a single model and call it agentic. DeepThinker runs an **orchestrated pipeline of 4 specialized models** with a loop-based Actor-Critic architecture, a live polyglot code execution sandbox, real-time web search with deep scraping, and a 3D visualization engine — all adapting to your hardware.

| Feature | DeepThinker | Typical Local LLM App |
|---|---|---|
| Number of models | 4 (dynamically routed) | 1 |
| Code execution | ✅ Isolated polyglot sandbox (9 languages) | ❌ |
| Self-correction | ✅ Reflexion + Nuclear Reset loops | ❌ |
| Web search | ✅ Deep scraping with deduplication | ❌ or cloud API |
| UI Artifacts | ✅ Claude-style Frontend Sandboxing | ❌ |
| Memory / RAG | ✅ ChromaDB + SQLite vector store | ❌ |
| 3D Visualization | ✅ Three.js / Plotly.js auto-generation | ❌ |
| Hardware scaling | ✅ Auto-adapts to GPU compute capability | Usually NVIDIA only |
| Cloud dependency | ❌ 100% local | Often cloud-backed |

---

## 🤖 The Agent Pipeline

```
User Prompt
    │
    ▼
┌─────────────────────────────────┐
│  Phi-3.5-Mini (Master Router)  │  ← Classifies: SIMPLE / CODING / REASONING
└───────────────┬─────────────────┘
                │
    ┌───────────┼──────────────────────────────┐
    ▼           ▼                              ▼
 SIMPLE      CODING                        REASONING
    │           │                              │
    │           ▼                              ▼
    │     DeepSeek-R1 (Logic Plan)       DeepSeek-R1 (Draft Answer)
    │           │                              │
    │           ▼                              ▼
    │     Reasoning Sandbox             Reasoning Playground
    │       (Logic Check)                 (Python Asserts Check)
    │           │                              │
    │           ├─► Fail: Correction           ├─► Fail: DeepSeek-R1
    │           │   Loop                       │   (Correction Loop ×3)
    │           ▼                              ▼
    │     DeepSeek-R1 (Write Code)       3D Gate Check
    │           │                              │
    │           ▼                              ▼
    │     Execution Sandbox (Polyglot)  OpenCodeInterpreter
    │           │                       (Three.js/Plotly.js Visual)
    │           ├─► Fail: Reflexion
    │           │   Loop ×3
    │           │
    │           ├─► Critical Fail:
    │           │   Nuclear Reset +
    │           │   Emergency Web Search
    │           ▼
    └───────────┴──────────────────────────────┘
                       │
                       ▼
               Streamed Response
           (Answer + Code + 3D Visual)
```

---

## 🧩 The Four Agents

| Agent | Model | Role | VRAM |
|---|---|---|---|
| **Master Router** | Phi-3.5-Mini-Instruct Q6_K | Intent classification, search query optimization, prompt compression, 3D gate checks | ~2.8 GB |
| **Reasoning Engine** | DeepSeek-R1-Distill-Qwen-7B Q6_K | Chain-of-thought logic, math proofs, code writing, deep analysis, Actor-Critic debate | ~6.0 GB |
| **3D Visualization** | OpenCodeInterpreter-DS-6.7B Q6_K | Plotly 3D charts, Three.js simulations, interactive HTML artifact generation | ~5.0 GB |
| **Vision Parser** | Qwen2.5-VL-7B-Instruct Q6_K_XL | Multimodal image/document OCR, circuit diagrams, handwritten equations | ~7.0 GB |

All models are loaded on-demand (lazy loading) and managed by the Dynamic Memory Allocator. Only one model occupies VRAM at a time in EVM mode.

---

## 🛡️ Core Systems

### 1 — Dynamic Memory Allocator (DMA)

A custom memory manager built for running multiple large models on consumer hardware:

- **Auto-Calibration:** Detects total RAM and VRAM at startup, reserves 25% of RAM and 40% of VRAM as safety buffers
- **LRU Eviction:** When memory is tight, evicts the least recently used model first — one at a time, until room is available
- **iGPU Unified Memory Guard:** Prevents `corrupted size vs. prev_size` glibc heap crashes on Intel Iris Xe by blocking simultaneous large allocations
- **Lazy Loading:** Models load only when the pipeline needs them, never all at once
- **Multi-Backend VRAM Detection:** Reads VRAM from `torch.cuda` (NVIDIA), `torch.xpu` (Intel), and Linux sysfs `/sys/class/drm/` (AMD/Intel without ROCm)
- **Automatic CPU Fallback:** If GPU context creation fails, transparently retries with `n_gpu_layers=0` to keep the pipeline alive

### 2 — GPU-Aware Dynamic Scaling

The entire system auto-scales to the detected GPU compute capability:

| Hardware | Compute Cap | Context Ceiling | Scrape Depth | Batch Size |
|---|---|---|---|---|
| **H100/B200** (80 GB) | SM ≥ 9.0 | **65,536 tokens** | 8,000 chars/page | Default (2048) |
| **A100/RTX 4090** (40-48 GB) | SM ≥ 8.0 | **32,768 tokens** | 8,000 chars/page | Default (2048) |
| **P100/T4/V100** (16 GB) | SM < 8.0 | **8,192 tokens** | ~2,949 chars/page | 512 (n_ubatch=256) |
| **Intel Iris Xe iGPU** | N/A | **16,384 tokens** | ~5,898 chars/page | Default |
| **CPU only** | N/A | **16,384 tokens** | ~5,898 chars/page | Default |

This scaling chain flows through every subsystem: **GPU Compute Capability → Context Ceiling → Web Scrape char_limit → Prompt Budget → Semantic Cruncher thresholds → Generation token allocation.**

On older GPUs (SM < 8.0), the system also:
- Restricts llama.cpp `n_batch` to 512 and `n_ubatch` to 256 to prevent quadratic self-attention VRAM spikes
- Explicitly disables `flash_attn` to avoid buggy fallback paths

### 3 — Enterprise VRAM Multiplexing (EVM)

A hardware-aware hot-swap technology for environments where System RAM far exceeds GPU VRAM (e.g., Kaggle P100 with 32 GB RAM + 16 GB VRAM):

**Auto-Activation Criteria:**
- System RAM ≥ 24 GB
- GPU VRAM ≤ 16 GB (discrete NVIDIA/CUDA GPU)

When active, EVM:
- Loads all models into the OS page cache (System RAM)
- Aggressively hot-swaps models one-by-one into GPU VRAM via PCIe
- Evicts dormant agents to guarantee the active model gets **100% of VRAM** for its KV cache
- Reduces RAM safety threshold to 5% (from 25%) and VRAM safety to 5% (from 40%)

The **Load All Models** sidebar button pre-loads all models into the OS page cache, reducing subsequent hot-swaps to **1-2 seconds** instead of minutes.

### 4 — Polyglot Execution Sandbox

A three-layer isolated code execution environment supporting **9 programming languages:**

**Supported Languages:** Python, C, C++, Java, JavaScript, Bash, Go, Rust, TypeScript

**Three Layers of Protection:**
1. **Process Isolation:** Code runs in a subprocess — crashes can't kill the server
2. **Restricted Builtins:** Custom `__import__` whitelist blocks `open`, `exec`, `eval`, `os`, `subprocess`, `socket`
3. **Linux Resource Limits:** 2 GB RAM cap, 300s CPU limit, 200 child process limit, 100 MB file write limit

**Additional Features:**
- **Auto-pip Package Installer:** Detects `ModuleNotFoundError`, auto-installs the missing pip package, and retries execution
- **Unrestricted Fallback:** If the restricted sandbox blocks a legitimate library, falls back to unrestricted execution with a `⚠️ [Unrestricted Fallback]` prefix
- **GUI Detection:** Identifies `pygame`, `tkinter`, `turtle`, etc. and returns a friendly message instead of hanging
- **Infinite Loop Detection:** Catches `while True`, `for(;;)`, `.mainloop()` patterns

**Whitelisted Science Libraries (Restricted Mode):**
`numpy`, `sympy`, `scipy`, `pandas`, `plotly`, `sklearn`, `statsmodels`, `pint`, `z3` (theorem prover), `networkx`, `astropy`, `Bio` (Biopython), `rdkit` (cheminformatics), `rocketpy`, `qiskit`, `qutip`, `cryptography`, `scapy`, `requests`, `urllib`

### 5 — Reflexion Self-Fix Loop

When generated code fails, the system doesn't just report the error — it fixes it:

1. **Shallow Fix:** Attempt a targeted correction based on the error traceback
2. **Deep Escalation:** Rewrite the entire script with a stronger prompt including the failure context
3. **Nuclear Reset:** Extract lessons from all failures and start over from scratch with a new plan
4. Up to **2 reflexion loops** and **1 Nuclear Reset cycle** before returning best-effort output

### 6 — Long-Term RAG Memory

- **ChromaDB** vector database for semantic similarity search (primary)
- **SQLite** with cosine similarity and keyword search as fallback
- Stores **compact solution summaries** — not raw code dumps — to avoid context bloat
- Stores **mistake-fix patterns** from the Reflexion loop to prevent regressing on similar tasks
- **Smart Deduplication:** Uses NLP stopword filtering and punctuation stripping to merge near-identical tasks (>80% content word overlap)
- **Concurrency-safe:** Dynamic connection pooling with 30-second lock timeouts for SQLite

### 7 — Web Search Integration

Real-time web search with deep scraping and intelligent deduplication:

**Search Provider Chain:** Google Custom Search API → SearXNG (4 instance fallback) → DuckDuckGo Library → DuckDuckGo HTML Scraper

**Deep Scraping Pipeline:**
- **LLM Query Optimizer:** Phi-3.5-Mini transforms the user prompt into a concise search query, preserving locations, names, and timeframes
- **Context-Aware Scraping:** Char limits per page scale dynamically with the GPU context ceiling (not raw VRAM), preventing oversized payloads
- **Jaccard Deduplication:** Computes word-level Jaccard similarity between scraped pages — any page with >65% overlap against previously collected content is skipped
- **Cloudflare/Bot Detection:** Automatically skips pages with captcha, access denied, or DDoS protection markers

**Predictive Scraping Mode:** For queries containing "predict", "forecast", or "prediction", the system expands to 8-20 pages with targeted character limits to harvest dense datasets for statistical analysis.

### 8 — Semantic Prompt Cruncher

Prevents context window crashes when prompts exceed the model's capacity:

- **Tokenizer-Precise Estimation:** Uses the loaded model's actual tokenizer for exact token counts (not char/3 approximation)
- **Smart Budget Allocation:** Splits the prompt into 25% top, 55% bottom, and 20% middle summary — preserving the user's question and the most recent context
- **Context-Safe Summarization:** Dynamically truncates the middle chunk to fit inside the router model's context window (with 300 token overhead for chat template markers) before summarization
- **Overflow-Safe Routing:** All internal summarization calls go through `_call_model` which has built-in overflow protection
- **Post-Crunch Verification:** After summarization, the final crunched output is verified against the budget; if still over, a hard-truncate preserves head and tail
- **Model-Aware Generation Sizing:** Allocates up to 40% of context for generation (capped 2048-8192), with a minimum 1500-token prompt headroom guarantee

### 9 — Claude-Style UI Artifacts & 3D Visualization

After every `CODING` or `REASONING` response, a **3D Gate Check** runs automatically:

- The Router decides if the task involves mathematical graphing, physics equations, data plots, or molecular structures
- If yes, **OpenCodeInterpreter** generates a self-contained **HTML/JS Artifact** using Three.js or Plotly.js:
  - **Physics:** Orbits, trajectories, barycenter markers, gravity vectors
  - **Biology/Chemistry:** DNA double-helices, membrane channels, molecular compounds
  - **Interactive Controls:** Glassmorphic sliders to tweak physical settings in real-time
- Rendered inside a **secure, isolated iframe sandbox** — exactly like Anthropic's Claude Artifacts
- **Real-time Console Overlay:** Captures `console.log`, `console.warn`, `console.error` from the iframe for instant debugging
- **Dual-Mode Fallback:** If HTML generation fails, falls back to Python Plotly executor

### 10 — Vision / Multimodal Input

Upload any image alongside a prompt for multimodal analysis:
- **Qwen2.5-VL 7B** parses images and extracts text, diagrams, and logic
- Extracted content is prepended to the prompt and passed through the full pipeline
- Works with circuit diagrams, handwritten equations, screenshots, charts, and documents

### 11 — Live Streaming Agent Timeline

The React UI streams the pipeline status in real-time:
- Each agent step shows a status badge: `info` / `warning` / `success` / `error`
- Progress bar advances through pipeline stages
- Logs collapse into a clean accordion once the final response arrives
- **Cancel button** stops generation mid-stream while keeping models loaded for instant reuse
- **Offload Memory** button unloads all models from RAM/VRAM on demand

---

## 🖥️ Hardware Compatibility

| Hardware | Status | Notes |
|---|---|---|
| **Intel Iris Xe iGPU** | ✅ Full support | Uses system RAM as VRAM via Level Zero API |
| **Intel Arc dGPU** | ✅ Full support | Same SYCL/Level Zero path |
| **NVIDIA (CUDA)** | ✅ Full support | cuBLAS acceleration via llama.cpp, auto-scales by compute capability |
| **AMD (ROCm)** | ✅ Full support | ROCm path via PyTorch + llama.cpp |
| **Apple Silicon (M1-M4)** | ✅ Full support | Metal backend |
| **CPU only** | ✅ Works | Slow but functional (switch to `device_mode: cpu` in settings) |

### Intel Iris Xe — Special Notes

Running 7B+ models on iGPUs is notoriously unstable. These fixes are baked into the codebase:

- **Error 45 (`UR_RESULT_ERROR_INVALID_ARGUMENT`):** Fixed via `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in `app.py`
- **`corrupted size vs. prev_size` heap crash:** Fixed via the DMA iGPU Guard preventing simultaneous large allocations
- **Float16 kernel failures:** Models fall back to `float32` on XPU with IPEX optimization applied automatically

### NVIDIA Older GPUs (P100/T4/V100) — Special Notes

On SM < 8.0 GPUs that lack hardware Flash Attention:

- **Batch size restriction:** `n_batch=512`, `n_ubatch=256` to prevent quadratic VRAM scratch buffer spikes
- **Flash attention disabled:** `flash_attn=False` to avoid buggy fallback code paths
- **Context ceiling capped:** Maximum 8,192 tokens to prevent CUDA OOM during prompt evaluation

---

## 🧠 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **RAM** | 16 GB | 32 GB |
| **GPU VRAM** | 8 GB (NVIDIA/AMD dGPU) | 16 GB+ |
| **Storage** | 25 GB free | 40 GB free |
| **OS** | Ubuntu 22.04 / Win 10 / macOS 13 | Ubuntu 24.04 / Win 11 / macOS 14 |
| **Python** | 3.10 | 3.12 |
| **Node.js** | 18 | 20 |

> **Intel iGPU (Iris Xe / Arc):** Uses system RAM as VRAM via the Level Zero API. Works with **16 GB RAM** minimum (using aggressive LRU model swapping), but 32 GB is recommended. Always export `SYCL_DEVICE_FILTER=level_zero` before starting the backend.

---

## 📁 Project Structure

```
Team_Trenches/
├── backend/
│   ├── app.py            # FastAPI server, REST endpoints, SSE streaming, cancel/offload controls
│   ├── orchestrator.py   # Core multi-agent pipeline, DMA, GPU-aware scaling, Reflexion loops
│   ├── sandbox.py        # Polyglot code execution sandbox (9 languages), auto-pip installer
│   ├── memory.py         # Long-term RAG memory (ChromaDB + SQLite), deduplication engine
│   ├── search.py         # Web search (Google → SearXNG → DuckDuckGo), deep scraper
│   └── downloader.py     # Model downloader with HuggingFace Hub, retry logic, shard support
├── frontend/
│   └── src/
│       └── App.jsx       # React UI: live streaming, 3D artifacts, glassmorphic design
├── models/               # Downloaded GGUF weights (git-ignored, ~18 GB total)
│   ├── text/
│   │   ├── router/       # Phi-3.5-Mini-Instruct Q6_K (~2.8 GB)
│   │   ├── deepseek_r1/  # DeepSeek-R1-Distill-Qwen-7B Q6_K (~6.0 GB)
│   │   └── opencode/     # OpenCodeInterpreter-DS-6.7B Q6_K (~5.0 GB)
│   └── image_to_text/
│       └── qwen_vl/      # Qwen2.5-VL-7B-Instruct Q6_K_XL (~7.0 GB)
├── start.sh              # One-command Linux launcher (backend + frontend)
├── STARTUP.md            # Cross-platform startup guide
├── README_SETUP.md       # Hardware acceleration installation guide
└── requirements.txt      # Python dependencies
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | `POST` | Main chat endpoint — streams SSE status updates + final response |
| `/api/status` | `GET` | System health, model status, RAM/CPU/GPU usage, EVM state |
| `/api/download/{model_key}` | `POST` | Trigger background model download from HuggingFace |
| `/api/cancel` | `POST` | Stop active generation (models stay loaded in memory) |
| `/api/offload` | `POST` | Unload all models from RAM/VRAM |
| `/api/load_all` | `POST` | Pre-load all downloaded models into RAM page cache (EVM only) |
| `/api/settings` | `POST` | Update context length, temperature, GPU layers, web search toggle |
| `/api/memory/count` | `GET` | Number of stored long-term memories |
| `/api/memory/clear` | `POST` | Reset the vector database and memory store |
| `/api/unload` | `POST` | Unload all models to free system RAM |

---

## 🧪 Example Prompts to Try

**Weather & Real-Time Data:**
> *How is the weather in Jharsuguda today?*

**3D Physics Simulation:**
> *Simulate the orbital mechanics of a binary star system. Give me the mathematical explanation, and then generate an interactive 3D visualization showing the orbital paths.*

**Live Prediction Playground:**
> *Perform a predictive analysis on the global adoption of renewable energy over the next 5 years. Based on historical growth rates, forecast the percentage share of renewables by 2030.*

**Deep Reasoning + Verification:**
> *Prove that the sum of the first N natural numbers is N(N+1)/2. Verify it computationally for N=100.*

**Reflexion Loop Test:**
> *Write a Python implementation of a Red-Black Tree with insert, delete, and search operations. Include test cases.*

**Vision + Coding (upload an image):**
> *Analyze this circuit diagram and write the Python simulation for it.*

---

## 🆕 Pipeline Optimizations (June 2026)

<details>
<summary><strong>Click to expand full optimization changelog</strong></summary>

- **GPU-Aware Dynamic Context Scaling:** Context windows, batch sizes, scraping depths, and generation budgets all auto-scale based on detected GPU compute capability (SM 6.0 → SM 9.0+)
- **Token Overflow Prevention (4-Bug Fix):** Comprehensive audit fixing char_limit ignoring context ceiling, Semantic Cruncher bypassing overflow protection, chat template overhead underestimation, and post-crunch output verification
- **Context-Derived Scraping:** Web scrape char_limits now derive from `ds_ctx_est × 0.60 × 3 / pages` instead of raw VRAM/RAM, preventing oversized payloads
- **LLM Search Query Optimizer:** Transforms user prompts into concise Google queries while preserving locations, names, and timeframes — prevents city/location details from being discarded
- **Older GPU Batch Restriction:** On SM < 8.0 GPUs, restricts `n_batch=512`, `n_ubatch=256`, and disables `flash_attn` to prevent quadratic VRAM scratch buffer crashes
- **EVM Warm-Up Preloading:** Background page-cache warming via `/api/load_all` for 1-2 second hot-swaps
- **RAG Memory Contamination Fix:** Prioritizes successful solutions over intermediate mistake logs
- **Deterministic Playground Verification:** Test scripts routed to Phi-3.5-Mini instead of verbose DeepSeek-R1
- **Emergency Search Verification:** Post emergency web search, runs 1 round of sandbox verification with correction
- **RAG Variable Prioritization:** Forces models to use past memories only for algorithmic structure, not numerical values
- **Aligned Coding & Reasoning Loops:** Identical nested retry structure (2 outer resets, 2 inner drafts per reset)
- **Multi-Tiered Intent Classification:** 4-stage classifier: structural patterns → domain overrides → few-shot prompting → fallback scans
- **Auto-pip Package Installer:** Detects `ModuleNotFoundError`, pip-installs on the fly, and re-executes
- **Memory Contamination Guard:** Strips past experience during playground verification for first-principles derivation

</details>

---

## 📦 Setup & Installation

See the dedicated guides:

| Guide | What it covers |
|---|---|
| **[README_SETUP.md](./README_SETUP.md)** | System prerequisites, Python deps, GPU acceleration (Mac/NVIDIA/Intel) |
| **[STARTUP.md](./STARTUP.md)** | Downloading models, starting the backend and frontend, troubleshooting |

**Quick Start (Linux):**
```bash
# 1. Clone the repository
git clone https://github.com/Bshdhorrhh/Team_Trenches.git
cd Team_Trenches

# 2. Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Launch everything
./start.sh
```

The web UI will be available at `http://localhost:5173`. Models download automatically on first use.

---

## 👥 Team

**Team Trenches** — India Agentic AI Open Hackathon 2026

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
