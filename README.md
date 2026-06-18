# 🧠 DeepThinker Multi-Agent Hub

> **India Agentic AI Open Hackathon 2026 — Team Trenches**

A production-grade, fully local multi-agent AI system that runs **five heavyweight LLMs simultaneously** on consumer hardware — including Intel Iris Xe integrated GPUs. DeepThinker routes every query through a dynamic pipeline of specialized agents that reason, verify, code, fix, and visualize — all without a single API call to the cloud.

---

## ✨ What Makes This Different

Most "local AI" projects run a single model and call it agentic. DeepThinker runs an **orchestrated pipeline of 5 models** with a loop-based Actor-Critic architecture, a live code execution sandbox, and a 3D visualization engine — all on a 32 GB laptop.

| Feature | DeepThinker | Typical Local LLM App |
|---|---|---|
| Number of models | 5 (routed) | 1 |
| Code execution | ✅ Isolated polyglot sandbox | ❌ |
| Self-correction | ✅ Reflexion + Nuclear Reset | ❌ |
| 3D Visualization | ✅ Live Plotly rendering | ❌ |
| Memory / RAG | ✅ ChromaDB + SQLite | ❌ |
| Hardware | Intel iGPU / NVIDIA / Mac M-series | Usually NVIDIA only |
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
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 SIMPLE      CODING     REASONING
    │           │           │
    │    ┌──────┴──────┐    └──────────────┐
    │    ▼             ▼                   ▼
    │ DeepSeek-R1  DeepSeek-R1       DeepSeek-R1
    │ (Logic Plan) (Logic Plan)      (Draft Answer)
    │    │             │                   │
    │    ▼             ▼                   ▼
    │ Reasoning    Reasoning          Reasoning
    │ Sandbox      Sandbox ──────► Playground
    │ (Verify)     (Verify)        (Verify via Python)
    │    │             │                   │
    │    ▼             ▼                   ▼
    │ VibeThinker  VibeThinker        VibeThinker
    │ (Write Code) (Fix Logic)       (Critique & Refine)
    │    │             │                   │
    │    ▼             ▼                   ▼
    │ Execution    Execution         3D Gate Check
    │ Sandbox      Sandbox               │
    │    │         (Reflexion             ▼
    │    │          Loop ×3)    OpenCodeInterpreter
    │    │             │        (Plotly 3D Chart)
    │    ▼             ▼                   │
    └────┴─────────────┴───────────────────┘
                       │
                       ▼
              Streamed Response
          (Answer + Code + 3D Chart)
```

---

## 🧩 The Five Agents

| Agent | Model | Role | Size |
|---|---|---|---|
| **Master Router** | Phi-3.5-Mini Q6_K | Classifies intent, prompt compression, 3D gate | ~3 GB |
| **Reasoning Engine** | DeepSeek-R1-Distill-Qwen-7B Q6_K | Chain-of-thought logic, math proofs, deep analysis | ~6 GB |
| **Omni AGI Core** | VibeThinker 1.5B Q6_K | Code writing, Reflexion self-fix, Actor-Critic debate | ~1.4 GB |
| **3D Visualization** | OpenCodeInterpreter-DS 6.7B Q6_K | Plotly 3D chart generation, code execution | ~5 GB |
| **Vision Parser** | Qwen2.5-VL 7B Q6_K_XL | Multimodal image/document OCR and analysis | ~7 GB |

---

## 🛡️ Core Systems

### Dynamic Memory Allocator (DMA)
The DMA is a custom memory manager built specifically for running multiple large models on consumer hardware. It:
- Auto-detects available RAM and sets a safety threshold (25% of total)
- Uses **LRU eviction** — evicts the least recently used model when memory is tight
- Has an **iGPU Unified Memory Guard** that prevents glibc heap corruption when multiple 7B models coexist on Intel iGPUs
- Supports **lazy loading** — models load only when the pipeline actually needs them, not all at startup

### Polyglot Execution Sandbox
A three-layer isolated code execution environment:
- **Layer 1:** Process isolation (code runs in a subprocess, can't crash the server)
- **Layer 2:** Restricted builtins (custom `__import__` whitelist, no `open`/`exec`/`os`)  
- **Layer 3:** Linux resource limits (1 GB RAM cap, 120s CPU limit, 50 child process limit)
- Auto-detects and executes **Python, C, C++, Java, JavaScript, and Bash**
- Falls back to unrestricted mode if the restricted sandbox blocks a legitimate library

### Reflexion Self-Fix Loop
When generated code fails, the system doesn't just report the error — it fixes it:
1. VibeThinker attempts a shallow fix
2. If that fails, deep escalation with a stronger prompt
3. If that fails, **Nuclear Reset**: extract lessons from all failures and rewrite from scratch
4. Up to **3 complete reset cycles** before returning best-effort output

### Long-Term Memory (RAG)
- **ChromaDB** vector database for semantic similarity search
- **SQLite** fallback with keyword-based search and cosine similarity
- Stores compact solution summaries (not raw code dumps) to avoid context bloat
- Stores mistake-fix patterns to prevent regression on similar future prompts

---

## 🖥️ Hardware Compatibility

| Hardware | Status | Notes |
|---|---|---|
| **Intel Iris Xe iGPU** | ✅ Full support | Uses system RAM as VRAM via Level Zero API |
| **Intel Arc dGPU** | ✅ Full support | Same SYCL/Level Zero path |
| **NVIDIA (CUDA)** | ✅ Full support | cuBLAS acceleration via llama.cpp |
| **AMD (ROCm)** | ✅ Full support | ROCm path via PyTorch + llama.cpp |
| **Apple Silicon (M1-M4)** | ✅ Full support | Metal backend, no crashes |
| **CPU only** | ✅ Works | Slow but functional (switch to `device_mode: cpu` in settings) |

### Intel Iris Xe — Special Notes
Running 7B+ models on iGPUs is notoriously unstable. These fixes are baked into the codebase:

- **Error 45 (`UR_RESULT_ERROR_INVALID_ARGUMENT`)**: Fixed via `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in `app.py`
- **`corrupted size vs. prev_size` heap crash**: Fixed via the DMA iGPU Guard which prevents simultaneous large allocations
- **Float16 kernel failures**: Models fall back to `float32` on XPU with IPEX optimization applied automatically

---

## 🔑 Key Features In Detail

### 1 — Three-Way Intelligent Routing
Every prompt is classified into one of three execution paths by the Phi-3.5 Router before any heavy model is loaded:
- **`SIMPLE`** — Factual questions, definitions, greetings. Answered directly by the Router itself. No large model loaded.
- **`CODING`** — Any task requiring writing, debugging, or executing code in any language. Routed to the full Actor-Critic pipeline.
- **`REASONING`** — Math proofs, physics analysis, logic puzzles, deep explanations. Routed to the Playground-Verified pipeline.

A keyword-based fallback scanner ensures no prompt is misrouted even if the LLM produces an unexpected output.

---

### 2 — Actor-Critic Coding Pipeline
For coding tasks, two agents work as a check-and-balance pair:
- **DeepSeek-R1 (Actor)** produces the initial logic plan using chain-of-thought reasoning
- **VibeThinker (Critic)** reviews the plan, writes the code, and runs it
- If code fails, VibeThinker attempts a **shallow fix** first
- If shallow fix fails, a **deep escalation** rewrites the entire script
- If deep escalation fails, a **Nuclear Reset** extracts lessons from all failures and starts over from scratch with a new plan
- Up to **3 full reset cycles** are attempted before returning the best-effort output

---

### 3 — Dual Sandbox Verification
The pipeline has two distinct sandboxes that run at different stages:
- **Reasoning Sandbox** — Runs before code is written. DeepSeek-R1 writes a Python verification script to check its own logic plan (using `sympy`, `scipy`, `z3`, `astropy`, `pint`). If the plan fails logical verification, VibeThinker steps in to correct it before any code is written.
- **Execution Sandbox** — Runs the final generated code in a fully isolated subprocess with three layers of protection:
  - Process isolation (crashes can't kill the server)
  - Restricted `__import__` whitelist (no `os`, `subprocess`, `socket`)
  - Linux resource caps (1 GB RAM, 120s CPU, 50 child processes)
- Supports **Python, C, C++, Java, JavaScript, and Bash** — auto-detected from code signatures

---

### 4 — Interactive 3D Visualization Engine
After every `CODING` or `REASONING` response, a **3D Gate Check** runs automatically:
- The Router decides if the task involves mathematical graphing, physics equations, or data plots
- If yes, **OpenCodeInterpreter** generates a complete Plotly 3D visualization script
- The script is executed in the sandbox, and the resulting JSON is sent to the React frontend
- The frontend renders it as a fully **interactive 3D chart** (zoom, rotate, hover) inside the chat
- If the visualization script fails, the Reflexion loop auto-fixes it before returning the chart

---

### 5 — Dynamic Memory Allocator (DMA)
A custom memory manager that makes running 5 LLMs on consumer hardware possible:
- **Auto-calibrates** safety thresholds based on detected RAM (25% reserved for OS)
- **LRU Eviction** — When memory is tight, the least recently used model is evicted first
- **iGPU Unified Memory Guard** — Prevents glibc heap corruption on Intel Iris Xe by preventing simultaneous large allocations
- **Lazy Loading** — Models are loaded only at the moment the pipeline needs them, never all at once
- **Reads VRAM from three sources**: `torch.cuda` (NVIDIA), `torch.xpu` (Intel), and Linux sysfs (AMD/Intel without ROCm)
- **XPU Fallback** — If a model crashes on XPU during generation, it automatically falls back to CPU for that prompt

---

### 6 — Long-Term RAG Memory
The system remembers what it has solved before and uses it to improve future responses:
- **ChromaDB** vector database for semantic similarity search (primary)
- **SQLite** with cosine similarity and keyword search as fallback
- Stores **compact solution summaries** — not raw code dumps — to avoid context bloat
- Stores **mistake-fix patterns** from the Reflexion loop to prevent regressing on similar tasks
- Automatic **deduplication** — if a very similar task (>60% word overlap) is already stored, it skips saving
- Memory can be viewed (count), cleared, and inspected via the UI sidebar and REST API

---

### 7 — Web Search Integration
Optionally enriches prompts with live web context before routing:
- **Priority chain**: Google Custom Search API → SearXNG → DuckDuckGo library → DuckDuckGo HTML scraper
- Fetches top 3 snippets and prepends them to the prompt as context
- Falls back gracefully through the chain — if Google API key is missing, it uses SearXNG, then DDG
- Toggle on/off from the UI Settings panel or via the `/api/settings` endpoint

---

### 8 — Vision / Multimodal Input
Upload any image alongside a prompt for multimodal analysis:
- **Qwen2.5-VL 7B** parses the image and extracts all text, diagrams, and logic
- The extracted content is prepended to the user's prompt and passed through the full pipeline
- Works with circuit diagrams, handwritten equations, screenshots, charts, and documents

---

### 9 — Prompt Cruncher
Prevents context overflow when very long documents or code are pasted:
- Estimates token count from character length
- If the prompt exceeds the model's context limit, it **summarizes the middle section** using the Router
- Preserves the start and end of the prompt intact (where the most important context usually lives)
- Safety net caps at 50,000 characters for extreme-length inputs

---

### 10 — Live Streaming Agent Timeline
The React UI streams the pipeline status in real-time:
- Each agent step shows a status badge: `info` / `warning` / `success` / `error`
- Progress bar advances through the pipeline stages
- Logs collapse into a clean accordion once the final response arrives
- **Cancel button** stops generation mid-stream while keeping models loaded in memory for instant reuse
- **Offload Memory** button unloads all models from RAM/VRAM on demand

---

## 📦 Setup & Installation

See the dedicated guides:

| Guide | What it covers |
|---|---|
| **[README_SETUP.md](./README_SETUP.md)** | System prerequisites, Python deps, GPU acceleration (Mac/NVIDIA/Intel) |
| **[STARTUP.md](./STARTUP.md)** | Downloading models, starting the backend and frontend, troubleshooting |

---

## 🧠 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **RAM** | 16 GB | 32 GB |
| **GPU VRAM** | 8 GB (NVIDIA/AMD dGPU) | 12 GB+ |
| **Storage** | 25 GB free | 40 GB free |
| **OS** | Ubuntu 22.04 / Win 10 / macOS 13 | Ubuntu 24.04 / Win 11 / macOS 14 |
| **Python** | 3.10 | 3.11 |
| **Node.js** | 18 | 20 |

> **Intel iGPU (Iris Xe / Arc):** Uses system RAM as VRAM via the Level Zero API. 32 GB RAM strongly recommended. Always export `SYCL_DEVICE_FILTER=level_zero` before starting the backend.

---

## 📁 Project Structure

```
Team_Trenches/
├── backend/
│   ├── app.py            # FastAPI server, all REST endpoints, SSE streaming
│   ├── orchestrator.py   # Core multi-agent pipeline, DMA, Reflexion loops
│   ├── sandbox.py        # Polyglot code execution sandbox (Python/C/C++/Java/JS/Bash)
│   ├── memory.py         # Long-term RAG memory (ChromaDB + SQLite)
│   ├── search.py         # Web search (Google → SearXNG → DuckDuckGo fallback)
│   └── downloader.py     # Model downloader with retry logic and progress bar
├── frontend/
│   └── src/
│       └── App.jsx       # React UI with live streaming, 3D Plotly rendering, agent timeline
├── models/               # Downloaded GGUF weights (git-ignored, ~18 GB)
├── start.sh              # One-command Linux launcher
├── STARTUP.md            # Cross-platform startup guide
├── README_SETUP.md       # Hardware acceleration installation guide
└── requirements.txt      # Python dependencies
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | `POST` | Main chat endpoint, streams SSE status + final response |
| `/api/status` | `GET` | System health, model status, RAM/CPU usage |
| `/api/download/{model_key}` | `POST` | Trigger background model download |
| `/api/cancel` | `POST` | Stop active generation (models stay loaded) |
| `/api/offload` | `POST` | Unload all models from RAM/VRAM |
| `/api/settings` | `POST` | Update context length, temperature, GPU layers |
| `/api/memory/count` | `GET` | Number of stored long-term memories |
| `/api/memory/clear` | `POST` | Reset the vector database |

---

## 🧪 Example Prompts to Try

**3D Physics Simulation:**
> *Simulate the orbital mechanics of a binary star system. Give me the mathematical explanation, and then generate an interactive 3D visualization showing the orbital paths of the two stars.*

**Deep Reasoning + Verification:**
> *Prove that the sum of the first N natural numbers is N(N+1)/2. Verify it computationally for N=100.*

**Reflexion Loop Test:**
> *Write a Python implementation of a Red-Black Tree with insert, delete, and search operations. Include test cases.*

**Vision + Coding (upload an image):**
> *Analyze this circuit diagram and write the Python simulation for it.*

---

## 👥 Team

**Team Trenches** — India Agentic AI Open Hackathon 2026

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.
