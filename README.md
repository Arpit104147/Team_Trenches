<div align="center">
  
# 🧠 DeepThinker Multi-Agent Hub

### A Local Multi-Agent Pipeline with Dual Sandbox Verification

*Running an orchestrated fleet of 5 heavyweight LLMs simultaneously on consumer hardware*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0%2B-blue)
![License AGPL 3.0](https://img.shields.io/badge/License-AGPL%203.0-green)
![Architecture Multi-Agent MoE](https://img.shields.io/badge/Architecture-Multi--Agent%20MoE-purple)

</div>

---

DeepThinker is a production-grade, fully local multi-agent AI system that runs **five heavyweight LLMs simultaneously** on consumer hardware — including Intel Iris Xe integrated GPUs. It routes every query through a dynamic pipeline of specialized agents that reason, verify, code, fix, and visualize — all without a single API call to the cloud.

> [!CAUTION]
> ### ⚠️ Experimental Status
> This project was developed as a submission for the **India Agentic AI Open Hackathon 2026**. The dual-sandbox architecture, Reflexion loops, and Dynamic Memory Allocator (DMA) push consumer hardware to its absolute limits. 

---

## ✨ What Makes This Different

Most "local AI" projects run a single model and call it agentic. DeepThinker runs an **orchestrated pipeline of 5 models** with a loop-based Actor-Critic architecture, a live code execution sandbox, and a 3D visualization engine — all on a 32 GB laptop.

| Feature | DeepThinker | Typical Local LLM App |
|---|---|---|
| Number of models | 5 (routed) | 1 |
| Code execution | ✅ Isolated polyglot sandbox | ❌ |
| Self-correction | ✅ Reflexion + Nuclear Reset | ❌ |
| UI Artifacts | ✅ Claude-style Frontend Sandboxing | ❌ |
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
    │           ├─► Fail: VibeThinker          ├─► Fail: DeepSeek-R1
    │           │   (Fix Logic Plan)           │   (Correction Loop ×3)
    │           ▼                              ▼
    │     VibeThinker (Write Code)        3D Gate Check
    │           │                              │
    │           ▼                              ▼
    │     Execution Sandbox (Polyglot)  OpenCodeInterpreter
    │           │                       (Three.js/Plotly.js Visual)
    │           ├─► Fail: VibeThinker/R1       │
    │           │   (Reflexion Loop ×3)        │
    │           │                              │
    │           ├─► Critical Fail:             ├─► Critical Fail:
    │           │   Emergency Search           │   Emergency Search
    │           │   + Playground Correction    │   + Playground Correction
    │           ▼                              ▼
    └───────────┴──────────────────────────────┘
                       │
                       ▼
               Streamed Response
           (Answer + Code + 3D Visual)
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
- **Kaggle dGPU Hot-Swap Mode** — On dGPU environments (like the 16GB Kaggle P100 with 32GB RAM), the DMA pre-loads all models into System RAM and aggressively hot-swaps them into the GPU one-by-one, ensuring complex reasoning models get 100% of the VRAM for their KV Cache.
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

### Enterprise VRAM Multiplexing (EVM)
This codebase features **Enterprise VRAM Multiplexing (EVM)**, a hardware-aware hot-swap technology designed for environments where System RAM far exceeds discrete GPU VRAM (e.g., Kaggle instances). 

**Minimum Hardware to Automatically Activate EVM:**
- **System RAM:** ≥ 24 GB
- **GPU VRAM:** ≤ 16 GB (Discrete NVIDIA/CUDA GPU)

When the Orchestrator detects this specific hardware profile, EVM activates autonomously. It loads the entire 5-model Swarm into the host System RAM. During generation, it aggressively hot-swaps models one-by-one via the PCIe bus into the GPU, forcefully evicting dormant agents. This guarantees the active reasoning model (like DeepSeek-R1) commands 100% of the VRAM for its massive KV Cache, effectively neutralizing Out-Of-Memory crashes during deep chain-of-thought processing.

- **EVM Preloading (Load All Models):** To eliminate the initial 2-4 minute cold-start loading time during active chat, the sidebar includes a **Load All Models** button (enabled only when EVM mode is active). Clicking this triggers background parsing and loads all downloaded models into the Operating System Page Cache (System RAM). Subsequent PCIe hot-swaps during generation take only **1-2 seconds** instead of minutes.

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
- Up to **2 reflexion loops**, **1 Nuclear Reset cycle** (incorporating failure lessons), and **1 emergency web search cycle** are executed in an optimized workflow before returning the best-effort output

---

### 3 — Dual Sandbox Verification
The pipeline has two distinct sandboxes that run at different stages:
- **Reasoning Sandbox** — Runs before code is written. DeepSeek-R1 writes a Python verification script to check its own logic plan. It has access to advanced math and science tools (like `sympy`, `scipy` ODE solvers for metabolic/enzyme kinetics, `Bio` (Biopython) for genomics/sequence verification, `rdkit` for cheminformatics/bond properties, `rocketpy` for trajectory/missile physics, `qiskit` & `qutip` for quantum dynamics/circuits, `z3` theorem prover, `astropy`, and `pint`). If the plan fails logical verification, VibeThinker steps in to correct it before any code is written.
- **Execution Sandbox** — Runs the final generated code in a fully isolated subprocess with three layers of protection:
  - Process isolation (crashes can't kill the server)
  - Restricted `__import__` whitelist (no `os`, `subprocess`, `socket`)
  - Linux resource caps (2 GB RAM, 120s CPU, 200 child processes)
- **Unrestricted Fallback Mode:** If the secure restricted sandbox blocks a legitimate external library (e.g. PyTorch, external REST APIs), the orchestrator elegantly falls back to unrestricted execution, prefixing outputs with `⚠️ [Unrestricted Fallback]` to ensure transparency.
- Supports **Python, C, C++, Java, JavaScript, Bash, Go, Rust, and TypeScript** — auto-detected from code signatures

---

### 4 — Claude-Style UI Artifacts & 3D Visualization
After every `CODING` or `REASONING` response, a **3D Gate Check** runs automatically:
- The Router decides if the task involves mathematical graphing, physics equations, data plots, or biological/molecular 3D structures.
- If yes, **OpenCodeInterpreter** generates a complete, self-contained **HTML/JS Artifact** using Three.js or Plotly.js:
  - **Astrophysics/Physics:** Renders orbits, barycenter center-of-mass markers, and dynamic gravity/speed vector indicators.
  - **Genetics/Bio/Chemistry:** Renders parametric DNA double-helices (`THREE.CatmullRomCurve3`), active transport membrane channels, lipid bilayers, and molecular compounds.
  - **Controls:** Includes glassmorphic sliders to interactively tweak physical/organic settings (e.g. mass ratio, speed, pH levels, ATP concentration) in real-time.
- The artifact is sent to the React frontend, which renders it inside a **secure, isolated iframe sandbox** — exactly like Anthropic's Claude Artifacts.
- The UI features a premium glassmorphic artifact window with expand-to-fullscreen, manual hot-reloading (`↻`), and open-in-new-tab capabilities.
- **Real-time Sandbox Console Overlay:** A live slide-up drawer inside the React UI that captures `console.log`, `console.warn`, and `console.error` directly from the secure iframe, enabling instant visual debugging of client-side Javascript simulations.
- **Dual-Mode Fallback:** If the AI HTML generation fails validation, it falls back to a Python Plotly executor, running in the backend Sandbox and sending the clean JSON dataset to the frontend.

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
- **Smart Deduplication** — uses NLP stopword filtering and punctuation stripping to intelligently merge near-identical tasks (>80% content word overlap) without falsely merging unique variable tasks.
- **Concurrency-safe Architecture** — Uses dynamic connection pooling and 30-second lock timeouts for SQLite to gracefully handle multi-threaded async FastAPI requests.
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

### 9 — Context Overflow & Prompt Cruncher
Prevents context window crashes when extremely long documents or code blocks are pasted:
- **Smart Token Allocation:** Dynamically balances generation space (`max_tokens`) against prompt length. If context is tight, it automatically shrinks the generation ceiling (down to a safe minimum of 512 tokens) to preserve your prompt entirely without truncating it.
- If the prompt still exceeds the absolute context limit, it safely truncates the middle section while preserving the start and end of the prompt intact (where the most important instructions live).
- Safety net caps at 50,000 characters for extreme-length inputs to protect VRAM.

---

### 10 — Live Streaming Agent Timeline
The React UI streams the pipeline status in real-time:
- Each agent step shows a status badge: `info` / `warning` / `success` / `error`
- Progress bar advances through the pipeline stages
- Logs collapse into a clean accordion once the final response arrives
- **Cancel button** stops generation mid-stream while keeping models loaded in memory for instant reuse
- **Offload Memory** button unloads all models from RAM/VRAM on demand

---

### 11 — Predictive Playground & Live API Data
The sandboxed python environment supports live prediction and data analysis:
- **Real-Time Data Fetching:** Whitelisted network access via `requests`, `urllib`, and `http` allowing the code executor to query open REST APIs (e.g., Yahoo Finance, Open-Meteo, Alpha Vantage).
- **Exact Numerical Forecasting:** The agent can write python code utilizing `numpy` and `pandas` to run moving averages, regressions, standard deviations, and Monte Carlo trend simulations on the retrieved real-world datasets rather than guessing or hallucinating numbers.
- **Dynamic 3D Plotting:** Renders predictive forecasts into responsive WebGL (Three.js) or Plotly interactive 3D charts.

---

## 🆕 Pipeline Optimizations (June 2026 Update)

To maximize accuracy, performance, and stability on consumer hardware, the following pipeline optimizations have been implemented:

* **EVM Warm-Up Preloading:** Introduced background page-cache warming (`/api/load_all`) to speed up subsequent hot-swapping down to 1–2 seconds.
* **RAG Memory Contamination Fix:** Enhanced `memory.recall()` to selectively prioritize successful solutions over intermediate mistake logs, preventing dirty code snippets from leaking into reasoning contexts.
* **Output Bloat Control:** Integrated regex sanitization to strip raw Plotly JSON data from agent execution streams before final synthesis, preventing context window inflation.
* **Fast-Pass Error Correction:** Swapped first-pass syntax repairs to run on the lightweight 1.5B `vibethinker` model, reserving the larger 7B model for deep architectural logic rewrites.
* **Dynamic Context Scaling (RAM/VRAM-Aware):** Beyond the base 8k token limit, the context window dynamically expands up to **32k tokens** if the system has RAM/VRAM headroom (leaving a strict 5% memory safety margin). This allows the models to swallow massive web scraped pages or long local files when memory is clear.
* **Deterministic Playground Verification:** Test script writing is routed to the **Router (Phi-3.5-Mini)** rather than the verbose DeepSeek-R1-7B. This prevents DeepSeek's long `<think>...</think>` tokens from consuming the context window and causing code truncation/syntax errors.
* **Emergency Search Verification & Correction Loop:** After executing an emergency web search, the pipeline runs exactly 1 round of sandbox verification on the healed result. If it fails, DeepSeek gets the error traceback to perform a final correction round before returning the answer.
* **RAG Variable Prioritization (No Parameter Drift):** Restructured system prompts with strict instructions forcing the models to use past memories *only* for algorithmic structure, ensuring they always prioritize the active user prompt's exact velocities, coordinates, and parameters.
* **Memory Swap Safety:** Solved model-reuse memory faults by implementing safe model pointer re-acquisition in the orchestrator execution loops, completely preventing GPU memory segfaults during hot-swaps.
* **Model-Specific Physical Context Ceilings:** Prevents VRAM OOM and RoPE overflows on discrete GPUs by dynamically limiting context allocation to the models' physical training limits: `router` (8k), `vibethinker` (8k), `opencode` (16k), and `deepseek_r1` (16k).
* **Automatic CPU Fallback Loader:** Intercepts GPU context creation failures and automatically retries loading the model with `n_gpu_layers=0` (CPU mode) to keep the pipeline alive.
* **Visual Sandbox Error Overlay:** Captures client-side JavaScript/Three.js sandbox runtime exceptions and displays them inside a visual warning panel within the iframe viewport, eliminating silent black-screen failures.
* **Aligned Coding & Reasoning Loops:** Re-engineered the coding pipeline to follow the exact same nested retry structure as the reasoning pipeline (2 outer Nuclear Resets and 2 inner draft attempts per reset 0, with immediate error feedback propagation).
* **Multi-Tiered Intent Classification:** Overhauled task classification into four stages: Tier 1 structural pattern checks (raw code block/traceback fast-tracks), Tier 2 deterministic domain overrides (automatically routing data science prompts containing terms like pandas, numpy, or plotly layout to CODING), Tier 3 few-shot prompting, and Tier 4 fallback scans.
* **Auto-pip Package Installer:** The sandbox executor now dynamically detects runtime `ModuleNotFoundError` during code execution, triggers a pip installation on the fly, and re-executes the script seamlessly.
* **Interactive 3D Gate Suggestion:** Rather than forcing rendering overhead on every compatible prompt, the orchestrator detects 3D options and prompts the user dynamically, allowing them to trigger 3D generation by responding `generate 3d`.

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

> **Intel iGPU (Iris Xe / Arc):** Uses system RAM as VRAM via the Level Zero API. Works with **16 GB RAM** minimum (using aggressive LRU model swapping), but 32 GB is recommended for maximum speed. Always export `SYCL_DEVICE_FILTER=level_zero` before starting the backend.

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
| `/api/load_all` | `POST` | Pre-load all downloaded models into RAM/page cache |
| `/api/settings` | `POST` | Update context length, temperature, GPU layers |
| `/api/memory/count` | `GET` | Number of stored long-term memories |
| `/api/memory/clear` | `POST` | Reset the vector database |

---

## 🧪 Example Prompts to Try

**3D Physics Simulation:**
> *Simulate the orbital mechanics of a binary star system. Give me the mathematical explanation, and then generate an interactive 3D visualization showing the orbital paths of the two stars.*

**Live Prediction Playground (Weather / Stocks):**
> *Fetch the current weather outlook for Mumbai and use polynomial regression to predict the temperature curve for the next 7 days, plotting it in an interactive 3D WebGL chart.*
>
> *Fetch historical AAPL stock prices from the web, compute a 30-day Exponential Moving Average (EMA) to project next week's buy/sell targets, and render a 3D Plotly visual.*

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

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
