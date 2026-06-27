<div align="center">
  
# 🧠 DeepThinker Multi-Agent Hub

### A Fully Local Multi-Agent AI System with Dual Sandbox Verification & Dynamic Hardware Scaling

*Running an orchestrated fleet of specialized LLMs on local hardware — from Intel iGPUs to NVIDIA H100s*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Architecture Multi-Agent MoE](https://img.shields.io/badge/Architecture-Multi--Agent%20MoE-purple)
![Hardware Adaptive](https://img.shields.io/badge/Hardware-Adaptive-orange)

</div>

---

DeepThinker is a **production-grade, fully local multi-agent AI system** designed to run multiple specialized LLMs locally on consumer and enterprise hardware. By leveraging a high-performance **6-Way Routing Pipeline**, DeepThinker analyzes user intent to route queries through optimal reasoning, coding, math, predictive, and visual paths—**all executing 100% offline with zero cloud API dependencies.**

At startup, the system auto-calibrates to the available compute environment (CUDA, Intel XPU/SYCL, or CPU), scaling context windows, batch sizes, scraping depth, memory thresholds, and execution configurations dynamically.

> [!CAUTION]
> ### ⚠️ Experimental Status
> This project was developed as a submission for the **India Agentic AI Open Hackathon 2026**. The dual-sandbox architecture, Reflexion loops, and Dynamic Memory Allocator (DMA) push consumer hardware to its absolute limits.

---

## 🚀 Key Highlights & Achievements

* **India Agentic AI Open Hackathon 2026 Submission** (Shortlisting Round: July 9, 2026)
* **6-Way Intelligent Orchestration:** Transitions away from simple chat/search toggles to an intent-aware routing network.
* **Dual-Sandbox Code & Logic Verification:** Built-in isolated polyglot runtime environment verifying code outputs across 9 languages.
* **Dynamic Memory Allocator (DMA):** Out-of-memory (OOM) protection allowing large models (up to 7B parameters) to run seamlessly on consumer hardware (e.g. 16GB RAM / Intel Iris Xe iGPUs) using LRU-based swapping.
* **Claude-Style Interactive Visual Sandbox:** Generates live, glassmorphic Three.js 3D physics engines and Plotly.js charts rendered in secure iframe sandboxes.

---

## 🤖 6-Way Agentic Pipeline Architecture

```
                       User Prompt + Uploaded Image
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │ Qwen-2.5-VL Vision Parser │ (Extract text/diagrams)
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Phi-3.5-Mini Router    │ (Intent & Mode Classifier)
                      └─────────────┬─────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬────────────────┬──────────────┐
    ▼               ▼               ▼               ▼                ▼              ▼
  [PATH A]       [PATH B]        [PATH C]        [PATH D]         [PATH E]       [PATH F]
   SIMPLE         CODING        REASONING       PREDICTION      EXTREME SEARCH    3D VIZ
    │               │               │               │                │              │
    ▼               ▼               ▼               ▼                ▼              ▼
Phi-3.5-Mini   OpenCodeDS      VibeThinker 3B  OpenCodeDS       DeepSeek R1    OpenCodeDS
 Direct Ans    Actor-Critic     Reasoning Plan  ML Regression   Deep Analysis   WebGL & JS
    │               │               │               │                │              │
    ▼               ▼               ▼               ▼                ▼              ▼
Web Search    Execution       Sandbox         Data Cleaning    OpenCodeDS     Reflexion
 (Simple)     Sandbox         Verification     Loop (SciPy)    Plotly Charts  Sandbox Loop
    │               │               │               │                │              │
    └───────────────┴───────────────┴───────────────┴────────────────┴──────────────┘
                                    │
                                    ▼
                             Streamed Response
                         (Answer + Code + 3D Visual)
```

```mermaid
flowchart TD
    %% ── INPUT & VISION ──
    USER([👨‍💻 User Prompt & Uploaded Image]) --> INPUT{Has Image?}
    
    INPUT -->|Yes| QWEN_VL[👁️ Qwen-2.5-VL 7B Vision Parser]
    INPUT -->|No| ROUTER
    QWEN_VL -->|Transcribed Text Context| ROUTER[🧭 Phi-3.5 Mini Intent Router]

    %% ── INTENT ROUTER ──
    ROUTER -->|Classifies Intent| PATHS{Select Pathway}

    %% ── 6-WAY PIPELINE PATHWAYS ──
    PATHS -->|1. SIMPLE| PATH_SIMPLE[General & Simple Queries]
    PATHS -->|2. CODING| PATH_CODING[Software Engineering Tasks]
    PATHS -->|3. REASONING| PATH_REASONING[Mathematical & Logic Problems]
    PATHS -->|4. PREDICTION| PATH_PREDICTION[Data Analysis & Forecasting]
    PATHS -->|5. EXTREME SEARCH| PATH_EXTREME[Deep Research & Document Synthesis]
    PATHS -->|6. 3D VIZ| PATH_3D[WebGL & Three.js 3D Simulations]

    %% ── PATHWAY INTERNAL LOOPS ──
    subgraph PathSimple [Simple Pipeline]
        PATH_SIMPLE --> DDG[🌐 DuckDuckGo Search]
        DDG --> PHI_ANS[Phi-3.5 Mini Direct Answer]
    end

    subgraph PathCoding [Coding Pipeline]
        PATH_CODING --> DS_DRAFT[⚡ DeepSeek R1 Logic Draft]
        DS_DRAFT --> OC_GEN[💻 OpenCodeInterpreter 7B Gen]
        OC_GEN --> PY_SANDBOX{🛡️ Isolated Polyglot Sandbox}
        PY_SANDBOX -->|Code Error| OC_FIX[OpenCodeInterpreter Fix Loop]
        OC_FIX --> PY_SANDBOX
        PY_SANDBOX -->|Passed| CODING_OUT[Verified Code Output]
    end

    subgraph PathReasoning [Reasoning Pipeline]
        PATH_REASONING --> VT_PLAN[🧠 VibeThinker 3B Math Logic Plan]
        VT_PLAN --> RS_VERIFY{🧮 Reasoning Sandbox Verify}
        RS_VERIFY -->|Math Assertion Fail| VT_PLAN
        RS_VERIFY -->|Passed| REASONING_OUT[Logical Proof & Derivations]
    end

    subgraph PathPrediction [Prediction Pipeline]
        PATH_PREDICTION --> PREDICT_SCRAPE[🌐 DuckDuckGo Web Scrape]
        PREDICT_SCRAPE --> SK_PLAN[💻 OpenCodeInterpreter 7B Data Sci Script]
        SK_PLAN --> DS_SANDBOX{📊 Scikit-Learn Data Cleaning Sandbox}
        DS_SANDBOX -->|NaN / Infinite Residues| SK_CLEAN[Data Cleaning Loop]
        SK_CLEAN --> DS_SANDBOX
        DS_SANDBOX -->|Passed| PREDICT_OUT[Forecasting Plots & Stats]
    end

    subgraph PathExtreme [Extreme Search Pipeline]
        PATH_EXTREME --> EXT_SCRAPE[🌐 Multi-Source Deep Search]
        EXT_SCRAPE --> DS_R1_SYN[⚡ DeepSeek R1 Synthesize & Compare]
        DS_R1_SYN --> OC_PLOT[💻 OpenCode Plotly Script Gen]
        OC_PLOT --> PLOTLY_SANDBOX{📈 Plotly JSON Execution Sandbox}
        PLOTLY_SANDBOX -->|Plotly Error| OC_PLOT
        PLOTLY_SANDBOX -->|Passed| EXTREME_OUT[Deep Analytical Report + Interactive Charts]
    end

    subgraph Path3D [3D Visualization Pipeline]
        PATH_3D --> OC_WebGL[💻 OpenCode HTML/WebGL Canvas Gen]
        OC_WebGL --> WebGL_SANDBOX{🎨 Three.js / Plotly.js Sandbox}
        WebGL_SANDBOX -->|WebGL / DOM Error| OC_WebGL
        WebGL_SANDBOX -->|Passed| WebGL_OUT[Live Interactive Glassmorphic Simulation]
    end

    %% ── CONTEXT MEMORY RAG ──
    CHROMA[(🗄️ ChromaDB Vector RAG)] <-->|Save / Recall Logic| ROUTER

    %% ── HARDWARE CONTROL ──
    subgraph Hardware [Hardware Allocation Engine]
        DMA[⚖️ Dynamic Memory Allocator]
        DMA <-->|LRU Model Swap: RAM ↔ VRAM| GPU_VRAM[🟩 GPU VRAM / Intel XPU]
    end

    PHI_ANS & CODING_OUT & REASONING_OUT & PREDICT_OUT & EXTREME_OUT & WebGL_OUT --> OUT_RENDER[💻 React Frontend UI]

    %% ── STYLING ──
    classDef default fill:#1E1E1E,stroke:#4A4A4A,stroke-width:2px,color:#FFF;
    classDef router fill:#5E2750,stroke:#9C27B0,stroke-width:2px,color:#FFF;
    classDef sandbox fill:#E65100,stroke:#FF9800,stroke-width:2px,color:#FFF;
    classDef memory fill:#1B5E20,stroke:#4CAF50,stroke-width:2px,color:#FFF;
    classDef hardware fill:#45475a,stroke:#94e2d5,stroke-width:2px,color:#FFF;
    
    class ROUTER,PATHS router;
    class PY_SANDBOX,RS_VERIFY,DS_SANDBOX,PLOTLY_SANDBOX,WebGL_SANDBOX sandbox;
    class CHROMA memory;
    class DMA,GPU_VRAM hardware;
```

### Routing Pathways

1. **SIMPLE:** Fast, direct answers utilizing Phi-3.5-Mini. Best for general conversations and basic tasks.
2. **CODING:** OpenCodeInterpreter 6.7B acts as the generator, with an execution-driven feedback loop verifying code correctness.
3. **REASONING:** Powered by VibeThinker 3B, providing chain-of-thought mathematical and logic verification.
4. **PREDICTION:** A specialized ML pipeline that scrapes target web sources, cleans data frames, performs model fitting (Scikit-Learn/SciPy), and outputs regression/forecast statistics.
5. **EXTREME WEBSEARCH:** Employs DeepSeek R1-7B to perform deep thematic synthesis over scraped pages, feeding key data arrays into OpenCodeInterpreter to produce Plotly visualizations.
6. **3D VISUALIZATION:** Generates interactive Three.js physics scenes or Plotly.js canvases rendered directly in the web frontend.

---

## 🧩 The Fleet of Local Models

| Model Key | Base Model | Parameter Size | Quantization | Primary Role |
|---|---|---|---|---|
| **router** | Phi-3.5-Mini-Instruct | 3.8B | Q6_K | Intent classification, routing decision, search query generation |
| **vibethinker** | WeiboAI VibeThinker 3B | 3.0B | Q6_K | Mathematical reasoning, logic puzzles, step-by-step verification |
| **deepseek_r1** | DeepSeek-R1-Distill-Qwen-7B | 7.0B | Q6_K | Complex multi-turn logic, chain-of-thought analysis, data synthesis |
| **opencode** | OpenCodeInterpreter-DS-6.7B | 6.7B | Q6_K | High-quality code generation, ML scripting, 3D HTML artifact creation |
| **qwen_vl** | Qwen-2.5-VL-7B-Instruct | 7.0B | Q6_K_XL + mmproj | Vision parsing, text/code transcription from screenshots and charts |

---

## 🛡️ Key System Components

### 1 — Dynamic Memory Allocator (DMA)
Designed to make local multi-model pipelines possible on resource-constrained systems:
* **LRU Model Eviction:** Hot-swaps models dynamically from System RAM to GPU VRAM to ensure only one active model consumes GPU memory.
* **Unified Memory Guard:** Protects Intel Iris Xe iGPUs (running via Level Zero APIs) from memory allocation failures and host crash issues.
* **Automatic CPU Fallback:** Gracefully redirects model execution to CPU threads if VRAM allocations fail.

### 2 — Dynamic Scaling
Automatically scales execution parameters based on detected GPU compute capability:
* **H100 / B200:** Context ceiling up to 64K tokens, maximum search query depth.
* **RTX 3090 / 4090 / Intel Arc:** Context ceiling up to 32K tokens.
* **T4 / Intel Iris Xe:** Caps context ceiling to 8K-16K tokens, restricts batch sizes (`n_batch=512`, `n_ubatch=256`), and disables flash attention to maintain system stability.

### 3 — Isolated Polyglot Sandbox
Executes generated code safely inside a 3-tier isolated environment:
* **Languages Supported:** Python, C, C++, Java, JavaScript, Go, Rust, Bash, TypeScript.
* **Libraries Pre-Whitelisted:** `numpy`, `scipy`, `pandas`, `sklearn`, `plotly`, `sympy`, `networkx`, `z3-solver`.
* **Reflexion Loop:** Code syntax or runtime errors automatically spawn self-correction loops to repair the code before outputting results.

---

## 🖥️ System Requirements & Setup

| Resource | Minimum | Recommended |
|---|---|---|
| **RAM** | 16 GB | 32 GB |
| **Storage** | 25 GB free space | 45 GB free space |
| **OS** | Ubuntu 22.04+ / Windows 11 / macOS 14 | Ubuntu 24.04 / macOS 15 |
| **GPU** | NVIDIA/Intel/AMD (8GB VRAM) | NVIDIA RTX 3090/4090 or Apple Silicon |

### Quick Start (Linux/macOS)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Bshdhorrhh/Team_Trenches.git
   cd Team_Trenches
   ```

2. **Setup Python Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies:**
   ```bash
   cd frontend && npm install && cd ..
   ```

4. **Launch Backend and Frontend:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

Open `http://localhost:5173` in your web browser. Model weights will download automatically upon first request or can be pre-downloaded via settings.

---

## 👥 Team
**Team Trenches** — Submission for the India Agentic AI Open Hackathon 2026.
