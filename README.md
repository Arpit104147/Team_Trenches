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

## 🛠️ Complete Technology Stack

The system is constructed with a highly decoupled client-server architecture utilizing the following tools:

### Frontend (User Interface)
* **Framework:** React 19 (Functional components, hooks) with Vite (Ultra-fast build/dev tooling)
* **Styling & Layout:** Vanilla CSS (Glassmorphism, custom dark theme, premium transitions, responsive flexbox/grid)
* **Icons:** Lucide React
* **Markdown & Parsing:** `react-markdown` (supports live rendering of LaTeX blocks, code tables, and syntax highlighting)
* **Interactive Visualization:** Dynamic loading inside an isolated iframe sandbox utilizing `Plotly.js` and `Three.js` (WebGL)

### Backend (Model Orchestrator & API Gateway)
* **Framework:** FastAPI (Asynchronous, type-safe REST API endpoint validation)
* **ASGI Server:** Uvicorn (Asynchronous web runner)
* **Model Inference Engine:** Local `llama.cpp` server & HuggingFace integration (`transformers`, `accelerate`)
* **Vector Database (RAG):** ChromaDB (Local SQLite-backed semantic vector storage for indexing past verified solutions)
* **Real-time Web Access:** DuckDuckGo Search API (`ddgs`)
* **System Metrics Monitoring:** `psutil` (Tracks active processes, CPU cores, and RAM utilization for EVM hot-swapping)

### Verification Sandbox (Pre-Whitelisted Scientific & Cryptographic Libraries)
* **Scientific Computing & Math:** `numpy`, `scipy` (Numerical solvers & ODE integrations), `sympy` (Symbolic algebraic solver), `pint` (Physical units validation)
* **Theorem Proving & Constraints:** `z3-solver` (Z3 SMT Solver for math/logic boundaries)
* **Biological & Molecular Physics:** `biopython`, `rdkit` (Molecular structures and sequence parsing)
* **Data Science & ML:** `scikit-learn` (Regression, regression trees, forecasting), `statsmodels` (Time series models)
* **Space Dynamics & Astrodynamics:** `astropy`, `rocketpy`
* **Quantum Computing Simulations:** `qiskit`, `qutip`
* **Cybersecurity & Cryptography:** `cryptography` (Fernet, AES, RSA), `scapy` (Interactive packet crafting/sniffing), `pyjwt`, `pycryptodome`

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
    %% ── TOP-LEVEL INGESTION ──
    USER([User Prompt]) --> ROUTER["Router: Phi-3.5 Mini"]
    
    ROUTER -->|Search Query Triggered| OPT_QUERY["Phi 3.5 Mini: Generate optimized query"]
    OPT_QUERY --> SCRAPE["Scrape Web Snippets & Live Data"]
    SCRAPE --> CLASSIFY["Phi-3.5 Mini: Intent Classification"]
    
    ROUTER -->|No Search| CLASSIFY

    %% ── Intent Classification Branches ──
    CLASSIFY --> PATH_SIMPLE["1. SIMPLE"]
    CLASSIFY --> PATH_REASONING["2. REASONING"]
    CLASSIFY --> PATH_CODING["3. CODING"]
    CLASSIFY --> PATH_PREDICT["4. PREDICTION"]
    CLASSIFY --> PATH_3D["5. 3D VIZ"]
    CLASSIFY --> PATH_EXTREME["6. EXTREME WEBSEARCH"]

    %% ── 1. SIMPLE PATHWAY ──
    PATH_SIMPLE --> SIMPLE_ANS["Phi-3.5 Mini: Answer directly with web context"]

    %% ── 2. REASONING PATHWAY ──
    PATH_REASONING --> VT_DRAFT["VibeThinker 3B: Draft Math/Logic Plan"]
    VT_DRAFT --> OC_TEST["OpenCodeInterpreter 7B: Write Python Verification Script"]
    OC_TEST --> REASON_SB{"Execution Sandbox"}
    
    REASON_SB -->|Verified Success| REASON_PASS["Pass logic plan to Coding or Output"]
    REASON_SB -->|Crash/Failure| REASON_FAIL["Emergency Web Search"]
    REASON_FAIL --> DS_MATH_FIX["DeepSeek R1-7B: Correct logic plan using SearchQA web facts"]
    DS_MATH_FIX --> REASON_SB

    %% ── 3. CODING PATHWAY ──
    PATH_CODING --> OC_DRAFT["OpenCodeInterpreter 7B: Draft Python/JS Script"]
    OC_DRAFT --> CODING_SB{"Execution Sandbox"}
    
    CODING_SB -->|Verified Success| CODE_PASS["Output final Verified Code Block"]
    CODING_SB -->|Syntax Error| CODE_LINT["OpenCode Linter: Patch specific error"]
    CODE_LINT --> CODING_SB
    
    CODING_SB -->|Logic Error| CODE_FIX["Reflexion Loop: OpenCode attempts logic fix"]
    CODE_FIX --> CODING_SB
    CODE_FIX -->|Escalate| DS_CODE_FIX["DeepSeek R1-7B corrects code using traceback"]
    DS_CODE_FIX --> CODING_SB

    %% ── 4. PREDICTION PATHWAY ──
    PATH_PREDICT --> P_VRAM["Expand VRAM Context Limits"]
    P_VRAM --> P_SPEC["OpenCodeInterpreter 7B: ML/Data Science Specialization Prompt"]
    P_SPEC --> P_DRAFT["Draft Pandas/Scikit-learn Regression Script"]
    P_DRAFT --> P_SB{"Sandbox Execution: Load Web Scraped Arrays"}
    
    P_SB -->|Verified Success| P_PASS["Parse 'PREDICTIVE_METRICS' JSON & Render Forecast UI"]
    P_SB -->|Data Format Error/NaN| P_CLEAN["Data Cleaning Loop: OpenCode fixes array shapes"]
    P_CLEAN --> P_SB

    %% ── 5. 3D VIZ PATHWAY ──
    PATH_3D --> VIZ_DRAFT["OpenCodeInterpreter 7B: Generate HTML (JS WebGL)"]
    VIZ_DRAFT --> VIZ_SB{"Node.js Sandbox: Verify syntax and DOM API logic"}
    
    VIZ_SB -->|Syntax Error| VIZ_LINT["OpenCode Linter: Fix JavaScript Error"]
    VIZ_LINT --> VIZ_SB
    VIZ_SB -->|Success| VIZ_PASS["Output Interactive HTML to Frontend Frame"]

    %% ── 6. EXTREME WEBSEARCH PATHWAY ──
    PATH_EXTREME --> EXT_VRAM["Expand VRAM to Absolute Max Limit"]
    EXT_VRAM --> DS_COMPARE["DeepSeek R1-7B: Deep Document Comparison & Data Structuring"]
    DS_COMPARE --> EXT_REPORT["Generate Comprehensive Analytical Report"]
    EXT_REPORT --> EXT_PASS_OC["Pass structured data to OpenCodeInterpreter"]
    EXT_PASS_OC --> EXT_PLOT["Draft Python Plotly Script: Piecharts, Barcharts, Trends"]
    EXT_PLOT --> EXT_SB{"Execution Sandbox: Verify JSON Output"}
    EXT_SB -->|Success| EXT_PASS["Output Deep Analysis Report + Interactive Charts"]

    %% ── FINAL RENDERING TERMINUS ──
    SIMPLE_ANS & REASON_PASS & CODE_PASS & P_PASS & VIZ_PASS & EXT_PASS --> RENDER_UI["💻 React Frontend UI / Chat Output"]

    %% ── STYLING ──
    classDef default fill:#1E1E1E,stroke:#4A4A4A,stroke-width:2px,color:#FFF;
    classDef gateway fill:#2D3748,stroke:#4A5568,stroke-width:2px,color:#FFF;
    classDef routing fill:#5E2750,stroke:#9C27B0,stroke-width:2px,color:#FFF;
    classDef sandbox fill:#E65100,stroke:#FF9800,stroke-width:2px,color:#FFF;
    classDef terminal fill:#1B5E20,stroke:#4CAF50,stroke-width:2px,color:#FFF;
    
    class USER,ROUTER,OPT_QUERY,SCRAPE,CLASSIFY gateway;
    class PATH_SIMPLE,PATH_REASONING,PATH_CODING,PATH_PREDICT,PATH_3D,PATH_EXTREME routing;
    class REASON_SB,CODING_SB,P_SB,VIZ_SB,EXT_SB sandbox;
    class RENDER_UI terminal;
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
