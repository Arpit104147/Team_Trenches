<div align="center">

# 🧠 DeepThink AIOS

### Fully Local Multi-Agent AI Operating System

*An orchestrated fleet of specialized LLMs running on consumer hardware — zero cloud dependencies.*

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18.0%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Architecture Multi-Agent](https://img.shields.io/badge/Architecture-Multi--Agent-purple)

</div>

---

DeepThink AIOS is a **production-grade, fully offline multi-agent system** that routes user queries through specialized LLM pipelines for coding, reasoning, data science, 3D visualization, and semiconductor chip design — all running locally with dynamic hardware scaling from Intel iGPUs to NVIDIA H100s.

> [!CAUTION]
> This project is in active development. The multi-sandbox architecture and Dynamic Memory Allocator push consumer hardware to its limits.

---

## ✨ Key Features

- **7-Way Intelligent Routing** — Intent-aware pipeline selection across coding, reasoning, prediction, search, 3D viz, and chip design
- **Self-Scaffolding Code Generation** — Ornith 9B autonomously plans and writes code in a single unified trajectory
- **AST-Aware Self-Healing** — Surgical patching via Python AST extraction instead of fixed-line windows
- **Parallel Web Scraping** — ThreadPoolExecutor-based concurrent page fetching (N×timeout → 1×timeout)
- **Dual Sandbox Verification** — Polyglot execution across 13 languages with kernel-level isolation
- **Chip Design EDA Pipeline** — Full Verilog/SPICE synthesis with SkyWater 130nm PDK mapping
- **Dynamic Memory Allocator (DMA)** — LRU model swapping enabling 7B+ models on 16GB RAM systems

---

## 🤖 Model Fleet

| Model | Size | Role |
|---|---|---|
| **Ornith 1.0-9B** | 9B (Q6_K) | Primary code generation, 3D visualization, ML scripting, self-correction |
| **DeepSeek-R1-7B** | 7B (Q6_K) | Deep reasoning, chain-of-thought, logic planning, pedagogical synthesis |
| **VibeThinker 3B** | 3B (Q6_K) | Agent IDE syntax linter — surgical AST-aware patching |
| **Phi-3.5-Mini** | 3.8B (Q6_K) | Intent classification, routing, search query generation |
| **Qwen-2.5-VL-7B** | 7B (Q6_K_XL) | Vision parsing, OCR, screenshot/chart transcription |

---

## 🔀 Pipeline Architecture

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
    CLASSIFY --> PATH_CODING["2. CODING"]
    CLASSIFY --> PATH_REASONING["3. REASONING (PAL)"]
    CLASSIFY --> PATH_PREDICT["4. PREDICTION"]
    CLASSIFY --> PATH_3D["5. 3D VIZ"]
    CLASSIFY --> PATH_EXTREME["6. EXTREME WEBSEARCH"]
    CLASSIFY --> PATH_CHIP["7. CHIP DESIGN"]

    %% ── 1. SIMPLE PATHWAY ──
    PATH_SIMPLE --> SIMPLE_ANS["Phi-3.5 Mini: Answer directly with web context"]

    %% ── 2. REASONING PATHWAY (PAL) ──
    PATH_REASONING --> REASON_BRANCH{"Playground Verifiable?"}
    REASON_BRANCH -->|Yes| PAL_DRAFT["Ornith 9B: Write SymPy/Verification Script"]
    PAL_DRAFT --> PAL_SB{"Execution Sandbox"}
    
    PAL_SB -->|Verified Success| DS_SYNTH["DeepSeek R1-7B: Pedagogical LaTeX Synthesis"]
    DS_SYNTH --> REASON_PASS["Pass final verified math solution"]
    
    PAL_SB -->|Syntax/Linter Error| VT_LINT["VibeThinker 3B: Rapid Agent IDE patch"]
    VT_LINT --> PAL_SB
    
    PAL_SB -->|Logic / Formula Error| DS_FIX["DeepSeek R1-7B: Adjust logic & retry"]
    DS_FIX --> PAL_DRAFT

    REASON_BRANCH -->|No| DS_THEORY["DeepSeek R1-7B: Direct detailed academic LaTeX derivation"]
    DS_THEORY --> REASON_PASS

    %% ── 3. CODING PATHWAY ──
    PATH_CODING --> C_DRAFT["Ornith 9B: Self-Scaffold & Code Generation"]
    C_DRAFT --> CODING_SB{"Execution Sandbox"}
    
    CODING_SB -->|Verified Success| CODE_PASS["Output final Verified Code Block"]
    
    CODING_SB -->|Syntax/Linter Error| VT_CODE_LINT["VibeThinker 3B: Agent IDE surgical patch"]
    VT_CODE_LINT --> CODING_SB
    
    CODING_SB -->|Logic / Runtime Bug| C_FIX["Ornith 9B: Logic self-correction loop"]
    C_FIX --> C_DRAFT
    
    CODING_SB -->|"Escalation (Max Retries)"| DS_CODE_FIX["DeepSeek R1-7B: Emergency traceback patch"]
    DS_CODE_FIX --> CODING_SB

    %% ── 4. PREDICTION PATHWAY ──
    PATH_PREDICT --> P_VRAM["Expand VRAM Context Limits"]
    P_VRAM --> P_DRAFT["Ornith 9B: Draft Pandas/Scikit-learn Regression Script"]
    
    P_DRAFT -->|Empty / Failed Draft| P_DS_FALLBACK["DeepSeek R1-7B Fallback Draft"]
    P_DRAFT -->|Successful Draft| P_SB{"Sandbox Execution"}
    P_DS_FALLBACK --> P_SB
    
    P_SB -->|Verified Success| P_PASS["Parse 'PREDICTIVE_METRICS' JSON & Render Forecast UI"]
    P_SB -->|Partial Success| P_BEST_EFFORT["Return Best-Effort Text Results"]
    P_SB -->|Syntax/Linter Error| P_LINT["VibeThinker 3B: Agent IDE surgical patch"]
    P_LINT --> P_SB
    
    P_SB -->|Runtime / Logic Error| P_CLEAN["Data Cleaning Loop: Ornith full rewrite"]
    P_CLEAN --> P_SB
    
    REASON_PASS & CODE_PASS & P_PASS & P_BEST_EFFORT --> P_3D_GATE{"Triggers 3D Visuals?"}
    P_3D_GATE -->|Yes| VIZ_DRAFT
    P_3D_GATE -->|No| RENDER_UI

    %% ── 5. 3D VIZ PATHWAY ──
    PATH_3D --> VIZ_DRAFT["Ornith 9B: Generate HTML (JS WebGL / Three.js)"]
    VIZ_DRAFT --> VIZ_SB{"Node.js Sandbox: Verify syntax and DOM API logic"}
    
    VIZ_SB -->|Syntax Error| VIZ_LINT["VibeThinker 3B: Agent IDE patch"]
    VIZ_LINT --> VIZ_SB
    VIZ_SB -->|DOM/Logic Error| VIZ_FIX["Ornith 9B: Fix JS execution logic"]
    VIZ_FIX --> VIZ_DRAFT
    VIZ_SB -->|Success| VIZ_PASS["Output Interactive HTML to Frontend Frame"]

    %% ── 6. EXTREME WEBSEARCH ──
    PATH_EXTREME --> EXT_VRAM["Expand VRAM to Absolute Max Limit"]
    EXT_VRAM --> DS_COMPARE["DeepSeek R1-7B: Deep Comparison & Data Structuring"]
    DS_COMPARE --> EXT_REPORT["Generate Comprehensive Analytical Report"]
    EXT_REPORT --> EXT_PLOT["DeepSeek R1-7B: Draft Plotly Script directly"]
    EXT_PLOT --> EXT_SB{"Execution Sandbox: Verify JSON Output"}
    EXT_SB -->|Success| EXT_PASS["Output Deep Analysis Report + Interactive Charts"]

    %% ── 7. CHIP DESIGN PATHWAY ──
    PATH_CHIP --> CHIP_ARCH["DeepSeek R1-7B: Architecture Decomposition"]
    CHIP_HDL{"Analog or Digital?"}
    CHIP_ARCH --> CHIP_HDL
    CHIP_HDL -->|Digital| CHIP_VERILOG["DeepSeek R1-7B: Verilog RTL + Testbench"]
    CHIP_HDL -->|Analog| CHIP_SPICE["DeepSeek R1-7B: SPICE Netlist"]
    CHIP_VERILOG --> CHIP_EDA{"iverilog + Yosys Sandbox"}
    CHIP_SPICE --> CHIP_NGSPICE{"Ngspice Sandbox"}
    
    CHIP_EDA -->|Syntax/Compile Error| CHIP_VT_LINT["VibeThinker 3B: Agent IDE patch"]
    CHIP_VT_LINT --> CHIP_EDA
    CHIP_EDA -->|DRC/Logic Error| CHIP_FIX["Reflexion: DeepSeek R1 Auto-correct HDL"]
    CHIP_FIX --> CHIP_EDA
    
    CHIP_EDA -->|Verified| CHIP_3D["Ornith 9B: Three.js 3D Chip Layer Viz"]
    CHIP_NGSPICE --> CHIP_3D
    CHIP_3D --> CHIP_PASS["Output HDL + Synthesis Stats + 3D Visual"]

    %% ── FINAL RENDERING TERMINUS ──
    SIMPLE_ANS & VIZ_PASS & EXT_PASS & CHIP_PASS --> RENDER_UI["💻 React Frontend UI / Chat Output"]

    %% ── STYLING ──
    classDef default fill:#1E1E1E,stroke:#4A4A4A,stroke-width:2px,color:#FFF;
    classDef gateway fill:#2D3748,stroke:#4A5568,stroke-width:2px,color:#FFF;
    classDef routing fill:#5E2750,stroke:#9C27B0,stroke-width:2px,color:#FFF;
    classDef sandbox fill:#E65100,stroke:#FF9800,stroke-width:2px,color:#FFF;
    classDef terminal fill:#1B5E20,stroke:#4CAF50,stroke-width:2px,color:#FFF;
    
    class USER,ROUTER,OPT_QUERY,SCRAPE,CLASSIFY gateway;
    class PATH_SIMPLE,PATH_CODING,PATH_REASONING,PATH_PREDICT,PATH_3D,PATH_EXTREME,PATH_CHIP,P_3D_GATE routing;
    class REASON_SB,CODING_SB,P_SB,VIZ_SB,EXT_SB,CHIP_EDA,CHIP_NGSPICE,PAL_SB sandbox;
    class RENDER_UI terminal;
```

### Pipeline Details

| # | Pipeline | Generator | Linter | Description |
|---|---|---|---|---|
| 1 | **Simple** | Phi-3.5 | — | Direct answers with optional web context |
| 2 | **Coding** | Ornith 9B | VibeThinker | Self-scaffolded code gen → Sandbox verify → AST patch loop |
| 3 | **Reasoning** | DeepSeek-R1 | VibeThinker | SymPy/SciPy verification scripts → LaTeX synthesis by R1 |
| 4 | **Prediction** | Ornith 9B | VibeThinker | ML regression with pandas/scikit-learn, data cleaning loops |
| 5 | **Extreme Search** | DeepSeek-R1 | — | Parallel scraping + deep thematic synthesis + Plotly charts |
| 6 | **3D Visualization** | Ornith 9B | VibeThinker | Three.js / Plotly.js interactive scenes in iframe sandbox |
| 7 | **Chip Design** | Ornith 9B + R1 | VibeThinker | 3-stage EDA: Architecture → HDL verify → 3D chip layout |

---

## 🛡️ System Components

### Sandbox Isolation
- **3-Tier Security:** Linux `unshare` namespaces + `chroot` jailing + resource limits
- **13 Languages:** Python, C, C++, Java, JS, Go, Rust, Bash, TS, Verilog, SystemVerilog, SPICE, Yosys TCL
- **Pre-Execution SAST:** Static security scanning blocks injection, reverse shells, and exfiltration

### Self-Healing Loop
```
Draft Code → Sandbox Execute → [Success] → Output
                                [Failure] ↓
                    AST Context Extraction (exact broken function)
                                ↓
                    VibeThinker: Surgical Search/Replace Patch
                                ↓
                    [Fixed] → Re-execute → Output
                    [Failed] → Ornith Full Rewrite → Re-execute
                    [Failed] → DeepSeek-R1 Escalation → Nuclear Reset
```

### Dynamic Memory Allocator (DMA)
- **LRU Eviction:** Hot-swaps models between VRAM ↔ System RAM
- **KV Cache Quantization:** INT8 KV cache halves VRAM usage
- **GPU Offloading:** KV cache pinned to VRAM via `offload_kqv`
- **Auto-Scaling:** Context windows scale from 8K (iGPU) → 64K (H100)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, Vite, Vanilla CSS (Glassmorphism), react-markdown, Plotly.js, Three.js |
| **Backend** | FastAPI, Uvicorn, llama-cpp-python, ChromaDB (RAG), DuckDuckGo Search |
| **Sandbox** | numpy, scipy, sympy, z3-solver, scikit-learn, biopython, rdkit, astropy, cryptography |
| **EDA** | Icarus Verilog, Yosys, Ngspice, gdstk, KLayout |

---

## 📂 Project Structure

```
Team_Trenches/
├── backend/
│   ├── app.py              # FastAPI server & endpoints
│   ├── orchestrator.py     # Core 7-Way Pipeline orchestrator
│   ├── downloader.py       # HuggingFace model downloader
│   ├── memory.py           # ChromaDB RAG memory & HW registry
│   ├── sandbox.py          # Polyglot sandbox (13 langs) & EDA verify
│   ├── search.py           # Web search & parallel scraping
│   ├── eda_setup.py        # EDA toolchain auto-installer
│   ├── repo_map.py         # AST-based repository mapper
│   └── git_agent.py        # Automated Git & PR agent
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React UI
│   │   └── components/     # Modular UI components
│   ├── package.json
│   └── vite.config.js
├── start.sh                # One-click launcher
├── requirements.txt
└── README.md
```

---

## 🖥️ Requirements & Setup

| Resource | Minimum | Recommended |
|---|---|---|
| **RAM** | 16 GB | 32 GB |
| **Storage** | 25 GB | 45 GB |
| **OS** | Ubuntu 22.04+ / macOS 14 | Ubuntu 24.04 |
| **GPU** | 8GB VRAM (any vendor) | NVIDIA RTX 3090/4090 |

### Quick Start

```bash
# Clone & setup
git clone https://github.com/Bshdhorrhh/Team_Trenches.git
cd Team_Trenches
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Launch
chmod +x start.sh && ./start.sh
```

Open `http://localhost:5173`. Models download automatically on first use.

**Optional — EDA Toolchain:**
```bash
sudo apt-get install -y iverilog yosys ngspice
pip install gdstk
```

---

## 👥 Team

**Team Trenches** — Local Multi-Agent AIOS Development Team.
