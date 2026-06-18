# 🚀 DeepThinker — Startup Guide

This guide shows you how to run DeepThinker on any laptop or desktop. It uses **two separate terminals** — one for the backend (AI engine) and one for the frontend (web UI). This is the most reliable method across all operating systems.

> **Prerequisites:** Ensure you have completed the one-time installation steps in [README_SETUP.md](./README_SETUP.md) first.

---

## 🖥️ Quick Reference

| Platform | Backend Command | Frontend Command |
|---|---|---|
| **Ubuntu / Linux** | `source venv/bin/activate && python backend/app.py` | `cd frontend && npm run dev` |
| **Windows (CMD)** | `venv\Scripts\activate && python backend\app.py` | `cd frontend && npm run dev` |
| **Windows (PowerShell)** | `venv\Scripts\Activate.ps1; python backend\app.py` | `cd frontend; npm run dev` |
| **Mac (Apple Silicon)** | `source venv/bin/activate && python backend/app.py` | `cd frontend && npm run dev` |

---

## Step 1 — Open Two Terminals

You need **two terminal windows** open at the same time, both pointed at the project folder.

**Ubuntu / Mac:**
```bash
cd /path/to/your/deepthinker
```

**Windows:**
```cmd
cd C:\path\to\your\deepthinker
```

---

## Step 2 — Terminal 1: Start the Backend (AI Engine)

### 🐧 Ubuntu / Linux

```bash
# Activate the virtual environment
source venv/bin/activate

# (Intel Iris Xe iGPU only) Set hardware environment
export SYCL_DEVICE_FILTER=level_zero
export IPEX_OPTIMIZE_TRANSFORMERS=1

# Start the backend server
python backend/app.py
```

✅ You will see:
```
Memory Engine: Successfully initialized ChromaDB...
🧠 DMA: Detected XX GB RAM → Safety threshold = X.X GB
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### 🍎 Mac (Apple Silicon M1/M2/M3/M4)

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the backend server
python backend/app.py
```

✅ You will see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### 🪟 Windows (Command Prompt)

```cmd
REM Activate the virtual environment
venv\Scripts\activate

REM Start the backend server
python backend\app.py
```

### 🪟 Windows (PowerShell)

```powershell
# Activate the virtual environment
venv\Scripts\Activate.ps1

# Start the backend server
python backend\app.py
```

✅ You will see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Step 3 — Terminal 2: Start the Frontend (Web UI)

Open a **second terminal** in the same project folder.

### 🐧 Ubuntu / Linux & 🍎 Mac

```bash
cd frontend
npm run dev
```

### 🪟 Windows (CMD or PowerShell)

```cmd
cd frontend
npm run dev
```

✅ You will see:
```
  VITE v8.x.x  ready in 300 ms
  ➜  Local:   http://localhost:5173/
```

---

## Step 4 — Open the UI

Open your browser and go to:
```
http://localhost:5173
```

The backend API is available at:
```
http://127.0.0.1:8000
```

---

## 📥 Downloading Models (First Time Only)

Before running the system for the first time, download the AI model weights (~18GB total):

```bash
# Activate your virtual environment first
source venv/bin/activate   # Mac / Linux
# venv\Scripts\activate    # Windows

# Download all models
python backend/downloader.py

# Or download specific models individually
python backend/downloader.py router          # Phi-3.5 Mini (3 GB)
python backend/downloader.py deepseek_r1    # DeepSeek-R1 7B (6 GB)
python backend/downloader.py vibethinker   # VibeThinker 1.5B (1.4 GB)
python backend/downloader.py opencode      # OpenCodeInterpreter 6.7B (5 GB)
python backend/downloader.py qwen_vl       # Qwen2.5-VL 7B (7 GB)
```

---

## 🛠️ Troubleshooting

### ❌ Error: `Address already in use` (Port 8000 or 5173)

A previous server instance is still running. Kill it first:

**Ubuntu / Linux / Mac:**
```bash
# Kill whatever is on port 8000
fuser -k 8000/tcp

# Kill whatever is on port 5173
fuser -k 5173/tcp
```

**Windows (PowerShell):**
```powershell
# Find the process using port 8000
netstat -ano | findstr :8000
# Then kill it by PID (replace 12345 with actual PID)
taskkill /PID 12345 /F

# Find the process using port 5173
netstat -ano | findstr :5173
taskkill /PID 12345 /F
```

Then re-run the backend and frontend commands.

---

### ❌ Error: `NetworkError when attempting to fetch resource`

The frontend cannot reach the backend. Check:
1. Is the **backend running** in Terminal 1? (You should see `Uvicorn running on http://127.0.0.1:8000`)
2. Did the backend **crash on startup**? (Look for error messages in Terminal 1)
3. Go to the UI **Settings** and confirm the API URL is set to `http://127.0.0.1:8000`

---

### ❌ Error: `Module not found` or `No module named 'llama_cpp'`

Your virtual environment is not activated, or dependencies weren't installed. Run:
```bash
source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

---

### ❌ Error: `CUDA out of memory` or `Aborted (core dumped)` (Linux Intel iGPU)

The Dynamic Memory Allocator (DMA) will handle this automatically by evicting the least recently used model. If it still crashes:

1. Click **"Offload Memory"** in the UI sidebar to free all loaded models.
2. Retry the prompt — the DMA will reload only what is needed.

---

### ❌ Frontend opens on port 5174 instead of 5173

Port 5173 is occupied. Either:
- Kill it: `fuser -k 5173/tcp` (Linux/Mac) or use `taskkill` (Windows)
- Or just use `http://localhost:5174` — the UI works the same on any port.

---

## 🧠 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **RAM** | 16 GB | 32 GB |
| **GPU VRAM** | 8 GB (NVIDIA/AMD) | 12 GB+ |
| **Storage** | 25 GB free | 40 GB free |
| **OS** | Ubuntu 22.04 / Win 10 / macOS 13 | Ubuntu 24.04 / Win 11 / macOS 14 |
| **Python** | 3.10 | 3.11 |
| **Node.js** | 18 | 20 |

> **Intel iGPU (Iris Xe / Arc):** The system maps your system RAM as VRAM. 32 GB RAM recommended. Set `SYCL_DEVICE_FILTER=level_zero` before starting the backend.

---

## ✅ Everything Working — What to Expect

When both servers are running and you submit a prompt, you will see the multi-agent pipeline streaming live in the UI:

1. **Phi-3.5 Router** classifies your prompt → `CODING`, `REASONING`, or `SIMPLE`
2. **DeepSeek-R1** drafts the logic plan
3. **Reasoning Sandbox** verifies the plan with Python assertions
4. **VibeThinker** writes the code
5. **Execution Sandbox** runs the code in an isolated environment
6. **OpenCodeInterpreter** generates an interactive 3D Plotly chart (if applicable)
7. Final answer streams to the UI with code, output, and visualization
