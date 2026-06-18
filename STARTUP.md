# 🚀 DeepThinker — Startup Guide

> **Prerequisites:** Complete the one-time installation in [README_SETUP.md](./README_SETUP.md) first.

---

## ✅ Recommended: Manual Two-Terminal Method

This is the most reliable way to run DeepThinker on any operating system. Open **two separate terminal windows** in the project root folder.

---

### 🐧 Ubuntu / Linux

**Terminal 1 — Start the Backend (AI Engine):**
```bash
cd "/path/to/Team_Trenches"
source venv/bin/activate

# Intel Iris Xe / Arc iGPU only — skip on NVIDIA or Mac
export SYCL_DEVICE_FILTER=level_zero
export IPEX_OPTIMIZE_TRANSFORMERS=1

python backend/app.py
```

Wait until you see this before moving to Terminal 2:
```
🧠 DMA: Detected XX GB RAM → Safety threshold = X.X GB
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2 — Start the Frontend (Web UI):**
```bash
cd "/path/to/Team_Trenches/frontend"
npm run dev
```

You will see:
```
  VITE v8.x.x  ready in 300 ms
  ➜  Local:   http://localhost:5173/
```

Open **`http://localhost:5173`** in your browser. ✅

---

### 🍎 Mac (Apple Silicon M1/M2/M3/M4)

**Terminal 1 — Backend:**
```bash
cd /path/to/Team_Trenches
source venv/bin/activate
python backend/app.py
```

Wait for: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 — Frontend:**
```bash
cd /path/to/Team_Trenches/frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser. ✅

---

### 🪟 Windows (Command Prompt)

**Terminal 1 — Backend:**
```cmd
cd C:\path\to\Team_Trenches
venv\Scripts\activate
python backend\app.py
```

**Terminal 2 — Frontend:**
```cmd
cd C:\path\to\Team_Trenches\frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser. ✅

### 🪟 Windows (PowerShell)

**Terminal 1 — Backend:**
```powershell
cd C:\path\to\Team_Trenches
venv\Scripts\Activate.ps1
python backend\app.py
```

**Terminal 2 — Frontend:**
```powershell
cd C:\path\to\Team_Trenches\frontend
npm run dev
```

---

## 📥 Downloading Models (First Time Only)

Before running for the first time, download the AI model weights (~18 GB total):

```bash
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows

# Check download status
python backend/downloader.py

# Download all models at once
python backend/downloader.py router deepseek_r1 vibethinker opencode qwen_vl

# Or download individually
python backend/downloader.py router          # Phi-3.5 Mini Router     (~3 GB)
python backend/downloader.py deepseek_r1    # DeepSeek-R1 7B           (~6 GB)
python backend/downloader.py vibethinker    # VibeThinker 1.5B         (~1.4 GB)
python backend/downloader.py opencode       # OpenCodeInterpreter 6.7B (~5 GB)
python backend/downloader.py qwen_vl        # Qwen2.5-VL 7B Vision     (~7 GB)
```

---

## ⏱️ What to Expect on First Run

When you send your **first prompt**, the system does a cold start — it loads the AI models from disk into GPU memory. This is normal:

| Situation | Time |
|---|---|
| **First prompt ever** (cold load) | 2–4 minutes |
| **Subsequent prompts** (models cached) | 5–30 seconds |
| **After "Offload Memory"** | 2–4 minutes again |

> 💡 **Tip:** Send a simple "hi" as your first message to warm up the Router model. Once you see a response, all other prompts will be fast.

---

## 🛠️ Troubleshooting

### ❌ `Address already in use` (Port 8000 or 5173)

A previous server is still running in the background. Kill it first:

**Linux / Mac:**
```bash
pkill -f "backend/app.py"
pkill -f "vite"
sleep 3
```

Then re-run both terminal commands from the top.

**Windows (PowerShell):**
```powershell
# Find PID on port 8000 and kill it
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Find PID on port 5173 and kill it
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

---

### ❌ `Cannot reach backend` / `NetworkError`

The frontend cannot reach the backend. Check:
1. Is Terminal 1 showing `Uvicorn running on http://127.0.0.1:8000`?
2. Did the backend crash? Look for error messages in Terminal 1.
3. Is it your **first prompt**? The backend may still be loading models (2–4 min). Wait and try again.
4. In the UI, click **Settings** and confirm the API URL is `http://127.0.0.1:8000`.

---

### ❌ `No module named 'llama_cpp'` or `ModuleNotFoundError`

Virtual environment not activated or dependencies missing:
```bash
source venv/bin/activate    # Linux / Mac
pip install -r requirements.txt
```

---

### ❌ `Aborted (core dumped)` or GPU crash on model load (Intel iGPU)

The Dynamic Memory Allocator (DMA) handles this automatically. If a crash still occurs:
1. Click **"Offload Memory"** in the UI sidebar to free all loaded models.
2. Retry your prompt — the DMA loads models one at a time.

---

### ❌ Frontend opens on port 5174 instead of 5173

Port 5173 is occupied by a stale process. Kill it:
```bash
pkill -f "vite"    # Linux / Mac
```
Then re-run `npm run dev`. Or just use `http://localhost:5174` — works the same.

---

> **System Requirements:** See [README.md](./README.md#-system-requirements) for the full hardware requirements table.

---

## ✅ What Happens When It's Working

Once both servers are up and you submit a prompt, the pipeline streams live in the UI:

1. **Phi-3.5 Router** classifies: `CODING` / `REASONING` / `SIMPLE`
2. **DeepSeek-R1** drafts a logic plan or detailed answer
3. **Reasoning Sandbox** verifies with Python assertions
4. **VibeThinker** writes and self-corrects the code
5. **Execution Sandbox** runs code in an isolated environment
6. **OpenCodeInterpreter** generates an interactive 3D Plotly chart (if applicable)
7. Final answer streams to the UI with code, output, and visualization

---

> **Note:** `start.sh` is included in the repo as a convenience script for Linux but may be unreliable depending on your system state. The manual two-terminal method above is always reliable and recommended.
