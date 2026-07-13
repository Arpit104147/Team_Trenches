# 🚀 DeepThink AIOS — Startup Guide

> **Prerequisites:** Complete the one-time installation in [README_SETUP.md](./README_SETUP.md) first.

---

## ✅ Recommended: Manual Two-Terminal Method

This is the most reliable way to run DeepThink AIOS on any operating system. Open **two separate terminal windows** in the project root folder.

---

### 🐧 Ubuntu / Linux

**Terminal 1 — Start the Backend (AI Engine):**
```bash
cd "/path/to/Team_Trenches"
source venv/bin/activate

# Intel Iris Xe / Arc iGPU only — skip on NVIDIA or Mac
export SYCL_DEVICE_FILTER=level_zero
export IPEX_OPTIMIZE_TRANSFORMERS=1

# Optional: Enable enterprise security features
# export AIOS_AUTH_ENABLED=1
# export AIOS_AIR_GAP=1

python backend/app.py
```

Wait until you see this before moving to Terminal 2:
```
🔓 Authentication: DISABLED — Open access (local development mode).
Sandbox: ✅ Native kernel isolation available (unshare/chroot/seccomp)
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

Open **`http://localhost:5173`** in your browser. ✅

---

## 📥 Downloading Models (First Time Only)

Before running for the first time, download the AI model weights (~24 GB total):

```bash
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows

# Check download status
python backend/downloader.py

# Download all models at once
python backend/downloader.py router deepseek_r1 vibethinker ornith qwen_vl

# Or download individually
python backend/downloader.py router          # Phi-3.5 Mini Router     (~3 GB)
python backend/downloader.py deepseek_r1    # DeepSeek-R1 7B           (~6 GB)
python backend/downloader.py vibethinker    # VibeThinker 3B           (~2.5 GB)
python backend/downloader.py ornith         # Ornith 1.0-9B (Coder)    (~7.5 GB)
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

1. **Phi-3.5 Router** classifies the query into one of the 7 paths: `SIMPLE`, `CODING`, `REASONING`, `PREDICTION`, `3D_VIZ`, `CHIP_DESIGN`, or `EXTREME_WEBSEARCH`.
2. **SAST Security Scanner** checks generated code for command injection, reverse shells, and exfiltration patterns before execution.
3. **DeepSeek-R1** or **VibeThinker** drafts the step-by-step logic, math derivation, or analytical comparison.
4. **Execution/Node.js Sandbox** performs logic, coordinate, or API checks on the draft (inside kernel-isolated namespaces when available).
5. **Ornith 1.0-9B** generates target code, data frames, or HTML visualization elements.
6. **Polyglot Sandbox** compiles and executes scripts in isolated environments with `unshare` network/PID isolation.
7. **Self-Correction/Linter loops** fix runtime syntax errors on the fly.
8. Final answer streams to the UI with code, output, and visual elements.

---

## 🔐 Security Features (Active by Default)

The following security features run automatically with no configuration required:

| Feature | Status | Description |
|---------|--------|-------------|
| **SAST Code Scanning** | ✅ Active | Blocks command injection, reverse shells, pickle attacks before execution |
| **Kernel Isolation** | ✅ Active (if available) | `unshare` network/PID/IPC namespaces isolate sandbox processes |
| **Resource Limits** | ✅ Active | 2GB RAM, 120s CPU, 0 child processes, 256 file descriptors |
| **3-Layer Sandbox** | ✅ Active | Process isolation + builtins stripping + import whitelisting |
| **JWT Authentication** | Optional | Enable with `AIOS_AUTH_ENABLED=1` |
| **Air-Gap Mode** | Optional | Enable with `AIOS_AIR_GAP=1` for offline-only deployments |

### UI Indicators
- **Connection Badge** (top of sidebar): Green pulsing dot = Backend connected, Red dot = Disconnected
- **Settings → Security tab**: Shows real-time status of SAST, Isolation, and Air-Gap mode
- **Settings → Workspace tab**: Configure GitHub token for automated PR creation

---

> **Note:** `start.sh` is included in the repo as a convenience script for Linux but may be unreliable depending on your system state. The manual two-terminal method above is always reliable and recommended.
