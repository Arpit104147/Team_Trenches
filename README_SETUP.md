# DeepThink AIOS Setup & Hardware Acceleration Guide

Because DeepThink AIOS runs fully local 7B reasoning agents, it requires **hardware acceleration (GPU)**. 

The Python code is OS-agnostic, but PyTorch and `llama.cpp` require different installation commands depending on if you are using an Apple MacBook, a Windows NVIDIA PC, or a Linux Intel machine.

Please follow the instructions for your specific operating system to ensure the models run on your GPU rather than your CPU.

---

## Step 1: System Prerequisites (For Polyglot Sandbox)
DeepThink AIOS can execute code in multiple languages. For this to work, ensure your system has the necessary native compilers installed.

*   **Mac:** Run `xcode-select --install` in terminal (installs `gcc`/`g++`).
*   **Linux:** Run `sudo apt install build-essential openjdk-17-jdk nodejs` (installs C/C++, Java, and Node.js).
*   **Windows:** Install [MinGW](https://www.mingw-w64.org/) for C/C++ and Node.js for Javascript.

*(Note: If a compiler is missing, the AI engine will not crash; it will gracefully fall back to Python execution).*

### 🔧 EDA Toolchain (Optional — For Chip Design Pipeline)
If you plan to use the Chip Design EDA Sandbox for Verilog/SPICE simulation:

```bash
# Linux (Debian/Ubuntu)
sudo apt install iverilog yosys ngspice klayout

# Mac (Homebrew)
brew install icarus-verilog yosys ngspice
```

*(These tools are optional. If missing, the AIOS will skip hardware simulation steps and still function for all other pipelines.)*

### 🔒 Kernel Isolation Prerequisites (Optional — Enhanced Security)
The sandbox automatically uses Linux kernel namespaces (`unshare`) for network/process isolation when available. No extra setup is needed on most modern Linux systems.

*   **Root users:** Full namespace isolation (`--net --pid --ipc`) is automatically enabled.
*   **Non-root users:** User namespaces are probed automatically. If not available, the existing 3-layer sandbox (process isolation + builtins stripping + resource limits) remains fully active.

---

## Step 2: Install Base Python Dependencies
First, install the core dependencies that work on all operating systems:

```bash
# Clone the repository and enter it
git clone https://github.com/Bshdhorrhh/Team_Trenches.git
cd Team_Trenches

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all package requirements
pip install -r requirements.txt
```

---

## Step 3: Install Hardware Acceleration

Run the specific block below that matches your computer's hardware to enable GPU execution.

### 🍎 Mac (Apple Silicon M1/M2/M3)
Apple uses the "Metal" framework for GPU acceleration.

```bash
# 1. Install Mac-optimized PyTorch
pip install torch torchvision torchaudio

# 2. Install Mac-optimized Llama.cpp (Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

### 🪟 Windows / Linux (NVIDIA GPUs)
NVIDIA uses "CUDA" for GPU acceleration. Ensure you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed (v12.1+ recommended).

#### 1. Setup CUDA Environment Variables (Linux only - e.g. Kaggle/L4/L40S)
If the installer cannot find your CUDA compiler (`nvcc`), make sure it is in your path:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. Install NVIDIA-optimized PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Compile and Install Llama.cpp with CUDA & Flash Attention Support
To enable GPU acceleration and full support for hardware-accelerated Flash Attention:
```bash
# On Linux:
export CMAKE_ARGS="-DGGML_CUDA=on"
# On Windows (cmd):
set CMAKE_ARGS="-DGGML_CUDA=on"
# On Windows (PowerShell):
$env:CMAKE_ARGS="-DGGML_CUDA=on"

# Force compile and install from source
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

> [!NOTE]
> **Flash Attention Automatic Activation:** The orchestrator dynamically queries your GPU computing capabilities. If running on modern NVIDIA architectures (Ampere or newer, such as L4, L40S, A100, RTX 3000/4000+), the system automatically initializes models with `flash_attn=True` and optimized batch size thresholds.

### 🐧 Linux (Intel Iris Xe / Arc GPUs)
*This is the default configuration used during the Hackathon development phase.*

```bash
# 1. Install Intel-optimized PyTorch (IPEX)
pip install torch==2.8.0+xpu intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://download.pytorch.org/whl/xpu

# 2. Install Llama.cpp
pip install llama-cpp-python
```

---

## Step 4: Download Models & Start the System

Once all dependencies are installed, head to **[STARTUP.md](./STARTUP.md)** for:
- How to download the AI model weights (~18 GB)
- How to start the backend and frontend on Ubuntu, Mac, and Windows
- How to configure enterprise security features (JWT auth, air-gap mode)
- Troubleshooting common errors

---

## 🔐 Enterprise Configuration (Optional)

These environment variables enable enterprise-grade features. They are **entirely optional** for local development:

| Variable | Default | Description |
|----------|---------|-------------|
| `AIOS_AUTH_ENABLED` | `0` | Set to `1` to require JWT tokens on all API endpoints |
| `AIOS_JWT_SECRET` | auto-generated | Custom secret key for JWT token signing |
| `AIOS_ADMIN_PASSWORD` | `admin` | Password for the `/api/auth/login` endpoint |
| `AIOS_AIR_GAP` | `0` | Set to `1` to disable all outbound network features |
| `GITHUB_TOKEN` | empty | GitHub Personal Access Token for automated PR creation |

Example (air-gapped deployment with auth):
```bash
export AIOS_AUTH_ENABLED=1
export AIOS_JWT_SECRET="your-secret-key-here"
export AIOS_ADMIN_PASSWORD="strong-password"
export AIOS_AIR_GAP=1
python backend/app.py
```
