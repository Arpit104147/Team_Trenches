#!/bin/bash

# Terminate background processes on exit
trap "kill 0" EXIT

echo "============================================="
echo "   Starting DeepThinker Multi-Agent Hub...   "
echo "============================================="

# Intel Iris Xe iGPU environment
export SYCL_DEVICE_FILTER=level_zero
export IPEX_OPTIMIZE_TRANSFORMERS=1

# Sourcing compiler variables if Intel environment exists
if [ -f "/opt/intel/oneapi/compiler/latest/env/vars.sh" ]; then
    echo "Sourcing Intel oneAPI Compiler environment variables..."
    source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1
elif [ -f "/opt/intel/oneapi/compiler/2026.0/env/vars.sh" ]; then
    echo "Sourcing Intel oneAPI 2026.0 Compiler environment variables..."
    source /opt/intel/oneapi/compiler/2026.0/env/vars.sh >/dev/null 2>&1
fi

if [ -f "/opt/intel/oneapi/mkl/2026.0/env/vars.sh" ]; then
    echo "Sourcing Intel MKL environment variables..."
    source /opt/intel/oneapi/mkl/2026.0/env/vars.sh >/dev/null 2>&1
fi

# Kill any leftover processes from a previous run before binding
echo "Cleaning up any previous server instances..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true
sleep 1

# 1. Start backend server
echo "Launching FastAPI Backend..."
venv/bin/python backend/app.py &
BACKEND_PID=$!

# Wait for backend to fully boot before starting frontend
sleep 3

# 2. Start frontend dev server
echo "Launching React/Vite Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "============================================="
echo "🟢 Both servers are running!"
echo "👉 Web UI URL: http://localhost:5173"
echo "👉 Backend API URL: http://127.0.0.1:8000"
echo "👉 Monitor downloads, GPU caching, and RAM in the dashboard!"
echo "Press Ctrl+C to stop both servers."
echo "============================================="

# Keep script running
wait
