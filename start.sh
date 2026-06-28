#!/bin/bash

# Cleanly terminate all child processes on exit
trap "echo ''; echo 'Shutting down DeepThink AIOS...'; pkill -f 'backend/app.py' 2>/dev/null; pkill -f 'vite' 2>/dev/null; sleep 1; exit 0" SIGINT SIGTERM EXIT

echo "============================================="
echo "   Starting DeepThink AIOS...   "
echo "============================================="

# Intel Iris Xe iGPU environment variables
export SYCL_DEVICE_FILTER=level_zero
export IPEX_OPTIMIZE_TRANSFORMERS=1

# Source Intel oneAPI compiler if available
if [ -f "/opt/intel/oneapi/compiler/latest/env/vars.sh" ]; then
    echo "Sourcing Intel oneAPI Compiler environment variables..."
    source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1
elif [ -f "/opt/intel/oneapi/compiler/2026.0/env/vars.sh" ]; then
    source /opt/intel/oneapi/compiler/2026.0/env/vars.sh >/dev/null 2>&1
fi
if [ -f "/opt/intel/oneapi/mkl/2026.0/env/vars.sh" ]; then
    echo "Sourcing Intel MKL environment variables..."
    source /opt/intel/oneapi/mkl/2026.0/env/vars.sh >/dev/null 2>&1
fi

# ── Step 1: Kill any stale processes by name ─────────────────────────────────
echo "Stopping any previous instances..."
pkill -f "backend/app.py" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -f "vite" 2>/dev/null

# ── Step 2: Wait until ports are actually free (up to 15s) ───────────────────
echo "Waiting for ports 8000 and 5173 to be released..."
for i in {1..15}; do
    PORT_8000=$(ss -tlnp 2>/dev/null | grep ':8000' | wc -l)
    PORT_5173=$(ss -tlnp 2>/dev/null | grep ':5173' | wc -l)
    if [ "$PORT_8000" -eq 0 ] && [ "$PORT_5173" -eq 0 ]; then
        echo "✅ Ports are free."
        break
    fi
    echo "  Still waiting... ($i/15)"
    sleep 1
done

# ── Step 3: Start Backend ─────────────────────────────────────────────────────
echo "Launching FastAPI Backend..."
venv/bin/python backend/app.py &
BACKEND_PID=$!

# ── Step 4: Wait until backend is actually listening on port 8000 ────────────
echo "Waiting for backend to initialize (model warm-up)..."
BACKEND_OK=0
for i in {1..60}; do
    if ss -tlnp 2>/dev/null | grep -q ':8000'; then
        BACKEND_OK=1
        echo "✅ Backend is up on http://127.0.0.1:8000"
        break
    fi
    # Check if the process died
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process exited unexpectedly. Check the error above."
        exit 1
    fi
    sleep 1
done

if [ "$BACKEND_OK" -eq 0 ]; then
    echo "❌ Backend did not start within 60 seconds. Check errors above."
    exit 1
fi

# ── Step 5: Start Frontend ────────────────────────────────────────────────────
echo "Launching React/Vite Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend port
sleep 3

echo "============================================="
echo "🟢 Both servers are running!"
echo "👉 Web UI URL:     http://localhost:5173"
echo "👉 Backend API:    http://127.0.0.1:8000"
echo "Press Ctrl+C to stop both servers."
echo "============================================="

wait
