# 📊 DeepThink AIOS — Benchmark Evaluation Guide

This guide describes how to run and monitor the automated, parallel benchmark suite integrated into DeepThink AIOS. 

Benchmarks are executed under physical sandbox isolation (with resource limits and network blocks) and graded using strict ground-truth verification.

---

## 📋 Prerequisites

Before running the benchmarks, ensure:
1. The virtual environment is active:
   ```bash
   source venv/bin/activate
   ```
2. All Python dependencies are installed. This is required — the `datasets`
   library is what downloads the official evaluation sets. If it is missing,
   the runner silently falls back to placeholder **mock data** and the reported
   scores will **not** be real:
   ```bash
   pip install -r requirements.txt
   ```
3. The benchmark models are downloaded:
   ```bash
   python backend/downloader.py router deepseek_r1 vibethinker opencode
   ```
4. The backend is running (serves the API and dashboard on port `8000`):
   ```bash
   python backend/app.py
   ```


---

## 🖥️ Method 1: Web Dashboard (Recommended)

DeepThink AIOS includes a dedicated, responsive HTML5 dashboard to trigger and monitor benchmark sessions visually.

1. Start the backend app as described above.
2. Open your web browser and navigate to:
   ```
   http://localhost:8000/benchmark
   ```
3. **Configure the Evaluation:**
   * **Category:** Select the benchmark suite. Supported categories are: `HumanEval`, `MBPP`, `GSM8K`, `MATH`, `GPQA (PhD Science)`, `AIME (Olympiad Logic)`, `MuSR (PhD Logic)`, `MMLU-Pro (Prof STEM)`, `SWE-bench Lite`, `SWE-bench Pro`, and `SearchQA / HotpotQA`.

   * **Sample Size:** Specify the number of questions to sample. Leave it blank or set to `0` to run the entire dataset.
4. Click **Start Evaluation**.

### What you will see:
* **Live Worker Console:** Real-time tracking of individual TPU/GPU/CPU workers, showing which problem they are executing, their live status (`Idle`, `Processing`, or the active agent/model stage), and latency. Per-task pass/fail results are reported in the console logs and the accuracy chart.

* **Accuracy Chart:** Real-time updates showing current accuracy compared against industry baselines (GPT-4, Claude 3.5 Sonnet, Llama-3-70B, and DeepThink AIOS).
* **Speed Metrics:** Throughput (average tokens generated per second per core) and average query latency.
* **Console Logs:** A scrolling, live logging console capturing exact test steps, warnings, compile failures, and final grading assertions.

---

## ⚡ Method 2: API (REST / CLI)

If you are running in a headless container or wish to script evaluations, you can trigger benchmarks using standard API endpoints.

### 1. Start a Benchmark Session
Send a `POST` request to start the parallel evaluation runner in the background:
```bash
curl -X POST http://127.0.0.1:8000/api/benchmark/start \
     -H "Content-Type: application/json" \
     -d '{"category": "GSM8K", "sample_size": 20}'
```

### 2. Poll Live Status & Logs
Get the current execution state, workers status, speed metrics, accuracy, and console output:
```bash
curl http://127.0.0.1:8000/api/benchmark/status
```

### 3. Stop an Active Run
Terminate the active benchmark immediately and release all background workers:
```bash
curl -X POST http://127.0.0.1:8000/api/benchmark/stop
```

---

## ⚙️ How It Works (Behind the Scenes)

* **Proactive Memory Flush:** Before starting, the runner automatically issues an `unload_all_models()` call to clear the GPU/TPU VRAM, ensuring maximum headroom for parallel workers.
* **Automatic Thread Mapping:** The system queries your hardware and sets the optimal number of parallel workers:
  * **NVIDIA 80GB GPUs (e.g. H100/A100/L40S):** 12 parallel workers.
  * **NVIDIA 40GB GPUs (e.g. A10G/L4):** 4 parallel workers.
  * **NVIDIA 24GB GPUs (e.g. RTX 3090/4090):** 2 parallel workers.
  * **Intel Arc/Iris Xe / CPUs:** Auto-scaled based on system CPU cores (`cores / 4`).
* **Accuracy Booster (PAL):** For math and logic suites (`GSM8K`, `MATH`, `AIME`), the system temporarily activates Program-Aided Language solving + self-consistency voting ($k=3$), executing logic code inside the sandbox to calculate exact outputs.
* **Strict Sandbox Grading:** Python/JS coding questions (`HumanEval`/`MBPP`/`SWE-bench`) write files directly to isolated sandbox folders and execute tests. The task is marked as `Passed` only if the script terminates with exit code `0`.
