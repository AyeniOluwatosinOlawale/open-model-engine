# vLLM v0.30.0 — New Feature Labs

Four labs targeting the major new features in the v0.30.0 release.

## Quick Start

```bash
pip install vllm==0.30.0
```

---

## Lab 1 — Fast Start (IPC Weight Cache)

**File:** `fast_start.py`

Measures engine restart time with and without `--load-format ipc_cache`.

```bash
# Start weight cache daemon
python -m vllm.tools.weight_cache_daemon --model <model> --port 8100

# Run the benchmark (manages server starts/stops automatically)
python fast_start.py --model <model> --n-restarts 3
```

**Expected:** Restart time drops from minutes (full disk reload) to seconds (IPC map).
**K8s impact:** Pod rolling restart time dramatically reduced.

---

## Lab 2 — HiSparse Host KV Cache

**File:** `hisparse.py`

Measures KV cache pressure handling with `HiSparseConnector` enabled.

```bash
# Baseline
python -m vllm.entrypoints.openai.api_server --model <model> --port 8000

# HiSparse
python -m vllm.entrypoints.openai.api_server --model <model> --port 8001 \
  --kv-connector HiSparseConnector --kv-connector-host-cache-size-gb 16

python hisparse.py --baseline-url http://localhost:8000 --hisparse-url http://localhost:8001 --model <model>
```

**Expected:** HiSparse spills to pinned host memory; short request TTFT maintained under GPU pressure.

---

## Lab 3 — MRV2 CUDA Graph Capture

**File:** `mrv2_graphs.py`

Measures graph capture time and decode throughput with/without CUDA graphs.

```bash
# CUDA graphs (default MRV2)
python -m vllm.entrypoints.openai.api_server --model <model> --port 8000

# Eager mode
python -m vllm.entrypoints.openai.api_server --model <model> --port 8001 --enforce-eager

python mrv2_graphs.py --model <model> --graph-url http://localhost:8000 --eager-url http://localhost:8001
```

**Expected:** Graph capture 12s → 2s; engine init 28.9s → 8.2s (H200). Throughput +30–50% at bs≥4.

---

## Lab 4 — Gumbel-Max Watermarking

**File:** `watermarking.py`

Tests watermark generation, detection endpoint, and per-request opt-out.

```bash
python -m vllm.entrypoints.openai.api_server --model <model> --port 8000 \
  --watermark-scheme gumbel --watermark-key "my-secret-key"

python watermarking.py --url http://localhost:8000 --model <model>
```

**Expected:** High true positive rate; near-zero false positives; negligible latency overhead.
