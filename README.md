# open-model-engine

A GPU-first inference engineering harness for self-hosted open model servers — vLLM, SGLang, and TensorRT-LLM. Deep profiling, bottleneck analysis, and optimization benchmarks for production inference stacks.

```
open-model-engine
        │
        ▼
┌───────────────────────┐
│   Server Backends     │
│  vLLM · SGLang · TRT  │
└──────────┬────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
GPU Profiling  Serving Metrics
pynvml         Prometheus /metrics
Nsight Systems SGLang /get_server_info
NVTX ranges    KV cache hit rate
               Queue depth · Batch size
    │             │
    └──────┬──────┘
           ▼
  Latency · Throughput
  Prefill · Decode
  Bottleneck Analysis
  Optimization Benchmarks
```

---

## Features

- **3 open model backends** — vLLM, SGLang, TensorRT-LLM via OpenAI-compatible HTTP API
- **Deep GPU profiling** — pynvml (SM clock, HBM clock, power, temperature, utilization) with nvidia-smi XML fallback
- **CUDA-level profiling** — Nsight Systems CLI subprocess with NVTX range annotations for prefill/decode phase attribution
- **Kernel-level bottleneck detection** — classifies prefill-bound vs decode-bound per request using explicit server metrics and CUDA kernel timing
- **Serving metrics** — KV cache hit rate, cache utilization, request queue depth, batch size via Prometheus and server info endpoints
- **Optimization benchmarks** — quantization (FP16/INT8/INT4/FP8), speculative decoding, prefix cache hit rate, multi-GPU tensor parallel scaling, continuous batching
- **Concurrency sweep** — TTFT degradation and throughput scaling under concurrent load
- **Export** — JSON and CSV output with full GPU and CUDA metrics

---

## Installation

```bash
git clone https://github.com/your-org/open-model-engine
cd open-model-engine
pip install -e .

# Optional: CUDA profiling (NVTX annotations)
pip install -e ".[cuda]"

# Optional: CUPTI kernel tracing (advanced)
pip install -e ".[cupti]"

# Nsight Systems CLI (for full CUDA profiling):
# https://developer.nvidia.com/nsight-systems
```

### Requirements

- Python ≥ 3.10
- NVIDIA GPU (for GPU metrics; CPU-only servers work but GPU metrics will be empty)
- A running vLLM, SGLang, or TensorRT-LLM server

---

## Quick Start

### Benchmark a running server

```bash
# Start vLLM (example)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Run benchmark
open-model-engine bench \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000 \
  --runs 10
```

### Quantization comparison

```bash
open-model-engine optimize quantization \
  --backend vllm \
  --model Llama-3.1-8B \
  --formats fp16,int8,int4
```

### Speculative decoding benchmark

```bash
open-model-engine optimize speculative \
  --backend vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --draft meta-llama/Llama-3.1-1B-Instruct \
  --url http://localhost:8000
```

### Multi-GPU tensor parallel scaling

```bash
open-model-engine optimize multi-gpu \
  --backend vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tp 1,2,4
```

### CUDA profiling with Nsight Systems

```bash
open-model-engine bench \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile cuda \
  --nsys-output ./profiles/run1 \
  --scenario medium \
  --runs 5
```

---

## CLI Reference

### `bench` — Baseline benchmark

```
open-model-engine bench [OPTIONS]

  --backend        TEXT     vllm | sglang | tensorrt
  --model          TEXT     Model name
  --url            TEXT     Server base URL  (default: http://localhost:8000)
  --scenario       TEXT     short | medium | long | complex  (default: all)
  --runs           INT      Runs per scenario  (default: 5)
  --no-warmup              Skip warm-up request
  --no-gpu-metrics         Skip GPU profiling
  --profile        TEXT    none | system | cuda  (default: system)
  --nsys-output    PATH    Nsight Systems report output directory  (default: ./profiles)
  --gpu-cost-per-hour FLOAT  GPU cost USD/hr for amortized cost tracking
  --concurrency            Run concurrency sweep after baseline
  --conc-levels    TEXT    Comma-separated levels  (default: 1,2,4,8,16,32)
  --output         PATH    Save full results to JSON
  --csv            PATH    Save aggregated metrics to CSV
```

### `optimize` — Optimization benchmarks

```
open-model-engine optimize OPTIMIZATION [OPTIONS]

OPTIMIZATION:
  quantization      Compare FP16 / INT8 / INT4 / FP8 at same serving config
  speculative       Draft + target model vs target alone (accepted token rate, speedup)
  prefix-cache      Shared-prefix hit rate and TTFT reduction
  multi-gpu         Throughput and latency at TP=1,2,4,8
  continuous-batch  Batch utilization and throughput at varying load

Shared options:
  --backend  TEXT   vllm | sglang | tensorrt
  --model    TEXT   Model name
  --url      TEXT   Server URL
  --runs     INT    Runs per configuration

Optimization-specific:
  # quantization
  --formats  TEXT   Comma-separated formats  (default: fp16,int8,int4)

  # speculative
  --draft    TEXT   Draft model name (required)

  # multi-gpu
  --tp       TEXT   Tensor parallel degrees  (default: 1,2,4)
```

---

## Metrics Reference

### Latency

| Metric | Description | Source |
|---|---|---|
| `ttft_ms` | Time To First Token | Measured (first stream chunk) |
| `tpot_ms` | Time Per Output Token | `(total - ttft) / output_tokens` |
| `prefill_ms` | Prefill phase duration | vLLM: `time_to_first_token_seconds` · TRT-LLM: `first_token_latency` |
| `decode_ms` | Decode phase total | vLLM: `time_per_output_token × tokens` · SGLang: derived from decode_throughput |
| `queue_time_ms` | Time in server queue | vLLM: `request_queue_time_seconds` · TRT-LLM: `request_queue_latency` |

### Throughput

| Metric | Description | Source |
|---|---|---|
| `throughput_tps` | Output tokens/sec | Measured |
| `prefill_throughput_tps` | Prompt tokens processed/sec | vLLM: `avg_prompt_throughput_toks_per_s` · SGLang: `prefill_throughput` |
| `decode_throughput_tps` | Generated tokens/sec | vLLM: `avg_generation_throughput_toks_per_s` · SGLang: `decode_throughput` |

### GPU Hardware

| Metric | Description | Source |
|---|---|---|
| `gpu_memory_used_mb` | HBM used | pynvml / nvidia-smi |
| `gpu_utilization_pct` | SM utilization | pynvml / nvidia-smi |
| `gpu_power_watts` | Power draw | pynvml |
| `gpu_sm_clock_mhz` | SM clock frequency | pynvml |
| `gpu_memory_clock_mhz` | HBM clock (decode BW indicator) | pynvml |
| `gpu_temperature_c` | Die temperature | pynvml / nvidia-smi |

### Serving Engine

| Metric | Description | vLLM Key | SGLang Key | TRT-LLM Key |
|---|---|---|---|---|
| `kv_cache_hit_rate` | Prefix cache hit fraction | `vllm:cache_hit_rate` | `cache_hit_rate` | `tensorrtllm_kv_cache_hit_rate` |
| `kv_cache_utilization` | KV cache blocks in use | `vllm:gpu_cache_usage_perc` | — | `tensorrtllm_kv_cache_fraction_used` |
| `num_running_requests` | Active in-flight requests | `vllm:num_requests_running` | `num_running_reqs` | `tensorrtllm_inflight_requests` |
| `num_queued_requests` | Requests waiting in queue | `vllm:num_requests_waiting` | `num_waiting_reqs` | — |

### CUDA (Nsight, opt-in)

| Metric | Description |
|---|---|
| `cuda_prefill_kernel_ms` | Total CUDA kernel time in prefill NVTX range |
| `cuda_decode_kernel_ms` | Total CUDA kernel time in decode NVTX range |
| `hbm_read_bandwidth_gbps` | HBM read bandwidth (decode bottleneck indicator) |
| `hbm_write_bandwidth_gbps` | HBM write bandwidth |
| `memory_transfer_ms` | PCIe / NVLink transfer time |

---

## Bottleneck Detection

Every request is classified as **prefill-bound**, **decode-bound**, or **balanced**.

```
If explicit server metrics available (prefill_ms, decode_ms):
    ratio = prefill_ms / (prefill_ms + decode_ms)

Else (fall back to TTFT proxy):
    ratio = ttft_ms / (ttft_ms + tpot_ms × output_tokens)

ratio > 0.6  →  PREFILL-bound
ratio < 0.4  →  DECODE-bound
otherwise    →  BALANCED
```

### What this means

**Prefill-bound** — context processing (attention over input tokens) dominates.
- Common when: prompt is very long, no prefix caching, insufficient tensor parallelism
- Fix: chunked prefill, larger TP degree, prefix caching, FlashAttention-3 / FlashInfer

**Decode-bound** — token generation (KV cache memory reads) dominates.
- Common when: long output sequences, small batch size, large model, high HBM BW pressure
- Fix: speculative decoding, INT4/FP8 quantization, larger batch, PagedAttention tuning

### Terminal output example

```
  BOTTLENECK ANALYSIS
  ════════════════════════════════════════════════════════════════
  short    Prefill:   42ms    Decode:   180ms  →  ⚠ DECODE
  medium   Prefill:  310ms    Decode:   220ms  →  ⚠ PREFILL
  long     Prefill:  280ms    Decode:  1420ms  →  ⚠ DECODE
  complex  Prefill:  190ms    Decode:   310ms  →  ✓ BALANCED

  DECODE-bound (short, long):
    → Speculative decoding  |  INT4/FP8 quantization  |  Increase batch size
    → Enable PagedAttention prefix caching  |  Check HBM bandwidth saturation

  PREFILL-bound (medium):
    → Chunked prefill  |  Increase tensor parallelism  |  Prefix caching
  ════════════════════════════════════════════════════════════════
```

---

## CUDA Profiling with Nsight Systems

When `--profile cuda` is set, each benchmark run is wrapped in Nsight Systems. NVTX ranges annotate the prefill and decode phases so kernel time can be attributed to each:

```
nsys profile --trace cuda,nvtx,osrt ...
    │
    ├─ [NVTX: prefill]
    │    flash_attn_fwd          92ms  ← context attention (compute-bound)
    │    ampere_fp16_s16816       8ms
    │
    └─ [NVTX: decode]
         paged_attention_v2     620ms  ← KV cache reads (memory-BW-bound)
         sampling                 4ms
```

Output includes:
- `cuda_prefill_kernel_ms` — total CUDA time in the prefill phase
- `cuda_decode_kernel_ms` — total CUDA time in the decode phase
- Raw `.nsys-rep` report for visualization in Nsight Systems GUI

---

## Optimization Benchmarks

### Quantization (`quantization`)

Compares the same model at different precisions. Requires separate server instances (different ports) per quantization format, or sequential launches.

| Metric | FP16 | INT8 | INT4 | FP8 |
|---|---|---|---|---|
| TTFT p50 | baseline | ~same | ~same | ~same |
| Decode TPS | baseline | +10–20% | +30–50% | +20–40% |
| GPU Memory | baseline | −40% | −70% | −40% |
| Quality | reference | −1–3% | −3–8% | −1–3% |

### Speculative Decoding (`speculative`)

Measures accepted token rate and effective throughput with a small draft model generating candidate tokens.

```
Metrics reported:
  accepted_token_rate    fraction of draft tokens accepted by target model
  effective_speedup      throughput_tps(spec) / throughput_tps(baseline)
  ttft_change_pct        TTFT increase from draft overhead (usually +5–15%)
```

### Prefix Cache (`prefix-cache`)

Sends requests with a long shared system prompt to measure RadixAttention / prefix caching effectiveness.

```
Metrics reported:
  kv_cache_hit_rate      fraction of KV blocks served from cache
  ttft_cached_ms         TTFT with warm cache
  ttft_cold_ms           TTFT without cache
  ttft_reduction_pct     how much TTFT drops with caching enabled
```

### Multi-GPU Scaling (`multi-gpu`)

Measures throughput and latency scaling as tensor parallelism degree increases (TP=1→2→4→8).

```
Metrics reported per TP degree:
  throughput_tps         output tokens/sec
  scaling_efficiency     throughput(TP=N) / (N × throughput(TP=1))
  ttft_p50_ms            latency change from communication overhead
  gpu_memory_per_device  HBM used per GPU
```

---

## Python Library Usage

```python
import asyncio
from open_model_engine import OpenModelEngine, ScenarioDef

async def main():
    async with OpenModelEngine(
        "vllm",
        "meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000",
        profile="system",           # "none" | "system" | "cuda"
        gpu_cost_per_hour=2.50,     # optional: amortized cost tracking
    ) as engine:

        # Single scenario benchmark
        results = await engine.bench(ScenarioDef.medium(), runs=10)
        agg = engine.aggregate(results)

        print(f"TTFT p50:         {agg.ttft_p50_ms:.1f}ms")
        print(f"Prefill p50:      {agg.prefill_p50_ms:.1f}ms")
        print(f"Decode p50:       {agg.decode_p50_ms:.1f}ms")
        print(f"Decode TPS:       {agg.decode_throughput_mean_tps:.1f} tok/s")
        print(f"KV cache hit:     {agg.kv_cache_hit_rate_mean * 100:.1f}%")
        print(f"Bottleneck:       {agg.dominant_bottleneck}")

        # All scenarios
        all_results, all_agg = await engine.bench_all(runs=5)

        # Concurrency sweep
        points = await engine.sweep_concurrency(
            ScenarioDef.medium(), levels=[1, 4, 8, 16, 32]
        )

        # Point-in-time GPU snapshot
        gpu = await engine.gpu_snapshot()
        print(f"GPU: {gpu['memory_used_mb']:.0f}MB used, {gpu['utilization_pct']:.1f}% util")

asyncio.run(main())
```

---

## Docker

Docker Compose files for each backend are in `docker/`.

### vLLM

```bash
# Single GPU
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=hf_... \
docker compose -f docker/vllm/compose.yaml up -d

# Multi-GPU (TP=4)
MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct \
TP_DEGREE=4 \
docker compose -f docker/vllm/compose.yaml up -d
```

### SGLang

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
docker compose -f docker/sglang/compose.yaml up -d
```

### TensorRT-LLM

```bash
# Requires pre-built TRT-LLM engine in ./models/
TRTLLM_MODEL_DIR=./models \
docker compose -f docker/trtllm/compose.yaml up -d
```

---

## Output Format

### Terminal (baseline)

```
  Benchmark: Latency, Throughput & Bottleneck
  ┌──────────┬──────────┬────┬──────────┬──────────┬────────────┬────────────┬───────────┬───────────┬──────────┐
  │ Scenario │ Backend  │ N  │ TTFT p50 │ TTFT p95 │ Prefill p50│ Decode p50 │ TPS (mean)│ Decode TPS│Bottleneck│
  ├──────────┼──────────┼────┼──────────┼──────────┼────────────┼────────────┼───────────┼───────────┼──────────┤
  │ short    │ vllm     │ 10 │   120ms  │   145ms  │    42ms    │   180ms    │   28.1    │   27.4    │⚠ DECODE  │
  │ medium   │ vllm     │ 10 │   310ms  │   380ms  │   310ms    │   220ms    │   31.4    │   31.2    │⚠ PREFILL │
  │ long     │ vllm     │ 10 │   280ms  │   320ms  │   280ms    │  1420ms    │   25.0    │   24.8    │⚠ DECODE  │
  └──────────┴──────────┴────┴──────────┴──────────┴────────────┴────────────┴───────────┴───────────┴──────────┘

  GPU Metrics
  ┌──────────┬─────────────┬────────────┬──────────────┬─────────────────┬─────────────────┐
  │ Scenario │ GPU Mem (MB)│ GPU Util % │ KV Hit Rate %│ CUDA Prefill ms │ CUDA Decode ms  │
  ├──────────┼─────────────┼────────────┼──────────────┼─────────────────┼─────────────────┤
  │ short    │    14200    │   68.2     │    82.4      │      38.1       │     572.3       │
  │ medium   │    14200    │   71.5     │    45.1      │     298.4       │     198.2       │
  └──────────┴─────────────┴────────────┴──────────────┴─────────────────┴─────────────────┘
```

### JSON export (`--output results.json`)

```json
{
  "metadata": {
    "timestamp": "2025-09-05T12:00:00Z",
    "backend": "vllm",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "profile": "cuda"
  },
  "aggregated": [
    {
      "scenario": "medium",
      "backend": "vllm",
      "ttft_p50_ms": 310.2,
      "prefill_p50_ms": 310.2,
      "decode_p50_ms": 220.8,
      "throughput_mean_tps": 31.4,
      "prefill_throughput_mean_tps": 985.0,
      "decode_throughput_mean_tps": 31.2,
      "gpu_memory_mean_mb": 14200.0,
      "gpu_utilization_mean_pct": 71.5,
      "kv_cache_hit_rate_mean": 0.451,
      "cuda_prefill_kernel_mean_ms": 298.4,
      "cuda_decode_kernel_mean_ms": 198.2,
      "hbm_read_bw_mean_gbps": 847.3,
      "dominant_bottleneck": "prefill",
      "prefill_pct": 0.9
    }
  ],
  "runs": [ ... ]
}
```

---

## Project Structure

```
open-model-engine/
├── pyproject.toml
├── .env.example
├── docker/
│   ├── vllm/
│   │   ├── Dockerfile
│   │   └── compose.yaml
│   ├── sglang/
│   │   ├── Dockerfile
│   │   └── compose.yaml
│   └── trtllm/
│       ├── Dockerfile
│       └── compose.yaml
└── src/open_model_engine/
    ├── __init__.py               # Public API
    ├── __main__.py               # CLI (typer)
    ├── types.py                  # OpenModelRunResult, AggregatedMetrics, ScenarioDef
    ├── engine.py                 # OpenModelEngine — main library class
    ├── report.py                 # Terminal tables, GPU table, JSON/CSV export
    ├── servers/
    │   ├── base.py               # ServerBackend (ABC)
    │   ├── vllm_server.py        # vLLM: streaming + Prometheus scraping
    │   ├── sglang_server.py      # SGLang: streaming + /get_server_info
    │   └── trtllm_server.py      # TRT-LLM: streaming + Prometheus scraping
    ├── profiling/
    │   ├── gpu_profiler.py       # pynvml continuous polling + nvidia-smi fallback
    │   └── nsight_profiler.py    # nsys subprocess + NVTX range annotations
    └── benchmarks/
        ├── runner.py             # run_scenario, run_all_scenarios
        └── optimizations/
            ├── quantization.py   # FP16 vs INT8 vs INT4 vs FP8
            ├── speculative.py    # Draft + target model comparison
            ├── prefix_cache.py   # Shared prefix hit rate benchmark
            ├── multi_gpu.py      # TP scaling benchmark
            └── continuous_batch.py  # Batch utilization sweep
```

---

## Relationship to inference-harness

`open-model-engine` is a standalone project focused exclusively on GPU-hosted open models. It goes deeper than `inference-harness` on:

| | inference-harness | open-model-engine |
|---|---|---|
| Closed API backends | ✓ 5 providers | — |
| Open model backends | ✓ vLLM, SGLang, TRT | ✓ vLLM, SGLang, TRT |
| System GPU profiling | ✓ | ✓ |
| CUDA kernel profiling | ✓ basic | ✓ full (NVTX phase attribution) |
| Optimization benchmarks | basic | ✓ quantization, speculative, multi-GPU, prefix cache |
| Web dashboard | ✓ | — |
| Cross-backend compare | ✓ | — |

Both projects share a compatible `RunResult` schema — results can be loaded and compared across repos.

---

## License

MIT
