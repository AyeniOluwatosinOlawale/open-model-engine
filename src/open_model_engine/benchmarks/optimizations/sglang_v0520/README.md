# SGLang v0.5.20 — New Feature Labs

Four labs targeting the major new features in the v0.5.20 release.  
**Model used throughout:** `Qwen/Qwen3-8B` — the exact model cited in the v0.5.20 release notes for sampling mask benchmarks (+17% @ batch 1, +52% @ batch 64).

## Quick Start

```bash
pip install sglang==0.5.20

# Qwen3-8B: disable thinking mode for clean latency benchmarks
python -m sglang.launch_server \
  --model Qwen/Qwen3-8B \
  --port 30000 \
  --chat-template qwen3 \
  --reasoning-parser qwen3
```

> **Qwen3 thinking mode note:** Qwen3-8B supports `<think>` reasoning tokens.
> For throughput/latency benchmarks, disable it by passing `/no_think` at the start
> of prompts, or set `enable_thinking=False` in sampling params.

---

## Lab 1 — HRRN Scheduler

**File:** `hrrn_scheduler.py`

Tests `--schedule-policy hrrn` vs the default FCFS on a mixed short/long workload.

```bash
# FCFS server (default)
python -m sglang.launch_server \
  --model Qwen/Qwen3-8B --port 30000 --chat-template qwen3

# HRRN server (new in v0.5.20)
python -m sglang.launch_server \
  --model Qwen/Qwen3-8B --port 30001 --chat-template qwen3 --schedule-policy hrrn

python -m open_model_engine.benchmarks.optimizations.sglang_v0520.hrrn_scheduler \
  --fcfs-url http://localhost:30000 \
  --hrrn-url http://localhost:30001 \
  --model Qwen/Qwen3-8B
```

**Expected:** Mean TTFT -69%, p99 -8% vs FCFS on mixed short/long workloads.

---

## Lab 2 — Sampling Masks for RL

**File:** `sampling_masks.py`

Tests `return_sampling_mask` — captures per-step token support and log-probs for RL rollout replay.
Benchmarks from the v0.5.20 release notes were run on **this exact model** (Qwen3-8B).

```bash
python -m sglang.launch_server \
  --model Qwen/Qwen3-8B --port 30000 \
  --chat-template qwen3 \
  --sampling-mask-max-tokens 4096

python -m open_model_engine.benchmarks.optimizations.sglang_v0520.sampling_masks \
  --url http://localhost:30000 \
  --model Qwen/Qwen3-8B
```

**Expected:** +17% decode throughput at batch 1, +52% at batch 64 (overlap scheduling).

---

## Lab 3 — Radix Cache & Branching-Point Caching

**File:** `radix_cache.py`

Measures token hit rate and TTFT across shared-prefix, unique-prefix, and branching workloads.

```bash
python -m open_model_engine.benchmarks.optimizations.sglang_v0520.radix_cache \
  --url http://localhost:30000 \
  --model Qwen/Qwen3-8B
```

**Expected:** 43.8% → 60.8% token hit rate; TTFT 1.57s → 1.07s on shared-prefix workloads.

---

## Lab 4 — CPU Simulator

**File:** `simulator.py`

Runs the SGLang simulator (no GPU needed) and compares predictions to real Qwen3-8B server measurements.

```bash
pip install sglang[simulator]

python -m open_model_engine.benchmarks.optimizations.sglang_v0520.simulator \
  --model Qwen/Qwen3-8B \
  --real-url http://localhost:30000
```

**Expected:** TTFT prediction within ~6%; prefix reuse within 0.05pp.
