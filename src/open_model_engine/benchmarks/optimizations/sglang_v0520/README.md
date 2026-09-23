# SGLang v0.5.20 — New Feature Labs

Four labs targeting the major new features in the v0.5.20 release.

## Quick Start

```bash
# Start SGLang v0.5.20
pip install sglang==0.5.20
python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 30000
```

---

## Lab 1 — HRRN Scheduler

**File:** `hrrn_scheduler.py`

Tests `--schedule-policy hrrn` vs the default FCFS on a mixed short/long workload.

```bash
# Start two servers: FCFS (port 30000) and HRRN (port 30001)
python -m sglang.launch_server --model <model> --port 30000
python -m sglang.launch_server --model <model> --port 30001 --schedule-policy hrrn

python hrrn_scheduler.py --fcfs-url http://localhost:30000 --hrrn-url http://localhost:30001 --model <model>
```

**Expected:** Mean TTFT -69%, p99 -8% vs FCFS on mixed workloads.

---

## Lab 2 — Sampling Masks for RL

**File:** `sampling_masks.py`

Tests `return_sampling_mask` — captures per-step token support and log-probs for RL rollout replay.

```bash
python -m sglang.launch_server --model <model> --port 30000 --sampling-mask-max-tokens 4096
python sampling_masks.py --url http://localhost:30000 --model <model>
```

**Expected:** Mask overhead negligible under overlap scheduling; +17–52% decode throughput.

---

## Lab 3 — Radix Cache & Branching-Point Caching

**File:** `radix_cache.py`

Measures token hit rate and TTFT across shared-prefix, unique-prefix, and branching workloads.

```bash
python radix_cache.py --url http://localhost:30000 --model <model>
```

**Expected:** 43.8% → 60.8% token hit rate; TTFT 1.57s → 1.07s on shared-prefix workloads.

---

## Lab 4 — CPU Simulator

**File:** `simulator.py`

Runs the SGLang simulator against a synthetic workload and compares predictions to real server measurements.

```bash
pip install sglang[simulator]
python simulator.py --model <model-path> --real-url http://localhost:30000
```

**Expected:** TTFT prediction within ~6%; prefix reuse within 0.05pp. Runs CPU-only.
