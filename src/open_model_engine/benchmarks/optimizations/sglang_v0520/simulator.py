"""
SGLang v0.5.20 — Lab: CPU Inference Simulator
==============================================
Exercises the new SGLang Simulator introduced in v0.5.20.
The simulator reuses the real scheduler, radix cache, and hierarchical
cache with a latency predictor in place of the model forward pass.

Accuracy from release notes:
  - TTFT prediction within ~6% on most traces
  - Prefix reuse within 0.05pp
  - No GPU required

This lab:
  1. Runs the simulator against a synthetic workload
  2. Runs the real server against the same workload
  3. Compares predicted vs actual TTFT and prefix hit rates

Setup:
  # Install SGLang simulator (CPU-only, no GPU needed)
  pip install sglang[simulator]

  # Also start the real server for comparison
  python -m sglang.launch_server --model <model> --port 30000

Usage:
  python -m open_model_engine.benchmarks.optimizations.sglang_v0520.simulator \\
    --model <model-path> \\
    --real-url http://localhost:30000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import openai


@dataclass
class SimulatorTrace:
    request_id: str
    arrival_time_s: float
    prompt_tokens: int
    output_tokens: int
    predicted_ttft_ms: float = 0.0
    actual_ttft_ms: float = 0.0
    predicted_hit_rate: float = 0.0
    actual_hit_rate: float = 0.0


def generate_synthetic_workload(
    n_requests: int = 50,
    arrival_rate_rps: float = 2.0,
    shared_prefix_pct: float = 0.6,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a synthetic workload trace matching the format expected
    by the SGLang simulator.
    shared_prefix_pct: fraction of requests that share a long system prompt.
    """
    import random
    rng = random.Random(seed)

    shared_system = (
        "You are an expert AI assistant with deep knowledge of software engineering, "
        "mathematics, and science. Always be precise and cite your reasoning. " * 8
    )
    unique_topics = [
        "Python async programming", "Rust ownership model", "SQL query optimisation",
        "Docker networking", "Kubernetes autoscaling", "Redis caching strategies",
        "gRPC vs REST", "B-tree indexes", "MVCC in PostgreSQL", "TCP congestion control",
        "TLS certificate pinning", "OAuth2 flows", "WebSocket vs SSE", "CRDT data structures",
        "consistent hashing", "Bloom filters", "LSM trees", "Raft consensus", "Paxos algorithm",
        "vector clocks", "event sourcing", "CQRS pattern", "saga pattern", "outbox pattern",
    ]

    requests = []
    t = 0.0
    for i in range(n_requests):
        inter_arrival = rng.expovariate(arrival_rate_rps)
        t += inter_arrival

        use_shared = rng.random() < shared_prefix_pct
        topic = rng.choice(unique_topics)
        if use_shared:
            prompt = shared_system + f"\n\nUser: Explain {topic} in detail."
        else:
            prompt = f"You are an expert. Explain {topic} in detail."

        prompt_tokens = max(50, len(prompt.split()) + rng.randint(-10, 20))
        output_tokens = rng.randint(50, 300)

        requests.append({
            "request_id": f"req_{i:04d}",
            "arrival_time_s": round(t, 3),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_output_tokens": output_tokens,
            "use_shared_prefix": use_shared,
        })

    return requests


def run_simulator(
    model_path: str,
    workload: list[dict],
    output_path: str,
) -> dict | None:
    """
    Invoke the SGLang simulator CLI.
    Falls back gracefully if simulator is not installed.
    """
    try:
        import sglang.tools.simulator as _  # noqa: F401 — check import
    except ImportError:
        return None

    trace_file = Path(output_path) / "trace.jsonl"
    result_file = Path(output_path) / "sim_results.json"

    with open(trace_file, "w") as f:
        for req in workload:
            f.write(json.dumps({
                "request_id": req["request_id"],
                "arrival_time": req["arrival_time_s"],
                "prompt_len": req["prompt_tokens"],
                "output_len": req["max_output_tokens"],
            }) + "\n")

    cmd = [
        sys.executable, "-m", "sglang.tools.simulator",
        "--model", model_path,
        "--trace", str(trace_file),
        "--output", str(result_file),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        with open(result_file) as f:
            return json.load(f)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


async def run_real_server(
    url: str,
    model: str,
    workload: list[dict],
    concurrency: int = 8,
) -> list[dict]:
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(req: dict) -> dict:
        async with sem:
            t0 = time.perf_counter()
            first_token = None
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": req["prompt"]}],
                max_tokens=req["max_output_tokens"],
                stream=True,
                temperature=0.0,
            )
            async for chunk in stream:
                if first_token is None and chunk.choices and chunk.choices[0].delta.content:
                    first_token = time.perf_counter()
                    break
            ttft = ((first_token or time.perf_counter()) - t0) * 1000
            return {"request_id": req["request_id"], "actual_ttft_ms": round(ttft, 1)}

    tasks = [one(req) for req in workload]
    results = await asyncio.gather(*tasks)
    await client.close()
    return results


async def get_hit_rate(url: str) -> float:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/server_info")
            data = r.json()
            return float(
                data.get("cache_hit_rate")
                or data.get("kv_cache_hit_rate")
                or data.get("token_hit_rate")
                or 0.0
            )
    except Exception:
        return 0.0


def compare_predictions(sim_results: dict | None, actual_results: list[dict]) -> dict:
    """Compare simulator predictions against actual server measurements."""
    if not sim_results:
        return {"simulator_available": False}

    predicted = {r["request_id"]: r["predicted_ttft_ms"] for r in sim_results.get("requests", [])}
    actual = {r["request_id"]: r["actual_ttft_ms"] for r in actual_results}

    common = set(predicted) & set(actual)
    if not common:
        return {"simulator_available": True, "common_requests": 0}

    errors = [
        abs(predicted[rid] - actual[rid]) / actual[rid] * 100
        for rid in common
        if actual[rid] > 0
    ]

    import statistics
    return {
        "simulator_available": True,
        "common_requests": len(common),
        "mean_ttft_error_pct": round(statistics.mean(errors), 1),
        "p95_ttft_error_pct": round(sorted(errors)[int(len(errors) * 0.95)], 1),
        "predicted_hit_rate": sim_results.get("summary", {}).get("cache_hit_rate", 0.0),
    }


async def main(args: argparse.Namespace):
    print("=== SGLang v0.5.20: CPU Inference Simulator ===\n")

    print("Generating synthetic workload...")
    workload = generate_synthetic_workload(
        n_requests=args.n_requests,
        arrival_rate_rps=args.arrival_rate,
        shared_prefix_pct=args.shared_prefix_pct,
    )
    shared = sum(1 for r in workload if r["use_shared_prefix"])
    print(f"  {len(workload)} requests — {shared} shared-prefix ({shared/len(workload):.0%}), {len(workload)-shared} unique")
    print(f"  Arrival rate: {args.arrival_rate} rps")
    print(f"  Duration: {workload[-1]['arrival_time_s']:.1f}s simulated time\n")

    # Simulator
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Running SGLang simulator (CPU, no GPU)...", end=" ", flush=True)
        t0 = time.perf_counter()
        sim_results = run_simulator(args.model, workload, tmpdir)
        sim_elapsed = time.perf_counter() - t0
        if sim_results:
            print(f"done in {sim_elapsed:.1f}s")
        else:
            print("simulator not installed (pip install sglang[simulator])")

    # Real server
    actual_results = []
    actual_hit_rate = 0.0
    if args.real_url:
        print("Running real server...", end=" ", flush=True)
        actual_results = await run_real_server(args.real_url, args.model_name, workload, args.concurrency)
        actual_hit_rate = await get_hit_rate(args.real_url)
        print(f"done ({len(actual_results)} results)")

    # Comparison
    comparison = compare_predictions(sim_results, actual_results)

    print("\n--- Results ---")
    if comparison.get("simulator_available"):
        print(f"  Simulator TTFT mean error:  {comparison.get('mean_ttft_error_pct', 'N/A')}%")
        print(f"  Simulator TTFT p95 error:   {comparison.get('p95_ttft_error_pct', 'N/A')}%")
        print(f"  Predicted hit rate:         {comparison.get('predicted_hit_rate', 0):.1%}")
        if actual_hit_rate:
            hr_delta = abs(comparison.get("predicted_hit_rate", 0) - actual_hit_rate)
            print(f"  Actual hit rate:            {actual_hit_rate:.1%}  (delta: {hr_delta:.3f}pp)")
    else:
        print("  Simulator not available — install with: pip install sglang[simulator]")
        if actual_results:
            import statistics
            ttfts = [r["actual_ttft_ms"] for r in actual_results]
            print(f"\n  Real server results ({len(ttfts)} requests):")
            print(f"  TTFT mean:   {statistics.mean(ttfts):.1f} ms")
            print(f"  TTFT p95:    {sorted(ttfts)[int(len(ttfts)*0.95)]:.1f} ms")
            print(f"  Cache hit:   {actual_hit_rate:.1%}")

    print("\nExpected from v0.5.20 release notes:")
    print("  TTFT predicted within ~6% on most traces (up to 10% on 32K–128K inputs)")
    print("  Prefix reuse within 0.05 percentage points")
    print("  Simulator runs on CPU only — no GPU needed for scheduling/cache studies")

    if args.output:
        out = {
            "sglang_version": "0.5.20",
            "feature": "cpu_simulator",
            "workload_size": len(workload),
            "comparison": comparison,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang v0.5.20 simulator benchmark")
    parser.add_argument("--model", required=True, help="Model path (for simulator)")
    parser.add_argument("--model-name", help="Model name for real server (if different)")
    parser.add_argument("--real-url", help="Real SGLang server URL for comparison")
    parser.add_argument("--n-requests", type=int, default=50)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--shared-prefix-pct", type=float, default=0.6)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = args.model
    asyncio.run(main(args))
