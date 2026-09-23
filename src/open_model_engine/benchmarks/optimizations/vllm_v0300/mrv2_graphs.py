"""
vLLM v0.30.0 — Lab: Model Runner V2 CUDA Graph Capture Time
============================================================
Measures the CUDA graph capture time improvement in MRV2 (Model Runner V2).

v0.30.0 freezes the garbage collector during graph capture, cutting:
  - Graph capture:  12s → 2s
  - Engine init:    28.9s → 8.2s  (on H200 with Qwen3.5)

This lab:
  1. Measures total engine startup time (includes graph capture)
  2. Times the first request (proxy for warmup completeness)
  3. Measures CUDA graph replay throughput vs eager mode

Setup:
  # MRV2 is the default in v0.30.0
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8000

  # Eager mode (no CUDA graphs) for comparison
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8001 --enforce-eager

Usage:
  python -m open_model_engine.benchmarks.optimizations.vllm_v0300.mrv2_graphs \\
    --model <model> \\
    --graph-url http://localhost:8000 \\
    --eager-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time

import httpx
import openai


async def wait_for_ready(url: str, timeout: float = 300.0) -> float:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = t0 + timeout
        while time.perf_counter() < deadline:
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return time.perf_counter() - t0
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Server not ready within {timeout}s")


async def throughput_sweep(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    batch_sizes: list[int],
    n_runs: int = 5,
) -> dict[int, dict]:
    """Measure decode throughput at different batch sizes."""
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    results: dict[int, dict] = {}

    for bs in batch_sizes:
        tpots: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            first_token = None
            output_tokens = 0
            tasks = []

            async def single_request() -> tuple[float | None, int]:
                nonlocal first_token
                ft = None
                toks = 0
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=0.0,
                )
                async for chunk in stream:
                    if ft is None and chunk.choices and chunk.choices[0].delta.content:
                        ft = time.perf_counter()
                    if chunk.choices and chunk.choices[0].delta.content:
                        toks += 1
                return ft, toks

            responses = await asyncio.gather(*[single_request() for _ in range(bs)])
            total_time = time.perf_counter() - t0

            total_output = sum(toks for _, toks in responses)
            tps = total_output / total_time if total_time > 0 else 0
            tpots.append(tps)

        results[bs] = {
            "throughput_tps": round(statistics.mean(tpots), 1),
            "throughput_std": round(statistics.stdev(tpots) if len(tpots) > 1 else 0, 1),
        }

    await client.close()
    return results


def start_server(model: str, port: int, extra_args: list[str]) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
    ] + extra_args
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


async def measure_server(
    model: str,
    port: int,
    mode: str,
    extra_args: list[str],
    batch_sizes: list[int],
) -> dict:
    url = f"http://localhost:{port}"
    print(f"  Starting {mode}...", end=" ", flush=True)
    proc = start_server(model, port, extra_args)
    try:
        startup_s = await wait_for_ready(url)
        print(f"ready in {startup_s:.1f}s")

        print(f"  Running throughput sweep ({batch_sizes})...", end=" ", flush=True)
        sweep = await throughput_sweep(
            url, model,
            prompt="Explain the theory of relativity in detail.",
            max_tokens=100,
            batch_sizes=batch_sizes,
        )
        print("done")
        return {"startup_s": round(startup_s, 1), "throughput": sweep}
    finally:
        stop_server(proc)
        await asyncio.sleep(3)


async def main(args: argparse.Namespace):
    print("=== vLLM v0.30.0: Model Runner V2 CUDA Graph Capture ===\n")

    # If servers already running, use them; otherwise start managed
    using_live_servers = bool(args.graph_url or args.eager_url)

    if using_live_servers:
        graph_url = args.graph_url or "http://localhost:8000"
        eager_url = args.eager_url or "http://localhost:8001"
        batch_sizes = [1, 4, 8, 16]

        async def sweep(url: str, label: str) -> dict:
            print(f"  [{label}] running throughput sweep...", end=" ", flush=True)
            r = await throughput_sweep(
                url, args.model,
                "Explain gradient descent in machine learning.",
                max_tokens=100,
                batch_sizes=batch_sizes,
            )
            print("done")
            return r

        graph_sweep = await sweep(graph_url, "CUDA graphs")
        eager_sweep = await sweep(eager_url, "eager")

        print("\n--- Throughput Comparison ---")
        print(f"{'Batch':>8} {'CUDA graphs (tps)':>20} {'Eager (tps)':>14} {'Speedup':>10}")
        print("-" * 56)
        for bs in batch_sizes:
            g = graph_sweep.get(bs, {}).get("throughput_tps", 0)
            e = eager_sweep.get(bs, {}).get("throughput_tps", 0)
            speedup = g / e if e > 0 else 0
            print(f"{bs:>8} {g:>20.1f} {e:>14.1f} {speedup:>9.2f}×")

        if args.output:
            out = {
                "vllm_version": "0.30.0",
                "feature": "mrv2_cuda_graphs",
                "cuda_graphs": {"throughput": graph_sweep},
                "eager": {"throughput": eager_sweep},
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        batch_sizes = [1, 4, 8]
        print("Starting managed servers (measuring startup time)...\n")

        print("--- MRV2 with CUDA graphs (default) ---")
        graph_result = await measure_server(
            args.model, 8000, "MRV2+graphs", [], batch_sizes
        )

        print("\n--- MRV2 eager mode (--enforce-eager) ---")
        eager_result = await measure_server(
            args.model, 8001, "eager", ["--enforce-eager"], batch_sizes
        )

        print("\n--- Results ---")
        print(f"{'Mode':>25} {'Startup':>12} {'Throughput @bs=1':>18} {'@bs=8':>10}")
        print("-" * 68)
        for label, result in [("CUDA graphs (MRV2)", graph_result), ("Eager mode", eager_result)]:
            bs1 = result["throughput"].get(1, {}).get("throughput_tps", 0)
            bs8 = result["throughput"].get(8, {}).get("throughput_tps", 0)
            print(f"{label:>25} {result['startup_s']:>10.1f}s {bs1:>17.1f} {bs8:>10.1f}")

    print("\nExpected from v0.30.0 release notes (H200, Qwen3.5):")
    print("  Graph capture:  12s → 2s  (GC frozen during capture)")
    print("  Engine init:    28.9s → 8.2s")
    print("  CUDA graphs consistently outperform eager at batch ≥ 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM v0.30.0 MRV2 CUDA graph benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph-url", help="URL of running CUDA-graph server (optional)")
    parser.add_argument("--eager-url", help="URL of running eager server (optional)")
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
