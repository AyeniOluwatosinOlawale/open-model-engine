"""
vLLM v0.30.0 — Lab: Fast Start (IPC Weight Cache)
==================================================
Tests the new --load-format ipc_cache flag introduced in v0.30.0.

A persistent per-GPU weight-cache daemon holds post-quantized, TP-sharded
weights in GPU memory. Restarting the engine maps them over CUDA IPC
instead of reloading from disk.

This directly addresses Kubernetes pod restart latency — critical for:
  - Rolling updates
  - HPA scale-up cold starts
  - Crash recovery

Setup:
  # Start the weight cache daemon (run ONCE, stays running)
  python -m vllm.tools.weight_cache_daemon --model <model> --port 8100

  # Start vLLM WITHOUT IPC cache (baseline)
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8000 --load-format auto

  # Start vLLM WITH IPC cache (fast start)
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8001 --load-format ipc_cache

Usage:
  python -m open_model_engine.benchmarks.optimizations.vllm_v0300.fast_start \\
    --model <model> --n-restarts 3
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


HEALTH_ENDPOINT = "/health"
MODEL_ENDPOINT = "/v1/models"
READY_TIMEOUT = 300  # seconds


async def wait_for_ready(url: str, timeout: float = READY_TIMEOUT) -> float:
    """Poll health endpoint until server is ready. Returns startup time in seconds."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=5.0) as client:
        deadline = t0 + timeout
        while time.perf_counter() < deadline:
            try:
                r = await client.get(f"{url}{HEALTH_ENDPOINT}")
                if r.status_code == 200:
                    return time.perf_counter() - t0
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(1.0)
    raise TimeoutError(f"Server {url} did not become ready within {timeout}s")


async def first_token_latency(url: str, model: str) -> float:
    """Time to first token after server is ready — measures true serving readiness."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.perf_counter()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "stream": True,
        }
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    return (time.perf_counter() - t0) * 1000
    return 0.0


def start_server(model: str, port: int, load_format: str, extra_args: list[str]) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--load-format", load_format,
        "--max-model-len", "4096",
    ] + extra_args
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_server(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


async def measure_startup(
    model: str,
    port: int,
    load_format: str,
    extra_args: list[str],
    n_restarts: int,
    label: str,
) -> list[dict]:
    url = f"http://localhost:{port}"
    results = []

    for i in range(n_restarts):
        print(f"  [{label}] restart {i+1}/{n_restarts}...", end=" ", flush=True)
        proc = start_server(model, port, load_format, extra_args)
        try:
            startup_s = await wait_for_ready(url)
            first_tok_ms = await first_token_latency(url, model)
            print(f"ready in {startup_s:.1f}s  (first token: {first_tok_ms:.0f}ms)")
            results.append({
                "restart": i + 1,
                "startup_s": round(startup_s, 1),
                "first_token_ms": round(first_tok_ms, 1),
            })
        except TimeoutError:
            print("TIMEOUT")
            results.append({"restart": i + 1, "startup_s": None, "first_token_ms": None})
        finally:
            stop_server(proc)
            await asyncio.sleep(3)  # allow port to free

    return results


def summarise(results: list[dict]) -> dict:
    valid = [r for r in results if r["startup_s"] is not None]
    if not valid:
        return {"n": 0}
    startups = [r["startup_s"] for r in valid]
    first_toks = [r["first_token_ms"] for r in valid if r["first_token_ms"]]
    return {
        "n": len(valid),
        "startup_mean_s": round(statistics.mean(startups), 1),
        "startup_min_s": round(min(startups), 1),
        "first_token_mean_ms": round(statistics.mean(first_toks), 0) if first_toks else None,
    }


async def main(args: argparse.Namespace):
    print("=== vLLM v0.30.0: Fast Start (IPC Weight Cache) ===\n")
    print(f"Model:     {args.model}")
    print(f"Restarts:  {args.n_restarts} per mode\n")

    # Baseline — standard disk load
    print("--- Mode 1: Standard load (--load-format auto) ---")
    baseline_results = await measure_startup(
        model=args.model,
        port=8000,
        load_format="auto",
        extra_args=[],
        n_restarts=args.n_restarts,
        label="baseline",
    )

    # Fast start — IPC cache
    print("\n--- Mode 2: IPC cache (--load-format ipc_cache) ---")
    print("  Note: requires weight-cache daemon running on port 8100")
    ipc_results = await measure_startup(
        model=args.model,
        port=8001,
        load_format="ipc_cache",
        extra_args=["--weight-cache-port", "8100"],
        n_restarts=args.n_restarts,
        label="ipc_cache",
    )

    baseline_summary = summarise(baseline_results)
    ipc_summary = summarise(ipc_results)

    print("\n--- Comparison ---")
    print(f"{'':>30} {'Standard':>12} {'IPC Cache':>12} {'Speedup':>10}")
    print("-" * 68)
    if baseline_summary["n"] > 0 and ipc_summary["n"] > 0:
        speedup = baseline_summary["startup_mean_s"] / ipc_summary["startup_mean_s"]
        ft_speedup = (
            (baseline_summary["first_token_mean_ms"] / ipc_summary["first_token_mean_ms"])
            if baseline_summary["first_token_mean_ms"] and ipc_summary["first_token_mean_ms"]
            else None
        )
        print(f"{'Startup time (mean)':>30}  {baseline_summary['startup_mean_s']:>10.1f}s  {ipc_summary['startup_mean_s']:>10.1f}s  {speedup:>9.1f}×")
        if ft_speedup:
            print(f"{'First token latency':>30}  {baseline_summary['first_token_mean_ms']:>9.0f}ms  {ipc_summary['first_token_mean_ms']:>9.0f}ms  {ft_speedup:>9.1f}×")
    else:
        print("  One or both modes failed — check server logs")

    print("\nExpected from v0.30.0 release notes:")
    print("  IPC cache eliminates model reload from disk on restart")
    print("  K8s impact: rolling update pod restart time drops from minutes to seconds")

    if args.output:
        out = {
            "vllm_version": "0.30.0",
            "feature": "fast_start_ipc_cache",
            "baseline": {"results": baseline_results, "summary": baseline_summary},
            "ipc_cache": {"results": ipc_results, "summary": ipc_summary},
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM v0.30.0 Fast Start benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
