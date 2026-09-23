"""
vLLM v0.30.0 — Lab: HiSparse (Host-Resident KV Cache Tier)
===========================================================
Tests the new HiSparseConnector introduced in v0.30.0.

HiSparse adds a host-resident tier for sparse-MLA decode that spills KV
pages to pinned host memory under GPU memory pressure and serves top-k
misses from a per-request GPU hot buffer.

This is most relevant for DeepSeek-V3/V4 (MLA architecture) but the
cache pressure patterns apply broadly.

Setup:
  # Baseline: standard KV cache (no HiSparse)
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8000 --kv-cache-dtype fp8_e5m2

  # With HiSparse: host-resident tier for KV overflow
  python -m vllm.entrypoints.openai.api_server \\
    --model <model> --port 8001 \\
    --kv-cache-dtype fp8_e5m2 \\
    --kv-connector HiSparseConnector \\
    --kv-connector-host-cache-size-gb 16

Usage:
  python -m open_model_engine.benchmarks.optimizations.vllm_v0300.hisparse \\
    --baseline-url http://localhost:8000 \\
    --hisparse-url http://localhost:8001 \\
    --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx
import openai


LONG_CONTEXT_PROMPTS = [
    # These prompts produce long KV cache entries, stressing GPU memory
    "Read the following text and summarise the key points: " + ("The transformer architecture " * 200),
    "Analyse this code and find all bugs: " + ("def process(x): return x * 2\n" * 150),
    "Translate the following document: " + ("Machine learning models require data. " * 180),
    "Explain each step of this derivation: " + ("Given f(x) = x^2, then f'(x) = 2x. " * 160),
    "Extract all entities from: " + ("John Smith works at Google in London. " * 170),
    "Compare and contrast: " + ("Neural networks use layers of neurons. " * 190),
    "Summarise this research paper: " + ("Abstract: We present a novel approach. " * 175),
    "Review this document for errors: " + ("The system processes requests asynchronously. " * 165),
]

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name a planet.",
    "What colour is the sky?",
    "Say hello.",
    "Count to 5.",
    "What day is Monday?",
    "Name a fruit.",
    "Say goodbye.",
]


async def get_kv_metrics(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/metrics")
            if r.status_code != 200:
                return {}
            text = r.text
            metrics = {}
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                for key in [
                    "vllm:gpu_cache_usage_perc",
                    "vllm:cpu_cache_usage_perc",
                    "hisparse_host_cache_hit_total",
                    "hisparse_host_cache_miss_total",
                    "vllm:num_requests_running",
                ]:
                    if line.startswith(key + "{") or line.startswith(key + " "):
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            try:
                                metrics[key] = float(parts[1])
                            except ValueError:
                                pass
            return metrics
    except Exception:
        return {}


async def run_mixed_workload(
    url: str,
    model: str,
    n_long: int,
    n_short: int,
    concurrency: int,
) -> dict:
    """
    Run a mixed workload of long-context and short requests concurrently.
    Long requests fill the KV cache; short requests measure latency under pressure.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    long_ttfts: list[float] = []
    short_ttfts: list[float] = []
    preemptions = 0

    async def send(prompt: str, max_tokens: int, is_long: bool) -> float:
        async with sem:
            t0 = time.perf_counter()
            first_token = None
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=0.0,
                )
                async for chunk in stream:
                    if first_token is None and chunk.choices and chunk.choices[0].delta.content:
                        first_token = time.perf_counter()
                        break
            except Exception:
                pass
            ttft = ((first_token or time.perf_counter()) - t0) * 1000
            if is_long:
                long_ttfts.append(ttft)
            else:
                short_ttfts.append(ttft)
            return ttft

    long_prompts = (LONG_CONTEXT_PROMPTS * max(1, n_long // len(LONG_CONTEXT_PROMPTS)))[:n_long]
    short_prompts_list = (SHORT_PROMPTS * max(1, n_short // len(SHORT_PROMPTS)))[:n_short]

    tasks = (
        [send(p, 200, True) for p in long_prompts]
        + [send(p, 20, False) for p in short_prompts_list]
    )

    metrics_before = await get_kv_metrics(url)
    await asyncio.gather(*tasks)
    metrics_after = await get_kv_metrics(url)
    await client.close()

    def safe_stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0, "p95": 0, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 1),
            "p95": round(sorted(vals)[int(len(vals) * 0.95)], 1) if len(vals) >= 20 else round(max(vals), 1),
            "n": len(vals),
        }

    return {
        "long_ttft": safe_stats(long_ttfts),
        "short_ttft": safe_stats(short_ttfts),
        "gpu_cache_usage_after": metrics_after.get("vllm:gpu_cache_usage_perc", 0.0),
        "cpu_cache_usage_after": metrics_after.get("vllm:cpu_cache_usage_perc", 0.0),
        "hisparse_hits": metrics_after.get("hisparse_host_cache_hit_total", 0),
        "hisparse_misses": metrics_after.get("hisparse_host_cache_miss_total", 0),
    }


async def main(args: argparse.Namespace):
    print("=== vLLM v0.30.0: HiSparse Host-Resident KV Cache ===\n")
    print("Workload: mixed long-context (fills GPU KV cache) + short requests")
    print(f"  Long requests:  {args.n_long}  (stress the KV cache)")
    print(f"  Short requests: {args.n_short} (measure latency under pressure)")
    print(f"  Concurrency:    {args.concurrency}\n")

    print("--- Baseline (standard KV cache) ---")
    baseline = await run_mixed_workload(
        args.baseline_url, args.model, args.n_long, args.n_short, args.concurrency
    )

    print("--- HiSparse (host-resident tier) ---")
    hisparse = await run_mixed_workload(
        args.hisparse_url, args.model, args.n_long, args.n_short, args.concurrency
    )

    print("\n--- Comparison ---")
    print(f"{'Metric':>30} {'Baseline':>12} {'HiSparse':>12} {'Delta':>10}")
    print("-" * 68)

    metrics = [
        ("Short TTFT mean (ms)", baseline["short_ttft"]["mean"], hisparse["short_ttft"]["mean"]),
        ("Short TTFT p95  (ms)", baseline["short_ttft"]["p95"],  hisparse["short_ttft"]["p95"]),
        ("Long  TTFT mean (ms)", baseline["long_ttft"]["mean"],  hisparse["long_ttft"]["mean"]),
        ("GPU cache usage (%)",  baseline["gpu_cache_usage_after"] * 100, hisparse["gpu_cache_usage_after"] * 100),
        ("CPU cache usage (%)",  baseline["cpu_cache_usage_after"] * 100, hisparse["cpu_cache_usage_after"] * 100),
    ]

    for name, b_val, h_val in metrics:
        if b_val > 0:
            delta = (h_val - b_val) / b_val * 100
            sign = "▼" if delta < 0 else "▲"
            print(f"{name:>30}  {b_val:>10.1f}  {h_val:>10.1f}  {sign}{abs(delta):.1f}%")
        else:
            print(f"{name:>30}  {'N/A':>10}  {h_val:>10.1f}  {'—':>10}")

    if hisparse["hisparse_hits"] or hisparse["hisparse_misses"]:
        total = hisparse["hisparse_hits"] + hisparse["hisparse_misses"]
        hit_rate = hisparse["hisparse_hits"] / total if total > 0 else 0
        print(f"\n  HiSparse host cache hit rate: {hit_rate:.1%} ({hisparse['hisparse_hits']:.0f} hits / {total:.0f} total)")

    print("\nExpected from v0.30.0 release notes:")
    print("  HiSparse spills KV pages to pinned host memory under GPU pressure")
    print("  Serves top-k misses from per-request GPU hot buffer")
    print("  Enables higher effective KV capacity per GPU")

    if args.output:
        out = {
            "vllm_version": "0.30.0",
            "feature": "hisparse",
            "baseline": baseline,
            "hisparse": hisparse,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM v0.30.0 HiSparse benchmark")
    parser.add_argument("--baseline-url", default="http://localhost:8000")
    parser.add_argument("--hisparse-url", default="http://localhost:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-long", type=int, default=20)
    parser.add_argument("--n-short", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
