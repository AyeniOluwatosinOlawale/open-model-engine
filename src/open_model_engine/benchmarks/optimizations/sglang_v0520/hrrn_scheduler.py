"""
SGLang v0.5.20 — Lab: HRRN vs FCFS Scheduler Comparison
=========================================================
Tests the new --schedule-policy hrrn flag introduced in v0.5.20.

HRRN (Highest Response Ratio Next) prioritises requests that have waited
the longest relative to their estimated service time, dramatically reducing
mean TTFT for workloads with mixed short and long requests.

Reported improvement: mean TTFT -69%, p99 -8% vs FCFS on GLM-5.2 production trace.

Setup:
  # Start SGLang with FCFS (default)
  python -m sglang.launch_server --model <model> --port 30000

  # Start SGLang with HRRN (new in v0.5.20)
  python -m sglang.launch_server --model <model> --port 30001 --schedule-policy hrrn

Usage:
  python -m open_model_engine.benchmarks.optimizations.sglang_v0520.hrrn_scheduler \\
    --fcfs-url http://localhost:30000 \\
    --hrrn-url http://localhost:30001 \\
    --model <model-name> \\
    --concurrency 16 \\
    --runs 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx
import openai


@dataclass
class RequestResult:
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tpot_ms: float
    queued_at: float
    completed_at: float


@dataclass
class SchedulerResult:
    policy: str
    url: str
    results: list[RequestResult] = field(default_factory=list)

    def ttft_stats(self) -> dict:
        ttfts = [r.ttft_ms for r in self.results]
        return {
            "mean": round(statistics.mean(ttfts), 1),
            "median": round(statistics.median(ttfts), 1),
            "p95": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1),
            "p99": round(sorted(ttfts)[int(len(ttfts) * 0.99)], 1),
            "max": round(max(ttfts), 1),
        }

    def tpot_stats(self) -> dict:
        tpots = [r.tpot_ms for r in self.results]
        return {
            "mean": round(statistics.mean(tpots), 1),
            "p95": round(sorted(tpots)[int(len(tpots) * 0.95)], 1),
        }


# Mixed workload: some short requests (fast), some long (slow)
# This is where HRRN shines — short requests don't wait behind long ones
WORKLOAD: list[dict] = (
    [{"prompt": "What is 2+2?", "max_tokens": 20}] * 10           # short
    + [{"prompt": "Write a detailed essay on the history of AI, covering all major milestones from 1950 to 2024.", "max_tokens": 600}] * 5  # long
    + [{"prompt": "Name three planets.", "max_tokens": 30}] * 10  # short
    + [{"prompt": "Explain in detail how transformer attention works, including the mathematical formulation.", "max_tokens": 500}] * 5  # long
)


async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    prompt_tokens = len(prompt.split()) + 10  # rough estimate

    queued_at = time.perf_counter()
    first_token_at: float | None = None
    output_tokens = 0

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        temperature=0.0,
    )

    async for chunk in stream:
        if first_token_at is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_at = time.perf_counter()
        if chunk.choices and chunk.choices[0].delta.content:
            output_tokens += 1

    completed_at = time.perf_counter()
    ttft_ms = ((first_token_at or completed_at) - queued_at) * 1000
    total_ms = (completed_at - queued_at) * 1000
    tpot_ms = (total_ms - ttft_ms) / max(output_tokens - 1, 1)

    return RequestResult(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tpot_ms=tpot_ms,
        queued_at=queued_at,
        completed_at=completed_at,
    )


async def run_workload(
    url: str,
    model: str,
    workload: list[dict],
    concurrency: int,
    policy: str,
) -> SchedulerResult:
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    result = SchedulerResult(policy=policy, url=url)
    sem = asyncio.Semaphore(concurrency)

    async def bounded(req: dict) -> RequestResult:
        async with sem:
            return await send_request(client, model, req["prompt"], req["max_tokens"])

    tasks = [bounded(req) for req in workload]
    result.results = await asyncio.gather(*tasks)
    await client.close()
    return result


async def check_sglang_version(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/get_server_info")
            data = r.json()
            return data.get("version", "unknown")
    except Exception:
        return "unreachable"


async def main(args: argparse.Namespace):
    print("=== SGLang v0.5.20: HRRN vs FCFS Scheduler ===\n")

    # Verify servers
    for label, url in [("FCFS", args.fcfs_url), ("HRRN", args.hrrn_url)]:
        ver = await check_sglang_version(url)
        status = "✓" if ver != "unreachable" else "✗ UNREACHABLE"
        print(f"  [{label}] {url}  — SGLang {ver}  {status}")
    print()

    workload = (WORKLOAD * max(1, args.runs // len(WORKLOAD)))[:args.runs]
    print(f"Workload: {len(workload)} requests ({args.concurrency} concurrent)")
    short = sum(1 for r in workload if r["max_tokens"] <= 50)
    long = len(workload) - short
    print(f"  Short requests (≤50 tokens): {short}")
    print(f"  Long  requests (>50 tokens): {long}")
    print()

    print("Running FCFS...", end=" ", flush=True)
    fcfs = await run_workload(args.fcfs_url, args.model, workload, args.concurrency, "FCFS")
    print(f"done ({len(fcfs.results)} results)")

    print("Running HRRN...", end=" ", flush=True)
    hrrn = await run_workload(args.hrrn_url, args.model, workload, args.concurrency, "HRRN")
    print(f"done ({len(hrrn.results)} results)\n")

    # Print comparison
    print(f"{'Metric':>20}  {'FCFS':>12}  {'HRRN':>12}  {'Delta':>10}")
    print("-" * 60)

    fcfs_ttft = fcfs.ttft_stats()
    hrrn_ttft = hrrn.ttft_stats()

    for key in ["mean", "median", "p95", "p99"]:
        f_val = fcfs_ttft[key]
        h_val = hrrn_ttft[key]
        delta = (h_val - f_val) / f_val * 100
        sign = "▼" if delta < 0 else "▲"
        print(f"{'TTFT ' + key:>20}  {f_val:>10.1f}ms  {h_val:>10.1f}ms  {sign}{abs(delta):.1f}%")

    fcfs_tpot = fcfs.tpot_stats()
    hrrn_tpot = hrrn.tpot_stats()
    for key in ["mean"]:
        f_val = fcfs_tpot[key]
        h_val = hrrn_tpot[key]
        delta = (h_val - f_val) / f_val * 100
        sign = "▼" if delta < 0 else "▲"
        print(f"{'TPOT ' + key:>20}  {f_val:>10.1f}ms  {h_val:>10.1f}ms  {sign}{abs(delta):.1f}%")

    mean_delta = (hrrn_ttft["mean"] - fcfs_ttft["mean"]) / fcfs_ttft["mean"] * 100
    print(f"\nConclusion: HRRN mean TTFT {'improved' if mean_delta < 0 else 'degraded'} by {abs(mean_delta):.1f}%")
    print("(Expected from SGLang v0.5.20 release notes: -69% mean TTFT on production traces)")

    if args.output:
        out = {
            "sglang_version": "0.5.20",
            "feature": "hrrn_scheduler",
            "workload_size": len(workload),
            "concurrency": args.concurrency,
            "fcfs": {"ttft": fcfs_ttft, "tpot": fcfs_tpot},
            "hrrn": {"ttft": hrrn_ttft, "tpot": hrrn_tpot},
            "ttft_mean_delta_pct": round(mean_delta, 1),
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang v0.5.20 HRRN vs FCFS benchmark")
    parser.add_argument("--fcfs-url", default="http://localhost:30000")
    parser.add_argument("--hrrn-url", default="http://localhost:30001")
    parser.add_argument("--model", required=True, help="Model name served by both servers")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", help="Save JSON results to file")
    asyncio.run(main(parser.parse_args()))
