"""
SGLang v0.5.20 — Lab: Unified Radix Tree & Branching-Point Caching
===================================================================
Tests the new branching-point caching for the SWA (Sliding Window Attention)
component introduced in v0.5.20.

Branches that diverge from a shared prefix now reuse the SWA state computed
at the fork point instead of recomputing it.

Reported improvement on DeepSeek-V4-Flash:
  - Token hit rate: 43.8% → 60.8%
  - Mean TTFT:      1.57s  → 1.07s

Setup:
  # Both old and new behaviour controlled by the same server in v0.5.20.
  # We measure cache hit rate on shared-prefix vs unique-prefix workloads.
  python -m sglang.launch_server --model <model> --port 30000

Usage:
  python -m open_model_engine.benchmarks.optimizations.sglang_v0520.radix_cache \\
    --url http://localhost:30000 --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx
import openai

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialised in software engineering. "
    "You provide concise, accurate answers. Always explain your reasoning step by step. "
    "When writing code, prefer Python 3.11+ syntax. Focus on readability and correctness. "
    "If asked about a bug, identify the root cause before suggesting a fix. "
    * 3  # ~200 tokens — long enough to benefit from caching
)

UNIQUE_QUESTIONS = [
    "How do I reverse a linked list in Python?",
    "What is the time complexity of QuickSort?",
    "Explain dependency injection.",
    "What is a deadlock and how do you prevent it?",
    "How does garbage collection work in Python?",
    "What is the difference between TCP and UDP?",
    "Explain the SOLID principles.",
    "What is a race condition?",
    "How does a hash table handle collisions?",
    "What is memoization?",
    "Explain the CAP theorem.",
    "What is idempotency in REST APIs?",
    "How does TLS handshake work?",
    "What is tail call optimisation?",
    "Explain eventual consistency.",
    "What is a B-tree?",
]


async def get_cache_stats(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            # /server_info is the current endpoint; /get_server_info is deprecated
            r = await c.get(f"{url}/server_info")
            data = r.json()
            hit_rate = (
                data.get("cache_hit_rate")
                or data.get("kv_cache_hit_rate")
                or data.get("token_hit_rate")
                or 0.0
            )
            return {
                "cache_hit_rate": round(float(hit_rate), 4),
                "num_cached_tokens": data.get("num_cached_tokens", 0),
                "num_total_tokens": data.get("num_total_tokens", 0),
            }
    except Exception:
        return {"cache_hit_rate": 0.0, "num_cached_tokens": 0, "num_total_tokens": 0}


async def flush_cache(url: str):
    """Ask SGLang to flush its prefix cache between experiments."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(f"{url}/flush_cache")
    except Exception:
        pass  # endpoint may not exist on all versions


async def run_batch(
    client: openai.AsyncOpenAI,
    model: str,
    messages_list: list[list[dict]],
    concurrency: int = 8,
) -> list[float]:
    """Send a batch of requests and return TTFT for each."""
    sem = asyncio.Semaphore(concurrency)

    async def one(messages: list[dict]) -> float:
        async with sem:
            t0 = time.perf_counter()
            first_token = None
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,        # measure prefill (TTFT) only, not decode
                stream=True,
                temperature=0.0,
            )
            async for chunk in stream:
                if first_token is None and chunk.choices and chunk.choices[0].delta.content:
                    first_token = time.perf_counter()
                    break
            return ((first_token or time.perf_counter()) - t0) * 1000

    return await asyncio.gather(*[one(msgs) for msgs in messages_list])


async def experiment_shared_prefix(
    url: str,
    model: str,
    questions: list[str],
    concurrency: int,
) -> dict:
    """All requests share the same long system prompt — exercises prefix cache."""
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    await flush_cache(url)

    # Cold batch — fills the cache
    cold_messages = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        for q in questions
    ]
    stats_before = await get_cache_stats(url)
    cold_ttfts = await run_batch(client, model, cold_messages, concurrency)
    stats_after_cold = await get_cache_stats(url)

    # Warm batch — all share the same prefix, now cached
    warm_ttfts = await run_batch(client, model, cold_messages, concurrency)
    stats_after_warm = await get_cache_stats(url)

    await client.close()
    return {
        "cold_ttft_mean_ms": round(statistics.mean(cold_ttfts), 1),
        "warm_ttft_mean_ms": round(statistics.mean(warm_ttfts), 1),
        "cache_hit_rate_cold": stats_after_cold["cache_hit_rate"],
        "cache_hit_rate_warm": stats_after_warm["cache_hit_rate"],
        "speedup": round(statistics.mean(cold_ttfts) / statistics.mean(warm_ttfts), 2),
    }


async def experiment_unique_prefix(
    url: str,
    model: str,
    questions: list[str],
    concurrency: int,
) -> dict:
    """Each request has a unique system prompt — cache provides no benefit."""
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    await flush_cache(url)

    unique_messages = [
        [
            {"role": "system", "content": f"You are assistant #{i}. " + SYSTEM_PROMPT[:100]},
            {"role": "user", "content": q},
        ]
        for i, q in enumerate(questions)
    ]
    ttfts = await run_batch(client, model, unique_messages, concurrency)
    stats = await get_cache_stats(url)
    await client.close()
    return {
        "ttft_mean_ms": round(statistics.mean(ttfts), 1),
        "cache_hit_rate": stats["cache_hit_rate"],
    }


async def experiment_branching(
    url: str,
    model: str,
    questions: list[str],
    concurrency: int,
) -> dict:
    """
    Branching-point caching experiment (key new feature in v0.5.20).
    All requests share a prefix up to a fork, then diverge.
    Branching-point caching reuses the SWA state at the fork.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")

    shared_prefix = SYSTEM_PROMPT
    branch_a = "Focus only on Python solutions."
    branch_b = "Focus only on JavaScript solutions."
    branch_c = "Focus only on Rust solutions."
    branches = [branch_a, branch_b, branch_c]

    await flush_cache(url)

    # First: prime the shared prefix
    prime_msgs = [
        [
            {"role": "system", "content": shared_prefix},
            {"role": "user", "content": questions[0]},
        ]
    ]
    await run_batch(client, model, prime_msgs, 1)

    # Now run branching requests — should reuse up to the fork
    branch_messages = [
        [
            {"role": "system", "content": shared_prefix},
            {"role": "user", "content": branch},
            {"role": "assistant", "content": "Sure, I'll focus on that language."},
            {"role": "user", "content": q},
        ]
        for branch in branches
        for q in questions[:4]
    ]

    stats_before = await get_cache_stats(url)
    ttfts = await run_batch(client, model, branch_messages, concurrency)
    stats_after = await get_cache_stats(url)

    await client.close()
    return {
        "ttft_mean_ms": round(statistics.mean(ttfts), 1),
        "cache_hit_rate": stats_after["cache_hit_rate"],
        "n_requests": len(branch_messages),
    }


async def main(args: argparse.Namespace):
    print("=== SGLang v0.5.20: Unified Radix Tree & Branching-Point Caching ===\n")

    questions = UNIQUE_QUESTIONS[:args.n_questions]

    print("--- Experiment 1: Shared Prefix (standard prefix caching) ---")
    shared = await experiment_shared_prefix(args.url, args.model, questions, args.concurrency)
    print(f"  Cold batch TTFT (cache cold):  {shared['cold_ttft_mean_ms']:>8.1f} ms  (hit rate: {shared['cache_hit_rate_cold']:.1%})")
    print(f"  Warm batch TTFT (cache warm):  {shared['warm_ttft_mean_ms']:>8.1f} ms  (hit rate: {shared['cache_hit_rate_warm']:.1%})")
    print(f"  Speedup: {shared['speedup']}×")

    print("\n--- Experiment 2: Unique Prefixes (no cache benefit baseline) ---")
    unique = await experiment_unique_prefix(args.url, args.model, questions, args.concurrency)
    print(f"  TTFT mean: {unique['ttft_mean_ms']:>8.1f} ms  (hit rate: {unique['cache_hit_rate']:.1%})")

    print("\n--- Experiment 3: Branching-Point Caching (new in v0.5.20) ---")
    branching = await experiment_branching(args.url, args.model, questions, args.concurrency)
    print(f"  TTFT mean: {branching['ttft_mean_ms']:>8.1f} ms  (hit rate: {branching['cache_hit_rate']:.1%})")
    print(f"  Requests:  {branching['n_requests']} (3 branches × {len(questions[:4])} questions)")

    print("\n--- Summary ---")
    print(f"{'Workload':>30} {'TTFT mean':>12} {'Hit rate':>10}")
    print("-" * 56)
    print(f"{'Shared prefix (cold)':>30} {shared['cold_ttft_mean_ms']:>10.1f}ms {shared['cache_hit_rate_cold']:>9.1%}")
    print(f"{'Shared prefix (warm)':>30} {shared['warm_ttft_mean_ms']:>10.1f}ms {shared['cache_hit_rate_warm']:>9.1%}")
    print(f"{'Unique prefixes':>30} {unique['ttft_mean_ms']:>10.1f}ms {unique['cache_hit_rate']:>9.1%}")
    print(f"{'Branching (v0.5.20)':>30} {branching['ttft_mean_ms']:>10.1f}ms {branching['cache_hit_rate']:>9.1%}")

    print("\nExpected from v0.5.20 release notes:")
    print("  Token hit rate: 43.8% → 60.8% on shared-prefix workloads")
    print("  Mean TTFT:      1.57s → 1.07s on DeepSeek-V4-Flash")

    if args.output:
        out = {
            "sglang_version": "0.5.20",
            "feature": "branching_point_cache",
            "shared_prefix": shared,
            "unique_prefix": unique,
            "branching": branching,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang v0.5.20 radix cache benchmark")
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--n-questions", type=int, default=16)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
