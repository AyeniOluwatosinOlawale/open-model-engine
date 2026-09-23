"""
SGLang v0.5.20 — Lab: Sampling Masks for RL Rollouts
======================================================
Tests the new return_sampling_mask parameter introduced in v0.5.20.

Each decode step returns the exact token support the sampler drew from
and the log-probability of the sampled token, so an RL trainer can replay
the rollout without reconstructing top-k or top-p.

Reported improvement: +17% decode throughput at batch 1, +52% at batch 64
on Qwen3-8B (overlap scheduling).

Setup:
  python -m sglang.launch_server --model <model> --port 30000 \\
    --sampling-mask-max-tokens 4096

Usage:
  python -m open_model_engine.benchmarks.optimizations.sglang_v0520.sampling_masks \\
    --url http://localhost:30000 --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RolloutStep:
    token_id: int
    log_prob: float
    mask_size: int        # number of tokens in the sampling support
    mask_present: bool


@dataclass
class RolloutResult:
    prompt: str
    steps: list[RolloutStep] = field(default_factory=list)
    total_ms: float = 0.0
    decode_throughput_tps: float = 0.0


ROLLOUT_PROMPTS = [
    "Solve step by step: what is 15 * 23?",
    "Write a Python function to reverse a string.",
    "Explain gradient descent in 3 sentences.",
    "What are the three laws of thermodynamics?",
    "Describe the attention mechanism in transformers.",
]


async def run_rollout_with_mask(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> RolloutResult:
    """
    Call SGLang's native /generate endpoint with return_sampling_mask=true.
    The response includes per-step mask_ids and log_probs for RL replay.
    """
    result = RolloutResult(prompt=prompt)
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_logprob": True,
                "return_sampling_mask": True,    # new in v0.5.20
                "logprob_start_len": 0,
            },
        }

        r = await client.post(f"{url}/generate", json=payload)
        r.raise_for_status()
        data = r.json()

    elapsed = (time.perf_counter() - t0) * 1000
    result.total_ms = round(elapsed, 1)

    # Parse output_token_logprobs and sampling_masks
    token_logprobs = data.get("meta_info", {}).get("output_token_logprobs", [])
    sampling_masks = data.get("meta_info", {}).get("sampling_masks", [])

    for i, (tok_id, log_p, _) in enumerate(token_logprobs):
        mask = sampling_masks[i] if i < len(sampling_masks) else None
        result.steps.append(RolloutStep(
            token_id=tok_id,
            log_prob=log_p,
            mask_size=len(mask) if mask else 0,
            mask_present=mask is not None,
        ))

    n_tokens = len(result.steps)
    result.decode_throughput_tps = round(n_tokens / (elapsed / 1000), 1) if elapsed > 0 else 0.0
    return result


async def benchmark_mask_overhead(
    url: str,
    model: str,
    n_runs: int = 10,
) -> dict:
    """Compare throughput with and without sampling masks."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async def timed_generate(return_mask: bool) -> float:
            t0 = time.perf_counter()
            payload = {
                "text": "Write a detailed explanation of how neural networks learn.",
                "sampling_params": {
                    "max_new_tokens": 300,
                    "temperature": 0.8,
                    "return_logprob": True,
                    "return_sampling_mask": return_mask,
                },
            }
            r = await client.post(f"{url}/generate", json=payload)
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000

        # Warmup
        for _ in range(2):
            await timed_generate(False)
            await timed_generate(True)

        no_mask_times = [await timed_generate(False) for _ in range(n_runs)]
        mask_times = [await timed_generate(True) for _ in range(n_runs)]

    def stats(times: list[float]) -> dict:
        import statistics
        return {
            "mean_ms": round(statistics.mean(times), 1),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 1),
        }

    return {
        "without_mask": stats(no_mask_times),
        "with_mask": stats(mask_times),
    }


async def verify_replay(result: RolloutResult) -> dict:
    """
    Verify that stored log_probs and masks are sufficient for RL replay.
    A trainer can recompute policy gradient loss from (token_id, log_prob, mask).
    """
    if not result.steps:
        return {"replayable": False, "reason": "no steps"}

    steps_with_mask = sum(1 for s in result.steps if s.mask_present)
    steps_with_logprob = sum(1 for s in result.steps if s.log_prob != 0.0)
    avg_mask_size = (
        sum(s.mask_size for s in result.steps if s.mask_present) / max(steps_with_mask, 1)
    )

    return {
        "total_steps": len(result.steps),
        "steps_with_mask": steps_with_mask,
        "steps_with_logprob": steps_with_logprob,
        "mask_coverage_pct": round(steps_with_mask / len(result.steps) * 100, 1),
        "avg_mask_size": round(avg_mask_size, 0),
        "replayable": steps_with_mask == len(result.steps) and steps_with_logprob == len(result.steps),
    }


async def main(args: argparse.Namespace):
    print("=== SGLang v0.5.20: Sampling Masks for RL Rollouts ===\n")

    # Part 1: rollout capture
    print("--- Part 1: Rollout Capture ---")
    for prompt in ROLLOUT_PROMPTS[:3]:
        result = await run_rollout_with_mask(args.url, args.model, prompt)
        replay = await verify_replay(result)
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"\nPrompt: {short_prompt!r}")
        print(f"  Tokens generated:    {replay['total_steps']}")
        print(f"  Mask coverage:       {replay['mask_coverage_pct']}%")
        print(f"  Avg mask size:       {replay['avg_mask_size']:.0f} tokens in support")
        print(f"  Decode throughput:   {result.decode_throughput_tps} tok/s")
        print(f"  Replayable for RL:   {'✓ YES' if replay['replayable'] else '✗ NO (masks missing)'}")

    # Part 2: overhead measurement
    print("\n--- Part 2: Mask Overhead Measurement ---")
    print("Running throughput comparison (with vs without masks)...", end=" ", flush=True)
    overhead = await benchmark_mask_overhead(args.url, args.model, n_runs=args.runs)
    print("done\n")

    print(f"{'':>25} {'Mean':>12} {'p95':>12}")
    print("-" * 52)
    for label, stats in overhead.items():
        print(f"{label:>25} {stats['mean_ms']:>10.1f}ms {stats['p95_ms']:>10.1f}ms")

    overhead_pct = (
        (overhead["with_mask"]["mean_ms"] - overhead["without_mask"]["mean_ms"])
        / overhead["without_mask"]["mean_ms"] * 100
    )
    print(f"\nMask overhead: {overhead_pct:+.1f}% latency")
    print("(Expected from v0.5.20 release notes: overhead negligible under overlap scheduling)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"overhead": overhead}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang v0.5.20 sampling masks benchmark")
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", help="Save JSON results to file")
    asyncio.run(main(parser.parse_args()))
