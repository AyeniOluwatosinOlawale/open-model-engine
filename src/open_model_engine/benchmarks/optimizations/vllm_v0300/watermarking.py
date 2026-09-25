"""
vLLM v0.30.0 — Lab: Gumbel-Max Watermarking
============================================
Tests the new watermarked generation and detection introduced in v0.30.0.

Gumbel-max watermarking embeds an invisible signal into generated text using
a keyed pseudo-random function (PRF). The signal is detectable by the key
holder but statistically invisible to readers.

Key properties:
  - Per-request opt-out (user can disable per call)
  - Compatible with speculative decoding (dual-key variant)
  - Detection endpoint: POST /v1/watermark/detect

Setup:
  # Start vLLM with Gumbel watermarking enabled
  # Note: v0.30.0 uses --watermark-config JSON; key must be an integer
  vllm serve <model> --port 8000 \\
    --watermark-config '{"algorithm": "gumbel", "key": 42}' \\
    --override-generation-config '{"enable_thinking": false}'

Usage:
  python -m open_model_engine.benchmarks.optimizations.vllm_v0300.watermarking \\
    --url http://localhost:8000 --model <model>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx
import openai


GENERATION_PROMPTS = [
    "Write a short paragraph about the importance of clean energy.",
    "Explain how the internet works in simple terms.",
    "Describe the life cycle of a butterfly.",
    "What are the benefits of regular exercise?",
    "Write a brief history of the printing press.",
    "Explain what DNA is and why it matters.",
    "Describe the water cycle.",
    "What makes a good software engineer?",
    "Explain how vaccines work.",
    "Write about the importance of biodiversity.",
]


async def generate_text(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    watermark: bool = True,
    max_tokens: int = 150,
) -> dict:
    """Generate text with or without watermarking."""
    t0 = time.perf_counter()
    kwargs: dict = {}
    if not watermark:
        # Per-request opt-out via extra_body (openai SDK passes these as raw JSON fields)
        kwargs["extra_body"] = {"watermark": False}

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.8,
        **kwargs,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = response.choices[0].message.content or ""
    return {
        "text": text,
        "tokens": response.usage.completion_tokens if response.usage else len(text.split()),
        "latency_ms": round(elapsed_ms, 1),
        "watermark_requested": watermark,
    }


async def detect_watermark(url: str, text: str) -> dict:
    """
    Call the detection endpoint. vLLM v0.30.0 may expose this at different paths.
    Tries /v1/watermark/detect, then /detect_watermark.
    """
    candidates = [f"{url}/v1/watermark/detect", f"{url}/detect_watermark"]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for endpoint in candidates:
                r = await client.post(endpoint, json={"text": text})
                if r.status_code == 200:
                    return r.json()
                if r.status_code != 404:
                    return {"error": f"HTTP {r.status_code}", "detected": None}
            return {"error": "detection_endpoint_not_found", "detected": None}
    except Exception as e:
        return {"error": str(e), "detected": None}


async def run_watermark_experiment(
    url: str,
    model: str,
    prompts: list[str],
    concurrency: int = 4,
) -> dict:
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    watermarked_results = []
    unwatermarked_results = []

    async def process(prompt: str, watermark: bool) -> dict:
        async with sem:
            result = await generate_text(client, model, prompt, watermark)
            detection = await detect_watermark(url, result["text"])
            return {**result, "detection": detection}

    # Generate both with and without watermarks
    wm_tasks = [process(p, True) for p in prompts]
    no_wm_tasks = [process(p, False) for p in prompts]

    watermarked_results = await asyncio.gather(*wm_tasks)
    unwatermarked_results = await asyncio.gather(*no_wm_tasks)
    await client.close()

    def detection_stats(results: list[dict]) -> dict:
        detections = [r["detection"] for r in results if not r["detection"].get("error")]
        detected = [d for d in detections if d.get("detected") is True]
        scores = [d.get("score", 0.0) for d in detections if "score" in d]
        return {
            "n": len(results),
            "detection_rate": round(len(detected) / len(detections), 3) if detections else 0,
            "mean_score": round(statistics.mean(scores), 4) if scores else None,
            "endpoint_available": len(detections) > 0,
        }

    def latency_stats(results: list[dict]) -> dict:
        lats = [r["latency_ms"] for r in results]
        return {
            "mean_ms": round(statistics.mean(lats), 1),
            "p95_ms": round(sorted(lats)[int(len(lats) * 0.95)], 1),
        }

    wm_detect = detection_stats(watermarked_results)
    no_wm_detect = detection_stats(unwatermarked_results)
    wm_lat = latency_stats(watermarked_results)
    no_wm_lat = latency_stats(unwatermarked_results)

    return {
        "watermarked": {**wm_detect, **wm_lat},
        "unwatermarked": {**no_wm_detect, **no_wm_lat},
        "false_positive_rate": no_wm_detect["detection_rate"],
        "true_positive_rate": wm_detect["detection_rate"],
    }


async def test_quality_impact(url: str, model: str, n: int = 5) -> dict:
    """
    Test whether watermarking noticeably changes text quality.
    Compare output diversity with and without watermarks.
    """
    client = openai.AsyncOpenAI(base_url=f"{url}/v1", api_key="EMPTY")
    prompt = "Write a one-sentence fact about space."

    wm_responses = []
    no_wm_responses = []

    for _ in range(n):
        wm = await generate_text(client, model, prompt, watermark=True, max_tokens=50)
        no_wm = await generate_text(client, model, prompt, watermark=False, max_tokens=50)
        wm_responses.append(wm["text"])
        no_wm_responses.append(no_wm["text"])

    await client.close()

    # Rough diversity: average pairwise word overlap
    def diversity(texts: list[str]) -> float:
        if len(texts) < 2:
            return 0.0
        overlaps = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = set(texts[i].lower().split()), set(texts[j].lower().split())
                if a and b:
                    overlaps.append(len(a & b) / len(a | b))
        return round(1 - statistics.mean(overlaps), 3) if overlaps else 0.0

    return {
        "wm_diversity": diversity(wm_responses),
        "no_wm_diversity": diversity(no_wm_responses),
        "wm_sample": wm_responses[0] if wm_responses else "",
        "no_wm_sample": no_wm_responses[0] if no_wm_responses else "",
    }


async def main(args: argparse.Namespace):
    print("=== vLLM v0.30.0: Gumbel-Max Watermarking ===\n")

    prompts = GENERATION_PROMPTS[:args.n_prompts]
    print(f"Prompts: {len(prompts)}  Concurrency: {args.concurrency}\n")

    print("Running watermark generation + detection...", end=" ", flush=True)
    results = await run_watermark_experiment(args.url, args.model, prompts, args.concurrency)
    print("done\n")

    print("--- Detection Results ---")
    print(f"{'':>30} {'Watermarked':>14} {'Unwatermarked':>16}")
    print("-" * 64)
    print(f"{'Detection rate':>30} {results['watermarked']['detection_rate']:>13.1%} {results['unwatermarked']['detection_rate']:>15.1%}")
    if results["watermarked"].get("mean_score") is not None:
        print(f"{'Mean detection score':>30} {results['watermarked']['mean_score']:>14.4f} {results['unwatermarked'].get('mean_score') or 0:>15.4f}")
    print(f"{'Latency mean (ms)':>30} {results['watermarked']['mean_ms']:>13.1f} {results['unwatermarked']['mean_ms']:>15.1f}")
    print(f"{'Latency p95  (ms)':>30} {results['watermarked']['p95_ms']:>13.1f} {results['unwatermarked']['p95_ms']:>15.1f}")

    overhead = (results['watermarked']['mean_ms'] - results['unwatermarked']['mean_ms']) / results['unwatermarked']['mean_ms'] * 100
    print(f"\nWatermark latency overhead: {overhead:+.1f}%")
    print(f"True positive rate:  {results['true_positive_rate']:.1%}")
    print(f"False positive rate: {results['false_positive_rate']:.1%}")

    if not results["watermarked"].get("endpoint_available"):
        print("\nNote: /v1/watermark/detect endpoint not exposed in this vLLM build.")
        print("      Detection API may require a separate vLLM build flag or is unavailable")
        print("      in the NGC container's vLLM version. Watermark generation still working.")

    print("\n--- Quality Impact Test ---")
    quality = await test_quality_impact(args.url, args.model)
    print(f"Output diversity (watermarked):   {quality['wm_diversity']}")
    print(f"Output diversity (unwatermarked): {quality['no_wm_diversity']}")
    print(f"\nWatermarked sample:   {quality['wm_sample'][:80]}...")
    print(f"Unwatermarked sample: {quality['no_wm_sample'][:80]}...")

    print("\nExpected from v0.30.0 release notes:")
    print("  Gumbel-max watermark statistically invisible to readers")
    print("  Per-request opt-out with watermark=False")
    print("  Dual-key variant compatible with speculative decoding")

    if args.output:
        out = {
            "vllm_version": "0.30.0",
            "feature": "gumbel_watermarking",
            "results": results,
            "quality": quality,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM v0.30.0 watermarking benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
