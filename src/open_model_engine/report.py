from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime

from .types import AggregatedMetrics, OpenModelRunResult

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


def _ms(v: float) -> str:
    return f"{v:>8.1f}ms"

def _tps(v: float) -> str:
    return f"{v:>7.1f}"

def _bottleneck(b: str) -> str:
    return {"prefill": "⚠ PREFILL", "decode": "⚠ DECODE", "balanced": "✓ BALANCED"}.get(b, "  —")


def print_baseline_table(aggregated: list[AggregatedMetrics]) -> None:
    has_prefill = any(m.prefill_p50_ms > 0 for m in aggregated)

    if _RICH:
        t = Table(title="Benchmark: Latency, Throughput & Bottleneck", box=box.SIMPLE_HEAVY)
        t.add_column("Scenario", style="bold")
        t.add_column("Backend")
        t.add_column("N", justify="right")
        t.add_column("TTFT p50", justify="right")
        t.add_column("TTFT p95", justify="right")
        if has_prefill:
            t.add_column("Prefill p50", justify="right")
            t.add_column("Decode p50", justify="right")
        t.add_column("TPOT p50", justify="right")
        t.add_column("TPS (mean)", justify="right")
        t.add_column("Prefill TPS", justify="right")
        t.add_column("Decode TPS", justify="right")
        t.add_column("Bottleneck")
        for m in aggregated:
            row = [m.scenario, m.backend, str(m.n),
                   _ms(m.ttft_p50_ms), _ms(m.ttft_p95_ms)]
            if has_prefill:
                row += [_ms(m.prefill_p50_ms), _ms(m.decode_p50_ms)]
            row += [_ms(m.tpot_p50_ms), _tps(m.throughput_mean_tps),
                    _tps(m.prefill_throughput_mean_tps), _tps(m.decode_throughput_mean_tps),
                    _bottleneck(m.dominant_bottleneck)]
            t.add_row(*row)
        _console.print(t)
    else:
        print("\nBENCHMARK RESULTS")
        for m in aggregated:
            print(f"  {m.scenario:<10} TTFT p50={_ms(m.ttft_p50_ms)} p95={_ms(m.ttft_p95_ms)}"
                  f"  TPS={_tps(m.throughput_mean_tps)}  {_bottleneck(m.dominant_bottleneck)}")


def print_gpu_table(aggregated: list[AggregatedMetrics]) -> None:
    if _RICH:
        t = Table(title="GPU Metrics", box=box.SIMPLE_HEAVY)
        for col in ["Scenario", "GPU Mem (MB)", "GPU Util %", "KV Hit Rate %",
                    "CUDA Prefill ms", "CUDA Decode ms", "HBM BW (GB/s)"]:
            t.add_column(col, justify="right" if col != "Scenario" else "left")
        for m in aggregated:
            t.add_row(
                m.scenario,
                f"{m.gpu_memory_mean_mb:.0f}",
                f"{m.gpu_utilization_mean_pct:.1f}",
                f"{m.kv_cache_hit_rate_mean * 100:.1f}",
                f"{m.cuda_prefill_kernel_mean_ms:.1f}",
                f"{m.cuda_decode_kernel_mean_ms:.1f}",
                f"{m.hbm_read_bw_mean_gbps:.1f}",
            )
        _console.print(t)
    else:
        print("\nGPU METRICS")
        for m in aggregated:
            print(f"  {m.scenario:<10} GPU={m.gpu_memory_mean_mb:.0f}MB"
                  f"  Util={m.gpu_utilization_mean_pct:.1f}%"
                  f"  KV-hit={m.kv_cache_hit_rate_mean*100:.1f}%"
                  f"  CUDA-prefill={m.cuda_prefill_kernel_mean_ms:.1f}ms"
                  f"  CUDA-decode={m.cuda_decode_kernel_mean_ms:.1f}ms")


def print_bottleneck_summary(aggregated: list[AggregatedMetrics]) -> None:
    print("\n" + "=" * 80)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 80)
    for m in aggregated:
        pf = _ms(m.prefill_p50_ms) if m.prefill_p50_ms > 0 else f"{_ms(m.ttft_p50_ms)} (TTFT proxy)"
        dc = _ms(m.decode_p50_ms) if m.decode_p50_ms > 0 else f"TPOT×tok proxy"
        print(f"  {m.scenario:<10}  Prefill: {pf}  Decode: {dc}  → {_bottleneck(m.dominant_bottleneck)}")

    decode_bound = [m.scenario for m in aggregated if m.dominant_bottleneck == "decode"]
    prefill_bound = [m.scenario for m in aggregated if m.dominant_bottleneck == "prefill"]
    print()
    if decode_bound:
        print(f"  DECODE-bound ({', '.join(decode_bound)}):")
        print("    → Speculative decoding  |  INT4/FP8 quantization  |  Increase batch size")
        print("    → Enable PagedAttention prefix caching  |  Check HBM bandwidth saturation")
    if prefill_bound:
        print(f"  PREFILL-bound ({', '.join(prefill_bound)}):")
        print("    → Chunked prefill  |  Increase tensor parallelism  |  Prefix caching")
    print("=" * 80)


def print_concurrency_table(points: list[dict]) -> None:
    if _RICH:
        t = Table(title="Concurrency Sweep", box=box.SIMPLE_HEAVY)
        for col in ["Concurrency", "TTFT p50", "TTFT p95", "TPS", "Prefill TPS",
                    "Decode TPS", "GPU Util %", "KV Hit%", "Bottleneck"]:
            t.add_column(col, justify="right" if col != "Bottleneck" else "center")
        for p in points:
            t.add_row(
                str(p["concurrency"]),
                _ms(p.get("ttft_p50_ms", 0)),
                _ms(p.get("ttft_p95_ms", 0)),
                _tps(p.get("throughput_mean_tps", 0)),
                _tps(p.get("prefill_throughput", 0)),
                _tps(p.get("decode_throughput", 0)),
                f"{p.get('gpu_util', 0):.1f}",
                f"{p.get('kv_hit_rate', 0)*100:.1f}",
                _bottleneck(p.get("bottleneck", "unknown")),
            )
        _console.print(t)
    else:
        print("\nCONCURRENCY SWEEP")
        for p in points:
            print(f"  c={p['concurrency']:>2}  TTFT p50={_ms(p.get('ttft_p50_ms',0))}"
                  f"  TPS={_tps(p.get('throughput_mean_tps',0))}"
                  f"  {_bottleneck(p.get('bottleneck','unknown'))}")


def export_json(path: str, results: list[OpenModelRunResult],
                aggregated: list[AggregatedMetrics], metadata: dict | None = None) -> None:
    payload = {
        "metadata": {"timestamp": datetime.utcnow().isoformat() + "Z", **(metadata or {})},
        "aggregated": [asdict(m) for m in aggregated],
        "runs": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Results saved → {path}")


def export_csv(path: str, aggregated: list[AggregatedMetrics]) -> None:
    if not aggregated:
        return
    rows = [asdict(m) for m in aggregated]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {path}")
