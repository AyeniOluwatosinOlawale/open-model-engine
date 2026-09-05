from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

# Re-export shared types from the same definitions
# (open-model-engine is standalone — types are duplicated intentionally)
# Compatible with inference-harness RunResult schema for cross-repo comparison.


@dataclass
class ScenarioDef:
    name: str
    system: str
    user: str
    expected_output_tokens: int = 256

    @classmethod
    def short(cls) -> "ScenarioDef":
        return cls(name="short", system=_SYSTEM, user="What is the capital of France? One sentence.", expected_output_tokens=30)

    @classmethod
    def medium(cls) -> "ScenarioDef":
        return cls(name="medium", system=_SYSTEM, user="Compare REST and GraphQL APIs covering data fetching, over/under-fetching, versioning, and when to choose each.", expected_output_tokens=300)

    @classmethod
    def long(cls) -> "ScenarioDef":
        return cls(name="long", system=_SYSTEM, user="Explain transformer self-attention: QKV projections, scaled dot-product, multi-head attention, positional encodings, and why attention replaced RNNs.", expected_output_tokens=700)

    @classmethod
    def complex(cls) -> "ScenarioDef":
        return cls(name="complex", system=_SYSTEM, user="Design a distributed rate-limiting system for an API gateway: 10M req/day, 20 regions, P99 < 5ms, sliding-window + token-bucket, partition tolerance. Include architecture, tech choices, failure modes, capacity estimate.", expected_output_tokens=800)

    @classmethod
    def all(cls) -> list["ScenarioDef"]:
        return [cls.short(), cls.medium(), cls.long(), cls.complex()]


_SYSTEM = (
    "You are an expert software engineer and systems architect. "
    "Answers are technically precise, well-structured, and concise. "
    "Cite trade-offs, give concrete examples, avoid filler. "
    "Satisfy every constraint stated in the question."
)


@dataclass
class RunResult:
    """Base result — compatible with inference-harness RunResult schema."""
    scenario: str
    backend: str            # "vllm" | "sglang" | "tensorrt"
    model: str
    run_index: int

    ttft_ms: float
    total_ms: float
    tpot_ms: float
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    queue_time_ms: float = 0.0

    throughput_tps: float = 0.0
    prefill_throughput_tps: float = 0.0
    decode_throughput_tps: float = 0.0

    bottleneck: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    gpu_power_watts: float = 0.0
    kv_cache_hit_rate: float = 0.0
    kv_cache_utilization: float = 0.0

    response_text: str = ""
    error: str | None = None
    request_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def __post_init__(self) -> None:
        if not self.bottleneck:
            self.bottleneck = classify_bottleneck(
                self.prefill_ms, self.decode_ms,
                self.ttft_ms, self.tpot_ms, self.output_tokens,
            )


@dataclass
class OpenModelRunResult(RunResult):
    """Extended result with deep GPU + serving engine metrics."""

    # Serving engine state at request time
    batch_size: int = 0
    num_running_requests: int = 0
    num_queued_requests: int = 0

    # GPU hardware (from pynvml, sampled during request)
    gpu_sm_clock_mhz: float = 0.0
    gpu_memory_clock_mhz: float = 0.0
    gpu_temperature_c: float = 0.0

    # KV cache detail
    kv_cache_num_blocks_used: int = 0
    kv_cache_num_blocks_total: int = 0

    # Nsight CUDA profiling (populated only with --profile cuda)
    nsight_report_path: str | None = None
    cuda_prefill_kernel_ms: float = 0.0    # kernel time in prefill NVTX range
    cuda_decode_kernel_ms: float = 0.0     # kernel time in decode NVTX range
    hbm_read_bandwidth_gbps: float = 0.0   # HBM read BW (decode bottleneck signal)
    hbm_write_bandwidth_gbps: float = 0.0
    memory_transfer_ms: float = 0.0        # PCIe / NVLink transfer time


def classify_bottleneck(
    prefill_ms: float,
    decode_ms: float,
    ttft_ms: float,
    tpot_ms: float,
    output_tokens: int,
) -> str:
    if prefill_ms > 0 and decode_ms > 0:
        ratio = prefill_ms / (prefill_ms + decode_ms)
        if ratio > 0.6:
            return "prefill"
        if ratio < 0.4:
            return "decode"
        return "balanced"
    if ttft_ms > 0 and tpot_ms > 0 and output_tokens > 0:
        decode_total = tpot_ms * max(output_tokens - 1, 1)
        ratio = ttft_ms / (ttft_ms + decode_total)
        if ratio > 0.6:
            return "prefill"
        if ratio < 0.4:
            return "decode"
        return "balanced"
    return "unknown"


@dataclass
class AggregatedMetrics:
    scenario: str
    backend: str
    model: str
    n: int

    ttft_p50_ms: float
    ttft_p95_ms: float
    prefill_p50_ms: float
    prefill_p95_ms: float
    decode_p50_ms: float
    decode_p95_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float
    queue_p50_ms: float

    throughput_mean_tps: float
    throughput_p5_tps: float
    prefill_throughput_mean_tps: float
    decode_throughput_mean_tps: float

    input_tokens_mean: float
    output_tokens_mean: float
    cost_mean_usd: float
    cost_total_usd: float

    dominant_bottleneck: str
    prefill_pct: float
    decode_pct: float

    # GPU averages (open model only)
    gpu_memory_mean_mb: float = 0.0
    gpu_utilization_mean_pct: float = 0.0
    kv_cache_hit_rate_mean: float = 0.0
    cuda_prefill_kernel_mean_ms: float = 0.0
    cuda_decode_kernel_mean_ms: float = 0.0
    hbm_read_bw_mean_gbps: float = 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def aggregate(results: list[RunResult]) -> AggregatedMetrics:
    good = [r for r in results if r.succeeded]
    if not good:
        raise ValueError("No successful runs to aggregate")

    def _mean(seq):
        lst = list(seq)
        return statistics.mean(lst) if lst else 0.0

    bottlenecks = [r.bottleneck for r in good]
    dominant = max(["prefill", "decode", "balanced", "unknown"], key=lambda b: bottlenecks.count(b))

    open_results = [r for r in good if isinstance(r, OpenModelRunResult)]

    return AggregatedMetrics(
        scenario=good[0].scenario,
        backend=good[0].backend,
        model=good[0].model,
        n=len(good),
        ttft_p50_ms=_pct([r.ttft_ms for r in good], 50),
        ttft_p95_ms=_pct([r.ttft_ms for r in good], 95),
        prefill_p50_ms=_pct([r.prefill_ms for r in good], 50),
        prefill_p95_ms=_pct([r.prefill_ms for r in good], 95),
        decode_p50_ms=_pct([r.decode_ms for r in good], 50),
        decode_p95_ms=_pct([r.decode_ms for r in good], 95),
        tpot_p50_ms=_pct([r.tpot_ms for r in good], 50),
        tpot_p95_ms=_pct([r.tpot_ms for r in good], 95),
        total_p50_ms=_pct([r.total_ms for r in good], 50),
        total_p95_ms=_pct([r.total_ms for r in good], 95),
        queue_p50_ms=_pct([r.queue_time_ms for r in good], 50),
        throughput_mean_tps=_mean(r.throughput_tps for r in good),
        throughput_p5_tps=_pct([r.throughput_tps for r in good], 5),
        prefill_throughput_mean_tps=_mean(r.prefill_throughput_tps for r in good),
        decode_throughput_mean_tps=_mean(r.decode_throughput_tps for r in good),
        input_tokens_mean=_mean(r.input_tokens for r in good),
        output_tokens_mean=_mean(r.output_tokens for r in good),
        cost_mean_usd=_mean(r.cost_usd for r in good),
        cost_total_usd=sum(r.cost_usd for r in good),
        dominant_bottleneck=dominant,
        prefill_pct=bottlenecks.count("prefill") / len(good),
        decode_pct=bottlenecks.count("decode") / len(good),
        gpu_memory_mean_mb=_mean(r.gpu_memory_used_mb for r in good),
        gpu_utilization_mean_pct=_mean(r.gpu_utilization_pct for r in good),
        kv_cache_hit_rate_mean=_mean(r.kv_cache_hit_rate for r in good),
        cuda_prefill_kernel_mean_ms=_mean(r.cuda_prefill_kernel_ms for r in open_results),
        cuda_decode_kernel_mean_ms=_mean(r.cuda_decode_kernel_ms for r in open_results),
        hbm_read_bw_mean_gbps=_mean(r.hbm_read_bandwidth_gbps for r in open_results),
    )


@dataclass
class GPUSnapshot:
    gpu_index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_pct: float
    power_watts: float
    sm_clock_mhz: float
    memory_clock_mhz: float
    temperature_c: float
    timestamp_ms: float


@dataclass
class ServerMetrics:
    kv_cache_hit_rate: float = 0.0
    kv_cache_utilization: float = 0.0
    num_running_requests: int = 0
    num_queued_requests: int = 0
    prefill_throughput_tps: float = 0.0
    decode_throughput_tps: float = 0.0
    avg_batch_size: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
