from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .profiling.gpu_profiler import snapshot_gpus, summarize_snapshots
from .profiling.nsight_profiler import NsightProfiler
from .servers import ServerBackend, get_server
from .types import AggregatedMetrics, OpenModelRunResult, ScenarioDef, aggregate


@dataclass
class ThroughputPoint:
    batch_size: int
    input_len: int
    output_len: int
    throughput_tps: float
    ttft_p50_ms: float
    decode_throughput_tps: float
    gpu_utilization_pct: float


class OpenModelEngine:
    """
    Main library entry point for open-model-engine.

    Usage:
        engine = OpenModelEngine("vllm", "meta-llama/Llama-3.1-8B",
                                 base_url="http://localhost:8000")
        results = await engine.bench(ScenarioDef.medium(), runs=10)
        engine.print_summary(results)
    """

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        base_url: str = "",
        gpu_cost_per_hour: float = 0.0,
        profile: str = "system",       # "none" | "system" | "cuda"
        nsight_output_dir: str = "./profiles",
    ) -> None:
        self.backend_name = backend
        self.model = model
        self.profile = profile
        self._server: ServerBackend = get_server(backend, model, base_url=base_url,
                                                  gpu_cost_per_hour=gpu_cost_per_hour)
        self._nsight = NsightProfiler(nsight_output_dir) if profile in ("cuda", "both") else None

    async def bench(
        self,
        scenario: ScenarioDef,
        runs: int = 5,
        warm_up: bool = True,
        collect_gpu: bool = True,
    ) -> list[OpenModelRunResult]:
        if warm_up:
            await self._server.warm_up(scenario)
        results: list[OpenModelRunResult] = []
        for i in range(runs):
            r = await self._server.measure_single(
                scenario, run_index=i, collect_gpu_metrics=collect_gpu
            )
            results.append(r)
        return results

    async def bench_all(
        self,
        scenarios: list[ScenarioDef] | None = None,
        runs: int = 5,
        warm_up: bool = True,
    ) -> tuple[list[OpenModelRunResult], list[AggregatedMetrics]]:
        s = scenarios or ScenarioDef.all()
        all_results: list[OpenModelRunResult] = []
        aggregated: list[AggregatedMetrics] = []
        for scenario in s:
            results = await self.bench(scenario, runs=runs, warm_up=warm_up)
            all_results.extend(results)
            try:
                aggregated.append(aggregate(results))
            except ValueError:
                pass
        return all_results, aggregated

    async def sweep_concurrency(
        self,
        scenario: ScenarioDef,
        levels: list[int] | None = None,
        runs_per_level: int = 5,
    ) -> list[dict]:
        lvls = levels or [1, 2, 4, 8, 16, 32]
        points = []
        for c in lvls:
            tasks = [
                self._server.measure_single(scenario, run_index=i)
                for i in range(c * runs_per_level)
            ]
            all_r: list[OpenModelRunResult] = []
            for batch_start in range(0, len(tasks), c):
                batch = tasks[batch_start:batch_start + c]
                batch_r = await asyncio.gather(*batch, return_exceptions=True)
                for r in batch_r:
                    if isinstance(r, OpenModelRunResult) and r.succeeded:
                        all_r.append(r)
            if all_r:
                try:
                    agg = aggregate(all_r)
                    points.append({
                        "concurrency": c,
                        "ttft_p50_ms": agg.ttft_p50_ms,
                        "ttft_p95_ms": agg.ttft_p95_ms,
                        "throughput_mean_tps": agg.throughput_mean_tps,
                        "prefill_throughput": agg.prefill_throughput_mean_tps,
                        "decode_throughput": agg.decode_throughput_mean_tps,
                        "bottleneck": agg.dominant_bottleneck,
                        "gpu_util": agg.gpu_utilization_mean_pct,
                        "kv_hit_rate": agg.kv_cache_hit_rate_mean,
                    })
                except ValueError:
                    pass
        return points

    async def gpu_snapshot(self) -> dict:
        snaps = await snapshot_gpus()
        if not snaps:
            return {}
        mem, util, power, sm, mem_clk, temp = summarize_snapshots(snaps)
        return {
            "gpu_count": len(snaps),
            "memory_used_mb": mem,
            "utilization_pct": util,
            "power_watts": power,
            "sm_clock_mhz": sm,
            "temperature_c": temp,
        }

    def aggregate(self, results: list[OpenModelRunResult]) -> AggregatedMetrics:
        return aggregate(results)

    async def health_check(self) -> bool:
        return await self._server.health_check()

    async def close(self) -> None:
        await self._server.close()

    async def __aenter__(self) -> "OpenModelEngine":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
