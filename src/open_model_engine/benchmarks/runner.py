from __future__ import annotations

import asyncio

from ..servers.base import ServerBackend
from ..types import AggregatedMetrics, OpenModelRunResult, ScenarioDef, aggregate


async def run_scenario(
    server: ServerBackend,
    scenario: ScenarioDef,
    runs: int = 5,
    warm_up: bool = True,
    collect_gpu: bool = True,
) -> list[OpenModelRunResult]:
    if warm_up:
        await server.warm_up(scenario)
    results: list[OpenModelRunResult] = []
    for i in range(runs):
        r = await server.measure_single(scenario, run_index=i, collect_gpu_metrics=collect_gpu)
        results.append(r)
    return results


async def run_all_scenarios(
    server: ServerBackend,
    scenarios: list[ScenarioDef],
    runs: int = 5,
    warm_up: bool = True,
) -> tuple[list[OpenModelRunResult], list[AggregatedMetrics]]:
    all_results: list[OpenModelRunResult] = []
    aggregated: list[AggregatedMetrics] = []
    for scenario in scenarios:
        results = await run_scenario(server, scenario, runs=runs, warm_up=warm_up)
        all_results.extend(results)
        try:
            aggregated.append(aggregate(results))
        except ValueError:
            pass
    return all_results, aggregated
