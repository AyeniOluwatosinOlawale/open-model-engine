from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

import typer

from .engine import OpenModelEngine
from .report import (
    export_csv,
    export_json,
    print_baseline_table,
    print_bottleneck_summary,
    print_concurrency_table,
    print_gpu_table,
)
from .servers import get_server
from .types import ScenarioDef, aggregate

app = typer.Typer(name="open-model-engine", help="GPU-first inference harness for vLLM, SGLang, TensorRT-LLM.")


@app.command("bench")
def cmd_bench(
    backend: str = typer.Option(..., "--backend", "-b", help="vllm | sglang | tensorrt"),
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server base URL"),
    scenario: Optional[str] = typer.Option(None, "--scenario", "-s", help="short | medium | long | complex"),
    runs: int = typer.Option(5, "--runs", "-r"),
    no_warmup: bool = typer.Option(False, "--no-warmup"),
    no_gpu: bool = typer.Option(False, "--no-gpu-metrics"),
    profile: str = typer.Option("system", "--profile", help="none | system | cuda"),
    nsight_output: str = typer.Option("./profiles", "--nsys-output"),
    gpu_cost: float = typer.Option(0.0, "--gpu-cost-per-hour"),
    output_json: Optional[str] = typer.Option(None, "--output", "-o"),
    output_csv: Optional[str] = typer.Option(None, "--csv"),
    concurrency: bool = typer.Option(False, "--concurrency", help="Run concurrency sweep after baseline"),
    conc_levels: str = typer.Option("1,2,4,8,16,32", "--conc-levels"),
) -> None:
    """Benchmark an open model server with full latency, throughput, and GPU profiling."""
    scenarios = [getattr(ScenarioDef, scenario)()] if scenario else ScenarioDef.all()
    levels = [int(x) for x in conc_levels.split(",")]

    async def _main() -> None:
        engine = OpenModelEngine(
            backend, model, base_url=url,
            gpu_cost_per_hour=gpu_cost,
            profile=profile,
            nsight_output_dir=nsight_output,
        )

        healthy = await engine.health_check()
        if not healthy:
            typer.echo(f"  WARNING: {backend} server at {url} did not respond to health check.", err=True)

        gpu_info = await engine.gpu_snapshot()
        if gpu_info:
            typer.echo(f"\n  GPU: {gpu_info.get('gpu_count', '?')} device(s)"
                       f"  Mem: {gpu_info.get('memory_used_mb', 0):.0f} MB used"
                       f"  Util: {gpu_info.get('utilization_pct', 0):.1f}%"
                       f"  Temp: {gpu_info.get('temperature_c', 0):.0f}°C")

        typer.echo(f"\n  Backend : {backend} / {model} @ {url}")
        typer.echo(f"  Scenarios: {[s.name for s in scenarios]}  runs={runs}  profile={profile}")

        all_results, aggregated = await engine.bench_all(
            scenarios, runs=runs, warm_up=not no_warmup
        )

        print_baseline_table(aggregated)
        print_bottleneck_summary(aggregated)

        has_gpu_data = any(r.gpu_memory_used_mb > 0 for r in all_results)
        if has_gpu_data and not no_gpu:
            print_gpu_table(aggregated)

        if concurrency:
            typer.echo("\n  Running concurrency sweep…")
            points = await engine.sweep_concurrency(ScenarioDef.medium(), levels)
            print_concurrency_table(points)

        if output_json:
            export_json(output_json, all_results, aggregated,
                        {"backend": backend, "model": model, "url": url, "profile": profile})
        if output_csv:
            export_csv(output_csv, aggregated)

        await engine.close()

    asyncio.run(_main())


@app.command("optimize")
def cmd_optimize(
    opt: str = typer.Argument(..., help="quantization | speculative | prefix-cache | multi-gpu | continuous-batch"),
    backend: str = typer.Option(..., "--backend", "-b"),
    model: str = typer.Option(..., "--model", "-m"),
    url: str = typer.Option("http://localhost:8000", "--url"),
    runs: int = typer.Option(5, "--runs"),
    # quantization
    formats: str = typer.Option("fp16,int8,int4", "--formats", help="Comma-separated quant formats"),
    # speculative
    draft_model: Optional[str] = typer.Option(None, "--draft"),
    # multi-gpu
    tp_degrees: str = typer.Option("1,2,4", "--tp", help="Tensor parallel degrees"),
) -> None:
    """Run optimization benchmarks (quantization, speculative decoding, multi-GPU scaling, etc.)."""
    typer.echo(f"\n  Optimization: {opt}")
    typer.echo(f"  Backend: {backend} / {model}")

    if opt == "quantization":
        from .benchmarks.optimizations.quantization import run_quantization_benchmark
        fmt_list = [f.strip() for f in formats.split(",")]
        asyncio.run(run_quantization_benchmark(backend, model, url, fmt_list, runs))

    elif opt == "speculative":
        from .benchmarks.optimizations.speculative import run_speculative_benchmark
        if not draft_model:
            typer.echo("--draft is required for speculative decoding benchmark", err=True)
            raise typer.Exit(1)
        asyncio.run(run_speculative_benchmark(backend, model, draft_model, url, runs))

    elif opt == "prefix-cache":
        from .benchmarks.optimizations.prefix_cache import run_prefix_cache_benchmark
        asyncio.run(run_prefix_cache_benchmark(backend, model, url, runs))

    elif opt == "multi-gpu":
        from .benchmarks.optimizations.multi_gpu import run_multi_gpu_benchmark
        tp_list = [int(t) for t in tp_degrees.split(",")]
        asyncio.run(run_multi_gpu_benchmark(backend, model, tp_list))

    elif opt == "continuous-batch":
        from .benchmarks.optimizations.continuous_batch import run_continuous_batch_benchmark
        asyncio.run(run_continuous_batch_benchmark(backend, model, url, runs))

    else:
        typer.echo(f"Unknown optimization: {opt}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
