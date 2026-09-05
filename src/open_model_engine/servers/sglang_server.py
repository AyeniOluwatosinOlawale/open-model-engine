from __future__ import annotations

import asyncio
import time

import httpx
import openai

from ..types import OpenModelRunResult, ScenarioDef, ServerMetrics, classify_bottleneck
from .base import ServerBackend

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


async def _get_sglang_metrics(base_url: str, timeout: float = 2.0) -> ServerMetrics:
    sm = ServerMetrics()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/get_server_info")
            r.raise_for_status()
            data = r.json()
            sm.kv_cache_hit_rate = float(data.get("cache_hit_rate") or data.get("kv_cache_hit_rate") or 0.0)
            sm.prefill_throughput_tps = float(data.get("prefill_throughput", 0.0))
            sm.decode_throughput_tps = float(data.get("decode_throughput", 0.0))
            sm.num_running_requests = int(data.get("num_running_reqs", 0))
            sm.num_queued_requests = int(data.get("num_waiting_reqs", 0))
            sm.avg_batch_size = float(data.get("avg_batch_size", 0.0))
    except Exception:
        pass
    return sm


class SGLangServer(ServerBackend):
    backend = "sglang"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:30000",
        api_key: str = "EMPTY",
        gpu_cost_per_hour: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self._client = openai.AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout,
        )

    async def measure_single(
        self,
        scenario: ScenarioDef,
        *,
        run_index: int = 0,
        collect_gpu_metrics: bool = True,
    ) -> OpenModelRunResult:
        from ..profiling.gpu_profiler import snapshot_gpus, summarize_snapshots

        max_tokens = _MAX_TOKENS.get(scenario.name, 1024)
        messages = [
            {"role": "system", "content": scenario.system},
            {"role": "user",   "content": scenario.user},
        ]
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        input_tok = output_tok = 0

        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens,
                stream=True, stream_options={"include_usage": True},
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content if chunk.choices else ""
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    chunks.append(content)
                if getattr(chunk, "usage", None):
                    input_tok = chunk.usage.prompt_tokens or 0
                    output_tok = chunk.usage.completion_tokens or 0

            total_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            total_ms = (time.perf_counter() - t0) * 1000
            return OpenModelRunResult(
                scenario=scenario.name, backend=self.backend, model=self.model,
                run_index=run_index, ttft_ms=0.0, total_ms=total_ms, tpot_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        if ttft_ms is None:
            ttft_ms = total_ms
        if output_tok == 0:
            output_tok = max(1, len("".join(chunks)) // 4)

        tpot_ms = (total_ms - ttft_ms) / max(output_tok - 1, 1)
        tps = output_tok / (total_ms / 1000) if total_ms > 0 else 0.0
        cost = (total_ms / 1000 / 3600) * self.gpu_cost_per_hour

        gpu_mem = gpu_util = gpu_power = sm_clock = mem_clock = temp = 0.0
        sm = ServerMetrics()
        if collect_gpu_metrics:
            snaps_task = asyncio.create_task(snapshot_gpus())
            sm_task = asyncio.create_task(_get_sglang_metrics(self.base_url))
            snaps, sm = await asyncio.gather(snaps_task, sm_task, return_exceptions=True)
            if isinstance(snaps, list):
                gpu_mem, gpu_util, gpu_power, sm_clock, mem_clock, temp = summarize_snapshots(snaps)
            if isinstance(sm, Exception):
                sm = ServerMetrics()

        prefill_ms = (input_tok / sm.prefill_throughput_tps * 1000) if sm.prefill_throughput_tps > 0 and input_tok > 0 else 0.0
        decode_ms = (output_tok / sm.decode_throughput_tps * 1000) if sm.decode_throughput_tps > 0 and output_tok > 0 else 0.0

        return OpenModelRunResult(
            scenario=scenario.name, backend=self.backend, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            prefill_ms=prefill_ms, decode_ms=decode_ms,
            throughput_tps=tps,
            prefill_throughput_tps=sm.prefill_throughput_tps or (input_tok / (ttft_ms / 1000) if ttft_ms > 0 else 0.0),
            decode_throughput_tps=sm.decode_throughput_tps or tps,
            bottleneck=classify_bottleneck(prefill_ms, decode_ms, ttft_ms, tpot_ms, output_tok),
            input_tokens=input_tok, output_tokens=output_tok,
            cost_usd=cost,
            gpu_memory_used_mb=gpu_mem,
            gpu_utilization_pct=gpu_util,
            gpu_power_watts=gpu_power,
            gpu_sm_clock_mhz=sm_clock,
            gpu_memory_clock_mhz=mem_clock,
            gpu_temperature_c=temp,
            kv_cache_hit_rate=sm.kv_cache_hit_rate,
            num_running_requests=sm.num_running_requests,
            num_queued_requests=sm.num_queued_requests,
            response_text="".join(chunks),
        )

    async def get_server_metrics(self) -> ServerMetrics:
        return await _get_sglang_metrics(self.base_url)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
