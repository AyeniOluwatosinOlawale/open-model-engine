from __future__ import annotations

import asyncio
import time

import httpx
import openai

from ..types import OpenModelRunResult, ScenarioDef, ServerMetrics, classify_bottleneck
from .base import ServerBackend

_MAX_TOKENS: dict[str, int] = {"short": 512, "medium": 1024, "long": 4096, "complex": 8192}


async def _get_trtllm_metrics(base_url: str, timeout: float = 2.0) -> ServerMetrics:
    sm = ServerMetrics()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/metrics")
            r.raise_for_status()
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, raw = parts[0], parts[-1]
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if "tensorrtllm_kv_cache_hit_rate" in key:
                    sm.kv_cache_hit_rate = val
                elif "tensorrtllm_kv_cache_fraction_used" in key:
                    sm.kv_cache_utilization = val
                elif "tensorrtllm_inflight_requests" in key:
                    sm.num_running_requests = int(val)
                elif "tensorrtllm_first_token_latency" in key:
                    sm.prefill_ms = val * 1000
                elif "tensorrtllm_time_per_output_token" in key:
                    sm.decode_ms = val * 1000
    except Exception:
        pass
    return sm


class TRTLLMServer(ServerBackend):
    backend = "tensorrt"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001",
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
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    stream=True, stream_options={"include_usage": True},
                )
            except (openai.BadRequestError, openai.APIError):
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens, stream=True,
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
            sm_task = asyncio.create_task(_get_trtllm_metrics(self.base_url))
            snaps, sm = await asyncio.gather(snaps_task, sm_task, return_exceptions=True)
            if isinstance(snaps, list):
                gpu_mem, gpu_util, gpu_power, sm_clock, mem_clock, temp = summarize_snapshots(snaps)
            if isinstance(sm, Exception):
                sm = ServerMetrics()

        prefill_ms = sm.prefill_ms if sm.prefill_ms > 0 else 0.0
        decode_ms = (sm.decode_ms * output_tok) if sm.decode_ms > 0 else 0.0

        return OpenModelRunResult(
            scenario=scenario.name, backend=self.backend, model=self.model,
            run_index=run_index,
            ttft_ms=ttft_ms, total_ms=total_ms, tpot_ms=tpot_ms,
            prefill_ms=prefill_ms, decode_ms=decode_ms,
            throughput_tps=tps,
            prefill_throughput_tps=input_tok / (ttft_ms / 1000) if ttft_ms > 0 and input_tok > 0 else 0.0,
            decode_throughput_tps=tps,
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
            kv_cache_utilization=sm.kv_cache_utilization,
            num_running_requests=sm.num_running_requests,
            response_text="".join(chunks),
        )

    async def get_server_metrics(self) -> ServerMetrics:
        return await _get_trtllm_metrics(self.base_url)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
