from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import OpenModelRunResult, ScenarioDef, ServerMetrics


class ServerBackend(ABC):
    backend: str = "base"
    base_url: str = ""
    model: str = ""

    @abstractmethod
    async def measure_single(
        self,
        scenario: ScenarioDef,
        *,
        run_index: int = 0,
        collect_gpu_metrics: bool = True,
    ) -> OpenModelRunResult: ...

    @abstractmethod
    async def get_server_metrics(self) -> ServerMetrics: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    async def warm_up(self, scenario: ScenarioDef) -> None:
        """Send one request to warm up KV cache / JIT compilation."""
        await self.measure_single(scenario, run_index=-1, collect_gpu_metrics=False)

    async def close(self) -> None:
        pass
