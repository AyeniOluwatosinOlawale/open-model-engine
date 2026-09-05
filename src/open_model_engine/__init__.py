"""
open-model-engine — GPU-first inference harness for vLLM, SGLang, TensorRT-LLM.

Quick start:
    from open_model_engine import OpenModelEngine, ScenarioDef
    engine = OpenModelEngine("vllm", "meta-llama/Llama-3.1-8B",
                              base_url="http://localhost:8000")
    results = await engine.bench(ScenarioDef.medium(), runs=5)
"""
__version__ = "0.1.0"

from .engine import OpenModelEngine
from .types import (
    AggregatedMetrics,
    GPUSnapshot,
    OpenModelRunResult,
    RunResult,
    ScenarioDef,
    ServerMetrics,
    aggregate,
    classify_bottleneck,
)
from .servers import get_server

__all__ = [
    "OpenModelEngine",
    "get_server",
    "OpenModelRunResult",
    "RunResult",
    "AggregatedMetrics",
    "ScenarioDef",
    "GPUSnapshot",
    "ServerMetrics",
    "aggregate",
    "classify_bottleneck",
]
