from .base import ServerBackend
from .vllm_server import VLLMServer
from .sglang_server import SGLangServer
from .trtllm_server import TRTLLMServer

__all__ = ["ServerBackend", "VLLMServer", "SGLangServer", "TRTLLMServer"]


def get_server(backend: str, model: str, base_url: str = "", **kwargs) -> ServerBackend:
    b = backend.lower()
    if b == "vllm":
        return VLLMServer(model=model, base_url=base_url or "http://localhost:8000", **kwargs)
    if b == "sglang":
        return SGLangServer(model=model, base_url=base_url or "http://localhost:30000", **kwargs)
    if b in ("tensorrt", "trtllm", "trt"):
        return TRTLLMServer(model=model, base_url=base_url or "http://localhost:8001", **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose from: vllm, sglang, tensorrt")
