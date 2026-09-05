from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NsightReport:
    report_path: str
    cuda_prefill_kernel_ms: float = 0.0
    cuda_decode_kernel_ms: float = 0.0
    hbm_read_bandwidth_gbps: float = 0.0
    hbm_write_bandwidth_gbps: float = 0.0
    memory_transfer_ms: float = 0.0
    raw_stats: dict = field(default_factory=dict)


class NsightProfiler:
    """
    Nsight Systems CLI wrapper for deep CUDA kernel profiling.
    Annotates prefill and decode phases with NVTX ranges so that
    nsys stats can attribute kernel time to each phase separately.

    Requires: Nsight Systems CLI (`nsys`) on PATH.
    Optional: `pip install nvtx` for in-process range annotations.
    """

    def __init__(self, output_dir: str = "./profiles") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._nsys = shutil.which("nsys")
        self._nvtx_ok = False
        try:
            import nvtx  # noqa: F401
            self._nvtx_ok = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._nsys is not None

    @contextlib.contextmanager
    def nvtx_range(self, name: str, color: str = "blue"):
        """
        Annotates a code block with an NVTX range.
        Use 'prefill' and 'decode' as names to enable phase-level attribution.
        """
        if self._nvtx_ok:
            import nvtx
            with nvtx.annotate(name, color=color):
                yield
        else:
            yield

    async def profile_subprocess(
        self,
        cmd: list[str],
        output_name: str = "profile",
        trace: str = "cuda,nvtx,osrt",
    ) -> NsightReport:
        """
        Wrap `cmd` with nsys profile. If nsys not available, runs cmd directly.
        Returns NsightReport with parsed kernel times.
        """
        rep = str(self.output_dir / output_name)

        if not self._nsys:
            await asyncio.to_thread(subprocess.run, cmd, check=False)
            return NsightReport(report_path=rep)

        nsys_cmd = [
            self._nsys, "profile",
            "--output", rep,
            "--trace", trace,
            "--force-overwrite", "true",
            "--export", "json",
            "--cuda-memory-usage", "true",
        ] + cmd

        proc = await asyncio.to_thread(
            subprocess.run, nsys_cmd, capture_output=True, check=False
        )
        return await self._parse(rep)

    async def _parse(self, rep: str) -> NsightReport:
        report = NsightReport(report_path=rep)
        stats_out = rep + "_kern_stats.json"

        if not self._nsys:
            return report

        try:
            await asyncio.to_thread(
                subprocess.run,
                [self._nsys, "stats",
                 "--report", "cuda_gpu_kern_sum",
                 "--format", "json",
                 "--output", stats_out,
                 rep + ".nsys-rep"],
                capture_output=True, check=False, timeout=120,
            )
            if os.path.exists(stats_out):
                with open(stats_out) as f:
                    data = json.load(f)
                report.raw_stats = data
                report = _parse_kernel_stats(report, data)
        except Exception:
            pass

        return report


def _parse_kernel_stats(report: NsightReport, data: dict) -> NsightReport:
    """
    Attribute CUDA kernel time to prefill vs decode based on kernel names.

    Prefill kernels (compute-bound, context processing):
      flash_attn_fwd, fmha_forward, context_attention, ampere_fp16_s16816*

    Decode kernels (memory-bandwidth-bound, token generation):
      paged_attention_v*, decode_attention, single_query_cached_kv_attention
    """
    try:
        kernels = data.get("NvtxKernelSum") or data.get("CudaGpuKernSum") or []
        prefill_ms = decode_ms = 0.0
        for k in kernels:
            name = (k.get("Name") or k.get("name") or "").lower()
            dur_ns = float(k.get("Total Time (ns)") or k.get("totalTimeNs") or 0)
            dur_ms = dur_ns / 1_000_000
            if any(t in name for t in [
                "flash_attn_fwd", "fmha_forward", "context_attn",
                "flash_forward", "ampere_fp16_s16816",
            ]):
                prefill_ms += dur_ms
            elif any(t in name for t in [
                "paged_attention", "decode_attention",
                "single_query_cached_kv", "generation_attention",
            ]):
                decode_ms += dur_ms

        report.cuda_prefill_kernel_ms = prefill_ms
        report.cuda_decode_kernel_ms = decode_ms
    except Exception:
        pass
    return report
