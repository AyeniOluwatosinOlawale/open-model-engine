from __future__ import annotations

import asyncio
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ..types import GPUSnapshot


async def snapshot_gpus() -> list[GPUSnapshot]:
    """
    Collect GPU metrics. Tries pynvml first (lower overhead), falls back to
    nvidia-smi XML subprocess. Returns [] gracefully on any failure.
    """
    # pynvml path
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        snaps: list[GPUSnapshot] = []
        ts = time.perf_counter() * 1000
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW → W
            except pynvml.NVMLError:
                power = 0.0
            try:
                sm_clock = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            except pynvml.NVMLError:
                sm_clock = 0.0
            try:
                mem_clock = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            except pynvml.NVMLError:
                mem_clock = 0.0
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                temp = 0.0
            snaps.append(GPUSnapshot(
                gpu_index=i,
                memory_used_mb=mem.used / 1024 / 1024,
                memory_total_mb=mem.total / 1024 / 1024,
                utilization_pct=float(util.gpu),
                power_watts=power,
                sm_clock_mhz=sm_clock,
                memory_clock_mhz=mem_clock,
                temperature_c=temp,
                timestamp_ms=ts,
            ))
        return snaps
    except Exception:
        pass

    # nvidia-smi XML fallback
    return await _snapshot_via_smi()


async def _snapshot_via_smi() -> list[GPUSnapshot]:
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        root = ET.fromstring(result.stdout)
        snaps: list[GPUSnapshot] = []
        ts = time.perf_counter() * 1000
        for i, gpu in enumerate(root.findall("gpu")):
            def _text(path: str, default: str = "0") -> str:
                el = gpu.find(path)
                return el.text.strip() if el is not None and el.text else default

            used_str = _text("fb_memory_usage/used").replace(" MiB", "").replace(" GiB", "")
            total_str = _text("fb_memory_usage/total").replace(" MiB", "").replace(" GiB", "")
            util_str = _text("utilization/gpu_util").replace(" %", "")
            power_str = _text("power_readings/power_draw", "0.00 W").replace(" W", "")

            snaps.append(GPUSnapshot(
                gpu_index=i,
                memory_used_mb=float(used_str) if used_str else 0.0,
                memory_total_mb=float(total_str) if total_str else 0.0,
                utilization_pct=float(util_str) if util_str else 0.0,
                power_watts=float(power_str) if power_str else 0.0,
                sm_clock_mhz=0.0,
                memory_clock_mhz=0.0,
                temperature_c=float(_text("temperature/gpu_temp", "0").replace(" C", "")),
                timestamp_ms=ts,
            ))
        return snaps
    except Exception:
        return []


def summarize_snapshots(snaps: list[GPUSnapshot]) -> tuple[float, float, float, float, float, float]:
    """Returns (mem_used_mb, util_pct, power_w, sm_clock_mhz, mem_clock_mhz, temp_c) averaged across GPUs."""
    if not snaps:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(snaps)
    return (
        sum(s.memory_used_mb for s in snaps) / n,
        sum(s.utilization_pct for s in snaps) / n,
        sum(s.power_watts for s in snaps) / n,
        sum(s.sm_clock_mhz for s in snaps) / n,
        sum(s.memory_clock_mhz for s in snaps) / n,
        sum(s.temperature_c for s in snaps) / n,
    )
