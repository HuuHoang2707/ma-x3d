"""Model cost: parameters, FLOPs, latency, throughput and peak memory.

FLOPs follow the fvcore convention used in the X3D paper and the thesis: one
multiply-accumulate counts as one FLOP.
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from ..models.wide_kernel import fuse_wide_kernels


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_gflops(model: nn.Module, clip: torch.Tensor) -> float:
    """GFLOPs (MACs) for one forward pass of `clip` with batch size 1."""
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(clip[:1])
    return fc.get_total_flops() / 2 / 1e9


@torch.no_grad()
def time_forward(model: nn.Module, clip: torch.Tensor, iters: int = 50, warmup: int = 10,
                 amp: torch.dtype | None = None) -> float:
    """Median seconds per forward pass."""
    model.eval()
    dev = clip.device
    times = []
    for i in range(warmup + iters):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            model(clip)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def benchmark(model: nn.Module, device: torch.device, frames: int = 16, size: int = 224,
              batch: int = 16, iters: int = 50, amp: torch.dtype | None = None) -> dict:
    fused = copy.deepcopy(model).cpu()
    n_fused = fuse_wide_kernels(fused)
    one = torch.rand(1, 3, frames, size, size)
    out = {
        "input": [3, frames, size, size],
        "params_train": count_params(model),
        "params_infer": count_params(fused),
        "fused_convs": n_fused,
        "gflops": count_gflops(fused, one),
    }
    fused = fused.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lat = time_forward(fused, one.to(device), iters, amp=amp)
    many = torch.rand(batch, 3, frames, size, size, device=device)
    thr = time_forward(fused, many, max(10, iters // 5), amp=amp)
    out.update({
        "latency_ms": lat * 1e3,
        "throughput_clips_s": batch / thr,
        "throughput_frames_s": batch * frames / thr,
        "batch": batch,
        "amp": str(amp).replace("torch.", "") if amp else "fp32",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    })
    if device.type == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out
