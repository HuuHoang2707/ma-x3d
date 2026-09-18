"""Multi-GPU helpers (torchrun + DistributedDataParallel).

A run started with `torchrun --nproc_per_node=N` trains one model on N GPUs. Each GPU
gets batch_size / N clips per step and BatchNorm is synchronised across GPUs, so the
result matches a single-GPU run with the same batch size. Rank 0 evaluates, logs and
writes files; the other ranks wait.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def init() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank); starts the process group under torchrun."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return 0, 1, 0
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        # rank 0 evaluates alone between epochs, so the others may wait for minutes
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60),
                                device_id=torch.device("cuda", local))
    return dist.get_rank(), world, local


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def sum_all(values: list[float], device: torch.device) -> list[float]:
    """Sum numbers over all ranks."""
    if not dist.is_initialized():
        return values
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def broadcast_bool(value: bool, device: torch.device) -> bool:
    """Rank 0's value on every rank."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([int(value)], device=device)
    dist.broadcast(t, 0)
    return bool(t.item())


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
