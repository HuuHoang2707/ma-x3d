"""Small training helpers: seeding, EMA, CutMix, schedulers, run metadata."""

from __future__ import annotations

import copy
import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_dtype(name: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[name]


class ModelEMA:
    """Exponential moving average of weights and buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(src[k])


def cutmix(clip: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Paste the same box from a shuffled batch into every frame of each clip.

    Returns (clip, labels_b, lam): the loss is lam * CE(labels) + (1 - lam) * CE(labels_b).
    """
    b, _, _, h, w = clip.shape
    perm = torch.randperm(b, device=clip.device)
    lam = float(np.random.beta(alpha, alpha))
    rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
    cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
    y1, y2 = max(0, cy - rh // 2), min(h, cy + rh // 2)
    x1, x2 = max(0, cx - rw // 2), min(w, cx + rw // 2)
    clip = clip.clone()
    clip[..., y1:y2, x1:x2] = clip[perm][..., y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return clip, labels[perm], lam


def make_scheduler(opt: torch.optim.Optimizer, phase: str, epochs: int):
    """Per-epoch schedule. Probe: one warm-up epoch at 0.1x, then cosine. Finetune: cosine."""
    lr = torch.optim.lr_scheduler
    if phase == "probe" and epochs >= 2:
        warm = lr.LinearLR(opt, start_factor=0.1, total_iters=1)
        cos = lr.CosineAnnealingLR(opt, T_max=epochs - 1)
        return lr.SequentialLR(opt, [warm, cos], milestones=[1])
    if phase == "probe":
        return lr.LinearLR(opt, start_factor=0.1, total_iters=1)
    return lr.CosineAnnealingLR(opt, T_max=max(1, epochs))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def environment_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "host": platform.node(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
    return info
