"""Which parameters train in each phase, and at which learning rate.

probe:    backbone frozen; head + new modules train at lr_probe.
finetune: stages in `train.unfreeze` train at lr_backbone; the head, the new modules
          and the wide-kernel ring train at lr_new.

The X3D head holds two pre-trained convolutions (about 970K parameters) before the new
linear layer. `train.head_new="proj"` treats only that linear layer as new; the head
convolutions then count as backbone (frozen in the probe, lr_backbone afterwards).

`param_match="legacy"` reproduces the Kaggle notebook, which selected parameters by
substring. There "blocks.5" also matched "blocks.3.res_blocks.5" (a Res4 block), and
"blocks.1" matched "res_blocks.1" and "res_blocks.10", so several Res4/Res5 blocks
trained when they were meant to be frozen, some of them at the high learning rate.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import TrainConfig
from ..models.ma_x3d import STAGES

HEAD = f"blocks.{STAGES['head']}."


def _in_stage(name: str, stage: str) -> bool:
    if stage == "all":  # every pre-trained weight (used for the VideoMAE teacher)
        return True
    return name.startswith(f"blocks.{STAGES[stage]}.")


def _is_ring(name: str) -> bool:
    return name.endswith("delta_weight")


def _is_new(name: str, cfg: TrainConfig) -> bool:
    if name.startswith(("motion_attn.", "eaa.", "net.classifier.", "net.fc.",
                        "diff_residual.")):
        return True
    if "pool.score" in name:  # BurstPool
        return True
    if name.startswith(HEAD):
        return cfg.head_new == "all" or name.startswith(HEAD + "proj.")
    return False


def _exact(name: str, phase: str, cfg: TrainConfig) -> tuple[bool, bool]:
    """Return (trainable, uses the new-module learning rate)."""
    new = _is_new(name, cfg)
    ring = _is_ring(name)
    if phase == "probe":
        return new, True
    trainable = new or ring or any(_in_stage(name, s) for s in cfg.unfreeze)
    return trainable, new or (ring and cfg.ring_lr == "new")


def _legacy(name: str, phase: str, has_wide: bool) -> tuple[bool, bool]:
    new = any(k in name for k in ("motion_attn", "eaa", "blocks.5", "blocks.6")) or _is_ring(name)
    if phase == "probe":
        return "blocks.5" in name or "motion_attn" in name or "eaa" in name, True
    keys = ["blocks.4", "blocks.5"] + (["blocks.1", "blocks.2"] if has_wide else [])
    trainable = any(k in name for k in keys) or "motion_attn" in name or "eaa" in name
    return trainable or _is_ring(name), new


def configure_phase(model: nn.Module, phase: str, cfg: TrainConfig) -> list[dict]:
    """Set requires_grad for `phase` and return AdamW parameter groups."""
    has_wide = any((n.endswith("conv_b.weight") and p.shape[-1] > 3) or _is_ring(n)
                   for n, p in model.named_parameters())
    backbone, new = [], []
    for name, p in model.named_parameters():
        if cfg.param_match == "legacy":
            trainable, is_new = _legacy(name, phase, has_wide)
        elif cfg.param_match == "exact":
            trainable, is_new = _exact(name, phase, cfg)
        else:
            raise ValueError(f"param_match must be exact|legacy, got {cfg.param_match!r}")
        p.requires_grad_(trainable)
        if trainable:
            (new if is_new else backbone).append(p)

    if phase == "probe":
        return [{"params": backbone + new, "lr": cfg.lr_probe, "name": "probe"}]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr_backbone, "name": "backbone"})
    if new:
        groups.append({"params": new, "lr": cfg.lr_new, "name": "new"})
    return groups


def trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def freeze_frozen_bn(model: nn.Module) -> None:
    """Put BatchNorm layers whose parameters are all frozen into eval mode."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()
