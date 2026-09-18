from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .eaa import EfficientAdditiveAttention
from .ma_x3d import MAX3D, STAGE_CHANNELS, STAGES
from .motion_attention import MotionAttention
from .wide_kernel import widen_stage


def x3d_m_blocks(pretrained: bool, num_classes: int, head_dropout: float) -> nn.ModuleList:
    """Kinetics-400 X3D-M from pytorchvideo with a new classification layer."""
    from pytorchvideo.models.hub import x3d_m

    net = x3d_m(pretrained=pretrained)
    head = net.blocks[STAGES["head"]]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout.p = head_dropout
    return net.blocks


def build_model(cfg: ModelConfig) -> MAX3D:
    blocks = x3d_m_blocks(cfg.pretrained, cfg.num_classes, cfg.head_dropout)
    if cfg.wide_kernel != "none":
        if cfg.wide_kernel not in ("reparam", "dense"):
            raise ValueError(f"wide_kernel must be none|reparam|dense, got {cfg.wide_kernel!r}")
        for stage in cfg.wk_stages:
            widen_stage(blocks[STAGES[stage]], cfg.wide_kernel, cfg.wk_size)
    ma = None
    if cfg.motion_attention:
        ma = MotionAttention(STAGE_CHANNELS[cfg.ma_stage], cfg.ma_modes, cfg.ma_temporal_kernel,
                             cfg.ma_reduction)
    eaa = EfficientAdditiveAttention(STAGE_CHANNELS["res5"]) if cfg.eaa else None
    return MAX3D(blocks, ma, cfg.ma_stage, eaa, cfg.normalize_input, cfg.motion_multiscale,
                 cfg.motion_clip)
