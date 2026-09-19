from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .eaa import EfficientAdditiveAttention
from .ma_x3d import MAX3D, STAGE_CHANNELS, STAGES
from .motion_attention import MotionAttention
from .teacher import TEACHERS, VideoMAEClassifier
from .wide_kernel import widen_stage

# Kinetics-400 checkpoints and depth factors of pytorchvideo's X3D-M and X3D-L.
BACKBONES = {"x3d_m": ("X3D_M.pyth", 2.2), "x3d_l": ("X3D_L.pyth", 5.0)}


def x3d_blocks(pretrained: bool, num_classes: int, head_dropout: float,
               backbone: str = "x3d_m") -> nn.ModuleList:
    """Kinetics-400 X3D from pytorchvideo with a new classification layer.

    Built for 224x224 input. X3D-L was trained at 312, but only its (weight-free) head
    pooling depends on the crop size, so the checkpoint still loads.
    """
    from pytorchvideo.models.x3d import create_x3d
    from torch.hub import load_state_dict_from_url

    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {list(BACKBONES)}, got {backbone!r}")
    ckpt, depth = BACKBONES[backbone]
    net = create_x3d(input_clip_length=16, input_crop_size=224, depth_factor=depth)
    if pretrained:
        url = f"https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/{ckpt}"
        net.load_state_dict(load_state_dict_from_url(url, map_location="cpu")["model_state"])
    head = net.blocks[STAGES["head"]]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout.p = head_dropout
    return net.blocks


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone in TEACHERS:
        if cfg.motion_attention or cfg.wide_kernel != "none" or cfg.eaa:
            raise ValueError(f"{cfg.backbone} takes no MA-X3D modules")
        return VideoMAEClassifier(cfg.backbone, cfg.num_classes, cfg.pretrained)
    blocks = x3d_blocks(cfg.pretrained, cfg.num_classes, cfg.head_dropout, cfg.backbone)
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
