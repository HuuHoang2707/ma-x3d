from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .apn import AttentionCrops
from .eaa import EfficientAdditiveAttention
from .interaction import BurstPool, FeatureDiffResidual
from .interaction_tokens import MotionPeakInteraction
from .ma_x3d import MAX3D, STAGE_CHANNELS, STAGES
from .motion_attention import MotionAttention
from .teacher import TEACHERS, VideoMAEClassifier
from .tv import TV_MODELS, TorchvisionVideo
from .wide_kernel import widen_stage
from .zoom import MotionZoom

# Kinetics-400 checkpoints of pytorchvideo's X3D: (file, depth factor). XS, S and M share
# one architecture and differ in input (XS: 4x160x160, S: 13x160x160, M: 16x224x224).
BACKBONES = {"x3d_xs": ("X3D_XS.pyth", 2.2), "x3d_s": ("X3D_S.pyth", 2.2),
             "x3d_m": ("X3D_M.pyth", 2.2), "x3d_l": ("X3D_L.pyth", 5.0)}


def x3d_blocks(pretrained: bool, num_classes: int, head_dropout: float,
               backbone: str = "x3d_m") -> nn.ModuleList:
    """Kinetics-400 X3D from pytorchvideo with a new classification layer.

    Built for 224x224 input. X3D-L was trained at 312, but only its (weight-free) head
    pooling depends on the crop size, so the checkpoint still loads. XS and S run at
    their own input size; their head pools globally, which equals their original
    pooling at that size.
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
    if backbone in ("x3d_xs", "x3d_s"):
        head.pool.pool = nn.AdaptiveAvgPool3d(1)
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout.p = head_dropout
    return net.blocks


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone in TEACHERS:
        if cfg.motion_attention or cfg.wide_kernel != "none" or cfg.eaa or cfg.apn_crops:
            raise ValueError(f"{cfg.backbone} takes no MA-X3D modules")
        return VideoMAEClassifier(cfg.backbone, cfg.num_classes, cfg.pretrained)
    if cfg.backbone in TV_MODELS:
        if cfg.motion_attention or cfg.wide_kernel != "none" or cfg.eaa or cfg.apn_crops:
            raise ValueError(f"{cfg.backbone} takes no MA-X3D modules")
        return TorchvisionVideo(cfg.backbone, cfg.num_classes, cfg.pretrained)
    blocks = x3d_blocks(cfg.pretrained, cfg.num_classes, cfg.head_dropout, cfg.backbone)
    if cfg.wide_kernel != "none":
        if cfg.wide_kernel not in ("reparam", "dense"):
            raise ValueError(f"wide_kernel must be none|reparam|dense, got {cfg.wide_kernel!r}")
        for stage in cfg.wk_stages:
            widen_stage(blocks[STAGES[stage]], cfg.wide_kernel, cfg.wk_size,
                        cfg.wk_tsize)
    ma = None
    if cfg.motion_attention:
        ma = MotionAttention(STAGE_CHANNELS[cfg.ma_stage], cfg.ma_modes, cfg.ma_temporal_kernel,
                             cfg.ma_reduction)
    eaa = EfficientAdditiveAttention(STAGE_CHANNELS["res5"]) if cfg.eaa else None
    diff = nn.ModuleDict({str(STAGES[st]): FeatureDiffResidual(STAGE_CHANNELS[st])
                          for st in cfg.diff_residual})
    apn = None
    if cfg.apn_crops:
        apn = AttentionCrops(cfg.apn_crops, cfg.apn_size)
        pool = blocks[STAGES["head"]].pool  # crops are smaller, so pool adaptively
        pool.pool = nn.AdaptiveAvgPool3d(1)
    if cfg.zoom not in ("", "motion"):
        raise ValueError(f"zoom must be ''|motion, got {cfg.zoom!r}")
    zoom = MotionZoom() if cfg.zoom == "motion" else None
    inter = (MotionPeakInteraction(STAGE_CHANNELS["res4"], cfg.num_classes,
                                   cfg.interaction_peaks) if cfg.interaction else None)
    if cfg.burst_pool:
        pool = blocks[STAGES["head"]].pool
        pool.pool = BurstPool(pool.post_conv.in_channels)
    if cfg.motion_input not in ("frames", "zero"):
        raise ValueError(f"motion_input must be frames|zero, got {cfg.motion_input!r}")
    return MAX3D(blocks, ma, cfg.ma_stage, eaa, cfg.normalize_input, cfg.motion_multiscale,
                 cfg.motion_clip, cfg.motion_input, cfg.input_size, diff, inter, zoom, apn)
