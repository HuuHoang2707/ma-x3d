"""MA-X3D: X3D-M with optional Motion Attention, wide kernels and EAA.

Module names follow the Kaggle notebook (`blocks`, `motion_attn`, `eaa`), so its
checkpoints load directly. The model takes clips in [0, 1] and computes the motion
map and the input normalisation itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eaa import EfficientAdditiveAttention
from .motion_attention import MotionAttention, motion_map
from .slowfast import FastPathway

# Index of each X3D stage in `blocks`.
STAGES = {"stem": 0, "res2": 1, "res3": 2, "res4": 3, "res5": 4, "head": 5}
STAGE_CHANNELS = {"res2": 24, "res3": 48, "res4": 96, "res5": 192}
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


class MAX3D(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        motion_attn: MotionAttention | None = None,
        ma_stage: str = "res4",
        eaa: EfficientAdditiveAttention | None = None,
        normalize_input: bool = True,
        motion_multiscale: bool = True,
        motion_clip: float = 0.2,
        motion_input: str = "frames",
        input_size: int = 224,
        diff_residual: nn.ModuleDict | None = None,
        interaction: nn.Module | None = None,
        zoom: nn.Module | None = None,
        apn: nn.Module | None = None,
        fast: FastPathway | None = None,
    ):
        super().__init__()
        self.blocks = blocks
        self.motion_attn = motion_attn
        self.eaa = eaa
        self.ma_after = STAGES[ma_stage]
        self.normalize_input = normalize_input
        self.motion_multiscale = motion_multiscale
        self.motion_clip = motion_clip
        self.motion_input = motion_input
        self.input_size = input_size
        self.diff_residual = diff_residual or nn.ModuleDict()
        self.interaction = interaction
        self.zoom = zoom
        self.apn = apn
        self.fast = fast
        if apn is not None:  # zero-initialised: the local branches start silent
            self.apn_weight = nn.Parameter(torch.zeros(apn.crops))
        self.register_buffer("mean", torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1), False)
        self.register_buffer("std", torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1), False)

    def motion(self, clip: torch.Tensor) -> torch.Tensor:
        m = motion_map(clip, self.motion_multiscale, self.motion_clip)
        return torch.zeros_like(m) if self.motion_input == "zero" else m

    def _run(self, clip: torch.Tensor, last: int, resize: bool = True) -> torch.Tensor:  # noqa: C901, E501
        if self.zoom is not None:  # scale normalisation, before anything else
            clip = self.zoom(clip)
        fast_feats = None
        if self.fast is not None:  # the fast pathway keeps every frame, the main stream
            fast_feats = self.fast(clip)  # takes one in alpha
            clip = clip[:, :, :: self.fast.alpha]
        if resize and clip.shape[-1] != self.input_size:  # e.g. X3D-XS/S at 160x160
            size = (clip.shape[2], self.input_size, self.input_size)
            clip = F.interpolate(clip, size=size, mode="trilinear", align_corners=False)
        needs_motion = self.motion_attn is not None or self.interaction is not None
        motion = self.motion(clip) if needs_motion else None
        res4_feat = None
        x = (clip - self.mean) / self.std if self.normalize_input else clip
        for i, block in enumerate(self.blocks[: last + 1]):
            x = block(x)
            if fast_feats is not None and i in (STAGES["res2"], STAGES["res3"]):
                name = "res2" if i == STAGES["res2"] else "res3"
                x = x + FastPathway.match(fast_feats[name], x)
            if str(i) in self.diff_residual:
                x = self.diff_residual[str(i)](x)
            if i == self.ma_after and self.motion_attn is not None:
                x = self.motion_attn(x, motion)
            if i == STAGES["res4"] and self.interaction is not None:
                res4_feat = x
            if i == STAGES["res5"] and self.eaa is not None:
                x = self.eaa(x)
        if self.interaction is not None and last == len(self.blocks) - 1:
            x = x + self.interaction(res4_feat, motion)  # logit correction
        return x

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [B, 3, T, H, W] in [0, 1] -> logits [B, num_classes]."""
        logits = self._run(clip, len(self.blocks) - 1)
        if self.apn is not None:  # local branches share the backbone weights
            for k, crop in enumerate(self.apn(clip)):
                local = self._run(crop, len(self.blocks) - 1, resize=False)
                logits = logits + self.apn_weight[k] * local
        return logits

    def features(self, clip: torch.Tensor, stage: str = "res5") -> torch.Tensor:
        """Feature map after `stage` (used for Grad-CAM)."""
        return self._run(clip, STAGES[stage])
