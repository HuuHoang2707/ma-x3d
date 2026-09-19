"""MA-X3D: X3D-M with optional Motion Attention, wide kernels and EAA.

Module names follow the Kaggle notebook (`blocks`, `motion_attn`, `eaa`), so its
checkpoints load directly. The model takes clips in [0, 1] and computes the motion
map and the input normalisation itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eaa import EfficientAdditiveAttention
from .motion_attention import MotionAttention, motion_map

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
        self.register_buffer("mean", torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1), False)
        self.register_buffer("std", torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1), False)

    def motion(self, clip: torch.Tensor) -> torch.Tensor:
        m = motion_map(clip, self.motion_multiscale, self.motion_clip)
        return torch.zeros_like(m) if self.motion_input == "zero" else m

    def _run(self, clip: torch.Tensor, last: int) -> torch.Tensor:
        motion = self.motion(clip) if self.motion_attn is not None else None
        x = (clip - self.mean) / self.std if self.normalize_input else clip
        for i, block in enumerate(self.blocks[: last + 1]):
            x = block(x)
            if i == self.ma_after and self.motion_attn is not None:
                x = self.motion_attn(x, motion)
            if i == STAGES["res5"] and self.eaa is not None:
                x = self.eaa(x)
        return x

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [B, 3, T, H, W] in [0, 1] -> logits [B, num_classes]."""
        return self._run(clip, len(self.blocks) - 1)

    def features(self, clip: torch.Tensor, stage: str = "res5") -> torch.Tensor:
        """Feature map after `stage` (used for Grad-CAM)."""
        return self._run(clip, STAGES[stage])
