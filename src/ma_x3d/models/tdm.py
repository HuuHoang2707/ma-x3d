"""Temporal difference features, added to a stage instead of gating it.

Motion Attention turns a motion map into a multiplicative gate, and its own control
shows that the gate works just as well with the motion input removed: the network
gets nothing from the motion signal itself. This module takes the opposite route,
following the short-term branch of TDN (Wang et al., CVPR 2021): the differences
between neighbouring feature maps are convolved and added to the features, so motion
enters as a signal the next layers can read rather than as a scaling factor.

The output projection is zero-initialised, so the stage starts exactly as it was.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDifference(nn.Module):
    """y = x + w * conv(x[t] - x[t-1]), with w zero at initialisation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv3d(channels, hidden, 1, bias=False)
        self.spatial = nn.Conv3d(hidden, hidden, (1, 3, 3), padding=(0, 1, 1),
                                 groups=hidden, bias=False)
        self.norm = nn.BatchNorm3d(hidden)
        self.expand = nn.Conv3d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.expand.weight)  # identity at the start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, :, 1:] - x[:, :, :-1]                 # [B, C, T-1, H, W]
        d = F.pad(d, (0, 0, 0, 0, 1, 0))               # keep T, first frame has no past
        d = self.expand(F.relu(self.norm(self.spatial(self.reduce(d))), inplace=True))
        return x + d


class WithTemporalDifference(nn.Module):
    """Wraps an X3D stage so its output carries the difference features."""

    def __init__(self, stage: nn.Module, channels: int, reduction: int = 4):
        super().__init__()
        self.stage = stage
        self.tdm = TemporalDifference(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tdm(self.stage(x))
