"""Modules that add information to the backbone instead of re-weighting it.

FeatureDiffResidual: a temporal feature difference, filtered and *added* to the main
path. The Motion Attention gate of `motion_attention.py` can only rescale features
between 0 and 2; this injects a motion-derived signal the backbone cannot produce.

BurstPool: attention pooling over time in place of the head's average pooling, so a
short violent burst is not diluted over a five-second clip.

Both start as an identity: the residual is zero-initialised, and the pooling weights
start uniform, which is exactly the average.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureDiffResidual(nn.Module):
    """y = x + W_p * (depthwise conv of the temporal difference of x)."""

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.depthwise = nn.Conv3d(channels, channels, (3, kernel, kernel),
                                   padding=(1, pad, pad), groups=channels, bias=False)
        self.norm = nn.BatchNorm3d(channels)
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv3d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1, H, W]
        d = torch.cat([torch.zeros_like(d[:, :, :1]), d], dim=2)
        return x + self.project(self.act(self.norm(self.depthwise(d))))


class BurstPool(nn.Module):
    """Spatial mean, then a learned weighting over time (uniform at initialisation)."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, 1, bias=True)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean((3, 4), keepdim=True)  # [B, C, T, 1, 1]
        w = self.score(z).softmax(dim=2)  # [B, 1, T, 1, 1]
        return (z * w).sum(dim=2, keepdim=True)
