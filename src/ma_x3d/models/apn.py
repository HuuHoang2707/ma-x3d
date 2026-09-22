"""Attention prediction crops, after Varghese et al. (2025).

Their framework predicts several spatiotemporal regions of interest, processes each
one separately and fuses the local features with the global ones. We keep that idea
but stay inside the budget of a compact model: the regions are predicted by a small
network from a downsampled clip, the crops are taken with a differentiable sampler,
and the *same* backbone processes them at half resolution, so n crops cost about
n/4 of the global pass instead of n full backbones.

The local logits are added to the global ones through a zero-initialised weight, so
the model starts exactly as the backbone it extends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCrops(nn.Module):
    """Predict `crops` boxes per clip and sample them from the input."""

    def __init__(self, crops: int = 2, size: int = 112, hidden: int = 32,
                 min_scale: float = 0.3):
        super().__init__()
        self.crops, self.size, self.min_scale = crops, size, min_scale
        self.net = nn.Sequential(  # small 3D CNN on a 64x64 version of the clip
            nn.Conv3d(3, hidden, (3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(True),
            nn.Conv3d(hidden, hidden, (3, 3, 3), stride=(1, 2, 2), padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.head = nn.Linear(hidden, crops * 3)  # centre x, centre y, scale per crop
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def boxes(self, clip: torch.Tensor) -> torch.Tensor:
        """[B, crops, 3] with centre in [-1, 1] and scale in [min_scale, 1]."""
        small = F.interpolate(clip, size=(min(clip.shape[2], 8), 64, 64), mode="trilinear",
                              align_corners=False)
        p = self.head(self.net(small)).view(-1, self.crops, 3)
        centre = torch.tanh(p[..., :2]) * 0.5                       # stay inside the frame
        scale = self.min_scale + (1 - self.min_scale) * torch.sigmoid(p[..., 2])
        return torch.cat([centre, scale[..., None]], dim=2)

    def forward(self, clip: torch.Tensor) -> list[torch.Tensor]:
        """-> `crops` clips of shape [B, 3, T, size, size], differentiable in the box."""
        b, c, t, h, w = clip.shape
        box = self.boxes(clip)
        frames = clip.transpose(1, 2).reshape(b * t, c, h, w)
        out = []
        for k in range(self.crops):
            cx, cy, s = box[:, k, 0], box[:, k, 1], box[:, k, 2]
            theta = torch.zeros(b, 2, 3, device=clip.device, dtype=clip.dtype)
            theta[:, 0, 0] = s
            theta[:, 1, 1] = s
            theta[:, 0, 2] = cx
            theta[:, 1, 2] = cy
            theta = theta.repeat_interleave(t, dim=0)
            grid = F.affine_grid(theta, (b * t, c, self.size, self.size), align_corners=False)
            crop = F.grid_sample(frames, grid, align_corners=False)
            out.append(crop.view(b, t, c, self.size, self.size).transpose(1, 2))
        return out
