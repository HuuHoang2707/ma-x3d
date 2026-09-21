"""Scale normalisation inside the model.

The error analysis shows that the clips the network misses hold much smaller actors
than the ones it gets right (largest person 4.0% of the frame against 7.5%), because
the crop made during preprocessing often keeps the whole scene. This module finds the
region that holds the bulk of the movement, crops it and resizes it back to the input
size, so the actors arrive at a comparable scale.

It needs nothing but the input clip: the frame difference is the same signal the
network already computes, so deployment adds no detector and about 0.01 GFLOPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion_attention import motion_map


class MotionZoom(nn.Module):
    def __init__(self, quantile: float = 0.95, spread: float = 0.1, margin: float = 0.25,
                 min_side: int = 96, size: int = 64, frames: int = 8):
        super().__init__()
        self.quantile, self.spread, self.margin = quantile, spread, margin
        self.min_side, self.size, self.frames = min_side, size, frames

    @torch.no_grad()
    def boxes(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [B, 3, T, H, W] in [0, 1] -> boxes [B, 4] (x1, y1, x2, y2) in pixels.

        The box only needs to be approximate, so it is computed on a small version of
        the clip (8 frames at 64x64): a few MFLOPs instead of the full-resolution map.
        """
        b, _, _, h, w = clip.shape
        small = F.interpolate(clip, size=(self.frames, self.size, self.size),
                              mode="trilinear", align_corners=False)
        m = motion_map(small).mean(2)                                  # [B, 1, s, s]
        m = F.avg_pool2d(m, 5, 1, 2)[:, 0]                             # [B, s, s]
        flat = m.flatten(1)
        keep = flat >= flat.quantile(self.quantile, dim=1, keepdim=True)
        mask = keep.view(b, self.size, self.size).float()

        def span(profile: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            total = profile.sum(1, keepdim=True).clamp(min=1e-6)
            cum = profile.cumsum(1) / total
            lo = torch.searchsorted(cum, torch.full_like(total, self.spread))
            hi = torch.searchsorted(cum, torch.full_like(total, 1 - self.spread))
            scale = length / self.size
            return lo.squeeze(1).float() * scale, hi.squeeze(1).float() * scale

        x1, x2 = span(mask.sum(1), w)   # column profile
        y1, y2 = span(mask.sum(2), h)   # row profile
        dx, dy = (x2 - x1) * self.margin, (y2 - y1) * self.margin
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = torch.clamp(torch.maximum(x2 - x1, y2 - y1) / 2, min=self.min_side / 2)
        box = torch.stack([cx - half, cy - half, cx + half, cy + half], dim=1)
        shift_x = (-box[:, 0]).clamp(min=0) - (box[:, 2] - w).clamp(min=0)
        shift_y = (-box[:, 1]).clamp(min=0) - (box[:, 3] - h).clamp(min=0)
        box = box + torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        return box.clamp(0, max(h, w))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import roi_align

        b, c, t, h, w = clip.shape
        box = self.boxes(clip)
        index = torch.arange(b, device=clip.device).repeat_interleave(t).float()
        rois = torch.cat([index[:, None], box.repeat_interleave(t, dim=0)], dim=1)
        frames = clip.transpose(1, 2).reshape(b * t, c, h, w)
        out = roi_align(frames, rois, output_size=(h, w), aligned=True)
        return out.view(b, t, c, h, w).transpose(1, 2)
