"""Motion map and the Motion Attention module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_map(clip: torch.Tensor, multiscale: bool = True, clip_at: float = 0.2) -> torch.Tensor:
    """Channel-averaged absolute frame difference, scaled to about [0, 1].

    clip: [B, C, T, H, W] in [0, 1]. Returns [B, 1, T, H, W].
    With multiscale, takes the max of the step-1 and step-2 differences.
    The first frame(s) reuse the first difference so T is preserved.
    """
    d1 = (clip[:, :, 1:] - clip[:, :, :-1]).abs().mean(1, keepdim=True)
    m = torch.cat([d1[:, :, :1], d1], dim=2)
    if multiscale:
        d2 = (clip[:, :, 2:] - clip[:, :, :-2]).abs().mean(1, keepdim=True)
        d2 = torch.cat([d2[:, :, :1], d2[:, :, :1], d2], dim=2)
        m = torch.maximum(m, d2)
    return m.clamp(0.0, clip_at) / clip_at


class MotionAttention(nn.Module):
    """Gate a feature map with the motion map: y = f * (1 + g(M)).

    A temporal conv turns the 1-channel motion map into `modes` temporal patterns;
    a small 1x1x1 bottleneck with tanh maps them to one gain per feature channel.
    Mode 0 starts as a copy of the motion map and the last gate layer starts near
    zero, so the module is close to an identity at initialisation.
    """

    def __init__(self, channels: int = 96, modes: int = 4, temporal_kernel: int = 3,
                 reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        if modes > 0:
            self.temporal_conv = nn.Conv3d(
                1, modes, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0), bias=False
            )
            with torch.no_grad():
                w = self.temporal_conv.weight
                w.zero_()
                w[0, 0, temporal_kernel // 2] = 1.0
                if modes > 1:
                    w[1:].normal_(0.0, 0.05)
            gate_in = modes
        else:  # raw motion map straight into the gate (ablation)
            self.temporal_conv = None
            gate_in = 1
        self.gate = nn.Sequential(
            nn.Conv3d(gate_in, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Tanh(),
        )
        nn.init.normal_(self.gate[2].weight, std=0.01)
        nn.init.zeros_(self.gate[2].bias)

    def gain(self, feat: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        m = self.temporal_conv(motion) if self.temporal_conv is not None else motion
        m = F.interpolate(m, size=feat.shape[2:], mode="trilinear", align_corners=False)
        return self.gate(m)

    def forward(self, feat: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return feat * (1.0 + self.gain(feat, motion))
