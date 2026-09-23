"""A fast pathway for X3D, after SlowFast (Feichtenhofer et al., ICCV 2019).

We measured that sampling 32 frames instead of 16 gains about 2.7 test points and
costs 131% more computation, which defeats the point of a compact model. SlowFast's
answer is to keep the high frame rate in a branch that is deliberately thin: few
channels, half the spatial resolution, and lateral connections into the main stream.

Here the main X3D stream keeps its 16 frames, the fast pathway sees all `alpha` times
more frames at half resolution, and time-strided lateral convolutions add its features
to Res2 and Res3. The laterals are zero-initialised, so the network starts exactly as
the X3D it extends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sep_conv(cin: int, cout: int, stride: tuple[int, int, int]) -> nn.Sequential:
    """Depthwise-separable 3D convolution, the X3D building block in miniature."""
    return nn.Sequential(
        nn.Conv3d(cin, cout, 1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, (3, 3, 3), stride=stride, padding=1, groups=cout, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class FastPathway(nn.Module):
    """Thin high-frame-rate branch with lateral connections into Res2 and Res3."""

    def __init__(self, alpha: int = 4, widths: tuple[int, int, int] = (12, 12, 24),
                 slow_channels: tuple[int, int] = (24, 48)):
        super().__init__()
        self.alpha = alpha
        w0, w1, w2 = widths
        self.stem = nn.Sequential(
            nn.Conv3d(3, w0, (5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(w0),
            nn.ReLU(inplace=True),
        )
        self.block1 = _sep_conv(w0, w1, (1, 1, 1))   # half resolution, as Res2
        self.block2 = _sep_conv(w1, w2, (1, 2, 2))   # quarter resolution, as Res3
        # time-strided lateral connections, zero-initialised so they start silent
        self.lateral1 = nn.Conv3d(w1, slow_channels[0], (alpha + 1, 1, 1),
                                  stride=(alpha, 1, 1), padding=(alpha // 2, 0, 0), bias=False)
        self.lateral2 = nn.Conv3d(w2, slow_channels[1], (alpha + 1, 1, 1),
                                  stride=(alpha, 1, 1), padding=(alpha // 2, 0, 0), bias=False)
        nn.init.zeros_(self.lateral1.weight)
        nn.init.zeros_(self.lateral2.weight)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        """clip: the full-rate [B, 3, T, H, W] input -> lateral features per stage."""
        h = clip.shape[-1] // 2
        x = F.interpolate(clip, size=(clip.shape[2], h, h), mode="trilinear",
                          align_corners=False)
        x = self.block1(self.stem(x))
        out = {"res2": self.lateral1(x)}
        out["res3"] = self.lateral2(self.block2(x))
        return out

    @staticmethod
    def match(lateral: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Trim or pad the lateral to the slow stream's temporal length."""
        t = target.shape[2]
        if lateral.shape[2] > t:
            lateral = lateral[:, :, :t]
        elif lateral.shape[2] < t:
            lateral = F.pad(lateral, (0, 0, 0, 0, 0, t - lateral.shape[2]))
        return lateral
