"""Training augmentation for a clip tensor [C, T, H, W] in [0, 1].

Every random parameter is drawn once per clip and applied to all frames, so the
frames stay registered and the frame-difference motion map only sees real motion.
Policy adapted from SepConvLSTM (Islam et al., 2021) with a smaller rotation and
no temporal reversal.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ClipAugment:
    def __init__(self, rotation_deg: float = 10.0, temporal_inverse: bool = False, rng=None):
        self.rotation_deg = rotation_deg
        self.temporal_inverse = temporal_inverse
        # The `random` module is reseeded per DataLoader worker; a private Random()
        # would be copied into every forked worker with the same state.
        self.rng = rng or random

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r = self.rng
        _, T, H, W = x.shape

        # Photometric: brightness, saturation, value shift.
        x = (x * r.uniform(0.5, 1.5)).clamp(0, 1)
        gray = x.mean(dim=0, keepdim=True)
        x = (gray + (1.0 + r.uniform(-0.3, 0.3)) * (x - gray)).clamp(0, 1)
        x = (x + r.uniform(-40, 40) / 255.0).clamp(0, 1)

        if r.random() < 0.5:
            x = x.flip(-1)

        if r.random() < 0.8:  # random crop, resized back
            scale = r.uniform(0.7, 1.0)
            nh, nw = int(H * scale), int(W * scale)
            top, left = r.randint(0, H - nh), r.randint(0, W - nw)
            crop = x[:, :, top : top + nh, left : left + nw].permute(1, 0, 2, 3)
            crop = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
            x = crop.permute(1, 0, 2, 3)

        if self.rotation_deg > 0 and r.random() < 0.8:
            angle = r.uniform(-self.rotation_deg, self.rotation_deg)
            frames = TF.rotate(
                x.permute(1, 0, 2, 3), angle, interpolation=TF.InterpolationMode.BILINEAR
            )
            x = frames.permute(1, 0, 2, 3)

        if self.temporal_inverse and r.random() < 0.15:
            x = x.flip(1)

        if r.random() < 0.5:  # temporal resampling: speed up or slow down
            new_t = max(int(T * r.uniform(0.7, 1.3)), 16)
            x = x[:, torch.linspace(0, T - 1, new_t).long()]
            x = x[:, torch.linspace(0, new_t - 1, T).long()]

        if r.random() < 0.2:
            sigma = r.uniform(1, 2)
            ksize = max(3, int(2 * math.ceil(2 * sigma) + 1))
            ksize += 1 - ksize % 2
            frames = TF.gaussian_blur(x.permute(1, 0, 2, 3), kernel_size=ksize, sigma=sigma)
            x = frames.permute(1, 0, 2, 3)

        return x.clamp(0, 1)
