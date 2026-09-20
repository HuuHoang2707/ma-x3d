"""Motion-peak interaction tokens.

Violence is an interaction between people, which a globally pooled 3D feature cannot
express. Instead of using the frame-difference map to re-weight features (a gate,
which we measure to have no effect), we use it to *select* the few locations that
move most, treat each as an actor token, and model the relations between pairs of
them: where they are relative to each other and how correlated their motion is over
time.

No detector and no pose model are needed at inference: the motion map is already
computed by the network. The pair features are mapped to a logit correction through a
zero-initialised layer, so the module is an identity at initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionPeakInteraction(nn.Module):
    """Top-k motion peaks -> actor tokens -> pairwise relations -> logit correction.

    feat:   [B, C, T, H, W] backbone features (Res4)
    motion: [B, 1, T, Hm, Wm] motion map in [0, 1]
    """

    def __init__(self, channels: int, num_classes: int = 2, peaks: int = 6,
                 dim: int = 64, min_distance: int = 2):
        super().__init__()
        self.peaks, self.min_distance = peaks, min_distance
        self.token = nn.Sequential(nn.Linear(channels + 3, dim), nn.ReLU(inplace=True))
        # pair input: two tokens, their difference, geometry (dx, dy, distance) and
        # the correlation of their motion profiles
        self.pair = nn.Sequential(nn.Linear(3 * dim + 4, dim), nn.ReLU(inplace=True),
                                  nn.Linear(dim, dim), nn.ReLU(inplace=True))
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, num_classes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _select(self, energy: torch.Tensor) -> torch.Tensor:
        """energy [B, H, W] -> indices [B, k] of the k strongest locations, spread out
        by suppressing a neighbourhood around each pick."""
        b, h, w = energy.shape
        e = energy.clone().flatten(1)
        picks = []
        for _ in range(self.peaks):
            idx = e.argmax(dim=1)
            picks.append(idx)
            y, x = idx // w, idx % w
            yy = torch.arange(h, device=e.device).view(1, h, 1)
            xx = torch.arange(w, device=e.device).view(1, 1, w)
            near = ((yy - y.view(b, 1, 1)).abs() <= self.min_distance) & \
                   ((xx - x.view(b, 1, 1)).abs() <= self.min_distance)
            e = e.masked_fill(near.flatten(1), -1e4)
        return torch.stack(picks, dim=1)

    def forward(self, feat: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = feat.shape
        m = F.adaptive_avg_pool3d(motion, (t, h, w))            # [B, 1, T, H, W]
        profile = m.squeeze(1).flatten(2)                        # [B, T, HW]
        energy = profile.mean(1).view(b, h, w)                   # [B, H, W]
        idx = self._select(energy)                               # [B, k]

        flat = feat.mean(2).flatten(2)                           # [B, C, HW]
        tokens = flat.gather(2, idx.unsqueeze(1).expand(b, c, self.peaks))  # [B, C, k]
        tokens = tokens.transpose(1, 2)                          # [B, k, C]
        ys = (idx // w).float() / max(h - 1, 1)
        xs = (idx % w).float() / max(w - 1, 1)
        peak_energy = energy.flatten(1).gather(1, idx)           # [B, k]
        z = self.token(torch.cat([tokens, ys[..., None], xs[..., None],
                                  peak_energy[..., None]], dim=2))  # [B, k, dim]

        # motion profile of each peak over time, for the correlation feature
        prof = profile.transpose(1, 2).gather(1, idx.unsqueeze(2).expand(b, self.peaks, t))
        prof = F.normalize(prof - prof.mean(2, keepdim=True), dim=2)  # [B, k, T]

        i, j = torch.triu_indices(self.peaks, self.peaks, offset=1, device=feat.device)
        zi, zj = z[:, i], z[:, j]
        geom = torch.stack([xs[:, i] - xs[:, j], ys[:, i] - ys[:, j],
                            ((xs[:, i] - xs[:, j]) ** 2 + (ys[:, i] - ys[:, j]) ** 2).sqrt(),
                            (prof[:, i] * prof[:, j]).sum(2)], dim=2)  # [B, P, 4]
        r = self.pair(torch.cat([zi, zj, zi - zj, geom], dim=2))       # [B, P, dim]
        weight = self.score(r).softmax(dim=1)
        return self.out((r * weight).sum(dim=1))                       # [B, num_classes]
