"""Efficient additive attention over a 3D feature map (SwiftFormer / CUE-Net style).

Optional experiment from the Kaggle notebook v5; not part of the paper model.
LayerScale starts at zero, so the block is an identity at initialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EfficientAdditiveAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.w_a = nn.Parameter(torch.randn(dim) * dim**-0.5)
        self.proj = nn.Linear(dim, dim)
        self.final = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # [B, N, C]
        n = self.norm(x)
        q, k = self.to_q(n), self.to_k(n)
        a = torch.softmax((q @ self.w_a) * c**-0.5, dim=1).unsqueeze(-1)
        g = (a * q).sum(dim=1, keepdim=True)  # global context [B, 1, C]
        x = x + self.gamma * self.final(self.proj(k * g) + q)
        return x.transpose(1, 2).reshape(b, c, t, h, w)
