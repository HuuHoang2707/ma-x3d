"""Kinetics-400 video CNNs from torchvision, used as reproduced baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (torchvision builder, input side the weights were trained at)
TV_MODELS = {
    "tv_r3d_18": ("r3d_18", 112),
    "tv_mc3_18": ("mc3_18", 112),
    "tv_r2plus1d_18": ("r2plus1d_18", 112),
    "tv_s3d": ("s3d", 224),
}
# normalisation used by the torchvision video weights
TV_MEAN = (0.43216, 0.394666, 0.37645)
TV_STD = (0.22803, 0.22145, 0.216989)


class TorchvisionVideo(nn.Module):
    """Same input contract as MAX3D: RGB in [0, 1], [B, 3, T, 224, 224].

    The clip is resized to the input side of the pre-trained weights and normalised
    inside the model; the classification layer is replaced.
    """

    def __init__(self, name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import video

        builder, self.size = TV_MODELS[name]
        self.net = getattr(video, builder)(weights="KINETICS400_V1" if pretrained else None)
        if builder == "s3d":
            old = self.net.classifier[1]
            self.net.classifier[1] = nn.Conv3d(old.in_channels, num_classes, 1)
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.register_buffer("mean", torch.tensor(TV_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(TV_STD).view(1, 3, 1, 1, 1))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.shape[-1] != self.size:
            size = (clip.shape[2], self.size, self.size)
            clip = F.interpolate(clip, size=size, mode="trilinear", align_corners=False)
        return self.net((clip - self.mean) / self.std)
