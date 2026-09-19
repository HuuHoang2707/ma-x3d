"""Large video transformer used as a distillation teacher (not part of MA-X3D)."""

from __future__ import annotations

import torch
import torch.nn as nn

# Hugging Face checkpoints fine-tuned on Kinetics-400 (16 frames, 224x224)
TEACHERS = {"videomae_b": "MCG-NJU/videomae-base-finetuned-kinetics"}


class VideoMAEClassifier(nn.Module):
    """Same input contract as MAX3D: RGB in [0, 1], [B, 3, 16, 224, 224]."""

    def __init__(self, name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        repo = TEACHERS[name]
        if pretrained:
            self.net = VideoMAEForVideoClassification.from_pretrained(
                repo, num_labels=num_classes, ignore_mismatched_sizes=True,
                attn_implementation="sdpa")
        else:
            cfg = VideoMAEConfig.from_pretrained(repo, num_labels=num_classes)
            self.net = VideoMAEForVideoClassification(cfg)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        x = ((clip - self.mean) / self.std).transpose(1, 2)  # [B, T, 3, H, W]
        return self.net(pixel_values=x).logits
