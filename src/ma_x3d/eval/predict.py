from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..progress import Progress
from .metrics import classification_metrics


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: torch.dtype | None = None,
            label_smoothing: float = 0.0, desc: str = "eval") -> dict:
    """Run `model` over `loader`; return probabilities, labels, mean loss and metrics."""
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for clip, y in Progress(loader, desc=desc, every=0.5):
        clip = clip.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(clip)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, label_smoothing=label_smoothing,
                                    reduction="sum").item()
        probs.append(logits.softmax(1).cpu())
        labels.append(y.cpu())
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    metrics = classification_metrics(labels, probs.argmax(1), probs[:, 1])
    metrics["loss"] = loss_sum / len(labels)
    return {"probs": probs, "labels": labels, "metrics": metrics}


def save_predictions(path, out: dict) -> None:
    np.savez_compressed(path, probs=out["probs"], labels=out["labels"])
