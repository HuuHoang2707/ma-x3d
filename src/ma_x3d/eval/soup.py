"""Fold soup: average the weights of the k fold models into one model.

The k-fold protocol already trains k models from the same pre-trained weights, and
their fold ensemble is consistently more accurate than any one of them. If they stay
in one basin, averaging their weights (a uniform model soup, Wortsman et al., ICML
2022) keeps much of that gain in a single network, with no extra training and no
extra inference cost. The recipe is fixed in advance (uniform, every fold), so
scoring it on the test split involves no selection.

BatchNorm running statistics are averaged with the weights. `recalibrate` re-estimates
them on training clips for the layers where the folds differ; it helped on the
experiment it was developed on and was a wash on eight replications (four better,
four worse), so the recipe is the plain uniform soup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def average_states(paths: list[Path]) -> dict:
    """Uniform average of the `model` state dicts in the given best.pt files."""
    states = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in paths]
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(s[k].double() for s in states).div(len(states)).to(v.dtype)
        else:  # counters such as num_batches_tracked
            out[k] = v.clone()
    return out


def differing_bn(paths: list[Path], atol: float = 1e-6) -> set[str]:
    """Names of BatchNorm layers whose running statistics differ between the models.

    Layers of frozen stages keep their Kinetics statistics in every fold, so they are
    identical and must be left alone.
    """
    states = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in paths]
    names = set()
    for k in states[0]:
        if k.endswith("running_mean"):
            ref = states[0][k]
            if any(not torch.allclose(s[k], ref, atol=atol) for s in states[1:]):
                names.add(k[: -len(".running_mean")])
    return names


@torch.no_grad()
def recalibrate(model, x_path: str, indices: np.ndarray, frames: int, device,
                only: set[str] | None = None, batch: int = 16) -> int:
    """Re-estimate the BatchNorm statistics of the layers in `only` on training clips."""
    from ..data.sampling import uniform_indices

    bns = [m for n, m in model.named_modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
           and (only is None or n in only)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches
    model.eval()
    for m in bns:  # only these layers update their statistics
        m.train()
    x = np.load(x_path, mmap_mode="r")
    t = uniform_indices(x.shape[1], frames)
    for start in range(0, len(indices), batch):
        ids = indices[start : start + batch]
        clip = torch.from_numpy(np.stack([x[i, t] for i in ids]))
        model(clip.permute(0, 2, 1, 3, 4).float().div_(255).to(device))
    model.eval()
    return len(bns)
