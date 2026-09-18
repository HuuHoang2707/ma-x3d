"""Train/validation/test index splits.

RWF-2000 clips were cut from longer YouTube videos, and the HDF5 files store clips
in sorted file-name order, so clips from the same source video sit next to each
other. A plain random hold-out would put sibling clips on both sides and inflate
validation scores. We instead hold out whole contiguous blocks of indices per class.
"""

from __future__ import annotations

import numpy as np


def blocked_holdout(
    labels: np.ndarray, val_fraction: float, n_blocks: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx), stratified by class, holding out contiguous blocks."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    k = max(1, round(val_fraction * n_blocks))
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        blocks = np.array_split(idx, n_blocks)
        chosen = set(rng.choice(n_blocks, size=k, replace=False).tolist())
        for b, block in enumerate(blocks):
            (val if b in chosen else train).append(block)
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def make_splits(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    protocol: str,
    val_fraction: float,
    n_blocks: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Indices into the train array ("train", "val") and the test array ("test")."""
    test_idx = np.arange(len(test_labels))
    if protocol == "holdout":
        tr, va = blocked_holdout(train_labels, val_fraction, n_blocks, seed)
        return {"train": tr, "val": va, "test": test_idx}
    if protocol == "test_as_val":
        return {"train": np.arange(len(train_labels)), "val": None, "test": test_idx}
    raise ValueError(f"unknown protocol {protocol!r}")
