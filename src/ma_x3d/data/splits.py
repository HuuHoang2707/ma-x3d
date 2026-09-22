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


def group_kfold(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int,
                pool: np.ndarray | None = None) -> list[np.ndarray]:
    """Stratified group K-fold: a group never spans two folds, classes stay balanced.

    Groups are placed largest first (ties in random order), each into the fold that
    currently holds the fewest clips of the group's majority class.
    """
    pool = np.arange(len(labels)) if pool is None else np.asarray(pool)
    rng = np.random.default_rng(seed)
    ids = np.unique(groups[pool])
    members = {g: pool[groups[pool] == g] for g in ids}
    order = sorted(ids, key=lambda g: (-len(members[g]), rng.random()))
    counts = np.zeros((n_folds, 2), dtype=int)
    folds = [[] for _ in range(n_folds)]
    for g in order:
        m = members[g]
        c = int(np.round(labels[m].mean()))
        f = int(np.argmin(counts[:, c] + 1e-3 * counts.sum(1)))
        folds[f].extend(m.tolist())
        counts[f] += np.bincount(labels[m], minlength=2)
    return [np.sort(np.array(f, dtype=np.int64)) for f in folds]


def make_splits(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    protocol: str,
    val_fraction: float,
    n_blocks: int,
    seed: int,
    groups: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    n_folds: int = 4,
    fold: int = 0,
) -> dict[str, np.ndarray]:
    """Indices into the train array ("train", "val") and the test array ("test").

    exclude: train clips never used (duplicates of test clips, second copies).
    """
    test_idx = np.arange(len(test_labels))
    keep = np.setdiff1d(np.arange(len(train_labels)), exclude if exclude is not None else [])
    if protocol == "kfold":
        if groups is None:
            raise ValueError("kfold needs train groups; run `ma-x3d audit` first")
        folds = group_kfold(train_labels, groups, n_folds, seed, keep)
        va = folds[fold]
        tr = np.sort(np.concatenate([f for i, f in enumerate(folds) if i != fold]))
        return {"train": tr, "val": va, "test": test_idx}
    if protocol == "holdout":
        tr, va = blocked_holdout(train_labels, val_fraction, n_blocks, seed)
        return {"train": np.intersect1d(tr, keep), "val": np.intersect1d(va, keep),
                "test": test_idx}
    if protocol in ("test_as_val", "full"):
        # "full": train on every training clip, no selection at all (the epoch budget
        # comes from the cross-validation); "test_as_val": the notebook's selection.
        return {"train": keep, "val": None, "test": test_idx}
    raise ValueError(f"unknown protocol {protocol!r}")
