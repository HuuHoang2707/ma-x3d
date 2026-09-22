"""Duplicate / same-scene audit of the prepared arrays.

Each clip gets a small appearance descriptor (4 frames, grey, 16x16, standardised).
From cosine similarities between descriptors we derive:
  * train clips that duplicate a test clip -> excluded from training (test stays unseen)
  * exact duplicates inside train          -> the second copy is excluded
  * same-scene groups inside train         -> kept together in one K-fold fold
Writes {root}/audit.json, {root}/train_groups.npy and {root}/train_exclude.npy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def descriptors(path: str | Path) -> torch.Tensor:
    x = np.load(path, mmap_mode="r")
    frames = np.linspace(0, x.shape[1] - 1, 4).astype(int)
    out = []
    for i in range(len(x)):
        f = torch.from_numpy(np.ascontiguousarray(x[i, frames])).float()
        g = F.adaptive_avg_pool2d(f.mean(1, keepdim=True), 16).flatten()
        out.append((g - g.mean()) / (g.std() + 1e-6))
    return F.normalize(torch.stack(out), dim=1)


def _components(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = [find(i) for i in range(n)]
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def audit(root: str | Path, dup_cos: float = 0.99, scene_cos: float = 0.9) -> dict:
    root = Path(root)
    tr = descriptors(root / "train_x.npy")
    te = descriptors(root / "test_x.npy")
    ytr = np.load(root / "train_y.npy")

    s_te = (te @ tr.T).numpy()
    test_dups = sorted({int(j) for j in np.where(s_te >= dup_cos)[1]})

    s_tr = (tr @ tr.T).numpy()
    np.fill_diagonal(s_tr, -1)
    ii, jj = np.where(np.triu(s_tr >= scene_cos))
    groups = _components(len(tr), list(zip(ii.tolist(), jj.tolist(), strict=True)))
    exact = [(int(i), int(j)) for i, j in zip(ii, jj, strict=True) if s_tr[i, j] >= dup_cos]
    second_copies = sorted({j for _, j in exact})
    bad_path = root / "train_bad.npy"   # clips build_rwf could not decode
    bad = np.load(bad_path).tolist() if bad_path.exists() else []
    exclude = np.array(sorted(set(test_dups) | set(second_copies) | set(bad)),
                       dtype=np.int64)

    conflicts = [(int(i), int(j)) for i, j in zip(ii, jj, strict=True) if ytr[i] != ytr[j]]
    sizes = np.bincount(groups)
    report = {
        "train_clips": len(tr),
        "train_duplicates_of_test": test_dups,
        "train_exact_duplicate_pairs": exact,
        "excluded_from_training": exclude.tolist(),
        "unreadable_clips": [int(i) for i in bad],
        "same_scene_pairs": int(len(ii)),
        "same_scene_pairs_with_different_labels": conflicts,
        "groups": int(groups.max() + 1),
        "largest_group": int(sizes.max()),
        "thresholds": {"duplicate_cos": dup_cos, "same_scene_cos": scene_cos},
    }
    np.save(root / "train_groups.npy", groups.astype(np.int64))
    np.save(root / "train_exclude.npy", exclude)
    (root / "audit.json").write_text(json.dumps(report, indent=2))
    return report


def grouped_test_split(src: str | Path, out: str | Path, test_fraction: float = 0.2,
                       seed: int = 0, dup_cos: float = 0.99, scene_cos: float = 0.9) -> dict:
    """For datasets without an official split: all_x.npy -> train/test arrays.

    Exact duplicates are removed (one copy kept), same-scene clips are grouped, and the
    test set is drawn by whole groups with both classes balanced, so no scene appears
    on both sides. Writes out/{train,test}_{x,y}.npy and out/split.json.
    """
    from .splits import group_kfold

    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(src / "all_y.npy")
    bad = np.load(src / "bad.npy") if (src / "bad.npy").exists() else np.zeros(len(y), bool)
    d = descriptors(src / "all_x.npy")
    s = (d @ d.T).numpy()
    np.fill_diagonal(s, -1)
    ii, jj = np.where(np.triu(s >= scene_cos))
    groups = _components(len(y), list(zip(ii.tolist(), jj.tolist(), strict=True)))
    second = {int(j) for i, j in zip(ii, jj, strict=True) if s[i, j] >= dup_cos}
    keep = np.array([i for i in range(len(y)) if i not in second and not bad[i]])
    folds = group_kfold(y, groups, round(1 / test_fraction), seed, keep)
    test = folds[0]
    train = np.sort(np.setdiff1d(keep, test))
    x = np.load(src / "all_x.npy", mmap_mode="r")
    for name, idx in (("train", train), ("test", test)):
        arr = np.lib.format.open_memmap(out / f"{name}_x.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(idx), *x.shape[1:]))
        for k, i in enumerate(idx):
            arr[k] = x[i]
        arr.flush()
        np.save(out / f"{name}_y.npy", y[idx])
    report = {"clips": int(len(y)), "unreadable": int(bad.sum()),
              "exact_duplicates_removed": len(second), "same_scene_pairs": int(len(ii)),
              "groups": int(groups.max() + 1), "train": int(len(train)), "test": int(len(test)),
              "train_fight": int(y[train].sum()), "test_fight": int(y[test].sum()),
              "train_index": train.tolist(), "test_index": test.tolist()}
    (out / "split.json").write_text(json.dumps(report))
    return report
