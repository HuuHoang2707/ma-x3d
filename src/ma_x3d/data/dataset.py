"""Clip dataset backed by uint8 .npy arrays (see `ma-x3d prepare`).

Arrays are memory-mapped, so only the sampled frames are read from disk and the
page cache is shared between DataLoader workers and between parallel runs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .augment import ClipAugment
from .sampling import motion_indices, random_indices, uniform_indices
from .splits import make_splits

CLASS_NAMES = ("NonFight", "Fight")  # label 0, label 1


def array_paths(root: str | Path, split: str) -> tuple[Path, Path]:
    root = Path(root)
    return root / f"{split}_x.npy", root / f"{split}_y.npy"


class ClipDataset(Dataset):
    def __init__(
        self,
        x_path: str | Path,
        labels: np.ndarray,
        indices: np.ndarray,
        frames: int,
        training: bool,
        augment: ClipAugment | None = None,
        multiplier: int = 1,
        span_frac: tuple[float, float] | None = None,
        boxes: np.ndarray | None = None,
        profiles: np.ndarray | None = None,
        alt_paths: list[str] | None = None,
    ):
        self.x_path = str(x_path)
        # other preprocessings of the same clips; training picks one per sample, which
        # gives a single model the diversity that an ensemble over preprocessings has
        self.alt_paths = [str(a) for a in (alt_paths or [])]
        self._alt = None
        self.labels = labels
        self.indices = np.asarray(indices)
        self.frames = frames
        self.training = training
        self.augment = augment if training else None
        self.multiplier = multiplier if training else 1
        self.span_frac = tuple(span_frac) if span_frac else None
        self.boxes = boxes
        self.profiles = profiles
        self._x = None

    def __len__(self) -> int:
        return len(self.indices) * self.multiplier

    @property
    def x(self) -> np.ndarray:
        if self._x is None:  # opened lazily so each worker gets its own handle
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._x

    def _source(self) -> np.ndarray:
        """The array this sample is read from: a random preprocessing while training."""
        if not self.alt_paths or not self.training:
            return self.x
        if self._alt is None:
            self._alt = [np.load(a, mmap_mode="r") for a in self.alt_paths]
        pick = random.randrange(len(self._alt) + 1)
        return self.x if pick == 0 else self._alt[pick - 1]

    def __getitem__(self, i: int):
        idx = int(self.indices[i % len(self.indices)])
        src = self._source()
        stored = src.shape[1]
        if self.profiles is not None:  # frames where the clip moves most
            t = motion_indices(self.profiles[idx], self.frames,
                               random if self.training else None)
        elif self.training:
            t = random_indices(stored, self.frames, random, span_frac=self.span_frac)
        else:
            t = uniform_indices(stored, self.frames)
        clip = torch.from_numpy(np.ascontiguousarray(src[idx, t]))  # [T, C, H, W] uint8
        clip = clip.permute(1, 0, 2, 3).float().div_(255.0)  # [C, T, H, W] in [0, 1]
        if self.boxes is not None:  # one static box per clip: actors at a common scale
            x1, y1, x2, y2 = self.boxes[idx]
            if x2 - x1 > 8 and y2 - y1 > 8:
                size = clip.shape[-1]
                clip = torch.nn.functional.interpolate(
                    clip[:, :, y1:y2, x1:x2].permute(1, 0, 2, 3), size=(size, size),
                    mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        if self.augment is not None:
            clip = self.augment(clip)
        return clip, int(self.labels[idx])


def subsample(idx: np.ndarray, labels: np.ndarray, groups, fraction: float,
              seed: int) -> np.ndarray:
    """A fixed fraction of the training clips, by whole groups, class balance kept."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        pool = idx[labels[idx] == c]
        ids = np.unique(groups[pool]) if groups is not None else pool
        order = rng.permutation(len(ids))
        target, taken = int(round(fraction * len(pool))), []
        for k in order:
            if len(taken) >= target:
                break
            taken += (pool[groups[pool] == ids[k]].tolist() if groups is not None
                      else [int(ids[k])])
        keep += taken
    return np.sort(np.array(keep, dtype=np.int64))


class JointDataset(torch.utils.data.ConcatDataset):
    """Training clips of several datasets. `indices` refers to the main one, which is
    what the fold split, the teacher check and the logs are about."""

    def __init__(self, main: ClipDataset, extras: list[ClipDataset]):
        super().__init__([main, *extras])
        self.main = main

    @property
    def indices(self) -> np.ndarray:
        return self.main.indices


def _needs(path: Path, how: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `{how}`")
    return path


def build_datasets(cfg: Config) -> dict[str, ClipDataset | None]:
    d = cfg.data
    xtr, ytr = array_paths(d.root, "train")
    xte, yte = array_paths(d.root, "test")
    ytr, yte = np.load(ytr), np.load(yte)
    root = Path(d.root)
    groups = np.load(root / "train_groups.npy") if (root / "train_groups.npy").exists() else None
    exclude = None
    if d.exclude_duplicates and (root / "train_exclude.npy").exists():
        exclude = np.load(root / "train_exclude.npy")
    split = make_splits(ytr, yte, d.protocol, d.val_fraction, d.val_blocks, d.split_seed,
                        groups, exclude, d.n_folds, d.fold)
    if d.train_fraction < 1.0:
        split["train"] = subsample(split["train"], ytr, groups, d.train_fraction, d.split_seed)
    boxes = {}
    if d.roi_zoom:
        suffix = "" if d.roi_zoom == "largest" else f"_{d.roi_zoom}"
        for part in ("train", "test"):
            f = root / f"person_box{suffix}_{part}.npy"
            if not f.exists():
                raise FileNotFoundError(
                    f"{f} missing; run `ma-x3d boxes --root {d.root} --mode {d.roi_zoom}`")
            boxes[part] = np.load(f)
    profiles = {}
    if d.sampling == "motion":
        for part in ("train", "test"):
            f = root / f"motion_profile_{part}.npy"
            if not f.exists():
                raise FileNotFoundError(f"{f} missing; run `ma-x3d profiles --root {d.root}`")
            profiles[part] = np.load(f)
    elif d.sampling != "uniform":
        raise ValueError(f"sampling must be uniform|motion, got {d.sampling!r}")
    aug = ClipAugment(d.rotation_deg, d.temporal_inverse) if d.augment else None
    out = {
        "train": ClipDataset(xtr, ytr, split["train"], d.frames, True, aug, d.augment_multiplier,
                             d.train_span, boxes.get("train"), profiles.get("train"),
                             [str(array_paths(a, "train")[0]) for a in d.alt_roots]),
        "test": ClipDataset(xte, yte, split["test"], d.frames, False,
                            boxes=boxes.get("test"), profiles=profiles.get("test")),
        "val": None,
    }
    if d.extra_roots:  # joint training: add the training clips of other datasets
        extras = []
        for other in d.extra_roots:
            ox, oy = array_paths(other, "train")
            oy = np.load(oy)
            keep = np.arange(len(oy))
            ex = Path(other) / "train_exclude.npy"
            if d.exclude_duplicates and ex.exists():
                keep = np.setdiff1d(keep, np.load(ex))
            # the extra clips need the same boxes and profiles as the main ones
            ob = op = None
            if d.roi_zoom:
                suffix = "" if d.roi_zoom == "largest" else f"_{d.roi_zoom}"
                ob = np.load(_needs(Path(other) / f"person_box{suffix}_train.npy",
                                    f"ma-x3d boxes --root {other} --mode {d.roi_zoom}"))
            if d.sampling == "motion":
                op = np.load(_needs(Path(other) / "motion_profile_train.npy",
                                    f"ma-x3d profiles --root {other}"))
            extras.append(ClipDataset(ox, oy, keep, d.frames, True, aug,
                                      d.augment_multiplier, d.train_span, ob, op))
        out["train"] = JointDataset(out["train"], extras)
    if split["val"] is not None:
        out["val"] = ClipDataset(xtr, ytr, split["val"], d.frames, False,
                                 boxes=boxes.get("train"), profiles=profiles.get("train"))
    return out


def make_loader(ds: Dataset, batch_size: int, train: bool, workers: int, seed: int = 0,
                sampler=None):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train and len(ds) > batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=g,
    )
