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
from .sampling import random_indices, uniform_indices
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
    ):
        self.x_path = str(x_path)
        self.labels = labels
        self.indices = np.asarray(indices)
        self.frames = frames
        self.training = training
        self.augment = augment if training else None
        self.multiplier = multiplier if training else 1
        self._x = None

    def __len__(self) -> int:
        return len(self.indices) * self.multiplier

    @property
    def x(self) -> np.ndarray:
        if self._x is None:  # opened lazily so each worker gets its own handle
            self._x = np.load(self.x_path, mmap_mode="r")
        return self._x

    def __getitem__(self, i: int):
        idx = int(self.indices[i % len(self.indices)])
        stored = self.x.shape[1]
        if self.training:
            t = random_indices(stored, self.frames, random)
        else:
            t = uniform_indices(stored, self.frames)
        clip = torch.from_numpy(np.ascontiguousarray(self.x[idx, t]))  # [T, C, H, W] uint8
        clip = clip.permute(1, 0, 2, 3).float().div_(255.0)  # [C, T, H, W] in [0, 1]
        if self.augment is not None:
            clip = self.augment(clip)
        return clip, int(self.labels[idx])


def build_datasets(cfg: Config) -> dict[str, ClipDataset | None]:
    d = cfg.data
    xtr, ytr = array_paths(d.root, "train")
    xte, yte = array_paths(d.root, "test")
    ytr, yte = np.load(ytr), np.load(yte)
    split = make_splits(ytr, yte, d.protocol, d.val_fraction, d.val_blocks, d.split_seed)
    aug = ClipAugment(d.rotation_deg, d.temporal_inverse) if d.augment else None
    out = {
        "train": ClipDataset(xtr, ytr, split["train"], d.frames, True, aug, d.augment_multiplier),
        "test": ClipDataset(xte, yte, split["test"], d.frames, False),
        "val": None,
    }
    if split["val"] is not None:
        out["val"] = ClipDataset(xtr, ytr, split["val"], d.frames, False)
    return out


def make_loader(ds: Dataset, batch_size: int, train: bool, workers: int, seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train and len(ds) > batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=g,
    )
