import random

import h5py
import numpy as np
import pytest
import torch

from conftest import ROOT
from ma_x3d.config import load_config
from ma_x3d.data.augment import ClipAugment
from ma_x3d.data.dataset import build_datasets
from ma_x3d.data.prepare import prepare
from ma_x3d.data.sampling import random_indices, uniform_indices
from ma_x3d.data.splits import blocked_holdout, make_splits
from ma_x3d.models.motion_attention import motion_map


def test_uniform_indices():
    idx = uniform_indices(64, 16)
    assert idx[0] == 0 and idx[-1] == 63 and len(idx) == 16


def test_random_indices_sorted_and_in_range():
    rng = random.Random(0)
    for _ in range(200):
        idx = random_indices(64, 16, rng)
        assert len(idx) == 16
        assert (np.diff(idx) >= 0).all()
        assert idx.min() >= 0 and idx.max() <= 63


def test_augment_is_clip_consistent():
    # A clip with no motion must still have no motion after augmentation.
    frame = torch.rand(3, 1, 64, 64)
    clip = frame.repeat(1, 16, 1, 1)
    aug = ClipAugment(rotation_deg=10, rng=random.Random(1))
    for _ in range(20):
        out = aug(clip)
        assert out.shape == clip.shape
        assert 0.0 <= out.min() and out.max() <= 1.0
        assert motion_map(out[None]).abs().max() < 1e-5


def test_blocked_holdout():
    labels = np.array([1] * 800 + [0] * 800)
    tr, va = blocked_holdout(labels, 0.1, 20, seed=0)
    assert len(np.intersect1d(tr, va)) == 0
    assert len(tr) + len(va) == 1600
    assert (labels[va] == 1).sum() == 80 and (labels[va] == 0).sum() == 80
    # held-out clips come in contiguous runs of 40
    runs = np.split(va, np.flatnonzero(np.diff(va) != 1) + 1)
    assert all(len(r) % 40 == 0 for r in runs)
    tr2, va2 = blocked_holdout(labels, 0.1, 20, seed=0)
    assert (va == va2).all()


def test_test_as_val_protocol():
    s = make_splits(np.zeros(10), np.zeros(4), "test_as_val", 0.1, 5, 0)
    assert s["val"] is None and len(s["train"]) == 10


def test_dataset_items(small_arrays):
    cfg = load_config(overrides=[f"data.root={small_arrays}", "data.val_blocks=10"])
    ds = build_datasets(cfg)
    assert len(ds["train"]) == len(ds["train"].indices) * 2  # augment multiplier
    assert len(ds["val"]) == 4 and len(ds["test"]) == 8
    clip, label = ds["train"][0]
    assert clip.shape == (3, 16, 32, 32) and clip.dtype == torch.float32
    assert 0.0 <= clip.min() and clip.max() <= 1.0
    assert label in (0, 1)
    a, _ = ds["test"][3]
    b, _ = ds["test"][3]
    assert torch.equal(a, b)  # evaluation is deterministic


def test_prepare_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    for name, n in (("rwf2000_train.h5", 6), ("rwf2000_val.h5", 4)):
        with h5py.File(tmp_path / name, "w") as f:
            for cls in ("Fight", "NonFight"):
                data = rng.integers(0, 256, (n // 2, 8, 3, 16, 16), dtype=np.uint8)
                f.create_dataset(cls, data=data, chunks=(2, 4, 1, 8, 8), compression="gzip")
    prepare(tmp_path, tmp_path / "out")
    x = np.load(tmp_path / "out/train_x.npy")
    y = np.load(tmp_path / "out/train_y.npy")
    with h5py.File(tmp_path / "rwf2000_train.h5") as f:
        assert np.array_equal(x[:3], f["Fight"][:]) and np.array_equal(x[3:], f["NonFight"][:])
    assert y.tolist() == [1, 1, 1, 0, 0, 0]


H5, NPY = ROOT / "dataset/rwf2000_val.h5", ROOT / "dataset/rwf2000/test_x.npy"


@pytest.mark.skipif(not (H5.exists() and NPY.exists()), reason="real RWF-2000 files not present")
def test_real_arrays_match_hdf5():
    x = np.load(NPY, mmap_mode="r")
    with h5py.File(H5) as f:
        assert np.array_equal(x[5], f["Fight"][5])
        assert np.array_equal(x[200 + 7], f["NonFight"][7])


def test_group_kfold_keeps_groups_and_classes():
    from ma_x3d.data.splits import group_kfold

    rng = np.random.default_rng(0)
    labels = np.array([1] * 100 + [0] * 100)
    groups = np.arange(200)
    groups[10:13] = 10  # a 3-clip group
    groups[150:152] = 150
    folds = group_kfold(labels, groups, 4, seed=0, pool=np.setdiff1d(np.arange(200), [5, 6]))
    allv = np.concatenate(folds)
    assert len(allv) == 198 and len(set(allv.tolist())) == 198 and 5 not in allv
    for f in folds:
        assert abs(labels[f].mean() - 0.5) < 0.05
    owner = {i: k for k, f in enumerate(folds) for i in f}
    assert owner[10] == owner[11] == owner[12] and owner[150] == owner[151]
    del rng


def test_grouped_test_split_keeps_scenes_apart(tmp_path):
    from ma_x3d.data.audit import grouped_test_split

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (40, 8, 3, 32, 32), dtype=np.uint8)
    x = np.concatenate([base, base[:4]])  # clips 40-43 duplicate clips 0-3
    y = np.array([1] * 20 + [0] * 20 + [1] * 4)
    np.save(tmp_path / "all_x.npy", x)
    np.save(tmp_path / "all_y.npy", y)
    rep = grouped_test_split(tmp_path, tmp_path / "out", 0.25, seed=0)
    assert rep["exact_duplicates_removed"] == 4
    tr, te = rep["train_index"], rep["test_index"]
    assert not set(tr) & set(te) and len(tr) + len(te) == 40
    assert rep["test_fight"] == len(te) // 2  # balanced
    assert np.load(tmp_path / "out/test_x.npy").shape == (len(te), 8, 3, 32, 32)


def test_subsample_keeps_groups_and_balance():
    from ma_x3d.data.dataset import subsample

    labels = np.array([1] * 100 + [0] * 100)
    groups = np.arange(200)
    groups[10:13] = 10  # one 3-clip group
    idx = np.arange(200)
    sub = subsample(idx, labels, groups, 0.5, seed=0)
    assert 90 <= len(sub) <= 110
    assert abs(labels[sub].mean() - 0.5) < 0.1
    owner = set(sub.tolist())
    assert {10, 11, 12} <= owner or not ({10, 11, 12} & owner)  # the group stays together
    assert (subsample(idx, labels, groups, 0.5, seed=0) == sub).all()  # deterministic
