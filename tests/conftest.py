from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_arrays(root: Path, n_train=8, n_test=4, frames=16, size=224, seed=0) -> Path:
    """Synthetic arrays in the `ma-x3d prepare` layout (half Fight, half NonFight)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        x = rng.integers(0, 256, size=(n, frames, 3, size, size), dtype=np.uint8)
        y = np.array([1] * (n // 2) + [0] * (n - n // 2), dtype=np.int64)
        np.save(root / f"{split}_x.npy", x)
        np.save(root / f"{split}_y.npy", y)
    return root


@pytest.fixture
def small_arrays(tmp_path) -> Path:
    return write_arrays(tmp_path / "arrays", n_train=40, n_test=8, frames=64, size=32)


@pytest.fixture(scope="session")
def x3d_blocks():
    """Randomly initialised X3D-M (no download), built once per session."""
    from ma_x3d.models.builder import x3d_blocks

    return x3d_blocks(pretrained=False, num_classes=2, head_dropout=0.5)
