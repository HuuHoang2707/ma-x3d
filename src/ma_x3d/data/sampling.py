"""Choose which of the stored frames a clip uses."""

from __future__ import annotations

import random

import numpy as np


def uniform_indices(num_stored: int, num_out: int) -> np.ndarray:
    """Deterministic, evenly spaced indices (evaluation)."""
    return np.linspace(0, num_stored - 1, num_out).astype(int)


def random_indices(
    num_stored: int,
    num_out: int,
    rng: random.Random,
    speed: tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    """Random window, random playback speed, +-1 frame jitter, kept in temporal order."""
    start = rng.randint(0, max(0, num_stored - num_out))
    span = int(num_out * rng.uniform(*speed))
    span = int(np.clip(span, num_out // 2, num_stored - start))
    if span < 2:
        return uniform_indices(num_stored, num_out)
    base = np.linspace(start, start + span - 1, num_out)
    jitter = np.array([rng.randint(-1, 1) for _ in range(num_out)])
    return np.sort(np.clip(base + jitter, 0, num_stored - 1).astype(int))
