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
    span_frac: tuple[float, float] | None = None,
) -> np.ndarray:
    """Random window, +-1 frame jitter, kept in temporal order.

    span_frac=(lo, hi): the window covers lo..hi of the stored clip, so training sees
    the same time scale as evaluation (which spreads the frames over the whole clip).
    span_frac=None: the thesis sampler, a window of num_out * speed stored frames
    (11-24 of 64 for 16 frames, i.e. much denser than evaluation).
    """
    if span_frac is not None:
        span = int(np.clip(num_stored * rng.uniform(*span_frac), num_out, num_stored))
        start = rng.randint(0, num_stored - span)
    else:
        start = rng.randint(0, max(0, num_stored - num_out))
        span = int(num_out * rng.uniform(*speed))
        span = int(np.clip(span, num_out // 2, num_stored - start))
    if span < 2:
        return uniform_indices(num_stored, num_out)
    base = np.linspace(start, start + span - 1, num_out)
    jitter = np.array([rng.randint(-1, 1) for _ in range(num_out)])
    return np.sort(np.clip(base + jitter, 0, num_stored - 1).astype(int))


def motion_indices(profile: np.ndarray, num_out: int, rng: random.Random | None = None,
                   span_frac: tuple[float, float] = (0.4, 0.8)) -> np.ndarray:
    """Frames concentrated where the clip moves most (Song et al., 2019: key frames
    instead of uniform sampling).

    profile: motion energy per stored frame. At evaluation (rng=None) the window with
    the most motion is taken; during training a window is drawn with probability
    proportional to the motion it contains, which keeps the sampler stochastic.
    """
    stored = len(profile)
    span = int(np.clip(round(stored * (span_frac[0] + span_frac[1]) / 2), num_out, stored))
    if rng is not None:
        span = int(np.clip(round(stored * rng.uniform(*span_frac)), num_out, stored))
    weight = np.convolve(profile, np.ones(span), mode="valid")  # motion inside each window
    if weight.sum() <= 0:
        start = 0 if rng is None else rng.randint(0, len(weight) - 1)
    elif rng is None:
        start = int(weight.argmax())
    else:
        p = weight / weight.sum()
        start = int(np.searchsorted(np.cumsum(p), rng.random(), side="right"))
        start = min(start, len(weight) - 1)
    base = np.linspace(start, start + span - 1, num_out)
    if rng is not None:
        base = base + np.array([rng.randint(-1, 1) for _ in range(num_out)])
    return np.sort(np.clip(base, 0, stored - 1).astype(int))
