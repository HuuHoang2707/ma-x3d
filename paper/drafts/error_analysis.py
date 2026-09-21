"""Look at the clips the model gets wrong.

Uses the out-of-fold predictions (never the test split) of a finished run: lists the
most confident mistakes, writes a strip of frames for each so they can be inspected,
and reports simple statistics of the wrong clips against the right ones (motion energy,
brightness, sharpness, how much of the frame moves).

Usage:
  .venv/bin/python paper/drafts/error_analysis.py runs/n1_diff_residual/seed0 [n]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ma_x3d.data.sampling import uniform_indices  # noqa: E402
from ma_x3d.eval.cv import _oof  # noqa: E402
from ma_x3d.models.motion_attention import motion_map  # noqa: E402

X = ROOT / "dataset/rwf2000/train_x.npy"
OUT = ROOT / "reports/errors"


def clip_stats(idx: np.ndarray) -> dict[str, np.ndarray]:
    x = np.load(X, mmap_mode="r")
    t = uniform_indices(x.shape[1], 16)
    motion, bright, sharp, area = [], [], [], []
    for i in idx:
        clip = torch.from_numpy(np.ascontiguousarray(x[i, t])).permute(1, 0, 2, 3)[None].float()
        clip /= 255
        m = motion_map(clip)
        motion.append(m.mean().item())
        area.append((m > 0.3).float().mean().item())  # share of the frame that moves
        bright.append(clip.mean().item())
        g = clip.mean(1)  # [1, T, H, W]
        sharp.append((g[:, :, 1:] - g[:, :, :-1]).abs().mean().item())
    return {"motion": np.array(motion), "moving_area": np.array(area),
            "brightness": np.array(bright), "detail": np.array(sharp)}


def strip(i: int, path: Path, frames: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.load(X, mmap_mode="r")
    t = uniform_indices(x.shape[1], frames)
    fig, axes = plt.subplots(1, frames, figsize=(frames * 1.6, 1.8))
    for ax, k in zip(axes, t, strict=True):
        ax.imshow(x[i, k].transpose(1, 2, 0))
        ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    run = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    probs, y, _ = _oof(run, "1clip")
    idx = np.sort(np.concatenate([np.load(f)["val_idx"]
                                  for f in sorted(run.glob("fold*/cv_predictions.npz"))]))
    pred = (probs >= 0.5).astype(int)
    wrong = np.flatnonzero(pred != y)
    print(f"{len(wrong)} of {len(y)} out-of-fold clips wrong ({100 * len(wrong) / len(y):.1f}%)")
    miss = wrong[y[wrong] == 1]   # violent clips called non-violent
    false = wrong[y[wrong] == 0]  # non-violent clips called violent
    print(f"  missed fights: {len(miss)}   false alarms: {len(false)}")

    sw, sr = clip_stats(idx[wrong]), clip_stats(idx[np.flatnonzero(pred == y)])
    print(f"\n{'statistic':12s} {'wrong':>8s} {'correct':>8s}")
    for k in sw:
        print(f"{k:12s} {sw[k].mean():8.3f} {sr[k].mean():8.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    conf = np.abs(probs - 0.5)
    for name, group in (("missed_fight", miss), ("false_alarm", false)):
        order = group[np.argsort(-conf[group])][:n]
        for rank, w in enumerate(order):
            strip(int(idx[w]), OUT / f"{name}_{rank}_clip{idx[w]}_p{probs[w]:.2f}.png")
        print(f"wrote {min(n, len(order))} strips for {name}")


if __name__ == "__main__":
    main()
