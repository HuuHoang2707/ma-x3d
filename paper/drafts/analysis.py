"""Analyses for the paper that need no training.

  motion   OOF accuracy of two experiments per quartile of clip motion
           (mean of the motion map over the 16 evaluation frames).
  gate     statistics of the Motion Attention gain 1 + G on the OOF clips of one run:
           mean |G| for violent and non-violent clips.

Usage:
  .venv/bin/python paper/drafts/analysis.py motion runs/p0_x3d_m/seed0 runs/e11_kd_long/seed0
  .venv/bin/python paper/drafts/analysis.py gate runs/e11_kd_long/seed0
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


def clip_motion(indices: np.ndarray, frames: int = 16) -> np.ndarray:
    x = np.load(X, mmap_mode="r")
    t = uniform_indices(x.shape[1], frames)
    out = np.zeros(len(indices))
    for k, i in enumerate(indices):
        clip = torch.from_numpy(np.ascontiguousarray(x[i, t])).permute(1, 0, 2, 3)[None]
        out[k] = motion_map(clip.float() / 255).mean().item()
    return out


def motion(a: str, b: str) -> None:
    pa, y, _ = _oof(Path(a), "1clip")
    pb, _, _ = _oof(Path(b), "1clip")
    idx = np.sort(np.concatenate([np.load(f)["val_idx"]
                                  for f in sorted(Path(a).glob("fold*/cv_predictions.npz"))]))
    m = clip_motion(idx)
    q = np.quantile(m, [0.25, 0.5, 0.75])
    bins = np.digitize(m, q)
    ca, cb = (pa >= 0.5) == y, (pb >= 0.5) == y
    print(f"{'motion quartile':16s} {'clips':>5s} {'fight %':>7s} {'A acc':>6s} {'B acc':>6s} "
          f"{'B-A':>6s}")
    for k in range(4):
        s = bins == k
        print(f"Q{k + 1} ({m[s].min():.3f}-{m[s].max():.3f}) {s.sum():5d} "
              f"{100 * y[s].mean():7.1f} {100 * ca[s].mean():6.2f} {100 * cb[s].mean():6.2f} "
              f"{100 * (cb[s].mean() - ca[s].mean()):+6.2f}")


@torch.no_grad()
def gate(run_dir: str) -> None:
    from ma_x3d.cli import _load_run

    device = torch.device("cuda")
    x = np.load(X, mmap_mode="r")
    y_all = np.load(ROOT / "dataset/rwf2000/train_y.npy")
    t = uniform_indices(x.shape[1], 16)
    stats = {0: [], 1: []}
    grab = {}

    def hook(_m, _i, out):
        grab["g"] = out

    for fold in sorted(Path(run_dir).glob("fold*")):
        cfg, model = _load_run(fold, device)
        idx = np.load(fold / "cv_predictions.npz")["val_idx"]
        h = model.motion_attn.gate.register_forward_hook(hook)
        for s in range(0, len(idx), 16):
            ids = idx[s:s + 16]
            clip = torch.from_numpy(np.stack([x[i, t] for i in ids])).permute(0, 2, 1, 3, 4)
            model(clip.float().div(255).to(device))
            g = grab["g"].abs().mean(dim=(1, 2, 3, 4)).cpu().numpy()
            for i, v in zip(ids, g, strict=True):
                stats[int(y_all[i])].append(float(v))
        h.remove()
    for c, name in ((1, "violent"), (0, "non-violent")):
        v = np.array(stats[c])
        print(f"{name:12s} clips {len(v):4d}  mean |G| {v.mean():.4f}  median {np.median(v):.4f}")


if __name__ == "__main__":
    {"motion": motion, "gate": gate}[sys.argv[1]](*sys.argv[2:])
