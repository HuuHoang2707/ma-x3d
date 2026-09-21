"""How much of the stored frame do the people occupy, and does it explain the errors?

Runs YOLOv8n on the stored (already cropped) clips, measures the area of the largest
person box and of the union of person boxes, and compares those numbers between the
clips a model gets right and wrong. If the preprocessing crop failed to isolate the
actors, the wrong clips should have much smaller person boxes.

Usage: .venv/bin/python paper/drafts/person_area.py runs/n1_diff_residual/seed0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ma_x3d.data.sampling import uniform_indices  # noqa: E402
from ma_x3d.eval.cv import _oof  # noqa: E402

X = ROOT / "dataset/rwf2000/train_x.npy"


def person_area(indices: np.ndarray, frames: int = 6) -> np.ndarray:
    """[N, 3]: largest box area, union area, number of people (median over frames)."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    x = np.load(X, mmap_mode="r")
    t = uniform_indices(x.shape[1], frames)
    out = np.zeros((len(indices), 3))
    for k, i in enumerate(indices):
        batch = [x[i, f].transpose(1, 2, 0) for f in t]
        largest, union, count = [], [], []
        for res in model(batch, classes=[0], verbose=False, conf=0.25):
            b = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            count.append(len(b))
            if len(b) == 0:
                largest.append(0.0), union.append(0.0)
                continue
            areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (224 * 224)
            largest.append(float(areas.max()))
            mask = np.zeros((224, 224), dtype=bool)
            for x1, y1, x2, y2 in b.astype(int):
                mask[max(y1, 0):y2, max(x1, 0):x2] = True
            union.append(float(mask.mean()))
        out[k] = [np.median(largest), np.median(union), np.median(count)]
    return out

def main() -> None:
    run = Path(sys.argv[1])
    probs, y, _ = _oof(run, "1clip")
    idx = np.sort(np.concatenate([np.load(f)["val_idx"]
                                  for f in sorted(run.glob("fold*/cv_predictions.npz"))]))
    pred = (probs >= 0.5).astype(int)
    correct = pred == y
    sub = np.concatenate([np.flatnonzero(~correct),
                          np.flatnonzero(correct)[:400]])  # all errors, 400 correct
    area = person_area(idx[sub])
    n_wrong = int((~correct).sum())
    w, r = area[:n_wrong], area[n_wrong:]
    miss = np.flatnonzero(~correct)
    is_miss = y[miss] == 1
    print(f"{'group':16s} {'largest box':>12s} {'union':>8s} {'people':>7s}  n")
    for name, a in (("wrong", w), ("  missed fight", w[is_miss]), ("  false alarm", w[~is_miss]),
                    ("correct", r)):
        print(f"{name:16s} {a[:, 0].mean():12.3f} {a[:, 1].mean():8.3f} "
              f"{a[:, 2].mean():7.2f}  {len(a)}")
    small = (w[:, 0] < 0.05).mean(), (r[:, 0] < 0.05).mean()
    none = (w[:, 2] == 0).mean(), (r[:, 2] == 0).mean()
    print(f"\nclips whose largest person covers < 5% of the frame: "
          f"{100 * small[0]:.0f}% of wrong, {100 * small[1]:.0f}% of correct")
    print(f"clips with no person detected: {100 * none[0]:.0f}% of wrong, "
          f"{100 * none[1]:.0f}% of correct")
    np.save(ROOT / "reports/person_area.npy", np.vstack([idx[sub], area.T]).T)


if __name__ == "__main__":
    main()
