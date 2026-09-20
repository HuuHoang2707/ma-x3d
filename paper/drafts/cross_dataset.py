"""Cross-dataset transfer: models trained on one dataset, tested on another.

No training and no adaptation: every fold model of a run is applied to the target
test clips through the same pipeline (16 evenly spaced frames, threshold 0.5).
Prints the single-model mean +- std and the fold ensemble.

Usage:
  .venv/bin/python paper/drafts/cross_dataset.py            # the full matrix
  .venv/bin/python paper/drafts/cross_dataset.py dataset/hockey runs/p0_x3d_m/seed0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ma_x3d.cli import _load_run  # noqa: E402
from ma_x3d.eval.cv import VARIANTS, metrics_at, predict_variant  # noqa: E402
from ma_x3d.train.utils import amp_dtype  # noqa: E402

# source dataset -> (label, runs trained on it); all use the final recipe
SOURCES = {
    "rwf2000": [("X3D-M", "p0_x3d_m"), ("MA-X3D", "e11_kd_long"), ("VideoMAE-B", "t01_videomae")],
    "hockey": [("X3D-M", "hk_p0_x3d_m_kd"), ("MA-X3D", "hk_e11_ma_x3d_kd"),
               ("VideoMAE-B", "hk_t01_videomae")],
    "rlvs": [("X3D-M", "rl_p0_x3d_m_kd"), ("MA-X3D", "rl_e11_ma_x3d_kd"),
             ("VideoMAE-B", "rl_t01_videomae")],
}


def evaluate(run_seed_dir: Path, target_root: Path, device) -> dict | None:
    y = np.load(target_root / "test_y.npy")
    x = target_root / "test_x.npy"
    probs = []
    for fold in sorted(p for p in run_seed_dir.glob("fold*") if (p / "best.pt").exists()):
        cfg, model = _load_run(fold, device)
        probs.append(predict_variant(model, str(x), np.arange(len(y)), cfg.data.frames,
                                     VARIANTS["1clip"], device, amp_dtype(cfg.train.amp)))
        del model
        torch.cuda.empty_cache()
    if not probs:
        return None
    single = [metrics_at(y, p, 0.5) for p in probs]
    ens = metrics_at(y, np.mean(probs, axis=0), 0.5)
    return {"acc": float(np.mean([m["accuracy"] for m in single])),
            "std": float(np.std([m["accuracy"] for m in single])),
            "f1": float(np.mean([m["f1"] for m in single])),
            "auc": float(np.mean([m["auc"] for m in single])),
            "ens": ens["accuracy"]}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) > 2:  # single target, explicit runs
        target = Path(sys.argv[1])
        for run in sys.argv[2:]:
            r = evaluate(Path(run), target, device)
            print(f"{run:34s} -> {target.name:9s} acc {100 * r['acc']:.1f} ± "
                  f"{100 * r['std']:.1f}  F1 {r['f1']:.3f}  AUC {r['auc']:.3f}")
        return
    for src, runs in SOURCES.items():
        for label, run in runs:
            seed = ROOT / "runs" / run / "seed0"
            if not seed.exists():
                continue
            row = []
            for tgt in SOURCES:
                r = evaluate(seed, ROOT / "dataset" / tgt, device)
                mark = " (in-domain)" if tgt == src else ""
                row.append(f"{tgt}: {100 * r['acc']:.1f}±{100 * r['std']:.1f} "
                           f"AUC {r['auc']:.3f}{mark}")
            print(f"{src:8s} {label:11s} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
