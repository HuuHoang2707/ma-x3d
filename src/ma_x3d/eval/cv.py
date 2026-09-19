"""K-fold evaluation: out-of-fold (OOF) predictions, inference variants, threshold,
fold-model ensemble on the test set.

Decisions (inference variant, threshold) are made on OOF predictions only. The test
set is scored with the 4 fold models averaged, using those decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .metrics import classification_metrics

# name -> (temporal windows, window span as fraction of the stored clip,
#          sub-stride offsets per window, horizontal-flip TTA)
VARIANTS = {
    "1clip": (1, 1.0, 1, False),
    "1clip+flip": (1, 1.0, 1, True),
    "2off": (1, 1.0, 2, False),
    "4off": (1, 1.0, 4, False),
    "2off+flip": (1, 1.0, 2, True),
    "4off+flip": (1, 1.0, 4, True),
    "3win": (3, 0.7, 1, False),
    "3win+flip": (3, 0.7, 1, True),
}
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)


def view_indices(stored: int, frames: int, windows: int, span: float,
                 offsets: int) -> list[np.ndarray]:
    """Frame indices of every temporal view. (1, 1.0, 1) is the standard evaluation."""
    length = max(frames, int(round(stored * span)))
    starts = np.linspace(0, stored - length, windows) if windows > 1 else [0.0]
    views = []
    for s in starts:
        if offsets == 1:  # same spacing as the standard evaluation
            idx = np.linspace(s, s + length - 1, frames)
            views.append(idx.astype(int))
            continue
        step = length / frames  # interleaved clips: shift by a fraction of the stride
        for k in range(offsets):
            idx = s + step * (np.arange(frames) + k / offsets)
            views.append(np.clip(np.floor(idx), 0, stored - 1).astype(int))
    return views


@torch.no_grad()
def predict_variant(model, x_path: str, indices: np.ndarray, frames: int, variant: tuple,
                    device, amp=None, batch: int = 16) -> np.ndarray:
    """Fight probability per clip, averaged over the variant's views."""
    windows, span, offsets, flip = variant
    x = np.load(x_path, mmap_mode="r")
    views = view_indices(x.shape[1], frames, windows, span, offsets)
    model.eval()
    probs = np.zeros(len(indices))
    for start in range(0, len(indices), batch):
        ids = indices[start : start + batch]
        acc = torch.zeros(len(ids), device=device)
        n = 0
        for v in views:
            clip = torch.from_numpy(np.stack([x[i, v] for i in ids]))  # [B, T, C, H, W]
            clip = clip.permute(0, 2, 1, 3, 4).float().div_(255).to(device)
            for c in ([clip, clip.flip(-1)] if flip else [clip]):
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    acc += model(c).float().softmax(1)[:, 1]
                n += 1
        probs[start : start + len(ids)] = (acc / n).cpu().numpy()
    return probs


def metrics_at(labels, probs, threshold: float) -> dict:
    return classification_metrics(labels, (probs >= threshold).astype(int), probs)


def best_threshold(labels, probs, key: str = "accuracy") -> float:
    """Threshold on the grid that maximises `key`; ties go to the one closest to 0.5."""
    scores = [(metrics_at(labels, probs, t)[key], -abs(t - 0.5), t) for t in THRESHOLDS]
    return float(max(scores)[2])


def evaluate_cv(exp_dir: str | Path, device, variants=None, key: str = "accuracy") -> dict:
    """exp_dir = runs/<name>/seed<s>, holding fold0..foldK-1 with best.pt."""
    from ..cli import _load_run
    from ..data.dataset import array_paths, build_datasets
    from ..train.utils import amp_dtype

    exp_dir = Path(exp_dir)
    folds = sorted(p for p in exp_dir.glob("fold*") if (p / "best.pt").exists())
    if not folds:
        raise FileNotFoundError(f"no finished folds in {exp_dir}")
    variants = variants or list(VARIANTS)
    per_fold = []
    for fdir in folds:
        cache = fdir / "cv_predictions.npz"
        cached = dict(np.load(cache)) if cache.exists() else {}
        cfg, model = _load_run(fdir, device)
        ds = build_datasets(cfg)
        amp = amp_dtype(cfg.train.amp)
        xtr, _ = array_paths(cfg.data.root, "train")
        xte, _ = array_paths(cfg.data.root, "test")
        out = {"val_idx": ds["val"].indices, "val_y": ds["val"].labels[ds["val"].indices],
               "test_y": ds["test"].labels}
        for name in variants:
            for split, xp, idx in (("val", xtr, ds["val"].indices),
                                   ("test", xte, ds["test"].indices)):
                k = f"{split}:{name}"
                out[k] = cached[k] if k in cached else predict_variant(
                    model, str(xp), idx, cfg.data.frames, VARIANTS[name], device, amp)
        np.savez(cache, **out)
        per_fold.append(out)
        del model
        torch.cuda.empty_cache()

    oof_y = np.concatenate([f["val_y"] for f in per_fold])
    test_y = per_fold[0]["test_y"]
    rows = {}
    for name in variants:
        oof_p = np.concatenate([f[f"val:{name}"] for f in per_fold])
        th = best_threshold(oof_y, oof_p, key)
        test_p = np.mean([f[f"test:{name}"] for f in per_fold], axis=0)
        single = [metrics_at(test_y, f[f"test:{name}"], 0.5) for f in per_fold]
        rows[name] = {
            "views": _n_views(VARIANTS[name]),
            "oof@0.5": metrics_at(oof_y, oof_p, 0.5),
            "threshold": th,
            "oof@th": metrics_at(oof_y, oof_p, th),
            "test_ensemble@0.5": metrics_at(test_y, test_p, 0.5),
            "test_ensemble@th": metrics_at(test_y, test_p, th),
            "test_single_mean@0.5": {
                m: float(np.mean([s[m] for s in single])) for m in ("accuracy", "f1")},
            "test_single_std@0.5": {
                m: float(np.std([s[m] for s in single])) for m in ("accuracy", "f1")},
        }
    res = {"exp": str(exp_dir), "folds": len(per_fold), "oof_clips": int(len(oof_y)),
           "selection_key": key, "variants": rows}
    (exp_dir / "cv_eval.json").write_text(json.dumps(res, indent=2))
    return res


def _n_views(v: tuple) -> int:
    windows, _, offsets, flip = v
    return windows * offsets * (2 if flip else 1)


def format_cv(res: dict) -> str:
    """Table: decisions by OOF; test columns are the fold ensemble."""
    lines = [f"{res['exp']}  ({res['folds']} folds, {res['oof_clips']} OOF clips)",
             f"{'variant':12s} {'views':>5s} | {'OOF acc':>7s} {'OOF F1':>6s} {'th':>4s} "
             f"{'OOF acc@th':>10s} | {'TEST acc':>8s} {'P':>5s} {'R':>5s} {'F1':>5s} "
             f"{'AUC':>5s} | single-model test acc"]
    for name, r in res["variants"].items():
        o, oth, t = r["oof@0.5"], r["oof@th"], r["test_ensemble@th"]
        sm, ss = r["test_single_mean@0.5"], r["test_single_std@0.5"]
        lines.append(
            f"{name:12s} {r['views']:5d} | {o['accuracy']:7.4f} {o['f1']:6.4f} "
            f"{r['threshold']:4.2f} {oth['accuracy']:10.4f} | {t['accuracy']:8.4f} "
            f"{t['precision']:5.3f} {t['recall']:5.3f} {t['f1']:5.3f} {t['auc']:5.3f} | "
            f"{sm['accuracy']:.4f} ± {ss['accuracy']:.4f}")
    return "\n".join(lines)


def experiment_table(exp_dirs: list[str | Path], variant: str = "1clip") -> str:
    """Markdown table over experiments: OOF metrics (used for decisions) and the fold
    ensemble on test (reported, never used for decisions)."""
    import yaml

    head = ("| Experiment | Change | Variant | OOF Acc | OOF P | OOF R | OOF F1 | th | "
            "Test Acc | Test P | Test R | Test F1 | Test AUC |")
    lines = [head, "|" + " --- |" * (head.count("|") - 1)]
    for d in exp_dirs:
        d = Path(d)
        f = d / "cv_eval.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        v = variant
        if v == "best":  # best variant by OOF accuracy at the OOF threshold
            v = max(res["variants"], key=lambda k: res["variants"][k]["oof@th"]["accuracy"])
        r = res["variants"][v]
        cfg = yaml.safe_load(next(d.glob("fold*/config.yaml")).read_text())
        o, t = r["oof@th"], r["test_ensemble@th"]
        lines.append(
            f"| {cfg['name']} | {cfg.get('description', '')} | {v} | {100 * o['accuracy']:.2f} | "
            f"{o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {r['threshold']:.2f} | "
            f"{100 * t['accuracy']:.2f} | {t['precision']:.3f} | {t['recall']:.3f} | "
            f"{t['f1']:.3f} | {t['auc']:.3f} |")
    return "\n".join(lines)


def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired per-clip correctness."""
    from math import comb

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def _oof(exp_dir: Path, variant: str) -> tuple[np.ndarray, np.ndarray, list]:
    """OOF probability per train clip index, labels, and per-fold test probabilities."""
    folds = sorted(Path(exp_dir).glob("fold*/cv_predictions.npz"))
    if not folds:
        raise FileNotFoundError(f"run `ma-x3d cv {exp_dir}` first")
    idx, y, p, test = [], [], [], []
    for f in folds:
        d = np.load(f)
        idx.append(d["val_idx"]), y.append(d["val_y"]), p.append(d[f"val:{variant}"])
        test.append((d["test_y"], d[f"test:{variant}"]))
    order = np.argsort(np.concatenate(idx))
    return np.concatenate(p)[order], np.concatenate(y)[order], test


def compare(exp_a, exp_b, variant: str = "1clip", threshold: float = 0.5,
            n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of two k-fold experiments (B minus A).

    OOF: accuracy difference, 95% bootstrap interval over clips, exact McNemar p.
    Test: difference of the fold models (mean, std), and McNemar on the fold ensemble.
    Both experiments must use the same folds (same split_seed and exclusions).
    """
    pa, ya, ta = _oof(exp_a, variant)
    pb, yb, tb = _oof(exp_b, variant)
    if len(ya) != len(yb) or (ya != yb).any():
        raise ValueError("the two experiments do not share the same folds")
    ca, cb = (pa >= threshold) == ya, (pb >= threshold) == yb
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(ya), (n_boot, len(ya)))
    diffs = cb[boot].mean(1) - ca[boot].mean(1)
    fold_diff = [((b >= threshold) == y).mean() - ((a >= threshold) == y).mean()
                 for (y, a), (_, b) in zip(ta, tb, strict=True)]
    y = ta[0][0]
    ea = (np.mean([a for _, a in ta], 0) >= threshold) == y
    eb = (np.mean([b for _, b in tb], 0) >= threshold) == y
    return {
        "a": str(exp_a), "b": str(exp_b), "variant": variant, "clips": int(len(ya)),
        "oof_acc_a": float(ca.mean()), "oof_acc_b": float(cb.mean()),
        "oof_diff": float(cb.mean() - ca.mean()),
        "oof_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "oof_mcnemar_p": mcnemar_p(ca, cb),
        "test_single_diff_mean": float(np.mean(fold_diff)),
        "test_single_diff_std": float(np.std(fold_diff)),
        "test_ensemble_diff": float(eb.mean() - ea.mean()),
        "test_ensemble_mcnemar_p": mcnemar_p(ea, eb),
    }


def format_compare(r: dict) -> str:
    lo, hi = r["oof_diff_ci95"]
    return (f"{r['b']}  vs  {r['a']}  ({r['variant']})\n"
            f"  OOF  ({r['clips']} clips): {100 * r['oof_acc_a']:.2f} -> "
            f"{100 * r['oof_acc_b']:.2f}  diff {100 * r['oof_diff']:+.2f} "
            f"[{100 * lo:+.2f}, {100 * hi:+.2f}]  McNemar p = {r['oof_mcnemar_p']:.4f}\n"
            f"  test (single models): diff {100 * r['test_single_diff_mean']:+.2f} "
            f"± {100 * r['test_single_diff_std']:.2f}; ensemble diff "
            f"{100 * r['test_ensemble_diff']:+.2f}, McNemar p = "
            f"{r['test_ensemble_mcnemar_p']:.4f}")
