"""Print the numbers of the paper tables from the k-fold results in runs/.

Usage: .venv/bin/python paper/drafts/make_tables.py > paper/drafts/tables.txt

For every experiment and every seed that has cv_predictions.npz (run `ma-x3d cv` first):
  OOF accuracy/F1 (one prediction per training clip, threshold 0.5),
  test accuracy/P/R/F1/AUC of the single fold models (mean and std over folds x seeds).
Paired comparisons: difference of OOF accuracy with a 95% bootstrap interval over clips
(seeds averaged), and the exact McNemar p-value for each common seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ma_x3d.eval.cv import _oof, mcnemar_p, metrics_at  # noqa: E402

RUNS = ROOT / "runs"
VARIANT = "1clip"
TH = 0.5


def seeds(exp: str) -> list[Path]:
    return sorted(d for d in (RUNS / exp).glob("seed*")
                  if len(list(d.glob("fold*/cv_predictions.npz"))) == 4)


def summary(exp: str) -> dict | None:
    ds = seeds(exp)
    if not ds:
        return None
    oof, test = [], []
    for d in ds:
        p, y, t = _oof(d, VARIANT)
        oof.append(metrics_at(y, p, TH))
        test += [metrics_at(ty, tp, TH) for ty, tp in t]
    out = {"seeds": len(ds)}
    for k in ("accuracy", "f1"):
        out[f"oof_{k}"] = float(np.mean([m[k] for m in oof]))
    out["oof_acc_std"] = float(np.std([m["accuracy"] for m in oof]))
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        out[f"test_{k}"] = float(np.mean([m[k] for m in test]))
    out["test_acc_std"] = float(np.std([m["accuracy"] for m in test]))
    return out


def paired(a: str, b: str, n_boot: int = 10000) -> dict | None:
    sa = {d.name: d for d in seeds(a)}
    sb = {d.name: d for d in seeds(b)}
    common = sorted(set(sa) & set(sb))
    if not common:
        return None
    ca, cb, ps = [], [], []
    for s in common:
        pa, y, _ = _oof(sa[s], VARIANT)
        pb, yb, _ = _oof(sb[s], VARIANT)
        assert (y == yb).all()
        ca.append((pa >= TH) == y)
        cb.append((pb >= TH) == y)
        ps.append(mcnemar_p(ca[-1], cb[-1]))
    ca, cb = np.mean(ca, 0), np.mean(cb, 0)  # per-clip accuracy averaged over seeds
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ca), (n_boot, len(ca)))
    diffs = cb[idx].mean(1) - ca[idx].mean(1)
    return {"seeds": len(common), "diff": float(cb.mean() - ca.mean()),
            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
            "p_max": float(max(ps)), "p_seed0": float(ps[0])}


def row(exp: str, label: str) -> str:
    s = summary(exp)
    if s is None:
        return f"{label:44s} (not finished)"
    return (f"{label:44s} seeds {s['seeds']} | OOF {100 * s['oof_accuracy']:.2f} "
            f"(±{100 * s['oof_acc_std']:.2f}) F1 {s['oof_f1']:.3f} | test "
            f"{100 * s['test_accuracy']:.1f} ± {100 * s['test_acc_std']:.1f} "
            f"P {s['test_precision']:.3f} R {s['test_recall']:.3f} F1 {s['test_f1']:.3f} "
            f"AUC {s['test_auc']:.3f}")


def cmp(a: str, b: str, label: str) -> str:
    r = paired(a, b)
    if r is None:
        return f"{label:44s} (not finished)"
    return (f"{label:44s} seeds {r['seeds']} | dOOF {100 * r['diff']:+.2f} "
            f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] McNemar p {r['p_seed0']:.4f} "
            f"(max over seeds {r['p_max']:.4f})")


def main() -> None:
    print("== Main comparison, RWF-2000 (recipe R1 unless stated)")
    for exp, label in [("b1_x3d_xs", "X3D-XS"), ("b2_x3d_s", "X3D-S"), ("b3_x3d_m", "X3D-M"),
                       ("b4_s3d", "S3D"), ("b5_mc3_18", "MC3-18"),
                       ("b6_r2plus1d_18", "R(2+1)D-18"), ("e05_lr_backbone5e-5", "MA-X3D (R1)"),
                       ("p0_x3d_m", "X3D-M, final recipe"), ("e11_kd_long", "MA-X3D, final"),
                       ("t01_videomae", "VideoMAE-B teacher")]:
        print(row(exp, label))
    print("\n== 2x2 network x recipe")
    for exp, label in [("q0_x3d_m_conventional", "X3D-M conventional"),
                       ("e00_baseline", "MA-X3D conventional"),
                       ("p0_x3d_m", "X3D-M final"), ("e11_kd_long", "MA-X3D final")]:
        print(row(exp, label))
    print(cmp("q0_x3d_m_conventional", "e00_baseline", "modules, conventional recipe"))
    print(cmp("p0_x3d_m", "e11_kd_long", "modules, final recipe"))
    print("\n== Components and controls (final recipe), each against X3D-M")
    for exp, label in [("p1_ma", "+ MA"), ("p6_ma_no_motion", "+ MA gate, zero motion"),
                       ("p2_wk", "+ WK"), ("p4_wk_dense", "+ WK dense 5x5"),
                       ("p5_ring_lr_backbone", "+ WK ring at backbone lr"),
                       ("e11_kd_long", "+ MA + WK")]:
        print(row(exp, label))
        print(cmp("p0_x3d_m", exp, "   vs X3D-M"))
    print(cmp("p6_ma_no_motion", "p1_ma", "MA vs zero-motion control (needs p1 with MA only)"))
    print("\n== Training recipe (MA-X3D)")
    for exp, label in [("e00_baseline", "R0 conventional"), ("e05_lr_backbone5e-5", "R1 lr 5e-5"),
                       ("e19_long_no_kd", "R1' 48 ep, no teacher"),
                       ("e09_kd_lr5e-5", "R2 distillation 24 ep"),
                       ("e11_kd_long", "R3 distillation 48 ep")]:
        print(row(exp, label))
    print(cmp("e00_baseline", "e05_lr_backbone5e-5", "R1 vs R0"))
    print(cmp("e05_lr_backbone5e-5", "e11_kd_long", "R3 vs R1"))
    print(cmp("e19_long_no_kd", "e11_kd_long", "R3 vs R1' (effect of the teacher)"))
    print("\n== Motion Attention design (vs MA-X3D)")
    for exp, label in [("p7_ma_single_scale", "step 1 only"),
                       ("p8_ma_raw_map", "no temporal modes"),
                       ("p9_ma_after_res3", "after Res3"), ("p10_ma_after_res5", "after Res5")]:
        print(row(exp, label))
        print(cmp(exp, "e11_kd_long", "   MA-X3D vs this"))
    for ds, pre in (("Hockey", "hk"), ("RLVS", "rl")):
        print(f"\n== {ds}")
        for exp, label in [(f"{pre}_b3_x3d_m", "X3D-M R1"), (f"{pre}_e05_ma_x3d", "MA-X3D R1"),
                           (f"{pre}_p0_x3d_m_kd", "X3D-M final"),
                           (f"{pre}_e11_ma_x3d_kd", "MA-X3D final"),
                           (f"{pre}_t01_videomae", "VideoMAE-B teacher")]:
            print(row(exp, label))
        print(cmp(f"{pre}_b3_x3d_m", f"{pre}_e05_ma_x3d", "modules, R1"))
        print(cmp(f"{pre}_p0_x3d_m_kd", f"{pre}_e11_ma_x3d_kd", "modules, final"))


if __name__ == "__main__":
    main()
