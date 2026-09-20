"""Build the data figures of the paper into paper/figures/.

  fig1_cost.pdf       test accuracy against GFLOPs for every model we trained
  fig3_effects.pdf    effect sizes with 95% intervals (modules, recipe, protocol)
  fig4_selection.pdf  test accuracy per epoch, validation-chosen vs best test epoch

Usage: .venv/bin/python paper/drafts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper/drafts"))

from make_tables import paired, summary  # noqa: E402

OUT = ROOT / "paper/figures"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# model -> (GFLOPs, run name, label, marker)
COST = [
    (0.60, "b1_x3d_xs", "X3D-XS", "o"),
    (1.96, "b2_x3d_s", "X3D-S", "o"),
    (4.73, "b3_x3d_m", "X3D-M", "o"),
    (17.98, "b4_s3d", "S3D", "s"),
    (43.34, "b5_mc3_18", "MC3-18", "s"),
    (40.52, "b6_r2plus1d_18", "R(2+1)D-18", "s"),
    (5.46, "e05_lr_backbone5e-5", "MA-X3D", "^"),
    (4.73, "p0_x3d_m", "X3D-M, final", "D"),
    (5.46, "e11_kd_long", "MA-X3D, final", "D"),
    (135.0, "t01_videomae", "VideoMAE-B", "*"),
]


def fig_cost() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for gflops, run, label, marker in COST:
        s = summary(run)
        if s is None:
            continue
        acc, err = 100 * s["test_accuracy"], 100 * s["test_acc_std"]
        ax.errorbar(gflops, acc, yerr=err, fmt=marker, ms=5, capsize=2, color="#1f4e79")
        dx = 1.12 if label != "VideoMAE-B" else 0.45
        ax.annotate(label, (gflops * dx, acc), fontsize=6.5, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs per view (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlim(0.35, 400)
    fig.savefig(OUT / "fig1_cost.pdf")
    plt.close(fig)


def fig_effects() -> None:
    rows = [
        ("modules, RWF-2000 (final, 3 seeds)", paired("p0_x3d_m", "e11_kd_long")),
        ("modules, RWF-2000 (conventional)", paired("q0_x3d_m_conventional", "e00_baseline")),
        ("modules, Hockey (3 seeds)", paired("hk_b3_x3d_m", "hk_e05_ma_x3d")),
        ("modules, RLVS", paired("rl_p0_x3d_m_kd", "rl_e11_ma_x3d_kd")),
        ("modules, 25% of the data", paired("d25_x3d_m", "d25_ma_x3d")),
        ("modules, 50% of the data", paired("d50_x3d_m", "d50_ma_x3d")),
        ("Motion Attention alone", paired("p0_x3d_m", "p1_ma")),
        ("wide kernel alone", paired("p0_x3d_m", "p2_wk")),
        ("backbone lr $10^{-5}\\rightarrow 5\\times10^{-5}$",
         paired("e00_baseline", "e05_lr_backbone5e-5")),
        ("24 $\\rightarrow$ 48 epochs", paired("e05_lr_backbone5e-5", "e19_long_no_kd")),
        ("distillation (same schedule)", paired("e19_long_no_kd", "e11_kd_long")),
    ]
    rows = [(n, r) for n, r in rows if r]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(y, rows, strict=True):
        colour = "#1f4e79" if r["lo"] > 0 or r["hi"] < 0 else "#888888"
        ax.plot([100 * r["lo"], 100 * r["hi"]], [yi, yi], color=colour, lw=1.4)
        ax.plot(100 * r["diff"], yi, "o", ms=4, color=colour)
    # the two effects that are not part of any model
    bias = selection_bias_mean()
    ax.plot(bias, -1, "s", ms=4, color="#a33")
    ax.plot([0.38, 1.38], [-1, -1], color="#a33", lw=1.4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y) + [-1])
    ax.set_yticklabels([n for n, _ in rows] + ["epoch chosen on test"], fontsize=6.5)
    ax.set_xlabel("difference in out-of-fold accuracy (points)")
    fig.savefig(OUT / "fig3_effects.pdf")
    plt.close(fig)


def selection_bias_mean() -> float:
    from ma_x3d.eval.cv import selection_bias

    runs = [ROOT / "runs" / n / "seed0" for n in
            ("b1_x3d_xs", "b2_x3d_s", "b3_x3d_m", "b4_s3d", "b5_mc3_18", "p0_x3d_m")]
    rows = selection_bias([r for r in runs if r.exists()])
    return 100 * float(np.mean([r["gap"] for r in rows]))


def fig_selection() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, fold in enumerate(sorted((ROOT / "runs/p0_x3d_m/seed0").glob("fold*"))):
        ep = [json.loads(line) for line in (fold / "metrics.jsonl").read_text().splitlines()]
        ep = [e for e in ep if "test_acc" in e]
        if not ep:
            continue
        x = [e["epoch"] for e in ep]
        acc = [100 * e["test_acc"] for e in ep]
        ax.plot(x, acc, lw=1, alpha=0.8, label=f"fold {k}")
        best_val = max(e["val_f1"] for e in ep)
        chosen = next(e for e in ep if e["val_f1"] == best_val)
        ax.plot(chosen["epoch"], 100 * chosen["test_acc"], "o", ms=5,
                color=ax.lines[-1].get_color())
        top = max(ep, key=lambda e: e["test_acc"])
        ax.plot(top["epoch"], 100 * top["test_acc"], "x", ms=6,
                color=ax.lines[-1].get_color())
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.savefig(OUT / "fig4_selection.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_cost()
    fig_effects()
    fig_selection()
    print("written:", *(p.name for p in sorted(OUT.glob("*.pdf"))))
