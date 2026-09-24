"""Test accuracy against computation, uncropped pipeline, one figure for the paper.

Single-model test accuracy is the mean of the four fold models (seed 0); the final
model is the fold soup, mean of three seeds. GFLOPs per 224x224 view, fused kernels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# (GFLOPs, test acc, label, family, label offset in points)
POINTS = [
    (4.73, 88.37, "X3D-M", "other", (6, -3)),
    (5.46, 89.13, "MA-X3D", "other", (6, -10)),
    (10.94, 90.56, "32 frames", "other", (-4, 7)),
    (5.45, 89.44, "5×5 Res2–3", "wk", (6, 2)),
    (5.20, 90.75, "5×5 Res4–5", "wk", (6, -7)),
    (7.15, 90.06, "5×5×5", "wk", (6, -3)),
    (7.70, 90.31, "7×7 dilated", "wk", (6, -3)),
    (5.92, 91.25, "5×5 Res2–5", "wk", (7, -4)),
]
FINAL = (5.92, 91.58, "5×5 Res2–5, fold soup (final)")

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for fam, color, name in (("other", ORANGE, "other models"), ("wk", BLUE, "wide kernels")):
    pts = [p for p in POINTS if p[3] == fam]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=color, label=name,
               edgecolor="white", linewidth=1.2, zorder=3)
for x, y, label, _, off in POINTS:
    ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=8,
                color=MUTED, va="center")
ax.scatter([FINAL[0]], [FINAL[1]], s=90, marker="D", color=BLUE, edgecolor=INK,
           linewidth=0.8, zorder=4, label="final model")
ax.annotate(FINAL[2], (FINAL[0], FINAL[1]), xytext=(9, 3), textcoords="offset points",
            fontsize=8, color=INK, va="center")
ax.set_xlabel("GFLOPs per view")
ax.set_ylabel("RWF-2000 test accuracy (%)")
ax.set_xlim(4.2, 12.2)
ax.set_ylim(87.8, 92.2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("paper/soict2026/fig_cost.pdf")
fig.savefig("paper/soict2026/fig_cost.png", dpi=200)
