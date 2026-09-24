"""Schematic of the reparameterised wide kernel: pre-trained 3x3 + zero-init ring = 5x5."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, MUTED, EMPTY = "#0b0b0b", "#52514e", "#f1f0ec"

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6.2, 2.0))
ax.set_xlim(0, 31)
ax.set_ylim(-1.8, 5.6)
ax.set_aspect("equal")
ax.axis("off")


def grid(x0, fill):
    for i in range(5):
        for j in range(5):
            ring = i in (0, 4) or j in (0, 4)
            c = fill(ring)
            ax.add_patch(mp.Rectangle((x0 + j, i), 0.92, 0.92, facecolor=c, edgecolor="white",
                                      linewidth=1.2))


grid(0, lambda ring: EMPTY if ring else BLUE)                 # padded pre-trained kernel
grid(8.5, lambda ring: ORANGE if ring else EMPTY)             # ring, zero at the start
grid(19, lambda ring: ORANGE if ring else BLUE)               # fused kernel
ax.text(6.7, 2.4, "+", fontsize=16, ha="center", va="center", color=INK)
ax.text(16.1, 3.0, "fuse", fontsize=9, ha="center", va="bottom", color=INK)
ax.annotate("", xy=(18.3, 2.4), xytext=(13.9, 2.4),
            arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.0))
kw = dict(ha="center", va="top", fontsize=8, color=MUTED)
ax.text(2.45, -0.3, "pre-trained 3×3,\nzero-padded", **kw)
ax.text(10.95, -0.3, "ring Δ, zero-initialised,\nown learning rate", **kw)
ax.text(21.45, -0.3, "one 5×5 kernel\nat inference", **kw)
ax.text(27.3, 2.4, "applied to every\nblock of Res2–Res5;\ntrains even while\nRes4 is frozen",
        ha="left", va="center", fontsize=8, color=INK)
ax.set_xlim(-0.5, 33)
fig.tight_layout()
fig.savefig("paper/soict2026/fig_kernel.pdf", bbox_inches="tight")
fig.savefig("paper/soict2026/fig_kernel.png", dpi=200, bbox_inches="tight")
