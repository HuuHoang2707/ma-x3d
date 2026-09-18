"""Qualitative maps: the Motion Attention gate and Grad-CAM for the Fight score."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.ma_x3d import MAX3D, STAGES


def _upsample(m: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(m[:, None], size=size, mode="trilinear", align_corners=False)[:, 0]


@torch.no_grad()
def gate_map(model: MAX3D, clip: torch.Tensor) -> torch.Tensor:
    """Channel-mean gate g in [-1, 1], upsampled to [B, T, H, W]."""
    feats = {}
    hook = model.blocks[model.ma_after].register_forward_hook(
        lambda m, i, o: feats.__setitem__("f", o))
    model.eval()
    model(clip)
    hook.remove()
    g = model.motion_attn.gain(feats["f"], model.motion(clip)).mean(1)
    return _upsample(g, clip.shape[2:])


def grad_cam(model: MAX3D, clip: torch.Tensor, target: int = 1, stage: str = "res5"):
    """Grad-CAM on the output of `stage`, normalised to [0, 1] per clip: [B, T, H, W]."""
    model.eval()
    acts = {}
    hook = model.blocks[STAGES[stage]].register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o))
    with torch.enable_grad():
        clip = clip.clone().requires_grad_(True)
        score = model(clip)[:, target].sum()
        a = acts["a"]
        grad = torch.autograd.grad(score, a)[0]
    hook.remove()
    cam = F.relu((grad.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1)).detach()
    cam = _upsample(cam, clip.shape[2:])
    flat = cam.flatten(1)
    lo, hi = flat.min(1)[0].view(-1, 1, 1, 1), flat.max(1)[0].view(-1, 1, 1, 1)
    return (cam - lo) / (hi - lo + 1e-8)


def save_figure(clip: torch.Tensor, rows: dict[str, torch.Tensor], path: str,
                frames: int = 5, titles: list[str] | None = None) -> None:
    """One column per sampled frame; first row RGB, then one row per map (single clip)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = clip.shape[1]
    idx = np.linspace(0, t - 1, frames).astype(int)
    rgb = clip.permute(1, 2, 3, 0).cpu().numpy()
    fig, axes = plt.subplots(1 + len(rows), frames, figsize=(2.2 * frames, 2.2 * (1 + len(rows))),
                             squeeze=False)
    for j, k in enumerate(idx):
        axes[0, j].imshow(rgb[k])
        axes[0, j].set_title(f"t={k}", fontsize=8)
        for i, (name, m) in enumerate(rows.items(), start=1):
            m = m.cpu().numpy()
            signed = m.min() < 0
            axes[i, j].imshow(rgb[k])
            axes[i, j].imshow(m[k], cmap="coolwarm" if signed else "jet", alpha=0.5,
                              vmin=-np.abs(m).max() if signed else 0,
                              vmax=np.abs(m).max() if signed else 1)
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    if titles:
        fig.suptitle(" | ".join(titles), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
