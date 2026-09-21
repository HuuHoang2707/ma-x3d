"""One person box per stored clip, used to normalise the scale of the actors.

The error analysis shows that the clips the model misses hold much smaller people than
the ones it gets right (largest box 4.0% of the frame against 7.5%), because the crop
made during preprocessing often keeps the whole scene. This module detects people on a
few frames of the stored clip and saves the union box, so training and evaluation can
zoom into the actors. The box is constant over the clip, which keeps the background
registered and the frame difference meaningful.

Writes {root}/person_box_{split}.npy, one row per clip: x1, y1, x2, y2 in pixels
(the full frame when no person is found).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .sampling import uniform_indices


def motion_box(x: np.ndarray, frames: int = 16, quantile: float = 0.95,
               spread: float = 0.1) -> np.ndarray:
    """Region that holds the bulk of the movement of the stored clip.

    The motion map is smoothed, the strongest `1 - quantile` of pixels are kept, and the
    box spans their central range (the `spread` to `1 - spread` quantiles of the
    coordinates), so scattered noise does not open the box to the whole frame.
    """
    import torch
    import torch.nn.functional as F

    from ..models.motion_attention import motion_map

    size = x.shape[-1]
    t = uniform_indices(x.shape[1], frames)
    clip = torch.from_numpy(np.ascontiguousarray(x[t])).permute(1, 0, 2, 3)[None].float() / 255
    m = motion_map(clip)[0, 0].mean(0)[None, None]           # [1, 1, H, W]
    m = F.avg_pool2d(m, 15, stride=1, padding=7)[0, 0]       # smooth out speckle
    ys, xs = torch.nonzero(m >= torch.quantile(m.flatten(), quantile), as_tuple=True)
    if len(ys) < 16:
        return np.array([0, 0, size, size], dtype=np.float64)
    q = torch.tensor([spread, 1 - spread])
    y1, y2 = torch.quantile(ys.float(), q)
    x1, x2 = torch.quantile(xs.float(), q)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def clip_boxes(x: np.ndarray, detector, frames: int = 8, margin: float = 0.25,
               min_side: int = 96, conf: float = 0.25, mode: str = "largest") -> np.ndarray:
    """A single crop box for the clip: the largest person, the union of people, or the
    strongest-moving region ("motion", no detector)."""
    size = x.shape[-1]
    if mode == "motion":
        x1, y1, x2, y2 = motion_box(x)
    else:
        t = uniform_indices(x.shape[1], frames)
        boxes = []
        for res in detector([x[f].transpose(1, 2, 0) for f in t], classes=[0], verbose=False,
                            conf=conf):
            if res.boxes is not None and len(res.boxes):
                boxes.append(res.boxes.xyxy.cpu().numpy())
        if not boxes:
            return np.array([0, 0, size, size], dtype=np.int32)
        b = np.concatenate(boxes)
        if mode == "largest":  # the biggest person, which sets the scale
            areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b[areas.argmax()][None]
        x1, y1 = b[:, 0].min(), b[:, 1].min()
        x2, y2 = b[:, 2].max(), b[:, 3].max()
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    # keep a minimum size so a single distant person is not blown up beyond recognition
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max((x2 - x1) / 2, (y2 - y1) / 2, min_side / 2)
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    shift_x = max(0, -x1) - max(0, x2 - size)
    shift_y = max(0, -y1) - max(0, y2 - size)
    box = np.array([x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y])
    return np.clip(box, 0, size).astype(np.int32)


def build(root: str | Path, split: str, device: str = "cuda",
          mode: str = "largest") -> Path:
    from ultralytics import YOLO

    from .dataset import array_paths

    detector = YOLO("yolov8n.pt")
    detector.to(device)
    xp, _ = array_paths(root, split)
    x = np.load(xp, mmap_mode="r")
    out = np.stack([clip_boxes(x[i], detector, mode=mode)
                    for i in tqdm(range(len(x)), desc=f"boxes {split}")])
    suffix = "" if mode == "largest" else f"_{mode}"
    path = Path(root) / f"person_box{suffix}_{split}.npy"
    np.save(path, out)
    side = (out[:, 2] - out[:, 0]) / x.shape[-1]
    print(f"{split}: median crop side {np.median(side):.2f} of the frame, "
          f"{(side > 0.99).mean() * 100:.0f}% keep the full frame")
    return path
