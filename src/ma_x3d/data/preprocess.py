"""Build the HDF5 clip files from raw RWF-2000 videos.

Needs the `preprocess` extra (OpenCV, scikit-learn, ultralytics). Steps per video:
64 frames from a 5 s window around the middle, one static person-centred crop
(YOLOv8n on 12 frames + DBSCAN on box centres), bilateral filter + CLAHE, resize.

`roi="none"` keeps the full frame, which is what the ROI ablation needs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def extract_frames(path: str, n: int = 64, window_s: float | None = 5.0) -> np.ndarray | None:
    """n evenly spaced RGB frames [n, H, W, 3] from a centred window (None = whole video)."""
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        return None
    start, end = 0, total - 1
    if window_s:
        half = int(fps * window_s) // 2
        start, end = max(0, total // 2 - half), min(total - 1, total // 2 + half)
    wanted = np.linspace(start, end, n, dtype=int).tolist()
    keep, store, cur = set(wanted), {}, 0
    while cur <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in keep:
            store[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    frames = [store[i] for i in wanted if i in store]
    if not frames:
        return None
    frames += [frames[-1]] * (n - len(frames))
    return np.stack(frames[:n])


def enhance(frame: np.ndarray, clahe) -> np.ndarray:
    """Bilateral denoising, then CLAHE on the L channel.

    The released HDF5 files were built by passing RGB frames through the BGR<->LAB
    conversions, so CLAHE saw a luminance with the R and B weights swapped. That is
    kept here so rebuilt files match the released ones.
    """
    import cv2

    frame = cv2.bilateralFilter(frame, 5, 50, 50)
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def person_roi(frames: np.ndarray, detector, n_sample: int = 12, margin: float = 0.2,
               mode: str = "cluster") -> tuple[int, int, int, int]:
    """One static crop box (x1, y1, x2, y2) for the whole clip."""
    from sklearn.cluster import DBSCAN

    h, w = frames.shape[1:3]
    if mode == "none":
        return 0, 0, w, h
    sample = [frames[i] for i in np.linspace(0, len(frames) - 1, n_sample, dtype=int)]
    boxes = []
    for res in detector(sample, classes=[0], verbose=False):
        if res.boxes is not None:
            boxes += [tuple(map(int, b.xyxy[0])) for b in res.boxes]
    if not boxes:
        return 0, 0, w, h

    if mode == "union":  # CUE-Net style: all people, full frame if fewer than two
        if len(boxes) < 2:
            return 0, 0, w, h
        chosen = boxes
    elif len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0]
        return max(0, x1 - 50), max(0, y1 - 50), min(w, x2 + 50), min(h, y2 + 50)
    else:  # largest DBSCAN cluster of box centres; lone detections are noise
        centres = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes]
        labels = DBSCAN(eps=int(0.15 * max(w, h)), min_samples=2).fit(centres).labels_
        clusters = [c for c in set(labels) if c != -1]
        if clusters:
            best = max(clusters, key=lambda c: int(np.sum(labels == c)))
            chosen = [b for b, c in zip(boxes, labels, strict=True) if c == best]
        else:
            chosen = boxes

    x1, y1 = min(b[0] for b in chosen), min(b[1] for b in chosen)
    x2, y2 = max(b[2] for b in chosen), max(b[3] for b in chosen)
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def build_split(src_dir: str | Path, out_path: str | Path, detector, n: int = 64,
                size: int = 224, window_s: float | None = 5.0, roi: str = "cluster") -> None:
    """src_dir/{Fight,NonFight}/*.avi -> HDF5 with datasets Fight, NonFight [N, n, 3, S, S]."""
    import cv2
    import h5py

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    with h5py.File(out_path, "a") as hf:
        for cls in ("Fight", "NonFight"):
            cls_dir = Path(src_dir) / cls
            videos = sorted(v for v in os.listdir(cls_dir) if v.endswith((".mp4", ".avi")))
            dset = hf[cls] if cls in hf else hf.create_dataset(
                cls, shape=(len(videos), n, 3, size, size), dtype=np.uint8,
                compression="gzip", compression_opts=4, chunks=(1, n, 3, size, size))
            names = [v.encode() for v in videos]
            if f"{cls}_names" not in hf:
                hf.create_dataset(f"{cls}_names", data=names)
            for i, name in enumerate(tqdm(videos, desc=f"{Path(out_path).name}:{cls}")):
                if dset[i, 0].any():  # already written (resume)
                    continue
                frames = extract_frames(str(cls_dir / name), n, window_s)
                if frames is None:
                    continue
                x1, y1, x2, y2 = person_roi(frames, detector, mode=roi)
                out = []
                for f in frames:
                    crop = f[y1:y2, x1:x2]
                    crop = f if crop.size == 0 else crop
                    out.append(cv2.resize(enhance(crop, clahe), (size, size)).transpose(2, 0, 1))
                dset[i] = np.stack(out)


def build_dataset(videos_root: str | Path, out_dir: str | Path, roi: str = "cluster",
                  device: str = "cuda") -> None:
    from ultralytics import YOLO

    detector = YOLO("yolov8n.pt")
    detector.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if roi == "cluster" else f"_{roi}"
    for split in ("train", "val"):
        build_split(Path(videos_root) / split, out_dir / f"rwf2000_{split}{suffix}.h5", detector,
                    roi=roi)
