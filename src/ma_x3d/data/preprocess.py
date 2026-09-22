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
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    if mode == "adaptive":
        # Zooming all the way onto the actors removes the context the network needs
        # (measured: a tight crop costs 1.8 points). Keep at least `min_frac` of the
        # frame and cap the magnification at `max_zoom`.
        min_frac, max_zoom = 0.5, 2.0
        side = max(x2 - x1, y2 - y1, min_frac * min(h, w), min(h, w) / max_zoom)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = int(x1 + side), int(y1 + side)
        x1, x2 = (0, min(w, int(side))) if x1 < 0 else ((max(0, w - int(side)), w)
                                                        if x2 > w else (x1, x2))
        y1, y2 = (0, min(h, int(side))) if y1 < 0 else ((max(0, h - int(side)), h)
                                                        if y2 > h else (y1, y2))
    return x1, y1, x2, y2


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


_DETECTOR = None


def _clip_array(args) -> np.ndarray | None:
    """One video -> [n, 3, size, size] uint8 (worker process, CPU detector)."""
    import cv2
    import torch

    global _DETECTOR
    path, roi, n, size, window_s, enhance_frames = args
    torch.set_num_threads(2)  # NMS has a time limit; one thread can hit it under load
    cv2.setNumThreads(1)
    if _DETECTOR is None:
        from ultralytics import YOLO

        _DETECTOR = YOLO("yolov8n.pt")
    frames = extract_frames(path, n, window_s)
    if frames is None:
        return None
    x1, y1, x2, y2 = person_roi(frames, _DETECTOR, mode=roi)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance_frames else None
    out = []
    for f in frames:
        crop = f[y1:y2, x1:x2]
        crop = f if crop.size == 0 else crop
        if clahe is not None:
            crop = enhance(crop, clahe)
        out.append(cv2.resize(crop, (size, size)).transpose(2, 0, 1))
    return np.stack(out)


def build_clip_arrays(class_dirs: dict[int, str | Path], out_dir: str | Path,
                      roi: str = "cluster", workers: int = 16, n: int = 64, size: int = 224,
                      window_s: float | None = 5.0, enhance_frames: bool = True) -> None:
    """Videos in {label: folder} -> out_dir/all_x.npy [N, n, 3, size, size], all_y.npy,
    names.txt, bad.npy (videos that could not be read). Same processing as build_split,
    run in parallel on CPU."""
    from multiprocessing import get_context

    from ultralytics import YOLO

    YOLO("yolov8n.pt")  # download once before the workers start
    videos = [(str(p), label) for label, d in sorted(class_dirs.items(), reverse=True)
              for p in sorted(Path(d).iterdir()) if p.suffix.lower() in (".mp4", ".avi")]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(out_dir / "all_x.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(videos), n, 3, size, size))
    bad = np.zeros(len(videos), dtype=bool)
    jobs = [(p, roi, n, size, window_s, enhance_frames) for p, _ in videos]
    with get_context("spawn").Pool(workers) as pool:
        for i, arr in enumerate(tqdm(pool.imap(_clip_array, jobs, chunksize=2),
                                     total=len(jobs), desc="clips")):
            if arr is None:
                bad[i] = True
            else:
                x[i] = arr
    x.flush()
    np.save(out_dir / "all_y.npy", np.array([label for _, label in videos], dtype=np.int64))
    np.save(out_dir / "bad.npy", bad)
    (out_dir / "names.txt").write_text("\n".join(Path(p).name for p, _ in videos))



def _rwf_clip(args) -> tuple[int, np.ndarray | None]:
    """One video of the official RWF-2000 layout -> [n, 3, size, size] uint8."""
    import cv2
    import torch

    global _DETECTOR
    index, path, roi, n, size, window_s, enhance_frames = args
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if _DETECTOR is None and roi != "none":
        from ultralytics import YOLO

        _DETECTOR = YOLO("yolov8n.pt")
    frames = extract_frames(path, n, window_s)
    if frames is None:
        return index, None
    x1, y1, x2, y2 = person_roi(frames, _DETECTOR, mode=roi)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if enhance_frames else None
    out = []
    for f in frames:
        crop = f[y1:y2, x1:x2]
        crop = f if crop.size == 0 else crop
        if clahe is not None:
            crop = enhance(crop, clahe)
        out.append(cv2.resize(crop, (size, size)).transpose(2, 0, 1))
    return index, np.stack(out)


def build_rwf(videos_root: str | Path, out_dir: str | Path, roi: str = "cluster",
              workers: int = 12, n: int = 64, size: int = 224,
              window_s: float | None = 5.0, enhance_frames: bool = True) -> None:
    """Official RWF-2000 layout (train|val / Fight|NonFight) -> train/test npy arrays.

    One array per split, Fight clips first, which is the order the audit and the folds
    assume. `roi` selects the crop: none, cluster (thesis), union (CUE-Net), adaptive.
    """
    from multiprocessing import get_context

    from ultralytics import YOLO

    if roi != "none":
        YOLO("yolov8n.pt")  # fetch the weights once, before the workers start
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, folder in (("train", "train"), ("test", "val")):
        videos, labels = [], []
        for label, cls in ((1, "Fight"), (0, "NonFight")):
            d = Path(videos_root) / folder / cls
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".avi", ".mp4"):
                    videos.append(str(f))
                    labels.append(label)
        x = np.lib.format.open_memmap(out_dir / f"{split}_x.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(videos), n, 3, size, size))
        jobs = [(i, v, roi, n, size, window_s, enhance_frames) for i, v in enumerate(videos)]
        bad = []
        with get_context("spawn").Pool(workers) as pool:
            for i, arr in tqdm(pool.imap_unordered(_rwf_clip, jobs, chunksize=2),
                               total=len(jobs), desc=f"{split} ({roi})"):
                if arr is None:
                    bad.append(i)
                else:
                    x[i] = arr
        x.flush()
        np.save(out_dir / f"{split}_y.npy", np.array(labels, dtype=np.int64))
        (out_dir / f"{split}_names.txt").write_text("\n".join(Path(v).name for v in videos))
        print(f"{split}: {len(videos)} clips ({sum(labels)} fight), {len(bad)} unreadable")
