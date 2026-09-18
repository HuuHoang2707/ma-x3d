"""Convert the RWF-2000 HDF5 files into memory-mappable uint8 .npy arrays.

The HDF5 files are gzip-compressed with chunks that span 25-50 clips, so reading a
single clip decompresses data for many others (about 4 s per clip on this machine).
Converting once, reading whole chunk rows, makes per-clip reads cheap.

Output layout (labels: NonFight=0, Fight=1; Fight clips first, as in the notebook):
    {out}/train_x.npy  (1600, 64, 3, 224, 224) uint8
    {out}/train_y.npy  (1600,) int64
    {out}/test_x.npy   (400, 64, 3, 224, 224)  uint8   <- RWF-2000 "val" folder
    {out}/test_y.npy   (400,) int64
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

CLASSES = (("Fight", 1), ("NonFight", 0))


def convert(h5_path: str | Path, out_x: Path, out_y: Path) -> dict:
    names = []
    with h5py.File(h5_path, "r", rdcc_nbytes=4 * 1024**3, rdcc_nslots=1_000_003) as f:
        shapes = [f[c].shape for c, _ in CLASSES]
        n = sum(s[0] for s in shapes)
        x = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.uint8, shape=(n, *shapes[0][1:]))
        y = np.empty(n, dtype=np.int64)
        pos = 0
        for cls, label in CLASSES:
            d = f[cls]
            step = max(d.chunks[0] if d.chunks else 1, 16)  # whole chunk rows at a time
            for i in tqdm(range(0, d.shape[0], step), desc=f"{Path(h5_path).name}:{cls}"):
                block = d[i : i + step]
                x[pos : pos + len(block)] = block
                pos += len(block)
            y[pos - d.shape[0] : pos] = label
            if f"{cls}_names" in f:  # only in files built by ma_x3d.data.preprocess
                names += [s.decode() for s in f[f"{cls}_names"][:]]
        x.flush()
        del x
    np.save(out_y, y)
    if names:
        out_x.with_name(out_x.name.replace("_x.npy", "_names.json")).write_text(json.dumps(names))
    return {"source": str(h5_path), "n": int(n), "fight": int(y.sum()), "shape": list(shapes[0])}


def prepare(h5_dir: str | Path, out_dir: str | Path, suffix: str = "",
            overwrite: bool = False) -> None:
    """suffix selects rebuilt variants, e.g. "_none" for rwf2000_train_none.h5."""
    h5_dir, out_dir = Path(h5_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for split, name in (("train", "rwf2000_train"), ("test", "rwf2000_val")):
        out_x, out_y = out_dir / f"{split}_x.npy", out_dir / f"{split}_y.npy"
        if out_x.exists() and out_y.exists() and not overwrite:
            print(f"[skip] {out_x} exists")
            continue
        meta[split] = convert(h5_dir / f"{name}{suffix}.h5", out_x, out_y)
        print(f"[ok] {split}: {meta[split]}")
    meta_path.write_text(json.dumps(meta, indent=2))
