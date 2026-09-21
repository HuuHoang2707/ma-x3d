"""Edge deployment measurements: CPU latency, model size, streams per device.

Every model is exported to ONNX and timed with ONNX Runtime at several thread budgets,
which is what an edge CPU offers. A five-second clip must be processed in less than five
seconds for one real-time stream, so streams-per-core follows directly from the latency.

Usage: .venv/bin/python paper/drafts/edge_benchmark.py [threads ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MODELS = [  # label -> a finished run and the frames it takes
    ("X3D-XS (4x160)", "b1_x3d_xs/seed0/fold0", 4),
    ("X3D-S (13x160)", "b2_x3d_s/seed0/fold0", 13),
    ("X3D-M", "b3_x3d_m/seed0/fold0", 16),
    ("MA-X3D", "e05_lr_backbone5e-5/seed0/fold0", 16),
    ("X3D-M + diff residual", "n1_diff_residual/seed0/fold0", 16),
    ("VideoMAE-B (teacher)", "t01_videomae/seed0/fold0", 16),
]
CLIP_SECONDS = 5.0


def export(run: Path, out: Path, frames: int) -> Path:
    import torch

    from ma_x3d.cli import _load_run
    from ma_x3d.models.wide_kernel import fuse_wide_kernels

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    _, model = _load_run(run, torch.device("cpu"))
    fuse_wide_kernels(model)
    model.eval()
    dummy = torch.rand(2, 3, frames, 224, 224)
    torch.onnx.export(model, (dummy,), str(out), input_names=["clip"],
                      output_names=["logits"], dynamo=True,
                      dynamic_shapes={"clip": {0: torch.export.Dim.DYNAMIC}})
    return out


def latency(path: Path, frames: int, threads: int, iters: int = 20) -> float:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, frames, 224, 224).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"clip": x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"clip": x})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def main() -> None:
    threads = [int(t) for t in sys.argv[1:]] or [1, 2, 4, 8]
    rows = []
    for label, run, frames in MODELS:
        path = export(ROOT / "runs" / run, ROOT / "exports/edge" / f"{run.split('/')[0]}.onnx",
                      frames)
        size = path.stat().st_size / 2**20
        lat = {t: latency(path, frames, t) for t in threads}
        rows.append({"model": label, "onnx_mb": size, "latency_ms": lat,
                     "streams_per_core": CLIP_SECONDS * 1000 / lat[1]})
        print(f"{label:24s} {size:6.1f} MB | " +
              " ".join(f"{t}t {lat[t]:7.0f} ms" for t in threads) +
              f" | {rows[-1]['streams_per_core']:5.1f} streams/core", flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/edge_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
