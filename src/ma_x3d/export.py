"""Export a trained model for edge inference (ONNX), check it, time it.

The exported graph takes clips in [0, 1] ([B, 3, T, 224, 224]) and returns logits;
the motion map, the input normalisation and the fused 5x5 kernels are inside the
graph. ONNX Runtime folds BatchNorm into the convolutions when it loads the model.
Needs the `edge` extra (onnx, onnxscript, onnxruntime).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models.wide_kernel import fuse_wide_kernels


def to_onnx(model: nn.Module, path: Path, frames: int = 16, size: int = 224,
            opset: int = 18) -> Path:
    model = copy.deepcopy(model).cpu().eval()
    fuse_wide_kernels(model)
    dummy = torch.rand(1, 3, frames, size, size)
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(model, (dummy,), str(path), input_names=["clip"], output_names=["logits"],
                      opset_version=opset, dynamo=True, dynamic_shapes=({0: batch},))
    return path


def ort_session(path: Path, threads: int = 0):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_latency(sess, frames: int = 16, batch: int = 1, iters: int = 20) -> float:
    x = np.random.rand(batch, 3, frames, 224, 224).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"clip": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"clip": x})
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def quantize_int8(fp32_path: Path, out_path: Path, clips: list[np.ndarray]) -> Path:
    """Static INT8 (QDQ) quantisation calibrated on a few real clips."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(clips)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"clip": x[None].astype(np.float32)}

    pre = out_path.with_name(out_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    quantize_static(str(pre), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True)
    pre.unlink(missing_ok=True)
    return out_path


def ort_evaluate(sess, dataset, limit: int | None = None) -> dict:
    from .eval.metrics import classification_metrics

    n = len(dataset) if limit is None else min(limit, len(dataset))
    probs, labels = [], []
    for i in range(n):
        clip, y = dataset[i]
        logits = sess.run(None, {"clip": clip[None].numpy()})[0][0]
        e = np.exp(logits - logits.max())
        probs.append(e / e.sum())
        labels.append(y)
    probs = np.array(probs)
    return classification_metrics(labels, probs.argmax(1), probs[:, 1])


def export_run(model: nn.Module, cfg, out_dir: Path, int8: bool = False,
               eval_limit: int | None = None, threads: int = 0) -> dict:
    from .data.dataset import ClipDataset, build_datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = cfg.data.frames
    fp32 = to_onnx(model, out_dir / "model_fp32.onnx", frames)
    ds = build_datasets(cfg)
    report = {"frames": frames, "onnx_fp32": str(fp32)}

    # numerical check against PyTorch on one real clip
    sess = ort_session(fp32, threads)
    clip, _ = ds["test"][0]
    with torch.no_grad():
        ref = model.cpu().eval()(clip[None]).numpy()
    got = sess.run(None, {"clip": clip[None].numpy()})[0]
    report["max_abs_logit_diff"] = float(np.abs(ref - got).max())
    report["fp32_latency_ms"] = 1e3 * ort_latency(sess, frames)
    report["fp32_test"] = ort_evaluate(sess, ds["test"], eval_limit)

    if int8:
        # calibrate on un-augmented training clips, never on the test split
        tr = ds["train"]
        plain = ClipDataset(tr.x_path, tr.labels, tr.indices, frames, training=False)
        pick = np.linspace(0, len(plain) - 1, 32).astype(int)
        calib = [plain[int(i)][0].numpy() for i in pick]
        q = quantize_int8(fp32, out_dir / "model_int8.onnx", calib)
        qs = ort_session(q, threads)
        report["onnx_int8"] = str(q)
        report["int8_latency_ms"] = 1e3 * ort_latency(qs, frames)
        report["int8_test"] = ort_evaluate(qs, ds["test"], eval_limit)
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    return report
