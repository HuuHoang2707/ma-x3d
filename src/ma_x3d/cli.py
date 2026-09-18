"""Command-line interface: `ma-x3d <command> --help` for details."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _device(name: str | None):
    import torch

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_run(run: Path, device):
    """Model from a run directory (best.pt) with its config."""
    import torch

    from .config import load_config
    from .models import build_model

    cfg = load_config(run / "config.yaml")
    cfg.model.pretrained = False
    model = build_model(cfg.model)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    if device.type == "cuda" and cfg.train.fast_depthwise:
        from .ops.depthwise3d import use_fast_depthwise

        use_fast_depthwise(model)
    return cfg, model.to(device).eval()


def cmd_prepare(a):
    from .data.prepare import prepare

    prepare(a.h5_dir, a.out, a.suffix, a.overwrite)


def cmd_build_dataset(a):
    from .data.preprocess import build_dataset

    build_dataset(a.videos, a.out, a.roi, a.device or "cuda")


def cmd_train(a):
    from .config import load_config
    from .train.engine import train

    overrides = list(a.overrides) + ([f"seed={a.seed}"] if a.seed is not None else [])
    cfg = load_config(a.config, overrides)
    train(cfg, _device(a.device), resume=a.resume, overwrite=a.overwrite)


def cmd_eval(a):
    from .data.dataset import build_datasets, make_loader
    from .eval.metrics import format_report
    from .eval.predict import predict
    from .models.wide_kernel import ring_disabled
    from .train.utils import amp_dtype

    device = _device(a.device)
    cfg, model = _load_run(Path(a.run), device)
    if a.fuse:
        from .models.wide_kernel import fuse_wide_kernels

        fuse_wide_kernels(model)
    ds = build_datasets(cfg)[a.split]
    loader = make_loader(ds, cfg.train.batch_size, False, cfg.data.num_workers)
    amp = amp_dtype(cfg.train.amp)
    if a.ring_off:
        with ring_disabled(model):
            out = predict(model, loader, device, amp)
    else:
        out = predict(model, loader, device, amp)
    print(format_report(out["metrics"]))
    if a.out:
        Path(a.out).write_text(json.dumps(out["metrics"], indent=2))


def cmd_bench(a):
    import torch

    from .config import load_config
    from .eval.benchmark import benchmark
    from .models import build_model
    from .train.utils import amp_dtype

    device = _device(a.device)
    if a.run:
        cfg, model = _load_run(Path(a.run), device)
    else:
        cfg = load_config(a.config, a.overrides)
        cfg.model.pretrained = False
        model = build_model(cfg.model).to(device)
    if a.triton:
        from .ops.depthwise3d import use_fast_depthwise

        use_fast_depthwise(model)
    torch.backends.cudnn.benchmark = True
    res = benchmark(model, device, a.frames or cfg.data.frames, 224, a.batch, a.iters,
                    amp_dtype(a.amp))
    res["config"] = cfg.name
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


def cmd_report(a):
    from .report import write_report

    print(write_report(a.runs, a.out))
    print(f"\nwritten to {a.out}/")


def cmd_visualize(a):
    import torch

    from .data.dataset import build_datasets
    from .eval.visualize import gate_map, grad_cam, save_figure

    device = _device(a.device)
    cfg, model = _load_run(Path(a.run), device)
    ds = build_datasets(cfg)[a.split]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in a.index:
        clip, label = ds[i]
        x = clip[None].to(device)
        rows = {}
        if model.motion_attn is not None:
            rows["gate"] = gate_map(model, x)[0]
        rows["Grad-CAM"] = grad_cam(model, x)[0]
        with torch.no_grad():
            p = model(x).softmax(1)[0, 1].item()
        path = out / f"{a.split}_{i:03d}.png"
        save_figure(clip, rows, str(path), titles=[f"{a.split}[{i}] label={label}",
                                                   f"p(Fight)={p:.2f}"])
        print(path)


def cmd_export(a):
    from .export import export_run

    cfg, model = _load_run(Path(a.run), _device("cpu"))
    rep = export_run(model, cfg, Path(a.out), a.int8, a.eval_limit, a.threads)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_test")}, indent=2))
    for k in ("fp32_test", "int8_test"):
        if k in rep:
            print(f"{k}: acc {rep[k]['accuracy']:.4f} f1 {rep[k]['f1']:.4f} (n={rep[k]['n']})")


def cmd_sweep(a):
    """Run every (config, seed) pair, one job per GPU at a time."""
    jobs = [(c, s) for c in a.configs for s in a.seeds]
    free, running = list(a.gpus), []
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs on GPUs {a.gpus}")
    while jobs or running:
        while jobs and free:
            cfg, seed = jobs.pop(0)
            gpu = free.pop(0)
            env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, "-m", "ma_x3d.cli", "train", cfg, "--seed", str(seed),
                   *(["--resume"] if a.resume else []),
                   *(["--set", *a.overrides] if a.overrides else [])]
            log = open(log_dir / f"{Path(cfg).stem}_seed{seed}.log", "a")
            running.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT),
                            gpu, cfg, seed))
            print(f"[start] gpu {gpu}: {cfg} seed {seed}")
        time.sleep(10)
        for job in list(running):
            proc, gpu, cfg, seed = job
            if proc.poll() is not None:
                running.remove(job)
                free.append(gpu)
                status = "done" if proc.returncode == 0 else "FAIL"
                print(f"[{status}] gpu {gpu}: {cfg} seed {seed}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ma-x3d", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="convert RWF-2000 HDF5 files to .npy arrays")
    s.add_argument("--h5-dir", default="dataset")
    s.add_argument("--out", default="dataset/rwf2000")
    s.add_argument("--suffix", default="", help='e.g. "_none" for full-frame rebuilt files')
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("build-dataset", help="raw RWF-2000 videos -> HDF5 (needs [preprocess])")
    s.add_argument("--videos", required=True, help="folder with train/ and val/")
    s.add_argument("--out", default="dataset")
    s.add_argument("--roi", default="cluster", choices=["cluster", "union", "none"])
    s.add_argument("--device")
    s.set_defaults(fn=cmd_build_dataset)

    s = sub.add_parser("train", help="train one config")
    s.add_argument("config")
    s.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                   metavar="KEY=VALUE", help="dotted overrides, e.g. --set train.epochs=2")
    s.add_argument("--seed", type=int)
    s.add_argument("--device", help="default: cuda if available (use HIP_VISIBLE_DEVICES)")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="evaluate the best checkpoint of a run")
    s.add_argument("run", help="run directory, e.g. runs/ma_x3d/seed0")
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    s.add_argument("--ring-off", action="store_true", help="zero the wide-kernel ring")
    s.add_argument("--fuse", action="store_true", help="fuse wide kernels before evaluating")
    s.add_argument("--device")
    s.add_argument("--out", help="write metrics JSON here")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("bench", help="parameters, GFLOPs, latency, throughput")
    s.add_argument("config", nargs="?", default="configs/ma_x3d.yaml")
    s.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                   metavar="KEY=VALUE")
    s.add_argument("--run", help="benchmark a trained run instead of a config")
    s.add_argument("--frames", type=int)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--iters", type=int, default=50)
    s.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    s.add_argument("--triton", action="store_true", help="Triton depthwise convs (GPU)")
    s.add_argument("--device")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("report", help="tables and curves from finished runs")
    s.add_argument("--runs", default="runs")
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("visualize", help="gate and Grad-CAM figures for some clips")
    s.add_argument("run")
    s.add_argument("--split", default="test", choices=["val", "test"])
    s.add_argument("--index", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--out", default="reports/figures")
    s.add_argument("--device")
    s.set_defaults(fn=cmd_visualize)

    s = sub.add_parser("export", help="ONNX export + check + CPU timing (needs [edge])")
    s.add_argument("run")
    s.add_argument("--out", default="exports/model")
    s.add_argument("--int8", action="store_true", help="also static INT8 quantisation")
    s.add_argument("--eval-limit", type=int, help="evaluate on the first N test clips only")
    s.add_argument("--threads", type=int, default=0, help="ONNX Runtime threads (0 = all)")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("sweep", help="run configs x seeds across GPUs")
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--gpus", type=int, nargs="+", default=[0])
    s.add_argument("--resume", action="store_true")
    s.add_argument("--log-dir", default="runs/logs")
    s.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                   metavar="KEY=VALUE", help="applied to every job")
    s.set_defaults(fn=cmd_sweep)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
