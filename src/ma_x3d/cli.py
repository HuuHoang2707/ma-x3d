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


def cmd_audit(a):
    from .data.audit import audit

    rep = audit(a.root)
    short = {k: (v if not isinstance(v, list) or len(v) < 12 else f"{len(v)} items")
             for k, v in rep.items()}
    print(json.dumps(short, indent=2))


def cmd_build_dataset(a):
    from .data.preprocess import build_dataset

    build_dataset(a.videos, a.out, a.roi, a.device or "cuda")


def cmd_boxes(a):
    from .data.person_boxes import build

    for split in ("train", "test"):
        build(a.root, split, a.device or "cuda", a.mode)


def cmd_build_rwf(a):
    from .data.preprocess import build_rwf

    build_rwf(a.videos, a.out, a.roi, a.workers, size=a.size, enhance_frames=not a.no_enhance)


def cmd_profiles(a):
    from .data.person_boxes import build_profiles

    for split in ("train", "test"):
        build_profiles(a.root, split)


def cmd_build_clips(a):
    from .data.audit import grouped_test_split
    from .data.preprocess import build_clip_arrays

    build_clip_arrays({1: a.fight, 0: a.nonfight}, a.all_dir, a.roi, a.workers)
    rep = grouped_test_split(a.all_dir, a.out, a.test_fraction, a.seed)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_index")}, indent=2))


def cmd_train(a):
    from .config import load_config
    from .train.engine import train

    overrides = list(a.overrides) + ([f"seed={a.seed}"] if a.seed is not None else [])
    cfg = load_config(a.config, overrides)
    from .train import dist

    try:
        train(cfg, _device(a.device), resume=a.resume, overwrite=a.overwrite)
    finally:
        dist.cleanup()


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


def cmd_cv(a):
    from .eval.cv import (
        compare,
        evaluate_cv,
        experiment_table,
        format_compare,
        format_cv,
        format_selection_bias,
        selection_bias,
    )

    if a.selection_bias:
        print(format_selection_bias(selection_bias(a.exp)))
        return
    if a.compare:
        base, *others = a.exp
        for other in others:
            print(format_compare(compare(base, other, a.compare)))
        return
    if a.table:
        print(experiment_table(a.exp, a.table))
        return
    for d in a.exp:
        print(format_cv(evaluate_cv(d, _device(a.device), a.variants, a.key)))
        print()


def _smi_status() -> dict[int, dict]:
    """rocm-smi view: {smi_id: {bus, mem, use}}."""
    out = subprocess.run(["rocm-smi", "--showmemuse", "--showuse", "--showbus"],
                         capture_output=True, text=True).stdout
    info: dict[int, dict] = {}
    for line in out.splitlines():
        if not (line.startswith("GPU[") and ":" in line):
            continue
        gpu = int(line[4 : line.index("]")])
        val = line.rsplit(": ", 1)[1].strip()
        d = info.setdefault(gpu, {})
        if "VRAM%" in line and val.isdigit():
            d["mem"] = int(val)
        elif "GPU use (%)" in line and val.isdigit():
            d["use"] = int(val)
        elif "PCI Bus" in line:
            d["bus"] = int(val.split(":")[1], 16)
    return info


def _hip_ids() -> dict[int, int]:
    """smi_id -> HIP id. On this node the two numberings differ (matched by PCI bus)."""
    code = ("import torch; print(*[torch.cuda.get_device_properties(i).pci_bus_id "
            "for i in range(torch.cuda.device_count())])")
    env = {k: v for k, v in os.environ.items()
           if k not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    buses = [int(b) for b in out.stdout.split()]
    by_bus = {bus: hip for hip, bus in enumerate(buses)}
    return {smi: by_bus[d["bus"]] for smi, d in _smi_status().items() if d.get("bus") in by_bus}


def _free_gpus(exclude: set[int], hip_of: dict[int, int], max_mem_pct: int = 5) -> list[int]:
    """HIP ids of GPUs with (almost) no memory allocated by anyone."""
    return [hip_of[smi] for smi, d in sorted(_smi_status().items())
            if smi in hip_of and hip_of[smi] not in exclude and d.get("mem", 100) <= max_mem_pct]


def cmd_gpus(a):
    hip_of = _hip_ids()
    print("HIP_VISIBLE_DEVICES id | rocm-smi id | memory % | use %")
    for smi, d in sorted(_smi_status().items()):
        print(f"{hip_of.get(smi, '?'):>22} | {smi:>11} | {d.get('mem', '?'):>8} | "
              f"{d.get('use', '?'):>5}")


def cmd_sweep(a):
    """Run every (config, seed[, fold]) job, one job per GPU at a time.

    --gpus auto: use any GPU that is idle (other users' jobs are left alone).
    --folds 0 1 2 3: K-fold jobs (sets data.protocol=kfold and data.fold).
    """
    folds = a.folds or [None]
    jobs = [(c, s, f) for c in a.configs for s in a.seeds for f in folds]
    auto = a.gpus == ["auto"]
    free = [] if auto else [int(g) for g in a.gpus]
    hip_of = _hip_ids() if auto else {}
    running = []
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs on GPUs {a.gpus}", flush=True)
    while jobs or running:
        if auto:
            busy = {j[1] for j in running}
            free = _free_gpus(busy, hip_of)
        while jobs and free:
            cfg, seed, fold = jobs.pop(0)
            gpu = free.pop(0)
            env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
            extra = list(a.overrides)
            if fold is not None:
                extra += ["data.protocol=kfold", f"data.fold={fold}"]
            cmd = [sys.executable, "-m", "ma_x3d.cli", "train", cfg, "--seed", str(seed),
                   *(["--resume"] if a.resume else []), *(["--set", *extra] if extra else [])]
            tag = f"{Path(cfg).stem}_seed{seed}" + (f"_fold{fold}" if fold is not None else "")
            log = open(log_dir / f"{tag}.log", "a")
            running.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT),
                            gpu, tag))
            print(f"[start] gpu {gpu}: {tag}", flush=True)
            if auto:
                time.sleep(90)  # let the job allocate memory before counting free GPUs again
                break
        time.sleep(20)
        for job in list(running):
            proc, gpu, tag = job
            if proc.poll() is not None:
                running.remove(job)
                if not auto:
                    free.append(gpu)
                status = "done" if proc.returncode == 0 else "FAIL"
                print(f"[{status}] gpu {gpu}: {tag}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ma-x3d", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="convert RWF-2000 HDF5 files to .npy arrays")
    s.add_argument("--h5-dir", default="dataset")
    s.add_argument("--out", default="dataset/rwf2000")
    s.add_argument("--suffix", default="", help='e.g. "_none" for full-frame rebuilt files')
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("audit", help="duplicates, same-scene groups, clips to exclude")
    s.add_argument("--root", default="dataset/rwf2000")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("build-dataset", help="raw RWF-2000 videos -> HDF5 (needs [preprocess])")
    s.add_argument("--videos", required=True, help="folder with train/ and val/")
    s.add_argument("--out", default="dataset")
    s.add_argument("--roi", default="cluster", choices=["cluster", "union", "none"])
    s.add_argument("--device")
    s.set_defaults(fn=cmd_build_dataset)

    s = sub.add_parser("boxes", help="one person box per stored clip (needs [preprocess])")
    s.add_argument("--root", default="dataset/rwf2000")
    s.add_argument("--mode", default="largest", choices=["largest", "union", "motion"])
    s.add_argument("--device")
    s.set_defaults(fn=cmd_boxes)

    s = sub.add_parser("build-rwf", help="raw RWF-2000 videos -> npy arrays (needs [preprocess])")
    s.add_argument("--videos", required=True, help="folder with train/ and val/")
    s.add_argument("--out", required=True)
    s.add_argument("--roi", default="cluster", choices=["none", "cluster", "union", "adaptive"])
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--no-enhance", action="store_true", help="skip the bilateral filter + CLAHE")
    s.set_defaults(fn=cmd_build_rwf)

    s = sub.add_parser("profiles", help="motion energy per stored frame (motion sampling)")
    s.add_argument("--root", default="dataset/rwf2000")
    s.set_defaults(fn=cmd_profiles)

    s = sub.add_parser("build-clips", help="videos without an official split -> grouped "
                       "train/test arrays (needs [preprocess]); run `audit` afterwards")
    s.add_argument("--fight", required=True, help="folder of violent videos")
    s.add_argument("--nonfight", required=True, help="folder of non-violent videos")
    s.add_argument("--all-dir", required=True, help="where all_x.npy is written")
    s.add_argument("--out", required=True, help="dataset root with train_x.npy / test_x.npy")
    s.add_argument("--roi", default="cluster", choices=["cluster", "union", "none"])
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--test-fraction", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_build_clips)

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

    s = sub.add_parser("gpus", help="free GPUs, with HIP ids (use these) and rocm-smi ids")
    s.set_defaults(fn=cmd_gpus)

    s = sub.add_parser("cv", help="K-fold evaluation: OOF metrics, threshold, TTA, ensemble")
    s.add_argument("exp", nargs="+", help="runs/<name>/seed0 folders holding fold*/")
    s.add_argument("--variants", nargs="+", help="inference variants (default: all)")
    s.add_argument("--key", default="accuracy", help="OOF metric for the threshold")
    s.add_argument("--table", metavar="VARIANT",
                   help="only print the experiment table (a variant name, or best)")
    s.add_argument("--compare", metavar="VARIANT",
                   help="paired test of every other exp against the first (McNemar, bootstrap)")
    s.add_argument("--selection-bias", action="store_true",
                   help="test accuracy at the validation-picked epoch vs the best test epoch")
    s.add_argument("--device")
    s.set_defaults(fn=cmd_cv)

    s = sub.add_parser("sweep", help="run configs x seeds across GPUs")
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--gpus", nargs="+", default=["auto"], help="GPU ids, or auto")
    s.add_argument("--folds", type=int, nargs="+", help="K-fold jobs, e.g. 0 1 2 3")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--log-dir", default="runs/logs")
    s.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                   metavar="KEY=VALUE", help="applied to every job")
    s.set_defaults(fn=cmd_sweep)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
