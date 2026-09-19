"""Two-phase training: a frozen-backbone probe, then discriminative fine-tuning.

Each run writes to runs/<name>/seed<seed>/:
    config.yaml, env.json, split.json   what was run
    metrics.jsonl                       one line per epoch
    last.pt, best.pt                    resume state / weights picked on validation F1
    results.json, test_predictions.npz  final evaluation of best.pt
    train.log                           console output
"""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from ..config import Config, save_config
from ..data.dataset import build_datasets, make_loader
from ..eval.metrics import format_report
from ..eval.predict import predict, save_predictions
from ..models import build_model
from ..models.wide_kernel import (
    WideKernelConv,
    fuse_wide_kernels,
    ring_disabled,
    ring_energy,
    wide_convs,
)
from ..ops.depthwise3d import use_fast_depthwise
from ..progress import Progress
from . import dist
from .params import configure_phase, freeze_frozen_bn
from .utils import ModelEMA, amp_dtype, cutmix, environment_info, make_scheduler, seed_everything


def run_dir(cfg: Config) -> Path:
    out = Path(cfg.output_dir) / cfg.name / f"seed{cfg.seed}"
    return out / f"fold{cfg.data.fold}" if cfg.data.protocol == "kfold" else out


class _Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")


def _train_epoch(model, fwd, loader, opt, scaler, cfg: Config, device, amp, ema,
                 epoch: int) -> dict:
    t = cfg.train
    model.train()
    if t.freeze_bn:
        freeze_frozen_bn(model)
    params = [p for g in opt.param_groups for p in g["params"]]
    loss_sum, correct, n = 0.0, 0, 0
    if hasattr(loader.sampler, "set_epoch"):  # DistributedSampler: new shuffle each epoch
        loader.sampler.set_epoch(epoch)
    bar = Progress(loader, desc=f"train ep {epoch}", enabled=dist.is_main())
    for clip, y in bar:
        clip, y = clip.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mixed = t.cutmix_prob > 0 and len(y) > 1 and random.random() < t.cutmix_prob
        if mixed:
            clip, y_b, lam = cutmix(clip, y, t.cutmix_alpha)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = fwd(clip).float()
        loss = F.cross_entropy(logits, y, label_smoothing=t.label_smoothing)
        if mixed:
            loss = lam * loss + (1 - lam) * F.cross_entropy(
                logits, y_b, label_smoothing=t.label_smoothing)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if t.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, t.grad_clip)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        bar.set(loss=loss_sum / n, acc=correct / n)
    loss_sum, correct, n = dist.sum_all([loss_sum, correct, n], device)
    return {"loss": loss_sum / n, "acc": correct / n}


def _enter_phase(model, phase: str, cfg: Config):
    t = cfg.train
    groups = configure_phase(model, phase, t)
    opt = torch.optim.AdamW(groups, betas=tuple(t.betas), weight_decay=t.weight_decay)
    span = t.probe_epochs if phase == "probe" else t.epochs - t.probe_epochs
    return opt, make_scheduler(opt, phase, span)


def _wrap(model, cfg: Config, device):
    """Forward module for training. DDP only tracks parameters that require grad when
    it is built, so it is rebuilt after every phase switch."""
    fwd = model
    if torch.distributed.is_initialized():
        fwd = DDP(model, device_ids=[device.index], forward_sync_buffers=False)
    return torch.compile(fwd) if cfg.train.compile else fwd


def _final_eval(model, loaders, cfg: Config, device, amp, out: Path, log) -> dict:
    t = cfg.train
    res = {"protocol": cfg.data.protocol}
    test = predict(model, loaders["test"], device, amp, t.label_smoothing)
    res["test"] = test["metrics"]
    save_predictions(out / "test_predictions.npz", test)
    log("test (best checkpoint)\n" + format_report(test["metrics"]))

    convs = wide_convs(model)
    if convs:
        energies = {name: ring_energy(m) for name, m in convs.items()}
        res["ring_energy"] = energies
        res["ring_energy_mean"] = sum(energies.values()) / len(energies)
        log(f"outer-ring energy (mean over {len(convs)} convs): {res['ring_energy_mean']:.4f}")
    if any(isinstance(m, WideKernelConv) for m in model.modules()):
        with ring_disabled(model):
            off = predict(model, loaders["test"], device, amp, t.label_smoothing)
        res["test_ring_off"] = off["metrics"]
        log(f"test with ring zeroed: acc {off['metrics']['accuracy']:.4f} "
            f"f1 {off['metrics']['f1']:.4f}")
        # fp32 on one batch: under bf16 the two paths use different kernels and round
        # differently, which would hide a real mismatch
        fused = copy.deepcopy(model)
        fuse_wide_kernels(fused)
        clip, _ = next(iter(loaders["test"]))
        with torch.no_grad():
            clip = clip.to(device)
            a = model.eval()(clip).float().softmax(1)
            b = fused.eval()(clip).float().softmax(1)
        diff = float((a - b).abs().max())
        res["fused_max_prob_diff"] = diff
        log(f"fused vs. reparam (fp32, one batch): max |dp| = {diff:.2e}")
    return res


def train(cfg: Config, device: torch.device, resume: bool = False, overwrite: bool = False) -> Path:
    """Train one run. Under torchrun the run is split over all started GPUs."""
    t = cfg.train
    rank, world, local = dist.init()
    main = rank == 0
    if world > 1:
        device = torch.device("cuda", local)
        if t.batch_size % world:
            raise ValueError(f"batch_size {t.batch_size} is not divisible by {world} GPUs")
    out = run_dir(cfg)
    if (out / "results.json").exists() and not overwrite:
        if main:
            print(f"[skip] {out} already finished (use --overwrite to rerun)")
        return out
    if main:
        if out.exists() and overwrite:
            shutil.rmtree(out)
        if (out / "last.pt").exists() and not resume:
            raise FileExistsError(f"{out} has a partial run; pass --resume or --overwrite")
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    log = _Log(out / "train.log") if main else (lambda msg: None)
    if main:
        save_config(cfg, out / "config.yaml")
        env = environment_info(device) | {"world_size": world}
        (out / "env.json").write_text(json.dumps(env, indent=2))

    seed_everything(cfg.seed)
    ds = build_datasets(cfg)
    if main:
        (out / "split.json").write_text(json.dumps(
            {k: (v.indices.tolist() if v is not None else None) for k, v in ds.items()}))
    w = cfg.data.num_workers
    sampler = (DistributedSampler(ds["train"], world, rank, shuffle=True, seed=cfg.seed)
               if world > 1 else None)
    loaders = {
        "train": make_loader(ds["train"], t.batch_size // world, True, w,
                             cfg.seed + 1000 * rank, sampler),
        "test": make_loader(ds["test"], t.batch_size, False, w),
    }
    if ds["val"] is not None:
        loaders["val"] = make_loader(ds["val"], t.batch_size, False, w)
    else:  # test_as_val protocol
        loaders["val"] = loaders["test"]
    log(f"run {out} | protocol {cfg.data.protocol} | train {len(ds['train'])} samples "
        f"({len(ds['train'].indices)} clips) | val {len(loaders['val'].dataset)} | "
        f"test {len(ds['test'])}" + (f" | {world} GPUs x {t.batch_size // world} clips"
                                     if world > 1 else ""))

    model = build_model(cfg.model).to(device)
    if t.fast_depthwise and device.type == "cuda":
        log(f"triton depthwise convs: {use_fast_depthwise(model)}")
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    amp = amp_dtype(t.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=t.amp == "fp16")
    ema = ModelEMA(model, t.ema_decay) if t.ema_decay > 0 else None
    n_params = sum(p.numel() for p in model.parameters())

    start, phase, opt, sched, fwd = 0, None, None, None, None
    best_f1, best_loss, no_improve = -1.0, math.inf, 0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.module.load_state_dict(ck["ema"])
        start, best_f1, best_loss, no_improve = (ck["epoch"], ck["best_f1"], ck["best_loss"],
                                                 ck["no_improve"])
        # phase of the last finished epoch: its optimizer state is what was saved
        phase = "probe" if start - 1 < t.probe_epochs else "finetune"
        opt, sched = _enter_phase(model, phase, cfg)
        fwd = _wrap(model, cfg, device)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        log(f"resumed at epoch {start} ({phase})")

    stopped = False
    for epoch in range(start, t.epochs):
        want = "probe" if epoch < t.probe_epochs else "finetune"
        if want != phase:
            phase = want
            opt, sched = _enter_phase(model, phase, cfg)
            fwd = None  # drop the old DDP wrapper before building the new one
            fwd = _wrap(model, cfg, device)
            no_improve = 0
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            lrs = ", ".join(f"{g['name']}={g['lr']:.0e}" for g in opt.param_groups)
            log(f"phase {phase}: {n_train:,}/{n_params:,} trainable | {lrs}")

        t0 = time.time()
        lrs = [g["lr"] for g in opt.param_groups]
        tr = _train_epoch(model, fwd, loaders["train"], opt, scaler, cfg, device, amp, ema,
                          epoch + 1)
        sched.step()
        stop = False
        if main:
            judge = ema.module if ema is not None else model
            va = predict(judge, loaders["val"], device, amp, t.label_smoothing)["metrics"]
            dt = time.time() - t0
            improved = va["f1"] > best_f1
            if improved:
                best_f1, best_loss, no_improve = va["f1"], va["loss"], 0
                torch.save({"model": judge.state_dict(), "epoch": epoch + 1, "val": va},
                           out / "best.pt")
            else:
                no_improve += 1
            row = {"epoch": epoch + 1, "phase": phase, "lr": lrs, "train_loss": tr["loss"],
                   "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["accuracy"],
                   "val_f1": va["f1"], "best": improved, "sec": round(dt, 1)}
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            eta = dt * (t.epochs - epoch - 1) / 60
            log(f"ep {epoch + 1:02d}/{t.epochs} {phase:8s} | "
                f"train {tr['loss']:.4f}/{tr['acc']:.3f} | "
                f"val {va['loss']:.4f}/{va['accuracy']:.3f} f1 {va['f1']:.4f}"
                f"{' *' if improved else '  '} | {dt:.0f}s | eta {eta:.0f} min")
            torch.save({"model": model.state_dict(),
                        "ema": ema.module.state_dict() if ema else None,
                        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1,
                        "best_loss": best_loss, "no_improve": no_improve}, out / "last.pt")
            stop = no_improve >= t.patience
        if dist.broadcast_bool(stop, device):
            log(f"early stop after epoch {epoch + 1}")
            stopped = True
            break

    fwd = None
    if main:
        best = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        results = _final_eval(model, loaders, cfg, device, amp, out, log)
        results.update({"name": cfg.name, "seed": cfg.seed, "best_epoch": best["epoch"],
                        "val": best["val"], "early_stopped": stopped, "params": n_params,
                        "world_size": world,
                        "peak_mem_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                        if device.type == "cuda" else None)})
        (out / "results.json").write_text(json.dumps(results, indent=2))
        (out / "last.pt").unlink(missing_ok=True)  # best.pt is kept
    dist.barrier()
    return out
