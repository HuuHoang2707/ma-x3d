"""Experiment configuration: typed dataclasses loaded from YAML.

A YAML file may name a parent with `_base_: other.yaml` (path relative to the file).
Command-line overrides use dotted keys, e.g. `train.epochs=10 model.wide_kernel=none`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "dataset/rwf2000"  # output of `ma-x3d prepare`
    frames_stored: int = 64
    frames: int = 16  # frames sampled per clip
    # "holdout": pick the checkpoint on a validation split carved out of the training set,
    # then evaluate the official test set once.
    # "test_as_val": the Kaggle notebook protocol (checkpoint picked on the test set).
    protocol: str = "holdout"
    val_fraction: float = 0.1
    val_blocks: int = 20  # contiguous index blocks per class, see data/splits.py
    split_seed: int = 0
    # "kfold": grouped stratified K-fold on train (groups from `ma-x3d audit`)
    n_folds: int = 4
    fold: int = 0
    # drop train clips that duplicate test clips (and second copies of train duplicates)
    exclude_duplicates: bool = True
    augment: bool = True
    augment_multiplier: int = 2  # each training clip appears this many times per epoch
    # fraction of the fold's training clips actually used (1.0 = all); the subset is
    # drawn by same-scene group and keeps the class balance (low-data study)
    train_fraction: float = 1.0
    # Training window as a fraction of the stored clip; null = thesis sampler (see sampling.py)
    train_span: list[float] | None = field(default_factory=lambda: [0.6, 1.0])
    rotation_deg: float = 10.0
    temporal_inverse: bool = False
    num_workers: int = 12


@dataclass
class ModelConfig:
    num_classes: int = 2
    # x3d_xs | x3d_s | x3d_m | x3d_l | tv_s3d | tv_mc3_18 | tv_r2plus1d_18 | tv_r3d_18
    # | videomae_b (teacher)
    backbone: str = "x3d_m"
    input_size: int = 224  # X3D input side; clips are resized inside the model if different
    pretrained: bool = True
    normalize_input: bool = True  # Kinetics mean/std; the notebook fed raw [0, 1] frames
    head_dropout: float = 0.5
    motion_attention: bool = True
    ma_stage: str = "res4"  # stage whose output is gated
    motion_multiscale: bool = True  # max over frame steps 1 and 2
    motion_clip: float = 0.2
    # "zero" feeds an all-zero motion map to the gate (control: same parameters, no motion)
    motion_input: str = "frames"
    ma_modes: int = 4  # 0 feeds the raw motion map to the gate (no temporal conv)
    ma_temporal_kernel: int = 3
    ma_reduction: int = 4
    wide_kernel: str = "reparam"  # none | reparam | dense
    wk_stages: list[str] = field(default_factory=lambda: ["res2", "res3"])
    wk_size: int = 5
    eaa: bool = False  # efficient additive attention after Res5 (notebook v5 experiment)
    # stages whose output gets a zero-initialised temporal-difference residual
    diff_residual: list[str] = field(default_factory=list)
    burst_pool: bool = False  # attention pooling over time in the head
    interaction: bool = False  # motion-peak interaction tokens (logit correction)
    interaction_peaks: int = 6


@dataclass
class TrainConfig:
    epochs: int = 24
    probe_epochs: int = 4
    patience: int = 6
    batch_size: int = 12
    lr_probe: float = 1e-3
    lr_backbone: float = 1e-5
    lr_new: float = 5e-4
    weight_decay: float = 5e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    amp: str = "bf16"  # bf16 | fp16 | none
    # Stages trained in the fine-tuning phase (new modules are always trained).
    unfreeze: list[str] = field(default_factory=lambda: ["res2", "res3", "res5", "head"])
    # "exact" matches module prefixes; "legacy" copies the notebook's substring matching.
    param_match: str = "exact"
    ring_lr: str = "new"  # learning-rate group of the wide-kernel ring: new | backbone
    head_new: str = "proj"  # part of the X3D head trained as new: all | proj (see params.py)
    freeze_bn: bool = True  # keep BatchNorm of frozen stages in eval mode
    ema_decay: float = 0.999  # 0 disables EMA
    cutmix_prob: float = 0.5
    cutmix_alpha: float = 1.0
    compile: bool = False
    # Triton kernels for depthwise 3D convs (same result, ~2x faster training on MI250)
    fast_depthwise: bool = True
    # knowledge distillation: teacher run dir, "{seed}" and "{fold}" are filled in.
    # The teacher must have been trained on exactly the same training clips.
    distill: str = ""
    distill_alpha: float = 0.5  # weight of the teacher term
    distill_temp: float = 2.0
    # also score the test split after every epoch (written to metrics.jsonl only, never
    # used for selection); measures how much picking the epoch on test would inflate
    log_test_curve: bool = False


@dataclass
class Config:
    name: str = "ma_x3d"
    description: str = ""
    seed: int = 0
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("_base_", None)
    if base is None:
        return raw
    return _merge(_read_yaml((path.parent / base).resolve()), raw)


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    data = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return data


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}


def _coerce(value: Any, default: Any) -> Any:
    """Match the type of the default. YAML reads 1e-3 (no dot) as a string."""
    if isinstance(default, bool) or value is None:
        return value
    if isinstance(default, float) and isinstance(value, (int, str)):
        return float(value)
    if isinstance(default, int) and isinstance(value, str):
        return int(value)
    if isinstance(default, list) and isinstance(value, list) and default:
        return [_coerce(v, default[0]) for v in value]
    return value


def _build(cls, values: dict):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    defaults = cls()
    return cls(**{k: _coerce(v, getattr(defaults, k)) for k, v in values.items()})


def from_dict(values: dict) -> Config:
    values = dict(values)
    sections = {k: _build(c, values.pop(k, None) or {}) for k, c in _SECTIONS.items()}
    return _build(Config, {**values, **sections})


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    values = _read_yaml(Path(path)) if path else {}
    values = apply_overrides(values, overrides or [])
    return from_dict(values)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
