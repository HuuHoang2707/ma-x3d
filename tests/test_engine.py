"""End-to-end smoke test: train, evaluate and report on a few synthetic clips (CPU)."""

import copy
import json

import pytest
import torch

from conftest import write_arrays
from ma_x3d.config import load_config
from ma_x3d.report import write_report
from ma_x3d.train import engine

pytestmark = pytest.mark.slow


def test_train_evaluate_report(tmp_path, monkeypatch, x3d_blocks):
    from ma_x3d.models import builder

    monkeypatch.setattr(builder, "x3d_blocks", lambda *a: copy.deepcopy(x3d_blocks))
    root = write_arrays(tmp_path / "data", n_train=8, n_test=4, frames=16)
    cfg = load_config(overrides=[
        f"data.root={root}", f"output_dir={tmp_path / 'runs'}", "name=smoke",
        "data.frames=16", "data.frames_stored=16", "data.val_fraction=0.5",
        "data.val_blocks=2", "data.num_workers=0", "data.augment_multiplier=1",
        "train.epochs=2", "train.probe_epochs=1", "train.batch_size=2", "train.amp=none",
    ])
    out = engine.train(cfg, torch.device("cpu"))

    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["probe", "finetune"]
    res = json.loads((out / "results.json").read_text())
    assert res["test"]["n"] == 4 and "test_ring_off" in res
    assert res["fused_max_prob_diff"] < 1e-4
    assert (out / "best.pt").exists() and not (out / "last.pt").exists()

    md = write_report(tmp_path / "runs", tmp_path / "reports")
    assert "smoke" in md and (out / "curves.png").exists()


def test_every_added_module_trains():
    """A module that matches no stage and is not listed as new would never train."""
    from ma_x3d.config import ModelConfig, TrainConfig
    from ma_x3d.models.builder import build_model
    from ma_x3d.train.params import configure_phase

    cases = {
        "fast.": ModelConfig(motion_attention=False, wide_kernel="none", slowfast=True,
                             frames=64),
        ".tdm.": ModelConfig(motion_attention=False, wide_kernel="none",
                             tdm_stages=["res2", "res3", "res4"]),
        "motion_attn.": ModelConfig(wide_kernel="none"),
        "diff_residual.": ModelConfig(motion_attention=False, wide_kernel="none",
                                      diff_residual=["res3"]),
    }
    for key, mc in cases.items():
        model = build_model(mc)
        for phase in ("probe", "finetune"):
            configure_phase(model, phase, TrainConfig())
            flags = [p.requires_grad for n, p in model.named_parameters() if key in n]
            assert flags and all(flags), f"{key} does not train in the {phase} phase"
