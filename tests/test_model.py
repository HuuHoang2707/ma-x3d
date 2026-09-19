"""Tests on the full X3D-M network (randomly initialised, CPU)."""

import copy

import pytest
import torch

from ma_x3d.config import ModelConfig, TrainConfig
from ma_x3d.models.eaa import EfficientAdditiveAttention
from ma_x3d.models.ma_x3d import MAX3D
from ma_x3d.models.motion_attention import MotionAttention
from ma_x3d.models.wide_kernel import WideKernelConv, fuse_wide_kernels, widen_stage
from ma_x3d.train.params import configure_phase, trainable_names

pytestmark = pytest.mark.slow


def make(blocks, ma=True, wk="reparam", eaa=False):
    blocks = copy.deepcopy(blocks)
    if wk != "none":
        for s in (1, 2):
            widen_stage(blocks[s], wk)
    return MAX3D(blocks, MotionAttention(96) if ma else None, "res4",
                 EfficientAdditiveAttention(192) if eaa else None)


@pytest.fixture(scope="module")
def clip():
    torch.manual_seed(0)
    return torch.rand(1, 3, 16, 224, 224)


def test_forward_shape_and_parameter_counts(x3d_blocks, clip):
    base = make(x3d_blocks, ma=False, wk="none").eval()
    full = make(x3d_blocks).eval()
    with torch.no_grad():
        assert full(clip).shape == (1, 2)
    n_base = sum(p.numel() for p in base.parameters())
    fused = copy.deepcopy(full)
    fuse_wide_kernels(fused)
    n_fused = sum(p.numel() for p in fused.parameters())
    assert n_fused - n_base == 2508 + 33696  # MA module + outer rings (thesis Table 3.4)


def test_new_modules_start_as_identity(x3d_blocks, clip):
    base = make(x3d_blocks, ma=False, wk="none").eval()
    full = make(x3d_blocks, eaa=True).eval()
    with torch.no_grad():
        full.motion_attn.gate[2].weight.zero_()  # remove the tiny random init
        assert torch.allclose(base(clip), full(clip), atol=1e-5)


def test_fused_model_matches(x3d_blocks, clip):
    full = make(x3d_blocks).eval()
    with torch.no_grad():
        for m in full.modules():
            if isinstance(m, WideKernelConv):
                m.delta_weight.normal_(0, 0.05)
        ref = full(clip)
        fuse_wide_kernels(full)
        assert torch.allclose(ref, full(clip), atol=1e-4)


def test_exact_phases(x3d_blocks):
    model = make(x3d_blocks)
    cfg = TrainConfig()
    configure_phase(model, "probe", cfg)
    names = trainable_names(model)
    assert names and all(n.startswith(("motion_attn.", "blocks.5.")) for n in names)

    groups = configure_phase(model, "finetune", cfg)
    names = trainable_names(model)
    assert not any(n.startswith(("blocks.0.", "blocks.3.")) for n in names)  # stem, Res4
    assert any(n.startswith("blocks.4.") for n in names)
    by_group = {g["name"]: {id(p) for p in g["params"]} for g in groups}
    params = dict(model.named_parameters())
    ring = params["blocks.1.res_blocks.0.branch2.conv_b.delta_weight"]
    base = params["blocks.1.res_blocks.0.branch2.conv_b.base_weight"]
    assert id(ring) in by_group["new"] and id(base) in by_group["backbone"]
    assert id(params["blocks.4.res_blocks.5.branch2.conv_a.weight"]) in by_group["backbone"]

    cfg.ring_lr = "backbone"
    groups = configure_phase(model, "finetune", cfg)
    assert id(ring) in {id(p) for p in groups[0]["params"]}


def test_head_new_proj_is_a_linear_probe(x3d_blocks):
    model = make(x3d_blocks)
    cfg = TrainConfig(head_new="proj")
    configure_phase(model, "probe", cfg)
    names = trainable_names(model)
    assert names and all(n.startswith(("motion_attn.", "blocks.5.proj.")) for n in names)
    groups = configure_phase(model, "finetune", cfg)
    new = {id(p) for g in groups if g["name"] == "new" for p in g["params"]}
    params = dict(model.named_parameters())
    assert id(params["blocks.5.proj.weight"]) in new
    assert id(params["blocks.5.pool.post_conv.weight"]) not in new
    assert params["blocks.5.pool.post_conv.weight"].requires_grad  # head is in unfreeze


def test_legacy_matching_reproduces_notebook_leak(x3d_blocks):
    model = make(x3d_blocks)
    cfg = TrainConfig(param_match="legacy")
    configure_phase(model, "probe", cfg)
    names = set(trainable_names(model))
    # "blocks.5" also matches the sixth block of Res4 and Res5
    assert "blocks.3.res_blocks.5.branch2.conv_a.weight" in names
    configure_phase(model, "finetune", cfg)
    names = set(trainable_names(model))
    assert "blocks.3.res_blocks.10.branch2.conv_a.weight" in names  # "blocks.1" in "res_blocks.10"
    assert "blocks.3.res_blocks.3.branch2.conv_a.weight" not in names


def test_build_model_uses_config(monkeypatch, x3d_blocks):
    from ma_x3d.models import builder

    monkeypatch.setattr(builder, "x3d_blocks", lambda *a: copy.deepcopy(x3d_blocks))
    m = builder.build_model(ModelConfig(ma_stage="res3", wide_kernel="dense"))
    assert m.motion_attn.gate[2].out_channels == 48
    assert m.blocks[1].res_blocks[0].branch2.conv_b.kernel_size == (3, 5, 5)


@pytest.mark.parametrize("name,frames", [("tv_r3d_18", 8), ("tv_mc3_18", 8),
                                         ("tv_r2plus1d_18", 8), ("tv_s3d", 16)])
def test_torchvision_baselines(name, frames):
    from ma_x3d.models.builder import build_model

    m = build_model(ModelConfig(backbone=name, pretrained=False, motion_attention=False,
                                wide_kernel="none")).eval()
    with torch.no_grad():
        assert m(torch.rand(1, 3, frames, 64, 64)).shape == (1, 2)
    new = [n for n, _ in m.named_parameters() if n.startswith(("net.fc.", "net.classifier."))]
    assert new  # the replaced head is found by the parameter grouping


@pytest.mark.parametrize("name,frames", [("x3d_xs", 4), ("x3d_s", 13)])
def test_x3d_small_run_at_160(name, frames):
    from ma_x3d.models.builder import build_model

    m = build_model(ModelConfig(backbone=name, pretrained=False, motion_attention=False,
                                wide_kernel="none", input_size=160)).eval()
    with torch.no_grad():
        feat = m.features(torch.rand(1, 3, frames, 224, 224), "res5")
        assert feat.shape[-2:] == (5, 5)  # 160 / 32
        assert m(torch.rand(1, 3, frames, 224, 224)).shape == (1, 2)
