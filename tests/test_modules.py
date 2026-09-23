import torch
import torch.nn as nn

from ma_x3d.eval.metrics import classification_metrics, roc_auc
from ma_x3d.models.motion_attention import MotionAttention, motion_map
from ma_x3d.models.wide_kernel import (
    WideKernelConv,
    fuse_wide_kernels,
    inflate_dense,
    ring_disabled,
    ring_energy,
)


def depthwise(c=8):
    torch.manual_seed(0)
    return nn.Conv3d(c, c, (3, 3, 3), padding=(1, 1, 1), groups=c, bias=False)


def test_motion_map_values():
    clip = torch.zeros(1, 3, 4, 2, 2)
    clip[:, :, 2] = 0.1  # frame 2 differs from frames 1 and 3 by 0.1
    m = motion_map(clip, multiscale=False)
    assert m.shape == (1, 1, 4, 2, 2)
    assert torch.allclose(m[0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 0.5, 0.5]))
    assert motion_map(torch.ones(1, 3, 4, 2, 2)).abs().max() == 0
    big = torch.zeros(1, 3, 3, 1, 1)
    big[:, :, 1] = 1.0
    assert motion_map(big).max() == 1.0  # clipped at 0.2, scaled by 5


def test_motion_attention_starts_near_identity():
    torch.manual_seed(0)
    for modes in (4, 0):
        ma = MotionAttention(channels=96, modes=modes)
        f = torch.randn(2, 96, 4, 7, 7)
        m = torch.rand(2, 1, 16, 56, 56)
        y = ma(f, m)
        assert y.shape == f.shape
        assert (y - f).abs().max() < 0.05 * f.abs().max()


def test_wide_kernel_identity_at_init():
    conv = depthwise()
    wide = WideKernelConv(conv)
    x = torch.randn(1, 8, 5, 12, 12)
    assert torch.allclose(conv(x), wide(x), atol=1e-6)
    assert torch.allclose(conv(x), inflate_dense(conv)(x), atol=1e-6)
    assert ring_energy(wide) == 0.0


def test_wide_kernel_fuse_matches_and_center_is_protected():
    conv = depthwise()
    wide = WideKernelConv(conv)
    with torch.no_grad():
        wide.delta_weight.normal_()
    # the mask keeps the centre equal to the pre-trained kernel
    assert torch.equal(wide.weight[..., 1:4, 1:4], conv.weight)
    x = torch.randn(1, 8, 5, 12, 12)
    fused = wide.fuse()
    assert isinstance(fused, nn.Conv3d) and fused.kernel_size == (3, 5, 5)
    assert torch.allclose(wide(x), fused(x), atol=1e-5)
    assert 0.0 < ring_energy(wide) < 1.0


def test_ring_disabled_restores_weights():
    wide = WideKernelConv(depthwise())
    with torch.no_grad():
        wide.delta_weight.normal_()
    before = wide.delta_weight.clone()
    with ring_disabled(wide):
        assert wide.delta_weight.abs().sum() == 0
    assert torch.equal(wide.delta_weight, before)


def test_fuse_in_a_block():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_b = WideKernelConv(depthwise())

    block = nn.Module()
    block.branch2 = Branch()
    assert fuse_wide_kernels(block) == 1
    assert isinstance(block.branch2.conv_b, nn.Conv3d)


def test_metrics_on_thesis_confusion_matrix():
    # Thesis Table 4.3: TN 172, FP 28, FN 10, TP 190.
    y = [0] * 200 + [1] * 200
    p = [0] * 172 + [1] * 28 + [0] * 10 + [1] * 190
    m = classification_metrics(y, p)
    assert abs(m["accuracy"] - 0.905) < 1e-9
    assert abs(m["f1"] - 380 / 418) < 1e-9
    assert m["confusion"] == [[172, 28], [10, 190]]
    assert abs(m["per_class"]["NonFight"]["recall"] - 0.86) < 1e-9


def test_auc_matches_pairwise_definition():
    torch.manual_seed(0)
    y = (torch.rand(50) > 0.5).int().numpy()
    s = torch.rand(50).round(decimals=1).numpy()  # with ties
    pos, neg = s[y == 1], s[y == 0]
    pairs = [(a > b) + 0.5 * (a == b) for a in pos for b in neg]
    assert abs(roc_auc(y, s) - sum(pairs) / len(pairs)) < 1e-9


def test_zero_motion_input_removes_motion(x3d_blocks):
    import copy

    from ma_x3d.models.ma_x3d import MAX3D
    from ma_x3d.models.motion_attention import MotionAttention

    ma = MotionAttention(96)
    m = MAX3D(copy.deepcopy(x3d_blocks), copy.deepcopy(ma), motion_input="zero")
    clip = torch.rand(1, 3, 16, 64, 64)
    assert m.motion(clip).abs().max() == 0
    assert MAX3D(copy.deepcopy(x3d_blocks), ma).motion(clip).abs().max() > 0


def test_diff_residual_and_burst_pool_start_as_identity():
    from ma_x3d.models.interaction import BurstPool, FeatureDiffResidual

    x = torch.rand(2, 24, 8, 16, 16)
    r = FeatureDiffResidual(24)
    assert torch.allclose(r(x), x, atol=1e-6)  # zero-initialised projection
    r.project.weight.data.normal_(0, 0.1)
    assert not torch.allclose(r(x), x, atol=1e-6)  # and it does something once trained

    z = torch.rand(2, 32, 8, 4, 4)
    b = BurstPool(32)
    assert torch.allclose(b(z), z.mean((2, 3, 4), keepdim=True), atol=1e-6)
    assert b(z).shape == (2, 32, 1, 1, 1)


def test_motion_peak_interaction():
    from ma_x3d.models.interaction_tokens import MotionPeakInteraction

    m = MotionPeakInteraction(96, peaks=4)
    feat, motion = torch.rand(2, 96, 8, 14, 14), torch.rand(2, 1, 8, 112, 112)
    out = m(feat, motion)
    assert out.shape == (2, 2)
    assert out.abs().max() == 0  # zero-initialised output: identity at start

    energy = torch.zeros(1, 14, 14)
    energy[0, 3, 3], energy[0, 10, 10] = 1.0, 0.9
    picks = [(int(i) // 14, int(i) % 14) for i in m._select(energy)[0]]
    assert picks[0] == (3, 3) and picks[1] == (10, 10)  # peaks, spread apart


def test_temporal_wide_kernel_is_identity_and_fuses():
    """Widening in time must also start as the pre-trained kernel and fuse exactly."""
    import torch
    import torch.nn as nn

    from ma_x3d.models.wide_kernel import WideKernelConv

    conv = nn.Conv3d(4, 4, (3, 3, 3), padding=(1, 1, 1), groups=4)
    x = torch.randn(2, 4, 8, 12, 12)
    for size, t_size in [(3, 5), (5, 5), (7, 3)]:
        wide = WideKernelConv(conv, size, t_size)
        assert wide.kernel_size == (t_size, size, size)
        assert torch.allclose(wide(x), conv(x), atol=1e-6)   # identity at init
        assert torch.allclose(wide.fuse()(x), wide(x), atol=1e-6)


def test_temporal_difference_starts_as_identity():
    """Motion enters as features, and the stage must start exactly as it was."""
    import torch
    import torch.nn as nn

    from ma_x3d.models.tdm import TemporalDifference, WithTemporalDifference

    x = torch.randn(2, 24, 8, 14, 14)
    tdm = TemporalDifference(24)
    assert torch.allclose(tdm(x), x, atol=1e-6)          # zero-init output projection
    stage = nn.Conv3d(24, 24, 1)
    assert torch.allclose(WithTemporalDifference(stage, 24)(x), stage(x), atol=1e-6)


def test_fast_pathway_starts_silent_and_sees_every_frame():
    """The lateral connections are zero-initialised, so the slow stream is untouched."""
    import torch

    from ma_x3d.config import ModelConfig
    from ma_x3d.models.builder import build_model

    base = build_model(ModelConfig(motion_attention=False, wide_kernel="none")).eval()
    sf = build_model(ModelConfig(motion_attention=False, wide_kernel="none",
                                 slowfast=True, frames=64)).eval()
    sf.load_state_dict(base.state_dict(), strict=False)
    clip = torch.rand(1, 3, 64, 224, 224)   # alpha 4 leaves the main stream 16
    with torch.no_grad():
        assert torch.allclose(sf(clip), base(clip[:, :, ::4]), atol=1e-5)
    assert sf.fast is not None and sf.fast.alpha == 4
