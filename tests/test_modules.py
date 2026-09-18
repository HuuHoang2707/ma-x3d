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
