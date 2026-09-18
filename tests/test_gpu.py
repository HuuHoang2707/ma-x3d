"""Triton depthwise kernels against F.conv3d (needs a GPU)."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from ma_x3d.models.wide_kernel import WideKernelConv  # noqa: E402
from ma_x3d.ops.depthwise3d import depthwise_conv3d, use_fast_depthwise  # noqa: E402

DEV = torch.device("cuda")
CASES = [  # (C, T, H, W, kernel, spatial stride) as used by X3D-M
    (24, 16, 112, 112, (5, 1, 1), 1),  # stem temporal conv
    (54, 16, 112, 112, (3, 3, 3), 2),  # first block of Res2
    (54, 16, 56, 56, (3, 5, 5), 1),  # widened Res2
    (216, 16, 14, 14, (3, 3, 3), 1),
    (8, 5, 13, 11, (3, 3, 3), 2),  # odd sizes
]


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_conv3d(case):
    c, t, h, w, k, s = case
    torch.manual_seed(0)
    x = torch.randn(2, c, t, h, w, device=DEV, requires_grad=True)
    wt = torch.randn(c, 1, *k, device=DEV, requires_grad=True)
    pad = tuple(v // 2 for v in k)
    ref = F.conv3d(x, wt, None, (1, s, s), pad, 1, c)
    gy = torch.randn_like(ref)
    gx_ref, gw_ref = torch.autograd.grad(ref, (x, wt), gy)
    out = depthwise_conv3d(x, wt, (1, s, s), pad)
    gx, gw = torch.autograd.grad(out, (x, wt), gy)
    assert rel(out, ref) < 1e-5 and rel(gx, gx_ref) < 1e-5 and rel(gw, gw_ref) < 1e-4


def test_modules_switch_and_match():
    torch.manual_seed(0)
    dw = nn.Conv3d(16, 16, 3, padding=1, groups=16, bias=False).to(DEV)
    wide = WideKernelConv(dw).to(DEV)
    with torch.no_grad():
        wide.delta_weight.normal_()
    model = nn.Sequential(copy.deepcopy(dw), wide)
    x = torch.randn(1, 16, 8, 20, 20, device=DEV)
    ref = model(x)
    assert use_fast_depthwise(model) == 2
    assert torch.allclose(model(x), ref, atol=1e-4, rtol=1e-4)
    assert set(model.state_dict()) == set(copy.deepcopy(model).state_dict())
