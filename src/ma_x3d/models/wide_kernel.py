"""Widen the depthwise kernels of chosen X3D stages, in space and/or in time.

Two variants:
  reparam: W = pad(W_base) + ring_mask * W_delta. W_base is the pre-trained 3x3
           kernel, W_delta is zero-initialised and only acts on the outer ring.
           They are separate parameters, so the ring can get its own learning rate.
  dense:   one 5x5 kernel with a zero outer ring (the naive baseline).
Both start as an exact copy of the 3x3 convolution. `fuse_wide_kernels` folds the
reparam form into a plain Conv3d for inference.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideKernelConv(nn.Module):
    def __init__(self, conv: nn.Conv3d, size: int = 5, t_size: int | None = None,
                 dilated: bool = False):
        super().__init__()
        self.dilated = dilated
        out_c, in_c, kt, kh, kw = conv.weight.shape
        t_size = kt if t_size is None else t_size
        assert kh == kw and size >= kh and (size - kh) % 2 == 0
        assert t_size >= kt and (t_size - kt) % 2 == 0 and (size > kh or t_size > kt)
        self.off = (size - kh) // 2
        self.off_t = (t_size - kt) // 2
        self.kernel_size = (t_size, size, size)
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.stride, self.dilation, self.groups = conv.stride, conv.dilation, conv.groups
        self.padding = (conv.padding[0] + self.off_t, conv.padding[1] + self.off,
                        conv.padding[2] + self.off)

        self.base_weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        if dilated:
            # RepLKNet's dilated re-parameterisation: the extra branch is another 3x3
            # whose taps are spread to reach the wide grid, so a 7x7 receptive field
            # costs 9 weights per channel instead of 40.
            assert (size - 1) % (kh - 1) == 0
            self.dil = (size - 1) // (kh - 1)
            self.delta_weight = nn.Parameter(torch.zeros(out_c, in_c, t_size, kh, kw))
            ring = torch.ones(1, 1, t_size, size, size)   # unused, kept for the buffer
        else:
            self.delta_weight = nn.Parameter(torch.zeros(out_c, in_c, t_size, size, size))
            ring = torch.ones(1, 1, t_size, size, size)   # everything outside the old kernel
            ring[:, :, self.off_t : self.off_t + kt,
                 self.off : self.off + kh, self.off : self.off + kw] = 0.0
        self.register_buffer("ring_mask", ring)
        self.fast = False  # set by ops.depthwise3d.use_fast_depthwise

    @property
    def weight(self) -> torch.Tensor:
        o, ot = self.off, self.off_t
        base = F.pad(self.base_weight, (o, o, o, o, ot, ot))
        if self.dilated:
            spread = torch.zeros_like(base)
            spread[..., :: self.dil, :: self.dil] = self.delta_weight
            return base + spread
        return base + self.delta_weight * self.ring_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast:
            from ..ops.depthwise3d import depthwise_conv3d, supported

            if supported(self, x):
                return depthwise_conv3d(x, self.weight, self.stride, self.padding)
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

    @torch.no_grad()
    def fuse(self) -> nn.Conv3d:
        conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                         self.padding, self.dilation, self.groups, bias=self.bias is not None)
        conv.weight.copy_(self.weight)
        if self.bias is not None:
            conv.bias.copy_(self.bias)
        return conv.to(self.base_weight.device)


@torch.no_grad()
def inflate_dense(conv: nn.Conv3d, size: int = 5) -> nn.Conv3d:
    _, _, kt, kh, kw = conv.weight.shape
    off = (size - kh) // 2
    new = nn.Conv3d(conv.in_channels, conv.out_channels, (kt, size, size), conv.stride,
                    (conv.padding[0], conv.padding[1] + off, conv.padding[2] + off),
                    conv.dilation, conv.groups, bias=conv.bias is not None)
    new.weight.zero_()
    new.weight[:, :, :, off : off + kh, off : off + kw] = conv.weight
    if conv.bias is not None:
        new.bias.copy_(conv.bias)
    return new


def depthwise_convs(stage: nn.Module):
    """Yield (res_block, conv_b) for every residual block of an X3D stage."""
    for rb in stage.res_blocks:
        yield rb, rb.branch2.conv_b


def widen_stage(stage: nn.Module, mode: str, size: int = 5, t_size: int | None = None) -> int:
    n = 0
    for rb, conv in depthwise_convs(stage):
        wider = conv.kernel_size[1] < size or (t_size or 0) > conv.kernel_size[0]
        if isinstance(conv, nn.Conv3d) and wider:
            if mode == "dense":
                wide = inflate_dense(conv, size)
            else:
                wide = WideKernelConv(conv, size, t_size, dilated=mode == "dilated")
            rb.branch2.conv_b = wide
            n += 1
    return n


def fuse_wide_kernels(model: nn.Module) -> int:
    """Replace every WideKernelConv in `model` by an equivalent Conv3d, in place."""
    n = 0
    for module in model.modules():
        branch = getattr(module, "branch2", None)
        if branch is not None and isinstance(getattr(branch, "conv_b", None), WideKernelConv):
            branch.conv_b = branch.conv_b.fuse()
            n += 1
    return n


def ring_energy(conv: nn.Module, center: int = 3) -> float:
    """Fraction of |weight| outside the central `center` x `center` window."""
    w = conv.weight.detach().abs().float()
    if w.shape[-1] <= center:
        return 0.0
    o = (w.shape[-1] - center) // 2
    ring = w.clone()
    ring[..., o : o + center, o : o + center] = 0
    return float(ring.sum() / (w.sum() + 1e-12))


def wide_convs(model: nn.Module) -> dict[str, nn.Module]:
    """Name -> widened conv (either variant, fused or not)."""
    return {
        name: m for name, m in model.named_modules()
        if name.endswith("conv_b") and getattr(m, "kernel_size", (0, 0, 0))[1] > 3
    }


@contextlib.contextmanager
def ring_disabled(model: nn.Module):
    """Temporarily zero the learned outer ring of every reparam kernel."""
    saved = []
    for m in model.modules():
        if isinstance(m, WideKernelConv):
            saved.append((m, m.delta_weight.detach().clone()))
            m.delta_weight.data.zero_()
    try:
        yield
    finally:
        for m, w in saved:
            m.delta_weight.data.copy_(w)
