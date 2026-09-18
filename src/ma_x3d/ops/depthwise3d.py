"""Depthwise 3D convolution as Triton kernels.

On ROCm, MIOpen runs X3D's depthwise 3D convolutions (and above all their backward
pass) through generic im2col or naive kernels, which take most of a training step.
These kernels compute the same operation directly. They support what X3D uses:
groups == channels, temporal stride 1, spatial stride s, no dilation, no bias.
Everything else, CPU tensors, and export/tracing fall back to F.conv3d.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

import os

# Tuned on one MI250 GCD (X3D-M shapes, batch 12); override to re-tune on other GPUs.
BLOCK = int(os.environ.get("MA_X3D_DW_BLOCK", 256))  # forward and input gradient
BLOCK_W = int(os.environ.get("MA_X3D_DW_BLOCK_W", 512))  # weight gradient


if triton is not None:

    @triton.jit
    def _fwd(x_ptr, w_ptr, y_ptr, C, T, H, W, Ho, Wo,
             KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, S: tl.constexpr,
             PT: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr, BLOCK: tl.constexpr):
        bct = tl.program_id(0)  # (b, c, t) of the output plane
        t = bct % T
        c = (bct // T) % C
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < Ho * Wo
        oh = offs // Wo
        ow = offs % Wo
        plane = (bct - t).to(tl.int64) * H * W  # start of (b, c, t=0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for kt in tl.static_range(KT):
            ti = t + kt - PT
            ok_t = (ti >= 0) & (ti < T)
            for ki in tl.static_range(KH):
                hi = oh * S + ki - PH
                ok_h = (hi >= 0) & (hi < H)
                for kj in tl.static_range(KW):
                    wi = ow * S + kj - PW
                    m = inside & ok_t & ok_h & (wi >= 0) & (wi < W)
                    xv = tl.load(x_ptr + plane + ti * H * W + hi * W + wi, mask=m, other=0.0)
                    wv = tl.load(w_ptr + c * KT * KH * KW + (kt * KH + ki) * KW + kj)
                    acc += xv.to(tl.float32) * wv.to(tl.float32)
        out = bct.to(tl.int64) * Ho * Wo + offs
        tl.store(y_ptr + out, acc.to(y_ptr.dtype.element_ty), mask=inside)

    @triton.jit
    def _bwd_input(gy_ptr, w_ptr, gx_ptr, C, T, H, W, Ho, Wo,
                   KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, S: tl.constexpr,
                   PT: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr, BLOCK: tl.constexpr):
        bct = tl.program_id(0)  # (b, c, t) of the input plane
        t = bct % T
        c = (bct // T) % C
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < H * W
        hi = offs // W
        wi = offs % W
        plane = (bct - t).to(tl.int64) * Ho * Wo
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for kt in tl.static_range(KT):
            to = t + PT - kt
            ok_t = (to >= 0) & (to < T)
            for ki in tl.static_range(KH):
                hh = hi + PH - ki
                oh = hh // S
                ok_h = (hh >= 0) & (hh % S == 0) & (oh < Ho)
                for kj in tl.static_range(KW):
                    ww = wi + PW - kj
                    ow = ww // S
                    m = inside & ok_t & ok_h & (ww >= 0) & (ww % S == 0) & (ow < Wo)
                    g = tl.load(gy_ptr + plane + to * Ho * Wo + oh * Wo + ow, mask=m, other=0.0)
                    wv = tl.load(w_ptr + c * KT * KH * KW + (kt * KH + ki) * KW + kj)
                    acc += g.to(tl.float32) * wv.to(tl.float32)
        out = bct.to(tl.int64) * H * W + offs
        tl.store(gx_ptr + out, acc.to(gx_ptr.dtype.element_ty), mask=inside)

    @triton.jit
    def _bwd_weight(gy_ptr, x_ptr, gw_ptr, C, T, H, W, Ho, Wo,
                    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, S: tl.constexpr,
                    PT: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
                    TAPS: tl.constexpr, BLOCK: tl.constexpr):
        bct = tl.program_id(0)  # one output plane; sums over its pixels, then atomics
        t = bct % T
        c = (bct // T) % C
        plane_x = (bct - t).to(tl.int64) * H * W
        plane_y = bct.to(tl.int64) * Ho * Wo
        tap_ids = tl.arange(0, TAPS)
        acc = tl.zeros([TAPS], dtype=tl.float32)
        for start in range(0, Ho * Wo, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            inside = offs < Ho * Wo
            oh = offs // Wo
            ow = offs % Wo
            g = tl.load(gy_ptr + plane_y + offs, mask=inside, other=0.0).to(tl.float32)
            for kt in tl.static_range(KT):
                ti = t + kt - PT
                ok_t = (ti >= 0) & (ti < T)
                for ki in tl.static_range(KH):
                    hi = oh * S + ki - PH
                    ok_h = (hi >= 0) & (hi < H)
                    for kj in tl.static_range(KW):
                        wi = ow * S + kj - PW
                        m = inside & ok_t & ok_h & (wi >= 0) & (wi < W)
                        xv = tl.load(x_ptr + plane_x + ti * H * W + hi * W + wi, mask=m,
                                     other=0.0)
                        s = tl.sum(g * xv.to(tl.float32), axis=0)
                        acc += tl.where(tap_ids == (kt * KH + ki) * KW + kj, s, 0.0)
        n_taps = KT * KH * KW
        tl.atomic_add(gw_ptr + c * n_taps + tap_ids, acc, mask=tap_ids < n_taps)


def _geometry(x, w, stride, padding):
    b, c, t, h, wd = x.shape
    kt, kh, kw = w.shape[2:]
    s = stride[1]
    ho = (h + 2 * padding[1] - kh) // s + 1
    wo = (wd + 2 * padding[2] - kw) // s + 1
    meta = dict(KT=kt, KH=kh, KW=kw, S=s, PT=padding[0], PH=padding[1], PW=padding[2])
    return (b, c, t, h, wd, ho, wo), meta


class _DepthwiseConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, stride, padding):
        x = x.contiguous()
        wc = w.contiguous()
        (b, c, t, h, wd, ho, wo), meta = _geometry(x, wc, stride, padding)
        y = torch.empty(b, c, t, ho, wo, device=x.device, dtype=x.dtype)
        grid = (b * c * t, triton.cdiv(ho * wo, BLOCK))
        _fwd[grid](x, wc, y, c, t, h, wd, ho, wo, BLOCK=BLOCK, **meta)
        ctx.save_for_backward(x, wc)
        ctx.stride, ctx.padding = stride, padding
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        (b, c, t, h, wd, ho, wo), meta = _geometry(x, w, ctx.stride, ctx.padding)
        gx = gw = None
        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            grid = (b * c * t, triton.cdiv(h * wd, BLOCK))
            _bwd_input[grid](gy, w, gx, c, t, h, wd, ho, wo, BLOCK=BLOCK, **meta)
        if ctx.needs_input_grad[1]:
            taps = w.shape[2] * w.shape[3] * w.shape[4]
            acc = torch.zeros(c, taps, device=x.device, dtype=torch.float32)
            _bwd_weight[(b * c * t,)](gy, x, acc, c, t, h, wd, ho, wo,
                                      TAPS=triton.next_power_of_2(taps), BLOCK=BLOCK_W, **meta)
            gw = acc.view_as(w).to(w.dtype)
        return gx, gw, None, None


def supported(conv_like, x: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and not torch.jit.is_tracing()
        and not torch.compiler.is_compiling()
        and conv_like.groups == conv_like.in_channels == conv_like.out_channels
        and conv_like.stride[0] == 1
        and conv_like.stride[1] == conv_like.stride[2]
        and tuple(conv_like.dilation) == (1, 1, 1)
        and getattr(conv_like, "bias", None) is None
    )


def depthwise_conv3d(x, w, stride, padding) -> torch.Tensor:
    return _DepthwiseConv3d.apply(x, w, tuple(stride), tuple(padding))


class FastDepthwiseConv3d(nn.Conv3d):
    """nn.Conv3d whose forward uses the Triton kernels when it can (same parameters)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if supported(self, x):
            return depthwise_conv3d(x, self.weight, self.stride, self.padding)
        return super().forward(x)


def use_fast_depthwise(model: nn.Module) -> int:
    """Switch every depthwise Conv3d (and wide-kernel conv) in `model` to the Triton path."""
    from ..models.wide_kernel import WideKernelConv

    n = 0
    for m in model.modules():
        if type(m) is nn.Conv3d and m.groups == m.in_channels == m.out_channels > 1:
            m.__class__ = FastDepthwiseConv3d
            n += 1
        elif isinstance(m, WideKernelConv):
            m.fast = True
            n += 1
    return n
