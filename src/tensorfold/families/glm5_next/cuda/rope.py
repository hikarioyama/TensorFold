"""Rotary positions on CUDA (Triton): the key part GLM-5.3 adds to MLA, and the DSA indexer's own keys.

GLM-5.3's MLA rotates a 64-wide key part that every head shares (``kv_a_proj_with_mqa``'s tail, which is not
LayerNormed), and its sparse indexer rotates its 128-wide queries and per-token keys with the same theta. The
arithmetic is the checkpoint's: ``interleave`` pairs element ``2i`` with ``2i + 1`` (GLM's ``rope_interleave``), the
other layout pairs ``i`` with ``i + nf``; each pair is turned by ``position * theta ** (-2i / dim)``.

Rows are independent: row ``r`` of a window sits at ``pos + r`` and reads only its own element pair, so a row's
bits do not depend on the window it arrived in.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(X, POS, INV, H: tl.constexpr, W: tl.constexpr, OFF: tl.constexpr, NF: tl.constexpr,
          INTER: tl.constexpr, B: tl.constexpr):
    """One program per (row, head): rotate ``NF`` pairs at ``OFF`` of a ``[R, H, W]`` row-major tensor."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.load(POS).to(tl.float64) + r                      # the window's first row is at POS, then one a row
    i = tl.arange(0, B)
    ok = i < NF
    inv = tl.load(INV + i, mask=ok, other=0.0).to(tl.float64)
    ang = p * inv
    c = tl.cos(ang).to(tl.float32)
    s = tl.sin(ang).to(tl.float32)
    base = (r * H + h) * W + OFF
    if INTER:
        i0, i1 = 2 * i, 2 * i + 1
    else:
        i0, i1 = i, NF + i
    x0 = tl.load(X + base + i0, mask=ok, other=0.0).to(tl.float32)
    x1 = tl.load(X + base + i1, mask=ok, other=0.0).to(tl.float32)
    tl.store(X + base + i0, (x0 * c - x1 * s).to(tl.bfloat16), mask=ok)
    tl.store(X + base + i1, (x0 * s + x1 * c).to(tl.bfloat16), mask=ok)


def apply_(x: torch.Tensor, inv_freq: torch.Tensor, pos: torch.Tensor, *, offset: int = 0, nf: int | None = None,
           interleave: bool = True) -> torch.Tensor:
    """Rotate ``x`` [R, H, W] bf16 in place: ``nf`` pairs (default: all of them) at ``offset``, rows at pos + r.

    ``pos`` is a device int32 tensor holding the window's first position (or the row's own, when R is 1).
    """

    if x.dim() != 3 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"rope: x must be a contiguous bf16 [R, H, W], got {tuple(x.shape)} {x.dtype}")
    R, H, W = x.shape
    nf = int(nf if nf is not None else inv_freq.numel())
    if R == 0 or nf == 0:
        return x
    if offset + (2 * nf if interleave else nf) > W:
        raise ValueError(f"rope: {2 * nf} rotated columns at {offset} do not fit a {W}-wide row")
    _rope[(R, H)](x, pos, inv_freq, H=H, W=W, OFF=offset, NF=nf, INTER=interleave,
                  B=int(2 ** math.ceil(math.log2(max(nf, 2)))), num_warps=1)
    return x


def apply(x: torch.Tensor, inv_freq: torch.Tensor, pos: torch.Tensor, **kwargs) -> torch.Tensor:
    """``apply_`` on a copy (for a caller that must keep the unrotated rows)."""

    return apply_(x.clone(), inv_freq, pos, **kwargs)


__all__ = ["apply", "apply_"]
