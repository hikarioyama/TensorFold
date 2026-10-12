"""Exact parallel radix selection; score arithmetic and ascending token order are unchanged."""
import torch
import triton
import triton.language as tl
from .sparse53 import _order_key

TILE = 1024


@triton.jit
def _hist(S, HIST, STATE, POS, stride, NP: tl.constexpr, NC: tl.constexpr, PASS: tl.constexpr, BLOCK: tl.constexpr):
    r, c = tl.program_id(0), tl.program_id(1)
    i = c * BLOCK + tl.arange(0, BLOCK)
    limit = tl.minimum(NP, tl.load(POS) + r + 1)
    ok = i < limit
    u = _order_key(tl.load(S + r * stride + i, mask=ok, other=float('-inf')))
    if PASS == 0:
        match = ok
    else:
        prefix = tl.load(STATE + 2 * r).to(tl.uint32, bitcast=True)
        fixed = tl.full((), (0xffffffff << (32 - 8 * PASS)) & 0xffffffff, tl.uint32)
        match = ok & ((u & fixed) == prefix)
    hist = tl.histogram(((u >> (24 - 8 * PASS)) & 255).to(tl.int32), 256, mask=match)
    bins = tl.arange(0, 256)
    tl.store(HIST + (r * NC + c) * 256 + bins, hist)


@triton.jit
def _threshold(HIST, STATE, NC: tl.constexpr, K: tl.constexpr, PASS: tl.constexpr):
    r = tl.program_id(0)
    bins = tl.arange(0, 256)
    hist = tl.zeros((256,), tl.int32)
    for base in range(0, NC, 32):
        c = base + tl.arange(0, 32)
        part = tl.load(HIST + (r * NC + c[:, None]) * 256 + bins[None, :], mask=c[:, None] < NC, other=0)
        hist += tl.sum(part, axis=0)
    if PASS == 0:
        prefix = tl.full((), 0, tl.uint32)
        need = tl.full((), K, tl.int32)
    else:
        prefix = tl.load(STATE + 2 * r).to(tl.uint32, bitcast=True)
        need = tl.load(STATE + 2 * r + 1)
    at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
    digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
    need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
    prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * PASS))
    tl.store(STATE + 2 * r, prefix)
    tl.store(STATE + 2 * r + 1, need)


@triton.jit
def _counts(S, HIST, STATE, POS, stride, NP: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    r, c = tl.program_id(0), tl.program_id(1)
    i = c * BLOCK + tl.arange(0, BLOCK)
    ok = i < tl.minimum(NP, tl.load(POS) + r + 1)
    u = _order_key(tl.load(S + r * stride + i, mask=ok, other=float('-inf')))
    prefix = tl.load(STATE + 2 * r).to(tl.uint32, bitcast=True)
    tl.store(HIST + (r * NC + c) * 256, tl.sum((ok & (u > prefix)).to(tl.int32), 0))
    tl.store(HIST + (r * NC + c) * 256 + 1, tl.sum((ok & (u == prefix)).to(tl.int32), 0))


@triton.jit
def _scatter(S, HIST, STATE, OUT, POS, stride, NP: tl.constexpr, NC: tl.constexpr,
             CP: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    r, c = tl.program_id(0), tl.program_id(1)
    chunks = tl.arange(0, CP)
    gt = tl.load(HIST + (r * NC + chunks) * 256, mask=chunks < c, other=0)
    eq = tl.load(HIST + (r * NC + chunks) * 256 + 1, mask=chunks < c, other=0)
    before_gt, before_eq = tl.sum(gt, 0), tl.sum(eq, 0)
    prefix = tl.load(STATE + 2 * r).to(tl.uint32, bitcast=True)
    need = tl.load(STATE + 2 * r + 1)
    i = c * BLOCK + tl.arange(0, BLOCK)
    ok = i < tl.minimum(NP, tl.load(POS) + r + 1)
    u = _order_key(tl.load(S + r * stride + i, mask=ok, other=float('-inf')))
    equal = (ok & (u == prefix)).to(tl.int32)
    take = (ok & (u > prefix)) | ((equal == 1) & (before_eq + tl.cumsum(equal, 0) - equal < need))
    t = take.to(tl.int32)
    at = before_gt + tl.minimum(before_eq, need) + tl.cumsum(t, 0) - t
    tl.store(OUT + r * K + at, i.to(tl.int32), mask=take)


class Scratch:
    """Buffer-owned arena, reused across serial layers and graph buckets; no host readback."""
    def __init__(self, rows, capacity, device):
        self.rows, self.capacity = rows, capacity
        self.hist = torch.empty(rows * triton.cdiv(capacity, TILE) * 256, dtype=torch.int32, device=device)
        self.state = torch.empty(rows * 2, dtype=torch.int32, device=device)


def select(scores, out, pos, npn, scratch):
    """Stable top-k of existing FP32 scores; selected IDs are ascending, ties prefer lower IDs.

    Four distributed histogram/reduce passes find the exact radix threshold. Tile
    counts then assign disjoint output ranges, retaining the legacy truncation and
    untouched output tails for dense rows. No atomics determine output ordering.
    """
    if (scores.dim() != 2 or scores.dtype != torch.float32 or scores.stride(1) != 1
            or out.dim() != 2 or out.dtype != torch.int32 or not out.is_contiguous()):
        raise ValueError("parallel selector requires FP32 score rows and contiguous int32 output")
    rows, k = out.shape
    if rows < 1 or k < 1 or npn < 1 or rows > scores.shape[0]:
        raise ValueError("parallel selector requires positive matching dimensions")
    if rows > scratch.rows or npn > scratch.capacity or npn > scores.shape[1]:
        raise ValueError('parallel selector workspace too small')
    nc = triton.cdiv(npn, TILE)
    for p in range(4):
        _hist[(rows, nc)](scores, scratch.hist, scratch.state, pos, scores.stride(0), npn, nc, p, TILE, num_warps=4)
        _threshold[(rows,)](scratch.hist, scratch.state, nc, k, p, num_warps=8)
    _counts[(rows, nc)](scores, scratch.hist, scratch.state, pos, scores.stride(0), npn, nc, TILE, num_warps=4)
    _scatter[(rows, nc)](scores, scratch.hist, scratch.state, out, pos, scores.stride(0), npn, nc,
                         triton.next_power_of_2(nc), k, TILE, num_warps=4)
