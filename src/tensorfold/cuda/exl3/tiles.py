"""Stable nonempty member tiles, with a graph-safe upper bound and no readback."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def tile_capacity(picks: int, experts: int) -> int:
    if picks < 0 or experts < 1:
        raise ValueError("nonnegative picks and positive experts required")
    # Sum ceil(stored_members/16) <= ceil(picks/16) + nonempty_experts.
    capacity = (picks + 15) // 16 + min(experts, picks)
    if capacity > 2**31 - 1:
        raise ValueError("tile capacity exceeds int32")
    return max(1, capacity)


@triton.jit
def _offsets(MEM, UC, OFF, COUNT, E: tl.constexpr, MAXM: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr):
    u = tl.arange(0, B)
    active = u < tl.minimum(tl.load(UC), E)
    # Grouping stores a contiguous prefix, including its truncation. Do not
    # count raw picks or inspect unowned member rows.
    lo = tl.zeros((B,), tl.int32)
    hi = tl.full((B,), MAXM, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(MEM + u * MAXM + mid, mask=active & (mid < MAXM), other=-1)
        present = active & (value >= 0)
        lo = tl.where(present, mid + 1, lo)
        hi = tl.where(present, hi, mid)
    nt = tl.where(active, tl.cdiv(lo, 16), 0)
    ends = tl.cumsum(nt)
    tl.store(OFF + u, ends - nt, mask=u < E)
    tl.store(OFF + E, tl.sum(nt, 0))
    tl.store(COUNT, tl.sum(nt, 0))


@triton.jit
def _descriptors(OFF, DESC, E: tl.constexpr, MT: tl.constexpr, B: tl.constexpr):
    u = tl.program_id(0)
    begin = tl.load(OFF + u)
    end = tl.load(OFF + u + 1)
    for base in range(0, MT, B):
        t = base + tl.arange(0, B)
        ok = (t < MT) & (begin + t < end)
        tl.store(DESC + 2 * (begin + t), u, mask=ok)
        tl.store(DESC + 2 * (begin + t) + 1, t, mask=ok)


class Scratch:
    def __init__(self, rows: int, slots: int, experts: int, device):
        self.capacity = tile_capacity(rows * slots, experts)
        self.descriptors = torch.empty((self.capacity, 2), device=device, dtype=torch.int32)
        self.count = torch.empty(1, device=device, dtype=torch.int32)
        self.offsets = torch.empty((min(rows * slots, experts) + 1,), device=device, dtype=torch.int32)

    def build(self, members: torch.Tensor, ucount: torch.Tensor, picks: int):
        if members.dtype != torch.int32 or members.ndim != 2 or not members.is_contiguous():
            raise ValueError("members must be contiguous int32 [experts, max_members]")
        e, maxm = members.shape
        if e < 1 or maxm < 1 or e + 1 > self.offsets.numel():
            raise ValueError("member shape exceeds tile workspace")
        capacity = tile_capacity(picks, e)
        if capacity > self.capacity:
            raise ValueError("picks exceed tile workspace")
        _offsets[(1,)](members, ucount, self.offsets, self.count, E=e, MAXM=maxm,
                        B=triton.next_power_of_2(e), STEPS=maxm.bit_length(), num_warps=4)
        _descriptors[(e,)](self.offsets, self.descriptors, E=e, MT=triton.cdiv(maxm, 16), B=128, num_warps=4)
        return self.descriptors[:capacity], self.count
