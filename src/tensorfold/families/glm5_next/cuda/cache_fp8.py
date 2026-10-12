"""Opt-in full-GLM MLA cache: E4M3FN payload + one FP32 scale per row.

A contiguous uint8 row holds W FP8 bytes followed by the scale's four raw bytes.
W is a multiple of four, so scales are aligned. Ordinary byte copies/clones and
prefix snapshots preserve both codes and scales together; no second allocator.
Main/MTP latents and rotary keys are quantized independently. Index keys remain
FP16; attention dequantizes to BF16 before the existing dot/softmax arithmetic.
This changes model numerics versus BF16; exactness is relative to FP8 serial.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

DTYPES = ("bf16", "fp8")


def row_bytes(width: int, dtype: str) -> int:
    if dtype not in DTYPES:
        raise ValueError(f"GLM KV dtype must be bf16 or fp8, not {dtype!r}")
    if dtype == "fp8" and (width <= 0 or width % 4):
        raise ValueError("FP8 cache width must be positive and divisible by four")
    return width * 2 if dtype == "bf16" else width + 4


def allocate(capacity: int, width: int, device, dtype: str):
    size = row_bytes(width, dtype)
    return torch.zeros((capacity, width if dtype == "bf16" else size),
                       dtype=torch.bfloat16 if dtype == "bf16" else torch.uint8, device=device)


@triton.jit
def load_rows(CACHE, rows, cols, mask, W: tl.constexpr):
    base = CACHE + rows.to(tl.int64) * (W + 4)
    value = tl.load(base[:, None].to(tl.pointer_type(tl.float8e4nv)) + cols[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
    scale = tl.load((base + W).to(tl.pointer_type(tl.float32)), mask=mask, other=0)
    return (value * scale[:, None]).to(tl.bfloat16)


@triton.jit
def _write(X, CACHE, POS, VALID, STRIDE: tl.constexpr, W: tl.constexpr, B: tl.constexpr, MASKED: tl.constexpr):
    r = tl.program_id(0)
    live = tl.load(VALID + r) != 0 if MASKED else True
    p = tl.load(POS).to(tl.int64) + r
    j = tl.arange(0, B)
    x = tl.load(X + r * STRIDE + j, mask=j < W, other=0).to(tl.float32)
    # FP32 division (not a reciprocal approximation) fixes the quantization rule.
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), 0), 1.0e-12), 448.0)
    q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0).to(tl.float8e4nv)
    base = CACHE + p * (W + 4)
    tl.store(base.to(tl.pointer_type(tl.float8e4nv)) + j, q, mask=live & (j < W))
    tl.store((base + W).to(tl.pointer_type(tl.float32)), scale, mask=live)


def write(x, cache, pos, valid=None):
    width = x.shape[1]
    if cache.dtype != torch.uint8 or cache.shape[1] != row_bytes(width, "fp8"):
        raise ValueError("FP8 cache must contain W codes and four scale bytes per row")
    if x.stride(1) != 1 or not cache.is_contiguous():
        raise ValueError("FP8 cache and input rows must be contiguous")
    _write[(x.shape[0],)](x, cache, pos, pos if valid is None else valid,
                         STRIDE=x.stride(0), W=width, B=triton.next_power_of_2(width),
                         MASKED=valid is not None, num_warps=4)
