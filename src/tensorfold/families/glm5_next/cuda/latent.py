"""GLM's MLA attention on the 512-wide latent cache; kernels compute each row alone, so window rows keep serial bits (docs/recipes/glm-5.3-flash.md)."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from . import cache_fp8
from . import LATENT as ENABLED     # off (TF_GLM_LATENT=0): per-head keys and values, 0.3.5's path, for A/B

L = 512            # GLM-5.3-Flash's latent width (kv_lora_rank); the kernels take the width from the tensors
CHUNK = 512        # keys per chunk program, merged in absolute order
# Heads one sparse/dense decode program scans at once. The tiles a program holds are a launch shape, never the
# arithmetic (a head's output and the per-chunk merge are its own), so a small window can trade heads for programs:
# at R=1 the grid is (rows, heads / HB, key chunks), and with 16 heads a program a 2,048-token row leaves 4 programs
# on a 188-SM GPU. 0 keeps ``head_block``'s value; probes sweep it (TF_GLM_SPARSE_HB sets the start value).
SPARSE_HB = int(os.environ.get("TF_GLM_SPARSE_HB", "4") or 0)   # measured at 2,100 tokens: 8.93 ms vs 10.00 at 16
MLA_POLICY = os.environ.get("TF_GLM_MLA_POLICY", "legacy")
if MLA_POLICY not in ("legacy", "decode443"):
    raise ValueError("TF_GLM_MLA_POLICY: legacy or decode443")
# Empty/unknown host buckets never select a candidate. This manifest is narrowed
# to successful full-model cells before deployment; no rank-local autotuning.
PREFILL_MLA_POLICY = os.environ.get("TF_GLM_PREFILL_MLA_POLICY", "legacy")
DENSE_POLICY = os.environ.get("TF_GLM_DENSE_POLICY", "legacy")
if PREFILL_MLA_POLICY not in ("legacy", "prefill1643") or DENSE_POLICY not in ("legacy", "decode443"):
    raise ValueError("invalid MLA prefill/dense policy")
PREFILL_MLA_ROWS = (32, 128, 512, 2048)
PREFILL_MLA_BUCKETS = (4096, 8192, 16384, 32768, 65536, 131072, 229376, 229384)
MLA_BUCKETS = tuple(int(b) for b in os.environ.get("TF_GLM_MLA_BUCKETS", "8192,16384,32768,65536,131072,229376,229384").split(",") if b)


def sparse_launch(rows, heads, width, rope, dtype, selections, bucket):
    if (PREFILL_MLA_POLICY == "prefill1643" and rows in PREFILL_MLA_ROWS and heads == 16 and width == 512
            and rope == 64 and dtype == torch.uint8 and selections == 2048 and bucket in PREFILL_MLA_BUCKETS
            and SPARSE_HB == 4):
        return 16, 4, 3
    if (MLA_POLICY == "decode443" and rows in (1, 2) and heads == 16 and width == 512
            and rope == 64 and dtype == torch.uint8 and selections == 2048 and bucket in MLA_BUCKETS
            and SPARSE_HB == 4):
        return 4, 4, 3
    return (head_block(rows, width) if SPARSE_HB <= 0 else min(SPARSE_HB, heads)), 8, 1


KT = 32            # keys per tensor-core tile
HB = 16            # heads per attention tile (all from one row)
HB_WIDE = 32       # prefill chunks: a rank's 32 heads in one tile read each key once (same bits as 16, tested)


WIDE_TILE_BYTES = 72 * 1024      # operand tiles a 32-head program may hold (this GPU's block limit is ~99 KiB)


def head_block(R: int, lw: int = L) -> int:
    """Heads per attention program: a rank's 32 in prompt chunks when the tiles fit, 16 in decode windows.

    A 32-head program keeps a 32 x lw query tile and a 32 x kt key tile (~``(2 * HBT + KT) * lw * 2`` bytes): at the
    512-wide latent that is 98 KiB, past the shared memory a block gets on this GPU, so the wide tile is only used
    where it fits. Both tiles compute a row the same way, so this is a launch shape, not arithmetic.
    """
    if R < 64:
        return HB
    return HB_WIDE if (2 * HB_WIDE + KT) * lw * 2 <= WIDE_TILE_BYTES else HB


# ------------------------------------------------------------------------------------------------ weights ---

def dequant_mlx4(w: torch.Tensor, s: torch.Tensor, b: torch.Tensor, group: int = 64) -> torch.Tensor:
    """MLX affine 4-bit rows -> fp32 [out, in]: w uint32 [out, in / 8], 8 nibbles low to high; s, b [out, in / group]."""

    out, words = w.shape
    shifts = torch.arange(0, 32, 4, device=w.device, dtype=torch.int32)
    q = (w.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF                        # [out, words, 8]
    q = q.reshape(out, words * 8).to(torch.float32)
    s = s.to(torch.float32).repeat_interleave(group, dim=1)
    b = b.to(torch.float32).repeat_interleave(group, dim=1)
    return q * s + b


class AbsorbW:
    """One DSA layer's kv_b_proj split per head for the latent path: wk [H, 256, 512], wv [H, 256, 512] bf16."""

    def __init__(self, wk: torch.Tensor, wv: torch.Tensor) -> None:
        if wk.dim() != 3 or wv.dim() != 3 or wk.shape[2] != wv.shape[2]:
            raise ValueError("AbsorbW: expected [heads, dim, latent] blocks")
        self.wk = wk.to(torch.bfloat16).contiguous()
        self.wv = wv.to(torch.bfloat16).contiguous()
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]

    @classmethod
    def from_rows(cls, k_rows: torch.Tensor, v_rows: torch.Tensor, heads: int) -> "AbsorbW":
        """k_rows [heads * qk_dim, latent], v_rows [heads * v_dim, latent], float, in head order."""
        lw = k_rows.shape[1]
        return cls(k_rows.reshape(heads, -1, lw), v_rows.reshape(heads, -1, lw))

    def nbytes(self) -> int:
        return self.wk.numel() * 2 + self.wv.numel() * 2


class AbsorbQ4:
    """kv_b split per head in MLX's affine 4-bit layout: words [H, dim, latent / 8], scales and biases [H, dim, latent / 64], key rows then value rows."""

    def __init__(self, k: tuple, v: tuple, heads: int) -> None:
        def per_head(t):
            w, sc, b = t
            return (w.reshape(heads, -1, w.shape[1]).contiguous(), sc.reshape(heads, -1, sc.shape[1]).contiguous(),
                    b.reshape(heads, -1, b.shape[1]).contiguous())

        self.wkw, self.wks, self.wkb = per_head(k)
        self.wvw, self.wvs, self.wvb = per_head(v)
        self.heads, self.qk_dim = self.wkw.shape[0], self.wkw.shape[1]
        self.v_dim = self.wvw.shape[1]
        self.lw = self.wkw.shape[2] * 8
        if self.wks.shape[2] * 64 != self.lw or self.wvw.shape[2] * 8 != self.lw:
            raise ValueError("AbsorbQ4: expected groups of 64 along the latent")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.wkw, self.wks, self.wkb, self.wvw, self.wvs, self.wvb))


# ----------------------------------------------------------------------------------------- absorb, expand ---

@triton.jit
def _absorb_q(Q, WK, QA, R, H: tl.constexpr, D: tl.constexpr, KP: tl.constexpr, QSTRIDE: tl.constexpr,
              LW: tl.constexpr, STRIDE: tl.constexpr, BN: tl.constexpr):
    """Program (head, column block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n] for every row r, k in one sum.

    ``LW`` is the latent's width (how the key blocks are stored); ``STRIDE`` is QA's row stride, larger than ``LW``
    when a rotary part follows the latent columns; ``KP`` pads ``D`` up to a power of two (192 keys, 256 lanes).
    """

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, KP)
    kk = tl.where(k < D, k, 0)
    ok = k < D
    n = n0 + tl.arange(0, BN)
    w = tl.load(WK + (h * D + kk[:, None]) * LW + n[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    for r in range(R):
        q = tl.load(Q + (r * H + h) * QSTRIDE + kk, mask=ok, other=0.0).to(tl.float32)
        acc = tl.sum(q[:, None] * w, axis=0)
        tl.store(QA + (r * H + h) * STRIDE + n, acc.to(tl.bfloat16))


@triton.jit
def _expand_v(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, STRIDE: tl.constexpr,
               BN: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k] for every row r."""

    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    k = tl.arange(0, LW)
    n = n0 + tl.arange(0, BN)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)          # [BN, LW]
    for r in range(R):
        o = tl.load(OL + (r * H + h) * STRIDE + k).to(tl.float32)
        acc = tl.sum(w * o[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


@triton.jit
def _absorb_q4(Q, WW, WS, WB, QA, R, H: tl.constexpr, D: tl.constexpr, LW: tl.constexpr, STRIDE: tl.constexpr,
               RB: tl.constexpr):
    """Program (head, 64-latent group, RB rows): QA = Q @ W with W = q * s + b per group, weights loaded once for the block and each row summed alike."""
    h = tl.program_id(0)
    g = tl.program_id(1)
    rb = tl.program_id(2)
    d = tl.arange(0, D)
    j = tl.arange(0, 8)
    shifts = tl.arange(0, 8) * 4
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    words = tl.load(WW + (h * D + d[:, None]) * KW + g * 8 + j[None, :])                   # [D, 8]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (D, 64)).to(tl.float32)
    sc = tl.load(WS + (h * D + d) * KG + g).to(tl.float32)
    bi = tl.load(WB + (h * D + d) * KG + g).to(tl.float32)
    n = g * 64 + tl.arange(0, 64)
    for i in tl.static_range(RB):
        r = rb * RB + i
        ok = r < R
        qv = tl.load(Q + (r * H + h) * D + d, mask=ok & (d >= 0), other=0).to(tl.float32)
        acc = tl.sum((qv * sc)[:, None] * qint, axis=0) + tl.sum(qv * bi, axis=0)
        tl.store(QA + (r * H + h) * STRIDE + n, acc.to(tl.bfloat16), mask=ok & (n >= 0))


@triton.jit
def _expand_v4(OL, WW, WS, WB, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr,
               STRIDE: tl.constexpr, BN: tl.constexpr):
    """Program (head, BN outputs): the 4-bit rows unpacked once to fp32, then each row's output as one fp32 sum over the latent, whatever R is."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    KW: tl.constexpr = LW // 8
    KG: tl.constexpr = LW // 64
    kw = tl.arange(0, KW)
    shifts = tl.arange(0, 8) * 4
    words = tl.load(WW + (h * DV + n[:, None]) * KW + kw[None, :])                          # [BN, KW]
    qint = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (BN, KG, 64)).to(tl.float32)
    gi = tl.arange(0, KG)
    sc = tl.load(WS + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)            # [BN, KG]
    bi = tl.load(WB + (h * DV + n[:, None]) * KG + gi[None, :]).to(tl.float32)
    w = tl.reshape(qint * sc[:, :, None] + bi[:, :, None], (BN, LW))
    k = tl.arange(0, LW)
    for r in range(R):
        x = tl.load(OL + (r * H + h) * STRIDE + k).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(OUT + (r * H + h) * DV + n, acc.to(tl.bfloat16))


def row_block(R: int) -> int:
    """Rows per program: 1 for decode windows (most parallel), 16 for prefill chunks (weights reused)."""
    return 1 if R <= 16 else 16


def absorb_q(q: torch.Tensor, a, out: torch.Tensor, d: int | None = None) -> torch.Tensor:
    """q [R, H, q.stride(1)] bf16 -> out's first ``latent`` columns, [R, H, latent] bf16.

    ``d`` (default: the query's width) says how many key columns of a row the absorbed weights cover: with a rotary
    part, only the first ``qk_nope`` of a ``qk_dim``-wide query row is absorbed.
    """

    R, H, D = q.shape
    if isinstance(a, AbsorbQ4):
        rb = row_block(R)
        _absorb_q4[(H, a.lw // 64, triton.cdiv(R, rb))](q, a.wkw, a.wks, a.wkb, out, R, H=H, D=D, LW=a.lw, RB=rb,
                                                        STRIDE=out.stride(1), num_warps=4)
        return out
    BN = 32
    _absorb_q[(H, a.lw // BN)](q, a.wk, out, R, H=H, D=(d or D), KP=triton.next_power_of_2(d or D),
                                  QSTRIDE=q.stride(1), LW=a.lw, STRIDE=out.stride(1), BN=BN, num_warps=4)
    return out


def expand_v(o_lat: torch.Tensor, a, out: torch.Tensor) -> torch.Tensor:
    """o_lat [R, H, latent] bf16 (stride from the tensor) -> out [R, H, v_dim] bf16."""

    R, H, _ = o_lat.shape
    if isinstance(a, AbsorbQ4):
        BN = 16
        _expand_v4[(H, a.v_dim // BN)](o_lat, a.wvw, a.wvs, a.wvb, out, R, H=H, DV=a.v_dim, LW=a.lw,
                                       STRIDE=o_lat.stride(1), BN=BN, num_warps=4)
        return out
    BN = 16
    _expand_v[(H, a.v_dim // BN)](o_lat, a.wv, out, R, H=H, DV=a.v_dim, LW=a.lw, STRIDE=o_lat.stride(1), BN=BN,
                                 num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ caches ---

@triton.jit
def _lat_write(LAT, lat_stride, LC, POS, VALID, W: tl.constexpr, OFF: tl.constexpr, CSTRIDE: tl.constexpr, MASKED: tl.constexpr):
    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    k = tl.arange(0, W)
    live = tl.load(VALID + r) != 0 if MASKED else True
    tl.store(LC + (P + r) * CSTRIDE + OFF + k, tl.load(LAT + r * lat_stride + k), mask=live)


def latent_write(lat: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, *, off: int = 0, valid: torch.Tensor | None = None) -> None:
    """lat [R, W] bf16 rows into cache slots pos .. pos + R - 1 at column ``off`` (pos read on the device).

    The cache row is wider than the latent when a rotary key part follows it (GLM-5.3), so the copy's width and the
    cache's stride are separate.
    """

    if cache.dtype == torch.uint8:
        if off:
            raise ValueError("FP8 cache writes must cover the whole row")
        return cache_fp8.write(lat, cache, pos, valid)
    if lat.stride(1) != 1 or cache.stride(1) != 1:
        raise ValueError("latent_write: rows must be contiguous")
    _lat_write[(lat.shape[0],)](lat, lat.stride(0), cache, pos, pos if valid is None else valid, W=lat.shape[1], OFF=off,
                                CSTRIDE=cache.shape[1], MASKED=valid is not None, num_warps=4)


# --------------------------------------------------------------------------------------------- attention ---

@triton.jit
def _tile(q, kv, m, l, o, valid, qp, kp, SCALE: tl.constexpr, ROPE: tl.constexpr):
    """One key tile for HB heads of one row: kv [KT, 512] is both key and value (the latent).

    GLM-5.3 adds a second term, ``qp . kp``, over the 64-wide rotary part every head shares (``ROPE`` is its width,
    0 for a NoPE model); both terms carry the same ``SCALE``, as one dot over the concatenated rows would.
    """
    scores = tl.dot(q, tl.trans(kv)).to(tl.float32)
    if ROPE > 0:
        scores += tl.dot(qp, tl.trans(kp)).to(tl.float32)
    scores = scores * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _dense_chunks(QA, LC, QP, KP, POS, PO, PM, PL, R, H: tl.constexpr, LW: tl.constexpr, CH: tl.constexpr,
                  SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr, ROPE: tl.constexpr, FP8: tl.constexpr):
    """Program (row, head block, chunk): causal attention of HB heads of row r over keys [c CH, (c + 1) CH)."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    P = tl.load(POS)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H                                                      # tile rows past the last head are padding
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    start = c * CH
    limit = P + r                                                     # keys 0 .. P + r are visible to row r
    if start <= limit:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        f = tl.arange(0, ROPE if ROPE > 0 else 2)
        qp = tl.load(QP + (r * H + hh[:, None]) * ROPE + f[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        # only the tiles this row can still see: an all-masked tile adds nothing (its p is zero and m is unchanged),
        # and at a decode window the tail of a chunk is almost all of it (one row, up to 512 keys masked)
        nt = tl.cdiv(tl.minimum(limit, start + CH - 1) + 1 - start, KTT)
        for t in range(nt):                    # t is a tile index; apply KTT exactly once below
            ki = start + t * KTT + tl.arange(0, KTT)
            ok = ki <= limit
            if FP8:
                kv = cache_fp8.load_rows(LC, ki, k, ok, LW)
                kp = cache_fp8.load_rows(KP, ki, f, ok, ROPE)
            else:
                kv = tl.load(LC + ki[:, None].to(tl.int64) * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
                kp = tl.load(KP + ki[:, None].to(tl.int64) * ROPE + f[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, qp, kp, SCALE, ROPE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _sparse_chunks(QA, LC, QP, KP, TOK, CNT, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, LW: tl.constexpr,
                   CH: tl.constexpr, SCALE: tl.constexpr, HBT: tl.constexpr, KTT: tl.constexpr,
                   ROPE: tl.constexpr, FP8: tl.constexpr):
    """Program (row, head block, chunk): HB heads of row r over its selected tokens [c CH, (c + 1) CH) in list order."""
    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    k = tl.arange(0, LW)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, LW), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        f = tl.arange(0, ROPE if ROPE > 0 else 2)
        qp = tl.load(QP + (r * H + hh[:, None]) * ROPE + f[None, :], mask=hok[:, None], other=0).to(tl.bfloat16)
        # only the tiles that hold selected tokens (a masked tile contributes nothing, as in the dense path)
        # Each chunk owns at most CH selections; scanning the tail here would count later chunks twice.
        nt = tl.cdiv(tl.minimum(n - c * CH, CH), KTT)
        for t in range(nt):                    # t is a tile index; apply KTT exactly once below
            idx = c * CH + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            tok = tl.load(TOK + r * W + idx, mask=ok, other=0).to(tl.int64)
            if FP8:
                kv = cache_fp8.load_rows(LC, tok, k, ok, LW)
                kp = cache_fp8.load_rows(KP, tok, f, ok, ROPE)
            else:
                kv = tl.load(LC + tok[:, None] * LW + k[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
                kp = tl.load(KP + tok[:, None] * ROPE + f[None, :], mask=ok[:, None], other=0).to(tl.bfloat16)
            m, l, o = _tile(q, kv, m, l, o, ok, qp, kp, SCALE, ROPE)
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * LW + k[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _merge(PO, PM, PL, OUT, CNT, R, H: tl.constexpr, LW: tl.constexpr, NCH: tl.constexpr, SPARSE: tl.constexpr):
    """Program (row, head): the row's chunk partials in chunk order -> OUT[r, h] bf16. Sparse: rows with CNT 0 skip."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    if SPARSE:
        if tl.load(CNT + r) == 0:
            return
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * LW + k)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    tl.store(OUT + (r * H + h) * LW + k, (o / l).to(tl.bfloat16))


class LatentScratch:
    """Chunk partials for ``part_rows`` (one ``attention`` call) x heads x chunks; queries and latents for ``rows``."""

    def __init__(self, rows: int, heads: int, chunks: int, device, lw: int = L, part_rows: int | None = None,
                 rope: int = 0) -> None:
        part_rows = rows if part_rows is None else min(rows, part_rows)
        self.rows, self.part_rows, self.heads, self.nch, self.lw = rows, part_rows, heads, chunks, lw
        self.po = torch.empty((chunks * part_rows * heads * lw,), dtype=torch.float32, device=device)
        self.pm = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.pl = torch.empty((chunks * part_rows * heads,), dtype=torch.float32, device=device)
        self.qa = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.qpe = torch.empty((rows, heads, max(rope, 1)), dtype=torch.bfloat16, device=device)
        self.ol = torch.empty((rows, heads, lw), dtype=torch.bfloat16, device=device)
        self.dummy = torch.zeros((1,), dtype=torch.int32, device=device)


def attention(qa: torch.Tensor, cache: torch.Tensor, pos: torch.Tensor, s: LatentScratch, *, scale: float,
              nch: int, out: torch.Tensor, hb: int | None = None, qpe: torch.Tensor | None = None,
              kpe: torch.Tensor | None = None) -> torch.Tensor:
    """Dense causal attention of qa [R, H, 512] through pos + R - 1 in nch 512-key chunks; a row ignores the others."""
    R, H, LW = qa.shape
    if nch > s.nch or R > s.part_rows or LW != s.lw:
        raise ValueError(f"latent attention: {R} rows, {nch} chunks, width {LW} past the scratch's "
                         f"{s.part_rows}, {s.nch}, {s.lw}")
    n = nch * R * H
    automatic = hb is None
    hb = head_block(R, LW) if automatic else hb
    if hb not in (HB, HB_WIDE):
        raise ValueError(f"latent attention: {hb} heads a program, not {HB} or {HB_WIDE}")
    rope = 0 if qpe is None else qpe.shape[2]
    if cache.dtype == torch.uint8 and (rope <= 0 or kpe is None or kpe.dtype != torch.uint8):
        raise ValueError("FP8 MLA requires full GLM latent and rotary caches")
    warps, stages = 8, 1
    if (automatic and DENSE_POLICY == "decode443" and R in (1, 2) and H == 16 and LW == 512
            and rope == 64 and cache.dtype == torch.uint8 and nch <= 6):
        hb, warps, stages = 4, 4, 3
    _dense_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, s.qpe if qpe is None else qpe,
                                                s.qpe if kpe is None else kpe, pos, s.po[:n * LW], s.pm[:n],
                                                s.pl[:n], R, H=H, LW=LW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT,
                                                ROPE=rope, FP8=cache.dtype == torch.uint8, num_warps=warps, num_stages=stages)
    _merge[(R, H)](s.po, s.pm, s.pl, out, s.dummy, R, H=H, LW=LW, NCH=nch, SPARSE=False, num_warps=4)
    return out


def sparse_attention(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor,
                     out: torch.Tensor, scale: float, qpe: torch.Tensor | None = None,
                     kpe: torch.Tensor | None = None, scratch: LatentScratch | None = None, *, bucket: int | None = None) -> None:
    """Attention of rows with counts > 0 over their selected tokens (ascending, -1 padded), written into out; other rows are left alone."""
    R, H, LW = qa.shape
    W = tokens.shape[1]
    nch = triton.cdiv(W, CHUNK)
    n = nch * R * H
    if scratch is not None and R <= scratch.part_rows and nch <= scratch.nch:
        po, pm, pl = scratch.po[:n * LW], scratch.pm[:n], scratch.pl[:n]
    else:
        po = torch.empty((n * LW,), dtype=torch.float32, device=qa.device)
        pm = torch.empty((n,), dtype=torch.float32, device=qa.device)
        pl = torch.empty((n,), dtype=torch.float32, device=qa.device)
    rope = 0 if qpe is None else qpe.shape[2]
    hb, warps, stages = sparse_launch(R, H, LW, rope, cache.dtype, W, bucket)
    if cache.dtype == torch.uint8 and (rope <= 0 or kpe is None or kpe.dtype != torch.uint8):
        raise ValueError("FP8 MLA requires full GLM latent and rotary caches")
    # No scratch object exists on this path. With ROPE=0 Triton eliminates the rotary loads.
    _sparse_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, qa if qpe is None else qpe,
                                                 cache if kpe is None else kpe, tokens, counts, po, pm, pl, R,
                                                 W=W, H=H, LW=LW, CH=CHUNK, SCALE=scale, HBT=hb, KTT=KT, ROPE=rope, FP8=cache.dtype == torch.uint8,
                                                 num_warps=warps, num_stages=stages)
    _merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)


def chunks_for(length: int) -> int:
    return triton.cdiv(length, CHUNK)
