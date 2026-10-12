"""GLM-5.3's DSA indexer (per token, no pooling): normalise and rotate each window's index keys into the cache, then
select every row's top ``index_topk`` visible tokens.

GLM-5.3-Flash folds four tokens into one index pool (``index_kpool``, a learned pool bias, TensorFold's
``sparse.py``); GLM-5.3 has no pooling tensors at all and scores every visible token instead, one layer in four
(``indexer_types``: 21 ``full``, the rest reuse their block's selection). The score is DeepSeek's indexer, as the
reference implements it (``modules/mla_attn.py``: ``_indexer_keys``/``_indexer_topk``, ``attention_fn/dsa_triton.py``):

    q = wq_b(q_a) rotated on its first qk_rope columns   (no LayerNorm)
    w = weights_proj(x)                                  (raw head weights, fp16 in the reference)
    k = Layernorm(wk(x)) rotated on its first qk_rope columns
    s = sum_h relu(q_h . k) * w_h * (index_dim ** -0.5 * index_heads ** -0.5)   over the row's visible tokens

The rotation is the reference's: only the leading ``qk_rope_head_dim`` (64) columns of the 128-wide heads turn, with
the main attention's own frequency table (``inv_freq`` holds ``qk_rope / 2`` = 32 pairs), interleaved pairs
(``indexer_rope_interleave``). Rotating all 128 columns with a 128-wide table would be a different model.

Dense/sparse boundary: the reference computes sparsely once a row's visible length exceeds ``index_topk``
(``sparse = max(host_seqlens) + seqlen > index_topk``); ``Config.dense_limit`` is that same number while
``kpool == 1``, so ``select_tokens`` serves exactly the rows past it and reports 0 for the rest.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

SEL_BN = 256          # tokens a scoring program covers: its key tile is BN x dim fp16, and 512 needed 128 KiB of
#                      shared memory (past this GPU's ~99 KiB block limit). A block only splits the scan, so the
#                      scores, and with them every row's selection, are the same at any block size.
SEL_BLOCK = 1024      # tokens a radix-select pass scans
SELECTOR_MODE = os.environ.get("TF_GLM_SELECTOR_MODE", "legacy")
if SELECTOR_MODE not in ("legacy", "parallel"):
    raise ValueError("TF_GLM_SELECTOR_MODE: legacy or parallel")


def selection_scratch(rows, capacity, device):
    """One reusable integer workspace per decode buffer, shared by its serial layers."""
    if SELECTOR_MODE != "parallel":
        return None
    from .selection53 import Scratch
    return Scratch(min(rows, 8), capacity, device)


@triton.jit
def _index_write(KR, GR, LNW, LNB, INV, IK, IG, POS, VALID, eps, D: tl.constexpr, W: tl.constexpr, NF: tl.constexpr,
                 WSTR: tl.constexpr, MASKED: tl.constexpr):
    """One program a row: LayerNorm the index key, rotate its leading pairs, and write the row's key and head weights.

    The rotation is elementwise: element ``j``'s partner is ``j ^ 1``, both carrying the same angle (``inv[j // 2]``),
    so ``out[j] = y[j] * cos -/+ y[j ^ 1] * sin`` with the sign of an odd ``j`` flipped. Only pairs below ``NF`` turn
    (``2 * NF`` columns, the reference's ``qk_rope_head_dim``); both elements are normalised under the row's one mean
    and variance. The key is cached in fp16 (the reference's plane dtype) and the head weights in fp32.
    """

    r = tl.program_id(0)
    P = tl.load(POS).to(tl.int64) + r
    d = tl.arange(0, D)
    x = tl.load(KR + r * D + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)
    if NF > 0:
        o = d ^ 1                                        # this element's rotary partner, also normalised
        xo = tl.load(KR + r * D + o).to(tl.float32)
        xoc = xo - mean
        yo = xoc * rstd * tl.load(LNW + o).to(tl.float32) + tl.load(LNB + o).to(tl.float32)
        i = d // 2
        inv = tl.load(INV + i, mask=i < NF, other=0.0).to(tl.float64)
        ang = P.to(tl.float64) * inv
        c = tl.cos(ang).to(tl.float32)
        s = tl.sin(ang).to(tl.float32)
        sign = tl.where((d & 1) == 1, 1.0, -1.0)
        y = tl.where(i < NF, y * c + sign * yo * s, y)
    live = tl.load(VALID + r) != 0 if MASKED else True
    tl.store(IK + P * D + d, y.to(tl.float16), mask=live)
    if WSTR:                                 # the latent path selects with the *current* row's weights (forward.py's
        g = tl.arange(0, W)                  # _selection), so a caller that never reads the stored ones skips both
        tl.store(IG + P * W + g, tl.load(GR + r * W + g), mask=live)


def index_update(k_raw: torch.Tensor, wts: torch.Tensor | None, ln_w: torch.Tensor, ln_b: torch.Tensor,
                 inv_freq: torch.Tensor, interleave: bool, ik: torch.Tensor, ig: torch.Tensor | None = None,
                 pos: torch.Tensor = None, eps: float = 1e-6, valid: torch.Tensor | None = None) -> None:
    """Window rows' index keys (normalised and rotated) and their head weights into the caches at pos..

    ``k_raw`` [R, dim] bf16 (``wk`` of the normed rows), ``wts`` [R, heads] fp32 (``weights_proj`` of them),
    ``ik`` [capacity, dim] fp16 and ``ig`` [capacity, heads] fp32 the caches, ``pos`` the device position of row 0.
    ``ig=None`` caches the keys alone: the latent path (``_dsa_latent``) selects with the current row's weights and
    never reads the stored ones, so a caller there passes no weights and skips both the projection and the store.
    """

    if not interleave:
        raise NotImplementedError("GLM-5.3's indexer uses interleaved rotary pairs (indexer_rope_interleave)")
    R, dim = k_raw.shape
    heads = 0 if ig is None else int(wts.shape[1])
    if k_raw.dtype != torch.bfloat16 or not k_raw.is_contiguous() or (ig is not None and wts.dtype != torch.float32):
        raise ValueError(f"index_update: k_raw must be contiguous bf16 and wts fp32, got {k_raw.dtype} {wts.dtype}")
    if ik.dtype != torch.float16 or (ig is not None and ig.dtype != torch.float32):
        raise ValueError(f"index_update: the caches must be fp16 keys and fp32 weights, got {ik.dtype} {ig.dtype}")
    nf = int(inv_freq.numel())
    if 2 * nf > dim:
        raise ValueError(f"index_update: {2 * nf} rotated columns do not fit a {dim}-wide key")
    _index_write[(R,)](k_raw, wts if ig is not None else k_raw, ln_w, ln_b, inv_freq, ik,
                       ig if ig is not None else ik, pos, pos if valid is None else valid, eps, D=dim, W=heads, NF=nf,
                       WSTR=ig is not None, MASKED=valid is not None, num_warps=4)


@triton.jit
def _sel_scores(QI, W, w_stride, IK, OUT, score_stride, POS, R, NP, scale, H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr,
                BN: tl.constexpr):
    """Program (row, token block): ``s_t = scale * sum_h relu(q_h . k_t) * w_h`` over the row's visible tokens.

    The reference's few-query kernel: heads are the dot's M dimension, the dot runs in fp16 (the query is bf16, which
    fp16 holds exactly; the key plane is fp16), the relu-weighted reduction in fp32, and one multiply by
    ``index_dim ** -0.5 * index_heads ** -0.5`` after the sum. Tokens at or past the row's bound score ``-inf``, so a
    row's selection never depends on which rows share its window.
    """

    r = tl.program_id(0)
    pb = tl.program_id(1)
    P = tl.load(POS)
    offs = pb * BN + tl.arange(0, BN)
    ok = offs < NP
    bound = tl.minimum(P + r + 1, NP)
    hh = tl.arange(0, HP)
    hok = hh < H
    d = tl.arange(0, D)
    qh = tl.load(QI + (r * H + hh[:, None]) * D + d[None, :], mask=hok[:, None], other=0.0).to(tl.float16)
    kt = tl.load(IK + offs[None, :] * D + d[:, None], mask=ok[None, :], other=0.0)
    logits = tl.dot(qh, kt)
    wh = tl.load(W + r * w_stride + hh, mask=hok, other=0.0)
    acc = tl.sum(tl.maximum(logits, 0.0) * wh[:, None].to(tl.float32), axis=0) * scale
    acc = tl.where(ok & (offs < bound), acc, float("-inf"))
    tl.store(OUT + r * score_stride + offs, acc, mask=ok)


@triton.jit
def _order_key(s):
    """A float32 score as a uint32 whose unsigned order is the scores' order (-0 counted as +0)."""
    bits = (s + 0.0).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _select_rows(S, OUT, score_stride, NP, POS, K: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: a radix select (8 bits a pass) finds the K-th best score, then one pass in token order writes the
    scores above it and the lowest ties, so the row's tokens come out ascending.

    Ties keep the lower token index (the pooled selector's rule): the second pass walks tokens in order and takes
    equal keys until K is filled. Only tokens below the row's own bound are visible; the rest scored ``-inf``.
    """

    r = tl.program_id(0).to(tl.int64)
    row = S + r * score_stride
    limit = tl.minimum(NP, tl.load(POS) + r + 1)
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, limit, BLOCK):
            i = c + tl.arange(0, BLOCK)
            ok = i < limit
            u = _order_key(tl.load(row + i, mask=ok, other=float("-inf")))
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist         # keys with this digit or a higher one
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, limit, BLOCK):
        i = c + tl.arange(0, BLOCK)
        ok = i < limit
        u = _order_key(tl.load(row + i, mask=ok, other=float("-inf")))
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * K + written + tl.cumsum(t, 0) - t, i.to(tl.int32), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)


def score_bucket(pos: int, R: int, np_max: int) -> int:
    """Tokens a scoring pass covers for rows ``pos .. pos + R - 1``: the visible ones rounded up to a power of two
    (at least 1024, at most the cache's), so a captured graph is reused across nearby positions.

    A bucket never drops a visible token; a shorter one would select from an incomplete set, which is why the
    captured path picks its bucket from the running position rather than a fixed constant.
    """

    visible = pos + R
    return min(np_max, max(1024, 1 << (visible - 1).bit_length()))


def buckets(np_max: int) -> list[int]:
    """Every value ``score_bucket`` can return up to ``np_max`` (the graph capture set)."""
    out, b = [], 1024
    while b < np_max:
        out.append(b)
        b *= 2
    return out + [np_max]


@triton.jit
def _counts(POS, OUT, R: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(POS).to(tl.int64)
    tl.store(OUT + r, tl.where(pos + r + 1 > K, K, 0), mask=r < R)


def select_tokens(qi: torch.Tensor, wts: torch.Tensor, ik: torch.Tensor, pos_dev: torch.Tensor, R: int, k: int,
                  np_max: int, *, bucket: int | None = None, scores: torch.Tensor | None = None,
                  tokens: torch.Tensor | None = None, counts: torch.Tensor | None = None,
                  parallel_scratch=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's ``k`` best visible tokens [R, k] int32 ascending (-1 padded) and its attention count.

    ``counts[r]`` is ``k`` for rows whose visible length passes ``k`` (the sparse rows) and 0 for the rows the dense
    pass covers, so a window straddling the boundary attends densely below it and sparsely above. ``qi`` [R, heads,
    dim] bf16 holds rotated queries, ``wts`` [R, heads] fp32 the raw head weights, ``ik`` [capacity, dim] fp16 the
    rotated keys. ``bucket`` fixes the tokens a pass scans (a captured graph; ``score_bucket`` sizes it); ``scores``
    and ``tokens`` are a slot's preallocated scratch, so a graph replay allocates nothing.
    """

    if qi.dim() != 3 or qi.dtype != torch.bfloat16 or not qi.is_contiguous():
        raise ValueError(f"select_tokens: qi must be a contiguous bf16 [R, heads, dim], got {tuple(qi.shape)} "
                         f"{qi.dtype}")
    if wts.dtype != torch.float32 or wts.stride(1) != 1 or wts.shape[0] != R:
        raise ValueError("select_tokens: wts must be fp32 rows with unit-stride columns")
    H, D = wts.shape[1], qi.shape[2]
    if ik.dtype != torch.float16 or ik.dim() != 2 or ik.shape[1] != D or not ik.is_contiguous():
        raise ValueError("select_tokens: the key cache must be contiguous fp16 [capacity, dim]")
    if R < 1 or k < 1 or qi.shape[:2] != (R, H):
        raise ValueError("select_tokens: positive R/k and matching query/weight rows and heads required")
    # A captured graph fixes the tokens a pass scans (``bucket``). A row the sparse pass consumes needs ``k`` ids
    # from that scan: with a narrower one the write loop stops short of k and leaves stale ids in ``tokens`` while the
    # row's count still says k, and the sparse attention gathers keys at those ids (an illegal access for large stale
    # values). Scanning past the row's position is safe — those tokens score -inf — so the scan is never narrower
    # than k. In production ``score_bucket`` already returns at least the visible length, so this only widens the
    # degenerate case (a graph captured at a position deeper than its bucket).
    # Shared prefill scratch can be much larger than this request's private key plane.
    # Causal score masking happens AFTER the key load; only this physical bound makes it safe.
    capacity = min(int(np_max), int(ik.shape[0]))
    if capacity < 1:
        raise ValueError("select_tokens: positive scan capacity required")
    npn = capacity if bucket is None else min(capacity, max(int(bucket), int(k)))
    if scores is not None and (scores.dim() != 2 or scores.dtype != torch.float32
                               or scores.shape[0] < 1 or scores.shape[1] < npn
                               or scores.stride(1) != 1 or scores.stride(0) < npn):
        raise ValueError("select_tokens: scores must be fp32 rows with unit-stride columns and scan capacity")
    out = tokens if tokens is not None else torch.empty((R, k), dtype=torch.int32, device=qi.device)
    if out.shape != (R, k) or out.dtype != torch.int32 or not out.is_contiguous():
        raise ValueError(f"select_tokens: tokens must be an int32 [{R}, {k}], got {tuple(out.shape)} {out.dtype}")
    slab = out.shape[0] if scores is None else scores.shape[0]
    slab = max(1, min(slab, R))
    sc = (scores if scores is not None
          else torch.empty((slab, npn), dtype=torch.float32, device=qi.device))[:, :npn]
    scale = float(D) ** -0.5 * float(H) ** -0.5
    for a in range(0, R, slab):
        n = min(slab, R - a)
        at = pos_dev if a == 0 else pos_dev + a
        _sel_scores[(n, triton.cdiv(npn, SEL_BN))](qi[a:a + n], wts[a:a + n], wts.stride(0), ik, sc[:n], sc.stride(0), at, n, npn,
                                                   scale, H=H, HP=max(16, triton.next_power_of_2(H)), D=D,
                                                   BN=SEL_BN, num_warps=4)
        if SELECTOR_MODE == "parallel" and parallel_scratch is not None and n <= 8 and npn >= 16384:
            # The scores and tie rule are unchanged. Large prefill slabs retain
            # their existing row-parallel selector and allocate no extra arena.
            from .selection53 import select
            select(sc[:n], out[a:a + n], at, npn, parallel_scratch)
        else:
            _select_rows[(n,)](sc[:n], out[a:a + n], sc.stride(0), npn, at, K=k, BLOCK=SEL_BLOCK, num_warps=4)
    counts = counts if counts is not None else torch.empty((R,), dtype=torch.int32, device=qi.device)
    _counts[(triton.cdiv(R, 128),)](pos_dev, counts, R, k, 128)
    return out, counts


__all__ = ["buckets", "index_update", "score_bucket", "select_tokens"]
