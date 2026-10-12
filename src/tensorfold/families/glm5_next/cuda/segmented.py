"""GLM full's packed forward: shared projections/MoE/collectives, private contiguous MLA caches.

This is a model adapter, not a second scheduler or allocator. Segments use TensorFold's
(state, first_row, end_row) convention. Solo kernels and their arithmetic remain the oracle.
"""
from __future__ import annotations

import copy
from typing import Sequence

import torch

from . import forward as fw, glue, latent, rope


def layout(w, b, windows, *, mtp=False):
    if not w.cfg.full or not latent.ENABLED:
        raise ValueError("segmented GLM requires full GLM and latent attention")
    if not 1 <= len(windows) <= 4 or len({id(x[0]) for x in windows}) != len(windows):
        raise ValueError("one nonempty window per distinct State, at most four")
    segs, ids = [], []
    for st, tokens, *_ in windows:
        n = len(tokens)
        if not n:
            raise ValueError("empty segment")
        fw.check_room(w, st, n, pos=st.mtp_len if mtp else st.pos)
        segs.append((st, len(ids), len(ids) + n))
        ids.extend(tokens)
    if len(ids) > b.rows:
        raise ValueError(f"packed rows {len(ids)} exceed buffers {b.rows}")
    return segs, ids


def stage_streams(w, b, windows, *, padded=False):
    """Stage exact windows, optionally bucketed to reuse private-State graph layouts.

    Dummy rows are excluded from routing and must never be sampled or committed.
    A padding bucket never exceeds the existing physical cache or buffer capacity.
    """
    original = windows
    if padded and windows:
        largest = max(len(tokens) for _, tokens in windows)
        width = 1 if largest == 1 else 3 if largest <= 3 else 8
        padded = (largest <= 8 and width * len(windows) <= b.rows and
                  all(st.pos + width <= st.capacity for st, _ in windows))
        if padded:
            windows = [(st, list(tokens) + [0] * (width - len(tokens))) for st, tokens in windows]
    segs, ids = layout(w, b, windows)
    b.staged.synchronize()
    b.ids_host[:len(ids)].numpy()[:] = ids
    b.ids[:len(ids)].copy_(b.ids_host[:len(ids)], non_blocking=True)
    if padded:
        if b.valid is None:
            b.valid = torch.empty(b.rows, device=w.device, dtype=torch.int32)
        b.valid[:len(ids)].zero_()
        for (_, tokens), (_, lo, _) in zip(original, segs):
            b.valid[lo:lo + len(tokens)].fill_(1)
    elif b.valid is not None:
        b.valid[:len(ids)].fill_(1)
    b.staged.record()
    return segs


def logical_segments(windows, physical):
    """The original row ranges inside a possibly padded physical layout."""
    if len(windows) != len(physical):
        raise ValueError("one logical window per physical segment")
    result = []
    for (st, tokens), (owner, lo, z) in zip(windows, physical):
        if st is not owner or not 0 < len(tokens) <= z - lo:
            raise ValueError("logical rows must be a nonempty prefix of their own segment")
        result.append((st, lo, lo + len(tokens)))
    return result


def sampling_logits(logits, physical, logical):
    """Compact only valid logits; the shared sampler still gets original positions/seeds."""
    if all(p[2] == q[2] for p, q in zip(physical, logical)):
        return logits
    return torch.cat([logits[lo:z] for _, lo, z in logical])


def row_view(b, a, z):
    """Non-owning activation views; only sequential attention calls share chunk partials."""
    v = copy.copy(b)
    names = ("normed", "xs", "dp", "qr", "xs_qr", "lat", "xs_lat", "q", "kpe", "vn",
             "ikr", "igr", "qi", "sel_tok", "sel_cnt", "valid")
    for name in names:
        t = getattr(b, name)
        setattr(v, name, None if t is None else t[a:z])
    v.rows = z - a
    v.lat_s = copy.copy(b.lat_s)
    for name in ("qa", "qpe", "ol"):
        setattr(v.lat_s, name, getattr(b.lat_s, name)[a:z])
    v.sel_owner = None
    return v


def attention_block(layer, w, segs, b, views, *, mtp=False, modes=None):
    """Only position/cache/selection/attention are segmented; two collectives/layer total."""
    c, a, R = w.cfg, layer.dsa, segs[-1][2]
    fw.dsa_prepare(layer, w, b, R)
    sparse = [((st.mtp_len if mtp else st.pos) + z - lo > c.dense_limit)
              for st, lo, z in segs] if modes is None else [m is not None for m in modes]
    # The indexer's projections are also shared across requests. Unused short rows are harmless scratch.
    if a.index is not None and any(st.index for st, _, _ in segs):
        ix = a.index
        fw.mm(b, b.normed[:R], ix.kw, fw.sums(w, b.xs[:R]), b.ikr[:R])
        if any(sparse) or not fw.SKIP_INDEX_WSTR:
            fw.mm(b, b.normed[:R], ix.wproj, fw.sums(w, b.xs[:R]), b.igr[:R])
        if any(sparse):
            fw.mm(b, b.qr[:R], ix.qb, fw.sums(w, b.xs_qr[:R]), b.qi[:R])
    for i, ((st, lo, z), v) in enumerate(zip(segs, views)):
        n = z - lo
        v.sel_np = st.capacity  # never scan past this request's smaller private index plane
        pos = st.mtp_len if mtp else st.pos
        pos_dev = st.mtp_pos_dev if mtp else st.pos_dev
        di = None if mtp else st.dsa_index[layer.index]
        kc = st.mtp_kc if mtp else st.kc[di]
        kpe = st.mtp_kpe if mtp else st.kpe[di]
        v.kpe[:n].copy_(v.dp[:n, c.q_lora + c.kv_lora:c.q_lora + c.kv_lora + c.qk_rope])
        rope.apply_(v.kpe[:n].view(n, 1, c.qk_rope), w.inv_freq, pos_dev, interleave=c.rope_interleave)
        rope.apply_(v.q[:n], w.inv_freq, pos_dev, offset=c.qk_nope, interleave=c.rope_interleave)
        # Captured dense graphs use a fixed chunk bound and device causal positions. Sparse graphs use buckets.
        bucket = None if modes is None else modes[i]
        host_pos = pos if modes is None else None
        fw._dsa_latent_context(a, w, layer, kc, kpe, pos_dev, v, n,
                               v.lat_s.nch if modes is not None else latent.chunks_for(pos + n),
                               st.planes(layer.index), host_pos, bucket, prepared=True)
    out = latent.expand_v(b.lat_s.ol[:R], a.absorb, b.vn[:R]).view(R, a.heads * c.v_dim)
    return fw.out_proj(w, b, out, a.o, fw.sums(w, b.xs_ao[:R]), R)


@torch.no_grad()
def compute_streams(w, segs, b, *, modes=None):
    c, R = w.cfg, segs[-1][2]
    views = [row_view(b, lo, z) for _, lo, z in segs]
    x = b.x[:R]
    glue.embed(b.ids[:R], w.embed, c.hidden, 1, x)
    for layer in w.layers:
        glue.rmsnorm(x, layer.in_norm, c.eps, b.normed[:R], fw.sums(w, b.xs[:R]))
        g = attention_block(layer, w, segs, b, views, modes=modes)
        glue.residual_add(x, x, g)
        glue.rmsnorm(x, layer.post_norm, c.eps, b.normed[:R], fw.sums(w, b.xs[:R]))
        g = fw.mlp_block(layer, w, b, R) if layer.mlp is not None else fw.moe_block(layer, w, b, R)
        glue.residual_add(x, x, g)
    glue.stream_mean(x, b.hidden[:R])
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], fw.sums(w, b.fxs[:R]))
    return fw.mm(b, b.fnormed[:R], w.head, fw.sums(w, b.fxs[:R]), b.logits[:R])


def commit_streams(w, segs, b, paths):
    if len(paths) != len(segs):
        raise ValueError("one committed path per segment")
    for (st, lo, z), path in zip(segs, paths):
        if not path or list(path) != list(range(len(path))) or len(path) > z - lo or st.cur:
            raise ValueError("full GLM commits a nonempty local chain prefix only")
    for (st, _, _), path in zip(segs, paths):
        st.set_pos(st.pos + len(path))


def mtp_stage_streams(w, b, windows, *, padded=False, chain=False):
    original = windows
    if padded:
        width = 1 if chain else (3 if max(len(x[1]) for x in windows) <= 3 else 8)
        windows = [(st, list(tokens) + [0] * (width - len(tokens)), hidden) for st, tokens, hidden in windows]
    segs, ids = layout(w, b, windows, mtp=True)
    for (_, tokens, hidden), (_, lo, z) in zip(original, segs):
        if hidden.shape != (len(tokens), w.cfg.hidden):
            raise ValueError("MTP carry hidden must match its next-token rows")
    b.staged.synchronize()
    b.ids_host[:len(ids)].numpy()[:] = ids
    b.ids[:len(ids)].copy_(b.ids_host[:len(ids)], non_blocking=True)
    if padded:
        if b.valid is None:
            b.valid = torch.empty(b.rows, device=w.device, dtype=torch.int32)
        b.valid[:len(ids)].zero_()
    elif b.valid is not None:
        b.valid[:len(ids)].fill_(1)
    for (_, tokens, hidden), (_, lo, z) in zip(original, segs):
        n = len(tokens)
        b.hin[lo:lo + n].copy_(hidden)
        if padded:
            b.hin[lo + n:z].zero_()
            b.valid[lo:lo + n].fill_(1)
    if not hasattr(b, "tail_ids"):
        b.tail_ids = torch.empty((4,), device=w.device, dtype=torch.int64)
    b.tail_ids[:len(segs)].copy_(torch.tensor([lo + len(tokens) - 1 for (_, tokens, _), (_, lo, _) in zip(original, segs)], device=w.device))
    b.staged.record()
    return segs


@torch.no_grad()
def mtp_compute_streams(w, segs, b, *, modes=None, zero=None, head=True):
    c, m, R, B = w.cfg, w.mtp, segs[-1][2], len(segs)
    D = c.hidden
    views = [row_view(b, lo, z) for _, lo, z in segs]
    glue.embed(b.ids[:R], w.embed, D, 1, b.me[:R])
    for i, (st, lo, _) in enumerate(segs):
        if st.mtp_len == 0 if zero is None else zero[i]:
            b.me[lo].zero_()
    glue.rmsnorm(b.me[:R], m.enorm, c.eps, b.mcat[:R, :D])
    glue.rmsnorm(b.hin[:R], m.hnorm, c.eps, b.mcat[:R, D:])
    fw.mm(b, b.mcat[:R], m.eh, fw.sums(w, b.mxs[:R]), b.mx[:R])
    layer = m.layer
    glue.rmsnorm(b.mx[:R], layer.in_norm, c.eps, b.normed[:R], fw.sums(w, b.xs[:R]))
    g = attention_block(layer, w, segs, b, views, mtp=True, modes=modes)
    glue.residual_add(b.mx[:R], b.mx[:R], g)
    glue.rmsnorm(b.mx[:R], layer.post_norm, c.eps, b.normed[:R], fw.sums(w, b.xs[:R]))
    g = fw.moe_block(layer, w, b, R)
    glue.residual_add(b.mx[:R], b.mx[:R], g)
    glue.rmsnorm(b.mx[:R], m.norm, c.eps, b.hidden[:R])
    torch.index_select(b.hidden[:R], 0, b.tail_ids[:B], out=b.fnormed[:B])
    if not head:
        return None
    return fw.mm(b, b.fnormed[:B], w.head, None, b.logits[:B])
