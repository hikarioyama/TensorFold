"""GLM-5.3-Flash's tensor-parallel forward and commit; kernels keep rows apart, so row r has the serial step's bits."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
from tensorfold.cuda.exl3.linear import Exl3Linear
from tensorfold.cuda.geometry import MLA_PROMPT_ATT_ROWS as PROMPT_ATT_ROWS   # a dense latent call's prompt rows
from tensorfold.cuda.kernels import prefill_attention, qmm as shared

from . import cache_fp8, exl3_generic, glue, kda as kda_mod, latent, prof, qmm, rope, sparse, sparse53
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights
from .weights53 import Concat

EXL3_ROWS = 128      # rows one EXL3 call takes: a wider window runs tiles (never a row of its own)
# Experimental A candidates. A process restart selects the same policy for every rank/graph.
EXPERT_TILE = os.environ.get("TF_GLM_EXPERT_TILE", "baseline")
DOWN_COMBINE = os.environ.get("TF_GLM_DOWN_COMBINE", "0") == "1"
INPLACE_GATHER = os.environ.get("TF_GLM_INPLACE_GATHER", "0")
if INPLACE_GATHER not in ("0", "1"):
    raise ValueError("TF_GLM_INPLACE_GATHER must be 0 or 1")
INPLACE_GATHER = INPLACE_GATHER == "1"
GROUPED_MODE = os.environ.get("TF_GLM_GROUPED_MODE", "legacy")
GROUPED_MODES = {"legacy": 0, "swizzle": 1, "smallm": 2, "transpose": 3}
ROUTE_MODE = os.environ.get("TF_GLM_ROUTE_MODE", "legacy")
ROUTE_MODES = {"legacy": 0, "ballot": 1, "parallel": 2}
if ROUTE_MODE not in ROUTE_MODES:
    raise ValueError("TF_GLM_ROUTE_MODE: legacy, ballot or parallel")
if GROUPED_MODE not in GROUPED_MODES:
    raise ValueError("TF_GLM_GROUPED_MODE: legacy, swizzle, smallm or transpose")
if EXPERT_TILE not in ("baseline", "nt4pf2", "nt4_at8"):
    raise ValueError("TF_GLM_EXPERT_TILE: baseline, nt4pf2 or nt4_at8")



@dataclass
class Cut:
    """The recurrent state and convolution windows at an interior prompt row."""

    point: int
    rec: torch.Tensor
    conv: torch.Tensor


def configure_expert_tiles(scratch, *, full):
    scratch.adaptive_tiles = full and GROUPED_MODE in ("smallm", "transpose")
    scratch.transpose_nt4_at8 = full and EXPERT_TILE == "nt4_at8"
    if full and EXPERT_TILE == "nt4pf2":
        # An explicit tile must not be silently replaced by adaptive dispatch.
        scratch.adaptive_tiles = False
        for name in ("cfg_gu", "cfg_d"):
            _, warps, sk, _ = getattr(scratch, name)
            setattr(scratch, name, (4, warps, sk, 2))


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    SEL_SLAB = 32      # rows scored at once by the per-token selector (a prompt chunk runs many)

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.rows, self.prefill = rows, prefill
        self.valid = None
        head_rows = 1 if prefill else rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        if latent.ENABLED:
            # Dense attention only ever covers contexts up to the dense limit; longer rows go sparse.
            self.attn = None
            # a prompt chunk's dense pass runs PROMPT_ATT_ROWS rows at a time: its fp32 partials hold that many rows
            self.lat_s = latent.LatentScratch(rows, HL, latent.chunks_for(min(capacity, 2560) + rows), dev,
                                              lw=c.kv_lora, rope=c.qk_rope,
                                              part_rows=PROMPT_ATT_ROWS if prefill else rows)
        else:
            self.attn = AttnScratch(1 if prefill else rows, HL, c.qk_dim, capacity, dev)
        if prefill:
            kda_layers = [l for l in w.layers if l.kind == "kda"]
            width = kda_layers[0].kda.proj.n if kda_layers else 0
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
        self.hin = torch.empty((rows, c.hidden), dtype=torch.bfloat16, device=dev)      # MTP input rows
        self.zero_first = False          # MTP: this step starts at position 0 (its embedding is zeroed)
        self.x = torch.empty((rows, S * D), dtype=bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.xs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.post = torch.empty((rows, S), dtype=f32, device=dev)
        self.comb = torch.empty((rows, S * S), dtype=f32, device=dev)
        self.hcpart = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=f32, device=dev)
        # KDA
        self.ka = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.xs_fa = torch.empty((rows, 2), dtype=f32, device=dev)
        self.xs_ga = torch.empty((rows, 2), dtype=f32, device=dev)
        self.kxs = torch.empty((rows, LL * 128 // 64), dtype=f32, device=dev)
        # DSA
        self.dp = torch.empty((rows, c.q_lora + c.kv_a_width), dtype=bf, device=dev)
        self.kpe = torch.empty((rows, max(c.qk_rope, 1)), dtype=bf, device=dev)   # the shared rotary key part
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.xs_qr = torch.empty((rows, c.q_lora // 64), dtype=f32, device=dev)
        self.lat = torch.empty((rows, c.kv_lora), dtype=bf, device=dev)
        self.xs_lat = torch.empty((rows, c.kv_lora // 64), dtype=f32, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.kn = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.xs_ao = torch.empty((rows, HL * c.v_dim // 64), dtype=f32, device=dev)
        # DSA indexer (long contexts): the key projection's 128-wide rows, the head weights, and the queries
        self.ikr = torch.empty((rows, c.index_dim), dtype=bf, device=dev)
        self.igr = torch.empty((rows, c.index_heads), dtype=f32, device=dev)
        self.qi = torch.empty((rows, c.index_heads * c.index_dim), dtype=bf, device=dev)
        # GLM-5.3's per-token selection: the rows' top index_topk tokens, their counts, and one slab's fp32 scores.
        # The owner layer of a block fills them; the block's other three layers attend to the same tokens.
        self.sel_np = max(capacity, c.index_topk) if c.full else 0
        self.sel_tok = (torch.empty((rows, c.index_topk), dtype=torch.int32, device=dev) if c.full else None)
        self.sel_cnt = torch.empty((rows,), dtype=torch.int32, device=dev) if c.full else None
        self.sel_score = (torch.empty((min(rows, self.SEL_SLAB), self.sel_np), dtype=f32, device=dev)
                          if c.full else None)
        self.sel_parallel = (sparse53.selection_scratch(rows, self.sel_np, dev)
                             if c.full and not prefill else None)
        self.sel_owner: int | None = None       # the layer whose selection sel_tok/sel_cnt hold
        # dense MLP
        dl = c.dense_width // w.world
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, dl // 64), dtype=f32, device=dev)
        # MoE
        slots = c.top_k + 1
        ml = c.moe_width // w.world
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        self.eact = torch.empty((rows * slots, ml), dtype=bf, device=dev)
        exl3 = c.quant == "exl3"
        self.ey = None if exl3 else torch.empty((rows, slots, D), dtype=bf if prefill else f32, device=dev)
        self.plan = None if exl3 else grouped.Plan(rows, slots, c.experts + 1, dev, prefill=prefill)
        self.exl3 = None
        if c.quant == "exl3":            # EXL3 routed experts, and the shared expert as a BF16 MLP
            sl = c.shared_width // w.world
            shape = SimpleNamespace(dims=D, width=ml, count=c.experts)
            self.exl3 = Exl3Scratch(shape, rows, slots, device=dev,
                grouped_mode=GROUPED_MODES[GROUPED_MODE] if c.full else 0,
                route_mode=ROUTE_MODES[ROUTE_MODE] if c.full else 0)
            configure_expert_tiles(self.exl3, full=c.full)
            self.ey = self.exl3.y.view(rows, slots, D)
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        if prefill and prof.ENABLED and self.exl3 is not None:
            from tensorfold.cuda.exl3.experts import grouped_config
            from .weights53 import LINEAR_MODE
            print(f"[tensorfold] prefill dispatch: rows={rows}, EXL3_ROWS={EXL3_ROWS}, "
                  f"expert=ext.grouped, grouped_mode={self.exl3.grouped_mode}, linear_mode={LINEAR_MODE}, "
                  f"gate_up={grouped_config(self.exl3, self.exl3.cfg_gu, rows)}, "
                  f"down={grouped_config(self.exl3, self.exl3.cfg_d, rows)}, "
                  f"route_mode={self.exl3.route_mode}, down_combine={DOWN_COMBINE}, "
                  f"HB_WIDE={latent.HB_WIDE}, SEL_BN={sparse53.SEL_BN}", flush=True)
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sk = torch.empty((1 if prefill and not exl3 else 8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.me = torch.empty((rows, D), dtype=bf, device=dev)
        self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
        self.mxs = torch.empty((rows, 2 * D // 64), dtype=f32, device=dev)
        self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        self._parents: dict[int, torch.Tensor] = {}
        self.weights = w                          # the slot's own reference: scratch is keyed by a layer's id
        self._ws: dict[tuple, tuple] = {}           # one EXL3 projection's (xh, z, counters) a slot
        self._cat: dict[int, list] = {}           # a fused projection's part outputs (the copy fallback only)
        self._side = None                          # a second stream for a layer's independent half (made on use)
        self.side_in = self.side_out = None
        self.pair_in = self.pair_out = None
        # DFlash2 taps: the mean of the streams after chosen layers (``set_taps``), filled by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.experts = c.experts
        self.top_k = c.top_k

    def set_taps(self, layers: tuple[int, ...], hidden: int) -> None:
        self.tap_at = {}
        for i, layer in enumerate(layers):
            self.tap_at.setdefault(layer, []).append(i)
        self.taps = [torch.empty((self.rows, hidden), dtype=torch.bfloat16, device=self.ids.device) for _ in layers]

    def cat_scratch(self, q: Concat) -> list[torch.Tensor]:
        """A fused projection's part outputs for this slot (the copy fallback of ``weights53.Concat``)."""

        got = self._cat.get(id(q))
        if got is None:
            got = self._cat[id(q)] = q.make_scratch(self.ids.device, min(self.rows, EXL3_ROWS))
        return got

    def exl3_ws(self, layer, lane: int = 0, part: int = 0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """One EXL3 projection's per-call scratch, owned by this slot: the weights keep none of their own."""

        # Same-stream calls have disjoint lifetimes. Pair halves and the shared-expert
        # side stream have explicit independent lanes; all joins remain in moe_block.
        key = (lane, part, layer.k, layer.n, layer.split[0])
        got = self._ws.get(key)
        if got is None:
            got = self._ws[key] = layer.workspace(self.ids.device, min(self.rows, EXL3_ROWS))
        return got

    def side_pair(self):
        """(stream, in-event, out-event) for a fused projection's second part, or None on one stream."""

        side = self.side_stream()
        return None if side is None else (side, self.pair_in, self.pair_out)

    def side_stream(self):
        """The second stream a layer's independent half runs on, or None where there is no CUDA (tests, one GPU)."""

        if not torch.cuda.is_available() or os.environ.get("TF_GLM_ONE_STREAM", "") == "1":
            return None
        if self._side is None:
            self._side = torch.cuda.Stream()
            self.side_in = torch.cuda.Event()
            self.side_out = torch.cuda.Event()
            self.pair_in = torch.cuda.Event()
            self.pair_out = torch.cuda.Event()
            self._side.wait_stream(torch.cuda.current_stream())     # fork: nothing on the side stream yet
        return self._side

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer)."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.conv = torch.zeros((n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch_set = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.scratch = self.scratch_set.views if n else []
        self.latent = latent.ENABLED
        self.kv_dtype = w.meta.get("kv_dtype", "bf16")
        if self.kv_dtype != "bf16" and not (c.full and self.latent):
            raise ValueError("FP8 KV cache requires full GLM latent attention")
        cache_fp8.row_bytes(c.kv_lora, self.kv_dtype)
        if self.latent:              # one 512-wide latent a token and layer (kc), no separate values (vc)
            self.kc = [(cache_fp8.allocate(capacity, c.kv_lora, dev, self.kv_dtype) if self.kv_dtype == "fp8" else
                       torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)) for _ in dsa_layers]
            # the shared rotary key part, one 64-wide row atoken, beside each layer's latents
            self.kpe = [(cache_fp8.allocate(capacity, max(c.qk_rope, 1), dev, self.kv_dtype) if self.kv_dtype == "fp8" else
                        torch.zeros((capacity, max(c.qk_rope, 1)), dtype=torch.bfloat16, device=dev))
                        for _ in dsa_layers]
            self.vc = [None for _ in dsa_layers]
        else:
            self.kc = [torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
            self.vc = [torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            if self.latent:
                self.mtp_kc = (cache_fp8.allocate(capacity, c.kv_lora, dev, self.kv_dtype) if self.kv_dtype == "fp8" else
                       torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev))
                self.mtp_kpe = (cache_fp8.allocate(capacity, max(c.qk_rope, 1), dev, self.kv_dtype) if self.kv_dtype == "fp8" else
                        torch.zeros((capacity, max(c.qk_rope, 1)), dtype=torch.bfloat16, device=dev))
                self.mtp_vc = None
            else:
                self.mtp_kc = torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev)
                self.mtp_vc = torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev)
        # DSA indexer planes (long contexts only). GLM-5.3 owns them on one layer in four (``indexer_types ==
        # "full"``, plus the MTP layer): per-token fp16 keys and fp32 head weights, and a shared layer reads its
        # block owner's plane and selection. Flash owns a pooled plane per DSA layer (keys, gates, pool keys).
        self.index_source = list(w.meta.get("index_source") or [])
        self.index_full = tuple(w.meta.get("index_full") or ())
        self.index_np = capacity if c.full else capacity // 4      # entries a sparse pass can cover
        self.index = {}
        if w.meta.get("long_context"):
            if c.full:
                keys = lambda: torch.zeros((capacity, c.index_dim), dtype=torch.float16, device=dev)    # noqa: E731
                wts = lambda: torch.zeros((capacity, c.index_heads), dtype=torch.float32, device=dev)   # noqa: E731
                owners = sorted(set(self.index_source))
                if w.mtp is not None:
                    owners.append(c.layers)                       # the MTP layer's own indexer
                self.index = {i: (keys(), wts()) for i in owners}
            else:
                mk = lambda: torch.zeros((capacity, c.index_dim), dtype=torch.bfloat16, device=dev)     # noqa: E731
                mkp = lambda: torch.zeros((capacity // 4 + 2, c.index_dim), dtype=torch.bfloat16,   # noqa: E731
                                          device=dev)
                self.index = {self.dsa_index[l.index]: (mk(), mk(), mkp()) for l in dsa_layers}
                if w.mtp is not None:
                    self.index[len(dsa_layers)] = (mk(), mk(), mkp())

    def planes(self, layer: int):
        """The indexer planes layer ``layer`` scores with: its own, or its block owner's when it only shares one."""

        if not self.index:
            return None
        if self.index_source:
            owner = layer if layer >= len(self.index_source) else self.index_source[layer]
        else:                                   # Flash: every DSA layer owns one, the MTP layer's is last
            owner = self.dsa_index.get(layer, len(self.dsa_index))
        return self.index.get(owner)

    def index_rows(self, n: int, m: int) -> list[torch.Tensor]:
        """The indexer planes' rows a snapshot of ``n`` tokens (``m`` in the MTP head) depends on, in save/load order.

        One entry per owning layer: GLM-5.3's per-token keys and head weights; Flash's keys, gates and complete pools.
        """

        out = []
        mtp_at = len(self.index_source) if self.index_source else len(self.kc)
        for owner, planes in self.index.items():
            rows = m if owner == mtp_at else n
            if self.index_source:
                if rows > 0:
                    out += [planes[0][:rows], planes[1][:rows]]
            else:
                out += [planes[0][:n], planes[1][:n], planes[2][:n // 4 + 1]]
        return out

    def sparse_buckets(self) -> list[int]:
        """The entry counts a captured sparse graph is built for (``select_bucket``'s reachable values)."""

        if not self.index_source:
            from .graphs import pool_buckets           # graphs imports this module: the list lives there

            return pool_buckets(self.index_np)
        return sparse53.buckets(self.index_np)

    def select_bucket(self, pos: int, R: int) -> int:
        """Entries a sparse pass scores for rows ``pos .. pos + R - 1`` (a captured graph's fixed size)."""

        if not self.index_source:               # Flash: pooled keys, four tokens a pool
            return sparse.pool_bucket(pos, R, self.index_np)
        return sparse53.score_bucket(pos, R, self.index_np)

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        for trio in self.index.values():
            for plane in trio:
                plane.zero_()
        self.cur = [0] * len(self.cur)
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() if x is not None else None for x in self.vc]
        if self.latent:
            other.kpe = [x.clone() for x in self.kpe]
        other.index = {i: tuple(x.clone() for x in trio) for i, trio in self.index.items()}
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone() if self.mtp_vc is not None else None
            if self.latent:
                other.mtp_kpe = self.mtp_kpe.clone()
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def tp_partial(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Own-rank receive slice at actual R, never capacity R; producer writes before native AllGather."""
    if not INPLACE_GATHER or not w.cfg.full or w.comm is None:
        return b.part[:R]
    if not 0 <= R <= b.part.shape[0] or not 0 <= w.rank < b.world:
        raise ValueError("invalid in-place partial rows/rank")
    n = R * b.part.shape[1]
    return b.gath[w.rank * n:(w.rank + 1) * n].view(R, b.part.shape[1])


def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Every rank's fp32 partial tp_partial(w, b, R) in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = b.part.shape[1]
    if w.comm is None:
        return tp_partial(w, b, R).view(1, R, d)
    out = b.gath[:b.world * R * d]
    w.comm.all_gather(tp_partial(w, b, R).reshape(-1), out)
    return out.view(b.world, R, d)


SKIP_DEAD_SUMS = os.environ.get("TF_GLM_DEAD_SUMS", "1").strip() != "0"    # K1-a, off for an A/B or a rollback
# K1-a2: the indexer's stored head weights are written and never read in the latent path (plan §10.2), so the
# weights_proj runs only for a window that selects and the cached copy is skipped. Off restores the old work.
SKIP_INDEX_WSTR = os.environ.get("TF_GLM_INDEX_WSTR", "1").strip() != "0"


def sums(w: Weights, buf: torch.Tensor) -> torch.Tensor | None:
    """The 64-group row-sums a projection would precompute, or None where no kernel of this model reads them.

    MLX Q4's MMA path reads them (``qmm.matmul(x, q, xs)``); every EXL3 and BF16 projection computes without them, so
    an EXL3-only checkpoint (``w.meta['norm_sums']`` False) can skip the producer launch entirely (plan K1-a). A Q4
    call handed None computes the sums itself, so this never changes a bit of a Q4 model.
    """

    return buf if not SKIP_DEAD_SUMS or w.meta.get("norm_sums", True) else None


def mm(b: Buffers, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False, lane: int = 0) -> torch.Tensor:
    """A projection: EXL3 on its own kernels (tiled by row), MLX 4-bit of a prompt chunk on the shared prefill matmul, the rest on ``qmm.matmul``.

    EXL3 takes at most ``EXL3_ROWS`` rows a call, which is an internal tile: a prompt chunk or a wide window runs
    tiles of that size into their own rows of ``out``. The kernels are row-invariant and their split-K order is fixed
    by the shape, so a row's bits do not depend on the tile it arrived in.
    """

    if isinstance(q, (Exl3Linear, Concat)):
        dt = torch.float32 if f32 else torch.bfloat16
        rows = x.shape[0]
        for r0 in range(0, rows, EXL3_ROWS):
            r1 = min(rows, r0 + EXL3_ROWS)
            if isinstance(q, Concat):
                # one stream for a fused projection's parts: on this hardware the fork/join costs more than the
                # two ~10 us latency-bound halves gain by overlapping, and the MoE split (moe_block) does pay off
                q(x[r0:r1], out=out[r0:r1], out_dtype=dt, scratch=b.cat_scratch(q),
                  ws=[b.exl3_ws(part, lane, i) for i, part in enumerate(q.layers)])
            else:
                xh, z, counters = b.exl3_ws(q, lane)
                q(x[r0:r1], out=out[r0:r1], out_dtype=dt, xh=xh, z=z, counters=counters)
        return out
    if b.prefill and isinstance(q, qmm.Q4):
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return qmm.matmul(x, q, sums(b.weights, xs) if xs is not None else None, out=out, f32=f32, part=b.sk)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor | None, R: int, *, profile: str | None = None) -> torch.Tensor:
    """The rank's fp32 partial of a projection: ``xs`` is the sums *buffer* to precompute into, or None to skip."""

    with prof.timed(f"{profile}: output projection") if profile else nullcontext():
        mm(b, x, q, None if xs is None else qmm.group_sums(x, xs), tp_partial(w, b, R), f32=True)
    with prof.timed(f"{profile}: all-gather") if profile else nullcontext():
        return gather(w, b, R)


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, cut: Cut | None = None) -> torch.Tensor:
    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    p = b.kproj[0, :R] if b.prefill else st.proj[li, :R]
    mm(b, b.normed[:R], k.proj, sums(w, b.xs[:R]), p)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    pre = b.prefill
    mm(b, fa, k.fb, None if pre else sums(w, b.xs_fa[:R]), b.ka[:R])
    mm(b, ga, k.gb, None if pre else sums(w, b.xs_ga[:R]), b.kg[:R])
    cur = st.cur[li]
    if cut is None:
        out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                            k.dt_bias, k.norm, c.eps, c.lower, R, b.kscratch if pre else st.scratch[li],
                            st.rec[1 - cur, li])
    else:
        n = cut.point
        first = kda_mod.chain(p[:n], k.b_off, b.ka[:n], b.kg[:n], st.conv[li], k.conv, st.rec[cur, li],
                              k.a_log, k.dt_bias, k.norm, c.eps, c.lower, n, b.kscratch, cut.rec[li]).clone()
        _shift_conv(cut.conv[li:li + 1], b.kproj[:, :n], n)
        rest = kda_mod.chain(p[n:], k.b_off, b.ka[n:R], b.kg[n:R], cut.conv[li], k.conv, cut.rec[li],
                             k.a_log, k.dt_bias, k.norm, c.eps, c.lower, R - n, b.kscratch, st.rec[1 - cur, li])
        out = torch.cat((first, rest))
    if pre:                              # a prompt chunk keeps every row: the layer commits now
        st.cur[li] = 1 - cur
        _shift_conv(st.conv[li:li + 1], b.kproj[:, :R], R)
    return out_proj(w, b, out, k.o, None if pre else sums(w, b.kxs[:R]), R)


def dsa_prepare(layer: LayerW, w: Weights, b: Buffers, R: int) -> None:
    """Stateless projections shared by solo and segmented MLA."""

    c = w.cfg
    a = layer.dsa
    with prof.timed("dsa: prepare proj"):
        mm(b, b.normed[:R], a.proj, sums(w, b.xs[:R]), b.dp[:R])
    xs_qr = sums(w, b.xs_qr[:R])
    with prof.timed("dsa: prepare q norm"):
        glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], xs_qr)
    # the latent path never reads the kv_a sums (the absorbed key is built from the normed latent rows)
    with prof.timed("dsa: prepare kv norm"):
        glue.rmsnorm(b.dp[:R, c.q_lora:c.q_lora + c.kv_lora], a.kv_norm, c.eps, b.lat[:R],
                     None if a.absorb is not None else sums(w, b.xs_lat[:R]))
    HL = a.heads
    with prof.timed("dsa: prepare q_b"):
        mm(b, b.qr[:R], a.q_b, xs_qr, b.q[:R].view(R, HL * c.qk_dim))


def dsa_block(layer: LayerW, w: Weights, kc: torch.Tensor, vc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers,
              R: int, nch: int | None, index=None, host_pos: int | None = None,
              sparse_np: int | None = None, kpe: torch.Tensor | None = None) -> torch.Tensor:
    """Write every index key and pool; rows past the dense limit attend to their top-512 pools (host_pos eager, or sparse_np in a captured graph)."""

    c, a = w.cfg, layer.dsa
    HL = a.heads
    dsa_prepare(layer, w, b, R)
    xs_qr = sums(w, b.xs_qr[:R])
    if c.qk_rope and a.absorb is not None:     # GLM-5.3: the shared rotary key part, and the query's own
        with prof.timed("dsa: rope"):
            b.kpe[:R].copy_(b.dp[:R, c.q_lora + c.kv_lora:c.q_lora + c.kv_lora + c.qk_rope])
            rope.apply_(b.kpe[:R].view(R, 1, c.qk_rope), w.inv_freq, pos_dev, interleave=c.rope_interleave)
            rope.apply_(b.q[:R], w.inv_freq, pos_dev, offset=c.qk_nope, interleave=c.rope_interleave)
    if a.absorb is not None:
        return _dsa_latent(a, w, layer, kc, kpe, pos_dev, b, R, nch, index, host_pos, sparse_np)
    if sparse_np is not None:
        raise ValueError("sparse CUDA graphs need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.lat[:R], a.kv_k, sums(w, b.xs_lat[:R]), b.kn[:R].view(R, HL * c.qk_dim))
    mm(b, b.lat[:R], a.kv_v, sums(w, b.xs_lat[:R]), b.vn[:R].view(R, HL * c.v_dim))
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        if len(index) != 3:
            raise ValueError("the per-head DSA path is Flash's pooled indexer; GLM-5.3 runs on the latent cache")
        ik, ig, pk = index
        ix = a.index
        mm(b, b.normed[:R], ix.kw, sums(w, b.xs[:R]), b.ikr[:R])
        glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:     # every row sparse: the dense pass is skipped
        o = torch.empty((R, HL, c.v_dim), dtype=torch.bfloat16, device=b.q.device) if b.prefill else b.attn.out[:R]
    elif b.prefill:
        o = prefill_attention.attention(b.q[:R], kc, vc, host_pos, scale=c.qk_dim ** -0.5)
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        mm(b, b.qr[:R], ix.qb, xs_qr, b.qi[:R])
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, None if b.prefill else sums(w, b.xs_ao[:R]), R, profile="dsa")


def dense_attention(qa: torch.Tensor, lc: torch.Tensor, pos_dev: torch.Tensor, s: latent.LatentScratch, *,
                    scale: float, nch: int, out: torch.Tensor, qpe: torch.Tensor | None = None,
                    kpe: torch.Tensor | None = None) -> torch.Tensor:
    """``latent.attention`` in blocks of ``part_rows`` rows, each at its first row's position: one call's bits a row."""

    R, step = qa.shape[0], s.part_rows
    hb = latent.head_block(R, qa.shape[2])
    if R <= step:
        return latent.attention(qa, lc, pos_dev, s, scale=scale, nch=nch, out=out, hb=hb, qpe=qpe, kpe=kpe)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        at = pos_dev if r0 == 0 else pos_dev + r0          # the block's first row's position, on the device
        latent.attention(qa[r0:r1], lc, at, s, scale=scale, nch=nch, out=out[r0:r1], hb=hb,
                         qpe=None if qpe is None else qpe[r0:r1], kpe=kpe)
    return out


def _selection(a, w: Weights, layer: LayerW, index, b: Buffers, R: int, pos_dev: torch.Tensor,
               sparse_np: int | None, prepared: bool = False):
    """The rows' attended tokens: the owner layer scores and selects them, a shared layer reuses its block's.

    GLM-5.3 computes the DSA top-k on one layer in four (``indexer_types == "full"``); the other three layers of the
    block attend to the same tokens. The owner's rotated queries and head weights stay in the slot while its block
    runs, so a shared layer neither projects nor scores again (the reference's ``indexer_mode == "shared"``), and the
    owner's identity is checked so no layer can read another block's selection.
    """

    c = w.cfg
    ix = a.index
    if ix is not None:
        if not prepared:
            mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
        if c.qk_rope:              # the reference rotates the head's leading qk_rope columns with the main table
            rope.apply_(b.qi[:R].view(R, c.index_heads, c.index_dim), w.index_inv_freq, pos_dev,
                        nf=int(w.index_inv_freq.numel()), interleave=c.indexer_interleave)
        tokens, counts = sparse53.select_tokens(
            b.qi[:R].view(R, c.index_heads, c.index_dim), b.igr[:R], index[0], pos_dev, R, c.index_topk,
            min(b.sel_np, int(index[0].shape[0])),
            bucket=sparse_np, scores=b.sel_score, tokens=b.sel_tok[:R], counts=b.sel_cnt[:R],
            parallel_scratch=getattr(b, "sel_parallel", None))
        b.sel_cnt[:R].copy_(counts)
        b.sel_owner = layer.index
    else:
        wanted = w.meta["index_source"][layer.index]
        if b.sel_owner != wanted:
            raise RuntimeError(f"layer {layer.index} attends to layer {wanted}'s DSA selection, but the slot holds "
                               f"layer {b.sel_owner}'s; the block's owner has to run first")
    return b.sel_tok[:R], b.sel_cnt[:R]


def _dsa_latent_context(a, w: Weights, layer: LayerW, lc: torch.Tensor, kc: torch.Tensor, pos_dev: torch.Tensor,
                b: Buffers, R: int, nch: int | None, index, host_pos: int | None,
                sparse_np: int | None = None, prepared: bool = False) -> torch.Tensor:
    """DSA on the latent cache: the same indexer and selection, attention over latents with kv_b's key blocks absorbed into the query."""

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.latent_write(b.lat[:R], lc, pos_dev, valid=b.valid)
        if c.qk_rope:
            latent.latent_write(b.kpe[:R], kc, pos_dev, valid=b.valid)
    # sparse_np: every row is past the dense limit (a captured graph); else the host position decides
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    ix = a.index
    if index is not None and ix is not None:      # only the block's owner projects and writes index keys: a shared
        ik, ig = index[0], index[1]              # layer has no indexer of its own and reads its owner's plane
        with prof.timed("dsa: indexer update"):
            if not prepared:
                mm(b, b.normed[:R], ix.kw, sums(w, b.xs[:R]), b.ikr[:R])
            if ix.gate is None:
                # GLM-5.3: the key plane is what later rows read; a row's head weights are used by the selection of the
                # *same* window (``_selection`` scores with b.igr), and the cached copy is never read. So the
                # weights_proj runs only for a window that selects, and the stored weights are skipped entirely
                # (plan §10.2's audit: written and never read).
                if not prepared and (sparse_rows or not SKIP_INDEX_WSTR):
                    mm(b, b.normed[:R], ix.wproj, sums(w, b.xs[:R]), b.igr[:R])
                sparse53.index_update(b.ikr[:R, :c.index_dim], None if SKIP_INDEX_WSTR else b.igr[:R], ix.ln_w,
                                      ix.ln_b, w.index_inv_freq, c.indexer_interleave, ik,
                                      None if SKIP_INDEX_WSTR else ig, pos_dev, valid=b.valid)
            else:
                glue.router(b.normed[:R], ix.gate, b.igr[:R])
                sparse53.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, w.index_inv_freq,
                                      c.indexer_interleave, ik, ig, pos_dev)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R], d=c.qk_nope or None)
        if c.qk_rope:                          # the query's own rotary part, scored against the shared key part
            s.qpe[:R].copy_(b.q[:R, :, c.qk_nope:])
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        # Rows past the dense limit are recomputed sparsely below, so the dense pass needs only the chunks up to it.
        with prof.timed("dsa: dense attention"):
            dense_attention(qa, lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol,
                            qpe=s.qpe[:R] if c.qk_rope else None, kpe=kc if c.qk_rope else None)
    if sparse_rows:
        if c.full:
            # An eager prefill has no captured bucket, and a None bucket makes select_tokens score the whole
            # key plane: a 43K prompt at a 256K window then scores 256K entries a row, which is what made
            # prefill crawl (~450 tok/s). Size the pass to the rows' own visible range instead.
            # score_bucket() rounds up to a power of two so a captured graph is reused across nearby
            # positions; eager has no graph, so the exact scoring tile is used and up to half the scan is
            # not paid. Entries past a row are masked to -inf and the scan is never narrower than k, so the
            # selected top-k is identical either way.
            sel_bucket = sparse_np
            if sel_bucket is None and host_pos is not None:
                plane = min(b.sel_np, int(index[0].shape[0]))
                span = host_pos + R
                tile = -(-span // sparse53.SEL_BN) * sparse53.SEL_BN
                sel_bucket = min(plane, max(tile, sparse53.SEL_BN, c.index_topk))
            with prof.timed("dsa: select tokens"):
                tokens, counts = _selection(a, w, layer, index, b, R, pos_dev, sel_bucket, prepared)
        else:
            with prof.timed("dsa: select tokens"):
                mm(b, b.qr[:R], ix.qb, sums(w, b.xs_qr[:R]), b.qi[:R])
                tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R], index[2], host_pos, R,
                                                      index[2].shape[0] - 2, pos_dev, bucket=sparse_np)
        with prof.timed("dsa: sparse attention"):
            latent.sparse_attention(qa, lc, tokens, counts, ol, scale,
                                    qpe=s.qpe[:R] if c.qk_rope else None, kpe=kc if c.qk_rope else None, scratch=s)
    return ol


def _dsa_latent(a, w: Weights, layer: LayerW, lc: torch.Tensor, kc: torch.Tensor, pos_dev: torch.Tensor,
                b: Buffers, R: int, nch: int | None, index, host_pos: int | None,
                sparse_np: int | None = None) -> torch.Tensor:
    ol = _dsa_latent_context(a, w, layer, lc, kc, pos_dev, b, R, nch, index, host_pos, sparse_np)
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, a.heads * w.cfg.v_dim)
    return out_proj(w, b, o, a.o, sums(w, b.xs_ao[:R]), R, profile="dsa")


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    mm(b, b.normed[:R], m.gu, sums(w, b.xs[:R]), b.gu[:R])
    xs = sums(w, b.xs_act[:R])
    glue.swiglu(b.gu[:R], b.act[:R], xs, w.cfg.limit)
    return out_proj(w, b, b.act[:R], m.down, xs, R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        if b.valid is not None:
            glue.mask_routes(b.pick[:R], b.wts[:R], b.valid[:R], c.experts)
        if m.shared is None:
            grouped.route(b.pick[:R], b.plan)
    if m.shared is not None:
        # EXL3: the routed slots through the trellis kernels, the shared expert (last slot) through BF16 matmuls.
        # The two are independent (both read the normed rows into their own buffers), so the shared expert runs on a
        # second stream beside the routed experts: its latency-bound matmuls fill the gaps the routed kernels leave.
        s = m.shared
        fused_down = c.full and DOWN_COMBINE
        side = b.side_stream()
        if side is None:
            exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit, defer_down=fused_down)
            with prof.stream_timed("moe: shared expert"):
                mm(b, b.normed[:R], s.gu, sums(w, b.xs[:R]), b.sgu[:R], lane=1)
                xs_s = sums(w, b.sxs[:R])
                glue.swiglu(b.sgu[:R], b.sact[:R], xs_s, c.limit)
                mm(b, b.sact[:R], s.down, xs_s, b.sy[:R], f32=True, lane=1)
        else:
            main = torch.cuda.current_stream()
            b.side_in.record(main)                 # the normed rows, the picks and the weights are ready
            side.wait_event(b.side_in)
            with torch.cuda.stream(side):
                with prof.stream_timed("moe: shared expert"):
                    mm(b, b.normed[:R], s.gu, sums(w, b.xs[:R]), b.sgu[:R], lane=1)
                    xs_s = sums(w, b.sxs[:R])
                    glue.swiglu(b.sgu[:R], b.sact[:R], xs_s, c.limit)
                    mm(b, b.sact[:R], s.down, xs_s, b.sy[:R], f32=True, lane=1)
                b.side_out.record(side)
            exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit, defer_down=fused_down)
            main.wait_event(b.side_out)            # the combine reads both, in slot order
        with prof.stream_timed("moe: combine"):
            b.ey[:R, c.top_k].copy_(b.sy[:R])
            if fused_down:
                exl3_generic.finish(b.pick, b.wts, m.experts, b.exl3, tp_partial(w, b, R), R)
            else:
                glue.combine(b.ey[:R], b.wts[:R], tp_partial(w, b, R))
        with prof.stream_timed("moe: all-gather"):
            return gather(w, b, R)
    with prof.timed("moe: gate/up"):
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)
    with prof.timed("moe: combine"):
        glue.combine(b.ey[:R], b.wts[:R], tp_partial(w, b, R))
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None, cut: Cut | None = None) -> None:
    c = w.cfg
    x = b.x[:R]
    if layer.attn_hc is None:                  # one residual stream (GLM-5.3, and the MTP layer)
        g = _plain_block(layer, w, st, b, R, nch, host_pos, sparse_np)
        return
    h = layer.attn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    if layer.kind == "kda":
        with prof.timed("kda"):
            g = kda_block(layer, w, st, b, R, cut)
    else:
        di = st.dsa_index[layer.index]
        with prof.timed("dsa (total)"):
            g = dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch, st.planes(layer.index),
                          host_pos, sparse_np)
    with prof.timed("hc"):
        glue.hc_post(x, x, g, b.post[:R], b.comb[:R])
        h = layer.ffn_hc
        glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])


def _plain_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None,
                 host_pos: int | None, sparse_np: int | None) -> None:
    """One layer of a one-stream model: LayerNorm, attention, add, LayerNorm, MLP or MoE, add.

    The same sequence MTP's own layer runs (``mtp.py``), so a plain model and the MTP head share their arithmetic.
    """

    c, x = w.cfg, b.x[:R]
    glue.rmsnorm(x, layer.in_norm, c.eps, b.normed[:R], sums(w, b.xs[:R]))
    if layer.kind == "kda":
        g = kda_block(layer, w, st, b, R)
    else:
        di = st.dsa_index[layer.index]
        with prof.timed("dsa (total)"):
            g = dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch, st.planes(layer.index),
                          host_pos, sparse_np, st.kpe[di])
    glue.residual_add(x, x, g)
    glue.rmsnorm(x, layer.post_norm, c.eps, b.normed[:R], sums(w, b.xs[:R]))
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    glue.residual_add(x, x, g)


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and not st.index:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    check_room(w, st, R)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = list(tokens)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return R


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None, cut: Cut | None = None):
    """Run capturable GPU work on static buffers and device positions; eager long contexts use host_pos (graphs sparse_np) to select sparse attention."""

    if cut is not None and (not b.prefill or not 0 < cut.point < R):
        raise ValueError("a prompt cut must lie inside a prefill chunk")
    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, cut)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    glue.stream_mean(b.x[:R], b.hidden[:R])
    if not logits:
        return None
    xs_f = sums(w, b.fxs[:R])
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], xs_f)
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        return mm(b, b.fnormed[R - 1:R], w.head, None if xs_f is None else xs_f[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, xs_f, b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits and hidden buffer views for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@triton.jit
def _row(CONV, PROJ, l, src, c, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr):
    old = tl.load(CONV + l * conv_layer + src * C + c, mask=(src < TAPS) & (c < C), other=0.0)
    new = tl.load(PROJ + l * proj_layer + (src - TAPS) * proj_row + c, mask=(src >= TAPS) & (c < C), other=0.0)
    return tl.where(src < TAPS, old, new)


@triton.jit
def _conv_shift(CONV, PROJ, keep, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program (layer, channel block): the 3 window rows become rows keep .. keep + 2 of [old window; new rows]."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v0 = _row(CONV, PROJ, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v1 = _row(CONV, PROJ, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v2 = _row(CONV, PROJ, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
    tl.store(CONV + l * conv_layer + c, v0, mask=c < C)
    tl.store(CONV + l * conv_layer + C + c, v1, mask=c < C)
    tl.store(CONV + l * conv_layer + 2 * C + c, v2, mask=c < C)


def _shift_conv(conv: torch.Tensor, proj: torch.Tensor, keep: int) -> None:
    """conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [L, R, W >= C])."""

    n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    _conv_shift[(n, triton.cdiv(C, 1024))](conv, proj, keep, conv.stride(0), proj.stride(0), proj.stride(1), C=C,
                                           TAPS=taps, BLOCK=1024, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
