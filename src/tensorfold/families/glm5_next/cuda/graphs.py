"""CUDA graphs preserve eager bits by replaying static buffers, device positions, and collectives, with empty attention chunks skipped."""

from __future__ import annotations

import os

import torch

from . import latent, prof
from .forward import compute
from .mtp import mtp_compute

# Bounded opt-in cache capacity, not a new graph pool or request/slot allocator.
PACKED_CACHE_LIMIT = int(os.environ.get("TF_GLM_GRAPH_CACHE", "16"))
SAMPLING_GRAPH = os.environ.get("TF_GLM_SAMPLING_GRAPH", "0")
if SAMPLING_GRAPH not in ("0", "1"):
    raise ValueError("TF_GLM_SAMPLING_GRAPH must be 0 or 1")
SAMPLING_GRAPH = SAMPLING_GRAPH == "1"
SAMPLING_CACHE_LIMIT = int(os.environ.get("TF_GLM_SAMPLING_CACHE", "8"))
if not 1 <= SAMPLING_CACHE_LIMIT <= 64:
    raise ValueError("TF_GLM_SAMPLING_CACHE must be in 1..64")
GRAPH_MISS_MODE = os.environ.get("TF_GLM_GRAPH_MISS_MODE", "legacy")
if GRAPH_MISS_MODE not in ("legacy", "once"):
    raise ValueError("TF_GLM_GRAPH_MISS_MODE: legacy or once")
if not 1 <= PACKED_CACHE_LIMIT <= 128:
    raise ValueError("TF_GLM_GRAPH_CACHE must be in 1..128")


def pool_buckets(np_max: int) -> list[int]:
    """Every value ``sparse.pool_bucket`` can return for a sparse step (past TOPK_POOLS complete pools)."""
    out, b = [], 1024
    while b < np_max:
        out.append(b)
        b *= 2
    return out + [np_max]


class Graphs:
    def __init__(self, e, main_rows=(1, 2, 3, 4), mtp_rows=(1, 2, 3, 4)) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.main: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[int, torch.cuda.CUDAGraph] = {}
        self.sparse: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}      # (rows, parity, pool bucket)
        self.sparse_mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}       # (rows, pool bucket)
        w, st = e.w, e.st
        prof.active = False
        saved = list(st.cur)
        parities = (0, 1) if st.cur else (0,)
        with torch.no_grad():
            for R in main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(2):
                        compute(w, st, e.buf, R)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R)
                    self.main[(R, parity)] = g
            st.cur = saved
            if w.mtp is not None:
                e.mbuf.zero_first = False
                for n in mtp_rows:
                    for _ in range(2):
                        mtp_compute(w, st, e.mbuf, n)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n)
                    self.mtp[n] = g
            if st.index and latent.ENABLED:
                self._capture_sparse(e, main_rows, mtp_rows, parities)
        torch.cuda.synchronize()

    def _capture_sparse(self, e, main_rows, mtp_rows, parities) -> None:
        w, st = e.w, e.st
        saved = list(st.cur)
        for bucket in st.sparse_buckets():
            for R in main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(2):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    self.sparse[(R, parity, bucket)] = g
            if w.mtp is not None:
                for n in mtp_rows:
                    for _ in range(2):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    self.sparse_mtp[(n, bucket)] = g
        st.cur = saved


    def sampling_stage(self, logits, temperature, count, offset, id_map, gather):
        """Bounded initial nucleus stage; all native collectives and CPU draws are shared."""
        from tensorfold.cuda.sampling import NucleusStage
        if not hasattr(self, "sampling_stages"):
            self.sampling_stages = {}
            self.sampling_hits = self.sampling_misses = 0
        owner = getattr(gather, "_comm", gather)
        key = (tuple(logits.shape), logits.dtype, logits.device, float(temperature), count, offset,
               None if id_map is None else id_map.data_ptr(), id(owner))
        if key in self.sampling_stages:
            self.sampling_hits += 1
            stage = self.sampling_stages.pop(key)
            self.sampling_stages[key] = stage
        else:
            self.sampling_misses += 1
            if len(self.sampling_stages) >= SAMPLING_CACHE_LIMIT:
                del self.sampling_stages[next(iter(self.sampling_stages))]
            stage = NucleusStage(logits, temperature, count, offset, id_map, gather)
            self.sampling_stages[key] = stage
        return stage(logits)

    def packed_forward(self, w, segs, b, *, mtp=False):
        """Bounded shape/layout cache for private contiguous State pointers, same graph framework.

        Positions are device scalars. A crossing of the dense/sparse boundary stays eager;
        graphs are invalidated before a State is released, never reused for a new request.
        """
        from . import segmented as seg

        fn = seg.mtp_compute_streams if mtp else seg.compute_streams
        modes, zeros = [], []
        for st, lo, z in segs:
            pos = st.mtp_len if mtp else st.pos
            n = z - lo
            if pos < w.cfg.dense_limit < pos + n:
                return fn(w, segs, b)
            modes.append(st.select_bucket(pos, n) if pos >= w.cfg.dense_limit else None)
            zeros.append(pos == 0)
        key = (mtp, id(b), b.valid is not None, tuple((id(st), st.capacity, lo, z) for st, lo, z in segs),
               tuple(modes), tuple(zeros) if mtp else ())
        if not hasattr(self, "packed"):
            self.packed = {}
            self.packed_hits = self.packed_misses = 0
        kwargs = {"modes": modes, **({"zero": zeros} if mtp else {})}
        g = self.packed.get(key)
        if g is None:
            self.packed_misses += 1
            for _ in range(1 if GRAPH_MISS_MODE == "once" else 2):
                fn(w, segs, b, **kwargs)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                fn(w, segs, b, **kwargs)
            if len(self.packed) >= PACKED_CACHE_LIMIT:
                self.packed.pop(next(iter(self.packed)))
            self.packed[key] = g
            if GRAPH_MISS_MODE == "once":
                # Capture records operations but does not execute them. The fully
                # synchronized warm pass already wrote these persistent buffers.
                return b.logits[:len(segs) if mtp else segs[-1][2]]
        else:
            self.packed_hits += 1
            if GRAPH_MISS_MODE == "once":
                self.packed.pop(key)
                self.packed[key] = g
        g.replay()
        return b.logits[:len(segs) if mtp else segs[-1][2]]

    def release_packed(self):
        """Call at a round boundary, after sampling synchronized, before releasing cache pointers."""
        if hasattr(self, "packed"):
            self.packed.clear()
