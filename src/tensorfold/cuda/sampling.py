"""Position-keyed CUDA target sampling with the Metal engine's host-side rule, for every CUDA family."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from dataclasses import replace
import math
import os
from functools import lru_cache

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, uniform

MASS = 2.0 ** 40        # a token's share of the mass in fixed point: shard sums are exact, so ranks agree bit for bit
NUCLEUS = 1024          # candidates a rank reads for a top_k-off draw; a row they don't cover reads whole shards


def _switch(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


AMAX = _switch("TF_NUCLEUS_AMAX")
UNSORTED = _switch("TF_NUCLEUS_UNSORTED")
SINGLETON = _switch("TF_NUCLEUS_SINGLETON")


def sample_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """Sample each row from its logits and absolute position; serial and verify-window rows share this one path."""

    if logits.ndim != 2 or not logits.is_cuda or len(positions) != logits.shape[0]:
        raise ValueError("expected CUDA logits [rows, vocab] and one position per row")
    if sampling is None or sampling.temperature <= 0:
        return [int(x) for x in logits.argmax(dim=-1).cpu().tolist()]
    if not sampling.top_k:
        return nucleus_rows(logits, positions, sampling)
    width = logits.shape[1]
    count = min(width, int(sampling.top_k) + MARGIN) if sampling.top_k else width
    if count < width:
        values, ids = torch.topk(logits.float(), count, dim=-1, sorted=False)
        values_np = values.cpu().numpy()
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False)
    else:
        values_np = logits.float().cpu().numpy()
        ids_np = np.broadcast_to(np.arange(width, dtype=np.int64), values_np.shape)
    return choose_rows(values_np, ids_np, positions, sampling)


def sample_streams(logits: torch.Tensor, starts: Sequence[int], positions: Sequence[Sequence[int]],
                   samplings: Sequence[Sampling | None]) -> list[list[int]]:
    """``sample_rows`` for several streams, grouped by candidate count so each row gets its own stream's call."""

    groups: dict[int, list[int]] = {}
    width = logits.shape[1]
    out: list[list[int]] = [[] for _ in samplings]
    for s, smp in enumerate(samplings):
        greedy = smp is None or smp.temperature <= 0
        if not greedy and not smp.top_k:              # top_k off: the nucleus rule, stream by stream
            out[s] = nucleus_rows(logits[starts[s]:starts[s + 1]], positions[s], smp)
            continue
        groups.setdefault(0 if greedy else min(width, int(smp.top_k) + MARGIN), []).append(s)
    launched = []
    for count, members in groups.items():
        rows = torch.cat([logits[starts[s]:starts[s + 1]] for s in members]) if len(members) > 1 \
            else logits[starts[members[0]]:starts[members[0] + 1]]
        if count == 0:
            launched.append((count, members, rows.argmax(dim=-1), None))
        elif count < width:
            values, ids = torch.topk(rows.float(), count, dim=-1, sorted=False)
            launched.append((count, members, ids, values))
        else:
            launched.append((count, members, None, rows.float()))
    for count, members, ids, values in launched:
        ids_np = ids.cpu().numpy().astype(np.int64, copy=False) if ids is not None else None
        values_np = values.cpu().numpy() if values is not None else None
        row = 0
        for s in members:
            n = starts[s + 1] - starts[s]
            if count == 0:
                out[s] = [int(x) for x in ids_np[row:row + n]]
            else:
                v = values_np[row:row + n]
                i = ids_np[row:row + n] if ids_np is not None else np.broadcast_to(np.arange(width, dtype=np.int64),
                                                                                   v.shape)
                out[s] = choose_rows(v, i, positions[s], samplings[s])
            row += n
    return out


def _stacked(gather: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """Every rank's copy of ``t`` [world, *t.shape], this rank's included, through ``gather`` of its float32 words."""

    words = t.contiguous().view(torch.float32).view(-1)
    return gather(words).reshape(-1, words.numel()).view(t.dtype).reshape(-1, *t.shape)


def one_rank(words: torch.Tensor) -> torch.Tensor:
    return words[None]


def comm_gather(comm) -> Callable[[torch.Tensor], torch.Tensor]:
    """``nucleus_rows``'s gather over a family's NCCL ``comm`` (``tensorfold.cuda.comm``)."""

    def gather(words: torch.Tensor) -> torch.Tensor:
        got = torch.empty((comm.world * words.numel(),), dtype=words.dtype, device=words.device)
        comm.all_gather(words, got)
        return got.view(comm.world, -1)

    gather._comm = comm  # stable owner for an optional GPU-stage cache; closures themselves are per call
    return gather


def dist_gather(words: torch.Tensor) -> torch.Tensor:
    """``nucleus_rows``'s gather over torch.distributed (the 27B's two ranks)."""

    import torch.distributed as dist

    got = torch.empty((dist.get_world_size(), words.numel()), dtype=words.dtype, device=words.device)
    dist.all_gather_into_tensor(got, words)
    return got


def nucleus_rows(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling, *, offset: int = 0,
                 id_map: torch.Tensor | None = None, gather: Callable = one_rank,
                 probs: list[float] | None = None, fallback_candidates: int = 0,
                 row_seeds: Sequence[int] | None = None, initial_candidates: int | None = None, stage: Callable | None = None) -> list[int]:
    """Exact fixed-point nucleus; intermediate candidates must pass the SAME coverage proof.

    0 preserves the old whole-shard fallback. All participating ranks must use the
    same budget. Existing callers/families remain unchanged unless they opt in.
    ``row_seeds`` batches independent keys under common sampling parameters; it
    reuses _draw unchanged and widens only rows whose strict coverage proof fails.
    A smaller initial budget must pass the same proof; failures still visit the
    original 1024 budget before the optional intermediate/whole-shard fallbacks.
    """
    initial_candidates = NUCLEUS if initial_candidates is None else initial_candidates
    if not 1 <= initial_candidates <= NUCLEUS:
        raise ValueError("initial_candidates must be in 1..1024")
    if fallback_candidates and fallback_candidates <= NUCLEUS:
        raise ValueError("fallback_candidates must be 0 or greater than the initial candidate count")

    if row_seeds is not None and (len(row_seeds) != logits.shape[0] or len(positions) != logits.shape[0]):
        raise ValueError("row_seeds and positions must contain one key per logits row")
    if stage is None:
        scaled, mass = _scaled_mass(logits, sampling.temperature, gather)
        got = _shares(gather, scaled, mass, initial_candidates, offset, id_map)
    else:
        scaled, mass, got = stage(logits, sampling.temperature, initial_candidates, offset, id_map, gather)
    whole = int(got[4].max())
    can_cut = 0.0 < sampling.top_p < 1.0 or sampling.min_p > 0.0
    budgets = ([v for v in (NUCLEUS, fallback_candidates) if initial_candidates < v < whole] if can_cut else []) + [whole]
    if row_seeds is not None:
        settings = {int(seed): replace(sampling, seed=int(seed)) for seed in row_seeds}
        samplings = [settings[int(seed)] for seed in row_seeds]
        drawn = _draw_seeded(got, positions, samplings)
        for budget in budgets:
            missing = [r for r, value in enumerate(drawn) if value is None]
            if not missing:
                break
            indices = torch.tensor(missing, dtype=torch.int64, device=logits.device)
            wider = _shares(gather, scaled.index_select(0, indices), mass.index_select(0, indices),
                            budget, offset, id_map)
            values = _draw_seeded(wider, [positions[r] for r in missing], [samplings[r] for r in missing])
            for r, value in zip(missing, values):
                drawn[r] = value
        if any(value is None for value in drawn):
            raise RuntimeError("whole-shard nucleus coverage failed")
    else:
        drawn = _draw(got, positions, sampling)
        for budget in budgets:
            if drawn is not None:
                break
            drawn = _draw(_shares(gather, scaled, mass, budget, offset, id_map), positions, sampling)
    if probs is not None:
        probs.extend(share for _, share in drawn)
    return [token for token, _ in drawn]


def _draw_seeded(got, positions, samplings):
    """Apply the unchanged exact draw to each row; retain its individual coverage result."""
    out = []
    for r, (position, sampling) in enumerate(zip(positions, samplings)):
        value = _draw(tuple(a[:, r:r + 1] for a in got), [position], sampling)
        out.append(None if value is None else value[0])
    return out


def _scaled_mass(logits, temperature, gather):
    scaled = logits.float().double() / max(float(temperature), 1e-6)
    if AMAX:
        top = torch.amax(_stacked(gather, torch.amax(scaled, dim=-1)), dim=0)
    else:
        top = _stacked(gather, scaled.max(dim=-1).values).max(dim=0).values
    mass = torch.floor(torch.exp(scaled - top[:, None]) * MASS).to(torch.int64)
    return scaled, mass


def _shares(gather, scaled, mass, count, offset, id_map):
    return _unpack_candidates(_candidate_device(gather, scaled, mass, count, offset, id_map), count)


def _candidate_device(gather, scaled, mass, count, offset, id_map, *, capture=False):
    """Every rank's padded top (value, id, mass) per row, plus the shard's mass and width sums."""

    rows, width = scaled.shape
    vals, cols = torch.topk(scaled, min(count, width), dim=-1, sorted=not UNSORTED)
    ids = id_map[cols].to(torch.int64) if id_map is not None else cols + int(offset)
    pad = count - vals.shape[1]
    if pad:
        vals = torch.cat([vals, vals.new_full((rows, pad), float("-inf"))], dim=1)
        ids = torch.cat([ids, ids.new_full((rows, pad), -1)], dim=1)
    kept = mass.gather(1, cols)
    kept = torch.cat([kept, kept.new_zeros((rows, pad))], dim=1) if pad else kept
    # A CUDA graph must not create a CPU literal followed by a synchronous H2D copy.
    shard = (torch.full((rows, 1), width, dtype=torch.int64, device=scaled.device) if capture else
             torch.tensor([[width]], dtype=torch.int64, device=scaled.device).expand(rows, 1))
    packed = torch.cat([vals.view(torch.int64), ids, kept, mass.sum(dim=-1, keepdim=True), shard], dim=1)
    return _stacked(gather, packed)


def _unpack_candidates(device_candidates, count):
    both = device_candidates.cpu().numpy()
    return (np.ascontiguousarray(both[:, :, :count]).view(np.float64), both[:, :, count:2 * count],
            both[:, :, 2 * count:3 * count], both[:, :, 3 * count], both[:, :, 3 * count + 1])


class NucleusStage:
    """Fixed GPU arithmetic/collectives only; seeds, positions, coverage and draws stay outside.

    Owned and bounded by the existing model Graphs cache. Each stage has a private
    CUDA graph pool, so model graphs and other row shapes cannot alias its live
    scaled/mass tensors during a selective fallback. No allocation occurs on replay.
    Every participating rank must capture/replay the same stage sequence.
    """

    def __init__(self, logits, temperature, count, offset, id_map, gather):
        self.input = torch.empty_like(logits)
        self.input.copy_(logits)
        self.count, self.id_map, self.gather = count, id_map, gather
        for _ in range(2):
            scaled, mass = _scaled_mass(self.input, temperature, gather)
            _candidate_device(gather, scaled, mass, count, offset, id_map, capture=True)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.scaled, self.mass = _scaled_mass(self.input, temperature, gather)
            self.device_candidates = _candidate_device(gather, self.scaled, self.mass, count, offset, id_map, capture=True)

    def __call__(self, logits):
        self.input.copy_(logits)
        self.graph.replay()
        return self.scaled, self.mass, _unpack_candidates(self.device_candidates, self.count)


@lru_cache(maxsize=64)
def _singleton_thresholds(p):
    q=Fraction(p)
    return q.numerator,q.denominator,((2**40)*q.denominator)//q.numerator


def _singleton_draw(got, positions, s):
    if len(positions)!=1 or not 0.<s.top_p<1.:
        return None
    vals,ids,mass,sums,widths=got
    total=int(sums[:,0].sum())
    if total<=0:return None
    numerator,denominator,limit=_singleton_thresholds(s.top_p)
    if total>limit:return None
    need=(numerator*total+denominator-1)//denominator
    v,i,m=vals[:,0].reshape(-1),ids[:,0].reshape(-1),mass[:,0].reshape(-1)
    top=int(np.argmax(v))
    if i[top]<0 or not np.isfinite(v[top]) or int(m[top])<need:
        return None
    if np.any(m<0) or np.count_nonzero(v==v[top])!=1:
        return None
    if s.min_p>0. and not v[top]>=vals[:,0,:].max()+s.min_log:
        return None
    for k in range(vals.shape[0]):
        real=ids[k,0]>=0
        if int(real.sum())==widths[k,0]:continue
        if not v[top]>vals[k,0][real].min():
            return None
    # Preserve the original keyed generator's input-domain errors as well.
    if type(s.seed) is not int:return None
    try:position=int(positions[0])
    except (TypeError,ValueError,OverflowError):return None
    if not 0<=position<=0xffffffffffffffff:return None
    return [(int(i[top]),float(m[top])/total)]


def _draw(got, positions, s: Sampling) -> list[tuple[int, float]] | None:
    """Each row's (token, its share of the mass) from every rank's candidates; None if a row needs whole shards."""

    if SINGLETON:
        single = _singleton_draw(got, positions, s)
        if single is not None:
            return single
    vals, ids, mass, sums, widths = got
    cut = 0.0 < s.top_p < 1.0
    drawn = []
    for r, position in enumerate(positions):
        total = int(sums[:, r].sum())
        need = math.ceil(Fraction(s.top_p) * total) if cut else None
        floor = vals[:, r, :].max() + s.min_log                   # the min_p cut (-inf when off)
        # Certify coverage from the GLOBAL candidate prefix, not from each shard
        # independently reaching the global top_p mass. The latter forces whole-
        # vocabulary transfers whenever probability is spread over multiple ranks.
        v, i, m = vals[:, r].reshape(-1), ids[:, r].reshape(-1), mass[:, r].reshape(-1)
        order = np.lexsort((i, -v))
        order = order[i[order] >= 0]
        v, i, m = v[order], i[order], m[order]
        cumulative = np.cumsum(m)
        boundary = None
        if cut:
            at = np.nonzero(cumulative >= need)[0]
            if len(at):
                boundary = v[int(at[0])]
        for k in range(vals.shape[0]):
            real = ids[k, r] >= 0
            if int(real.sum()) == widths[k, r]:
                continue                         # the entire shard is already present
            last = vals[k, r][real].min()
            # topk can return an arbitrary subset of a boundary tie. Strict > is
            # essential: equality must still read the whole shard and order by ID.
            covered = (boundary is not None and boundary > last) if cut else (s.min_p > 0 and last < floor)
            if not covered:
                return None
        keep = len(v)
        if cut:
            keep = int((cumulative < need).sum()) + 1
        if s.min_p > 0.0:
            keep = min(keep, int((v >= floor).sum()))
        score = v[:keep] - np.log(-np.log(uniform(s.seed, int(position), i[:keep])))
        best = int(np.argmax(score))
        drawn.append((int(i[best]), float(m[best]) / total))
    return drawn
