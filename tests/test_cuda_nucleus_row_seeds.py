"""Shared exact sampler batching: per-row seeds and strict selective fallback."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import sampling as shared
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode


@pytest.mark.parametrize("top_p,min_p", [(.5, 0.), (.95, 0.), (1., 0.), (1., .01)])
@pytest.mark.parametrize("temperature", [.9, 1.])
@pytest.mark.parametrize("initial", [8, 64, 1024])
def test_row_keys_and_probabilities_match_independent_calls(top_p, min_p, temperature, initial):
    gen = torch.Generator().manual_seed(273)
    logits = (torch.randn(4, 5000, generator=gen) * 2.5).bfloat16()
    logits[1].zero_()
    logits[3, -24:] = -torch.inf
    seeds = [0, 2**64 - 1, 5300, -1]
    positions = [0, 2047, 2048, 40960]
    s = Sampling(5, temperature, 0, top_p, min_p)
    want, want_probs, got_probs = [], [], []
    for r in range(4):
        want += shared.nucleus_rows(logits[r:r + 1], positions[r:r + 1], replace(s, seed=seeds[r]),
                                    offset=10000, probs=want_probs)
    got = shared.nucleus_rows(logits, positions, s, offset=10000, probs=got_probs,
                              fallback_candidates=4096, row_seeds=seeds, initial_candidates=initial)
    assert got == want and got_probs == want_probs


def test_only_uncovered_rows_widen(monkeypatch):
    logits = torch.full((3, 5000), -10.)
    logits[0, 17] = logits[2, 29] = 10.
    logits[1].zero_()
    calls = []
    original = shared._shares
    def shares(gather, scaled, mass, count, offset, id_map):
        calls.append((scaled.shape[0], count))
        return original(gather, scaled, mass, count, offset, id_map)
    monkeypatch.setattr(shared, "_shares", shares)
    got = shared.nucleus_rows(logits, [1, 2, 3], Sampling(8, 1., 0, .95),
                              fallback_candidates=4096, row_seeds=[11, 12, 13])
    assert got[0] == 17 and got[2] == 29
    assert calls == [(3, 1024), (1, 4096), (1, 5000)]


@pytest.mark.parametrize("seeds,positions", [([1], [2, 3]), ([1, 2], [3])])
def test_bad_row_keys_fail_before_shares(monkeypatch, seeds, positions):
    def unexpected(*a, **kw):
        pytest.fail("invalid keys must not reach a collective")
    monkeypatch.setattr(shared, "_shares", unexpected)
    with pytest.raises(ValueError, match="row_seeds"):
        shared.nucleus_rows(torch.zeros(2, 4), positions, Sampling(1), row_seeds=seeds)


@pytest.mark.parametrize("kind", ["greedy", "nucleus", "mixed", "topk", "different_params"])
def test_glm_segments_keep_seeds_positions_and_physical_vocab(kind):
    gen = torch.Generator().manual_seed(72)
    logits = torch.randn(6, 5000, generator=gen).bfloat16()
    logits[:, -24:] = 1000.
    w = SimpleNamespace(vocab_offset=0, comm=None, world=1, meta={"valid_vocab": 4976})
    positions = [[5], [17, 18, 19], [2048, 2049]]
    samplings = [Sampling(seed, 1., 0, .95) for seed in (0, 2**64-1, 11)]
    if kind == "greedy": samplings = [None, Sampling(4, 0.), None]
    if kind == "mixed": samplings[0] = None
    if kind == "topk": samplings = [replace(s, top_k=20) for s in samplings]
    if kind == "different_params": samplings[1] = replace(samplings[1], temperature=.7)
    want, start = [], 0
    for pos, s in zip(positions, samplings):
        want.append(decode.sample_rows(w, logits[start:start + len(pos)], pos, s))
        start += len(pos)
    got = decode.sample_many(w, logits, positions, samplings)
    assert got == want and all(t < 4976 for segment in got for t in segment)


def test_sample_many_invalid_layout_rejected():
    with pytest.raises(ValueError, match="sample_many"):
        decode.sample_many(None, torch.zeros(3, 4), [[1]], [None])


@pytest.mark.parametrize("initial", [0, 1025])
def test_invalid_initial_budget(initial):
    with pytest.raises(ValueError, match="initial_candidates"):
        shared.nucleus_rows(torch.zeros(1, 4), [1], Sampling(1), initial_candidates=initial)


@pytest.mark.parametrize("scale", [1., 3., .01, -1.])
def test_four_shards_agree_on_unresolved_rows_and_keys(scale):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    logits = (torch.randn(3, 4096, generator=torch.Generator().manual_seed(711)) * scale).bfloat16()
    logits[0].fill_(-12.)
    logits[0, 17] = 12.
    logits[1].zero_()
    positions, seeds = [2047, 2048, 40960], [531, 0, 2**64-1]
    sampling = Sampling(8, 1., 0, .95)
    want, probs = [], []
    for r in range(3):
        want += shared.nucleus_rows(logits[r:r+1], [positions[r]], replace(sampling, seed=seeds[r]), probs=probs)
    cuts, slots = [0, 23, 1000, 2300, 4096], [None] * 4
    barrier = threading.Barrier(4, timeout=30)
    def rank(r):
        calls = []
        def gather(words):
            calls.append(words.numel())
            slots[r] = words.clone()
            barrier.wait()
            got = torch.stack(slots)
            barrier.wait()
            return got
        result_probs = []
        ids = shared.nucleus_rows(logits[:, cuts[r]:cuts[r+1]], positions, sampling, offset=cuts[r],
                                  gather=gather, probs=result_probs, row_seeds=seeds,
                                  initial_candidates=64, fallback_candidates=4096)
        return ids, result_probs, calls
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(rank, r) for r in range(4)]
        results = [j.result(timeout=60) for j in jobs]
    assert all(ids == want and values == probs for ids, values, calls in results)
    assert all(calls == results[0][2] for ids, values, calls in results)


@pytest.mark.parametrize("kind", ["nucleus", "greedy", "mixed", "topk"])
def test_probability_sinks_preserve_existing_confidence(kind):
    logits=torch.randn(6,512,generator=torch.Generator().manual_seed(514)).bfloat16()
    w=SimpleNamespace(vocab_offset=0,comm=None,world=1,meta={})
    positions=[[11],[30,31,32],[2047,2048]]
    settings=[Sampling(seed,1.,0,.95) for seed in (5300,0,2**64-1)]
    if kind=="greedy": settings=[None]*3
    if kind=="mixed": settings[1]=None
    if kind=="topk": settings=[replace(s,top_k=20) for s in settings]
    want,want_p,start=[],[[],None,[]],0
    for i,(pos,smp) in enumerate(zip(positions,settings)):
        want.append(decode.sample_rows(w,logits[start:start+len(pos)],pos,smp,probs=want_p[i]))
        start+=len(pos)
    got_p=[[],None,[]]
    assert decode.sample_many(w,logits,positions,settings,prob_lists=got_p)==want
    assert got_p==want_p
