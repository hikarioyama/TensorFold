"""Intermediate candidates use the existing coverage proof and preserve whole-shard draws."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from tensorfold.cuda import sampling as cs
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda.engine import kernel_namespace
from tensorfold.families.glm5_next.cuda import decode as glm_decode


def draw(logits, sampling, *, id_map=None, fallback_candidates=0):
    probs = []
    tokens = cs.nucleus_rows(logits, list(range(111, 111 + len(logits))), sampling, id_map=id_map, probs=probs,
                             fallback_candidates=fallback_candidates)
    return tokens, probs


@pytest.mark.parametrize("top_p,min_p", [(.5, 0.), (.95, 0.), (1., 0.), (1., .02)])
@pytest.mark.parametrize("rows", [1, 3, 12])
def test_intermediate_matches_original_ids_and_probability_bits(monkeypatch, top_p, min_p, rows):
    gen = torch.Generator().manual_seed(rows)
    logits = (torch.randn(rows, 4096, generator=gen) * 3).bfloat16()
    mapping = torch.randperm(logits.shape[1], generator=gen) + 7777
    sampling = Sampling(2**64-1, .9, 0, top_p, min_p)
    monkeypatch.setattr(cs, "NUCLEUS", 16)
    want = draw(logits, sampling, id_map=mapping)
    for budget in (64, 512, 4096, 8192):
        assert draw(logits, sampling, id_map=mapping, fallback_candidates=budget) == want


def test_intermediate_avoids_full_transfer_when_coverage_is_proven(monkeypatch):
    logits = torch.linspace(8, -8, 512)[None].bfloat16()
    sampling = Sampling(43, 1., 0, .95)
    monkeypatch.setattr(cs, "NUCLEUS", 16)
    want = draw(logits, sampling)
    reads, real = [], cs._shares
    monkeypatch.setattr(cs, "_shares", lambda *a: reads.append(a[3]) or real(*a))
    assert draw(logits, sampling, fallback_candidates=128) == want and reads == [16, 128]


@pytest.mark.parametrize("top_p,expected", [(.05, [16, 128, 512]), (.95, [16, 128, 512]), (1., [16, 512])])
def test_boundary_ties_and_uncut_mass_still_use_whole_shard(monkeypatch, top_p, expected):
    logits = torch.zeros((3, 512), dtype=torch.bfloat16)
    sampling = Sampling(84, 1., 0, top_p)
    monkeypatch.setattr(cs, "NUCLEUS", 16)
    want = draw(logits, sampling)
    reads, real = [], cs._shares
    monkeypatch.setattr(cs, "_shares", lambda *a: reads.append(a[3]) or real(*a))
    assert draw(logits, sampling, fallback_candidates=128) == want and reads == expected


@pytest.mark.parametrize("flat", [False, True])
def test_four_uneven_ranks_agree_and_keep_sampling_bits(monkeypatch, flat):
    logits = (torch.randn(3, 4096, generator=torch.Generator().manual_seed(200)) * 3).bfloat16()
    if flat:
        logits.zero_()
    sampling = Sampling(531, 1., 0, .95)
    monkeypatch.setattr(cs, "NUCLEUS", 16)
    want = draw(logits, sampling)
    cuts, slots = [0, 23, 1000, 2300, 4096], [None] * 4
    barrier = threading.Barrier(4, timeout=60)
    def rank(r):
        def gather(words):
            slots[r] = words.clone()
            barrier.wait()
            result = torch.stack(slots)
            barrier.wait()
            return result
        probs = []
        got = cs.nucleus_rows(logits[:, cuts[r]:cuts[r + 1]], [111, 112, 113], sampling,
                              offset=cuts[r], gather=gather, probs=probs, fallback_candidates=512)
        return got, probs
    with ThreadPoolExecutor(4) as pool:
        jobs = [pool.submit(rank, r) for r in range(4)]
        assert all(job.result(timeout=90) == want for job in jobs)


def test_candidate_budget_is_in_rank_namespace(monkeypatch):
    monkeypatch.setattr(glm_decode, "NUCLEUS_FALLBACK", 0)
    before = kernel_namespace("test-runtime")
    monkeypatch.setattr(glm_decode, "NUCLEUS_FALLBACK", 4096)
    assert kernel_namespace("test-runtime") != before


@pytest.mark.parametrize("budget", [-1, 1, 1024])
def test_invalid_candidate_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="fallback_candidates"):
        draw(torch.zeros((1, 8)), Sampling(1, 1., 0, .95), fallback_candidates=budget)
