"""CUDA fixed-point mass and keyed draws stay exact when compatible streams share transfers."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sampling import nucleus_rows
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 2, 3, 6, 9, 12, 32])
@pytest.mark.parametrize("initial", [64, 1024])
@pytest.mark.parametrize("scale", [1., 3., .01, -1.])
@pytest.mark.parametrize("shape", ["random", "flat", "mixed"])
@pytest.mark.parametrize("top_p,min_p", [(.95, 0.), (1., .01)])
def test_original_gpu_mass_rows_and_probabilities(rows, shape, top_p, min_p, initial, scale):
    gen = torch.Generator(device="cuda").manual_seed(271 + rows)
    logits = (torch.randn(rows, 38720, device="cuda", generator=gen) * (2.5 * scale)).bfloat16()
    if shape == "flat": logits.zero_()
    if shape == "mixed":
        logits[::2].fill_(-12.)
        logits[::2, 71] = 10.
        logits[1::2].zero_()
    logits[:, -24:] = -torch.inf
    seeds = [(2**64-1, 0, 5300)[i % 3] for i in range(rows)]
    positions = list(range(2047, 2047 + rows))
    sampling = Sampling(91, .9, 0, top_p, min_p)
    expected, expected_probs, probs = [], [], []
    for r in range(rows):
        expected += nucleus_rows(logits[r:r + 1], positions[r:r + 1], replace(sampling, seed=seeds[r]),
                                  offset=116160, probs=expected_probs)
    got = nucleus_rows(logits, positions, sampling, offset=116160, row_seeds=seeds,
                        fallback_candidates=4096, probs=probs, initial_candidates=initial)
    assert got == expected and probs == expected_probs


@pytest.mark.parametrize("greedy", [True, False])
def test_glm_sample_many_matches_existing_shard_draw(greedy):
    logits = torch.randn(9, 38720, device="cuda", dtype=torch.bfloat16)
    logits[:, -24:] = 1000.
    w = SimpleNamespace(vocab_offset=116160, comm=None, world=1, meta={"valid_vocab":154856})
    positions = [[31, 32, 33], [2047], [98, 99], [517, 518, 519]]
    samplings = [None if greedy else Sampling(i + 5300, 1., 0, .95) for i in range(4)]
    expected, start = [], 0
    for pos, smp in zip(positions, samplings):
        expected.append(decode.sample_rows(w, logits[start:start + len(pos)], pos, smp))
        start += len(pos)
    assert decode.sample_many(w, logits, positions, samplings) == expected
