"""Fixed GPU sampling stage retains eager IDs/probabilities and bounded per-shape ownership."""
from types import SimpleNamespace
import pytest
import torch

from tensorfold.cuda.sampling import nucleus_rows
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode, graphs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1,2,3,6,9,12,32])
@pytest.mark.parametrize("scale", [1.,3.,.01,-1.])
@pytest.mark.parametrize("kind", ["random","flat","mixed"])
def test_capture_replay_keeps_keys_mass_and_selective_fallback(rows, scale, kind):
    owner = graphs.Graphs.__new__(graphs.Graphs)
    gen = torch.Generator(device="cuda").manual_seed(517 + rows)
    x = (torch.randn(rows, 38720, device="cuda", generator=gen) * scale).bfloat16()
    if kind == "flat": x.zero_()
    if kind == "mixed":
        x[::2].fill_(-12.)
        x[::2, 17] = 12.
        x[1::2].zero_()
    x[:, -24:] = -torch.inf
    seeds = [(0, 2**64-1, 5300)[r % 3] for r in range(rows)]
    positions = list(range(2047,2047+rows))
    smp = Sampling(7,.9,0,.95)
    for replay in range(2):
        want_probs, got_probs = [], []
        want = nucleus_rows(x, positions, smp, offset=116160, initial_candidates=64,
                            fallback_candidates=4096, row_seeds=seeds, probs=want_probs)
        got = nucleus_rows(x, positions, smp, offset=116160, initial_candidates=64,
                           fallback_candidates=4096, row_seeds=seeds, probs=got_probs, stage=owner.sampling_stage)
        assert got == want and got_probs == want_probs
        x[:, 31 + replay] = 15.  # changed values must replace prior graph outputs
    assert owner.sampling_misses == 1 and owner.sampling_hits == 1


def test_cache_eviction_and_noncontiguous_logits_do_not_reuse_stale_outputs(monkeypatch):
    monkeypatch.setattr(graphs, "SAMPLING_CACHE_LIMIT", 8)
    owner = graphs.Graphs.__new__(graphs.Graphs)
    for rows in (1,2,3,4,5,6,7,8,9,12,32,1,12,1,3):
        storage = torch.randn(rows, 320, dtype=torch.bfloat16, device="cuda")
        x = storage[:,32:288]
        positions = list(range(rows));smp=Sampling(53,1.,0,.95)
        want = nucleus_rows(x,positions,smp,initial_candidates=64)
        got = nucleus_rows(x,positions,smp,initial_candidates=64,stage=owner.sampling_stage)
        assert got == want and len(owner.sampling_stages) <= graphs.SAMPLING_CACHE_LIMIT
    assert owner.sampling_hits == 2 and owner.sampling_misses == 13


@pytest.mark.parametrize("greedy", [True,False])
def test_confidence_sinks_match_existing_shard_calls(greedy):
    owner = graphs.Graphs.__new__(graphs.Graphs)
    w = SimpleNamespace(vocab_offset=116160,comm=None,world=1,meta={"valid_vocab":154856})
    logits = torch.randn(6,38720,device="cuda",dtype=torch.bfloat16)
    logits[:,-24:]=1000.
    positions=[[31],[1023,1024,1025],[2048,2049]]
    smps=[None if greedy else Sampling(i+5300,1.,0,.95) for i in range(3)]
    want, want_p, start = [], [[],None,[]], 0
    for i,(pos,smp) in enumerate(zip(positions,smps)):
        want.append(decode.sample_rows(w,logits[start:start+len(pos)],pos,smp,probs=want_p[i]))
        start += len(pos)
    got_p=[[],None,[]]
    got=decode.sample_many(w,logits,positions,smps,prob_lists=got_p,stage=owner.sampling_stage)
    assert got == want and got_p == want_p
