"""P0 contracts, using real decoder control flow and CPU tensors (no CUDA context)."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tensorfold.cuda.geometry import split_weights
from tensorfold.engine.exact_sampling import Sampling, choose_rows
from tensorfold.families.glm5_next.cuda import decode as dec
from tensorfold.families.glm5_next.cuda import drafter_choice as auto
from tensorfold.families.glm5_next.cuda.split import rule, rule_full
from tensorfold.families.glm5_next.cuda.weights53 import private_head


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_head_private_storage_and_single_split_estimate(world, dtype):
    full = torch.arange(64 * 16).reshape(64, 16).to(dtype)
    for rank in range(world):
        head = private_head(full, rank, world, 64, "cpu")
        assert torch.equal(head.weight, full[rank * 64 // world:(rank + 1) * 64 // world].bfloat16())
        assert head.weight.untyped_storage().nbytes() == head.weight.numel() * 2
        assert head.weight.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()
    estimate = split_weights(rule_full, world, draft_head=False)
    assert estimate("lm_head.weight", {"shape": [64, 16], "dtype": "BF16"}) == (64 * 16 * 2 // world, 0)
    assert estimate("lm_head.weight", {"shape": [64 // world, 16], "dtype": "BF16", "split": True}) == (64 * 16 * 2 // world, 0)
    # Flash still builds its additional Q4 head, including on pre-split folders where the head was replicated.
    flash = split_weights(rule, world)
    assert flash("lm_head.weight", {"shape": [64, 16], "dtype": "BF16", "split": True}) == (64 * 16 * 2 // world + 64 * 16 // world * 9 // 16, 0)


def run_decoder(kind, count, depth, reject_at, eos_position):
    st = SimpleNamespace(pos=10, mtp_drafted=0)
    emitted, followed, committed, draft_counts = [], [], [], []
    eos = [] if eos_position is None else [100 + eos_position]
    w = SimpleNamespace(cfg=SimpleNamespace(eos=eos, hidden=2), device="cpu")
    e = SimpleNamespace(w=w, st=st, buf=SimpleNamespace(taps=[]), last_hidden=torch.zeros(1, 2),
                        verify_window=lambda t: t, follow=followed.extend,
                        sample=lambda tokens, positions, sampling: [t + 1 for t in tokens],
                        main_hidden=lambda sl: torch.zeros(sl.stop, 2), tap_rows=lambda n: torch.empty(n, 0))
    def forward(tokens):
        e.window_tokens = list(tokens)
        return tokens
    e.forward = forward
    def draft(engine, hidden, nxt, position, number, sampling, confidence):
        draft_counts.append(number)
        return [nxt[-1] + i + 1 + (1000 if i >= reject_at else 0) for i in range(number)]
    def commit(weights, state, buffers, rows, keep):
        assert 1 <= keep <= rows <= depth + 1
        committed.extend(e.window_tokens[:keep])
        state.pos += keep
    drafter = SimpleNamespace(propose=lambda pending, depth, sampling, confidence: draft(
        e, None, [pending], st.pos + 1, depth, sampling, confidence), add_taps=lambda taps: None)
    policy = dec.DepthPolicy(depth, fixed=True)
    with patch.object(dec, "_sync", lambda w: None), patch.object(auto, "_sync", lambda w: None), \
         patch.object(torch.cuda, "synchronize", lambda: None), \
         patch.object(dec, "draft", draft), patch.object(auto, "draft", draft), \
         patch.object(dec, "commit", commit), patch.object(auto, "commit", commit):
        if kind == "auto":
            result = auto.auto_decode(e, None, 100, count, None, choice=None, m_policy=policy,
                                      f_policy=policy, stop_eos=True, on_tokens=emitted.extend)
        elif kind == "dflash":
            result = dec.dflash_decode(e, drafter, 100, count, None, policy=policy,
                                       stop_eos=True, on_tokens=emitted.extend)
        else:
            result = dec.mtp_decode(e, 100, count, None, policy=policy,
                                    stop_eos=True, on_tokens=emitted.extend)
    n = count if eos_position is None else min(count, eos_position + 1)
    expected = list(range(100, 100 + n))
    assert result.tokens == expected
    assert emitted == followed == expected[1:]
    assert committed == expected[:-1]
    assert st.pos == 10 + len(expected) - 1
    assert sum(result.keeps) == len(committed)
    assert all(k <= depth for k in draft_counts)
    if n == 1:
        assert not draft_counts
    return result


@pytest.mark.parametrize("kind", ["mtp", "dflash", "auto"])
def test_terminal_room_rejection_and_every_eos_position(kind):
    # GPU arithmetic is stubbed, but each real decoder's draft/verify/accept/commit control flow runs.
    for depth in range(8):
        for count in (1, 2, 3, 4, 5, 8, 9, 16, 17):
            for reject_at in range(depth + 1):
                for eos in [None, *range(10)]:
                    run_decoder(kind, count, depth, reject_at, eos)


@pytest.mark.parametrize("sampling", [None, Sampling(42, 1.0, 4, 0.95), Sampling(42, 1.0, 0, 0.95)])
def test_padding_does_not_enter_sampling_or_mutate_logits(sampling):
    w = SimpleNamespace(comm=None, vocab_offset=12, meta={"valid_vocab": 18})
    logits = torch.tensor([[0., 1., 2., 3., 4., 5., 999., 999.]], dtype=torch.bfloat16)
    saved = logits.clone()
    got = dec.sample_rows(w, logits, [17], sampling)
    plain = SimpleNamespace(comm=None, vocab_offset=12)
    want = dec.sample_rows(plain, logits[:, :6], [17], sampling)
    assert got == want and got[0] < 18
    assert torch.equal(logits, saved)


def test_large_ties_use_smallest_ids_for_greedy_and_topk():
    w = SimpleNamespace(comm=None, vocab_offset=100)
    logits = torch.ones(4, 256, dtype=torch.bfloat16)
    pos = [7, 8, 9, 10]
    assert dec.sample_rows(w, logits, pos, None) == [100] * 4
    probs = []
    assert dec.sample_rows(w, logits, pos, None, probs=probs) == [100] * 4
    smp = Sampling(123, 1.0, 20, 0.95)
    ids = np.broadcast_to(np.arange(100, 356), (4, 256))
    assert dec.sample_rows(w, logits, pos, smp) == choose_rows(np.ones((4, 256)), ids, pos, smp)


@pytest.mark.parametrize("parallel", [0, -1, 5, 8])
def test_out_of_range_parallel_never_loads(monkeypatch, parallel):
    from tensorfold.families import glm5_next as family
    from tensorfold.families.glm5_next.cuda import engine
    monkeypatch.setattr(family, "_is_full", lambda _: True)
    def forbidden(*a, **kw):
        pytest.fail("must reject parallel before loading the model or creating CUDA state")
    monkeypatch.setattr(engine, "GlmEngine", forbidden)
    with pytest.raises(ValueError, match="parallel"):
        family.cuda_engine("unused", tp=4, master="127.0.0.1", parallel=parallel)


@pytest.mark.parametrize("parallel", [None, "auto", 1])
def test_cuda_auto_keeps_single_stream(monkeypatch, parallel):
    from tensorfold.families import glm5_next as family
    from tensorfold.families.glm5_next.cuda import engine
    monkeypatch.setattr(family, "_is_full", lambda _: True)
    monkeypatch.setattr(engine, "GlmEngine", lambda *a, **kw: kw)
    result = family.cuda_engine("unused", tp=4, master="127.0.0.1", parallel=parallel)
    assert result["world"] == 4


@pytest.mark.parametrize("parallel", [2, 3, 4])
def test_explicit_parallel_is_wired_to_glm(monkeypatch, parallel):
    from tensorfold.families import glm5_next as family
    from tensorfold.families.glm5_next.cuda import engine
    monkeypatch.setattr(family, "_is_full", lambda _: True)
    monkeypatch.setattr(engine, "GlmEngine", lambda *a, **kw: kw)
    result = family.cuda_engine("unused", tp=4, master="127.0.0.1", parallel=parallel)
    assert result["parallel"] == parallel and result["world"] == 4
