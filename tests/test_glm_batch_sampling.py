"""Exercise actual MTP proposal control with CPU logits, including skipped targets/confidence."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode, multi


@dataclass
class State:
    pos: int
    mtp_len: int
    mtp_drafted: int = 1

    def set_mtp_len(self, value):
        self.mtp_len = value


@pytest.mark.parametrize("temperature_scale", [1., .8, 1.2])
@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("confidence", [0., .7])
def test_propose_batch_keeps_targets_keys_and_suffix_rollback(monkeypatch, greedy, confidence, temperature_scale):
    def streams():
        out = []
        for i, target in enumerate([0, 1, 3]):
            pos = 100 + i * 17
            policy = SimpleNamespace(temperature_scale=temperature_scale, confidence=confidence, next=lambda *args, target=target: target)
            carry = SimpleNamespace(policy=policy, tokens=[7, 9], hidden=torch.ones(2, 8),
                                    last_drafted=2, last_accepted=1)
            out.append(SimpleNamespace(sid=i, st=State(pos, pos-1), snap=carry, count=16, out=[9],
                                       sampling=None if greedy else Sampling(5300+i, 1., 0, .95)))
        return out
    def mtp(windows, *, chain=False):
        logits = torch.stack([((torch.arange(11).float() + st.pos + st.mtp_len) % 7) * .25
                              for st, tokens, hidden in windows])
        return logits, torch.zeros(len(windows), 8)
    decoder = SimpleNamespace(w=SimpleNamespace(vocab_offset=0, comm=None, world=1, meta={"valid_vocab":11}), _mtp=mtp, graphs=None)
    results, calls, scalar_calls, scalar_by_mode = [], [], [], []
    scalar = decode.sample_rows
    def counted(w, logits, positions, sampling, *args, **kwargs):
        scalar_calls.append(logits.shape[0])
        return scalar(w, logits, positions, sampling, *args, **kwargs)
    monkeypatch.setattr(decode, "sample_rows", counted)
    original = decode.sample_many
    def batched(w, logits, positions, samplings, **kwargs):
        calls.append((tuple(tuple(p) for p in positions), [None if s is None else s.seed for s in samplings]))
        return original(w, logits, positions, samplings, **kwargs)
    monkeypatch.setattr(decode, "sample_many", batched)
    for enabled in (False, True):
        monkeypatch.setattr(decode, "BATCH_SAMPLING", enabled)
        scalar_calls.clear()
        live = streams()
        multi.MultiDecoder._propose(decoder, live)
        scalar_by_mode.append(list(scalar_calls))
        results.append([(s.drafts, s.st.mtp_len, s.st.mtp_drafted) for s in live])
    assert results[0] == results[1]
    assert results[1][0][0] == []
    if confidence == 0.:
        assert [len(v[0]) for v in results[1]] == [0, 1, 3]
        assert calls[0][0] == ((118,), (135,))
        if not greedy:
            assert calls[0][1] == [5301, 5302]
    else:
        assert len(calls) == 1  # same exact probabilities stop every stream after its first draft

    if not greedy:
        assert scalar_by_mode[1] == []  # no discarded batched draw followed by a duplicate scalar draw
    elif confidence == 0.:
        assert scalar_by_mode[1] == [2, 1, 1]
    else:
        assert scalar_by_mode[1] == [1, 1]  # exact greedy confidence retains its scalar fallback
