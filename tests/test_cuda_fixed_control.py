"""Bounded control uses native collectives and poisons every rank on bad root data."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.comm import FixedIntBroadcast
from tensorfold.families.glm5_next.cuda.multi import MultiDecoder


def frames(world=4):
    wire = []
    def comm(rank):
        cursor = 0
        def gather(send, recv):
            nonlocal cursor
            if rank == 0:
                wire.append(send.clone())
            recv.zero_()
            recv[:send.numel()].copy_(wire[cursor])
            cursor += 1
        return SimpleNamespace(world=world, all_gather=gather)
    return [FixedIntBroadcast(comm(r), 'cpu', rank=r) for r in range(world)]


def test_repeated_frames_have_no_stale_payload():
    ranks = frames()
    for payload, limit, allow_empty in [(list(range(64)),64,False), ([],4,True),
                                       ([-2**31,2**31-1],4,True), ([2],64,False),
                                       (list(range(4)),4,True), ([],64,True)]:
        for r, frame in enumerate(ranks):
            assert frame.broadcast(payload if r == 0 else None, limit=limit,
                                   allow_empty=allow_empty) == payload


@pytest.mark.parametrize('payload,limit,allow_empty', [
    (list(range(65)),64,False), (list(range(5)),4,True),
    ([],64,False), ([2**31],64,False), ([-2**31-1],64,False),
    ([1.5],64,False), (None,64,False), ([object()],64,False),
])
def test_bad_root_fails_all_ranks_and_prevents_reuse(payload, limit, allow_empty):
    ranks = frames()
    for r, frame in enumerate(ranks):
        with pytest.raises(RuntimeError, match='invalid bounded control frame'):
            frame.broadcast(payload if r == 0 else None, limit=limit, allow_empty=allow_empty)
        assert frame.failed
        with pytest.raises(RuntimeError, match='control frame failed'):
            frame.broadcast([1])


def test_adapter_retains_legacy_fallback_and_limits():
    calls = []
    decoder = object.__new__(MultiDecoder)
    decoder.engine = SimpleNamespace(_share=lambda x: calls.append(('legacy', x)) or x,
                                     _ring=lambda: calls.append(('ring',)))
    assert decoder._control_share([3]) == [3]
    decoder.control = SimpleNamespace(broadcast=lambda x, **kw: calls.append((x, kw)) or x)
    decoder._send([2])
    assert decoder._control_share([], ended=True) == []
    assert calls == [('legacy',[3]), ('ring',), ([2], {'limit':64,'allow_empty':False}),
                     ([], {'limit':4,'allow_empty':True})]
