"""First-miss execution count, replay freshness, LRU and crossing bypass."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import graphs, segmented


@pytest.mark.parametrize('mode,warm,replays', [('legacy',2,1),('once',1,0)])
@pytest.mark.parametrize('mtp', [False,True])
def test_miss_output_and_changed_input_replay(monkeypatch, mode, warm, replays, mtp):
    calls = dict(warm=0,capture=0,replay=0)
    capturing = []
    b = SimpleNamespace(valid=None, logits=torch.zeros(3,4), value=7.)
    def compute(*args, **kwargs):
        if capturing:
            calls['capture'] += 1
        else:
            calls['warm'] += 1
            b.logits.fill_(b.value)
    class Graph:
        def replay(self):
            calls['replay'] += 1
            b.logits.fill_(b.value)
    @contextmanager
    def capture(g, **kwargs):
        capturing.append(g)
        try: yield
        finally: capturing.pop()
    monkeypatch.setattr(graphs,'GRAPH_MISS_MODE',mode)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda: None)
    monkeypatch.setattr(torch.cuda,'CUDAGraph',Graph)
    monkeypatch.setattr(torch.cuda,'graph',capture)
    monkeypatch.setattr(segmented,'mtp_compute_streams' if mtp else 'compute_streams',compute)
    owner = object.__new__(graphs.Graphs); owner.pool = None
    st = SimpleNamespace(pos=1,mtp_len=1,capacity=64)
    w = SimpleNamespace(cfg=SimpleNamespace(dense_limit=2048))
    segs = [(st,0,3)]
    out = owner.packed_forward(w,segs,b,mtp=mtp)
    assert out.shape == (1 if mtp else 3,4) and (out == 7).all()
    assert calls == dict(warm=warm,capture=1,replay=replays)
    b.value = 11.
    assert (owner.packed_forward(w,segs,b,mtp=mtp) == 11).all()
    assert calls == dict(warm=warm,capture=1,replay=replays+1)
    assert (owner.packed_hits,owner.packed_misses) == (1,1)
    owner.release_packed()
    assert not owner.packed


def test_once_cache_is_lru_and_crossing_does_not_capture(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(graphs,'GRAPH_MISS_MODE','once')
    monkeypatch.setattr(graphs,'PACKED_CACHE_LIMIT',2)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda: None)
    monkeypatch.setattr(torch.cuda,'CUDAGraph',lambda: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(torch.cuda,'graph',lambda *a,**kw: nullcontext())
    calls = []
    monkeypatch.setattr(segmented,'compute_streams',lambda *a,**kw: calls.append(kw) or 'eager')
    owner=object.__new__(graphs.Graphs);owner.pool=None
    w=SimpleNamespace(cfg=SimpleNamespace(dense_limit=2048))
    b=SimpleNamespace(valid=None,logits=torch.zeros(3,4))
    states=[SimpleNamespace(pos=1,capacity=64) for _ in range(3)]
    for i in (0,1,0,2,0):
        owner.packed_forward(w,[(states[i],0,1)],b)
    assert (owner.packed_hits,owner.packed_misses)==(2,3)
    owner.packed_forward(w,[(states[1],0,1)],b)
    assert owner.packed_misses==4
    crossing=SimpleNamespace(pos=2047,capacity=4096)
    assert owner.packed_forward(w,[(crossing,0,3)],b)=='eager'
    assert owner.packed_misses==4 and calls[-1]=={}
