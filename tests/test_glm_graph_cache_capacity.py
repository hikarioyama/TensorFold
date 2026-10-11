"""The existing graph cache remains bounded and releases every pointer on invalidation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import graphs, segmented
from tensorfold.families.glm5_next.cuda.engine import kernel_namespace


@pytest.mark.parametrize("limit", [1, 16, 64])
def test_configured_bound_eviction_replay_and_release(monkeypatch, limit):
    calls = []
    monkeypatch.setattr(graphs, "GRAPH_MISS_MODE", "legacy")
    monkeypatch.setattr(graphs, "PACKED_CACHE_LIMIT", limit)
    monkeypatch.setattr(segmented, "compute_streams", lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: nullcontext())
    g = object.__new__(graphs.Graphs)
    g.pool = None
    w = SimpleNamespace(cfg=SimpleNamespace(dense_limit=2048))
    b = SimpleNamespace(valid=None, logits=torch.zeros(1, 3))
    states = [SimpleNamespace(pos=1, capacity=32) for _ in range(limit + 1)]
    for st in states:
        g.packed_forward(w, [(st, 0, 1)], b)
        assert len(g.packed) <= limit
    assert g.packed_misses == limit + 1 and len(calls) == 3 * (limit + 1)
    g.packed_forward(w, [(states[-1], 0, 1)], b)
    assert g.packed_hits == 1
    g.packed_forward(w, [(states[0], 0, 1)], b)
    assert g.packed_misses == limit + 2
    g.release_packed()
    assert not g.packed


def test_graph_bound_is_in_rank_namespace(monkeypatch):
    monkeypatch.setattr(graphs, "PACKED_CACHE_LIMIT", 16)
    before = kernel_namespace("test-runtime")
    monkeypatch.setattr(graphs, "PACKED_CACHE_LIMIT", 64)
    assert kernel_namespace("test-runtime") != before
