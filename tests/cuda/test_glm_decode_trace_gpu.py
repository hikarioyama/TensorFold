"""The optional steady-state profiler observes an already captured graph."""
import json
from types import SimpleNamespace

import pytest
import torch
from tensorfold.families.glm5_next.cuda import decode_trace as trace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cupti_trace_contains_replays_not_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "CUPTI", True)
    monkeypatch.setattr(trace, "TRACE_LIMIT", 1)
    x = torch.ones(2, 1024, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1)
    e = SimpleNamespace(w=SimpleNamespace(rank=0), st=SimpleNamespace(pos=7400))
    result = SimpleNamespace(seconds=0., rounds=30, tokens=list(range(31)), drafted=30, accepted=0)
    @trace.traced
    def run(engine):
        for _ in range(30):
            with trace.step(engine, "main", 2, "sparse", True, 8192, 0):
                graph.replay()
            # This is the fake engine's existing dependency wait, not a trace tag.
            torch.cuda.synchronize()
        return result
    assert run(e) is result
    assert torch.all(x == 31)
    events = json.loads(next(tmp_path.glob("*.trace.json")).read_text())["traceEvents"]
    assert any(e.get("cat") == "kernel" for e in events)
    assert any(e.get("name", "").startswith("tf.decode/main/R2/sparse/hit") for e in events)
    assert not any("cudaStreamBeginCapture" in e.get("name", "") for e in events)
    assert not any("cudaGraphInstantiate" in e.get("name", "") for e in events)
