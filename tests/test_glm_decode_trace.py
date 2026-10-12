"""Host-only telemetry must not call synchronization or read CUDA tensors."""
import json
from types import SimpleNamespace

import pytest
import torch
from tensorfold.families.glm5_next.cuda import decode_trace as trace


def test_disabled_decorator_is_identity(monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", "")
    def f(e):
        return 3
    assert trace.traced(f) is f
    with trace.step(SimpleNamespace(), "main", 2, "main", True):
        pass


def test_observer_counts_actual_rows_without_synchronizing(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("observer synchronized CUDA")
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    observer = trace.Observer()
    e = SimpleNamespace(_decode_observer=observer)
    with trace.step(e, "main", 2, "sparse", True, 8192, 0):
        pass
    with trace.step(e, "main", 1, "main", False, None, 0):
        pass
    with trace.step(e, "mtp", 1, "sparse_mtp", True, 8192):
        pass
    assert observer.main_steps == 2
    assert observer.paths == {
        "main/R2/sparse/hit/bucket=8192/parity=0": 1,
        "main/R1/main/eager/bucket=None/parity=0": 1,
        "mtp/R1/sparse_mtp/hit/bucket=8192/parity=None": 1,
    }
    assert set(observer.host_seconds) == set(observer.paths)


def test_request_summary_and_engine_lifetime(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "CUPTI", False)
    previous = object()
    e = SimpleNamespace(w=SimpleNamespace(rank=2), st=SimpleNamespace(pos=7400), _decode_observer=previous)
    result = SimpleNamespace(seconds=1.2, rounds=2, tokens=[1, 2, 3], drafted=1, accepted=0)
    @trace.traced
    def run(engine):
        with trace.step(engine, "main", 2, "sparse", True, 8192, 0):
            pass
        return result
    assert run(e) is result
    assert e._decode_observer is previous
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["completed"] and data["rank"] == 2 and data["start_pos"] == 7400
    assert data["rounds"] == 2 and data["main_steps"] == 1
    assert "not GPU" in data["timing_contract"]


def test_request_failure_restores_observer(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "CUPTI", False)
    e = SimpleNamespace(w=SimpleNamespace(), st=SimpleNamespace(pos=0))
    @trace.traced
    def fail(engine):
        raise ValueError("test failure")
    with pytest.raises(ValueError, match="test failure"):
        fail(e)
    assert e._decode_observer is None
    assert not json.loads(next(tmp_path.glob("*.json")).read_text())["completed"]


def test_trace_limit_is_shared_across_decode_policies(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "CUPTI", False)
    monkeypatch.setattr(trace, "TRACE_LIMIT", 1)
    e = SimpleNamespace(w=SimpleNamespace(rank=0), st=SimpleNamespace(pos=0))
    result = SimpleNamespace(seconds=1., rounds=1, tokens=[1, 2], drafted=0, accepted=0)
    first = trace.traced(lambda e: result)
    second = trace.traced(lambda e: result)
    assert first(e) is result and second(e) is result
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert e._decode_trace_requests == 1


def test_export_is_after_decode_and_callback_cycle_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(trace, "CUPTI", True)
    order, profiles = [], []
    class FakeProfile:
        def __init__(self, **kwargs):
            self.callback = kwargs["on_trace_ready"]
            profiles.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.callback(self)
        def step(self):
            pass
        def export_chrome_trace(self, path):
            order.append("export")
    monkeypatch.setattr(torch.profiler, "profile", FakeProfile)
    e = SimpleNamespace(w=SimpleNamespace(rank=0), st=SimpleNamespace(pos=0))
    result = SimpleNamespace(seconds=1., rounds=1, tokens=[1, 2], drafted=0, accepted=0)
    @trace.traced
    def run(engine):
        order.append("decode")
        return result
    assert run(e) is result
    assert order == ["decode", "export"]
    assert profiles[0].callback.__self__ == []
    assert e._decode_observer is None
