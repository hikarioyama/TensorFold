"""CPU-only checks for prefill observers; no CUDA context or arithmetic changes."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import forward as fw, prof
from tensorfold.cuda.exl3 import experts


@pytest.mark.parametrize("enabled,active", [(False, False), (False, True), (True, False)])
def test_inactive_stream_observer_never_touches_cuda(monkeypatch, enabled, active):
    monkeypatch.setattr(prof, "ENABLED", enabled)
    monkeypatch.setattr(prof, "active", active)
    def forbidden(*a, **kw):
        raise AssertionError("inactive observer touched CUDA")
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    with prof.stream_timed("moe: gate/up"):
        pass


def test_stream_events_defer_sync_until_report(monkeypatch, capsys):
    calls = []
    class Event:
        def __init__(self, **kw):
            assert kw == {"enable_timing": True}
        def record(self):
            calls.append("record")
        def elapsed_time(self, other):
            assert calls[-1] == "sync"
            return 125.
    monkeypatch.setattr(prof, "ENABLED", True)
    monkeypatch.setattr(prof, "active", True)
    monkeypatch.setattr(prof, "events", [])
    monkeypatch.setattr(prof, "totals", {"moe (total)": 1.})
    monkeypatch.setattr(prof, "peaks", {})
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    with prof.stream_timed("moe: gate/up"):
        calls.append("work")
    assert calls == ["record", "work", "record"]
    prof.report(2048)
    assert "moe: gate/up 0.12s" in capsys.readouterr().out
    assert not prof.events and not prof.totals


def test_dsa_prepare_tags_preserve_call_order(monkeypatch):
    calls, tags = [], []
    @contextmanager
    def timed(name):
        tags.append(name)
        yield
    monkeypatch.setattr(prof, "timed", timed)
    monkeypatch.setattr(fw, "mm", lambda *a, **kw: calls.append("mm"))
    monkeypatch.setattr(fw.glue, "rmsnorm", lambda *a, **kw: calls.append("rmsnorm"))
    monkeypatch.setattr(fw, "sums", lambda w, x: None)
    a = SimpleNamespace(proj=None, q_norm=None, kv_norm=None, absorb=True, heads=2, q_b=None)
    w = SimpleNamespace(cfg=SimpleNamespace(q_lora=4, kv_lora=4, qk_dim=4, eps=1e-6))
    b = SimpleNamespace(**{k: torch.empty(3, 8) for k in ("normed", "xs", "dp", "xs_qr", "qr", "lat", "xs_lat", "q")})
    fw.dsa_prepare(SimpleNamespace(dsa=a), w, b, 3)
    assert calls == ["mm", "rmsnorm", "rmsnorm", "mm"]
    assert tags == ["dsa: prepare proj", "dsa: prepare q norm", "dsa: prepare kv norm", "dsa: prepare q_b"]


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("rows", [1, 128, 2048])
def test_exl3_real_branch_emits_tags_without_changing_calls(monkeypatch, deferred, rows):
    calls, tags = [], []
    class Extension:
        def __getattr__(self, name):
            return lambda *a: calls.append(name)
    @contextmanager
    def timed(name):
        tags.append(name)
        yield
    monkeypatch.setattr(experts, "_ext", lambda: Extension())
    ex = SimpleNamespace(dims=8, width=8, count=2, cb=2, k2_gu=(6, 6), k2_d=(6, 6))
    for name in ("suh_g", "suh_u", "gate_ptr", "up_ptr", "gate_k2", "up_k2", "svh_g", "svh_u", "suh_d", "down_ptr", "down_k2", "svh_d"):
        setattr(ex, name, None)
    s = SimpleNamespace(slots=2, rows=rows, route_mode=0, adaptive_tiles=False,
                        grouped_mode=3, cfg_gu=(8, 4, 4, 1), cfg_d=(8, 4, 1, 1),
                        window=lambda n: (None, None), count=None, xg=None, xu=None, z=None,
                        xd=None, y=torch.empty(rows * 2, 8))
    x = torch.empty(rows, 8)
    for observer in (None, timed):
        calls.clear()
        kw = {} if observer is None else {"timed": observer}
        experts.routed(x, None, None, ex, s, None, rows, defer_down=deferred, **kw)
        assert calls == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped"] + ([] if deferred else ["down_epilogue"])
    assert tags == ["moe: expert grouping", "moe: gate/up", "moe: down"] + ([] if deferred else ["moe: down epilogue"])


@pytest.mark.parametrize("prefix", [None, "dsa"])
def test_output_projection_tags_keep_projection_then_gather(monkeypatch, prefix):
    calls, tags = [], []
    @contextmanager
    def timed(name):
        tags.append(name)
        yield
    monkeypatch.setattr(prof, "timed", timed)
    monkeypatch.setattr(fw, "tp_partial", lambda *a: "partial")
    monkeypatch.setattr(fw, "mm", lambda *a, **kw: calls.append(("mm", a[-1], kw)))
    def gather(*a):
        calls.append("gather")
        return "output"
    monkeypatch.setattr(fw, "gather", gather)
    assert fw.out_proj(None, None, None, None, None, 2048, profile=prefix) == "output"
    assert calls == [("mm", "partial", {"f32": True}), "gather"]
    assert tags == ([] if prefix is None else ["dsa: output projection", "dsa: all-gather"])
