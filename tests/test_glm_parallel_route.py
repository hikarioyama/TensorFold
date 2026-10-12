"""The stable route mode is explicit and participates in TP configuration identity."""
from types import SimpleNamespace

from tensorfold.cuda.exl3 import experts
from tensorfold.families.glm5_next.cuda import engine, forward


def test_parallel_route_mode_mapping_and_scratch():
    assert forward.ROUTE_MODES["parallel"] == 2
    ex = SimpleNamespace(dims=128, width=128, count=16)
    scratch = experts.Scratch(ex, 2, 9, device="cpu", route_mode=2)
    assert scratch.route_mode == 2
    assert scratch.window(2)[1].shape == (16, 2)


def test_parallel_route_changes_tp_namespace(monkeypatch):
    monkeypatch.setattr(forward, "ROUTE_MODE", "legacy")
    legacy = engine.kernel_namespace("stable-route-test")
    monkeypatch.setattr(forward, "ROUTE_MODE", "parallel")
    assert engine.kernel_namespace("stable-route-test") != legacy
