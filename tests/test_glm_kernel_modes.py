"""CPU checks of opt-in layouts: fixed reductions, early validation and TP namespace."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda import decode, engine, forward, graphs, weights53


@pytest.mark.parametrize("mode, expected", [("legacy", 0), ("transpose", 1), ("transpose_cooperative", 3)])
def test_projection_layout_selected_after_legacy_tuning(monkeypatch, mode, expected):
    calls = []
    def tune(layer, m):
        calls.append((layer.kernel_mode, m))
        layer.split = (3, 4)
    monkeypatch.setattr(weights53, "LINEAR_MODE", mode)
    monkeypatch.setattr(weights53, "tune", tune)
    layer = SimpleNamespace(kernel_mode=1, split=None)
    assert weights53._configure_linear(layer) is layer
    assert calls == [(0, weights53.DECODE_ROWS)]
    assert layer.split == (3, 4) and layer.kernel_mode == expected


@pytest.mark.parametrize("module, name, value", [
    (forward, "GROUPED_MODE", "transpose"),
    (forward, "ROUTE_MODE", "ballot"),
    (decode, "BATCH_SAMPLING", True),
    (decode, "NUCLEUS_INITIAL", 64),
    (graphs, "SAMPLING_GRAPH", True),
    (graphs, "SAMPLING_CACHE_LIMIT", 4),
    (graphs, "GRAPH_MISS_MODE", "once"),
    (engine, "CONTROL_FRAME", True),
    (engine, "MAIN_PAD", True),
    (weights53, "LINEAR_MODE", "transpose"),
    (weights53, "LINEAR_MODE", "transpose_cooperative"),
    (weights53, "CONCAT_DIRECT", False),
    (weights53, "CONCAT_PAIR", False),
    (engine, "GRAPH_ROWS", tuple(range(1, 9))),
])
def test_effective_kernel_flags_change_rank_namespace(monkeypatch, module, name, value):
    monkeypatch.setattr(forward, "GROUPED_MODE", "legacy")
    monkeypatch.setattr(forward, "ROUTE_MODE", "legacy")
    monkeypatch.setattr(decode, "BATCH_SAMPLING", False)
    monkeypatch.setattr(decode, "NUCLEUS_INITIAL", 1024)
    monkeypatch.setattr(graphs, "SAMPLING_GRAPH", False)
    monkeypatch.setattr(graphs, "SAMPLING_CACHE_LIMIT", 8)
    monkeypatch.setattr(graphs, "GRAPH_MISS_MODE", "legacy")
    monkeypatch.setattr(engine, "CONTROL_FRAME", False)
    monkeypatch.setattr(engine, "MAIN_PAD", False)
    monkeypatch.setattr(weights53, "LINEAR_MODE", "legacy")
    monkeypatch.setattr(weights53, "CONCAT_DIRECT", True)
    monkeypatch.setattr(weights53, "CONCAT_PAIR", True)
    monkeypatch.setattr(engine, "GRAPH_ROWS", (1, 2, 3, 4, 5, 6))
    before = engine.kernel_namespace("test-runtime")
    monkeypatch.setattr(module, name, value)
    after = engine.kernel_namespace("test-runtime")
    assert len(before) == len(after) == 8 and before != after
    assert after == engine.kernel_namespace("test-runtime")


@pytest.mark.parametrize("variable", ["TF_GLM_GROUPED_MODE", "TF_GLM_LINEAR_MODE", "TF_GLM_ROUTE_MODE"])
def test_unknown_kernel_mode_rejected_before_load(variable):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_GLM_GROUPED_MODE="legacy", TF_GLM_LINEAR_MODE="legacy")
    env[variable] = "typo"
    run = subprocess.run([sys.executable, "-c", "from tensorfold.families.glm5_next.cuda import forward"],
                         env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode != 0 and variable in run.stderr and "ValueError" in run.stderr


def test_kernel_modes_are_opt_in():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("TF_GLM_GROUPED_MODE", None)
    env.pop("TF_GLM_LINEAR_MODE", None)
    env.pop("TF_GLM_ROUTE_MODE", None)
    run = subprocess.run([sys.executable, "-c", "from tensorfold.families.glm5_next.cuda import forward as f, weights53 as w; "
                          "assert f.GROUPED_MODES[f.GROUPED_MODE] == w.LINEAR_MODES[w.LINEAR_MODE] == f.ROUTE_MODES[f.ROUTE_MODE] == 0"],
                         env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr


def test_batch_sampling_default_and_invalid_setting():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("TF_GLM_BATCH_SAMPLING", None)
    command = [sys.executable, "-c", "from tensorfold.families.glm5_next.cuda import decode; assert not decode.BATCH_SAMPLING"]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    env["TF_GLM_BATCH_SAMPLING"] = "typo"
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode != 0 and "TF_GLM_BATCH_SAMPLING" in run.stderr and "ValueError" in run.stderr


def test_sampling_graph_default_and_invalid_setting():
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("TF_GLM_SAMPLING_GRAPH",None)
    cmd=[sys.executable,"-c","from tensorfold.families.glm5_next.cuda import graphs; assert not graphs.SAMPLING_GRAPH"]
    run=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    assert run.returncode==0,run.stderr
    env["TF_GLM_SAMPLING_GRAPH"]="typo"
    run=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    assert run.returncode!=0 and "TF_GLM_SAMPLING_GRAPH" in run.stderr and "ValueError" in run.stderr


@pytest.mark.parametrize("variable,bad", [
    ("TF_GLM_CONTROL_FRAME","typo"), ("TF_GLM_INPLACE_GATHER","typo"), ("TF_GLM_MAIN_PAD","typo"), ("TF_GLM_GRAPH_MISS_MODE","typo"),
    ("TF_GLM_SAMPLING_CACHE","0"), ("TF_GLM_SAMPLING_CACHE","65"),
])
def test_control_and_cache_defaults_and_validation(variable, bad):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for name in ("TF_GLM_CONTROL_FRAME","TF_GLM_MAIN_PAD","TF_GLM_GRAPH_MISS_MODE","TF_GLM_SAMPLING_CACHE"):
        env.pop(name,None)
    cmd = [sys.executable,"-c","from tensorfold.families.glm5_next.cuda import engine, graphs; "
           "assert not engine.CONTROL_FRAME; assert not engine.MAIN_PAD; assert graphs.GRAPH_MISS_MODE == 'legacy'; "
           "assert graphs.SAMPLING_CACHE_LIMIT == 8"]
    run = subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    assert run.returncode == 0, run.stderr
    env[variable] = bad
    run = subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    assert run.returncode != 0 and variable in run.stderr and "ValueError" in run.stderr
