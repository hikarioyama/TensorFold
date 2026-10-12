"""CPU tests of the existing fault monitor lifecycle and TP request wire format."""
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda.engine import GlmEngine
from tensorfold.families.glm5_next.cuda.multi import pack_request, unpack_request


def engine():
    e = object.__new__(GlmEngine)
    e.rank, e.world, e.e = 0, 4, SimpleNamespace(st=SimpleNamespace(pos=9))
    return e


def test_finished_command_never_kills_a_later_idle_server(monkeypatch, tmp_path):
    from tensorfold.families.glm5_next.cuda import engine as module
    called = []
    monkeypatch.setattr(module, 'WATCHDOG_DIR', str(tmp_path))
    monkeypatch.setattr(module.os, '_exit', called.append)
    e = engine()
    token = e._watchdog(time.perf_counter(), 17, timeout_s=.05)
    e._disarm_watchdog(token)
    time.sleep(.12)
    e._watchdog_stop.set()
    assert called == [] and not list(tmp_path.iterdir())


def test_old_disarm_cannot_cancel_a_new_command(monkeypatch, tmp_path):
    from tensorfold.families.glm5_next.cuda import engine as module
    event = threading.Event()
    monkeypatch.setattr(module, 'WATCHDOG_DIR', str(tmp_path))
    monkeypatch.setattr(module.os, '_exit', lambda code: event.set())
    e = engine()
    first = e._watchdog(time.perf_counter(), 17, timeout_s=10)
    second = e._watchdog(time.perf_counter(), 18, timeout_s=.05)
    e._disarm_watchdog(first)
    assert second != first and event.wait(1)
    e._watchdog_stop.set()
    assert (tmp_path / 'hang_rank0.json').is_file()


@pytest.mark.parametrize('seed', [0, 2**63-1, 2**64-1, -1])
def test_request_wire_preserves_float64_and_seed_bits(seed):
    s = Stream([1, 2], 7, Sampling(seed, .987654321, 0, .95123456789, .0000001), stop_eos=False)
    code = [3, 7, 350000, 0]
    header = pack_request(s, code, 29, 1)
    got, policy, cached = unpack_request(header, s.prompt)
    assert got.sampling.seed == seed & ((1 << 64) - 1)
    assert got.sampling.temperature == s.sampling.temperature
    assert got.sampling.top_p == s.sampling.top_p and got.sampling.min_p == s.sampling.min_p
    assert policy == code and cached == 1 and got.sid == 29 and not got.stop_eos
