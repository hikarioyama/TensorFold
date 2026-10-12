"""Scheduler.close: the worker stops once idle and lets go of its decoder, so an engine's GPU memory can be freed."""

import gc
import weakref

from tensorfold.cuda.scheduler import Scheduler


class Idle:
    """A decoder with nothing to decode."""

    def live(self):
        return 0

    def round(self):
        return []

    def finish(self, done):
        pass

    def drop(self):
        return []


def test_close_stops_the_worker_and_frees_the_decoder():
    decoder = Idle()
    ref = weakref.ref(decoder)
    scheduler = Scheduler(decoder)
    del decoder
    gc.collect()
    assert ref() is not None                             # the running worker holds it
    scheduler.close()
    assert not scheduler.thread.is_alive()
    gc.collect()
    assert ref() is None


def test_close_during_active_decode_drains_and_rejects_new_requests():
    import threading
    import time
    import pytest
    class Slow:
        def __init__(self):
            self.streams = {}
            self.entered, self.release = threading.Event(), threading.Event()
        def live(self):
            return len(self.streams)
        def admit(self, s):
            s.started = time.perf_counter()
            self.streams[id(s)] = s
        def round(self):
            self.entered.set()
            assert self.release.wait(2)
            done = []
            for s in self.streams.values():
                s.counted(1)
                s.take([7])
                if s.done:
                    done.append(s)
            return done
        def finish(self, done):
            for s in done:
                self.streams.pop(id(s))
        def drop(self):
            old = list(self.streams.values())
            self.streams.clear()
            return old
    decoder = Slow()
    scheduler = Scheduler(decoder, max_streams=4)
    out, stats = [], []
    caller = threading.Thread(target=lambda: stats.append(scheduler.submit([1], 2, None, True, out.extend)), daemon=True)
    caller.start()
    assert decoder.entered.wait(2)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (scheduler.close(), closed.set()), daemon=True)
    closer.start()
    assert scheduler._closing.wait(2)
    assert not closed.is_set()
    with pytest.raises(RuntimeError, match='closing'):
        scheduler.submit([1], 2, None, True, out.extend)
    decoder.release.set()
    caller.join(2)
    closer.join(2)
    assert closed.is_set() and not caller.is_alive()
    assert out == [7, 7] and stats[0]['rounds'] == 2
    scheduler.close()  # idempotent
