"""Opt-in decode telemetry and steady-state CUPTI traces, without per-tag device syncs.

TF_GLM_DECODE_TRACE_DIR writes per-request/rank JSON. TF_GLM_DECODE_CUPTI=1
also traces 16 main steps after 4 skipped + 2 warmup steps. Profiling is NOT
an uninstrumented benchmark; export and profiler setup are outside decode seconds.
TF_GLM_DECODE_TRACE_LIMIT bounds traced requests per engine (0 means unlimited).
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

TRACE_DIR = os.environ.get("TF_GLM_DECODE_TRACE_DIR", "")
CUPTI = os.environ.get("TF_GLM_DECODE_CUPTI", "0")
if CUPTI not in ("0", "1"):
    raise ValueError("TF_GLM_DECODE_CUPTI must be 0 or 1")
CUPTI = CUPTI == "1"
TRACE_LIMIT = int(os.environ.get("TF_GLM_DECODE_TRACE_LIMIT", "0"))
if TRACE_LIMIT < 0:
    raise ValueError("TF_GLM_DECODE_TRACE_LIMIT must be nonnegative")


class Observer:
    def __init__(self, profiler=None):
        self.profiler = profiler
        self.paths = Counter()
        self.host_seconds = Counter()
        self.main_steps = 0

    @contextmanager
    def step(self, model, rows, kind, hit, bucket, parity):
        if model == "main":
            self.main_steps += 1
            if self.profiler is not None:
                self.profiler.step()
        key = f"{model}/R{rows}/{kind}/{'hit' if hit else 'eager'}/bucket={bucket}/parity={parity}"
        self.paths[key] += 1
        marker = nullcontext()
        if self.profiler is not None:
            import torch
            marker = torch.profiler.record_function("tf.decode/" + key)
        start = time.perf_counter()
        try:
            with marker:
                yield
        finally:
            # This is CPU submission wall time, NOT GPU/kernel execution time.
            self.host_seconds[key] += time.perf_counter() - start


def step(engine, model, rows, kind, hit, bucket=None, parity=None):
    observer = getattr(engine, "_decode_observer", None)
    return nullcontext() if observer is None else observer.step(model, rows, kind, hit, bucket, parity)


def traced(fn):
    if not TRACE_DIR:
        return fn

    @wraps(fn)
    def run(engine, *args, **kwargs):
        used = getattr(engine, "_decode_trace_requests", 0)
        if TRACE_LIMIT and used >= TRACE_LIMIT:
            return fn(engine, *args, **kwargs)
        engine._decode_trace_requests = used + 1
        root = Path(TRACE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        rank = getattr(engine.w, "rank", 0)
        stem = f"decode-rank{rank}-pid{os.getpid()}-{time.time_ns()}"
        observer = Observer()
        previous = getattr(engine, "_decode_observer", None)
        profiler = nullcontext()
        ready = []
        if CUPTI:
            import torch
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=4, warmup=2, active=16, repeat=1),
                on_trace_ready=ready.append,
            )
            observer.profiler = profiler
        result = None
        start_pos = engine.st.pos
        try:
            engine._decode_observer = observer
            with profiler:
                result = fn(engine, *args, **kwargs)
            # Serialize only after decode: writing a large trace must not create a
            # fake round-boundary stall at the scheduled active-window boundary.
            for captured in ready:
                captured.export_chrome_trace(str(root / f"{stem}.trace.json"))
            return result
        finally:
            # on_trace_ready holds ready.append; release its profiler references
            # explicitly rather than waiting for cyclic GC after a large trace.
            ready.clear()
            observer.profiler = None
            engine._decode_observer = previous
            record = {
                "rank": rank, "pid": os.getpid(), "decode": fn.__name__, "start_pos": start_pos,
                "cupti": CUPTI, "paths": dict(observer.paths), "main_steps": observer.main_steps,
                "host_submission_seconds": dict(observer.host_seconds),
                "timing_contract": "host submission, not GPU duration; CUPTI run is instrumented",
                "completed": result is not None,
            }
            if result is not None:
                record.update(seconds=result.seconds, rounds=result.rounds, tokens=len(result.tokens),
                              drafted=result.drafted, accepted=result.accepted)
            (root / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return run
