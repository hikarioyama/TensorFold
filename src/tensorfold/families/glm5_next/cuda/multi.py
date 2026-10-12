"""GLM family adapter for TensorFold's shared Scheduler/Stream/PrefixCache/MemoryGate.

Only rank 0 owns a scheduler. Other ranks mirror its model commands on the existing
GLM doorbell and integer collective channel. Weights and workspaces are never copied per request.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from tensorfold.cuda.capacity import available_bytes
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.streams import PrefixCache, Stream, accept, next_fill
from tensorfold.engine.exact_sampling import Sampling
from . import cache_fp8, decode as dec, engine as engine_cfg, forward as fw, graphs as graph_mod, segmented as seg
from .engine import _f64_ints, _ints_f64, decode_policy, encode_policy

ADMIT, ROUND, FINISH, DROP, STOP, SCORE = range(1, 7)


@dataclass
class Carry:
    hidden: torch.Tensor | None
    tokens: list[int]
    policy: object
    bytes: int = 0
    last_drafted: int = 0
    last_accepted: int = 0


def state_bytes(w, capacity):
    c = w.cfg
    layers = len(w.layers) + int(w.mtp is not None)
    owners = len(set(w.meta.get("index_source", ()))) + int(w.mtp is not None)
    dtype = w.meta.get("kv_dtype", "bf16")
    per = layers * (cache_fp8.row_bytes(c.kv_lora, dtype) + cache_fp8.row_bytes(c.qk_rope, dtype))
    if w.meta.get("long_context"):
        per += owners * (c.index_dim * 2 + c.index_heads * 4)
    return capacity * per + 8  # position scalars; full GLM has no recurrent state


def proposal_sampling(s):
    """Optional native-draft calibration only. Never mutate the target Sampling object."""
    scale = getattr(s.snap.policy, "temperature_scale", 1.)
    if s.sampling is None or scale == 1.:
        return s.sampling
    return replace(s.sampling, temperature=s.sampling.temperature * scale)


def pack_request(s, code, sid, cached):
    smp = s.sampling
    seed = (smp.seed if smp else 0) & ((1 << 64) - 1)
    return [ADMIT, sid, s.count, int(s.draft), int(s.stop_eos), int(s.background), cached,
            seed & 0x7fffffff, (seed >> 31) & 0x7fffffff, seed >> 62,
            *_f64_ints(smp.temperature if smp else 0), int(smp.top_k if smp else 0),
            *_f64_ints(smp.top_p if smp else 1), *_f64_ints(smp.min_p if smp else 0),
            int(s.constraint is not None), *code]


def unpack_request(header, prompt, constraint=None):
    _, sid, count, draft, stop, background, cached, lo, hi, top, t0, t1, k, p0, p1, m0, m1, shaped, *code = header
    temperature = _ints_f64(t0, t1)
    smp = Sampling((top << 62) | (hi << 31) | lo, temperature, k, _ints_f64(p0, p1), _ints_f64(m0, m1)) if temperature > 0 else None
    s = Stream(prompt, count, smp, draft=bool(draft), stop_eos=bool(stop), background=bool(background),
               constraint=constraint, sid=sid, options={"policy_code": code})
    return s, code, cached


class MultiDecoder:
    def __init__(self, engine, *, max_streams=4):
        if not engine.full or not 1 <= max_streams <= 4:
            raise ValueError("full GLM supports at most four streams")
        self.engine, self.w, self.e = engine, engine.w, engine.e
        self.max_streams = max_streams
        self.streams, self.filling = {}, []
        self.cache_limit = getattr(engine, "cache_bytes", 3 << 30)
        self.cache = PrefixCache(engine.cache_entries if self.cache_limit > 0 else 0)
        self.next_id = 1
        self.lock = threading.RLock()
        self.failed = False
        self.idle_states = [self.e.st]
        self.state_charges = {id(self.e.st): 0}
        self.b = fw.Buffers(self.w, 32, self.e.st.capacity)
        self.mb = fw.Buffers(self.w, 32, self.e.st.capacity) if self.w.mtp is not None else None
        # Prefill and packed main/MTP never overlap. Their workspace arenas share the
        # already warmed 128-row storage; each paired/side lane inside it remains distinct.
        self.b._ws = self.e.pbuf._ws
        if self.mb is not None:
            self.mb._ws = self.e.pbuf._ws
        self.ready_in = torch.empty(1, dtype=torch.int32, device=self.w.device)
        self.ready_out = torch.empty(engine.world, dtype=torch.int32, device=self.w.device)
        self.control = None
        if engine_cfg.CONTROL_FRAME:
            from tensorfold.cuda.comm import FixedIntBroadcast
            self.control = FixedIntBroadcast(engine.comm, self.w.device, rank=engine.rank)
        self.gate = MemoryGate(max(0, engine.capacity_plan["budget_bytes"] - torch.cuda.memory_allocated()),
                               256 << 20, torch_live(torch, available_bytes))
        self.frames = 0
        self.rows_seen = set()
        self.graphs = self.e.graphs

    @contextmanager
    def command(self):
        if self.failed:
            raise RuntimeError("GLM TP group failed; restart every rank")
        watch = self.engine._watchdog(time.perf_counter(), 0,
                    timeout_s=float(os.environ.get("TF_GLM_COMMAND_TIMEOUT_S", "120")))
        try:
            yield
        except NoRoom:
            self.engine._disarm_watchdog(watch)  # every rank agreed before model collectives
            raise
        except Exception:
            self.failed = True
            # Keep the fault monitor armed: a peer may be stuck inside a collective.
            # Neither caches nor the communicator may be reused after this failure.
            raise
        else:
            self.engine._disarm_watchdog(watch)

    def live(self):
        return len(self.streams) + len(self.filling)

    def _ready(self, value):
        self.ready_in.fill_(value)
        self.engine.comm.all_gather(self.ready_in, self.ready_out)
        return self.ready_out.cpu().tolist()

    def _control_share(self, payload, *, ended=False):
        control = getattr(self, "control", None)
        if control is None:
            return self.engine._share(payload)
        return control.broadcast(payload, limit=4 if ended else 64, allow_empty=ended)

    def _send(self, payload):
        self.engine._ring()
        self._control_share(payload)

    def _all(self):
        return list(self.filling) + list(self.streams.values())

    def _evict(self, among=None):
        if not self.cache.entries:
            return False
        before = sum(dec.snapshot_bytes(e[2]) for e in self.cache.entries)
        self.cache.evict(among)
        after = sum(dec.snapshot_bytes(e[2]) for e in self.cache.entries)
        self.gate.give(before - after)
        return True

    @torch.no_grad()
    def admit(self, s):
        with self.lock:
            if self.failed:
                raise RuntimeError("GLM TP decoder failed; restart all ranks before reuse")
            if not s.prompt or len(s.prompt) >= self.engine.limit:
                raise ValueError("nonempty prompt must fit the GLM context window")
            s.count = min(s.count, self.engine.limit - len(s.prompt))
            code = s.options.get("policy_code", encode_policy(self.engine.policy)) if s.draft else encode_policy("0")
            if code[0] >= 10:
                raise ValueError("full GLM uses MTP, not an external drafter")
            hit = self.cache.longest(s.prompt) if s.draft else None
            if hit and code[0] and hit[2].mtp_len < 0:
                hit = None
            cached = len(hit[0]) if hit else 0
            if self.next_id > 0x7fffffff:
                raise OverflowError("GLM int32 request IDs exhausted; restart the rank group")
            s.sid = self.next_id
            self.next_id += 1
            with self.command():
                self._send(pack_request(s, code, s.sid, cached))
                self.engine._share(list(s.prompt))
                if s.constraint is not None:
                    from tensorfold.engine.grammar import pack
                    self.engine._share(pack(s.constraint))
                self._admit_local(s, code, cached, hit)

    def _admit_local(self, s, code, cached, hit=None, preparation_error=None):
        # Plan and memory readiness precede all model collectives; no State is published early.
        if hit is None and cached:
            hit = self.cache.named(s.prompt, cached)
        ok = preparation_error is None and (not cached or hit is not None) and self.live() < self.max_streams
        if not all(self._ready(int(ok))):
            raise RuntimeError("GLM TP ranks disagree on admission/prefix/grammar plan")
        capacity = min(self.e.st.capacity, len(s.prompt) + s.count + 8)
        reusable = (self.e.st if self.e.st in self.idle_states else
                    next((st for st in self.idle_states if st.capacity >= capacity), None))
        if reusable is None:
            # Retire an undersized idle State at a round boundary, invalidating its
            # captured pointers first. This is ordinary contiguous-cache reservation,
            # not paging or an independent KV allocator.
            for st in list(self.idle_states):
                if st is self.e.st:
                    continue
                if self.graphs is not None:
                    self.graphs.release_packed()
                self.idle_states.remove(st)
                self.gate.give(self.state_charges.pop(id(st)))
                del st
        need = 0 if reusable is not None else state_bytes(self.w, capacity)
        # A selected prefix remains referenced even if evicted: don't pretend its bytes were freed.
        while not all(self._ready(int(self.gate.fits(need)))):
            eligible = [entry for entry in self.cache.entries if entry is not hit]
            if not all(self._ready(int(bool(eligible)))):
                self.gate.waits += 1
                raise NoRoom(f"GLM request needs {need} private cache bytes per rank")
            self._evict(eligible)
        error, view = None, None
        try:
            s.st = reusable if reusable is not None else fw.State(self.w, capacity, 8)
            if reusable is not None:
                s.st.reset()
            policy = decode_policy(code)
            if isinstance(policy, tuple):
                greedy = s.sampling is None or s.sampling.temperature <= 0
                policy = dec.DepthPolicy(3, fixed=greedy, confidence=.35 if greedy else 0, low=.6, high=.85)
            if policy is not None and self.w.mtp is None:
                raise ValueError("MTP requested without MTP weights")
            s.snap = Carry(None, [], policy, self.state_charges.get(id(s.st), need))
            if hit is not None:
                view = SimpleNamespace(st=s.st)
                dec.load_rows(view, hit[2])
                dec.restore(view, hit[2])
                s.cached = cached

        except Exception as exc:
            error = exc
        if not all(self._ready(int(error is None))):
            s.st = s.snap = view = None
            if error is not None:
                error.__traceback__ = None
            torch.cuda.empty_cache()
            raise NoRoom("one or more GLM ranks could not prepare the private State") from None
        if reusable is not None:
            self.idle_states.remove(reusable)
        else:
            self.state_charges[id(s.st)] = need
        if hit is not None and s.snap.policy is not None:
            dec._absorb_rows(SimpleNamespace(w=self.w, st=s.st, pbuf=self.e.pbuf), hit[2].pending,
                             s.prompt[cached:cached + 1])
        self.gate.take(need)
        s.context = list(s.prompt)
        self.filling.append(s)

    def _snapshot(self, s, pending):
        if self.cache.keep <= 0 or not s.draft or s.st.pos < 1:
            return
        view = SimpleNamespace(st=s.st)
        snap = dec.take_snapshot(view, s.prompt[:s.st.pos], pending,
                                 mtp=s.snap.policy is not None)
        if s.snap.policy is not None:
            snap.mtp_len = s.st.pos - 1
        need = dec.row_bytes(view, snap) + dec.snapshot_bytes(snap)
        if need > self.cache_limit:
            return
        while sum(dec.snapshot_bytes(e[2]) for e in self.cache.entries) + need > self.cache_limit:
            if not self._evict():
                return
        if not all(self._ready(int(self.gate.fits(need)))):
            return
        error = None
        try:
            dec.save_rows(view, snap)
        except torch.OutOfMemoryError as exc:
            error = exc
        if not all(self._ready(int(error is None))):
            return
        before = sum(dec.snapshot_bytes(e[2]) for e in self.cache.entries)
        self.cache.add(list(snap.ids), None, snap)
        after = sum(dec.snapshot_bytes(e[2]) for e in self.cache.entries)
        if after >= before:
            self.gate.take(after - before)
        else:
            self.gate.give(before - after)

    def _fill(self):
        s = next_fill(self.filling)
        t0 = time.perf_counter()
        st, b, c = s.st, self.e.pbuf, self.w.cfg
        start, n = st.pos, len(s.prompt)
        stop = min(n, start + min(b.rows, 128))
        if self.cache.keep > 0 and start < n - 1 < stop and s.draft:
            stop = n - 1
        tokens = s.prompt[start:stop]
        rows = fw.stage(self.w, st, b, tokens)
        logits = fw.compute(self.w, st, b, rows, nch=fw.chunks_for(st, rows), host_pos=st.pos).clone()
        hidden = b.fnormed[rows - 1:rows].clone()
        if s.snap.policy is not None:
            nxt = s.prompt[start + 1:stop + 1]
            if nxt:
                dec._absorb_rows(SimpleNamespace(w=self.w, st=st, pbuf=b), b.fnormed[:len(nxt)], nxt)
        fw.commit(self.w, st, b, rows, rows)
        if stop == n - 1:
            self._snapshot(s, hidden)
        s.prefill_s += time.perf_counter() - t0
        if stop != n:
            return
        if s.constraint is not None:
            win = s.constraint.window([0], [-1])
            s.constraint.mask(logits, win, self.w.vocab_offset)
        first = dec.sample_rows(self.w, logits, [n], s.sampling)[0]
        if s.constraint is not None:
            s.constraint.advance([first])
        s.snap.hidden, s.snap.tokens = hidden, [first]
        s.started = time.perf_counter()
        self.filling.remove(s)
        self.streams[s.sid] = s
        s.take([first], self._ends(s))

    def _ends(self, s):
        return self.engine.eos if s.stop_eos else ()

    def _mtp(self, windows, *, chain=False):
        if len(windows) == 1 and windows[0][0] is self.e.st:
            _, tokens, hidden = windows[0]
            logits = self.e.mtp(tokens, hidden)
            return logits, self.e.mbuf.fnormed[:1].clone()
        segs = seg.mtp_stage_streams(self.w, self.mb, windows, padded=True, chain=chain)
        logits = (self.graphs.packed_forward(self.w, segs, self.mb, mtp=True) if self.graphs is not None
                  else seg.mtp_compute_streams(self.w, segs, self.mb))
        return logits, self.mb.fnormed[:len(windows)].clone()

    def _propose(self, live):
        active = [s for s in live if s.snap.policy is not None]
        for s in live:
            s.drafts = []
        if not active:
            return
        targets = {}
        for s in active:
            targets[s.sid] = max(0, min(s.snap.policy.next(s.snap.last_drafted, s.snap.last_accepted), s.count - len(s.out) - 1))
            s.st.set_mtp_len(s.st.mtp_len - s.st.mtp_drafted)
            s.st.mtp_drafted = 0
        logits, hidden = self._mtp([(s.st, s.snap.tokens, s.snap.hidden) for s in active])
        for s in active:
            s.st.set_mtp_len(s.st.mtp_len + len(s.snap.tokens))
        depth, confidence = 0, {s.sid: 1. for s in active}
        while active:
            batched = None
            if dec.BATCH_SAMPLING:
                eligible = [i for i, s in enumerate(active) if depth < targets[s.sid]]
                batched = {}
                if eligible:
                    rows = logits if len(eligible) == len(active) else logits[eligible]
                    selected = [active[i] for i in eligible]
                    prob_lists = [[] if s.snap.policy.confidence > 0 else None for s in selected]
                    stage = (self.graphs.sampling_stage if graph_mod.SAMPLING_GRAPH and self.graphs is not None else None)
                    tokens = dec.sample_many(self.w, rows, [[s.st.pos + 1 + depth] for s in selected],
                                             [proposal_sampling(s) for s in selected], prob_lists=prob_lists, stage=stage)
                    batched = {s.sid: (t[0], p[0] if p is not None else 1.)
                               for s, t, p in zip(selected, tokens, prob_lists)}
            going, inputs = [], []
            for i, s in enumerate(active):
                if depth >= targets[s.sid]:
                    continue
                probs = []
                threshold = s.snap.policy.confidence
                if batched is not None:
                    token, prob = batched[s.sid]
                else:
                    token = dec.sample_rows(self.w, logits[i:i + 1], [s.st.pos + 1 + depth], proposal_sampling(s),
                                            probs=probs if threshold > 0 else None)[0]
                    prob = probs[0] if probs else 1.
                if threshold > 0 and depth and confidence[s.sid] * prob < threshold:
                    continue
                s.drafts.append(token)
                confidence[s.sid] *= prob
                if depth + 1 < targets[s.sid] and confidence[s.sid] >= threshold:
                    going.append(s)
                    inputs.append((s.st, [token], hidden[i:i + 1]))
            if not going:
                break
            logits, hidden = self._mtp(inputs, chain=True)
            for s in going:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active, depth = going, depth + 1

    @torch.no_grad()
    def round(self):
        with self.lock:
            if not self.live():
                return []
            with self.command():
                self._send([ROUND])
                return self._round_local()

    def _round_local(self):
        if self.filling:
            self._fill()
        # Cancellation after the first token is a leader decision, shared before selecting a collective layout.
        ended = self._control_share([s.sid for s in self._all() if s.done] if self.engine.rank == 0 else None, ended=True)
        for s in self._all():
            s.done = s.sid in ended
        live = [s for s in self.streams.values() if not s.done]
        if live:
            self._propose(live)
            windows, masks = [], []
            for s in live:
                tokens = [s.out[-1]] + s.drafts
                mask = s.constraint.window(tokens, list(range(-1, len(tokens) - 1))) if s.constraint else None
                windows.append((s.st, mask.tokens if mask else tokens))
                masks.append(mask)
            if len(windows) == 1 and windows[0][0] is self.e.st:
                logits = self.e.forward(windows[0][1])
                segs = [(self.e.st, 0, len(windows[0][1]))]
                main_buf = self.e.buf
            else:
                segs = seg.stage_streams(self.w, self.b, windows, padded=engine_cfg.MAIN_PAD)
                logits = (self.graphs.packed_forward(self.w, segs, self.b) if self.graphs is not None
                          else seg.compute_streams(self.w, segs, self.b))
                main_buf = self.b
            self.rows_seen.add(segs[-1][2])
            self.frames += 1
            paths, news = [], []
            sampled_batches = None
            logical = seg.logical_segments(windows, segs)
            if dec.BATCH_SAMPLING:
                positions = []
                for s, (_, lo, z), mask in zip(live, logical, masks):
                    if mask is not None:
                        s.constraint.mask(logits[lo:z], mask, self.w.vocab_offset)
                    positions.append(list(range(s.st.pos + 1, s.st.pos + 1 + z - lo)))
                stage = (self.graphs.sampling_stage if graph_mod.SAMPLING_GRAPH and self.graphs is not None else None)
                sampled_batches = dec.sample_many(self.w, seg.sampling_logits(logits, segs, logical), positions,
                                                   [s.sampling for s in live], stage=stage)
            for s, (_, tokens), (_, lo, z), mask in zip(live, windows, logical, masks):
                rows = logits[lo:z]
                if mask is not None and sampled_batches is None:
                    s.constraint.mask(rows, mask, self.w.vocab_offset)
                sampled = (sampled_batches[len(paths)] if sampled_batches is not None else
                           dec.sample_rows(self.w, rows, list(range(s.st.pos + 1, s.st.pos + 1 + z - lo)), s.sampling))
                path, terminal = accept(tokens, list(range(-1, len(tokens) - 1)), sampled,
                                        s.count - len(s.out), self._ends(s))
                paths.append(path)
                news.append([tokens[r] for r in path[1:]] + [terminal])
            seg.commit_streams(self.w, segs, self.b, paths)
            for s, (_, tokens), (_, lo, _), path, new in zip(live, windows, segs, paths, news):
                s.committed.extend(tokens[r] for r in path)
                s.counted(len(tokens))
                s.snap.hidden = main_buf.fnormed[lo:lo + len(path)].clone()
                s.snap.tokens = list(new)
                s.snap.last_drafted, s.snap.last_accepted = len(tokens) - 1, len(path) - 1
                if s.constraint is not None:
                    s.constraint.advance(new)
                s.take(new, self._ends(s))
        ended = self._control_share([s.sid for s in self._all() if s.done] if self.engine.rank == 0 else None, ended=True)
        done = [s for s in self._all() if s.sid in ended]
        for s in done:
            s.done = True
        return done

    def _finish_local(self, ids):
        for s in self._all():
            if s.sid in ids:
                self.idle_states.append(s.st)  # keep stable graph pointers for the next private request
                self.streams.pop(s.sid, None)
                self.filling = [x for x in self.filling if x is not s]
                s.st = s.snap = None

    def finish(self, done):
        if done:
            with self.lock, self.command():
                ids = [s.sid for s in done]
                self._send([FINISH, *ids])
                self._finish_local(ids)

    def drop(self):
        # Never continue an NCCL group after an unknown model failure.
        self.failed = True
        old = self._all()
        self._quarantine = old
        self.streams, self.filling = {}, []
        return old  # retain State/scratch until the whole rank group is restarted

    def close(self):
        with self.lock, self.command():
            self._send([STOP])

    def follow(self):
        while True:
            self.engine._await_bell()
            try:
                with self.command():
                    header = self._control_share(None)
                    op = header[0]
                    if op == STOP:
                        return
                    if op == ADMIT:
                        prompt = self.engine._share(None)
                        packed = self.engine._share(None) if header[17] else []
                        constraint, error = None, None
                        try:
                            if packed:
                                from tensorfold.engine import grammar
                                constraint = grammar.compiler(self.engine, self.engine.model_dir, self.engine.eos).follow(packed)
                        except Exception as exc:
                            error = exc
                        stream, code, cached = unpack_request(header, prompt, constraint)
                        self._admit_local(stream, code, cached, preparation_error=error)
                    elif op == ROUND:
                        self._round_local()
                    elif op == FINISH:
                        self._finish_local(header[1:])
                    elif op == SCORE:
                        prompt, labels = self.engine._share(None), self.engine._share(None)
                        self.engine._score_local(prompt, labels)
                    else:
                        raise RuntimeError(f"unknown GLM model command {op}")
            except NoRoom:
                continue  # all ranks agreed on rollback before publishing any State
