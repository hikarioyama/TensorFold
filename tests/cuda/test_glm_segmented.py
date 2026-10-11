"""GPU integration of segmented GLM MLA/DSA/MoE/MTP using a small full-architecture fixture."""
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tiny_weights():
    from tensorfold.families.glm5_next.cuda.weights import Config, Weights, DSAW, IndexW, LayerW, MLPW, MoEW, MTPW
    from tensorfold.families.glm5_next.cuda.qmm import make_b16
    from tensorfold.families.glm5_next.cuda.latent import AbsorbW
    from tensorfold.families.glm5_next.cuda.weights53 import inv_freq
    from tensorfold.cuda.exl3 import experts
    c = Config(hidden=256, layers=2, vocab=256, eps=1e-5, heads=4, q_lora=128, kv_lora=512,
               qk_dim=256, v_dim=256, lin_heads=4, lin_dim=128, conv=4, lower=-5,
               experts=8, top_k=2, moe_width=128, shared_width=128, dense_width=128,
               routed_scale=2.5, norm_topk=True, streams=1, hc_iters=20, hc_eps=1e-6,
               index_heads=32, index_dim=128, index_topk=2048, kpool=1, limit=10.,
               kinds=["dsa"] * 2, mlp_kinds=["dense", "moe"], eos=(255,), mtp_layers=1,
               group_size=0, bits=0, quant="exl3", qk_nope=192, qk_rope=64, full=True)
    gen = torch.Generator(device="cuda").manual_seed(513)
    def rand(*shape):
        return (torch.randn(shape, generator=gen, device="cuda") * .03).bfloat16()
    def norm(n):
        return (1 + rand(n)).bfloat16()
    def lin(n, k):
        return make_b16(rand(n, k))
    def attention(owner):
        ix = IndexW(lin(128, 256), lin(4096, 128), norm(128).half(), rand(128).half(), wproj=lin(32, 256)) if owner else None
        wk, wv = rand(4 * 192, 512), rand(4 * 256, 512)
        return DSAW(lin(128 + 640, 256), norm(128), norm(512), lin(4 * 256, 128),
                    make_b16(wk), make_b16(wv), lin(256, 4 * 256), 4, ix,
                    AbsorbW.from_rows(wk.float(), wv.float(), 4))
    def mlp():
        return MLPW(lin(256, 256), lin(256, 128), 128)
    def triples(k, n):
        return [(torch.randint(-32768, 32768, (k // 16, n // 16, 48), dtype=torch.int16, device="cuda", generator=gen),
                 rand(k).half(), rand(n).half()) for _ in range(8)]
    ex = experts.prepare(triples(256, 128), triples(256, 128), triples(128, 256), "mul1", device="cuda")
    moe = MoEW(rand(8, 256).half(), rand(8).float(), ex, mlp())
    layers = [LayerW(0, "dsa", None, None, norm(256), norm(256), dsa=attention(True), mlp=mlp()),
              LayerW(1, "dsa", None, None, norm(256), norm(256), dsa=attention(False), moe=moe)]
    m = LayerW(2, "dsa", None, None, norm(256), norm(256), dsa=attention(True), moe=moe)
    mtp = MTPW(norm(256), norm(256), lin(256, 512), norm(256), m)
    w = Weights(c, rand(256, 256), layers, norm(256), lin(256, 256), mtp, 0, 1, torch.device("cuda"),
                meta={"norm_sums": False, "long_context": True, "index_source": [0, 0], "index_full": (0,)},
                inv_freq=inv_freq(64, 10000, "cuda"), index_inv_freq=inv_freq(64, 10000, "cuda"))
    return w


@pytest.fixture(scope="module")
def weights():
    return tiny_weights()


def states(w, positions):
    from tensorfold.families.glm5_next.cuda.forward import State
    out = []
    for i, pos in enumerate(positions):
        st = State(w, max(2112, pos + 32), 8)
        gen = torch.Generator(device="cuda").manual_seed(701 + i)
        for cache in st.kc + st.kpe + [st.mtp_kc, st.mtp_kpe] + [p[0] for p in st.index.values()]:
            cache[:pos].copy_(torch.randn(cache[:pos].shape, generator=gen, device="cuda").to(cache.dtype))
        st.set_pos(pos)
        st.set_mtp_len(pos)
        out.append(st)
    return out


@pytest.mark.parametrize("widths", [(1,), (3, 3), (3, 3, 3), (3, 3, 3, 3), (8, 8, 8, 8), (2, 1, 4, 3)])
@pytest.mark.parametrize("sparse", [False, True])
def test_main_solo_packed_bit_exact_and_private(weights, widths, sparse):
    from tensorfold.families.glm5_next.cuda import forward as fw, segmented as seg
    w = weights
    pos = [0, 31, 65, 127] if not sparse else [2047, 2048, 2053, 65]
    sts = states(w, pos[:len(widths)])
    b = fw.Buffers(w, 32, 2112)
    wins = [(st, [12 + i + j for j in range(n)]) for i, (st, n) in enumerate(zip(sts, widths))]
    expected, cached = [], []
    for st, tokens in wins:
        expected.append(fw.forward(w, st, b, tokens).clone())
        cached.append([t[:st.pos + len(tokens)].clone() for t in st.kc + st.kpe])
    segs = seg.stage_streams(w, b, wins)
    got = seg.compute_streams(w, segs, b)
    assert torch.equal(got, torch.cat(expected)), (widths, sparse, (got - torch.cat(expected)).abs().max())
    for (st, tokens), ref in zip(wins, cached):
        assert all(torch.equal(t[:st.pos + len(tokens)], r) for t, r in zip(st.kc + st.kpe, ref))
    old = [s.pos for s in sts]
    paths = [list(range(max(1, n - 1))) for n in widths]
    seg.commit_streams(w, segs, b, paths)
    assert [s.pos for s in sts] == [p + len(path) for p, path in zip(old, paths)]


@pytest.mark.parametrize("positions", [(0, 7, 31, 65), (2047, 2048, 2053, 65)])
def test_mtp_zero_position_and_each_stream_tail(weights, positions):
    from tensorfold.families.glm5_next.cuda import forward as fw, mtp, segmented as seg
    w = weights
    sts = states(w, positions)
    b = fw.Buffers(w, 32, 2112)
    widths = [3, 1, 2, 3]
    wins = [(st, [11 + i] * n, torch.full((n, w.cfg.hidden), .1 * (i + 1), device="cuda", dtype=torch.bfloat16))
            for i, (st, n) in enumerate(zip(sts, widths))]
    logits, hidden = [], []
    for st, tokens, h in wins:
        logits.append(mtp.mtp_forward(w, st, b, tokens, h).clone())
        hidden.append(b.fnormed[:1].clone())
    ss = seg.mtp_stage_streams(w, b, wins)
    got = seg.mtp_compute_streams(w, ss, b)
    assert torch.equal(got, torch.cat(logits))
    assert torch.equal(b.fnormed[:4], torch.cat(hidden))


def test_r9_r12_graph_replay_changes_device_positions(weights):
    from tensorfold.families.glm5_next.cuda import forward as fw, segmented as seg
    from tensorfold.families.glm5_next.cuda.graphs import Graphs
    w = weights
    b = fw.Buffers(w, 32, 2112)
    g = object.__new__(Graphs)
    g.pool = torch.cuda.graph_pool_handle()
    for count in (3, 4):
        sts = states(w, [31, 65, 127, 511][:count])
        wins = [(s, [11, 12, 13]) for s in sts]
        for step in range(2):
            ss = seg.stage_streams(w, b, wins)
            want = seg.compute_streams(w, ss, b).clone()
            got = g.packed_forward(w, ss, b).clone()
            assert torch.equal(got, want)
            for st in sts:
                st.set_pos(st.pos + 3)
        g.release_packed()
    assert g.packed_hits == 2


def test_shared_scheduler_five_requests_prefix_and_cancel(weights):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.glm5_next.cuda.multi import MultiDecoder
    class Comm:
        world = 1
        def all_gather(self, send, recv):
            recv.copy_(send.reshape(-1))
    w = weights
    w.comm = Comm()
    engine = object.__new__(GlmEngine)
    engine.w, engine.rank, engine.world, engine.comm, engine.torch = w, 0, 1, w.comm, torch
    engine.full, engine.limit, engine.parallel, engine.concurrent = True, 2112, 4, True
    engine.policy, engine.serial_only, engine.drafter = "2", False, None
    engine.eos, engine.request, engine.cache_entries = w.cfg.eos, threading.local(), 4
    engine.capacity_plan = {"budget_bytes": int(torch.cuda.mem_get_info()[1] * .85)}
    engine.e = Engine(w, capacity=2120, max_rows=8, prefill_rows=128, graphs=True, long_context=True)
    engine.multi = MultiDecoder(engine)
    engine.scheduler = Scheduler(engine.multi, max_streams=4)
    prompts = [[10 + i] * (33 + i) for i in range(5)]
    expected = {}
    try:
        for i, prompt in enumerate(prompts):
            out = []
            engine.generate(prompt, 16, None, lambda t: out.extend(t), draft=False)
            expected[i] = out
        barrier = threading.Barrier(5)
        peak = [0]
        original = engine.multi._admit_local
        def tracked(*a, **kw):
            result = original(*a, **kw)
            peak[0] = max(peak[0], engine.multi.live())
            return result
        engine.multi._admit_local = tracked
        def run(i):
            out = []
            barrier.wait()
            stats = engine.generate(prompts[i], 16, None, lambda t: out.extend(t))
            return out, stats
        with ThreadPoolExecutor(5) as pool:
            jobs = [pool.submit(run, i) for i in range(5)]
            results = [f.result(timeout=float(os.environ.get("TF_TEST_TIMEOUT_S", "90"))) for f in jobs]
        assert 2 <= peak[0] <= 4
        assert [r[0] for r in results] == [expected[i] for i in range(5)]
        assert not engine.multi.live()
        out = []
        stats = engine.generate(prompts[0], 16, None, lambda t: out.extend(t))
        assert out == expected[0] and stats["cached"] > 0
        stopped = []
        engine.generate(prompts[0], 64, None, lambda t: (stopped.extend(t), True)[1])
        assert len(stopped) < 64
        out = []
        engine.generate(prompts[0], 16, None, lambda t: out.extend(t))
        assert out == expected[0]
    finally:
        engine.close()
        w.comm = None


def test_padded_mtp_reuses_graph_and_never_writes_invalid_cache_rows(weights):
    from tensorfold.families.glm5_next.cuda import forward as fw, mtp, segmented as seg
    from tensorfold.families.glm5_next.cuda.graphs import Graphs
    w = weights
    sts = states(w, [31, 65, 2053, 39])
    b, solo = fw.Buffers(w, 32, 4096), fw.Buffers(w, 8, 4096)
    graph = object.__new__(Graphs)
    graph.pool = torch.cuda.graph_pool_handle()
    for turn, widths in enumerate(([1, 3, 2, 1], [3, 1, 1, 2], [2, 2, 3, 3])):
        wins = [(st, [11 + i] * n, torch.full((n, w.cfg.hidden), .1 * (i + 1), device="cuda", dtype=torch.bfloat16))
                for i, (st, n) in enumerate(zip(sts, widths))]
        expected, unchanged = [], []
        for st, tokens, hidden in wins:
            end = st.mtp_len + len(tokens)
            caches = [st.mtp_kc, st.mtp_kpe, st.index[w.cfg.layers][0]]
            for cache in caches:
                cache[end:st.mtp_len + 3].fill_(float('nan'))
            expected.append(mtp.mtp_forward(w, st, solo, tokens, hidden).clone())
            unchanged.append([cache[end:st.mtp_len + 3].clone() for cache in caches])
        ss = seg.mtp_stage_streams(w, b, wins, padded=True)
        got = graph.packed_forward(w, ss, b, mtp=True).clone()
        assert torch.equal(got, torch.cat(expected))
        assert b.valid[:12].tolist() == [int(j < n) for n in widths for j in range(3)]
        for (st, tokens, _), saved in zip(wins, unchanged):
            caches = [st.mtp_kc, st.mtp_kpe, st.index[w.cfg.layers][0]]
            end = st.mtp_len + len(tokens)
            assert all(torch.equal(cache[end:st.mtp_len + 3].view(torch.int16), ref.view(torch.int16))
                       for cache, ref in zip(caches, saved))
            st.set_mtp_len(end)
        invalid = b.valid[:12] == 0
        assert torch.all(b.pick[:12][invalid] == w.cfg.experts)
        assert torch.all(b.wts[:12][invalid] == 0)
    assert graph.packed_misses == 1 and graph.packed_hits == 2


@pytest.mark.parametrize('rows', [1, 3, 6, 9, 12, 32])
@pytest.mark.parametrize('zero', [False, True])
def test_fused_down_keeps_slot_order_and_fp32_bits(weights, monkeypatch, rows, zero):
    from tensorfold.families.glm5_next.cuda import forward as fw
    w = weights
    b = fw.Buffers(w, rows, 2112)
    b.normed.copy_(torch.zeros_like(b.normed) if zero else torch.randn_like(b.normed))
    monkeypatch.setattr(fw, 'DOWN_COMBINE', False)
    want = fw.moe_block(w.layers[1], w, b, rows).clone()
    monkeypatch.setattr(fw, 'DOWN_COMBINE', True)
    got = fw.moe_block(w.layers[1], w, b, rows).clone()
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_r7_r8_solo_graphs_preserve_main_and_mtp_bits(weights):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    e = Engine(weights, capacity=2112, max_rows=8, prefill_rows=128, graphs=True,
               graph_rows=(7, 8), long_context=True)
    prefill(e, [11, 12, 13] * 12, None, mtp=True)
    graph = e.graphs
    for rows in (7, 8):
        tokens = [20 + i for i in range(rows)]
        e.graphs = None
        want = e.forward(tokens).clone()
        hidden = e.buf.fnormed[:rows].clone()
        want_mtp = e.mtp(tokens, hidden).clone()
        e.graphs = graph
        assert torch.equal(e.forward(tokens), want)
        assert torch.equal(e.mtp(tokens, hidden), want_mtp)
    assert e.replays['main'] == 2 and e.replays['mtp'] == 2
