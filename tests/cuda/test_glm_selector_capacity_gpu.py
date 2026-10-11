"""Capacity-mismatched DSA fixtures for direct execution and compute-sanitizer."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import decode as dec, forward as fw, mtp, sparse53
from test_glm_segmented import tiny_weights, states


@pytest.mark.parametrize("pos", [2047, 2048, 2049])
@pytest.mark.parametrize("rows", [1, 2, 3, 128])
@pytest.mark.parametrize("bucket", [None, 4096])
def test_shared_scores_private_keys_and_canaries(pos, rows, bucket):
    capacity = 2112 if rows < 128 else 2560
    gen = torch.Generator(device="cuda").manual_seed(451)
    qi = torch.randn(rows, 32, 128, generator=gen, device="cuda", dtype=torch.bfloat16)
    wts = torch.randn(rows, 32, generator=gen, device="cuda", dtype=torch.float32)
    keys = torch.randn(capacity, 128, generator=gen, device="cuda", dtype=torch.float16)
    position = torch.tensor([pos], device="cuda", dtype=torch.int64)
    tight = torch.empty(rows, capacity, device="cuda")
    want, wc = sparse53.select_tokens(qi, wts, keys, position, rows, 2048, capacity, scores=tight)
    # Deliberately noncompact workspace with an offset and multiple slabs.
    storage = torch.full((min(rows, 7), 6152 + 13), 123456., device="cuda")
    shared = storage[:, 5:6157]
    got, gc = sparse53.select_tokens(qi, wts, keys, position, rows, 2048, 6152,
                                     bucket=bucket, scores=shared)
    assert torch.equal(got, want) and torch.equal(gc, wc)
    assert gc.tolist() == [2048 if pos + r + 1 > 2048 else 0 for r in range(rows)]
    assert torch.all(storage[:, :5] == 123456.)
    assert torch.all(storage[:, 5 + capacity:] == 123456.)
    for r in range(rows):
        # Independent stable ordering of the actual FP32 score bits.
        visible = pos + r + 1
        order = torch.argsort(tight[r, :visible], descending=True, stable=True)[:2048]
        assert torch.equal(got[r].long(), order.sort().values)


def test_ties_and_graph_replay_with_shared_stride():
    qi = torch.zeros(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    wts = torch.ones(3, 32, device="cuda")
    keys = torch.zeros(2112, 128, device="cuda", dtype=torch.float16)
    pos = torch.tensor([2047], device="cuda", dtype=torch.int64)
    scores = torch.full((3, 6144), 999., device="cuda")
    tokens = torch.full((3, 2048), 999999, device="cuda", dtype=torch.int32)
    counts = torch.empty(3, device="cuda", dtype=torch.int32)
    def run():
        sparse53.select_tokens(qi, wts, keys, pos, 3, 2048, 6144, bucket=4096,
                               scores=scores, tokens=tokens, counts=counts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for position in (2047, 2048, 2049):
        pos.fill_(position)
        graph.replay()
        assert torch.equal(tokens, torch.arange(2048, device="cuda").expand(3, -1))
        assert counts.tolist() == [2048 if position + r + 1 > 2048 else 0 for r in range(3)]
        assert torch.all(scores[:, 2112:] == 999.)


@pytest.fixture(scope="module")
def weights():
    return tiny_weights()


@pytest.mark.parametrize("pos,rows", [(2047, 1), (2047, 2), (2048, 3), (2049, 128)])
def test_main_mtp_absorb_shared_prefill_buffer(weights, pos, rows):
    w = weights
    st = states(w, [pos])[0]
    if st.capacity < pos + rows:
        st = fw.State(w, 2560, 8)
        original = states(w, [pos])[0]
        for dst, src in zip(dec._row_views(st, pos, pos), dec._row_views(original, pos, pos)):
            dst.copy_(src)
        st.set_pos(pos)
        st.set_mtp_len(pos)
    tight = fw.Buffers(w, rows, st.capacity, prefill=True)
    shared = fw.Buffers(w, rows, 6144, prefill=True)
    tokens = [12 + j % 20 for j in range(rows)]
    want = fw.forward(w, st, tight, tokens).clone()
    hidden = tight.fnormed[:rows].clone()
    got = fw.forward(w, st, shared, tokens).clone()
    assert torch.equal(want, got)
    assert torch.equal(hidden, shared.fnormed[:rows])
    want = mtp.mtp_forward(w, st, tight, tokens, hidden).clone()
    cache = [t[pos:pos + rows].clone() for t in (st.mtp_kc, st.mtp_kpe, st.index[w.cfg.layers][0])]
    dec._absorb_rows(SimpleNamespace(w=w, st=st, pbuf=shared), hidden, tokens)
    assert torch.equal(want, shared.logits[:1])
    assert st.mtp_len == pos + rows
    for t, ref in zip((st.mtp_kc, st.mtp_kpe, st.index[w.cfg.layers][0]), cache):
        assert torch.equal(t[pos:pos + rows], ref)


def test_private_prefix_resume_through_shared_prefill(weights):
    """Saved main/MTP keys copied into a smaller State, then actual prefill resume/absorb."""
    w = weights
    source = states(w, [2049])[0]
    source.set_mtp_len(2048)
    pending = torch.full((1, w.cfg.hidden), .125, dtype=torch.bfloat16, device="cuda")
    snap = dec.take_snapshot(SimpleNamespace(st=source), [12] * 2049, pending, mtp=True)
    dec.save_rows(SimpleNamespace(st=source), snap)
    results = []
    for buffer_capacity in (2560, 6152):
        e = dec.Engine(w, capacity=2560, max_rows=8, prefill_rows=128, long_context=True)
        e.pbuf = fw.Buffers(w, 128, buffer_capacity, prefill=True)
        dec.load_rows(e, snap)
        token = dec.prefill(e, snap.ids + [13, 14, 15], None, mtp=True, resume=snap)
        results.append((token, e.last_hidden.clone(),
                        [t.clone() for t in dec._row_views(e.st, e.st.pos, e.st.mtp_len)]))
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])
    assert all(torch.equal(a, b) for a, b in zip(results[0][2], results[1][2]))
