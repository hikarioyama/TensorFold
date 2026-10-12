"""FP8 storage oracle, dequantized-BF16 attention oracle and private State checks."""
import copy
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def unpack(cache, width):
    codes = cache[:, :width].contiguous().view(torch.float8_e4m3fn).float()
    scales = cache[:, width:].contiguous().view(torch.float32)
    return (codes * scales).bfloat16()


@pytest.mark.parametrize("width", [64, 512])
@pytest.mark.parametrize("rows", [1, 3, 8, 128])
def test_quantization_mask_snapshot_and_graph(width, rows):
    from tensorfold.families.glm5_next.cuda import cache_fp8 as cf, latent
    torch.manual_seed(15)
    x = torch.randn(rows, width + 16, device="cuda").bfloat16()[:, :width]
    x[0].zero_()
    if rows > 1:
        x[1].mul_(1000)
    cache = cf.allocate(rows + 10, width, "cuda", "fp8")
    pos = torch.tensor([2], device="cuda", dtype=torch.int32)
    valid = torch.ones(rows, device="cuda", dtype=torch.int32)
    if rows > 2:
        valid[-1] = 0
    latent.latent_write(x, cache, pos, valid=valid)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent.latent_write(x, cache, pos, valid=valid)
    x.mul_(.125)
    pos.fill_(4)
    graph.replay()
    live = rows if rows < 3 else rows - 1
    src = x[:live].float()
    scales = (src.abs().amax(1, keepdim=True).clamp_min(1.e-12).double() / 448.).float()
    codes = (src.double() / scales.double()).float().clamp(-448, 448).to(torch.float8_e4m3fn)
    got = cache[4:4 + live]
    assert torch.equal(got[:, :width], codes.view(torch.uint8))
    assert torch.equal(got[:, width:].contiguous().view(torch.float32), scales)
    assert torch.equal(unpack(got, width), (codes.float() * scales).bfloat16())
    if live < rows:
        assert not cache[4 + live].any()
    restored = torch.empty_like(cache)
    restored.copy_(cache.clone())
    assert torch.equal(restored, cache)
    error = (unpack(got, width).float() - src).square().sum() / src.square().sum().clamp_min(1e-30)
    assert error.sqrt().item() < .035


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("R", [1, 3, 8])
def test_attention_equals_dequantized_oracle_and_serial(sparse, R):
    from tensorfold.families.glm5_next.cuda import cache_fp8 as cf, latent
    torch.manual_seed(20)
    N, H, L, RP = (131080 if sparse else 1600), 4, 512, 64
    pos = torch.tensor([N - R], dtype=torch.int32, device="cuda")
    zero = torch.zeros_like(pos)
    lc, kp = cf.allocate(N, L, "cuda", "fp8"), cf.allocate(N, RP, "cuda", "fp8")
    for width, cache in [(L, lc), (RP, kp)]:
        latent.latent_write(torch.randn(N, width, device="cuda").bfloat16(), cache, zero)
    q = torch.randn(R, H, L, device="cuda").bfloat16()
    qp = torch.randn(R, H, RP, device="cuda").bfloat16()
    s = latent.LatentScratch(R, H, 4, "cuda", rope=RP)
    out, want = torch.empty_like(q), torch.empty_like(q)
    lref, pref = unpack(lc, L), unpack(kp, RP)
    counts = torch.tensor([2048 - r for r in range(R)], dtype=torch.int32, device="cuda")
    tokens = torch.full((R, 2048), -1, dtype=torch.int32, device="cuda")
    for r in range(R):
        n = 2048 - r
        if sparse:
            tokens[r, :n] = torch.linspace(0, N - R + r, n, device="cuda").to(torch.int32)
    def run(queries, rotary, cache, rope, output, p, tok, cnt):
        if sparse:
            latent.sparse_attention(queries, cache, tok, cnt, output, 256**-.5, qpe=rotary, kpe=rope, scratch=s)
        else:
            latent.attention(queries, cache, p, s, scale=256**-.5, nch=4, out=output, qpe=rotary, kpe=rope)
    run(q, qp, lc, kp, out, pos, tokens, counts)
    run(q, qp, lref, pref, want, pos, tokens, counts)
    # Inlining dequantization changes Triton scheduling; compare the independent
    # BF16-storage oracle numerically, but require exact FP8 serial/graph bits below.
    delta = (out.float() - want.float())
    assert delta.square().sum().sqrt() / want.float().square().sum().sqrt() < .001
    assert delta.abs().max() < .005 * want.float().abs().max()
    for r in range(R):
        one = torch.empty_like(q[:1])
        run(q[r:r+1], qp[r:r+1], lc, kp, one, pos + r, tokens[r:r+1], counts[r:r+1])
        assert torch.equal(out[r], one[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(q, qp, lc, kp, out, pos, tokens, counts)
    q.mul_(.5)
    graph.replay()
    run(q, qp, lref, pref, want, pos, tokens, counts)
    delta = out.float() - want.float()
    assert delta.square().sum().sqrt() / want.float().square().sum().sqrt() < .001
    eager = torch.empty_like(out)
    run(q, qp, lc, kp, eager, pos, tokens, counts)
    assert torch.equal(out, eager)


@pytest.fixture(scope="module")
def fp8_weights():
    from test_glm_segmented import tiny_weights
    w = tiny_weights()
    w.meta["kv_dtype"] = "fp8"
    return w


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("C", [1, 2, 4])
def test_full_fixture_main_mtp_packed_private_and_bytes(fp8_weights, C, sparse):
    from tensorfold.families.glm5_next.cuda import forward as fw, segmented as seg, latent, multi, decode, mtp
    w = fp8_weights
    sts = []
    for i in range(C):
        st = fw.State(w, 2112, 8)
        pos = 2053 + i if sparse else 13 + i
        zero = torch.zeros(1, device="cuda", dtype=torch.int32)
        for cache, width in [(x, 512) for x in st.kc + [st.mtp_kc]] + [(x, 64) for x in st.kpe + [st.mtp_kpe]]:
            latent.latent_write(torch.randn(pos, width, device="cuda").bfloat16(), cache, zero)
        for planes in st.index.values():
            planes[0][:pos].normal_()
        st.set_pos(pos); st.set_mtp_len(pos)
        assert all(x.dtype == torch.uint8 for x in st.kc + st.kpe + [st.mtp_kc, st.mtp_kpe])
        actual = sum(x.numel() * x.element_size() for x in decode._row_views(st, st.capacity, st.capacity)) + 8
        assert actual == multi.state_bytes(w, st.capacity)
        clone = st.clone()
        assert all(torch.equal(x, y) and x.data_ptr() != y.data_ptr() for x, y in
                   zip(decode._row_views(st, st.capacity, st.capacity), decode._row_views(clone, st.capacity, st.capacity)))
        sts.append(st)
    b = fw.Buffers(w, 32, 2112)
    wins = [(st, [12, 17, 25][:1 + i % 3]) for i, st in enumerate(sts)]
    want = torch.cat([fw.forward(w, st, b, ids).clone() for st, ids in wins])
    physical = seg.stage_streams(w, b, wins, padded=True)
    got = seg.compute_streams(w, physical, b)
    logical = seg.logical_segments(wins, physical)
    assert torch.equal(seg.sampling_logits(got, physical, logical), want)
    b = fw.Buffers(w, 32, 2112)  # no stale main-padding validity mask in the solo MTP oracle
    mwins = [(st, ids, torch.randn(len(ids), 256, device="cuda").bfloat16()) for st, ids in wins]
    expected = torch.cat([mtp.mtp_forward(w, st, b, ids, h).clone() for st, ids, h in mwins])
    ms = seg.mtp_stage_streams(w, b, mwins)
    assert torch.equal(seg.mtp_compute_streams(w, ms, b), expected)
