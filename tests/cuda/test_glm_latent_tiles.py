"""Independent attention oracles at key-tile/chunk boundaries (with and without GLM-5.3 RoPE).

Window-vs-serial comparisons alone cannot detect a key traversal bug shared by both paths.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def inputs(rows, length, rope):
    gen = torch.Generator(device="cuda").manual_seed(5300 + rows + length)
    def rand(*shape):
        return torch.randn(shape, device="cuda", generator=gen).to(torch.bfloat16)
    qa, cache = rand(rows, 16, 512), rand(length, 512)
    qp = rand(rows, 16, rope) if rope else None
    kp = rand(length, rope) if rope else None
    return qa, cache, qp, kp


def reference(q, cache, qp=None, kp=None):
    # CPU fp64, independent of Triton, chunking and tensor-core rounding.
    q, cache = q.double().cpu(), cache.double().cpu()
    score = q @ cache.T
    if qp is not None:
        score += qp.double().cpu() @ kp.double().cpu().T
    return torch.softmax(score * 0.0625, dim=-1) @ cache


def assert_attention(got, want):
    got, want = got.double().cpu(), want.double().cpu()
    assert torch.isfinite(got).all()
    rel = (got - want).norm() / want.norm()
    assert rel < 0.008, f"relative L2 error {rel.item():.6f}"
    assert (got - want).abs().max() < 0.015 * want.abs().max()


@pytest.mark.parametrize("rope", [0, 64])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 39, 64, 65, 511, 512, 513, 1025, 2048])
def test_dense_key_tiles_against_fp64(length, rope):
    from tensorfold.families.glm5_next.cuda import latent

    qa, cache, qp, kp = inputs(1, length, rope)
    nch = latent.chunks_for(length)
    scratch = latent.LatentScratch(1, 16, nch, "cuda", rope=rope)
    out = torch.empty_like(qa)
    latent.attention(qa, cache, torch.tensor([length - 1], device="cuda", dtype=torch.int32), scratch,
                     scale=0.0625, nch=nch, out=out, qpe=qp, kpe=kp)
    assert_attention(out[0], reference(qa[0], cache, None if qp is None else qp[0], kp))


@pytest.mark.parametrize("rope", [0, 64])
@pytest.mark.parametrize("length", [31, 32, 33, 39, 65, 511, 512, 513, 1025, 2048])
def test_sparse_key_tiles_against_fp64(length, rope):
    from tensorfold.families.glm5_next.cuda import latent

    # Sorted nonconsecutive IDs, one partial row, one empty row, and poisoned padding.
    qa, cache, qp, kp = inputs(3, 2 * length + 17, rope)
    tokens = torch.full((3, length + 7), -1, device="cuda", dtype=torch.int32)
    tokens[0, :length] = 2 * torch.arange(length, device="cuda", dtype=torch.int32)
    shorter = max(1, length - 13)
    tokens[1, :shorter] = 2 * torch.arange(shorter, device="cuda", dtype=torch.int32) + 1
    counts = torch.tensor([length, shorter, 0], device="cuda", dtype=torch.int32)
    out = torch.full_like(qa, 7)
    latent.sparse_attention(qa, cache, tokens, counts, out, 0.0625, qpe=qp, kpe=kp)
    for r, count in enumerate([length, shorter]):
        ids = tokens[r, :count].long()
        want = reference(qa[r], cache[ids], None if qp is None else qp[r], None if kp is None else kp[ids])
        assert_attention(out[r], want)
    assert torch.equal(out[2], torch.full_like(out[2], 7))


@pytest.mark.parametrize("pos,rows", [(0, 65), (30, 9), (510, 7), (1020, 9)])
def test_causal_prefill_rows_against_fp64(pos, rows):
    from tensorfold.families.glm5_next.cuda import latent

    qa, cache, qp, kp = inputs(rows, pos + rows, 64)
    nch = latent.chunks_for(pos + rows)
    scratch = latent.LatentScratch(rows, 16, nch, "cuda", rope=64)
    out = torch.empty_like(qa)
    latent.attention(qa, cache, torch.tensor([pos], device="cuda", dtype=torch.int32), scratch,
                     scale=0.0625, nch=nch, out=out, qpe=qp, kpe=kp)
    for r in sorted({0, 1, rows // 2, rows - 1}):
        n = pos + r + 1
        assert_attention(out[r], reference(qa[r], cache[:n], qp[r], kp[:n]))


def test_uniform_attention_visits_each_key_once():
    from tensorfold.families.glm5_next.cuda import latent

    # Different chunk means expose duplicate coverage as well as skipped tiles.
    length = 1025
    qa = torch.zeros((1, 16, 512), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((length, 512), device="cuda", dtype=torch.bfloat16)
    cache[32:512] = 1
    cache[512:1024] = 2
    cache[1024:] = 4
    qp = torch.zeros((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    kp = torch.zeros((length, 64), device="cuda", dtype=torch.bfloat16)
    want = cache.double().mean(0).expand(16, -1)
    scratch = latent.LatentScratch(1, 16, 3, "cuda", rope=64)
    out = torch.empty_like(qa)
    latent.attention(qa, cache, torch.tensor([length - 1], device="cuda", dtype=torch.int32), scratch,
                     scale=0.0625, nch=3, out=out, qpe=qp, kpe=kp)
    assert_attention(out[0], want)
    tokens = torch.arange(length, device="cuda", dtype=torch.int32).view(1, -1)
    counts = torch.tensor([length], device="cuda", dtype=torch.int32)
    latent.sparse_attention(qa, cache, tokens, counts, out, 0.0625, qpe=qp, kpe=kp)
    assert_attention(out[0], want)
