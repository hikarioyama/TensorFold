"""A distributed nucleus is certified globally without altering mass, tie, or keyed RNG rules."""
import numpy as np
import pytest
import torch

from tensorfold.cuda import sampling as cs
from tensorfold.engine.exact_sampling import Sampling


def shares(width=32, count=2, tie=False):
    vals = np.zeros((4, 1, width), dtype=np.float64)
    vals[:, :, 0] = 10
    if tie:
        vals[:] = 10
    ids = np.arange(4 * width, dtype=np.int64).reshape(4, 1, width)
    mass = np.floor(np.exp(vals - 10) * cs.MASS).astype(np.int64)
    return vals[:, :, :count], ids[:, :, :count], mass[:, :, :count], mass.sum(axis=-1), np.full((4, 1), width)


def test_shards_need_not_each_cover_global_mass():
    s = Sampling(17, 1., 0, .95)
    narrow = shares()
    assert all(int(x) < int(narrow[3].sum()) * .95 for x in narrow[3].reshape(-1))
    assert cs._draw(narrow, [100], s) == cs._draw(shares(count=32), [100], s)
    assert cs._draw(narrow, [100], s) is not None


def test_global_boundary_tie_still_requires_whole_shards():
    assert cs._draw(shares(count=31, tie=True), [100], Sampling(17, 1., 0, .5)) is None


@pytest.mark.parametrize('top_p', [.01, .3, .8, .95, 1.])
@pytest.mark.parametrize('min_p', [0., .01, .1])
def test_certified_candidates_equal_whole_vocab(monkeypatch, top_p, min_p):
    gen = torch.Generator().manual_seed(940)
    for scale in (0., 1., 3., 10.):
        logits = (torch.randn((12, 3000), generator=gen) * scale).bfloat16()
        s = Sampling(335, .9, 0, top_p, min_p)
        positions = list(range(91, 103))
        monkeypatch.setattr(cs, 'NUCLEUS', 64)
        narrow = cs.nucleus_rows(logits, positions, s)
        monkeypatch.setattr(cs, 'NUCLEUS', 3000)
        whole = cs.nucleus_rows(logits, positions, s)
        assert narrow == whole
