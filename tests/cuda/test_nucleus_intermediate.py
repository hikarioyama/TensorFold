"""GPU masses and candidate transfers keep original shared-sampler IDs/probabilities."""
import pytest
import torch

from tensorfold.cuda.sampling import nucleus_rows
from tensorfold.engine.exact_sampling import Sampling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 3, 12])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("top_p,min_p", [(.5, 0.), (.95, 0.), (1., 0.), (1., .02)])
def test_gpu_intermediate_is_exact(rows, flat, top_p, min_p):
    gen = torch.Generator(device="cuda").manual_seed(200 + rows)
    logits = (torch.randn(rows, 38720, generator=gen, device="cuda") * 2.5).bfloat16()
    if flat:
        logits.zero_()
    logits[:, -24:] = -torch.inf
    sampling = Sampling(2**64-1, .9, 0, top_p, min_p)
    positions = list(range(1024, 1024 + rows))
    want_probs, got_probs = [], []
    want = nucleus_rows(logits, positions, sampling, offset=116160, probs=want_probs)
    got = nucleus_rows(logits, positions, sampling, offset=116160, probs=got_probs, fallback_candidates=4096)
    assert got == want and got_probs == want_probs
    assert all(116160 <= token < 154856 for token in got)
