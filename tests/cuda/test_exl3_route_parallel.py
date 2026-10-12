"""Stable multi-CTA route plans: independent oracle, guards, graph lifetime, FP32 bits."""
import pytest
import torch

from tensorfold.cuda.exl3 import experts
from test_exl3_experts import _layer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(pick, count, maxm):
    found = {}
    for row, values in enumerate(pick.tolist()):
        for slot, expert in enumerate(values):
            if 0 <= expert < count:
                bucket = found.setdefault(expert, [])
                if len(bucket) < maxm:
                    bucket.append(row * 32 + slot)
    ids = sorted(found)
    return ids, [found[e] + [-1] * (maxm - len(found[e])) for e in ids]


SHAPES = [(0, 9, 256), (1, 9, 256), (2, 9, 256), (8, 9, 256),
          (127, 9, 256), (128, 9, 256), (129, 9, 256), (257, 9, 256),
          (2048, 9, 256), (2049, 9, 256), (2048, 32, 256),
          (513, 1, 1), (257, 8, 63), (129, 16, 257), (33, 32, 4096)]


@pytest.mark.parametrize("rows,slots,count", SHAPES)
@pytest.mark.parametrize("pattern", ["random", "concentrated", "separate", "invalid"])
@pytest.mark.parametrize("truncate", [False, True])
def test_plan_oracle_legacy_canaries(rows, slots, count, pattern, truncate):
    gen = torch.Generator().manual_seed(1092 + rows + count)
    pick = torch.randint(-2, count + 2, (rows, slots), generator=gen, dtype=torch.int32)
    if pattern == "concentrated":
        pick.fill_(count - 1)
    elif pattern == "separate":
        pick = (torch.arange(rows * slots, dtype=torch.int32) % count).view(rows, slots)
    elif pattern == "invalid":
        pick.fill_(-1)
    if slots > 1:
        pick[:, -1] = count
    maxm = max(1, rows // 3 if truncate else rows + 3)
    expected_ids, expected_members = oracle(pick, count, maxm)
    picks = pick.cuda()
    maxu = min(rows * slots, count)
    fns = [experts._ext().group_parallel]
    if rows * slots <= 24000:
        fns.append(experts._ext().group)
    for fn in fns:
        id_storage = torch.full((maxu + 64,), 987654, dtype=torch.int32, device="cuda")
        member_storage = torch.full((maxu * maxm + 64,), 987654, dtype=torch.int32, device="cuda")
        ids, members = id_storage[32:-32], member_storage[32:-32].view(maxu, maxm)
        count_storage = torch.full((65,), 987654, dtype=torch.int32, device="cuda")
        uc = count_storage[32:33]
        fn(picks, ids, uc, members, rows, slots, count)
        used = len(expected_ids)
        assert uc.item() == used
        assert ids[:used].cpu().tolist() == expected_ids
        assert members[:used].cpu().tolist() == expected_members
        # Unused IDs/member rows are not part of the plan and must remain untouched.
        assert torch.all(ids[used:] == 987654)
        assert torch.all(members[used:] == 987654)
        for storage in (id_storage, member_storage, count_storage):
            assert torch.all(storage[:32] == 987654)
            assert torch.all(storage[-32:] == 987654)


@pytest.mark.parametrize("rows,slots,count", [(1, 9, 256), (2, 9, 256), (129, 9, 256), (2048, 9, 256), (2048, 32, 257)])
def test_changed_input_graph_replay(rows, slots, count):
    picks = torch.zeros(rows, slots, dtype=torch.int32, device="cuda")
    ids = torch.empty(min(rows * slots, count), dtype=torch.int32, device="cuda")
    members = torch.empty(ids.numel(), rows, dtype=torch.int32, device="cuda")
    uc = torch.empty(1, dtype=torch.int32, device="cuda")
    fn = experts._ext().group_parallel
    fn(picks, ids, uc, members, rows, slots, count)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(picks, ids, uc, members, rows, slots, count)
    for kind in (0, 1, 2, 0):
        if kind == 0:
            picks.copy_((torch.arange(rows * slots, device="cuda") % count).view(rows, slots))
        elif kind == 1:
            picks.fill_(count)
        else:
            picks.fill_(count - 1)
        graph.replay()
        expected_ids, expected_members = oracle(picks.cpu(), count, rows)
        assert uc.item() == len(expected_ids)
        assert ids[:len(expected_ids)].tolist() == expected_ids
        assert members[:len(expected_ids)].tolist() == expected_members


@pytest.fixture(scope="module")
def weights():
    return _layer(16, 512, 256, [(6, 8, 6), (8, 6, 8)] * 8, "mul1", 713)[0]


@pytest.mark.parametrize("rows", [1, 2, 8, 129, 512, 2048])
@pytest.mark.parametrize("scale", [1., 3., .01, -1.])
@pytest.mark.parametrize("grouped_mode", [0, 3])
def test_routed_four_distributions_bit_exact(weights, rows, scale, grouped_mode):
    ex = weights
    gen = torch.Generator(device="cuda").manual_seed(985 + rows)
    x = (torch.randn(rows, ex.dims, generator=gen, device="cuda") * scale).bfloat16()
    pick = (torch.arange(8, device="cuda")[None, :] + 3 * torch.arange(rows, device="cuda")[:, None]) % ex.count
    pick = torch.cat([pick, torch.full((rows, 1), ex.count, device="cuda")], dim=1).int()
    wts = torch.rand(rows, 9, generator=gen, device="cuda")
    scratch = experts.Scratch(ex, rows, 9, grouped_mode=grouped_mode)
    shared = torch.randn(rows, ex.dims, generator=gen, device="cuda")
    outputs = []
    for mode in (0, 2):
        scratch.route_mode = mode
        scratch.z.fill_(float("nan"))
        scratch.y.fill_(float("nan"))
        scratch.y.view(rows, 9, ex.dims)[:, -1].copy_(shared)
        out = experts.routed(x, pick, wts, ex, scratch, None, rows, 10., act_mode=experts.ACT_BF16)
        outputs.append(out.clone())
    assert torch.isfinite(outputs[0]).all()
    assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))


@pytest.mark.parametrize("R,slots,E", [(-1, 9, 256), (1, 0, 256), (1, 33, 256), (1, 9, 0), (1, 9, 4097), (2**31, 9, 256)])
def test_invalid_dimensions_rejected(R, slots, E):
    pick = torch.zeros(32, dtype=torch.int32, device="cuda")
    ids = torch.empty(32, dtype=torch.int32, device="cuda")
    uc = torch.empty(1, dtype=torch.int32, device="cuda")
    members = torch.empty(32, 1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        experts._ext().group_parallel(pick, ids, uc, members, R, slots, E)
