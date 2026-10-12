"""Integer-only route plans: stable IDs/members, canaries, graphs and FP32 output bits."""
import pytest
import torch

from tensorfold.cuda.exl3 import experts
from test_exl3_experts import _layer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(pick, experts_count, maxm):
    rows = pick.tolist()
    ids = sorted({e for row in rows for e in row if 0 <= e < experts_count})
    members = []
    for e in ids:
        found = [r * 32 + slot for r, row in enumerate(rows) for slot, value in enumerate(row) if value == e][:maxm]
        members.append(found + [-1] * (maxm - len(found)))
    return ids, members


@pytest.mark.parametrize("rows", [1, 2, 3, 6, 9, 12, 32, 33, 128])
@pytest.mark.parametrize("count", [1, 16, 63, 256, 257])
@pytest.mark.parametrize("pattern", ["random", "concentrated", "separate", "invalid"])
def test_plan_matches_oracle_and_legacy_with_canaries(rows, count, pattern):
    slots = 9
    g = torch.Generator().manual_seed(302 + rows)
    pick = torch.randint(-1, count + 2, (rows, slots), generator=g, dtype=torch.int32)
    if pattern == "concentrated":
        pick.fill_(count - 1)  # duplicates also preserve the generic maxm truncation contract
    elif pattern == "separate":
        pick = (torch.arange(rows * slots, dtype=torch.int32) % count).view(rows, slots)
    elif pattern == "invalid":
        pick.fill_(-1)
    pick[:, -1] = count  # shared-expert/invalid sentinel
    ids_ref, members_ref = oracle(pick, count, rows)
    picks = pick.cuda()
    maxu = min(rows * slots, count)
    for fn in (experts._ext().group, experts._ext().group_ballot):
        id_storage = torch.full((maxu + 64,), 987654, dtype=torch.int32, device="cuda")
        member_storage = torch.full((maxu * rows + 64,), 987654, dtype=torch.int32, device="cuda")
        ids, members = id_storage[32:-32], member_storage[32:-32].view(maxu, rows)
        uc = torch.full((1,), -17, dtype=torch.int32, device="cuda")
        fn(picks, ids, uc, members, rows, slots, count)
        assert uc.item() == len(ids_ref)
        assert ids[:len(ids_ref)].cpu().tolist() == ids_ref
        assert members[:len(ids_ref)].cpu().tolist() == members_ref
        assert torch.all(id_storage[:32] == 987654) and torch.all(id_storage[-32:] == 987654)
        assert torch.all(member_storage[:32] == 987654) and torch.all(member_storage[-32:] == 987654)


@pytest.mark.parametrize("slots", [1, 8, 16, 32])
@pytest.mark.parametrize("rows", [32, 128])
def test_graph_replay_changes_routes_and_counts(slots, rows):
    count = 256
    picks = torch.zeros(rows, slots, dtype=torch.int32, device="cuda")
    ids = torch.empty(min(rows * slots, count), dtype=torch.int32, device="cuda")
    members = torch.empty(ids.numel(), rows, dtype=torch.int32, device="cuda")
    uc = torch.empty(1, dtype=torch.int32, device="cuda")
    fn = experts._ext().group_ballot
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
            picks.fill_(0)
        graph.replay()
        want_ids, want_members = oracle(picks.cpu(), count, rows)
        assert uc.item() == len(want_ids)
        assert ids[:len(want_ids)].tolist() == want_ids
        assert members[:len(want_ids)].tolist() == want_members


@pytest.fixture(scope="module")
def weights():
    return _layer(16, 512, 256, [(6, 8, 6), (8, 6, 8)] * 8, "mul1", 419)[0]


@pytest.mark.parametrize("rows", [1, 3, 6, 9, 12, 32, 128])
@pytest.mark.parametrize("grouped_mode", [0, 3])
@pytest.mark.parametrize("scale", [1., 3., .01, -1.])
def test_routed_fp32_bits_and_invalid_slots(weights, rows, grouped_mode, scale):
    w = weights
    gen = torch.Generator(device="cuda").manual_seed(432 + rows)
    x = (torch.randn(rows, w.dims, generator=gen, device="cuda") * scale).bfloat16()
    pick = (torch.arange(8, device="cuda")[None, :] + 3 * torch.arange(rows, device="cuda")[:, None]) % w.count
    pick = torch.cat([pick, torch.full((rows, 1), w.count, device="cuda")], dim=1).int()
    pick[::7] = w.count
    wt = torch.rand(rows, 9, generator=gen, device="cuda")
    shared = torch.randn(rows, w.dims, generator=gen, device="cuda")
    outputs = []
    for mode in (0, 1):
        s = experts.Scratch(w, rows, 9, grouped_mode=grouped_mode, route_mode=mode)
        s.z.fill_(float("nan"))
        s.y.view(rows, 9, w.dims)[:, -1].copy_(shared)
        out = experts.routed(x, pick, wt, w, s, None, rows, 10., act_mode=experts.ACT_BF16)
        outputs.append(out.clone())
    assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
