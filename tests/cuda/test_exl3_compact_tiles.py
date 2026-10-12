"""Nonempty scheduling: integer oracle, truncation, canaries, changed-input replay."""
import pytest
import torch
from tensorfold.cuda.exl3 import experts, tiles
from test_exl3_experts import _layer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows,slots,count", [(1,9,256),(128,9,256),(129,9,63),(2048,9,256),(2049,32,257)])
@pytest.mark.parametrize("maxm", [1,8,15,16,17,33,2048])
def test_descriptor_oracle_graph_canaries(rows, slots, count, maxm):
    maxu = min(rows * slots, count)
    picks = torch.empty(rows, slots, dtype=torch.int32, device="cuda")
    ids = torch.empty(maxu, dtype=torch.int32, device="cuda")
    uc = torch.empty(1, dtype=torch.int32, device="cuda")
    members = torch.full((maxu,maxm), -123, dtype=torch.int32, device="cuda")
    s = tiles.Scratch(rows, slots, count, "cuda")
    storage = torch.full((s.capacity * 2 + 64,), 987654, dtype=torch.int32, device="cuda")
    s.descriptors = storage[32:-32].view(s.capacity,2)
    def run():
        experts._ext().group_parallel(picks, ids, uc, members, rows, slots, count)
        s.build(members, uc, rows * slots)
    picks.zero_(); run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): run()
    for kind in (0,1,2,3,0):
        if kind == 0:
            picks.copy_((torch.arange(rows * slots, device="cuda") % count).view(rows,slots))
        elif kind == 1: picks.fill_(count - 1)
        elif kind == 2: picks.fill_(count)
        else:
            gen = torch.Generator(device="cuda").manual_seed(789)
            picks.copy_(torch.randint(-1,count+1,picks.shape,device="cuda",generator=gen,dtype=torch.int32))
        g.replay()
        found = {}
        for row, values in enumerate(picks.cpu().tolist()):
            for slot, expert in enumerate(values):
                if 0 <= expert < count:
                    found[expert] = min(maxm, found.get(expert,0) + 1)
        expected = [[u,t] for u,e in enumerate(sorted(found)) for t in range((found[e]+15)//16)]
        assert s.count.item() == len(expected) <= s.capacity
        assert s.descriptors[:len(expected)].cpu().tolist() == expected
        assert torch.all(storage[:32] == 987654) and torch.all(storage[-32:] == 987654)


@pytest.fixture(scope="module", params=[False,True])
def weights(request):
    widths = [(6,6,6)]*16 if not request.param else [(6,8,6),(8,6,8)]*8
    return _layer(16,512,256,widths,"mul1",713)[0]


@pytest.mark.parametrize("rows", [1,2,8,127,128,129,257,512,2048])
@pytest.mark.parametrize("scale", [1.,3.,.01,-1.])
def test_routed_partials_and_replay(weights, rows, scale):
    ex = weights
    gen = torch.Generator(device="cuda").manual_seed(987)
    x = (torch.randn(rows,ex.dims,device="cuda",generator=gen)*scale).bfloat16()
    pick = ((torch.arange(rows,device="cuda")[:,None]*3 + torch.arange(8,device="cuda")) % ex.count).int()
    pick = torch.cat([pick,torch.full((rows,1),ex.count,device="cuda",dtype=torch.int32)],1)
    s = experts.Scratch(ex, rows,9,grouped_mode=3,route_mode=2,compact_tiles=True)
    s.adaptive_tiles=True
    def run(): experts.routed(x,pick,None,ex,s,None,rows,10.,act_mode=experts.ACT_BF16)
    s.compact_tiles=True;run();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    for turn in range(3):
        s.compact_tiles=False;s.z.fill_(float("nan"));run()
        want_z=s.z.clone();want_y=s.y.clone();want_xd=s.xd.clone()
        s.z.fill_(float("nan"));s.compact_tiles=True;g.replay()
        for got,want in ((s.z,want_z),(s.y,want_y)):
            assert torch.equal(got.view(torch.int32),want.view(torch.int32))
        assert torch.equal(s.xd.view(torch.int16),want_xd.view(torch.int16))
        x[:,0].add_(.25);pick[:,:8].add_(1).remainder_(ex.count)
