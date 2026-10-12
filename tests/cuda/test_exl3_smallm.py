"""Shared-memory permutations and small-M resource specialization preserve every valid FP32 bit."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required', allow_module_level=True)

from tensorfold.cuda.exl3 import experts
from test_exl3_experts import _layer


@pytest.fixture(scope='module', params=[6, 8, 'mixed'])
def weights(request):
    bits = [(request.param,)*3]*16 if request.param != 'mixed' else [(6,8,6),(8,6,8)]*8
    return _layer(16, 512, 256, bits, 'mul1', 734)[0]


@pytest.mark.parametrize('rows', [1,2,3,4,5,7,8,9,12,16,17,24,32,64,128])
@pytest.mark.parametrize('route', ['separate', 'concentrated', 'invalid'])
def test_valid_rows_exact_with_poisoned_scratch(weights, rows, route):
    w = weights
    gen = torch.Generator(device='cuda').manual_seed(711+rows)
    x = torch.randn((rows,w.dims),device='cuda',generator=gen).bfloat16()
    x[::5].zero_()
    x[1::7].mul_(16)
    ids = torch.arange(8,device='cuda',dtype=torch.int32)[None,:].expand(rows,-1).clone()
    if route == 'separate': ids = (ids + 3*torch.arange(rows,device='cuda')[:,None]) % w.count
    if route == 'invalid': ids.fill_(w.count)
    ids = torch.cat([ids, torch.full((rows,1),w.count,device='cuda',dtype=torch.int32)],1).int()
    wt = torch.rand((rows,9),device='cuda',generator=gen); wt[:,-1]=1.
    shared = torch.randn((rows,w.dims),device='cuda',generator=gen)
    expected = None
    for mode in (0,1,2,3):
        s = experts.Scratch(w,rows,9,grouped_mode=mode)
        s.y.view(rows,9,w.dims)[:,-1].copy_(shared)
        s.z.fill_(float('nan'))
        s.xg.fill_(float('nan'));s.xu.fill_(float('nan'));s.xd.fill_(float('nan'))
        got = experts.routed(x,ids,wt,w,s,None,rows,10.,act_mode=experts.ACT_BF16).clone()
        assert torch.isfinite(got).all()
        if expected is None: expected=got
        else: assert torch.equal(got.view(torch.int32),expected.view(torch.int32)), (rows,route,mode)
        if mode >= 2:
            again = experts.routed(x,ids,wt,w,s,None,rows,10.,act_mode=experts.ACT_BF16).clone()
            assert torch.equal(again.view(torch.int32),expected.view(torch.int32))


@pytest.mark.parametrize('rows',[1,3,6,9,12,17,32])
@pytest.mark.parametrize('mode',[2,3])
def test_optimized_graph_and_row_permutation(weights, rows, mode):
    w=weights
    gen=torch.Generator(device='cuda').manual_seed(987)
    x=torch.randn((rows,w.dims),device='cuda',generator=gen).bfloat16()
    pick=torch.cat([(torch.arange(8,device='cuda',dtype=torch.int32)[None,:]+torch.arange(rows,device='cuda')[:,None])%16,
                     torch.full((rows,1),16,device='cuda',dtype=torch.int32)],1).int()
    wt=torch.ones((rows,9),device='cuda')
    s=experts.Scratch(w,rows,9,grouped_mode=mode)
    out=torch.empty((rows,w.dims),device='cuda')
    def run():return experts.routed(x,pick,wt,w,s,out,rows,10.,act_mode=experts.ACT_BF16)
    want=run().clone()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    for _ in range(3):g.replay()
    assert torch.equal(out.view(torch.int32),want.view(torch.int32))
    other=experts.routed(x.flip(0).contiguous(),pick.flip(0).contiguous(),wt,w,s,out,rows,10.,act_mode=experts.ACT_BF16)
    assert torch.equal(other.flip(0).contiguous().view(torch.int32),want.view(torch.int32))


@pytest.mark.parametrize('rows',[1,3,6,9,12,17,32])
def test_tile_policy_keeps_logical_reduction_and_bits(weights, rows):
    w=weights
    gen=torch.Generator(device='cuda').manual_seed(702)
    x=torch.randn((rows,w.dims),device='cuda',generator=gen).bfloat16()
    ids=torch.arange(8,device='cuda',dtype=torch.int32)[None,:].expand(rows,8).contiguous()
    wt=torch.full((rows,8),.125,device='cuda')
    baseline=experts.Scratch(w,rows,8)
    candidate=experts.Scratch(w,rows,8,grouped_mode=2)
    candidate.adaptive_tiles=True
    a=experts.routed(x,ids,wt,w,baseline,None,rows,10.,act_mode=experts.ACT_BF16)
    b=experts.routed(x,ids,wt,w,candidate,None,rows,10.,act_mode=experts.ACT_BF16)
    assert torch.equal(a.view(torch.int32),b.view(torch.int32))
