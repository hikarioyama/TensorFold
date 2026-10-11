"""Transposed small-M MMA: valid output bits, counters, pair strides and graph replay."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required', allow_module_level=True)

from tensorfold.cuda.exl3.linear import Exl3Linear, pair_call
from test_exl3_linear import _tensors


@pytest.mark.parametrize('bits',[3,4,5,6])
@pytest.mark.parametrize('rows',[1,2,3,6,8,9,16,17,32,128])
@pytest.mark.parametrize('split',[(1,4),(4,4),(8,2),(4,8)])
def test_projection_bits_and_counter_lifecycle(bits, rows, split):
    t,su,sv=_tensors('mul1',bits,kt=32,nt=16)
    w=Exl3Linear.from_tensors(t,su,sv,'mul1')
    w.split=split
    g=torch.Generator(device='cuda').manual_seed(743)
    x=torch.randn((rows,w.k),generator=g,device='cuda').bfloat16()
    x[::5].zero_()
    want=w(x,out_dtype=torch.float32).clone()
    w.kernel_mode=1
    ws=w.workspace(x.device,rows)
    out=torch.empty_like(want)
    for _ in range(3):
        w(x,out=out,xh=ws[0],z=ws[1],counters=ws[2])
        assert torch.equal(out.view(torch.int32),want.view(torch.int32))
        assert not ws[2].count_nonzero().item()


@pytest.mark.parametrize('rows',[1,3,6,8,9,17])
@pytest.mark.parametrize('bits',[3,4,5,6])
def test_pair_graph_and_strided_output(rows,bits):
    a=Exl3Linear.from_tensors(*_tensors('mul1',bits,kt=32,nt=16,seed=8),'mul1')
    b=Exl3Linear.from_tensors(*_tensors('mul1',bits,kt=32,nt=8,seed=19),'mul1')
    a.split,b.split=(2,4),(4,4)
    x=torch.randn((rows,a.k),device='cuda').bfloat16()
    want_a,want_b=a(x).clone(),b(x).clone()
    a.kernel_mode=b.kernel_mode=1
    wide=torch.full((rows,a.n+b.n+128),7.,device='cuda',dtype=torch.bfloat16)
    left,right=wide[:,:a.n],wide[:,a.n:a.n+b.n]
    wa,wb=a.workspace(x.device,rows),b.workspace(x.device,rows)
    def run():pair_call(a,b,x,left,right,wa,wb)
    run();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for _ in range(3):graph.replay()
    assert torch.equal(left.view(torch.int16),want_a.view(torch.int16))
    assert torch.equal(right.view(torch.int16),want_b.view(torch.int16))
    assert torch.all(wide[:,a.n+b.n:]==7)
    assert not wa[2].count_nonzero().item() and not wb[2].count_nonzero().item()
