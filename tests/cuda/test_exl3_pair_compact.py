"""Flat pair grid preserves each logical partial, pass counter and output bits."""
import pytest
import torch
from tensorfold.cuda.exl3 import linear
from test_exl3_linear import _tensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("bits", [4,5,6])
@pytest.mark.parametrize("rows", [1,2,8,9,15,16,17,31,32,127,128])
@pytest.mark.parametrize("scale", [1.,3.,.01,-1.])
def test_pair_workspace_changed_graph(bits,rows,scale,monkeypatch):
    a=linear.Exl3Linear.from_tensors(*_tensors("mul1",bits,kt=128,nt=16,seed=8),"mul1")
    b=linear.Exl3Linear.from_tensors(*_tensors("mul1",bits,kt=128,nt=8,seed=19),"mul1")
    a.split,b.split=(8,8),(16,8)
    a.kernel_mode=b.kernel_mode=7
    gen=torch.Generator(device="cuda").manual_seed(933)
    x=(torch.randn(rows,a.k,device="cuda",generator=gen)*scale).bfloat16()
    records=[]
    for compact in (False,True):
        monkeypatch.setattr(linear,"PAIR_COMPACT",compact)
        out=torch.full((rows,a.n+b.n+128),7.,device="cuda",dtype=torch.bfloat16)
        wa,wb=a.workspace(x.device,rows),b.workspace(x.device,rows)
        wa[1].fill_(float("nan"));wb[1].fill_(float("nan"))
        def run():linear.pair_call(a,b,x,out[:,:a.n],out[:,a.n:a.n+b.n],wa,wb)
        run();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):run()
        records.append((out,wa,wb,g))
    for _ in range(3):
        for _,_,_,g in records:g.replay()
        ref,ra,rb,_=records[0]
        for out,wa,wb,_ in records:
            assert torch.equal(out.view(torch.int16),ref.view(torch.int16))
            assert torch.all(out[:,a.n+b.n:]==7)
            for got,want in ((wa,ra),(wb,rb)):
                assert torch.equal(got[1].view(torch.int32),want[1].view(torch.int32))
                assert not got[2].count_nonzero().item()
        x[:,0].add_(.25)
