"""Value-only FP64 maxima and unsorted candidates retain integer masses and original draws."""
import pytest
import torch
from tensorfold.cuda import sampling as smp
from tensorfold.engine.exact_sampling import Sampling
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def flags(monkeypatch,on):
    for name in ('AMAX','UNSORTED','SINGLETON'):monkeypatch.setattr(smp,name,on)


@pytest.mark.parametrize('rows',[1,3,8,32])
@pytest.mark.parametrize('kind',['peaked','flat','boundary','random'])
@pytest.mark.parametrize('temperature',[.7,1.,1.3])
def test_optimized_nucleus_matches_scalar_fallback_and_fresh_graph(monkeypatch,rows,kind,temperature):
    x=torch.randn(rows,4096,device='cuda').bfloat16()
    if kind=='peaked':x[:,13]=30
    if kind=='flat':x.zero_()
    if kind=='boundary':x[:,:65]=12
    x[:,-24:]=-float('inf')
    positions=list(range(1024,1024+rows));seeds=[2**63+i for i in range(rows)]
    sampling=Sampling(3,temperature,0,.95)
    flags(monkeypatch,False)
    want=smp.nucleus_rows(x,positions,sampling,row_seeds=seeds,initial_candidates=64,fallback_candidates=2048)
    a,b=smp._scaled_mass(x,temperature,smp.one_rank)
    flags(monkeypatch,True)
    c,d=smp._scaled_mass(x,temperature,smp.one_rank)
    assert torch.equal(a.view(torch.int64),c.view(torch.int64)) and torch.equal(b,d)
    stage=smp.NucleusStage(x,temperature,64,0,None,smp.one_rank)
    call=lambda logits,*args:stage(logits)
    assert smp.nucleus_rows(x,positions,sampling,row_seeds=seeds,initial_candidates=64,fallback_candidates=2048,stage=call)==want
    y=torch.roll(x,7,1)
    flags(monkeypatch,False)
    want=smp.nucleus_rows(y,positions,sampling,row_seeds=seeds,initial_candidates=64,fallback_candidates=2048)
    flags(monkeypatch,True)
    assert smp.nucleus_rows(y,positions,sampling,row_seeds=seeds,initial_candidates=64,fallback_candidates=2048,stage=call)==want


@pytest.mark.parametrize('value',[0.,-0.,float('nan'),float('-inf')])
def test_mass_extremes_match_original_bits(monkeypatch,value):
    x=torch.full((3,129),value,device='cuda',dtype=torch.float64)
    def ranks(words):
        v=words.view(torch.float64)
        return torch.stack([v,v+.125,v-.125,v+.0625]).view(torch.float32).reshape(4,-1)
    flags(monkeypatch,False);a,b=smp._scaled_mass(x,1.,ranks)
    flags(monkeypatch,True);c,d=smp._scaled_mass(x,1.,ranks)
    assert torch.equal(a.view(torch.int64),c.view(torch.int64)) and torch.equal(b,d)
