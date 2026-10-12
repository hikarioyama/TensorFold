import pytest
import torch
from tensorfold.families.glm5_next.cuda import sparse53
from tensorfold.families.glm5_next.cuda.selection53 import Scratch, select

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize('rows,npn', [(1,2048),(2,8192),(2,16384),(2,43024),(2,76000),(8,229376)])
@pytest.mark.parametrize('scale', [1.,3.,.01,-1.])
@pytest.mark.parametrize('pattern', ['random','ties','infinities','zeros'])
def test_exact_cpu_and_legacy(rows,npn,scale,pattern):
    gen=torch.Generator().manual_seed(546+rows+npn)
    cpu=torch.randn(rows,npn+31,generator=gen)*scale
    if pattern=='zeros': cpu.zero_()
    elif pattern=='ties': cpu=(cpu*2).round()
    elif pattern=='infinities':
        cpu[:,::7]=float('inf'); cpu[:,::11]=float('-inf'); cpu[:,::13]=-0.; cpu[:,::17]=0.
    scores=cpu.cuda()
    k=min(2048,npn)
    pos=torch.tensor([npn-rows-19],dtype=torch.int64,device='cuda')
    scratch=Scratch(rows,npn,'cuda')
    outputs=[]
    for fn in ('legacy','parallel'):
        storage=torch.full((rows*k+64,),-719,dtype=torch.int32,device='cuda')
        out=storage[32:-32].view(rows,k)
        if fn=='legacy': sparse53._select_rows[(rows,)](scores,out,scores.stride(0),npn,pos,K=k,BLOCK=sparse53.SEL_BLOCK,num_warps=4)
        else: select(scores,out,pos,npn,scratch)
        outputs.append(out.cpu())
        assert torch.all(storage[:32]==-719) and torch.all(storage[-32:]==-719)
    assert torch.equal(*outputs)
    for r in range(rows):
        limit=min(npn,npn-rows-19+r+1)
        ids=torch.argsort(cpu[r,:limit],descending=True,stable=True)[:k].sort().values.int()
        assert torch.equal(outputs[1][r,:len(ids)],ids)
        assert torch.all(outputs[1][r,len(ids):]==-719)


@pytest.mark.parametrize('rows,npn',[(1,16384),(2,76000),(8,43024)])
def test_graph_replay_new_values_and_positions(rows,npn):
    scores=torch.zeros(rows,npn,device='cuda')
    out=torch.empty(rows,2048,dtype=torch.int32,device='cuda')
    ref=torch.empty_like(out)
    pos=torch.tensor([npn-10],device='cuda',dtype=torch.int64)
    scratch=Scratch(rows,npn,'cuda')
    select(scores,out,pos,npn,scratch)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): select(scores,out,pos,npn,scratch)
    for seed,p in [(1,2047),(2,npn-1),(3,17),(4,2048)]:
        torch.manual_seed(seed)
        scores.normal_(); pos.fill_(p); out.fill_(-719); ref.fill_(-719)
        g.replay()
        sparse53._select_rows[(rows,)](scores,ref,scores.stride(0),npn,pos,K=2048,BLOCK=sparse53.SEL_BLOCK,num_warps=4)
        assert torch.equal(out,ref)


@pytest.mark.parametrize('rows,npn', [(1,16384),(2,43024),(8,16384),(9,16384)])
@pytest.mark.parametrize('scale',[1.,3.,.01,-1.])
def test_public_selector_preserves_scores_tokens_counts(monkeypatch,rows,npn,scale):
    monkeypatch.setattr(sparse53,'SELECTOR_MODE','parallel')
    torch.manual_seed(1943+rows)
    qi=(torch.randn(rows,4,128,device='cuda')*scale).bfloat16()
    w=torch.randn(rows,4,device='cuda')
    keys=torch.randn(npn,128,device='cuda').half()
    pos=torch.tensor([npn-rows-17],device='cuda',dtype=torch.int64)
    # np_max and bucket deliberately exceed the private physical key plane.
    outputs=[]; score_outputs=[]; counts_outputs=[]
    for scratch in [None,Scratch(min(rows,8),npn,'cuda')]:
        tokens=torch.full((rows,2048),-719,device='cuda',dtype=torch.int32)
        score=torch.full((rows,npn+127),float('nan'),device='cuda')
        got,count=sparse53.select_tokens(qi,w,keys,pos,rows,2048,npn*2,bucket=npn*2,
                                         scores=score,tokens=tokens,parallel_scratch=scratch)
        outputs.append(got.clone()); score_outputs.append(score[:,:npn].clone()); counts_outputs.append(count.clone())
        assert torch.isnan(score[:,npn:]).all()
    assert torch.equal(*outputs) and torch.equal(*counts_outputs)
    assert torch.equal(score_outputs[0].view(torch.int32),score_outputs[1].view(torch.int32))
