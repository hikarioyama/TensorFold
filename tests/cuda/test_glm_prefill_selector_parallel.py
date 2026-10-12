"""32-row exact selection: independent CPU stable ordering and changing graph inputs."""
import pytest,torch
from tensorfold.families.glm5_next.cuda import sparse53,selection53
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("rows",[9,32])
@pytest.mark.parametrize("width",[8192,16384,76000,229376])
@pytest.mark.parametrize("factor",[1.,3.,.01,-1.])
def test_independent_ordering_and_graph(rows,width,factor):
    gen=torch.Generator().manual_seed(947)
    cpu=torch.randn(rows,width,generator=gen)*factor
    cpu[:,::17]=0.;cpu[:,::19]=-0.;cpu[:,::23]=1.;cpu[:,::97]=float('inf');cpu[:,::103]=float('-inf')
    scores=cpu.cuda();pos=torch.tensor([width-rows],device='cuda',dtype=torch.int32)
    storage=torch.full((rows*2048+64,),987654,device='cuda',dtype=torch.int32);out=storage[32:-32].view(rows,2048)
    s=selection53.Scratch(rows,width,'cuda')
    def run():selection53.select(scores,out,pos,width,s)
    run();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    for turn in range(2):
        g.replay()
        for r in range(rows):
            limit=int(pos.cpu()[0])+r+1
            ids=torch.argsort(scores[r,:limit].cpu(),descending=True,stable=True)[:2048].sort().values
            assert torch.equal(out[r].cpu().long(),ids)
        assert torch.all(storage[:32]==987654) and torch.all(storage[-32:]==987654)
        scores.neg_();pos.sub_(3)
