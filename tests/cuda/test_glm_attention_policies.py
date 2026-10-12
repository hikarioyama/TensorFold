"""Dense boundary and large sparse slabs preserve every chunk partial and masked row."""
import pytest,torch
from tensorfold.families.glm5_next.cuda import latent,cache_fp8
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

def bits(t):return t.view(torch.int32) if t.dtype==torch.float32 else t

def inputs(rows,length,factor):
    gen=torch.Generator(device='cuda').manual_seed(1300+rows)
    q=(torch.randn(rows,16,512,device='cuda',generator=gen)*factor).bfloat16();qp=(torch.randn(rows,16,64,device='cuda',generator=gen)*factor).bfloat16()
    lc=cache_fp8.allocate(length,512,'cuda','fp8');kp=cache_fp8.allocate(length,64,'cuda','fp8');zero=torch.zeros(1,dtype=torch.int32,device='cuda')
    for cache,width in ((lc,512),(kp,64)):latent.latent_write((torch.randn(length,width,device='cuda',generator=gen)*factor).bfloat16(),cache,zero)
    return q,qp,lc,kp

@pytest.mark.parametrize('rows',[1,2])
@pytest.mark.parametrize('factor',[1.,3.,.01,-1.])
def test_dense_dynamic_boundary_bits(rows,factor,monkeypatch):
    q,qp,lc,kp=inputs(rows,3072,factor);s=latent.LatentScratch(rows,16,6,'cuda',rope=64);out=torch.empty_like(q);pos=torch.zeros(1,device='cuda',dtype=torch.int32)
    def run():latent.attention(q,lc,pos,s,scale=.0625,nch=6,out=out,qpe=qp,kpe=kp)
    for at in (0,30,31,32,510,511,512,1022,1023,1024,2046,2047,2048):
        pos.fill_(at);monkeypatch.setattr(latent,'DENSE_POLICY','legacy');run();want=[t.clone() for t in (s.po,s.pm,s.pl,out)]
        monkeypatch.setattr(latent,'DENSE_POLICY','decode443');run()
        assert all(torch.equal(bits(t),bits(w)) for t,w in zip((s.po,s.pm,s.pl,out),want))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    q.neg_();qp.mul_(.5);pos.fill_(2047);g.replay();want=out.clone();monkeypatch.setattr(latent,'DENSE_POLICY','legacy');run();assert torch.equal(want,out)

@pytest.mark.parametrize('rows',[32,128,512,2048])
@pytest.mark.parametrize('length',[4096,16384,76000])
@pytest.mark.parametrize('factor',[1.,3.,.01,-1.])
def test_prefill_partials_counts_and_graph(rows,length,factor,monkeypatch):
    q,qp,lc,kp=inputs(rows,length,factor);s=latent.LatentScratch(rows,16,4,'cuda',rope=64);out=torch.full_like(q,7)
    tok=torch.linspace(0,length-2,2048,device='cuda').int().repeat(rows,1)
    cnt=torch.tensor([0,31,32,33,511,512,513,2048],device='cuda',dtype=torch.int32).repeat((rows+7)//8)[:rows]
    bucket=min(229384,1 << (length+rows-1).bit_length())
    def run():latent.sparse_attention(q,lc,tok,cnt,out,.0625,qpe=qp,kpe=kp,scratch=s,bucket=bucket)
    monkeypatch.setattr(latent,'PREFILL_MLA_POLICY','legacy');run();want=[t.clone() for t in (s.po,s.pm,s.pl,out)]
    monkeypatch.setattr(latent,'PREFILL_MLA_POLICY','prefill1643');run();assert all(torch.equal(bits(t),bits(w)) for t,w in zip((s.po,s.pm,s.pl,out),want))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    q.neg_();qp.mul_(.5);cnt.fill_(513);tok.add_(1);g.replay();want=out.clone();monkeypatch.setattr(latent,'PREFILL_MLA_POLICY','legacy');run();assert torch.equal(want,out)
