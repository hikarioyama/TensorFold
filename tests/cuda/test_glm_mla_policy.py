"""Qualified dispatch, four query/KV distributions and changed graph inputs."""
import pytest
import torch
from tensorfold.families.glm5_next.cuda import latent,cache_fp8

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")

@pytest.mark.parametrize("length",[4096,16384,34000,76000,131072,228800])
@pytest.mark.parametrize("rows",[1,2])
@pytest.mark.parametrize("factor",[1.,3.,.01,-1.])
def test_policy_partials_bit_exact(length,rows,factor,monkeypatch):
    gen=torch.Generator(device="cuda").manual_seed(length+rows)
    q=(torch.randn(rows,16,512,device="cuda",generator=gen)*factor).bfloat16()
    qp=(torch.randn(rows,16,64,device="cuda",generator=gen)*factor).bfloat16()
    lc=cache_fp8.allocate(length,512,"cuda","fp8");kp=cache_fp8.allocate(length,64,"cuda","fp8")
    zero=torch.zeros(1,device="cuda",dtype=torch.int32)
    for cache,width in ((lc,512),(kp,64)):
        latent.latent_write((torch.randn(length,width,device="cuda",generator=gen)*factor).bfloat16(),cache,zero)
    tok=torch.linspace(0,length-2,2048,device="cuda").int().repeat(rows,1)
    cnt=torch.empty(rows,device="cuda",dtype=torch.int32);s=latent.LatentScratch(rows,16,4,"cuda",rope=64)
    out=torch.empty_like(q);bucket=min(229384,1 << (length+rows-1).bit_length())
    def run():latent.sparse_attention(q,lc,tok,cnt,out,.0625,qpe=qp,kpe=kp,scratch=s,bucket=bucket)
    for count in (0,31,32,33,511,512,513,2048):
        cnt.fill_(count);out.fill_(7);monkeypatch.setattr(latent,"MLA_POLICY","legacy");run()
        want=[t.clone() for t in (s.po,s.pm,s.pl,out)]
        monkeypatch.setattr(latent,"MLA_POLICY","decode443");run()
        for got,ref in zip((s.po,s.pm,s.pl,out),want):
            assert torch.equal(got.view(torch.int32) if got.dtype==torch.float32 else got,ref.view(torch.int32) if ref.dtype==torch.float32 else ref)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    q.neg_();qp.mul_(.5);cnt.fill_(513);tok.add_(1)
    g.replay();want=out.clone();monkeypatch.setattr(latent,"MLA_POLICY","legacy");run();assert torch.equal(want,out)
