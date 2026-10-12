"""Separate G2 binary: fixed K/warp order, all tails, pair/single and F32."""
import importlib.util,os
from pathlib import Path
import pytest,torch
from tensorfold.cuda.exl3 import linear
from test_exl3_linear import _tensors
pytestmark=[pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required"),pytest.mark.skipif(not os.environ.get("TF_OPT_G2_ROOT"),reason="experimental G2 artifact required")]

@pytest.fixture(scope="module")
def probe():
    name=os.environ.get('TF_OPT_G2_NAME','tensorfold_exl3_g2_probe')
    path=Path(os.environ['TF_OPT_G2_ROOT'])/name/(name+'.so')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize("bits",[4,5,6])
@pytest.mark.parametrize("rows",[1,2,8,9,15,16,17,31,32,33,64,127,128])
@pytest.mark.parametrize("factor",[1.,3.,.01,-1.])
@pytest.mark.parametrize("pair",[False,True])
def test_g2_bits_and_counters(probe,bits,rows,factor,pair,monkeypatch):
    baseline=linear._ext()
    a=linear.Exl3Linear.from_tensors(*_tensors('mul1',bits,kt=128,nt=16,seed=8),'mul1')
    b=linear.Exl3Linear.from_tensors(*_tensors('mul1',bits,kt=128,nt=8,seed=19),'mul1')
    a.split,b.split=(8,8),(16,8)
    gen=torch.Generator(device='cuda').manual_seed(993)
    x=(torch.randn(rows,a.k,device='cuda',generator=gen)*factor).bfloat16()
    output=[]
    for ext,mode in ((baseline,7),(probe,9)):
        monkeypatch.setattr(linear,'_ext',lambda:ext);a.kernel_mode=b.kernel_mode=mode
        out=torch.full((rows,a.n+(b.n if pair else 0)+128),7.,device='cuda',dtype=torch.float32)
        wa,wb=a.workspace('cuda',rows),b.workspace('cuda',rows)
        def run():
            if pair:linear.pair_call(a,b,x,out[:,:a.n],out[:,a.n:a.n+b.n],wa,wb)
            else:
                xh,z,c=wa;a(x,out=out[:,:a.n],xh=xh,z=z,counters=c)
        run();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):run()
        output.append((out,wa,wb,g))
    for _ in range(3):
        for out,wa,wb,g in output:g.replay()
        ref,ra,rb,_=output[0];got,ga,gb,_=output[1]
        assert torch.equal(got.view(torch.int32),ref.view(torch.int32))
        for w,z in ((ga,ra),(gb,rb)) if pair else ((ga,ra),):
            assert torch.equal(w[1].view(torch.int32),z[1].view(torch.int32))
            assert w[2].count_nonzero().item()==0
        assert torch.all(got[:,-128:]==7)
        x[:,0].add_(.25)
