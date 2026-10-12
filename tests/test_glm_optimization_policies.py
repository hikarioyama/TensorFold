"""CPU dispatch and graph-namespace checks; unknown shapes retain the baseline."""
import pytest,torch
from tensorfold.cuda.exl3.tiles import tile_capacity
from tensorfold.families.glm5_next.cuda import latent,engine,forward,sparse53

@pytest.mark.parametrize('picks,experts,expected',[(0,256,1),(1,256,2),(16384,256,1280),(18432,256,1408)])
def test_integer_capacity(picks,experts,expected):assert tile_capacity(picks,experts)==expected

@pytest.mark.parametrize('picks,experts',[(-1,256),(2,0),(2**36,256)])
def test_capacity_rejects_overflow(picks,experts):
    with pytest.raises(ValueError):tile_capacity(picks,experts)

@pytest.mark.parametrize('field,value',[('rows',4),('heads',8),('width',256),('rope',0),('dtype',torch.bfloat16),('selections',1024),('bucket',None),('bucket',12345)])
def test_unknown_decode_shape_falls_back(field,value,monkeypatch):
    monkeypatch.setattr(latent,'MLA_POLICY','decode443');args=dict(rows=2,heads=16,width=512,rope=64,dtype=torch.uint8,selections=2048,bucket=32768)
    assert latent.sparse_launch(**args)==(4,4,3)
    args[field]=value;assert latent.sparse_launch(**args)[1:]==(8,1)

@pytest.mark.parametrize('module,field,value',[(latent,'MLA_POLICY','decode443'),(latent,'PREFILL_MLA_POLICY','prefill1643'),(latent,'DENSE_POLICY','decode443'),(forward,'COMPACT_TILES',True),(sparse53,'PREFILL_SELECTOR','parallel')])
def test_variant_has_own_rank_namespace(module,field,value,monkeypatch):
    monkeypatch.setattr(module,field,'legacy' if isinstance(value,str) else False)
    before=engine.kernel_namespace('test-runtime');monkeypatch.setattr(module,field,value)
    after=engine.kernel_namespace('test-runtime');assert before!=after;assert after==engine.kernel_namespace('test-runtime')

@pytest.mark.parametrize('rows,mode',[(1,7),(2,7),(16,7),(17,7),(31,7),(32,9),(64,9),(127,7),(128,9)])
def test_g2_validated_rows_only(rows,mode,monkeypatch):
    from types import SimpleNamespace
    from tensorfold.cuda.exl3 import linear
    monkeypatch.setattr(linear,'PREFILL_G2',True)
    layer=SimpleNamespace(bits=5,codebook='mul1',kernel_mode=7)
    assert linear.launch_mode(layer,rows)==mode

@pytest.mark.parametrize('bits,cb,kernel_mode',[(3,'mul1',7),(5,'mcg',7),(5,'mul1',1)])
def test_g2_unverified_weight_falls_back(bits,cb,kernel_mode,monkeypatch):
    from types import SimpleNamespace
    from tensorfold.cuda.exl3 import linear
    monkeypatch.setattr(linear,'PREFILL_G2',True)
    assert linear.launch_mode(SimpleNamespace(bits=bits,codebook=cb,kernel_mode=kernel_mode),128)==kernel_mode

def test_g2_has_distinct_namespace(monkeypatch):
    from tensorfold.cuda.exl3 import linear
    monkeypatch.setattr(linear,'PREFILL_G2',False);before=engine.kernel_namespace('test-runtime')
    monkeypatch.setattr(linear,'PREFILL_G2',True);assert before!=engine.kernel_namespace('test-runtime')
