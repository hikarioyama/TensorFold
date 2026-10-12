"""Physical bucket bounds and logical sampler rows are independent of dummy padding."""
from types import SimpleNamespace
import pytest
import torch
from tensorfold.families.glm5_next.cuda import segmented as seg


def fixture(rows=32):
    w=SimpleNamespace(cfg=SimpleNamespace(full=True,dense_limit=2048),device=torch.device('cpu'))
    b=SimpleNamespace(rows=rows,valid=None,ids_host=torch.empty(rows,dtype=torch.int64),
                      ids=torch.empty(rows,dtype=torch.int64),
                      staged=SimpleNamespace(synchronize=lambda:None,record=lambda:None))
    st=lambda pos=7,cap=64:SimpleNamespace(pos=pos,capacity=cap,index={})
    return w,b,st


@pytest.mark.parametrize('lengths,width', [([1,1],1),([2,1,3],3),([8,2,6,7],8),([7],8)])
def test_physical_bucket_excludes_dummy_sampler_rows(lengths,width):
    w,b,st=fixture();windows=[(st(),[i+1]*n) for i,n in enumerate(lengths)]
    physical=seg.stage_streams(w,b,windows,padded=True)
    assert [(lo,z) for _,lo,z in physical]==[(i*width,(i+1)*width) for i in range(len(lengths))]
    logical=seg.logical_segments(windows,physical)
    assert [(lo,z) for _,lo,z in logical]==[(i*width,i*width+n) for i,n in enumerate(lengths)]
    want=[v for i,n in enumerate(lengths) for v in [1]*n+[0]*(width-n)]
    assert b.valid[:len(want)].tolist()==want
    logits=torch.arange(len(want)*4).reshape(len(want),4)
    compact=seg.sampling_logits(logits,physical,logical)
    assert torch.equal(compact,logits[torch.tensor(want,dtype=torch.bool)])
    seg.stage_streams(w,b,windows)
    assert b.valid[:sum(lengths)].tolist()==[1]*sum(lengths)


@pytest.mark.parametrize('reason',['cache','rows','large'])
def test_never_expand_past_existing_physical_capacity(reason):
    w,b,st=fixture(3 if reason=='rows' else 32)
    windows=[(st(62,64),[1,2])] if reason=='cache' else ([(st(),[1,2]),(st(),[3])] if reason=='rows' else [(st(),list(range(9)))])
    physical=seg.stage_streams(w,b,windows,padded=True)
    assert [z-lo for _,lo,z in physical]==[len(t) for _,t in windows]
    assert b.valid is None


def test_logical_state_identity_and_length_are_checked():
    _,_,st=fixture();a,b=st(),st()
    with pytest.raises(ValueError):seg.logical_segments([(a,[1])],[(b,0,1)])
    with pytest.raises(ValueError):seg.logical_segments([(a,[1,2])],[(a,0,1)])
    with pytest.raises(ValueError):seg.logical_segments([(a,[1])],[])
