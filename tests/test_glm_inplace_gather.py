"""The producer and native AllGather share the current-R rank slice, not the capacity stride."""
from types import SimpleNamespace
import pytest
import torch
from tensorfold.families.glm5_next.cuda import forward as fw,engine,weights53


@pytest.mark.parametrize('rank',range(4))
def test_actual_r_offset_changes_without_stale_capacity_alias(monkeypatch,rank):
    monkeypatch.setattr(fw,'INPLACE_GATHER',True)
    b=SimpleNamespace(part=torch.full((32,16),-991.),gath=torch.full((4*32*16,),-777.),world=4)
    w=SimpleNamespace(cfg=SimpleNamespace(full=True),rank=rank,world=4)
    for rows in (1,3,8,12,32,2,0):
        b.gath.fill_(-777.)
        def gather(send,recv):
            if rows:assert send.data_ptr()==recv.data_ptr()+rank*rows*16*4
            assert torch.all(send==rank+1)
            for r in range(4):recv.view(4,rows,16)[r].fill_(r+1)
        w.comm=SimpleNamespace(all_gather=gather)
        part=fw.tp_partial(w,b,rows)
        assert part.shape==(rows,16) and part.is_contiguous()
        part.fill_(rank+1)
        got=fw.gather(w,b,rows)
        assert got.shape==(4,rows,16)
        for r in range(4):assert torch.all(got[r]==r+1)
        assert torch.all(b.gath[4*rows*16:]==-777.)
        assert torch.all(b.part==-991.)


@pytest.mark.parametrize('full,comm,enabled',[(True,None,True),(False,object(),True),(True,object(),False)])
def test_legacy_no_comm_and_other_family_paths_stay_separate(monkeypatch,full,comm,enabled):
    monkeypatch.setattr(fw,'INPLACE_GATHER',enabled)
    b=SimpleNamespace(part=torch.ones(8,16))
    w=SimpleNamespace(cfg=SimpleNamespace(full=full),comm=comm)
    assert fw.tp_partial(w,b,3).data_ptr()==b.part.data_ptr()


def test_tuner_binary_bytes_are_in_namespace(monkeypatch,tmp_path):
    p=tmp_path/'tuner.so';p.write_bytes(b'one')
    monkeypatch.setenv('NCCL_TUNER_PLUGIN',str(p))
    a=engine.kernel_namespace('test')
    p.write_bytes(b'two')
    assert a!=engine.kernel_namespace('test')
    assert weights53.LINEAR_MODES['transpose_narrow']==7
