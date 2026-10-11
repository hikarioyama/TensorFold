from types import SimpleNamespace
import pytest
from tensorfold.cuda.exl3.experts import grouped_config
from tensorfold.families.glm5_next.cuda import forward as fw


@pytest.mark.parametrize('tile',['baseline','nt4pf2','nt4_at8'])
@pytest.mark.parametrize('rows',[1,2,3,4,5,6,7,8,9,12,16,24,32,128])
def test_effective_tile_preserves_k_and_warp_partitions(monkeypatch,tile,rows):
    monkeypatch.setattr(fw,'GROUPED_MODE','transpose');monkeypatch.setattr(fw,'EXPERT_TILE',tile)
    s=SimpleNamespace(cfg_gu=(8,4,4,1),cfg_d=(8,4,1,1),grouped_mode=3)
    fw.configure_expert_tiles(s,full=True)
    nt=4 if tile=='nt4pf2' or tile=='nt4_at8' and rows==8 else 8
    assert grouped_config(s,s.cfg_gu,rows)==(nt,4,4,2)
    assert grouped_config(s,s.cfg_d,rows)==(nt,4,1,2)


def test_other_family_does_not_enable_full_glm_override(monkeypatch):
    monkeypatch.setattr(fw,'GROUPED_MODE','transpose');monkeypatch.setattr(fw,'EXPERT_TILE','nt4pf2')
    s=SimpleNamespace(cfg_gu=(8,4,4,1),cfg_d=(8,4,1,1),grouped_mode=0)
    fw.configure_expert_tiles(s,full=False)
    assert grouped_config(s,s.cfg_gu,8)==(8,4,4,1)
