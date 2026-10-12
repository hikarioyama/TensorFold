"""Certified singleton uses the original rule for ties, incomplete coverage and invalid keys."""
import numpy as np
import pytest
from tensorfold.cuda import sampling as smp
from tensorfold.engine.exact_sampling import Sampling


def got(tie=False,width=2):
    return (np.array([[[20.,20. if tie else 0.]]]),np.array([[[7,9]]]),
            np.array([[[2**40,2**40 if tie else 1]]],dtype=np.int64),
            np.array([[2*2**40 if tie else 2**40+1]],dtype=np.int64),np.array([[width]]))


@pytest.mark.parametrize('seed',[0,-1,2**64-1,2**100])
@pytest.mark.parametrize('position',[0,40960,2**64-1])
def test_exact_singleton_skips_only_pure_keyed_draw(monkeypatch,seed,position):
    s=Sampling(seed,1.,0,.95)
    monkeypatch.setattr(smp,'SINGLETON',False);want=smp._draw(got(width=64),[position],s)
    monkeypatch.setattr(smp,'SINGLETON',True)
    monkeypatch.setattr(smp,'uniform',lambda *a:pytest.fail('certified singleton must not need Gumbel scores'))
    assert smp._draw(got(width=64),[position],s)==want


def test_boundary_tie_still_requires_widening(monkeypatch):
    monkeypatch.setattr(smp,'SINGLETON',True)
    assert smp._draw(got(tie=True,width=64),[4],Sampling(1,1.,0,.1)) is None


@pytest.mark.parametrize('position',[-1,2**64])
def test_out_of_domain_position_still_fails(monkeypatch,position):
    monkeypatch.setattr(smp,'SINGLETON',True)
    with pytest.raises(OverflowError):smp._draw(got(),[position],Sampling(1,1.,0,.95))
