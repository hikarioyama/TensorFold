"""Experimental proposal choices must not mutate target sampling or exact acceptance."""
from types import SimpleNamespace
import pytest
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode,engine,multi


def test_native_policy_encoding_and_legacy_defaults():
    assert engine.encode_policy('t7:0.1:0.8')==[6,7,100000,800000]
    assert engine.encode_policy('g7:0.7:0.16')==[7,7,700000,160000]
    t=engine.decode_policy(engine.encode_policy('t7:0.1:0.8'))
    assert t.most==7 and t.confidence==.1 and t.temperature_scale==.8
    assert t.next(7,0)==7
    g=engine.decode_policy(engine.encode_policy('g7:0.7:0.16'))
    assert isinstance(g,decode.GainDepthPolicy)
    assert engine.encode_policy('c7:0.1')==[3,7,100000,0]
    assert engine.decode_policy([3,7,100000,0]).temperature_scale==1.


@pytest.mark.parametrize('spec',['t0:.1:1','t8:.1:1','t7:.1:0','t7:.1:nan','t7:.1:inf',
                                't7:1.1:1','g7:1:.1','g7:.7:0','g7:nan:1','g7:.7:inf',
                                'ft7:.1:1','fg7:.7:.1'])
def test_invalid_or_external_policy_refused(spec):
    with pytest.raises(ValueError):engine.encode_policy(spec)


@pytest.mark.parametrize('full,mtp',[(False,object()),(True,None)])
def test_experimental_policies_require_full_native_mtp(full,mtp):
    e=SimpleNamespace(full=full,w=SimpleNamespace(mtp=mtp))
    with pytest.raises(ValueError,match='full GLM native MTP'):
        engine.GlmEngine._effective(e,[6,7,100000,800000])


def test_draft_temperature_keeps_target_object_and_keys():
    target=Sampling(2**64-1,1.,0,.95)
    s=SimpleNamespace(sampling=target,snap=SimpleNamespace(policy=decode.DepthPolicy(7,temperature_scale=.8)))
    draft=multi.proposal_sampling(s)
    assert draft is not target and draft.temperature==.8
    assert (draft.seed,draft.top_k,draft.top_p,draft.min_p)==(target.seed,target.top_k,target.top_p,target.min_p)
    assert target.temperature==1.
    s.snap.policy.temperature_scale=1.
    assert multi.proposal_sampling(s) is target
    s.sampling=None;s.snap.policy.temperature_scale=.8
    assert multi.proposal_sampling(s) is None


def test_censored_rejection_counts_only_observed_failure():
    p=decode.GainDepthPolicy(7,decay=0.,marginal_cost=.16)
    p.next(7,2)
    assert p.successes==2 and p.trials==3  # NOT two successes and five failures
    for _ in range(10):k=p.next(7,7)
    assert k==7
    for _ in range(10):k=p.next(7,0)
    assert k==1
    with pytest.raises(ValueError):p.next(1,2)


def test_gain_depth_is_deterministic_and_bounded():
    a=decode.GainDepthPolicy(7);b=decode.GainDepthPolicy(7)
    for d,n in [(0,0),(7,7),(7,2),(2,0),(1,1),(4,4)]*8:
        assert 1 <= a.next(d,n)==b.next(d,n) <= 7
