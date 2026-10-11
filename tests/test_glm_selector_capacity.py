"""CPU dispatch regressions: shared scratch must never enlarge a private key scan."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import forward as fw, sparse53


class Launches:
    def __init__(self, kernel):
        self.names = kernel.arg_names
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kw):
            call = dict(zip(self.names, args))
            call.update(kw)
            self.calls.append((grid, call))
        return launch


@pytest.fixture
def launches(monkeypatch):
    result = {}
    for name in ("_sel_scores", "_select_rows", "_counts"):
        result[name] = Launches(getattr(sparse53, name))
        monkeypatch.setattr(sparse53, name, result[name])
    return result


@pytest.mark.parametrize("capacity", [2112, 2560])
@pytest.mark.parametrize("rows", [1, 2, 3, 128])
@pytest.mark.parametrize("bucket", [None, 1024, 4096, 8192])
def test_private_scan_shared_score_stride(launches, capacity, rows, bucket):
    qi = torch.empty(rows, 32, 128, dtype=torch.bfloat16)
    wts = torch.empty(rows, 32, dtype=torch.float32)
    ik = torch.empty(capacity, 128, dtype=torch.float16)
    # Offset view, more than one slab, noncompact rows, sentinel outside the scan.
    scores = torch.empty(7, 6152 + 13)[:, 5:6157]
    sparse53.select_tokens(qi, wts, ik, torch.tensor([2048]), rows, 2048, 6152,
                           bucket=bucket, scores=scores)
    expected = capacity if bucket is None else min(capacity, max(bucket, 2048))
    producers = launches["_sel_scores"].calls
    consumers = launches["_select_rows"].calls
    assert len(producers) == len(consumers) == (rows + 6) // 7
    for (_, p), (_, c) in zip(producers, consumers):
        assert p["NP"] == c["NP"] == expected <= capacity
        assert p["score_stride"] == c["score_stride"] == scores.stride(0)
        assert p["OUT"].data_ptr() == c["S"].data_ptr() == scores.data_ptr()
    assert len(launches["_counts"].calls) == 1


@pytest.mark.parametrize("kind", ["short", "empty", "columns", "dtype"])
def test_invalid_scores_rejected_before_launch(launches, kind):
    qi = torch.empty(3, 32, 128, dtype=torch.bfloat16)
    ik = torch.empty(2560, 128, dtype=torch.float16)
    scores = {"short": torch.empty(3, 2048), "empty": torch.empty(0, 6144),
              "columns": torch.empty(3, 12288)[:, ::2],
              "dtype": torch.empty(3, 6144, dtype=torch.float16)}[kind]
    with pytest.raises(ValueError, match="scores"):
        sparse53.select_tokens(qi, torch.empty(3, 32), ik, torch.tensor([2048]), 3, 2048, 6144,
                               scores=scores)
    assert all(not x.calls for x in launches.values())


@pytest.mark.parametrize("layer_id", [0, 78])
@pytest.mark.parametrize("rows", [1, 2, 3, 128])
def test_family_owner_passes_private_capacity(monkeypatch, layer_id, rows):
    calls = []
    def select(*args, **kw):
        calls.append((args, kw))
        return kw["tokens"], kw["counts"]
    monkeypatch.setattr(sparse53, "select_tokens", select)
    b = SimpleNamespace(sel_np=6152, qi=torch.empty(rows, 4096, dtype=torch.bfloat16),
                        igr=torch.empty(rows, 32), sel_score=torch.empty(8, 6152),
                        sel_tok=torch.empty(rows, 2048, dtype=torch.int32),
                        sel_cnt=torch.zeros(rows, dtype=torch.int32))
    w = SimpleNamespace(cfg=SimpleNamespace(index_heads=32, index_dim=128, index_topk=2048, qk_rope=0))
    keys = torch.empty(2560, 128, dtype=torch.float16)
    fw._selection(SimpleNamespace(index=object()), w, SimpleNamespace(index=layer_id), (keys, None),
                  b, rows, torch.tensor([2048]), None, prepared=True)
    assert calls[0][0][5] == 2048
    assert calls[0][0][6] == 2560
    assert calls[0][1]["scores"] is b.sel_score
    assert b.sel_np == 6152 and b.sel_owner == layer_id


@pytest.mark.parametrize("pos,rows", [(2047, 1), (2047, 2), (2048, 3), (2049, 128)])
def test_multi_prefill_and_absorb_dispatch(launches, monkeypatch, pos, rows):
    from tensorfold.families.glm5_next.cuda import decode as dec, multi
    b = SimpleNamespace(rows=rows, sel_np=6152, qi=torch.empty(rows, 4096, dtype=torch.bfloat16),
                        igr=torch.empty(rows, 32), sel_score=torch.empty(7, 6152),
                        sel_tok=torch.empty(rows, 2048, dtype=torch.int32),
                        sel_cnt=torch.zeros(rows, dtype=torch.int32), fnormed=torch.zeros(rows, 8))
    w = SimpleNamespace(cfg=SimpleNamespace(index_heads=32, index_dim=128, index_topk=2048, qk_rope=0))
    st = SimpleNamespace(pos=pos, mtp_len=pos, capacity=2560)
    st.set_mtp_len = lambda n: setattr(st, "mtp_len", n)
    main_keys = torch.empty(2560, 128, dtype=torch.float16)
    mtp_keys = torch.empty_like(main_keys)
    def select(keys, layer_id, count, position):
        fw._selection(SimpleNamespace(index=object()), w, SimpleNamespace(index=layer_id), (keys, None),
                      b, count, torch.tensor([position]), None, prepared=True)
        return torch.zeros(1, 8)
    monkeypatch.setattr(fw, "stage", lambda w, st, b, tokens: len(tokens))
    monkeypatch.setattr(fw, "chunks_for", lambda *args: 1)
    monkeypatch.setattr(fw, "compute", lambda w, st, b, count, **kw: select(main_keys, 0, count, st.pos))
    monkeypatch.setattr(dec, "mtp_forward", lambda w, st, b, tokens, hidden:
                        select(mtp_keys, 78, len(tokens), st.mtp_len))
    monkeypatch.setattr(fw, "commit", lambda w, st, b, count, keep: setattr(st, "pos", st.pos + keep))
    monkeypatch.setattr(multi, "next_fill", lambda filling: filling[0])
    s = SimpleNamespace(st=st, prompt=[12] * (pos + rows + 8), draft=True,
                        snap=SimpleNamespace(policy=2), prefill_s=0.)
    decoder = SimpleNamespace(w=w, e=SimpleNamespace(pbuf=b), filling=[s], cache=SimpleNamespace(keep=0))
    multi.MultiDecoder._fill(decoder)
    calls = [call for _, call in launches["_sel_scores"].calls]
    assert calls and all(call["NP"] == 2560 and call["score_stride"] == 6152 for call in calls)
    assert {call["IK"].data_ptr() for call in calls} == {main_keys.data_ptr(), mtp_keys.data_ptr()}
    assert st.pos == st.mtp_len == pos + rows
    assert b.sel_np == 6152
