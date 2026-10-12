from tensorfold.families.glm5_next.cuda import engine, sparse53


def test_selector_workspace_is_opt_in(monkeypatch):
    monkeypatch.setattr(sparse53,'SELECTOR_MODE','legacy')
    assert sparse53.selection_scratch(8,229376,'cpu') is None
    monkeypatch.setattr(sparse53,'SELECTOR_MODE','parallel')
    scratch=sparse53.selection_scratch(32,229376,'cpu')
    assert scratch.rows==8 and scratch.capacity==229376
    assert scratch.hist.numel()*4 + scratch.state.numel()*4 < 2*1024**2


def test_selector_mode_changes_rank_namespace(monkeypatch):
    monkeypatch.setattr(sparse53,'SELECTOR_MODE','legacy')
    before=engine.kernel_namespace('selector-test')
    monkeypatch.setattr(sparse53,'SELECTOR_MODE','parallel')
    assert engine.kernel_namespace('selector-test')!=before
