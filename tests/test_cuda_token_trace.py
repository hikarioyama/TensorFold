"""Opt-in HTTP provenance: original IDs are distinct from visible text and hidden markup."""
import pytest

from tests.test_cuda_stop_strings import END_ID, Engine, body, events, http_server, ids, make_app, post


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_token_trace_tracks_original_ids_and_visible_channels(tmp_path, chat, draft):
    text = "考える🙂</think>\n\nAnswer STOP not delivered"
    reply = ids(text) + [END_ID]
    app = make_app(tmp_path, Engine(reply))
    fields = {"draft": draft, "return_token_ids": True, "stop": "STOP",
              "chat_template_kwargs": {"enable_thinking": True}}
    with http_server(app) as port:
        status, wire = post(port, body(chat, True, **fields), chat)
    assert status == 200
    cumulative, reasoning, content, traces = [], "", "", []
    for event in events(wire):
        for choice in event["choices"]:
            delta = choice.get("delta", {})
            assert "_token_trace" not in delta
            reasoning += delta.get("reasoning_content", "")
            content += delta.get("content", "") if chat else choice.get("text", "")
        block = event.get("tensorfold", {})
        if "token_trace" not in block:
            continue
        trace = block["token_trace"]
        assert trace["start"] == len(cumulative)
        cumulative.extend(trace["ids"])
        assert trace["generated"] == len(cumulative)
        assert trace["visible_chars"] == {"reasoning": len(reasoning), "content": len(content)}
        assert trace["eos_ids"] == [END_ID] and trace["tools"] is False
        traces.append(trace)
    assert traces[-1]["final"] and traces[-1]["ids"] == []
    assert cumulative == reply[:len(ids(text[:text.index("STOP") + 4]))]
    assert "STOP" not in content
    if chat:
        assert reasoning == "考える🙂" and content == "Answer "
    else:
        assert reasoning == "" and content == text[:text.index("STOP")]
    # Deferred UTF-8 bytes and hidden markers are recorded but not mislabeled as visible IDs.
    assert any(t["decoded"] < t["generated"] for t in traces)


@pytest.mark.parametrize("requested", [False, True])
def test_token_trace_records_eos_but_does_not_make_it_visible(tmp_path, requested):
    reply = ids("hello") + [END_ID]
    app = make_app(tmp_path, Engine(reply))
    with http_server(app) as port:
        status, wire = post(port, body(False, True, return_token_ids=requested), False)
    chunks = events(wire)
    traces = [c["tensorfold"]["token_trace"] for c in chunks if "token_trace" in c.get("tensorfold", {})]
    assert status == 200
    assert bool(traces) is requested
    if requested:
        assert [t for trace in traces for t in trace["ids"]] == reply
        assert traces[-1]["decoded"] == len(reply) - 1
        assert traces[-1]["visible_chars"] == {"reasoning": 0, "content": 5}
    assert "".join(c["choices"][0].get("text", "") for c in chunks if c["choices"]) == "hello"
