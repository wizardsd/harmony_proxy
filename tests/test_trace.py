import json

from harmony_proxy import trace


def test_trace_record_writes_json(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace.configure(str(path))
    trace.record(
        "proxy_request",
        request_id="test-123",
        endpoint="chat_completions",
        payload={"messages": [{"role": "user", "content": "Hi"}]},
        stream=False,
    )

    contents = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    entry = json.loads(contents[0])
    assert entry["event"] == "proxy_request"
    assert entry["request_id"] == "test-123"
    assert entry["endpoint"] == "chat_completions"
    assert entry["payload"]["messages"][0]["content"] == "Hi"
    assert entry["stream"] is False
    assert "timestamp" in entry
    trace.reset_for_test()


def test_trace_noop_without_configuration(tmp_path):
    trace.reset_for_test()
    path = tmp_path / "trace.jsonl"
    trace.record(
        "proxy_request",
        request_id="test-456",
        endpoint="chat_completions",
        payload={"foo": "bar"},
        stream=False,
    )
    assert not path.exists()
    assert trace.current_path() is None


def test_trace_compacts_strings(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace.configure(str(path), max_string_length=10)
    trace.record(
        "proxy_response",
        request_id="test-789",
        endpoint="chat_completions",
        payload={"message": "Hello world, this is a long string."},
        stream=True,
    )

    contents = path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(contents[0])
    compacted = entry["payload"]["message"]
    assert compacted.startswith("Hello worl")
    assert compacted.endswith("chars)")
    assert "+" in compacted
    trace.reset_for_test()
