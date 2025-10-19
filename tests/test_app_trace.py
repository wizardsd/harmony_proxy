import asyncio
import json

from harmony_proxy import app as app_module, trace
from harmony_proxy.config import ProxyConfig, ProxyMode
from harmony_proxy.upstream import UpstreamSSEEvent


class DummyRequest:
    async def is_disconnected(self) -> bool:
        return False


class DummyUpstream:
    def __init__(self, events):
        self._events = events

    async def stream_chat_completions(self, payload, trace_id=None):
        for event in self._events:
            yield event


def test_streaming_response_logs_proxy_events(tmp_path):
    harmony_payload = (
        "<|start|>assistant"
        "<|channel|>final"
        "<|message|>Trace output"
        "<|end|>"
    )
    events = [
        UpstreamSSEEvent(
            event="message",
            data=json.dumps({"choices": [{"delta": {"content": harmony_payload}}]}),
        ),
        UpstreamSSEEvent(event="message", data="[DONE]"),
    ]

    log_path = tmp_path / "proxy_trace.jsonl"
    trace.configure(str(log_path))

    original_config = getattr(app_module.app.state, "config", None)
    original_upstream = getattr(app_module.app.state, "upstream", None)

    config = ProxyConfig()
    config.proxy_mode = ProxyMode.FINAL_ONLY
    config.no_final_timeout = 0.0
    app_module.app.state.config = config
    app_module.app.state.upstream = DummyUpstream(events)

    payload = {"model": "dummy-model", "stream": True}
    request = DummyRequest()

    try:
        async def consume():
            chunks = []
            async for chunk in app_module._streaming_response(payload, request, "trace-app"):
                chunks.append(chunk)
            return chunks

        emitted = asyncio.run(consume())
    finally:
        if original_config is not None:
            app_module.app.state.config = original_config
        if original_upstream is not None:
            app_module.app.state.upstream = original_upstream

    assert emitted, "expected proxy to emit SSE chunks"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    related = [entry for entry in lines if entry["request_id"] == "trace-app"]
    events_logged = {entry["event"] for entry in related}
    assert "proxy_stream_start" in events_logged
    proxy_events = [entry for entry in related if entry["event"] == "proxy_event"]
    assert proxy_events, "expected proxy_event entries"
    first_payload = proxy_events[0]["payload"]
    assert first_payload["id"] == "trace-app"
