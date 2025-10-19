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


def test_streaming_tool_call_emits_role_delta():
    harmony_tool_call = (
        "<|start|>assistant"
        "<|channel|>commentary to=read_file"
        "<|message|>{\"path\":\"workspace/trace.log\"}"
        "<|call|>"
        "<|end|>"
    )
    events = [
        UpstreamSSEEvent(
            event="message",
            data=json.dumps(
                {
                    "id": "call-only",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": harmony_tool_call},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
        ),
        UpstreamSSEEvent(
            event="message",
            data=json.dumps(
                {
                    "id": "call-only",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
        ),
        UpstreamSSEEvent(event="message", data="[DONE]"),
    ]

    original_config = getattr(app_module.app.state, "config", None)
    original_upstream = getattr(app_module.app.state, "upstream", None)

    config = ProxyConfig()
    config.proxy_mode = ProxyMode.OPENAI_TOOL_CALLS
    config.no_final_timeout = 0.0
    app_module.app.state.config = config
    app_module.app.state.upstream = DummyUpstream(events)

    payload = {"model": "dummy-model", "stream": True}
    request = DummyRequest()

    try:
        async def consume():
            chunks = []
            async for chunk in app_module._streaming_response(payload, request, "tool-role"):
                chunks.append(chunk)
            return chunks

        emitted = asyncio.run(consume())
    finally:
        if original_config is not None:
            app_module.app.state.config = original_config
        if original_upstream is not None:
            app_module.app.state.upstream = original_upstream

    assert emitted, "expected SSE output from streaming pipeline"

    def _parse_chunk(raw: bytes):
        text = raw.decode("utf-8").strip()
        if not text:
            return None
        lines = text.splitlines()
        data_lines = [line for line in lines if line.startswith("data: ")]
        if not data_lines:
            return None
        data = data_lines[0][len("data: ") :]
        if data == "[DONE]":
            return None
        return json.loads(data)

    payloads = [parsed for chunk in emitted if (parsed := _parse_chunk(chunk)) is not None]
    assert payloads, "expected at least one JSON SSE payload"

    first_delta = payloads[0]["choices"][0]["delta"]
    assert first_delta.get("role") == "assistant", "stream must announce assistant role before tool call"

    tool_deltas = [
        choice["delta"]
        for payload in payloads
        for choice in payload["choices"]
        if choice.get("delta", {}).get("tool_calls")
    ]
    assert tool_deltas, "expected tool_calls delta"
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "read_file"

    finish_reasons = [
        choice["finish_reason"]
        for payload in payloads
        for choice in payload["choices"]
        if choice.get("finish_reason")
    ]
    assert finish_reasons and finish_reasons[0] == "tool_calls"
