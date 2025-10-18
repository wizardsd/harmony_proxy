import asyncio

import httpx
import pytest

from harmony_proxy import upstream as upstream_module
from harmony_proxy.upstream import UpstreamClient, UpstreamError


async def _collect_stream(client: UpstreamClient, payload):
    events = []
    async for event in client.stream_chat_completions(payload):
        events.append(event)
    return events


def test_chat_completions_retries_on_502():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )

    result = asyncio.run(client.chat_completions({"messages": []}))
    asyncio.run(client.aclose())

    assert calls["count"] == 2
    assert result["choices"][0]["message"]["content"] == "ok"


def test_stream_chat_completions_retries_before_data():
    calls = {"count": 0}
    sse_body = (
        "event: message\n"
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        "\n"
        "event: message\n"
        "data: [DONE]\n"
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )

    events = asyncio.run(_collect_stream(client, {"messages": [], "stream": True}))
    asyncio.run(client.aclose())

    assert calls["count"] == 2
    assert events[0].data.startswith('{"choices"')
    assert events[-1].data == "[DONE]"


def test_stream_does_not_retry_after_events(monkeypatch):
    original_iter = upstream_module._iter_sse_events

    async def failing_iter(response, read_timeout):
        agen = original_iter(response, read_timeout)
        try:
            first = await agen.__anext__()
        except StopAsyncIteration:  # pragma: no cover - defensive
            return
        yield first
        raise UpstreamError("boom", retryable=True)

    monkeypatch.setattr("harmony_proxy.upstream._iter_sse_events", failing_iter)

    sse_body = (
        "event: message\n"
        'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=3,
        retry_backoff_seconds=0,
        transport=transport,
    )

    async def consume():
        events = []
        error_raised = False
        try:
            async for event in client.stream_chat_completions({"stream": True}):
                events.append(event)
        except UpstreamError:
            error_raised = True
        return events, error_raised

    collected, failed = asyncio.run(consume())
    asyncio.run(client.aclose())

    assert collected, "expected at least one event before failure"
    assert failed, "expected failure propagated without retry after events"


def test_check_readiness_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )
    result = asyncio.run(client.check_readiness())
    asyncio.run(client.aclose())
    assert result is True


def test_check_readiness_not_ready_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )
    result = asyncio.run(client.check_readiness())
    asyncio.run(client.aclose())
    assert result is False


def test_check_readiness_retries_on_503():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=2,
        retry_backoff_seconds=0,
        transport=transport,
    )
    result = asyncio.run(client.check_readiness())
    asyncio.run(client.aclose())
    assert result is True
    assert calls["count"] == 2


def test_check_readiness_raises_after_exhausting_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    transport = httpx.MockTransport(handler)
    client = UpstreamClient(
        base_url="http://upstream.test",
        connect_timeout=0.1,
        read_timeout=0.1,
        max_retries=1,
        retry_backoff_seconds=0,
        transport=transport,
    )

    def run():
        return asyncio.run(client.check_readiness())

    try:
        with pytest.raises(UpstreamError):
            run()
    finally:
        asyncio.run(client.aclose())
