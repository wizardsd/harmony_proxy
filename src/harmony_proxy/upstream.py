from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx


log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    pass


@dataclass
class UpstreamSSEEvent:
    event: str
    data: str


class UpstreamClient:
    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=None,
        )
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._read_timeout = read_timeout

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat_completions(self, payload: Dict[str, Any]) -> AsyncIterator[UpstreamSSEEvent]:
        url = "/chat/completions"
        async with self._client.stream("POST", url, json=payload) as response:
            await self._raise_for_status(response)
            async for event in _iter_sse_events(response, self._read_timeout):
                yield event

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post("/chat/completions", json=payload)
        await self._raise_for_status(response)
        return response.json()

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": response.text}
        message = payload.get("error") or payload.get("message") or response.text
        raise UpstreamError(f"Upstream error {response.status_code}: {message}")


async def _iter_sse_events(
    response: httpx.Response,
    read_timeout: float,
) -> AsyncIterator[UpstreamSSEEvent]:
    data_lines: list[str] = []
    event_name = "message"
    iterator = response.aiter_lines()
    while True:
        try:
            raw_line = await asyncio.wait_for(iterator.__anext__(), timeout=read_timeout)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            raise UpstreamError("Timed out while reading from upstream SSE stream.")

        if raw_line is None:
            continue

        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield UpstreamSSEEvent(event=event_name, data="\n".join(data_lines))
                data_lines = []
                event_name = "message"
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
            continue
        data_lines.append(line)

    if data_lines:
        yield UpstreamSSEEvent(event=event_name, data="\n".join(data_lines))
