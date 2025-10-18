from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from . import metrics


log = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


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
        max_retries: int,
        retry_backoff_seconds: float,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=None,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )
        self._read_timeout = read_timeout
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat_completions(self, payload: Dict[str, Any]) -> AsyncIterator[UpstreamSSEEvent]:
        url = "/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            had_events = False
            try:
                async with self._client.stream("POST", url, json=payload) as response:
                    await self._raise_for_status(response)
                    async for event in _iter_sse_events(response, self._read_timeout):
                        had_events = True
                        yield event
                return
            except (UpstreamError, httpx.HTTPError) as exc:
                last_error = exc
                if not self._should_retry(exc, attempt, had_events=had_events):
                    raise self._normalize_error(exc) from exc
                await self._sleep_before_retry(attempt, exc)
                continue
        if last_error is not None:
            raise self._normalize_error(last_error) from last_error

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                await self._raise_for_status(response)
                return response.json()
            except (UpstreamError, httpx.HTTPError) as exc:
                last_error = exc
                if not self._should_retry(exc, attempt, had_events=False):
                    raise self._normalize_error(exc) from exc
                await self._sleep_before_retry(attempt, exc)
                continue
        if last_error is not None:
            raise self._normalize_error(last_error) from last_error
        raise UpstreamError("Upstream returned no response", retryable=False)

    async def check_readiness(self) -> bool:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                response = await self._client.get("/models")
            except httpx.HTTPError as exc:
                last_error = exc
                if not self._should_retry(exc, attempt, had_events=False):
                    raise self._normalize_error(exc) from exc
                await self._sleep_before_retry(attempt, exc)
                continue

            status = response.status_code
            if status == 200:
                return True
            if status in {401, 403}:
                # Upstream is reachable but credentials are invalid; consider it ready.
                return True
            if status >= 500:
                err = UpstreamError(
                    f"Upstream readiness check failed with status {status}",
                    status_code=status,
                    retryable=True,
                )
                last_error = err
                if not self._should_retry(err, attempt, had_events=False):
                    raise err
                await self._sleep_before_retry(attempt, err)
                continue
            if status == 404:
                # Endpoint missing but server reachable; treat as not ready.
                return False
            return False

        if last_error is not None:
            raise self._normalize_error(last_error) from last_error
        return False

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"error": response.text}
        message = payload.get("error") or payload.get("message") or response.text
        status = response.status_code
        retryable = status in {408, 429, 500, 502, 503, 504}
        raise UpstreamError(
            f"Upstream error {status}: {message}",
            status_code=status,
            retryable=retryable,
        )

    def _should_retry(
        self,
        exc: Exception,
        attempt: int,
        *,
        had_events: bool,
    ) -> bool:
        if had_events:
            return False
        if attempt > self._max_retries:
            return False
        if isinstance(exc, UpstreamError):
            return bool(exc.retryable)
        if isinstance(exc, httpx.HTTPError):
            return True
        return False

    async def _sleep_before_retry(self, attempt: int, exc: Exception) -> None:
        delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
        metrics.observe_retry()
        log.warning(
            "Retrying upstream request (attempt %s/%s, delay %.2fs): %s",
            attempt,
            self._max_retries + 1,
            delay,
            exc,
        )
        if delay > 0:
            await asyncio.sleep(delay)

    def _normalize_error(self, exc: Exception) -> UpstreamError:
        if isinstance(exc, UpstreamError):
            return exc
        if isinstance(exc, httpx.HTTPError):
            return UpstreamError(
                f"Transport error contacting upstream: {exc}",
                retryable=isinstance(exc, httpx.TimeoutException),
            )
        return UpstreamError(str(exc))


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
            raise UpstreamError(
                "Timed out while reading from upstream SSE stream.",
                retryable=True,
            )

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
