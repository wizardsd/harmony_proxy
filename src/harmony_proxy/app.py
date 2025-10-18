from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Iterable, List

try:  # ensure starlette imports python_multipart without deprecation warning
    import python_multipart.multipart as _python_multipart

    sys.modules.setdefault("multipart", _python_multipart)
except ImportError:  # pragma: no cover - optional dependency
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import metrics
from .config import ProxyConfig, ProxyMode, load_config
from .harmony_fallback import extract_final_text
from .parser import FinalChunk, HarmonyStreamParser, ToolCallChunk
from .sse import format_sse_done, format_sse_event
from .upstream import UpstreamClient, UpstreamError


config: ProxyConfig = load_config()
logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))
logger = logging.getLogger("harmony_proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config
    config = load_config()
    app.state.config = config
    metrics.configure(config.metrics_enabled)
    app.state.upstream = UpstreamClient(
        base_url=config.upstream_base_url,
        connect_timeout=config.connect_timeout,
        read_timeout=config.read_timeout,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    try:
        yield
    finally:
        upstream: UpstreamClient = app.state.upstream
        await upstream.aclose()


app = FastAPI(title="Harmony Parser Proxy", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Response:
    config: ProxyConfig = app.state.config
    upstream: UpstreamClient = app.state.upstream
    try:
        ok = await upstream.check_readiness()
    except UpstreamError as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(status_code=503, detail="Upstream not ready")
    return JSONResponse({"status": "ready"})


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    config: ProxyConfig = app.state.config
    if not metrics.is_enabled() or not config.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    body = metrics.render_metrics()
    return Response(content=body, media_type=metrics.content_type())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await _read_json(request)
    _apply_stop_tokens(payload, config)

    stream = bool(payload.get("stream"))
    upstream: UpstreamClient = app.state.upstream
    metrics.observe_request("chat_completions", stream, config.proxy_mode.value)

    if stream:
        generator = _streaming_response(payload, request)
        headers = {"Content-Type": "text/event-stream"}
        return StreamingResponse(generator, headers=headers)

    try:
        upstream_result = await upstream.chat_completions(payload)
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    normalized = _normalize_non_streaming(upstream_result, config.proxy_mode)
    return JSONResponse(normalized)


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    payload = await _read_json(request)
    upstream: UpstreamClient = app.state.upstream
    metrics.observe_request("responses", bool(payload.get("stream")), config.proxy_mode.value)
    try:
        upstream_result = await upstream.chat_completions(payload)
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(upstream_result)


async def _streaming_response(payload: Dict[str, Any], request: Request):
    config: Any = app.state.config
    upstream: UpstreamClient = app.state.upstream

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = payload.get("model", "gpt-oss")
    parser = HarmonyStreamParser(
        encoding_name=config.harmony_encoding_name,
        prepend_missing_start=config.prepend_missing_start,
    )

    role_sent = False
    finish_sent = False
    tool_indices: Dict[str, int] = {}
    no_final_timeout = config.no_final_timeout
    final_output_emitted = False
    endpoint_name = "chat_completions"
    forwarded_final_chunks = 0
    forwarded_tool_chunks = 0
    fallback_used = False

    def build_delta(delta: Dict[str, Any]) -> bytes:
        body = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }
            ],
        }
        return format_sse_event(body)

    def build_finish(reason: str = "stop") -> bytes:
        nonlocal finish_sent
        if finish_sent:
            return b""
        finish_sent = True
        body = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": reason,
                }
            ],
        }
        return format_sse_event(body)

    last_final_seen = time.monotonic()

    try:
        async for event in upstream.stream_chat_completions(payload):
            if await request.is_disconnected():
                logger.info("Client disconnected; aborting stream")
                break

            data = event.data.strip()
            if data == "[DONE]":
                break
            try:
                payload_json = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON SSE chunk: %s", data)
                continue

            choices = payload_json.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    for chunk in parser.feed(content):
                        last_final_seen = time.monotonic()
                        if isinstance(chunk, FinalChunk):
                            if not role_sent:
                                role_sent = True
                                yield build_delta({"role": "assistant"})
                            if chunk.text:
                                final_output_emitted = True
                                forwarded_final_chunks += 1
                                yield build_delta({"content": chunk.text})
                                metrics.observe_final_chunk(endpoint_name, True)
                            if chunk.is_terminal:
                                finish_chunk = build_finish("stop")
                                if finish_chunk:
                                    yield finish_chunk
                        elif isinstance(chunk, ToolCallChunk):
                            tool_outputs = _emit_tool_chunk(
                                chunk,
                                config.proxy_mode,
                                build_delta,
                                tool_indices,
                                endpoint_name,
                            )
                            if tool_outputs:
                                forwarded_tool_chunks += len(tool_outputs)
                                for out_chunk in tool_outputs:
                                    yield out_chunk
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    finish_chunk = build_finish(str(finish_reason))
                    if finish_chunk:
                        yield finish_chunk

            if no_final_timeout:
                elapsed = time.monotonic() - last_final_seen
                if elapsed > no_final_timeout:
                    logger.warning(
                        "Upstream stream stalled without final output for %.2fs", elapsed
                    )
                    finish_chunk = build_finish("timeout")
                    if finish_chunk:
                        yield finish_chunk
                    break
    except UpstreamError as exc:
        logger.exception("Error while streaming from upstream: %s", exc)
        error_payload = {
            "error": {
                "type": "upstream_error",
                "message": str(exc),
            }
        }
        yield format_sse_event(error_payload)
    finally:
        for chunk in parser.close():
            if isinstance(chunk, FinalChunk):
                if chunk.text:
                    final_output_emitted = True
                    forwarded_final_chunks += 1
                    yield build_delta({"content": chunk.text})
                    metrics.observe_final_chunk(endpoint_name, True)
                if chunk.is_terminal:
                    finish_chunk = build_finish("stop")
                    if finish_chunk:
                        yield finish_chunk
            elif isinstance(chunk, ToolCallChunk):
                tool_outputs = _emit_tool_chunk(
                    chunk,
                    config.proxy_mode,
                    build_delta,
                    tool_indices,
                    endpoint_name,
                )
                if tool_outputs:
                    forwarded_tool_chunks += len(tool_outputs)
                    for out_chunk in tool_outputs:
                        yield out_chunk
        if not final_output_emitted:
            fallback_text = extract_final_text(parser.get_raw_text())
            if fallback_text:
                logger.warning("Falling back to heuristic FINAL extraction due to missing parsed output")
                if not role_sent:
                    role_sent = True
                    yield build_delta({"role": "assistant"})
                yield build_delta({"content": fallback_text})
                metrics.observe_final_chunk(endpoint_name, True)
                metrics.observe_parser_fallback(endpoint_name)
                forwarded_final_chunks += 1
                fallback_used = True
                finish_chunk = build_finish("stop")
                if finish_chunk:
                    yield finish_chunk
        parser.reset_history()
        logger.info(
            "stream_complete id=%s final_chunks=%d tool_chunks=%d fallback_used=%s",
            chunk_id,
            forwarded_final_chunks,
            forwarded_tool_chunks,
            fallback_used,
        )
        yield format_sse_done()


def _emit_tool_chunk(
    chunk: ToolCallChunk,
    mode: ProxyMode,
    build_delta: Callable[[Dict[str, Any]], bytes],
    tool_indices: Dict[str, int],
    endpoint_name: str,
) -> List[bytes]:
    if mode == ProxyMode.FINAL_ONLY:
        return []

    if mode == ProxyMode.FINAL_PLUS_TOOLS_TEXT:
        text = f"\n[tool:{chunk.name}] {chunk.arguments}\n"
        metrics.observe_tool_chunk(endpoint_name, mode.value)
        return [build_delta({"content": text})]

    if mode == ProxyMode.OPENAI_TOOL_CALLS:
        if chunk.call_id not in tool_indices:
            tool_indices[chunk.call_id] = len(tool_indices)

        index = tool_indices[chunk.call_id]
        metrics.observe_tool_chunk(endpoint_name, mode.value)
        tool_payload = {
            "tool_calls": [
                {
                    "index": index,
                    "id": chunk.call_id,
                    "type": "function",
                    "function": {
                        "name": chunk.name,
                        "arguments": chunk.arguments,
                    },
                }
            ]
        }
        return [build_delta(tool_payload)]

    return []


def _normalize_non_streaming(
    response_payload: Dict[str, Any],
    mode: ProxyMode,
) -> Dict[str, Any]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return response_payload

    for choice in choices:
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        final_text, tools, used_fallback = _parse_full_text(content)
        if mode == ProxyMode.FINAL_ONLY:
            message["content"] = final_text
            message.pop("tool_calls", None)
        elif mode == ProxyMode.FINAL_PLUS_TOOLS_TEXT:
            tool_text = "".join(
                f"\n[tool:{t.name}] {t.arguments}\n" for t in tools
            )
            message["content"] = f"{final_text}{tool_text}"
            message.pop("tool_calls", None)
        elif mode == ProxyMode.OPENAI_TOOL_CALLS:
            message["content"] = final_text
            if tools:
                message["tool_calls"] = [
                    {
                        "id": tool.call_id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": tool.arguments,
                        },
                    }
                    for tool in tools
                ]
            else:
                message.pop("tool_calls", None)

        if final_text:
            metrics.observe_final_chunk("chat_completions", False)
        if used_fallback and final_text:
            metrics.observe_parser_fallback("chat_completions")
        if tools and mode != ProxyMode.FINAL_ONLY:
            for _ in tools:
                metrics.observe_tool_chunk("chat_completions", mode.value)

    return response_payload


def _parse_full_text(content: str) -> tuple[str, List[ToolCallChunk], bool]:
    parser = HarmonyStreamParser(
        encoding_name=config.harmony_encoding_name,
        prepend_missing_start=config.prepend_missing_start,
    )
    final_parts: List[str] = []
    tool_chunks: List[ToolCallChunk] = []

    for chunk in parser.feed(content):
        if isinstance(chunk, FinalChunk):
            final_parts.append(chunk.text)
        elif isinstance(chunk, ToolCallChunk):
            tool_chunks.append(chunk)
    for chunk in parser.close():
        if isinstance(chunk, FinalChunk):
            final_parts.append(chunk.text)
        elif isinstance(chunk, ToolCallChunk):
            tool_chunks.append(chunk)

    raw_text = parser.get_raw_text() or content
    parser.reset_history()

    if not final_parts:
        return extract_final_text(raw_text), tool_chunks, True
    return "".join(final_parts), tool_chunks, False


async def _read_json(request: Request) -> Dict[str, Any]:
    try:
        return await request.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - FastAPI handles
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc


def _apply_stop_tokens(payload: Dict[str, Any], config) -> None:
    if not config.harmony_stops_enabled:
        return

    harmony_stops = ["<|return|>", "<|call|>"]
    harmony_stops.extend(config.extra_stop_sequences)

    existing = payload.get("stop")
    if existing is None:
        payload["stop"] = harmony_stops
        return
    if isinstance(existing, list):
        payload["stop"] = list(dict.fromkeys(existing + harmony_stops))
        return
    if isinstance(existing, str):
        combined = [existing] + harmony_stops
        payload["stop"] = list(dict.fromkeys(combined))
