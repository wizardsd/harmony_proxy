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

from .config import ProxyConfig, ProxyMode, load_config
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
    app.state.upstream = UpstreamClient(
        base_url=config.upstream_base_url,
        connect_timeout=config.connect_timeout,
        read_timeout=config.read_timeout,
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await _read_json(request)
    _apply_stop_tokens(payload, config)

    stream = bool(payload.get("stream"))
    upstream: UpstreamClient = app.state.upstream

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
    )

    role_sent = False
    finish_sent = False
    tool_indices: Dict[str, int] = {}
    no_final_timeout = config.no_final_timeout

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
                                yield build_delta({"content": chunk.text})
                            if chunk.is_terminal:
                                finish_chunk = build_finish("stop")
                                if finish_chunk:
                                    yield finish_chunk
                        elif isinstance(chunk, ToolCallChunk):
                            for out_chunk in _emit_tool_chunk(
                                chunk,
                                config.proxy_mode,
                                build_delta,
                                tool_indices,
                            ):
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
                    yield build_delta({"content": chunk.text})
                if chunk.is_terminal:
                    finish_chunk = build_finish("stop")
                    if finish_chunk:
                        yield finish_chunk
            elif isinstance(chunk, ToolCallChunk):
                for out_chunk in _emit_tool_chunk(
                    chunk,
                    config.proxy_mode,
                    build_delta,
                    tool_indices,
                ):
                    yield out_chunk
        yield format_sse_done()


def _emit_tool_chunk(
    chunk: ToolCallChunk,
    mode: ProxyMode,
    build_delta: Callable[[Dict[str, Any]], bytes],
    tool_indices: Dict[str, int],
) -> List[bytes]:
    if mode == ProxyMode.FINAL_ONLY:
        return []

    if mode == ProxyMode.FINAL_PLUS_TOOLS_TEXT:
        text = f"\n[tool:{chunk.name}] {chunk.arguments}\n"
        return [build_delta({"content": text})]

    if mode == ProxyMode.OPENAI_TOOL_CALLS:
        if chunk.call_id not in tool_indices:
            tool_indices[chunk.call_id] = len(tool_indices)

        index = tool_indices[chunk.call_id]
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
        final_text, tools = _parse_full_text(content)
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

    return response_payload


def _parse_full_text(content: str) -> tuple[str, List[ToolCallChunk]]:
    parser = HarmonyStreamParser(encoding_name=config.harmony_encoding_name)
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

    if not final_parts:
        return content, tool_chunks
    return "".join(final_parts), tool_chunks


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
