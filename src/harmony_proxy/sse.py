from __future__ import annotations

import json


def format_sse_event(data: dict, event: str = "message") -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def format_sse_done(event: str = "message") -> bytes:
    return f"event: {event}\ndata: [DONE]\n\n".encode("utf-8")
