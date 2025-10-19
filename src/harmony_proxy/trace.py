from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


_lock = threading.Lock()
_logger: Optional[logging.Logger] = None
_enabled = False
_current_path: Optional[str] = None
_max_string_length: Optional[int] = None


def configure(path: Optional[str], *, max_string_length: Optional[int] = None) -> None:
    """
    Configure trace logging. When ``path`` is ``None`` or empty, tracing is disabled.
    """
    global _logger, _enabled, _current_path, _max_string_length

    if not path:
        _disable()
        return

    abs_path = os.path.abspath(path)
    directory = os.path.dirname(abs_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    logger = logging.getLogger("harmony_proxy.trace")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(abs_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    _logger = logger
    _enabled = True
    _current_path = abs_path
    _max_string_length = max_string_length if max_string_length and max_string_length > 0 else None


def record(
    event: str,
    *,
    request_id: str,
    endpoint: str,
    payload: Dict[str, Any],
    stream: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a trace event as a JSON object. This function never raises; failures are swallowed.
    """
    if not _enabled or _logger is None:
        return

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "endpoint": endpoint,
        "request_id": request_id,
        "payload": payload,
    }
    if stream is not None:
        entry["stream"] = bool(stream)
    if metadata:
        entry["meta"] = metadata

    if _max_string_length is not None:
        entry = _compact(entry, _max_string_length)

    serialized = _safe_dumps(entry)
    if serialized is None:
        return

    with _lock:
        try:
            _logger.info(serialized)
        except Exception:
            # Do not let trace failures interfere with the main request flow.
            return


def is_enabled() -> bool:
    return _enabled


def current_path() -> Optional[str]:
    return _current_path


def reset_for_test() -> None:
    """
    Reset tracing to a disabled, handler-free state (test helper only).
    """
    _disable()


def _safe_dumps(entry: Dict[str, Any]) -> Optional[str]:
    try:
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=_stringify)
    except Exception:
        return None


def _stringify(value: Any) -> str:
    return repr(value)


def _disable() -> None:
    global _logger, _enabled, _current_path, _max_string_length
    if _logger is not None:
        for handler in list(_logger.handlers):
            _logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    _logger = None
    _enabled = False
    _current_path = None
    _max_string_length = None


def _compact(value: Any, limit: int):
    if isinstance(value, dict):
        return {key: _compact(val, limit) for key, val in value.items()}
    if isinstance(value, list):
        return [_compact(item, limit) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact(item, limit) for item in value)
    if isinstance(value, str):
        if len(value) > limit:
            overflow = len(value) - limit
            return f"{value[:limit]}... (+{overflow} chars)"
        return value
    return value
