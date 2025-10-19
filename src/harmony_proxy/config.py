import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ProxyMode(str, Enum):
    FINAL_ONLY = "final_only"
    FINAL_PLUS_TOOLS_TEXT = "final_plus_tools_text"
    OPENAI_TOOL_CALLS = "openai_tool_calls"
    KILOCODE = "kilocode"


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_positive_int_or_none(name: str) -> Optional[int]:
    value = _parse_int(name, 0)
    if value <= 0:
        return None
    return value


def _parse_proxy_mode() -> ProxyMode:
    raw = os.getenv("PROXY_MODE", ProxyMode.FINAL_ONLY.value)
    if raw is None:
        return ProxyMode.FINAL_ONLY
    normalized = raw.strip().lower()
    for mode in ProxyMode:
        if normalized == mode.value:
            return mode
    return ProxyMode.FINAL_ONLY


def _parse_stops() -> List[str]:
    raw = os.getenv("EXTRA_STOP_SEQUENCES")
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


@dataclass
class ProxyConfig:
    upstream_base_url: str = os.getenv("UPSTREAM_BASE_URL", "http://localhost:8080/v1")
    proxy_mode: ProxyMode = _parse_proxy_mode()
    connect_timeout: float = _parse_float("CONNECT_TIMEOUT", 10.0)
    read_timeout: float = _parse_float("READ_TIMEOUT", 120.0)
    no_final_timeout: float = _parse_float("NO_FINAL_TIMEOUT", 90.0)
    max_retries: int = _parse_int("MAX_RETRIES", 3)
    retry_backoff_seconds: float = _parse_float("RETRY_BACKOFF_SECONDS", 0.5)
    metrics_enabled: bool = _get_env_bool("METRICS_ENABLED", True)
    prepend_missing_start: bool = _get_env_bool("PREPEND_MISSING_HARMONY_START", True)
    harmony_stops_enabled: bool = _get_env_bool("HARMONY_STOPS_ENABLED", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    extra_stop_sequences: List[str] = field(default_factory=_parse_stops)
    harmony_encoding_name: str = os.getenv("HARMONY_ENCODING_NAME", "HarmonyGptOss")
    trace_log_path: Optional[str] = os.getenv("TRACE_LOG_PATH") or None
    trace_max_string_length: Optional[int] = _parse_positive_int_or_none("TRACE_MAX_STRING_LENGTH")


def load_config() -> ProxyConfig:
    """
    Load configuration from the current environment.

    The original implementation relied on the default values of the
    ``ProxyConfig`` dataclass, which are evaluated at import time.
    This caused tests that monkey‑patch environment variables to see stale
    values because the defaults had already been materialised.

    By constructing the ``ProxyConfig`` instance explicitly we ensure that
    each call reads the environment afresh, making the function deterministic
    and test‑friendly.
    """
    return ProxyConfig(
        upstream_base_url=os.getenv("UPSTREAM_BASE_URL", "http://localhost:8080/v1"),
        proxy_mode=_parse_proxy_mode(),
        connect_timeout=_parse_float("CONNECT_TIMEOUT", 10.0),
        read_timeout=_parse_float("READ_TIMEOUT", 120.0),
        no_final_timeout=_parse_float("NO_FINAL_TIMEOUT", 90.0),
        max_retries=_parse_int("MAX_RETRIES", 3),
        retry_backoff_seconds=_parse_float("RETRY_BACKOFF_SECONDS", 0.5),
        metrics_enabled=_get_env_bool("METRICS_ENABLED", True),
        prepend_missing_start=_get_env_bool("PREPEND_MISSING_HARMONY_START", True),
        harmony_stops_enabled=_get_env_bool("HARMONY_STOPS_ENABLED", True),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        extra_stop_sequences=_parse_stops(),
        harmony_encoding_name=os.getenv("HARMONY_ENCODING_NAME", "HarmonyGptOss"),
        trace_log_path=os.getenv("TRACE_LOG_PATH") or None,
        trace_max_string_length=_parse_positive_int_or_none("TRACE_MAX_STRING_LENGTH"),
    )
