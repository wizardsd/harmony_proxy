import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ProxyMode(str, Enum):
    FINAL_ONLY = "final_only"
    FINAL_PLUS_TOOLS_TEXT = "final_plus_tools_text"
    OPENAI_TOOL_CALLS = "openai_tool_calls"


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


def _parse_proxy_mode() -> ProxyMode:
    raw = os.getenv("PROXY_MODE", ProxyMode.FINAL_ONLY.value)
    try:
        return ProxyMode(raw)
    except ValueError:
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
    harmony_stops_enabled: bool = _get_env_bool("HARMONY_STOPS_ENABLED", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    extra_stop_sequences: List[str] = field(default_factory=_parse_stops)
    harmony_encoding_name: str = os.getenv("HARMONY_ENCODING_NAME", "HarmonyGptOss")


def load_config() -> ProxyConfig:
    return ProxyConfig()
