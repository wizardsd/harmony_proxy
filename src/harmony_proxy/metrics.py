from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest


_enabled = True


def _init_registry() -> None:
    global REGISTRY
    global REQUEST_COUNTER
    global FINAL_CHUNKS_COUNTER
    global TOOL_CHUNKS_COUNTER
    global PARSER_FALLBACK_COUNTER
    global RETRY_COUNTER

    REGISTRY = CollectorRegistry()
    REQUEST_COUNTER = Counter(
        "harmony_proxy_requests_total",
        "Total number of requests handled by the proxy",
        ("endpoint", "mode", "stream"),
        registry=REGISTRY,
    )
    FINAL_CHUNKS_COUNTER = Counter(
        "harmony_proxy_final_chunks_total",
        "Number of FINAL channel chunks forwarded to clients",
        ("endpoint", "stream"),
        registry=REGISTRY,
    )
    TOOL_CHUNKS_COUNTER = Counter(
        "harmony_proxy_tool_chunks_total",
        "Number of tool-call chunks forwarded",
        ("endpoint", "mode"),
        registry=REGISTRY,
    )
    PARSER_FALLBACK_COUNTER = Counter(
        "harmony_proxy_parser_fallback_total",
        "Count of times heuristic Harmony fallback was triggered",
        ("endpoint",),
        registry=REGISTRY,
    )
    RETRY_COUNTER = Counter(
        "harmony_proxy_upstream_retries_total",
        "Number of upstream retry attempts executed",
        registry=REGISTRY,
    )


_init_registry()


def configure(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def observe_request(endpoint: str, stream: bool, mode: str) -> None:
    if not _enabled:
        return
    REQUEST_COUNTER.labels(endpoint=endpoint, mode=mode, stream=_bool_label(stream)).inc()


def observe_final_chunk(endpoint: str, stream: bool) -> None:
    if not _enabled:
        return
    FINAL_CHUNKS_COUNTER.labels(endpoint=endpoint, stream=_bool_label(stream)).inc()


def observe_tool_chunk(endpoint: str, mode: str) -> None:
    if not _enabled:
        return
    TOOL_CHUNKS_COUNTER.labels(endpoint=endpoint, mode=mode).inc()


def observe_parser_fallback(endpoint: str) -> None:
    if not _enabled:
        return
    PARSER_FALLBACK_COUNTER.labels(endpoint=endpoint).inc()


def observe_retry() -> None:
    if not _enabled:
        return
    RETRY_COUNTER.inc()


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


def is_enabled() -> bool:
    return _enabled


def content_type() -> str:
    return CONTENT_TYPE_LATEST


def reset_for_test() -> None:
    """
    Reset counters for test isolation. This touches internals of prometheus_client
    and must never be called from production code.
    """
    current_flag = _enabled
    _init_registry()
    configure(current_flag)


def _bool_label(flag: bool) -> str:
    return "true" if flag else "false"
