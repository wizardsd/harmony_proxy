import os

import pytest

from src.harmony_proxy import config as cfg


def test_get_env_bool(monkeypatch):
    monkeypatch.setenv("BOOL_TRUE", "true")
    assert cfg._get_env_bool("BOOL_TRUE", False) is True

    monkeypatch.setenv("BOOL_FALSE", "0")
    assert cfg._get_env_bool("BOOL_FALSE", True) is False

    monkeypatch.delenv("BOOL_MISSING", raising=False)
    assert cfg._get_env_bool("BOOL_MISSING", True) is True


def test_parse_int(monkeypatch):
    monkeypatch.setenv("INT_VAL", "42")
    assert cfg._parse_int("INT_VAL", 0) == 42

    monkeypatch.setenv("INT_INVALID", "not_an_int")
    assert cfg._parse_int("INT_INVALID", 7) == 7

    monkeypatch.delenv("INT_MISSING", raising=False)
    assert cfg._parse_int("INT_MISSING", 5) == 5


def test_parse_float(monkeypatch):
    monkeypatch.setenv("FLOAT_VAL", "3.14")
    assert cfg._parse_float("FLOAT_VAL", 0.0) == pytest.approx(3.14)

    monkeypatch.setenv("FLOAT_INVALID", "nan")
    # float conversion will succeed for "nan", but we expect fallback on ValueError only
    # Ensure that a non‑numeric string falls back to default
    monkeypatch.setenv("FLOAT_INVALID", "not_a_float")
    assert cfg._parse_float("FLOAT_INVALID", 2.5) == 2.5

    monkeypatch.delenv("FLOAT_MISSING", raising=False)
    assert cfg._parse_float("FLOAT_MISSING", 1.1) == 1.1


def test_parse_proxy_mode(monkeypatch):
    monkeypatch.setenv("PROXY_MODE", cfg.ProxyMode.OPENAI_TOOL_CALLS.value)
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.OPENAI_TOOL_CALLS

    monkeypatch.setenv("PROXY_MODE", cfg.ProxyMode.KILOCODE.value)
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.KILOCODE

    monkeypatch.setenv("PROXY_MODE", "KILOCODE")
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.KILOCODE

    monkeypatch.setenv("PROXY_MODE", "  FINAL_PLUS_TOOLS_TEXT  ")
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.FINAL_PLUS_TOOLS_TEXT

    monkeypatch.setenv("PROXY_MODE", "invalid_mode")
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.FINAL_ONLY

    monkeypatch.delenv("PROXY_MODE", raising=False)
    assert cfg._parse_proxy_mode() == cfg.ProxyMode.FINAL_ONLY


def test_parse_stops(monkeypatch):
    monkeypatch.setenv("EXTRA_STOP_SEQUENCES", "stop1, stop2 ,stop3")
    assert cfg._parse_stops() == ["stop1", "stop2", "stop3"]

    monkeypatch.setenv("EXTRA_STOP_SEQUENCES", "")
    assert cfg._parse_stops() == []

    monkeypatch.delenv("EXTRA_STOP_SEQUENCES", raising=False)
    assert cfg._parse_stops() == []


def test_load_config(monkeypatch):
    # Set a representative subset of environment variables
    env_vars = {
        "UPSTREAM_BASE_URL": "http://example.com/v1",
        "PROXY_MODE": cfg.ProxyMode.FINAL_PLUS_TOOLS_TEXT.value,
        "CONNECT_TIMEOUT": "5",
        "READ_TIMEOUT": "30",
        "NO_FINAL_TIMEOUT": "60",
        "MAX_RETRIES": "2",
        "RETRY_BACKOFF_SECONDS": "0.2",
        "METRICS_ENABLED": "false",
        "PREPEND_MISSING_HARMONY_START": "0",
        "HARMONY_STOPS_ENABLED": "1",
        "LOG_LEVEL": "debug",
        "EXTRA_STOP_SEQUENCES": "foo,bar",
        "HARMONY_ENCODING_NAME": "CustomEncoding",
        "TRACE_LOG_PATH": "/tmp/trace.log",
        "TRACE_MAX_STRING_LENGTH": "128",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    cfg_obj = cfg.load_config()

    assert cfg_obj.upstream_base_url == "http://example.com/v1"
    assert cfg_obj.proxy_mode == cfg.ProxyMode.FINAL_PLUS_TOOLS_TEXT
    assert cfg_obj.connect_timeout == 5.0
    assert cfg_obj.read_timeout == 30.0
    assert cfg_obj.no_final_timeout == 60.0
    assert cfg_obj.max_retries == 2
    assert cfg_obj.retry_backoff_seconds == 0.2
    assert cfg_obj.metrics_enabled is False
    assert cfg_obj.prepend_missing_start is False
    assert cfg_obj.harmony_stops_enabled is True
    assert cfg_obj.log_level == "DEBUG"
    assert cfg_obj.extra_stop_sequences == ["foo", "bar"]
    assert cfg_obj.harmony_encoding_name == "CustomEncoding"
    assert cfg_obj.trace_log_path == "/tmp/trace.log"
    assert cfg_obj.trace_max_string_length == 128

    monkeypatch.setenv("TRACE_MAX_STRING_LENGTH", "0")
    assert cfg._parse_positive_int_or_none("TRACE_MAX_STRING_LENGTH") is None
