import copy
from types import SimpleNamespace

from harmony_proxy.app import (
    ChannelPrefixStripper,
    _apply_stop_tokens,
    _emit_tool_chunk,
    _normalize_non_streaming,
    strip_channel_prefix,
)
from harmony_proxy.config import ProxyMode
from harmony_proxy.parser import ToolCallChunk


HARMONY_WITH_TOOL = (
    "<|start|>assistant"
    "<|channel|>commentary to=tool_x"
    "<|message|>{\"arg\":1}"
    "<|call|>"
    "<|end|>"
    "<|start|>assistant"
    "<|channel|>final"
    "<|message|>Result text"
    "<|end|>"
)

HARMONY_WITH_TOOL_PREFIXED = (
    "<|start|>assistant"
    "<|channel|>commentary to=tool_x"
    "<|message|>{\"arg\":1}"
    "<|call|>"
    "<|end|>"
    "<|start|>assistant"
    "<|channel|>final"
    "<|message|>analysis: Result text"
    "<|end|>"
)

MALFORMED_FINAL = (
    "<|start|>assistant"
    "<|channel|>final"
    "<|message|>Recovered output"
)

HARMONY_WITH_LEADING_PLAIN = (
    "Result text"
    "<|start|>assistant"
    "<|channel|>analysis"
    "<|message|>analysis block"
    "<|start|>assistant"
    "<|channel|>final"
    "<|message|>Result text"
    "<|end|>"
)


def make_config(**overrides):
    base = {
        "harmony_stops_enabled": True,
        "extra_stop_sequences": ["<|custom|>"],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_channel_prefix_stripper_handles_chunked_prefix():
    stripper = ChannelPrefixStripper()
    assert stripper.feed("ana") == ""
    assert stripper.feed("lysis") == ""
    assert stripper.feed(":  Result") == "Result"
    assert stripper.feed(" continues") == " continues"


def test_strip_channel_prefix_helper():
    assert strip_channel_prefix("analysis\nAnswer") == "Answer"
    assert strip_channel_prefix("final Response") == "Response"
    assert strip_channel_prefix("Hello") == "Hello"


def test_apply_stop_tokens_inserts_defaults_when_missing():
    payload = {}
    cfg = make_config()
    _apply_stop_tokens(payload, cfg)
    assert payload["stop"][:2] == ["<|return|>", "<|call|>"]
    assert "<|custom|>" in payload["stop"]


def test_apply_stop_tokens_merges_with_list():
    payload = {"stop": ["END"]}
    cfg = make_config()
    _apply_stop_tokens(payload, cfg)
    assert payload["stop"][0] == "END"
    assert "<|return|>" in payload["stop"]
    assert payload["stop"].count("<|return|>") == 1


def test_apply_stop_tokens_converts_string_to_list():
    payload = {"stop": "DONE"}
    cfg = make_config()
    _apply_stop_tokens(payload, cfg)
    assert payload["stop"][0] == "DONE"
    assert "<|call|>" in payload["stop"]


def test_normalize_non_streaming_final_only():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": HARMONY_WITH_TOOL,
                }
            }
        ]
    }
    result = _normalize_non_streaming(copy.deepcopy(payload), ProxyMode.FINAL_ONLY)
    assert result["choices"][0]["message"]["content"] == "Result text"
    assert "tool_calls" not in result["choices"][0]["message"]


def test_normalize_non_streaming_openai_tool_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": HARMONY_WITH_TOOL,
                }
            }
        ]
    }
    result = _normalize_non_streaming(copy.deepcopy(payload), ProxyMode.OPENAI_TOOL_CALLS)
    message = result["choices"][0]["message"]
    assert message["content"] == "Result text"
    assert message["tool_calls"][0]["function"]["name"] == "tool_x"
    assert message["tool_calls"][0]["function"]["arguments"] == "{\"arg\":1}"


def test_normalize_non_streaming_kilocode():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": HARMONY_WITH_TOOL_PREFIXED,
                }
            }
        ]
    }
    result = _normalize_non_streaming(copy.deepcopy(payload), ProxyMode.KILOCODE)
    message = result["choices"][0]["message"]
    assert message["content"] == "Result text"
    tool_call = message["tool_calls"][0]["function"]
    assert tool_call["name"] == "tool_x"
    assert tool_call["arguments"] == "{\"arg\":1}"


def test_normalize_non_streaming_kilocode_handles_plain_prefix():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": HARMONY_WITH_LEADING_PLAIN,
                }
            }
        ]
    }
    result = _normalize_non_streaming(copy.deepcopy(payload), ProxyMode.KILOCODE)
    content = result["choices"][0]["message"]["content"]
    assert content == "Result text"


def test_emit_tool_chunk_modes():
    tool = ToolCallChunk(call_id="call_42", name="execute", arguments="{}")
    captured = []

    def fake_build(delta):
        captured.append(delta)
        return b"bytes"

    # FINAL_ONLY -> empty list
    assert (
        _emit_tool_chunk(
            tool,
            ProxyMode.FINAL_ONLY,
            fake_build,
            {},
            "chat_completions",
        )
        == []
    )

    # FINAL_PLUS_TOOLS_TEXT -> single content delta
    captured.clear()
    result = _emit_tool_chunk(
        tool,
        ProxyMode.FINAL_PLUS_TOOLS_TEXT,
        fake_build,
        {},
        "chat_completions",
    )
    assert result == [b"bytes"]
    assert captured[0]["content"].startswith("\n[tool:execute]")

    # OPENAI_TOOL_CALLS -> tool_calls delta
    captured.clear()
    result = _emit_tool_chunk(
        tool,
        ProxyMode.OPENAI_TOOL_CALLS,
        fake_build,
        {},
        "chat_completions",
    )
    assert result == [b"bytes"]
    assert captured[0]["tool_calls"][0]["function"]["name"] == "execute"

    # KILOCODE -> same as OPENAI_TOOL_CALLS
    captured.clear()
    result = _emit_tool_chunk(
        tool,
        ProxyMode.KILOCODE,
        fake_build,
        {},
        "chat_completions",
    )
    assert result == [b"bytes"]
    assert captured[0]["tool_calls"][0]["function"]["name"] == "execute"


def test_normalize_non_streaming_fallback_on_malformed_final():
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": MALFORMED_FINAL,
                }
            }
        ]
    }
    result = _normalize_non_streaming(payload, ProxyMode.FINAL_ONLY)
    assert result["choices"][0]["message"]["content"] == "Recovered output"
