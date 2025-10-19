import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harmony_proxy.harmony_fallback import extract_final_text
from harmony_proxy.parser import FinalChunk, HarmonyStreamParser
from harmony_proxy.upstream import UpstreamClient
from harmony_proxy.app import app


pytestmark = pytest.mark.integration


def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.exists():
        return
    with dotenv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_roundtrip_hi():
    base_url = os.environ["UPSTREAM_BASE_URL"]
    connect_timeout = float(os.getenv("CONNECT_TIMEOUT", "10"))
    read_timeout = float(os.getenv("READ_TIMEOUT", "60"))
    model = os.getenv("LLM_MODEL", "gpt-oss")
    proxy_mode = os.getenv("LLM_PROXY_MODE", "final_only")

    async def run_llm_request() -> tuple[str, bool]:
        client = UpstreamClient(
            base_url=base_url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=int(os.getenv("MAX_RETRIES", "1")),
            retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "0.5")),
        )
        parser = HarmonyStreamParser()
        final_parts: list[str] = []
        parser_had_error = False

        payload = {
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Hi"},
            ],
        }

        try:
            async for event in client.stream_chat_completions(payload):
                data = event.data.strip()
                if not data or data == "[DONE]":
                    break
                try:
                    payload_json = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in payload_json.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        for chunk in parser.feed(content):
                            if isinstance(chunk, FinalChunk):
                                final_parts.append(chunk.text)
                    if parser.had_error:
                        parser_had_error = True
        finally:
            for chunk in parser.close():
                if isinstance(chunk, FinalChunk) and chunk.text:
                    final_parts.append(chunk.text)
            await client.aclose()

        combined = "".join(final_parts).strip()
        if not combined:
            # Fall back to heuristic extraction if the parser failed to yield text.
            combined = extract_final_text(parser.get_raw_text())
        return combined.strip(), parser_had_error or parser.had_error

    response_text, parser_had_errors = asyncio.run(run_llm_request())
    assert not parser_had_errors, "Harmony stream contained unexpected formatting; see logs"
    assert response_text, "LLM returned an empty response for 'Hi'"
    assert any(token in response_text.lower() for token in ("hi", "hello")), response_text


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_kilocode_history_is_normalised_before_upstream():
    model = os.getenv("LLM_MODEL", "gpt-oss")

    harmony_history = (
        "<|start|>assistant"
        "<|channel|>analysis"
        "<|message|>Considering how to answer the greeting.<|end|>"
        "<|start|>assistant"
        "<|channel|>final"
        "<|message|>Hello from the legacy bridge!<|end|>"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Please respond with a friendly greeting."},
            {"role": "assistant", "content": harmony_history},
            {"role": "user", "content": "Say hello again."},
        ],
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, "proxy returned no choices"
    message = choices[0].get("message", {})
    assert isinstance(message.get("content"), str) and message["content"].strip(), "empty assistant reply"


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_kilocode_multisegment_user_content_is_flattened():
    class RecordingUpstream:
        def __init__(self) -> None:
            self.captured_payload: dict[str, Any] | None = None

        async def chat_completions(self, payload, trace_id=None):  # type: ignore[override]
            self.captured_payload = payload
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<|start|>assistant<|channel|>final<|message|>Ready<|end|>",
                        }
                    }
                ]
            }

        async def stream_chat_completions(self, payload, trace_id=None):  # pragma: no cover
            raise AssertionError("streaming not expected in this test")

        async def aclose(self):  # pragma: no cover
            pass

    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-oss"),
        "messages": [
            {"role": "system", "content": "Act as a planner."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>Summarise recent patches</task>"},
                    {
                        "type": "text",
                        "text": "<environment_details>Repository contains src/ and tests/</environment_details>",
                    },
                ],
            },
        ],
    }

    recorder = RecordingUpstream()
    with TestClient(app) as client:
        client.app.state.upstream = recorder  # type: ignore[attr-defined]
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert recorder.captured_payload is not None, "proxy did not forward payload to upstream"
    forwarded_messages = recorder.captured_payload["messages"]
    assert isinstance(forwarded_messages[1]["content"], str)
    assert "<task>Summarise recent patches</task>" in forwarded_messages[1]["content"]
    assert "<environment_details>" in forwarded_messages[1]["content"]


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_kilocode_plain_tool_xml_becomes_tool_call():
    class PlainToolUpstream:
        async def chat_completions(self, payload, trace_id=None):  # type: ignore[override]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<|start|>assistant"
                                "<|channel|>commentary"
                                "<|message|><update_todo_list>\n"
                                "<todos>\n"
                                "[ ] Investigate\n"
                                "</todos>\n"
                                "</update_todo_list>"
                            ),
                        }
                    }
                ]
            }

        async def stream_chat_completions(self, payload, trace_id=None):  # pragma: no cover
            raise AssertionError("streaming not expected in this test")

        async def aclose(self):  # pragma: no cover
            pass

    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-oss"),
        "messages": [
            {"role": "user", "content": "Trigger a todo list update."},
        ],
    }

    with TestClient(app) as client:
        client.app.state.upstream = PlainToolUpstream()  # type: ignore[attr-defined]
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"].startswith("<update_todo_list>")
    assert message["tool_calls"][0]["function"]["name"] == "update_todo_list"
    assert "<todos>" in message["tool_calls"][0]["function"]["arguments"]


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_kilocode_plain_codebase_search_becomes_tool_call():
    class CodebaseToolUpstream:
        async def chat_completions(self, payload, trace_id=None):  # type: ignore[override]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<|start|>assistant"
                                "<|channel|>commentary"
                                "<|message|><codebase_search>\n"
                                "<query>__builtin_</query>\n"
                                "</codebase_search>"
                            ),
                        }
                    }
                ]
            }

        async def stream_chat_completions(self, payload, trace_id=None):  # pragma: no cover
            raise AssertionError("streaming not expected in this test")

        async def aclose(self):  # pragma: no cover
            pass

    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-oss"),
        "messages": [
            {"role": "user", "content": "Search for intrinsics."},
        ],
    }

    with TestClient(app) as client:
        client.app.state.upstream = CodebaseToolUpstream()  # type: ignore[attr-defined]
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"].startswith("<codebase_search>")
    assert message["tool_calls"][0]["function"]["name"] == "codebase_search"
    assert "<query>__builtin_</query>" in message["tool_calls"][0]["function"]["arguments"]


@pytest.mark.skipif(
    not os.getenv("UPSTREAM_BASE_URL"),
    reason="UPSTREAM_BASE_URL must point at a running llama.cpp-compatible server.",
)
def test_llm_kilocode_full_trace_sequence():
    class TraceUpstream:
        async def chat_completions(self, payload, trace_id=None):  # type: ignore[override]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<|start|>assistant"
                                "<|channel|>commentary"
                                "<|message|>We will search the repo.<|start|>assistant"
                                "<|channel|>commentary"
                                "<|message|><codebase_search>\n<query>asm</query>\n</codebase_search>"
                                "<|start|>assistant"
                                "<|channel|>commentary"
                                "<|message|><codebase_search>\n<query>__builtin_</query>\n</codebase_search>"
                                "<|start|>assistant"
                                "<|channel|>final"
                                "<|message|>The task will be completed by performing a semantic search across the repository for the specified patterns."
                            ),
                        }
                    }
                ]
            }

        async def stream_chat_completions(self, payload, trace_id=None):  # pragma: no cover
            raise AssertionError("streaming not expected in this test")

        async def aclose(self):  # pragma: no cover
            pass

    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-oss"),
        "messages": [
            {"role": "user", "content": "First orchestrator request."},
        ],
    }

    with TestClient(app) as client:
        client.app.state.upstream = TraceUpstream()  # type: ignore[attr-defined]
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == "codebase_search"
    assert "<query>__builtin_</query>" in message["tool_calls"][0]["function"]["arguments"]
    assert message["content"].strip().startswith("<codebase_search>")
