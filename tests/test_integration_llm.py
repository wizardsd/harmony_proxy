import asyncio
import json
import os
from pathlib import Path

import pytest

from harmony_proxy.harmony_fallback import extract_final_text
from harmony_proxy.parser import FinalChunk, HarmonyStreamParser
from harmony_proxy.upstream import UpstreamClient


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
