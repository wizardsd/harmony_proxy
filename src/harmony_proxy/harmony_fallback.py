"""
Heuristic helpers that recover FINAL channel content when Harmony parsing fails.

These functions operate on raw Harmony-formatted text. They are intentionally
lenient: if the structure is broken, they attempt best-effort extraction so the
proxy can still return usable OpenAI Chat Completions to clients.
"""

from __future__ import annotations

import re
from typing import Optional


FINAL_CHANNEL_PATTERN = re.compile(
    r"<\|channel\|>final(?P<body>.*?)(?=(?:<\|channel\|>|<\|start\|>|$))",
    re.DOTALL,
)
MESSAGE_PATTERN = re.compile(r"<\|message\|>(?P<content>.*)", re.DOTALL)
TAG_PATTERN = re.compile(r"<\|[^>]*\|>")


def extract_final_text(raw: str) -> str:
    """
    Return the FINAL channel text from a Harmony-formatted string.

    If no FINAL channel can be found the original text is returned unchanged.
    """
    if not raw:
        return ""

    match: Optional[re.Match[str]] = None
    for match in FINAL_CHANNEL_PATTERN.finditer(raw):
        pass

    if not match:
        # Fallback: take the last <|message|> content in the transcript.
        last_message = None
        for candidate in MESSAGE_PATTERN.finditer(raw):
            last_message = candidate.group("content")
        if last_message:
            stop_index = last_message.find("<|")
            if stop_index != -1:
                last_message = last_message[:stop_index]
            stripped = _strip_tags(last_message)
            if stripped:
                return stripped
        return _strip_tags(raw)

    body = match.group("body")
    message_match = MESSAGE_PATTERN.search(body)
    if not message_match:
        return _strip_tags(body)

    content = message_match.group("content")
    stop_index = content.find("<|")
    if stop_index != -1:
        content = content[:stop_index]

    return _strip_tags(content)


def _strip_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = TAG_PATTERN.sub("", text)
    return cleaned.strip()
