from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from openai_harmony import (
    StreamableParser as HarmonyStreamableParser,
    HarmonyEncodingName,
    HarmonyError,
    load_harmony_encoding,
)

from .harmony_fallback import extract_final_text


log = logging.getLogger(__name__)

RETURN_TOKEN_ID = 200002
END_TOKEN_ID = 200007
CALL_TOKEN_ID = 200012


@dataclass
class FinalChunk:
    text: str
    is_terminal: bool = False


@dataclass
class ToolCallChunk:
    call_id: str
    name: str
    arguments: str


ParsedChunk = Union[FinalChunk, ToolCallChunk]


class HarmonyStreamParser:
    """
    Streaming parser backed exclusively by the official openai_harmony bindings.
    Accepts Harmony-formatted text deltas and emits parsed channel chunks suitable
    for OpenAI-compatible streaming responses.
    """

    def __init__(
        self,
        encoding_name: str = "HarmonyGptOss",
        role: Optional[str] = None,
        prepend_missing_start: bool = True,
    ) -> None:
        self._encoding_name = encoding_name
        self._encoding = self._load_encoding(encoding_name)
        self._parser = HarmonyStreamableParser(self._encoding, role=None)

        self._buffer: str = ""
        self._cursor: int = 0

        self._active_channel: Optional[str] = None
        self._active_recipient: Optional[str] = None
        self._tool_buffer: List[str] = []
        self._tool_counter: int = 0
        self._raw_segments: List[str] = []
        self._had_error: bool = False

        self._prepend_missing_start = prepend_missing_start
        self._saw_harmony_start: bool = False
        self._synthetic_mode: bool = False
        self._synthetic_buffer: List[str] = []

    def feed(self, delta: str) -> List[ParsedChunk]:
        if not delta:
            return []

        if (
            self._prepend_missing_start
            and not self._synthetic_mode
            and not self._saw_harmony_start
        ):
            if "<|start|>assistant" in delta:
                self._saw_harmony_start = True
            else:
                self._synthetic_mode = True

        self._raw_segments.append(delta)

        if self._synthetic_mode:
            self._synthetic_buffer.append(delta)
            return []

        self._buffer += delta
        chunks: List[ParsedChunk] = []

        while True:
            segment = self._pop_processable_segment()
            if not segment:
                break
            chunks.extend(self._process_segment(segment))

        self._compact_buffer()
        return chunks

    def close(self) -> List[ParsedChunk]:
        if self._synthetic_mode:
            raw = "".join(self._synthetic_buffer)
            text = extract_final_text(raw)
            if not text.strip():
                text = raw.strip()
            chunks: List[ParsedChunk] = []
            if text:
                chunks.append(FinalChunk(text=text))
                chunks.append(FinalChunk(text="", is_terminal=True))
            self._synthetic_buffer.clear()
            return chunks

        chunks: List[ParsedChunk] = []
        if self._cursor < len(self._buffer):
            remaining = self._buffer[self._cursor :]
            if remaining:
                chunks.extend(self._process_segment(remaining))
            self._cursor = len(self._buffer)

        try:
            self._parser.process_eos()
        except HarmonyError as exc:  # pragma: no cover - defensive
            log.debug("Harmony parser process_eos failed: %s", exc)
            self._had_error = True

        self._buffer = ""
        self._cursor = 0
        return chunks

    def _process_segment(self, segment: str) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []

        try:
            tokens = self._encoding.encode(segment, allowed_special="all")
        except HarmonyError as exc:
            log.warning("Failed to encode harmony segment; dropping text. segment=%r error=%s", segment, exc)
            self._had_error = True
            return chunks

        for token in tokens:
            chunks.extend(self._process_token(token))

        return chunks

    def _process_token(self, token: int) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []

        try:
            self._parser.process(token)
        except HarmonyError as exc:
            log.warning("Harmony parser failed on token %s: %s", token, exc)
            self._had_error = True
            return chunks

        channel = self._parser.current_channel
        if channel is not None:
            self._active_channel = channel

        recipient = self._parser.current_recipient
        if recipient is not None:
            self._active_recipient = recipient

        delta = self._parser.last_content_delta
        if delta:
            if self._active_channel == "final":
                chunks.append(FinalChunk(text=delta))
            elif self._active_channel == "commentary" and self._active_recipient:
                self._tool_buffer.append(delta)

        if token == RETURN_TOKEN_ID:
            if self._active_channel == "final":
                chunks.append(FinalChunk(text="", is_terminal=True))
            self._reset_message_state()
        elif token == CALL_TOKEN_ID:
            tool_chunk = self._flush_tool_chunk()
            if tool_chunk:
                chunks.append(tool_chunk)
            self._reset_message_state()
        elif token == END_TOKEN_ID:
            self._reset_message_state()

        return chunks

    def _flush_tool_chunk(self) -> Optional[ToolCallChunk]:
        if not self._tool_buffer or not self._active_recipient:
            return None
        self._tool_counter += 1
        arguments = "".join(self._tool_buffer).strip()
        chunk = ToolCallChunk(
            call_id=f"call_{self._tool_counter}",
            name=self._active_recipient,
            arguments=arguments,
        )
        return chunk

    def _reset_message_state(self) -> None:
        self._active_channel = None
        self._active_recipient = None
        self._tool_buffer.clear()

    def _pop_processable_segment(self) -> str:
        if self._cursor >= len(self._buffer):
            return ""

        pending = self._buffer[self._cursor :]
        marker_index = pending.rfind("<|")
        if marker_index != -1:
            tail = pending[marker_index:]
            if "|>" not in tail:
                segment = pending[:marker_index]
                self._cursor += len(segment)
                return segment

        segment = pending
        self._cursor = len(self._buffer)
        return segment

    def _compact_buffer(self) -> None:
        if self._cursor == 0:
            return
        if self._cursor >= len(self._buffer):
            self._buffer = ""
            self._cursor = 0
            return
        self._buffer = self._buffer[self._cursor :]
        self._cursor = 0

    def _load_encoding(self, encoding_name: str | HarmonyEncodingName):
        name = encoding_name
        if isinstance(encoding_name, str):
            try:
                name = HarmonyEncodingName(encoding_name)
            except ValueError:
                name = encoding_name
        return load_harmony_encoding(name)

    @property
    def had_error(self) -> bool:
        return self._had_error

    def get_raw_text(self) -> str:
        return "".join(self._raw_segments)

    def reset_history(self) -> None:
        self._raw_segments.clear()
        self._had_error = False
        self._saw_harmony_start = False
        self._synthetic_mode = False
        self._synthetic_buffer.clear()
