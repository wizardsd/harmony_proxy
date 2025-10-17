from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Union


log = logging.getLogger(__name__)


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
    Wrapper around the official openai_harmony StreamableParser when available.
    Falls back to a permissive heuristic parser when the library cannot be imported.
    """

    def __init__(self, encoding_name: str = "HarmonyGptOss", role: str = "assistant") -> None:
        self._encoding_name = encoding_name
        self._role = role
        self._native_parser = self._load_native_parser()
        self._naive = _NaiveHarmonyParser()

    def feed(self, delta: str) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []
        if not delta:
            return chunks

        native_chunks = self._feed_native(delta)
        if native_chunks:
            return native_chunks

        chunks.extend(self._naive.feed(delta))
        return chunks

    def close(self) -> List[ParsedChunk]:
        native_chunks = self._close_native()
        if native_chunks:
            return native_chunks
        return self._naive.close()

    def _load_native_parser(self) -> Any:
        try:
            harmony = importlib.import_module("openai_harmony")
        except ImportError:
            log.warning(
                "openai_harmony not installed; falling back to heuristic Harmony parser.",
            )
            return None

        load_encoding = getattr(harmony, "load_harmony_encoding", None)
        parser_cls = getattr(harmony, "StreamableParser", None)
        if not callable(load_encoding) or parser_cls is None:
            log.warning(
                "openai_harmony available but missing StreamableParser or encoding loader; "
                "falling back to heuristic parser.",
            )
            return None

        try:
            encoding = load_encoding(self._encoding_name)
            return parser_cls(encoding=encoding, role=self._role)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to instantiate openai_harmony StreamableParser: %s", exc)
        return None

    def _feed_native(self, delta: str) -> List[ParsedChunk]:
        parser = self._native_parser
        if parser is None:
            return []
        try:
            events = parser.feed(delta)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("openai_harmony parser raised %s; disabling native parser.", exc)
            self._native_parser = None
            return []
        converted: List[ParsedChunk] = []
        for event in _ensure_sequence(events):
            converted.extend(self._convert_native_event(event))
        if converted:
            return converted
        return []

    def _close_native(self) -> List[ParsedChunk]:
        parser = self._native_parser
        if parser is None:
            return []
        close = getattr(parser, "close", None)
        if not callable(close):
            return []
        try:
            events = close()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("openai_harmony parser close() failed: %s", exc)
            self._native_parser = None
            return []
        converted: List[ParsedChunk] = []
        for event in _ensure_sequence(events):
            converted.extend(self._convert_native_event(event))
        return converted

    def _convert_native_event(self, event: Any) -> List[ParsedChunk]:
        """
        Attempts to adapt an event from openai_harmony into FinalChunk/ToolCallChunk objects.
        Falls back to heuristic handling if the event layout is unknown.
        """
        channel = _get_attr(event, "channel")
        if isinstance(channel, bytes):
            channel = channel.decode("utf-8", errors="ignore")

        text = _get_attr(event, "text")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")

        event_type = _get_attr(event, "type") or _get_attr(event, "kind")

        converted: List[ParsedChunk] = []
        if channel == "final" and isinstance(text, str) and text:
            is_terminal = bool(_get_attr(event, "is_terminal") or _get_attr(event, "is_finished"))
            converted.append(FinalChunk(text=text, is_terminal=is_terminal))
            return converted

        tool_name = _get_attr(event, "tool_name") or _get_attr(event, "name")
        arguments = _get_attr(event, "arguments") or _get_attr(event, "args")
        call_id = _get_attr(event, "call_id") or _get_attr(event, "id")

        if channel == "commentary" and tool_name and arguments is not None:
            if not isinstance(arguments, str):
                arguments = str(arguments)
            if not isinstance(call_id, str) or not call_id:
                call_id = _build_tool_id(tool_name)
            converted.append(
                ToolCallChunk(call_id=call_id, name=str(tool_name), arguments=arguments),
            )
            return converted

        if event_type == "final" and isinstance(text, str) and text:
            converted.append(FinalChunk(text=text, is_terminal=False))
            return converted

        return []


class _NaiveHarmonyParser:
    """
    Heuristic Harmony parser that understands the minimal subset required for final/tool chunks.
    Keeps enough state to emit incremental deltas for the final channel.
    """

    FINAL_MARKER = "<|channel|>final"
    MESSAGE_MARKER = "<|message|>"
    END_MARKER = "<|end|>"
    TOOL_MARKER = "<|channel|>commentary to="
    CALL_MARKER = "<|call|>"

    def __init__(self) -> None:
        self._buffer: List[str] = []
        self._final_emitted_len = 0
        self._final_closed = False
        self._tool_cursor = 0
        self._tool_counter = 0

    def feed(self, delta: str) -> List[ParsedChunk]:
        self._buffer.append(delta)
        full_text = "".join(self._buffer)
        chunks: List[ParsedChunk] = []

        final_text, final_complete = self._extract_final(full_text)
        if final_text is not None:
            new_text = final_text[self._final_emitted_len :]
            if new_text:
                chunks.append(FinalChunk(text=new_text))
                self._final_emitted_len += len(new_text)
            if final_complete and not self._final_closed:
                chunks.append(FinalChunk(text="", is_terminal=True))
                self._final_closed = True

        chunks.extend(self._extract_tools(full_text))
        return chunks

    def close(self) -> List[ParsedChunk]:
        if self._final_closed:
            return []
        full_text = "".join(self._buffer)
        final_text, final_complete = self._extract_final(full_text)
        chunks: List[ParsedChunk] = []
        if final_text is not None:
            new_text = final_text[self._final_emitted_len :]
            if new_text:
                chunks.append(FinalChunk(text=new_text))
                self._final_emitted_len += len(new_text)
        if final_complete and not self._final_closed:
            chunks.append(FinalChunk(text="", is_terminal=True))
        self._final_closed = True
        return chunks

    def _extract_final(self, full_text: str) -> tuple[Optional[str], bool]:
        marker_idx = full_text.find(self.FINAL_MARKER)
        if marker_idx == -1:
            return None, False
        msg_idx = full_text.find(self.MESSAGE_MARKER, marker_idx)
        if msg_idx == -1:
            return "", False
        start = msg_idx + len(self.MESSAGE_MARKER)
        end = full_text.find(self.END_MARKER, start)
        if end == -1:
            return full_text[start:], False
        return full_text[start:end], True

    def _extract_tools(self, full_text: str) -> List[ParsedChunk]:
        chunks: List[ParsedChunk] = []
        idx = self._tool_cursor
        while True:
            idx = full_text.find(self.TOOL_MARKER, idx)
            if idx == -1:
                break
            name_start = idx + len(self.TOOL_MARKER)
            name_match = re.match(r"([^\s<]+)", full_text[name_start:])
            if not name_match:
                break
            tool_name = name_match.group(1)
            msg_idx = full_text.find(self.MESSAGE_MARKER, idx)
            if msg_idx == -1:
                break
            args_start = msg_idx + len(self.MESSAGE_MARKER)
            call_idx = full_text.find(self.CALL_MARKER, args_start)
            if call_idx == -1:
                break
            arguments = full_text[args_start:call_idx]
            self._tool_counter += 1
            call_id = f"call_{self._tool_counter}"
            chunks.append(
                ToolCallChunk(call_id=call_id, name=tool_name, arguments=arguments.strip()),
            )
            idx = call_idx + len(self.CALL_MARKER)
            self._tool_cursor = idx
        return chunks


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _build_tool_id(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return f"call_{sanitized or 'tool'}"
