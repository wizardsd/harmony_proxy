from harmony_proxy.parser import FinalChunk, HarmonyStreamParser, ToolCallChunk


HARMONY_SAMPLE = (
    "<|start|>assistant"
    "<|channel|>analysis"
    "<|message|>thinking"
    "<|end|>"
    "<|start|>assistant"
    "<|channel|>commentary to=read_file"
    "<|message|>{\"path\":\"foo.txt\"}"
    "<|call|>"
    "<|end|>"
    "<|start|>assistant"
    "<|channel|>final"
    "<|message|>Hello"
    "<|end|>"
)


def collect_chunks(parser: HarmonyStreamParser, text: str):
    chunks = []
    for idx in range(0, len(text), 15):
        chunks.extend(parser.feed(text[idx : idx + 15]))
    chunks.extend(parser.close())
    return chunks


def test_naive_parser_emits_tool_and_final_chunks():
    parser = HarmonyStreamParser()
    parser._native_parser = None  # force heuristic parser for deterministic test
    chunks = collect_chunks(parser, HARMONY_SAMPLE)

    assert any(isinstance(chunk, ToolCallChunk) for chunk in chunks), "tool call missing"
    assert any(
        isinstance(chunk, FinalChunk) and chunk.text == "Hello" for chunk in chunks
    ), "final channel text missing"
    assert any(
        isinstance(chunk, FinalChunk) and chunk.is_terminal for chunk in chunks
    ), "final terminal marker missing"


def test_parser_supports_multiple_feed_calls_without_overlap():
    parser = HarmonyStreamParser()
    parser._native_parser = None
    first_pass_chunks = parser.feed(HARMONY_SAMPLE[:30])
    second_pass_chunks = parser.feed(HARMONY_SAMPLE[30:60])
    remaining_chunks = parser.feed(HARMONY_SAMPLE[60:])
    parser.close()

    assert not any(
        isinstance(chunk, FinalChunk) and chunk.text for chunk in first_pass_chunks
    ), "final text should only appear after reaching final channel body"
    combined = first_pass_chunks + second_pass_chunks + remaining_chunks
    assert any(isinstance(chunk, ToolCallChunk) for chunk in combined)
