from harmony_proxy.parser import FinalChunk, HarmonyStreamParser, ToolCallChunk


def test_parser_emits_final_chunks_incrementally():
    parser = HarmonyStreamParser()
    segments = [
        "<|start|>assistant",
        "<|channel|>final",
        "<|message|>Hel",
        "lo<|return|>",
    ]

    chunks = []
    for piece in segments:
        chunks.extend(parser.feed(piece))
    chunks.extend(parser.close())

    final_text = "".join(
        chunk.text for chunk in chunks if isinstance(chunk, FinalChunk) and chunk.text
    )
    assert final_text == "Hello"
    assert any(
        isinstance(chunk, FinalChunk) and chunk.is_terminal for chunk in chunks
    ), "terminal marker missing"


def test_parser_extracts_tool_calls():
    parser = HarmonyStreamParser()
    sample = (
        "<|start|>assistant"
        "<|channel|>commentary to=read_file"
        "<|message|>{\"path\":\"foo.txt\"}"
        "<|call|>"
    )

    chunks = parser.feed(sample)
    chunks.extend(parser.close())

    tool_chunks = [chunk for chunk in chunks if isinstance(chunk, ToolCallChunk)]
    assert len(tool_chunks) == 1
    tool_chunk = tool_chunks[0]
    assert tool_chunk.name == "read_file"
    assert tool_chunk.arguments == '{"path":"foo.txt"}'


def test_parser_injects_start_token_when_missing():
    parser = HarmonyStreamParser()
    sample = "<|channel|>final<|message|>Injected<|return|>"

    chunks = parser.feed(sample)
    chunks.extend(parser.close())

    final_text = "".join(
        chunk.text for chunk in chunks if isinstance(chunk, FinalChunk) and chunk.text
    )
    assert final_text == "Injected"
    assert not parser.had_error
