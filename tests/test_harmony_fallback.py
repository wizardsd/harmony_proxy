from harmony_proxy.harmony_fallback import extract_final_text


def test_extract_final_text_recovers_truncated_final():
    raw = (
        "<|start|>assistant"
        "<|channel|>final"
        "<|message|>Heuristic Hello"
    )
    assert extract_final_text(raw) == "Heuristic Hello"


def test_extract_final_text_strips_tags_when_missing_final():
    raw = (
        "<|start|>assistant"
        "<|channel|>analysis"
        "<|message|>thinking"
        "<|end|>"
    )
    assert extract_final_text(raw) == "thinking"
