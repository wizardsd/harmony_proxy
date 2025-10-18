from harmony_proxy import metrics


def test_observe_request_increments_counter():
    metrics.observe_request("chat_completions", True, "final_only")
    output = metrics.render_metrics().decode()
    assert 'harmony_proxy_requests_total{endpoint="chat_completions",mode="final_only",stream="true"} 1.0' in output


def test_metrics_disabled_no_increments():
    metrics.configure(False)
    metrics.observe_request("chat_completions", True, "final_only")
    metrics.observe_final_chunk("chat_completions", True)
    metrics.observe_tool_chunk("chat_completions", "final_plus_tools_text")
    metrics.observe_parser_fallback("chat_completions")
    metrics.observe_retry()
    output = metrics.render_metrics().decode()
    assert "harmony_proxy_requests_total" not in output or " 0.0" in output
