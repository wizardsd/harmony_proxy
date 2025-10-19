# Fix KiloCode Streaming Role Emission for Tool Calls

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

Refer to `.agent/PLANS.md` for the governing rules; this plan adheres to those requirements.

## Purpose / Big Picture

KiloCode clients currently surface `Error The model's response ended unexpectedly (no assistant messages)` when the upstream Harmony stream contains only tool call content and no final text. By ensuring the proxy emits an explicit assistant-role delta before forwarding tool call chunks, KiloCode will always see a well-formed assistant message and can continue its tool workflow without interruption. The fix will be demonstrated via a regression test that reproduces the traced payload and confirms the emitted SSE sequence contains the role delta.

## Progress

- [x] (2025-10-19 18:20Z) Reviewed `trace.log` and `_streaming_response` to confirm the role delta is only sent for `FinalChunk`, explaining the missing assistant messages for tool-only streams.
- [x] (2025-10-19 18:46Z) Added regression test for tool-only Harmony stream and confirmed it fails against current implementation.
- [x] (2025-10-19 19:01Z) Updated `_streaming_response` to emit the assistant role before tool chunks in both streaming and flush paths.
- [x] (2025-10-19 19:04Z) Re-ran the regression test and nearby suites to cover the new behavior.
- [x] (2025-10-19 19:05Z) Verified clean pytest runs without fallback warnings, confirming the regression is closed.
- [x] (2025-10-19 19:22Z) Investigated trace evidence that non-streaming tool-only replies keep `finish_reason` as `"stop"`, causing KiloCode to retry and eventually fail.
- [x] (2025-10-19 19:22Z) Adjusted finish-reason normalization (streaming and non-streaming) and extended regression tests to ensure `"tool_calls"` is emitted when only tool calls are forwarded.
- [x] (2025-10-19 19:35Z) Suppressed fallback-only natural language finals when no markup is present, preventing premature status blurbs from reaching KiloCode while retaining XML snippets needed for tool execution.
- [x] (2025-10-19 19:47Z) Restored KiloCode-visible XML tool directives in `message.content` while still emitting structured `tool_calls`, so the client sees compliant markup alongside the JSON tool metadata.
- [x] (2025-10-19 19:55Z) Updated integration coverage to expect XML content plus `tool_calls` for full trace sequences, aligning tests with the proxy's current KiloCode normalization.

## Surprises & Discoveries

- Observation: `_streaming_response` only sends `{"role": "assistant"}` when a `FinalChunk` arrives or a fallback kicks in; tool-only sequences bypass that code path, so KiloCode never receives an assistant delta.
  Evidence: Manual inspection of `src/harmony_proxy/app.py` showed the `role_sent` guard located inside the `FinalChunk` branch, leaving `_emit_tool_chunk` unreachable for role emission.
- Observation: Replay of the tool-only fixture triggers the parser fallback path, confirming the proxy currently tries to synthesize final text when none exists.
  Evidence: Pytest run logs included `Harmony parser failed on token 200007` followed by the proxy warning about heuristic FINAL extraction.
- Observation: After emitting the role delta ahead of tool chunks, the regression test completes without fallback warnings, indicating the proxy no longer tries to synthesize final text.
  Evidence: `PYTHONPATH=src python -m pytest tests/test_app_trace.py::test_streaming_tool_call_emits_role_delta` finished green with no warning logs.
- Observation: Trace `chatcmpl-2f1425acbf984a6fb32942bda06b0d41` shows the proxy forwarding a tool-only message with `finish_reason: "stop"`, prompting the client to issue identical retries and eventually raise an error despite the `tool_calls` array being present.
  Evidence: `trace.log` entries reveal consecutive `proxy_response` payloads where `message.content == ""`, `tool_calls` exists, and `finish_reason` remains `"stop"`.
- Observation: Updated traces reveal Harmony sometimes emits a plain-language status message without markup when only a tool call is intended, and forwarding that text confuses KiloCode's tool workflow.
  Evidence: `chatcmpl-bb7ee1027715470ea4dff56400c2f971` shows fallback-derived final text “The task is now in progress…” alongside a single tool call; suppressing such markup-free fallbacks prevents spurious status messages.
- Observation: Latest trace `chatcmpl-237d3e21867a4f888d529f0fb3c316c5` shows the proxy returning an empty `content` field despite a required `<update_todo_list>` directive, because the XML snippet was lifted into `tool_calls`; KiloCode still expects the textual XML to be present in the message body.
  Evidence: Trace payload records `message.content==""` with `tool_calls` populated, indicating our normalization removed the XML block entirely.
- Observation: After reinstating XML snippets for plain tool chunks in KiloCode mode, the proxy again surfaces `<update_todo_list>` while keeping `tool_calls` populated, satisfying both JSON and textual contracts.
  Evidence: Unit tests now confirm `message.content` includes `<codebase_search>` for reasoning-driven tool calls in `ProxyMode.KILOCODE`.

## Decision Log

- Decision: Fix the issue within `_streaming_response` by injecting a role delta when emitting the first tool chunk rather than altering `_emit_tool_chunk`.
  Rationale: Keeps role handling local to stream orchestration, avoids coupling `_emit_tool_chunk` to stateful concerns, and covers both live streaming and parser flush paths.
  Date/Author: 2025-10-19 / Codex
- Decision: Normalize tool-only completions to emit `finish_reason="tool_calls"` so downstream agents detect the pending tool invocation even when Harmony omits a final channel.
  Rationale: Matches OpenAI semantics, resolves the repeated retry observed in traces, and keeps compatibility for responses that include both final text and tool calls.
  Date/Author: 2025-10-19 / Codex
- Decision: Only surface fallback-derived final text when it carries structured markup (e.g., XML snippets); suppress plain-language fallbacks so tool-only turns remain silent until tool results arrive.
  Rationale: Keeps informative snippets needed for tool execution while preventing misleading progress blurbs from appearing before the tool completes.
  Date/Author: 2025-10-19 / Codex
- Decision: Reintroduce serialized XML tool directives into `content` for KiloCode mode while preserving deduped `tool_calls`, ensuring downstream parsers that rely on textual markup continue to function.
  Rationale: Allows both OpenAI-style consumers (via `tool_calls`) and KiloCode’s XML parser to succeed without double-counting tool executions.
  Date/Author: 2025-10-19 / Codex

## Outcomes & Retrospective

Tool-only Harmony streams now begin with an assistant-role delta and complete with `finish_reason="tool_calls"`, allowing KiloCode to process tool workflows without raising errors or misclassifying pending tool invocations. The regression tests guard both streaming and non-streaming paths, and the surrounding suites remain green. No additional configuration or interface changes were required, and the proxy avoids unnecessary fallback parsing in these scenarios.

## Context and Orientation

`src/harmony_proxy/app.py` drives FastAPI endpoints and the streaming bridge; `_streaming_response` funnels Harmony deltas through `HarmonyStreamParser` from `src/harmony_proxy/parser.py`. Tool call chunks surface as `ToolCallChunk` instances. KiloCode sets `PROXY_MODE=kilocode`, which strips channel prefixes but depends on OpenAI-compatible SSE framing. Current logic only emits the assistant role when a `FinalChunk` arrives, so sessions that contain only `<|channel|>commentary ... <|call|>` never announce the assistant role. Tests live under `tests/`, with `tests/test_app_trace.py` already exercising streaming traces via `DummyUpstream`. This plan targets those modules to adjust role emission and codify the regression.

## Plan of Work

First, craft a failing async test (likely in `tests/test_app_trace.py`) that feeds `_streaming_response` an SSE event representing a Harmony tool call only. The test should assert that at least one emitted SSE payload contains `delta.role == "assistant"` before the tool call delta; without the fix it will fail. Next, modify `_streaming_response` in `src/harmony_proxy/app.py` so that, whenever a `ToolCallChunk` is about to be forwarded—both during the main streaming loop and in the `finally` flush—it emits the assistant role if it has not already done so. Ensure the same guard applies when fallback text is absent. Finally, rerun the new test and any focused suites, updating the plan as results arrive.

With the latest traces showing `finish_reason: "stop"` on tool-only replies, extend the plan to adjust both normalization paths. Update `_normalize_non_streaming` to override the finish reason when tool calls are emitted without final text, and tweak `_streaming_response` so the final SSE event reports `"tool_calls"` whenever only tool chunks were forwarded. Add regression coverage that mimics the observed trace (non-streaming reasoning-only payload) and a streaming test where the upstream finish reason is `"stop"` yet the proxy emits a tool call.

## Concrete Steps

1. Ensure dependencies are available inside the virtual environment: `.venv/bin/pip install -r requirements.txt` and `.venv/bin/pip install pytest` (run once per environment).
2. Implement the regression test in `tests/test_app_trace.py`, using `DummyUpstream` to replay the Harmony tool-call-only payload captured from `trace.log`.
3. Run `PYTHONPATH=src python -m pytest tests/test_app_trace.py::test_streaming_tool_call_emits_role_delta` to observe the pre-fix failure and collect fallback warnings.
4. Update `_streaming_response` so it emits the assistant role before forwarding tool call chunks during streaming and flush-time parsing.
5. Re-run the targeted pytest, then execute `PYTHONPATH=src python -m pytest tests/test_app_trace.py tests/test_app_utils.py` (or the `.venv`-scoped equivalent) to confirm nearby behaviors stay intact.
6. Introduce a non-streaming regression in `tests/test_app_utils.py` that mirrors the recorded `reasoning_content`-only payload, asserting that `finish_reason` becomes `"tool_calls"` and the tool call is exposed.
7. Adjust `tests/test_app_trace.py::test_streaming_tool_call_emits_role_delta` to feed an upstream finish reason of `"stop"` and expect the proxy to emit `"tool_calls"` after the fix.
8. Modify `_normalize_non_streaming` and `_streaming_response` per the updated plan, then rerun the focused pytest commands from steps 3 and 5 to validate the new behavior.
9. Extend normalization so that, in KiloCode mode, the textual `<tool>` XML fragment (or synthesized openai-style block) is placed back into `message.content` alongside a concise status, while continuing to emit structured `tool_calls`.
10. Add regression tests for both non-streaming and streaming paths ensuring `content` contains the XML snippet and meets the KiloCode formatting rules when tool calls are present.

## Validation and Acceptance

Acceptance requires the new regression test to pass, demonstrating that an SSE sequence with only tool call content now includes an assistant role delta. Additionally, the focused pytest subset must succeed without modifying unrelated behavior, ensuring non-tool streaming still works. Manual verification can inspect the first SSE payload emitted by `_streaming_response` to confirm the role field is present.

## Idempotence and Recovery

The test harness is deterministic: rerunning the pytest command replays the same stubbed upstream events. If edits introduce syntax errors, revert the specific changes in `src/harmony_proxy/app.py` and the new test file portions; no database or external side effects occur. The plan can be resumed by re-running the targeted pytest until it passes.

## Artifacts and Notes

Keep the key SSE payload from the regression test handy to compare pre- and post-fix behavior, noting the addition of `{"delta":{"role":"assistant"}}`. Capture any pytest failure output in this section as implementation progresses.

- Latest confirmation:

        PYTHONPATH=src python -m pytest tests/test_app_trace.py::test_streaming_tool_call_emits_role_delta
        ============================= test session starts ==============================
        tests/test_app_trace.py .                                                [100%]

- Additional regression sweep:

        PYTHONPATH=src python -m pytest tests/test_app_trace.py tests/test_app_utils.py
        ============================= test session starts ==============================
        tests/test_app_trace.py ..                                               [ 11%]
        tests/test_app_utils.py ................                                 [100%]

- Virtualenv verification:

        .venv/bin/python -m pytest tests/test_app_trace.py tests/test_app_utils.py
        ============================= test session starts ==============================
        tests/test_app_trace.py ..                                               [ 11%]
        tests/test_app_utils.py ................                                 [100%]

- KiloCode XML restoration check:

        .venv/bin/python -m pytest tests/test_app_utils.py::test_normalize_non_streaming_sets_tool_finish_reason_from_reasoning
        ============================= test session starts ==============================
        tests/test_app_utils.py .                                                [100%]

- Full-trace expectation update:

        .venv/bin/python -m pytest tests/test_integration_llm.py::test_llm_kilocode_full_trace_sequence
        ============================= test session starts ==============================
        tests/test_integration_llm.py .                                          [100%]

## Interfaces and Dependencies

`HarmonyStreamParser.feed` and `.close` already surface `ToolCallChunk` instances; we rely on that behavior unchanged. The interface to `_emit_tool_chunk` remains the same, accepting `(chunk, mode, build_delta, tool_indices, endpoint_name)` and returning byte chunks. No external dependencies change; FastAPI and Prometheus integrations remain untouched.

---
Plan created on 2025-10-19 by Codex to address missing assistant role emission for tool-only Harmony streams.
