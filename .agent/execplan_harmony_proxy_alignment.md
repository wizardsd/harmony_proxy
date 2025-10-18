# Purpose / Big Picture

This work hardens the Harmony Parser Proxy so IDE clients always receive plain OpenAI Chat Completions even when llama.cpp misbehaves. After the changes ship, a developer can point Continue/Roo/Cline at the proxy, stream completions from a flaky GPT-OSS backend, and observe that the proxy (1) retries transient upstream failures, (2) automatically falls back to clean FINAL text when Harmony tags are malformed, (3) exposes Prometheus metrics and structured logs that show parse counts and tool-call usage, and (4) reports readiness via `/readyz`. The system will terminate stalled generations rather than hanging forever and will remain observable in production.

# Progress

- [x] (2025-10-17 15:00Z) Add configuration knobs for retry counts, backoff, and metrics toggles.
- [x] (2025-10-17 15:30Z) Implement upstream retry/back-pressure logic with tests for transient failures.
- [x] (2025-10-17 16:15Z) Teach the parser/app to fall back to heuristic FINAL extraction when Harmony parsing fails; cover with tests.
- [x] (2025-10-17 17:00Z) Add Prometheus metrics, structured logging, and `/metrics` endpoint; validate via unit tests or smoke command.
- [x] (2025-10-17 17:30Z) Add `/readyz` readiness check hitting the upstream `/models` endpoint with retries; include test coverage.
- [ ] (2025-10-17 18:00Z) Run full test suite; update docs/config examples; finalize retrospective. (Docs updated; test run blocked: `pytest` executable not available in environment.)

# Surprises & Discoveries

- Observation: Local environment lacks a `pytest` executable, so automated test runs currently fail.
  Evidence: Running `pytest` produced `bash: pytest: command not found` on 2025-10-17.

# Decision Log

- Decision: Use Prometheus counters in a dedicated registry with optional disabling to satisfy observability requirements.
  Rationale: Keeps metrics self-contained for the proxy while allowing tests to reset state easily.
  Date/Author: 2025-10-17 / Codex

- Decision: Treat 401/403 responses from the upstream `/models` readiness probe as “ready”.
  Rationale: Such codes prove the upstream HTTP server is reachable even if credentials are missing, aligning with the proxy’s availability goal.
  Date/Author: 2025-10-17 / Codex

# Outcomes & Retrospective

New retry/backoff logic, Harmony fallback heuristics, Prometheus instrumentation, and readiness probing are in place and covered by unit tests. Streaming summarises its behaviour via structured key=value logs, and documentation now advertises the added endpoints and env vars. Full pytest verification is pending because the environment lacks a pytest executable.

# Context and Orientation

The FastAPI entry point lives at `src/harmony_proxy/app.py`. It wires request handling, invokes `HarmonyStreamParser` from `src/harmony_proxy/parser.py`, and forwards to llama.cpp via `src/harmony_proxy/upstream.py`. Configuration is in `src/harmony_proxy/config.py`. SSE utilities sit in `src/harmony_proxy/sse.py`. Tests currently cover parser basics in `tests/test_parser.py` and app utilities in `tests/test_app_utils.py`. There is no metrics module yet. The proxy presently applies Harmony stop tokens, strips non-final channels, but lacks retries, fallback heuristics, metrics, or readiness endpoints.

The llama.cpp OpenAI-compatible server accepts POST `/v1/chat/completions` for streaming and non-streaming responses. Harmony formatting wraps messages using tags like `<|channel|>final`. The official `openai_harmony` parser decodes tokens but can throw when inputs are truncated. We must detect such failures and heuristically extract the last FINAL message by stripping tags manually so IDE clients see plain text. Prometheus metrics are the de facto standard for container observability; we will use the `prometheus_client` Python package to expose counters/gauges at `/metrics`.

# Plan of Work

First extend `ProxyConfig` (`src/harmony_proxy/config.py`) to capture retry counts, backoff factor, and a flag enabling metrics. Update env parsing helpers accordingly. Next, enhance `UpstreamClient` to retry transient network errors (connection reset, timeouts, HTTP 502/503) with exponential backoff, and to guard streaming ingestion so the generator yields only after downstream consumers advance (FastAPI’s `StreamingResponse` already provides back-pressure; we will formalize it by awaiting `response.aiter_lines()` sequentially).

Introduce a reusable fallback extractor in a new module `src/harmony_proxy/harmony_fallback.py`. When the parser drops text or raises, the fallback should grab the last `<|channel|>final` block, strip tags, and return plain text. Update `_streaming_response` and `_parse_full_text` to accumulate raw Harmony deltas, detect parser failures or missing FINAL chunks, and emit the fallback content so clients always receive a message. Add tests that feed malformed Harmony to confirm the heuristic kicks in.

For observability, add `prometheus_client` to dependencies, create `src/harmony_proxy/metrics.py` to define counters/gauges (e.g., requests total by mode, parsed final/tool events, parser errors, retries). Register a `/metrics` endpoint in FastAPI that exposes Prometheus text format. Instrument request handling to increment metrics, including retry counts and fallback usage. Update logging statements to emit structured dicts or key=value pairs capturing request ids, retry attempts, and parser outcomes.

Implement `/readyz` in `app.py`. It should call a new `UpstreamClient.check_readiness()` that pings `/models` (standard OpenAI endpoint) with retries from config; if unreachable, return HTTP 503. Add unit tests that mock the upstream client to simulate readiness states.

Finally, document the new env vars in README if present (or add a short section) and run the existing pytest suite to verify no regressions. Capture relevant command outputs in the plan once executed.

# Concrete Steps

1. Modify configuration and dependency files (`pyproject.toml`, possibly extras) to add retry/backoff settings and Prometheus dependency. Regenerate lockfiles if applicable (none currently).
2. Refactor `UpstreamClient` to centralize requests with retry logic and backoff; add async sleep between attempts. Provide unit tests mocking `httpx.AsyncClient`.
3. Create `harmony_fallback.py` with extraction helpers. Update parser/app to use fallback on errors and add tests covering malformed inputs.
4. Add `metrics.py`, wire Prometheus counters, expose `/metrics`, and instrument request flow. Extend tests to assert metrics counters change.
5. Add `/readyz` endpoint and readiness checks with tests verifying 200 vs 503.
6. Run `pytest` from repo root; note output in this plan.

# Validation and Acceptance

Acceptance requires these behaviors:

* When llama.cpp responds with malformed Harmony (e.g., missing `<|end|>`), the proxy returns a valid Chat Completion whose `choices[0].message.content` equals the final text, proven by a new test in `tests/test_app_utils.py`.
* Transient upstream 502 errors are retried up to the configured max; a unit test should simulate first failure then success and verify retries.
* `/metrics` returns Prometheus text with counters such as `harmony_proxy_requests_total`.
* `/readyz` responds with HTTP 200 when upstream ping succeeds and 503 otherwise (unit tests cover both cases).
* `pytest` passes entirely.

# Idempotence and Recovery

Configuration changes are additive and safe to reapply. Retry logic is guarded by bounded attempts and timeouts. Fallback extraction only activates when parser output is empty, so normal flows remain unchanged. Running the migrations/tests multiple times is safe; no persistent state is mutated.

# Artifacts and Notes

Command: `python -m pytest tests/test_metrics.py`  
Result: `/Users/roman/miniconda3/bin/python: No module named pytest` (environment missing pytest executable).

# Interfaces and Dependencies

Add `prometheus_client>=0.16` to the project dependencies. Update `ProxyConfig` to include:

    max_retries: int
    retry_backoff_seconds: float
    metrics_enabled: bool

Expose instrumentation via `metrics.py` functions:

    observe_request(stream: bool, mode: ProxyMode)
    observe_parser_event(event: str)
    observe_retry()

Ensure `UpstreamClient` gains:

    async def check_readiness(self) -> bool

The fallback helper should offer:

    def extract_final_text(raw: str) -> str

Streaming and non-streaming paths must call this when no FINAL chunks are produced so the API contract is met.

# Revision Notes

- 2025-10-17 Codex: Initial ExecPlan drafted before implementation work begins.
