# Debugging Playbook for `harmony_proxy`

## Fast Recon
- **Inspect live traces:** `tail -n 200 trace.log` for recent requests; use `jq`/`python -m json.tool` to prettify payloads.
- **Extract payloads quickly:** `python - <<'PY'` snippets that load `trace.log` and print `payload["messages"]` for reproducible test fixtures.
- **Diff config vs. runtime:** `cat .env` or the `_load_dotenv()` helper to confirm `UPSTREAM_BASE_URL`, `PROXY_MODE`, and other overrides.

## Reproducing Upstream Behaviour
- **Stubbed FastAPI flow:** wrap the app with `TestClient` and inject a custom `StubUpstream` to replay trace snippets. Handy template:
  ```python
  from fastapi.testclient import TestClient
  from harmony_proxy.app import app

  class StubUpstream:
      async def chat_completions(self, payload, trace_id=None):
          return responses.pop(0)
      async def stream_chat_completions(self, payload, trace_id=None):
          raise AssertionError("streaming not expected")
      async def aclose(self): pass

  client = TestClient(app)
  client.app.state.upstream = StubUpstream()
  resp = client.post("/v1/chat/completions", json=payload)
  ```
- **Direct upstream probe:** when network access is allowed, call the real model with `UpstreamClient` to validate raw responses.

## Parser & Normaliser Checks
- **Token-level issues:** run `_parse_full_text()` directly in a REPL with suspicious content to see how Harmony parsing behaves.
- **Plain XML tool detection:** feed strings like `<codebase_search>...</codebase_search>` into `_detect_plain_tool_chunks()` to ensure fallbacks trigger.
- **Final normalisation:** compare `_normalize_non_streaming()` output for different `ProxyMode` values when adding new fields (e.g., `thinking`).

## Testing Workflow
- **Focused unit tests:** extend `tests/test_app_utils.py` with crafted Harmony strings for specific edge cases (missing `<|start|>`, duplicated XML, etc.).
- **Integration snapshots:** add new scenarios to `tests/test_integration_llm.py` mirroring `trace.log` sequences so regressions are caught automatically.
- **Full suite check:** `PYTHONPATH=src .venv/bin/pytest tests/test_app_utils.py tests/test_integration_llm.py` for quick validation.

## Useful One-Liners / Snippets
- Pretty-print trace events:
  ```bash
  python - <<'PY'
  import json
  with open("trace.log") as fh:
      for line in fh:
          evt = json.loads(line)
          if evt["event"] == "proxy_response":
              print(json.dumps(evt, indent=2)[:1000])
  PY
  ```
- Rapid reload of app module to pick up config changes:
  ```python
  import harmony_proxy.app as app_module
  from importlib import reload
  reload(app_module)
  ```
- Mock multiple upstream replies:
  ```python
  responses = [first_response_json, second_response_json, ...]
  client.app.state.upstream = StubUpstream(responses)
  for _ in responses:
      client.post("/v1/chat/completions", json=payload)
  ```

## When Adding Features
- **Trace first:** reproduce the issue via trace-driven tests *before* patching.
- **Document edge cases:** note token IDs, unexpected channels, or missing metadata in the new test fixture comments.
- **Keep scripts nearby:** if a REPL snippet proved useful, drop it here for reuse rather than reinventing it later.

