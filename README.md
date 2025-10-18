# Harmony Parser Proxy

This service sits between IDE clients and a GPT-OSS (Harmony-format) backend such as
`llama.cpp` or `vLLM`. It consumes Harmony channel streams from the upstream server,
filters them with the official `openai_harmony` parser, and emits plain OpenAI-style
chat completion responses (JSON or SSE) so that existing OpenAI SDKs and IDE agents run
without modifications.

## Features

- FastAPI HTTP edge that implements `/v1/chat/completions` and proxies directly to an
  upstream OpenAI-compatible server.
- Streaming pipeline that normalises upstream Harmony deltas into RFC-compliant Server-
  Sent Events. `analysis`/`commentary` channels are removed; only `final` (and optional
  tool calls) reach the client.
- Uses the `openai_harmony.StreamableParser` for robust token-level
  parsing. If Harmony output is malformed, the proxy heuristically recovers the last
  `final` channel so IDEs still receive a usable answer.
- Optional tool-call bridges:
  - `final_only` (default): discard Harmony tool calls entirely.
  - `final_plus_tools_text`: renders tool calls as human-readable bracket blocks.
  - `openai_tool_calls`: maps Harmony tool calls to OpenAI `tool_calls` objects.
- Supports timeout guards, Harmony stop tokens (`<|return|>`, `<|call|>`), and extra
  stop sequences via configuration.
- Resilient upstream adapter with bounded retries/backoff and a `/readyz` endpoint that
  confirms upstream reachability.
- Prometheus metrics exposed at `/metrics` including request counts, forwarded chunks,
  tool-call usage, and parser fallbacks.

## Getting Started

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # installs deps plus the harmony_proxy package
python -m harmony_proxy --host 0.0.0.0 --port 8001
# or: uvicorn harmony_proxy.app:app --host 0.0.0.0 --port 8001 --reload
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # installs deps plus the harmony_proxy package
python -m harmony_proxy --host 0.0.0.0 --port 8001
# or: uvicorn harmony_proxy.app:app --host 0.0.0.0 --port 8001 --reload
```

> Prefer `python -m harmony_proxy` on Windows to avoid issues with resolving the `uvicorn` console script when the environment is not activated.

### Configuration

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `http://localhost:8080/v1` | Base URL of the upstream OpenAI-compatible server. |
| `PROXY_MODE` | `final_only` | Output mode: `final_only`, `final_plus_tools_text`, `openai_tool_calls`. |
| `HARMONY_ENCODING_NAME` | `HarmonyGptOss` | Harmony encoding to initialise the parser with. |
| `CONNECT_TIMEOUT` | `10.0` | Connection timeout for upstream requests (seconds). |
| `READ_TIMEOUT` | `120.0` | Read timeout for upstream responses (seconds). |
| `NO_FINAL_TIMEOUT` | `90.0` | Aborts streaming responses if no `final` channel token is seen in this window. |
| `MAX_RETRIES` | `3` | Maximum number of retry attempts for upstream requests (streaming and batched). |
| `RETRY_BACKOFF_SECONDS` | `0.5` | Initial backoff delay (seconds) for retries; doubles on each attempt. |
| `HARMONY_STOPS_ENABLED` | `true` | Automatically add `<|return|>` and `<|call|>` stop sequences. |
| `EXTRA_STOP_SEQUENCES` | *(empty)* | Comma-separated list of extra stop strings. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `METRICS_ENABLED` | `true` | Toggle Prometheus metrics collection and `/metrics`. |
| `PREPEND_MISSING_HARMONY_START` | `true` | When true, injects `<|start|>assistant` if upstream omits it (workaround for non-compliant streams). |

Environment variables can be provided via `.env`, exported in the shell, or using
PowerShell's `setx`/`$env:` for Windows users.

### Installing Prebuilt Wheels from Artifactory

If your team publishes releases to Artifactory (see `BUILDING.md`), install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
  --index-url https://artifactory.example.com/artifactory/api/pypi/harmony-pypi/simple \
  --extra-index-url https://pypi.org/simple \
  harmony-proxy==<desired-version>
```

Replace `https://artifactory.example.com/artifactory/api/pypi/harmony-pypi/simple`
with your repository URL. The `--extra-index-url` flag keeps the standard PyPI
available for transitive dependencies that are not mirrored in Artifactory.

## Development

- `python -m compileall src` ensures syntax correctness.
- Use `uvicorn harmony_proxy.app:app --reload` during development.
- Extend `UpstreamClient` if you need custom authentication headers or retries.

## Known Limitations

- The proxy requires the `openai_harmony` package at runtime. Without it, the parser import fails.
- Only the `/v1/chat/completions` endpoint is normalised. `/v1/responses` is forwarded
  untouched for compatibility testing.
