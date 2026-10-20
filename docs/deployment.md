# Deploying the HTTP service

`structured-output-agent` ships an optional HTTP service that exposes
schema-enforced, validated and retried model runs to any language. This guide
covers how it is built, how to configure, run, secure, observe and scale it,
and what to do when it misbehaves.

- Install: `pip install 'structured-output-agent[server,anthropic]'`
- Run: `structured-agent serve --host 0.0.0.0 --port 8000`
  (or `python -m structured_output_agent.server`)
- Container: `ghcr.io/imranmali/structured-output-agent` serves the API by default.

## Architecture

```
            clients
               |  HTTPS (TLS terminated at the ingress / load balancer)
        +------v-------+
        | LB / Ingress |  retries 503s on another replica
        +--+--------+--+
           |        |
     +-----v--+  +--v-----+      each replica: 1 uvicorn worker, 1 event loop,
     | pod 1  |  | pod N  |  ... 1 AsyncStructuredOutputAgent, up to
     +---+----+  +---+----+      SOA_MAX_IN_FLIGHT concurrent runs
         |           |
         +-----+-----+
               v
        Claude Messages API
```

- **Stateless replicas.** A replica keeps no state between requests other
  than metrics and a bounded in-memory window of failure events. Any replica
  can serve any request, so you scale by adding replicas (Kubernetes HPA on
  CPU) and callers own retries across replicas. There is no job queue: a run
  lives exactly as long as its HTTP request.
- **One worker per container.** `serve` runs uvicorn with a single worker
  process. Async I/O gives concurrency inside the process; replicas give
  parallelism across cores and machines. This keeps Prometheus metrics
  per-process and correct without multi-process mode.
- **Load shedding instead of queueing.** Each replica counts runs in flight
  with a non-blocking counter. When `SOA_MAX_IN_FLIGHT` is reached, new run
  and batch requests get `503` with `Retry-After: 1` immediately rather than
  waiting in an unbounded queue, so latency stays bounded, the load balancer
  or client retries elsewhere and the HPA sees the pressure.
- **Two retry layers.** Transient provider failures (rate limits, overload,
  5xx, timeouts, connection errors) are retried per model call with capped,
  jittered exponential backoff honouring `Retry-After`
  (`SOA_PROVIDER_RETRIES`). Invalid model *output* is repaired by re-prompting
  the model with itemised errors (`SOA_MAX_ATTEMPTS`). The Anthropic SDK's own
  retries are disabled so the layers do not multiply.
- **Lazy provider client.** The Claude client is created on the first run, so
  the process starts and answers probes without credentials. A warning is
  logged at startup when `ANTHROPIC_API_KEY` is not set.
- **Graceful shutdown.** On `SIGTERM` uvicorn stops accepting connections and
  drains in-flight requests for up to 30 seconds; readiness reports
  `shutting down` and the provider client is closed. In Kubernetes a `preStop`
  sleep keeps the pod serving while it is removed from Service endpoints.

## Quick start

```bash
# Docker (the demo contract; replace SOA_MODELS with your own models)
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e SOA_API_KEYS="$(openssl rand -hex 32)" \
  -e SOA_MODELS=structured_output_agent.cli:WeatherReport \
  ghcr.io/imranmali/structured-output-agent:latest

# The offline demo still works from the same image
docker run --rm ghcr.io/imranmali/structured-output-agent:latest demo

# docker compose (see "Docker Compose" below)
cp deploy/.env.example deploy/.env && $EDITOR deploy/.env
docker compose -f deploy/docker-compose.yml up --build

# Kubernetes (see "Kubernetes" below)
kubectl create secret generic structured-output-agent \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --from-literal=SOA_API_KEYS="$(openssl rand -hex 32)"
kubectl apply -k deploy/kubernetes
```

Your output contracts are ordinary Pydantic models. To serve them, install the
package that defines them into the image (for example
`FROM ghcr.io/imranmali/structured-output-agent` plus
`RUN pip install your-models`, or your own image that installs both) and list
them in `SOA_MODELS`. Contracts are addressed by class name, so two models with
the same class name cannot be served together.

## Configuration reference

All configuration comes from environment variables (`ServerSettings.from_env`).
Unset or empty variables keep their defaults. Invalid values stop the process
at startup with an error naming the variable. `SOA_MODELS` and `SOA_TOOLS` are
imported at startup too, so a typo fails fast rather than on the first request.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOA_MODELS` | *(none)* | Comma-separated `module:Class` Pydantic models to serve. Addressed by class name. With none, `/readyz` fails. |
| `SOA_TOOLS` | *(none)* | Optional `module:attribute` of a `ToolRegistry` whose tools the model may call. |
| `SOA_LLM_MODEL` | `claude-opus-5-5` | Claude model id. |
| `SOA_LLM_TIMEOUT` | `120` | Seconds allowed for one provider call (must be > 0). |
| `SOA_PROVIDER_RETRIES` | `3` | Retries of transient provider failures per model call (`0` disables). |
| `SOA_MAX_ATTEMPTS` | `3` | Generation attempts per run: the first answer plus repairs of invalid output. |
| `SOA_RUN_TIMEOUT` | `300` | Deadline for a whole run in seconds, including retries and tools; `0` disables it. |
| `SOA_MAX_IN_FLIGHT` | `32` | Run slots per replica; beyond it requests are shed with `503`. |
| `SOA_MAX_PROMPT_CHARS` | `100000` | Maximum characters in a prompt or system prompt (`422` beyond). |
| `SOA_MAX_BATCH` | `50` | Maximum prompts in one batch request (`422` beyond). |
| `SOA_BATCH_CONCURRENCY` | `8` | Prompts of one batch run concurrently (also capped by `SOA_MAX_IN_FLIGHT`). |
| `SOA_STRICT` | `false` | Forbid unknown fields in model output by default (requests can override). Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`. |
| `SOA_SYSTEM_PROMPT` | *(none)* | Default system prompt placed before the contract instructions (requests can override). |
| `SOA_API_KEYS` | *(none)* | Comma-separated API keys accepted on `/v1/*`. Empty disables authentication. Never logged or shown in `repr`. |
| `SOA_LOG_FORMAT` | `json` | `json` (one JSON object per line) or `text`. The image sets `json`. |
| `SOA_LOG_LEVEL` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, ...). `DEBUG` adds probe access logs. |
| `SOA_FAILURE_LOG` | *(none)* | Optional JSON lines file receiving every validation failure event (redacted). |
| `SOA_REDACT_EMAILS` | `true` | Replace email addresses with `[REDACTED]` in failure events (memory, file and logs). |
| `SOA_MAX_EVENTS` | `1000` | Failure events kept in memory per replica (oldest dropped first). |

Provider credentials are read by the Anthropic SDK itself: set
`ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`), and optionally
`ANTHROPIC_BASE_URL` for a gateway.

Programmatic use: `create_app(settings=None, *, llm=None, models=None,
tools=None, llm_factory=None)` builds the FastAPI app; pass `models`/`tools`
to bypass `SOA_MODELS`/`SOA_TOOLS`, `llm` to inject any `AsyncLLMClient` (it
is owned by you and never closed by the app) or `llm_factory(settings)` to
build one lazily. Run the app with any ASGI server, one worker per process.

## API reference

Interactive docs are served at `/docs` (OpenAPI at `/openapi.json`). With
authentication enabled every `/v1/*` request needs
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

### `POST /v1/runs`

```bash
curl -sS http://localhost:8000/v1/runs \
  -H "Authorization: Bearer $SOA_KEY" -H "Content-Type: application/json" \
  -d '{"model": "WeatherReport", "prompt": "Weather report for Paris"}'
```

Body: `model` (contract name, required), `prompt` (required, 1 to
`SOA_MAX_PROMPT_CHARS` characters), `system_prompt` (optional override),
`strict` (optional override). Unknown fields are rejected.

`200` response:

```json
{
  "run_id": "4f0c2d0e9b7a4e3c8f1a2b3c4d5e6f70",
  "model": "WeatherReport",
  "output": {"city": "Paris", "temperature_c": 18.5, "conditions": "light rain",
             "advice": "Take an umbrella."},
  "attempts": 2,
  "failures": [
    {"attempt": 1, "kind": "validation_error",
     "errors": [{"loc": "advice", "msg": "Field required", "type": "missing"}]}
  ],
  "tool_calls": 0,
  "usage": {"input_tokens": 812, "output_tokens": 64, "total_tokens": 876},
  "duration_ms": 2380.4
}
```

`failures` lists the attempts that were repaired. Raw model output and the
offending input values are deliberately not returned; they are available,
redacted, in the failure log.

### `POST /v1/batches`

```bash
curl -sS http://localhost:8000/v1/batches \
  -H "Authorization: Bearer $SOA_KEY" -H "Content-Type: application/json" \
  -d '{"model": "WeatherReport", "prompts": ["Paris", "Oslo"]}'
```

Body: `model`, `prompts` (1 to `SOA_MAX_BATCH` non-empty strings),
`system_prompt`, `strict`. The response is always `200` once the batch ran,
with one entry per prompt in input order:

```json
{
  "model": "WeatherReport",
  "results": [
    {"status": "ok", "run_id": "...", "model": "WeatherReport", "output": {...},
     "attempts": 1, "failures": [], "tool_calls": 0, "usage": {...}, "duration_ms": 950.1},
    {"status": "error", "status_code": 422,
     "error": {"type": "retries_exhausted", "message": "...", "attempts": 3, "failures": [...]}}
  ]
}
```

`status_code` is what the same failure would have returned from
`/v1/runs`. **Concurrency accounting:** a batch holds
`min(len(prompts), SOA_BATCH_CONCURRENCY, SOA_MAX_IN_FLIGHT)` in-flight slots
for its whole duration and runs that many prompts at a time, so a replica
never runs more than `SOA_MAX_IN_FLIGHT` model runs at once whatever the mix
of single runs and batches. If those slots are not free the whole batch is
shed with `503`. A batch can take up to
`ceil(len(prompts) / slots) * SOA_RUN_TIMEOUT`; size client and ingress
timeouts accordingly, or send many small batches.

### `GET /v1/models`, `GET /v1/models/{name}`

```bash
curl -sS http://localhost:8000/v1/models -H "X-API-Key: $SOA_KEY"
# {"models": [{"name": "WeatherReport", "schema": {...JSON Schema...}}]}
```

The schema is exactly the one the model is shown (respecting `SOA_STRICT`).

### Operational endpoints (never authenticated)

| Endpoint | Success | Failure |
| --- | --- | --- |
| `GET /healthz` | `200 {"status": "ok"}` while the process serves | - (no response means dead) |
| `GET /readyz` | `200 {"status": "ready", "models": N}` | `503 {"status": "unavailable", "reason": "no models configured" \| "shutting down"}` |
| `GET /metrics` | Prometheus text format | - |

Readiness deliberately does not fail when the replica is saturated: removing
busy pods from the Service would push their load onto the others and flap.
Saturation is handled by load shedding and the HPA.

### Errors

Every error has the same shape, and every response carries `X-Request-ID`:

```json
{"error": {"type": "unknown_model", "message": "Unknown model 'Nope'; available: WeatherReport",
           "model": "Nope"},
 "request_id": "1a686ad4822d4438a07c028f5e1a48da"}
```

| Status | `error.type` | When | Client action |
| --- | --- | --- | --- |
| 401 | `unauthorized` | Missing or wrong API key (`WWW-Authenticate: Bearer`). | Fix credentials; do not retry. |
| 404 | `unknown_model` | `model` is not configured (also `GET /v1/models/{name}`). | Check `GET /v1/models`. |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or method. | Fix the call. |
| 413 | `request_too_large` | `Content-Length` above `4 x SOA_MAX_PROMPT_CHARS x (SOA_MAX_BATCH + 1) + 64 KiB`. | Send less. |
| 422 | `invalid_request` | Malformed JSON, missing/unknown fields, empty prompt (`details` lists the problems). | Fix the request. |
| 422 | `prompt_too_long` | A prompt or system prompt exceeds `SOA_MAX_PROMPT_CHARS` (`field`, `limit`). | Shorten it. |
| 422 | `batch_too_large` | More than `SOA_MAX_BATCH` prompts (`limit`). | Split the batch. |
| 422 | `retries_exhausted` | Every attempt produced invalid output (`attempts`, `failures`). | Improve the prompt/contract; retrying may help. |
| 422 | `tool_round_limit` | The model kept calling tools past the round limit. | Inspect tools/prompt. |
| 500 | `internal_error` | Unexpected server error (details only in server logs). | Retry with backoff; report with `request_id`. |
| 500 | `llm_unavailable` | The provider client could not be created (configuration). | Operator action. |
| 502 | `provider_error` | The provider failed permanently or stayed unavailable after retries (`retryable`, `provider_status`; `Retry-After` when the provider sent one). | Retry later if `retryable`. |
| 503 | `overloaded` | The replica is at `SOA_MAX_IN_FLIGHT` (`Retry-After: 1`). | Retry with backoff and jitter. |
| 504 | `run_timeout` | The run exceeded `SOA_RUN_TIMEOUT`. | Simplify or raise the timeout. |

Invalid requests (`401`, `404`, `413`, `422` validation errors) are rejected
before any model call.

## Security

- **Authentication.** Set `SOA_API_KEYS` to one or more long random keys
  (`openssl rand -hex 32`). Keys are compared in constant time against every
  configured key; list two keys during rotation, then remove the old one.
  Probes, `/metrics`, `/docs` and `/openapi.json` stay open so orchestrators
  and scrapers work; restrict them at the network layer (NetworkPolicy,
  internal-only ingress paths) if needed. API keys authenticate callers but do
  not provide per-tenant quotas.
- **TLS and proxies.** Terminate TLS at the ingress or load balancer. The
  server trusts `X-Forwarded-*` headers from any peer (it is meant to sit
  behind a proxy), so never expose the container port directly to the
  internet.
- **Secrets.** `ANTHROPIC_API_KEY` and `SOA_API_KEYS` belong in a Kubernetes
  Secret (or your secret manager) and `deploy/.env` for compose (git-ignored,
  excluded from the Docker build context). `ServerSettings` never prints API
  keys. `deploy/kubernetes/secret.example.yaml` only contains placeholders.
- **No prompt or output logging.** Access logs contain method, path, status,
  duration and request ID only. Failure events are logged with error
  locations and messages but without raw model output or offending input
  values; the full events (raw output truncated to 2000 characters) go only to
  the optional `SOA_FAILURE_LOG` file and the bounded in-memory window.
- **Redaction.** With `SOA_REDACT_EMAILS=true` (default) email addresses are
  replaced in failure events before they are stored, written or logged. For
  other data (account numbers, phone numbers) build the app in your own entry
  point and replace the redaction function before serving, e.g.
  `app.state.service.logger.redact = redact_patterns(EMAIL_PATTERN, r"\b\d{12,19}\b")`
  (both from `structured_output_agent.validation_log`), or keep
  `SOA_FAILURE_LOG` unset.
- **Request limits.** Prompt length, batch size and body size are bounded;
  run duration is bounded by `SOA_RUN_TIMEOUT`; in-flight runs by
  `SOA_MAX_IN_FLIGHT`. Also set request-size and timeout limits at the
  ingress (the body-size check relies on `Content-Length`).
- **Container hardening.** Non-root uid 10001, read-only root filesystem
  (only `/tmp` is writable), all capabilities dropped, no privilege
  escalation, RuntimeDefault seccomp and no service-account token, as in
  `deploy/kubernetes/deployment.yaml`.

## Observability

### Metrics (`GET /metrics`)

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `soa_runs_total` | counter | `contract`, `outcome` | Finished runs; `outcome` is `success`, `retries_exhausted`, `tool_round_limit`, `timeout`, `provider_error`, `cancelled` or `error`. |
| `soa_run_attempts` | histogram | `contract` | Generation attempts per run (1 = first answer valid). |
| `soa_failures_total` | counter | `contract`, `kind` | Recorded failures: `parse_error`, `validation_error`, `unknown_tool`, `tool_argument_error`, `tool_execution_error`, `tool_response_error`, `provider_error`. |
| `soa_llm_latency_seconds` | histogram | `contract` | Latency of model calls that returned (including provider-level retries). |
| `soa_tokens_total` | counter | `contract`, `direction` | Tokens reported by the provider (`input` / `output`). |
| `soa_run_duration_seconds` | histogram | `contract` | Wall-clock duration of finished runs. |
| `soa_runs_in_flight` | gauge | `contract` | Runs executing now. |
| `soa_tool_latency_seconds` | histogram | `contract`, `ok` | Tool call latency by success. |
| `soa_provider_retries_total` | counter | `error` | Provider retries by exception type (e.g. `TransientLLMError`). |
| `soa_http_requests_total` | counter | `method`, `route`, `status` | Requests by route *template* (e.g. `/v1/models/{name}`; unknown paths are `unmatched`). |
| `soa_http_request_duration_seconds` | histogram | `route` | Request duration by route template. |
| `soa_http_rejected_total` | counter | `reason` | Requests rejected before running: `overload`, `auth`, `too_large`. |
| `soa_in_flight_limit` | gauge | - | Configured `SOA_MAX_IN_FLIGHT`. |
| `soa_in_flight_slots_used` | gauge | - | Slots held by runs and batches now. |

All label values are bounded (configured contracts, fixed enums, route
templates), so cardinality does not grow with traffic. The run metrics come
from `PrometheusHooks`, a `RunHooks` implementation you can also attach to
your own agents (`hooks=[PrometheusHooks(registry)]`).

Useful queries:

```promql
# success ratio per contract
sum by (contract) (rate(soa_runs_total{outcome="success"}[5m]))
  / sum by (contract) (rate(soa_runs_total[5m]))
# first-attempt success (repairs needed)
histogram_quantile(0.9, sum by (le, contract) (rate(soa_run_attempts_bucket[15m])))
# shed load ratio
sum(rate(soa_http_rejected_total{reason="overload"}[5m])) / sum(rate(soa_http_requests_total{route=~"/v1/(runs|batches)"}[5m]))
# saturation per pod
soa_in_flight_slots_used / soa_in_flight_limit
# p95 model latency
histogram_quantile(0.95, sum by (le) (rate(soa_llm_latency_seconds_bucket[5m])))
```

### Logs

With `SOA_LOG_FORMAT=json` every line on stdout is one JSON object:

```json
{"timestamp": "2026-10-20T14:55:45.448542+00:00", "level": "INFO",
 "logger": "structured_output_agent.server.access", "message": "POST /v1/runs 200 2380.4ms",
 "request_id": "a2f1521cb88c4ea3a85680640f61c0cd",
 "access": {"method": "POST", "path": "/v1/runs", "route": "/v1/runs", "status": 200, "duration_ms": 2380.4}}
```

Records may also carry `failure_event` (a validation/provider failure without
raw output), `run_outcome` (a failed run), `exc_info` (server errors). Probe
and metrics requests are logged at `DEBUG` to keep logs quiet. uvicorn's own
access log is disabled in favour of this one.

### Request IDs

Send `X-Request-ID` (1-128 characters of `A-Z a-z 0-9 . _ -`) to correlate
your logs with the service's; otherwise, or when the value is invalid, the
service generates one. It is returned in the `X-Request-ID` header and in
every error body, and is attached to every log line written while handling
the request. Each run additionally has a `run_id`, returned on success and
present on its failure events.

### Failure log

`SOA_FAILURE_LOG=/path/file.jsonl` appends every failure event (contract,
attempt, kind, itemised errors, redacted raw output, run ID) as a JSON line.
In Kubernetes only `/tmp` is writable and it is ephemeral: point it at
`/tmp/failures.jsonl` and ship the file with a log agent sidecar, or rely on
the `failure_event` log records instead.

## Scaling and capacity

- **One worker per container; scale with replicas.** Do not run multiple
  uvicorn workers in a container: metrics would be per-worker and the
  in-flight limit would no longer describe the container.
- **What `SOA_MAX_IN_FLIGHT` means.** A run spends almost all its time
  waiting on the provider, so one replica can hold many concurrent runs with
  little CPU. Memory grows with concurrency (each run holds its conversation,
  and prompts can be up to `SOA_MAX_PROMPT_CHARS`). Start with the default 32
  for 256-512 MiB, and validate with a load test using a slow scripted model
  (`AsyncScriptedLLM(..., delay=2.0)` injected into `create_app`). Raise it
  when CPU and memory are low while 503s occur; lower it if memory approaches
  the limit or event-loop latency grows. CPU utilisation is the HPA signal
  (JSON parsing, validation and TLS dominate CPU).
- **Little's law.** Required in-flight capacity is roughly
  `requests per second x mean run duration`. For example 20 req/s with 6 s runs
  needs ~120 slots, i.e. 4 replicas at 32 plus headroom. Keep
  `minReplicas x SOA_MAX_IN_FLIGHT` above your steady-state need so the HPA
  only handles bursts.
- **Provider rate limits are the real ceiling.** Total concurrency across all
  replicas is `replicas x SOA_MAX_IN_FLIGHT`; make sure it, multiplied by
  your tokens per run, fits your Anthropic rate limits. Scaling replicas past
  the provider limit only converts 503s into 502s and retries. Use
  `maxReplicas` as the cap.
- **Timeouts line up outermost-longest.** Client timeout > ingress timeout >
  `SOA_RUN_TIMEOUT` > `SOA_LLM_TIMEOUT`. With defaults set the ingress read
  timeout above 300 s for runs (and more for batches).
- **Retries.** Clients should retry `503` and retryable `502` with jittered
  backoff, and `500`/`504` sparingly. Do not retry `4xx` other than as noted.

## Kubernetes

`deploy/kubernetes/` contains a kustomization with a ConfigMap (every `SOA_*`
setting), Deployment, Service (ClusterIP 80 -> 8000), HPA (autoscaling/v2,
2-20 replicas at 70% CPU, fast scale-up and slow scale-down) and a PDB
(`minAvailable: 1`), plus `secret.example.yaml`, which is **not** applied.

```bash
kubectl create secret generic structured-output-agent \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --from-literal=SOA_API_KEYS="$(openssl rand -hex 32)"
# edit deploy/kubernetes/configmap.yaml (SOA_MODELS!) and pin the image tag
kubectl apply -k deploy/kubernetes
kubectl rollout status deploy/structured-output-agent
kubectl port-forward svc/structured-output-agent 8000:80
```

Notes:

- The Deployment uses a rolling update with `maxUnavailable: 0`; a startup
  probe on `/healthz`, liveness on `/healthz` and readiness on `/readyz`.
- Shutdown sequence: the pod is marked terminating, `preStop` sleeps 10 s
  while endpoints are removed, then `SIGTERM` starts uvicorn's drain (up to
  30 s, readiness fails, the provider client closes); the grace period is
  45 s. Runs longer than ~30 s at shutdown are cut off and their clients get
  an error to retry; raise `terminationGracePeriodSeconds` (and keep the
  preStop sleep) if your runs are long and rollouts must not interrupt them.
- Prometheus scrape annotations are set on the pod template; with the
  Prometheus Operator create a `PodMonitor` selecting
  `app.kubernetes.io/name: structured-output-agent` on port `http`.
- The ConfigMap is not hashed into the pod template; after changing it run
  `kubectl rollout restart deploy/structured-output-agent` (or use a kustomize
  `configMapGenerator`).
- The HPA needs the metrics-server (or another resource-metrics provider).
- Replace `:latest` with a released tag or digest for production.

## Docker Compose

`deploy/docker-compose.yml` builds the root `Dockerfile` and runs the API on
port 8000 with the same hardening (read-only filesystem, tmpfs `/tmp`, no
capabilities) and a health check. It reads `deploy/.env` (copy
`deploy/.env.example`; the file is git-ignored):

```bash
cp deploy/.env.example deploy/.env   # set ANTHROPIC_API_KEY, SOA_API_KEYS, SOA_MODELS
docker compose -f deploy/docker-compose.yml up --build -d
curl -sS localhost:8000/readyz
# with Prometheus on :9090 scraping the API (deploy/prometheus.yml)
docker compose -f deploy/docker-compose.yml --profile monitoring up --build -d
```

## Operations runbook

Start every investigation from the request ID (client side) or from these
metrics, then filter logs by `request_id` / `run_id`.

### Many 503 responses (`overloaded`)

1. Confirm: `rate(soa_http_rejected_total{reason="overload"}[5m])` and
   `soa_in_flight_slots_used / soa_in_flight_limit` near 1.
2. Is the HPA scaling? `kubectl get hpa structured-output-agent`. If it is at
   `maxReplicas`, raise it (if provider limits allow). If CPU stays low while
   slots are full, the HPA will not react: runs are I/O bound, so raise
   `SOA_MAX_IN_FLIGHT` (check memory) or add replicas manually.
3. Did run duration grow? Check `soa_run_duration_seconds` and
   `soa_llm_latency_seconds`; slower provider calls hold slots longer
   (Little's law). See "Timeouts" below.
4. Large batches hold up to `SOA_BATCH_CONCURRENCY` slots each; lower it or
   ask clients for smaller batches.
5. Make sure clients retry 503s with jitter instead of hammering.

### Many 502 responses (`provider_error`)

1. Check `error.retryable` / `error.provider_status` in responses and
   `failure_event` logs (`kind: provider_error`).
2. `429` / `529` / `5xx` (retryable): provider rate limit or overload. Check
   `soa_provider_retries_total`; reduce total concurrency
   (`replicas x SOA_MAX_IN_FLIGHT`), request a higher rate limit or spread
   load over time. Raising `SOA_PROVIDER_RETRIES` trades latency for success.
3. `400` / `401` / `403` / `404` (permanent): bad credentials, model id
   (`SOA_LLM_MODEL`) or request; check the Secret and recent config changes.
4. Check the provider's status page.

### Rising retries (invalid output)

1. `soa_failures_total{kind=~"parse_error|validation_error"}` and the
   `soa_run_attempts` distribution show repairs; `soa_runs_total{outcome="retries_exhausted"}`
   shows give-ups (422 for clients).
2. Correlate with deploys: contract (model) changes, `SOA_STRICT`, system
   prompt or `SOA_LLM_MODEL` changes.
3. Inspect the itemised errors in `failure_event` logs or the failure log
   (`loc`, `msg`) to see which fields fail; tighten field descriptions or
   relax the contract. Tool failures (`tool_*` kinds) point to tool argument
   schemas or tool bugs.
4. Each repair is another model call: rising retries also raise latency,
   tokens (`soa_tokens_total`) and slot usage.

### Timeouts (504 `run_timeout`)

1. Compare `soa_run_duration_seconds` with `SOA_RUN_TIMEOUT`, and
   `soa_llm_latency_seconds` with `SOA_LLM_TIMEOUT`.
2. Slow provider: provider-level retries with backoff can consume the run
   budget; check `soa_provider_retries_total`.
3. Many attempts or tool rounds: see "Rising retries"; slow tools show in
   `soa_tool_latency_seconds`.
4. Raise `SOA_RUN_TIMEOUT` only together with the ingress and client
   timeouts, and remember it lengthens how long slots are held.

### Process restarts or 5xx `internal_error`

1. `kubectl describe pod` for OOMKilled (raise memory limit or lower
   `SOA_MAX_IN_FLIGHT`) or failing liveness probes (event loop blocked: look
   for slow synchronous tools; prefer async tools).
2. Search logs for `level: ERROR` with `exc_info` and the request ID.
3. A run failing with `internal_error` right after deploy usually means
   missing credentials: look for the startup warning
   `ANTHROPIC_API_KEY is not set`.

### Readiness failing

`/readyz` returns the reason: `no models configured` (set `SOA_MODELS`) or
`shutting down` (normal during rollouts). A process that cannot import
`SOA_MODELS`/`SOA_TOOLS` or has invalid settings exits at startup with the
variable named in the error (`kubectl logs --previous`).
