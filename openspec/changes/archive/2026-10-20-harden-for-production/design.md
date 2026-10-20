# Design

## Context

See `proposal.md` for motivation. Today `StructuredOutputAgent.run` is one
synchronous loop that calls the LLM, executes tools and drives retries
inline. Adding an async variant by copying that loop would let the two drift,
so the loop is first extracted into a transport-independent engine.

## Goals / Non-Goals

**Goals:**

- One run engine, two thin drivers (sync and async) with identical semantics.
- Throughput scales with concurrency in one process (async) and with replicas
  across processes (stateless HTTP service).
- Bounded resource use: in-flight runs, memory held by logs, run duration.
- Production operations: probes, metrics, structured logs, auth, graceful
  shutdown, vulnerability scanning.
- The existing public sync API keeps working unchanged.

**Non-Goals:**

- Durable job queues or cross-replica coordination; scaling is horizontal and
  stateless, and callers own retries across replicas.
- Prometheus multi-process mode: one worker process per container, scale by
  replicas.

## Decisions

### Sans-IO run engine (`engine.py`)

`RunEngine` holds the conversation, attempt and tool-round counters, usage,
failures and timing, and exposes:

- `request() -> LLMRequest` (system, messages, tools) for the next model call,
  checking the run deadline first.
- `handle(response) -> Step`, which returns `ExecuteTools(calls)`,
  `Retry(delay)` or `Finished(result)`, or raises `RetryExhaustedError` /
  `ToolRoundLimitError`.
- `add_tool_results(results)`, `provider_failed(exc)`, `timed_out()`.

The engine never sleeps, calls the LLM or runs tools; the sync driver
(`StructuredOutputAgent`) and async driver (`AsyncStructuredOutputAgent`) do.
Alternative considered: making the sync agent call `asyncio.run` on the async
one. Rejected because it breaks callers that already run an event loop and
forces sync LLM clients into threads.

### Concurrency model

- Async tool calls of one turn run with `asyncio.gather`; sync tool functions
  run in `asyncio.to_thread`; coroutine tools are awaited. Per-tool `timeout`
  uses `asyncio.wait_for` (async) or a worker thread with a result timeout
  (sync driver).
- `AsyncStructuredOutputAgent.run_many` bounds concurrency with an
  `asyncio.Semaphore`; the sync `run_many` uses a `ThreadPoolExecutor`. Both
  return results in input order and capture per-item exceptions.
- Run deadline: the async driver wraps the run in `asyncio.wait_for`; both
  drivers also check the deadline before each model call.

### Provider resilience (`resilience.py`)

`ProviderRetryPolicy(max_retries=3, initial_delay=0.5, multiplier=2.0,
max_delay=20.0, jitter=True)` and wrappers `RetryingLLM` / `AsyncRetryingLLM`
around any client. Classification is by exception: `LLMError.retryable`
(set by `TransientLLMError`) and `LLMError.retry_after`. `AnthropicClient` and
`AsyncAnthropicClient` map SDK errors: 429, 408, 409, 5xx and 529, timeouts and
connection errors become `TransientLLMError` (with `retry-after` parsed from
headers); other API errors become permanent `LLMError`. Full jitter
(`uniform(0, backoff)`) avoids synchronized retry storms across replicas.
The SDK's own retries are configurable via `max_retries` on the clients;
when wrapping in `RetryingLLM` set them to 0 to avoid multiplying retries.

### Observability (`hooks.py`)

`RunHooks` is a base class with no-op methods (`on_run_start`,
`on_llm_request`, `on_llm_response`, `on_tool_result`, `on_failure`,
`on_run_end`). Agents accept `hooks=[...]`; a `HookDispatcher` calls each hook
inside `try/except` and logs exceptions on the library logger. The HTTP
service's Prometheus metrics are a `RunHooks` implementation, so library
users get the same integration point. `Usage(input_tokens, output_tokens)` is
attached to every `LLMResponse` and summed into `RunResult.usage`; the result
also carries `run_id` and `duration_s`.

### Logging hardening

`ValidationLogger(max_events=1000, redact=None, ...)`: events live in a
`collections.deque(maxlen=max_events)`; `redact(str) -> str` is applied to raw
output and to string error inputs (recursively inside containers) before the
event is stored, written or logged. Statistics add `input_tokens`,
`output_tokens`, `total_duration_s` and `mean_duration_s`.

### HTTP service (`server/`, extra `server`)

FastAPI app built by `create_app(settings, *, llm=None, models=None,
tools=None)`; settings come from `SOA_*` environment variables
(`SOA_MODELS`, `SOA_TOOLS`, `SOA_LLM_MODEL`, `SOA_MAX_ATTEMPTS`,
`SOA_MAX_IN_FLIGHT`, `SOA_RUN_TIMEOUT`, `SOA_MAX_PROMPT_CHARS`,
`SOA_MAX_BATCH`, `SOA_API_KEYS`, `SOA_STRICT`, `SOA_LOG_FORMAT`, ...).
Endpoints: `POST /v1/runs`, `POST /v1/batches`, `GET /v1/models`,
`GET /v1/models/{name}`, `GET /healthz`, `GET /readyz`, `GET /metrics`.

- Load shedding uses a non-blocking in-flight counter, so a saturated replica
  answers 503 + `Retry-After` immediately and the load balancer or HPA reacts,
  instead of hiding latency in an unbounded queue.
- Error mapping: unknown contract 404, invalid request 422, retries exhausted
  422 (with failure history), provider error 502, run timeout 504, overload
  503, missing or wrong API key 401.
- API keys are compared in constant time; probes and metrics stay open for
  orchestrators.
- The LLM client is created lazily so probes work without credentials, and
  readiness flips to failing during shutdown so in-flight requests drain.
- `structured-agent serve` runs uvicorn with one worker; containers scale by
  replica count (Kubernetes HPA on CPU), which keeps Prometheus metrics
  per-process and correct.

### Deployment and supply chain

- Image: installs `anthropic,server` extras by default, `CMD ["serve"]`,
  `EXPOSE 8000`, a `HEALTHCHECK`, non-root, multi-arch (amd64/arm64), SBOM and
  provenance attestations, Trivy scan in CI (fail on fixable critical CVEs).
- `deploy/kubernetes/`: Deployment (probes, resource requests/limits,
  read-only root filesystem, non-root, topology spread), Service, HPA, PDB,
  ConfigMap and an example Secret; `deploy/docker-compose.yml` for local runs.
- CI adds Python 3.14, `pip-audit`, CodeQL, and installs all extras so the
  service is tested. CD optionally publishes to PyPI with trusted publishing
  when the repository variable `PUBLISH_TO_PYPI` is `true`.
- Version is single-sourced from `structured_output_agent.__version__`.

### Work split

| Branch | Scope |
| --- | --- |
| `feature/run-engine` | types, errors, hooks, engine, sync/async agents, async tool execution, `AsyncScriptedLLM`, stats usage/duration (integrator) |
| `feature/provider-resilience` | `resilience.py`, `AsyncAnthropicClient`, SDK error mapping, usage extraction |
| `feature/logging-hardening` | bounded retention, redaction |
| `feature/http-service` | `server/`, `serve` CLI, Dockerfile, `deploy/`, deployment docs |
| `ci/production-hardening` | CI/CD, CodeQL, audits, release, SECURITY.md, CHANGELOG.md |

## Risks / Trade-offs

- [Sync tool timeouts cannot kill a running thread] → the call fails fast for
  the model, but the thread finishes in the background; documented, and async
  tools are recommended for slow I/O.
- [Load shedding returns 503 under bursts] → clients and load balancers retry
  with backoff; HPA adds replicas; the limit is configurable per replica.
- [Jittered retries add latency] → bounded by `max_delay` and the run deadline.
- [Redaction is user supplied] → defaults to none; documented with an example.

## Migration Plan

Backward compatible for library users. The container's default command
changes from `--help` to `serve`; run `docker run IMAGE demo` explicitly for
the demo. Release as 0.2.0.
