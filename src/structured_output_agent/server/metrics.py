"""Prometheus metrics for runs (as ``RunHooks``) and for the HTTP layer.

Every metric is registered on an explicit ``CollectorRegistry`` so several
apps (for example in tests) never collide on the process-global registry.
Label values are bounded: ``contract`` is one of the configured models,
``route`` is the route template (never the raw path) and ``kind``,
``outcome``, ``direction`` and ``reason`` come from fixed sets.
"""

from __future__ import annotations

import asyncio

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ..errors import LLMError, RetryExhaustedError, RunTimeoutError, ToolRoundLimitError
from ..hooks import RunHooks
from ..tools import ToolResult
from ..types import Usage
from ..validation_log import FailureEvent

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)
RUN_BUCKETS = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0)
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0)
ATTEMPT_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10)


def run_outcome(success: bool, error: BaseException | None) -> str:
    """Classify a finished run for the ``outcome`` label."""
    if success:
        return "success"
    if isinstance(error, RetryExhaustedError):
        return "retries_exhausted"
    if isinstance(error, ToolRoundLimitError):
        return "tool_round_limit"
    if isinstance(error, RunTimeoutError):
        return "timeout"
    if isinstance(error, LLMError):
        return "provider_error"
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    return "error"


class PrometheusHooks(RunHooks):
    """``RunHooks`` exporting run metrics to a Prometheus registry.

    Usable outside the HTTP service too: pass it to an agent's ``hooks`` and
    expose ``registry`` with ``prometheus_client``.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.runs = Counter(
            "soa_runs_total", "Finished runs by outcome.", ["contract", "outcome"], registry=r
        )
        self.attempts = Histogram(
            "soa_run_attempts",
            "Generation attempts per finished run.",
            ["contract"],
            buckets=ATTEMPT_BUCKETS,
            registry=r,
        )
        self.failures = Counter(
            "soa_failures_total",
            "Recorded failures (parse, validation, tool, provider) by kind.",
            ["contract", "kind"],
            registry=r,
        )
        self.llm_latency = Histogram(
            "soa_llm_latency_seconds",
            "Latency of model calls that returned a response.",
            ["contract"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "soa_tokens_total",
            "Tokens reported by the provider.",
            ["contract", "direction"],
            registry=r,
        )
        self.run_duration = Histogram(
            "soa_run_duration_seconds",
            "Wall-clock duration of finished runs.",
            ["contract"],
            buckets=RUN_BUCKETS,
            registry=r,
        )
        self.in_flight = Gauge(
            "soa_runs_in_flight", "Runs currently executing.", ["contract"], registry=r
        )
        self.tool_latency = Histogram(
            "soa_tool_latency_seconds",
            "Latency of tool calls by success (tool names are model-controlled, so not a label).",
            ["contract", "ok"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )

    def on_run_start(self, *, run_id: str, contract: str, prompt: str) -> None:
        self.in_flight.labels(contract=contract).inc()

    def on_llm_response(
        self, *, run_id: str, contract: str, attempt: int, latency_s: float, usage: Usage
    ) -> None:
        self.llm_latency.labels(contract=contract).observe(latency_s)
        if usage.input_tokens:
            self.tokens.labels(contract=contract, direction="input").inc(usage.input_tokens)
        if usage.output_tokens:
            self.tokens.labels(contract=contract, direction="output").inc(usage.output_tokens)

    def on_tool_result(
        self, *, run_id: str, contract: str, result: ToolResult, latency_s: float
    ) -> None:
        ok = "true" if result.ok else "false"
        self.tool_latency.labels(contract=contract, ok=ok).observe(latency_s)

    def on_failure(self, *, run_id: str, contract: str, event: FailureEvent) -> None:
        self.failures.labels(contract=contract, kind=event.kind.value).inc()

    def on_run_end(
        self,
        *,
        run_id: str,
        contract: str,
        success: bool,
        attempts: int,
        usage: Usage,
        duration_s: float,
        error: BaseException | None,
    ) -> None:
        self.in_flight.labels(contract=contract).dec()
        self.runs.labels(contract=contract, outcome=run_outcome(success, error)).inc()
        self.attempts.labels(contract=contract).observe(attempts)
        self.run_duration.labels(contract=contract).observe(duration_s)


class HTTPMetrics:
    """Request, rejection and provider-retry metrics of the HTTP layer."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "soa_http_requests_total",
            "HTTP requests by route template and status.",
            ["method", "route", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "soa_http_request_duration_seconds",
            "HTTP request duration by route template.",
            ["route"],
            buckets=HTTP_BUCKETS,
            registry=registry,
        )
        self.rejected = Counter(
            "soa_http_rejected_total",
            "Requests rejected before running (overload, auth, too_large).",
            ["reason"],
            registry=registry,
        )
        self.provider_retries = Counter(
            "soa_provider_retries_total",
            "Retries of transient provider failures by error type.",
            ["error"],
            registry=registry,
        )
        self.in_flight_limit = Gauge(
            "soa_in_flight_limit", "Configured per-replica in-flight run limit.", registry=registry
        )
        self.in_flight_slots = Gauge(
            "soa_in_flight_slots_used",
            "In-flight slots currently held by run and batch requests.",
            registry=registry,
        )
