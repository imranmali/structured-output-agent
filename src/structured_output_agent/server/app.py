"""The FastAPI application: run, batch and catalogue endpoints plus probes.

``create_app`` builds a self-contained app (its own agent, metrics registry and
in-flight limiter), so a process can host several apps, for example in tests.
Replicas are stateless: scale them horizontally behind a load balancer.
"""

from __future__ import annotations

import hmac
import logging
import math
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from pydantic import BaseModel
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..agent import AsyncStructuredOutputAgent
from ..errors import (
    LLMError,
    RetryExhaustedError,
    RunTimeoutError,
    ToolRoundLimitError,
)
from ..llm import AsyncLLMClient
from ..loading import load_model, load_object
from ..retry import RetryPolicy
from ..schema import OutputContract
from ..tools import ToolRegistry
from ..types import RunResult
from ..validation_log import EMAIL_PATTERN, ValidationLogger, redact_patterns
from .logging_config import request_id_var
from .metrics import HTTPMetrics, PrometheusHooks
from .schemas import (
    BatchItemError,
    BatchItemOk,
    BatchRequest,
    BatchResponse,
    ErrorItem,
    FailureOut,
    ModelInfo,
    ModelList,
    RunRequest,
    RunResponse,
    UsageOut,
)
from .settings import ServerSettings

log = logging.getLogger("structured_output_agent.server")
access_log = logging.getLogger("structured_output_agent.server.access")

LLMFactory = Callable[[ServerSettings], AsyncLLMClient]

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_PROBE_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class APIError(Exception):
    """An error answered with ``{"error": {...}, "request_id": ...}``."""

    def __init__(
        self,
        status_code: int,
        type: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.type = type
        self.message = message
        self.headers = dict(headers or {})
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        return {"type": self.type, "message": self.message, **self.extra}

    def response(self) -> JSONResponse:
        return error_response(self.status_code, self.payload(), headers=self.headers)


def error_response(
    status_code: int, error: dict[str, Any], *, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": error, "request_id": request_id_var.get()},
        status_code=status_code,
        headers=dict(headers or {}),
    )


def _failures(result_failures: Sequence[Any]) -> list[FailureOut]:
    # Raw model output and offending input values are deliberately left out:
    # they can contain sensitive data and are available, redacted, in the
    # failure log.
    return [
        FailureOut(
            attempt=failure.number,
            kind=failure.kind.value,
            errors=[ErrorItem(loc=e.loc, msg=e.msg, type=e.type) for e in failure.errors],
        )
        for failure in result_failures
    ]


def map_run_error(exc: BaseException) -> APIError:
    """Translate a run failure into the API error answered for it."""
    if isinstance(exc, RetryExhaustedError):
        return APIError(
            422,
            "retries_exhausted",
            str(exc),
            attempts=len(exc.attempts),
            failures=[f.model_dump() for f in _failures(exc.attempts)],
        )
    if isinstance(exc, ToolRoundLimitError):
        return APIError(422, "tool_round_limit", str(exc))
    if isinstance(exc, RunTimeoutError):
        return APIError(504, "run_timeout", str(exc))
    if isinstance(exc, LLMError):
        headers = {}
        if exc.retryable and exc.retry_after is not None and exc.retry_after > 0:
            headers["Retry-After"] = str(max(1, math.ceil(exc.retry_after)))
        return APIError(
            502,
            "provider_error",
            f"The model provider failed: {exc}",
            headers=headers,
            retryable=exc.retryable,
            provider_status=exc.status_code,
        )
    log.error("unexpected error during run", exc_info=(type(exc), exc, exc.__traceback__))
    return APIError(500, "internal_error", "Internal server error")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class InFlightLimiter:
    """Non-blocking counter of in-flight run slots for one replica.

    Acquisition never waits: when the slots are taken the request is shed so
    the client or load balancer can retry elsewhere, instead of queueing
    without bound. The app runs on one event loop, so no lock is needed.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def try_acquire(self, slots: int = 1) -> bool:
        if self.used + slots > self.limit:
            return False
        self.used += slots
        return True

    def release(self, slots: int = 1) -> None:
        self.used = max(0, self.used - slots)


async def _close(obj: Any, depth: int = 0) -> None:
    """Close an LLM client: ``aclose``/``close`` or, failing that, its inner client."""
    for name in ("aclose", "close"):
        method = getattr(obj, name, None)
        if callable(method):
            outcome = method()
            if hasattr(outcome, "__await__"):
                await outcome
            return
    if depth < 3:
        for attr in ("inner", "_client", "client"):
            inner = getattr(obj, attr, None)
            if inner is not None:
                await _close(inner, depth + 1)
                return


def default_llm_factory(
    settings: ServerSettings,
    *,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> AsyncLLMClient:
    """Claude via ``AsyncAnthropicClient``, retried by ``AsyncRetryingLLM``.

    Imported lazily so the service starts (and answers probes) without the
    ``anthropic`` extra or credentials; the SDK's own retries are disabled so
    they do not multiply with ``SOA_PROVIDER_RETRIES``.
    """
    from ..llm import AsyncAnthropicClient
    from ..resilience import AsyncRetryingLLM, ProviderRetryPolicy

    client = AsyncAnthropicClient(settings.llm_model, timeout=settings.llm_timeout, max_retries=0)
    return AsyncRetryingLLM(
        client, ProviderRetryPolicy(max_retries=settings.provider_retries), on_retry=on_retry
    )


class ServiceState:
    """Everything one app instance owns."""

    def __init__(
        self,
        settings: ServerSettings,
        *,
        models: Mapping[str, type[BaseModel]],
        tools: ToolRegistry | None,
        llm: AsyncLLMClient | None,
        llm_factory: LLMFactory,
    ) -> None:
        self.settings = settings
        self.models = dict(models)
        self.schemas = {
            name: OutputContract(model, strict=settings.strict).json_schema()
            for name, model in self.models.items()
        }
        self.tools = tools
        self.registry = CollectorRegistry()
        self.hooks = PrometheusHooks(self.registry)
        self.http = HTTPMetrics(self.registry)
        self.http.in_flight_limit.set(settings.max_in_flight)
        self.limiter = InFlightLimiter(settings.max_in_flight)
        self.logger = ValidationLogger(
            path=settings.failure_log,
            max_events=settings.max_events,
            redact=redact_patterns(EMAIL_PATTERN) if settings.redact_emails else None,
        )
        self.llm = llm
        self.owns_llm = False
        self.llm_factory = llm_factory
        self._agent: AsyncStructuredOutputAgent | None = None
        self.shutting_down = False

    @property
    def agent(self) -> AsyncStructuredOutputAgent:
        """The app's agent, creating the LLM client on first use."""
        if self._agent is None:
            if self.llm is None:
                try:
                    self.llm = self.llm_factory(self.settings)
                except Exception as exc:
                    log.exception("could not create the LLM client")
                    raise APIError(
                        500, "llm_unavailable", "The model provider client is not configured"
                    ) from exc
                self.owns_llm = True
            self._agent = AsyncStructuredOutputAgent(
                self.llm,
                tools=self.tools,
                retry=RetryPolicy(max_attempts=self.settings.max_attempts),
                logger=self.logger,
                hooks=[self.hooks],
                system_prompt=self.settings.system_prompt,
                strict=self.settings.strict,
                run_timeout=self.settings.run_deadline,
            )
        return self._agent

    def model(self, name: str) -> type[BaseModel]:
        try:
            return self.models[name]
        except KeyError:
            available = ", ".join(sorted(self.models)) or "none"
            raise APIError(
                404,
                "unknown_model",
                f"Unknown model {name!r}; available: {available}",
                model=name,
            ) from None

    def check_text(self, field: str, text: str | None) -> None:
        limit = self.settings.max_prompt_chars
        if text is not None and len(text) > limit:
            raise APIError(
                422,
                "prompt_too_long",
                f"{field} has {len(text)} characters; the limit is {limit}",
                field=field,
                limit=limit,
            )

    @contextmanager
    def slots(self, count: int) -> Iterator[None]:
        if not self.limiter.try_acquire(count):
            self.http.rejected.labels(reason="overload").inc()
            raise APIError(
                503,
                "overloaded",
                f"This replica is at capacity ({self.limiter.limit} runs in flight); retry later",
                headers={"Retry-After": "1"},
            )
        self.http.in_flight_slots.set(self.limiter.used)
        try:
            yield
        finally:
            self.limiter.release(count)
            self.http.in_flight_slots.set(self.limiter.used)

    async def close(self) -> None:
        self.shutting_down = True
        if self.owns_llm and self.llm is not None:
            try:
                await _close(self.llm)
            except Exception:
                log.exception("error closing the LLM client")


def _run_response(name: str, result: RunResult[BaseModel]) -> RunResponse:
    return RunResponse(
        run_id=result.run_id,
        model=name,
        output=result.output.model_dump(mode="json"),
        attempts=result.attempts,
        failures=_failures(result.failures),
        tool_calls=result.tool_calls,
        usage=UsageOut(**result.usage.to_dict()),
        duration_ms=round(result.duration_s * 1000, 3),
    )


# ---------------------------------------------------------------------------
# Middleware: request IDs, auth, body size, access log, HTTP metrics
# ---------------------------------------------------------------------------


def _route_template(scope: Scope, routes: Sequence[BaseRoute]) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    for candidate in routes:
        match, _ = candidate.matches(scope)
        if match == Match.FULL:
            template = getattr(candidate, "path", None)
            if isinstance(template, str):
                return template
    return "unmatched"


class RequestMiddleware:
    """Pure ASGI middleware wrapping every HTTP request."""

    def __init__(self, app: ASGIApp, state: ServiceState, routes: Sequence[BaseRoute]) -> None:
        self.app = app
        self.state = state
        self.routes = routes
        settings = state.settings
        self.api_keys = [key.encode() for key in settings.api_keys]
        # Generous bound on a valid body: the largest batch of the longest
        # prompts (4 bytes per character worst case) plus JSON overhead.
        self.max_body_bytes = 4 * settings.max_prompt_chars * (settings.max_batch + 1) + 65_536

    def _authorized(self, headers: Headers) -> bool:
        presented: str | None = headers.get("x-api-key")
        if presented is None:
            scheme, _, credentials = headers.get("authorization", "").partition(" ")
            if scheme.lower() == "bearer" and credentials.strip():
                presented = credentials.strip()
        if not presented:
            return False
        candidate = presented.encode()
        matched = False
        for key in self.api_keys:  # compare against every key: constant time per key
            matched |= hmac.compare_digest(candidate, key)
        return matched

    def _reject(self, scope: Scope) -> JSONResponse | None:
        headers = Headers(scope=scope)
        path: str = scope["path"]
        protected = path == "/v1" or path.startswith("/v1/")
        if self.api_keys and protected and not self._authorized(headers):
            self.state.http.rejected.labels(reason="auth").inc()
            return error_response(
                401,
                {"type": "unauthorized", "message": "A valid API key is required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        length = headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > self.max_body_bytes:
            self.state.http.rejected.labels(reason="too_large").inc()
            return error_response(
                413,
                {
                    "type": "request_too_large",
                    "message": f"Request body exceeds {self.max_body_bytes} bytes",
                },
            )
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = (
            supplied
            if supplied is not None and _REQUEST_ID_RE.fullmatch(supplied)
            else uuid.uuid4().hex
        )
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            rejection = self._reject(scope)
            if rejection is not None:
                await rejection(scope, receive, send_wrapper)
            else:
                await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("unhandled error")
            if response_started:
                raise
            await error_response(
                500, {"type": "internal_error", "message": "Internal server error"}
            )(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - started
            route = _route_template(scope, self.routes)
            method: str = scope["method"]
            self.state.http.requests.labels(method=method, route=route, status=str(status)).inc()
            self.state.http.duration.labels(route=route).observe(duration)
            path: str = scope["path"]
            access_log.log(
                logging.DEBUG if path in _PROBE_PATHS else logging.INFO,
                "%s %s %d %.1fms",
                method,
                path,
                status,
                duration * 1000,
                extra={
                    "request_id": request_id,
                    "access": {
                        "method": method,
                        "path": path,
                        "route": route,
                        "status": status,
                        "duration_ms": round(duration * 1000, 3),
                    },
                },
            )
            request_id_var.reset(token)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _load_models(specs: Sequence[str]) -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for spec in specs:
        try:
            model = load_model(spec)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"SOA_MODELS: cannot load {spec!r}: {exc}") from exc
        if model.__name__ in models:
            raise ValueError(f"SOA_MODELS: two models are named {model.__name__!r}")
        models[model.__name__] = model
    return models


def _load_tools(spec: str) -> ToolRegistry:
    try:
        registry = load_object(spec)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"SOA_TOOLS: cannot load {spec!r}: {exc}") from exc
    if not isinstance(registry, ToolRegistry):
        raise ValueError(f"SOA_TOOLS: {spec!r} is not a ToolRegistry")
    return registry


def create_app(
    settings: ServerSettings | None = None,
    *,
    llm: AsyncLLMClient | None = None,
    models: Mapping[str, type[BaseModel]] | None = None,
    tools: ToolRegistry | None = None,
    llm_factory: LLMFactory | None = None,
) -> FastAPI:
    """Build the service.

    ``settings`` default to :meth:`ServerSettings.from_env`. ``models`` (keyed
    by contract name) and ``tools`` override ``SOA_MODELS`` and ``SOA_TOOLS``.
    ``llm`` injects a client (owned by the caller, never closed by the app);
    otherwise ``llm_factory`` (default :func:`default_llm_factory`) creates
    one on the first run, and the app closes it on shutdown.
    """
    settings = settings if settings is not None else ServerSettings.from_env()
    resolved_models = dict(models) if models is not None else _load_models(settings.models)
    if tools is None and settings.tools:
        tools = _load_tools(settings.tools)
    state = ServiceState(
        settings,
        models=resolved_models,
        tools=tools,
        llm=llm,
        llm_factory=llm_factory or default_llm_factory,
    )
    if llm_factory is None:
        # Count provider retries; done after construction as it needs the metrics.
        state.llm_factory = partial(default_llm_factory, on_retry=_retry_counter(state))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state.shutting_down = False
        log.info("service started with models: %s", ", ".join(sorted(state.models)) or "none")
        try:
            yield
        finally:
            log.info("service shutting down")
            await state.close()

    app = FastAPI(
        title="Structured Output Agent",
        version=__version__,
        description="Schema-enforced, validated and retried LLM output over HTTP.",
        lifespan=lifespan,
    )
    app.state.service = state
    app.add_middleware(RequestMiddleware, state=state, routes=app.router.routes)

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: Exception) -> Response:
        assert isinstance(exc, APIError)
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        details = [
            {
                "loc": ".".join(str(part) for part in error.get("loc", ())),
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        return error_response(
            422,
            {"type": "invalid_request", "message": "Request validation failed", "details": details},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        kinds = {404: "not_found", 405: "method_not_allowed"}
        return error_response(
            exc.status_code,
            {"type": kinds.get(exc.status_code, "http_error"), "message": str(exc.detail)},
            headers=exc.headers,
        )

    @app.post("/v1/runs", tags=["runs"])
    async def create_run(body: RunRequest) -> RunResponse:
        """Run one prompt against a contract and return the validated output."""
        model = state.model(body.model)
        state.check_text("prompt", body.prompt)
        state.check_text("system_prompt", body.system_prompt)
        with state.slots(1):
            agent = state.agent
            try:
                result = await agent.run(
                    body.prompt, model, strict=body.strict, system_prompt=body.system_prompt
                )
            except Exception as exc:
                raise map_run_error(exc) from exc
        return _run_response(body.model, result)

    @app.post("/v1/batches", tags=["runs"])
    async def create_batch(body: BatchRequest) -> BatchResponse:
        """Run several prompts against one contract; one result per prompt, in order.

        The batch holds ``min(len(prompts), SOA_BATCH_CONCURRENCY,
        SOA_MAX_IN_FLIGHT)`` in-flight slots for its whole duration and runs
        that many prompts at a time, so per-replica concurrency stays bounded
        by ``SOA_MAX_IN_FLIGHT``.
        """
        model = state.model(body.model)
        if len(body.prompts) > settings.max_batch:
            raise APIError(
                422,
                "batch_too_large",
                f"Batch has {len(body.prompts)} prompts; the limit is {settings.max_batch}",
                limit=settings.max_batch,
            )
        for index, prompt in enumerate(body.prompts):
            if not prompt:
                raise APIError(
                    422, "invalid_request", f"prompts.{index} must not be empty", field="prompts"
                )
            state.check_text(f"prompts.{index}", prompt)
        state.check_text("system_prompt", body.system_prompt)
        slots = min(len(body.prompts), settings.batch_concurrency, settings.max_in_flight)
        with state.slots(slots):
            agent = state.agent
            outcomes = await agent.run_many(
                body.prompts,
                model,
                concurrency=slots,
                strict=body.strict,
                system_prompt=body.system_prompt,
            )
        results: list[BatchItemOk | BatchItemError] = []
        for outcome in outcomes:
            if isinstance(outcome, RunResult):
                results.append(BatchItemOk(**_run_response(body.model, outcome).model_dump()))
            else:
                error = map_run_error(outcome)
                results.append(BatchItemError(status_code=error.status_code, error=error.payload()))
        return BatchResponse(model=body.model, results=results)

    @app.get("/v1/models", tags=["models"])
    async def list_models() -> ModelList:
        """The configured output contracts and the JSON Schema each one is shown as."""
        return ModelList(
            models=[ModelInfo(name=name, schema_=state.schemas[name]) for name in state.models]
        )

    @app.get("/v1/models/{name}", tags=["models"])
    async def get_model(name: str) -> ModelInfo:
        """One configured output contract."""
        state.model(name)
        return ModelInfo(name=name, schema_=state.schemas[name])

    @app.get("/healthz", tags=["operations"])
    async def healthz() -> dict[str, str]:
        """Liveness: the process is serving requests."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"], response_model=None)
    async def readyz() -> Response | dict[str, Any]:
        """Readiness: at least one contract is configured and not shutting down."""
        reason = None
        if state.shutting_down:
            reason = "shutting down"
        elif not state.models:
            reason = "no models configured"
        if reason is not None:
            return JSONResponse({"status": "unavailable", "reason": reason}, status_code=503)
        return {"status": "ready", "models": len(state.models)}

    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(state.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _retry_counter(state: ServiceState) -> Callable[[int, BaseException, float], None]:
    def count(_: int, exc: BaseException, __: float) -> None:
        state.http.provider_retries.labels(error=type(exc).__name__).inc()

    return count


__all__ = ["APIError", "create_app", "default_llm_factory", "map_run_error"]
