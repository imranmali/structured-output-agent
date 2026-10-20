"""HTTP service tests: every scenario of specs/agent-service plus edge cases."""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from structured_output_agent import (
    AsyncScriptedLLM,
    LLMError,
    LLMResponse,
    Message,
    StructuredOutputError,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    TransientLLMError,
    Usage,
)
from structured_output_agent.server import PrometheusHooks, ServerSettings, create_app
from structured_output_agent.server.app import (
    InFlightLimiter,
    _close,
    default_llm_factory,
    map_run_error,
)
from structured_output_agent.server.metrics import run_outcome


class Invoice(BaseModel):
    number: str
    total: float


VALID = '{"number": "INV-1", "total": 12.5}'
INVALID = '{"number": "INV-1"}'

invoice_tools = ToolRegistry()


@invoice_tools.register
def lookup(number: str) -> dict[str, str]:
    """Look up an invoice."""
    return {"number": number}


not_a_registry = object()


def make_app(
    llm: Any = None,
    *,
    models: dict[str, type[BaseModel]] | None = None,
    tools: ToolRegistry | None = None,
    llm_factory: Any = None,
    **settings: Any,
) -> Any:
    return create_app(
        ServerSettings(**settings),
        llm=llm,
        models={"Invoice": Invoice} if models is None else models,
        tools=tools,
        llm_factory=llm_factory,
    )


@asynccontextmanager
async def client_for(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def post_run(app: Any, body: dict[str, Any], **kwargs: Any) -> httpx.Response:
    async with client_for(app) as client:
        return await client.post("/v1/runs", json=body, **kwargs)


def metric(text: str, line_prefix: str) -> float:
    for line in text.splitlines():
        if line.startswith(line_prefix + " "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{line_prefix} not found")


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------


async def test_list_models_includes_json_schema() -> None:
    async with client_for(make_app()) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    (entry,) = response.json()["models"]
    assert entry["name"] == "Invoice"
    assert entry["schema"]["required"] == ["number", "total"]
    assert entry["schema"]["properties"]["total"]["type"] == "number"


async def test_get_one_model_and_unknown_model() -> None:
    async with client_for(make_app(strict=True)) as client:
        ok = await client.get("/v1/models/Invoice")
        missing = await client.get("/v1/models/Nope")

    assert ok.status_code == 200
    assert ok.json()["schema"]["additionalProperties"] is False
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "unknown_model"


async def test_run_for_unknown_model_is_404_naming_it() -> None:
    llm = AsyncScriptedLLM([VALID])
    response = await post_run(make_app(llm), {"model": "Nope", "prompt": "x"})

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "unknown_model"
    assert "Nope" in error["message"]
    assert error["model"] == "Nope"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Run endpoint
# ---------------------------------------------------------------------------


async def test_successful_run_returns_output_and_metadata() -> None:
    llm = AsyncScriptedLLM(
        [INVALID, LLMResponse(text=VALID, usage=Usage(input_tokens=10, output_tokens=5))]
    )
    response = await post_run(make_app(llm), {"model": "Invoice", "prompt": "Invoice INV-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == {"number": "INV-1", "total": 12.5}
    assert body["model"] == "Invoice"
    assert body["attempts"] == 2
    assert re.fullmatch(r"[0-9a-f]{32}", body["run_id"])
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert body["duration_ms"] >= 0
    assert body["tool_calls"] == 0
    (failure,) = body["failures"]
    assert failure["attempt"] == 1
    assert failure["kind"] == "validation_error"
    assert failure["errors"][0]["loc"] == "total"
    assert set(failure["errors"][0]) == {"loc", "msg", "type"}


async def test_run_overrides_system_prompt_and_strict() -> None:
    llm = AsyncScriptedLLM(['{"number": "1", "total": 1, "extra": true}', VALID])
    response = await post_run(
        make_app(llm, system_prompt="default system"),
        {"model": "Invoice", "prompt": "x", "system_prompt": "custom system", "strict": True},
    )

    assert response.status_code == 200
    assert response.json()["attempts"] == 2
    assert llm.calls[0].system.startswith("custom system")


async def test_run_uses_configured_default_system_prompt() -> None:
    llm = AsyncScriptedLLM([VALID])
    await post_run(
        make_app(llm, system_prompt="default system"), {"model": "Invoice", "prompt": "x"}
    )
    assert llm.calls[0].system.startswith("default system")


async def test_retries_exhausted_is_422_with_failure_history() -> None:
    llm = AsyncScriptedLLM(["not json", INVALID])
    response = await post_run(make_app(llm, max_attempts=2), {"model": "Invoice", "prompt": "x"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "retries_exhausted"
    assert error["attempts"] == 2
    assert [f["kind"] for f in error["failures"]] == ["parse_error", "validation_error"]
    assert [f["attempt"] for f in error["failures"]] == [1, 2]
    assert error["failures"][1]["errors"][0]["loc"] == "total"


async def test_tool_round_limit_is_422() -> None:
    calls = [
        LLMResponse(text="", tool_calls=[ToolCall(f"c{i}", "lookup", {"number": "1"})])
        for i in range(10)
    ]
    response = await post_run(
        make_app(AsyncScriptedLLM(calls), tools=invoice_tools), {"model": "Invoice", "prompt": "x"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "tool_round_limit"


async def test_tool_calls_are_counted_and_measured() -> None:
    llm = AsyncScriptedLLM(
        [LLMResponse(text="", tool_calls=[ToolCall("c1", "lookup", {"number": "1"})]), VALID]
    )
    app = make_app(llm, tools=invoice_tools)
    async with client_for(app) as client:
        response = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
        text = (await client.get("/metrics")).text

    assert response.json()["tool_calls"] == 1
    assert metric(text, 'soa_tool_latency_seconds_count{contract="Invoice",ok="true"}') == 1.0


async def test_permanent_provider_failure_is_502() -> None:
    llm = AsyncScriptedLLM([LLMError("invalid request", status_code=400)])
    response = await post_run(make_app(llm), {"model": "Invoice", "prompt": "x"})

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "provider_error"
    assert error["retryable"] is False
    assert error["provider_status"] == 400
    assert "Retry-After" not in response.headers


async def test_transient_provider_failure_after_retries_is_502_with_retry_after() -> None:
    llm = AsyncScriptedLLM([TransientLLMError("overloaded", retry_after=2.5, status_code=529)])
    response = await post_run(make_app(llm), {"model": "Invoice", "prompt": "x"})

    assert response.status_code == 502
    assert response.json()["error"]["retryable"] is True
    assert response.headers["Retry-After"] == "3"


async def test_run_timeout_is_504() -> None:
    llm = AsyncScriptedLLM([VALID], delay=1.0)
    response = await post_run(make_app(llm, run_timeout=0.05), {"model": "Invoice", "prompt": "x"})

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "run_timeout"


async def test_unexpected_error_is_500_without_internals(caplog: pytest.LogCaptureFixture) -> None:
    llm = AsyncScriptedLLM([RuntimeError("database password is hunter2")])
    with caplog.at_level(logging.ERROR, logger="structured_output_agent.server"):
        response = await post_run(make_app(llm), {"model": "Invoice", "prompt": "x"})

    assert response.status_code == 500
    assert response.json()["error"] == {
        "type": "internal_error",
        "message": "Internal server error",
    }
    assert "hunter2" not in response.text
    assert any(record.exc_info for record in caplog.records)


def test_map_run_error_treats_other_library_errors_as_internal() -> None:
    error = map_run_error(StructuredOutputError("odd"))
    assert (error.status_code, error.type) == (500, "internal_error")


# ---------------------------------------------------------------------------
# Request validation limits
# ---------------------------------------------------------------------------


async def test_oversized_prompt_is_422_and_model_not_called() -> None:
    llm = AsyncScriptedLLM([VALID])
    app = make_app(llm, max_prompt_chars=1000)
    async with client_for(app) as client:
        too_long = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "a" * 1001})
        assert llm.calls == []
        at_limit = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "a" * 1000})

    assert too_long.status_code == 422
    error = too_long.json()["error"]
    assert error["type"] == "prompt_too_long"
    assert error["limit"] == 1000
    assert at_limit.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"model": "Invoice", "prompt": ""},
        {"model": "Invoice"},
        {"model": "Invoice", "prompt": "x", "unexpected": 1},
        {"model": "Invoice", "prompt": "x", "system_prompt": "s" * 11},
    ],
)
async def test_invalid_run_requests_are_422_without_calling_the_model(body: dict[str, Any]) -> None:
    llm = AsyncScriptedLLM([VALID])
    response = await post_run(make_app(llm, max_prompt_chars=10), body)

    assert response.status_code == 422
    assert response.json()["error"]["type"] in {"invalid_request", "prompt_too_long"}
    assert llm.calls == []


async def test_malformed_json_is_422() -> None:
    async with client_for(make_app()) as client:
        response = await client.post(
            "/v1/runs", content=b"{not json", headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"
    assert response.json()["error"]["details"]


async def test_oversized_body_is_413() -> None:
    llm = AsyncScriptedLLM([VALID])
    app = make_app(llm, max_prompt_chars=10, max_batch=1)
    async with client_for(app) as client:
        response = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x" * 100_000})
        text = (await client.get("/metrics")).text
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"
    assert metric(text, 'soa_http_rejected_total{reason="too_large"}') == 1.0
    assert llm.calls == []


async def test_unknown_route_and_method_use_error_shape() -> None:
    async with client_for(make_app()) as client:
        missing = await client.get("/nope")
        wrong_method = await client.get("/v1/runs")
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["type"] == "method_not_allowed"


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------


async def test_mixed_batch_keeps_order_with_success_then_error() -> None:
    llm = AsyncScriptedLLM([VALID, INVALID])
    app = make_app(llm, max_attempts=1, batch_concurrency=1)
    async with client_for(app) as client:
        response = await client.post(
            "/v1/batches", json={"model": "Invoice", "prompts": ["first", "second"]}
        )

    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["status"] == "ok"
    assert first["output"] == {"number": "INV-1", "total": 12.5}
    assert second["status"] == "error"
    assert second["status_code"] == 422
    assert second["error"]["type"] == "retries_exhausted"
    assert second["error"]["failures"][0]["kind"] == "validation_error"


async def test_batch_runs_concurrently_in_input_order() -> None:
    def echo(_: str, messages: Sequence[Message]) -> str:
        return '{"number": "' + messages[0].content + '", "total": 1}'

    llm = AsyncScriptedLLM([echo] * 4, delay=0.01)
    app = make_app(llm, batch_concurrency=4)
    async with client_for(app) as client:
        response = await client.post(
            "/v1/batches", json={"model": "Invoice", "prompts": ["a", "b", "c", "d"]}
        )
    numbers = [item["output"]["number"] for item in response.json()["results"]]
    assert numbers == ["a", "b", "c", "d"]


async def test_batch_reports_unexpected_item_errors_generically() -> None:
    llm = AsyncScriptedLLM([RuntimeError("internal detail")])
    async with client_for(make_app(llm)) as client:
        response = await client.post("/v1/batches", json={"model": "Invoice", "prompts": ["a"]})
    (item,) = response.json()["results"]
    assert item == {
        "status": "error",
        "status_code": 500,
        "error": {"type": "internal_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        ({"model": "Invoice", "prompts": ["a", "b", "c"]}, "batch_too_large"),
        ({"model": "Invoice", "prompts": []}, "invalid_request"),
        ({"model": "Invoice", "prompts": ["a", ""]}, "invalid_request"),
        ({"model": "Invoice", "prompts": ["a", "x" * 11]}, "prompt_too_long"),
        ({"model": "Invoice", "prompts": ["a"], "system_prompt": "x" * 11}, "prompt_too_long"),
    ],
)
async def test_invalid_batches_are_422_without_calling_the_model(
    body: dict[str, Any], error_type: str
) -> None:
    llm = AsyncScriptedLLM([VALID])
    async with client_for(make_app(llm, max_batch=2, max_prompt_chars=10)) as client:
        response = await client.post("/v1/batches", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["type"] == error_type
    assert llm.calls == []


async def test_batch_for_unknown_model_is_404() -> None:
    async with client_for(make_app()) as client:
        response = await client.post("/v1/batches", json={"model": "Nope", "prompts": ["a"]})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Load shedding
# ---------------------------------------------------------------------------


async def test_saturated_replica_sheds_with_503_and_retry_after() -> None:
    llm = AsyncScriptedLLM([VALID, VALID], delay=0.3)
    app = make_app(llm, max_in_flight=1)
    async with client_for(app) as client:
        first = asyncio.create_task(
            client.post("/v1/runs", json={"model": "Invoice", "prompt": "one"})
        )
        await asyncio.sleep(0.05)
        second = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "two"})
        first_response = await first
        third = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "three"})
        text = (await client.get("/metrics")).text

    assert second.status_code == 503
    assert second.headers["Retry-After"] == "1"
    assert second.json()["error"]["type"] == "overloaded"
    assert first_response.status_code == 200
    assert third.status_code == 200  # the slot was released
    assert len(llm.calls) == 2
    assert metric(text, 'soa_http_rejected_total{reason="overload"}') == 1.0
    assert metric(text, "soa_in_flight_slots_used") == 0.0
    assert metric(text, "soa_in_flight_limit") == 1.0


async def test_batch_holds_bounded_slots() -> None:
    llm = AsyncScriptedLLM([VALID] * 3, delay=0.3)
    app = make_app(llm, max_in_flight=2, batch_concurrency=8)
    async with client_for(app) as client:
        batch = asyncio.create_task(
            client.post("/v1/batches", json={"model": "Invoice", "prompts": ["a", "b", "c"]})
        )
        await asyncio.sleep(0.05)
        single = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
        batch_response = await batch

    assert single.status_code == 503
    assert batch_response.status_code == 200
    assert [item["status"] for item in batch_response.json()["results"]] == ["ok"] * 3


def test_in_flight_limiter() -> None:
    limiter = InFlightLimiter(3)
    assert limiter.try_acquire(2)
    assert not limiter.try_acquire(2)
    assert limiter.try_acquire(1)
    limiter.release(3)
    limiter.release(1)
    assert limiter.used == 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_missing_key_is_401() -> None:
    llm = AsyncScriptedLLM([VALID])
    app = make_app(llm, api_keys=("k1", "k2"))
    async with client_for(app) as client:
        response = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
        catalogue = await client.get("/v1/models")
        text = (await client.get("/metrics")).text

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error"]["type"] == "unauthorized"
    assert catalogue.status_code == 401
    assert metric(text, 'soa_http_rejected_total{reason="auth"}') == 2.0
    assert llm.calls == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong"},
        {"X-API-Key": "k1x"},
        {"Authorization": "Basic azE6"},
        {"Authorization": "Bearer "},
    ],
)
async def test_wrong_key_is_401(headers: dict[str, str]) -> None:
    response = await post_run(
        make_app(AsyncScriptedLLM([VALID]), api_keys=("k1",)),
        {"model": "Invoice", "prompt": "x"},
        headers=headers,
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": "Bearer k2"}, {"Authorization": "bearer k1"}, {"X-API-Key": "k1"}],
)
async def test_valid_key_is_accepted(headers: dict[str, str]) -> None:
    response = await post_run(
        make_app(AsyncScriptedLLM([VALID]), api_keys=("k1", "k2")),
        {"model": "Invoice", "prompt": "x"},
        headers=headers,
    )
    assert response.status_code == 200


async def test_probes_and_metrics_stay_open_with_keys() -> None:
    async with client_for(make_app(api_keys=("k1",))) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200


# ---------------------------------------------------------------------------
# Probes and lifecycle
# ---------------------------------------------------------------------------


async def test_no_models_live_but_not_ready() -> None:
    async with client_for(make_app(models={})) as client:
        live = await client.get("/healthz")
        ready = await client.get("/readyz")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 503
    assert ready.json() == {"status": "unavailable", "reason": "no models configured"}


class ClosableLLM(AsyncScriptedLLM):
    closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_lifespan_shutdown_fails_readiness_and_closes_owned_client() -> None:
    created: list[ClosableLLM] = []

    def factory(settings: ServerSettings) -> ClosableLLM:
        llm = ClosableLLM([VALID, VALID])
        created.append(llm)
        return llm

    app = make_app(llm_factory=factory)
    async with client_for(app) as client:
        async with app.router.lifespan_context(app):
            assert (await client.get("/readyz")).status_code == 200
            assert created == []  # probes never create the client
            for _ in range(2):
                response = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
                assert response.status_code == 200
        ready = await client.get("/readyz")

    assert len(created) == 1
    assert created[0].closed
    assert ready.status_code == 503
    assert ready.json()["reason"] == "shutting down"


async def test_injected_client_is_not_closed() -> None:
    llm = ClosableLLM([VALID])
    app = make_app(llm)
    async with client_for(app) as client, app.router.lifespan_context(app):
        await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
    assert not llm.closed


async def test_close_errors_are_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        async def complete(self, **_: Any) -> LLMResponse:
            return LLMResponse(text=VALID)

        def close(self) -> None:
            raise RuntimeError("boom")

    app = make_app(llm_factory=lambda settings: Broken())
    async with client_for(app) as client, app.router.lifespan_context(app):
        await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
    assert "error closing the LLM client" in caplog.text


async def test_llm_factory_failure_is_500() -> None:
    def factory(settings: ServerSettings) -> Any:
        raise RuntimeError("no credentials")

    response = await post_run(make_app(llm_factory=factory), {"model": "Invoice", "prompt": "x"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "llm_unavailable"
    assert "credentials" not in response.text


async def test_close_walks_to_the_closable_client() -> None:
    events: list[str] = []

    class SDK:
        async def close(self) -> None:
            events.append("sdk")

    class Client:
        def __init__(self) -> None:
            self._client = SDK()

    class Wrapper:
        def __init__(self) -> None:
            self.inner = Client()

    class SyncClosable:
        def close(self) -> None:
            events.append("sync")

    await _close(Wrapper())
    await _close(SyncClosable())
    await _close(object())
    assert events == ["sdk", "sync"]


async def test_default_llm_factory_wraps_anthropic_client_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import structured_output_agent.llm as llm_module
    import structured_output_agent.resilience as resilience

    created: list[dict[str, Any]] = []

    class FakeAnthropic:
        def __init__(self, model: str, **kwargs: Any) -> None:
            created.append({"model": model, **kwargs})
            self.failures = 1

        async def complete(
            self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> LLMResponse:
            if self.failures:
                self.failures -= 1
                raise TransientLLMError("rate limited", status_code=429)
            return LLMResponse(text=VALID)

    monkeypatch.setattr(llm_module, "AsyncAnthropicClient", FakeAnthropic)
    monkeypatch.setattr(
        resilience,
        "ProviderRetryPolicy",
        functools.partial(resilience.ProviderRetryPolicy, initial_delay=0.0),
    )
    app = create_app(
        ServerSettings(llm_model="claude-test", llm_timeout=7.0, provider_retries=2),
        models={"Invoice": Invoice},
    )
    async with client_for(app) as client:
        assert (await client.get("/readyz")).status_code == 200
        assert created == []
        response = await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
        text = (await client.get("/metrics")).text

    assert response.status_code == 200
    assert created == [{"model": "claude-test", "timeout": 7.0, "max_retries": 0}]
    assert metric(text, 'soa_provider_retries_total{error="TransientLLMError"}') == 1.0
    state = app.state.service
    assert isinstance(state.llm, resilience.AsyncRetryingLLM)
    assert state.llm.policy.max_retries == 2


@pytest.mark.parametrize(("key", "warned"), [("", True), ("sk-test", False)])
async def test_startup_warns_when_credentials_are_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, key: str, warned: bool
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    app = create_app(ServerSettings(), models={"Invoice": Invoice})
    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            pass
    assert ("ANTHROPIC_API_KEY is not set" in caplog.text) is warned


def test_default_llm_factory_builds_real_clients() -> None:
    import structured_output_agent.resilience as resilience

    pytest.importorskip("anthropic")
    llm = default_llm_factory(ServerSettings(llm_model="claude-test", provider_retries=1))
    assert isinstance(llm, resilience.AsyncRetryingLLM)
    assert llm.inner.model == "claude-test"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


async def test_metrics_after_a_run() -> None:
    llm = AsyncScriptedLLM(
        [INVALID, LLMResponse(text=VALID, usage=Usage(input_tokens=3, output_tokens=2))]
    )
    app = make_app(llm)
    async with client_for(app) as client:
        await client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"})
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'soa_runs_total{contract="Invoice",outcome="success"} 1.0' in text
    assert metric(text, 'soa_failures_total{contract="Invoice",kind="validation_error"}') == 1.0
    assert metric(text, 'soa_tokens_total{contract="Invoice",direction="input"}') == 3.0
    assert metric(text, 'soa_tokens_total{contract="Invoice",direction="output"}') == 2.0
    assert metric(text, 'soa_run_attempts_sum{contract="Invoice"}') == 2.0
    assert metric(text, 'soa_llm_latency_seconds_count{contract="Invoice"}') == 2.0
    assert metric(text, 'soa_run_duration_seconds_count{contract="Invoice"}') == 1.0
    assert metric(text, 'soa_runs_in_flight{contract="Invoice"}') == 0.0
    assert (
        metric(text, 'soa_http_requests_total{method="POST",route="/v1/runs",status="200"}') == 1.0
    )
    assert metric(text, 'soa_http_request_duration_seconds_count{route="/v1/runs"}') == 1.0


async def test_http_metrics_use_route_templates() -> None:
    async with client_for(make_app()) as client:
        await client.get("/v1/models/Invoice")
        await client.get("/v1/models/whatever")
        await client.get("/unknown/path")
        text = (await client.get("/metrics")).text
    assert (
        metric(text, 'soa_http_requests_total{method="GET",route="/v1/models/{name}",status="200"}')
        == 1.0
    )
    assert (
        metric(text, 'soa_http_requests_total{method="GET",route="/v1/models/{name}",status="404"}')
        == 1.0
    )
    assert 'route="unmatched"' in text
    assert "whatever" not in text


async def test_separate_apps_have_separate_registries() -> None:
    first, second = make_app(AsyncScriptedLLM([VALID])), make_app()
    await post_run(first, {"model": "Invoice", "prompt": "x"})
    async with client_for(second) as client:
        text = (await client.get("/metrics")).text
    assert "soa_runs_total{" not in text


@pytest.mark.parametrize(
    ("success", "error", "outcome"),
    [
        (True, None, "success"),
        (False, StructuredOutputError("x"), "error"),
        (False, LLMError("x"), "provider_error"),
        (False, asyncio.CancelledError(), "cancelled"),
    ],
)
def test_run_outcome_labels(success: bool, error: BaseException | None, outcome: str) -> None:
    assert run_outcome(success, error) == outcome


def test_prometheus_hooks_default_registry() -> None:
    hooks = PrometheusHooks()
    hooks.on_run_start(run_id="r", contract="C", prompt="p")
    assert hooks.registry.get_sample_value("soa_runs_in_flight", {"contract": "C"}) == 1.0


# ---------------------------------------------------------------------------
# Request correlation and logging
# ---------------------------------------------------------------------------


async def test_client_request_id_is_echoed() -> None:
    llm = AsyncScriptedLLM([VALID])
    response = await post_run(
        make_app(llm), {"model": "Invoice", "prompt": "x"}, headers={"X-Request-ID": "abc-123"}
    )
    assert response.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("supplied", ["bad id!", "x" * 129, ""])
async def test_invalid_request_id_is_replaced(supplied: str) -> None:
    async with client_for(make_app()) as client:
        response = await client.get("/healthz", headers={"X-Request-ID": supplied})
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


async def test_error_bodies_carry_the_request_id() -> None:
    async with client_for(make_app(api_keys=("k",))) as client:
        unauthorized = await client.get("/v1/models", headers={"X-Request-ID": "req.1"})
        not_found = await client.get(
            "/v1/models/Nope", headers={"X-Request-ID": "req.2", "X-API-Key": "k"}
        )
    assert unauthorized.headers["X-Request-ID"] == "req.1"
    assert unauthorized.json()["request_id"] == "req.1"
    assert not_found.json()["request_id"] == "req.2"


async def test_unhandled_exceptions_become_500_with_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = make_app()

    async def boom() -> None:
        raise RuntimeError("secret detail")

    app.add_api_route("/boom", boom)
    async with client_for(app) as client:
        response = await client.get("/boom", headers={"X-Request-ID": "r-500"})
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "r-500"
    assert response.json() == {
        "error": {"type": "internal_error", "message": "Internal server error"},
        "request_id": "r-500",
    }
    assert "unhandled error" in caplog.text


async def test_access_log_has_request_fields_but_no_prompt_or_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = AsyncScriptedLLM(['{"number": "SECRET-OUTPUT", "total": 1}'])
    with caplog.at_level(logging.DEBUG):
        await post_run(
            make_app(llm),
            {"model": "Invoice", "prompt": "SECRET-PROMPT"},
            headers={"X-Request-ID": "log-1"},
        )
    access = [r for r in caplog.records if r.name == "structured_output_agent.server.access"]
    (record,) = access
    assert record.levelno == logging.INFO
    assert record.request_id == "log-1"  # type: ignore[attr-defined]
    fields = record.access  # type: ignore[attr-defined]
    assert fields["method"] == "POST"
    assert fields["path"] == "/v1/runs"
    assert fields["status"] == 200
    assert fields["duration_ms"] >= 0
    for logged in caplog.records:
        assert "SECRET" not in logged.getMessage()


async def test_probe_access_logs_are_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        async with client_for(make_app()) as client:
            await client.get("/healthz")
    (record,) = [r for r in caplog.records if r.name.endswith(".access")]
    assert record.levelno == logging.DEBUG


async def test_failure_log_file_and_email_redaction(tmp_path: Any) -> None:
    path = tmp_path / "failures.jsonl"
    llm = AsyncScriptedLLM(['{"number": "ada@example.com"}', VALID])
    await post_run(make_app(llm, failure_log=str(path)), {"model": "Invoice", "prompt": "x"})
    content = path.read_text()
    assert "[REDACTED]" in content
    assert "ada@example.com" not in content


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


async def test_create_app_reads_settings_and_models_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SOA_MODELS", "tests.test_server:Invoice, structured_output_agent.cli:Weather"
    )
    monkeypatch.setenv("SOA_TOOLS", "tests.test_server:invoice_tools")
    app = create_app(llm=AsyncScriptedLLM([VALID]))
    async with client_for(app) as client:
        names = [m["name"] for m in (await client.get("/v1/models")).json()["models"]]
    assert names == ["Invoice", "Weather"]
    assert app.state.service.tools is invoice_tools


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"models": ("tests.test_server:Missing",)}, "SOA_MODELS"),
        ({"models": ("json:dumps",)}, "SOA_MODELS"),
        (
            {"models": ("tests.test_server:Invoice", "tests.test_server:Invoice")},
            "two models are named 'Invoice'",
        ),
        ({"tools": "tests.test_server:not_a_registry"}, "not a ToolRegistry"),
        ({"tools": "tests.no_such_module:tools"}, "SOA_TOOLS"),
    ],
)
def test_bad_model_or_tool_specs_fail_fast(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_app(ServerSettings(**settings))


def test_server_package_requires_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name.startswith("structured_output_agent.server"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError, match=r"pip install 'structured-output-agent\[server\]'"):
        import structured_output_agent.server  # noqa: F401


def test_full_asgi_lifespan_through_the_middleware() -> None:
    from fastapi.testclient import TestClient

    app = make_app(AsyncScriptedLLM([VALID]))
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        assert client.post("/v1/runs", json={"model": "Invoice", "prompt": "x"}).status_code == 200
    assert app.state.service.shutting_down


async def test_errors_after_the_response_started_propagate() -> None:
    from fastapi.responses import StreamingResponse

    async def chunks() -> AsyncIterator[bytes]:
        yield b"partial"
        raise RuntimeError("mid-stream")

    app = make_app()
    app.add_api_route("/stream", lambda: StreamingResponse(chunks()))
    async with client_for(app) as client:
        with pytest.raises(RuntimeError, match="mid-stream"):
            await client.get("/stream")


def test_python_dash_m_runs_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    import structured_output_agent.server.runner as runner

    calls: list[bool] = []
    monkeypatch.setattr(runner, "run", lambda: calls.append(True))
    runpy.run_module("structured_output_agent.server", run_name="__main__")
    assert calls == [True]
