"""ServerSettings parsing/validation, JSON logging and the uvicorn runner."""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import Any

import pytest

from structured_output_agent.server import ServerSettings
from structured_output_agent.server import runner as runner_module
from structured_output_agent.server.logging_config import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    request_id_var,
)
from structured_output_agent.server.settings import ENV_VARS, setting_names


def test_defaults() -> None:
    settings = ServerSettings.from_env({})
    assert settings == ServerSettings()
    assert settings.models == ()
    assert settings.llm_model == "claude-opus-5-5"
    assert settings.llm_timeout == 120.0
    assert settings.provider_retries == 3
    assert settings.max_attempts == 3
    assert settings.run_timeout == 300.0
    assert settings.run_deadline == 300.0
    assert settings.max_in_flight == 32
    assert settings.max_prompt_chars == 100_000
    assert settings.max_batch == 50
    assert settings.batch_concurrency == 8
    assert settings.strict is False
    assert settings.api_keys == ()
    assert not settings.auth_enabled
    assert settings.log_format == "json"
    assert settings.log_level == "INFO"
    assert settings.failure_log is None
    assert settings.redact_emails is True
    assert settings.max_events == 1000


def test_from_env_parses_every_variable() -> None:
    env = {
        "SOA_MODELS": "a.b:C, d:E ,",
        "SOA_TOOLS": "pkg.tools:registry",
        "SOA_LLM_MODEL": "claude-test",
        "SOA_LLM_TIMEOUT": "30.5",
        "SOA_PROVIDER_RETRIES": "0",
        "SOA_MAX_ATTEMPTS": "5",
        "SOA_RUN_TIMEOUT": "0",
        "SOA_MAX_IN_FLIGHT": "4",
        "SOA_MAX_PROMPT_CHARS": "2000",
        "SOA_MAX_BATCH": "10",
        "SOA_BATCH_CONCURRENCY": "2",
        "SOA_STRICT": "Yes",
        "SOA_SYSTEM_PROMPT": "Be terse.",
        "SOA_API_KEYS": "k1,k2",
        "SOA_LOG_FORMAT": "TEXT",
        "SOA_LOG_LEVEL": "debug",
        "SOA_FAILURE_LOG": "/tmp/f.jsonl",
        "SOA_REDACT_EMAILS": "off",
        "SOA_MAX_EVENTS": "10",
    }
    settings = ServerSettings.from_env(env)
    assert settings == ServerSettings(
        models=("a.b:C", "d:E"),
        tools="pkg.tools:registry",
        llm_model="claude-test",
        llm_timeout=30.5,
        provider_retries=0,
        max_attempts=5,
        run_timeout=0.0,
        max_in_flight=4,
        max_prompt_chars=2000,
        max_batch=10,
        batch_concurrency=2,
        strict=True,
        system_prompt="Be terse.",
        api_keys=("k1", "k2"),
        log_format="text",
        log_level="DEBUG",
        failure_log="/tmp/f.jsonl",
        redact_emails=False,
        max_events=10,
    )
    assert settings.run_deadline is None
    assert settings.auth_enabled


def test_every_field_has_one_env_var() -> None:
    assert sorted(name for name, _ in ENV_VARS.values()) == sorted(setting_names())


def test_reads_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOA_MAX_BATCH", "7")
    monkeypatch.setenv("SOA_SYSTEM_PROMPT", "   ")
    settings = ServerSettings.from_env()
    assert settings.max_batch == 7
    assert settings.system_prompt is None


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("SOA_MAX_IN_FLIGHT", "many"),
        ("SOA_MAX_IN_FLIGHT", "0"),
        ("SOA_LLM_TIMEOUT", "0"),
        ("SOA_LLM_TIMEOUT", "nan"),
        ("SOA_RUN_TIMEOUT", "-1"),
        ("SOA_PROVIDER_RETRIES", "-1"),
        ("SOA_MAX_ATTEMPTS", "0"),
        ("SOA_MAX_PROMPT_CHARS", "0"),
        ("SOA_MAX_BATCH", "0"),
        ("SOA_BATCH_CONCURRENCY", "0"),
        ("SOA_MAX_EVENTS", "0"),
        ("SOA_STRICT", "maybe"),
        ("SOA_REDACT_EMAILS", "2"),
        ("SOA_LOG_FORMAT", "xml"),
        ("SOA_LOG_LEVEL", "LOUD"),
        ("SOA_MODELS", "no_colon"),
        ("SOA_TOOLS", "module:"),
    ],
)
def test_invalid_values_name_the_variable(var: str, value: str) -> None:
    with pytest.raises(ValueError, match=var):
        ServerSettings.from_env({var: value})


def test_empty_llm_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="SOA_LLM_MODEL"):
        ServerSettings(llm_model="")


def test_api_keys_never_appear_in_repr() -> None:
    settings = ServerSettings.from_env({"SOA_API_KEYS": "super-secret-key"})
    assert "super-secret-key" not in repr(settings)
    assert "api_keys" not in repr(settings)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _record(**extra: Any) -> logging.LogRecord:
    record = logging.LogRecord("soa.test", logging.WARNING, __file__, 1, "hello %s", ("x",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_basic_fields() -> None:
    token = request_id_var.set("req-9")
    try:
        document = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)
    assert document["level"] == "WARNING"
    assert document["logger"] == "soa.test"
    assert document["message"] == "hello x"
    assert document["request_id"] == "req-9"
    assert document["timestamp"].endswith("+00:00")


def test_json_formatter_extras_scrub_raw_output_and_inputs() -> None:
    failure = {
        "kind": "validation_error",
        "raw": "model output with secrets",
        "errors": [{"loc": "a", "msg": "bad", "type": "t", "input": "secret"}, "odd"],
    }
    record = _record(
        failure_event=failure,
        run_outcome={"success": False},
        access={"status": 200},
        request_id="explicit",
    )
    document = json.loads(JsonFormatter().format(record))
    assert document["failure_event"] == {
        "kind": "validation_error",
        "errors": [{"loc": "a", "msg": "bad", "type": "t"}, "odd"],
    }
    assert document["run_outcome"] == {"success": False}
    assert document["access"] == {"status": 200}
    assert document["request_id"] == "explicit"
    assert "secret" not in json.dumps(document)


def test_json_formatter_passes_through_non_dict_failure_event() -> None:
    document = json.loads(JsonFormatter().format(_record(failure_event="text")))
    assert document["failure_event"] == "text"


def test_json_formatter_exceptions_and_stacks() -> None:
    try:
        raise RuntimeError("bad")
    except RuntimeError:
        record = logging.LogRecord(
            "soa", logging.ERROR, __file__, 1, "failed", (), sys.exc_info(), sinfo="stack here"
        )
    document = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: bad" in document["exc_info"]
    assert document["stack_info"] == "stack here"


def test_request_id_filter_fills_in_context() -> None:
    token = request_id_var.set("ctx")
    try:
        record = _record()
        assert RequestIdFilter().filter(record)
        assert record.request_id == "ctx"  # type: ignore[attr-defined]
    finally:
        request_id_var.reset(token)


@pytest.fixture
def restore_root_logging() -> Any:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.mark.usefixtures("restore_root_logging")
def test_configure_logging_json_is_idempotent() -> None:
    stream = io.StringIO()
    configure_logging(ServerSettings(log_level="DEBUG"), stream=stream)
    handler = configure_logging(ServerSettings(log_level="DEBUG"), stream=stream)
    root = logging.getLogger()
    assert [h for h in root.handlers if getattr(h, "_soa_handler", False)] == [handler]
    assert root.level == logging.DEBUG
    assert logging.getLogger("uvicorn.access").disabled
    logging.getLogger("structured_output_agent.test").info("ready")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "ready"


@pytest.mark.usefixtures("restore_root_logging")
def test_configure_logging_text_includes_request_id() -> None:
    stream = io.StringIO()
    configure_logging(ServerSettings(log_format="text"), stream=stream)
    token = request_id_var.set("rid-1")
    try:
        logging.getLogger("structured_output_agent.test").warning("plain")
    finally:
        request_id_var.reset(token)
    assert "[rid-1] plain" in stream.getvalue()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_root_logging")
def test_run_starts_uvicorn_with_production_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    monkeypatch.setenv("SOA_MODELS", "structured_output_agent.cli:WeatherReport")

    runner_module.run(host="127.0.0.1", port=9000)

    ((app, kwargs),) = calls
    assert list(app.state.service.models) == ["WeatherReport"]
    assert kwargs == {
        "host": "127.0.0.1",
        "port": 9000,
        "log_config": None,
        "access_log": False,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
        "server_header": False,
        "timeout_graceful_shutdown": 30,
    }


@pytest.mark.usefixtures("restore_root_logging")
def test_run_accepts_explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    apps: list[Any] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    runner_module.run(ServerSettings(max_in_flight=3))
    assert apps[0].state.service.limiter.limit == 3
