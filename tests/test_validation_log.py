import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from structured_output_agent.types import ErrorDetail, FailureKind, Usage
from structured_output_agent.validation_log import (
    EMAIL_PATTERN,
    LOGGER_NAME,
    REDACTION_FAILED,
    FailureEvent,
    ReliabilityStats,
    ValidationLogger,
    redact_patterns,
)

AGE_MISSING = ErrorDetail(loc="age", msg="Field required", type="missing", input={"name": "Ada"})


def test_validation_failure_recorded() -> None:
    log = ValidationLogger()
    event = log.record(
        FailureKind.VALIDATION_ERROR,
        contract="Person",
        attempt=2,
        errors=[AGE_MISSING],
        raw='{"name": "Ada"}',
        run_id="r1",
    )
    assert list(log.events) == [event]
    assert event.kind is FailureKind.VALIDATION_ERROR
    assert event.contract == "Person"
    assert event.attempt == 2
    assert event.errors == [AGE_MISSING]
    assert event.raw == '{"name": "Ada"}'
    assert event.truncated is False
    assert event.run_id == "r1"
    ts = datetime.fromisoformat(event.timestamp)
    assert ts.utcoffset() == timezone.utc.utcoffset(None)
    assert log.stats.failures_by_kind == {"validation_error": 1}


def test_event_to_dict_is_json_serialisable() -> None:
    log = ValidationLogger()
    odd = ErrorDetail(loc="when", msg="bad", type="x", input=object())
    event = log.record(
        FailureKind.VALIDATION_ERROR, contract="Person", attempt=1, errors=[AGE_MISSING, odd]
    )
    data = event.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["kind"] == "validation_error"
    assert data["contract"] == "Person"
    assert data["errors"][0] == {
        "loc": "age",
        "msg": "Field required",
        "type": "missing",
        "input": {"name": "Ada"},
    }
    assert isinstance(data["errors"][1]["input"], str)
    assert data["errors"][1]["input"].startswith("<object")
    assert data["raw"] == ""
    assert data["truncated"] is False
    assert data["run_id"] is None


def test_raw_output_truncated() -> None:
    log = ValidationLogger(max_raw_chars=10)
    event = log.record(
        FailureKind.PARSE_ERROR, contract="Person", attempt=1, errors=[], raw="a" * 25
    )
    assert event.raw == "a" * 10
    assert event.truncated is True


def test_raw_output_at_limit_not_truncated() -> None:
    log = ValidationLogger(max_raw_chars=10)
    event = log.record(
        FailureKind.PARSE_ERROR, contract="Person", attempt=1, errors=[], raw="a" * 10
    )
    assert event.raw == "a" * 10
    assert event.truncated is False


def test_default_truncation_limit() -> None:
    log = ValidationLogger()
    event = log.record(FailureKind.PARSE_ERROR, contract="P", attempt=1, errors=[], raw="b" * 2500)
    assert len(event.raw) == 2000
    assert event.truncated is True


def test_negative_max_raw_chars_rejected() -> None:
    with pytest.raises(ValueError, match="max_raw_chars"):
        ValidationLogger(max_raw_chars=-1)


def test_events_written_to_file(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "nested" / "failures.jsonl"
    log = ValidationLogger(path=path)
    log.record(FailureKind.PARSE_ERROR, contract="Person", attempt=1, errors=[], raw="nope")
    log.record(
        FailureKind.VALIDATION_ERROR, contract="Person", attempt=2, errors=[AGE_MISSING], raw="{}"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["kind"] == "parse_error"
    assert first["raw"] == "nope"
    assert second["kind"] == "validation_error"
    assert second["attempt"] == 2
    assert second["errors"][0]["loc"] == "age"


def test_file_sink_appends_and_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "failures.jsonl"
    path.write_text('{"existing": true}\n', encoding="utf-8")
    log = ValidationLogger(path=str(path))
    log.record(FailureKind.UNKNOWN_TOOL, contract="search", attempt=1, errors=[], raw="é")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["raw"] == "é"


def test_no_file_written_without_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log = ValidationLogger()
    log.record(FailureKind.PARSE_ERROR, contract="P", attempt=1, errors=[])
    assert log.path is None
    assert list(tmp_path.iterdir()) == []


def test_warning_emitted(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    log = ValidationLogger()
    log.record(FailureKind.VALIDATION_ERROR, contract="Person", attempt=1, errors=[AGE_MISSING])
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == LOGGER_NAME
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "validation_error" in message
    assert "Person" in message
    assert "age: Field required" in message
    payload = record.failure_event  # type: ignore[attr-defined]
    assert payload["contract"] == "Person"
    assert payload["kind"] == "validation_error"


def test_warning_summarises_many_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    errors = [ErrorDetail(loc=f"f{i}", msg="bad", type="x") for i in range(5)]
    ValidationLogger().record(FailureKind.VALIDATION_ERROR, contract="P", attempt=1, errors=errors)
    message = caplog.records[0].getMessage()
    assert "f2: bad" in message
    assert "f3" not in message
    assert "(2 more)" in message


def test_warning_without_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    ValidationLogger().record(FailureKind.PARSE_ERROR, contract="P", attempt=1, errors=[])
    assert "no details" in caplog.records[0].getMessage()


def test_custom_logger(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    custom = logging.getLogger("my.app")
    ValidationLogger(logger=custom).record(
        FailureKind.PARSE_ERROR, contract="P", attempt=1, errors=[]
    )
    assert [r.name for r in caplog.records] == ["my.app"]


def test_statistics_after_mixed_runs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    log = ValidationLogger()
    log.record_run(contract="Person", success=True, attempts=1)
    log.record(FailureKind.VALIDATION_ERROR, contract="Person", attempt=1, errors=[AGE_MISSING])
    log.record_run(contract="Person", success=True, attempts=2)
    stats = log.stats
    assert stats.runs == 2
    assert stats.successes == 2
    assert stats.failures == 0
    assert stats.attempts == 3
    assert stats.first_attempt_successes == 1
    assert stats.first_attempt_success_rate == 0.5
    assert stats.success_rate == 1.0
    assert stats.failures_by_kind == {"validation_error": 1}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_exhausted_run_logged_at_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    log = ValidationLogger()
    log.record_run(contract="Person", success=False, attempts=3, run_id="abc")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "run failed" in errors[0].getMessage()
    assert "Person" in errors[0].getMessage()
    assert errors[0].run_outcome["run_id"] == "abc"  # type: ignore[attr-defined]
    stats = log.stats
    assert (stats.runs, stats.successes, stats.failures, stats.attempts) == (1, 0, 1, 3)
    assert stats.success_rate == 0.0
    assert stats.first_attempt_success_rate == 0.0


def test_failures_by_kind_counts_each_kind() -> None:
    log = ValidationLogger()
    for kind in (FailureKind.PARSE_ERROR, FailureKind.PARSE_ERROR, FailureKind.TOOL_ARGUMENT_ERROR):
        log.record(kind, contract="c", attempt=1, errors=[])
    assert log.stats.failures_by_kind == {"parse_error": 2, "tool_argument_error": 1}


def test_stats_is_a_snapshot() -> None:
    log = ValidationLogger()
    log.record(FailureKind.PARSE_ERROR, contract="c", attempt=1, errors=[])
    snapshot = log.stats
    snapshot.failures_by_kind["parse_error"] = 99
    snapshot.runs = 42
    assert log.stats.failures_by_kind == {"parse_error": 1}
    assert log.stats.runs == 0


def test_empty_stats_rates_are_zero() -> None:
    stats = ReliabilityStats()
    assert stats.success_rate == 0.0
    assert stats.first_attempt_success_rate == 0.0
    assert stats.to_dict() == {
        "runs": 0,
        "successes": 0,
        "failures": 0,
        "attempts": 0,
        "first_attempt_successes": 0,
        "failures_by_kind": {},
        "success_rate": 0.0,
        "first_attempt_success_rate": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_duration_s": 0.0,
        "mean_duration_s": 0.0,
    }


def test_stats_to_dict_includes_rates() -> None:
    stats = ReliabilityStats(
        runs=4,
        successes=3,
        failures=1,
        attempts=7,
        first_attempt_successes=2,
        failures_by_kind={"parse_error": 3},
    )
    data = stats.to_dict()
    assert data["success_rate"] == 0.75
    assert data["first_attempt_success_rate"] == 0.5
    assert data["failures_by_kind"] == {"parse_error": 3}
    assert json.loads(json.dumps(data)) == data


def test_clear_resets_events_and_stats() -> None:
    log = ValidationLogger()
    log.record(FailureKind.PARSE_ERROR, contract="c", attempt=1, errors=[])
    log.record_run(contract="c", success=True, attempts=2)
    log.clear()
    assert list(log.events) == []
    assert log.stats == ReliabilityStats()


def test_new_run_id_is_unique_hex() -> None:
    first, second = ValidationLogger.new_run_id(), ValidationLogger.new_run_id()
    assert first != second
    assert len(first) == 32
    int(first, 16)


def test_failure_event_direct_construction() -> None:
    event = FailureEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        kind=FailureKind.TOOL_EXECUTION_ERROR,
        contract="search",
        attempt=1,
        errors=[],
        raw="",
        truncated=False,
    )
    assert event.run_id is None
    assert event.to_dict()["kind"] == "tool_execution_error"


def test_usage_and_duration_totals() -> None:
    logger = ValidationLogger()
    logger.record_run(contract="P", success=True, attempts=1, usage=Usage(100, 10), duration_s=1.0)
    logger.record_run(contract="P", success=True, attempts=1, usage=Usage(50, 5), duration_s=3.0)
    stats = logger.stats
    assert stats.input_tokens == 150
    assert stats.output_tokens == 15
    assert stats.total_duration_s == 4.0
    assert stats.mean_duration_s == 2.0
    assert stats.to_dict()["mean_duration_s"] == 2.0


def _fail(log: ValidationLogger, n: int) -> FailureEvent:
    return log.record(FailureKind.PARSE_ERROR, contract="c", attempt=n, errors=[], raw=str(n))


def test_retention_limit_keeps_most_recent() -> None:
    log = ValidationLogger(max_events=2)
    for n in (1, 2, 3):
        _fail(log, n)
    assert len(log.events) == 2
    assert [e.attempt for e in log.events] == [2, 3]
    assert log.events[0].attempt == 2
    assert log.events[-1].raw == "3"
    assert log.stats.failures_by_kind == {"parse_error": 3}


def test_default_retention_limit() -> None:
    log = ValidationLogger()
    assert log.max_events == 1000
    assert log.events.maxlen == 1000


def test_unbounded_retention() -> None:
    log = ValidationLogger(max_events=None)
    for n in range(1500):
        _fail(log, n)
    assert log.events.maxlen is None
    assert len(log.events) == 1500
    assert log.stats.failures_by_kind == {"parse_error": 1500}


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_max_events_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="max_events"):
        ValidationLogger(max_events=bad)


def test_clear_keeps_retention_bound() -> None:
    log = ValidationLogger(max_events=2)
    for n in (1, 2, 3):
        _fail(log, n)
    log.clear()
    assert len(log.events) == 0
    assert log.events.maxlen == 2
    for n in (4, 5, 6):
        _fail(log, n)
    assert [e.attempt for e in log.events] == [5, 6]


def test_email_redacted_everywhere(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    path = tmp_path / "failures.jsonl"
    log = ValidationLogger(path=path, redact=redact_patterns(EMAIL_PATTERN))
    error = ErrorDetail(
        loc="email",
        msg="ada@example.com is not allowed",
        type="value_error",
        input="ada@example.com",
    )
    event = log.record(
        FailureKind.VALIDATION_ERROR,
        contract="Person",
        attempt=1,
        errors=[error],
        raw='{"email": "ada@example.com"}',
    )
    assert event.raw == '{"email": "[REDACTED]"}'
    assert event.errors == [
        ErrorDetail(
            loc="email", msg="[REDACTED] is not allowed", type="value_error", input="[REDACTED]"
        )
    ]
    assert log.events[0] is event
    text = path.read_text(encoding="utf-8")
    assert "[REDACTED]" in text
    assert "ada@example.com" not in text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "[REDACTED]" in record.getMessage()
    assert "ada@example.com" not in record.getMessage()
    payload = json.dumps(record.failure_event)  # type: ignore[attr-defined]
    assert "[REDACTED]" in payload
    assert "ada@example.com" not in payload
    # the caller's error object is untouched
    assert error.input == "ada@example.com"


def test_redaction_of_nested_inputs() -> None:
    log = ValidationLogger(redact=lambda text: text.replace("secret", "***"))
    nested = {
        "name": "secret",
        "tags": ["a", "secret", 3],
        "pair": ("secret", None),
        "inner": {"deep": ["x-secret"], "n": 1.5, "flag": True},
    }
    event = log.record(
        FailureKind.VALIDATION_ERROR,
        contract="c",
        attempt=1,
        errors=[ErrorDetail(loc="$", msg="bad secret", type="x", input=nested)],
    )
    assert event.errors[0].msg == "bad ***"
    assert event.errors[0].input == {
        "name": "***",
        "tags": ["a", "***", 3],
        "pair": ("***", None),
        "inner": {"deep": ["x-***"], "n": 1.5, "flag": True},
    }
    assert nested["name"] == "secret"


def test_non_string_input_untouched_by_redaction() -> None:
    calls: list[str] = []

    def redact(text: str) -> str:
        calls.append(text)
        return text

    log = ValidationLogger(redact=redact)
    event = log.record(
        FailureKind.VALIDATION_ERROR,
        contract="c",
        attempt=1,
        errors=[ErrorDetail(loc="n", msg="m", type="x", input=42)],
    )
    assert event.errors[0].input == 42
    assert calls == ["", "m"]


def test_truncation_applied_after_redaction() -> None:
    log = ValidationLogger(max_raw_chars=10, redact=redact_patterns("x+", replacement="-"))
    event = log.record(FailureKind.PARSE_ERROR, contract="c", attempt=1, errors=[], raw="x" * 50)
    assert event.raw == "-"
    assert event.truncated is False
    event = log.record(
        FailureKind.PARSE_ERROR, contract="c", attempt=1, errors=[], raw="ab" + "x" * 3 + "y" * 20
    )
    assert event.raw == "ab-yyyyyyy"
    assert event.truncated is True


def test_failing_redaction_replaces_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    def broken(text: str) -> str:
        raise RuntimeError(f"cannot redact {text}")

    path = tmp_path / "failures.jsonl"
    log = ValidationLogger(path=path, redact=broken)
    event = log.record(
        FailureKind.VALIDATION_ERROR,
        contract="c",
        attempt=1,
        errors=[ErrorDetail(loc="f", msg="leak@example.com", type="x", input=["leak"])],
        raw="leak@example.com",
    )
    assert event.raw == REDACTION_FAILED
    assert event.errors[0].msg == REDACTION_FAILED
    assert event.errors[0].input == [REDACTION_FAILED]
    assert "leak" not in path.read_text(encoding="utf-8")
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "RuntimeError" in messages[0]
    assert all("leak" not in m for m in messages)
    assert log.stats.failures_by_kind == {"validation_error": 1}


def test_redaction_returning_non_string_treated_as_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    def bad(text: str) -> str:
        return None  # type: ignore[return-value]

    log = ValidationLogger(redact=bad)
    event = log.record(FailureKind.PARSE_ERROR, contract="c", attempt=1, errors=[], raw="data")
    assert event.raw == REDACTION_FAILED
    assert "returned NoneType" in caplog.records[0].getMessage()


def test_redact_patterns_with_string_and_compiled_patterns() -> None:
    redact = redact_patterns(r"\d{3}-\d{4}", re.compile("token=\\w+", re.IGNORECASE))
    assert redact("call 555-1234, TOKEN=abc123") == "call [REDACTED], [REDACTED]"
    custom = redact_patterns(EMAIL_PATTERN, replacement="<email>")
    assert custom("a.b+c@mail.example.org and x@y.io") == "<email> and <email>"
    assert custom("no address here") == "no address here"
    assert redact_patterns()("unchanged") == "unchanged"
