import inspect
import time

import pytest

from structured_output_agent.retry import RetryPolicy, build_repair_message
from structured_output_agent.types import ErrorDetail, FailureKind


def test_defaults() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.initial_delay == 0.0
    assert policy.multiplier == 2.0
    assert policy.max_delay == 10.0
    assert [policy.delay_for(n) for n in (1, 2, 3)] == [0.0, 0.0, 0.0]


def test_exponential_backoff_is_capped() -> None:
    policy = RetryPolicy(max_attempts=4, initial_delay=0.5, multiplier=2, max_delay=1.5)
    assert policy.delay_for(1) == 0.0
    assert [policy.delay_for(n) for n in (2, 3, 4)] == [0.5, 1.0, 1.5]
    assert policy.delay_for(10) == 1.5


def test_constant_delay_with_multiplier_one() -> None:
    policy = RetryPolicy(initial_delay=0.25, multiplier=1)
    assert [policy.delay_for(n) for n in (2, 3, 4)] == [0.25, 0.25, 0.25]


def test_delay_for_rejects_non_positive_attempt() -> None:
    with pytest.raises(ValueError, match="attempt"):
        RetryPolicy().delay_for(0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"initial_delay": -0.1}, "initial_delay"),
        ({"max_delay": -1.0}, "max_delay"),
        ({"multiplier": 0.5}, "multiplier"),
    ],
)
def test_invalid_bounds_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_single_attempt_policy_is_valid() -> None:
    assert RetryPolicy(max_attempts=1).max_attempts == 1


def test_policy_is_frozen() -> None:
    policy = RetryPolicy()
    with pytest.raises(AttributeError):
        policy.max_attempts = 5  # type: ignore[misc]


def test_wait_sleeps_for_delay() -> None:
    slept: list[float] = []
    policy = RetryPolicy(initial_delay=0.5, multiplier=2, max_delay=1.5)
    assert policy.wait(1, sleep=slept.append) == 0.0
    assert policy.wait(2, sleep=slept.append) == 0.5
    assert policy.wait(3, sleep=slept.append) == 1.0
    assert slept == [0.5, 1.0]


def test_wait_skips_sleep_when_zero() -> None:
    slept: list[float] = []
    assert RetryPolicy().wait(3, sleep=slept.append) == 0.0
    assert slept == []


def test_wait_uses_time_sleep_by_default() -> None:
    assert inspect.signature(RetryPolicy.wait).parameters["sleep"].default is time.sleep
    start = time.monotonic()
    assert RetryPolicy(initial_delay=0.01).wait(2) == 0.01
    assert time.monotonic() - start >= 0.009


def test_repair_message_for_validation_error_mentions_missing_field() -> None:
    previous = '{"name": "Ada"}'
    errors = [ErrorDetail(loc="age", msg="Field required", type="missing")]
    message = build_repair_message(
        previous, errors, FailureKind.VALIDATION_ERROR, contract_name="Person"
    )
    assert "failed validation against the `Person` output contract" in message
    assert previous in message
    assert "- age: Field required" in message
    assert "ONLY a corrected JSON object" in message
    assert "schema" in message


def test_repair_message_for_parse_error() -> None:
    errors = [ErrorDetail(loc="$", msg="Expecting value", type="json_parse")]
    message = build_repair_message("not json", errors, FailureKind.PARSE_ERROR)
    assert "could not be parsed as JSON" in message
    assert "not json" in message
    assert "- $: Expecting value" in message
    assert "the output contract" in message
    assert "`" not in message


def test_repair_message_lists_every_error_in_order() -> None:
    errors = [
        ErrorDetail(loc="name", msg="Field required", type="missing"),
        ErrorDetail(loc="tags.0", msg="Input should be a valid string", type="string_type"),
    ]
    message = build_repair_message("{}", errors, FailureKind.VALIDATION_ERROR)
    first = message.index("- name: Field required")
    second = message.index("- tags.0: Input should be a valid string")
    assert first < second


def test_repair_message_truncates_long_previous_output() -> None:
    previous = "x" * 600
    message = build_repair_message(previous, [], FailureKind.PARSE_ERROR)
    assert "x" * 500 in message
    assert "x" * 501 not in message
    assert "truncated, 100 more characters" in message


def test_repair_message_keeps_output_at_limit() -> None:
    previous = "y" * 500
    message = build_repair_message(previous, [], FailureKind.PARSE_ERROR)
    assert previous in message
    assert "truncated" not in message


def test_repair_message_handles_empty_output_and_no_errors() -> None:
    message = build_repair_message("  ", [], FailureKind.PARSE_ERROR)
    assert "(empty response)" in message
    assert "- $: unknown error" in message


def test_repair_message_for_other_kind() -> None:
    errors = [ErrorDetail(loc="$", msg="boom", type="x")]
    message = build_repair_message("{}", errors, FailureKind.TOOL_RESPONSE_ERROR)
    assert "was rejected (tool_response_error)" in message
    assert "- $: boom" in message
