import logging
from collections.abc import Iterator

import pytest

from structured_output_agent import (
    AsyncScriptedLLM,
    LLMError,
    LLMResponse,
    Message,
    ScriptedLLM,
    ToolSpec,
    TransientLLMError,
)
from structured_output_agent.llm import AsyncLLMClient, LLMClient
from structured_output_agent.resilience import (
    AsyncRetryingLLM,
    ProviderRetryPolicy,
    RetryingLLM,
    is_retryable,
)

NO_JITTER = ProviderRetryPolicy(initial_delay=1.0, multiplier=2.0, max_delay=5.0, jitter=False)
MESSAGES = [Message(role="user", content="q")]


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class _AsyncSleeps(_Sleeps):
    async def __call__(self, delay: float) -> None:  # type: ignore[override]
        self.delays.append(delay)


def _overloaded() -> TransientLLMError:
    return TransientLLMError("overloaded", status_code=529)


# --- policy -----------------------------------------------------------------


def test_policy_defaults() -> None:
    policy = ProviderRetryPolicy()
    assert (policy.max_retries, policy.initial_delay, policy.multiplier) == (3, 0.5, 2.0)
    assert (policy.max_delay, policy.jitter) == (20.0, True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"initial_delay": -0.1}, "initial_delay"),
        ({"initial_delay": float("nan")}, "initial_delay"),
        ({"multiplier": 0.5}, "multiplier"),
        ({"multiplier": float("inf")}, "multiplier"),
        ({"max_delay": -1.0}, "max_delay"),
        ({"max_delay": float("inf")}, "max_delay"),
    ],
)
def test_policy_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ProviderRetryPolicy(**kwargs)  # type: ignore[arg-type]


def test_backoff_without_jitter_is_capped_exponential() -> None:
    assert [NO_JITTER.backoff(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_backoff_bounds_with_jitter() -> None:
    policy = ProviderRetryPolicy(initial_delay=1.0, multiplier=2.0, max_delay=5.0, jitter=True)
    for retry, upper in [(1, 1.0), (2, 2.0), (3, 4.0)]:
        assert policy.backoff(retry, rng=lambda: 0.0) == 0.0
        assert policy.backoff(retry, rng=lambda: 1.0) == upper
        assert 0.0 <= policy.backoff(retry) <= upper
    assert policy.backoff(2, rng=lambda: 0.25) == 0.5


def test_backoff_honours_retry_after_capped_at_max_delay() -> None:
    policy = ProviderRetryPolicy(initial_delay=1.0, max_delay=5.0, jitter=False)
    assert policy.backoff(1, retry_after=3.0) == 3.0
    assert policy.backoff(1, retry_after=0.2) == 1.0
    assert policy.backoff(1, retry_after=60.0) == 5.0
    jittered = ProviderRetryPolicy(initial_delay=1.0, max_delay=5.0)
    assert jittered.backoff(1, retry_after=3.0, rng=lambda: 0.1) == 3.0


def test_backoff_rejects_non_positive_retry_and_survives_overflow() -> None:
    with pytest.raises(ValueError, match="1-based"):
        NO_JITTER.backoff(0)
    assert NO_JITTER.backoff(100_000) == 5.0


def test_is_retryable() -> None:
    assert is_retryable(TransientLLMError("busy")) is True
    assert is_retryable(LLMError("busy", retryable=True)) is True
    assert is_retryable(LLMError("bad request", status_code=400)) is False
    assert is_retryable(ValueError("x")) is False
    assert is_retryable(TimeoutError()) is False


# --- sync wrapper -----------------------------------------------------------


def test_rate_limited_then_success() -> None:
    inner = ScriptedLLM([TransientLLMError("rate limited", status_code=429), "{}"])
    sleeps = _Sleeps()
    llm = RetryingLLM(inner, NO_JITTER, sleep=sleeps)
    tools = [ToolSpec(name="t", description="d", input_schema={})]

    resp = llm.complete(system="s", messages=MESSAGES, tools=tools)

    assert resp.text == "{}"
    assert len(inner.calls) == 2
    assert inner.calls[1].tools == tools
    assert inner.calls[1].messages == MESSAGES
    assert sleeps.delays == [1.0]
    assert isinstance(llm, LLMClient)


def test_retries_exhausted_raises_last_transient_error() -> None:
    errors = [_overloaded() for _ in range(5)]
    inner = ScriptedLLM(errors)
    sleeps = _Sleeps()
    llm = RetryingLLM(inner, ProviderRetryPolicy(max_retries=2, jitter=False), sleep=sleeps)

    with pytest.raises(TransientLLMError) as info:
        llm.complete(system="s", messages=MESSAGES)

    assert len(inner.calls) == 3
    assert info.value is errors[2]
    assert sleeps.delays == [0.5, 1.0]


def test_permanent_error_is_not_retried() -> None:
    auth = LLMError("invalid x-api-key", status_code=401)
    inner = ScriptedLLM([auth, "{}"])
    sleeps = _Sleeps()
    llm = RetryingLLM(inner, sleep=sleeps)

    with pytest.raises(LLMError) as info:
        llm.complete(system="s", messages=MESSAGES)

    assert info.value is auth
    assert len(inner.calls) == 1
    assert sleeps.delays == []


def test_non_llm_exception_is_not_retried_by_default() -> None:
    inner = ScriptedLLM([RuntimeError("bug"), "{}"])
    llm = RetryingLLM(inner, sleep=_Sleeps())
    with pytest.raises(RuntimeError, match="bug"):
        llm.complete(system="s", messages=MESSAGES)
    assert len(inner.calls) == 1


def test_retry_after_is_respected() -> None:
    inner = ScriptedLLM([TransientLLMError("rate limited", retry_after=3.0), "{}"])
    sleeps = _Sleeps()
    policy = ProviderRetryPolicy(initial_delay=1.0, max_delay=5.0, jitter=False)
    RetryingLLM(inner, policy, sleep=sleeps).complete(system="s", messages=MESSAGES)
    assert sleeps.delays == [3.0]


def test_zero_retries_makes_a_single_call() -> None:
    inner = ScriptedLLM([_overloaded(), "{}"])
    llm = RetryingLLM(inner, ProviderRetryPolicy(max_retries=0), sleep=_Sleeps())
    with pytest.raises(TransientLLMError):
        llm.complete(system="s", messages=MESSAGES)
    assert len(inner.calls) == 1


def test_custom_classifier_and_on_retry_callback() -> None:
    first, second = ConnectionResetError("reset"), ConnectionResetError("reset again")
    inner = ScriptedLLM([first, second, "{}"])
    seen: list[tuple[int, BaseException, float]] = []
    llm = RetryingLLM(
        inner,
        NO_JITTER,
        sleep=_Sleeps(),
        classify=lambda exc: isinstance(exc, ConnectionError),
        on_retry=lambda retry, exc, delay: seen.append((retry, exc, delay)),
    )

    assert llm.complete(system="s", messages=MESSAGES).text == "{}"
    assert seen == [(1, first, 1.0), (2, second, 2.0)]


def test_retries_are_logged_and_failing_callback_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(retry: int, exc: BaseException, delay: float) -> None:
        raise RuntimeError("callback bug")

    inner = ScriptedLLM([TransientLLMError("rate limited", status_code=429), "{}"])
    llm = RetryingLLM(inner, NO_JITTER, sleep=_Sleeps(), on_retry=broken)

    with caplog.at_level(logging.WARNING, logger="structured_output_agent"):
        assert llm.complete(system="s", messages=MESSAGES).text == "{}"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == "structured_output_agent"
    text = warnings[0].getMessage()
    assert "attempt 1" in text
    assert "1.00s" in text
    assert "rate limited" in text
    assert "TransientLLMError" in text
    assert any("on_retry callback raised" in r.getMessage() for r in caplog.records)


def test_wrapper_works_with_any_client() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.outcomes: Iterator[bool] = iter([False, True])

        def complete(self, *, system: str, messages: object, tools: object = ()) -> LLMResponse:
            if not next(self.outcomes):
                raise TransientLLMError("busy")
            return LLMResponse(text="ok")

    llm = RetryingLLM(Flaky(), NO_JITTER, sleep=_Sleeps())  # type: ignore[arg-type]
    assert llm.complete(system="s", messages=[]).text == "ok"


# --- async wrapper ----------------------------------------------------------


async def test_async_rate_limited_then_success() -> None:
    inner = AsyncScriptedLLM([TransientLLMError("rate limited", retry_after=3.0), "{}"])
    sleeps = _AsyncSleeps()
    llm = AsyncRetryingLLM(inner, NO_JITTER, sleep=sleeps)

    resp = await llm.complete(system="s", messages=MESSAGES)

    assert resp.text == "{}"
    assert len(inner.calls) == 2
    assert sleeps.delays == [3.0]
    assert isinstance(llm, AsyncLLMClient)


async def test_async_retries_exhausted() -> None:
    inner = AsyncScriptedLLM([_overloaded() for _ in range(5)])
    sleeps = _AsyncSleeps()
    seen: list[int] = []
    llm = AsyncRetryingLLM(
        inner,
        ProviderRetryPolicy(max_retries=2, jitter=False),
        sleep=sleeps,
        on_retry=lambda retry, exc, delay: seen.append(retry),
    )

    with pytest.raises(TransientLLMError):
        await llm.complete(system="s", messages=MESSAGES)

    assert len(inner.calls) == 3
    assert sleeps.delays == [0.5, 1.0]
    assert seen == [1, 2]


async def test_async_permanent_error_is_not_retried() -> None:
    inner = AsyncScriptedLLM([LLMError("invalid x-api-key", status_code=401), "{}"])
    sleeps = _AsyncSleeps()
    with pytest.raises(LLMError, match="x-api-key"):
        await AsyncRetryingLLM(inner, sleep=sleeps).complete(system="s", messages=MESSAGES)
    assert len(inner.calls) == 1
    assert sleeps.delays == []


async def test_async_default_sleep_is_asyncio_sleep() -> None:
    inner = AsyncScriptedLLM([TransientLLMError("busy"), "{}"])
    policy = ProviderRetryPolicy(initial_delay=0.0, jitter=False)
    llm = AsyncRetryingLLM(inner, policy)
    assert (await llm.complete(system="s", messages=MESSAGES)).text == "{}"
