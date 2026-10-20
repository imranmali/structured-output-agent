"""Provider resilience: retry transient LLM failures with jittered backoff.

``RetryingLLM`` and ``AsyncRetryingLLM`` wrap any ``LLMClient`` /
``AsyncLLMClient`` and retry calls that fail with a transient error (rate
limits, overload, server errors, timeouts, connection failures) according to a
``ProviderRetryPolicy``. Permanent errors (invalid requests, authentication
failures, refusals) are raised immediately.

Classification is by exception: by default a call is retried when it raises an
``LLMError`` whose ``retryable`` flag is set (as ``TransientLLMError`` does),
and the wait honours the error's ``retry_after``. ``AnthropicClient`` and
``AsyncAnthropicClient`` map SDK errors onto these exceptions; when wrapping
them, pass ``max_retries=0`` to the client so SDK retries do not multiply with
these ones.

This is distinct from ``RetryPolicy``, which repairs invalid model *output*;
this module only deals with the provider call failing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from .errors import LLMError
from .llm import AsyncLLMClient, LLMClient
from .types import LLMResponse, Message, ToolSpec

_log = logging.getLogger("structured_output_agent")

RetryCallback = Callable[[int, BaseException, float], None]


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """How often and how long to wait when a provider call fails transiently.

    ``max_retries`` is the number of retries after the first call, so at most
    ``max_retries + 1`` calls are made. The wait before retry ``n`` is
    ``min(initial_delay * multiplier ** (n - 1), max_delay)``; with ``jitter``
    a uniformly random fraction of it is used ("full jitter"), which keeps
    many clients from retrying in lockstep.
    """

    max_retries: int = 3
    initial_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 20.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if not math.isfinite(self.initial_delay) or self.initial_delay < 0:
            raise ValueError("initial_delay must be a finite number >= 0")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("multiplier must be a finite number >= 1")
        if not math.isfinite(self.max_delay) or self.max_delay < 0:
            raise ValueError("max_delay must be a finite number >= 0")

    def backoff(
        self,
        retry: int,
        *,
        retry_after: float | None = None,
        rng: Callable[[], float] = random.random,
    ) -> float:
        """Seconds to wait before retry number ``retry`` (1-based).

        A provider-requested ``retry_after`` raises the wait to at least that
        value, still capped at ``max_delay``.
        """
        if retry < 1:
            raise ValueError("retry is 1-based and must be >= 1")
        try:
            base = self.initial_delay * self.multiplier ** (retry - 1)
        except OverflowError:
            base = self.max_delay
        delay = min(base, self.max_delay)
        if self.jitter:
            delay *= rng()
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.max_delay))
        return delay


def is_retryable(exc: BaseException) -> bool:
    """Default classifier: retry ``LLMError`` instances flagged ``retryable``."""
    return isinstance(exc, LLMError) and exc.retryable


def _retry_after(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class _RetryingBase:
    def __init__(
        self,
        policy: ProviderRetryPolicy | None,
        classify: Callable[[BaseException], bool],
        on_retry: RetryCallback | None,
    ) -> None:
        self.policy = policy if policy is not None else ProviderRetryPolicy()
        self.classify = classify
        self.on_retry = on_retry

    def _should_retry(self, retries_done: int, exc: BaseException) -> bool:
        return retries_done < self.policy.max_retries and self.classify(exc)

    def _prepare_retry(self, retry: int, exc: BaseException) -> float:
        """Compute the wait before ``retry``, log it and notify ``on_retry``."""
        delay = self.policy.backoff(retry, retry_after=_retry_after(exc))
        _log.warning(
            "Provider call attempt %d failed with %s: %s; retrying in %.2fs (retry %d of %d)",
            retry,
            type(exc).__name__,
            exc,
            delay,
            retry,
            self.policy.max_retries,
        )
        if self.on_retry is not None:
            try:
                self.on_retry(retry, exc, delay)
            except Exception:
                _log.exception("on_retry callback raised; ignoring")
        return delay


class RetryingLLM(_RetryingBase):
    """``LLMClient`` that retries transient failures of an inner client.

    ``sleep`` and ``classify`` are injectable (for tests or custom provider
    errors); ``on_retry(retry, error, delay)`` is called before each wait.
    When retries are exhausted the last error is re-raised unchanged.
    """

    def __init__(
        self,
        inner: LLMClient,
        policy: ProviderRetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        classify: Callable[[BaseException], bool] = is_retryable,
        on_retry: RetryCallback | None = None,
    ) -> None:
        super().__init__(policy, classify, on_retry)
        self.inner = inner
        self.sleep = sleep

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        retries = 0
        while True:
            try:
                return self.inner.complete(system=system, messages=messages, tools=tools)
            except Exception as exc:
                if not self._should_retry(retries, exc):
                    raise
                retries += 1
                self.sleep(self._prepare_retry(retries, exc))


class AsyncRetryingLLM(_RetryingBase):
    """Async ``RetryingLLM`` wrapping an ``AsyncLLMClient``."""

    def __init__(
        self,
        inner: AsyncLLMClient,
        policy: ProviderRetryPolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        classify: Callable[[BaseException], bool] = is_retryable,
        on_retry: RetryCallback | None = None,
    ) -> None:
        super().__init__(policy, classify, on_retry)
        self.inner = inner
        self.sleep = sleep

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> LLMResponse:
        retries = 0
        while True:
            try:
                return await self.inner.complete(system=system, messages=messages, tools=tools)
            except Exception as exc:
                if not self._should_retry(retries, exc):
                    raise
                retries += 1
                await self.sleep(self._prepare_retry(retries, exc))
