"""Shared retry-with-backoff helper for network-bound calls."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger("vsa.retry")

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 2,
    backoff: Iterable[float] = (2, 8, 30),
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    label: str = "call",
) -> T:
    """Call ``fn``; on a ``retry_on`` exception sleep per ``backoff`` and retry.

    ``retries`` is the number of *additional* attempts after the first.
    The last sleep value is reused when attempts outnumber backoff entries.
    """
    delays = list(backoff) or [1.0]
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on as exc:
            attempt += 1
            if attempt > retries:
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)]
            logger.warning("%s failed (attempt %d/%d): %s; retrying in %.0fs",
                           label, attempt, retries + 1, exc, delay)
            time.sleep(delay)
