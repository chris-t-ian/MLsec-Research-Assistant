"""Per-session rate limit: at most ``limits.messages_per_minute`` messages per sliding minute.

The UI calls :meth:`SessionRateLimiter.check` once per incoming message, before any model
call, with its session key (the chat thread id). Rejected messages do not count against the
window, so a user who waits ``retry_after_s`` seconds can always continue.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from research_assistant.config import Settings, get_settings

WINDOW_S = 60.0
# Sessions idle for longer than this are dropped from memory on the next check.
_IDLE_DROP_S = 10 * WINDOW_S


class RateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    retry_after_s: int = 0

    @property
    def message(self) -> str:
        """User-facing text for a rejected message."""
        unit = "second" if self.retry_after_s == 1 else "seconds"
        return (
            "You're sending messages faster than this demo allows. "
            f"Please wait {self.retry_after_s} {unit} and try again."
        )


class SessionRateLimiter:
    """Sliding-window limiter keyed by session id (thread-safe, in memory)."""

    def __init__(
        self,
        max_messages: int,
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, session_id: str) -> RateDecision:
        """Count one message for ``session_id`` if it is within the limit."""
        now = self._clock()
        with self._lock:
            self._drop_idle(now)
            hits = self._hits.setdefault(session_id, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.max_messages:
                wait = self.window_s - (now - hits[0])
                return RateDecision(allowed=False, retry_after_s=max(1, math.ceil(wait)))
            hits.append(now)
            return RateDecision(allowed=True)

    def _drop_idle(self, now: float) -> None:
        idle = [k for k, h in self._hits.items() if not h or now - h[-1] > _IDLE_DROP_S]
        for k in idle:
            del self._hits[k]


_default: SessionRateLimiter | None = None
_default_lock = threading.Lock()


def default_rate_limiter(settings: Settings | None = None) -> SessionRateLimiter:
    """Process-wide limiter configured from ``limits.messages_per_minute``."""
    global _default
    with _default_lock:
        if _default is None:
            s = settings or get_settings()
            _default = SessionRateLimiter(s.limits.messages_per_minute)
        return _default


__all__ = ["RateDecision", "SessionRateLimiter", "default_rate_limiter"]
