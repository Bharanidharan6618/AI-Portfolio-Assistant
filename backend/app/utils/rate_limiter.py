"""Small in-memory limiters. No extra libraries needed."""

import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone

_SWEEP_THRESHOLD = 5000


class SlidingWindowLimiter:
    """Allow at most `max_requests` per `window_seconds` for each key (e.g. an IP)."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> int:
        """Record a request. Returns 0 if allowed, else seconds to wait."""
        now = time.monotonic()
        if len(self._hits) > _SWEEP_THRESHOLD:
            self._sweep(now)

        hits = self._hits.setdefault(key, deque())
        cutoff = now - self._window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._max:
            return max(1, math.ceil(hits[0] + self._window - now))

        hits.append(now)
        return 0

    def _sweep(self, now: float) -> None:
        """Forget keys that have been quiet for a full window (frees memory)."""
        cutoff = now - self._window
        stale = [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]
        for key in stale:
            del self._hits[key]


class DailyCounter:
    """Counts calls per UTC day and refuses once the daily limit is reached."""

    def __init__(self, max_per_day: int) -> None:
        self._max = max_per_day
        self._day = self._today()
        self._count = 0

    @staticmethod
    def _today():
        return datetime.now(timezone.utc).date()

    def try_increment(self) -> bool:
        today = self._today()
        if today != self._day:
            self._day, self._count = today, 0
        if self._count >= self._max:
            return False
        self._count += 1
        return True

    def seconds_until_reset(self) -> int:
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((tomorrow - now).total_seconds()))