"""A tiny in-memory cache with expiry."""

import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, max_items: int, ttl_seconds: int) -> None:
        self._max = max_items
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= time.monotonic():
            del self._data[key]
            return None
        return value

    def set(self, key: str, value: str) -> None:
        self._data[key] = (time.monotonic() + self._ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)  # drop the oldest entry