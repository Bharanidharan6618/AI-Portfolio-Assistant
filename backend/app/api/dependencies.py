"""Reusable FastAPI dependencies."""

from functools import lru_cache

from fastapi import HTTPException, Request

from app.config import get_settings
from app.utils.rate_limiter import SlidingWindowLimiter


@lru_cache
def get_rate_limiter() -> SlidingWindowLimiter:
    return SlidingWindowLimiter(
        max_requests=get_settings().rate_limit_per_minute, window_seconds=60
    )


async def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    wait_seconds = get_rate_limiter().check(client_ip)
    if wait_seconds:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again shortly.",
            headers={"Retry-After": str(wait_seconds)},
        )