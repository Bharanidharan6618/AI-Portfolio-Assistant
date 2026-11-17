"""Application configuration.

Reads settings from the .env file and environment variables.
Every other module imports settings from here.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_ORIGINS = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_model: str
    max_question_length: int
    gemini_timeout_seconds: int
    knowledge_file: Path
    allowed_origins: tuple[str, ...]
    max_request_bytes: int
    rate_limit_per_minute: int
    daily_gemini_limit: int
    enable_docs: bool


def _int_env(name: str, default: int) -> int:
    """Read a whole number setting of 1 or more, with a clear error if invalid."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number.") from None
    if value < 1:
        raise RuntimeError(f"{name} must be 1 or greater.")
    return value


def _parse_origins(raw: str) -> tuple[str, ...]:
    origins = tuple(o.strip().rstrip("/") for o in raw.split(",") if o.strip())
    if "*" in origins:
        raise RuntimeError("ALLOWED_ORIGINS must list exact origins, not '*'.")
    return origins


@lru_cache
def get_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to backend/.env "
            "(see .env.example)."
        )

    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL","").strip(),
        max_question_length=_int_env("MAX_QUESTION_LENGTH", 500),
        gemini_timeout_seconds=_int_env("GEMINI_TIMEOUT_SECONDS", 30),
        knowledge_file=BASE_DIR / "app" / "data" / "portfolio_knowledge.txt",
        allowed_origins=_parse_origins(os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS)),
        max_request_bytes=_int_env("MAX_REQUEST_BYTES", 8192),
        rate_limit_per_minute=_int_env("RATE_LIMIT_PER_MINUTE", 10),
        daily_gemini_limit=_int_env("DAILY_GEMINI_LIMIT", 200),
        enable_docs=os.getenv("ENABLE_DOCS", "true").strip().lower()
        in ("1", "true", "yes"),
    )