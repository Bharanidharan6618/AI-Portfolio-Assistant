"""Application configuration.

Reads settings from the .env file and environment variables.
Every other module imports settings from here.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# backend/ folder (two levels above this file: app/config.py -> backend/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env if it exists. On Render there is no .env file;
# the real environment variables are used instead.
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    max_question_length: int
    gemini_timeout_seconds: int
    knowledge_file: Path


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
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip(),
        max_question_length=int(os.getenv("MAX_QUESTION_LENGTH", "500")),
        gemini_timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")),
        knowledge_file=BASE_DIR / "app" / "data" / "portfolio_knowledge.txt",
    )