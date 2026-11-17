"""Shared test setup. Tests never use your real API key or the internet."""

import os

# These must be set BEFORE the app is imported. Real environment variables
# take priority over the .env file, so tests ignore your real settings.
os.environ["GEMINI_API_KEY"] = "test-key-not-real"
os.environ["GEMINI_MODEL"] = "test-model"
os.environ["GEMINI_FALLBACK_MODEL"] = ""
os.environ["MAX_QUESTION_LENGTH"] = "500"
os.environ["MAX_REQUEST_BYTES"] = "8192"
os.environ["RATE_LIMIT_PER_MINUTE"] = "5"
os.environ["DAILY_GEMINI_LIMIT"] = "1000"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:5500"
os.environ["ENABLE_DOCS"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.dependencies import get_rate_limiter  # noqa: E402
from app.main import app  # noqa: E402
from app.services.gemini_service import get_gemini_service  # noqa: E402
from app.services.knowledge_service import get_knowledge_service  # noqa: E402


class FakeKnowledge:
    section_titles = ["Education", "Skills"]

    def get_relevant_context(self, question: str) -> str:
        return "Skills: Python, FastAPI"


class FakeGemini:
    """Stands in for the real Gemini service. Can be told to raise an error."""

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    async def generate_answer(self, question: str, context: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "Fake answer"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    get_rate_limiter.cache_clear()


@pytest.fixture
def fake_gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def client(fake_gemini):
    app.dependency_overrides[get_knowledge_service] = lambda: FakeKnowledge()
    app.dependency_overrides[get_gemini_service] = lambda: fake_gemini
    # raise_server_exceptions=False lets us see the 500 response a visitor would get.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()