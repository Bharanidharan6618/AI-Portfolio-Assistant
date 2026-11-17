"""API tests. Gemini and the knowledge base are replaced by fakes."""

import pytest

from app.main import app
from app.services.gemini_service import GeminiServiceError
from app.services.knowledge_service import KnowledgeBaseError, get_knowledge_service

CHAT = "/api/chat"
JSON_HEADER = {"Content-Type": "application/json"}


def ask(client, question: str):
    return client.post(CHAT, json={"question": question})


# ---------- Normal endpoints ----------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_portfolio_sections(client):
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert data["sections"] == ["Education", "Skills"]
    assert data["total_sections"] == 2


def test_chat_returns_answer(client, fake_gemini):
    response = ask(client, "What are his technical skills?")
    assert response.status_code == 200
    assert response.json() == {"answer": "Fake answer"}
    assert fake_gemini.calls == 1


# ---------- Input validation ----------

@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "     "},
        {"question": "!!!???"},
        {},
        {"question": "Hi there", "extra": 1},
        {"question": "a" * 600},
        {"question": 123},
        {"question": None},
    ],
)
def test_invalid_bodies_return_422(client, fake_gemini, body):
    response = client.post(CHAT, json=body)
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request."
    assert set(response.json()["errors"][0]) == {"field", "message"}
    assert "aaaa" not in response.text  # the input is never echoed back
    assert fake_gemini.calls == 0  # Gemini was never called


def test_invalid_json_returns_422(client):
    response = client.post(CHAT, content="not json", headers=JSON_HEADER)
    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request."


def test_oversized_body_returns_413(client):
    response = client.post(CHAT, content=b"x" * 9000, headers=JSON_HEADER)
    assert response.status_code == 413


# ---------- Routing ----------

def test_unknown_endpoint_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_wrong_method_returns_405(client):
    assert client.get(CHAT).status_code == 405
    assert client.post("/health").status_code == 405


# ---------- Gemini errors become safe responses ----------

@pytest.mark.parametrize(
    "status_code, retry_after",
    [(429, 60), (500, None), (502, None), (503, None), (504, None)],
)
def test_gemini_errors_become_safe_responses(
    client, fake_gemini, status_code, retry_after
):
    fake_gemini.error = GeminiServiceError("Safe message.", status_code, retry_after)
    response = ask(client, "What are his skills?")
    assert response.status_code == status_code
    assert response.json() == {"detail": "Safe message."}
    if retry_after:
        assert response.headers["retry-after"] == str(retry_after)


def test_unexpected_error_returns_generic_500(client, fake_gemini):
    fake_gemini.error = RuntimeError("secret internal detail")
    response = ask(client, "What are his skills?")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error. Please try again later."
    }
    assert "secret" not in response.text


def test_missing_portfolio_file_returns_safe_503(client):
    def broken():
        raise KnowledgeBaseError("Knowledge file not found: C:\\secret\\path.txt")

    app.dependency_overrides[get_knowledge_service] = broken
    for response in (ask(client, "What are his skills?"), client.get("/api/portfolio")):
        assert response.status_code == 503
        assert "secret" not in response.text
        assert "C:" not in response.text


# ---------- Rate limiting and CORS ----------

def test_rate_limit_blocks_after_five_requests(client):
    responses = [ask(client, "What are his skills?") for _ in range(7)]
    assert [r.status_code for r in responses] == [200] * 5 + [429, 429]
    assert int(responses[5].headers["retry-after"]) >= 1


def test_health_is_not_rate_limited(client):
    assert all(client.get("/health").status_code == 200 for _ in range(10))


def test_cors_allows_listed_origin(client):
    response = client.options(
        CHAT,
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5500"


def test_cors_blocks_unknown_origin(client):
    response = client.options(
        CHAT,
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers