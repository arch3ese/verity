"""Tests for API Hardening: Pydantic bounds, Sanitized 500s, and CORS configuration (API-01, API-02, API-03)."""

import pytest
from fastapi.testclient import TestClient
from verity.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_origins_explicit_not_wildcard():
    """Verify CORS middleware uses explicit origins rather than wildcard with credentials (API-03)."""
    from starlette.middleware.cors import CORSMiddleware
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            allow_origins = middleware.kwargs.get("allow_origins", [])
            allow_credentials = middleware.kwargs.get("allow_credentials", False)
            if allow_credentials:
                assert "*" not in allow_origins, "Wildcard origin '*' must not be used when allow_credentials=True!"
                assert len(allow_origins) > 0, "Explicit origins must be provided!"


def test_pydantic_prompt_max_length_validation(client):
    """Verify prompt exceeding max_length is rejected with 422 Unprocessable Entity (API-01)."""
    # Default prompt max length is 50,000 chars. Send 50,001 chars.
    huge_prompt = "A" * 50001
    payload = {
        "prompt": huge_prompt,
        "response_a": "Normal response",
        "response_b": "Normal response",
        "provider": "mock",
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 422
    assert "length" in str(response.json()).lower() or "max_length" in str(response.json()).lower()


def test_sanitized_500_errors_do_not_leak_internal_details(client, monkeypatch):
    """Verify server 500 errors log details internally and return a generic sanitized message (API-02)."""
    # Monkeypatch DebiasingJudge to throw an internal error containing a simulated sensitive credential
    secret_leak = "FATAL_SECRET_DB_PASSWD_xyz987"

    from verity.judges.debiaser import DebiasingJudge

    def broken_evaluate(self, *args, **kwargs):
        raise RuntimeError(f"Database crash: {secret_leak}")

    monkeypatch.setattr(DebiasingJudge, "evaluate_with_debiasing", broken_evaluate)

    payload = {
        "prompt": "Test prompt",
        "response_a": "Response A",
        "response_b": "Response B",
        "provider": "mock",
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 500
    resp_text = response.text
    # Sensitive credential must NOT be in the client response
    assert secret_leak not in resp_text, "Internal exception detail was leaked to the client!"
    # Response must contain a generic sanitized message
    assert "internal" in resp_text.lower() or "error occurred" in resp_text.lower()
