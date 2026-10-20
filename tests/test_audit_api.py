"""Tests for the AlignCraft Web API and Human Annotator Audit Station."""

import pytest
from fastapi.testclient import TestClient
from aligncraft.web.app import app

client = TestClient(app)


def test_get_guidelines():
    response = client.get("/api/guidelines")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "guidelines" in data
    assert data["total"] >= 3


def test_submit_audit_requires_rationale_on_override():
    # Attempting to override AI verdict without explanatory rationale should fail (HTTP 400)
    payload = {
        "sample_id": "test-sample-01",
        "prompt": "Test prompt",
        "response_a": "Candidate A response",
        "response_b": "Candidate B response",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_b",
        "ai_verdict": "model_a",  # AI preferred A, Human prefers B -> Disagreement!
        "human_rubrics": [
            {
                "dimension": "instruction_following",
                "score_a": 2.0,
                "score_b": 4.5,
                "agreed_with_ai": False,
                "comment": "Candidate A failed constraint"
            }
        ],
        "human_rationale": "",  # Empty rationale!
        "ai_rubric_scores": {
            "instruction_following": {"score_a": 4.0, "score_b": 2.0}
        }
    }
    response = client.post("/api/audit/submit", json=payload)
    assert response.status_code == 400
    assert "Mandatory Annotator Rationale" in response.json()["detail"]


def test_submit_audit_succeeds_with_rationale_on_override():
    payload = {
        "sample_id": "test-sample-02",
        "prompt": "Test prompt with constraints",
        "response_a": "Candidate A response",
        "response_b": "Candidate B response",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_b",
        "ai_verdict": "model_a",
        "human_rubrics": [
            {
                "dimension": "instruction_following",
                "score_a": 2.0,
                "score_b": 4.5,
                "agreed_with_ai": False,
                "comment": "Candidate A failed constraint"
            }
        ],
        "human_rationale": "Candidate B adhered completely to formatting constraints while Candidate A omitted key details.",
        "ai_rubric_scores": {
            "instruction_following": {"score_a": 4.0, "score_b": 2.0}
        }
    }
    response = client.post("/api/audit/submit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "cohens_kappa" in data
    assert "rubric_mae" in data


def test_export_dpo_triplet():
    payload = {
        "sample_id": "test-sample-03",
        "prompt": "Explain gravity in one sentence.",
        "response_a": "Gravity is the attractive force between masses.",
        "response_b": "Gravity is magic that makes things fall down fast.",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_a",
        "ai_verdict": "model_a",
        "human_rubrics": [],
        "human_rationale": "Candidate A is scientifically accurate, B is inaccurate.",
        "ai_rubric_scores": {}
    }
    response = client.post("/api/audit/export-dpo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Explain gravity in one sentence."
    assert "chosen" in data
    assert "rejected" in data
    assert data["chosen"] == "Gravity is the attractive force between masses."
    assert data["rejected"] == "Gravity is magic that makes things fall down fast."
