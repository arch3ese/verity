"""Tests for the Verity Web API and Human Annotator Audit Station."""

import pytest
from fastapi.testclient import TestClient
from verity.web.app import app

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


def test_submit_audit_rejects_whitespace_and_short_rationale_on_disagreement():
    base_disagreement = {
        "sample_id": "test-disagree",
        "prompt": "Test prompt",
        "response_a": "A",
        "response_b": "B",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_b",
        "ai_verdict": "model_a",  # Disagreement
        "human_rubrics": [
            {"dimension": "instruction_following", "score_a": 2.0, "score_b": 4.5, "agreed_with_ai": False, "comment": ""}
        ],
        "ai_rubric_scores": {"instruction_following": {"score_a": 4.0, "score_b": 2.0}}
    }

    # Case 1: Whitespace-only rationale
    payload_spaces = dict(base_disagreement, human_rationale="       ")
    res_spaces = client.post("/api/audit/submit", json=payload_spaces)
    assert res_spaces.status_code == 400
    assert "Mandatory Annotator Rationale" in res_spaces.json()["detail"]

    # Case 2: Newlines and tabs only
    payload_newlines = dict(base_disagreement, human_rationale="\n\t  \r\n")
    res_newlines = client.post("/api/audit/submit", json=payload_newlines)
    assert res_newlines.status_code == 400

    # Case 3: Too short (<8 chars)
    payload_short = dict(base_disagreement, human_rationale="Too bad")
    res_short = client.post("/api/audit/submit", json=payload_short)
    assert res_short.status_code == 400

    # Case 4: When human AGREES with AI and no overrides, empty rationale is accepted
    base_agreement = {
        "sample_id": "test-agree",
        "prompt": "Agreement prompt",
        "response_a": "A",
        "response_b": "B",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_a",
        "ai_verdict": "model_a",  # Agreement!
        "human_rubrics": [
            {"dimension": "instruction_following", "score_a": 4.0, "score_b": 2.0, "agreed_with_ai": True, "comment": ""}
        ],
        "human_rationale": "",  # Empty is fine when agreeing with AI
        "ai_rubric_scores": {"instruction_following": {"score_a": 4.0, "score_b": 2.0}}
    }
    res_agree = client.post("/api/audit/submit", json=base_agreement)
    assert res_agree.status_code == 200



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


def test_export_dpo_triplet_rejects_tie_and_indeterminate():
    base_payload = {
        "sample_id": "test-sample-tie",
        "prompt": "Write a haiku.",
        "response_a": "Leaves fall in the wind / Cool autumn air all around / Silence fills the park",
        "response_b": "Snowflakes softly fall / Winter blankets the cold ground / Quiet starry night",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "tie",
        "ai_verdict": "tie",
        "human_rubrics": [],
        "human_rationale": "Both haikus are equally poetic.",
        "ai_rubric_scores": {}
    }

    # 1. Tie verdict must return HTTP 400
    resp_tie = client.post("/api/audit/export-dpo", json=base_payload)
    assert resp_tie.status_code == 400
    assert "Cannot export DPO" in resp_tie.json()["detail"]

    # 2. Indeterminate verdict must return HTTP 400
    base_payload["human_overall_winner"] = "indeterminate"
    resp_indet = client.post("/api/audit/export-dpo", json=base_payload)
    assert resp_indet.status_code == 400
    assert "Cannot export DPO" in resp_indet.json()["detail"]


def test_export_dpo_triplet_model_b_winner():
    payload_b = {
        "sample_id": "test-sample-b",
        "prompt": "Count to 3.",
        "response_a": "1, 2, 4",
        "response_b": "1, 2, 3",
        "model_a_name": "Candidate A",
        "model_b_name": "Candidate B",
        "human_overall_winner": "model_b",
        "ai_verdict": "model_b",
        "human_rubrics": [],
        "human_rationale": "Candidate B counted correctly.",
        "ai_rubric_scores": {}
    }
    resp = client.post("/api/audit/export-dpo", json=payload_b)
    assert resp.status_code == 200
    data = resp.json()
    assert data["chosen"] == payload_b["response_b"]
    assert data["rejected"] == payload_b["response_a"]

