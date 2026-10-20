"""Unit tests for the In-Context Alignment Guidelines Bank."""

import pytest
from aligncraft.alignment.guidelines import GuidelineBank, AlignmentGuideline


def test_guideline_bank_initialization():
    guidelines = GuidelineBank.get_all_guidelines()
    assert len(guidelines) >= 3
    # Check that default exemplars are present
    categories = [g.get("scenario_focus", "") for g in guidelines]
    assert any("Proportional" in c for c in categories)
    assert any("Safety" in c for c in categories)


def test_calibration_prompt_formatting():
    prompt_str = GuidelineBank.format_calibration_prompt()
    assert "SENIOR HUMAN ANNOTATOR ALIGNMENT GUIDELINES" in prompt_str
    assert "Alignment Principle" in prompt_str
    assert "Annotator Rationale" in prompt_str


def test_add_human_override_guideline():
    initial_count = len(GuidelineBank.get_all_guidelines())
    new_g = GuidelineBank.add_human_override_guideline(
        prompt="Test prompt for guideline addition",
        constraints=["constraint A"],
        human_winner="model_b",
        annotator_rationale="Candidate B followed constraints and exhibited clearer logic.",
        scenario_focus="Test Scenario Override"
    )
    assert new_g["human_preferred_winner"] == "model_b"
    assert len(GuidelineBank.get_all_guidelines()) == initial_count + 1
