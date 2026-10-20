"""
Unit tests for deterministic programmatic verifier (RuleVerifier).
"""

from aligncraft.judges.rule_verifier import RuleVerifier


def test_letter_negation_detection():
    prompt = "Explain physics in one sentence."
    constraints = ["Must NOT contain the letter 'e' or 'E'"]
    text_bad = "EeEeEeEeEeEeEeEe"
    text_good = "Physics holds truth without ambiguity."  # No 'e'

    report_bad = RuleVerifier.audit_candidate(text_bad, constraints, prompt)
    assert report_bad.hard_negative_violation is True
    assert report_bad.is_gibberish_or_noise is True

    report_good = RuleVerifier.audit_candidate(text_good, constraints, prompt)
    assert report_good.hard_negative_violation is False
    assert report_good.is_gibberish_or_noise is False


def test_sentence_count_constraint():
    prompt = "Summary"
    constraints = ["Must be exactly 1 sentence"]
    text_one = "This is a single complete sentence."
    text_three = "This is sentence one. This is sentence two! And sentence three?"

    rep_one = RuleVerifier.audit_candidate(text_one, constraints, prompt)
    assert rep_one.sentence_count == 1
    assert any(a.passed for a in rep_one.constraint_audits)

    rep_three = RuleVerifier.audit_candidate(text_three, constraints, prompt)
    assert rep_three.sentence_count == 3
    assert any(not a.passed for a in rep_three.constraint_audits)


def test_gibberish_and_semantic_void_detection():
    rep_empty = RuleVerifier.audit_candidate("", [])
    assert rep_empty.is_gibberish_or_noise is True

    rep_repeated = RuleVerifier.audit_candidate("noise noise noise noise noise noise", [])
    assert rep_repeated.is_gibberish_or_noise is True


def test_unparsed_semantic_constraints_report_unverified_not_passed():
    semantic_rules = ["Adopt a Shakespearean tone", "Explain quantum physics accurately"]
    rep = RuleVerifier.audit_candidate("Hello there world.", semantic_rules)
    
    # Unparsed semantic constraints must have passed=None (UNVERIFIED), not True
    assert len(rep.constraint_audits) == 2
    for item in rep.constraint_audits:
        assert item.passed is None, f"Expected passed=None for unparsed rule, got {item.passed}"
        assert "UNVERIFIED" in item.details

    # Verify telemetry formatting excludes unverified from pass ratio
    pair_report = RuleVerifier.verify_pair("Topic", semantic_rules, "Resp A", "Resp B")
    assert "2/2 constraints passed" not in pair_report.summary_audit_prompt
    assert "0 deterministic constraints checked; 2 unverified/semantic" in pair_report.summary_audit_prompt


def test_mixed_deterministic_and_semantic_constraints_split():
    mixed_rules = ["Must NOT contain the letter 'e' or 'E'", "Adopt a poetic style"]
    rep = RuleVerifier.audit_candidate("A quick brown fox.", mixed_rules)
    
    # Rule 1: 'e' constraint -> deterministically verified (clean text has no 'e' -> True)
    # Rule 2: poetic style -> unverified (None)
    assert rep.constraint_audits[0].passed is True
    assert rep.constraint_audits[1].passed is None

    pair_report = RuleVerifier.verify_pair("Topic", mixed_rules, "A quick brown fox.", "An elephant.")
    assert "1/1 verified constraints passed; 1 unverified/semantic" in pair_report.summary_audit_prompt

