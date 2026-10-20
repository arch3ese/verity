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
        assert "Requires subjective LLM judgment" in item.details

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


def test_telemetry_prompt_zero_deterministic_never_claims_passed_ratio():
    """
    COR-01 / Task 2.0c telemetry verification:
    Asserts that when all constraints are semantic (0 deterministic):
    1. Header is '### CANDIDATE INSTRUCTION & CONSTRAINT AUDIT' (not GROUND TRUTH VERIFICATION).
    2. Prompt does NOT claim 'verified deterministically' or 'Base your scores strictly on these verified facts'.
    3. Telemetry explicitly says '0 deterministic constraints checked; 4 unverified/semantic constraints require LLM judgment'.
    4. Telemetry NEVER claims '4/4 passed', '0/0 passed', or 'N/N passed'.
    5. No doubled label '[UNVERIFIED] (UNVERIFIED (Requires ...))'.
    6. In mixed case, GROUND TRUTH VERIFICATION and verified fact directives ARE present.
    """
    semantic_rules = [
        "Adopt a Shakespearean tone",
        "Explain quantum physics accurately",
        "Be concise and clear",
        "Show empathy"
    ]
    pair_report = RuleVerifier.verify_pair("Physics topic", semantic_rules, "Candidate A response", "Candidate B response")
    prompt_text = pair_report.summary_audit_prompt

    # Header and grounding checks for zero-deterministic case
    assert "### CANDIDATE INSTRUCTION & CONSTRAINT AUDIT" in prompt_text
    assert "GROUND TRUTH VERIFICATION" not in prompt_text
    assert "verified deterministically" not in prompt_text
    assert "Base your scores strictly on these verified facts" not in prompt_text

    # Wording check
    assert "0 deterministic constraints checked; 4 unverified/semantic constraints require LLM judgment" in prompt_text
    
    # Must never claim N/N passed or 4/4 passed
    assert "4/4" not in prompt_text
    assert "0/0" not in prompt_text
    assert "passed constraints" not in prompt_text.lower()
    assert "verified constraints passed" not in prompt_text

    # No doubled label
    assert "[UNVERIFIED] (UNVERIFIED" not in prompt_text
    for rule in semantic_rules:
        assert f"- Constraint '{rule}': [UNVERIFIED] (Requires subjective LLM judgment.)" in prompt_text

    # Mixed case verification
    mixed_rules = [
        "Must NOT contain the letter e or E",
        "Adopt a Shakespearean tone",
        "Must be exactly 1 sentence"
    ]
    mixed_report = RuleVerifier.verify_pair("Physics topic", mixed_rules, "A quick fox jumps.", "An elephant was here.")
    mixed_prompt = mixed_report.summary_audit_prompt

    assert "### DETERMINISTIC PROGRAMMATIC AUDIT (GROUND TRUTH VERIFICATION)" in mixed_prompt
    assert "The following programmatic telemetry has been verified deterministically using exact string analysis:" in mixed_prompt
    assert "Base your scores strictly on these verified facts. Do not contradict verified character occurrences." in mixed_prompt
    assert "2/2 verified constraints passed; 1 unverified/semantic" in mixed_prompt
    assert "1/2 verified constraints passed; 1 unverified/semantic" in mixed_prompt

def test_rule_verifier_oversize_input_marked_unverified_never_passed():
    """
    Task E3: Input exceeding MAX_VERIFIER_INPUT_CHARS (50,000) with a violation
    beyond 50,000 characters must return [UNVERIFIED] (passed=None), NEVER [PASSED].
    """
    # 54,000 characters without 'z', followed by 'z' at index 55,000, padded to 60,000 chars
    prefix = "The quick brown fox jumps over the lazy dog without letter twenty-six. " * 800  # ~56,800 chars
    # Ensure no 'z' or 'Z' in prefix
    clean_prefix = prefix.replace('z', 's').replace('Z', 'S')[:55_000]
    violation_tail = "Here is the forbidden character z at char 55001. " + ("safe text padding " * 300)
    oversize_text = clean_prefix + violation_tail
    assert len(oversize_text) > 55_000, f"Expected > 55,000 chars, got {len(oversize_text)}"

    constraints = ["Must NOT contain the letter 'z' or 'Z'"]
    report = RuleVerifier.audit_candidate(oversize_text, constraints)

    # In the first 50,000 chars, 'z' does not appear. But because text > 50,000 chars,
    # the verifier must NOT mark this as passed=True (PASSED). It must mark passed=None ([UNVERIFIED]).
    audit = report.constraint_audits[0]
    assert audit.passed is not True, f"Expected audit.passed to NOT be True on truncated input, got {audit.passed}"
    assert audit.passed is None, f"Expected passed=None ([UNVERIFIED]), got {audit.passed}"
    assert "[UNVERIFIED]" in audit.details
    assert "Input exceeds verifier limit: 50,000 characters" in audit.details


def test_rule_verifier_ligature_and_special_character_lipograms():
    """
    Task E4: German eszett 'ß' folds to 'ss', 'æ' to 'ae', 'ø' to 'o', 'œ' to 'oe'.
    Lipograms forbidding 's', 'e', or 'o' must flag these edge characters even when
    the text contains no literal standard ASCII letters.
    """
    # 1. German 'ß' contains 's' (Fuß has no literal 's')
    rep_s = RuleVerifier.audit_candidate("Fuß", ["Must NOT contain the letter 's'"])
    assert rep_s.hard_negative_violation is True
    assert any(a.passed is False for a in rep_s.constraint_audits)

    # 2. Latin 'æ' contains 'e' (Cæsar has no literal 'e')
    rep_ae = RuleVerifier.audit_candidate("Cæsar", ["Must NOT contain the letter 'e'"])
    assert rep_ae.hard_negative_violation is True
    assert any(a.passed is False for a in rep_ae.constraint_audits)

    # 3. Latin 'ø' contains 'o' (Mjød has no literal 'o')
    rep_o = RuleVerifier.audit_candidate("Mjød", ["Must NOT contain the letter 'o'"])
    assert rep_o.hard_negative_violation is True
    assert any(a.passed is False for a in rep_o.constraint_audits)

    # 4. Latin 'œ' contains 'e' (Cœur has no literal 'e')
    rep_oe = RuleVerifier.audit_candidate("Cœur", ["Must NOT contain the letter 'e'"])
    assert rep_oe.hard_negative_violation is True
    assert any(a.passed is False for a in rep_oe.constraint_audits)


def test_word_count_emoji_and_standalone_symbols():
    """
    Task E5: Non-alphanumeric emojis and standalone symbols must not be counted as words.
    """
    from aligncraft.judges.rule_verifier import _calculate_word_count

    # "Hello world! 😃 🌟 - ... 123" -> words are "Hello", "world!", "123" (count = 3)
    text = "Hello world! 😃 🌟 - ... 123"
    assert _calculate_word_count(text) == 3

    # Compound hyphenated word is 1 word, emoji is 0 words
    text2 = "state-of-the-art AI 🎉"
    assert _calculate_word_count(text2) == 2
