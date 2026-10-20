"""
Tests for COR-03/04/05/06: Rule verifier regex, sentence counting, unicode lipogram,
word counting (hyphenation, emoji, CJK), and input size guard.
"""

import unicodedata
from aligncraft.judges.rule_verifier import RuleVerifier, CandidateVerificationReport


def test_sentence_splitting_decimals_abbreviations_ellipses_quotes():
    """
    COR-04: Verify sentence counting does not miscount decimals, abbreviations, ellipses, or quotes.
    """
    # 1. Decimals: 3.14 should not split
    text_decimals = "The ratio of circumference to diameter is 3.14. It is known as pi."
    rep = RuleVerifier.audit_candidate(text_decimals, ["Must be exactly 2 sentences"])
    assert rep.sentence_count == 2, f"Expected 2 sentences for decimals, got {rep.sentence_count}"

    # 2. Abbreviations: Dr., e.g., U.S. should not split
    text_abbrev = "Dr. Smith visited Washington, D.C. yesterday. He met with U.S. officials e.g. senators."
    rep_abbrev = RuleVerifier.audit_candidate(text_abbrev, ["Must be exactly 2 sentences"])
    assert rep_abbrev.sentence_count == 2, f"Expected 2 sentences for abbreviations, got {rep_abbrev.sentence_count}"

    # 3. Ellipses: ... should not create empty or false sentences
    text_ellipses = "Wait... what happened? I am not sure."
    rep_ellipses = RuleVerifier.audit_candidate(text_ellipses, ["Must be exactly 2 sentences"])
    assert rep_ellipses.sentence_count == 2, f"Expected 2 sentences for ellipses, got {rep_ellipses.sentence_count}"

    # 4. Quotes: terminal punctuation inside quotes followed by dialogue tag
    text_quotes = '"Stop right there!" yelled the guard. He drew his sword.'
    rep_quotes = RuleVerifier.audit_candidate(text_quotes, ["Must be exactly 2 sentences"])
    assert rep_quotes.sentence_count == 2, f"Expected 2 sentences for quotes, got {rep_quotes.sentence_count}"


def test_lipogram_unicode_nfkd_accented_characters():
    """
    COR-03: Accented characters (e.g. 'é' in 'café' or 'cliché') must be decomposed
    via NFKD normalization so that forbidden base letters are caught.
    """
    prompt = "Write without the letter e"
    rules = ["Must NOT contain the letter e or E"]

    # 'café' with precomposed é (\u00e9) has no standard ASCII 'e'; only accented é
    text_accented = "A quick brown fox jumps at a caf\u00e9."
    rep = RuleVerifier.audit_candidate(text_accented, rules, prompt)
    assert rep.hard_negative_violation is True, "Expected violation for accented e (é)!"
    assert any("Forbidden 'e'" in v for v in rep.character_violations)

    # Pure lipogram without any form of e
    text_clean = "A quick brown fox jumps on a lazy dog."
    rep_clean = RuleVerifier.audit_candidate(text_clean, rules, prompt)
    assert rep_clean.hard_negative_violation is False
    assert len(rep_clean.character_violations) == 0


def test_word_count_hyphenated_and_cjk():
    """
    COR-05: Hyphenated compound words treated consistently; CJK counted by characters.
    """
    # Hyphenated compounds: "state-of-the-art" is 1 compound token, "self-driving" is 1
    text_hyphen = "A state-of-the-art self-driving car."
    rep_hyphen = RuleVerifier.audit_candidate(text_hyphen, [])
    assert rep_hyphen.word_count == 4, f"Expected 4 words, got {rep_hyphen.word_count}"

    # CJK: Chinese string with 8 characters must not be counted as 1 word!
    text_cjk = "人工智能改变世界"
    rep_cjk = RuleVerifier.audit_candidate(text_cjk, [])
    assert rep_cjk.word_count == 8, f"Expected 8 CJK words/chars, got {rep_cjk.word_count}"


def test_twenty_phrasing_extraction_probe():
    """
    COR-03 / Task 2.5: 20-phrasing constraint probe.
    Tests extraction across diverse phrasings and verifies that token limits are NOT
    conflated with word limits.
    """
    phrasings = [
        ("Must NOT contain the letter e", "letter_neg"),
        ("avoid the letter e", "letter_neg"),
        ("no e's", "letter_neg"),
        ("do not use the letter e", "letter_neg"),
        ("without the letter e", "letter_neg"),
        ("never use the letter e", "letter_neg"),
        ("forbid the letter e", "letter_neg"),
        ("omit the letter e", "letter_neg"),
        ("lipogram on e", "letter_neg"),
        ("exclude the letter e", "letter_neg"),
        ("exactly 1 sentence", "sentence_count"),
        ("write exactly 3 sentences", "sentence_count"),
        ("limit to 2 sentences", "sentence_count"),
        ("at most 50 words", "word_count"),
        ("no more than 30 words", "word_count"),
        ("under 40 words", "word_count"),
        ("maximum of 25 words", "word_count"),
        ("less than 15 words", "word_count"),
        ("under 100 tokens", "unverified"),  # Must NOT be conflated with words!
        ("respond in JSON format", "format")
    ]

    sample_text = "A dog barked. A cat meowed."

    extracted_count = 0
    token_rule_conflated = False

    for phrase, expected_type in phrasings:
        rep = RuleVerifier.audit_candidate(sample_text, [phrase])
        item = rep.constraint_audits[0]
        if expected_type == "unverified":
            # "under 100 tokens" must remain UNVERIFIED (passed is None)
            if item.passed is not None:
                token_rule_conflated = True
        else:
            if item.passed is not None:
                extracted_count += 1

    assert token_rule_conflated is False, "Defect: 'under 100 tokens' was conflated with a word limit!"
    # Extraction rate should be high (> 15/19 recognized deterministic rules)
    assert extracted_count >= 15, f"Expected >= 15 extracted constraints out of 19, got {extracted_count}"


def test_input_size_guard_against_regex_blowup():
    """
    COR-06: Very large inputs (e.g. 100,000 characters) must not cause ReDoS or freeze.
    """
    huge_input = ("The quick brown fox jumps over the lazy dog. " * 2500)  # ~112,000 chars
    rules = ["Must NOT contain the letter e", "under 50 words"]

    rep = RuleVerifier.audit_candidate(huge_input, rules)
    assert rep.hard_negative_violation is True  # contains e
    assert rep.word_count > 50
