"""
Deterministic programmatic verifier for exact constraints, negative rules,
formatting guidelines, and semantic hygiene checks.

Solves the classic BPE Tokenization Blindness problem where LLMs cannot reliably
count characters, verify lipograms (e.g. forbidden letters), or calculate word counts.
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
from pydantic import BaseModel, Field

# ==============================================================================
# INPUT SIZE & GUARD CONSTANTS
# ==============================================================================
MAX_VERIFIER_INPUT_CHARS: int = 50_000
"""Maximum input characters audited to avoid ReDoS or catastrophic backtracking."""


class ConstraintAuditItem(BaseModel):
    constraint: str
    passed: Optional[bool] = None
    details: str


class CandidateVerificationReport(BaseModel):
    is_gibberish_or_noise: bool = False
    gibberish_reason: Optional[str] = None
    character_violations: List[str] = Field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    constraint_audits: List[ConstraintAuditItem] = Field(default_factory=list)
    hard_negative_violation: bool = False


class PairwiseVerificationReport(BaseModel):
    report_a: CandidateVerificationReport
    report_b: CandidateVerificationReport
    summary_audit_prompt: str


def _word_to_int(val: str) -> Optional[int]:
    """Converts number words (one..ten) or digits to integer."""
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    if val.isdigit():
        return int(val)
    return mapping.get(val.lower())


def _split_into_sentences(text: str) -> List[str]:
    """
    Robust sentence boundary detector handling:
    - Decimals (e.g. 3.14)
    - Common abbreviations (Dr., Mr., Mrs., Ms., Prof., Sr., Jr., e.g., i.e., U.S., vs., etc.)
    - Ellipses (...)
    - Terminal punctuation followed by closing quotation marks
    """
    if not text.strip():
        return []

    # 1. Protect numbers with decimals (e.g. 3.14)
    t = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL_DOT>\2', text)

    # 2. Protect common abbreviations
    abbrevs = [
        r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Sr\.', r'Jr\.',
        r'e\.g\.', r'i\.e\.', r'U\.S\.', r'vs\.', r'etc\.', r'Inc\.', r'Ltd\.', r'Corp\.', r'D\.C\.'
    ]
    for abb in abbrevs:
        t = re.sub(r'\b' + abb, abb.replace('.', '<ABBREV_DOT>'), t, flags=re.IGNORECASE)

    # 3. Protect ellipses (...)
    t = re.sub(r'\.{3,}', '<ELLIPSES>', t)

    # 4. Handle quotes followed by dialogue tags (e.g. '"Stop!" yelled the guard'): protect punctuation
    punct_map = {'.': '<QUOTE_DOT>', '!': '<QUOTE_EXCL>', '?': '<QUOTE_QMARK>'}
    def _quote_replace(m):
        p = punct_map.get(m.group(1), m.group(1))
        return f"{p}{m.group(2)} {m.group(3)}"
    t = re.sub(r'([.!?])(["\'])\s+([a-z])', _quote_replace, t)

    # 5. Split on terminal punctuation followed by whitespace or end of string
    raw_sentences = re.split(r'[.!?]+(?:\s+|$)', t)
    sentences = []
    for s in raw_sentences:
        clean_s = (
            s.replace('<DECIMAL_DOT>', '.')
            .replace('<ABBREV_DOT>', '.')
            .replace('<ELLIPSES>', '...')
            .replace('<QUOTE_DOT>', '.')
            .replace('<QUOTE_EXCL>', '!')
            .replace('<QUOTE_QMARK>', '?')
            .strip()
        )
        if clean_s:
            sentences.append(clean_s)
    return sentences


def _calculate_word_count(text: str) -> int:
    """
    Calculates word count with consistent handling of:
    - Hyphenated compound words (e.g. 'state-of-the-art', 'self-driving' counted as single words)
    - CJK characters (Chinese, Japanese, Korean counted per character)
    - Non-alphanumeric emoji/symbols
    """
    if not text.strip():
        return 0

    # CJK character ranges: Chinese/Kanji, Kana, Hangul
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
    cjk_chars = cjk_pattern.findall(text)

    # Remove CJK characters to count remaining Latin/alphanumeric tokens
    non_cjk_text = cjk_pattern.sub(' ', text)
    words = [w for w in non_cjk_text.split() if w]
    return len(words) + len(cjk_chars)


class RuleVerifier:
    """
    Deterministic rule verifier running ahead of the LLM Judge to ground
    the evaluation in verified, non-hallucinated facts.
    """

    @classmethod
    def audit_candidate(
        cls,
        text: str,
        constraints: List[str],
        prompt: str = ""
    ) -> CandidateVerificationReport:
        # Size guard against ReDoS or huge payloads
        raw_text = text[:MAX_VERIFIER_INPUT_CHARS] if len(text) > MAX_VERIFIER_INPUT_CHARS else text
        clean_text = raw_text.strip()

        word_count = _calculate_word_count(clean_text)
        sentences = _split_into_sentences(clean_text)
        sentence_count = len(sentences)

        report = CandidateVerificationReport(
            word_count=word_count,
            sentence_count=sentence_count
        )

        # 1. Gibberish / Repetition / Semantic Void Check
        if not clean_text:
            report.is_gibberish_or_noise = True
            report.gibberish_reason = "Empty candidate response."
        elif word_count == 1 and len(clean_text) > 8:
            unique_chars = set(clean_text.lower())
            if len(unique_chars) <= 3:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = (
                    f"Repetitive non-lexical character string ('{clean_text[:20]}...') "
                    "with zero communicative or topical value."
                )
        elif word_count > 0:
            words = clean_text.split()
            lower_words = [w.lower() for w in words]
            if len(lower_words) > 4 and len(set(lower_words)) == 1:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = f"Repeating identical word '{words[0]}' continuously."

        # 2. Check each constraint
        all_rules = list(constraints)
        if prompt:
            # Inline extraction from prompt
            prompt_lower = prompt.lower()
            if "one sentence" in prompt_lower or "1 sentence" in prompt_lower:
                all_rules.append("Must be exactly 1 sentence")

        # Decomposed text for Unicode-aware lipograms (accented chars like 'é' decompose to 'e' + combining mark)
        decomposed_text = unicodedata.normalize('NFKD', clean_text).lower()

        for rule in all_rules:
            rule_clean = rule.strip()
            rule_lower = rule_clean.lower()

            # Guard: token limits must NOT be conflated with word limits
            if "token" in rule_lower:
                report.constraint_audits.append(
                    ConstraintAuditItem(
                        constraint=rule_clean,
                        passed=None,
                        details="Token limit requires tokenizer/model inspection (Requires subjective LLM judgment)."
                    )
                )
                continue

            # A. Negative Character/Lipogram Constraints:
            # Phrasings: "Must NOT contain the letter e", "avoid the letter e", "no e's", "without the letter e",
            # "never use the letter e", "forbid the letter e", "omit the letter e", "lipogram on e", "exclude the letter e"
            letter_neg_matches = re.findall(
                r'(?:not\s+contain|without|no|must\s+not\s+have|never\s+use|forbid(?:den)?|avoid|omit|exclude|lipogram\s+(?:on|without))\s+(?:the\s+)?(?:letter|character)?\s*[\'"]?([a-zA-Z])[\'"]?',
                rule_lower
            )
            no_s_matches = re.findall(r"\bno\s+([a-zA-Z])(?:'s|\b)", rule_lower)
            alt_letters = re.findall(r'[\'"]([a-zA-Z])[\'"]', rule_clean)
            target_letters = set(
                [m.lower() for m in letter_neg_matches] +
                [m.lower() for m in no_s_matches] +
                [m.lower() for m in alt_letters if "letter" in rule_lower or "character" in rule_lower or "lipogram" in rule_lower]
            )

            if target_letters:
                for char in target_letters:
                    count = decomposed_text.count(char)
                    if count > 0:
                        detail = f"VIOLATION: Contains {count} instance(s) of forbidden character '{char}'."
                        report.character_violations.append(f"Forbidden '{char}': {count} occurrences")
                        report.hard_negative_violation = True
                        report.constraint_audits.append(
                            ConstraintAuditItem(constraint=rule_clean, passed=False, details=detail)
                        )
                    else:
                        detail = f"PASSED: 0 instances of forbidden character '{char}'."
                        report.constraint_audits.append(
                            ConstraintAuditItem(constraint=rule_clean, passed=True, details=detail)
                        )
                continue

            # B. Forbidden words: "do not use the word 'X'", "avoid the word X"
            word_neg_match = re.search(
                r'(?:do\s+not\s+use|must\s+not\s+mention|avoid(?:\s+the\s+word)?|forbid(?:\s+the\s+word)?)\s+(?:the\s+word\s+)?[\'"]?([a-zA-Z0-9_\-]+)[\'"]?',
                rule_lower
            )
            if word_neg_match and "letter" not in rule_lower and "character" not in rule_lower:
                forb_word = word_neg_match.group(1).lower()
                if re.search(r'\b' + re.escape(forb_word) + r'\b', clean_text.lower()):
                    report.hard_negative_violation = True
                    report.constraint_audits.append(
                        ConstraintAuditItem(
                            constraint=rule_clean,
                            passed=False,
                            details=f"VIOLATION: Contained forbidden word '{forb_word}'."
                        )
                    )
                else:
                    report.constraint_audits.append(
                        ConstraintAuditItem(
                            constraint=rule_clean,
                            passed=True,
                            details=f"PASSED: Forbidden word '{forb_word}' not found."
                        )
                    )
                continue

            # C. Sentence count checks:
            # "exactly N sentences", "write exactly N sentences", "limit to N sentences", "at most N sentences", "at least N sentences"
            sent_match = re.search(
                r'(?:exactly|write\s+exactly|limit\s+(?:your\s+response\s+)?to|at\s+most|maximum\s+of|no\s+more\s+than|at\s+least)?\s*(\d+|one|two|three|four|five)\s+sentences?',
                rule_lower
            )
            if sent_match:
                target_sent_str = sent_match.group(1)
                target_sent = _word_to_int(target_sent_str)
                if target_sent is not None:
                    is_at_most = any(w in rule_lower for w in ["at most", "maximum", "no more than", "limit"])
                    is_at_least = "at least" in rule_lower

                    if is_at_most:
                        passed = sentence_count <= target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Sentence count {sentence_count} is <= limit of {target_sent}."
                    elif is_at_least:
                        passed = sentence_count >= target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Sentence count {sentence_count} is >= minimum of {target_sent}."
                    else:
                        passed = sentence_count == target_sent
                        status = "PASSED" if passed else "VIOLATION"
                        detail = f"{status}: Expected exactly {target_sent} sentence(s), found {sentence_count}."

                    report.constraint_audits.append(
                        ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                    )
                    continue

            # D. Word count limit checks:
            # "under 50 words", "at most 50 words", "no more than 30 words", "maximum of 25 words", "less than 15 words"
            word_limit_match = re.search(
                r'(?:under|less\s+than|maximum(?:\s+of)?|no\s+more\s+than|at\s+most|limit\s+(?:your\s+response\s+)?to)\s+(\d+)\s+words',
                rule_lower
            )
            if word_limit_match:
                max_w = int(word_limit_match.group(1))
                passed = word_count <= max_w
                status = "PASSED" if passed else "VIOLATION"
                detail = f"{status}: Word count {word_count} is <= limit of {max_w}." if passed else f"{status}: Word count {word_count} exceeds limit of {max_w}."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=passed, details=detail)
                )
                continue

            # E. Format constraints: "json format", "valid json", "respond in json"
            if "json" in rule_lower and ("format" in rule_lower or "valid" in rule_lower or "respond in" in rule_lower):
                is_json = False
                try:
                    candidate_clean = clean_text
                    if candidate_clean.startswith("```json"):
                        candidate_clean = candidate_clean[7:]
                    elif candidate_clean.startswith("```"):
                        candidate_clean = candidate_clean[3:]
                    if candidate_clean.endswith("```"):
                        candidate_clean = candidate_clean[:-3]
                    json.loads(candidate_clean.strip())
                    is_json = True
                except Exception:
                    is_json = False

                detail = "PASSED: Valid JSON format verified." if is_json else "VIOLATION: Response does not parse as valid JSON."
                report.constraint_audits.append(
                    ConstraintAuditItem(constraint=rule_clean, passed=is_json, details=detail)
                )
                continue

            # Default: Qualitative / Semantic constraint left to subjective LLM judgment
            report.constraint_audits.append(
                ConstraintAuditItem(
                    constraint=rule_clean,
                    passed=None,
                    details="Requires subjective LLM judgment."
                )
            )

        return report

    @classmethod
    def verify_pair(
        cls,
        prompt: str,
        constraints: List[str],
        response_a: str,
        response_b: str
    ) -> PairwiseVerificationReport:
        rep_a = cls.audit_candidate(response_a, constraints, prompt)
        rep_b = cls.audit_candidate(response_b, constraints, prompt)

        verified_items_a = [item for item in rep_a.constraint_audits if item.passed is not None]
        unverified_items_a = [item for item in rep_a.constraint_audits if item.passed is None]
        passed_a = sum(1 for item in verified_items_a if item.passed is True)
        total_verified_a = len(verified_items_a)
        num_unverified_a = len(unverified_items_a)

        verified_items_b = [item for item in rep_b.constraint_audits if item.passed is not None]
        unverified_items_b = [item for item in rep_b.constraint_audits if item.passed is None]
        passed_b = sum(1 for item in verified_items_b if item.passed is True)
        total_verified_b = len(verified_items_b)
        num_unverified_b = len(unverified_items_b)

        has_verified_facts = (total_verified_a + total_verified_b) > 0 or rep_a.is_gibberish_or_noise or rep_b.is_gibberish_or_noise or rep_a.character_violations or rep_b.character_violations

        if has_verified_facts:
            lines = ["### DETERMINISTIC PROGRAMMATIC AUDIT (GROUND TRUTH VERIFICATION)"]
            lines.append("The following programmatic telemetry has been verified deterministically using exact string analysis:")
        else:
            lines = ["### CANDIDATE INSTRUCTION & CONSTRAINT AUDIT"]
            lines.append("No programmatic rules were applied. The following qualitative constraints require subjective LLM judgment:")

        # Candidate A section
        if total_verified_a > 0:
            lines.append(f"\n**Candidate A Telemetry ({passed_a}/{total_verified_a} verified constraints passed; {num_unverified_a} unverified/semantic)**:")
        else:
            lines.append(f"\n**Candidate A Telemetry (0 deterministic constraints checked; {num_unverified_a} unverified/semantic constraints require LLM judgment)**:")

        if rep_a.is_gibberish_or_noise:
            lines.append(f"- [CRITICAL FAILURE] Semantic Void / Pure Noise: {rep_a.gibberish_reason} (Cannot win over communicative text)")
        if rep_a.character_violations:
            lines.append(f"- Negative Character Findings: {', '.join(rep_a.character_violations)}")
        for item in rep_a.constraint_audits:
            if item.passed is True:
                status_str = "[PASSED]"
            elif item.passed is False:
                status_str = "[FAILED]"
            else:
                status_str = "[UNVERIFIED]"
            lines.append(f"- Constraint '{item.constraint}': {status_str} ({item.details})")

        # Candidate B section
        if total_verified_b > 0:
            lines.append(f"\n**Candidate B Telemetry ({passed_b}/{total_verified_b} verified constraints passed; {num_unverified_b} unverified/semantic)**:")
        else:
            lines.append(f"\n**Candidate B Telemetry (0 deterministic constraints checked; {num_unverified_b} unverified/semantic constraints require LLM judgment)**:")

        if rep_b.is_gibberish_or_noise:
            lines.append(f"- [CRITICAL FAILURE] Semantic Void / Pure Noise: {rep_b.gibberish_reason} (Cannot win over communicative text)")
        if rep_b.character_violations:
            lines.append(f"- Negative Character Findings: {', '.join(rep_b.character_violations)}")
        for item in rep_b.constraint_audits:
            if item.passed is True:
                status_str = "[PASSED]"
            elif item.passed is False:
                status_str = "[FAILED]"
            else:
                status_str = "[UNVERIFIED]"
            lines.append(f"- Constraint '{item.constraint}': {status_str} ({item.details})")

        if has_verified_facts:
            lines.append("\nBase your scores strictly on these verified facts. Do not contradict verified character occurrences.")
        else:
            lines.append("\nEvaluate all constraints according to your rubric judgment.")

        return PairwiseVerificationReport(
            report_a=rep_a,
            report_b=rep_b,
            summary_audit_prompt="\n".join(lines)
        )
