"""
Deterministic programmatic verifier for exact constraints, negative rules,
formatting guidelines, and semantic hygiene checks.

Solves the classic BPE Tokenization Blindness problem where LLMs cannot reliably
count characters, verify lipograms (e.g. forbidden letters), or calculate word counts.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


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
        clean_text = text.strip()
        words = clean_text.split()
        word_count = len(words)
        
        # Simple sentence count based on terminal punctuation
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
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
            # Single unbroken string like 'EeEeEeEeEeEeEeEe' or 'aaaaaaaaaaa'
            # Check unique character ratio
            unique_chars = set(clean_text.lower())
            if len(unique_chars) <= 3:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = (
                    f"Repetitive non-lexical character string ('{clean_text[:20]}...') "
                    "with zero communicative or topical value."
                )
        elif word_count > 0:
            # Check if text is just repeating the same token
            lower_words = [w.lower() for w in words]
            if len(lower_words) > 4 and len(set(lower_words)) == 1:
                report.is_gibberish_or_noise = True
                report.gibberish_reason = f"Repeating identical word '{words[0]}' continuously."

        # 2. Check each constraint
        all_rules = list(constraints)
        if prompt:
            # Extract inline rules from prompt if any
            if "one sentence" in prompt.lower() or "1 sentence" in prompt.lower():
                all_rules.append("Must be exactly 1 sentence")

        for rule in all_rules:
            rule_clean = rule.strip()
            rule_lower = rule_clean.lower()

            # A. Character/letter negation: "Must NOT contain the letter 'e' or 'E'", "without letter x", "no 'z'"
            letter_neg_matches = re.findall(
                r'(?:not\s+contain|without|no|must\s+not\s+have|never\s+use|forbid(?:den)?)\s+(?:the\s+)?(?:letter|character)?\s*[\'\"]?([a-zA-Z])[\'\"]?',
                rule_lower
            )
            # Also catch patterns like "letter 'e' or 'E'"
            alt_letters = re.findall(r'[\'\"]([a-zA-Z])[\'\"]', rule_clean)
            target_letters = set([m.lower() for m in letter_neg_matches] + [m.lower() for m in alt_letters if "letter" in rule_lower or "character" in rule_lower])

            if target_letters:
                for char in target_letters:
                    count = clean_text.lower().count(char)
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

            # B. Forbidden words: "do not use the word 'X'"
            word_neg_match = re.search(r'(?:do\s+not\s+use|must\s+not\s+mention|avoid\s+the\s+word)\s+[\'\"]?([a-zA-Z0-9_\-]+)[\'\"]?', rule_lower)
            if word_neg_match:
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

            # C. Sentence length check
            if "one sentence" in rule_lower or "1 sentence" in rule_lower:
                if sentence_count == 1:
                    report.constraint_audits.append(
                        ConstraintAuditItem(constraint=rule_clean, passed=True, details="PASSED: Exactly 1 sentence.")
                    )
                else:
                    report.constraint_audits.append(
                        ConstraintAuditItem(
                            constraint=rule_clean,
                            passed=False,
                            details=f"VIOLATION: Expected 1 sentence, but found {sentence_count} sentences."
                        )
                    )
                continue

            # D. Word count limit check
            word_limit_match = re.search(r'(?:under|less\s+than|maximum|no\s+more\s+than)\s+(\d+)\s+words', rule_lower)
            if word_limit_match:
                max_w = int(word_limit_match.group(1))
                if word_count <= max_w:
                    report.constraint_audits.append(
                        ConstraintAuditItem(
                            constraint=rule_clean,
                            passed=True,
                            details=f"PASSED: Word count {word_count} is <= {max_w}."
                        )
                    )
                else:
                    report.constraint_audits.append(
                        ConstraintAuditItem(
                            constraint=rule_clean,
                            passed=False,
                            details=f"VIOLATION: Word count {word_count} exceeds limit of {max_w}."
                        )
                    )
                continue

            # Default: Semantic constraint left to LLM
            report.constraint_audits.append(
                ConstraintAuditItem(
                    constraint=rule_clean,
                    passed=None,
                    details="UNVERIFIED (Requires subjective LLM judgment)."
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

        lines = ["### DETERMINISTIC PROGRAMMATIC AUDIT (GROUND TRUTH VERIFICATION)"]
        lines.append("The following programmatic telemetry has been verified deterministically using exact string analysis:")

        # Model A summary
        verified_items_a = [item for item in rep_a.constraint_audits if item.passed is not None]
        unverified_items_a = [item for item in rep_a.constraint_audits if item.passed is None]
        passed_a = sum(1 for item in verified_items_a if item.passed is True)
        total_verified_a = len(verified_items_a)
        num_unverified_a = len(unverified_items_a)

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

        # Model B summary
        verified_items_b = [item for item in rep_b.constraint_audits if item.passed is not None]
        unverified_items_b = [item for item in rep_b.constraint_audits if item.passed is None]
        passed_b = sum(1 for item in verified_items_b if item.passed is True)
        total_verified_b = len(verified_items_b)
        num_unverified_b = len(unverified_items_b)

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

        lines.append("\n**PROPORTIONAL SCORING GUIDELINES**:")
        lines.append("- Score 1.0 (Critical Failure): Non-communicative gibberish, refusal, or zero constraints followed.")
        lines.append("- Score 2.0 - 2.5 (Partial / Mixed): Met secondary formatting (e.g. sentence count) but breached a negative constraint.")
        lines.append("- Score 3.0 - 3.5 (Acceptable): Followed core intent and most constraints with minor lapses.")
        lines.append("- Score 4.0 - 5.0 (High / Flawless): Followed all positive instructions and all negative constraints.")
        lines.append("Base your scores strictly on these verified facts. Do not contradict verified character occurrences.")

        audit_prompt = "\n".join(lines)
        return PairwiseVerificationReport(
            report_a=rep_a,
            report_b=rep_b,
            summary_audit_prompt=audit_prompt
        )
