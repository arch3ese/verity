from aligncraft.judges.rule_verifier import RuleVerifier, _split_into_sentences

probes_e1 = [
    (1, "Write a haiku about rain without using the letter 'e'.", "DET [forbidden letter e]"),
    (2, "No e's allowed.", "DET [forbidden letter e]"),
    (3, "Avoid the letter E entirely.", "DET [forbidden letter e, case-insensitive]"),
    (4, "Your answer must be a lipogram that omits the letter z.", "DET [forbidden letter z]"),
    (5, "Write a paragraph in which the letter 'a' never appears.", "DET [forbidden letter a]"),
    (6, "Summarize this in a single sentence.", "DET [sentences == 1]"),
    (7, "Give me a 3-sentence summary.", "DET [sentences == 3]"),
    (8, "Reply in no more than two sentences.", "DET [sentences <= 2]"),
    (9, "Use between 4 and 6 sentences.", "DET [sentences in 4..6]"),
    (10, "Write exactly five sentences about cats.", "DET [sentences == 5]"),
    (11, "Answer in fewer than 40 words.", "DET [words < 40, strict]"),
    (12, "Keep it to 50 words or less.", "DET [words <= 50]"),
    (13, "Write at least 120 words.", "DET [words >= 120]"),
    (14, "Your response should be about 100 words.", "UNV [approximate]"),
    (15, "Use exactly 25 words.", "DET [words == 25]"),
    (16, "Respond only with valid JSON.", "DET [valid JSON]"),
    (17, "Format the answer as a bulleted list.", "DET [bullet list]"),
    (18, "Use exactly three bullet points.", "DET [bullets == 3]"),
    (19, "Write two paragraphs.", "DET [paragraphs == 2]"),
    (20, "Stay under 100 tokens.", "UNV [tokens are not words]"),
    (21, "Do not use any adjectives.", "UNV [needs POS tagging]"),
    (22, "Be formal and avoid slang.", "UNV"),
    (23, "Do not mention the company's competitors.", "UNV"),
    (24, "Explain why the letter 'e' is the most common letter in English.", "NONE"),
    (25, 'Summarize this article: "The story has 3 sentences. It was short."', "NONE"),
    (26, "The review says the product is not under 100 dollars. Rewrite it politely.", "NONE"),
    (27, "Write an essay about Gadsby, a 1939 novel written with no letter e.", "NONE"),
]

print("=== APPENDIX 1: HELD-OUT CONSTRAINT EXTRACTION PROBES (AFTER) ===")
det_expected_count = 0
det_correct = 0
unv_none_expected_count = 0
unv_none_correct = 0

for idx, text, expected in probes_e1:
    rep = RuleVerifier.audit_candidate("Dummy test response.", [text], "")
    audit_item = rep.constraint_audits[0] if rep.constraint_audits else None
    
    is_det_expected = expected.startswith("DET")
    if is_det_expected:
        det_expected_count += 1
    else:
        unv_none_expected_count += 1

    if not audit_item or audit_item.passed is None:
        extracted = "UNV / NONE"
        details = audit_item.details if audit_item else "None"
        if not is_det_expected:
            unv_none_correct += 1
            status = "MATCH"
        else:
            status = "MISSED DET"
    else:
        extracted = f"DET: {audit_item.details}"
        if is_det_expected:
            det_correct += 1
            status = "MATCH"
        else:
            status = "SPURIOUS DET"
        
    print(f"Probe {idx:2d}: Expected='{expected}' -> Extracted='{extracted}' [{status}]")

recall_det = det_correct / det_expected_count
precision_unv_none = unv_none_correct / unv_none_expected_count

print(f"\nAFTER Recall on Deterministic Constraints: {det_correct}/{det_expected_count} ({recall_det:.1%})")
print(f"AFTER Precision on UNV/NONE Constraints: {unv_none_correct}/{unv_none_expected_count} ({precision_unv_none:.1%})")

print("\n=== APPENDIX 2: SENTENCE SPLITTER PROBES (AFTER) ===")
probes_e2 = [
    ("S1", 'He said "Go." Then he left.', 2),
    ("S2", '"Is it true?" She left.', 2),
    ("S3", "The meeting is at 5 p.m. today. Bring coffee.", 2),
    ("S4", "He earned a Ph.D. in 2019. It took years.", 2),
    ("S5", "I live on Main St. in Pune. It is loud.", 2),
    ("S6", '"Stop!" yelled the guard.', 1),
    ("S7", "Pi is roughly 3.14. That is enough.", 2),
    ("S8", "Visit example.com for details. Then relax.", 2),
    ("S9", "Dr. Lee met Mr. Cho at 9 a.m. They talked.", 2),
    ("S10", 'She asked, "Why?" He shrugged.', 2),
]

e2_correct = 0
for sid, text, expected in probes_e2:
    sents = _split_into_sentences(text)
    actual = len(sents)
    status = "MATCH" if actual == expected else "MISMATCH"
    if status == "MATCH":
        e2_correct += 1
    print(f"{sid:3s}: Expected={expected}, Actual={actual} [{status}] -> {sents}")

print(f"\nAFTER Sentence Splitter Accuracy: {e2_correct}/{len(probes_e2)} ({e2_correct / len(probes_e2):.1%})")
