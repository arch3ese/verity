"""Concurrency tests for GuidelineBank atomic writes (CAL-02)."""

import json
import threading
from pathlib import Path
from aligncraft.alignment.guidelines import GuidelineBank


def test_concurrent_guideline_writes_no_data_loss(tmp_path, monkeypatch):
    """
    Spawns multiple concurrent threads adding guidelines simultaneously.
    Verifies that all entries are persisted, no entries are lost, and the file
    remains valid JSON (CAL-02).
    """
    test_file = tmp_path / "test_guidelines.json"
    initial_data = [
        {
            "id": "guide-001",
            "scenario_focus": "Baseline",
            "prompt": "Test prompt",
            "constraints": [],
            "human_preferred_winner": "model_a",
            "annotator_rationale": "Initial",
            "key_alignment_rule": "Rule 1",
            "is_active": True,
        }
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(initial_data, f, indent=2)

    monkeypatch.setattr("aligncraft.alignment.guidelines.GUIDELINES_FILE", test_file)

    num_threads = 20
    errors = []

    def writer_task(thread_id: int):
        try:
            GuidelineBank.add_human_override_guideline(
                prompt=f"Prompt from thread {thread_id}",
                constraints=[f"Constraint {thread_id}"],
                human_winner="model_a" if thread_id % 2 == 0 else "model_b",
                annotator_rationale=f"Rationale from thread {thread_id}",
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer_task, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors encountered: {errors}"

    # Verify file is valid JSON
    with open(test_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    # Initial (1) + num_threads (20) = 21 entries total
    assert len(saved_data) == 1 + num_threads, (
        f"Expected {1 + num_threads} entries, but found {len(saved_data)}. "
        f"Concurrent writes caused data loss!"
    )

    # Verify all thread prompts exist
    saved_prompts = {item["prompt"] for item in saved_data}
    for i in range(num_threads):
        assert f"Prompt from thread {i}" in saved_prompts
