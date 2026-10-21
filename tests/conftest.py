import sys
import pytest
import json
from pathlib import Path
import verity.alignment.guidelines
import verity.web.app
from verity.alignment.guidelines import DEFAULT_ALIGNMENT_GUIDELINES


@pytest.fixture(autouse=True)
def isolate_data_persistence(tmp_path, monkeypatch):
    """
    Isolates file persistence for every test run using tmp_path.
    Prevents tests from modifying or polluting data/alignment_guidelines.json
    and data/human_audits.jsonl.
    """
    test_guidelines_file = tmp_path / "alignment_guidelines.json"
    initial_data = [g.model_dump() for g in DEFAULT_ALIGNMENT_GUIDELINES]
    test_guidelines_file.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")
    
    gl_module = sys.modules["verity.alignment.guidelines"]
    monkeypatch.setattr(gl_module, "GUIDELINES_FILE", test_guidelines_file)

    test_audits_file = tmp_path / "human_audits.jsonl"
    app_module = sys.modules["verity.web.app"]
    monkeypatch.setattr(app_module, "AUDITS_FILE", test_audits_file)

    yield {
        "guidelines_file": test_guidelines_file,
        "audits_file": test_audits_file,
    }
