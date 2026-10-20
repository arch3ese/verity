"""
Tests for frontend template sanitization and XSS prevention (SEC-01).
"""

import re
from pathlib import Path

INDEX_HTML_PATH = Path(__file__).parents[1] / "src" / "aligncraft" / "web" / "static" / "index.html"


def test_escape_html_helper_defined_in_index():
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    assert "function escapeHtml(str)" in content
    assert ".replace(/</g, '&lt;')" in content
    assert ".replace(/>/g, '&gt;')" in content


def test_dynamic_interpolations_are_escaped():
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    
    # Justification interpolation in dimension bars must be escaped
    assert "${escapeHtml(ds.justification)}" in content
    assert "${ds.justification}" not in content.replace("${escapeHtml(ds.justification)}", "")

    # Error tags in candidate error cards must be escaped
    assert "${escapeHtml(e)}" in content

    # Benchmark table items must be escaped
    assert "${escapeHtml(p.sample.id)}" in content


def test_xss_payload_escaping_simulation():
    """Simulate the escapeHtml JS function in python to verify inert rendering."""
    payload = '<img src=x onerror=alert(1)>'
    
    def js_escape_html(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    
    escaped = js_escape_html(payload)
    assert "<" not in escaped
    assert ">" not in escaped
    assert escaped == "&lt;img src=x onerror=alert(1)&gt;"


def test_real_browser_xss_payload_in_justification():
    """Real browser test using Playwright Chromium to verify that XSS payloads remain inert."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file:///{INDEX_HTML_PATH.as_posix()}")

        xss_payload = '<img src=x onerror="window.__xss_fired=true">'
        page.evaluate("""(xss) => {
            window.__xss_fired = false;
            const mockData = {
                final_consensus_winner: "model_a",
                positional_bias_detected: false,
                verbosity_bias_flag: false,
                confidence_score: 0.95,
                forward_verdict: {
                    winner: "model_a",
                    margin: "clear",
                    overall_reasoning: "Safe reasoning.",
                    dimension_scores: [
                        { dimension: "instruction_following", score_a: 5.0, score_b: 2.0, justification: xss }
                    ],
                    detected_errors_model_a: [],
                    detected_errors_model_b: []
                }
            };
            displayEvaluationResults(mockData);
        }""", xss_payload)

        page.wait_for_timeout(300)
        xss_fired = page.evaluate("() => window.__xss_fired")
        assert xss_fired is False, "XSS executed inside browser!"

        img_nodes_count = page.evaluate("() => document.querySelectorAll('#dimension-bars img').length")
        assert img_nodes_count == 0, f"Expected 0 img nodes, found {img_nodes_count}"

        browser.close()

