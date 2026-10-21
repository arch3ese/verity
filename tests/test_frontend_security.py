"""
Tests for frontend template sanitization and XSS prevention (SEC-01).
"""

from pathlib import Path
import pytest

INDEX_HTML_PATH = Path(__file__).parents[1] / "src" / "verity" / "web" / "static" / "index.html"


def test_dynamic_interpolations_are_escaped():
    """Verify static HTML template embeds escapeHtml() wrapper on all dynamic text sinks."""
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # Justification interpolation in dimension bars must be escaped
    assert "${escapeHtml(ds.justification)}" in content
    assert "${ds.justification}" not in content.replace("${escapeHtml(ds.justification)}", "")

    # Error tags in candidate error cards must be escaped
    assert "${escapeHtml(e)}" in content

    # Benchmark table items must be escaped
    assert "${escapeHtml(p.sample.id)}" in content


def _get_browser_context():
    """Initializes Playwright and Chromium, skipping cleanly if missing."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright is not installed in the current environment")

    try:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file:///{INDEX_HTML_PATH.as_posix()}")
        return p, browser, page
    except Exception as exc:
        pytest.skip(f"Chromium browser could not be launched: {exc}")


def test_real_browser_xss_payload_in_justification():
    """Real browser test verifying that XSS payloads in dimension justifications remain inert."""
    p, browser, page = _get_browser_context()
    try:
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
        assert xss_fired is False, "XSS executed inside browser via dimension justification!"

        img_nodes_count = page.evaluate("() => document.querySelectorAll('#dimension-bars img').length")
        assert img_nodes_count == 0, f"Expected 0 img nodes in dimension bars, found {img_nodes_count}"
    finally:
        browser.close()
        p.stop()


def test_real_browser_xss_payload_in_errors_and_human_audit_grid():
    """Real browser test verifying that XSS payloads in error tags and human audit grid remain inert."""
    p, browser, page = _get_browser_context()
    try:
        err_xss = '<img src=x onerror="window.__err_xss_fired=true">'
        grid_xss = '<img src=x onerror="window.__grid_xss_fired=true">'
        page.evaluate("""(args) => {
            window.__err_xss_fired = false;
            window.__grid_xss_fired = false;
            const mockData = {
                final_consensus_winner: "model_a",
                positional_bias_detected: false,
                verbosity_bias_flag: false,
                confidence_score: 0.90,
                forward_verdict: {
                    winner: "model_a",
                    margin: "clear",
                    overall_reasoning: "Evaluation completed.",
                    dimension_scores: [
                        { dimension: "instruction_following", score_a: 4.0, score_b: 2.0, justification: args.grid_xss },
                        { dimension: "factuality", score_a: 5.0, score_b: 1.0, justification: "Factuality ok" }
                    ],
                    detected_errors_model_a: [args.err_xss],
                    detected_errors_model_b: [args.err_xss]
                }
            };
            displayEvaluationResults(mockData);
        }""", {"err_xss": err_xss, "grid_xss": grid_xss})

        page.wait_for_timeout(300)

        err_xss_fired = page.evaluate("() => window.__err_xss_fired")
        assert err_xss_fired is False, "XSS executed inside candidate error tags!"

        grid_xss_fired = page.evaluate("() => window.__grid_xss_fired")
        assert grid_xss_fired is False, "XSS executed inside human audit station grid!"

        err_a_imgs = page.evaluate("() => document.querySelectorAll('#errors-model-a img').length")
        err_b_imgs = page.evaluate("() => document.querySelectorAll('#errors-model-b img').length")
        grid_imgs = page.evaluate("() => document.querySelectorAll('#human-rubric-grid img').length")

        assert err_a_imgs == 0, f"Expected 0 img nodes in errors-model-a, found {err_a_imgs}"
        assert err_b_imgs == 0, f"Expected 0 img nodes in errors-model-b, found {err_b_imgs}"
        assert grid_imgs == 0, f"Expected 0 img nodes in human-rubric-grid, found {grid_imgs}"
    finally:
        browser.close()
        p.stop()
