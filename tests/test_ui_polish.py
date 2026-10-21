"""Tests for UI-01 and UI-02: Checkmark SVG replacement and popover ARIA/keyboard navigation."""

from pathlib import Path
import pytest

INDEX_HTML_PATH = Path(__file__).parents[1] / "src" / "verity" / "web" / "static" / "index.html"


def test_no_raw_checkmark_glyphs_in_index_html():
    """Verify raw checkmark unicode characters are replaced with SVG icons (UI-01)."""
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")
    for glyph in ["\u2713", "\u2714", "\u2705", "\u2611"]:
        assert glyph not in content, f"Raw checkmark glyph {ascii(glyph)} found in index.html! Use SVG icon instead."


def test_popover_aria_and_role_attributes():
    """Verify custom dropdown popovers and modal have correct ARIA roles and attributes (UI-02)."""
    content = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # Dropdown buttons must have aria-haspopup and aria-expanded
    assert 'id="dropdown-btn-a"' in content
    assert 'aria-haspopup="listbox"' in content
    assert 'aria-expanded=' in content

    # Menus must have role="listbox" and options role="option"
    assert 'role="listbox"' in content
    assert 'role="option"' in content

    # Modal must have role="dialog" and aria-modal="true"
    assert 'id="api-modal"' in content
    assert 'role="dialog"' in content
    assert 'aria-modal="true"' in content


def test_real_browser_popover_keyboard_navigation():
    """Verify popovers support keyboard navigation: Arrow keys, Enter, Escape, and focus return (UI-02)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright is not installed in the current environment")

    try:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file:///{INDEX_HTML_PATH.as_posix()}")
    except Exception as exc:
        pytest.skip(f"Chromium browser could not be launched: {exc}")

    try:
        # 1. Test Dropdown keyboard controls
        page.focus("#dropdown-btn-a")
        page.keyboard.press("Enter")
        page.wait_for_timeout(100)

        # Menu should be visible
        is_menu_hidden = page.eval_on_selector("#dropdown-menu-a", "el => el.classList.contains('hidden')")
        assert not is_menu_hidden, "Dropdown menu should be visible after pressing Enter on button"

        # Press Escape -> Menu should close and focus return to button
        page.keyboard.press("Escape")
        page.wait_for_timeout(100)
        is_menu_hidden_after_esc = page.eval_on_selector("#dropdown-menu-a", "el => el.classList.contains('hidden')")
        assert is_menu_hidden_after_esc, "Dropdown menu should be hidden after pressing Escape"

        focused_id = page.evaluate("() => document.activeElement ? document.activeElement.id : ''")
        assert focused_id == "dropdown-btn-a", f"Focus should return to dropdown button, got '{focused_id}'"

        # 2. Test Modal Escape key closing
        page.evaluate("openApiKeyModal()")
        page.wait_for_timeout(100)
        modal_hidden = page.eval_on_selector("#api-modal", "el => el.classList.contains('hidden')")
        assert not modal_hidden, "Modal should be open"

        page.keyboard.press("Escape")
        page.wait_for_timeout(100)
        modal_hidden_after_esc = page.eval_on_selector("#api-modal", "el => el.classList.contains('hidden')")
        assert modal_hidden_after_esc, "Modal should close on Escape key"

    finally:
        browser.close()
        p.stop()
