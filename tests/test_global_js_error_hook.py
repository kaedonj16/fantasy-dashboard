"""Global JS error hook in static/app.js (pre-deployment checklist).

Uncaught errors and unhandled promise rejections forward to Sentry, but
only when the lazily loaded Sentry SDK is actually present. The hook
lives in the core section of app.js (before the @public-js:core-end
marker) so the public bundle built from that section carries it too.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
MARKER = "// @public-js:core-end"


def _hook_block() -> str:
    start = APP_JS.index("installGlobalErrorHook")
    end = APP_JS.index("})();", start)
    return APP_JS[start:end]


def test_error_hook_is_in_core_section_before_marker():
    assert MARKER in APP_JS
    assert APP_JS.index("installGlobalErrorHook") < APP_JS.index(MARKER)


def test_error_hook_listens_for_errors_and_unhandled_rejections():
    block = _hook_block()
    assert "window.addEventListener('error'" in block
    assert "window.addEventListener('unhandledrejection'" in block


def test_error_hook_forwards_to_sentry_only_when_present():
    block = _hook_block()
    assert "window.Sentry" in block
    assert "captureException" in block
    assert "captureMessage" in block
    # The Sentry calls sit behind the presence check, not at top level.
    assert block.index("if (!window.Sentry) return;") < block.index("captureException")


def test_error_hook_is_guarded_and_silent():
    block = _hook_block()
    assert "__brGlobalErrorHookInstalled" in block
    assert "console." not in block
