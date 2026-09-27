"""Regression tests for the generated public.js bundle (served as public.min.js).

Production incident 2026-09-26 (logged-out pages, e.g. home / sign-in):
1. The custom-select `sync()` called `sel.checkValidity()`, which synchronously
   fires an 'invalid' event when the select is invalid. The 'invalid' handler
   calls `sync()` -> infinite recursion -> "Maximum call stack size exceeded",
   killing all page JS (sign-in / welcome-back UI dead).
2. `initPageRoot` (included in the public bundle) called
   `normalizeClickableAccessibility`, whose definition lived below the
   `@public-js:core-end` marker, so public.min.js threw
   "ReferenceError: normalizeClickableAccessibility is not defined" and page
   init died partway (custom selects, mobile nav rebind, etc. never ran).

public.js = everything in static/app.js above `@public-js:core-end` (+ any
`@public-js:include-start/end` regions). These tests pin the invariants.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"
MARKER = "// @public-js:core-end"


def _read() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _core() -> str:
    src = _read()
    assert MARKER in src, "public-js core-end marker missing from app.js"
    return src.split(MARKER, 1)[0]


def test_no_checkValidity_call_in_app_js():
    """checkValidity() fires 'invalid' synchronously; the custom-select
    'invalid' handler calls sync(), so any checkValidity() inside sync()
    recurses until the stack blows. Use validity.valid (pure check) instead."""
    src = _read()
    hits = [
        line.strip()
        for line in src.splitlines()
        if "checkValidity()" in line and not line.strip().startswith(("//", "*", "#"))
    ]
    assert not hits, f"checkValidity() must not be used in app.js: {hits[:3]}"


def test_accessibility_helpers_defined_before_public_marker():
    """normalizeClickableAccessibility/_makeKeyboardActionable are called by
    public-bundle code (initPageRoot et al.); their definitions must live
    above the marker so the generated public.js includes them."""
    core = _core()
    for name in ("normalizeClickableAccessibility", "_makeKeyboardActionable"):
        assert re.search(rf"(?:^|\n)\s*function {name}\s*\(", core), (
            f"{name} must be defined before {MARKER!r}"
        )


def test_public_bundle_core_contains_helpers():
    """End-to-end on the build rule: the core slice that becomes public.js
    must contain the helper definitions (not just the call sites)."""
    core = _core()
    assert "function normalizeClickableAccessibility" in core
    assert "function _makeKeyboardActionable" in core


def test_no_unguarded_cross_marker_calls():
    """No bare NAME() call in the public bundle may resolve only to a
    top-level definition below the marker, unless guarded by a
    `typeof NAME === 'function'` check. (Nested/local shadowing is fine.)"""
    src = _read()
    core, rest = src.split(MARKER, 1)
    defs_below = set(re.findall(r"^function ([A-Za-z_$][\w$]*)\s*\(", rest, re.M))
    defs_above = set(re.findall(r"(?:^|\n)\s*function ([A-Za-z_$][\w$]*)\s*\(", core))
    defs_above |= set(re.findall(r"window\.([A-Za-z_$][\w$]*)\s*=\s*function", core))
    calls = set(re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", core))
    bad = []
    for name in sorted(calls):
        if name not in defs_below or name in defs_above:
            continue
        for m in re.finditer(r"(?<![\w$.])" + re.escape(name) + r"\s*\(", core):
            ctx = core[max(0, m.start() - 120):m.end()]
            if f"typeof {name}" not in ctx and f"typeof window.{name}" not in ctx:
                bad.append(name)
                break
    assert not bad, (
        f"calls in public bundle with no definition above the marker: {bad}"
    )
