"""Waivers page must survive an in-place soft swap (auto-revalidate / refresh).

brSwapPageRoot replaces #page-root and re-executes inline scripts via
reexecScripts AFTER DOMContentLoaded already fired. Two things must hold:

1. The inline script must be re-runnable: no top-level let/const/class, or the
   second execution throws "already declared" and the whole script dies.
2. Init must not wait for DOMContentLoaded alone: after a swap that event
   never fires again, so wvLoad() has to run immediately.

Regression test for the endlessly-spinning Start/Sit skeletons.
"""

import re

from dashboard_services.pages.waivers_page import build_waivers_body


def _inline_script():
    body = build_waivers_body("fleaflicker", 2026, "92916", {})
    scripts = re.findall(r"<script>(.*?)</script>", body, re.S)
    matches = [s for s in scripts if "function wvLoad(" in s]
    assert matches, "waivers inline script not found"
    return matches[0]


def test_waivers_script_has_no_top_level_lexical_declarations():
    """Top-level let/const/class throw on re-execution (re-runnable contract)."""
    js = _inline_script()
    bad = [
        line.strip()
        for line in js.splitlines()
        if re.match(r"^(let|const|class)\s", line)
    ]
    assert not bad, f"top-level lexical declarations break reexec: {bad[:5]}"


def test_waivers_init_runs_without_domcontentloaded():
    """wvLoad must init immediately when the script re-runs after a swap."""
    js = _inline_script()
    assert "document.readyState" in js
    # The non-loading branch must call wvLoad() directly, not just register it.
    assert re.search(
        r"else\s*\{\s*wvLoad\(\);", js
    ), "wvLoad() must run immediately when readyState is not 'loading'"
    # Every DOMContentLoaded registration of wvLoad must sit inside the
    # readyState guard -- a bare registration never fires after a swap.
    guard_at = js.find("if (document.readyState === 'loading')")
    assert guard_at != -1
    for m in re.finditer(r"addEventListener\('DOMContentLoaded', wvLoad\)", js):
        assert m.start() > guard_at, "unguarded DOMContentLoaded init of wvLoad"
