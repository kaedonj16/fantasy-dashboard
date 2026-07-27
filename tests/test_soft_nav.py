"""Tests for the SPA soft-navigation (fetch + swap #page-root) in static/app.js.

Soft-nav has two halves this suite locks down:

  * the JS ``softNavigable()`` allowlist — the gate that decides which pages get
    swapped in place vs. handed to the browser for a full load. A silent typo
    here (a page dropped from the list, or a heavy page wrongly added) breaks
    navigation with no error, so we run the *real* extracted source through Node
    against a case table rather than re-implementing it here.

  * the server-rendered contract the swap + focus management depend on: a
    focusable ``#page-root`` (so a soft-nav can move focus into the new content
    and the skip link works) and the skip link that targets it.

The Node half is skipped when Node.js isn't available; the render half is skipped
when Flask/pandas aren't installed. Both run in CI where the full stack is present.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_JS = os.path.join(_REPO_ROOT, "static", "app.js")

# Pull the SOFT_NAV_PAGES allowlist + softNavigable() verbatim out of app.js so a
# change to either flows straight into this test — we execute the shipped source,
# never a copy of it.
_SOFT_NAV_RE = re.compile(
    r"var SOFT_NAV_PAGES = \{.*?"
    r"function softNavigable\(href\) \{.*?return SOFT_NAV_PAGES\[seg\] === 1;\s*\}",
    re.DOTALL,
)


def _extract_soft_navigable() -> str:
    src = open(_APP_JS, encoding="utf-8").read()
    m = _SOFT_NAV_RE.search(src)
    assert m, "could not locate SOFT_NAV_PAGES/softNavigable() in static/app.js"
    return m.group(0)


# (href, expected) — href is resolved against a league dashboard URL, so bare and
# relative hrefs exercise the same base the browser uses.
_BASE = "https://ex.com/sleeper/2026/abc/dashboard"
_CASES = [
    ("/sleeper/2026/abc/dashboard", True),
    ("/sleeper/2026/abc/standings", True),
    ("/sleeper/2026/abc/teams", True),
    ("/sleeper/2026/abc/breakouts", True),
    ("/sleeper/2026/abc/waivers", True),
    ("/sleeper/2026/abc/schedule/", True),   # trailing slash still matches
    ("standings", True),                     # relative href resolves off the base
    ("/sleeper/2026/abc/draft", False),      # draft loads its own scripts
    ("/sleeper/2026/abc/draft/history", False),
    ("/sleeper/2026/abc/keeper", False),     # not on the allowlist
    ("/sleeper/2026/abc/redzone", False),
    ("/sleeper/2026/abc/trade", False),
    ("/sleeper/2026/abc/compare", False),
    ("/sleeper/2026/abc/metrics", False),
    ("/sleeper/2026/abc/prospects", False),
    ("/watchlist", False),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_soft_navigable_allowlist():
    import json

    harness = (
        "var location = { href: %s };\n" % json.dumps(_BASE)
        + _extract_soft_navigable()
        + "\nvar cases = " + json.dumps(_CASES) + ";\n"
        + "var bad = [];\n"
        + "cases.forEach(function (c) {\n"
        + "  var got = softNavigable(c[0]);\n"
        + "  if (got !== c[1]) bad.push(c[0] + ' -> ' + got + ' (want ' + c[1] + ')');\n"
        + "});\n"
        + "if (bad.length) { console.error(bad.join('\\n')); process.exit(1); }\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "soft_nav_check.js")
        with open(fp, "w", encoding="utf-8") as fh:
            fh.write(harness)
        res = subprocess.run(["node", fp], capture_output=True, text=True)
    assert res.returncode == 0, "softNavigable() mismatches:\n" + res.stderr


# ── Render contract the swap + focus management rely on ──────────────────────

pytest.importorskip("flask")
pytest.importorskip("pandas")

GRAPHS = "/sleeper/2026/tourdemo/graphs?tour=1"


def _html(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    return r.get_data(as_text=True)


def test_page_root_is_focusable(offline_client):
    # The soft-nav swap moves focus into #page-root after each navigation, so it
    # must be programmatically focusable (tabindex=-1) and the labelled main.
    html = _html(offline_client, GRAPHS)
    m = re.search(r"<main[^>]*id=\"page-root\"[^>]*>", html)
    assert m, "expected a <main id=\"page-root\"> element"
    tag = m.group(0)
    assert 'tabindex="-1"' in tag, "page-root must be focusable for soft-nav focus mgmt"
    assert 'role="main"' in tag


def test_skip_link_targets_page_root(offline_client):
    # The skip link is the keyboard entry into content; it targets #page-root,
    # which only actually moves focus now that page-root is focusable.
    html = _html(offline_client, GRAPHS)
    assert 'class="skip-link" href="#page-root"' in html
