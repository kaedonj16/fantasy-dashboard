"""Header CSS regressions for League Health + Rookie Prospects.

Both pages rendered a stray 300px sky bar under the title: dashboard.css's
global ``h1::after`` (a centered accent bar for standalone hero titles)
leaks into any h1 that does not opt out. Both pages' action buttons also
use ``.btn`` classes that dashboard.css never defines, so they fell back
to UA button padding (1px 6px, ~22px tall) next to full-height inputs.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_rookies_title_suppresses_global_h1_bar_and_export_button_is_styled():
    from dashboard_services.pages.rookies_page import build_prospects_body

    body = build_prospects_body(False)
    # Title opt-out from the global centered h1::after bar.
    assert "#rookiesTitle::after { display: none; }" in body
    # Export (.btn btn-ghost btn-sm) gets real geometry scoped to the card:
    # dashboard.css defines none of those classes.
    assert "#prospectsCard .btn-ghost" in body
    assert "min-height: 32px" in body
    assert "background: transparent" in body


def test_league_health_title_left_aligned_without_global_h1_bar():
    src = _read("dashboard_services/pages/commissioner_page.py")
    # The subtitle and content are left-aligned; the h1 must be too (the
    # global h1 rule centers it). The title now uses the shared .page-title
    # class, which carries the left-align + bar opt-out in dashboard.css.
    assert '<h1 class="page-title" style="margin-bottom:8px;">League Health</h1>' in src
    css = _read("static/dashboard.css")
    assert ".page-title::after { content: none; display: none; }" in css
    # Opt-out from the global centered h1::after sky bar, in the page's own
    # always-rendered style block.
    assert ".lh-wrap > h1::after {{ display:none; }}" in src


def test_league_health_invite_is_a_slim_row_and_copy_button_is_styled():
    src = _read("dashboard_services/pages/commissioner_page.py")
    # Direction A: the invite is a slim flex row at the bottom of the page,
    # not a headed card with a visible link input.
    assert 'id="lhLeagueInvite"' in src
    assert "#lhLeagueInvite {{ display:flex; align-items:center;" in src
    assert "lhInvitePath" not in src
    assert "data-invite-path=" in src
    # Copy button (.btn, undefined in dashboard.css) gets real geometry
    # instead of the 22px UA fallback.
    assert "#lhInviteCopy {{ padding:8px 14px; font-size:13px;" in src
