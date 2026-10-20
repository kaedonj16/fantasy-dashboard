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
    # global h1 rule centers it).
    assert 'margin:0 0 8px;text-align:left;">League Health</h1>' in src
    # Opt-out from the global centered h1::after sky bar, in the page's own
    # always-rendered style block.
    assert ".lh-wrap > h1::after {{ display:none; }}" in src


def test_league_health_invite_card_header_aligns_and_copy_button_is_styled():
    src = _read("dashboard_services/pages/commissioner_page.py")
    # Card header takes the body's 18px inset so header and body copy line up.
    assert "#lhLeagueInvite .card-header {{ padding:16px 18px 0; min-height:0; }}" in src
    # Copy button (.btn, undefined in dashboard.css) matches the invite
    # input's height/type instead of the 22px UA fallback.
    assert "#lhInviteCopy {{ padding:8px 14px; font-size:13px;" in src
