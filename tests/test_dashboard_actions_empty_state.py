"""The Season Hub Actions tab is the default-active panel, so it must never
render blank. Every action card (lineup, roster moves, trade window, waiver)
returns "" with no viewer roster or when the roster is clean this week. These
guards lock in the two fallbacks: a "link your team" prompt when unidentified,
and an all-clear card when identified but there is nothing to do.
"""
from __future__ import annotations

import re
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "dashboard_services" / "pages" / "dashboard_page.py"


def _src() -> str:
    return _PAGE.read_text(encoding="utf-8")


def test_action_queue_has_nonblank_fallback():
    src = _src()
    # The queue body is driven by a computed inner block, not four raw cards.
    assert "_action_inner" in src
    assert 'id="os-jump-actions">{_action_inner}' in src
    # Guard is "any card has content", covering the clean-roster case too.
    assert "any((c or " in src


def test_unlinked_viewer_gets_link_prompt():
    src = _src()
    assert "elif not viewer_roster_id:" in src
    assert "Link your team to see this week's actions" in src
    # CTA opens the existing link-my-team modal for this league.
    assert "linkMyTeam(" in src
    assert "openLinkModal()" in src


def test_clean_roster_gets_all_clear_card():
    src = _src()
    assert "You're all set for Week" in src
    # All-clear card still points to Start/Sit and Waivers.
    assert "?tab=startsit" in src
    assert "os-actions-empty" in src


def test_changelog_announces_actions_empty_state_fix():
    from dashboard_services.changelog import CHANGELOG

    entry = next(
        e for e in CHANGELOG
        if "actions tab" in e.get("text", "").lower()
        and "blank" in e.get("text", "").lower()
    )
    assert entry["tag"] == "fix"
    assert "—" not in entry["text"]
    assert "–" not in entry["text"]
