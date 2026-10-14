"""Player-modal color pass + mobile width/headshot fixes (Kaedon, 2026-10-14).

She approved the tasteful color pass from the legit-renderer mock (color
only where it carries meaning, ADP left stock) and reported three mobile
problems from her phone: the Playoff/Role/ADP cards rendered at content
width instead of full width, and the headshot floated in the header
instead of sitting flush above the tab bar.

Root cause of the widths: .pm-ov-cols is a grid with align-items: start
on desktop; the <=768px block turns it into a flex column but never reset
align-items, so every card shrank to its content width. The fix stretches
the column's items. The headshot fix stretches the mobile header so the
headshot fills its height with its bottom edge on the tab bar.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _mobile_ov_block() -> str:
    start = CSS.find("Phone: one column, mock order")
    assert start > 0
    return CSS[start:start + 1600]


def test_mobile_overview_column_stretches_cards():
    block = _mobile_ov_block()
    assert "flex-direction: column; align-items: stretch" in block


def test_playoff_ranks_carry_tier_colors():
    # Aggregate gets a tier class; week ranks are wrapped in a tier span.
    assert "pm-playoff-rank-${_rkTier(_ps.rank)}" in JS
    assert 'class="pm-rk pm-rk-${_rkTier(w.opp_rank)}"' in JS
    for tier, color in (("easy", "#059669"), ("mid", "#b45309"), ("hard", "#dc2626")):
        assert f".pm-rk-{tier} {{ color: {color}; }}" in CSS
        assert f".pm-playoff-rank-{tier} {{ color: {color}; }}" in CSS


def test_role_card_share_bars():
    assert "pm-ov-sharebar" in JS
    assert "n >= 60 ? '#10b981' : n >= 25 ? '#3b82f6' : '#f59e0b'" in JS
    assert ".pm-ov-sharebar {" in CSS
    # A bar follows each of the two share rows.
    assert JS.count("_roleRows.push(_shareBar(") == 2


def test_recent_form_best_game_is_green():
    assert "pm-recent-bar-best" in JS and "pm-recent-val-best" in JS
    assert ".pm-recent-bar-best { background: #10b981; }" in CSS
    assert ".pm-recent-val-best { color: #059669; }" in CSS
    # Only the first max game is flagged.
    assert "_bestUsed" in JS


def test_team_share_bar_palette():
    # Focus player takes his position color; teammates use the blue family.
    assert "{ WR: '#f97316', TE: '#8b5cf6' }" in JS
    assert "'#3b82f6', '#7aa5e8', '#a8c3ef'" in JS
    # The old neutral gray ramp is gone from the share bar.
    share_fn = JS[JS.find("function _pmTeamShareBar"):]
    share_fn = share_fn[:share_fn.find("\n}\n")]
    assert "color-mix(in srgb, var(--text-subtle)" not in share_fn


def test_adp_left_stock():
    # Her call: no added color on ADP. The color pass adds no pm-adp rules
    # beyond what already existed (no gradient line, no tinted chips).
    assert "linear-gradient(90deg,#10b981" not in CSS
    assert '.pm-adp-card[data-adp-fmt="dynasty"] .pm-adp-card-h' not in CSS


def test_mobile_headshot_flush_with_tab_bar():
    # The mobile header block: no bottom padding, stretched items, and the
    # headshot fills the header height.
    start = CSS.find("/* Headshot: fills the full header height on phone")
    assert start > 0
    block = CSS[start - 2600:start + 900]
    header_rule = re.search(r"\.player-modal-header \{(.*?)\}", block, re.S)
    assert header_rule and "padding: 10px 12px 0;" in header_rule.group(1)
    assert "align-items: stretch;" in header_rule.group(1)
    shot_rule = re.search(r"\.player-modal-headshot \{(.*?)\}", CSS[start:start + 900], re.S)
    assert shot_rule and "height: 100%;" in shot_rule.group(1)
    assert "width: 88px;" in shot_rule.group(1)
