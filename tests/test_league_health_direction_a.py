"""League Health Direction A structure pins.

Direction A (approved mock): one merged health card (96px ring + verdict +
the four driver bars), a compact Multi-Season card, the Team Overview table
with no parity strip, one-line Trade Fairness rows, and the slim invite row
last. Presentation only: every computed value on the page is unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "dashboard_services" / "pages" / "commissioner_page.py"
SRC = PAGE.read_text(encoding="utf-8")


def _load_page_module():
    spec = importlib.util.spec_from_file_location("commissioner_page_lh_a", PAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_page_order_is_health_history_teams_trades_invite():
    ret = SRC.split("return f'<div class=\"lh-wrap\">", 1)[1]
    order = [ret.index(f"{{{name}}}") for name in (
        "health_html", "history_panel", "roster_table", "trade_card", "invite_card")]
    assert order == sorted(order)


def test_health_card_merges_ring_verdict_and_all_four_drivers():
    health = SRC.split("health_html = f\"\"\"", 1)[1].split("\"\"\"", 1)[0]
    assert "lh-ring" in health
    assert "lh-verdict" in health
    assert "{_bars_html}" in health
    # The separate drivers card is gone from the whole page.
    assert "What's driving the score" not in SRC
    for label in ("Inactive Teams", "Trades This Season",
                  "Lopsided Trades", "Total Moves"):
        assert f'_lh_bar("{label}"' in SRC


def test_ring_is_96px():
    assert ".lh-ring {{ position:relative; width:96px; height:96px;" in SRC
    assert '<svg width="96" height="96" viewBox="0 0 96 96"' in SRC
    assert "_ring_r = 40" in SRC


def test_parity_strip_is_gone():
    assert "lh-parity" not in SRC
    assert "lh-pdot" not in SRC
    assert "parity_html" not in SRC
    assert "_p_x" not in SRC
    # The Team Overview table itself (and its sticky first column) stays.
    assert 'class="comm-team-table"' in SRC
    assert ".comm-team-table thead th:first-child" in SRC


def test_trade_fairness_rows_are_compact_one_liners():
    assert 'class="lh-trade-row"' in SRC
    assert "lh-tr-diff" in SRC and "lh-tr-label" in SRC
    assert "SIDE A" not in SRC and "SIDE B" not in SRC
    # Same verdict thresholds as before: lopsided flag, else diff > 40 uneven.
    assert '"UNEVEN" if t["diff"] > 40 else "FAIR"' in SRC


def test_invite_row_is_slim_and_copies_via_data_attribute():
    assert "Invite your league to PRO:</b> share this link so teammates can claim League PRO on this league." in SRC
    assert 'data-invite-path="{_inv}"' in SRC
    assert "getAttribute('data-invite-path')" in SRC


def _layer(seasons, owners=None):
    return {"seasons": seasons, "owners": owners or {}}


def test_history_renders_compact_lines_with_trend_and_current_marker():
    mod = _load_page_module()
    out = mod._render_commissioner_history(
        _layer([
            {"season": 2023, "moves": 100, "trades": 10, "inactive_owners": 0, "n_teams": 10},
            {"season": 2024, "moves": 130, "trades": 12, "inactive_owners": 0, "n_teams": 10},
        ]),
        current_season=2026, current_moves=50, current_trades=5,
        current_inactive_uids=set(), current_week=5, playoff_start=14,
    )
    assert 'class="card msh-card"' in out
    assert "msh-line" in out and "msh-seq" in out
    assert "msh-chip" not in out and "msh-stat" not in out
    assert "Rising" in out  # 100 -> 130 completed-season moves trend
    assert "in progress" in out  # 2026 is the in-progress season
    assert "No chronic inactivity. Owners stay engaged across seasons." in out
    assert "\u2014" not in out  # no em dashes in UI copy


def test_history_chronic_owner_gets_a_flag_line():
    mod = _load_page_module()
    out = mod._render_commissioner_history(
        _layer(
            [{"season": 2024, "moves": 90, "trades": 8, "inactive_owners": 1, "n_teams": 10}],
            owners={"u1": {"name": "Ghost Team", "seasons_present": 2, "inactive_seasons": 2}},
        ),
        current_season=2026, current_moves=40, current_trades=4,
        current_inactive_uids=set(), current_week=5, playoff_start=14,
    )
    assert "Ghost Team" in out
    assert "inactive 2 of last 3" in out
    assert "msh-chronic-flag" in out
