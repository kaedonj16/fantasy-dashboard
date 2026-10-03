"""Render tests for the Season Hub "Usage risers" card (_render_usage_movers).

Locks in the restyled player-row markup (position/team subtext, season vs
recent mini bars, delta chip) and the behavioral contracts around it: max 3
movers, delta >= 1.5 with >= 3 weeks played, "" when nobody qualifies or
there is no viewer, and "" (never a raise) when the trends lookup fails.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

from app import _render_usage_movers  # noqa: E402

# Patch the trends lookup on the module object itself: in a full-suite run
# the namespace package can sit in sys.modules with the submodule never
# bound as an attribute, which makes dotted-string monkeypatch targets
# ("data_building.weekly_metrics.get_usage_trends") fail to resolve.
import data_building.weekly_metrics as _weekly_metrics  # noqa: E402


def _ctx(players_index=None, pids=("100", "101", "102", "103", "104")):
    return {
        "rosters": [{"roster_id": 1, "players": list(pids)}],
        "current_season": 2026,
        "players_index": players_index or {},
    }


def _trends():
    return {
        # Qualifies: snap % riser (position via "position", no team).
        "100": {"stat": "snap_pct", "season_avg": 62.5, "recent_avg": 78.0,
                "delta": 15.5, "weeks_played": 6},
        # Qualifies: targets riser with full pos/team meta.
        "101": {"stat": "targets", "season_avg": 4.2, "recent_avg": 7.8,
                "delta": 3.6, "weeks_played": 6},
        # Below the delta threshold: excluded.
        "102": {"stat": "touches", "season_avg": 12.0, "recent_avg": 13.2,
                "delta": 1.2, "weeks_played": 6},
        # Too few weeks: excluded despite a big delta.
        "103": {"stat": "targets", "season_avg": 3.0, "recent_avg": 8.0,
                "delta": 5.0, "weeks_played": 2},
        # Qualifies but ranks 4th: cut by the top-3 cap.
        "104": {"stat": "touches", "season_avg": 10.0, "recent_avg": 12.0,
                "delta": 2.0, "weeks_played": 5},
        # Qualifies and ranks 3rd.
        "105": {"stat": "targets", "season_avg": 5.0, "recent_avg": 7.5,
                "delta": 2.5, "weeks_played": 4},
    }


def _index():
    return {
        "100": {"full_name": "Snap Riser", "position": "RB"},
        "101": {"full_name": "Target Riser", "pos": "WR", "team": "CIN"},
        "102": {"full_name": "Below Threshold", "pos": "RB", "team": "DAL"},
        "103": {"full_name": "Too Few Weeks", "pos": "TE", "team": "KC"},
        "104": {"full_name": "Fourth Riser", "pos": "WR", "team": "MIA"},
        "105": {"full_name": "Third Riser", "pos": "WR", "team": "BUF"},
    }


def _render(monkeypatch, ctx, viewer="1", trends=None):
    monkeypatch.setattr(
        _weekly_metrics,
        "get_usage_trends",
        lambda season: _trends() if trends is None else trends,
    )
    return _render_usage_movers(ctx, viewer)


def test_card_markup_rows_bars_and_chip(monkeypatch):
    ctx = _ctx(players_index=_index(), pids=("100", "101", "102", "103", "104", "105"))
    out = _render(monkeypatch, ctx)
    assert '<section class="os-card usage-movers-card">' in out
    assert "Usage risers" in out
    assert "last 3 weeks vs season average" in out
    # Top 3 only, sorted by delta desc.
    assert out.count('<li class="usage-mover player-clickable"') == 3
    assert "Snap Riser" in out and "Target Riser" in out and "Third Riser" in out
    assert "Fourth Riser" not in out
    assert "Below Threshold" not in out
    assert "Too Few Weeks" not in out
    assert out.index("Snap Riser") < out.index("Target Riser") < out.index("Third Riser")
    # Position/team subtext: "pos" and "position" keys both honored.
    assert '<span class="um-sub">WR · CIN</span>' in out
    assert '<span class="um-sub">RB</span>' in out
    # Legible trend readout with capitalized stat labels.
    assert "Targets 4.2 &rarr; 7.8" in out
    assert "Snap % 62.5 &rarr; 78" in out
    # Mini bars: snap % scales 0-100 absolute; volume stats scale to row max.
    assert 'um-bar um-bar-season" style="width:62.5%"' in out
    assert 'um-bar um-bar-recent" style="width:78.0%"' in out
    assert 'um-bar um-bar-season" style="width:53.8%"' in out  # 4.2 / 7.8
    assert 'um-bar um-bar-recent" style="width:100.0%"' in out
    # Delta is a chip, not bare text.
    assert '<span class="um-delta up">&#9650;3.6</span>' in out
    assert '<span class="um-delta up">&#9650;15.5</span>' in out


def test_rows_carry_real_player_ids_for_the_modal(monkeypatch):
    ctx = _ctx(players_index=_index(), pids=("100", "101", "105"))
    out = _render(monkeypatch, ctx)
    # Each row carries the global player-clickable wiring with the real
    # Sleeper id and display name, in delta order.
    assert (
        '<li class="usage-mover player-clickable" data-player-id="100" '
        'data-player-name="Snap Riser">'
    ) in out
    assert (
        '<li class="usage-mover player-clickable" data-player-id="101" '
        'data-player-name="Target Riser">'
    ) in out
    assert (
        '<li class="usage-mover player-clickable" data-player-id="105" '
        'data-player-name="Third Riser">'
    ) in out
    assert out.index('data-player-id="100"') < out.index('data-player-id="101"')
    assert out.index('data-player-id="101"') < out.index('data-player-id="105"')
    # Excluded players get no row and no clickable attrs.
    assert 'data-player-id="102"' not in out
    assert 'data-player-id="104"' not in out


def test_missing_meta_falls_back_to_name_only(monkeypatch):
    ctx = _ctx(players_index={}, pids=("101",))
    out = _render(monkeypatch, ctx)
    assert "Player 101" in out
    # The fallback name still rides the clickable row with the real id.
    assert 'data-player-id="101"' in out
    assert 'data-player-name="Player 101"' in out
    assert 'class="um-sub"' not in out
    assert 'class="um-bar um-bar-recent"' in out
    assert '<span class="um-delta up">' in out


def test_no_viewer_renders_nothing(monkeypatch):
    ctx = _ctx(players_index=_index())
    assert _render(monkeypatch, ctx, viewer=None) == ""
    assert _render(monkeypatch, ctx, viewer="") == ""


def test_unknown_viewer_roster_renders_nothing(monkeypatch):
    ctx = _ctx(players_index=_index())
    assert _render(monkeypatch, ctx, viewer="99") == ""


def test_nobody_qualifies_renders_nothing(monkeypatch):
    ctx = _ctx(players_index=_index(), pids=("102", "103"))
    assert _render(monkeypatch, ctx) == ""


def test_trends_failure_is_swallowed(monkeypatch):
    def _boom(season):
        raise RuntimeError("metrics table unavailable")

    monkeypatch.setattr(_weekly_metrics, "get_usage_trends", _boom)
    assert _render_usage_movers(_ctx(players_index=_index()), "1") == ""
