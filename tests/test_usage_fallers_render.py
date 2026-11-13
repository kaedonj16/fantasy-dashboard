"""Render tests for the mobile dashboard "Usage fallers" card.

The Fallers card is the Usage risers card's mirror: same trends data
(data_building.weekly_metrics.get_usage_trends), same row grammar, but
delta <= -1.5 with >= 3 weeks played, biggest drop first, top 3. It
renders as its own card (class usage-fallers-card) so the mobile stack
can place it directly after Usage risers; the desktop layout hides it
via CSS (see test_dashboard_mobile_reorder).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

from app import _render_usage_fallers, _render_usage_movers  # noqa: E402

import data_building.weekly_metrics as _weekly_metrics  # noqa: E402


def _ctx(players_index=None, pids=("100", "101", "102", "103", "104")):
    return {
        "rosters": [{"roster_id": 1, "players": list(pids)}],
        "current_season": 2026,
        "players_index": players_index or {},
    }


def _trends():
    return {
        # Qualifies: snap % faller (position via "position", no team).
        "100": {"stat": "snap_pct", "season_avg": 78.0, "recent_avg": 62.5,
                "delta": -15.5, "weeks_played": 6},
        # Qualifies: targets faller with full pos/team meta.
        "101": {"stat": "targets", "season_avg": 7.8, "recent_avg": 4.2,
                "delta": -3.6, "weeks_played": 6},
        # Above the (negative) delta threshold: excluded.
        "102": {"stat": "touches", "season_avg": 13.2, "recent_avg": 12.0,
                "delta": -1.2, "weeks_played": 6},
        # Too few weeks: excluded despite a big drop.
        "103": {"stat": "targets", "season_avg": 8.0, "recent_avg": 3.0,
                "delta": -5.0, "weeks_played": 2},
        # Qualifies but ranks 4th: cut by the top-3 cap.
        "104": {"stat": "touches", "season_avg": 12.0, "recent_avg": 10.0,
                "delta": -2.0, "weeks_played": 5},
        # Qualifies and ranks 3rd.
        "105": {"stat": "targets", "season_avg": 7.5, "recent_avg": 5.0,
                "delta": -2.5, "weeks_played": 4},
    }


def _index():
    return {
        "100": {"full_name": "Snap Faller", "position": "RB"},
        "101": {"full_name": "Target Faller", "pos": "WR", "team": "CIN"},
        "102": {"full_name": "Above Threshold", "pos": "RB", "team": "DAL"},
        "103": {"full_name": "Too Few Weeks", "pos": "TE", "team": "KC"},
        "104": {"full_name": "Fourth Faller", "pos": "WR", "team": "MIA"},
        "105": {"full_name": "Third Faller", "pos": "WR", "team": "BUF"},
    }


def _render(monkeypatch, ctx, viewer="1"):
    monkeypatch.setattr(
        _weekly_metrics, "get_usage_trends", lambda season: _trends()
    )
    return _render_usage_fallers(ctx, viewer)


def test_fallers_card_markup_rows_and_chip(monkeypatch):
    ctx = _ctx(players_index=_index(),
               pids=("100", "101", "102", "103", "104", "105"))
    out = _render(monkeypatch, ctx)
    assert '<section class="os-card usage-movers-card usage-fallers-card">' in out
    assert "Usage fallers" in out
    assert "trending down: last 3 weeks vs season average" in out
    # Top 3 only, biggest drop first.
    assert out.count('<li class="usage-mover player-clickable"') == 3
    assert "Snap Faller" in out and "Target Faller" in out and "Third Faller" in out
    assert "Fourth Faller" not in out
    assert "Above Threshold" not in out
    assert "Too Few Weeks" not in out
    assert out.index("Snap Faller") < out.index("Target Faller") < out.index("Third Faller")
    # Same row grammar as risers: subtext, readout, bars.
    assert '<span class="um-sub">WR · CIN</span>' in out
    assert "Targets 7.8 &rarr; 4.2" in out
    assert 'um-bar um-bar-recent" style="width:53.8%"' in out  # 4.2 / 7.8
    # Down chip shows the drop magnitude with a down arrow.
    assert '<span class="um-delta down">&#9660;3.6</span>' in out
    assert '<span class="um-delta down">&#9660;15.5</span>' in out
    assert 'um-delta up' not in out


def test_fallers_and_risers_are_disjoint(monkeypatch):
    ctx = _ctx(players_index=_index(),
               pids=("100", "101", "102", "103", "104", "105"))
    monkeypatch.setattr(
        _weekly_metrics, "get_usage_trends", lambda season: _trends()
    )
    # A fallers-only trend set renders nothing on the risers card.
    assert _render_usage_movers(ctx, "1") == ""


def test_no_viewer_renders_nothing(monkeypatch):
    ctx = _ctx(players_index=_index())
    assert _render(monkeypatch, ctx, viewer=None) == ""
    assert _render(monkeypatch, ctx, viewer="") == ""


def test_nobody_qualifies_renders_nothing(monkeypatch):
    ctx = _ctx(players_index=_index(), pids=("102", "103"))
    assert _render(monkeypatch, ctx) == ""


def test_trends_failure_is_swallowed(monkeypatch):
    def _boom(season):
        raise RuntimeError("metrics table unavailable")

    monkeypatch.setattr(_weekly_metrics, "get_usage_trends", _boom)
    assert _render_usage_fallers(_ctx(players_index=_index()), "1") == ""
