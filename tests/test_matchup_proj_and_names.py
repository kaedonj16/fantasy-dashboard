"""Matchup Preview projections + player-name clipping contracts."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("flask")
pytest.importorskip("requests")

from dashboard_services import matchups as mmod


def test_week_proj_map_accepts_string_week_keys_and_flat_maps():
    nested = {1: {"projections": {"4984": 20.1}}}
    assert mmod._week_proj_map_from_bundles(nested, 1)["4984"] == 20.1
    assert mmod._week_proj_map_from_bundles(nested, "1")["4984"] == 20.1

    flat = {"1": {"4984": 18.5, "_available": True}}
    out = mmod._week_proj_map_from_bundles(flat, 1)
    assert out["4984"] == 18.5
    assert "_available" not in out


def test_proj_value_falls_back_to_raw_week_map():
    raw = {
        "4984": {
            "raw_stats": {"pts_ppr": 22.0, "pts_std": 22.0, "pts_half_ppr": 22.0},
            "ppr": 22.0,
            "std": 22.0,
            "half_ppr": 22.0,
        }
    }
    # Empty bundle → Scout-style raw fallback.
    pts = mmod._proj_value_for_pid(
        {}, "4984",
        raw_week_map=raw,
        scoring_settings={"rec": 1.0, "pass_td": 4.0},
        pos="QB",
    )
    assert pts == pytest.approx(22.0)


def test_render_matchup_slide_uses_raw_fallback_when_bundle_empty(monkeypatch):
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "_allow_live_game_indicators", lambda *_a, **_k: False)

    raw = {
        "4984": {"ppr": 21.5, "std": 21.5, "half_ppr": 21.5,
                 "raw_stats": {"pts_ppr": 21.5, "pts_std": 21.5, "pts_half_ppr": 21.5}},
        "11564": {"ppr": 19.0, "std": 19.0, "half_ppr": 19.0,
                  "raw_stats": {"pts_ppr": 19.0, "pts_std": 19.0, "pts_half_ppr": 19.0}},
    }
    monkeypatch.setattr(
        "utils.utils.load_week_projection", lambda *_a, **_k: raw,
    )

    matchup = {
        "left": {
            "name": "Team A", "roster_id": "1", "record": "0-0", "username": "a",
            "avatar": "", "pts_total": None,
            "starters": [{"pid": "11564", "name": "Drake Maye", "pos": "QB", "nfl": "NE", "pts": None}],
        },
        "right": {
            "name": "Team B", "roster_id": "2", "record": "0-0", "username": "b",
            "avatar": "", "pts_total": None,
            "starters": [{"pid": "4984", "name": "Josh Allen", "pos": "QB", "nfl": "BUF", "pts": None}],
        },
    }
    html = mmod.render_matchup_slide(
        "2026", matchup, w=1, proj_week=0,
        status_by_pid={},
        projections={},  # empty bundle — previously painted 0.0 everywhere
        players={"11564": {"name": "Drake Maye"}, "4984": {"name": "Josh Allen"}},
        teams={},
        team_game_lookup={},
        scoring_settings={"rec": 1.0, "pass_td": 4.0},
    )
    assert "21.5" in html
    assert "19.0" in html
    assert "p-name-line" in html
    # Name ellipsis lives on .pname via CSS class, not a shared parent nowrap.
    assert "white-space:nowrap;text-overflow:ellipsis;\">" not in html.replace(" ", "")


def test_render_matchup_keeps_team_abbr_outside_name_ellipsis():
    matchup = {
        "left": {
            "name": "Team A", "roster_id": "1", "record": "0-0", "username": "a",
            "avatar": "", "pts_total": None,
            "starters": [{
                "pid": "6813", "name": "Jonathan Taylor", "pos": "RB", "nfl": "IND", "pts": None,
            }],
        },
        "right": {
            "name": "Team B", "roster_id": "2", "record": "0-0", "username": "b",
            "avatar": "", "pts_total": None,
            "starters": [],
        },
    }
    with mock.patch.object(mmod, "load_teams_index", return_value={}), \
         mock.patch.object(mmod, "build_offense_rankings", return_value={}), \
         mock.patch.object(mmod, "load_week_stats", return_value={}), \
         mock.patch.object(mmod, "load_week_schedule", return_value={}), \
         mock.patch.object(mmod, "build_team_schedule_lookup", return_value={}), \
         mock.patch.object(mmod, "_allow_live_game_indicators", return_value=False), \
         mock.patch("utils.utils.load_week_projection", return_value={}):
        html = mmod.render_matchup_slide(
            "2026", matchup, w=1, proj_week=0,
            status_by_pid={},
            projections={1: {"projections": {"6813": 14.2}}},
            players={"6813": {"name": "Jonathan Taylor"}},
            teams={},
            team_game_lookup={},
        )
    assert "p-name-line" in html
    assert "p-team" in html
    assert "IND" in html
    assert "Jonathan Taylor" in html
    assert "14.2" in html
