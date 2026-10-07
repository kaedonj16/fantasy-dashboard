"""Player Rankings PPG / total points must be the CURRENT season's numbers.

Regression for the 2026-09-30 report: /players showed 2025 PPG and total
points. The league-players payload enriched PPG from the newest prebuilt
usage_rows_<season>.json, but that file only exists for finished seasons
(2026's was never built), so the "current season, else last season" lookup
silently served 2025's full-year numbers all through the 2026 season.

The fix sources in-season PPG from the same completed-week Sleeper stats the
player modal uses (_load_season_weekly_points, full PPR). There is no
prior-season fallback at all: when current-season numbers are unavailable
the cells stay blank rather than showing a plausible-looking old number.
The rankings grid also renders Value and PPG as two always-visible columns.
"""
import pathlib

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app
from utils.season_qualification import QualificationPolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fake_policy(weeks, games_min):
    def _policy(season):
        return QualificationPolicy(int(season), tuple(weeks), games_min)
    return _policy


def _patch_season(monkeypatch, weekly, weeks=(1, 2), games_min=2):
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        _fake_policy(weeks, games_min),
    )
    monkeypatch.setattr(app, "_load_season_weekly_points",
                        lambda season, scoring, weeks=None: weekly)
    monkeypatch.setattr(app, "load_players_index", dict)
    # Per-player completed weeks (PR #2288): return the full mocked weeks
    # for every requested player so PPG is computed from the mocked data.
    monkeypatch.setattr(
        "utils.season_qualification.bulk_player_completed_weeks",
        lambda pids, season: {str(_pid): tuple(weeks) for _pid in pids},
    )


def test_current_season_ppg_comes_from_completed_weeks(monkeypatch):
    # The loader only appends actual appearances, so a bye/absence simply
    # isn't in the list. Any appearance qualifies (no games gate), matching
    # the player modal: pid 300's single game still yields a PPG.
    weekly = {"100": [30.0, 20.0], "200": [12.0, 8.0], "300": [40.0]}
    _patch_season(monkeypatch, weekly)
    positions = {"100": "RB", "200": "RB", "300": "WR"}

    result = app._current_season_ppg_map(2026, positions)

    assert result["100"]["ppg"] == 25.0
    assert result["100"]["total_pts"] == 50.0
    assert result["100"]["ppg_games"] == 2
    assert result["100"]["ppg_season"] == 2026
    assert result["100"]["ppg_rank"] == 1
    assert result["100"]["total_pts_rank"] == 1
    assert result["200"]["ppg"] == 10.0
    assert result["200"]["ppg_rank"] == 2
    assert result["300"]["ppg"] == 40.0
    assert result["300"]["ppg_games"] == 1
    assert result["300"]["ppg_rank"] == 1


def test_current_season_ppg_uses_real_totals_not_ppg_times_games(monkeypatch):
    # Totals come from the actual weekly points, so a player with fewer
    # appearances can trail on total while leading on PPG.
    weekly = {"100": [30.04], "200": [20.0, 20.0]}
    _patch_season(monkeypatch, weekly, weeks=(1, 2), games_min=1)
    positions = {"100": "WR", "200": "WR"}

    result = app._current_season_ppg_map(2026, positions)

    assert result["100"]["ppg"] == 30.0
    assert result["100"]["total_pts"] == 30.0
    assert result["100"]["ppg_rank"] == 1
    assert result["100"]["total_pts_rank"] == 2
    assert result["200"]["total_pts_rank"] == 1


def test_no_completed_rounds_returns_empty(monkeypatch):
    # Preseason: no current-season numbers exist, and nothing substitutes
    # for them (no prior-season fallback), so the map is empty.
    _patch_season(monkeypatch, {"100": [30.0]}, weeks=(), games_min=1)
    assert app._current_season_ppg_map(2026, {"100": "RB"}) == {}


def test_missing_weekly_data_midseason_returns_empty(monkeypatch):
    # Rounds ARE complete but the weekly files came back empty: still no
    # substitute. This is the state that used to leak 2025's numbers.
    _patch_season(monkeypatch, {}, weeks=(1, 2), games_min=1)
    assert app._current_season_ppg_map(2026, {"100": "RB"}) == {}


def test_weekly_loader_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        _fake_policy((1, 2), 1),
    )

    def _boom(season, scoring, weeks=None):
        raise RuntimeError("weekly files unavailable")

    monkeypatch.setattr(app, "_load_season_weekly_points", _boom)
    monkeypatch.setattr(app, "load_players_index", dict)
    assert app._current_season_ppg_map(2026, {"100": "RB"}) == {}


def test_current_season_ppg_is_scored_full_ppr(monkeypatch):
    seen = {}

    def _loader(season, scoring, weeks=None):
        seen["season"] = season
        seen["scoring"] = scoring
        return {"100": [10.0]}

    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        _fake_policy((1,), 1),
    )
    monkeypatch.setattr(app, "_load_season_weekly_points", _loader)
    monkeypatch.setattr(app, "load_players_index", dict)

    app._current_season_ppg_map(2026, {"100": "RB"})

    assert seen == {"season": 2026, "scoring": {"rec": 1.0}}


def test_ppg_map_with_ranks_shares_better_rank_on_ties():
    entries = {
        "1": {"ppg": 20.0, "total_pts": 80.0, "games": 4, "season": 2026, "position": "QB"},
        "2": {"ppg": 20.0, "total_pts": 60.0, "games": 3, "season": 2026, "position": "QB"},
        "3": {"ppg": 15.0, "total_pts": 90.0, "games": 6, "season": 2026, "position": "QB"},
    }
    result = app._ppg_map_with_ranks(entries)
    assert result["1"]["ppg_rank"] == 1
    assert result["2"]["ppg_rank"] == 1
    assert result["3"]["ppg_rank"] == 3
    assert result["3"]["total_pts_rank"] == 1
    # Exactly the payload fields, nothing extra (no internal "position").
    assert set(result["1"]) == {
        "ppg", "total_pts", "ppg_games", "ppg_season", "ppg_rank", "total_pts_rank",
    }


def test_payload_builder_uses_current_season_with_no_usage_fallback():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.index("def _build_league_players_payload_uncached")
    block = src[start:start + 60000]
    block = block[:block.index("\ndef ", 100)]
    # PPG enrichment comes from the current-season weekly build...
    assert "_current_season_ppg_map(_season_lp, _pos_lp)" in block
    # ...and never from a usage file. The old "newest usage file wins" loop
    # (the 2025 leak) is gone, and no other usage fallback remains: a
    # missing or failed current-season build leaves the cells blank.
    assert "_load_usage_rows_cached" not in block
    assert "_season_lp - 1" not in block


def test_rankings_grid_shows_value_and_ppg_side_by_side():
    page = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
    js = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
    src = (ROOT / "app.py").read_text(encoding="utf-8")

    # Header carries both columns, PPG immediately before Value.
    assert 'id="prPpgHeader"' in page
    assert page.index('id="prPpgHeader"') < page.index('id="prSortHeader"')
    # Desktop grid has a track for the PPG column.
    assert "88px 42px 1fr 52px 46px 46px 52px 60px" in page
    # Hydrated rows render the PPG cell before the Value cell.
    assert js.index('class="pr-ppg"') < js.index('class="pr-value"')
    # Sorting by PPG or Total Points must not swap Value out of the
    # trailing column; the header stays on Value for both sorts too.
    assert "const valueMeta = (sortBy === 'ppg' || sortBy === 'total_pts') ? PR_SORT_META.rank : sortMeta;" in js
    assert "const headerMeta0 = (sortBy === 'ppg' || sortBy === 'total_pts') ? PR_SORT_META.rank : sortMeta0;" in js
    # SSR first paint matches: rows include a PPG cell too.
    assert 'pr-ppg' in src
