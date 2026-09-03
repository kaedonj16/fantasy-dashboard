"""Week 1 preview must not paint last season's box scores before kickoff.

The 2026 draft-week hub showed Jayden Daniels as 233 yds / 1 td / 11 car /
68 yds — his 2025 Week 1 line — because:

1. Footballguys still publishes last year's Wk 1 under the new season, and
2. Sleeper stores Washington as WAS while Tank01 schedules use WSH, so
   build_status_by_pid marked Commanders as FINAL and the old
   "hide stats if not_started" gate never fired.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

# utils.utils pulls requests + bs4 + dashboard_services.api (Flask). Gate the
# file so the slim lint job skips instead of erroring at collection.
pytest.importorskip("requests")
pytest.importorskip("bs4")
pytest.importorskip("flask")

from utils.utils import (
    STATUS_NOT_STARTED,
    build_games_by_team,
    build_status_by_pid,
    game_has_started,
    lookup_team_map,
    team_abbr_keys,
)


def _scheduled_wsh_game(**extra):
    game = {
        "home": "NYG",
        "away": "WSH",
        "gameDate": "20260913",
        "gameTime": "1:00p",
        "gameStatus": "Scheduled",
        "gameStatusCode": "0",
        "gameTime_epoch": "1789318800.0",
    }
    game.update(extra)
    return game


def test_team_abbr_keys_was_wsh_and_jac():
    assert team_abbr_keys("WAS") == ("WAS", "WSH")
    assert team_abbr_keys("wsh") == ("WSH", "WAS")
    assert team_abbr_keys("JAC") == ("JAC", "JAX")
    assert team_abbr_keys("LAC") == ("LAC",)
    assert team_abbr_keys("") == ()


def test_lookup_team_map_resolves_alias():
    assert lookup_team_map({"WSH": {"ok": 1}}, "WAS") == {"ok": 1}
    assert lookup_team_map({"WAS": {"ok": 1}}, "WSH") == {"ok": 1}
    assert lookup_team_map({"LAC": {"ok": 1}}, "LAC") == {"ok": 1}
    assert lookup_team_map({"LAC": {"ok": 1}}, "WAS") is None
    assert lookup_team_map({}, "WAS") is None


def test_game_has_started_respects_scheduled_code_over_stale_epoch():
    # Status says scheduled; a year-old epoch must not count as "already played".
    stale = _scheduled_wsh_game(gameTime_epoch="1726189200.0")  # 2025-09-13
    assert game_has_started(stale) is False
    assert game_has_started(_scheduled_wsh_game()) is False
    assert game_has_started(None) is False
    assert game_has_started({"gameStatusCode": "1"}) is True
    assert game_has_started({"gameStatusCode": "2"}) is True


def test_normalize_fallback_when_code_missing():
    now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    future = {"gameTime_epoch": str(now.timestamp() + 7 * 86400)}
    past = {"gameTime_epoch": str(now.timestamp() - 7 * 86400)}
    assert game_has_started(future, now=now) is False
    assert game_has_started(past, now=now) is True


def test_build_games_by_team_indexes_was_and_wsh():
    by_team = build_games_by_team([_scheduled_wsh_game()])
    assert "WSH" in by_team and "WAS" in by_team
    assert by_team["WAS"]["status"] == "pre"
    assert by_team["WSH"]["game"]["away"] == "WSH"


def test_build_status_by_pid_was_player_on_wsh_schedule():
    games = build_games_by_team([_scheduled_wsh_game()])
    statuses = build_status_by_pid(
        {"11566": {"team": "WAS", "name": "Jayden Daniels"}},
        games,
        {"WAS": {"byeWeek": 9}},
        current_week=1,
    )
    assert statuses["11566"] == STATUS_NOT_STARTED


def test_build_status_looks_up_alias_when_map_not_preindexed():
    # Raw map keyed only as Tank01 sent it (WSH), no build_games_by_team.
    games = {"WSH": {"status": "pre", "game": _scheduled_wsh_game()}}
    statuses = build_status_by_pid(
        {"11566": {"team": "WAS"}},
        games,
        {},
        current_week=1,
    )
    assert statuses["11566"] == STATUS_NOT_STARTED


def _matchups():
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


DANIELS_STATS = {
    "WAS": {
        "QB": {
            "jayden daniels": {
                "pass_yds": 233,
                "pass_td": 1,
                "int": 0,
                "rush_att": 11,
                "rush_yds": 68,
                "rush_td": 0,
            }
        }
    }
}


def _daniels_matchup():
    return {
        "left": {
            "name": "JiggyJay30", "roster_id": "1", "record": "0-0",
            "username": "a", "avatar": "", "pts_total": 0.0,
            "starters": [{
                "pid": "4984", "name": "Justin Herbert", "pos": "QB",
                "nfl": "LAC", "pts": 0.0,
            }],
        },
        "right": {
            "name": "JJettas 2 Holiday", "roster_id": "2", "record": "0-0",
            "username": "b", "avatar": "", "pts_total": 0.0,
            "starters": [{
                "pid": "11566", "name": "Jayden Daniels", "pos": "QB",
                "nfl": "WAS", "pts": 0.0,
            }],
        },
    }


def test_matchup_hides_stale_was_box_score_before_kickoff(monkeypatch):
    mmod = _matchups()
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_a, **_k: DANIELS_STATS)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_a, **_k: [])
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "_allow_live_game_indicators", lambda *_a, **_k: False)
    monkeypatch.setattr(mmod, "get_nfl_scores_for_date", lambda *_a, **_k: None)

    html = mmod.render_matchup_slide(
        "2026", _daniels_matchup(), w=1, proj_week=1,
        # Reproduce the production bug: WAS miss-keyed as FINAL.
        status_by_pid={"11566": mmod.STATUS_FINAL, "4984": mmod.STATUS_NOT_STARTED},
        projections={},
        players={},
        teams={},
        team_game_lookup={"WSH": _scheduled_wsh_game(), "LAC": {
            "home": "LAC", "away": "ARI", "gameDate": "20260913",
            "gameTime": "4:25p", "gameStatusCode": "0",
        }},
    )
    assert "233" not in html
    assert "yds" not in html
    assert "Jayden Daniels" in html
    assert "m-cell-stats" not in html


def test_matchup_shows_box_score_once_game_is_live(monkeypatch):
    mmod = _matchups()
    live = _scheduled_wsh_game(gameStatusCode="1", gameStatus="In Progress")
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda *_a, **_k: DANIELS_STATS)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_a, **_k: [])
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_a, **_k: {})
    monkeypatch.setattr(mmod, "_allow_live_game_indicators", lambda *_a, **_k: True)
    monkeypatch.setattr(mmod, "get_nfl_scores_for_date", lambda *_a, **_k: None)

    html = mmod.render_matchup_slide(
        "2026", _daniels_matchup(), w=1, proj_week=1,
        status_by_pid={"11566": mmod.STATUS_IN_PROGRESS},
        projections={},
        players={},
        teams={},
        team_game_lookup={"WSH": live},
    )
    assert "233 yds" in html
    assert "1 td" in html
    assert "11 car" in html
    assert "68 yds" in html
    assert "m-cell-stats" in html


def test_week_stats_builder_writes_empty_before_kickoff(monkeypatch, tmp_path):
    import utils.utils as umod

    out = tmp_path / "week_stats.json"
    monkeypatch.setattr(umod, "path_week_stats", lambda *_a, **_k: str(out))
    monkeypatch.setattr(umod, "load_week_schedule", lambda *_a, **_k: [_scheduled_wsh_game()])

    def _boom(*_a, **_k):
        raise AssertionError("Footballguys scrape must not run before kickoff")

    monkeypatch.setattr(umod, "fetch_team_game_logs_html", _boom)

    path = umod.build_and_save_week_stats_for_league({"WAS": {}}, 2026, 1, live_game_ids=None)
    assert str(path) == str(out)
    assert out.read_text(encoding="utf-8").strip() == "{}"


def test_week_stats_builder_skips_when_schedule_missing(monkeypatch, tmp_path):
    import utils.utils as umod

    out = tmp_path / "week_stats.json"
    out.write_text('{"keep": true}', encoding="utf-8")
    monkeypatch.setattr(umod, "path_week_stats", lambda *_a, **_k: str(out))
    monkeypatch.setattr(umod, "load_week_schedule", lambda *_a, **_k: [])

    def _boom(*_a, **_k):
        raise AssertionError("must not scrape when schedule is missing")

    monkeypatch.setattr(umod, "fetch_team_game_logs_html", _boom)

    umod.build_and_save_week_stats_for_league({"WAS": {}}, 2026, 1)
    assert out.read_text(encoding="utf-8") == '{"keep": true}'
