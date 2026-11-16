"""Live freshness bounds folded into PR #3163.

Two live surfaces used to read provider matchups through the provider
ttl_cache default (300s) while ScoreZone and the Matchup live overlay read
the same rows on a 12s bound:

- Portfolio live cards + League Scores go through
  routes/user_pages_bp._compute_live_matchups -> build_matchup_preview.
  The 12s bound is threaded ONLY through that live chain: the default
  build_matchup_preview call (and build_matchups_by_week's ctx builds)
  must stay unbounded, exactly as before.
- The Start/Sit Lineup Lab's per-player live points re-read the week's
  rows on the 12s bound for the current week and fall back to the base
  rows on any failure/absence.

Fixtures mirror the real Sleeper matchup-row shape; gating is driven by
the injected ctx current_week and fixture game codes, never a calendar
date.
"""
from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("requests")


# ── build_matchup_preview bound threading ─────────────────────────────────

def _sleeper_rows():
    return [
        {"roster_id": 1, "matchup_id": 1, "points": 43.6,
         "players": ["1001", "1002"], "starters": ["1001", "1002"],
         "players_points": {"1001": 18.6, "1002": 25.0}},
        {"roster_id": 2, "matchup_id": 1, "points": 25.0,
         "players": ["1003"], "starters": ["1003"],
         "players_points": {"1003": 25.0}},
    ]


def _preview_stubs(record):
    def fake_get_matchups(platform, league_id, week, season, cache_ttl=None):
        record.append(cache_ttl)
        return _sleeper_rows()

    rosters = [
        {"roster_id": 1, "owner_id": "a", "players": ["1001", "1002"],
         "starters": ["1001", "1002"], "reserve": [],
         "settings": {"wins": 0, "losses": 0}},
        {"roster_id": 2, "owner_id": "b", "players": ["1003"],
         "starters": ["1003"], "reserve": [],
         "settings": {"wins": 0, "losses": 0}},
    ]
    users = [
        {"user_id": "a", "roster_id": 1, "display_name": "Ada"},
        {"user_id": "b", "roster_id": 2, "display_name": "Bea"},
    ]
    return fake_get_matchups, rosters, users


def _run_preview(record, **kwargs):
    from dashboard_services.matchups import build_matchup_preview

    fake, rosters, users = _preview_stubs(record)
    with mock.patch("dashboard_services.matchups.get_matchups", fake), \
            mock.patch("dashboard_services.matchups.get_rosters",
                       return_value=rosters), \
            mock.patch("dashboard_services.matchups.get_users",
                       return_value=users), \
            mock.patch("dashboard_services.matchups.get_league_settings",
                       return_value={}), \
            mock.patch("dashboard_services.matchups.team_avatar",
                       return_value=""):
        return build_matchup_preview(
            league_id="L1", week=10,
            roster_map={"1": "Owls", "2": "Bears"},
            players_map={
                "1001": {"name": "P1", "pos": "WR", "team": "DAL"},
                "1002": {"name": "P2", "pos": "RB", "team": "BUF"},
                "1003": {"name": "P3", "pos": "QB", "team": "BUF"},
            },
            season="2026", platform="sleeper", **kwargs,
        )


def test_preview_live_bound_reaches_provider_read():
    record: list = []
    preview = _run_preview(record, live_cache_ttl=12)
    assert record == [12]
    assert len(preview) == 1
    assert preview[0]["left"]["pts_total"] == pytest.approx(43.6)
    assert preview[0]["right"]["pts_total"] == pytest.approx(25.0)


def test_preview_default_call_stays_unbounded():
    record: list = []
    preview = _run_preview(record)
    # No cache_ttl kwarg reaches the provider read: the 300s default that
    # ctx builds have always used still governs this call.
    assert record == [None]
    assert len(preview) == 1


def test_build_matchups_by_week_does_not_pass_live_bound():
    from dashboard_services import service as service_mod

    seen: list = []

    def fake_preview(**kwargs):
        seen.append(kwargs)
        return []

    with mock.patch.object(service_mod, "build_matchup_preview", fake_preview):
        out = service_mod.build_matchups_by_week(
            "L1", [10], {}, {}, "2026", "sleeper")
    assert out == {10: []}
    assert seen and all(kw.get("live_cache_ttl") is None for kw in seen)


def test_compute_live_matchups_passes_live_bound():
    from dashboard_services import matchups as matchups_mod
    from routes import user_pages_bp
    import utils.utils as utils_mod

    seen: list = []

    def fake_preview(**kwargs):
        seen.append(kwargs)
        return []

    ctx = {"roster_map": {}, "players": {}, "players_index": {},
           "teams_index": {}, "raw_scoring_settings": {}}
    with mock.patch.object(matchups_mod, "build_matchup_preview", fake_preview), \
            mock.patch.object(utils_mod, "build_status_for_week",
                              return_value={}), \
            mock.patch.object(user_pages_bp, "build_projections_by_week",
                              return_value={}):
        user_pages_bp._compute_live_matchups(
            ("sleeper", "L1", 2026, 10), "sleeper", "L1", 2026, 10, ctx)
    assert seen and seen[0].get("live_cache_ttl") == 12


# ── Lineup Lab fresh live points ──────────────────────────────────────────

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.api as api_mod


_LAB_PROJ = {
    "1": 22.5, "2": 18.0, "3": 16.9, "4": 14.2, "5": 9.0,
    "6": 12.0, "7": 3.0, "8": 8.0, "9": 20.0, "10": 15.0,
}
_LAB_POS = {"1": "QB", "2": "RB", "3": "WR", "4": "WR", "5": "TE",
            "6": "RB", "7": "K", "8": "DEF", "9": "QB", "10": "RB"}
_LAB_TEAM = {"1": "LAR", "2": "DET", "3": "LAR", "4": "CIN", "5": "KC",
             "6": "SF", "7": "DAL", "8": "BUF", "9": "BUF", "10": "MIA"}
_LAB_SCHED = [
    {"home": "LAR", "away": "SF", "gameStatusCode": "2"},
    {"home": "DET", "away": "KC", "gameStatusCode": "1"},
    {"home": "CIN", "away": "MIA", "gameStatusCode": "2"},
    {"home": "DAL", "away": "BUF", "gameStatusCode": "2"},
]
_BASE_MINE = {"1": 20.0, "2": 10.0, "3": 18.2, "4": 6.4,
              "5": 7.5, "6": 12.0, "7": 9.0, "8": 11.0}
_BASE_OPP = {"9": 18.0, "10": 10.0}
_FRESH_MINE = {**_BASE_MINE, "1": 24.3}
_FRESH_OPP = {"9": 21.0, "10": 13.5}


def _lab_matchups(mine_points, opp_points):
    return [
        {"roster_id": 7, "matchup_id": 1,
         "starters": ["1", "2", "3", "5", "6", "7", "8"],
         "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
         "players_points": dict(mine_points)},
        {"roster_id": 3, "matchup_id": 1,
         "starters": ["9", "10"], "players": ["9", "10"],
         "players_points": dict(opp_points)},
    ]


def _lab_ctx():
    return {
        "current_week": 4,
        "rosters": [
            {"roster_id": 7, "owner_id": "u1",
             "players": ["1", "2", "3", "4", "5", "6", "7", "8"],
             "reserve": [], "taxi": []},
            {"roster_id": 3, "owner_id": "u2", "players": ["9", "10"],
             "reserve": [], "taxi": []},
        ],
        "users": [{"user_id": "u2", "display_name": "Pittsburgh Pilots"}],
        "players_index": {
            pid: {"position": _LAB_POS[pid], "pos": _LAB_POS[pid],
                  "team": _LAB_TEAM[pid], "full_name": f"Player {pid}",
                  "injury_status": ""}
            for pid in _LAB_PROJ
        },
        "players": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF",
                             "BN", "BN"],
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


def _lab_profiles(requests, season, week):
    out = {}
    for req in requests:
        pid = str(req["player_id"])
        mean = float(req["mean"])
        out[pid] = {
            "player_id": pid, "pos": req["pos"], "mean": mean,
            "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
            "dud_risk": 0.0, "n_games": 3.0, "factors": {},
        }
    return out


@pytest.fixture
def lab_mocks(monkeypatch):
    """Provider seam: the base read (no bound) serves stale points; the
    12s-bounded re-read (_cache_max_age=12 through the Sleeper provider)
    serves fresh points, raises, or returns nothing per ``fresh_mode``."""
    state = {"fresh_mode": "fresh"}

    def fake_get_matchups(league_id, week, **kwargs):
        if kwargs.get("_cache_max_age") == lab_mod._LIVE_POINTS_CACHE_TTL:
            if state["fresh_mode"] == "raise":
                raise RuntimeError("provider down")
            if state["fresh_mode"] == "empty":
                return []
            return _lab_matchups(_FRESH_MINE, _FRESH_OPP)
        return _lab_matchups(_BASE_MINE, _BASE_OPP)

    monkeypatch.setattr(api_mod, "get_matchups", fake_get_matchups)
    monkeypatch.setattr(pd_mod, "build_profiles", _lab_profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs",
                        lambda pids, season, ctx=None: {})
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched",
                        lambda season, week: list(_LAB_SCHED))
    monkeypatch.setattr(
        fs_mod, "weekly_projection_points",
        lambda raw, pid, scoring, pos="": _LAB_PROJ.get(str(pid)))
    return state


def _build_lab():
    return lab_mod.build_lineup_lab_payload(
        ctx=_lab_ctx(), league_id="123", viewer_roster_id=7,
        season=2026, week=4)


def test_lab_live_points_come_from_fresh_bounded_read(lab_mocks):
    payload = _build_lab()
    by_pid = {e["player_id"]: e for e in payload["you"]["lineup"]}
    # Player 1's game is final: locked at the FRESH 24.3, not the stale
    # base row's 20.0.
    assert by_pid["1"]["live"] == {"status": "final", "points": 24.3}
    # Opponent aggregate locks off the fresh rows too: 21.0 + 13.5.
    assert payload["opponent"]["live_points"] == 34.5


def test_lab_falls_back_to_base_rows_when_fresh_read_fails(lab_mocks):
    lab_mocks["fresh_mode"] = "raise"
    payload = _build_lab()
    by_pid = {e["player_id"]: e for e in payload["you"]["lineup"]}
    assert by_pid["1"]["live"] == {"status": "final", "points": 20.0}
    assert payload["opponent"]["live_points"] == 28.0


def test_lab_falls_back_to_base_rows_when_fresh_read_empty(lab_mocks):
    lab_mocks["fresh_mode"] = "empty"
    payload = _build_lab()
    by_pid = {e["player_id"]: e for e in payload["you"]["lineup"]}
    assert by_pid["1"]["live"] == {"status": "final", "points": 20.0}
    assert payload["opponent"]["live_points"] == 28.0
