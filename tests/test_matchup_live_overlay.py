"""Live overlay for the Matchup page (Weekly Hub).

The matchup slides render from the cached league context, so a player's
points and stat line used to trail ScoreZone by minutes -- and a plain page
refresh only re-rendered the same cached numbers. The overlay re-fetches
the week's provider matchups on ScoreZone's 12s cache and builds stat
lines from a fresh Sleeper week-stats fetch (Sleeper's stats endpoint is
live during games) behind a short in-memory TTL, current week and
live/final games only. These contracts guard that behaviour; fixtures
mirror the real Sleeper matchup-row and Sleeper stats-payload shapes, and
the live gating is driven entirely by fixture game codes / the injected
current week (never a hard-coded calendar date).
"""
from __future__ import annotations

import pytest

WEEK = 10
GAME_ID = "20261115_DAL@BUF"
SCORING = {
    "rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "fgm": 3.0, "xpm": 1.0,
}


def _mmod():
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


def _built_matchups():
    """Team-block matchups as build_matchups_by_week bakes them into ctx:
    Lamb's line is stale (3.0), Allen has no provider actual yet."""
    return [{
        "matchup_id": 1,
        "left": {
            "roster_id": "1", "name": "Team A", "avatar": "", "record": "5-4",
            "pts_total": 28.0, "proj_total": 110.0,
            "starters": [
                {"pid": "1001", "name": "CeeDee Lamb", "pos": "WR", "nfl": "DAL", "pts": 3.0},
                {"pid": "1005", "name": "Jake Ferguson", "pos": "TE", "nfl": "DAL", "pts": 5.0},
                {"pid": "1002", "name": "Justin Jefferson", "pos": "WR", "nfl": "MIN", "pts": 20.0},
                {"pid": "DAL", "name": "Cowboys", "pos": "DEF", "nfl": "DAL", "pts": 7.0},
            ],
            "bench": [],
        },
        "right": {
            "roster_id": "2", "name": "Team B", "avatar": "", "record": "4-5",
            "pts_total": 0.0, "proj_total": 105.0,
            "starters": [
                {"pid": "1003", "name": "Josh Allen", "pos": "QB", "nfl": "BUF", "pts": None},
                {"pid": "1004", "name": "Brandon Aubrey", "pos": "K", "nfl": "DAL", "pts": None},
            ],
            "bench": [],
        },
        "h2h": {},
    }]


def _fresh_rows():
    """Sleeper-shaped matchup rows from the 12s re-fetch. Lamb's provider
    actual (18.6) deliberately differs from his stat-line-derived points
    (22.8); Allen / Aubrey have no provider actual at all."""
    return [
        {
            "roster_id": 1, "matchup_id": 1, "points": 43.6,
            "players": ["1001", "1005", "1002", "DAL"],
            "starters": ["1001", "1005", "1002", "DAL"],
            "players_points": {"1001": 18.6, "1005": 5.0, "1002": 20.0},
        },
        {
            "roster_id": 2, "matchup_id": 1,
            "players": ["1003", "1004"],
            "starters": ["1003", "1004"],
            "players_points": {},
        },
    ]


def _sleeper_stats():
    """Sleeper stats payload for the week, keyed by pid (team defenses by
    team abbreviation), raw Sleeper key names and float values -- the real
    /v1/stats/nfl/regular/{season}/{week} shape. Ferguson (1005) is absent:
    he keeps his existing line and points. The DAL row also carries
    whole-team aggregates (td, off_yd) that are NOT the defense's line."""
    return {
        "1001": {
            "rec": 7.0, "rec_tgt": 9.0, "rec_yd": 92.0, "rec_td": 1.0,
            "rush_att": 1.0, "rush_yd": 6.0, "rush_td": 0.0,
            "pts_ppr": 20.2, "gp": 1.0,
        },
        "1003": {
            "pass_cmp": 24.0, "pass_att": 35.0, "pass_yd": 300.0,
            "pass_td": 2.0, "pass_int": 0.0,
            "rush_att": 6.0, "rush_yd": 40.0, "rush_td": 0.0,
        },
        "1004": {
            "fgm": 2.0, "fga": 2.0, "xpm": 3.0, "xpa": 3.0, "fgm_yds": 71.0,
        },
        "DAL": {
            "sack": 3.0, "int": 1.0, "pts_allow": 17.0, "def_td": 0.0,
            "fum_rec": 1.0, "ff": 2.0, "yds_allow": 305.0, "sack_yd": 21.0,
            "td": 4.0, "off_yd": 409.0, "tkl": 77.0,
        },
    }


def _game_lookup():
    live = {
        "gameID": GAME_ID, "gameStatusCode": "1",
        "home": "BUF", "away": "DAL",
        "lineScore": {"period": "3"}, "gameClock": "5:00",
    }
    scheduled = {"gameID": "20261115_MIN@GB", "gameStatusCode": "0",
                 "home": "GB", "away": "MIN"}
    return {"DAL": live, "BUF": live, "MIN": scheduled}


def _wire(monkeypatch, m, *, fresh=None, payload=None, get_matchups_raises=False,
          sleeper_raises=False):
    calls = {"get_matchups": 0, "sleeper": 0}

    def fake_get_matchups(platform, league_id, week, season, *, cache_ttl=None):
        calls["get_matchups"] += 1
        assert cache_ttl == 12  # ScoreZone's live cache bound
        if get_matchups_raises:
            raise RuntimeError("provider down")
        return fresh if fresh is not None else _fresh_rows()

    def fake_fetch_week_stats(season, week, force=False):
        calls["sleeper"] += 1
        assert force is True  # the in-progress week must bypass the file cache
        assert (season, week) == (2026, WEEK)
        if sleeper_raises:
            raise RuntimeError("sleeper down")
        return payload if payload is not None else _sleeper_stats()

    monkeypatch.setattr(m, "get_matchups", fake_get_matchups)
    # No fresh scoreboard body -> the overlay falls back to the caller's
    # team_game_lookup, which is the fixture above.
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a, **k: {})
    import data_building.external_data.sleeper_bulk_stats as sbs
    monkeypatch.setattr(sbs, "fetch_week_stats", fake_fetch_week_stats)
    m._sleeper_live_stats_memo.clear()
    return calls


def _apply(m, matchups, **overrides):
    kwargs = dict(
        platform="sleeper", league_id="L1", season=2026,
        week=WEEK, current_week=WEEK,
        team_game_lookup=_game_lookup(), scoring_settings=SCORING,
    )
    kwargs.update(overrides)
    return m.apply_live_matchup_overlay(matchups, **kwargs)


def _starter(matchups, side, pid):
    for p in matchups[0][side]["starters"]:
        if p and str(p.get("pid")) == pid:
            return p
    raise AssertionError(f"starter {pid} not found on {side}")


def test_overlay_replaces_stale_points_and_totals(monkeypatch):
    m = _mmod()
    calls = _wire(monkeypatch, m)
    built = _built_matchups()
    out, overlay = _apply(m, built)

    # Provider actual replaces the stale baked-in points; team total comes
    # from the fresh row's points.
    assert _starter(out, "left", "1001")["pts"] == pytest.approx(18.6)
    assert out[0]["left"]["pts_total"] == pytest.approx(43.6)
    # The cached context's dicts are never mutated.
    assert _starter(built, "left", "1001")["pts"] == pytest.approx(3.0)
    assert built[0]["left"]["pts_total"] == pytest.approx(28.0)
    # One Sleeper stats fetch serves every live game on the page.
    assert calls["sleeper"] == 1
    assert overlay is not None


def test_provider_points_win_over_sleeper_derived(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, _ = _apply(m, _built_matchups())
    # Lamb's line derives 22.8 (7 rec + 92 yds + TD + 6 rush yds) but the
    # provider's 18.6 is authoritative, exactly as in ScoreZone.
    assert _starter(out, "left", "1001")["pts"] == pytest.approx(18.6)


def test_missing_provider_actual_falls_back_to_sleeper_line(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, _ = _apply(m, _built_matchups())
    # Allen: 300 pass yds (12) + 2 pass TD (8) + 40 rush yds (4) = 24.0.
    assert _starter(out, "right", "1003")["pts"] == pytest.approx(24.0)
    # Aubrey (K): 2 FG (6) + 3 XP (3) = 9.0.
    assert _starter(out, "right", "1004")["pts"] == pytest.approx(9.0)
    # No fresh team total for roster 2, so it is recomputed from starters.
    assert out[0]["right"]["pts_total"] == pytest.approx(33.0)


def test_player_absent_from_sleeper_payload_keeps_existing(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, overlay = _apply(m, _built_matchups())
    # Ferguson is in the live game but not in the Sleeper payload: points
    # stay on the provider value and no week-stats entry is fabricated.
    assert _starter(out, "left", "1005")["pts"] == pytest.approx(5.0)
    te_bucket = (overlay or {}).get("DAL", {}).get("TE", {})
    assert not any("ferguson" in name for name in te_bucket)
    # Jefferson's game has not kicked off: untouched by the stats path.
    assert _starter(out, "left", "1002")["pts"] == pytest.approx(20.0)
    assert "MIN" not in (overlay or {})


def test_overlay_stat_line_shape_for_slide(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    _, overlay = _apply(m, _built_matchups())
    lamb = overlay["DAL"]["WR"]["ceedee lamb"]
    assert lamb["rec"] == 7 and lamb["tgt"] == 9
    assert lamb["rec_yds"] == 92 and lamb["rec_td"] == 1
    assert lamb["rush_att"] == 1 and lamb["rush_yds"] == 6
    # Sleeper-only bookkeeping keys never leak into the week_stats entry.
    assert "pts_ppr" not in lamb and "gp" not in lamb
    allen = overlay["BUF"]["QB"]["josh allen"]
    assert allen["pass_cmp"] == 24 and allen["pass_att"] == 35
    assert allen["pass_yds"] == 300 and allen["pass_td"] == 2
    assert allen["int"] == 0


def test_overlay_k_line_from_sleeper_fg_xp_keys(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    _, overlay = _apply(m, _built_matchups())
    aubrey = overlay["DAL"]["K"]["brandon aubrey"]
    assert aubrey["fgm"] == 2 and aubrey["fga"] == 2
    assert aubrey["xpm"] == 3 and aubrey["xpa"] == 3


def test_overlay_def_line_from_sleeper_team_entry(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, overlay = _apply(m, _built_matchups())
    # The team-defense row (pid = team abbr) lands in the IDP bucket's
    # "team defense" slot -- the shape the slide's DEF line sums.
    entry = overlay["DAL"]["IDP"]["team defense"]
    assert entry["sack"] == 3 and entry["int"] == 1
    assert entry["pts_allow"] == 17 and entry["yds_allow"] == 305
    assert entry["fum_rec"] == 1 and entry["ff"] == 2
    # Whole-team aggregates on the same payload row are not the DEF line.
    assert "td" not in entry and "off_yd" not in entry and "tkl" not in entry
    # DEF points come from the provider only: no actual in the fresh row,
    # so the baked-in points stand (no derivation from the DEF line, whose
    # "int" would mis-score as a QB interception).
    assert _starter(out, "left", "DAL")["pts"] == pytest.approx(7.0)


def test_sleeper_fetch_memoized_within_ttl(monkeypatch):
    m = _mmod()
    calls = _wire(monkeypatch, m)
    _apply(m, _built_matchups())
    _apply(m, _built_matchups())
    # The hub's 20s poll cannot hammer Sleeper: the second render inside
    # the TTL window reuses the first fetch.
    assert calls["sleeper"] == 1
    # Once the memo entry ages out, the next fetch goes upstream again.
    key = (2026, WEEK)
    ts, data = m._sleeper_live_stats_memo[key]
    m._sleeper_live_stats_memo[key] = (ts - m._SLEEPER_LIVE_STATS_TTL - 1, data)
    assert m._fetch_sleeper_live_week_stats(2026, WEEK)
    assert calls["sleeper"] == 2


def test_sleeper_fetch_failure_keeps_points_overlay_only(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m, sleeper_raises=True)
    built = _built_matchups()
    out, overlay = _apply(m, built)
    # Points still refresh from the provider re-fetch; stat lines fall
    # back to the snapshot (no overlay), and nothing raises.
    assert _starter(out, "left", "1001")["pts"] == pytest.approx(18.6)
    assert overlay is None


def test_sleeper_empty_payload_keeps_points_overlay_only(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m, payload={})
    out, overlay = _apply(m, _built_matchups())
    assert _starter(out, "left", "1001")["pts"] == pytest.approx(18.6)
    assert overlay is None


def test_past_week_untouched(monkeypatch):
    m = _mmod()
    calls = _wire(monkeypatch, m)
    built = _built_matchups()
    out, overlay = _apply(m, built, week=WEEK - 1)
    assert out is built
    assert overlay is None
    assert calls["get_matchups"] == 0
    assert calls["sleeper"] == 0


def test_overlay_failure_falls_back_to_current_rendering(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m, get_matchups_raises=True, sleeper_raises=True)
    built = _built_matchups()
    out, overlay = _apply(m, built)
    assert out is built
    assert overlay is None
    assert _starter(out, "left", "1001")["pts"] == pytest.approx(3.0)


def _render_slide(monkeypatch, m, matchup, live_week_stats, *,
                  status_by_pid=None, snapshot=None):
    monkeypatch.setattr("dashboard_services.api.get_nfl_state", lambda: {})
    if snapshot is None:
        snapshot = {
            "DAL": {"WR": {"ceedee lamb": {
                "rec": 2, "tgt": 3, "rec_yds": 21, "rec_td": 0,
                "rush_att": 0, "rush_yds": 0, "rush_td": 0,
            }}},
        }
    monkeypatch.setattr(m, "load_week_stats", lambda *a, **k: snapshot)
    monkeypatch.setattr(m, "overlay_idp_and_k_stats_from_sleeper", lambda **k: None)
    monkeypatch.setattr(m, "load_week_schedule", lambda *a, **k: [])
    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *a: {})
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a, **k: None)
    monkeypatch.setattr("utils.data_cache.load_week_projection", lambda *a, **k: {})

    game = {
        "gameID": GAME_ID, "gameStatusCode": "1",
        "home": "BUF", "away": "DAL",
        "lineScore": {"period": "3"}, "gameClock": "5:00",
    }
    return m.render_matchup_slide(
        2026, matchup, w=WEEK, proj_week=WEEK - 1,
        status_by_pid=status_by_pid or {"1001": m.STATUS_IN_PROGRESS},
        projections={WEEK: {"projections": {"1001": 20.0}}},
        players={}, teams={},
        team_game_lookup={"DAL": game, "BUF": game},
        scoring_settings=SCORING,
        live_week_stats=live_week_stats,
    )


def test_slide_renders_fresh_line_and_points_over_stale_snapshot(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, overlay = _apply(m, _built_matchups())
    html = _render_slide(monkeypatch, m, out[0], overlay)
    # Fresh Sleeper line replaces the stale snapshot line...
    assert "7/9 92 yds, 1 TD" in html
    assert "21 yds" not in html
    # ...and the fresh provider points are what the row scores.
    assert "18.6" in html


def test_slide_without_overlay_keeps_stale_snapshot_line(monkeypatch):
    m = _mmod()
    html = _render_slide(monkeypatch, m, _built_matchups()[0], None)
    assert "2/3 21 yds" in html


def test_slide_renders_fresh_def_line_without_double_counting(monkeypatch):
    m = _mmod()
    _wire(monkeypatch, m)
    out, overlay = _apply(m, _built_matchups())
    # Snapshot carries a stale IDP-bucket sum for DAL; the fresh team
    # entry must replace it, not add to it.
    snapshot = {
        "DAL": {"IDP": {"stale defender": {
            "sack": 1, "int": 0, "pts_allow": 40, "yds_allow": 500,
            "fum_rec": 0, "ff": 0,
        }}},
    }
    matchup = {
        "left": {
            "roster_id": "1", "name": "Team A", "avatar": "", "record": "5-4",
            "pts_total": 7.0, "proj_total": 100.0,
            "starters": [
                {"pid": "DAL", "name": "Cowboys", "pos": "DEF", "nfl": "DAL", "pts": 7.0},
            ],
            "bench": [],
        },
        "right": {
            "roster_id": "2", "name": "Team B", "avatar": "", "record": "4-5",
            "pts_total": 0.0, "proj_total": 100.0,
            "starters": [],
            "bench": [],
        },
        "h2h": {},
    }
    html = _render_slide(
        monkeypatch, m, matchup, overlay,
        status_by_pid={"DAL": m.STATUS_IN_PROGRESS}, snapshot=snapshot,
    )
    assert "17 pa, 3 sacks, 1 int, 1 fr, 2 ff, 305 ya" in html
    assert "40 pa" not in html
    assert "4 sacks" not in html


def test_hub_auto_refresh_interval_is_20s():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "dashboard_services" / "pages" / "weekly_hub_page.py").read_text()
    assert "}}, 20000);" in src
    assert "Auto-refresh the week every 20 s" in src
