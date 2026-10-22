"""League-wide rank fix for the tier-anchored start/sit index.

Regression: the Start/Sit page mapped each player's rank WITHIN THE
VIEWER'S OWN ROSTER through the tier bands, which are league-wide
(36 deep at RB/WR). Every rostered RB/WR landed in the 90-100 band
(RJ Harvey showed 98 as the roster's RB3). The index must come from
the player's rank in the full league-wide pool, on every surface.
"""

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


# ── Pool builder ──────────────────────────────────────────────────────────

def _fake_bundles(monkeypatch, bundles):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: "ppr")
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: dict(bundles))
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: {pid: {} for pid in bundles})
    appmod._SS_POOL_CACHE.clear()


def test_pool_from_bundles_is_league_wide_scores(monkeypatch):
    import app as appmod

    # 45 RBs, projections 30.0 down in 0.5 steps; score == proj here
    # (no form/usage/etc inputs, so every multiplier is neutral).
    bundles = {}
    for i in range(45):
        bundles[f"rb{i}"] = {"pos": "RB", "proj_pts": 30.0 - 0.5 * i}
    for i in range(5):
        bundles[f"wr{i}"] = {"pos": "WR", "proj_pts": 25.0 - i}
    _fake_bundles(monkeypatch, bundles)

    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["kind"] == "score"
    assert len(pool["by_pos"]["RB"]) == 45
    assert pool["by_pos"]["RB"][0] == 30.0
    assert pool["by_pos"]["RB"] == sorted(pool["by_pos"]["RB"],
                                          reverse=True)
    appmod._SS_POOL_CACHE.clear()


def test_pool_below_min_bundles_falls_back_to_projections(monkeypatch):
    import app as appmod
    import utils.fantasy_scoring as fs_mod
    import utils.utils as uu_mod

    # Only a handful of bundles: a partial batch, not a league pool.
    _fake_bundles(monkeypatch, {"rb0": {"pos": "RB", "proj_pts": 30.0}})
    projs = {f"rb{i}": 20.0 - i for i in range(12)}
    monkeypatch.setattr(uu_mod, "load_week_projection",
                        lambda *a, **k: {pid: {} for pid in projs})
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda m, pid, s, pos: projs.get(str(pid)))
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: {pid: {"pos": "RB"}
                                         for pid in projs})
    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["kind"] == "proj"
    assert pool["by_pos"]["RB"][0] == 20.0
    appmod._SS_POOL_CACHE.clear()


def test_pool_exotic_scoring_ranks_on_full_start_score(monkeypatch):
    """Stroud-shaped regression (2026-10-22): a league whose scoring is
    not a canonical variant (milestone bonuses) used to get a PROJECTION
    pool, so its index ranked players by projection while its own modal
    showed the factor-adjusted start score — Stroud printed QB7/89 off
    his projection rank when his start score ranked QB5. The pool must
    rank on the full start score for every scoring the bundles can be
    re-derived under."""
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import utils.fantasy_scoring as fs_mod
    import utils.utils as uu_mod

    exotic = {"rec": 1.0, "bonus_rush_yd_200": 5.0}
    # rbA projects BELOW rbB but out-scores him: strong positive form
    # (+10 PPG last-3 over season) and a rising role vs rbB's fade.
    proj = {"rbA": 11.5, "rbB": 12.0}
    pids = ["rbA", "rbB"] + [f"rb{i}" for i in range(43)]
    for i, pid in enumerate(pids[2:]):
        proj[pid] = 9.0 - 0.1 * i
    base = {pid: {"pos": "RB", "proj_pts": 5.0} for pid in pids}
    derived = {
        "rbA": {"recent_ppg": 20.0, "season_ppg": 10.0,
                "usage_delta": 3.0, "usage_season_avg": 20.0,
                "played_games": 6},
        "rbB": {"recent_ppg": 8.0, "season_ppg": 16.0,
                "usage_delta": -3.0, "usage_season_avg": 20.0,
                "played_games": 6},
    }
    for pid in pids[2:]:
        derived[pid] = {"recent_ppg": proj[pid], "season_ppg": proj[pid],
                        "usage_delta": None, "usage_season_avg": None,
                        "played_games": 6}
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: dict(base))
    monkeypatch.setattr(bundle_mod, "derive_start_sit_inputs",
                        lambda *a, **k: dict(derived))
    monkeypatch.setattr(bundle_mod, "_variant_week_points",
                        lambda *a, **k: ({}, {}))
    monkeypatch.setattr(uu_mod, "load_week_projection",
                        lambda *a, **k: {pid: {} for pid in pids})
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda m, pid, s, pos: proj.get(str(pid)))
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: {pid: {} for pid in pids})
    appmod._SS_POOL_CACHE.clear()

    pool = appmod.start_score_pool_scores(2026, 6, exotic)
    assert pool["kind"] == "score"
    scores = pool["by_pos"]["RB"]
    assert len(scores) == 45
    # Score order flips the projection order at the top.
    assert scores[0] == pytest.approx(12.41, abs=0.05)   # rbA
    assert scores[1] == pytest.approx(11.08, abs=0.05)   # rbB
    idx_a = appmod.start_score_league_index(
        2026, 6, exotic, "RB", scores[0], proj["rbA"])
    idx_b = appmod.start_score_league_index(
        2026, 6, exotic, "RB", scores[1], proj["rbB"])
    assert idx_a == appmod.start_score_tier_index("RB", 1) == 100
    assert idx_b == appmod.start_score_tier_index("RB", 2) == 99
    appmod._SS_POOL_CACHE.clear()


# ── League index (rank -> tier band) ──────────────────────────────────────

def _pool(monkeypatch, kind, by_pos):
    import app as appmod
    monkeypatch.setattr(appmod, "start_score_pool_scores",
                        lambda *a, **k: {"kind": kind, "by_pos": by_pos})


def test_league_index_uses_league_rank_not_roster_rank(monkeypatch):
    import app as appmod

    # 36-deep RB pool, scores 36 down to 1. A 13.0 score is league RB24.
    _pool(monkeypatch, "score",
          {"RB": [float(v) for v in range(36, 0, -1)]})
    idx = appmod.start_score_league_index(2026, 6, {}, "RB", 13.0, 13.0)
    # RB24 sits in the FLEX band (23-30 -> 40-69), nowhere near 98.
    assert idx == appmod.start_score_tier_index("RB", 24) == 65


def test_league_index_proj_kind_ranks_on_projection(monkeypatch):
    import app as appmod

    _pool(monkeypatch, "proj",
          {"RB": [float(v) for v in range(36, 0, -1)]})
    # Score says elite, projection says RB24: proj pool ranks the proj.
    idx = appmod.start_score_league_index(2026, 6, {}, "RB", 99.0, 13.0)
    assert idx == 65


def test_league_index_no_pool_returns_none(monkeypatch):
    import app as appmod

    _pool(monkeypatch, None, {})
    assert appmod.start_score_league_index(
        2026, 6, {}, "RB", 13.0, 13.0) is None
    # Positions without tier bands never get an index.
    _pool(monkeypatch, "score", {"K": [10.0]})
    assert appmod.start_score_league_index(
        2026, 6, {}, "K", 10.0, 10.0) is None
    # Zero / missing values never get an index (e.g. bye weeks).
    _pool(monkeypatch, "score", {"RB": [10.0]})
    assert appmod.start_score_league_index(
        2026, 6, {}, "RB", 0.0, 0.0) is None


# ── Start/Sit page end to end ─────────────────────────────────────────────

WEEK = 5
PROJ = {"rb1": 20.0, "rb2": 14.0, "rb3": 11.0, "rb4": 8.0, "qb1": 22.0,
        "rbF": 9.0}
NAMES = {"rb1": "Runner One", "rb2": "Runner Two", "rb3": "Runner Three",
         "rb4": "Runner Four", "qb1": "Quarter Back", "rbF": "Free Runner"}
TEAMS = {"rb1": "KC", "rb2": "BUF", "rb3": "MIA", "rb4": "DEN",
         "qb1": "KC", "rbF": "NYJ"}
POS = {p: ("QB" if p == "qb1" else "RB") for p in NAMES}
SCHED = [{"home": "KC", "away": "BUF"}, {"home": "MIA", "away": "DEN"},
         {"home": "DAL", "away": "NYJ"}, {"home": "GB", "away": "CHI"}]

# League-wide pool around the roster: extra RBs slotted between the
# roster's projections so league ranks are 5 / 11 / 14 / 17 (guest: 16),
# and the QB pool puts qb1 at league rank 4.
POOL_PROJ = dict(PROJ)
for _pid, _pts in [("x1", 28.0), ("x2", 26.0), ("x3", 24.0), ("x4", 22.5),
                   ("x5", 19.0), ("x6", 18.0), ("x7", 17.0), ("x8", 16.0),
                   ("x9", 15.0), ("x10", 13.0), ("x11", 12.0), ("x12", 10.0)]:
    POOL_PROJ[_pid] = _pts
for _pid, _pts in [("q2", 28.0), ("q3", 26.0), ("q4", 24.0)]:
    POOL_PROJ[_pid] = _pts


def _ctx():
    idx = {p: {"pos": POS[p], "position": POS[p], "team": TEAMS[p],
               "full_name": NAMES[p], "name": NAMES[p], "injury_status": ""}
           for p in NAMES}
    return {
        "viewer": {"viewer_roster_id": 1},
        "current_week": WEEK,
        "rosters": [
            {"roster_id": 1, "owner_id": "me-user",
             "players": ["rb1", "rb2", "rb3", "rb4", "qb1"],
             "reserve": [], "taxi": [], "metadata": {"team_name": "My Team"}},
        ],
        "users": [{"user_id": "me-user", "display_name": "Kaedon"}],
        "players_index": idx,
        "players": idx,
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN", "BN"],
        "scoring_settings": {},
        "raw_scoring_settings": {},
        "proj_by_week": {WEEK: {"projections": dict(PROJ)}},
    }


@pytest.fixture
def league_client(offline_client, monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import data_building.weekly_metrics as wm_mod
    import utils.fantasy_scoring as fs_mod
    import utils.game_conditions as gc_mod
    import utils.utils as uu_mod

    appmod._SS_POOL_CACHE.clear()
    monkeypatch.setattr(appmod, "_session_signed_in", lambda: True)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache",
                        lambda *a, **k: _ctx())
    rows = [{"id": p, "name": NAMES[p], "position": POS[p], "team": TEAMS[p],
             "value": 5000, "sf_value": 5000, "pos_rank_label": "RB1"}
            for p in NAMES]
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda *a, **k: rows)
    monkeypatch.setattr(appmod, "_matchup_rank_table",
                        lambda *a, **k: ({}, 32, None, None))
    monkeypatch.setattr(appmod, "_fpts_against_effective", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_team_play_volume", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_season_weekstat_points",
                        lambda *a, **k: ((), ()))
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_season_snap_totals",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "build_projections_by_week",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "start_score_pos_anchors", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_game_env", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_qb_situation", lambda *a, **k: None)
    # No bundle coverage: the pool must come from the projection fallback.
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda *a, **k: None)
    monkeypatch.setattr(bundle_mod, "load_start_score_bundles",
                        lambda *a, **k: {})
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda *a, **k: {})
    monkeypatch.setattr(gc_mod, "build_week_conditions", lambda *a, **k: {})
    monkeypatch.setattr(uu_mod, "load_week_projection",
                        lambda *a, **k: {pid: {} for pid in POOL_PROJ})
    monkeypatch.setattr(uu_mod, "load_week_sched", lambda *a, **k: list(SCHED))
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda m, pid, s, pos: POOL_PROJ.get(str(pid)))
    pool_index = {}
    for pid, pts in POOL_PROJ.items():
        pos = POS.get(pid) or ("QB" if pid.startswith("q") else "RB")
        pool_index[pid] = {"pos": pos, "position": pos}
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: pool_index)
    yield offline_client
    appmod._SS_POOL_CACHE.clear()


def test_page_index_is_league_wide_for_roster_and_guest(league_client):
    import app as appmod

    base = "/api/start-sit-options?platform=sleeper&league_id=L1&season=2026"
    d = league_client.get(base + "&guests=rbF").get_json()
    rows = {p["player_id"]: p for p in d["positions"]["RB"]}
    # League ranks 5 / 11 / 14 / 17 through the RB bands. The old
    # roster-local mapping gave 100 / 99 / 98 / 97 for these same rows.
    assert rows["rb1"]["start_score_pct"] == \
        appmod.start_score_tier_index("RB", 5) == 96
    assert rows["rb2"]["start_score_pct"] == \
        appmod.start_score_tier_index("RB", 11) == 89
    assert rows["rb3"]["start_score_pct"] == \
        appmod.start_score_tier_index("RB", 14) == 84
    assert rows["rb4"]["start_score_pct"] == \
        appmod.start_score_tier_index("RB", 17) == 79
    # Guests get the same league-wide index as roster players (league
    # rank 16), not a different formula.
    assert rows["rbF"]["start_score_pct"] == \
        appmod.start_score_tier_index("RB", 16) == 80
    qb = {p["player_id"]: p for p in d["positions"]["QB"]}
    assert qb["qb1"]["start_score_pct"] == \
        appmod.start_score_tier_index("QB", 4) == 94
