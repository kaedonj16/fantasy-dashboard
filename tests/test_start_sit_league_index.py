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


# ── One-computation pool (by_pid) ─────────────────────────────────────────
# Regression 2026-10-22 (94/94): after #2783 made the pool rank on full
# start scores, the modal kept ranking its own LIVE-computed score
# against the bundle-built pool. The two assemblies of "the same" score
# differ (live pace/role/Vegas stores, unrounded inputs, a different
# season-PPG source), and in a dense QB top band both Maye's and
# Stroud's live values landed in the same pocket of the pool — exactly
# three pool values above each — so both printed rank 4 / index 94,
# a rank neither computation earned. The index must rank the pool's
# OWN value for the player, on every surface.

def test_pool_by_pid_carries_the_ranked_values(monkeypatch):
    import app as appmod

    bundles = {}
    for i in range(45):
        bundles[f"rb{i}"] = {"pos": "RB", "proj_pts": 30.0 - 0.5 * i}
    for i in range(5):
        bundles[f"wr{i}"] = {"pos": "WR", "proj_pts": 25.0 - i}
    _fake_bundles(monkeypatch, bundles)

    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["kind"] == "score"
    assert pool["by_pid"]["rb0"] == 30.0
    assert len(pool["by_pid"]) == 50
    for pos in ("RB", "WR"):
        from_pid = sorted(
            (v for pid, v in pool["by_pid"].items()
             if bundles[pid]["pos"] == pos), reverse=True)
        assert from_pid == pool["by_pos"][pos]
    appmod._SS_POOL_CACHE.clear()


def test_pool_live_injury_status_overlays_bundle_status(monkeypatch):
    """The pool scores a player with the LIVE injury status, like every
    surface that scores him directly: a bundle written while he was on
    IR must not zero him once he is active again."""
    import app as appmod

    bundles = {}
    for i in range(44):
        bundles[f"rb{i}"] = {"pos": "RB", "proj_pts": 30.0 - 0.5 * i}
    bundles["rbBack"] = {"pos": "RB", "proj_pts": 25.0,
                         "injury_status": "IR"}
    _fake_bundles(monkeypatch, bundles)
    index = {pid: {} for pid in bundles}
    index["rbBack"] = {"injury_status": "Active"}
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda *a, **k: index)

    pool = appmod.start_score_pool_scores(2026, 6, {"rec": 1.0})
    assert pool["by_pid"]["rbBack"] == pytest.approx(25.0)
    appmod._SS_POOL_CACHE.clear()


def _qb_pool_94_94(monkeypatch):
    """Fake score pool carrying the verified Week 7 QB values."""
    import app as appmod

    by_pid = {"maye": 21.88, "burrow": 21.82, "purdy": 21.10,
              "lamar": 21.08, "stroud": 20.86, "goff": 20.04,
              "love": 20.01, "williams": 19.43}
    pool = {"kind": "score",
            "by_pos": {"QB": sorted(by_pid.values(), reverse=True)},
            "by_pid": dict(by_pid)}
    monkeypatch.setattr(appmod, "start_score_pool_scores",
                        lambda *a, **k: pool)
    return appmod


def test_league_index_ranks_pool_value_not_caller_score(monkeypatch):
    appmod = _qb_pool_94_94(monkeypatch)

    # The 94/94 inputs: both callers' live scores sit at 21.09, in the
    # pocket of the pool with exactly three values above. Ranked on the
    # pool's own values, Maye is QB1 and Stroud QB5 — the caller's
    # foreign score must not move either index.
    assert appmod.start_score_league_index(
        2026, 7, {}, "QB", 21.09, 20.3, player_id="maye") == 100
    assert appmod.start_score_league_index(
        2026, 7, {}, "QB", 21.09, 18.6, player_id="stroud") == 92
    # A player with no pool value gets no index, never a ranked guess.
    assert appmod.start_score_league_index(
        2026, 7, {}, "QB", 21.09, 18.6, player_id="ghost") is None


def test_league_index_proj_kind_ranks_pool_projection(monkeypatch):
    import app as appmod

    pool = {"kind": "proj",
            "by_pos": {"RB": [30.0, 25.0, 20.0]},
            "by_pid": {"rbA": 25.0}}
    monkeypatch.setattr(appmod, "start_score_pool_scores",
                        lambda *a, **k: pool)
    # Caller score says elite; the pool's projection for rbA ranks 2nd.
    # Since 2026-11-06 an RB ranks against the combined flex pool: this
    # pool's only flex values are the three RB projections, so rbA is
    # rank 2 of a 3-player flex pool (the old per-position mapping
    # asserted tier_index("RB", 2) == 99 here).
    assert appmod.start_score_league_index(
        2026, 6, {}, "RB", 99.0, 25.0, player_id="rbA") == \
        appmod.start_score_flex_index(2, 3) == 89


def test_qb_acceptance_maye_100_stroud_92_from_pool(monkeypatch):
    """End to end through the real pool builder, engineered to the
    verified Week 7 QB score set: with foreign live scores supplied by
    the caller, the indexes still come out Maye 100 / Stroud 92."""
    import app as appmod

    def qb(proj, recent, season, bust, oline):
        return {"pos": "QB", "proj_pts": proj, "recent_ppg": recent,
                "season_ppg": season, "bust_rate": bust,
                "oline_index": oline, "played_games": 6}

    bundles = {
        "maye": qb(20.27, 25.0, 19.5, 0.39, 33.75),
        "burrow": qb(19.33, 22.0, 18.45, 0.27, 53.75),
        "purdy": qb(19.31, 14.0, 18.58, 0.25, 93.75),
        "lamar": qb(20.02, 17.0, 18.49, 0.35, 57.5),
        "stroud": qb(18.61, 22.67, 18.86, 0.29, 52.5),
    }
    for i in range(40):
        p = 19.9 - 0.3 * i
        bundles[f"qb{i}"] = qb(p, p, p, None, None)
    _fake_bundles(monkeypatch, bundles)

    pool = appmod.start_score_pool_scores(2026, 7, {"rec": 1.0})
    assert pool["kind"] == "score"
    top5 = sorted(pool["by_pid"].items(), key=lambda kv: -kv[1])[:5]
    assert [pid for pid, _ in top5] == \
        ["maye", "burrow", "purdy", "lamar", "stroud"]
    # Foreign caller scores (the 94/94 pocket) cannot move the indexes.
    assert appmod.start_score_league_index(
        2026, 7, {"rec": 1.0}, "QB", 21.09, 20.3,
        player_id="maye") == 100
    assert appmod.start_score_league_index(
        2026, 7, {"rec": 1.0}, "QB", 21.09, 18.6,
        player_id="stroud") == 92
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
    # League ranks 5 / 11 / 14 / 17 (guest: 16). Since 2026-11-06 flex
    # players map through the combined flex pool; this fixture's pool
    # holds only RBs, so the combined pool is its 17 RB values and the
    # bands scale to 17. The old per-position mapping gave
    # 96 / 89 / 84 / 79 (guest 80) for these same rows, and the older
    # roster-local mapping gave 100 / 99 / 98 / 97.
    assert rows["rb1"]["start_score_pct"] == \
        appmod.start_score_flex_index(5, 17) == 90
    assert rows["rb2"]["start_score_pct"] == \
        appmod.start_score_flex_index(11, 17) == 69
    assert rows["rb3"]["start_score_pct"] == \
        appmod.start_score_flex_index(14, 17) == 40
    assert rows["rb4"]["start_score_pct"] == \
        appmod.start_score_flex_index(17, 17) == 1
    # Guests get the same league-wide index as roster players (league
    # rank 16), not a different formula.
    assert rows["rbF"]["start_score_pct"] == \
        appmod.start_score_flex_index(16, 17) == 20
    qb = {p["player_id"]: p for p in d["positions"]["QB"]}
    assert qb["qb1"]["start_score_pct"] == \
        appmod.start_score_tier_index("QB", 4) == 94


# ── Combined flex pool (2026-11-06) ───────────────────────────────────────
# RB/WR/TE compete for the same flex slots, so their index ranks ONE
# combined pool through the tier-band curve shape scaled to the pool
# size. Before this, one identical raw score printed 79 at RB, 52 at
# WR and 81 at TE — three numbers that could not be compared at all.

def test_flex_index_matches_tier_table_at_reference_depth():
    import app as appmod

    # At the reference depth (36) the scaled bands ARE the RB/WR table.
    for rank in range(1, 37):
        assert appmod.start_score_flex_index(rank, 36) == \
            appmod.start_score_tier_index("RB", rank)
    # Rank beyond the pool bottoms out; bad inputs get no index.
    assert appmod.start_score_flex_index(37, 36) == 1
    assert appmod.start_score_flex_index(0, 36) is None
    assert appmod.start_score_flex_index(1, 0) is None


def _flex_pool(monkeypatch, by_pos, by_pid):
    import app as appmod
    pool = {"kind": "score", "by_pos": by_pos, "by_pid": by_pid}
    monkeypatch.setattr(appmod, "start_score_pool_scores",
                        lambda *a, **k: pool)
    return appmod


def test_league_index_same_value_same_index_across_flex_positions(
        monkeypatch):
    # One raw score (15.0) held by an RB, a WR and a TE. Their
    # within-position ranks are 3 / 2 / 1 — the old mapping printed
    # 98 / 99 / 100 for them — but the combined rank is 4 for all
    # three, so all three print the same index.
    by_pos = {"RB": [30.0, 20.0, 15.0, 10.0],
              "WR": [28.0, 15.0, 12.0],
              "TE": [15.0, 9.0],
              "QB": [35.0, 22.0]}
    by_pid = {"rbS": 15.0, "wrS": 15.0, "teS": 15.0,
              "qbTop": 35.0, "qb2": 22.0}
    appmod = _flex_pool(monkeypatch, by_pos, by_pid)

    for pos, pid in (("RB", "rbS"), ("WR", "wrS"), ("TE", "teS")):
        assert appmod.start_score_league_index(
            2026, 9, {}, pos, 0.0, 0.0, player_id=pid) == \
            appmod.start_score_flex_index(4, 9) == 83
    # QB neither joins the flex pool (its 35.0 must not shift the flex
    # ranks above) nor is ranked by it: QB still maps within-position.
    assert appmod.start_score_league_index(
        2026, 9, {}, "QB", 0.0, 0.0, player_id="qbTop") == \
        appmod.start_score_tier_index("QB", 1) == 100
    assert appmod.start_score_league_index(
        2026, 9, {}, "QB", 0.0, 0.0, player_id="qb2") == \
        appmod.start_score_tier_index("QB", 2) == 98


def test_league_index_week9_shaped_pool_checkpoints(monkeypatch):
    """The verified Week 9 checkpoint set: a 90-player combined flex
    pool (36 RB / 36 WR / 18 TE shape) must print Gibbs 100,
    Smith-Njigba 99, Bowers 92, Rice 92, Cook 92 (distinct values,
    one printed tie), Moore 72, Downs 45, Charbonnet 28, Singletary 1.
    """
    checkpoints = {
        "gibbs": ("RB", 28.47), "jsn": ("WR", 26.16),
        "bowers": ("TE", 18.44), "rice": ("WR", 18.42),
        "cook": ("RB", 18.27), "moore": ("WR", 12.99),
        "downs": ("WR", 11.10), "charb": ("RB", 10.12),
        "sing": ("RB", 7.86),
    }
    expected = {"gibbs": 100, "jsn": 99, "bowers": 92, "rice": 92,
                "cook": 92, "moore": 72, "downs": 45, "charb": 28,
                "sing": 1}
    # Filler values strictly between checkpoint values, sized so the
    # combined ranks land on 1 / 3 / 19 / 20 / 21 / 52 / 72 / 80 / 90.
    spans = [(28.47, 26.16, 1), (26.16, 18.44, 15), (18.27, 12.99, 30),
             (12.99, 11.10, 19), (11.10, 10.12, 7), (10.12, 7.86, 9)]
    players = dict(checkpoints)
    i = 0
    for hi, lo, n in spans:
        for k in range(1, n + 1):
            players[f"f{i}"] = (("RB", "WR", "TE")[i % 3],
                                hi - (hi - lo) * k / (n + 1))
            i += 1
    assert len(players) == 90
    by_pos, by_pid = {}, {}
    for pid, (pos, value) in players.items():
        by_pos.setdefault(pos, []).append(value)
        by_pid[pid] = value
    for vals in by_pos.values():
        vals.sort(reverse=True)
    appmod = _flex_pool(monkeypatch, by_pos, by_pid)

    for pid, want in expected.items():
        pos, value = checkpoints[pid]
        assert appmod.start_score_league_index(
            2026, 9, {}, pos, 0.0, 0.0, player_id=pid) == want, pid
        # The ad-hoc float path (no player_id) agrees.
        assert appmod.start_score_league_index(
            2026, 9, {}, pos, value, value) == want, pid


def test_league_index_below_board_depth_flex_player_gets_index(
        monkeypatch):
    """Addison shape: a WR ranked 38th at his own position — past the
    old 36-deep table, where the tier map had already fallen off its
    cliff to a flat 1 — still earns a real interpolated index from
    the combined pool instead of the degenerate cliff value."""
    wr = [30.0 - 0.4 * i for i in range(40)]   # target: index 37 -> 15.2
    rb = [29.0 - 0.4 * i for i in range(40)]
    te = [25.0 - 0.5 * i for i in range(20)]
    by_pos = {"RB": rb, "WR": wr, "TE": te}
    by_pid = {"wrDeep": wr[37]}
    appmod = _flex_pool(monkeypatch, by_pos, by_pid)

    # 37 WR + 35 RB + 20 TE values sit above 15.2: combined rank 93
    # of 100, inside the scaled SIT band — interpolated, not the cliff.
    assert appmod.start_score_league_index(
        2026, 9, {}, "WR", 0.0, 0.0, player_id="wrDeep") == \
        appmod.start_score_flex_index(93, 100) == 18
