"""Tests for the daily precomputed start/sit score bundles.

Covers: the exact-scoring gate (variant_for_exact_scoring), the batch build
with stubbed data sources, the request-time read path, and the key equivalence
property — a score computed from bundle inputs must equal the score from the
live-gathered inputs the endpoint used before.
"""
import json

import pytest

from data_building import start_score_bundle as ssb
from utils.proj_variant import pick_proj_variant
from utils.start_sit_score import compute_start_score


# ---------------------------------------------------------------------------
# Exact-scoring gate
# ---------------------------------------------------------------------------

def test_variant_round_trip():
    for variant, scoring in ssb.VARIANT_SCORING.items():
        assert pick_proj_variant(scoring) == variant


@pytest.mark.parametrize("settings,want", [
    ({"rec": 1.0, "pass_td": 4}, "ppr"),
    ({"rec": 0.5}, "half_ppr"),
    ({"rec": 0.0, "pass_td": 4}, "std"),
    ({"rec": 1.0, "bonus_rec_te": 0.5}, "tep"),
    ({"rec": 1.0, "pass_td": 6}, "6pt_ppr"),
    ({"rec": 0.5, "pass_td": 6}, "6pt_half"),
    ({"rec": 1.0, "pass_td": 6, "bonus_rec_te": 0.5}, "6pt_tep"),
    # Exotic leagues must NOT use the fast path.
    ({"rec": 0.75}, None),                       # fractional PPR
    ({"rec": 1.0, "bonus_rec_te": 1.0}, None),   # non-canonical TE premium
    ({"rec": 1.0, "rec_yd": 0.15}, None),        # custom yardage rate
    ({"rec": 1.0, "bonus_rec_rb": 1}, None),     # exotic bonus
    ({"rec": 1.0, "pass_td": 5}, None),          # non-4/6 pass TD
    ({}, None),                                  # missing rec is ambiguous
    ({"rec": 1.0, "pass_td": 4, "rec_fd": 0.5}, None),  # first-down points
])
def test_variant_for_exact_scoring(settings, want):
    assert ssb.variant_for_exact_scoring(settings) == want


# ---------------------------------------------------------------------------
# Weekly scoring pass (fixture files)
# ---------------------------------------------------------------------------

def _write_stat_file(tmp_path, season, week, rows):
    p = tmp_path / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _line(**kw):
    base = {"position": "RB", "rush_yd": 50, "rush_td": 0, "rec": 2,
            "rec_yd": 15, "fum_lost": 0}
    base.update(kw)
    return base


def test_variant_week_points_matches_live_scorers(tmp_path, monkeypatch):
    from utils.fantasy_scoring import week_stat_points, score_stats
    from utils.season_qualification import qualification_policy

    season = 2026
    _write_stat_file(tmp_path, season, 1, {"1": _line(rush_yd=100, rush_td=1)})
    _write_stat_file(tmp_path, season, 2, {"1": _line(rush_yd=20)})
    _write_stat_file(tmp_path, season, 3, {"1": _line(rush_yd=0, rec=0, rec_yd=0)})
    monkeypatch.setattr(ssb, "_stat_files",
                        lambda s: [str(tmp_path / f"sleeper_stats_s{s}_w{w}.json")
                                   for w in (1, 2, 3)])
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        lambda s: type("Q", (), {"completed_weeks": (1, 2, 3)})())

    scoring = dict(ssb.VARIANT_SCORING["half_ppr"])
    weekstat, weekly = ssb._variant_week_points(season, scoring)

    # weekstat: >0 filter, week_stat_points scorer, no TE flag.
    exp1 = week_stat_points(_line(rush_yd=100, rush_td=1), scoring, "")
    exp2 = week_stat_points(_line(rush_yd=20), scoring, "")
    exp3 = week_stat_points(_line(rush_yd=0, rec=0, rec_yd=0), scoring, "")
    rec = weekstat["1"]
    assert rec["n"] == sum(1 for e in (exp1, exp2, exp3) if e > 0)
    assert rec["sum"] == pytest.approx(sum(e for e in (exp1, exp2, exp3) if e > 0))
    # weekly: every stat line counts (no >0 filter), score_stats with position.
    assert weekly["1"] == pytest.approx([
        score_stats(_line(rush_yd=100, rush_td=1), scoring, "RB"),
        score_stats(_line(rush_yd=20), scoring, "RB"),
        score_stats(_line(rush_yd=0, rec=0, rec_yd=0), scoring, "RB"),
    ])

    s_ppg, r_ppg = ssb._season_ppg_recent(weekstat, "1")
    pos_vals = [e for e in (exp1, exp2, exp3) if e > 0]
    assert s_ppg == round(sum(pos_vals) / len(pos_vals), 1)
    assert r_ppg == round(sum(pos_vals) / len(pos_vals), 1)
    assert ssb._season_ppg_recent(weekstat, "nope") == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Fake DB + build
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, rows=None, one=None):
        self._rows = rows or []
        self._one = one or {}

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


class _FakeConn:
    """Captures writes; serves canned reads. Context-manager compatible."""
    def __init__(self, read_rows=None, count=0):
        self.read_rows = read_rows or []
        self.count = count
        self.written = []   # executemany payloads
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if "COUNT(*)" in sql:
            return _FakeResult(one={"n": self.count})
        if sql.strip().upper().startswith("SELECT"):
            return _FakeResult(rows=self.read_rows)
        return _FakeResult()

    def executemany(self, sql, rows):
        self.statements.append(sql)
        self.written.extend(rows)
        return _FakeResult()


def _patch_build_deps(monkeypatch, tmp_path):
    """Stub every external data source the batch touches."""
    monkeypatch.setattr("dashboard_services.api.get_nfl_state",
                        lambda: {"season": 2026, "week": 4})
    monkeypatch.setattr(ssb, "init_start_score_db", lambda: None)
    import utils.utils as uu
    monkeypatch.setattr(uu, "load_players_index", lambda: {
        "1": {"pos": "RB", "team": "DET", "injury_status": "Questionable"},
        "2": {"pos": "WR", "team": "GB", "injury_status": ""},
    })
    monkeypatch.setattr(uu, "load_week_sched", lambda s, w: [
        {"home": "DET", "away": "GB", "gameDate": "2026-09-27"},
    ])
    import data_building.weekly_metrics as wm
    monkeypatch.setattr(wm, "get_usage_trends", lambda season: {
        "1": {"delta": 2.5, "season_avg": 18.0},
    })
    import utils.game_conditions as gc
    monkeypatch.setattr(gc, "build_week_conditions",
                        lambda s, w, games: {
                            "DET": {"implied_total": 24.5,
                                    "weather": {"kind": "wind"}},
                            "GB": {"implied_total": 21.0, "weather": {}},
                        })
    import dashboard_services.team_play_volume as tpv
    monkeypatch.setattr(tpv, "load_team_play_volume", lambda season: {
        "teams": {"DET": {}, "GB": {}}, "nfl_avg_plays_faced_pg": 63.0})
    monkeypatch.setattr(uu, "load_week_projection", lambda s, w: {
        "1": {"raw_stats": {"rush_yd": 80}, "ppr": 14.2, "half_ppr": 13.1,
              "std": 12.0, "tep": 14.2, "6pt_ppr": 14.2, "6pt_half": 13.1,
              "6pt_tep": 14.2},
    })
    import data_building.fetch_projections as fp
    monkeypatch.setattr(fp, "fetch_sleeper_season_ppg_variants",
                        lambda season, players_index=None: {})
    import utils.start_sit_context as ssc
    monkeypatch.setattr(ssc, "expected_plays_context",
                        lambda teams, team, opp, avg: {
                            "expected_team_plays": 65.0,
                            "league_average_plays": 63.0})
    monkeypatch.setattr(ssc, "role_confidence_from_trend", lambda ut: 0.8)
    import utils.fantasy_scoring as fs
    monkeypatch.setattr(fs, "weekly_projection_points",
                        lambda entry, scoring, pos="": float(entry.get("ppr") or 0))
    monkeypatch.setattr(ssb, "_variant_week_points",
                        lambda season, scoring: ({"1": {"sum": 40.0, "n": 4,
                                                       "last4": [10.0, 10.0, 10.0, 10.0]}},
                                                {"1": [10.0, 10.0, 10.0, 10.0]}))
    import utils.consistency as cons
    monkeypatch.setattr(cons, "blended_consistency_profile",
                        lambda cur, prior, pos="", **kw: {"bust_rate": 0.25})
    monkeypatch.setattr(ssb, "_load_oline_ratings",
                        lambda season: {"DET": {"run_block": 70.0}})


def test_build_writes_bundles_for_all_variants(tmp_path, monkeypatch):
    _patch_build_deps(monkeypatch, tmp_path)
    conns = []

    def _fake_get_conn(*a, **k):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            c = _FakeConn()
            conns.append(c)
            yield c
        return _cm()

    monkeypatch.setattr("dashboard_services.db.get_conn", _fake_get_conn)

    summary = ssb.build_start_score_bundles()
    assert summary["ok"], summary
    assert summary["players"] == 2
    assert summary["variants"] == 7
    assert summary["rows"] == 14

    writer = conns[-1]
    assert any("PRUNE" in s.upper() or "DELETE FROM start_score_bundle" in s
               for s in writer.statements)
    assert len(writer.written) == 14
    by_key = {(r[2], r[3]): json.loads(r[4]) for r in writer.written}
    ppr = by_key[("1", "ppr")]
    assert ppr["pos"] == "RB"
    assert ppr["team"] == "DET"
    assert ppr["opponent"] == "GB"
    assert ppr["on_bye"] is False
    assert ppr["proj_pts"] == 14.2
    assert ppr["season_ppg"] == 10.0
    assert ppr["recent_ppg"] == 10.0
    assert ppr["bust_rate"] == 0.25
    assert ppr["usage_delta"] == 2.5
    assert ppr["usage_season_avg"] == 18.0
    assert ppr["implied_total"] == 24.5
    assert ppr["weather_kind"] == "wind"
    assert ppr["oline_index"] == 70.0
    assert ppr["expected_team_plays"] == 65.0
    assert ppr["role_confidence"] == 0.8
    # Batch-time injury is stored for reference only.
    assert ppr["injury_status"] == "Questionable"
    # Player 2 has no projection entry and no usage trend: still gets a bundle.
    w2 = by_key[("2", "ppr")]
    assert w2["proj_pts"] == 0.0
    assert w2["usage_delta"] is None


def test_build_never_raises_without_db(monkeypatch):
    monkeypatch.setattr("dashboard_services.api.get_nfl_state",
                        lambda: {"season": 2026, "week": 4})

    def _boom(*a, **k):
        raise RuntimeError("no db")

    monkeypatch.setattr("dashboard_services.db.get_conn", _boom)
    # init is best-effort; build must surface ok=False, not raise.
    summary = ssb.build_start_score_bundles()
    assert summary["ok"] is False


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def test_load_bundles_parses_rows(monkeypatch):
    rows = [
        {"player_id": "1", "bundle": json.dumps({"proj_pts": 14.2})},
        {"player_id": "2", "bundle": {"proj_pts": 9.5}},
    ]
    conns = []

    def _fake_get_conn(*a, **k):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            c = _FakeConn(read_rows=rows)
            conns.append(c)
            yield c
        return _cm()

    monkeypatch.setattr("dashboard_services.db.get_conn", _fake_get_conn)
    out = ssb.load_start_score_bundles(2026, 4, "ppr", ["1", "2"])
    assert out == {"1": {"proj_pts": 14.2}, "2": {"proj_pts": 9.5}}


def test_load_bundles_empty_on_db_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr("dashboard_services.db.get_conn", _boom)
    assert ssb.load_start_score_bundles(2026, 4, "ppr", ["1"]) == {}
    assert ssb.start_score_bundles_ready(2026, 4, "ppr", ["1"]) is False


def test_bundles_ready_counts(monkeypatch):
    conns = []

    def _fake_get_conn(count):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            c = _FakeConn(count=count)
            conns.append(c)
            yield c
        return _cm()

    monkeypatch.setattr("dashboard_services.db.get_conn", lambda *a, **k: _fake_get_conn(2))
    assert ssb.start_score_bundles_ready(2026, 4, "ppr", ["1", "2"]) is True
    monkeypatch.setattr("dashboard_services.db.get_conn", lambda *a, **k: _fake_get_conn(1))
    assert ssb.start_score_bundles_ready(2026, 4, "ppr", ["1", "2"]) is False


# ---------------------------------------------------------------------------
# Equivalence: bundle inputs score identically to live-gathered inputs
# ---------------------------------------------------------------------------

def test_bundle_score_matches_live_score():
    """The request path must produce the same score from bundle inputs as the
    old live path did from the same values (same pure function, same args)."""
    live_inputs = dict(
        proj_pts=14.2, on_bye=False, recent_ppg=11.0, season_ppg=10.5,
        usage_delta=2.5, usage_season_avg=18.0, injury_status="Questionable",
        implied_total=24.5, bust_rate=0.25, weather_kind="wind",
        position="RB", oline_index=70.0,
        expected_team_plays=65.0, league_average_plays=63.0,
        role_confidence=0.8,
    )
    bundle = {
        "proj_pts": 14.2, "recent_ppg": 11.0, "season_ppg": 10.5,
        "usage_delta": 2.5, "usage_season_avg": 18.0,
        "implied_total": 24.5, "bust_rate": 0.25, "weather_kind": "wind",
        "oline_index": 70.0, "expected_team_plays": 65.0,
        "league_average_plays": 63.0, "role_confidence": 0.8,
        # Batch-time injury is ignored: the request overlays the live status.
        "injury_status": "",
    }
    live_score, live_factors, live_dem = compute_start_score(**live_inputs)
    bundle_score, bundle_factors, bundle_dem = compute_start_score(
        bundle["proj_pts"], on_bye=False,
        recent_ppg=bundle["recent_ppg"], season_ppg=bundle["season_ppg"],
        usage_delta=bundle["usage_delta"],
        usage_season_avg=bundle["usage_season_avg"],
        injury_status=live_inputs["injury_status"],  # live overlay
        implied_total=bundle["implied_total"], bust_rate=bundle["bust_rate"],
        weather_kind=bundle["weather_kind"], position="RB",
        oline_index=bundle["oline_index"],
        expected_team_plays=bundle["expected_team_plays"],
        league_average_plays=bundle["league_average_plays"],
        role_confidence=bundle["role_confidence"],
    )
    assert bundle_score == live_score
    assert bundle_factors == live_factors
    assert bundle_dem == live_dem


def test_live_injury_overlay_beats_batch_status():
    """A Q tag reported after the batch ran must still move the score."""
    kwargs = dict(proj_pts=14.2, recent_ppg=11.0, season_ppg=10.5,
                  position="RB")
    healthy, _, _ = compute_start_score(injury_status=None, **kwargs)
    # Batch saw him healthy (injury_status=""), live map now says Q.
    live_q, _, dem = compute_start_score(injury_status="Questionable", **kwargs)
    assert dem == "questionable"
    assert live_q < healthy
