"""Projection sanity cap (Kaedon, 2026-11-20): before start/sit
scoring, a projection far above what a player's recent workload was
worth is capped against the site's expected-points model. exp4 is the
player's expected PPG over his last 4 PLAYED games; when
exp4 < 0.70 x projection, the projection the rating scores on becomes
exp4 / 0.70. Displayed projection lines stay raw everywhere -- the cap
is scoring-internal, the same rule as the handcuff lead fix
(#3191/#3194).

Why: Rashee Rice's Week 11 pool rank (WR8) was built on a 15.3
projection while his snap-limited workload since returning was worth
8.6 expected PPG -- the projection had never gotten the news. The cap
takes him to ~12.2 and WR18. Backtest over Weeks 6-10 as a fixed
formula: Spearman 0.3616 vs the current model's 0.3469, tier hits
0.7292 vs 0.7167, touching 7.5% of player-weeks.

Layers covered:
- the pure helper (utils.start_sit.cap_projection): fires under the
  threshold with the exact capped value, stands at/above it, and
  leaves the projection untouched without evidence;
- the derivation (data_building.start_score_bundle
  .derive_start_sit_inputs): exp4 over the last 4 played weeks from
  the weekly expected-points table, a played week with no expected
  row counting 0.0, no value at all when the window carries no
  expected rows, the 4-point pass-TD guard, and the TE-premium
  rescale. The window minimum matches the backtest's feature build:
  exp4 exists with as few as ONE played game (the mean over however
  many played games exist, up to 4) -- there is no 2-game floor;
- the ranked paths (app.build_start_sit_pool_export +
  app.start_score_pool_scores): the scorer receives the capped
  projection on both paths while the exported "proj" display field
  stays the raw bundle value.
"""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from utils.start_sit import PROJ_CAP_RATIO, cap_projection


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def test_cap_fires_under_threshold_with_exact_value():
    # The Rice case: 15.3 projected, 8.6 expected on his last-4
    # workload -> capped to exp / 0.70.
    assert cap_projection(15.3, 8.6) == pytest.approx(8.6 / 0.70)
    assert cap_projection(15.3, 8.6) == pytest.approx(12.29, abs=0.01)


def test_cap_stands_at_and_above_threshold():
    # Exactly at the ratio: not below it, so the projection stands.
    assert cap_projection(10.0, 7.0) == 10.0
    assert cap_projection(10.0, 7.01) == 10.0
    assert cap_projection(10.0, 12.0) == 10.0


def test_cap_untouched_without_evidence_or_projection():
    assert cap_projection(15.3, None) == 15.3
    assert cap_projection(0.0, 0.0) == 0.0
    assert cap_projection(None, 5.0) is None
    assert cap_projection("junk", 5.0) == "junk"


def test_cap_zero_expected_caps_to_zero():
    # Played games with expected rows that sum to nothing: the
    # workload supports no projection at all.
    assert cap_projection(9.0, 0.0) == 0.0


def test_cap_ratio_constant():
    assert PROJ_CAP_RATIO == 0.70


# ---------------------------------------------------------------------------
# Expected-points column selection
# ---------------------------------------------------------------------------

def test_expected_column_follows_reception_rate():
    from data_building.start_score_bundle import (
        _expected_column_for_scoring)
    assert _expected_column_for_scoring({"rec": 1.0}) == "expected_ppr"
    assert _expected_column_for_scoring({"rec": 0.5}) == (
        "expected_half_ppr")
    assert _expected_column_for_scoring({"rec": 0.0}) == (
        "expected_standard")
    # Missing keys default to the validated configuration (full PPR,
    # 4-point passing TDs).
    assert _expected_column_for_scoring({}) == "expected_ppr"
    assert _expected_column_for_scoring(None) == "expected_ppr"


def test_expected_column_rejects_six_point_pass_td():
    from data_building.start_score_bundle import (
        _expected_column_for_scoring)
    # The stored expectation prices passing TDs at 4; under a 6-point
    # rate it is not like-for-like with the projection, so no cap.
    assert _expected_column_for_scoring(
        {"rec": 1.0, "pass_td": 6.0}) is None


# ---------------------------------------------------------------------------
# Derivation: exp4 from the weekly expected-points table
# ---------------------------------------------------------------------------

def _write_stat_file(tmp_path, season, week, rows):
    p = tmp_path / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _wr_line(rec=5):
    return {"position": "WR", "gp": 1, "off_snp": 50,
            "rec": rec, "rec_yd": 60, "rec_tgt": 7}


def _patch_derive(monkeypatch, ssb, tmp_path, weeks, index, exp_map,
                  seen_columns=None):
    monkeypatch.setattr(
        ssb, "_stat_files",
        lambda s: [str(tmp_path / f"sleeper_stats_s{s}_w{w}.json")
                   for w in weeks])
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)

    def _fake_expected(season, column):
        if seen_columns is not None:
            seen_columns.append(column)
        return exp_map

    monkeypatch.setattr(ssb, "_load_weekly_expected_map", _fake_expected)
    ssb._DERIVE_CACHE.clear()


def test_derive_exp4_last4_played_weeks(tmp_path, monkeypatch):
    from data_building import start_score_bundle as ssb

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"])
    index = {
        "100": {"pos": "WR", "team": "KC"},
        "101": {"pos": "WR", "team": "KC"},
        "102": {"pos": "RB", "team": "DET"},
        "103": {"pos": "QB", "team": "KC"},
        "104": {"pos": "WR", "team": "DET"},
    }
    for w in range(1, 6):
        rows = {
            "100": _wr_line(),
            "101": _wr_line(),
            "103": {"position": "QB", "gp": 1, "off_snp": 60,
                    "pass_yd": 250, "pass_td": 2},
            "104": _wr_line(),
        }
        if w == 5:
            rows["102"] = {"position": "RB", "gp": 1, "off_snp": 30,
                           "rush_att": 10, "rush_yd": 50,
                           "rec_tgt": 2, "rec": 1, "rec_yd": 5}
        _write_stat_file(tmp_path, season, w, rows)
    exp_map = {
        # Window is the last 4 PLAYED weeks (2-5); week 5 has no
        # expected row and counts 0.0: (10 + 12 + 14 + 0) / 4.
        "100": {2: 10.0, 3: 12.0, 4: 14.0},
        "101": {1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0, 5: 20.0},
        # One played game only: exp4 is that game (the backtest's
        # minimum -- no 2-game floor).
        "102": {5: 7.0},
        # QBs derive it too: the cap applies to every position.
        "103": {2: 18.0, 3: 18.0, 4: 18.0, 5: 18.0},
        # "104" has no expected rows at all: no evidence, no value.
    }
    seen_columns = []
    _patch_derive(monkeypatch, ssb, tmp_path, range(1, 6), index,
                  exp_map, seen_columns)
    out = ssb.derive_start_sit_inputs(season, scoring)
    ssb._DERIVE_CACHE.clear()

    assert seen_columns == ["expected_ppr"]
    assert out["100"]["exp_last4_ppg"] == pytest.approx(9.0)
    assert out["101"]["exp_last4_ppg"] == pytest.approx(20.0)
    assert out["102"]["exp_last4_ppg"] == pytest.approx(7.0)
    assert out["103"]["exp_last4_ppg"] == pytest.approx(18.0)
    assert "exp_last4_ppg" not in out["104"]


def test_derive_exp4_skipped_for_six_point_pass_td(tmp_path, monkeypatch):
    from data_building import start_score_bundle as ssb

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"], pass_td=6.0)
    index = {"100": {"pos": "WR", "team": "KC"}}
    for w in range(1, 6):
        _write_stat_file(tmp_path, season, w, {"100": _wr_line()})
    _patch_derive(monkeypatch, ssb, tmp_path, range(1, 6), index,
                  {"100": {2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0}})
    out = ssb.derive_start_sit_inputs(season, scoring)
    ssb._DERIVE_CACHE.clear()
    assert "exp_last4_ppg" not in out["100"]


def test_derive_exp4_te_premium_rescale(tmp_path, monkeypatch):
    from data_building import start_score_bundle as ssb

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"], bonus_rec_te=0.5)
    index = {
        "105": {"pos": "TE", "team": "GB"},
        "106": {"pos": "WR", "team": "GB"},
    }
    for w in range(1, 5):
        _write_stat_file(tmp_path, season, w, {
            "105": {"position": "TE", "gp": 1, "off_snp": 45,
                    "rec": 4, "rec_yd": 40, "rec_tgt": 5},
            "106": _wr_line(rec=4),
        })
    exp_map = {
        "105": {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
        "106": {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
    }
    _patch_derive(monkeypatch, ssb, tmp_path, range(1, 5), index, exp_map)
    out = ssb.derive_start_sit_inputs(season, scoring)
    ssb._DERIVE_CACHE.clear()
    # TE: 10.0 expected + 0.5 premium x 4.0 receptions/game = 12.0, so
    # the expectation is like-for-like with a TEP projection.
    assert out["105"]["exp_last4_ppg"] == pytest.approx(12.0)
    # The premium never leaks onto other positions.
    assert out["106"]["exp_last4_ppg"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Ranked paths: export (fake scorer) + pool parity (real scorer)
# ---------------------------------------------------------------------------

FAKE_INDEX = {
    "1": {"full_name": "Steady Receiver", "team": "KC"},
    "2": {"full_name": "Comeback Receiver", "team": "KC"},
    "3": {"full_name": "Unrated Receiver", "team": "KC"},
}


def _bundle(pos, proj, **over):
    b = {
        "pos": pos, "proj_pts": proj, "on_bye": False,
        "recent_ppg": 10.0, "season_ppg": 9.0, "last4_ppg": None,
        "last4_ppg_lead": None, "lead_pid": None, "played_games": None,
        "season_ppg_lead": None, "recent_ppg_lead": None,
        "usage_delta_lead": None, "snap_trend_lead": None,
        "exp_last4_ppg": None,
        "opponent": None, "usage_delta": None, "usage_season_avg": None,
        "injury_status": "", "implied_total": None, "bust_rate": None,
        "weather_kind": None, "oline_index": None,
        "expected_team_plays": None, "league_average_plays": None,
        "role_confidence": None, "snap_trend": None,
        "low_total_team": False,
    }
    b.update(over)
    return b


def _fake_compute_start_score(
    proj_pts, *, on_bye=False, recent_ppg=0.0, season_ppg=0.0,
    last4_ppg=None, def_rank=None, def_total=32, usage_delta=None,
    usage_season_avg=None, injury_status=None, played_games=None,
    implied_total=None, bust_rate=None, weather_kind=None, position=None,
    oline_index=None, expected_team_plays=None, league_average_plays=None,
    role_confidence=None, snap_trend=None, low_total_team=False,
):
    _fake_compute_start_score.seen.append(
        {"proj_pts": proj_pts, "last4_ppg": last4_ppg})
    return (float(proj_pts) * 2.0, {}, None)


_fake_compute_start_score.seen = []


@pytest.fixture()
def export_env(monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod
    import utils.start_sit_score as score_mod

    state = {"bundles": {}, "live_players": {}}
    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda scoring: "ppr")
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles",
        lambda season, week, variant, pids: dict(state["bundles"]))
    monkeypatch.setattr(score_mod, "compute_start_score",
                        _fake_compute_start_score)
    _fake_compute_start_score.seen = []
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(FAKE_INDEX))
    monkeypatch.setattr(appmod, "get_players_global",
                        lambda: dict(state["live_players"]))
    monkeypatch.setattr(
        appmod, "_matchup_rank_table",
        lambda season, position, scoring_settings=None: (
            {"DAL": 32, "NE": 20, "SF": 4}, 32, {}, False))
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    yield state
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()


def _cap_bundles():
    return {
        # Expected 16.0 >= 0.70 x 20.0: projection stands.
        "1": _bundle("WR", 20.0, exp_last4_ppg=16.0, played_games=9),
        # The Rice shape: 8.6 expected vs a 15.3 projection -> capped.
        "2": _bundle("WR", 15.3, exp_last4_ppg=8.6, played_games=7),
        # No expected evidence derived: projection stands.
        "3": _bundle("WR", 12.0, exp_last4_ppg=None, played_games=9),
    }


def test_export_scores_capped_projection_display_stays_raw(export_env):
    import app as appmod

    export_env["bundles"] = _cap_bundles()
    out = appmod.build_start_sit_pool_export(2026, 11)
    rows = {r["pid"]: r for r in out["WR"]}
    seen = sorted(s["proj_pts"] for s in _fake_compute_start_score.seen)
    # The scorer saw the capped projection for pid 2 and the raw
    # projections for pids 1 and 3.
    assert seen == pytest.approx([12.0, 8.6 / 0.70, 20.0])
    assert rows["2"]["value"] == pytest.approx(2.0 * (8.6 / 0.70), abs=0.01)
    # The displayed projection line stays the raw bundle value.
    assert rows["2"]["proj"] == 15.3
    assert rows["1"]["proj"] == 20.0


PARITY_INDEX = {
    "1": {"full_name": "Steady Receiver", "team": "KC",
          "injury_status": ""},
    "2": {"full_name": "Comeback Receiver", "team": "KC",
          "injury_status": ""},
}

PARITY_BUNDLES = {
    "1": _bundle("WR", 20.0, recent_ppg=19.0, season_ppg=18.0,
                 last4_ppg=19.5, exp_last4_ppg=16.0, played_games=9,
                 opponent="DAL"),
    "2": _bundle("WR", 15.3, recent_ppg=11.0, season_ppg=11.3,
                 last4_ppg=13.8, exp_last4_ppg=8.6, played_games=7,
                 opponent="DAL"),
}


@pytest.fixture()
def parity_env(monkeypatch):
    import app as appmod
    import data_building.start_score_bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "variant_for_exact_scoring",
                        lambda scoring: "ppr")
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles",
        lambda season, week, variant, pids: dict(PARITY_BUNDLES))
    monkeypatch.setattr(
        bundle_mod, "load_start_score_bundles_for_scoring",
        lambda season, week, scoring, pids: dict(PARITY_BUNDLES))
    monkeypatch.setattr(appmod, "get_players_index_global",
                        lambda: dict(PARITY_INDEX))
    monkeypatch.setattr(appmod, "get_players_global", lambda: {})
    monkeypatch.setattr(
        appmod, "_matchup_rank_table",
        lambda season, position, scoring_settings=None: (
            {"DAL": 32, "NE": 20, "SF": 4}, 32, {}, False))
    monkeypatch.setattr(appmod, "_SS_POOL_MIN_BUNDLES", 1)
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()
    yield
    appmod._SS_EXPORT_CACHE.clear()
    appmod._SS_EXPORT_CACHE_TS.clear()
    appmod._SS_POOL_CACHE.clear()
    appmod._SS_POOL_CACHE_TS.clear()


def test_pool_and_export_agree_on_capped_projection(parity_env):
    import app as appmod
    from utils.start_sit_score import compute_start_score

    export = appmod.build_start_sit_pool_export(2026, 11)
    pool = appmod.start_score_pool_scores(2026, 11, {"rec": 1.0})
    assert pool["kind"] == "score"
    exported = {r["pid"]: r["value"]
                for rows in export.values() for r in rows}
    assert exported["2"] == round(pool["by_pid"]["2"], 2)

    # The agreed value is the REAL scorer's output on the capped
    # projection -- and it is not the raw-projection value.
    common = dict(on_bye=False, recent_ppg=11.0, season_ppg=11.3,
                  def_rank=32, def_total=32, usage_delta=None,
                  usage_season_avg=None, injury_status="",
                  played_games=7, implied_total=None, bust_rate=None,
                  weather_kind=None, position="WR", oline_index=None,
                  expected_team_plays=None, league_average_plays=None,
                  role_confidence=None, snap_trend=None,
                  low_total_team=False)
    capped, _, _ = compute_start_score(8.6 / 0.70, last4_ppg=13.8,
                                       **common)
    raw, _, _ = compute_start_score(15.3, last4_ppg=13.8, **common)
    assert exported["2"] == round(capped, 2)
    assert exported["2"] != round(raw, 2)
    # The uncapped player's value is the raw-projection computation.
    lead_common = dict(common, recent_ppg=19.0, season_ppg=18.0,
                       played_games=9)
    lead_val, _, _ = compute_start_score(20.0, last4_ppg=19.5,
                                         **lead_common)
    assert exported["1"] == round(lead_val, 2)
