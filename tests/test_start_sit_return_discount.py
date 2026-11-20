"""First-game-back discount (Kaedon, 2026-11-20): before start/sit
scoring, the projection of a player returning from a multi-week
absence is shaded by a ladder on the absence length -- 1 missed week
x0.98, 2-3 x0.95, 4+ x0.90 -- because his output in his first game
back is genuinely uncertain. The absence count is consecutive
non-bye weeks missed immediately before this week: a week his team
did not play never counts (a bye is not an injury absence), and no
count is derived when a bye cannot be ruled out for the gap.
Displayed projection lines stay raw everywhere -- the discount is
scoring-internal, the same rule as the projection cap (#3208) and
the handcuff lead fix (#3191/#3194).

Why: Mike Evans was boarded WR11 in Week 11 on a 13.3 projection and
Weeks 3-6 form after missing Weeks 7-10. At four missed weeks the
scoring projection becomes 11.97 and his Week 11 score moves
15.68 -> ~14.5 (WR11 -> ~WR13). Backtest over Weeks 6-10 as a fixed
formula, applied to the projection ahead of scoring: Spearman
0.3493 vs the current model's 0.3469, tier hits 0.7188 vs 0.7167,
points captured 0.8363 vs 0.8356.

Layers covered:
- the pure helpers (utils.start_sit.return_discount_multiplier /
  return_adjusted_projection): the ladder rungs, the exact Evans
  math, and the untouched cases (no count, no projection, junk);
- the derivation (data_building.start_score_bundle
  .derive_start_sit_inputs): the tail-only count, the bye exclusion
  (a week his team shows no played row never counts), no value
  without a team on the index, and no value for a player whose last
  played game is the newest week;
- the ranked paths (app.build_start_sit_pool_export +
  app.start_score_pool_scores): the scorer receives the discounted
  projection on both paths while the exported "proj" display field
  stays the raw bundle value, and the discount composes AFTER the
  projection cap (cap judges the raw line, discount shades what
  survives).
"""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from utils.start_sit import (
    RETURN_DISCOUNT_LADDER,
    return_adjusted_projection,
    return_discount_multiplier,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_multiplier_ladder_rungs():
    assert return_discount_multiplier(0) == 1.0
    assert return_discount_multiplier(1) == 0.98
    assert return_discount_multiplier(2) == 0.95
    assert return_discount_multiplier(3) == 0.95
    assert return_discount_multiplier(4) == 0.90
    assert return_discount_multiplier(9) == 0.90


def test_multiplier_neutral_without_a_count():
    assert return_discount_multiplier(None) == 1.0
    assert return_discount_multiplier("junk") == 1.0
    assert return_discount_multiplier(-2) == 1.0


def test_ladder_constant():
    assert RETURN_DISCOUNT_LADDER == ((4, 0.90), (2, 0.95), (1, 0.98))


def test_adjusted_projection_evans_case():
    # Out since Week 6, boarded in Week 11: four missed weeks.
    assert return_adjusted_projection(13.3, 4) == pytest.approx(11.97)
    assert return_adjusted_projection(13.3, 1) == pytest.approx(13.3 * 0.98)
    assert return_adjusted_projection(13.3, 3) == pytest.approx(13.3 * 0.95)


def test_adjusted_projection_untouched_cases():
    assert return_adjusted_projection(13.3, 0) == 13.3
    assert return_adjusted_projection(13.3, None) == 13.3
    assert return_adjusted_projection(0.0, 4) == 0.0
    assert return_adjusted_projection(None, 4) is None
    assert return_adjusted_projection("junk", 4) == "junk"


# ---------------------------------------------------------------------------
# Derivation: absence_weeks from the weekly stat files
# ---------------------------------------------------------------------------

def _write_stat_file(tmp_path, season, week, rows):
    p = tmp_path / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _wr_line(rec=5):
    return {"position": "WR", "gp": 1, "off_snp": 50,
            "rec": rec, "rec_yd": 60, "rec_tgt": 7}


def _patch_derive(monkeypatch, ssb, tmp_path, weeks, index):
    monkeypatch.setattr(
        ssb, "_stat_files",
        lambda s: [str(tmp_path / f"sleeper_stats_s{s}_w{w}.json")
                   for w in weeks])
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)
    monkeypatch.setattr(ssb, "_load_weekly_expected_map", lambda s, c: {})
    ssb._DERIVE_CACHE.clear()


def test_derive_absence_weeks_tail_only_bye_excluded(tmp_path, monkeypatch):
    from data_building import start_score_bundle as ssb

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"])
    index = {
        "100": {"pos": "WR", "team": "KC"},
        "101": {"pos": "WR", "team": "KC"},
        "102": {"pos": "WR", "team": "DET"},
        "103": {"pos": "WR", "team": "DET"},
        # No team on the index: a bye cannot be ruled out for his
        # gap, so no count is derived.
        "104": {"pos": "WR", "team": ""},
        "105": {"pos": "WR", "team": "KC"},
        "106": {"pos": "WR", "team": "DET"},
    }
    # KC's bye is Week 3: no KC player has a played row that week.
    rows_by_week = {
        1: {"100": _wr_line(), "101": _wr_line(), "102": _wr_line(),
            "103": _wr_line(), "104": _wr_line(), "106": _wr_line()},
        2: {"100": _wr_line(), "101": _wr_line(),
            "103": _wr_line(), "104": _wr_line()},
        3: {"103": _wr_line(), "106": _wr_line()},
        4: {"101": _wr_line(), "103": _wr_line(), "106": _wr_line()},
        5: {"101": _wr_line(), "103": _wr_line(), "106": _wr_line()},
    }
    for w, rows in rows_by_week.items():
        _write_stat_file(tmp_path, season, w, rows)
    _patch_derive(monkeypatch, ssb, tmp_path, range(1, 6), index)
    out = ssb.derive_start_sit_inputs(season, scoring)
    ssb._DERIVE_CACHE.clear()

    # 100: last played Week 2; missed Week 3 (KC bye -- not counted),
    # Weeks 4 and 5 -> 2.
    assert out["100"]["absence_weeks"] == 2
    # 101: played the newest week -> no count.
    assert "absence_weeks" not in out["101"]
    # 102: last played Week 1, DET played every week -> 4.
    assert out["102"]["absence_weeks"] == 4
    # 103: played every week -> no count.
    assert "absence_weeks" not in out["103"]
    # 104: same gap as 100 but no team on the index -> no count.
    assert "absence_weeks" not in out["104"]
    # 105: never played -> not a return, no entry at all.
    assert "105" not in out
    # 106: missed Week 2 mid-season but has played since -> the
    # count is the tail only, and his tail is empty.
    assert "absence_weeks" not in out["106"]


# ---------------------------------------------------------------------------
# Ranked paths: export (fake scorer) + pool parity (real scorer)
# ---------------------------------------------------------------------------

FAKE_INDEX = {
    "1": {"full_name": "Steady Receiver", "team": "KC"},
    "2": {"full_name": "Comeback Receiver", "team": "KC"},
    "3": {"full_name": "Capped Comeback", "team": "KC"},
}


def _bundle(pos, proj, **over):
    b = {
        "pos": pos, "proj_pts": proj, "on_bye": False,
        "recent_ppg": 10.0, "season_ppg": 9.0, "last4_ppg": None,
        "last4_ppg_lead": None, "lead_pid": None, "played_games": None,
        "season_ppg_lead": None, "recent_ppg_lead": None,
        "usage_delta_lead": None, "snap_trend_lead": None,
        "exp_last4_ppg": None, "absence_weeks": None,
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


def _return_bundles():
    return {
        # No absence count: projection stands.
        "1": _bundle("WR", 20.0, played_games=9),
        # Four missed weeks: 20.0 x 0.90 = 18.0 to the scorer.
        "2": _bundle("WR", 20.0, absence_weeks=4, played_games=5),
        # Cap AND discount: the cap judges the raw 15.3 against the
        # 8.6 expectation (-> 8.6 / 0.70), then the four-week
        # discount shades what survives (x 0.90).
        "3": _bundle("WR", 15.3, exp_last4_ppg=8.6, absence_weeks=4,
                     played_games=7),
    }


def test_export_scores_discounted_projection_display_stays_raw(export_env):
    import app as appmod

    export_env["bundles"] = _return_bundles()
    out = appmod.build_start_sit_pool_export(2026, 11)
    rows = {r["pid"]: r for r in out["WR"]}
    seen = sorted(s["proj_pts"] for s in _fake_compute_start_score.seen)
    assert seen == pytest.approx([8.6 / 0.70 * 0.90, 18.0, 20.0])
    assert rows["2"]["value"] == pytest.approx(2.0 * 18.0)
    # The displayed projection line stays the raw bundle value.
    assert rows["2"]["proj"] == 20.0
    assert rows["3"]["proj"] == 15.3
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
    "2": _bundle("WR", 20.0, recent_ppg=11.0, season_ppg=11.3,
                 last4_ppg=13.8, exp_last4_ppg=16.0, absence_weeks=4,
                 played_games=5, opponent="DAL"),
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


def test_pool_and_export_agree_on_discounted_projection(parity_env):
    import app as appmod
    from utils.start_sit_score import compute_start_score

    export = appmod.build_start_sit_pool_export(2026, 11)
    pool = appmod.start_score_pool_scores(2026, 11, {"rec": 1.0})
    assert pool["kind"] == "score"
    exported = {r["pid"]: r["value"]
                for rows in export.values() for r in rows}
    assert exported["2"] == round(pool["by_pid"]["2"], 2)

    # The agreed value is the REAL scorer's output on the discounted
    # projection (20.0 x 0.90 = 18.0) -- not the raw-projection value.
    common = dict(on_bye=False, recent_ppg=11.0, season_ppg=11.3,
                  def_rank=32, def_total=32, usage_delta=None,
                  usage_season_avg=None, injury_status="",
                  played_games=5, implied_total=None, bust_rate=None,
                  weather_kind=None, position="WR", oline_index=None,
                  expected_team_plays=None, league_average_plays=None,
                  role_confidence=None, snap_trend=None,
                  low_total_team=False)
    discounted, _, _ = compute_start_score(18.0, last4_ppg=13.8,
                                           **common)
    raw, _, _ = compute_start_score(20.0, last4_ppg=13.8, **common)
    assert exported["2"] == round(discounted, 2)
    assert exported["2"] != round(raw, 2)
    # The non-returning player's value is the raw-projection
    # computation.
    lead_common = dict(common, recent_ppg=19.0, season_ppg=18.0,
                       played_games=9)
    lead_val, _, _ = compute_start_score(20.0, last4_ppg=19.5,
                                         **lead_common)
    assert exported["1"] == round(lead_val, 2)
