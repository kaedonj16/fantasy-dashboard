"""Handcuff form filter (Kaedon, 2026-11-19): when the lead at a
player's position on his team is ACTIVE for the current week, the
start/sit form window (last-4-played PPG fed to compute_start_score)
counts only games that lead also played -- a backup's rank must not
be built on the starter's absence (Emmett Johnson's 35.1 came in his
only start, with Kenneth Walker III out, yet his raw last-4 of 18.4
parked him at RB17 in the world where Walker plays).

Layers covered:
- the pure helpers (utils.start_sit): the shared-games window and its
  fallbacks, and the active/out/bye decision;
- the derivation (data_building.start_score_bundle
  .derive_start_sit_inputs): room-lead identification by season usage
  per played game, RB/WR/TE only, and the filtered value it derives;
- the ranked paths (app.build_start_sit_pool_export +
  app.start_score_pool_scores): the scorer receives the filtered
  form when the lead is active and the raw form when he is out or on
  bye, identically on both paths. The export assertions fail on
  pre-fix code, where the export always passed the raw last-4.
"""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from utils.start_sit import lead_adjusted_last4_ppg, lead_filtered_last4_ppg


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_filtered_window_counts_only_shared_games():
    # Player's played games W1..W6; the lead played W1..W5 only (the
    # W6 spike is the injury-replacement game and must not count).
    pts = [8.0, 5.0, 2.0, 4.0, 9.0, 35.0]
    weeks = [1, 2, 3, 4, 5, 6]
    # Shared = W1..W5; last 4 shared = W2..W5.
    assert lead_filtered_last4_ppg(pts, weeks, {1, 2, 3, 4, 5}) == (
        pytest.approx((5.0 + 2.0 + 4.0 + 9.0) / 4))


def test_filtered_window_uses_what_remains_below_two_games():
    # Exactly one shared game: use it, never zero the player out.
    assert lead_filtered_last4_ppg([10.0, 20.0, 30.0], [1, 2, 3], {2}) == (
        pytest.approx(20.0))


def test_filtered_window_no_shared_games_falls_back_to_raw():
    # Never on the field together in the sample: unfiltered last-4.
    assert lead_filtered_last4_ppg(
        [1.0, 2.0, 3.0, 4.0, 5.0], [1, 2, 3, 4, 5], {9}) == (
        pytest.approx((2.0 + 3.0 + 4.0 + 5.0) / 4))


def test_filtered_window_no_games_is_zero():
    assert lead_filtered_last4_ppg([], [], {1, 2}) == 0.0


def test_adjusted_swaps_only_when_lead_active():
    raw, filtered, lead = 18.4, 12.8, "8151"
    # Lead active (healthy or merely Questionable): filtered form.
    assert lead_adjusted_last4_ppg(raw, filtered, lead) == filtered
    assert lead_adjusted_last4_ppg(
        raw, filtered, lead, lead_status="Questionable") == filtered
    # Lead out / excluded or on bye: the replacement world is real.
    for status in ("IR", "Out", "PUP", "NFI"):
        assert lead_adjusted_last4_ppg(
            raw, filtered, lead, lead_status=status) == raw
    assert lead_adjusted_last4_ppg(
        raw, filtered, lead, lead_on_bye=True) == raw
    # No lead identified / nothing derived: untouched.
    assert lead_adjusted_last4_ppg(raw, filtered, None) == raw
    assert lead_adjusted_last4_ppg(raw, None, lead) == raw


# ---------------------------------------------------------------------------
# Derivation: lead identification + filtered value
# ---------------------------------------------------------------------------

def _write_stat_file(tmp_path, season, week, rows):
    p = tmp_path / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _rb_line(yd, att, tgt):
    # Points come from the rushing yards under PPR scoring; pts_ppr is
    # the played-week marker the derivation keys on.
    return {"position": "RB", "pts_ppr": yd / 10.0, "rush_att": att,
            "rec_tgt": tgt, "rush_yd": yd, "rec": 0, "rec_yd": 0,
            "fum_lost": 0}


def test_derive_identifies_lead_and_filters_handcuff(tmp_path, monkeypatch):
    from data_building import start_score_bundle as ssb
    from utils.fantasy_scoring import week_stat_points

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"])
    lead, cuff = "900", "901"          # KC RB room
    qb1, qb2 = "910", "911"            # KC QBs: never filtered
    solo = "930"                       # DET RB, alone in his room
    rows = {}
    lead_yd = {1: 200, 2: 210, 3: 190, 4: 220, 5: 205}   # sits week 6
    cuff_yd = {1: 40, 2: 30, 3: 50, 4: 20, 5: 45, 6: 350}  # W6 spike
    for w in range(1, 7):
        rows[w] = {
            cuff: _rb_line(cuff_yd[w], 3, 2),
            qb1: {"position": "QB", "pts_ppr": 20.0, "pass_yd": 250,
                  "pass_td": 2},
            qb2: {"position": "QB", "pts_ppr": 5.0, "pass_yd": 60,
                  "pass_td": 0},
            solo: _rb_line(150, 15, 3),
        }
        if w in lead_yd:
            rows[w][lead] = _rb_line(lead_yd[w], 20, 5)
    for w, payload in rows.items():
        _write_stat_file(tmp_path, season, w, payload)
    monkeypatch.setattr(ssb, "_stat_files",
                        lambda s: [str(tmp_path / f"sleeper_stats_s{s}_w{w}.json")
                                   for w in range(1, 7)])
    monkeypatch.setattr("utils.utils.load_players_index", lambda: {
        lead: {"pos": "RB", "team": "KC"},
        cuff: {"pos": "RB", "team": "KC"},
        qb1: {"pos": "QB", "team": "KC"},
        qb2: {"pos": "QB", "team": "KC"},
        solo: {"pos": "RB", "team": "DET"},
    })
    ssb._DERIVE_CACHE.clear()
    out = ssb.derive_start_sit_inputs(season, scoring)
    ssb._DERIVE_CACHE.clear()

    scored = {w: week_stat_points(rows[w][cuff], scoring, "")
              for w in range(1, 7)}
    entry = out[cuff]
    # Raw last-4 includes the week-6 spike (weeks 3,4,5,6)...
    assert entry["last4_ppg"] == pytest.approx(
        sum(scored[w] for w in (3, 4, 5, 6)) / 4)
    # ...the filtered value counts only weeks the lead played (last 4
    # of weeks 1-5 = weeks 2,3,4,5).
    assert entry["lead_pid"] == lead
    assert entry["last4_ppg_lead"] == pytest.approx(
        sum(scored[w] for w in (2, 3, 4, 5)) / 4)
    assert entry["last4_ppg_lead"] < entry["last4_ppg"]
    # The lead himself, the QBs, and the solo back carry no filter.
    assert "lead_pid" not in out[lead]
    assert "lead_pid" not in out[qb1] and "lead_pid" not in out[qb2]
    assert "lead_pid" not in out[solo]


# ---------------------------------------------------------------------------
# Ranked paths: export (fake scorer) + pool parity (real scorer)
# ---------------------------------------------------------------------------

FAKE_INDEX = {
    "1": {"full_name": "Lead Back", "team": "KC"},
    "2": {"full_name": "Cuff Back", "team": "KC"},
}


def _bundle(pos, proj, **over):
    b = {
        "pos": pos, "proj_pts": proj, "on_bye": False,
        "recent_ppg": 10.0, "season_ppg": 9.0, "last4_ppg": None,
        "last4_ppg_lead": None, "lead_pid": None, "played_games": None,
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


def _kc_bundles(**cuff_over):
    base = {"last4_ppg": 18.4, "last4_ppg_lead": 12.8, "lead_pid": "1",
            "played_games": 9}
    base.update(cuff_over)
    return {
        "1": _bundle("RB", 20.0, last4_ppg=22.0, played_games=9),
        "2": _bundle("RB", 14.0, **base),
    }


def test_export_scores_handcuff_on_filtered_form_when_lead_active(export_env):
    import app as appmod

    export_env["bundles"] = _kc_bundles()
    out = appmod.build_start_sit_pool_export(2026, 11)
    assert [r["pid"] for r in out["RB"]] == ["1", "2"]
    seen = {s["proj_pts"]: s for s in _fake_compute_start_score.seen}
    # Regression: pre-fix the export fed the raw 18.4 here.
    assert seen[14.0]["last4_ppg"] == 12.8
    assert seen[20.0]["last4_ppg"] == 22.0  # the lead himself untouched


def test_export_scores_handcuff_on_raw_form_when_lead_out(export_env):
    import app as appmod

    export_env["bundles"] = _kc_bundles()
    export_env["live_players"] = {"1": {"injury_status": "Out"}}
    out = appmod.build_start_sit_pool_export(2026, 11)
    # The lead is excluded from the pool; the handcuff's full form is
    # the real one in the injury-replacement world.
    assert [r["pid"] for r in out["RB"]] == ["2"]
    seen = {s["proj_pts"]: s for s in _fake_compute_start_score.seen}
    assert seen[14.0]["last4_ppg"] == 18.4


def test_export_scores_handcuff_on_raw_form_when_lead_on_bye(export_env):
    import app as appmod

    bundles = _kc_bundles()
    bundles["1"]["on_bye"] = True
    bundles["2"]["on_bye"] = True
    export_env["bundles"] = bundles
    appmod.build_start_sit_pool_export(2026, 11)
    seen = {s["proj_pts"]: s for s in _fake_compute_start_score.seen}
    assert seen[14.0]["last4_ppg"] == 18.4


PARITY_INDEX = {
    "1": {"full_name": "Lead Back", "team": "KC", "injury_status": ""},
    "2": {"full_name": "Cuff Back", "team": "KC", "injury_status": ""},
}

PARITY_BUNDLES = {
    "1": _bundle("RB", 22.0, recent_ppg=21.0, season_ppg=20.0,
                 last4_ppg=22.5, played_games=9, opponent="DAL"),
    "2": _bundle("RB", 15.0, recent_ppg=10.0, season_ppg=9.0,
                 last4_ppg=18.4, last4_ppg_lead=12.8, lead_pid="1",
                 played_games=9, opponent="DAL"),
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


def test_pool_and_export_agree_on_filtered_form(parity_env):
    import app as appmod
    from utils.start_sit_score import compute_start_score

    export = appmod.build_start_sit_pool_export(2026, 11)
    pool = appmod.start_score_pool_scores(2026, 11, {"rec": 1.0})
    assert pool["kind"] == "score"
    exported = {r["pid"]: r["value"]
                for rows in export.values() for r in rows}
    assert exported["2"] == round(pool["by_pid"]["2"], 2)

    # The agreed value is the one the REAL scorer produces from the
    # filtered form -- and it is not the raw-form value pre-fix code
    # produced on both paths.
    common = dict(on_bye=False, recent_ppg=10.0, season_ppg=9.0,
                  def_rank=32, def_total=32, usage_delta=None,
                  usage_season_avg=None, injury_status="",
                  played_games=9, implied_total=None, bust_rate=None,
                  weather_kind=None, position="RB", oline_index=None,
                  expected_team_plays=None, league_average_plays=None,
                  role_confidence=None, snap_trend=None,
                  low_total_team=False)
    filtered, _, _ = compute_start_score(15.0, last4_ppg=12.8, **common)
    raw, _, _ = compute_start_score(15.0, last4_ppg=18.4, **common)
    assert exported["2"] == round(filtered, 2)
    assert exported["2"] != round(raw, 2)
    # The lead's own value is the raw-form computation, unchanged.
    lead_common = dict(common, recent_ppg=21.0, season_ppg=20.0)
    lead_val, _, _ = compute_start_score(22.0, last4_ppg=22.5,
                                         **lead_common)
    assert exported["1"] == round(lead_val, 2)
