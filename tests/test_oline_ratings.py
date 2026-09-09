"""Unit tests for the nflverse-derived O-line ratings (no network).

Guards the pure logic -- line-yards weighting, opponent adjustment, prior
regression, time-to-throw residualisation, percentile scaling, and the
end-to-end assembly on a synthetic play-by-play frame -- so a broken formula is
caught without hitting nflverse.
"""
import pytest

pd = pytest.importorskip("pandas")

from data_building.oline_ratings import (  # noqa: E402
    _line_yards,
    _gap_of,
    _alt_adjust,
    _regress_to_prior,
    _residualize,
    _percentile_index,
    build_oline_ratings,
)


def test_line_yards_weighting():
    assert _line_yards(-2) == pytest.approx(-2.4)   # stuffed 1.2x
    assert _line_yards(3) == pytest.approx(3.0)      # 0-4 full
    assert _line_yards(4) == pytest.approx(4.0)
    assert _line_yards(10) == pytest.approx(7.0)     # 4 + 0.5*6
    assert _line_yards(11) == _line_yards(80) == pytest.approx(7.0)  # capped


def test_gap_bucketing():
    assert _gap_of("left", "guard") == "interior"
    assert _gap_of("middle", None) == "interior"
    assert _gap_of("right", "tackle") == "tackle"
    assert _gap_of("left", "end") == "end"
    assert _gap_of("left", None) is None


def test_alt_adjust_recovers_offense_effect():
    # Fully-crossed design (every offense faces every defense, so effects are
    # identifiable) with a known additive structure:
    #   value = 3 (league) + off_skill + def_effect
    #   off_skill: GOOD +1, BAD -1, REF 0   def_effect: TOUGH -1, SOFT +1
    league = 3.0
    off_skill = {"GOOD": 1.0, "BAD": -1.0, "REF": 0.0}
    def_eff = {"TOUGH": -1.0, "SOFT": 1.0}
    triples = []
    for o, os_ in off_skill.items():
        for d, de in def_eff.items():
            triples += [(o, d, league + os_ + de)] * 10
    adj, n, lg = _alt_adjust(triples)
    # adjusted value = league + off_skill, so ordering and magnitude recover
    assert adj["GOOD"] > adj["REF"] > adj["BAD"]
    assert adj["GOOD"] == pytest.approx(4.0, abs=0.05)
    assert adj["BAD"] == pytest.approx(2.0, abs=0.05)


def test_regress_to_prior_shrinks_small_samples():
    cur = {"A": 10.0, "B": 10.0}
    n = {"A": 200, "B": 5}          # A has a big sample, B tiny
    prior = {"A": 0.0, "B": 0.0}
    out = _regress_to_prior(cur, n, prior, league_cur=0.0, K=100.0)
    assert out["A"] > out["B"]      # B pulled harder toward its prior
    assert out["B"] < 5.0


def test_residualize_removes_time_to_throw_effect():
    # value rises purely with x; residuals should be ~0 (line not to blame).
    value = {"A": 2.0, "B": 4.0, "C": 6.0, "D": 8.0, "E": 10.0}
    x = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    resid = _residualize(value, x)
    assert all(abs(v) < 1e-6 for v in resid.values())


def test_percentile_index_orders_and_spreads():
    idx = _percentile_index({"A": 5.0, "B": 3.0, "C": 1.0}, higher_is_better=True)
    assert idx["A"] == 100.0 and idx["C"] == 0.0 and idx["B"] == 50.0
    inv = _percentile_index({"A": 5.0, "B": 3.0, "C": 1.0}, higher_is_better=False)
    assert inv["C"] == 100.0 and inv["A"] == 0.0


def _row(**kw):
    base = dict(season=2025, week=1, season_type="REG", posteam="GOOD",
                defteam="DEF", rush_attempt=0, pass_attempt=0, qb_dropback=0,
                rushing_yards=0, yards_gained=0, sack=0, qb_hit=0,
                was_pressure=0, time_to_throw=2.6, qb_kneel=0, qb_spike=0,
                qb_scramble=0, run_location="middle", run_gap="guard",
                success=0, passer_player_id=None, rusher_player_id=None,
                wp=0.5, score_differential=0)
    base.update(kw)
    return base


def _synthetic_pbp():
    """GOOD blocks well; BAD gets stuffed and pressured. Same time to throw,
    so the residualiser can't explain BAD's pressure away. Includes success
    flags so the run-block blend has both inputs."""
    rows = []
    for _ in range(60):
        rows.append(_row(posteam="GOOD", rush_attempt=1, rushing_yards=4, yards_gained=4, success=1))
        rows.append(_row(posteam="GOOD", qb_dropback=1, pass_attempt=1, was_pressure=0, sack=0))
    for i in range(60):
        rows.append(_row(posteam="BAD", rush_attempt=1, rushing_yards=-1, yards_gained=-1, success=0))
        rows.append(_row(posteam="BAD", qb_dropback=1, pass_attempt=1,
                         was_pressure=1 if i < 40 else 0,
                         sack=1 if i < 15 else 0, qb_hit=1 if i < 30 else 0))
    return pd.DataFrame(rows)


def test_is_qb_run_filters_scramble_and_keeper():
    from data_building.oline_ratings import _is_qb_run
    assert _is_qb_run(1, "QB1", "QB1") is True
    assert _is_qb_run(0, "QB1", "QB1") is True          # designed keep
    assert _is_qb_run(0, "QB1", "RB1") is False
    assert _is_qb_run(0, None, "RB1") is False
    assert _is_qb_run(None, None, None) is False


def test_build_end_to_end_ranks_good_over_bad(monkeypatch):
    frame = _synthetic_pbp()
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    out = build_oline_ratings(2025, through_week=1, save=False)
    r = out["ratings"]
    assert out["pressure_source"] == "was_pressure"
    assert "GOOD" in r and "BAD" in r and "DEF" not in r
    assert r["GOOD"]["composite"] > r["BAD"]["composite"]
    assert r["GOOD"]["pass_block"] > r["BAD"]["pass_block"]
    assert r["GOOD"]["run_block"] > r["BAD"]["run_block"]
    assert r["BAD"]["stuffed_rate"] > r["GOOD"]["stuffed_rate"]
    assert r["BAD"]["pressure_rate"] > r["GOOD"]["pressure_rate"]
    assert r["GOOD"]["success_rate"] > r["BAD"]["success_rate"]
    assert out["weights"]["run_ly"] + out["weights"]["run_success"] == pytest.approx(1.0)
    for row in r.values():
        assert 0.0 <= row["composite"] <= 100.0


def test_qb_runs_excluded_from_run_grade(monkeypatch):
    """A team whose only 'good' rushes are QB keeps should not get credit."""
    rows = []
    for _ in range(50):
        # RB rushes are stuffed
        rows.append(_row(posteam="MOBILE", rush_attempt=1, rushing_yards=-1,
                         yards_gained=-1, success=0, rusher_player_id="RB1",
                         passer_player_id="QB1"))
        # QB keeps look great but must be excluded
        rows.append(_row(posteam="MOBILE", rush_attempt=1, rushing_yards=8,
                         yards_gained=8, success=1, qb_scramble=0,
                         rusher_player_id="QB1", passer_player_id="QB1"))
        # Honest line with solid RB rushes
        rows.append(_row(posteam="SOLID", rush_attempt=1, rushing_yards=4,
                         yards_gained=4, success=1, rusher_player_id="RB2",
                         passer_player_id="QB2"))
        rows.append(_row(posteam="MOBILE", qb_dropback=1, pass_attempt=1, was_pressure=0, sack=0))
        rows.append(_row(posteam="SOLID", qb_dropback=1, pass_attempt=1, was_pressure=0, sack=0))
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    out = build_oline_ratings(2025, through_week=1, save=False)
    r = out["ratings"]
    assert r["SOLID"]["run_block"] > r["MOBILE"]["run_block"]
    # MOBILE's graded rushes are the stuffed RB runs only.
    assert r["MOBILE"]["line_yards"] < 0
    assert r["MOBILE"]["n_rush"] == 50


def test_talent_prior_decays_toward_zero_as_sample_grows():
    """The offseason adjustment lives in the prior term, so n_cur/K washes it out."""
    from data_building.oline_talent_prior import shift_prior
    from data_building.oline_ratings import _regress_to_prior, RUN_PRIOR_K

    cur = {"A": 5.0, "B": 5.0, "C": 5.0, "D": 5.0}
    prior_plain = {"A": 0.0, "B": 2.0, "C": 1.0, "D": 1.5}
    talent = {"A": 1.5, "B": 0.0, "C": 0.0, "D": 0.0}
    prior_talent = shift_prior(prior_plain, talent, weight=0.4, higher_is_better=True,
                               league=0.0)
    assert prior_talent["A"] != prior_plain["A"]

    small = {"A": 8, "B": 8, "C": 8, "D": 8}
    large = {"A": 2000, "B": 2000, "C": 2000, "D": 2000}
    K = RUN_PRIOR_K
    d_small = (
        _regress_to_prior(cur, small, prior_talent, 0.0, K)["A"]
        - _regress_to_prior(cur, small, prior_plain, 0.0, K)["A"]
    )
    d_large = (
        _regress_to_prior(cur, large, prior_talent, 0.0, K)["A"]
        - _regress_to_prior(cur, large, prior_plain, 0.0, K)["A"]
    )
    assert abs(d_small) > abs(d_large)
    assert abs(d_large) < 0.05
    # Influence equals K/(n+K) * prior_delta, so the large-n share is tiny.
    prior_delta = prior_talent["A"] - prior_plain["A"]
    assert d_small == pytest.approx(K / (8 + K) * prior_delta)
    assert d_large == pytest.approx(K / (2000 + K) * prior_delta)


def test_stuart_pick_value_ignores_pff_and_falls_back():
    from data_building.oline_talent_prior import stuart_pick_value
    chart = {1: 34.6, 32: 12.5, "pff": 999}
    assert stuart_pick_value(1, chart) == pytest.approx(34.6)
    assert stuart_pick_value(32, chart) == pytest.approx(12.5)
    # Unknown pick uses the exponential fallback, never a PFF column.
    v = stuart_pick_value(10, None)
    assert 18.0 < v < 27.0
    assert stuart_pick_value(0) == 0.0


def test_continuity_and_veteran_net_from_snaps():
    from data_building.oline_talent_prior import ol_continuity_and_veteran_net
    prior = [
        {"player_id": "ret", "team": "PHI", "snaps": 800},
        {"player_id": "gone", "team": "PHI", "snaps": 200},
        {"player_id": "arriv", "team": "DAL", "snaps": 700},
        {"player_id": "stay", "team": "DAL", "snaps": 100},
    ]
    current = {"ret": "PHI", "arriv": "PHI", "stay": "DAL"}  # gone left the league
    cont, vet, detail = ol_continuity_and_veteran_net(prior, current)
    assert cont["PHI"] == pytest.approx(0.8)
    assert vet["PHI"] == pytest.approx(700 - 200)
    assert cont["DAL"] == pytest.approx(100 / 800)
    assert vet["DAL"] == pytest.approx(0 - 700)
    assert detail["PHI"]["returning_snaps"] == pytest.approx(800)


def test_draft_value_sums_ol_picks_only():
    from data_building.oline_talent_prior import ol_draft_value
    chart = {1: 34.6, 50: 9.7, 100: 5.3}
    picks = [
        {"team": "PHI", "pick": 1, "position": "T", "category": "OL"},
        {"team": "PHI", "pick": 50, "position": "WR", "category": "WR"},
        {"team": "DAL", "pick": 100, "position": "C"},
    ]
    val, detail = ol_draft_value(picks, chart)
    assert val["PHI"] == pytest.approx(34.6)
    assert val["DAL"] == pytest.approx(5.3)
    assert detail["PHI"]["n_ol_picks"] == 1


def test_talent_scores_are_zscored_across_teams():
    from data_building.oline_talent_prior import compute_talent_scores
    prior = []
    current = {}
    for i, team in enumerate(("A", "B", "C", "D", "E", "F")):
        pid = f"p{team}"
        prior.append({"player_id": pid, "team": team, "snaps": 1000})
        # Half the roster walks; A-C keep their guy, D-F lose him.
        if i < 3:
            current[pid] = team
    draft = [
        {"team": "D", "pick": 1, "position": "T", "category": "OL"},
        {"team": "A", "pick": 200, "position": "G", "category": "OL"},
    ]
    scores, detail, used = compute_talent_scores(
        prior, current, draft, pick_chart={1: 34.6, 200: 0.9},
        w_continuity=0.2, w_draft=0.4, w_veteran=0.4)
    assert used
    assert set(scores) >= {"A", "B", "C", "D", "E", "F"}
    mean = sum(scores.values()) / len(scores)
    assert abs(mean) < 0.05
    # D drafted at 1 and lost its starter; A kept its starter and drafted late.
    # Continuity favors A; draft favors D. Just assert we produced a spread.
    assert max(scores.values()) - min(scores.values()) > 0.5


def test_shift_prior_flips_sign_for_pressure():
    from data_building.oline_talent_prior import shift_prior
    prior = {"A": 0.30, "B": 0.30, "C": 0.30, "D": 0.32}
    scores = {"A": 1.0, "B": -1.0, "C": 0.0, "D": 0.0}
    better = shift_prior(prior, scores, weight=0.5, higher_is_better=True)
    worse = shift_prior(prior, scores, weight=0.5, higher_is_better=False)
    assert better["A"] > prior["A"] > better["B"]
    assert worse["A"] < prior["A"] < worse["B"]


def test_pfr_snaps_match_gsis_roster_via_xwalk():
    """Snap counts key on PFR ids; rosters key on GSIS. The public players
    file is the join — without it continuity is identically zero."""
    from data_building.oline_talent_prior import (
        _canonical_pid, ol_continuity_and_veteran_net,
    )
    xwalk = {"WyliAn00": "00-0030001"}
    assert _canonical_pid("WyliAn00", None, "Andrew Wylie", xwalk) == "00-0030001"
    prior = [
        {"player_id": _canonical_pid("WyliAn00", None, "Andrew Wylie", xwalk),
         "team": "WAS", "snaps": 900},
        {"player_id": _canonical_pid("GoneXx00", None, "Gone Guy", xwalk),
         "team": "WAS", "snaps": 100},
    ]
    current = {"00-0030001": "WAS"}
    cont, vet, _ = ol_continuity_and_veteran_net(prior, current)
    assert cont["WAS"] == pytest.approx(0.9)


def test_flag_off_does_not_call_talent_loader(monkeypatch):
    frame = _synthetic_pbp()
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("talent prior should not load when flag is off")

    monkeypatch.setattr(
        "data_building.oline_talent_prior.compute_oline_talent_prior", _boom)
    out = build_oline_ratings(2025, through_week=1, save=False, use_talent_prior=False)
    assert called["n"] == 0
    assert "talent_prior" not in out
    for row in out["ratings"].values():
        assert "composite_realized" not in row


def test_results_composite_unchanged_when_flag_off(monkeypatch):
    frame = _synthetic_pbp()
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    # Injecting talent inputs must not matter when the flag is off.
    fake_inputs = {
        "prior_snaps": [{"player_id": "x", "team": "GOOD", "snaps": 1000}],
        "current_team_by_player": {"x": "GOOD"},
        "draft_picks": [{"team": "GOOD", "pick": 1, "position": "T", "category": "OL"}],
        "pick_chart": {1: 34.6},
        "roster_source": "week_1",
    }
    off = build_oline_ratings(2025, through_week=1, save=False, use_talent_prior=False)
    off2 = build_oline_ratings(
        2025, through_week=1, save=False, use_talent_prior=False, talent_inputs=fake_inputs)
    assert off["ratings"]["GOOD"]["composite"] == off2["ratings"]["GOOD"]["composite"]
    assert off["ratings"]["BAD"]["composite"] == off2["ratings"]["BAD"]["composite"]
    assert off["weights"] == off2["weights"]


def test_missing_roster_draft_falls_back_to_today(monkeypatch):
    frame = _synthetic_pbp()
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    off = build_oline_ratings(2025, through_week=1, save=False, use_talent_prior=False)
    # Empty inputs skip the network load and return no scores -> today's prior.
    on = build_oline_ratings(
        2025, through_week=1, save=False, use_talent_prior=True, talent_inputs={})
    assert on["ratings"]["GOOD"]["composite"] == off["ratings"]["GOOD"]["composite"]
    assert on["ratings"]["BAD"]["composite"] == off["ratings"]["BAD"]["composite"]
    assert "talent_prior" not in on
    assert on["weights"] == off["weights"]


def _four_team_row(posteam, **kw):
    return _row(posteam=posteam, **kw)


def _balanced_plays(team, rush_yds, press, n=40):
    rows = []
    for i in range(n):
        rows.append(_four_team_row(
            team, rush_attempt=1, rushing_yards=rush_yds, yards_gained=rush_yds,
            success=1 if rush_yds >= 4 else 0))
        rows.append(_four_team_row(
            team, qb_dropback=1, pass_attempt=1,
            was_pressure=1 if i < press else 0,
            sack=1 if i < max(0, press // 3) else 0))
    return rows


def test_talent_flag_on_shifts_grade_and_labels_realized(monkeypatch):
    """Prior-season spread + a large talent residual must move the primary grade
    and leave the unshifted measurement in *_realized."""
    # Current season: four teams look identical, so last-year prior decides rank.
    cur_rows = []
    for t in ("WINS", "OK", "MEH", "LOSE"):
        cur_rows += _balanced_plays(t, rush_yds=4, press=10, n=40)
    cur = pd.DataFrame(cur_rows)
    # Prior season: WINS >> LOSE.
    prior_rows = []
    prior_rows += _balanced_plays("WINS", rush_yds=6, press=2, n=40)
    prior_rows += _balanced_plays("OK", rush_yds=4, press=10, n=40)
    prior_rows += _balanced_plays("MEH", rush_yds=3, press=16, n=40)
    prior_rows += _balanced_plays("LOSE", rush_yds=1, press=28, n=40)
    prior = pd.DataFrame(prior_rows)

    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: cur if year == 2025 else prior if year == 2024 else None,
    )
    off = build_oline_ratings(2025, through_week=1, save=False, use_talent_prior=False)
    assert off["ratings"]["WINS"]["composite"] > off["ratings"]["LOSE"]["composite"]

    def _fake_talent(season, pd=None, nfl=None, **k):
        return {
            "scores": {"WINS": -2.0, "OK": 0.0, "MEH": 0.0, "LOSE": 2.0},
            "detail": {
                "WINS": {"continuity": 0.2, "draft_value": 0.0, "veteran_net": -900, "score": -2.0},
                "LOSE": {"continuity": 0.9, "draft_value": 34.6, "veteran_net": 800, "score": 2.0},
            },
            "weights_used": {"continuity": 0.2, "draft": 0.4, "veteran_net": 0.4},
            "meta": {
                "sources": ["test"],
                "omitted": "coaching",
                "roster_source": "week_1",
                "pick_chart": "stuart",
            },
        }

    monkeypatch.setattr(
        "data_building.oline_talent_prior.compute_oline_talent_prior", _fake_talent)
    on = build_oline_ratings(
        2025, through_week=1, save=False, use_talent_prior=True, talent_weight=1.0)
    assert on["talent_prior"]["kind"] == "projection"
    assert on["talent_prior"]["applied_to_ratings"] is True
    assert on["ratings"]["WINS"]["composite_realized"] == off["ratings"]["WINS"]["composite"]
    assert on["ratings"]["LOSE"]["composite_realized"] == off["ratings"]["LOSE"]["composite"]
    # LOSE's offseason upgrade pulls it up relative to WINS vs the realized grade.
    realized_gap = (off["ratings"]["WINS"]["composite"] - off["ratings"]["LOSE"]["composite"])
    talent_gap = (on["ratings"]["WINS"]["composite"] - on["ratings"]["LOSE"]["composite"])
    assert talent_gap < realized_gap
    assert on["ratings"]["LOSE"]["composite"] > off["ratings"]["LOSE"]["composite"]
    assert "talent_prior_w" in on["weights"]
    assert "talent_prior_w" not in off["weights"]
