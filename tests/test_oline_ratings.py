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
