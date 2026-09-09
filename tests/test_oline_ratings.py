"""Unit tests for the nflverse-derived O-line ratings (no network).

These guard the pure logic -- the line-yards weighting, the percentile scaling,
and the end-to-end assembly on a synthetic play-by-play frame -- so a broken
formula is caught without hitting nflverse.
"""
import pytest

pd = pytest.importorskip("pandas")

from data_building.oline_ratings import (  # noqa: E402
    _line_yards,
    _percentile_index,
    build_oline_ratings,
)


def test_line_yards_weighting():
    # Football Outsiders buckets: stuffed 1.2x, 0-4 full, 5-10 half, 11+ capped.
    assert _line_yards(-2) == pytest.approx(-2.4)   # 1.2 * -2
    assert _line_yards(3) == pytest.approx(3.0)     # full credit
    assert _line_yards(4) == pytest.approx(4.0)
    assert _line_yards(10) == pytest.approx(7.0)    # 4 + 0.5*6
    # long runs get no extra line credit past 11 yards
    assert _line_yards(11) == _line_yards(80) == pytest.approx(7.0)


def test_percentile_index_orders_and_spreads():
    idx = _percentile_index({"A": 5.0, "B": 3.0, "C": 1.0}, higher_is_better=True)
    assert idx["A"] == 100.0 and idx["C"] == 0.0 and idx["B"] == 50.0
    # lower-is-better flips it (fewer sacks = better line)
    inv = _percentile_index({"A": 5.0, "B": 3.0, "C": 1.0}, higher_is_better=False)
    assert inv["C"] == 100.0 and inv["A"] == 0.0


def _synthetic_pbp():
    """Two offenses vs one defense: GOOD blocks well, BAD blocks poorly."""
    rows = []
    # GOOD: consistent 3-4 yard runs, almost no sacks/hits
    for _ in range(60):
        rows.append(dict(season=2025, week=1, season_type="REG", posteam="GOOD",
                         defteam="DEF", rush_attempt=1, pass_attempt=0, qb_dropback=0,
                         rushing_yards=4, yards_gained=4, sack=0, qb_hit=0,
                         qb_kneel=0, qb_spike=0, two_point_attempt=0))
    for _ in range(60):
        rows.append(dict(season=2025, week=1, season_type="REG", posteam="GOOD",
                         defteam="DEF", rush_attempt=0, pass_attempt=1, qb_dropback=1,
                         rushing_yards=0, yards_gained=7, sack=0, qb_hit=0,
                         qb_kneel=0, qb_spike=0, two_point_attempt=0))
    # BAD: lots of stuffs, lots of sacks and hits
    for _ in range(60):
        rows.append(dict(season=2025, week=1, season_type="REG", posteam="BAD",
                         defteam="DEF", rush_attempt=1, pass_attempt=0, qb_dropback=0,
                         rushing_yards=-1, yards_gained=-1, sack=0, qb_hit=0,
                         qb_kneel=0, qb_spike=0, two_point_attempt=0))
    for i in range(60):
        rows.append(dict(season=2025, week=1, season_type="REG", posteam="BAD",
                         defteam="DEF", rush_attempt=0, pass_attempt=1, qb_dropback=1,
                         rushing_yards=0, yards_gained=0,
                         sack=1 if i < 15 else 0, qb_hit=1 if i < 30 else 0,
                         qb_kneel=0, qb_spike=0, two_point_attempt=0))
    return pd.DataFrame(rows)


def test_build_end_to_end_ranks_good_over_bad(monkeypatch):
    frame = _synthetic_pbp()
    monkeypatch.setattr(
        "data_building.oline_ratings._load_pbp_year",
        lambda year, pd_, nfl=None: frame if year == 2025 else None,
    )
    out = build_oline_ratings(2025, through_week=1, save=False)
    r = out["ratings"]
    assert set(r) == {"GOOD", "BAD", "DEF"} - {"DEF"} or "GOOD" in r
    assert r["GOOD"]["composite"] > r["BAD"]["composite"]
    assert r["GOOD"]["sack_rate"] < r["BAD"]["sack_rate"]
    assert r["BAD"]["stuffed_rate"] > r["GOOD"]["stuffed_rate"]
    # scale sanity: percentile indices land inside [0, 100]
    for row in r.values():
        assert 0.0 <= row["composite"] <= 100.0
