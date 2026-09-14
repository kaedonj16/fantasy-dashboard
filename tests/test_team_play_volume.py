"""Unit tests for the team play-volume aggregation ("opp plays faced").

Pure logic tests -- no pandas / nfl_data_py needed. They exercise
``aggregate_team_play_volume`` (per-team pace/possession from raw play rows) and
the display-only ``_play_volume_context`` helper the start/sit endpoint uses.
"""
from data_building.external_data.nflverse_metrics import aggregate_team_play_volume


def _row(game, defteam, posteam, week, ptype="pass"):
    return (game, defteam, posteam, week, ptype)


def test_plays_faced_and_off_plays_per_game():
    # BAL defense faces 3 plays in g1 and 5 in g2 -> 4.0/gm over 2 games.
    # BAL offense runs 2 in g1 and 4 in g2 -> 3.0/gm.
    rows = []
    rows += [_row("g1", "BAL", "CIN", 1)] * 3
    rows += [_row("g1", "CIN", "BAL", 1)] * 2
    rows += [_row("g2", "BAL", "PIT", 2)] * 5
    rows += [_row("g2", "PIT", "BAL", 2)] * 4
    out = aggregate_team_play_volume(rows)
    assert out["BAL"]["plays_faced_pg"] == 4.0
    assert out["BAL"]["off_plays_pg"] == 3.0
    assert out["BAL"]["games"] == 2


def test_total_counts_pass_and_rush_but_no_split_fields():
    # Pass + rush both count toward the single total; the split fields the
    # backtest showed had no predictive edge are intentionally not emitted.
    rows = []
    rows += [_row("g1", "BAL", "CIN", 1, "pass")] * 4
    rows += [_row("g1", "BAL", "CIN", 1, "run")] * 2
    rows += [_row("g2", "BAL", "PIT", 2, "pass")] * 6
    rows += [_row("g2", "BAL", "PIT", 2, "run")] * 4
    out = aggregate_team_play_volume(rows)["BAL"]
    assert out["plays_faced_pg"] == 8.0  # (6+10)/2 total scrimmage plays
    assert out["games"] == 2
    assert "pass_faced_pg" not in out and "rush_faced_pg" not in out


def test_non_scrimmage_plays_excluded():
    # Punts / kickoffs / kneels / spikes / no_play don't count as volume.
    rows = [_row("g1", "BAL", "CIN", 1, "pass"),
            _row("g1", "BAL", "CIN", 1, "run"),
            _row("g1", "BAL", "CIN", 1, "punt"),
            _row("g1", "BAL", "CIN", 1, "kickoff"),
            _row("g1", "BAL", "CIN", 1, "qb_kneel"),
            _row("g1", "BAL", "CIN", 1, "qb_spike"),
            _row("g1", "BAL", "CIN", 1, "no_play"),
            _row("g1", "BAL", "CIN", 1, None)]
    out = aggregate_team_play_volume(rows)
    assert out["BAL"]["plays_faced_pg"] == 2.0
    assert out["BAL"]["games"] == 1


def test_last_four_games_window_by_week():
    # Six games; last-4 average should use weeks 3-6 only.
    rows = []
    counts = {1: 10, 2: 20, 3: 60, 4: 62, 5: 64, 6: 66}
    for wk, n in counts.items():
        rows += [_row(f"g{wk}", "KC", "OPP", wk)] * n
    out = aggregate_team_play_volume(rows)
    assert out["KC"]["games"] == 6
    # season = (10+20+60+62+64+66)/6 = 47.0
    assert out["KC"]["plays_faced_pg"] == 47.0
    # last four = (60+62+64+66)/4 = 63.0
    assert out["KC"]["plays_faced_l4_pg"] == 63.0


def test_team_code_normalisation():
    # Feed aliases; they should collapse onto canonical codes.
    rows = [_row("g1", "WSH", "JAC", 1), _row("g1", "JAC", "WSH", 1)]
    out = aggregate_team_play_volume(rows)
    assert "WAS" in out and "JAX" in out
    assert "WSH" not in out and "JAC" not in out


def test_offense_only_team_not_emitted():
    # A team seen only on offense has no defensive headline -> skipped.
    rows = [_row("g1", "BAL", "CIN", 1)]  # only BAL on defense, CIN on offense
    out = aggregate_team_play_volume(rows)
    assert "BAL" in out
    assert "CIN" not in out


def test_empty_rows():
    assert aggregate_team_play_volume([]) == {}


def test_play_volume_context_builds_delta():
    import pytest
    pytest.importorskip("flask")
    from app import _play_volume_context

    teams = {"BAL": {"plays_faced_pg": 58.4, "plays_faced_l4_pg": 56.8,
                     "off_plays_pg": 61.2, "games": 5}}
    pv = _play_volume_context(teams, "BAL", 64.5)
    assert pv["plays_faced_pg"] == 58.4
    assert pv["vs_avg"] == -6.1
    assert pv["plays_faced_l4_pg"] == 56.8
    assert pv["off_plays_pg"] == 61.2
    assert pv["games"] == 5

    # Alias opponent code resolves to the canonical row.
    assert _play_volume_context(teams, "BLT", 64.5)["plays_faced_pg"] == 58.4
    # Unknown opponent / empty table -> None (stat omitted).
    assert _play_volume_context(teams, "XYZ", 64.5) is None
    assert _play_volume_context({}, "BAL", 64.5) is None
    # No league average available -> no vs_avg key, but still returns the row.
    assert "vs_avg" not in _play_volume_context(teams, "BAL", None)
