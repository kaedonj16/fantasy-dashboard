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


def test_pass_rush_split():
    # g1: BAL faces 4 pass + 2 rush; g2: BAL faces 6 pass + 4 rush.
    rows = []
    rows += [_row("g1", "BAL", "CIN", 1, "pass")] * 4
    rows += [_row("g1", "BAL", "CIN", 1, "run")] * 2
    rows += [_row("g2", "BAL", "PIT", 2, "pass")] * 6
    rows += [_row("g2", "BAL", "PIT", 2, "run")] * 4
    out = aggregate_team_play_volume(rows)["BAL"]
    assert out["pass_faced_pg"] == 5.0   # (4+6)/2
    assert out["rush_faced_pg"] == 3.0   # (2+4)/2
    assert out["plays_faced_pg"] == 8.0  # total
    assert out["games"] == 2


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


def test_play_volume_context_position_aware():
    import pytest
    pytest.importorskip("flask")
    from app import _play_volume_context

    teams = {"BAL": {"plays_faced_pg": 58.4, "plays_faced_l4_pg": 56.8,
                     "pass_faced_pg": 33.1, "rush_faced_pg": 25.3,
                     "pass_faced_l4_pg": 32.0, "rush_faced_l4_pg": 24.8,
                     "off_plays_pg": 61.2, "games": 5}}
    avgs = {"total": 64.5, "pass": 37.6, "rush": 26.9}

    # RB -> rush basis headline.
    rb = _play_volume_context(teams, "BAL", "RB", avgs)
    assert rb["basis"] == "rush"
    assert rb["faced_pg"] == 25.3
    assert rb["vs_avg"] == round(25.3 - 26.9, 1)      # -1.6
    assert rb["total_vs_avg"] == round(58.4 - 64.5, 1)  # -6.1 (for Compare)

    # WR -> pass basis headline.
    wr = _play_volume_context(teams, "BAL", "WR", avgs)
    assert wr["basis"] == "pass"
    assert wr["faced_pg"] == 33.1
    assert wr["vs_avg"] == round(33.1 - 37.6, 1)      # -4.5

    # Splits + total present regardless of basis (for the card detail/Compare).
    assert wr["pass_faced_pg"] == 33.1 and wr["rush_faced_pg"] == 25.3
    assert wr["plays_faced_pg"] == 58.4 and wr["games"] == 5

    # Alias opponent code resolves; unknown / empty table -> None.
    assert _play_volume_context(teams, "BLT", "WR", avgs)["faced_pg"] == 33.1
    assert _play_volume_context(teams, "XYZ", "WR", avgs) is None
    assert _play_volume_context({}, "BAL", "WR", avgs) is None


def test_play_volume_context_falls_back_to_total_when_split_missing():
    import pytest
    pytest.importorskip("flask")
    from app import _play_volume_context

    # A row with only the total (e.g. an older cache) still yields a headline.
    teams = {"BAL": {"plays_faced_pg": 58.4, "plays_faced_l4_pg": 56.8, "games": 5}}
    avgs = {"total": 64.5, "pass": 37.6, "rush": 26.9}
    pv = _play_volume_context(teams, "BAL", "RB", avgs)
    assert pv["basis"] == "total"
    assert pv["faced_pg"] == 58.4
    assert pv["vs_avg"] == round(58.4 - 64.5, 1)
