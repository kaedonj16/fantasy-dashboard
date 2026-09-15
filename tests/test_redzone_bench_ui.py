from pathlib import Path


def test_redzone_keeps_every_bench_player_accessible_and_unknown_scores_unknown():
    source = Path("static/redzone.js").read_text()
    roster = source[source.index("function _rosterCard("):source.index("function _myMatchups(")]
    teams = source[source.index("function _renderMyTeams("):source.index("function _renderScoreboard(")]
    scoring = source[source.index("function _playerPts("):source.index("function _totalPtsForPid(")]
    assert "bench.slice(" not in roster
    assert "bench.slice(" not in teams
    assert "matchup.bench_slots || matchup.bench" in roster
    assert "m.bench_slots || m.bench" in teams
    assert "return null" in scoring
    assert "if (!isNaN(platformN)) return platformN" in scoring
