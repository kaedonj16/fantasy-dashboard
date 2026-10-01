"""Recap additions: top scorers by position (Week at a Glance), the Injury
Report section, and the League Activity section."""
from pathlib import Path

import pytest

from dashboard_services.pages.recap_page import (
    _notable_injuries,
    _recent_activity,
    _starter_player_ids,
    _top_scorers_by_position,
)

ROOT = Path(__file__).resolve().parents[1]


def _team(rid, starters, *, historical=True):
    return {"roster_id": rid, "starters": starters,
            "lineup_is_historical": historical}


def _player(pid, name, pos, pts, nfl="KC"):
    return {"pid": pid, "name": name, "pos": pos, "nfl": nfl, "pts": pts}


def test_top_scorers_ranks_within_position_and_caps_at_three():
    matchups = [{
        "left": _team("1", [
            _player("q1", "Qb One", "QB", 30.0),
            _player("r1", "Rb One", "RB", 25.0),
            _player("r2", "Rb Two", "RB", 20.0),
        ]),
        "right": _team("2", [
            _player("q2", "Qb Two", "QB", 35.5),
            _player("r3", "Rb Three", "RB", 22.0),
            _player("r4", "Rb Four", "RB", 18.0),
            _player("r5", "Rb Five", "RB", 10.0),
        ]),
    }]

    result = dict(_top_scorers_by_position(matchups))

    assert [p["pid"] for p in result["QB"]] == ["q2", "q1"]
    # Four RBs started league-wide; only the top 3 survive, best first.
    assert [p["pid"] for p in result["RB"]] == ["r1", "r3", "r2"]
    assert result["RB"][0]["rid"] == "1"


def test_top_scorers_position_display_order():
    matchups = [{
        "left": _team("1", [
            _player("d1", "Denver Broncos", "DEF", 12.0, nfl="DEN"),
            _player("k1", "Kicker", "K", 9.0),
            _player("w1", "Wideout", "WR", 21.0),
        ]),
        "right": _team("2", [_player("t1", "Tight End", "TE", 14.0)]),
    }]

    positions = [pos for pos, _ in _top_scorers_by_position(matchups)]

    assert positions == ["WR", "TE", "K", "DEF"]


def test_top_scorers_ignores_bench_fallback_and_missing_points():
    matchups = [{
        # Current-roster fallback lineups never speak for a past week.
        "left": _team("1", [_player("new", "Current Addition", "RB", 40.0)],
                      historical=False),
        "right": _team("2", [
            _player("ghost", "No Points", "RB", None),
            _player("real", "Real Starter", "RB", 11.0),
            None,  # empty slot placeholder
        ]),
    }]

    result = dict(_top_scorers_by_position(matchups))

    assert [p["pid"] for p in result["RB"]] == ["real"]
    assert _starter_player_ids(matchups) == {"ghost", "real"}


def _injury_df(rows):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


def _inj_row(rid, pid, player, status, injury="", body="", pos="RB", nfl="KC",
             team="Team A"):
    return {"RosterID": rid, "Team": team, "PlayerID": pid, "Player": player,
            "Pos": pos, "NFL": nfl, "Status": status, "Injury": injury,
            "Body": body, "Last Updated": None, "NewsUrl": ""}


def test_notable_injuries_severe_plus_questionable_starters_only():
    df = _injury_df([
        _inj_row("1", "out1", "Out Starter", "Active", injury="Out", body="Knee"),
        _inj_row("1", "qstart", "Questionable Starter", "Questionable"),
        _inj_row("2", "qbench", "Questionable Bench", "Questionable", team="Team B"),
        _inj_row("2", "ir1", "Ir Back", "IR", team="Team B"),
        _inj_row("3", "fine", "Healthy", "Active", team="Team C"),
        _inj_row("", "fa", "Free Agent", "Out", team="Free Agent"),
        _inj_row("3", "dbt", "Doubtful Guy", "Active", injury="Doubtful", team="Team C"),
    ])

    rows = _notable_injuries(df, {"out1", "qstart"})

    # Severity first (Out/IR, then Doubtful, then Questionable); the
    # questionable bench player, the healthy player, and the free agent
    # are not league injury news.
    assert [r["pid"] for r in rows] == ["out1", "ir1", "dbt", "qstart"]
    assert rows[0]["status"] == "OUT"
    assert rows[0]["body"] == "Knee"
    assert rows[0]["started"] is True


def test_notable_injuries_handles_missing_report():
    assert _notable_injuries(None, {"x"}) == []


def _activity_df(rows):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


def test_recent_activity_filters_to_the_recap_week_newest_first():
    df = _activity_df([
        {"kind": "trade", "week": 4, "ts": None, "data": {"teams": []}},
        {"kind": "waiver", "week": 3, "ts": None, "data": {"adds": []}},
        {"kind": "waiver", "week": 4, "ts": None, "data": {"adds": []}},
    ])

    rows = _recent_activity(df, 4)

    assert [r["kind"] for r in rows] == ["trade", "waiver"]
    assert _recent_activity(df, 9) == []
    assert _recent_activity(None, 4) == []


def test_recap_source_contracts_for_new_sections():
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()

    # Week at a Glance carries the per-position top scorers.
    assert "Top scorers by position" in page
    assert "{pos_scorers_html}</section>" in page
    # Injury report: canonical lazy fill, gated to the latest week because
    # the underlying report is a current snapshot.
    assert "ensure_injury_bits" in page and "ensure_activity_bits" in page
    assert 'selected_week == available_weeks[-1]' in page
    assert "<h2>Injury Report</h2>" in page
    # League activity renders after standings, before Up Next.
    assert "<h2>League Activity</h2>" in page
    ret = page.split("return ('<main class=\"weekly-recap\">'", 1)[1]
    assert ret.index("standings_html") < ret.index("activity_html") < ret.index("up_next_html")
    assert ret.index("cards_html") < ret.index("injuries_html") < ret.index("story_html")
