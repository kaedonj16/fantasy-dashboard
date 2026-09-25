"""Unit tests for utils.streaming_targets (matchup-based K/DST streaming rankers)."""
import sys
import types

import pytest

from utils.streaming_targets import stream_score, streaming_targets

SCHEDULE = [
    {"home": "KC", "away": "BUF", "gameDate": "20260928"},
    {"home": "MIA", "away": "NE", "gameDate": "20260928"},
]

IMPLIED = {"KC": 28.0, "BUF": 24.0, "MIA": 17.0, "NE": 19.0}

PLAYERS = {
    "k_kc": {"pos": "K", "team": "KC", "name": "KC Kicker"},
    "k_buf": {"pos": "K", "team": "BUF", "name": "BUF Kicker"},
    "k_mia": {"pos": "K", "team": "MIA", "name": "MIA Kicker"},
    "k_kc2": {"pos": "K", "team": "KC", "name": "KC Kicker 2"},
    "k_dal": {"pos": "K", "team": "DAL", "name": "DAL Kicker"},  # DAL idle this week
    "qb1": {"pos": "QB", "team": "KC", "name": "Some QB"},
}

RPOS = ["QB", "RB", "WR", "TE", "K", "DEF", "BN", "BN"]


def _ctx(**over):
    ctx = {
        "current_week": 4,
        "roster_positions": list(RPOS),
        # BUF D/ST is rostered (pid == team abbr), so it must be excluded.
        "rosters": [{"roster_id": 1, "players": ["BUF"]}],
        "players_index": dict(PLAYERS),
    }
    ctx.update(over)
    return ctx


@pytest.fixture
def vegas(monkeypatch):
    """Stub the cached schedule + Vegas loaders the ranker reads."""
    utils_stub = types.ModuleType("utils.utils")
    utils_stub.load_week_sched = lambda season, week: list(SCHEDULE)
    conds_stub = types.ModuleType("utils.game_conditions")
    conds_stub.build_week_conditions = lambda season, week, games: {
        t: {"implied_total": v} for t, v in IMPLIED.items()
    }
    monkeypatch.setitem(sys.modules, "utils.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "utils.game_conditions", conds_stub)


def test_stream_score_scale():
    # Defenses: a low opponent total is good -> above the waiver floor.
    assert stream_score(17.0, lower_is_better=True) > 100.0
    assert stream_score(30.0, lower_is_better=True) < 100.0
    # Kickers: a high own total is good.
    assert stream_score(28.0, lower_is_better=False) > 100.0
    assert stream_score(18.0, lower_is_better=False) < 100.0
    # Missing data is mid-pack, never zero.
    assert stream_score(None, lower_is_better=True) == 75.0
    assert stream_score(None, lower_is_better=False) == 75.0


def test_defense_sorted_by_lowest_opp_implied(vegas):
    res = streaming_targets(_ctx(), 2026)
    assert res["in_season"] is True
    got = [r["player_id"] for r in res["defense"]]
    # NE faces MIA (17) < MIA faces NE (19) < KC faces BUF (24); BUF is rostered.
    assert got == ["NE", "MIA", "KC"]
    assert "BUF" not in got
    ne = res["defense"][0]
    assert ne["opp_implied"] == 17.0
    assert ne["stream_score"] > 100.0  # best matchup scores above replacement


def test_kicker_sorted_by_highest_own_implied(vegas):
    res = streaming_targets(_ctx(), 2026)
    got = [r["player_id"] for r in res["kicker"]]
    # KC (28) > BUF (24) > MIA (17); k_kc2 dropped (one per team); k_dal idle.
    assert got == ["k_kc", "k_buf", "k_mia"]
    assert res["kicker"][0]["own_implied"] == 28.0
    assert res["kicker"][0]["stream_score"] > 100.0


def test_rostered_kicker_pid_excluded(vegas):
    ctx = _ctx(rosters=[{"roster_id": 1, "players": ["BUF", "k_kc"]}])
    res = streaming_targets(ctx, 2026)
    # k_kc rostered -> k_kc2 becomes KC's representative (one per team kept).
    assert [r["player_id"] for r in res["kicker"]] == ["k_kc2", "k_buf", "k_mia"]


def test_position_gating(vegas):
    # League starts no K/DST: both lists empty, flags false, still in season.
    res = streaming_targets(_ctx(roster_positions=["QB", "RB", "WR", "TE", "BN"]), 2026)
    assert res["defense"] == [] and res["kicker"] == []
    assert res["uses_k"] is False and res["uses_def"] is False
    assert res["in_season"] is True
    # K-only league: kickers ranked, no defenses.
    res = streaming_targets(_ctx(roster_positions=["QB", "K", "BN"]), 2026)
    assert res["defense"] == [] and len(res["kicker"]) == 3
    assert res["uses_k"] is True and res["uses_def"] is False


def test_offseason_returns_empty():
    res = streaming_targets(_ctx(current_week=0), 2026)
    assert res["defense"] == [] and res["kicker"] == []
    assert res["in_season"] is False
    res = streaming_targets(_ctx(offseason_mode=True), 2026)
    assert res["in_season"] is False


def test_never_raises_on_missing_data(monkeypatch):
    """Schedule/Vegas failures degrade to empty lists, never exceptions."""
    utils_stub = types.ModuleType("utils.utils")
    utils_stub.load_week_sched = lambda season, week: (_ for _ in ()).throw(
        RuntimeError("sleeper down"))
    conds_stub = types.ModuleType("utils.game_conditions")
    conds_stub.build_week_conditions = lambda *a: (_ for _ in ()).throw(
        RuntimeError("odds down"))
    monkeypatch.setitem(sys.modules, "utils.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "utils.game_conditions", conds_stub)
    res = streaming_targets(_ctx(), 2026)
    assert res["defense"] == [] and res["kicker"] == []
