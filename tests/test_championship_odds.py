"""Championship odds: the separate playoff-bracket sim.

Covers the locked spec: clinched teams are in every run, eliminated
teams read exactly 0.0, bubble teams flex into the open spots weighted
by playoff_pct (which dilutes the current last seed and gives the
first team out a real number), a bye team needs one fewer win than an
equal-strength non-bye team, a locked bye never rotates, title pcts
sum to ~100, the format gate fails closed (Fleaflicker), and the
app-level visibility gating thresholds (page <= 5 weeks, hub <= 2).
Deterministic seeds throughout.
"""
import logging
import math
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from data_building.simulate_playoff_bracket import (  # noqa: E402
    bracket_format,
    simulate_championship_odds,
    simulate_playoff_bracket,
)
from data_building.simulate_playoff_odds import (  # noqa: E402
    mathematical_clinch_elimination,
)

ROOT = Path(__file__).resolve().parents[1]


def _teams(n=12, avg=120.0, std=20.0):
    """12-team league, standings order = roster_id order."""
    return [
        {
            "roster_id": i, "name": f"Team {i}",
            "wins": 13 - i, "losses": i - 1, "ties": 0,
            "pf": 1600.0 - 40 * i, "pa": 1100.0,
            "avg": avg, "std": std,
        }
        for i in range(1, n + 1)
    ]


def _by_rid(rows):
    return {r["roster_id"]: r for r in rows}


# ── Shared clinch / elimination definition ──────────────────────────────────

def test_shared_clinch_elimination_bounds():
    clinched, eliminated = mathematical_clinch_elimination(
        [12.0, 11.0, 2.0, 1.0], [2, 2, 2, 2], playoff_teams=2)
    assert bool(clinched[0]) is True
    assert bool(eliminated[3]) is True
    assert bool(clinched[3]) is False
    assert bool(eliminated[0]) is False


# ── Field flex ──────────────────────────────────────────────────────────────

def test_clinched_always_in_eliminated_exact_zero():
    teams = _teams()
    odds = {}
    for i in range(1, 13):
        odds[i] = {
            "playoff_pct": 100.0 if i <= 3 else (0.0 if i >= 11 else 50.0),
            "bye_pct": 60.0 if i <= 2 else 0.0,
            "first_seed_pct": 70.0 if i == 1 else 0.0,
        }
    rows = _by_rid(simulate_playoff_bracket(
        teams, odds, playoff_teams=6, n_sims=2000, seed=11))
    for rid in (11, 12):
        assert rows[rid]["title_pct"] == 0.0
        assert rows[rid]["in_field_pct"] == 0.0
        assert rows[rid]["title_if_in_pct"] == 0.0
    for rid in (1, 2, 3):
        # In every single run: raw 100 clamps to the 99.9 display cap.
        assert rows[rid]["in_field_pct"] == 99.9
        assert rows[rid]["title_pct"] > 0


def test_bubble_flex_dilutes_last_seed_and_credits_first_out():
    teams = _teams()
    base = {
        "playoff_pct": 100.0, "bye_pct": 90.0, "first_seed_pct": 90.0,
    }
    fixed = {i: dict(base) for i in range(1, 7)}
    fixed[6] = {"playoff_pct": 100.0, "bye_pct": 0.0, "first_seed_pct": 0.0}
    for i in range(7, 13):
        fixed[i] = {"playoff_pct": 0.0, "bye_pct": 0.0, "first_seed_pct": 0.0}
    fixed[1] = {"playoff_pct": 100.0, "bye_pct": 100.0, "first_seed_pct": 100.0}
    fixed[2] = {"playoff_pct": 100.0, "bye_pct": 100.0, "first_seed_pct": 0.0}

    flex = dict(fixed)
    flex[6] = {"playoff_pct": 50.0, "bye_pct": 0.0, "first_seed_pct": 0.0}
    flex[7] = {"playoff_pct": 50.0, "bye_pct": 0.0, "first_seed_pct": 0.0}

    r_fixed = _by_rid(simulate_playoff_bracket(
        teams, fixed, playoff_teams=6, n_sims=4000, seed=5))
    r_flex = _by_rid(simulate_playoff_bracket(
        teams, flex, playoff_teams=6, n_sims=4000, seed=5))

    # The current 6 seed keeps its per-appearance strength but loses
    # overall title share to the bubble team rotating in.
    assert r_flex[6]["title_pct"] < r_fixed[6]["title_pct"] - 1.0
    assert r_flex[7]["title_pct"] > 0.5
    assert abs(r_flex[6]["title_if_in_pct"]
               - r_fixed[6]["title_if_in_pct"]) < 6.0


def test_title_pcts_sum_to_about_100():
    teams = _teams()
    odds = {}
    for i in range(1, 13):
        odds[i] = {
            "playoff_pct": 100.0 if i <= 4 else (0.0 if i >= 12 else 45.0),
            "bye_pct": 50.0 if i <= 3 else 0.0,
            "first_seed_pct": 40.0 if i <= 2 else 0.0,
        }
    rows = simulate_playoff_bracket(
        teams, odds, playoff_teams=6, n_sims=4000, seed=3)
    total = sum(r["title_pct"] for r in rows)
    assert 95.0 <= total <= 105.0


def test_sim_is_deterministic_for_a_seed():
    teams = _teams()
    odds = {i: {"playoff_pct": 55.0, "bye_pct": 20.0, "first_seed_pct": 10.0}
            for i in range(1, 13)}
    a = simulate_playoff_bracket(teams, odds, playoff_teams=6,
                                 n_sims=1000, seed=42)
    b = simulate_playoff_bracket(teams, odds, playoff_teams=6,
                                 n_sims=1000, seed=42)
    assert a == b


# ── Bye / seed flex ─────────────────────────────────────────────────────────

def test_bye_team_needs_one_fewer_win():
    teams = _teams(n=6)
    odds = {i: {"playoff_pct": 100.0, "bye_pct": 0.0, "first_seed_pct": 0.0}
            for i in range(1, 7)}
    odds[1] = {"playoff_pct": 100.0, "bye_pct": 100.0, "first_seed_pct": 100.0}
    odds[2] = {"playoff_pct": 100.0, "bye_pct": 100.0, "first_seed_pct": 0.0}
    rows = _by_rid(simulate_playoff_bracket(
        teams, odds, playoff_teams=6, n_sims=6000, seed=9))
    # All six teams have identical scoring distributions; the bye teams
    # (1 and 2) only need two wins, everyone else needs three.
    assert rows[1]["title_pct"] > rows[3]["title_pct"] + 4.0
    assert rows[2]["title_pct"] > rows[4]["title_pct"] + 4.0
    assert rows[1]["semis_pct"] == 99.9  # starts in the semis every run


def test_locked_bye_does_not_rotate():
    teams = _teams(n=6)
    odds = {i: {"playoff_pct": 100.0, "bye_pct": 50.0, "first_seed_pct": 0.0}
            for i in range(1, 7)}
    odds[1] = {"playoff_pct": 100.0, "bye_pct": 50.0, "first_seed_pct": 100.0}
    rows = _by_rid(simulate_playoff_bracket(
        teams, odds, playoff_teams=6, n_sims=4000, seed=13,
        locked_bye_ids={2}, locked_first_id=1))
    # Team 2 is a bye in every run, so it starts in the semis every run
    # even though its bye weight was only 50%.
    assert rows[2]["semis_pct"] == 99.9
    # Team 3, never a bye here, must win its way in (~half the runs).
    assert rows[3]["semis_pct"] < 80.0


# ── Format gate (fail closed) ───────────────────────────────────────────────

def test_bracket_format_fails_closed_for_fleaflicker():
    ctx = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15}}
    assert bracket_format(ctx, "fleaflicker") is None
    assert bracket_format(ctx, "sleeper") is not None
    verified = {"league_settings": {
        "playoff_teams": 6, "playoff_week_start": 15,
        "playoff_format_verified": True,
    }}
    assert bracket_format(verified, "fleaflicker") is not None
    assert bracket_format({"league_settings": {}}, "sleeper") is None


def test_unknown_format_serves_no_championship_rows():
    ctx = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15}}
    odds = [{"roster_id": 1, "playoff_pct": 90.0, "is_complete": False}]
    assert simulate_championship_odds(ctx, odds, platform="fleaflicker") == []


# ── App-level gating (page <= 5 weeks, hub <= 2) ────────────────────────────

def _load_flags():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.index("# ── Championship odds: separate playoff-bracket sim cache")
    end = src.index("# ── Trade Playoff Impact", start)
    ns = {
        "math": math, "threading": threading, "time": time,
        "logger": logging.getLogger("test-champ"),
        "_PLAYOFF_SIM_CACHE_TTL": 3600,
    }
    exec(src[start:end], ns)
    return ns["_championship_flags"]


FLAGS = _load_flags()
_SLEEPER_CTX = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15}}
_FLEA_CTX = {"league_settings": {"playoff_teams": 6, "playoff_week_start": 15}}
_ROWS = [{"roster_id": 1, "playoff_pct": 80.0, "is_complete": False}]
_DONE = [{"roster_id": 1, "playoff_pct": 100.0, "is_complete": True}]


def test_gating_thresholds():
    f = FLAGS(_SLEEPER_CTX, "sleeper", _ROWS, 6)
    assert f["championship_format_ok"] is True
    assert f["championship_page_visible"] is False
    assert f["championship_hub_visible"] is False

    f = FLAGS(_SLEEPER_CTX, "sleeper", _ROWS, 5)
    assert f["championship_page_visible"] is True
    assert f["championship_hub_visible"] is False

    f = FLAGS(_SLEEPER_CTX, "sleeper", _ROWS, 2)
    assert f["championship_page_visible"] is True
    assert f["championship_hub_visible"] is True

    # Playoffs under way: both surfaces on regardless of weeks left.
    f = FLAGS(_SLEEPER_CTX, "sleeper", _DONE, 0)
    assert f["championship_page_visible"] is True
    assert f["championship_hub_visible"] is True

    # Unverified format hides everything at any week.
    f = FLAGS(_FLEA_CTX, "fleaflicker", _ROWS, 1)
    assert f == {
        "championship_format_ok": False,
        "championship_page_visible": False,
        "championship_hub_visible": False,
    }


# ── History: title snapshot rides the daily playoff-odds row ────────────────

def test_daily_snapshot_records_title_probability(monkeypatch):
    from dashboard_services import playoff_odds_history as H

    writes = []

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return self._rows

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            if "INSERT INTO playoff_odds_daily" in sql:
                writes.append(params)
                return _Cur([])
            if "MAX(snap_date)" in sql:
                return _Cur([{"d": None}])
            return _Cur([])

    monkeypatch.setattr(H, "_DAILY_TABLE_READY", True)
    monkeypatch.setattr(H, "get_conn", lambda: _Conn())
    src = Path(H.__file__).read_text(encoding="utf-8")
    assert "title_probability" in src
    assert "ADD COLUMN IF NOT EXISTS title_probability" in src

    H.record_daily_and_movement(
        "L", 2026,
        [{"roster_id": 1, "playoff_pct": 80.0, "avg_final_wins": 9.0,
          "title_pct": 12.5}],
        write=True,
    )
    assert len(writes) == 1
    assert writes[0][-1] == 12.5  # title_pct lands in title_probability
