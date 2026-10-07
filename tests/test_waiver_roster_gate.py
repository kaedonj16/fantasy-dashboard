"""Positional waiver roster gate.

A claim at position P must beat the worst healthy rostered player AT P by 5%
(or fill a starter hole at P) to be worth a nudge. Mirrors Kaedon's superflex
example: rostering Dak, Tua, Darnold and Lawrence, a backup QB like Spencer
Rattler must not be suggested.
"""
import pytest

pd = pytest.importorskip("pandas")

from utils.digest import (
    select_waiver_add,
    waiver_claim_beats_roster,
    waiver_drop_bar_value,
)


def test_drop_bar_is_per_position_and_skips_injured():
    pidx = {
        "q1": {"position": "QB"},
        "q2": {"position": "QB"},
        "q3": {"position": "QB", "injury_status": "IR"},
        "r1": {"position": "RB"},
        "k1": {"position": "K"},
    }
    bars = waiver_drop_bar_value(
        ["q1", "q2", "q3", "r1", "k1"],
        {"q1": 100.0, "q2": 90.0, "q3": 70.0, "r1": 80.0, "k1": 50.0},
        {},
        pidx,
    )
    # IR quarterback excluded (bar is 90, not 70); kickers are not skill pos.
    assert bars == {"QB": 90.0, "RB": 80.0}


def test_drop_bar_empty_roster_fails_open():
    assert waiver_drop_bar_value([], {}, {}, {}) == {}


def test_claim_beats_roster_truth_table():
    # No bar for the position: fail open.
    assert waiver_claim_beats_roster(10.0, None, 0.0)
    # Starter hole bypasses the margin check.
    assert waiver_claim_beats_roster(10.0, 100.0, 1.0)
    # Real upgrade clears the 5% margin.
    assert waiver_claim_beats_roster(106.0, 100.0, 0.0)
    # Lateral moves do not: equal, below, and exactly at the margin.
    assert not waiver_claim_beats_roster(100.0, 100.0, 0.0)
    assert not waiver_claim_beats_roster(90.0, 100.0, 0.0)
    assert not waiver_claim_beats_roster(105.0, 100.0, 0.0)


def _sf_rows(roster_values, candidate_value):
    """Superflex rows: rostered QBs plus a Rattler-style backup QB claim."""
    pidx = {}
    rows = []
    for pid, val in roster_values.items():
        pidx[pid] = {"position": "QB", "injury_status": ""}
        rows.append({
            "id": pid, "name": f"QB {pid}", "position": "QB", "team": "DAL",
            "value": val, "pos_rank": 10, "age": 28,
        })
    pidx["rattler"] = {"position": "QB", "injury_status": ""}
    rows.append({
        "id": "rattler", "name": "Spencer Rattler", "position": "QB",
        "team": "NO", "value": candidate_value, "pos_rank": 22, "age": 24,
    })
    roster_positions = [
        "QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
        "BN", "BN", "BN", "BN", "BN", "BN", "BN",
    ]
    return rows, pidx, roster_positions


def _select(rows, pidx, roster_positions, roster_pids):
    return select_waiver_add(
        rows,
        {str(p) for p in roster_pids},
        value_key="value",
        fallback_key="value",
        is_redraft=False,
        is_sf=True,
        n_teams=12,
        roster_players=list(roster_pids),
        roster_positions=roster_positions,
        pidx=pidx,
        min_value=10.0,
    )


def test_stacked_qb_room_filters_backup_qb():
    # Dak/Tua/Darnold/Lawrence rostered; Rattler (60) is below the worst
    # rostered QB (85 * 1.05 = 89.25), so no claim is surfaced.
    roster_values = {"dak": 100.0, "tua": 95.0, "darnold": 90.0, "lawrence": 85.0}
    rows, pidx, positions = _sf_rows(roster_values, candidate_value=60.0)
    assert _select(rows, pidx, positions, list(roster_values)) is None


def test_great_claim_beats_stacked_room():
    # Same room, but a claim at 95 clears the worst-QB bar with margin.
    roster_values = {"dak": 100.0, "tua": 95.0, "darnold": 90.0, "lawrence": 85.0}
    rows, pidx, positions = _sf_rows(roster_values, candidate_value=95.0)
    hit = _select(rows, pidx, positions, list(roster_values))
    assert hit is not None
    assert hit["player_id"] == "rattler"


def test_qb_hole_surfaces_backup_despite_lower_value():
    # One QB rostered with a QB + SUPER_FLEX to fill: the hole bypasses the
    # margin check even though 60 < 100 * 1.05.
    roster_values = {"dak": 100.0}
    rows, pidx, positions = _sf_rows(roster_values, candidate_value=60.0)
    hit = _select(rows, pidx, positions, list(roster_values))
    assert hit is not None
    assert hit["player_id"] == "rattler"
    assert hit["starter_gap"] > 0
