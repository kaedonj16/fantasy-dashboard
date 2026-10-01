"""Guards the true-strength power rankings rebuild.

In-season rank is 0.80 all-play + 0.20 injury-adjusted roster strength
(``starter_value_available``), not pure all-play. Out / IR / Doubtful players
cannot be fielded: their slots refill from the bench via the same slot-legal
derivation; Questionable counts at full value. With no injury data anywhere,
the adjusted value equals the raw starter value exactly.
"""
import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.power_score import (
    blended_team_scores,
    starter_lineup_value,
)

# available_starter_lineup_value / player_is_unavailable are imported inside
# the availability tests so the blend-ordering tests below still fail with
# plain assertion errors on pre-rebuild main (not a collection error).


def _lookup(rows):
    return {r["player_id"]: r for r in rows}


def _mv(pid, pos, redraft):
    return {"player_id": pid, "position": pos, "redraft_value_1qb": redraft}


# ── availability unit behavior ──────────────────────────────────────────────

def test_unavailable_statuses():
    from dashboard_services.power_score import player_is_unavailable

    assert player_is_unavailable("IR")
    assert player_is_unavailable("Out")
    assert player_is_unavailable("doubtful")
    assert not player_is_unavailable("Questionable")
    assert not player_is_unavailable("")
    assert not player_is_unavailable(None)


def test_ir_star_refills_from_bench():
    from dashboard_services.power_score import available_starter_lineup_value

    # 9 players; the star is on IR. Raw takes the top 8 (star included);
    # adjusted excludes the star and the 9th-best fills the gap.
    rows = [_mv("star", "RB", 900), _mv("q1", "QB", 400)]
    rows += [_mv(f"p{i}", "WR", v) for i, v in enumerate(
        (300, 280, 260, 240, 220, 200, 180), start=1)]
    lookup = _lookup(rows)
    pids = ["star", "q1"] + [f"p{i}" for i in range(1, 8)]

    raw = starter_lineup_value(pids, lookup, redraft_key="redraft_value_1qb")
    assert raw == 900 + 400 + 300 + 280 + 260 + 240 + 220 + 200

    adj = available_starter_lineup_value(
        pids, lookup, redraft_key="redraft_value_1qb",
        injury_by_pid={"star": "IR"},
    )
    assert adj == 400 + 300 + 280 + 260 + 240 + 220 + 200 + 180
    assert adj < raw


def test_questionable_counts_at_full_value():
    from dashboard_services.power_score import available_starter_lineup_value

    rows = [_mv("star", "RB", 900), _mv("q1", "QB", 400)]
    lookup = _lookup(rows)
    raw = starter_lineup_value(["star", "q1"], lookup, redraft_key="redraft_value_1qb")
    adj = available_starter_lineup_value(
        ["star", "q1"], lookup, redraft_key="redraft_value_1qb",
        injury_by_pid={"star": "Questionable"},
    )
    assert adj == raw


def test_no_injury_data_falls_back_to_raw_exactly():
    from dashboard_services.power_score import available_starter_lineup_value

    rows = [_mv("star", "RB", 900), _mv("q1", "QB", 400)]
    lookup = _lookup(rows)
    raw = starter_lineup_value(["star", "q1"], lookup, redraft_key="redraft_value_1qb")
    assert available_starter_lineup_value(
        ["star", "q1"], lookup, redraft_key="redraft_value_1qb",
        injury_by_pid=None,
    ) == raw
    assert available_starter_lineup_value(
        ["star", "q1"], lookup, redraft_key="redraft_value_1qb",
        injury_by_pid={},
    ) == raw
    # Designations present but blank for everyone = no injury data.
    assert available_starter_lineup_value(
        ["star", "q1"], lookup, redraft_key="redraft_value_1qb",
        injury_by_pid={"star": "", "q1": ""},
    ) == raw


def test_unfillable_slot_contributes_zero():
    from dashboard_services.power_score import available_starter_lineup_value

    # Star on IR with no bench behind him: the adjusted lineup is just the
    # healthy players; nothing is invented for the empty slot.
    rows = [_mv("star", "RB", 900), _mv("q1", "QB", 400)]
    lookup = _lookup(rows)
    adj = available_starter_lineup_value(
        ["star", "q1"], lookup, redraft_key="redraft_value_1qb",
        roster_positions=["QB", "RB", "RB"],
        injury_by_pid={"star": "Out"},
    )
    # QB fills; only one RB is fieldable so the second RB slot is empty.
    assert adj == 400


# ── blend ordering ─────────────────────────────────────────────────────────

def _team(name, ap, avg, raw_value, avail_value=None):
    t = {
        "team": name, "avg": avg, "luck_adj_win": ap,
        "starter_value": raw_value, "momentum": 0.0,
        "consistency": 0.0, "sos": 0.5,
    }
    if avail_value is not None:
        t["starter_value_available"] = avail_value
    return t


def test_equal_all_play_healthy_outranks_ir_depleted():
    # Two teams, identical all-play and PPG; the first is listed first and has
    # its star on IR. Old model: all-play + PPG only, so the injured team
    # stays on top. New model: healthy roster pulls ahead.
    injured = _team("Depleted", 0.55, 100.0, 2000.0, 900.0)
    healthy = _team("Healthy", 0.55, 100.0, 2000.0, 2000.0)
    ranked = blended_team_scores([dict(injured), dict(healthy)], phase="mid")
    assert ranked[0]["team"] == "Healthy"
    assert ranked[0]["power_score"] > ranked[1]["power_score"]


def test_depressed_all_play_healthy_roster_pulls_up():
    # Team A has worse all-play but a far stronger healthy roster; team B has
    # better all-play but an IR-depleted roster. The 0.20 value term pulls A
    # ahead of B. Pure all-play would rank B first.
    a = _team("NowHealthy", 0.50, 100.0, 2000.0, 2000.0)
    b = _team("Depleted", 0.52, 100.0, 900.0, 800.0)
    c = _team("Anchor", 0.60, 100.0, 900.0, 800.0)
    ranked = blended_team_scores([dict(a), dict(b), dict(c)], phase="mid")
    order = [t["team"] for t in ranked]
    assert order.index("NowHealthy") < order.index("Depleted")


def test_ppg_tiebreak_unchanged():
    a = _team("LowPPG", 0.5, 90.0, 100.0, 100.0)
    b = _team("HighPPG", 0.5, 140.0, 100.0, 100.0)
    ranked = blended_team_scores([dict(a), dict(b)], phase="early")
    assert ranked[0]["team"] == "HighPPG"


def test_preseason_uses_adjusted_value():
    # Team X's raw starter value is inflated by an IR'd star; its adjusted
    # value is much lower. Preseason must rank on the adjusted number.
    x = _team("IRHeavy", 0.5, 100.0, 2000.0, 1200.0)
    y = _team("Real", 0.5, 100.0, 1500.0, 1500.0)
    ranked = blended_team_scores([dict(x), dict(y)], phase="preseason")
    assert ranked[0]["team"] == "Real"


def test_value_term_falls_back_to_raw_starter_value():
    # Historical rows / callers without injury data carry no
    # starter_value_available; the blend must still use raw starter_value.
    a = _team("A", 0.5, 100.0, 300.0)
    b = _team("B", 0.5, 100.0, 100.0)
    ranked = blended_team_scores([dict(a), dict(b)], phase="mid")
    assert ranked[0]["team"] == "A"
