"""Frontier 6: league culture model (utils/league_culture.py).

Customs are computed from roster composition and transaction
history only, each signal carries its evidence, and the two claim
bends (QB stash in a hoarding league, DST rental in a streaming
league) fire only on evidenced signals.
"""
from utils.league_culture import (
    claim_culture_note,
    culture_profile,
    qb_hoarding,
    transaction_signals,
)


def _ctx(qbs_per_team, teams=10, sf=False):
    players = {}
    rosters = []
    for t in range(teams):
        pids = []
        for i in range(qbs_per_team):
            pid = f"q{t}_{i}"
            players[pid] = {"position": "QB", "team": "BUF"}
            pids.append(pid)
        for i in range(6):
            pid = f"s{t}_{i}"
            players[pid] = {"position": "WR", "team": "BUF"}
            pids.append(pid)
        rosters.append({"roster_id": t + 1, "players": pids,
                        "starters": pids[:4]})
    ctx = {"players": players, "rosters": rosters,
           "roster_positions": (["QB", "RB", "WR", "TE", "SUPER_FLEX"]
                                if sf else ["QB", "RB", "WR", "TE"])}
    return ctx


def test_qb_hoarding_detected_from_rosters():
    signal = qb_hoarding(_ctx(3))
    assert signal["hoarding"] is True
    assert signal["mean_qbs_per_team"] == 3.0
    assert qb_hoarding(_ctx(2))["hoarding"] is False
    # Superflex leagues legitimately carry more: 3 is normal there.
    assert qb_hoarding(_ctx(3, sf=True))["hoarding"] is False
    assert qb_hoarding(_ctx(4, sf=True))["hoarding"] is True


def test_qb_hoarding_needs_rosters_and_positions():
    assert qb_hoarding({"rosters": []}) is None
    assert qb_hoarding(_ctx(3, teams=3)) is None   # too few rosters


def test_transaction_signals_churn_streaming_trades():
    positions = {"d1": "DEF", "d2": "DEF", "p1": "WR", "p2": "RB"}
    rows = []
    for w in range(1, 5):
        rows.append({"type": "waiver", "week": w,
                     "adds": {"d1": 1}, "drops": {"p1": 1}})
        rows.append({"type": "free_agent", "week": w,
                     "adds": {"p2": 2}, "drops": {"d2": 2}})
    rows.append({"type": "trade", "status": "complete",
                 "adds": {"p1": 3}, "drops": {"p2": 3}})
    out = transaction_signals(rows, num_teams=10,
                              positions_by_id=positions)
    # 16 moves / 10 teams / 4 weeks = 0.4 per team-week: quiet wire.
    assert out["wire_churn"]["level"] == "low"
    # 8 of 16 moves are defenses: a streaming league.
    assert out["dst_streaming"]["streaming"] is True
    assert out["trade_activity"]["completed_trades"] == 1
    assert out["trade_activity"]["active"] is False


def test_transaction_signals_empty_without_rows():
    assert transaction_signals([], num_teams=10) == {}
    assert transaction_signals(None, num_teams=10) == {}


def test_claim_notes_fire_only_on_evidenced_customs():
    profile = culture_profile(_ctx(3))
    note = claim_culture_note(profile, {"position": "QB"})
    assert note and "hoarded" in note
    assert claim_culture_note(profile, {"position": "WR"}) is None
    quiet = culture_profile(_ctx(2))
    assert claim_culture_note(quiet, {"position": "QB"}) is None
    streaming = {"dst_streaming": {"streaming": True}}
    note = claim_culture_note(streaming, {"position": "DEF"})
    assert note and "one-week rental" in note
    assert claim_culture_note({}, {"position": "QB"}) is None
