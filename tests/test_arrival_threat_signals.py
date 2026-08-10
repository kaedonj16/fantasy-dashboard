"""A veteran's threat to an incumbent is not just his old-team box score. A
backup signed to start (thin prior usage) is a real threat, and two proxies for
the role he was brought in to fill should coalign: the contract the market gave
him (A) and the size of the starting job that opened on his new team (B). These
tests lock that behavior in the pure scoring functions.
"""
import pytest

comp = pytest.importorskip("data_building.breakout_engine.components")


def _backup_te(contract=None):
    """A TE with a backup's prior-season usage (thin targets, part-time snaps)."""
    arr = {
        "player_id": "vetTE",
        "player_name": "Backup TE",
        "change_type": "free_agent",
        "last_season_targets": 40,
        "last_season_carries": 0,
        "last_season_snap_share": 0.35,
    }
    if contract is not None:
        arr["contract_metadata"] = contract
    return arr


STARTER_DEAL = {"apy": 8_000_000, "guaranteed": 10_000_000, "years": 3}
BIG_OPENING = {"targets": 90, "carries": 0, "snap_share": 0.5}


def test_starter_contract_lifts_a_thin_usage_vet():
    # Contract (A) is the market's proxy for the role he was signed into: a
    # backup's box score alone reads as "low", the contract makes him a real
    # threat.
    no_contract, _ = comp._calculate_arrival_role_threat("TE", _backup_te())
    with_contract, det = comp._calculate_arrival_role_threat("TE", _backup_te(STARTER_DEAL))
    assert with_contract > no_contract
    assert with_contract >= 0.48           # at least "medium"
    assert det["contract_signal"] > 0.5


def test_vacated_opening_does_not_overthreaten_noncredible_depth():
    # B alone must not over-threaten: a low-usage arrival with no contract signal
    # stepping into a big opening is NOT credible to claim it, so the room's size
    # is gated by credibility and the threat stays low.
    depth = {
        "player_id": "campBody", "player_name": "Camp Body",
        "change_type": "free_agent",
        "last_season_targets": 8, "last_season_carries": 0,
        "last_season_snap_share": 0.10,
    }
    threat, _ = comp._calculate_arrival_role_threat("TE", depth, BIG_OPENING)
    assert threat < 0.25                    # "low"/"minimal", not inflated by B


def test_contract_and_opening_coalign_to_boost_confidence():
    contract_only, _ = comp._calculate_arrival_role_threat("TE", _backup_te(STARTER_DEAL))
    both, _ = comp._calculate_arrival_role_threat("TE", _backup_te(STARTER_DEAL), BIG_OPENING)
    # When A (contract) and B (opening) agree, the threat is at least as strong as
    # either signal alone.
    assert both >= contract_only


def test_proven_usage_is_a_floor():
    # A proven producer changing teams stays a strong threat on his own track
    # record even with no contract data.
    stud = {
        "player_id": "studWR", "player_name": "Stud WR",
        "change_type": "free_agent",
        "last_season_targets": 150, "last_season_carries": 0,
        "last_season_snap_share": 0.9,
    }
    threat, _ = comp._calculate_arrival_role_threat("WR", stud)
    assert threat >= 0.5


def test_penalty_end_to_end_uses_contract_and_vacated():
    # Through the public penalty function: a starter-contract TE arrival produces
    # a materially larger penalty than the same arrival with no contract data.
    def _run(arr):
        cache = {("CLE", "WR"): [], ("CLE", "TE"): [arr]}
        vac = {("CLE", "TE"): BIG_OPENING, ("CLE", "WR"): {}}
        return comp.calculate_competition_added_penalty(
            "incumbent", "CLE", "TE", 2025,
            arrivals_cache=cache, vacated_cache=vac,
        )

    pen_plain, _ = _run(_backup_te())
    pen_contract, det = _run(_backup_te(STARTER_DEAL))
    assert abs(pen_contract) > abs(pen_plain)
    assert det["threats_added"][0]["threat_level"] in ("medium", "high")
