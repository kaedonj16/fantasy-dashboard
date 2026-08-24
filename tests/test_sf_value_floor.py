"""Superflex non-QB value floor: guard against the market-inverting bug where
every non-QB's SF value was floored at its FULL 1QB value.

The floor must (a) protect against a cratered SF read but (b) never inflate a
non-QB above its true market SF value. Part two checks the market premise the
fix relies on — elite non-QBs are LOWER in SF than 1QB — against the checked-in
FantasyCalc feeds, so the assumption can't silently rot.
"""
import csv
import os

from data_building.value_guardrails import sf_nonqb_floor, SF_NONQB_FLOOR_RATIO

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_market_sf_value_is_preserved_not_inflated():
    # An elite RB whose market SF sits ~0.92x its 1QB value (the real market
    # position). The floor must NOT lift it up to the 1QB value.
    v1qb = 1050.0
    sf_market = 966.0  # ~0.92x
    out = sf_nonqb_floor(sf_market, v1qb)
    assert out == sf_market, "market SF value must be preserved, not floored up"
    assert out < v1qb, "a non-QB must be allowed to sit below its 1QB value in SF"


def test_floor_catches_a_cratered_read():
    # A bad/missing DP-2QB read craters SF to 0.4x — the safety net lifts it to
    # the ratio, but only to the ratio, never the full 1QB value.
    v1qb = 1000.0
    out = sf_nonqb_floor(400.0, v1qb)
    assert out == SF_NONQB_FLOOR_RATIO * v1qb
    assert out < v1qb


def test_floor_never_returns_full_1qb_when_sf_below_it():
    # The old bug: max(sf, 1qb) returned the full 1QB value. The floor ratio is
    # < 1, so a non-QB whose SF is below its 1QB value never gets the full 1QB.
    assert SF_NONQB_FLOOR_RATIO < 1.0
    for sf in (0.0, 100.0, 500.0, 900.0):
        assert sf_nonqb_floor(sf, 1000.0) < 1000.0


def _load(name):
    path = os.path.join(_DATA, name)
    with open(path, newline="", encoding="utf-8") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def test_market_premise_elite_nonqbs_are_lower_in_sf():
    """Ground truth: in the FantasyCalc feeds every elite non-QB is worth LESS in
    SF than 1QB (QBs absorb the value). This is exactly what the old floor fought,
    so if this ever flips the fix's premise needs revisiting."""
    oneqb = _load("fantasycalc_api_values.csv")
    sf = _load("fantasycalc_sf_api_values.csv")
    for name in ("Bijan Robinson", "Jahmyr Gibbs", "Ja'Marr Chase"):
        if name in oneqb and name in sf:
            v1 = float(oneqb[name]["value"])
            vs = float(sf[name]["value"])
            assert vs < v1, f"{name}: SF {vs} should be < 1QB {v1} in the market"
            # and the ratio is close to the floor, never wildly below it
            assert vs / v1 > SF_NONQB_FLOOR_RATIO - 0.05


def test_market_premise_qb_tops_sf_not_1qb():
    """A top QB (Josh Allen) is the SF #1 but not the 1QB #1 — confirming the SF
    board should be QB-led, which the old non-QB floor broke by lifting RBs up."""
    oneqb = _load("fantasycalc_api_values.csv")
    sf = _load("fantasycalc_sf_api_values.csv")
    if "Josh Allen" in sf and "Josh Allen" in oneqb:
        assert int(sf["Josh Allen"]["overall_rank"]) == 1
        assert int(oneqb["Josh Allen"]["overall_rank"]) > 1
