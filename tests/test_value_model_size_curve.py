"""The value board is market-only: no usage engine, and the league-size columns
come from FantasyCalc's per-numTeams values.

Guards two things:
  1. The multi-size FantasyCalc scrape round-trips (write -> load) with a
     value_{n} / sf_value_{n} column per league size — this is the market source
     for the size curve (value_n = base * FC@n / FC@10).
  2. The value board no longer references the retired usage engine, so the
     dependency can't silently creep back in.
"""
import os

import pytest

pytest.importorskip("requests")  # external_values_scraper imports requests at load

from data_building.external_data import external_values_scraper as evs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_size_values_roundtrip(tmp_path, monkeypatch):
    # Fake FantasyCalc: value scales with league size (so ratios differ) and SF
    # (numQbs=2) is richer than 1QB — enough to exercise every column.
    def fake_fetch(*, is_dynasty=True, num_qbs=1, num_teams=None, ppr=1.0):
        base = 100 * int(num_teams)
        mult = 2 if num_qbs == 2 else 1
        return [
            {"player": {"sleeperId": 111}, "value": base * mult},
            {"player": {"sleeperId": 222}, "value": base * mult / 2},
        ]

    monkeypatch.setattr(evs, "fetch_fantasycalc_api_values", fake_fetch)
    out = tmp_path / "fc_size.csv"
    n = evs.write_fantasycalc_size_values(sizes=(8, 10, 12), out_csv=out)
    assert n == 2

    rows = evs.load_fantasycalc_size_values(out)
    assert rows is not None and len(rows) == 2
    by = {r["sleeper_id"]: r for r in rows}

    # Per-size 1QB values (base = 100 * numTeams).
    assert float(by["111"]["value_8"]) == 800
    assert float(by["111"]["value_12"]) == 1200
    # SF is 2x in the fake, and every size has both a 1QB and SF column.
    assert float(by["111"]["sf_value_8"]) == 1600
    for size in (8, 10, 12):
        assert f"value_{size}" in by["111"]
        assert f"sf_value_{size}" in by["111"]

    # The size ratio that drives the curve: FC@12 / FC@10 for player 111.
    ratio = float(by["111"]["value_12"]) / float(by["111"]["value_10"])
    assert ratio == pytest.approx(1200 / 1000)


def test_missing_size_file_loads_none(tmp_path):
    assert evs.load_fantasycalc_size_values(tmp_path / "nope.csv") is None


def test_value_board_has_no_usage_engine():
    # The board is now market-only (FantasyCalc + DynastyProcess). Guard against
    # the retired usage-engine dependency creeping back into the blend.
    src = open(os.path.join(_ROOT, "data_building", "value_model_training.py"),
               encoding="utf-8").read()
    for banned in ("engine_values.csv", "engine_size_map", "sf_engine_map", "engine_1qb_map"):
        assert banned not in src, f"retired engine reference reintroduced: {banned}"
