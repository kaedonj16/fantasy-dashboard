"""Totals behind the per-X rates (Kaedon 2026-10-15).

"if I have any kind of 'x per x' stats, I should be able to see totals
too, like YACatch" (YAC / Reception). Every rate below gains a selectable
season-total companion in LEADERBOARD_METRICS, adjacent to its rate, with
the rate's category / positions / min_vol:

- stored numerator columns get a plain catalog entry (the key IS the
  column): yards_after_catch, avoided_tackles, explosive_runs_10_plus,
  ngs_rush_yards_over_expected;
- rates stored without a yardage total are derived with the existing
  computed_sql pattern (rate x volume), following total_pass_yards /
  total_yards: total_rush_yards, total_rec_yards, yards_after_contact,
  yards_before_contact.

No total is faked: rec_broken_tackles_per_reception keeps NO total
companion because only the (rounded) rate is stored - its numerator
count is never written to the snapshot.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS, PRO_METRICS

# rate key -> (total key, computed_sql or None for a direct stored column)
PAIRS = {
    "yards_after_catch_per_reception": ("yards_after_catch", None),
    "avoided_tackles_per_carry": ("avoided_tackles", None),
    "explosive_run_rate": ("explosive_runs_10_plus", None),
    "ngs_rush_yards_over_expected_per_att": ("ngs_rush_yards_over_expected", None),
    "yards_per_carry": (
        "total_rush_yards",
        "ROUND(m.yards_per_carry * m.total_carries)",
    ),
    "yards_per_reception": (
        "total_rec_yards",
        "ROUND(m.yards_per_reception * m.total_receptions)",
    ),
    "yac_per_carry": (
        "yards_after_contact",
        "ROUND(m.yac_per_carry * m.total_carries)",
    ),
    "ybc_per_carry": (
        "yards_before_contact",
        "ROUND(m.ybc_per_carry * m.total_carries)",
    ),
}

NEW_TOTAL_KEYS = [total for total, _ in PAIRS.values()]


def test_every_rate_has_its_total_companion():
    for rate, (total, _sql) in PAIRS.items():
        assert rate in LEADERBOARD_METRICS, f"rate missing: {rate}"
        assert total in LEADERBOARD_METRICS, f"total missing: {total}"


def test_totals_match_their_rate_spec():
    for rate, (total, _sql) in PAIRS.items():
        rspec = LEADERBOARD_METRICS[rate]
        tspec = LEADERBOARD_METRICS[total]
        assert tspec["label"] and tspec["label"] != rspec["label"]
        assert tspec["category"] == rspec["category"], total
        assert tspec["positions"] == rspec["positions"], total
        assert tspec.get("min_vol") == rspec.get("min_vol"), total
        assert tspec.get("desc"), total
        # A total is a counting stat, never itself a rate.
        assert not tspec.get("pct"), total
        assert not tspec.get("efficiency"), total


def test_total_derivations():
    for _rate, (total, sql) in PAIRS.items():
        tspec = LEADERBOARD_METRICS[total]
        if sql is None:
            # Direct stored column: no computed_sql, the key is the column.
            assert "computed_sql" not in tspec, total
        else:
            assert tspec["computed_sql"] == sql, total
            # The null gate names both inputs so partial rows stay NULL.
            assert tspec["computed_null"].count("IS NOT NULL") >= 2, total


def test_yardage_and_count_totals_are_integer():
    for total in (
        "yards_after_catch",
        "avoided_tackles",
        "explosive_runs_10_plus",
        "total_rush_yards",
        "total_rec_yards",
        "yards_after_contact",
        "yards_before_contact",
    ):
        assert LEADERBOARD_METRICS[total].get("integer") is True, total
    # RYOE is a decimal EPA-style total (and can be negative), not a count.
    assert not LEADERBOARD_METRICS["ngs_rush_yards_over_expected"].get("integer")


def test_totals_sit_adjacent_to_their_rate():
    keys = list(LEADERBOARD_METRICS)
    for rate, (total, _sql) in PAIRS.items():
        if total == "total_rec_yards":
            continue  # placed with Rec Yds/G, its per-game twin (checked below)
        assert keys.index(total) == keys.index(rate) + 1, (rate, total)
    assert keys.index("total_rec_yards") == keys.index("rec_yards_per_game") + 1


def test_totals_are_free_like_their_rates():
    for total in NEW_TOTAL_KEYS:
        assert total not in PRO_METRICS, total


def test_totals_appear_in_the_metric_pickers():
    # The table picker (and, from the same spec, the graph X/Y/bubble
    # pickers) renders one <option> per catalog key.
    html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    for total in NEW_TOTAL_KEYS:
        assert f'<option value="{total}">' in html, total


def test_no_faked_total_for_rec_broken_tackles():
    # Only the rounded rate is stored; the numerator count is not, so no
    # "Broken Tackles (Rec)" total may be invented from it.
    assert "rec_broken_tackles" not in LEADERBOARD_METRICS
    labels = [spec.get("label") for spec in LEADERBOARD_METRICS.values()]
    assert "Broken Tackles (Rec)" not in labels
