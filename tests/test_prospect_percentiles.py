"""Tests for the percentile display model (prospect page rework).

The board and modal show no numeric prospect grade: rank, a band tag from
the internal score, and component percentiles computed within position
against a reference population of graded prospects. These tests cover the
percentile function (population, position scoping, the <30 hide rule,
rounding), the tag band edges, the in-season data-state check, receipt
generation from stored seasons, and that the shipped page carries no
grade-rendering code for a fixture player.

Synthetic data only; no DB needed.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services import rookie_api as ra
from dashboard_services.pages.rookies_page import build_prospects_body


def _pop_rows(n=40, base=50.0, tested=0, year=2026):
    rows = []
    for i in range(n):
        rows.append({
            "draft_class_year": year,
            "production_score": base + i,
            "efficiency_score": 40.0 + i * 0.5,
            "breakout_profile_score": 30.0 + i,
            "competition_score": 20.0 + i,
            "athleticism_score": 60.0 + i,
            "tested": i < tested,
        })
    return rows


class TestComponentPercentile:
    def test_share_at_or_below(self):
        values = sorted(float(i) for i in range(40))  # 0..39
        assert ra.component_percentile(39.0, values) == 100
        assert ra.component_percentile(19.0, values) == 50
        assert ra.component_percentile(0.0, values) == 3  # 1/40 = 2.5 -> 3

    def test_rounding_half_up(self):
        values = sorted(float(i) for i in range(40))
        # 21 of 40 at or below 20.0 -> 52.5 -> 53 (half up, not banker's)
        assert ra.component_percentile(20.0, values) == 53

    def test_missing_value_is_none(self):
        values = sorted(float(i) for i in range(40))
        assert ra.component_percentile(None, values) is None

    def test_thin_population_hides(self):
        values = sorted(float(i) for i in range(29))
        assert ra.component_percentile(10.0, values) is None
        values30 = sorted(float(i) for i in range(30))
        assert ra.component_percentile(29.0, values30) == 100


class TestReferencePopulation:
    def test_values_classes_and_size(self):
        pop = ra.reference_population("WR", rows=_pop_rows(35, year=2025))
        assert pop["n"] == 35
        assert pop["classes"] == [2025]
        assert len(pop["values"]["production"]) == 35
        assert pop["values"]["production"] == sorted(
            pop["values"]["production"])

    def test_athleticism_population_tested_only(self):
        # Untested players carry a floor athleticism score in v3.x; they
        # must never anchor the athleticism scale.
        pop = ra.reference_population("WR", rows=_pop_rows(40, tested=12))
        assert len(pop["values"]["athleticism"]) == 12
        assert len(pop["values"]["production"]) == 40

    def test_position_scoping_via_injected_rows(self):
        # The DB path filters by position in SQL; injected rows are already
        # one position's rows, and each position builds its own population.
        wr = ra.reference_population("WR", rows=_pop_rows(40, base=50.0))
        te = ra.reference_population("TE", rows=_pop_rows(40, base=10.0))
        assert wr["values"]["production"][0] == 50.0
        assert te["values"]["production"][0] == 10.0


class TestComponentPercentiles:
    def test_untested_player_gets_no_athleticism(self):
        pop = ra.reference_population("WR", rows=_pop_rows(40, tested=40))
        row = {"production_score": 89.0, "efficiency_score": 59.5,
               "breakout_profile_score": 69.0, "competition_score": 59.0,
               "athleticism_score": 99.0,
               "forty_yard": None, "ras_score": None}
        out = ra.component_percentiles(row, pop)
        assert out["athleticism"] is None
        assert out["production"] == 100

    def test_tested_player_gets_athleticism(self):
        pop = ra.reference_population("RB", rows=_pop_rows(40, tested=40))
        row = {"production_score": 50.0, "efficiency_score": 40.0,
               "breakout_profile_score": 30.0, "competition_score": 20.0,
               "athleticism_score": 99.0,
               "forty_yard": 4.39, "ras_score": None}
        out = ra.component_percentiles(row, pop)
        assert out["athleticism"] == 100

    def test_thin_position_population_hides_all(self):
        pop = ra.reference_population("TE", rows=_pop_rows(12))
        row = {"production_score": 89.0, "forty_yard": 4.5}
        out = ra.component_percentiles(row, pop)
        assert all(v is None for v in out.values())


class TestProspectTag:
    @pytest.mark.parametrize("score,label", [
        (99.9, "Elite Prospect"),
        (92.0, "Elite Prospect"),      # top edge of the elite band
        (91.99, "Great Prospect"),
        (80.0, "Great Prospect"),      # top edge of the great band
        (79.9, "Average Prospect"),
        (70.0, "Average Prospect"),    # top edge of the average band
        (69.9, "Day 3"),
        (12.0, "Day 3"),
    ])
    def test_band_edges(self, score, label):
        assert ra.prospect_tag(score)["label"] == label

    def test_no_score_no_tag(self):
        assert ra.prospect_tag(None) is None
        assert ra.prospect_tag("") is None


class TestInSeason:
    def test_partial_current_season(self):
        assert ra.is_in_season(2026, 7, today=date(2026, 10, 17)) is True

    def test_completed_season_is_not_in_season(self):
        assert ra.is_in_season(2026, 13, today=date(2026, 10, 17)) is False
        assert ra.is_in_season(2026, 12, today=date(2026, 10, 17)) is False

    def test_older_latest_season_is_not_in_season(self):
        assert ra.is_in_season(2025, 7, today=date(2026, 10, 17)) is False

    def test_january_belongs_to_the_fall_season(self):
        assert ra.is_in_season(2026, 11, today=date(2027, 1, 10)) is True
        assert ra.is_in_season(2026, 14, today=date(2027, 1, 10)) is False

    def test_missing_data_is_not_in_season(self):
        assert ra.is_in_season(None, 5, today=date(2026, 10, 17)) is False
        assert ra.is_in_season(2026, None, today=date(2026, 10, 17)) is False
        assert ra.is_in_season(2026, 0, today=date(2026, 10, 17)) is False


class TestChipInSeasonSuccession:
    """v3.4: the stored grading-mode flag succeeds the data-state check."""

    def test_flag_true_wins_over_completed_data_state(self):
        # Graded in-season; the stored line later filled to 13 GP without
        # a re-grade. The grading record wins.
        assert ra.chip_in_season(True, 2026, 13,
                                 today=date(2026, 10, 17)) is True

    def test_flag_false_wins_over_partial_data_state(self):
        # Graded complete-mode (e.g. a February re-grade); the chip must
        # not resurrect from a partial-looking stored line.
        assert ra.chip_in_season(False, 2026, 7,
                                 today=date(2026, 10, 17)) is False

    def test_no_flag_falls_back_to_data_state(self):
        assert ra.chip_in_season(None, 2026, 7,
                                 today=date(2026, 10, 17)) is True
        assert ra.chip_in_season(None, 2026, 13,
                                 today=date(2026, 10, 17)) is False
        assert ra.chip_in_season(None, None, None,
                                 today=date(2026, 10, 17)) is False


class TestPoolLine:
    def test_generated_from_actual_classes(self, monkeypatch):
        pops = {
            "QB": {"classes": [2025, 2026, 2027]},
            "RB": {"classes": [2026, 2027]},
            "WR": {"classes": [2027]},
            "TE": {"classes": []},
        }
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, rows=None: pops[pos])
        assert ra.percentile_pool_line() == (
            "Percentiles vs graded prospects at the position, "
            "classes 2025-2027")

    def test_single_class(self, monkeypatch):
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, rows=None: {"classes": [2027]})
        assert ra.percentile_pool_line() == (
            "Percentiles vs graded prospects at the position, classes 2027")

    def test_no_population_no_line(self, monkeypatch):
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, rows=None: {"classes": []})
        assert ra.percentile_pool_line() is None


class TestReceipt:
    SEASONS = [
        {"season": 2024, "games_played": 14, "receiving_yards": 1315,
         "receptions": 79, "yds_per_reception": 16.6,
         "dominator_rating": 0.41},
        {"season": 2025, "games_played": 13, "receiving_yards": 1100,
         "receptions": 70, "yds_per_reception": 15.7,
         "dominator_rating": 0.35},
    ]

    def test_production_receipt_from_stored_season(self):
        rec = ra.strongest_evidence_receipt(
            "WR", self.SEASONS, {},
            {"production": 99, "efficiency": 96, "breakout": 98,
             "competition": 88, "athleticism": None})
        assert rec == ("99th percentile production: 1,315 receiving yards "
                       "in 2024.")

    def test_athleticism_receipt_from_testing(self):
        rec = ra.strongest_evidence_receipt(
            "RB",
            [{"season": 2025, "games_played": 12, "rush_yards": 1400,
              "yds_per_carry": 6.1, "dominator_rating": None}],
            {"forty_yard": 4.39},
            {"production": 80, "efficiency": 85, "athleticism": 97})
        assert rec == "97th percentile athleticism: 4.39 forty-yard dash."

    def test_no_evidence_no_receipt(self):
        assert ra.strongest_evidence_receipt("WR", [], {}, {}) is None
        assert ra.strongest_evidence_receipt(
            "WR", self.SEASONS, {}, {"production": None}) is None


class TestNoGradeRendered:
    """The shipped page must carry no code path that renders a prospect
    grade numeral on the board or in the modal."""

    FIXTURE_SCORE = 91.58

    def test_page_has_no_grade_rendering(self):
        html = build_prospects_body(False)
        # The old grade renderers are gone entirely.
        assert "rkScore1" not in html
        assert "rk-dense" not in html
        assert "rkDense" not in html
        assert 'pm-hero-label">Prospect Score' not in html
        assert "Grade Breakdown" not in html
        # The new display pieces are present.
        assert "rkBoardRow" in html
        assert "rkPoolLine" in html
        assert "tag-chip" in html
        assert "rkHeroReceipt" in html

    def test_board_uses_native_row_classes(self):
        html = build_prospects_body(False)
        # Native players-board grammar, not the bespoke rk-grid fork.
        assert "pr-player-row" in html
        assert "pr-grid-row" in html
        assert "data-pr-sort-col" in html
        assert "pr-tier-divider" in html
        assert "rk-grid-row" not in html
        assert "rkBoardDisc" not in html
        # Continuous list: pagination is gone.
        assert "rkPagination" not in html
        assert "rkLoadPage" not in html
        # Sorts are rank / the three board percentiles / name.
        for opt in ('value="rank"', 'value="prod"', 'value="eff"',
                    'value="brk"', 'value="name"'):
            assert opt in html, opt
        assert 'value="score"' not in html
        assert 'value="mock"' not in html
        # CSV export of the current list is wired in the card header.
        assert "rkExportCSV" in html

    def test_fixture_display_fields_carry_no_grade(self):
        # Everything the board/modal renders for a player is derived from
        # these fields; none of them may contain his grade.
        import json
        pop = ra.reference_population("WR", rows=_pop_rows(40, tested=40))
        row = {"production_score": 89.0, "efficiency_score": 59.5,
               "breakout_profile_score": 69.0, "competition_score": 59.0,
               "athleticism_score": 70.0, "forty_yard": None,
               "ras_score": None, "prospect_score": self.FIXTURE_SCORE}
        display = {
            "tag": ra.prospect_tag(row["prospect_score"]),
            "percentiles": ra.component_percentiles(row, pop),
            "receipt": ra.strongest_evidence_receipt(
                "WR", TestReceipt.SEASONS, {}, {"production": 99}),
        }
        blob = json.dumps(display)
        assert "91.58" not in blob
        assert "91.6" not in blob


class TestModalBandClasses:
    """The modal's band tint is one CSS variable driven by a band class;
    every band the API can emit has a matching class definition."""

    @pytest.mark.parametrize("band", ["elite", "great", "average", "day3"])
    def test_band_class_defined(self, band):
        html = build_prospects_body(False)
        assert f"#rkModalContent.rk-band-{band}" in html
        assert "--band:" in html

    def test_band_applied_from_player_tag(self):
        html = build_prospects_body(False)
        assert "rkApplyBand" in html
        assert "'rk-band-' + tag.band" in html
        # The API's band keys are exactly the four classes above.
        for score in (95.0, 85.0, 75.0, 60.0):
            assert ra.prospect_tag(score)["band"] in (
                "elite", "great", "average", "day3")


class TestContextStripGuards:
    """The context strip omits a side that does not exist: the positional
    #1 has nobody to chase, the last player has nobody chasing him."""

    def test_chasing_side_guarded_by_index(self):
        html = build_prospects_body(False)
        assert "if (idx > 0) html += side('Chasing'" in html
        assert "if (idx < peers.length - 1) html += side('Chased by'" in html


class TestRookieDraftCard:
    """The Rookie Draft hero card renders only from a real ADP row
    (avg_pick on the board payload); no row, no card."""

    def test_card_gated_on_avg_pick(self):
        html = build_prospects_body(False)
        assert "rkRookiePickLabel" in html
        assert "if (rookiePick)" in html
        assert "Dynasty rookie draft" in html


class TestAdvancedPercentileOnly:
    """Advanced rows carry a numeric value for percentile computation;
    the Efficiency/Production rows are component scores, flagged so the
    percentile-only Advanced tab excludes them."""

    SEASONS = [
        {"season": 2025, "games_played": 13, "receiving_yards": 1540,
         "receptions": 110, "targets": 140, "dominator_rating": 0.38,
         "market_share_yards": 0.30, "market_share_tds": 0.28,
         "yds_per_reception": 14.0},
    ]

    def test_component_rows_flagged(self):
        adv = ra._build_advanced_metrics(
            "WR", self.SEASONS, {},
            {"age": 21.0, "efficiency_score": 84.0,
             "production_score": 90.0, "weight_lbs": 205})
        by_label = {a["label"]: a for a in adv}
        assert by_label["Efficiency"]["component"] is True
        assert by_label["Efficiency"]["value"] is None
        assert by_label["Dominator"]["component"] is False
        assert by_label["Dominator"]["value"] == 0.38

    def test_advanced_tab_excludes_component_rows(self):
        html = build_prospects_body(False)
        assert "!a.component" in html


class TestSeasonLines:
    """Per-season percentiles: completed seasons only. An in-progress
    season shows the chip and NO percentile."""

    RATES = [float(x) for x in range(40, 140, 2)]  # 50 completed rates

    def _lines(self, monkeypatch, seasons):
        monkeypatch.setattr(
            ra, "_season_rate_distributions", lambda pos: list(self.RATES))
        return ra._season_lines("WR", seasons)

    def test_in_progress_season_has_no_percentile(self, monkeypatch):
        cur = ra.current_college_season()
        lines = self._lines(monkeypatch, [
            {"season": cur - 1, "games_played": 13,
             "receiving_yards": 1300, "team": "State"},
            {"season": cur, "games_played": 7,
             "receiving_yards": 700, "team": "State"},
        ])
        done, live = lines[0], lines[1]
        assert done["in_progress"] is False
        assert done["percentile"] is not None
        assert live["in_progress"] is True
        assert live["percentile"] is None
        assert live["rate"] == 100.0

    def test_primary_rates_by_position(self):
        qb = {"games_played": 10, "pass_yards": 3000}
        rb = {"games_played": 10, "rush_yards": 800, "receiving_yards": 400}
        wr = {"games_played": 10, "receiving_yards": 1100}
        assert ra._season_primary_rate("QB", qb) == 300.0
        assert ra._season_primary_rate("RB", rb) == 120.0
        assert ra._season_primary_rate("WR", wr) == 110.0
        assert ra._season_primary_rate("WR", {"games_played": 0}) is None


class TestTestingRows:
    """Testing percentiles come from the tested-only pool; a thin pool
    yields a dash (None), and timed tests invert (lower is better)."""

    def test_percentile_from_tested_pool(self, monkeypatch):
        dist = {"forty_yard": [4.40 + 0.01 * i for i in range(40)]}
        monkeypatch.setattr(
            ra, "_testing_distributions", lambda pos: dist)
        rows = ra._testing_rows({"forty_yard": 4.42}, "RB")
        assert rows[0]["label"] == "40-Yard Dash"
        assert rows[0]["result"] == "4.42"
        # 4.42 beats 38 of the 40 tested (only 4.40/4.41 are faster).
        assert rows[0]["percentile"] == 95

    def test_thin_pool_is_a_dash(self, monkeypatch):
        monkeypatch.setattr(
            ra, "_testing_distributions",
            lambda pos: {"forty_yard": [4.5] * 12})
        rows = ra._testing_rows({"forty_yard": 4.5}, "RB")
        assert rows[0]["percentile"] is None

    def test_untested_player_has_no_rows(self, monkeypatch):
        monkeypatch.setattr(ra, "_testing_distributions", lambda pos: {})
        assert ra._testing_rows({}, "RB") == []
        assert ra._testing_rows({"forty_yard": None}, "RB") == []

    def test_broad_jump_format(self):
        assert ra._fmt_broad_jump(124) == "10'4\""
        assert ra._fmt_broad_jump(131.6) == "11'0\""


class TestBreakoutPhrase:
    SEASONS = [{"season": 2024}, {"season": 2025}, {"season": 2026}]

    def test_first_season_true_freshman(self):
        assert ra._breakout_phrase(self.SEASONS, 2024, 18.4) == \
            "Broke out as a true freshman"

    def test_later_seasons(self):
        assert ra._breakout_phrase(self.SEASONS, 2025, 19.4) == \
            "Broke out as a sophomore"
        assert ra._breakout_phrase(self.SEASONS, 2026, 20.4) == \
            "Broke out as a junior"

    def test_no_breakout_no_phrase(self):
        assert ra._breakout_phrase(self.SEASONS, None, None) is None
        assert ra._breakout_phrase(self.SEASONS, 2024, None) is None


class TestPercentileOf:
    def test_lower_better_inverts(self):
        pool = [19.0 + 0.1 * i for i in range(40)]  # breakout ages
        assert ra._percentile_of(19.0, pool, lower_better=True) == 100
        assert ra._percentile_of(22.9, pool, lower_better=True) == 3
        assert ra._percentile_of(19.0, pool) == 3  # same value, raw share
