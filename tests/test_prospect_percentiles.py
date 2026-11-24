"""Tests for the percentile display model (prospect page rework).

The board and modal show rank, the numeric prospect grade, a band tag
from the internal score, and component percentiles computed within
position against a reference population of graded prospects. These tests
cover the percentile function (population, position scoping, the <30
hide rule, rounding), the tag band edges, the in-season data-state
check, receipt generation from stored seasons, and that the shipped
page carries the grade-rendering code (board Grade column, modal
Prospect Grade card).

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
            # Raw-driver anchors: one best completed-season rate and
            # best Advanced marks per drafted player, as the backfill
            # stores them. Production bests run 40..(39+n); the marks
            # step evenly so expected percentiles are exact.
            "season_rates": [40.0 + i],
            "advanced_marks": {
                "Yds/Rec": 10.0 + 0.25 * i,
                "Yds/Carry": 3.0 + 0.25 * i,
                "Yds/Att": 6.0 + 0.25 * i,
                "Dominator": round(0.10 + 0.01 * i, 2),
            },
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
        pop = ra.reference_population("WR", ref_rows=_pop_rows(35, year=2025))
        assert pop["n"] == 35
        assert pop["classes"] == [2025]
        assert len(pop["values"]["production"]) == 35
        assert pop["values"]["production"] == sorted(
            pop["values"]["production"])

    def test_athleticism_population_tested_only(self):
        # Untested players carry a floor athleticism score in v3.x; they
        # must never anchor the athleticism scale.
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40, tested=12))
        assert len(pop["values"]["athleticism"]) == 12
        assert len(pop["values"]["production"]) == 40

    def test_position_scoping_via_injected_rows(self):
        # The DB path filters by position in SQL; injected rows are already
        # one position's rows, and each position builds its own population.
        wr = ra.reference_population("WR", ref_rows=_pop_rows(40, base=50.0))
        te = ra.reference_population("TE", ref_rows=_pop_rows(40, base=10.0))
        assert wr["values"]["production"][0] == 50.0
        assert te["values"]["production"][0] == 10.0


class TestComponentPercentiles:
    def test_untested_player_gets_no_athleticism(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40, tested=40))
        row = {"position": "WR",
               "production_score": 89.0, "efficiency_score": 59.5,
               "breakout_profile_score": 69.0, "competition_score": 59.0,
               "athleticism_score": 99.0,
               "forty_yard": None, "ras_score": None}
        seasons = [{"season": 2024, "games_played": 13,
                    "receiving_yards": 780, "yds_per_reception": 13.0,
                    "dominator_rating": 0.25}]
        out = ra.component_percentiles(row, pop, seasons=seasons)
        assert out["athleticism"] is None
        # Production is a raw-driver percentile now: his best rate is
        # 60.0 rec yds/gm against bests of 40..79 -> 21/40 -> 53. The
        # old score share would have printed 100 for his 89.0 score.
        assert out["production"] == 53
        # Competition stays score-based: 59.0 tops the 20..59 pool.
        assert out["competition"] == 100

    def test_tested_player_gets_athleticism(self):
        pop = ra.reference_population("RB", ref_rows=_pop_rows(40, tested=40))
        row = {"production_score": 50.0, "efficiency_score": 40.0,
               "breakout_profile_score": 30.0, "competition_score": 20.0,
               "athleticism_score": 99.0,
               "forty_yard": 4.39, "ras_score": None}
        out = ra.component_percentiles(row, pop)
        assert out["athleticism"] == 100

    def test_thin_position_population_hides_all(self):
        pop = ra.reference_population("TE", ref_rows=_pop_rows(12))
        row = {"production_score": 89.0, "forty_yard": 4.5}
        out = ra.component_percentiles(row, pop)
        assert all(v is None for v in out.values())


class TestRawDriverPercentiles:
    """Production / efficiency / breakout percentiles rank the raw
    driver, not the clipped component score (2026-11-20). The pool
    plateaus at the score cap (many drafted players store a 100), so
    two maxed scores must still separate on the stat itself."""

    def test_population_drivers_built_from_reference_rows(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40))
        assert pop["drivers"]["production"] == [40.0 + i for i in range(40)]
        assert pop["drivers"]["efficiency"] == [
            10.0 + 0.25 * i for i in range(40)]
        assert pop["drivers"]["breakout"] == [
            round(0.10 + 0.01 * i, 2) for i in range(40)]

    def test_production_driver_is_best_season_not_every_season(self):
        refs = _pop_rows(39)
        refs.append({
            "draft_class_year": 2026, "production_score": 1.0,
            "efficiency_score": None, "breakout_profile_score": None,
            "competition_score": None, "athleticism_score": None,
            "tested": False, "advanced_marks": {},
            "season_rates": [55.0, 82.0, 61.0],
        })
        pop = ra.reference_population("WR", ref_rows=refs)
        # One value per drafted player: the appended row's three rates
        # contribute only its best (82.0, above the whole fixture
        # range), exactly once; 40 values for 40 players.
        assert pop["drivers"]["production"].count(82.0) == 1
        assert pop["drivers"]["production"][-1] == 82.0
        assert len(pop["drivers"]["production"]) == 40

    def test_plateau_separation(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40))
        base = {"position": "WR", "production_score": 100.0,
                "efficiency_score": 100.0, "breakout_profile_score": 100.0,
                "competition_score": 50.0, "athleticism_score": None,
                "forty_yard": None, "ras_score": None}
        star = dict(base)
        star_seasons = [{"season": 2024, "games_played": 13,
                         "receiving_yards": 1014,
                         "yds_per_reception": 18.0,
                         "dominator_rating": 0.40}]
        other = dict(base)
        other_seasons = [{"season": 2024, "games_played": 13,
                          "receiving_yards": 650,
                          "yds_per_reception": 12.0,
                          "dominator_rating": 0.20}]
        a = ra.component_percentiles(star, pop, seasons=star_seasons)
        b = ra.component_percentiles(other, pop, seasons=other_seasons)
        # Identical (maxed) scores, different raw drivers: the three
        # percentiles separate, at the driver-implied values.
        assert a["production"] == 98 and b["production"] == 28
        assert a["efficiency"] == 83 and b["efficiency"] == 23
        assert a["breakout"] == 78 and b["breakout"] == 28
        # Score-based components are untouched by the seasons.
        assert a["competition"] == b["competition"] == 78

    def test_driver_mapping_by_position(self):
        wr = ra._component_drivers("WR", [
            {"season": 2024, "games_played": 12, "receiving_yards": 1200,
             "yds_per_reception": 15.0, "dominator_rating": 0.33}])
        assert wr == {"production": 100.0, "efficiency": 15.0,
                      "breakout": 0.33}
        rb = ra._component_drivers("RB", [
            {"season": 2024, "games_played": 12, "rush_yards": 900,
             "receiving_yards": 300, "yds_per_carry": 5.5,
             "dominator_rating": 0.30}])
        assert rb == {"production": 100.0, "efficiency": 5.5,
                      "breakout": 0.30}
        qb = ra._component_drivers("QB", [
            {"season": 2024, "games_played": 12, "pass_yards": 3600,
             "yds_per_attempt": 9.2, "dominator_rating": 0.50}])
        assert qb == {"production": 300.0, "efficiency": 9.2,
                      "breakout": 0.50}
        te = ra._component_drivers("TE", [
            {"season": 2024, "games_played": 12, "receiving_yards": 960,
             "yds_per_reception": 12.0, "dominator_rating": 0.22}])
        assert te == {"production": 80.0, "efficiency": 12.0,
                      "breakout": 0.22}

    def test_qb_breakout_hides_without_dominator_pool(self):
        # QBs carry no Dominator advanced mark, so the historical pool
        # has no breakout distribution: the percentile hides rather
        # than falling back to the clipped score.
        refs = [{
            "draft_class_year": 2020, "production_score": 50.0 + i,
            "efficiency_score": 50.0, "breakout_profile_score": 50.0,
            "competition_score": 50.0, "athleticism_score": None,
            "tested": False, "season_rates": [200.0 + i],
            "advanced_marks": {"Yds/Att": 6.0 + 0.25 * i},
        } for i in range(40)]
        pop = ra.reference_population("QB", ref_rows=refs)
        assert pop["drivers"]["breakout"] == []
        row = {"position": "QB", "competition_score": 50.0,
               "forty_yard": None, "ras_score": None}
        seasons = [{"season": 2024, "games_played": 12,
                    "pass_yards": 3600, "yds_per_attempt": 9.0,
                    "dominator_rating": 0.50}]
        out = ra.component_percentiles(row, pop, seasons=seasons)
        assert out["breakout"] is None
        assert out["production"] == 100
        assert out["efficiency"] == 33

    def test_in_progress_season_never_anchors_production(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40))
        cur = ra.current_college_season()
        row = {"position": "WR", "competition_score": 50.0,
               "forty_yard": None, "ras_score": None}
        seasons = [
            {"season": cur - 1, "games_played": 13,
             "receiving_yards": 650, "yds_per_reception": 12.0,
             "dominator_rating": 0.20},
            {"season": cur, "games_played": 7,
             "receiving_yards": 1400, "yds_per_reception": 20.0,
             "dominator_rating": 0.50},
        ]
        drivers = ra._component_drivers("WR", seasons)
        # The completed season's 50.0 anchors production; the live
        # season's 200.0 pace never does.
        assert drivers["production"] == 50.0
        out = ra.component_percentiles(row, pop, seasons=seasons)
        assert out["production"] == 28
        live_only = ra.component_percentiles(
            row, pop, seasons=[seasons[1]])
        assert live_only["production"] is None

    def test_missing_driver_fields_hide(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40))
        row = {"position": "WR", "competition_score": 50.0,
               "forty_yard": None, "ras_score": None}
        seasons = [{"season": 2024, "games_played": 13,
                    "receiving_yards": 780}]
        out = ra.component_percentiles(row, pop, seasons=seasons)
        assert out["production"] == 53
        assert out["efficiency"] is None
        assert out["breakout"] is None

    def test_no_seasons_no_driver_percentiles(self):
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40))
        row = {"position": "WR", "production_score": 100.0,
               "efficiency_score": 100.0, "breakout_profile_score": 100.0,
               "competition_score": 59.0,
               "forty_yard": None, "ras_score": None}
        out = ra.component_percentiles(row, pop)
        assert out["production"] is None
        assert out["efficiency"] is None
        assert out["breakout"] is None
        assert out["competition"] == 100


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
            lambda pos, ref_rows=None: pops[pos])
        assert ra.percentile_pool_line() == (
            "Percentiles vs drafted players at the position, "
            "classes 2025-2027")

    def test_single_class(self, monkeypatch):
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, ref_rows=None: {"classes": [2027]})
        assert ra.percentile_pool_line() == (
            "Percentiles vs drafted players at the position, classes 2027")

    def test_no_population_no_line(self, monkeypatch):
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, ref_rows=None: {"classes": []})
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


class TestGradeRendered:
    """Grades are back (Kaedon, 2026-11-23): the shipped page renders the
    numeric prospect grade on the board (Grade column) and in the modal
    (the report-card header band). The pre-rework dense grade renderers
    stay gone. The board carries no prospect tag chips (Kaedon,
    2026-11-23): the labeled tier divider lines name each band group
    instead; the modal's band surfaces in the grade context line."""

    FIXTURE_SCORE = 91.58

    def test_page_renders_grade(self):
        html = build_prospects_body(False)
        # The old dense grade renderers are gone entirely.
        assert "rkScore1" not in html
        assert "rk-dense" not in html
        assert "rkDense" not in html
        assert 'pm-hero-label">Prospect Score' not in html
        # The Grade Breakdown section (component percentile rows) is
        # part of the report-card modal; it is not the old renderer.
        assert "Grade Breakdown" in html
        assert "rkRcRenderGrade" in html
        assert "rkRcGradeSection" in html
        # The grade display pieces are present.
        assert "rkGradeCell" in html
        assert "pr-grade" in html
        assert '<span class="th-r">Grade</span>' in html
        assert 'rc-grade-label">Prospect Grade' in html
        assert "rkRcGrade" in html
        # The other display pieces are present.
        assert "rkBoardRow" in html
        assert "rkPoolLine" in html
        assert "tag-chip" in html
        assert "rkRcHeroNote" in html
        # Archetype evidence is one plain-text line (middot-separated),
        # not pills: the chip class is gone.
        assert "rc-arch-ev" in html
        assert "rc-ev-sep" in html
        assert "rc-ev-chip" not in html
        # The board shows no prospect tag chip and no Tag column: the
        # tier lines carry the band. (The modal's tag-chip surfaces
        # above stay.)
        assert "pr-tagcell" not in html
        assert '<span class="th-r">Tag</span>' not in html
        # The CSV export matches the board: no Tag column either.
        assert "'Breakout Pct', 'Tag'" not in html
        assert "r.tag && r.tag.label ? r.tag.label : ''," not in html
        assert "rkBandDivider" in html
        assert "RK_BAND_SHORT" in html

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
        # The grade numeral renders from the raw prospect_score field;
        # the derived display surfaces (tag, percentiles, receipt) still
        # never embed it.
        import json
        pop = ra.reference_population("WR", ref_rows=_pop_rows(40, tested=40))
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


class TestReportCardBand:
    """The report-card header carries the band as the tag label in the
    grade context line (e.g. "WR1 · Great Prospect"); the API's band
    keys are unchanged."""

    def test_band_label_rendered_in_grade_context(self):
        html = build_prospects_body(False)
        assert "rc-band-name" in html
        assert "tag.label" in html
        assert "rkRcContext" in html

    def test_band_keys_unchanged(self):
        for score in (95.0, 85.0, 75.0, 60.0):
            assert ra.prospect_tag(score)["band"] in (
                "elite", "great", "average", "day3")


class TestClassStandingGuards:
    """The Class Standing panel omits the chased-by note for the last
    player at the position (nobody is behind him)."""

    def test_chased_by_guarded_by_index(self):
        html = build_prospects_body(False)
        assert "ctx.idx < ctx.peers.length - 1" in html
        assert "Chased by" in html


class TestClassStandingPanel:
    """The Class Standing panel replaces the old hero cards; its
    mock-pick row renders only from a real consensus row."""

    def test_standing_panel_and_pick_gate(self):
        html = build_prospects_body(False)
        assert "rkRcRenderStanding" in html
        assert "Class Standing" in html
        assert "if (mockPick != null)" in html
        assert "r.projected_pick" in html
        assert "r.projected_round" in html


class TestAdvancedPercentileOnly:
    """Advanced rows carry a numeric value for percentile computation;
    the Efficiency/Production rows are component scores, flagged so the
    report-card Metric Breakdown excludes them."""

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


class TestSeasonLineCells:
    """Rich per-season stat cells: per-game rates derived from stored
    totals / GP only. A cell whose source field is not stored for that
    season is None (the page dashes it); nothing is ever derived from
    an unrelated field."""

    RATES = [float(x) for x in range(40, 140, 2)]  # 50 completed rates

    def _lines(self, monkeypatch, pos, seasons):
        monkeypatch.setattr(
            ra, "_season_rate_distributions", lambda p: list(self.RATES))
        return ra._season_lines(pos, seasons)

    def test_wr_cells_math(self, monkeypatch):
        lines = self._lines(monkeypatch, "WR", [
            {"season": 2025, "games_played": 10, "targets": 80,
             "receptions": 60, "receiving_yards": 900,
             "receiving_tds": 8, "yds_per_reception": 15.0},
        ])
        cells = lines[0]["cells"]
        assert cells["tgt_g"] == 8.0
        assert cells["rec_g"] == 6.0
        assert cells["rec_yds_g"] == 90.0
        assert cells["yds_rec"] == 15.0
        assert cells["rec_td_g"] == 0.8

    def test_wr_targets_dash_when_not_stored(self, monkeypatch):
        lines = self._lines(monkeypatch, "WR", [
            {"season": 2024, "games_played": 16, "receptions": 76,
             "receiving_yards": 1315, "receiving_tds": 15,
             "yds_per_reception": 17.3},
        ])
        cells = lines[0]["cells"]
        assert cells["tgt_g"] is None
        assert cells["rec_g"] == 4.75
        assert cells["rec_yds_g"] == 82.188

    def test_wr_targets_below_receptions_is_not_data(self, monkeypatch):
        # The stored-targets pipeline failure (targets < receptions):
        # the cell dashes, exactly as _build_advanced_metrics treats it.
        lines = self._lines(monkeypatch, "WR", [
            {"season": 2026, "games_played": 7, "targets": 18,
             "receptions": 48, "receiving_yards": 916,
             "receiving_tds": 9},
        ])
        assert lines[0]["cells"]["tgt_g"] is None

    def test_wr_yds_rec_derived_from_totals(self, monkeypatch):
        lines = self._lines(monkeypatch, "WR", [
            {"season": 2025, "games_played": 13, "receptions": 90,
             "receiving_yards": 1500, "receiving_tds": 12},
        ])
        assert lines[0]["cells"]["yds_rec"] == 16.667

    def test_te_uses_receiving_columns(self, monkeypatch):
        cols = ra._season_line_columns("TE")
        assert [c["key"] for c in cols] == [
            "tgt_g", "rec_g", "rec_yds_g", "yds_rec", "rec_td_g"]

    def test_rb_cells(self, monkeypatch):
        lines = self._lines(monkeypatch, "RB", [
            {"season": 2025, "games_played": 10, "rush_attempts": 180,
             "rush_yards": 900, "rush_tds": 9, "receptions": 40,
             "receiving_yards": 320, "receiving_tds": 2},
        ])
        cells = lines[0]["cells"]
        assert cells["rush_att_g"] == 18.0
        assert cells["rush_yds_g"] == 90.0
        assert cells["yds_carry"] == 5.0  # derived: no stored rate
        assert cells["rec_g"] == 4.0
        assert cells["rec_yds_g"] == 32.0
        assert cells["td_g"] == 1.1  # (9 rush + 2 rec) / 10 GP

    def test_rb_td_g_dash_when_no_tds_stored(self, monkeypatch):
        lines = self._lines(monkeypatch, "RB", [
            {"season": 2025, "games_played": 10, "rush_attempts": 100,
             "rush_yards": 450},
        ])
        assert lines[0]["cells"]["td_g"] is None

    def test_qb_cells(self, monkeypatch):
        lines = self._lines(monkeypatch, "QB", [
            {"season": 2025, "games_played": 12, "pass_attempts": 420,
             "completions": 294, "pass_yards": 3600, "pass_tds": 30,
             "interceptions": 10, "rush_yards": 240, "team": "State"},
        ])
        cells = lines[0]["cells"]
        assert cells["att_g"] == 35.0
        assert cells["cmp_pct"] == 70.0  # derived from comp/att
        assert cells["pass_yds_g"] == 300.0
        assert cells["td_int"] == 3.0
        assert cells["rush_yds_g"] == 20.0

    def test_qb_cmp_pct_prefers_stored_rate(self, monkeypatch):
        lines = self._lines(monkeypatch, "QB", [
            {"season": 2025, "games_played": 12, "pass_attempts": 420,
             "completions": 294, "completion_pct": 68.4,
             "pass_yards": 3600},
        ])
        assert lines[0]["cells"]["cmp_pct"] == 68.4

    def test_qb_td_int_dash_against_zero_interceptions(self, monkeypatch):
        lines = self._lines(monkeypatch, "QB", [
            {"season": 2025, "games_played": 12, "pass_attempts": 420,
             "pass_yards": 3600, "pass_tds": 30, "interceptions": 0},
        ])
        assert lines[0]["cells"]["td_int"] is None
        # A stored ratio still wins when one exists.
        lines = self._lines(monkeypatch, "QB", [
            {"season": 2025, "games_played": 12, "pass_attempts": 420,
             "pass_yards": 3600, "pass_tds": 30, "interceptions": 0,
             "td_int_ratio": 30.0},
        ])
        assert lines[0]["cells"]["td_int"] == 30.0

    def test_columns_shape_and_overview_subset(self):
        cols = ra._season_line_columns("WR")
        assert [c["key"] for c in cols] == [
            "tgt_g", "rec_g", "rec_yds_g", "yds_rec", "rec_td_g"]
        ov = [c["key"] for c in cols if c["ov"]]
        assert ov == ["tgt_g", "rec_g", "rec_yds_g", "yds_rec"]
        qb = {c["key"]: c for c in ra._season_line_columns("QB")}
        assert qb["cmp_pct"]["ov"] is False
        assert qb["td_int"]["ov"] is True
        rb = {c["key"]: c for c in ra._season_line_columns("RB")}
        assert rb["td_g"]["dp"] == 2
        assert rb["rush_att_g"]["ov"] is False

    def test_in_progress_season_cells_present_percentile_none(
            self, monkeypatch):
        cur = ra.current_college_season()
        lines = self._lines(monkeypatch, "WR", [
            {"season": cur, "games_played": 7, "targets": 70,
             "receptions": 48, "receiving_yards": 916,
             "receiving_tds": 9},
        ])
        line = lines[0]
        assert line["in_progress"] is True
        assert line["percentile"] is None
        # Rates still print for the live season; only the percentile
        # and any totals stay out.
        assert line["cells"]["rec_yds_g"] == 130.857
        assert line["cells"]["tgt_g"] == 10.0

    def test_season_tables_render_from_api_columns(self):
        html = build_prospects_body(False)
        assert "season_line_columns" in html
        assert "rk-table-scroll" in html
        assert "c.ov" in html  # Overview renders the ov subset only


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
