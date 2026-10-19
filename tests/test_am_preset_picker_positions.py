"""Advanced Metrics: preset -> picker sync, multi-position filter, scrimmage yards.

Kaedon 2026-10-19, three asks in one sitting:
1. "when I click one of the presets, it should update the primary metric
   in the picker" - amLoadPreset set the hidden native select's value but
   never refreshed the custom picker's button label (which syncs only on
   a real 'change' event), so the button kept showing the old metric.
2. "I want to be able to select multiple positions for the filtering on
   the page, like rb and wr" - the position control was single-select.
   state.position now carries 'ALL' or a comma-joined subset ('RB,WR');
   the segmented buttons toggle membership.
3. "I should be able to do rushing yards and receiving yards and both
   per game" - the four singles already existed (Rush Yards, Rec Yards,
   Rush Yds/G, Rec Yds/G); what's added is the combined scrimmage pair
   (total + per game) for mixed-position boards, plus picker search that
   understands "rushing yards" / "receiving yards per game" phrasing.
"""
import pathlib
import sqlite3

import pytest

pytest.importorskip("flask")

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import (
    _PER_GAME_BOUNDS,
    _WEEKLY_METRICS,
    LEADERBOARD_METRICS,
)

PAGE = build_advanced_metrics_body(True, LEADERBOARD_METRICS)


# ── 1. Preset loads refresh the visible picker ─────────────────────────────

def test_picker_exposes_a_programmatic_label_sync():
    assert "selEl._amSyncLabel = syncLabel;" in PAGE
    assert "function _syncMetricPickerLabel()" in PAGE


def test_every_programmatic_metric_set_syncs_the_picker():
    # amLoadPreset, saved-set apply, and working-view restore all assign
    # metricSel.value directly; each must refresh the button label.
    assert PAGE.count("_syncMetricPickerLabel();") >= 3


def test_picker_label_falls_back_to_the_shared_resolver():
    # A primary with no <option> in this viewer's picker (PRO metric on a
    # free picker) must not blank the button: syncLabel resolves through
    # _mLabel, the page's one sanctioned label fallback.
    assert "const txt = opt ? opt.textContent : _mLabel(selEl.value);" in PAGE


# ── 2. Multi-position selection ────────────────────────────────────────────

def test_position_helpers_define_the_subset_encoding():
    assert "function _posList()" in PAGE
    assert "function _setPositions(list)" in PAGE
    assert "function _sanitizePosParam(raw)" in PAGE
    assert "function _narrowPositionsToMetric()" in PAGE


def test_render_filters_by_selected_subset():
    assert "_selPos.indexOf(up(r.position)) >= 0" in PAGE


def test_position_sources_are_sanitized_for_subsets():
    # The state literal itself stays harness-extractable (pinned by
    # test_staged_url_parity); sanitizing runs immediately after it.
    assert "state.position = _sanitizePosParam(state.position);" in PAGE
    assert "position: _sanitizePosParam(v.position)" in PAGE
    assert "position:_sanitizePosParam(raw.position)" in PAGE


def test_preset_position_goes_through_the_setter():
    assert "if (preset.position) _setPositions([preset.position]);" in PAGE


def test_field_average_label_uses_the_subset_label():
    assert PAGE.count("_posLabel()") >= 4  # helper + note + both bar titles
    assert "state.position !== 'ALL' ? state.position : 'Field'" not in PAGE


def test_graph_seeds_only_from_a_single_selected_position():
    assert "_pageSel.length === 1 ? _pageSel[0] : null" in PAGE
    assert "_posList().length === 1 ? _posList()[0] : null" in PAGE


# ── Picker search understands natural phrasing ─────────────────────────────

def test_picker_search_normalizes_rushing_receiving_yds():
    assert "function _normTerm(s)" in PAGE
    assert ".replace(/rushing/g, 'rush').replace(/receiving/g, 'rec')" in PAGE
    assert ".replace(/yds/g, 'yards')" in PAGE


# ── 3. Scrimmage yards pair ────────────────────────────────────────────────

def test_scrimmage_pair_registered_free_and_cross_position():
    for key in ("scrimmage_yards", "scrimmage_yards_per_game"):
        spec = LEADERBOARD_METRICS[key]
        assert spec["category"] == "General", key
        assert spec["positions"] == ["QB", "RB", "WR", "TE"], key
        assert spec.get("desc"), key
        assert not spec.get("pct"), key
    assert LEADERBOARD_METRICS["scrimmage_yards"].get("integer") is True
    keys = list(LEADERBOARD_METRICS)
    assert keys.index("scrimmage_yards") == keys.index("total_yards") + 1
    assert keys.index("scrimmage_yards_per_game") == keys.index("scrimmage_yards") + 1


def test_scrimmage_sql_is_rush_plus_rec_only():
    total = LEADERBOARD_METRICS["scrimmage_yards"]
    assert total["computed_sql"] == (
        "ROUND(COALESCE(m.yards_per_carry * m.total_carries, 0) + "
        "COALESCE(m.yards_per_reception * m.total_receptions, 0))"
    )
    assert "yards_per_attempt" not in total["computed_sql"]  # no passing
    pg = LEADERBOARD_METRICS["scrimmage_yards_per_game"]
    assert "NULLIF(m.games, 0)" in pg["computed_sql"]
    assert pg["computed_null"].count("IS NOT NULL") >= 4


def test_scrimmage_sql_computes_and_null_gates():
    total = LEADERBOARD_METRICS["scrimmage_yards"]
    pg = LEADERBOARD_METRICS["scrimmage_yards_per_game"]
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE m (yards_per_carry REAL, total_carries REAL,"
        " yards_per_reception REAL, total_receptions REAL, games REAL)"
    )
    # 5.0 ypc x 100 carries + 10.0 ypr x 50 recs = 1000 scrimmage yards.
    con.execute("INSERT INTO m VALUES (5.0, 100, 10.0, 50, 10)")
    # Receiving only: rushing side NULL -> still counts (COALESCE 0).
    con.execute("INSERT INTO m VALUES (NULL, NULL, 12.0, 40, 8)")
    # Nothing at all: the null gate must exclude the row entirely.
    con.execute("INSERT INTO m VALUES (NULL, NULL, NULL, NULL, 6)")
    q = "SELECT {sql} FROM m WHERE {null}"
    vals = [r[0] for r in con.execute(q.format(sql=total["computed_sql"], null=total["computed_null"]))]
    assert vals == [1000.0, 480.0]
    vals_pg = [r[0] for r in con.execute(q.format(sql=pg["computed_sql"], null=pg["computed_null"]))]
    assert vals_pg == [100.0, 60.0]


def test_scrimmage_weekly_and_bounds_wiring():
    assert _WEEKLY_METRICS["scrimmage_yards_per_game"]["sql"] == "AVG(rush_yards + rec_yards)"
    assert _PER_GAME_BOUNDS["scrimmage_yards_per_game"] == "scrimmage_yards"


def test_scrimmage_appears_in_the_picker():
    for key in ("scrimmage_yards", "scrimmage_yards_per_game",
                "total_rush_yards", "total_rec_yards",
                "rush_yards_per_game", "rec_yards_per_game"):
        assert f'<option value="{key}">' in PAGE, key


def test_weekly_range_aggregator_reports_scrimmage():
    src = pathlib.Path("data_building/advanced_metrics.py").read_text()
    assert 'out["scrimmage_yards"] = int(rush_yards + rec_yards)' in src
    assert 'out["scrimmage_yards_per_game"] = (rush_yards + rec_yards) / n' in src
