"""Composite Breakout Score (breakout_trend_score) + weekly chip contracts.

Part 1: the AM "Is This Breakout Real?" preset's primary metric becomes a
composite of xFP Trend + Usage Trend + FPOE/G (each min-max normalized
across the qualified pool, averaged, scaled 0-100). xFP Trend is required;
a player without it gets no composite, never a partial invention. The
metric is PRO-gated like all three of its components.

Part 2: while the breakout_check board is displayed, the page fetches the
current weekly Breakout Engine board once and chips matching AM rows.
"""
import pytest

pytest.importorskip("flask")

import data_building.advanced_metrics as am
from data_building.advanced_metrics import (
    LEADERBOARD_METRICS,
    PRO_METRICS,
    _compute_breakout_trend_scores,
)
from dashboard_services.pages.advanced_metrics_page import (
    ADVANCED_METRIC_PRESETS,
    _AM_JS,
)

XFP = "xfp_trend"
OPP = "opportunity_trend"
FPOE = "ppr_over_expected_per_game"


# ── Registration + gating ───────────────────────────────────────────────────

def test_composite_registered_with_label_and_positions():
    spec = LEADERBOARD_METRICS["breakout_trend_score"]
    assert spec["label"] == "Breakout Score"
    assert spec["positions"] == ["QB", "RB", "WR", "TE"]
    assert spec["min_vol"]["col"] == "games"
    assert spec["desc"]  # the page glossary ships the registry desc


def test_composite_is_pro_gated_like_its_components():
    assert "breakout_trend_score" in PRO_METRICS
    for key in (XFP, OPP, FPOE):
        assert key in PRO_METRICS


def test_composite_is_season_only_not_weekly():
    from data_building.advanced_metrics import (
        _WEEKLY_METRICS, ADV_WEEKLY_METRIC_KEYS, adv_weekly_metric_supported,
    )
    assert "breakout_trend_score" not in _WEEKLY_METRICS
    assert "breakout_trend_score" not in ADV_WEEKLY_METRIC_KEYS
    assert adv_weekly_metric_supported("breakout_trend_score") is False


def test_leaderboard_breakout_score_403_for_non_premium(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: False)
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=breakout_trend_score&season=2026")
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "pro_only"


def test_leaderboard_breakout_score_ok_for_premium(offline_client, monkeypatch):
    import routes.advanced_metrics_bp as bp
    monkeypatch.setattr(bp, "_request_has_premium", lambda season=None: True)
    monkeypatch.setattr(am, "get_metric_leaderboard", lambda *a, **k: [])
    resp = offline_client.get(
        "/api/advanced-metrics/leaderboard?metric=breakout_trend_score&season=2026")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data["metric"] == "breakout_trend_score"
    assert data["label"] == "Breakout Score"
    # Season-only: a week range must report weekly_capable=False (the route
    # then serves season values, like xfp_trend).
    assert data["weekly_capable"] is False


# ── Pure composite math ─────────────────────────────────────────────────────

def test_compute_scores_known_pool():
    scores = _compute_breakout_trend_scores({
        "a": {XFP: 0.30, OPP: 0.20, FPOE: 2.0},    # max on all three
        "b": {XFP: -0.10, OPP: -0.20, FPOE: -2.0},  # min on all three
        "c": {XFP: 0.10, OPP: 0.0, FPOE: 0.0},      # midpoint on all three
    })
    assert scores == {"a": 100.0, "b": 0.0, "c": 50.0}


def test_compute_scores_missing_xfp_gets_no_composite():
    scores = _compute_breakout_trend_scores({
        "a": {XFP: 0.30, OPP: 0.20, FPOE: 2.0},
        "b": {XFP: -0.10, OPP: -0.20, FPOE: -2.0},
        "ghost": {XFP: None, OPP: 0.99, FPOE: 9.9},
    })
    assert "ghost" not in scores
    assert scores["a"] == 100.0


def test_compute_scores_missing_secondary_averages_available():
    scores = _compute_breakout_trend_scores({
        "full": {XFP: 0.0, OPP: 0.0, FPOE: 0.0},
        "xfp_only": {XFP: 0.40, OPP: None, FPOE: None},
        "xfp_fpoe": {XFP: 0.20, OPP: None, FPOE: 4.0},
    })
    # xfp bounds [0.0, 0.40]; fpoe bounds [0.0, 4.0] (pool players who have it).
    assert scores["xfp_only"] == 100.0           # xfp normalized 1.0, only part
    assert scores["xfp_fpoe"] == 75.0            # (0.5 + 1.0) / 2
    # "full" is the only player with a Usage Trend value, so that component
    # is flat across its pool and contributes the neutral 0.5:
    # (0.0 + 0.5 + 0.0) / 3 -> 16.7.
    assert scores["full"] == pytest.approx(16.7)


def test_compute_scores_flat_component_is_neutral():
    scores = _compute_breakout_trend_scores({
        "a": {XFP: 1.0, OPP: 0.10, FPOE: None},
        "b": {XFP: 0.0, OPP: 0.10, FPOE: None},
    })
    # Usage Trend identical across the pool -> contributes the neutral 0.5.
    assert scores["a"] == 75.0
    assert scores["b"] == 25.0


def test_compute_scores_empty_pools():
    assert _compute_breakout_trend_scores({}) == {}
    assert _compute_breakout_trend_scores({"a": {XFP: None, OPP: 0.1}}) == {}


# ── Leaderboard builder + dispatch ──────────────────────────────────────────

def _fake_boards(monkeypatch, boards):
    calls = []

    def fake(metric, position=None, season=None, min_vol=None, limit=500, **kw):
        calls.append((metric, position, season, min_vol))
        return [dict(r) for r in boards.get(metric, [])]

    monkeypatch.setattr(am, "get_metric_leaderboard", fake)
    return calls


def _row(pid, value, name=None):
    return {"player_id": pid, "name": name or ("P" + pid), "team": "KC",
            "position": "WR", "value": value, "games": 6}


def test_breakout_leaderboard_rows_come_from_xfp_board(monkeypatch):
    boards = {
        XFP: [_row("1", 0.30), _row("2", -0.10), _row("3", 0.10)],
        OPP: [_row("1", 0.20), _row("2", -0.20), _row("3", 0.0)],
        FPOE: [_row("1", 2.0), _row("2", -2.0), _row("3", 0.0)],
    }
    calls = _fake_boards(monkeypatch, boards)
    rows = am.get_breakout_trend_leaderboard(position="WR", season=2026, min_vol=4)
    assert [r["player_id"] for r in rows] == ["1", "3", "2"]
    assert [r["value"] for r in rows] == [100.0, 50.0, 0.0]
    # Identity/context fields ride along from the xFP Trend board row.
    assert rows[0]["name"] == "P1" and rows[0]["team"] == "KC"
    assert rows[0]["games"] == 6
    # All three component boards were fetched with the same context.
    assert {c[0] for c in calls} == {XFP, OPP, FPOE}
    assert all(c[1:] == ("WR", 2026, 4) for c in calls)


def test_breakout_leaderboard_drops_players_without_xfp(monkeypatch):
    boards = {
        XFP: [_row("1", 0.30)],
        OPP: [_row("1", 0.20), _row("9", 0.99)],
        FPOE: [_row("1", 2.0), _row("9", 9.9)],
    }
    _fake_boards(monkeypatch, boards)
    rows = am.get_breakout_trend_leaderboard(season=2026)
    assert [r["player_id"] for r in rows] == ["1"]


def test_breakout_leaderboard_limit_applies_after_sort(monkeypatch):
    boards = {
        XFP: [_row("1", 0.30), _row("2", 0.20), _row("3", 0.10)],
        OPP: [_row("1", 0.30), _row("2", 0.20), _row("3", 0.10)],
        FPOE: [_row("1", 3.0), _row("2", 2.0), _row("3", 1.0)],
    }
    _fake_boards(monkeypatch, boards)
    rows = am.get_breakout_trend_leaderboard(season=2026, limit=2)
    assert [r["player_id"] for r in rows] == ["1", "2"]


def test_metric_leaderboard_dispatches_to_composite(monkeypatch):
    seen = {}

    def fake_builder(position=None, season=None, min_vol=None, limit=500):
        seen.update(position=position, season=season, min_vol=min_vol, limit=limit)
        return [{"player_id": "1", "name": "P1", "value": 88.8}]

    monkeypatch.setattr(am, "get_breakout_trend_leaderboard", fake_builder)
    rows = am.get_metric_leaderboard(
        "breakout_trend_score", position="RB", season=2026, min_vol=4, limit=25)
    assert seen == {"position": "RB", "season": 2026, "min_vol": 4, "limit": 25}
    assert rows[0]["value"] == 88.8
    assert rows[0]["season"] == 2026  # _stamp_season applied like other builders


# ── Preset contract ─────────────────────────────────────────────────────────

def test_breakout_check_preset_primary_is_composite():
    preset = ADVANCED_METRIC_PRESETS["breakout_check"]
    assert preset["primary"] == "breakout_trend_score"
    assert preset["metrics"][0] == "breakout_trend_score"
    # The previously displayed columns all remain on the board.
    for key in ("xfp_trend", "opportunity_trend", "snap_share",
                "ppr_over_expected_per_game", "target_share", "air_yards_share"):
        assert key in preset["metrics"], key


def test_waiver_wire_preset_unchanged():
    preset = ADVANCED_METRIC_PRESETS["waiver_wire"]
    assert preset["primary"] == "opportunity_trend"
    assert preset["metrics"][0] == "opportunity_trend"


# ── Weekly cross-reference chip (source contracts) ──────────────────────────

def test_weekly_chip_fetch_is_single_and_breakout_check_scoped():
    assert "function _ensureWeeklyBreakouts()" in _AM_JS
    assert "'/api/breakout/candidates?season='" in _AM_JS
    assert "_activePresetId !== 'breakout_check'" in _AM_JS
    # Cached per season; an in-flight or completed fetch is not repeated.
    assert "_weeklyBo.byId || _weeklyBo.inflight" in _AM_JS
    # Only weekly-engine rows chip; offseason candidates never do.
    assert "c.weekly !== true" in _AM_JS


def test_weekly_chip_renders_on_rows_and_links_to_board():
    assert "function _weeklyChipHtml(r)" in _AM_JS
    assert "+ _weeklyChipHtml(r)" in _AM_JS  # wired into the player cell
    assert 'class="am-chip am-weekly-chip"' in _AM_JS  # reuses the chip class
    assert 'href="/breakouts"' in _AM_JS
    assert "'Weekly: '" in _AM_JS
    # Fetch is triggered from the board load path, not per row.
    assert "_ensureWeeklyBreakouts();" in _AM_JS
