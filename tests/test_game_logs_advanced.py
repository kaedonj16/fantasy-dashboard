"""Regression tests for the game-log advanced-metrics feature.

Covers the weekly pipeline additions that feed the redesigned player game
logs: broken tackles + WOPR columns in the weekly advanced schema, carry
share in the weekly usage table, PFR broken-tackle aggregation, the weekly
WOPR formula, the sync caller's team-map argument, and frontend source
contracts (toggle, grouped table, no RR, no em dashes, responsive CSS).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── 1. Weekly advanced schema ──────────────────────────────────────────────

def test_weekly_adv_metric_cols_include_broken_tackles_and_wopr():
    from data_building.advanced_metrics import WEEKLY_ADV_METRIC_COLS

    assert "broken_tackles" in WEEKLY_ADV_METRIC_COLS
    assert "wopr" in WEEKLY_ADV_METRIC_COLS


# ── 2. Carry share ─────────────────────────────────────────────────────────

def test_carry_share_migration_and_formula_present():
    src = (ROOT / "data_building" / "weekly_metrics.py").read_text()
    assert "ADD COLUMN IF NOT EXISTS carry_share NUMERIC" in src
    # Player carries / team-week carries, as a percentage.
    assert "carries / team_carries[team] * 100" in src
    # A week with NULL carry_share is selected for rebuild.
    assert "carry_share IS NULL" in src


# ── 3. PFR broken tackles aggregation ──────────────────────────────────────

def _write_pfr_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pfr_player_id", "week", "game_type",
            "rushing_broken_tackles", "receiving_broken_tackles",
        ])
        w.writeheader()
        w.writerows(rows)


def test_pfr_broken_tackles_sums_rush_and_rec(tmp_path, monkeypatch):
    import data_building.external_data.nflverse_metrics as nm

    rush = tmp_path / "rush.csv"
    rec = tmp_path / "rec.csv"
    _write_pfr_csv(rush, [
        {"pfr_player_id": "TaylJo00", "week": "1", "game_type": "REG",
         "rushing_broken_tackles": "12", "receiving_broken_tackles": ""},
        {"pfr_player_id": "TaylJo00", "week": "2", "game_type": "REG",
         "rushing_broken_tackles": "8", "receiving_broken_tackles": ""},
        # Postseason rows must not leak into regular-season game logs.
        {"pfr_player_id": "TaylJo00", "week": "19", "game_type": "POST",
         "rushing_broken_tackles": "99", "receiving_broken_tackles": ""},
    ])
    _write_pfr_csv(rec, [
        {"pfr_player_id": "TaylJo00", "week": "1", "game_type": "REG",
         "rushing_broken_tackles": "", "receiving_broken_tackles": "6"},
        # Unknown PFR id (no crosswalk entry) is skipped.
        {"pfr_player_id": "NopeXX00", "week": "1", "game_type": "REG",
         "rushing_broken_tackles": "5", "receiving_broken_tackles": "5"},
    ])
    monkeypatch.setattr(nm, "_pfr_to_sleeper", lambda: {"TaylJo00": "4034"})
    monkeypatch.setattr(nm, "download_pfr_advstats_rush_csv", lambda season: str(rush))
    monkeypatch.setattr(nm, "download_pfr_advstats_rec_csv", lambda season: str(rec))

    out = nm.build_pfr_broken_tackles_weekly(2026)
    assert out[("4034", 1)]["broken_tackles"] == 18.0  # 12 rush + 6 rec
    assert out[("4034", 2)]["broken_tackles"] == 8.0
    assert ("4034", 19) not in out
    assert all(pid != "NopeXX00" for pid, _ in out)


def test_pfr_broken_tackles_skips_pre_2018():
    import data_building.external_data.nflverse_metrics as nm

    assert nm.build_pfr_broken_tackles_weekly(2017) == {}


# ── 4. Weekly WOPR ─────────────────────────────────────────────────────────

def test_weekly_wopr_formula():
    from data_building.external_data.nflverse_metrics import _apply_weekly_wopr

    out = {
        ("p1", 1): {"w_targets": 10.0, "w_rec_air_yards": 120.0, "avg_depth_of_target": 8.0},
        ("p2", 1): {"w_targets": 10.0, "w_rec_air_yards": 80.0, "avg_depth_of_target": 6.0},
        ("p3", 2): {"w_targets": 5.0, "w_rec_air_yards": 50.0, "avg_depth_of_target": 9.0},
    }
    _apply_weekly_wopr(out, {"p1": "IND", "p2": "IND", "p3": "IND"})
    # Week 1: team totals 20 targets / 200 air yards.
    assert out[("p1", 1)]["wopr"] == round(1.5 * 0.5 + 0.7 * 0.6, 3)
    assert out[("p2", 1)]["wopr"] == round(1.5 * 0.5 + 0.7 * 0.4, 3)
    # Week 2 is its own team-week: 100% shares.
    assert out[("p3", 2)]["wopr"] == round(1.5 + 0.7, 3)


def test_weekly_wopr_skips_players_without_team():
    from data_building.external_data.nflverse_metrics import _apply_weekly_wopr

    out = {("p1", 1): {"w_targets": 10.0, "w_rec_air_yards": 120.0}}
    _apply_weekly_wopr(out, {})  # no team map at all
    assert "wopr" not in out[("p1", 1)]
    _apply_weekly_wopr(out, {"p2": "IND"})  # p1 not in map
    assert "wopr" not in out[("p1", 1)]


# ── 5. Sync caller passes the team map ─────────────────────────────────────

def test_sync_weekly_passes_teams_by_pid():
    src = (ROOT / "scripts" / "sync_nflverse_metrics.py").read_text()
    assert "build_nflverse_weekly_metrics_for_season(season, teams_by_pid=teams_by_pid)" in src


# ── 6. Frontend source contracts ───────────────────────────────────────────

def _new_gamelog_js():
    src = (ROOT / "static" / "player_modal.js").read_text()
    start = src.index("window._glAvgTot = window._glAvgTot || 'avg';")
    end_marker = "window.glSetAvgTot = function (mode, el) {"
    end = src.index(end_marker)
    depth = 0
    i = end
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    return src[start:i]


def test_gamelog_js_has_toggle_and_grouped_table():
    js = _new_gamelog_js()
    assert "glSetAvgTot" in js
    assert "season-card" in js
    assert "mini-stats" in js
    assert "gl-adv" in js
    for group in ("Fantasy", "Passing", "Rushing", "Receiving", "Usage"):
        assert group in js
    for col in ("xFP", "BTK", "WOPR", "Car%", "SNP%"):
        assert col in js


def test_gamelog_js_excludes_rr_and_position_ranks():
    js = _new_gamelog_js()
    assert "'rr'" not in js and '"rr"' not in js
    assert "route run" not in js.lower()
    assert "pos_rank" not in js and "posRank" not in js


def test_gamelog_js_no_em_dashes():
    assert "—" not in _new_gamelog_js()


def test_gamelog_css_responsive_classes():
    css = (ROOT / "static" / "dashboard.css").read_text()
    for cls in (".season-card", ".mini-stats", ".gl-adv", ".seg-light",
                ".scroll-hint", ".season-team", ".t-g", ".sec-start"):
        assert cls in css
    assert "@media (max-width: 600px)" in css
    assert "grid-template-columns: repeat(4, 1fr)" in css


def test_gamelog_js_toggle_wired_in_both_modals():
    pm = (ROOT / "static" / "player_modal.js").read_text()
    assert "data.season_teams || {}" in pm
    app = (ROOT / "static" / "app.js").read_text()
    assert "_buildStatsHTML(logsByYear, true, position || '', data.season_teams || {})" in app
