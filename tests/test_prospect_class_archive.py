"""Tests for the past-class prospect archive (migration 058).

Past classes the live pipeline never stored get boards from
prospect_class_archive: one graded row per drafted QB/RB/WR/TE,
written by scripts/build_reference_population.py from the same
scores it grades for the percentile pools. These tests cover the
archive row shaping + rank computation, the combined (?year=all)
merge, the archive detail branch, the build script's archive row
assembly, and the migration. Synthetic data only; no DB needed.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)
pytest.importorskip("flask")

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services import rookie_api as ra

_spec = importlib.util.spec_from_file_location(
    "build_reference_population",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_reference_population.py"))
refpop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refpop)


# ── Fixtures ────────────────────────────────────────────────────────

def _store_rows(year=2020):
    return [
        {"player_name": "Alpha Wideout", "position": "WR",
         "draft_class_year": year, "school": "State U",
         "actual_pick": 12, "prospect_score": 88.0,
         "production_score": 90.0, "efficiency_score": 85.0,
         "breakout_profile_score": 80.0, "competition_score": 70.0,
         "model_version": "3.7"},
        {"player_name": "Beta Back", "position": "RB",
         "draft_class_year": year, "school": "Tech",
         "actual_pick": 40, "prospect_score": 91.5,
         "production_score": 92.0, "efficiency_score": 88.0,
         "breakout_profile_score": 84.0, "competition_score": 60.0,
         "model_version": "3.7"},
        {"player_name": "Gamma Wideout", "position": "WR",
         "draft_class_year": year, "school": "State U",
         "actual_pick": 90, "prospect_score": 75.0,
         "production_score": 70.0, "efficiency_score": 66.0,
         "breakout_profile_score": 61.0, "competition_score": 50.0,
         "model_version": "3.7"},
        {"player_name": "Delta Tight", "position": "TE",
         "draft_class_year": year, "school": "A&M",
         "actual_pick": 65, "prospect_score": 82.0,
         "production_score": 78.0, "efficiency_score": 74.0,
         "breakout_profile_score": 71.0, "competition_score": 55.0,
         "model_version": "3.7"},
    ]


def _fake_population(pos):
    # 40 values per distribution: component_percentile needs >= 30.
    return {
        "values": {
            "production": [float(i) for i in range(40)],
            "efficiency": [float(i) for i in range(40)],
            "breakout": [float(i) for i in range(40)],
            "competition": [float(i * 2) for i in range(40)],  # 0..78
            "athleticism": [],
        },
        "drivers": {
            "production": [float(i * 10) for i in range(40)],  # 0..390
            "efficiency": [float(i) for i in range(40)],
            "breakout": [i / 100.0 for i in range(40)],        # 0..0.39
        },
        "classes": [2016, 2024],
        "n": 40,
    }


def _fake_ref_rows(pos):
    if pos != "WR":
        return []
    return [{
        "player_name": "Alpha Wideout", "draft_class_year": 2020,
        "season_rates": [50.0, 90.0],
        "advanced_marks": {"Yds/Rec": 15.0, "Dominator": 0.35},
    }]


def _patch_archive(monkeypatch, rows_by_year):
    monkeypatch.setattr(
        ra, "_archive_store_rows",
        lambda year=None: list(rows_by_year.get(
            year, rows_by_year.get(None, []))) if year is not None
        else [r for rows in rows_by_year.values() for r in rows])
    monkeypatch.setattr(ra, "reference_population",
                        lambda pos, ref_rows=None: _fake_population(pos))
    monkeypatch.setattr(ra, "_reference_rows", _fake_ref_rows)
    ra._archive_cache.clear()


# ── Board shaping ───────────────────────────────────────────────────

class TestArchiveBoardShaping:
    def test_ranks_from_grade_order(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        board = ra._archive_board_rows(2020)
        assert [d["name"] for d in board] == [
            "Beta Back", "Alpha Wideout", "Delta Tight", "Gamma Wideout"]
        assert [d["overall_rank"] for d in board] == [1, 2, 3, 4]
        by_name = {d["name"]: d for d in board}
        assert by_name["Alpha Wideout"]["position_rank"] == 1
        assert by_name["Gamma Wideout"]["position_rank"] == 2
        assert by_name["Beta Back"]["position_rank"] == 1
        alpha = by_name["Alpha Wideout"]
        assert alpha["archive"] is True
        assert alpha["prospect_score"] == 88.0
        assert alpha["school"] == "State U"
        assert alpha["actual_pick"] == 12
        assert alpha["draft_class_year"] == 2020
        assert alpha["player_id"].startswith("ARCHIVE_2020_WR_")
        assert alpha["tag"]["band"] == "great"  # 80 <= 88 < 92

    def test_driver_percentiles_from_reference_marks(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        board = ra._archive_board_rows(2020)
        alpha = next(d for d in board if d["name"] == "Alpha Wideout")
        pct = alpha["percentiles"]
        # Production driver = best season rate 90.0; 10 of the 40
        # driver values (0..90 step 10) are at or below -> 25.
        assert pct["production"] == 25
        # Efficiency driver 15.0 vs 0..39 -> 16 at or below -> 40.
        assert pct["efficiency"] == 40
        # Breakout driver 0.35 vs 0..0.39 -> 36 at or below -> 90.
        assert pct["breakout"] == 90
        # Competition is score-based: 70.0 vs 0..78 step 2 -> 36 of
        # 40 at or below -> 90.
        assert pct["competition"] == 90

    def test_percentiles_hide_without_driver_marks(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        board = ra._archive_board_rows(2020)
        beta = next(d for d in board if d["name"] == "Beta Back")
        # No reference marks stored for Beta: the driver percentiles
        # hide (None) instead of falling back to the clipped scores.
        assert beta["percentiles"]["production"] is None
        assert beta["percentiles"]["efficiency"] is None
        assert beta["percentiles"]["breakout"] is None
        # Competition is score-based and still prints.
        assert beta["percentiles"]["competition"] is not None

    def test_unscored_and_non_skill_rows_never_surface(self, monkeypatch):
        rows = _store_rows() + [
            {"player_name": "No Grade", "position": "WR",
             "draft_class_year": 2020, "prospect_score": None},
            {"player_name": "A Corner", "position": "CB",
             "draft_class_year": 2020, "prospect_score": 99.0},
        ]
        _patch_archive(monkeypatch, {2020: rows})
        board = ra._archive_board_rows(2020)
        names = [d["name"] for d in board]
        assert "No Grade" not in names
        assert "A Corner" not in names


# ── Combined board ──────────────────────────────────────────────────

class TestCombinedBoard:
    def test_merge_dedupe_and_rerank(self, monkeypatch):
        archive = {2020: _store_rows(),
                   2025: [{"player_name": "Stored Star",
                           "position": "WR", "draft_class_year": 2025,
                           "school": "Big U", "actual_pick": 3,
                           "prospect_score": 70.0,
                           "production_score": 60.0,
                           "efficiency_score": 60.0,
                           "breakout_profile_score": 60.0,
                           "competition_score": 60.0,
                           "model_version": "3.7"}]}
        _patch_archive(monkeypatch, archive)
        stored_2025 = [{
            "player_id": "ROOKIE_2025_stored_star", "name": "Stored Star",
            "position": "WR", "draft_class_year": 2025,
            "prospect_score": 93.0, "overall_rank": 1, "position_rank": 1,
            "rank_delta": 4, "percentiles": {}, "tag": None,
        }]
        stored_2027 = [{
            "player_id": "ROOKIE_2027_top", "name": "Future Top",
            "position": "QB", "draft_class_year": 2027,
            "prospect_score": 95.0, "overall_rank": 1, "position_rank": 1,
            "rank_delta": 2, "percentiles": {}, "tag": None,
        }]
        monkeypatch.setattr(
            ra, "get_board_rows",
            lambda yr: stored_2025 if yr == 2025 else (
                stored_2027 if yr == 2027 else []))
        from data_building.rookie_pipeline import pipeline
        monkeypatch.setattr(pipeline, "get_active_rookie_class",
                            lambda: 2027)
        combined = ra._combined_board_rows()
        names = [d["name"] for d in combined]
        # Stored 2025 wins over its archive twin: one Stored Star,
        # carrying the stored grade, not the archive one.
        assert names.count("Stored Star") == 1
        star = next(d for d in combined if d["name"] == "Stored Star")
        assert star["prospect_score"] == 93.0
        # Ranked together by grade across classes.
        assert names[0] == "Future Top"      # 95.0
        assert names[1] == "Stored Star"     # 93.0
        assert names[2] == "Beta Back"       # 91.5 (archive, 2020)
        assert [d["overall_rank"] for d in combined] == list(
            range(1, len(combined) + 1))
        # Per-class deltas are meaningless on the combined board.
        assert all(d["rank_delta"] is None for d in combined)
        assert star["draft_class_year"] == 2025


# ── Detail branch ───────────────────────────────────────────────────

class TestArchiveDetail:
    def test_minimal_detail_no_fabricated_sections(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        pid = ra.archive_player_id(2020, "WR", "Alpha Wideout")
        detail = ra._archive_detail(pid)
        assert detail is not None
        assert detail["name"] == "Alpha Wideout"
        assert detail["prospect_score"] == 88.0
        assert detail["archive"] is True
        assert detail["class_size"] == 4
        assert detail["position_class_size"] == 2
        # Nothing the archive does not store is attached.
        for absent in ("seasons", "season_lines", "advanced",
                       "archetype", "key_reasons", "value",
                       "display_value"):
            assert absent not in detail

    def test_unknown_archive_player_is_none(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        assert ra._archive_detail("ARCHIVE_2020_WR_nobody-real") is None
        assert ra._archive_detail("ROOKIE_2027_x") is None

    def test_id_round_trip(self):
        pid = ra.archive_player_id(2019, "TE", "T.J. Hockenson")
        assert ra._parse_archive_player_id(pid) == (
            2019, "TE", "t-j-hockenson")


# ── Endpoints ───────────────────────────────────────────────────────

def _client():
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(ra.rookie_bp)
    return app.test_client()


class TestArchiveEndpoints:
    def test_rankings_falls_back_to_archive(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        monkeypatch.setattr(ra, "_get_rankings", lambda yr: [])
        resp = _client().get("/api/prospects/rankings?year=2020")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["archive"] is True
        assert payload["draft_class_year"] == 2020
        assert payload["total_players"] == 4
        assert payload["rankings"][0]["name"] == "Beta Back"
        # No dynasty values are fabricated for archive rows.
        assert "display_value" not in payload["rankings"][0]

    def test_rankings_all_combines(self, monkeypatch):
        row = {"player_id": "ARCHIVE_2020_RB_beta-back",
               "name": "Beta Back", "position": "RB",
               "draft_class_year": 2020, "prospect_score": 91.5,
               "overall_rank": 1, "position_rank": 1, "archive": True}
        monkeypatch.setattr(ra, "_combined_board_rows", lambda: [row])
        resp = _client().get("/api/prospects/rankings?year=all")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["draft_class_year"] == "all"
        assert payload["rankings"][0]["draft_class_year"] == 2020

    def test_player_detail_archive_branch(self, monkeypatch):
        _patch_archive(monkeypatch, {2020: _store_rows()})
        pid = ra.archive_player_id(2020, "WR", "Alpha Wideout")
        resp = _client().get(f"/api/prospects/player/{pid}")
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "Alpha Wideout"
        resp404 = _client().get(
            "/api/prospects/player/ARCHIVE_2020_WR_ghost-player")
        assert resp404.status_code == 404


# ── Build script + migration ────────────────────────────────────────

class TestBuildArchiveRows:
    def _inputs(self):
        draft_class = [
            {"gsis_id": "00-1", "name": "Tested Wideout",
             "position": "WR", "draft_pick": 15, "college": "State U"},
            {"gsis_id": "00-2", "name": "Untested Back",
             "position": "RB", "draft_pick": 88, "college": "Tech"},
        ]
        prospects = [
            {"player_id": "00-1", "name": "Tested Wideout",
             "position": "WR", "school": "State U"},
            {"player_id": "00-2", "name": "Untested Back",
             "position": "RB"},
        ]
        scores = [
            {"player_id": "00-1", "position": "WR",
             "prospect_score": 87.5, "production_score": 88.0,
             "efficiency_score": 81.0, "breakout_profile_score": 77.0,
             "competition_score": 66.0},
            {"player_id": "00-2", "position": "RB",
             "production_score": 74.0},  # no grade: no archive row
        ]
        return draft_class, prospects, scores

    def test_archive_row_shape(self):
        draft_class, prospects, scores = self._inputs()
        rows = refpop.build_archive_rows(
            2024, draft_class, prospects, scores, "3.7")
        assert len(rows) == 1
        row = rows[0]
        assert row["player_name"] == "Tested Wideout"
        assert row["prospect_score"] == 87.5
        assert row["competition_score"] == 66.0
        assert row["school"] == "State U"
        assert row["actual_pick"] == 15
        assert row["model_version"] == "3.7"

    def test_school_falls_back_to_draft_class_college(self):
        draft_class, prospects, scores = self._inputs()
        prospects[0].pop("school")
        scores[1]["prospect_score"] = 70.0
        rows = refpop.build_archive_rows(
            2024, draft_class, prospects, scores, "3.7")
        by_name = {r["player_name"]: r for r in rows}
        assert by_name["Tested Wideout"]["school"] == "State U"
        assert by_name["Untested Back"]["school"] == "Tech"

    def test_archive_upsert_targets_archive_table(self):
        assert "prospect_class_archive" in refpop._ARCHIVE_UPSERT_SQL
        assert ("ON CONFLICT (position, draft_class_year, player_name)"
                in refpop._ARCHIVE_UPSERT_SQL)

    def test_archive_years_span_stored_classes(self):
        assert refpop.ARCHIVE_YEARS[0] == 2016
        assert 2025 in refpop.ARCHIVE_YEARS
        assert 2026 in refpop.ARCHIVE_YEARS


class TestMigration058:
    def test_migration_shape(self):
        files = sorted(Path("migrations").glob("058_*.sql"))
        assert len(files) == 1
        sql = files[0].read_text().lower()
        assert "create table if not exists prospect_class_archive" in sql
        assert "unique (position, draft_class_year, player_name)" in sql
        for col in ("player_name", "position", "draft_class_year",
                    "school", "actual_pick", "prospect_score",
                    "production_score", "efficiency_score",
                    "breakout_profile_score", "competition_score",
                    "model_version"):
            assert col in sql, col


# ── Page ────────────────────────────────────────────────────────────

class TestProspectsPageClassSelector:
    def test_selector_markup_and_views(self):
        from dashboard_services.pages.rookies_page import (
            build_prospects_body)
        body = build_prospects_body(False)
        assert 'id="rkClass"' in body
        assert 'for="rkClass"' in body
        assert '<option value="all">All</option>' in body
        assert 'id="rkArchiveNote"' in body
        assert "Past classes graded with the current model" in body
        assert "rkSetClass" in body
        # The viewed class drives the CSV filename.
        assert "rkViewSel || rkDraftYear" in body
