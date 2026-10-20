"""Tests for the v4 GO-LIVE WIRING (pipeline scorer switch, strict
reference gate, backtest --model flag, calibration panel).

The wiring, not the model: prospect_model_v4's architecture is
covered by tests/test_prospect_v4.py. Here we verify that a run can
SELECT v4, that v4 without its recorded reference artifact fails
loudly at startup (never grades anchorless, never falls back
silently), that the fixture sample can never satisfy a production
check unless explicitly allowed, that the backtest's default (v3)
path is untouched by the new flag, and that v4 backtest rows are
the v4 model's own scores.

Synthetic data only; no DB, no network.
"""

import copy
import json
import os
import sys

import pytest

pytest.importorskip("pandas")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import pipeline  # noqa: E402
from data_building.rookie_pipeline import prospect_model_v4 as v4  # noqa: E402
from data_building.rookie_pipeline.prospect_model import (  # noqa: E402
    MODEL_VERSION as INCUMBENT_VERSION,
    score_all_prospects,
)
from test_prospect_v32 import GREEN, MESTEMAKER, SMITH, _dc  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(
    _REPO, "data_building", "rookie_pipeline",
    "v4_reference_fixture_sample.json")


@pytest.fixture(scope="module")
def sample_ref():
    return v4.load_reference(SAMPLE_PATH)


# ─────────────────────────────────────────────────────────────────────
# Strict reference loading
# ─────────────────────────────────────────────────────────────────────

class TestStrictReference:
    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(v4.V4ReferenceError, match="not found"):
            v4.load_reference_strict(str(tmp_path / "nope.json"))

    def test_fixture_sample_rejected_for_production(self):
        with pytest.raises(v4.V4ReferenceError, match="FIXTURE SAMPLE"):
            v4.load_reference_strict(SAMPLE_PATH)

    def test_fixture_sample_allowed_when_flagged(self, sample_ref):
        data = v4.load_reference_strict(SAMPLE_PATH, allow_fixture=True)
        assert data["provenance"]["fixture_sample"] is True
        assert data == sample_ref

    def test_recorded_artifact_accepted(self, tmp_path):
        path = tmp_path / "v4_reference.json"
        path.write_text(json.dumps({
            "provenance": {"fixture_sample": False,
                           "source": "recorded test artifact"},
            "anchors": {"WR": {"rec_yds_pg": [1, 2, 3]}},
            "athleticism": {}, "percentiles": {}, "populations": {},
        }))
        data = v4.load_reference_strict(str(path))
        assert data["anchors"]["WR"]["rec_yds_pg"] == [1, 2, 3]

    def test_recorded_but_anchorless_rejected(self, tmp_path):
        path = tmp_path / "v4_reference.json"
        path.write_text(json.dumps({
            "provenance": {"fixture_sample": False},
            "anchors": {}, "athleticism": {},
        }))
        with pytest.raises(v4.V4ReferenceError, match="no anchor"):
            v4.load_reference_strict(str(path))


# ─────────────────────────────────────────────────────────────────────
# The selection switch
# ─────────────────────────────────────────────────────────────────────

class TestSwitchSelection:
    def test_default_is_the_pipeline_default(self, monkeypatch):
        monkeypatch.delenv("PROSPECT_MODEL", raising=False)
        assert pipeline.selected_prospect_model() == (
            "v4" if pipeline.DEFAULT_PROSPECT_MODEL == "v4" else "v3")

    @pytest.mark.parametrize("raw,want", [
        ("v3", "v3"), ("3.4", "v3"), ("v4", "v4"), ("4.0", "v4"),
        (" V4 ", "v4"),
    ])
    def test_env_values(self, monkeypatch, raw, want):
        monkeypatch.setenv("PROSPECT_MODEL", raw)
        assert pipeline.selected_prospect_model() == want

    def test_unknown_selection_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("PROSPECT_MODEL", "v9")
        with pytest.raises(ValueError, match="Unrecognized"):
            pipeline.selected_prospect_model()


class TestStartupGate:
    def test_v4_missing_artifact_fails_at_startup(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROSPECT_MODEL", "v4")
        monkeypatch.delenv("PROSPECT_V4_ALLOW_FIXTURE", raising=False)
        monkeypatch.setattr(v4, "_REFERENCE_DEFAULT", str(tmp_path / "absent.json"))
        with pytest.raises(v4.V4ReferenceError):
            pipeline.assert_scorer_ready()
        # Scoring time re-checks: no path reaches grading anchorless.
        with pytest.raises(v4.V4ReferenceError):
            pipeline._score_prospects([copy.deepcopy(SMITH)], {})

    def test_v4_fixture_artifact_rejected_without_escape(self, monkeypatch):
        monkeypatch.setenv("PROSPECT_MODEL", "v4")
        monkeypatch.delenv("PROSPECT_V4_ALLOW_FIXTURE", raising=False)
        monkeypatch.setattr(v4, "_REFERENCE_DEFAULT", SAMPLE_PATH)
        with pytest.raises(v4.V4ReferenceError, match="FIXTURE SAMPLE"):
            pipeline.assert_scorer_ready()

    def test_v3_needs_no_artifact(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PROSPECT_MODEL", "v3")
        monkeypatch.setattr(v4, "_REFERENCE_DEFAULT", str(tmp_path / "absent.json"))
        assert pipeline.assert_scorer_ready() == "v3"


# ─────────────────────────────────────────────────────────────────────
# Scoring through the switch
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def v4_fixture_route(monkeypatch):
    """The test-only route to a v4 run: fixture sample as the
    artifact, explicitly allowed. Production never sets the escape."""
    monkeypatch.setenv("PROSPECT_MODEL", "v4")
    monkeypatch.setenv("PROSPECT_V4_ALLOW_FIXTURE", "1")
    monkeypatch.setattr(v4, "_REFERENCE_DEFAULT", SAMPLE_PATH)


class TestScoreThroughSwitch:
    def test_v4_run_stamps_version_and_storage_fields(self, v4_fixture_route):
        prospect = copy.deepcopy(SMITH)
        dc = _dc("WR", 2)
        scores = pipeline._score_prospects(
            [prospect], {prospect["player_id"]: dc}, draft_year=2027)
        assert len(scores) == 1
        row = scores[0]
        assert row["model_version"] == v4.MODEL_VERSION
        assert row["reference_kind"] == "fixture_sample"
        assert row["prospect_score"] is not None
        assert row["talent_score"] is not None
        # Storage/display fields the incumbent stamps inside its scorer.
        assert row["draft_class_year"] == 2027
        assert "archetype" in row and "archetype_receipt" in row
        assert isinstance(row["key_reasons"], str)
        assert row["fantasy_translation_score"] is not None

    def test_incumbent_path_unchanged(self, monkeypatch):
        monkeypatch.setenv("PROSPECT_MODEL", "v3")
        prospect = copy.deepcopy(SMITH)
        dc = _dc("WR", 2)
        scores = pipeline._score_prospects(
            [prospect], {prospect["player_id"]: dc})
        assert scores[0]["model_version"] == INCUMBENT_VERSION


# ─────────────────────────────────────────────────────────────────────
# Backtest: the flag must not move the default path
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def backtest(monkeypatch):
    import scripts.backtest_prospect_model as bt
    monkeypatch.setattr(bt, "_MODEL_CHOICE", "v3")
    monkeypatch.setattr(bt, "_V4_REFERENCE", None)
    return bt


def _fixture_class(bt):
    """A tiny drafted class built from the repo fixtures."""
    entries = [
        {"name": "Jeremiah Smith", "position": "WR", "gsis_id": "G1",
         "draft_pick": 2, "college": "Ohio State", "birth_date": "2005-11-28"},
        {"name": "Trey'Dez Green", "position": "TE", "gsis_id": "G2",
         "draft_pick": 20, "college": "LSU", "birth_date": "2004-05-01"},
        {"name": "Drew Mestemaker", "position": "QB", "gsis_id": "G3",
         "draft_pick": 8, "college": "Oklahoma State",
         "birth_date": "2005-01-15"},
    ]
    stats = {
        "G1": copy.deepcopy(SMITH["seasons"]),
        "G2": copy.deepcopy(GREEN["seasons"]),
        "G3": copy.deepcopy(MESTEMAKER["seasons"]),
    }
    return entries, stats


def _stub_loaders(monkeypatch, bt, entries, stats):
    monkeypatch.setattr(bt, "_load_draft_class", lambda year: entries)
    monkeypatch.setattr(bt, "_load_combine_athleticism", lambda year: {})
    monkeypatch.setattr(bt, "_load_cfbd_college_stats",
                        lambda year, dc, fetch_games_played=False: stats)
    monkeypatch.setattr(bt, "_build_nfl_ppr_per_player",
                        lambda ids, year: {})


class TestBacktestFlag:
    def test_default_choice_is_incumbent(self):
        import scripts.backtest_prospect_model as bt
        assert bt._MODEL_CHOICE == "v3"

    def test_default_path_matches_direct_incumbent_scoring(
            self, backtest, monkeypatch):
        bt = backtest
        entries, stats = _fixture_class(bt)
        _stub_loaders(monkeypatch, bt, entries, stats)
        rows = bt._run_draft_class_backtest(2099)
        prospects, cmap = bt._build_prospect_dicts(entries, {}, 2099, stats)
        expected = score_all_prospects(prospects, cmap, skip_sagarin=True)
        pid_to_name = {p["player_id"]: p["name"] for p in prospects}
        got = {r["name"]: r["model_score"] for r in rows}
        want = {pid_to_name[s["player_id"]]: s["prospect_score"]
                for s in expected}
        assert got == want

    def test_v4_path_matches_direct_v4_scoring(
            self, backtest, monkeypatch, sample_ref):
        bt = backtest
        entries, stats = _fixture_class(bt)
        _stub_loaders(monkeypatch, bt, entries, stats)
        monkeypatch.setattr(bt, "_MODEL_CHOICE", "v4")
        monkeypatch.setattr(bt, "_V4_REFERENCE", sample_ref)
        rows = bt._run_draft_class_backtest(2099)
        prospects, cmap = bt._build_prospect_dicts(entries, {}, 2099, stats)
        expected = v4.score_all_v4(prospects, cmap, reference=sample_ref,
                                   cycle_complete=True)
        pid_to_name = {p["player_id"]: p["name"] for p in prospects}
        got = {r["name"]: r["model_score"] for r in rows}
        want = {pid_to_name[s["player_id"]]: s["prospect_score"]
                for s in expected if s["prospect_score"] is not None}
        assert got == want
        assert all(r["model_score"] is not None for r in rows)

    def test_v4_run_requires_reference(self, backtest):
        with pytest.raises(ValueError, match="requires the loaded reference"):
            backtest.run_backtest([2099], save_grades=False, model="v4")


# ─────────────────────────────────────────────────────────────────────
# Percentile pool: stored v4 rows count toward the board population
# ─────────────────────────────────────────────────────────────────────

class TestPercentilePoolVersions:
    def test_v4_rows_count_toward_the_pool(self):
        from dashboard_services import rookie_api as ra
        assert "4.0" in ra._PCT_MODEL_VERSIONS
        # The reference-population query binds the same version set via
        # the constant (single source of truth since #2710's refactor);
        # keep the two from drifting apart again.
        import inspect
        src = inspect.getsource(ra.reference_population)
        assert "list(_PCT_MODEL_VERSIONS)" in src


# ─────────────────────────────────────────────────────────────────────
# Calibration panel printer
# ─────────────────────────────────────────────────────────────────────

class TestCalibrationPanel:
    def test_scored_and_unscored_players(self, backtest, capsys):
        bt = backtest
        rows = [
            {"name": "Trevor Lawrence", "position": "QB",
             "model_score": 91.5, "draft_year": 2021},
            {"name": "Kyle Pitts", "position": "TE",
             "model_score": 97.25, "draft_year": 2021},
        ]
        bt._print_calibration_panel(rows)
        out = capsys.readouterr().out
        assert "Trevor Lawrence" in out and "91.50" in out
        assert "Great Prospect" in out          # 91.5 is the great band
        assert "Kyle Pitts" in out and "Elite Prospect" in out
        # Absent from the population: unscored, never fabricated.
        assert "Saquon Barkley" in out and "unscored" in out
        assert "Jeremiah Smith" in out          # fixture: unscored here
