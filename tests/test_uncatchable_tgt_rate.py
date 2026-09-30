"""Uncatchable Tgt % (uncatchable_tgt_rate) from FTN charting.

Percent of a receiver's FTN-charted targets where is_catchable_ball is
false: (charted_targets - catchable) / charted_targets * 100. FTN
charting via the nflverse FTN release (the same source as drop_rate /
contested_catch_rate), not PFF. Free metric: the drop_rate /
contested_catch_rate precedent says a metric populated from free data
is public-safe, so it stays out of PREMIUM_METRICS and PRO_METRICS.

Covers, with small synthetic fixtures and no network:
- season builder math, including a 0-catchable week/season -> 100.0
  (drop_rate is guarded on catchable > 0; uncatchable must NOT be, or a
  fully-uncatchable sample would vanish instead of reading 100.0)
- no charted targets -> the key is absent, never a fabricated 0.0
- weekly builder math at (player, week) granularity
- metric spec (label / category / positions / lower_better / free),
  weekly column + w_targets weighting, and the JS glossary/weight
  wiring that mirrors drop_rate
"""

import inspect
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm
from data_building import advanced_metrics as am

GSIS = "00-0036212"
SLEEPER = "6000"
QB_GSIS = "00-0099999"  # not in the crosswalk; passer-side rows are skipped

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeNfl(types.SimpleNamespace):
    def __init__(self, ftn=None, pbp=None):
        super().__init__()
        self._ftn = ftn
        self._pbp = pbp

    def import_ftn_data(self, years):
        return self._ftn.copy()

    def import_pbp_data(self, years, columns=None, downcast=False):
        return self._pbp.copy()

    def import_ngs_data(self, stat_type=None, years=None):
        return pd.DataFrame({"season_type": [], "week": []})


@pytest.fixture(autouse=True)
def _patch_crosswalk_and_pfr(monkeypatch):
    monkeypatch.setattr(nvm, "_gsis_to_sleeper", lambda: {GSIS: SLEEPER})
    # The weekly builder also merges PFR weekly files (network downloads);
    # stub them out, they are not under test here.
    monkeypatch.setattr(nvm, "build_pfr_contact_yards_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_broken_tackles_weekly", lambda season: {})


def _install_nfl(monkeypatch, ftn=None, pbp=None):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))


def _target_plays(catchable_flags, week=1):
    """One pbp target play per flag, all targeted at the fixture receiver."""
    plays = []
    for i, _flag in enumerate(catchable_flags):
        plays.append({
            "game_id": "g1", "play_id": 100 + i, "week": week,
            "season_type": "REG", "play_type": "pass",
            "passer_player_id": QB_GSIS, "rusher_player_id": None,
            "receiver_player_id": GSIS,
            "pass_attempt": 1, "rush_attempt": 0, "qb_dropback": 1,
            "complete_pass": 1 if _flag else 0,
            "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0, "success": 1,
            "sack": 0, "qb_scramble": 0, "qb_hit": 0,
            "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
            "yards_after_catch": 2, "passing_yards": 8,
            "pass_touchdown": 0, "interception": 0,
        })
    return plays


def _ftn_frame(plays, catchable_flags):
    rows = []
    for i, p in enumerate(plays):
        rows.append({
            "nflverse_game_id": p["game_id"], "nflverse_play_id": p["play_id"],
            "week": p["week"],
            "is_catchable_ball": catchable_flags[i], "is_drop": 0,
            "is_contested_ball": 0, "is_throw_away": 0, "is_play_action": 0,
            "is_qb_out_of_pocket": 0, "n_defense_box": 6, "n_blitzers": 0,
            "n_pass_rushers": 4,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Season builder
# --------------------------------------------------------------------------- #
def test_season_uncatchable_rate_math(monkeypatch):
    flags = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 7 of 10 catchable
    plays = _target_plays(flags)
    _install_nfl(monkeypatch, ftn=_ftn_frame(plays, flags), pbp=pd.DataFrame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    assert out[SLEEPER]["uncatchable_tgt_rate"] == pytest.approx(30.0)


def test_season_zero_catchable_is_100_not_absent(monkeypatch):
    flags = [0, 0, 0, 0]
    plays = _target_plays(flags)
    _install_nfl(monkeypatch, ftn=_ftn_frame(plays, flags), pbp=pd.DataFrame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    assert out[SLEEPER]["uncatchable_tgt_rate"] == pytest.approx(100.0)
    # drop_rate keeps its catchable > 0 guard and stays absent here; the
    # uncatchable rate must not inherit that guard.
    assert "drop_rate" not in out[SLEEPER]


def test_season_all_catchable_is_zero(monkeypatch):
    flags = [1, 1, 1]
    plays = _target_plays(flags)
    _install_nfl(monkeypatch, ftn=_ftn_frame(plays, flags), pbp=pd.DataFrame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    assert out[SLEEPER]["uncatchable_tgt_rate"] == pytest.approx(0.0)


def test_season_no_targets_no_uncatchable_key(monkeypatch):
    plays = _target_plays([1, 0])
    for p in plays:
        p["receiver_player_id"] = None
    _install_nfl(monkeypatch, ftn=_ftn_frame(plays, [1, 0]), pbp=pd.DataFrame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    assert "uncatchable_tgt_rate" not in out.get(SLEEPER, {})


# --------------------------------------------------------------------------- #
# Weekly builder
# --------------------------------------------------------------------------- #
def test_weekly_uncatchable_rate_math(monkeypatch):
    wk1_flags = [1, 0, 0, 0]   # 1 of 4 catchable -> 75.0
    wk2_flags = [0, 0]         # 0 of 2 catchable -> 100.0, not absent
    plays = _target_plays(wk1_flags, week=1) + _target_plays(wk2_flags, week=2)
    for i, p in enumerate(plays):
        p["play_id"] = 100 + i
    flags = wk1_flags + wk2_flags
    _install_nfl(monkeypatch, ftn=_ftn_frame(plays, flags), pbp=pd.DataFrame(plays))

    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[(SLEEPER, 1)]["uncatchable_tgt_rate"] == pytest.approx(75.0)
    assert out[(SLEEPER, 2)]["uncatchable_tgt_rate"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Metric spec / wiring
# --------------------------------------------------------------------------- #
def test_metric_spec_matches_drop_rate_shape():
    spec = am.LEADERBOARD_METRICS["uncatchable_tgt_rate"]
    drop = am.LEADERBOARD_METRICS["drop_rate"]
    assert spec["label"] == "Uncatchable Tgt %"
    assert spec["category"] == "Receiving"
    assert spec["positions"] == ["WR", "RB", "TE"]
    assert spec["efficiency"] is True
    assert spec["pct"] is True
    assert spec["lower_better"] is True
    assert spec["min_vol"] == drop["min_vol"]
    assert "FTN" in spec["desc"]
    assert "\u2014" not in spec["desc"]  # no em dashes in UI copy


def test_uncatchable_is_free():
    assert "uncatchable_tgt_rate" not in am.PRO_METRICS
    assert "uncatchable_tgt_rate" not in am.PREMIUM_METRICS


def test_weekly_column_weight_and_ddl_wiring():
    assert "uncatchable_tgt_rate" in am.WEEKLY_ADV_METRIC_COLS
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["uncatchable_tgt_rate"] == "w_targets"
    assert "uncatchable_tgt_rate" in inspect.getsource(am.get_player_career_metrics)
    assert "ADD COLUMN IF NOT EXISTS uncatchable_tgt_rate" in inspect.getsource(am)


def test_js_glossary_and_weight_wiring():
    app_js = (REPO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    modal_js = (REPO_ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    page_py = (REPO_ROOT / "dashboard_services" / "pages"
               / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert "uncatchable_tgt_rate:'Uncatchable Tgt %'" in app_js
    assert "uncatchable_tgt_rate: 'w_targets'" in app_js
    assert "uncatchable_tgt_rate:" in modal_js
    assert "uncatchable_tgt_rate: 'w_targets'" in page_py
