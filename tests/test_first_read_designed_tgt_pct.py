"""First-Read Tgt % and Designed Tgt % from FTN charting (read_thrown).

first_read_tgt_pct = targets with read_thrown == "1" / charted targets
designed_tgt_pct   = targets with read_thrown == "DES" / charted targets

read_thrown is a string code in the FTN release via nflverse:
"1" = first (primary) read, "2" = later read, "CHK" = checkdown,
"DES" = designed play with no progression (screens, RPOs), "SD" =
scramble drill, "0" = non-throw default. The coding is validated for
2023-2026; 2022 primary reads are not coded, so both metrics are
2023+ only while the other FTN metrics keep the 2022 floor. A missing
read_thrown column skips both metrics with a warning, never zeros.
Free metrics (FTN via nflverse, not PFF), mirroring screen_target_rate.
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

QB, WR = "00-0000001", "00-0000002"
SL = {QB: "1001", WR: "1002"}

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
def _patch(monkeypatch):
    monkeypatch.setattr(nvm, "_gsis_to_sleeper", lambda: dict(SL))
    monkeypatch.setattr(nvm, "build_pfr_contact_yards_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_broken_tackles_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_rec_broken_tackles_weekly", lambda season: {})


def _pbp_row(play_id, **kw):
    base = {
        "game_id": "g1", "play_id": play_id, "week": 1, "season_type": "REG",
        "play_type": "pass", "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0,
        "success": 1, "sack": 0, "qb_scramble": 0, "qb_hit": 0,
        "rush_attempt": 0, "pass_attempt": 1, "qb_dropback": 1,
        "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
        "complete_pass": 1, "yards_after_catch": 2, "passing_yards": 8,
        "pass_touchdown": 0, "interception": 0,
        "first_down_pass": 0, "first_down_rush": 0,
        "posteam": "KC", "yardline_100": 50, "down": 1,
        "third_down_converted": 0,
        "passer_player_id": QB, "rusher_player_id": None,
        "receiver_player_id": WR,
    }
    base.update(kw)
    return base


def _ftn_row(play_id, **kw):
    base = {
        "nflverse_game_id": "g1", "nflverse_play_id": play_id,
        "is_catchable_ball": 1, "is_drop": 0, "is_contested_ball": 0,
        "is_screen_pass": 0, "is_interception_worthy": 0,
        "is_throw_away": 0, "is_play_action": 0, "is_qb_out_of_pocket": 0,
        "n_blitzers": 0, "n_defense_box": 6,
        "read_thrown": "2",
    }
    base.update(kw)
    return base


# 10 charted WR targets: 6 first reads, 2 designed, 1 later read,
# 1 scramble drill; 2 of the designed targets are also screen passes.
READS = ["1", "1", "1", "1", "1", "1", "DES", "DES", "2", "SD"]


def _frames(reads=READS):
    pbp, ftn = [], []
    for i, code in enumerate(reads):
        pid = i + 1
        pbp.append(_pbp_row(pid))
        ftn.append(_ftn_row(
            pid,
            read_thrown=code,
            is_screen_pass=1 if code == "DES" else 0,
        ))
    return pd.DataFrame(ftn), pd.DataFrame(pbp)


def _install(monkeypatch, ftn, pbp):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))


# --------------------------------------------------------------------------- #
# Season builder
# --------------------------------------------------------------------------- #
def test_ftn_season_first_read_and_designed(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2025)
    assert out["1002"]["first_read_tgt_pct"] == pytest.approx(60.0)
    assert out["1002"]["designed_tgt_pct"] == pytest.approx(20.0)
    # The existing screen metric on the same rows is unaffected.
    assert out["1002"]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_season_2022_has_no_read_metrics(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2022)
    # 2022 primary reads are not coded in FTN: neither key is emitted,
    # while the other FTN metrics keep their 2022 floor behavior.
    assert "first_read_tgt_pct" not in out.get("1002", {})
    assert "designed_tgt_pct" not in out.get("1002", {})
    assert out["1002"]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_season_missing_read_column_skips_both(monkeypatch, capsys):
    ftn, pbp = _frames()
    ftn = ftn.drop(columns=["read_thrown"])
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2025)
    assert "first_read_tgt_pct" not in out.get("1002", {})
    assert "designed_tgt_pct" not in out.get("1002", {})
    # No fabricated zeros, other metrics unaffected, and the skip is loud.
    assert out["1002"]["screen_target_rate"] == pytest.approx(20.0)
    assert "read_thrown" in capsys.readouterr().out


def test_ftn_season_only_exact_codes_count(monkeypatch):
    # "0" (non-throw default), "CHK" (checkdown) and "2" (later read)
    # must not count as a first read or a designed target; only the
    # codes "1" / "DES" do.
    ftn, pbp = _frames(reads=["1", "DES", "0", "CHK", "2"])
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2025)
    assert out["1002"]["first_read_tgt_pct"] == pytest.approx(20.0)
    assert out["1002"]["designed_tgt_pct"] == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Weekly builder
# --------------------------------------------------------------------------- #
def test_ftn_weekly_first_read_and_designed(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[("1002", 1)]["first_read_tgt_pct"] == pytest.approx(60.0)
    assert out[("1002", 1)]["designed_tgt_pct"] == pytest.approx(20.0)
    assert out[("1002", 1)]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_weekly_2022_has_no_read_metrics(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2022)
    assert "first_read_tgt_pct" not in out.get(("1002", 1), {})
    assert "designed_tgt_pct" not in out.get(("1002", 1), {})
    assert out[("1002", 1)]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_weekly_missing_read_column_skips_both(monkeypatch, capsys):
    ftn, pbp = _frames()
    ftn = ftn.drop(columns=["read_thrown"])
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert "first_read_tgt_pct" not in out.get(("1002", 1), {})
    assert "designed_tgt_pct" not in out.get(("1002", 1), {})
    assert out[("1002", 1)]["screen_target_rate"] == pytest.approx(20.0)
    assert "read_thrown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Metric spec / wiring
# --------------------------------------------------------------------------- #
def test_metric_specs_mirror_screen_target_rate():
    screen = am.LEADERBOARD_METRICS["screen_target_rate"]
    for key, label in (("first_read_tgt_pct", "First-Read Tgt %"),
                       ("designed_tgt_pct", "Designed Tgt %")):
        spec = am.LEADERBOARD_METRICS[key]
        assert spec["label"] == label
        assert spec["category"] == "Receiving"
        assert spec["positions"] == ["WR", "RB", "TE"]
        assert spec["efficiency"] is True
        assert spec["pct"] is True
        assert spec.get("lower_better") is not True
        assert spec["min_vol"] == screen["min_vol"]
        assert "FTN" in spec["desc"]
        assert "not PFF" in spec["desc"]
        assert "\u2014" not in spec["desc"]  # no em dashes in UI copy


def test_read_metrics_are_free():
    for key in ("first_read_tgt_pct", "designed_tgt_pct"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS


def test_weekly_column_weight_and_ddl_wiring():
    for key in ("first_read_tgt_pct", "designed_tgt_pct"):
        assert key in am.WEEKLY_ADV_METRIC_COLS
        assert am._ADV_WEEKLY_WEIGHTED_METRICS[key] == "w_targets"
        assert f"ADD COLUMN IF NOT EXISTS {key}" in inspect.getsource(am)
        assert key in inspect.getsource(am.get_player_career_metrics)


def test_js_and_page_wiring():
    app_js = (REPO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    modal_js = (REPO_ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    page_py = (REPO_ROOT / "dashboard_services" / "pages"
               / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert "first_read_tgt_pct:'First-Read Tgt %'" in app_js
    assert "designed_tgt_pct:'Designed Tgt %'" in app_js
    assert "first_read_tgt_pct: 'w_targets'" in app_js
    assert "designed_tgt_pct: 'w_targets'" in app_js
    assert "first_read_tgt_pct:" in modal_js
    assert "designed_tgt_pct:" in modal_js
    assert "first_read_tgt_pct: 'w_targets'" in page_py
    assert "designed_tgt_pct: 'w_targets'" in page_py


def test_target_quality_preset_includes_read_metrics():
    from dashboard_services.pages.advanced_metrics_page import (
        ADVANCED_METRIC_PRESETS,
    )
    metrics = ADVANCED_METRIC_PRESETS["target_quality"]["metrics"]
    assert "screen_target_rate" in metrics
    assert "first_read_tgt_pct" in metrics
    assert "designed_tgt_pct" in metrics
