"""First Read % (passer) and First-Read Share (receiver) from FTN.

first_read_pct   = a passer's targeted throws coded first read ("1")
                   or designed ("DES") / all of his targeted throws
first_read_share = a receiver's first-read + designed targets / his
                   team's first-read + designed targeted throws

Designed throws count as first-read throws in both metrics: the
designed target is the first look by construction. That matches the
definitions Fantasy Points publishes (First Read Percentage for QBs,
First Read Target Share for receivers), which are the versions
readers cross-check against. first_read_tgt_pct / designed_tgt_pct
stay separate so the two looks can still be told apart.

read_thrown is a string code in the FTN release via nflverse:
"1" = first (primary) read, "2" = later read, "CHK" = checkdown,
"DES" = designed play with no progression (screens, RPOs), "SD" =
scramble drill, "0" = non-throw default. The coding is validated for
2023-2026; 2022 primary reads are not coded, so both metrics are
2023+ only while the other FTN metrics keep the 2022 floor. A
missing read_thrown column skips both metrics with a warning, never
zeros. Free metrics (FTN via nflverse, not PFF), mirroring
first_read_tgt_pct / designed_tgt_pct.
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

QB1, QB2 = "00-0000011", "00-0000012"
WR1, WR2, WR3 = "00-0000021", "00-0000022", "00-0000023"
SL = {QB1: "1101", QB2: "1102", WR1: "1201", WR2: "1202", WR3: "1203"}

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
        "passer_player_id": QB1, "rusher_player_id": None,
        "receiver_player_id": WR1,
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


# KC (QB1 throwing): WR1 gets 10 targets (6 first reads, 2 designed,
# 2 later reads) and WR2 gets 10 targets (2 first reads, 8 later
# reads), so KC has 10 first-read + designed throws out of 20.
# BUF (QB2 throwing): WR3 gets 4 targets (3 first reads, 1 designed),
# so BUF has 4 first-read + designed throws out of 4.
_PLAYS = (
    [(QB1, WR1, "KC", code)
     for code in ["1", "1", "1", "1", "1", "1", "DES", "DES", "2", "2"]]
    + [(QB1, WR2, "KC", code)
       for code in ["1", "1", "2", "2", "2", "2", "2", "2", "2", "2"]]
    + [(QB2, WR3, "BUF", code)
       for code in ["1", "1", "1", "DES"]]
)


def _frames():
    pbp, ftn = [], []
    for i, (passer, receiver, team, code) in enumerate(_PLAYS):
        pid = i + 1
        pbp.append(_pbp_row(
            pid, passer_player_id=passer, receiver_player_id=receiver,
            posteam=team,
        ))
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
def test_ftn_season_first_read_pct_and_share(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2025)
    # QB: (first + designed) / targeted throws, DES folded in.
    assert out["1101"]["first_read_pct"] == pytest.approx(50.0)
    assert out["1102"]["first_read_pct"] == pytest.approx(100.0)
    # Share: player first + designed over HIS TEAM's first + designed
    # throws (KC denominator 10, BUF denominator 4, never pooled).
    assert out["1201"]["first_read_share"] == pytest.approx(80.0)
    assert out["1202"]["first_read_share"] == pytest.approx(20.0)
    assert out["1203"]["first_read_share"] == pytest.approx(100.0)
    # The sibling rates on the same rows are unaffected.
    assert out["1201"]["first_read_tgt_pct"] == pytest.approx(60.0)
    assert out["1201"]["designed_tgt_pct"] == pytest.approx(20.0)


def test_ftn_season_2022_has_no_new_read_metrics(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2022)
    # 2022 primary reads are not coded in FTN: neither new key (nor
    # the sibling keys) is emitted, while the other FTN metrics keep
    # their 2022 floor behavior.
    for pid in ("1101", "1102", "1201", "1202", "1203"):
        assert "first_read_pct" not in out.get(pid, {})
        assert "first_read_share" not in out.get(pid, {})
    assert "first_read_tgt_pct" not in out.get("1201", {})
    assert out["1201"]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_season_missing_read_column_skips_both(monkeypatch, capsys):
    ftn, pbp = _frames()
    ftn = ftn.drop(columns=["read_thrown"])
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_ftn_charting_for_season(2025)
    for pid in ("1101", "1102", "1201", "1202", "1203"):
        assert "first_read_pct" not in out.get(pid, {})
        assert "first_read_share" not in out.get(pid, {})
    # No fabricated zeros, other metrics unaffected, and the skip is loud.
    assert out["1201"]["screen_target_rate"] == pytest.approx(20.0)
    assert "read_thrown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Weekly builder
# --------------------------------------------------------------------------- #
def test_ftn_weekly_first_read_pct_and_share(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[("1101", 1)]["first_read_pct"] == pytest.approx(50.0)
    assert out[("1102", 1)]["first_read_pct"] == pytest.approx(100.0)
    assert out[("1201", 1)]["first_read_share"] == pytest.approx(80.0)
    assert out[("1202", 1)]["first_read_share"] == pytest.approx(20.0)
    assert out[("1203", 1)]["first_read_share"] == pytest.approx(100.0)
    assert out[("1201", 1)]["first_read_tgt_pct"] == pytest.approx(60.0)


def test_ftn_weekly_2022_has_no_new_read_metrics(monkeypatch):
    ftn, pbp = _frames()
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2022)
    for key in (("1101", 1), ("1102", 1), ("1201", 1), ("1202", 1),
                ("1203", 1)):
        assert "first_read_pct" not in out.get(key, {})
        assert "first_read_share" not in out.get(key, {})
    assert out[("1201", 1)]["screen_target_rate"] == pytest.approx(20.0)


def test_ftn_weekly_missing_read_column_skips_both(monkeypatch, capsys):
    ftn, pbp = _frames()
    ftn = ftn.drop(columns=["read_thrown"])
    _install(monkeypatch, ftn, pbp)
    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    for key in (("1101", 1), ("1102", 1), ("1201", 1), ("1202", 1),
                ("1203", 1)):
        assert "first_read_pct" not in out.get(key, {})
        assert "first_read_share" not in out.get(key, {})
    assert out[("1201", 1)]["screen_target_rate"] == pytest.approx(20.0)
    assert "read_thrown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Metric spec / wiring
# --------------------------------------------------------------------------- #
def test_metric_specs():
    qb = am.LEADERBOARD_METRICS["first_read_pct"]
    adj = am.LEADERBOARD_METRICS["adjusted_completion_rate"]
    assert qb["label"] == "First Read %"
    assert qb["category"] == "Passing"
    assert qb["positions"] == ["QB"]
    assert qb["efficiency"] is True
    assert qb["pct"] is True
    assert qb.get("lower_better") is not True
    # Throws-based minimum volume, same as the other QB throw rates.
    assert qb["min_vol"] == adj["min_vol"]
    assert "FTN" in qb["desc"]
    assert "not PFF" in qb["desc"]
    assert "\u2014" not in qb["desc"]  # no em dashes in UI copy

    share = am.LEADERBOARD_METRICS["first_read_share"]
    screen = am.LEADERBOARD_METRICS["screen_target_rate"]
    assert share["label"] == "First-Read Share"
    assert share["category"] == "Receiving"
    assert share["positions"] == ["WR", "RB", "TE"]
    assert share["efficiency"] is True
    assert share["pct"] is True
    assert share.get("lower_better") is not True
    assert share["min_vol"] == screen["min_vol"]
    assert "FTN" in share["desc"]
    assert "not PFF" in share["desc"]
    assert "\u2014" not in share["desc"]


def test_new_read_metrics_are_free():
    for key in ("first_read_pct", "first_read_share"):
        assert key not in am.PRO_METRICS
        assert key not in am.PREMIUM_METRICS


def test_weekly_column_weight_and_ddl_wiring():
    assert "first_read_pct" in am.WEEKLY_ADV_METRIC_COLS
    assert "first_read_share" in am.WEEKLY_ADV_METRIC_COLS
    # QB rate weights by pass attempts like adjusted completion;
    # the receiver share weights by targets like its siblings.
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["first_read_pct"] == "w_pass_att"
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["first_read_share"] == "w_targets"
    for key in ("first_read_pct", "first_read_share"):
        assert f"ADD COLUMN IF NOT EXISTS {key}" in inspect.getsource(am)
        assert key in inspect.getsource(am.get_player_career_metrics)


def test_js_and_page_wiring():
    app_js = (REPO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    modal_js = (REPO_ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    page_py = (REPO_ROOT / "dashboard_services" / "pages"
               / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert "first_read_pct:'First Read %'" in app_js
    assert "first_read_share:'First-Read Share'" in app_js
    assert "'first_read_pct': 95" in app_js
    assert "'first_read_share': 55" in app_js
    assert "first_read_pct: 'w_pass_att'" in app_js
    assert "first_read_share: 'w_targets'" in app_js
    assert "first_read_pct:" in modal_js
    assert "first_read_share:" in modal_js
    assert "first_read_pct: 'w_pass_att'" in page_py
    assert "first_read_share: 'w_targets'" in page_py
    assert "'first_read_pct': ['First read or designed', 'Later reads']" in page_py
    assert "'first_read_share': ['First-look throws', 'Rest of team']" in page_py


def test_target_quality_preset_includes_first_read_share():
    from dashboard_services.pages.advanced_metrics_page import (
        ADVANCED_METRIC_PRESETS,
    )
    metrics = ADVANCED_METRIC_PRESETS["target_quality"]["metrics"]
    assert "first_read_tgt_pct" in metrics
    assert "designed_tgt_pct" in metrics
    assert "first_read_share" in metrics


def test_qb_charting_preset_groups_ftn_qb_rates():
    from dashboard_services.pages.advanced_metrics_page import (
        ADVANCED_METRIC_PRESETS,
    )
    preset = ADVANCED_METRIC_PRESETS["qb_charting"]
    assert preset["primary"] == "first_read_pct"
    assert preset["position"] == "QB"
    # The FTN-charted QB rates that previously sat in no preset at all.
    assert preset["metrics"] == [
        "first_read_pct", "adjusted_completion_rate", "turnover_worthy_rate",
        "play_action_rate", "blitz_rate_faced", "out_of_pocket_rate",
    ]
    for key, other in ADVANCED_METRIC_PRESETS.items():
        if key != "qb_charting":
            assert "first_read_pct" not in other["metrics"]


def test_out_of_pocket_rate_registered():
    # The builders always computed/stored this FTN dropback rate; the
    # 2026-09-30 catalog trim dropped its registry entry (restored
    # 2026-10-14 at Kaedon's request for the QB Charting preset).
    spec = am.LEADERBOARD_METRICS["out_of_pocket_rate"]
    assert spec["label"] == "Out of Pocket Rate"
    assert spec["category"] == "Passing"
    assert spec["positions"] == ["QB"]
    assert spec["pct"] is True
    assert spec.get("lower_better") is not True
    # Dropback denominator: same minimum volume as its builder-block
    # siblings play_action_rate / blitz_rate_faced.
    assert spec["min_vol"] == am.LEADERBOARD_METRICS["play_action_rate"]["min_vol"]
    assert "FTN" in spec["desc"]
    assert "not PFF" in spec["desc"]
    assert "\u2014" not in spec["desc"]  # no em dashes in UI copy
    assert "out_of_pocket_rate" not in am.PRO_METRICS
    assert "out_of_pocket_rate" not in am.PREMIUM_METRICS
    # Weekly + career + DDL surfaces the builders already fed.
    assert "out_of_pocket_rate" in am.WEEKLY_ADV_METRIC_COLS
    assert am._ADV_WEEKLY_WEIGHTED_METRICS["out_of_pocket_rate"] == "w_dropbacks"
    assert "ADD COLUMN IF NOT EXISTS out_of_pocket_rate" in inspect.getsource(am)
    assert "out_of_pocket_rate" in inspect.getsource(am.get_player_career_metrics)


def test_out_of_pocket_rate_js_and_page_wiring():
    from dashboard_services.pages.advanced_metrics_page import (
        ADVANCED_METRIC_PRESETS,
    )
    app_js = (REPO_ROOT / "static" / "app.js").read_text(encoding="utf-8")
    modal_js = (REPO_ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    page_py = (REPO_ROOT / "dashboard_services" / "pages"
               / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert ADVANCED_METRIC_PRESETS["qb_charting"]["metrics"][-1] == "out_of_pocket_rate"
    assert "out_of_pocket_rate:'Out of Pocket %'" in app_js
    assert "'out_of_pocket_rate': 35" in app_js
    assert "out_of_pocket_rate: 'w_dropbacks'" in app_js
    assert "out_of_pocket_rate:" in modal_js
    assert "out_of_pocket_rate: 'w_dropbacks'" in page_py
    assert "'out_of_pocket_rate': ['Out of pocket', 'In pocket']" in page_py
