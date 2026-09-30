"""Regression tests for the FTN blitz + nflverse scramble fixes.

Audit findings (2026-09-30), verified against the real upstream files
(ftn_charting_2024/2025.parquet, play_by_play_2025.parquet):

3. FTN charting has NO is_blitz column and nfl_data_py does no renaming;
   blitzes arrive as counts in n_blitzers / n_pass_rushers. The builders
   read r.get("is_blitz"), which was always None, so blitz_rate_faced was
   0.0 for every QB and epa_vs_blitz never populated. The fix derives the
   flag from n_blitzers > 0 and warns + skips blitz outputs when the
   column is absent, instead of silently storing 0.0.

4. qb_scramble is alive in play-by-play (1,160 plays in 2025 REG), but
   passer_player_id is NULL on 100% of scramble plays (coded run/no_play),
   so the passer-grouped frame excluded every scramble before summing.
   rusher_player_id IS populated on the qb_dropback==1 subset. The fix
   counts scrambles from the full REG frame by rusher and keeps the
   dropback denominator from the passer frame.
"""

import sys
import types

import pytest

pytest.importorskip("pandas")

import pandas as pd

import data_building.external_data.nflverse_metrics as nvm

GSIS = "00-0036212"
SLEEPER = "6000"


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
    # stub them out — they are not under test here.
    monkeypatch.setattr(nvm, "build_pfr_contact_yards_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_catchable_weekly", lambda season: {})
    monkeypatch.setattr(nvm, "build_pfr_broken_tackles_weekly", lambda season: {})


def _install_nfl(monkeypatch, ftn=None, pbp=None):
    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl(ftn=ftn, pbp=pbp))


def _dropback_plays(n=10, blitz_eps=None, week=1):
    """Passer dropback plays for the fixture QB. blitz_eps, when given, maps
    play index -> epa for plays that were blitzed (n_blitzers=2)."""
    plays = []
    for i in range(n):
        plays.append({
            "game_id": "g1", "play_id": 100 + i, "week": week,
            "season_type": "REG", "play_type": "pass",
            "passer_player_id": GSIS, "rusher_player_id": None,
            "receiver_player_id": None,
            "pass_attempt": 1, "rush_attempt": 0, "qb_dropback": 1,
            "complete_pass": 1 if i % 2 == 0 else 0,
            "epa": 0.1, "qb_epa": 0.1, "cpoe": 1.0, "success": 1,
            "sack": 0, "qb_scramble": 0, "qb_hit": 0,
            "rushing_yards": 0, "air_yards": 8, "yards_gained": 8,
            "yards_after_catch": 2, "passing_yards": 8,
            "pass_touchdown": 0, "interception": 0,
        })
    return plays


def _ftn_frame(plays, blitz_eps, with_blitzers=True):
    rows = []
    for i, p in enumerate(plays):
        r = {
            "nflverse_game_id": p["game_id"], "nflverse_play_id": p["play_id"],
            "week": p["week"],
            "is_catchable_ball": 0, "is_drop": 0, "is_contested_ball": 0,
            "is_throw_away": 0, "is_play_action": 0, "is_qb_out_of_pocket": 0,
            "n_defense_box": 6,
        }
        if with_blitzers:
            r["n_blitzers"] = 2 if i in blitz_eps else 0
            r["n_pass_rushers"] = 5 if i in blitz_eps else 4
        rows.append(r)
    return pd.DataFrame(rows)


def _pbp_frame(plays):
    return pd.DataFrame(plays)


# --------------------------------------------------------------------------- #
# Finding 3 (season): blitz from n_blitzers, a column is_blitz never was
# --------------------------------------------------------------------------- #
def test_ftn_season_blitz_rate_from_n_blitzers(monkeypatch):
    plays = _dropback_plays(10)
    # Plays 0-3 blitzed with EPA 0.5 each; pbp epa must carry the blitz EPA
    # because the season builder reads epa from the merged (pbp) side.
    for i in range(4):
        plays[i]["epa"] = 0.5
    ftn = _ftn_frame(plays, blitz_eps={0, 1, 2, 3})
    assert "is_blitz" not in ftn.columns  # the verified upstream shape
    _install_nfl(monkeypatch, ftn=ftn, pbp=_pbp_frame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    assert out[SLEEPER]["blitz_rate_faced"] == pytest.approx(40.0)
    assert out[SLEEPER]["epa_vs_blitz"] == pytest.approx(0.5)


def test_ftn_season_missing_n_blitzers_skips_loudly(monkeypatch, capsys):
    plays = _dropback_plays(6)
    ftn = _ftn_frame(plays, blitz_eps=set(), with_blitzers=False)
    _install_nfl(monkeypatch, ftn=ftn, pbp=_pbp_frame(plays))

    out = nvm.build_ftn_charting_for_season(2025)
    # Skipped, not silently zeroed.
    assert "blitz_rate_faced" not in out.get(SLEEPER, {})
    assert "epa_vs_blitz" not in out.get(SLEEPER, {})
    assert "WARNING: FTN frame missing n_blitzers" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Finding 4 (season): scrambles counted by rusher, not the passer frame
# --------------------------------------------------------------------------- #
def test_pbp_season_scramble_rate_counts_rusher_rows(monkeypatch):
    plays = _dropback_plays(20)
    plays[0]["sack"] = 1
    plays[1]["sack"] = 1
    for j in range(3):
        plays.append({
            "game_id": "g1", "play_id": 900 + j, "week": 1,
            "season_type": "REG", "play_type": "run",
            "passer_player_id": None, "rusher_player_id": GSIS,
            "receiver_player_id": None,
            "pass_attempt": 0, "rush_attempt": 1, "qb_dropback": 1,
            "complete_pass": 0,
            "epa": 0.3, "qb_epa": None, "cpoe": None, "success": 1,
            "sack": 0, "qb_scramble": 1, "qb_hit": 0,
            "rushing_yards": 8, "air_yards": None, "yards_gained": 8,
            "yards_after_catch": 0, "passing_yards": 0,
            "pass_touchdown": 0, "interception": 0,
        })
    _install_nfl(monkeypatch, pbp=_pbp_frame(plays))

    out = nvm.build_pbp_metrics_for_season(2025)
    # 3 scrambles over 20 dropbacks; pre-fix this was 0.0 because the
    # scramble rows never enter the passer-grouped frame.
    assert out[SLEEPER]["scramble_rate"] == pytest.approx(15.0)
    # Denominator is untouched: sacks still resolve against the same 20.
    assert out[SLEEPER]["sack_rate"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Weekly builders: same two fixes at (player, week) granularity
# --------------------------------------------------------------------------- #
def test_weekly_scramble_rate_counts_rusher_rows(monkeypatch):
    plays = _dropback_plays(10, week=1)
    for j in range(2):
        plays.append({
            "game_id": "g1", "play_id": 950 + j, "week": 1,
            "season_type": "REG", "play_type": "run",
            "passer_player_id": None, "rusher_player_id": GSIS,
            "receiver_player_id": None,
            "pass_attempt": 0, "rush_attempt": 1, "qb_dropback": 1,
            "complete_pass": 0,
            "epa": 0.3, "qb_epa": None, "cpoe": None, "success": 1,
            "sack": 0, "qb_scramble": 1, "qb_hit": 0,
            "rushing_yards": 7, "air_yards": None, "yards_gained": 7,
            "yards_after_catch": 0, "passing_yards": 0,
            "pass_touchdown": 0, "interception": 0,
        })
    _install_nfl(monkeypatch, pbp=_pbp_frame(plays))

    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[(SLEEPER, 1)]["scramble_rate"] == pytest.approx(20.0)


def test_weekly_ftn_blitz_rate_from_n_blitzers(monkeypatch):
    plays = _dropback_plays(10, week=1)
    for i in range(3):
        plays[i]["epa"] = 0.4
    ftn = _ftn_frame(plays, blitz_eps={0, 1, 2})
    _install_nfl(monkeypatch, ftn=ftn, pbp=_pbp_frame(plays))

    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert out[(SLEEPER, 1)]["blitz_rate_faced"] == pytest.approx(30.0)
    assert out[(SLEEPER, 1)]["epa_vs_blitz"] == pytest.approx(0.4)


def test_weekly_ftn_missing_n_blitzers_skips_loudly(monkeypatch, capsys):
    plays = _dropback_plays(6, week=1)
    ftn = _ftn_frame(plays, blitz_eps=set(), with_blitzers=False)
    _install_nfl(monkeypatch, ftn=ftn, pbp=_pbp_frame(plays))

    out = nvm.build_nflverse_weekly_metrics_for_season(2025)
    assert "blitz_rate_faced" not in out.get((SLEEPER, 1), {})
    assert "WARNING: weekly FTN frame missing n_blitzers" in \
        capsys.readouterr().out
