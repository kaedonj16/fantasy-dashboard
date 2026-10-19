"""Did-not-play rows must not count as games played in the player modal.

Regression (reported 2026-10-19): Roman Wilson was a DNP in the Steelers'
2026 week 6 game (2026-10-18), but his player modal counted that week as
a game he played -- the hero showed an extra "G" with PPG diluted by a
phantom 0.0, and the Stats tab game log rendered the week as a played
0.0 game instead of DNP.

Root cause, verified against the real Sleeper feed: Sleeper emits a
weekly stat row for every ACTIVE player, played or not. Wilson's real
week 6 row is bookkeeping only --

    {"gms_active": 1.0, "pos_rank_half_ppr": 999.0,
     "pos_rank_ppr": 999.0, "pos_rank_std": 999.0}

-- no ``gp``, no snaps, no stats. Both modal loaders treated a present
row as an appearance: ``_load_season_weekly_points`` appended a scored
0.0 for it, and ``_week_stat_index_rows`` kept it because ``gms_active``
and the 999 rank sentinels are nonzero, so ``_sleeper_stats_by_week``
zero-filled its display stats and the client counted it a real game.

The fixtures below use Wilson's REAL 2026 rows (fetched from
https://api.sleeper.app/v1/stats/nfl/regular/2026/{week} on 2026-10-19,
trimmed only of unrelated stat keys): week 5 is a genuine 0-point game
(``gp`` 1, 20 offensive snaps, one target) and must keep counting;
week 6 is the DNP and must not.
"""
import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod
from utils.fantasy_scoring import stat_row_is_appearance
from utils.season_qualification import QualificationPolicy

WILSON_W5_ZERO_GAME = {  # played: gp + snaps + a target, scored 0.0
    "gms_active": 1.0, "gp": 1.0, "off_snp": 20.0,
    "pos_rank_half_ppr": 999.0, "pos_rank_ppr": 999.0, "pos_rank_std": 999.0,
    "rec_tgt": 1.0, "tm_def_snp": 71.0, "tm_off_snp": 60.0, "tm_st_snp": 30.0,
}
WILSON_W6_DNP = {  # active, zero snaps: bookkeeping keys only
    "gms_active": 1.0,
    "pos_rank_half_ppr": 999.0, "pos_rank_ppr": 999.0, "pos_rank_std": 999.0,
}
BACKUP_QB_DNP = {  # same shape plus team snap totals (Mason Rudolph, W6)
    "gms_active": 1.0,
    "pos_rank_half_ppr": 999.0, "pos_rank_ppr": 999.0, "pos_rank_std": 999.0,
    "tm_def_snp": 74.0, "tm_off_snp": 79.0, "tm_st_snp": 36.0,
}
SNAPS_NO_GP = {  # played (special teams / defense) but Sleeper set no gp
    "gms_active": 1.0, "def_snp": 1.0, "st_snp": 20.0,
    "pos_rank_half_ppr": 999.0, "pos_rank_ppr": 999.0, "pos_rank_std": 999.0,
    "tm_def_snp": 58.0, "tm_off_snp": 70.0, "tm_st_snp": 20.0,
}


# ── The predicate ────────────────────────────────────────────────────────

def test_predicate_dnp_rows_are_not_appearances():
    assert stat_row_is_appearance(WILSON_W6_DNP) is False
    assert stat_row_is_appearance(BACKUP_QB_DNP) is False
    assert stat_row_is_appearance({}) is False
    assert stat_row_is_appearance(None) is False
    assert stat_row_is_appearance("not-a-dict") is False


def test_predicate_played_rows_are_appearances():
    # Genuine 0-point game: gp/snaps prove he played.
    assert stat_row_is_appearance(WILSON_W5_ZERO_GAME) is True
    # Snaps without gp (real feed shape for ST/defensive players).
    assert stat_row_is_appearance(SNAPS_NO_GP) is True
    # Any recorded stat is evidence, incl. a lone negative rushing line.
    assert stat_row_is_appearance({"rush_att": 1.0, "rush_yd": -1.0}) is True
    assert stat_row_is_appearance({"gp": 1.0}) is True


# ── Hero weekly points (_load_season_weekly_points) ─────────────────────

def _patch_weekly_points(monkeypatch, tmp_path, weeks_stats):
    stats_dir = tmp_path / "sleeper_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for week, stats in weeks_stats.items():
        (stats_dir / f"sleeper_stats_s2026_w{week}.json").write_text(
            json.dumps(stats)
        )
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        lambda season: QualificationPolicy(int(season), (1, 5, 6), 4),
    )
    monkeypatch.setattr(appmod, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "_ensure_sleeper_week_files", lambda s: None)
    monkeypatch.setattr(appmod, "load_players_index", lambda: {})
    monkeypatch.setattr(appmod, "_WEEKLY_PTS_CACHE", {})


def test_weekly_points_skips_dnp_counts_zero_game(monkeypatch, tmp_path):
    _patch_weekly_points(monkeypatch, tmp_path, {
        1: {"11630": {"gms_active": 1.0, "gp": 1.0, "off_snp": 37.0,
                      "rec": 3.0, "rec_yd": 26.0}},
        5: {"11630": WILSON_W5_ZERO_GAME},
        6: {"11630": WILSON_W6_DNP},
    })
    out = appmod._load_season_weekly_points(2026, {"rec": 1.0})
    # Week 1 (5.6) and the genuine week-5 zero game count; the DNP does not.
    assert out["11630"] == [5.6, 0.0]
    from utils.fantasy_scoring import completed_points_summary
    summary = completed_points_summary(out["11630"])
    assert summary["games"] == 2
    assert summary["ppg"] == pytest.approx(2.8)


# ── Game-log stat index (_week_stat_index_rows / _sleeper_stats_by_week) ─

def test_stats_by_week_omits_dnp_keeps_zero_game(monkeypatch, tmp_path):
    stats_dir = tmp_path / "sleeper_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        1: {"11630": {"gms_active": 1.0, "gp": 1.0, "off_snp": 37.0,
                      "rec": 3.0, "rec_tgt": 6.0, "rec_yd": 26.0}},
        5: {"11630": WILSON_W5_ZERO_GAME},
        6: {"11630": WILSON_W6_DNP},
    }
    for week, stats in rows.items():
        (stats_dir / f"sleeper_stats_s2026_w{week}.json").write_text(
            json.dumps(stats)
        )
    monkeypatch.setattr(appmod, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(appmod, "_WEEK_STAT_INDEX", {})
    monkeypatch.setattr(appmod, "_ensure_sleeper_week_files", lambda s: None)

    index_rows = appmod._week_stat_index_rows(2026, 6)
    assert "11630" not in index_rows  # DNP: absent, not a zero-filled game

    out = appmod._sleeper_stats_by_week("11630", 2026)
    # Week 6 absent -> the game-log route renders it via its DNP branch
    # (fantasy_pts None / stats None) instead of a played 0.0 row.
    assert set(out) == {1, 5}
    # The genuine zero game stays, zero-filled for display.
    assert out[5]["rec"] == 0
    assert out[5]["rec_tgt"] == 1.0  # real target line preserved
    assert out[1]["rec_yd"] == 26.0
