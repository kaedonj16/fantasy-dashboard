"""Unit tests for the start/sit role-trend factor (V2, half strength).

The factor is the snap-share trend over a player's last 4 PLAYED weeks
(mean share of the last 2 minus mean share of the first 2), applied at
half strength: clamp(1 + 0.25 * trend, 0.96, 1.04), RB/WR/TE only.
Backtested 2026-11-13 over Weeks 6-9 (fully-ordered boards 6/16
baseline -> 9/16); the full-strength variant (0.5 / +/-8%) was worse.
"""
import json

import pytest

from utils.start_sit_score import compute_start_score, snap_share_trend


# ---------------------------------------------------------------------------
# Trend derivation
# ---------------------------------------------------------------------------

def test_trend_is_last2_minus_first2_of_last4():
    # Only the final four shares count; the older 0.90 is outside the window.
    trend = snap_share_trend([0.90, 0.40, 0.50, 0.60, 0.70])
    assert trend == pytest.approx(0.65 - 0.45)


def test_trend_flat_shares_is_zero():
    assert snap_share_trend([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_trend_needs_four_played_weeks():
    assert snap_share_trend([0.5, 0.6, 0.7]) is None
    assert snap_share_trend([]) is None
    assert snap_share_trend(None) is None
    # A week with no computable share does not fill the window.
    assert snap_share_trend([0.5, None, 0.6, 0.7]) is None


# ---------------------------------------------------------------------------
# Factor application
# ---------------------------------------------------------------------------

def test_rising_share_capped_at_1_04():
    # trend +0.40 -> 1 + 0.25 * 0.40 = 1.10 raw, capped at the half-strength max.
    _, factors, _ = compute_start_score(10.0, position="RB", snap_trend=0.40)
    assert factors["role_trend"] == 1.04


def test_falling_share_floored_at_0_96():
    _, factors, _ = compute_start_score(10.0, position="WR", snap_trend=-0.40)
    assert factors["role_trend"] == 0.96


def test_half_strength_scaling():
    # trend +0.08 -> 1 + 0.25 * 0.08 = 1.02 (full strength would give 1.04).
    score, factors, _ = compute_start_score(10.0, position="TE", snap_trend=0.08)
    assert factors["role_trend"] == 1.02
    assert abs(score - 10.2) < 1e-9


def test_qb_excluded():
    _, factors, _ = compute_start_score(20.0, position="QB", snap_trend=0.40)
    assert factors["role_trend"] == 1.0


def test_neutral_when_input_absent():
    base_score, base_factors, _ = compute_start_score(10.0, position="RB")
    none_score, none_factors, _ = compute_start_score(
        10.0, position="RB", snap_trend=None)
    assert base_factors["role_trend"] == 1.0
    assert none_factors["role_trend"] == 1.0
    assert none_score == base_score


def test_factor_present_in_neutral_factors_on_bye():
    _, factors, demotion = compute_start_score(10.0, on_bye=True, position="RB")
    assert demotion == "bye"
    assert factors["role_trend"] == 1.0


def test_end_to_end_from_shares():
    trend = snap_share_trend([0.50, 0.50, 0.60, 0.60])
    assert trend == pytest.approx(0.10)
    _, factors, _ = compute_start_score(10.0, position="RB", snap_trend=trend)
    assert factors["role_trend"] == 1.025


# ---------------------------------------------------------------------------
# Bundle derivation: snap shares ride along with the form/usage inputs
# ---------------------------------------------------------------------------

def _write_stat_file(tmp_path, season, week, rows):
    p = tmp_path / f"sleeper_stats_s{season}_w{week}.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_derive_start_sit_inputs_snap_trend(tmp_path, monkeypatch):
    """The derivation computes snap_trend from off_snp / tm_off_snp over
    played weeks (gp >= 1 or a snap count), including a played week with
    no points line; fewer than 4 share weeks stays None."""
    from data_building import start_score_bundle as ssb

    season = 2026
    scoring = dict(ssb.VARIANT_SCORING["ppr"])
    rb, short = "201", "202"

    def line(pts, off, tm=100):
        row = {"position": "RB", "gp": 1, "off_snp": off, "tm_off_snp": tm,
               "rush_att": 10, "rec_tgt": 2, "rush_yd": 50,
               "rec": 1, "rec_yd": 5}
        if pts is not None:
            row["pts_ppr"] = pts
        return row

    rows = {
        # RB shares: .40, .50, .60 (no points line — still a played week
        # for the trend), .70, .80 -> last 4 trend = .75 - .55 = +0.20.
        1: {rb: line(10.0, 40), short: line(9.0, 50)},
        2: {rb: line(12.0, 50), short: line(9.0, 50)},
        3: {rb: line(None, 60), short: line(9.0, 50)},
        4: {rb: line(14.0, 70)},
        5: {rb: line(16.0, 80)},
    }
    for w, payload in rows.items():
        _write_stat_file(tmp_path, season, w, payload)
    monkeypatch.setattr(ssb, "_stat_files",
                        lambda s: [str(tmp_path / f"sleeper_stats_s{s}_w{w}.json")
                                   for w in (1, 2, 3, 4, 5)])
    monkeypatch.setattr("utils.utils.load_players_index",
                        lambda: {rb: {"pos": "RB", "team": "DET"},
                                 short: {"pos": "RB", "team": "DET"}})
    ssb._DERIVE_CACHE.clear()

    out = ssb.derive_start_sit_inputs(season, scoring)
    assert out[rb]["snap_trend"] == pytest.approx(0.20)
    # Only 3 played weeks with shares: no trend is claimed.
    assert out[short]["snap_trend"] is None
    ssb._DERIVE_CACHE.clear()
