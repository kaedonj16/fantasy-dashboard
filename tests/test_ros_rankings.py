"""Focused tests for the Rest of Season rankings mode.

Covers the ROS compute (utils/ros_rankings.py): the backtested
position-aware formula (projection-only for RB/WR/TE, 40/30/30 with
form shrinkage for QB), the eligibility gates, weeks-out / bye handling
inside each window, per-window positional ranks (the full window by the
factor model, the sub-windows by total), tier thresholds derived from
the numbers each window displays, the payload whitelist, the public
/rest-of-season page bodies, the sitemap registration, and the
rankings.js / players-shell mode wiring (static guards, the established
style for this page).
"""
import os

import pytest

pytest.importorskip("pandas")

from utils.ros_rankings import (
    ROS_MODEL_WEIGHTS,
    apply_qb_calibration,
    assign_ros_ranks,
    availability_rate,
    ceiling_ppg,
    compute_player_ros,
    _ros_gap_tier_thresholds,
    compute_ros_tier_thresholds,
    compute_ros_vor_tier_thresholds,
    consistency_rate,
    divergence_guard_cap,
    first_projected_week,
    form_ppg,
    has_season_ending_note,
    is_in_window_returner,
    is_season_out,
    opportunity_form,
    order_overall,
    passes_role_gate,
    player_model_factors,
    qb_ros_ppg,
    role_share_series,
    role_trend,
    ros_flex_replacement_ppgs,
    ros_model_scores,
    ros_qb_replacement_ppg,
    ros_replacement_map,
    ros_vor_totals,
    window_projection_stats,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ros(position, **kw):
    base = dict(
        played_games=8, season_ppg=15.0, last4_ppg=16.0, last4_games=4,
        touches_per_game=14.0, targets_per_game=6.0, is_team_top=False,
        season_out=False,
        windows={"all": [20.0, 20.0, 20.0], "stretch": [20.0, 20.0],
                 "playoffs": [20.0]},
        sched_rank=None,
    )
    base.update(kw)
    return compute_player_ros(position, **base)


# ── QB blend ────────────────────────────────────────────────────────────────

def test_qb_blend_full_weights():
    # 40% projection + 30% form + 30% season
    assert qb_ros_ppg(20.0, 24.0, 4, 22.0) == pytest.approx(21.8)


def test_qb_blend_thin_form_shrinks_toward_projection():
    # 2 form games: form weight halves to 0.15, projection takes the rest.
    assert qb_ros_ppg(20.0, 24.0, 2, 22.0) == pytest.approx(21.2)
    # No form at all: 70% projection + 30% season.
    assert qb_ros_ppg(20.0, None, 0, 22.0) == pytest.approx(20.6)


def test_qb_ros_uses_blend_rb_ignores_form():
    qb = _ros("QB", is_team_top=True, season_ppg=22.0, last4_ppg=24.0,
              windows={"all": [20.0, 20.0], "stretch": [20.0], "playoffs": [20.0]})
    assert qb["ros_ppg"] == pytest.approx(21.8)
    assert qb["ros_total"] == pytest.approx(43.6)
    rb = _ros("RB", season_ppg=30.0, last4_ppg=30.0)
    assert rb["ros_ppg"] == pytest.approx(20.0)  # projection only


# ── Divergence guard (v2, factor 1.25) ──────────────────────────────────────
# DIVERGENCE_GUARD_V2_2026-11-23.md: cap the PPG input at 1.25x the
# better of season / last-4 PPG when the feed sits above BOTH by more
# than 1.25x, the player has 7+ games played, and he appeared in 3+
# of his team's last 4 games. Priced at -0.003 pooled on 2022-2025;
# Kaedon approved the spend 2026-11-23 ("the board needs fixed").

def test_divergence_guard_fires_on_jefferson_shape():
    # Jefferson 2026 through Week 10: feed 19.36, season 13.95,
    # last-4 11.97, 8 played games, 4 of his team's last 4 games.
    # Cap = max(13.95, 11.97) * 1.25 = 17.4375 -> stamped 17.44.
    out = _ros("WR", played_games=8, season_ppg=13.95, last4_ppg=11.97,
               played_last4_team_games=4,
               windows={"all": [19.36] * 7, "stretch": [19.36] * 4,
                        "playoffs": [19.36] * 3})
    assert out["ros_ppg"] == pytest.approx(17.44)
    assert out["ros_total"] == pytest.approx(17.4375 * 7, abs=0.1)


def test_divergence_guard_cap_value_exact():
    assert divergence_guard_cap(
        19.36, played_games=8, season_ppg=13.95, last4_ppg=11.97,
        played_last4_team_games=4) == pytest.approx(17.4375)
    # The better actual mark is the base: a hot-ER last-4 under the
    # factor still caps off the last-4 line when both prongs clear.
    assert divergence_guard_cap(
        30.0, played_games=9, season_ppg=10.0, last4_ppg=20.0,
        played_last4_team_games=3) == pytest.approx(25.0)


def test_divergence_guard_spares_hot_last4_jeudy_shape():
    # Jeudy-shaped breakout: cold season line, last-4 already turned.
    # The recency prong must NOT fire even though the feed clears the
    # season line by more than 1.25x.
    out = _ros("WR", played_games=8, season_ppg=8.56, last4_ppg=17.0,
               played_last4_team_games=4,
               windows={"all": [11.41] * 7, "stretch": [11.41] * 4,
                        "playoffs": [11.41] * 3})
    assert out["ros_ppg"] == pytest.approx(11.41)


def test_divergence_guard_needs_both_prongs():
    # Feed above season x1.25 but NOT above last-4 x1.25: no fire.
    assert divergence_guard_cap(
        20.0, played_games=8, season_ppg=15.0, last4_ppg=16.5,
        played_last4_team_games=4) == pytest.approx(20.0)
    # Feed above last-4 x1.25 but not season x1.25: no fire.
    assert divergence_guard_cap(
        20.0, played_games=8, season_ppg=16.5, last4_ppg=15.0,
        played_last4_team_games=4) == pytest.approx(20.0)


def test_divergence_guard_games_minimums():
    # Under 7 played games: no fire, however extreme the divergence.
    assert divergence_guard_cap(
        19.36, played_games=6, season_ppg=13.95, last4_ppg=11.97,
        played_last4_team_games=4) == pytest.approx(19.36)
    # Fewer than 3 of his team's last 4 games: no fire (a returner's
    # thin recent log is not evidence about the feed).
    assert divergence_guard_cap(
        19.36, played_games=9, season_ppg=13.95, last4_ppg=11.97,
        played_last4_team_games=2) == pytest.approx(19.36)
    # A caller that does not track team-game appearances: no fire.
    assert divergence_guard_cap(
        19.36, played_games=9, season_ppg=13.95, last4_ppg=11.97,
        played_last4_team_games=None) == pytest.approx(19.36)
    # Missing season or last-4 line: no fire, never a guess.
    assert divergence_guard_cap(
        19.36, played_games=9, season_ppg=None, last4_ppg=11.97,
        played_last4_team_games=4) == pytest.approx(19.36)
    assert divergence_guard_cap(
        19.36, played_games=9, season_ppg=13.95, last4_ppg=None,
        played_last4_team_games=4) == pytest.approx(19.36)


def test_divergence_guard_default_off_without_team_games_input():
    # compute_player_ros callers that pass no played_last4_team_games
    # (every pre-guard caller) keep the unguarded numbers byte-exact.
    out = _ros("WR", played_games=8, season_ppg=13.95, last4_ppg=11.97,
               windows={"all": [19.36] * 7, "stretch": [19.36] * 4,
                        "playoffs": [19.36] * 3})
    assert out["ros_ppg"] == pytest.approx(19.36)


def test_divergence_guard_caps_qb_blended_input():
    # QB: the cap reads the FINAL blended PPG (the backtest's feed_ppg
    # is the production input, blend included), not the raw forward
    # projection. Blend of fwd 34 / form 18 / season 19 at full
    # weights is 24.7; cap = max(19, 18) * 1.25 = 23.75.
    out = _ros("QB", is_team_top=True, played_games=9, season_ppg=19.0,
               last4_ppg=18.0, last4_games=4, played_last4_team_games=4,
               windows={"all": [34.0, 34.0], "stretch": [34.0],
                        "playoffs": [34.0]})
    assert out["ros_ppg"] == pytest.approx(23.75)


# ── Window series math ──────────────────────────────────────────────────────

def test_window_stats_leading_zero_run_is_weeks_out():
    stats = window_projection_stats([0.0, 0.4, 18.0, 20.0])
    assert stats["weeks_out"] == 2
    assert stats["games"] == 2
    assert stats["total"] == pytest.approx(38.0)
    assert stats["fwd_ppg"] == pytest.approx(19.0)


def test_window_stats_absent_weeks():
    # Absent mid-window is skipped, never scored as a zero game.
    assert window_projection_stats([None, 15.0])["games"] == 1
    stats = window_projection_stats([20.0, None, 18.0])
    assert stats["games"] == 2
    assert stats["fwd_ppg"] == pytest.approx(19.0)


def test_window_stats_missing_week_is_not_a_week_out():
    # treat_missing_as_out=False (audit D8): a missing (None) week is
    # unknown, not an absence. It stops the leading zero-run instead of
    # extending it, so the explicit zeros behind it stay in the count
    # as 0-point games rather than being deleted as "weeks out".
    stats = window_projection_stats([None, 0.0, 18.0])
    assert stats["weeks_out"] == 0
    assert stats["games"] == 2  # the explicit 0.0 counts, the None does not
    assert stats["total"] == pytest.approx(18.0)
    # An explicit leading zero still counts as out, up to the missing
    # week that stops the run.
    stats = window_projection_stats([0.0, None, 0.0, 18.0])
    assert stats["weeks_out"] == 1
    assert stats["games"] == 2


def test_window_stats_no_playable_week():
    assert window_projection_stats([]) is None
    assert window_projection_stats([0.0, 0.0]) is None
    assert window_projection_stats([None, None]) is None


def test_weeks_out_applies_inside_each_window_independently():
    # Out for the first playoff week only: stretch is untouched, the
    # playoff window drops that week from its own games count.
    fields = _ros("RB", windows={
        "all": [20.0, 20.0, 0.0, 18.0],
        "stretch": [20.0, 20.0],
        "playoffs": [0.0, 18.0],
    })
    assert fields["ros_stretch_games"] == 2
    assert fields["ros_playoff_games"] == 1
    assert fields["ros_playoff_total"] == pytest.approx(18.0)
    # The full window counts every present week: the zero here is
    # mid-series (not a leading weeks-out run), so it stays in the count.
    assert fields["ros_games"] == 4


# ── Expected games (availability) ───────────────────────────────────────────

def test_availability_rate_shrink_and_clamp():
    # Full attendance over two seasons still shrinks a touch toward the
    # positional mean: (34 + 17*0.9) / (34 + 17).
    assert availability_rate(34, 34, 0.9) == pytest.approx((34 + 15.3) / 51)
    # A half-missed record over the same span hits the 0.70 floor instead
    # of pricing the raw 0.29 share: availability alone never writes off
    # more than 30% of a player's remaining games.
    assert availability_rate(10, 34, 0.9) == pytest.approx(0.70)
    assert availability_rate(2, 34, 0.8) == pytest.approx(0.70)
    # No data -> no rate (caller prices scheduled games unweighted).
    assert availability_rate(0, 0, 0.9) is None
    assert availability_rate(None, 34, 0.9) is None
    assert availability_rate(5, 34, None) is None


def test_expected_games_scale_totals_not_ppg():
    fragile = _ros("RB", availability_rate=0.8)
    # 3 scheduled games x 0.8 = 2.4 expected games; PPG untouched, the
    # stamped games ARE the expected games the total is built on.
    assert fragile["ros_games"] == pytest.approx(2.4)
    assert fragile["ros_total"] == pytest.approx(20.0 * 2.4)
    assert fragile["ros_ppg"] == pytest.approx(20.0)
    # The rate applies inside each window on its own scheduled count.
    assert fragile["ros_stretch_games"] == pytest.approx(1.6)
    assert fragile["ros_playoff_games"] == pytest.approx(0.8)
    durable = _ros("RB", availability_rate=1.0)
    assert durable["ros_games"] == 3 and durable["ros_total"] == pytest.approx(60.0)
    unrated = _ros("RB")
    assert unrated["ros_games"] == 3 and unrated["ros_total"] == pytest.approx(60.0)


# ── Eligibility gates ───────────────────────────────────────────────────────

def test_role_gate_matrix():
    assert passes_role_gate("RB", is_team_top=True, touches_per_game=3.0,
                            targets_per_game=None)
    assert passes_role_gate("RB", is_team_top=False, touches_per_game=12.0,
                            targets_per_game=None)
    assert not passes_role_gate("RB", is_team_top=False, touches_per_game=11.9,
                                targets_per_game=None)
    assert passes_role_gate("WR", is_team_top=False, touches_per_game=None,
                            targets_per_game=5.5)
    assert not passes_role_gate("TE", is_team_top=False, touches_per_game=None,
                                targets_per_game=4.0)
    # A backup QB passes only as his team's #1 or on a starter's
    # attempt volume (>= 25 pass attempts/game): a relief arm's cameo
    # rate does not clear it.
    assert not passes_role_gate("QB", is_team_top=False, touches_per_game=None,
                                targets_per_game=None)
    assert passes_role_gate("QB", is_team_top=True, touches_per_game=None,
                            targets_per_game=None)
    assert passes_role_gate("QB", is_team_top=False, touches_per_game=None,
                            targets_per_game=None, pass_attempts_per_game=25.0)
    assert not passes_role_gate("QB", is_team_top=False, touches_per_game=None,
                                targets_per_game=None, pass_attempts_per_game=24.9)


def test_three_game_floor_and_season_out_exclusion():
    # Floor is 3 played games (2026-11-20: the 5-game floor kept
    # Josh Jacobs, GB's RB1 at 14.6 PPG over 3 games, off the board).
    assert _ros("RB", played_games=2) is None
    assert _ros("RB", played_games=3) is not None
    assert _ros("RB", season_out=True) is None
    assert is_season_out("IR") and is_season_out("PUP")
    assert not is_season_out("Out") and not is_season_out("Questionable")


def test_no_projection_no_fields():
    assert _ros("RB", windows={"all": [0.0, 0.0], "stretch": [], "playoffs": []}) is None


def test_in_window_returner_reads():
    # Puka shape (verified live 2026-11-20): IR, knee MCL sprain, the
    # feed dark for Weeks 11-14 and real projections back from Week 15.
    puka_series = [None, None, None, None, 18.2, 15.5, 16.8]
    assert first_projected_week(puka_series) == 4
    assert is_in_window_returner("IR", "Sprain", puka_series)
    # Explicit zeros are weeks out too: a resumption after them counts.
    assert is_in_window_returner("IR", None, [0.0, 0.0, 15.0, 15.0])
    # A surgery note is the feed's season-ending signal and outranks a
    # resuming series (Davante Adams shape: still projected weekly).
    assert has_season_ending_note("Surgery")
    assert has_season_ending_note("Out for the season")
    assert not has_season_ending_note("Sprain")
    assert not has_season_ending_note(None)
    assert not is_in_window_returner("IR", "Surgery", [None, 15.5, 15.2, 15.9])
    # A series that never resumes is a season-ender the status named.
    assert not is_in_window_returner("IR", None, [None] * 7)
    assert not is_in_window_returner("IR", "Sprain", [0.0, 0.0])
    assert first_projected_week([None, None]) is None
    # Projected from the first window week contradicts the designation:
    # the designation wins (the audit D2 shape).
    assert not is_in_window_returner("IR", None, [15.0, 15.0, 15.0])
    assert first_projected_week([0.4, 12.0]) == 1
    # No marker status: the returner question does not arise.
    assert not is_in_window_returner("Questionable", None, puka_series)
    assert not is_in_window_returner(None, None, puka_series)


def test_ros_coleman_shape_participation_week_counts(tmp_path, monkeypatch):
    """End-to-end (Coleman, DEN 2026): 4 points-line games plus one
    participation game (gp=1, 3 snaps, zero touches, NO pts_ppr line)
    must make 5 played games, so the back is ROS-eligible. His touches
    (60) divide by the 5 played games: exactly 12.0/game, which passes
    the RB role gate on its own — a higher-scoring teammate holds the
    team-top spot, so team-top cannot be what lets him in."""
    import json as _json
    import os as _os

    from data_building import start_score_bundle as ssb
    from utils import ros_rankings as rr

    season, pid, mate = 2026, "13345", "9999"
    cache = tmp_path / "sleeper_stats"
    cache.mkdir()
    # Coleman-shaped lines: W1 participation-only, W2/W7/W8/W9 real lines
    # (14.8 / 8.9 / 20.4 / 18.8 PPR points, 13/11/15/21 touches).
    coleman = {
        1: {"gp": 1, "gms_active": 1, "off_snp": 3},
        2: {"gp": 1, "gms_active": 1, "off_snp": 25, "pts_ppr": 14.8,
            "rush_att": 10, "rush_yd": 39, "rec": 3, "rec_tgt": 3, "rec_yd": 19},
        7: {"gp": 1, "gms_active": 1, "off_snp": 30, "pts_ppr": 8.9,
            "rush_att": 9, "rush_yd": 49, "rec": 2, "rec_tgt": 2, "rec_yd": 10},
        8: {"gp": 1, "gms_active": 1, "off_snp": 41, "pts_ppr": 20.4,
            "rush_att": 11, "rush_yd": 84, "rec": 4, "rec_tgt": 4, "rec_yd": 30},
        9: {"gp": 1, "gms_active": 1, "off_snp": 58, "pts_ppr": 18.8,
            "rush_att": 17, "rush_yd": 88, "rec": 4, "rec_tgt": 4, "rec_yd": 20},
    }
    # Teammate out-scores him on the season (80 > 62.9), taking team-top.
    mate_line = {"gp": 1, "gms_active": 1, "off_snp": 50, "pts_ppr": 20.0,
                 "rush_att": 15, "rush_yd": 100, "rec": 5, "rec_tgt": 5,
                 "rec_yd": 50}
    files = []
    for w in range(1, 10):
        rows = {}
        if w in coleman:
            rows[pid] = dict(coleman[w], position="RB")
        if w in (2, 7, 8, 9):
            rows[mate] = dict(mate_line, position="RB")
        p = cache / f"sleeper_stats_s{season}_w{w}.json"
        p.write_text(_json.dumps(rows))
        files.append(str(p))
    index = {pid: {"pos": "RB", "team": "DEN"},
             mate: {"pos": "RB", "team": "DEN"}}
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)
    monkeypatch.setattr("utils.utils.load_week_schedule", lambda s, w: [])
    monkeypatch.setattr(ssb, "_stat_files", lambda s: files)
    ssb._DERIVE_CACHE.clear()
    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.data_cache._read_week_projection_file",
        lambda s, w: {pid: {}, mate: {}})
    monkeypatch.setattr(
        "utils.projections.weekly_projection_points",
        lambda week_map, p, scoring, pos: 15.0 if str(p) in week_map else None)
    monkeypatch.setattr(rr, "live_injury_status_map", lambda: {})

    payload = rr._build_ros_map(season, 11, "November 20, 2026")
    ssb._DERIVE_CACHE.clear()
    assert payload is not None
    # The shared derivation now counts the participation week: 5 games.
    from data_building.start_score_bundle import derive_start_sit_inputs
    derived = derive_start_sit_inputs(season, {"rec": 1.0})
    ssb._DERIVE_CACHE.clear()
    assert derived[pid]["played_games"] == 5
    # Eligible, with fields stamped from the projection series.
    fields = payload["players"][pid]
    assert fields["ros_ppg"] == pytest.approx(15.0)
    assert fields["ros_pos_rank"] >= 1


# ── Audit fixes (2026-11-20): end-to-end shapes ────────────────────────────

def _fixture_build(tmp_path, monkeypatch, *, index, rows_by_year,
                   live_status=None, proj=15.0):
    """Drive _build_ros_map on shaped stat files (the Coleman test's
    pattern): rows_by_year = {year: {week: {pid: statrow}}}."""
    import json as _json

    from data_building import start_score_bundle as ssb
    from utils import ros_rankings as rr

    season = 2026
    cache = tmp_path / "sleeper_stats"
    cache.mkdir(exist_ok=True)
    files = []
    for yr, weeks in rows_by_year.items():
        for w, rows in weeks.items():
            p = cache / f"sleeper_stats_s{yr}_w{w}.json"
            p.write_text(_json.dumps(rows))
            if yr == season:
                files.append(str(p))
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)
    monkeypatch.setattr("utils.utils.load_week_schedule", lambda s, w: [])
    monkeypatch.setattr(ssb, "_stat_files", lambda s: files)
    ssb._DERIVE_CACHE.clear()
    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.data_cache._read_week_projection_file",
        lambda s, w: {pid: {} for pid in index})
    monkeypatch.setattr(
        "utils.projections.weekly_projection_points",
        lambda week_map, p, scoring, pos: proj if str(p) in week_map else None)
    monkeypatch.setattr(rr, "live_injury_status_map",
                        lambda: dict(live_status or {}))
    payload = rr._build_ros_map(season, 11, "November 20, 2026")
    ssb._DERIVE_CACHE.clear()
    assert payload is not None
    return payload


def _rb_line(rush_yd, att):
    # PPR points = rush_yd / 10 exactly; touches = att (no receptions).
    return {"gp": 1, "gms_active": 1, "off_snp": 50, "rush_att": att,
            "rush_yd": rush_yd, "rec": 0, "rec_tgt": 0, "rec_yd": 0,
            "position": "RB"}


def _qb_line(att, pass_yd, pass_td):
    # PPR points = pass_yd * 0.04 + pass_td * 4.
    return {"gp": 1, "gms_active": 1, "off_snp": 60, "pass_att": att,
            "pass_yd": pass_yd, "pass_td": pass_td, "position": "QB"}


def test_team_top_is_by_points_per_game_not_totals(tmp_path, monkeypatch):
    # Audit D1/D7: the healthy veteran owns the season total (10 games
    # x 9 pts = 90) but the takeover is the better per-game player
    # (4 x 20 = 80). Totals made the veteran team-top and gated the
    # takeover out (his 10 touches/game is under the RB bar); per-game
    # team-top admits the takeover and leaves the veteran to the
    # touches gate, which his 8/game fails.
    weeks = {}
    for w in range(1, 11):
        rows = {"vet": _rb_line(90, 8)}
        if w >= 7:
            rows["take"] = _rb_line(200, 10)
        weeks[w] = rows
    index = {"vet": {"pos": "RB", "team": "DEN"},
             "take": {"pos": "RB", "team": "DEN"}}
    payload = _fixture_build(tmp_path, monkeypatch, index=index,
                             rows_by_year={2026: weeks})
    assert "take" in payload["players"]
    assert "vet" not in payload["players"]


def test_qb_attempts_fallback_admits_takeover(tmp_path, monkeypatch):
    # McCarthy shape (audit D1): the incumbent leads NYG per game
    # (20 PPG) and is team-top; the takeover (12 PPG) is not, but his
    # 28 pass attempts/game clear the QB volume fallback. A true
    # backup (10 att/game, not his team's top) stays out.
    weeks = {}
    for w in range(1, 11):
        rows = {}
        if w <= 6:
            rows["inc"] = _qb_line(32, 300, 2)       # 20.0 pts
        if w >= 6:
            rows["take"] = _qb_line(28, 200, 1)      # 12.0 pts
        if w <= 5:
            rows["start"] = _qb_line(35, 300, 2)     # KC starter, 20.0
            rows["back"] = _qb_line(10, 100, 0)      # KC backup, 4.0
        weeks[w] = rows
    index = {"inc": {"pos": "QB", "team": "NYG"},
             "take": {"pos": "QB", "team": "NYG"},
             "start": {"pos": "QB", "team": "KC"},
             "back": {"pos": "QB", "team": "KC"}}
    payload = _fixture_build(tmp_path, monkeypatch, index=index,
                             rows_by_year={2026: weeks})
    assert "inc" in payload["players"]
    assert "take" in payload["players"]
    assert "start" in payload["players"]
    assert "back" not in payload["players"]


def test_three_game_floor_admits_jacobs_shape(tmp_path, monkeypatch):
    # Jacobs shape: GB's RB1 with exactly 3 played games at ~14.7 PPG
    # and 15 touches/game is in; a 2-game flash (20 PPG, 20 touches)
    # is still under the floor and out.
    weeks = {
        1: {"jj": _rb_line(150, 15), "thin": _rb_line(200, 20)},
        2: {"jj": _rb_line(140, 15), "thin": _rb_line(200, 20)},
        3: {"jj": _rb_line(150, 15)},
    }
    index = {"jj": {"pos": "RB", "team": "GB"},
             "thin": {"pos": "RB", "team": "GB"}}
    payload = _fixture_build(tmp_path, monkeypatch, index=index,
                             rows_by_year={2026: weeks})
    assert "jj" in payload["players"]
    assert "thin" not in payload["players"]


def test_rookie_availability_denominator_first_season_only(tmp_path, monkeypatch):
    # Audit D4: the rookie appeared in all 9 of his team's 2026 games.
    # His denominator must be those 9 games, not 9 + the team's 17
    # games from 2025, a season he was not in the league: at the old
    # denominator his rate clamped to 0.70 (4.9 expected games); on
    # the first-season denominator it is 1.0 (7.0 expected games).
    rows25 = {w: {"f25": {"gms_active": 1, "position": "RB"},
                  "vet": {"gms_active": 1, "position": "RB"}}
              for w in range(1, 18)}
    weeks26 = {}
    for w in range(1, 10):
        rows = {"vet": {"gms_active": 1, "position": "RB"},
                "rook": {"gms_active": 1, "position": "RB"}}
        if w <= 5:
            rows["rook"] = _rb_line(130, 13)
        if w <= 6:
            rows["vet"] = _rb_line(130, 13)
        weeks26[w] = rows
    index = {"rook": {"pos": "RB", "team": "DEN"},
             "vet": {"pos": "RB", "team": "DEN"},
             "f25": {"pos": "RB", "team": "DEN"}}
    payload = _fixture_build(tmp_path, monkeypatch, index=index,
                             rows_by_year={2025: rows25, 2026: weeks26})
    assert payload["players"]["rook"]["ros_games"] == pytest.approx(7.0)
    assert payload["players"]["vet"]["ros_games"] == pytest.approx(7.0)


def test_live_ir_status_excludes_despite_blank_index(tmp_path, monkeypatch):
    # Audit D2 (Harvey/Love shape): the index carries no designation
    # for "hurt", but the live Sleeper overlay says IR, so he is out.
    # A stale index IR with no live designation still excludes (the
    # fallback), and a healthy roommate is ranked.
    weeks = {w: {"hurt": _rb_line(150, 15), "fine": _rb_line(140, 14),
                 "back": _rb_line(130, 13)} for w in range(1, 6)}
    index = {"hurt": {"pos": "RB", "team": "DEN"},
             "fine": {"pos": "RB", "team": "DEN"},
             "back": {"pos": "RB", "team": "DEN", "injury_status": "IR"}}
    payload = _fixture_build(tmp_path, monkeypatch, index=index,
                             rows_by_year={2026: weeks},
                             live_status={"hurt": "IR"})
    assert "hurt" not in payload["players"]
    assert "back" not in payload["players"]
    assert "fine" in payload["players"]


def _fixture_build_weeks(tmp_path, monkeypatch, *, index, rows_by_year,
                         proj_by_week, live_status=None, live_notes=None):
    """_fixture_build variant with a per-week projection table:
    proj_by_week = {week: {pid: points}}; a pid absent from a week's
    dict is absent from that week's feed (a None week in his series)."""
    import json as _json

    from data_building import start_score_bundle as ssb
    from utils import ros_rankings as rr

    season = 2026
    cache = tmp_path / "sleeper_stats"
    cache.mkdir(exist_ok=True)
    files = []
    for yr, weeks in rows_by_year.items():
        for w, rows in weeks.items():
            p = cache / f"sleeper_stats_s{yr}_w{w}.json"
            p.write_text(_json.dumps(rows))
            if yr == season:
                files.append(str(p))
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)
    monkeypatch.setattr("utils.utils.load_week_schedule", lambda s, w: [])
    monkeypatch.setattr(ssb, "_stat_files", lambda s: files)
    ssb._DERIVE_CACHE.clear()
    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.data_cache._read_week_projection_file",
        lambda s, w: {pid: {"v": v}
                      for pid, v in (proj_by_week.get(w) or {}).items()})
    monkeypatch.setattr(
        "utils.projections.weekly_projection_points",
        lambda week_map, p, scoring, pos: (week_map.get(str(p)) or {}).get("v"))
    monkeypatch.setattr(rr, "live_injury_status_map",
                        lambda: dict(live_status or {}))
    monkeypatch.setattr(rr, "live_injury_notes_map",
                        lambda: dict(live_notes or {}))
    payload = rr._build_ros_map(season, 11, "November 20, 2026")
    ssb._DERIVE_CACHE.clear()
    assert payload is not None
    return payload


def _wr_line(rec, tgt, rec_yd):
    # PPR points = rec + rec_yd / 10, stamped as pts_ppr like the feed.
    return {"gp": 1, "gms_active": 1, "off_snp": 55, "rec": rec,
            "rec_tgt": tgt, "rec_yd": rec_yd,
            "pts_ppr": rec + rec_yd / 10.0, "position": "WR"}


def test_ir_returner_ranked_season_enders_stay_out(tmp_path, monkeypatch):
    # The 2026-11-20 shapes, all WRs with nine strong played games so
    # only the season-out verdict decides who is missing:
    #   puka: IR + sprain, feed dark Weeks 11-14, back Week 15 (in,
    #         priced on the three post-return weeks alone)
    #   surg: IR + surgery note, still projected every week (out)
    #   done: IR, feed dark for the whole window (out)
    #   flat: IR, projected from Week 11 (designation wins, out)
    #   mate: healthy roommate, projected all seven weeks (in)
    strong = _wr_line(8, 10, 110)  # 19.0 PPR, 10 targets a game
    weeks = {w: {"puka": strong, "mate": _wr_line(6, 8, 60),
                 "surg": strong, "done": strong, "flat": strong}
             for w in range(1, 10)}
    index = {"puka": {"pos": "WR", "team": "LAR"},
             "mate": {"pos": "WR", "team": "LAR"},
             "surg": {"pos": "WR", "team": "KC"},
             "done": {"pos": "WR", "team": "KC"},
             "flat": {"pos": "WR", "team": "KC"}}
    proj_by_week = {}
    for w in range(11, 18):
        proj_by_week[w] = {"mate": 12.0, "surg": 15.0, "flat": 15.0}
    for w, pts in ((15, 18.2), (16, 15.5), (17, 16.8)):
        proj_by_week[w]["puka"] = pts
    payload = _fixture_build_weeks(
        tmp_path, monkeypatch, index=index, rows_by_year={2026: weeks},
        proj_by_week=proj_by_week,
        live_status={pid: "IR" for pid in ("puka", "surg", "done", "flat")},
        live_notes={"puka": "Sprain", "surg": "Surgery"})
    players = payload["players"]
    assert "surg" not in players
    assert "done" not in players
    assert "flat" not in players
    assert "mate" in players
    puka = players["puka"]
    # Priced on the post-return weeks only: 3 expected games at the
    # resumed series' per-game rate, against the roommate's full slate.
    assert puka["ros_ppg"] == pytest.approx((18.2 + 15.5 + 16.8) / 3, abs=0.01)
    assert puka["ros_games"] == pytest.approx(3.0)
    assert players["mate"]["ros_games"] == pytest.approx(7.0)
    assert puka["ros_pos_rank"] >= 1


# ── Ranks + tiers ───────────────────────────────────────────────────────────

def test_assign_ros_ranks_per_window():
    pool = {
        "a": {"ros_total": 100.0, "ros_ppg": 20.0,
              "ros_playoff_total": 30.0, "ros_playoff_ppg": 15.0},
        "b": {"ros_total": 120.0, "ros_ppg": 18.0,
              "ros_playoff_total": 40.0, "ros_playoff_ppg": 20.0},
        "c": {"ros_total": 90.0, "ros_ppg": 22.0},
    }
    pos = {"a": "RB", "b": "RB", "c": "RB"}
    assign_ros_ranks(pool, pos)
    assert pool["b"]["ros_pos_rank"] == 1
    assert pool["a"]["ros_pos_rank"] == 2
    assert pool["b"]["ros_playoff_pos_rank"] == 1
    assert "ros_playoff_pos_rank" not in pool["c"]


# ── Factor model (full-window positional order) ─────────────────────────────

def test_model_weights_are_the_locked_set():
    # Tournament winner (2026-11-22, NEW_RANKING_METHOD_2026-11-22.md):
    # score = z(PPG) + w * z(expected games), w = 0.20 QB/RB, 0 WR,
    # 0.05 TE relative to PPG 1.0, normalized below to sum 1.0. Every
    # other factor — sched, consistency, offense, and the earlier
    # stripped form/ceiling/role — is out of the score at every
    # position; they stay computed as display/context signals.
    # Pooled +0.711 across 2022-2025 vs the stripped base +0.703.
    assert ROS_MODEL_WEIGHTS == {
        "QB": {"ppg": 0.833, "games": 0.167},
        "RB": {"ppg": 0.833, "games": 0.167},
        "WR": {"ppg": 1.000},
        "TE": {"ppg": 0.952, "games": 0.048},
    }
    for pos, weights in ROS_MODEL_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), pos
        # The zero-weight factors carry no score weight anywhere.
        for dropped in ("form", "ceiling", "role",
                        "sched", "consistency", "offense"):
            assert dropped not in weights, (pos, dropped)
    # WR is PPG-only; TE PPG outweighs every other factor combined.
    assert set(ROS_MODEL_WEIGHTS["WR"]) == {"ppg"}
    w = ROS_MODEL_WEIGHTS["TE"]
    assert w["ppg"] > sum(v for k, v in w.items() if k != "ppg")


def test_opportunity_form_last_four_played():
    # Mean share over the last 4 played games; None under 2 games.
    assert opportunity_form([10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(35.0)
    assert opportunity_form([25.0, 15.0]) == pytest.approx(20.0)
    assert opportunity_form([25.0]) is None
    assert opportunity_form([]) is None
    assert opportunity_form(None) is None


def test_form_ppg_last_four_played():
    assert form_ppg([10.0, 20.0, 30.0, 40.0, 50.0]) == pytest.approx(35.0)
    # Played games he scored 0.0 in count: they are real games.
    assert form_ppg([0.0, 0.0]) == pytest.approx(0.0)
    assert form_ppg([25.0]) is None
    assert form_ppg([]) is None


def test_ceiling_ppg_best_three():
    assert ceiling_ppg([5.0, 30.0, 10.0, 25.0]) == pytest.approx(
        (30.0 + 25.0 + 10.0) / 3)
    assert ceiling_ppg([10.0, 20.0]) is None


def test_consistency_rate_floors():
    assert consistency_rate([12.0, 8.0, 20.0, 4.0], "RB") == pytest.approx(0.5)
    assert consistency_rate([16.0, 15.9], "QB") == pytest.approx(0.5)
    assert consistency_rate([7.0, 6.9], "TE") == pytest.approx(0.5)
    assert consistency_rate([], "WR") is None


def test_role_trend_windows():
    # 6+ played games: the full 3v3 window.
    assert role_trend([10.0, 10.0, 10.0, 16.0, 16.0, 16.0]) == pytest.approx(6.0)
    # 4-5 played games: a partial 2v2 window (audit D6: the old
    # all-or-nothing 6-game rule neutralised the factor for every
    # 5-game player, exactly the takeover profile).
    assert role_trend([10.0, 10.0, 16.0, 16.0]) == pytest.approx(6.0)
    assert role_trend([10.0, 10.0, 10.0, 16.0, 16.0]) == pytest.approx(6.0)
    assert role_trend([10.0] * 5) == pytest.approx(0.0)
    # Under 4 played games there is no trend to read.
    assert role_trend([10.0, 16.0, 16.0]) is None
    assert role_trend([]) is None


def test_role_share_series_bases():
    usage = {1: (6.0, 2.0, 0.0), 2: (3.0, 4.0, 0.0)}
    tgt = {("DET", 1): 30.0, ("DET", 2): 30.0}
    rush = {("DET", 1): 20.0, ("DET", 2): 20.0}
    # WR/TE: target share of the team's targets.
    assert role_share_series("WR", [1, 2], usage, tgt, rush, "DET") == [
        pytest.approx(20.0), pytest.approx(10.0)]
    # RB: (rush attempts + targets) share of the team's combined volume.
    assert role_share_series("RB", [1, 2], usage, tgt, rush, "DET") == [
        pytest.approx(16.0), pytest.approx(14.0)]
    # QB: raw pass attempts, no share.
    qb_usage = {1: (0.0, 0.0, 33.0), 2: (0.0, 0.0, 41.0)}
    assert role_share_series("QB", [1, 2], qb_usage, tgt, rush, "DET") == [33.0, 41.0]
    # A week with no team denominator reads 0.0 rather than raising.
    assert role_share_series("WR", [3], usage, tgt, rush, "DET") == [0.0]


def test_player_model_factors_assembly():
    fac = player_model_factors(
        "WR", ppg=18.0, games=5.5, sched_rank=4,
        game_points={1: 10.0, 2: 20.0, 3: 30.0, 4: 12.0, 5: 22.0, 6: 26.0},
        role_series=[20.0, 20.0, 20.0, 26.0, 26.0, 26.0],
        offense=27.5)
    assert fac["ppg"] == 18.0 and fac["games"] == 5.5
    assert fac["sched"] == -4.0  # inverted: softer slate, larger number
    # v3: WR form is opportunity share, not points: mean target share
    # over the last 4 played games (20, 26, 26, 26).
    assert fac["form"] == pytest.approx(24.5)
    assert fac["ceiling"] == pytest.approx(26.0)    # best 3: 30, 26, 22
    assert fac["consistency"] == pytest.approx(1.0)
    assert fac["role"] == pytest.approx(6.0)
    assert fac["offense"] == 27.5


def test_player_model_factors_qb_keeps_points_form():
    # v3: QB form stays points-based (last-4 played PPG); the QB role
    # series is raw pass attempts and must NOT be read as form.
    fac = player_model_factors(
        "QB", ppg=21.0, games=6.0, sched_rank=10,
        game_points={1: 12.0, 2: 18.0, 3: 24.0, 4: 20.0, 5: 28.0, 6: 22.0},
        role_series=[30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
        offense=26.0)
    assert fac["form"] == pytest.approx(23.5)  # last 4 pts: 24, 20, 28, 22


def test_player_model_factors_rb_form_is_opportunity_share():
    # RB form = mean rush+target share over the last 4 played games,
    # even when his points went the other way.
    fac = player_model_factors(
        "RB", ppg=14.0, games=6.0, sched_rank=10,
        game_points={1: 30.0, 2: 28.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 3.0},
        role_series=[40.0, 42.0, 44.0, 46.0, 48.0, 50.0],
        offense=24.0)
    assert fac["form"] == pytest.approx(47.0)  # shares 44, 46, 48, 50


def test_model_order_beats_total_order():
    # The case that killed the total order: "volume" owns the bigger
    # total on expected games alone; "rate" is the better per-game
    # player with two fewer games. The factor model ranks rate first.
    factors = {
        "rate": {"ppg": 20.0, "ceiling": 30.0, "games": 5.0, "sched": -10.0,
                 "form": 21.0, "role": 2.0, "consistency": 0.8,
                 "offense": 25.0},
        "volume": {"ppg": 17.0, "ceiling": 24.0, "games": 7.0,
                   "sched": -12.0, "form": 16.0, "role": 0.0,
                   "consistency": 0.7, "offense": 24.0},
        "third": {"ppg": 12.0, "ceiling": 18.0, "games": 6.0,
                  "sched": -20.0, "form": 11.0, "role": -2.0,
                  "consistency": 0.5, "offense": 20.0},
    }
    scores = ros_model_scores(factors, "RB")
    assert scores["rate"] > scores["volume"] > scores["third"]


def test_model_missing_factor_is_neutral():
    # Probed on games, a factor the tournament-winner score reads:
    # a missing value fills with the pool mean (z = 0), so the player
    # with no value sits exactly between the high and low carriers.
    base = {"ppg": 15.0, "ceiling": 22.0, "sched": -15.0,
            "form": 15.0, "role": 0.0, "consistency": 0.6,
            "offense": 22.0}
    factors = {
        "hot": {**base, "games": 7.0},
        "none": {**base, "games": None},
        "cold": {**base, "games": 5.0},
    }
    scores = ros_model_scores(factors, "RB")
    assert scores["hot"] > scores["none"] > scores["cold"]


def test_model_ignores_stripped_factors():
    # Stripped base: form, ceiling and role are computed and passed in
    # but carry no score weight, so swinging them changes nothing.
    base = {"ppg": 15.0, "games": 6.0, "sched": -15.0,
            "consistency": 0.6, "offense": 22.0}
    factors = {
        "hot": {**base, "ceiling": 30.0, "form": 25.0, "role": 6.0},
        "cold": {**base, "ceiling": 5.0, "form": 5.0, "role": -6.0},
    }
    scores = ros_model_scores(factors, "RB")
    assert scores["hot"] == scores["cold"]


def test_model_zero_weight_factors_contribute_nothing():
    # Tournament winner (2026-11-22): sched, consistency and offense
    # carry zero score weight at every position (the earlier stripped
    # factors form/ceiling/role are covered by
    # test_model_ignores_stripped_factors). Two players identical in
    # the scored factors (ppg, games) but at opposite extremes of
    # every zero-weight factor must score exactly the same.
    base = {"ppg": 15.0, "games": 6.0}
    factors = {
        "soft": {**base, "sched": -1.0, "consistency": 1.0,
                 "offense": 32.0},
        "hard": {**base, "sched": -32.0, "consistency": 0.0,
                 "offense": 14.0},
    }
    for pos in ("QB", "RB", "WR", "TE"):
        scores = ros_model_scores(factors, pos)
        assert scores["soft"] == scores["hard"], pos


def test_build_sched_factor_uses_playoff_weeks_only(tmp_path, monkeypatch):
    """End-to-end: two identical WRs on different teams. AAA owns the
    softer FULL-window slate (soft near weeks, hard playoff weeks);
    BBB's slate is the reverse. The display ros_sched_rank follows
    the full window (AAA softer). Since the 2026-11-22 tournament
    weights, the schedule factor (built from the Weeks 15-17 slate)
    carries zero score weight, so the two identical players score
    exactly the same and the rank falls to the pid tie-break."""
    import json as _json

    from data_building import start_score_bundle as ssb
    from utils import ros_rankings as rr

    season = 2026
    cache = tmp_path / "sleeper_stats"
    cache.mkdir(exist_ok=True)
    line = {"gp": 1, "gms_active": 1, "off_snp": 50, "pts_ppr": 15.0,
            "rec": 5, "rec_tgt": 8, "rec_yd": 100, "position": "WR"}
    files = []
    for w in range(1, 9):
        p = cache / f"sleeper_stats_s{season}_w{w}.json"
        p.write_text(_json.dumps({"wa": dict(line), "wb": dict(line)}))
        files.append(str(p))
    index = {"wa": {"pos": "WR", "team": "AAA"},
             "wb": {"pos": "WR", "team": "BBB"}}
    monkeypatch.setattr("utils.utils.load_players_index", lambda: index)

    def _schedule(s, w):
        if 11 <= w <= 14:
            return [{"home_team": "AAA", "away_team": "NS"},
                    {"home_team": "BBB", "away_team": "NH"}]
        if 15 <= w <= 17:
            return [{"home_team": "AAA", "away_team": "FH"},
                    {"home_team": "BBB", "away_team": "FS"}]
        return []

    monkeypatch.setattr("utils.utils.load_week_schedule", _schedule)
    monkeypatch.setattr(ssb, "_stat_files", lambda s: files)
    ssb._DERIVE_CACHE.clear()
    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.data_cache._read_week_projection_file",
        lambda s, w: {pid: {} for pid in index})
    monkeypatch.setattr(
        "utils.projections.weekly_projection_points",
        lambda week_map, p, scoring, pos: 15.0 if str(p) in week_map else None)
    monkeypatch.setattr(rr, "live_injury_status_map", lambda: {})
    # Schedule Rankings values (higher = softer matchup): NS/FS soft,
    # NH/FH hard, the two home teams mid.
    info = {t: {"rank_value": v} for t, v in
            {"NS": 30.0, "NH": 1.0, "FS": 30.0, "FH": 1.0,
             "AAA": 15.0, "BBB": 15.0}.items()}
    monkeypatch.setattr(
        "app._matchup_rank_table",
        lambda s, pos: ({}, 0, info, False))

    payload = rr._build_ros_map(season, 11, "November 20, 2026")
    ssb._DERIVE_CACHE.clear()
    assert payload is not None
    wa, wb = payload["players"]["wa"], payload["players"]["wb"]
    # Display rank is the full-window slate: AAA's is softer.
    assert wa["ros_sched_rank"] < wb["ros_sched_rank"]
    # Schedule is display-only under the tournament weights: the
    # playoff-slate edge moves nothing, scores tie, pid breaks it.
    assert wb["ros_score"] == wa["ros_score"]
    assert wa["ros_pos_rank"] == 1 and wb["ros_pos_rank"] == 2


def test_assign_ros_ranks_model_order_full_window_only():
    pool = {
        "a": {"ros_total": 100.0, "ros_ppg": 20.0,
              "ros_playoff_total": 30.0, "ros_playoff_ppg": 15.0},
        "b": {"ros_total": 120.0, "ros_ppg": 18.0,
              "ros_playoff_total": 40.0, "ros_playoff_ppg": 20.0},
    }
    pos = {"a": "RB", "b": "RB"}
    assign_ros_ranks(pool, pos, model_scores={"a": 0.9, "b": 0.4})
    # Full window follows the model, not b's bigger total; score stamped.
    assert pool["a"]["ros_pos_rank"] == 1
    assert pool["a"]["ros_score"] == 0.9
    assert pool["b"]["ros_pos_rank"] == 2
    # Sub-windows keep the total-points order, untouched by the model.
    assert pool["b"]["ros_playoff_pos_rank"] == 1
    assert "ros_playoff_score" not in pool["b"]


def _clustered_pool():
    """60 players in six clusters of ten: small steps inside a cluster,
    a real drop-off between clusters (the shape tier breaks read)."""
    pool, pos = {}, {}
    for i in range(60):
        pid = str(i)
        cluster, within = divmod(i, 10)
        total = 200.0 - cluster * 30.0 - within * 1.0
        pool[pid] = {"ros_total": total, "ros_ppg": total / 7,
                     "ros_score": 2.0 - cluster * 0.3 - within * 0.01,
                     "ros_stretch_total": total * 0.6,
                     "ros_stretch_ppg": total / 7,
                     "ros_playoff_total": total * 0.4,
                     "ros_playoff_ppg": total / 7}
        pos[pid] = "RB" if i % 2 == 0 else "WR"
    return pool, pos


def test_tier_thresholds_all_window_uses_scores():
    pool, pos = _clustered_pool()
    out = compute_ros_tier_thresholds(pool, pos)
    # Full-window thresholds live on the score scale (max score 2.0),
    # not the 200-point total scale the sub-windows still use.
    assert out["all"] and out["all"] == sorted(out["all"], reverse=True)
    assert max(out["all"]) <= 2.0
    assert max(out["stretch"]) > 2.0


def test_tier_thresholds_come_from_ros_totals():
    pool, pos = _clustered_pool()
    for fields in pool.values():
        fields.pop("ros_score", None)
    out = compute_ros_tier_thresholds(pool, pos)
    assert set(out) == {"all", "stretch", "playoffs"}
    all_t = out["all"]
    assert all_t and all_t == sorted(all_t, reverse=True)
    # Thresholds live on the ROS-total scale (max total is 200), not on a
    # dynasty value scale (which runs to ~1000).
    assert max(all_t) <= 200.0
    # Stretch totals are smaller, so its top threshold sits below the
    # full-window one: thresholds track the window actually displayed.
    assert max(out["stretch"]) < max(all_t)


# ── Cross-position VOR: replacement convention + All-view tiers ────────────

def test_ros_replacement_definitions():
    # Replaced the old index-convention test: the replacement lines are
    # no longer per-position indices. QB 1QB is the streaming
    # replacement, QB SF keeps the scarcity index, and RB/WR/TE come
    # out of one shared flex-aware allocation (below).
    ladder = [26.0 - 0.5 * i for i in range(20)]
    # No weekly series on this call, so the 1QB line is the best-of
    # fallback: the mean of the top two of the six quarterbacks ranked
    # immediately below the starter line (QB11..QB16 at size 10):
    # mean(21, 20.5) = 20.75.
    assert ros_qb_replacement_ppg(ladder, 10, "1qb") == pytest.approx(20.75)
    # SF: the player at index 2 x 10 x 1.5 = 30 on a deeper ladder.
    deep = [26.0 - 0.5 * i for i in range(40)]
    assert ros_qb_replacement_ppg(deep, 10, "sf") == pytest.approx(11.0)
    assert ros_qb_replacement_ppg([], 10, "1qb") is None
    # The fallback line sits BELOW the worst starter (ladder[9] = 21.5)
    # by construction: it is the waiver tier under the starter line,
    # not a discounted slice of the starters themselves.
    assert ros_qb_replacement_ppg(ladder, 10, "1qb") < ladder[9]


def test_ros_qb_replacement_line_is_streaming_best_available():
    # The 1QB line is DERIVED from the waiver pool, never fitted: for
    # each week it takes the best projection among the quarterbacks
    # below the starter line (QB13+ at size 12) and averages the
    # weekly bests. Hand-computable pool: the below-line series peak
    # at 25 / 30 / 28 across the three weeks (a different streamer
    # each week), so the line is (25 + 30 + 28) / 3 — NOT the tier's
    # mean member, and not any single streamer's average.
    ladder = [26.0 - 0.5 * i for i in range(24)]
    series = [[0.0, 0.0, 0.0] for _ in range(12)]  # starters: unread
    series += [
        [10.0, 30.0, 20.0],   # QB13
        [25.0, 12.0, 18.0],   # QB14
        [15.0, 22.0, 28.0],   # QB15
    ] + [[5.0, 5.0, 5.0] for _ in range(9)]
    line = ros_qb_replacement_ppg(ladder, 12, "1qb",
                                  weekly_series_desc=series)
    assert line == pytest.approx((25.0 + 30.0 + 28.0) / 3)
    # Inflate every starter by 4 PPG: the pool below the line is
    # untouched, so the line is identical. A fitted factor on the
    # starter pool would move with them; a derived line cannot.
    boosted = [v + 4.0 for v in ladder[:12]] + ladder[12:]
    assert ros_qb_replacement_ppg(boosted, 12, "1qb",
                                  weekly_series_desc=series) == \
        pytest.approx(line)
    # The line responds to the tier's weekly PEAKS: raising QB14's
    # first week to 29 lifts that week's best from 25 to 29.
    peaked = [list(s) for s in series]
    peaked[13][0] = 29.0
    assert ros_qb_replacement_ppg(ladder, 12, "1qb",
                                  weekly_series_desc=peaked) == \
        pytest.approx((29.0 + 30.0 + 28.0) / 3)
    # No weekly series at all: the best-of fallback, the mean of the
    # top two below the line (QB13 20.0, QB14 19.5).
    assert ros_qb_replacement_ppg(ladder, 12, "1qb") == \
        pytest.approx(sum(ladder[12:14]) / 2)
    # A below-line quarterback carrying no series makes the weekly
    # read unusable, so the same fallback prices the line.
    broken = [list(s) for s in series]
    broken[15] = None
    assert ros_qb_replacement_ppg(ladder, 12, "1qb",
                                  weekly_series_desc=broken) == \
        pytest.approx(sum(ladder[12:14]) / 2)
    # Pool exhaustion: with nobody below the line the fallback is the
    # worst starter himself; with two below it, the tier is those two.
    assert ros_qb_replacement_ppg(ladder[:12], 12, "1qb") == \
        pytest.approx(ladder[11])
    assert ros_qb_replacement_ppg(ladder[:14], 12, "1qb") == \
        pytest.approx(sum(ladder[12:14]) / 2)


def test_build_ros_map_stamps_aligned_qb_weekly_series(tmp_path, monkeypatch):
    # The builder stamps each QB's calendar-aligned weekly projection
    # series next to his window aggregates: one entry per window week
    # (Weeks 11-17 here), the values the streaming replacement takes
    # its weekly bests from.
    weeks = {w: {"qb1": _qb_line(30, 250, 2), "qb2": _qb_line(28, 225, 2)}
             for w in range(1, 6)}
    index = {"qb1": {"pos": "QB", "team": "BUF"},
             "qb2": {"pos": "QB", "team": "KC"}}
    proj_by_week = {w: {"qb1": 20.0, "qb2": 18.0} for w in range(11, 18)}
    payload = _fixture_build_weeks(
        tmp_path, monkeypatch, index=index, rows_by_year={2026: weeks},
        proj_by_week=proj_by_week)
    qb1 = payload["players"]["qb1"]
    assert qb1["ros_weekly"] == [20.0] * 7
    assert qb1["ros_stretch_weekly"] == [20.0] * 4
    assert qb1["ros_playoff_weekly"] == [20.0] * 3
    assert payload["players"]["qb2"]["ros_weekly"] == [18.0] * 7


def test_ros_flex_replacement_shares_one_pool():
    # Synthetic pool where the shared allocation provably differs from
    # the old independent multiplier lines (RB idx 24 -> 12.8, WR idx 24
    # -> 11.8, TE idx 12 -> 8.4 on these ladders).
    by_pos = {"RB": [20.0 - 0.3 * i for i in range(30)],
              "WR": [19.0 - 0.3 * i for i in range(30)],
              "TE": [12.0 - 0.3 * i for i in range(15)]}
    out = ros_flex_replacement_ppgs(by_pos, 10)
    # Dedicated: RB 20, WR 20, TE 10. The 10 flex spots go to the best
    # combined leftovers: 7 RBs (14.0 down to 12.2) and 3 WRs (13.0,
    # 12.7, 12.4); no TE leftover cracks the flex. Replacement is the
    # next player at each position: RB idx 27, WR idx 23, TE idx 10.
    assert out["RB"] == pytest.approx(20.0 - 0.3 * 27)
    assert out["WR"] == pytest.approx(19.0 - 0.3 * 23)
    assert out["TE"] == pytest.approx(12.0 - 0.3 * 10)
    # The map combines the QB line with the flex lines.
    full = ros_replacement_map({**by_pos, "QB": [26.0 - 0.5 * i for i in range(20)]},
                               10, "1qb")
    # No weekly series on this call: the best-of fallback (top two
    # below the size-10 line, 21.0 and 20.5).
    assert full["QB"] == pytest.approx(20.75)
    assert full["RB"] == pytest.approx(out["RB"])


def _vor_pool():
    pool, pos = {}, {}
    for i in range(12):
        ppg = 26.0 - 0.6 * i
        pool[f"qb{i}"] = {"ros_ppg": ppg, "ros_total": ppg * 8, "ros_games": 8,
                          "ros_stretch_ppg": ppg, "ros_stretch_total": ppg * 5,
                          "ros_stretch_games": 5,
                          "ros_playoff_ppg": ppg, "ros_playoff_total": ppg * 3,
                          "ros_playoff_games": 3}
        pos[f"qb{i}"] = "QB"
    for i in range(30):
        ppg = 24.0 - 0.45 * i
        pool[f"rb{i}"] = {"ros_ppg": ppg, "ros_total": ppg * 8, "ros_games": 8,
                          "ros_stretch_ppg": ppg, "ros_stretch_total": ppg * 5,
                          "ros_stretch_games": 5,
                          "ros_playoff_ppg": ppg, "ros_playoff_total": ppg * 3,
                          "ros_playoff_games": 3}
        pos[f"rb{i}"] = "RB"
    for i in range(30):
        ppg = 22.0 - 0.4 * i
        pool[f"wr{i}"] = {"ros_ppg": ppg, "ros_total": ppg * 8, "ros_games": 8,
                          "ros_stretch_ppg": ppg, "ros_stretch_total": ppg * 5,
                          "ros_stretch_games": 5,
                          "ros_playoff_ppg": ppg, "ros_playoff_total": ppg * 3,
                          "ros_playoff_games": 3}
        pos[f"wr{i}"] = "WR"
    for i in range(12):
        ppg = 16.0 - 0.5 * i
        pool[f"te{i}"] = {"ros_ppg": ppg, "ros_total": ppg * 8, "ros_games": 8,
                          "ros_stretch_ppg": ppg, "ros_stretch_total": ppg * 5,
                          "ros_stretch_games": 5,
                          "ros_playoff_ppg": ppg, "ros_playoff_total": ppg * 3,
                          "ros_playoff_games": 3}
        pos[f"te{i}"] = "TE"
    return pool, pos


def test_ros_vor_totals_qb_discount_1qb_only():
    pool, pos = _vor_pool()
    vor_1qb = ros_vor_totals(pool, pos, 10, "1qb")["all"]
    vor_sf = ros_vor_totals(pool, pos, 10, "sf")["all"]
    # 1QB replacement QB is the streaming line: this pool carries no
    # weekly series and has only two quarterbacks below the size-10
    # starter line (20.0 and 19.4), so the best-of fallback line is
    # their mean, 19.7, and QB1's VOR is (26 - 19.7) x 8 = 50.4.
    assert vor_1qb["qb0"] == pytest.approx(50.4)
    # SF: idx 30 clamps to this pool's QB12 (19.4): a strictly bigger edge.
    assert vor_sf["qb0"] == pytest.approx((26.0 - 19.4) * 8)
    assert vor_sf["qb0"] > vor_1qb["qb0"]
    # RB replacement is format-independent: identical VOR either way.
    # Flex-aware line: 20 dedicated RBs + 6 of the 10 flex spots (the
    # combined leftovers order RB-heavy on this pool) -> RB idx 26.
    assert vor_1qb["rb0"] == vor_sf["rb0"] == pytest.approx((24.0 - 12.3) * 8)
    # The point of the streaming line: in 1QB QB1's edge is priced
    # against the waiver tier, so he no longer out-VORs the top RB on
    # raw totals; in SF he closes most of the gap back.
    assert vor_1qb["qb0"] < vor_1qb["rb0"]


def _tier_of(thresholds, value):
    for i, t in enumerate(thresholds):
        if value >= t:
            return i + 1
    return len(thresholds) + 1


def test_vor_tier_thresholds_shape_and_scale():
    pool, pos = _vor_pool()
    out = compute_ros_vor_tier_thresholds(pool, pos)
    assert set(out) == {"all", "stretch", "playoffs"}
    assert set(out["all"]) == {"1qb", "sf"}
    assert set(out["all"]["1qb"]) == {"8", "10", "12", "14"}
    t = out["all"]["1qb"]["10"]
    assert t and t == sorted(t, reverse=True)
    # Thresholds live on the VOR scale: the pool's max VOR (RB1, 93.6
    # under the flex-aware RB line) bounds the top threshold.
    assert max(t) <= 93.6


def test_all_view_tiers_come_from_vor_not_positional_totals():
    pool, pos = _vor_pool()
    # QB1 owns the pool's biggest raw total (208.0) and IS T1 by the
    # positional-total thresholds the position views use...
    totals_t = compute_ros_tier_thresholds(pool, pos)["all"]
    assert _tier_of(totals_t, pool["qb0"]["ros_total"]) == 1
    # ...but his VOR (50.4) is priced against the streaming line, well
    # behind the actual VOR leader, so the All view's VOR thresholds
    # must NOT stamp him T1, while RB1 (the actual VOR leader at 93.6)
    # is T1. One metric, one sequence.
    vor_t = compute_ros_vor_tier_thresholds(pool, pos)["all"]["1qb"]["10"]
    qb1_vor = ros_vor_totals(pool, pos, 10, "1qb")["all"]["qb0"]
    rb1_vor = ros_vor_totals(pool, pos, 10, "1qb")["all"]["rb0"]
    assert _tier_of(vor_t, rb1_vor) == 1
    assert _tier_of(vor_t, qb1_vor) != 1


def test_ros_tier_breaks_are_the_ros_lists_own_gaps():
    # Kaedon (2026-11-22): tiers are based on the ROS rankings. Three
    # tight clusters over a smooth tail: the breaks land exactly on the
    # cluster boundaries (midpoints 78.5 and 43.5) and nowhere else.
    vals = [100, 99, 98, 97, 60, 59, 58, 57, 30, 29, 28, 27] + list(range(26, 0, -1))
    assert _ros_gap_tier_thresholds(vals) == [78.5, 43.5]
    # The dynasty value algorithm (span cap / local significance on the
    # dynasty scale, rescaled the way the old ROS code fed it) puts the
    # second break at 29.5 instead: the ROS break set must NOT match a
    # dynasty-value-derived break set on the same players.
    from utils.tier_thresholds import FALLBACK_THRESHOLDS, compute_tier_thresholds
    factor = 1000.0 / max(vals)
    remapped = [{"position": "RB", "value": v * factor, "sf_value": v * factor}
                for v in vals]
    dyn = compute_tier_thresholds(remapped, "1qb", 10)
    dyn_back = [] if list(dyn) == list(FALLBACK_THRESHOLDS) else [round(v / factor, 1) for v in dyn]
    assert dyn_back != _ros_gap_tier_thresholds(vals)
    # A perfectly smooth ladder has no drop-offs: no manufactured tiers.
    assert _ros_gap_tier_thresholds([100.0 - 2.0 * i for i in range(40)]) == []


def _clustered_vor_pool():
    """Clustered PPG ladders per position, so every format/size VOR
    list carries real drop-offs to tier from."""
    pool, pos = {}, {}

    def add(prefix, position, ppgs):
        for i, ppg in enumerate(ppgs):
            pool[f"{prefix}{i}"] = {
                "ros_ppg": ppg, "ros_total": ppg * 8, "ros_games": 8,
                "ros_stretch_ppg": ppg, "ros_stretch_total": ppg * 5,
                "ros_stretch_games": 5,
                "ros_playoff_ppg": ppg, "ros_playoff_total": ppg * 3,
                "ros_playoff_games": 3}
            pos[f"{prefix}{i}"] = position

    add("qb", "QB", [30, 29.5, 29, 28.5, 24, 23.5, 23, 22.5,
                     19, 18.5, 18, 17.5, 14, 13.5, 13, 12.5,
                     10, 9.5, 9, 8.5])
    add("rb", "RB", [27, 26.5, 26, 25.5, 21, 20.5, 20, 19.5,
                     16, 15.5, 15, 14.5, 11, 10.5, 10, 9.5,
                     7, 6.5, 6, 5.5, 4, 3.5, 3, 2.5])
    add("wr", "WR", [25, 24.5, 24, 23.5, 20, 19.5, 19, 18.5,
                     15, 14.5, 14, 13.5, 10, 9.5, 9, 8.5,
                     6, 5.5, 5, 4.5, 3, 2.5, 2, 1.5])
    add("te", "TE", [18, 17.5, 17, 16.5, 13, 12.5, 12, 11.5,
                     9, 8.5, 8, 7.5, 5, 4.5, 4, 3.5])
    return pool, pos


def test_ros_tier_breaks_change_with_format_and_size():
    # The All-view thresholds are computed per format x size from that
    # setting's own VOR order: switching 1QB -> SF (QBs reprice against
    # a far worse replacement line) or 10 -> 12 teams re-derives the
    # breaks, so no two settings share a break set on this pool.
    pool, pos = _clustered_vor_pool()
    out = compute_ros_vor_tier_thresholds(pool, pos)["all"]
    sets = {f"{lt}/{sz}": out[lt][sz] for lt in ("1qb", "sf") for sz in ("10", "12")}
    for key, thresholds in sets.items():
        assert thresholds, key
    assert len({tuple(v) for v in sets.values()}) == len(sets)


# ── Payload path ────────────────────────────────────────────────────────────

def test_slim_payload_keeps_ros_fields_and_blocks():
    from dashboard_services.league_players_board import slim_rankings_payload

    payload = {
        "players": [{"id": "1", "name": "X", "position": "RB",
                     "ros_ppg": 20.0, "ros_total": 120.0, "ros_games": 6,
                     "ros_sched_rank": 3, "ros_pos_rank": 2, "ros_score": 1.25,
                     "ros_stretch_total": 80.0, "ros_playoff_total": 40.0,
                     "internal_only": 1},
                    {"id": "2", "name": "Q", "position": "QB",
                     "ros_ppg": 24.0, "ros_total": 144.0, "ros_games": 6,
                     "ros_weekly": [22.0, None, 25.0],
                     "ros_stretch_weekly": [22.0, None],
                     "ros_playoff_weekly": [25.0]}],
        "ros_window": {"start_week": 11, "end_week": 17},
        "tier_thresholds_ros": {"all": [100.0], "stretch": [60.0], "playoffs": [30.0]},
    }
    out = slim_rankings_payload(payload)
    row = out["players"][0]
    assert row["ros_total"] == 120.0 and row["ros_sched_rank"] == 3
    assert row["ros_score"] == 1.25
    assert row["ros_stretch_total"] == 80.0 and row["ros_playoff_total"] == 40.0
    assert "internal_only" not in row
    # The QB weekly series ride the slim payload too: the board's
    # streaming replacement reads them per window.
    qb_row = out["players"][1]
    assert qb_row["ros_weekly"] == [22.0, None, 25.0]
    assert qb_row["ros_stretch_weekly"] == [22.0, None]
    assert qb_row["ros_playoff_weekly"] == [25.0]
    assert out["ros_window"] == {"start_week": 11, "end_week": 17}
    assert out["tier_thresholds_ros"]["playoffs"] == [30.0]


def test_slim_payload_keeps_vor_tier_block():
    from dashboard_services.league_players_board import slim_rankings_payload

    payload = {"players": [],
               "tier_thresholds_ros_vor": {"all": {"1qb": {"10": [50.0, 30.0]}}}}
    out = slim_rankings_payload(payload)
    assert out["tier_thresholds_ros_vor"]["all"]["1qb"]["10"] == [50.0, 30.0]


# ── Public pages ────────────────────────────────────────────────────────────

def _fake_ros_payload():
    players = {
        "rb1": {"ros_ppg": 20.0, "ros_total": 140.0, "ros_games": 7,
                "ros_sched_rank": 4, "ros_pos_rank": 1},
        "rb2": {"ros_ppg": 18.0, "ros_total": 108.0, "ros_games": 6,
                "ros_sched_rank": 9, "ros_pos_rank": 2},
        "te1": {"ros_ppg": 14.0, "ros_total": 98.0, "ros_games": 7,
                "ros_sched_rank": 2, "ros_pos_rank": 1},
        "qb1": {"ros_ppg": 24.0, "ros_total": 168.0, "ros_games": 7,
                "ros_sched_rank": 6, "ros_pos_rank": 1},
    }
    table = [
        {"id": "rb1", "name": "Back One", "team": "DET", "position": "RB"},
        {"id": "rb2", "name": "Back Two", "team": "MIA", "position": "RB"},
        {"id": "te1", "name": "Tight End One", "team": "KC", "position": "TE"},
        {"id": "qb1", "name": "Quarterback One", "team": "BUF", "position": "QB"},
    ]
    return ({"players": players, "window": {"start_week": 11, "end_week": 17},
             "updated": "November 09, 2026"}, table)


def test_public_position_page_orders_by_ros_total_without_scores():
    # Fallback: a payload carrying no factor-model scores keeps the old
    # ROS-total order (the live payload always carries scores).
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position="RB")
    assert body.index("Back One") < body.index("Back Two")
    assert "Quarterback One" not in body
    assert "Updated November 09, 2026" in body or "updated November 09, 2026" in body
    assert "Full PPR" in body


def test_public_position_page_orders_by_model_score():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    # Back Two owns the smaller total but the better model score.
    payload["players"]["rb1"]["ros_score"] = 0.4
    payload["players"]["rb2"]["ros_score"] = 1.1
    body = build_ros_rankings_body(payload, table, position="RB")
    assert body.index("Back Two") < body.index("Back One")


def test_public_overall_page_discounts_qb1_in_1qb():
    # 20 QBs: under the old 1.5x index the replacement QB was QB19
    # (19.7 PPG) and QB1's VOR (44.1) beat the 20-PPG back's (42.0).
    # This payload carries no weekly series, so the derived streaming
    # line uses its best-of fallback, the mean of QB13 and QB14
    # (21.625 PPG); QB1's VOR is ~30.6, and the back orders ahead.
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    players, table = {}, []
    for i in range(20):
        pid = f"qb{i}"
        ppg = 26.0 - 0.35 * i
        players[pid] = {"ros_ppg": ppg, "ros_total": ppg * 7, "ros_games": 7,
                        "ros_sched_rank": 5, "ros_pos_rank": i + 1}
        table.append({"id": pid, "name": f"Quarterback {i + 1}",
                      "team": "BUF", "position": "QB"})
    for j, ppg in enumerate((24.0, 22.0, 20.0, 18.0, 16.0, 14.0)):
        pid = f"rb{j}"
        players[pid] = {"ros_ppg": ppg, "ros_total": ppg * 7, "ros_games": 7,
                        "ros_sched_rank": 5, "ros_pos_rank": j + 1}
        table.append({"id": pid, "name": f"Back {j + 1}",
                      "team": "DET", "position": "RB"})
    payload = {"players": players, "window": {"start_week": 11, "end_week": 17},
               "updated": "November 09, 2026"}
    body = build_ros_rankings_body(payload, table, position=None)
    # "Back 3" is the 20.0-PPG back: ahead of "Quarterback 1" under
    # the streaming replacement (under the old index line QB1's
    # VOR beat his).
    assert body.index("Back 3") < body.index("Quarterback 1")
    assert body.index("Back 1") < body.index("Back 3")


def _floor_payload():
    """Fixture shaped like the old overall board's forward-floor cases.

    Replacement lines are pinned by monkeypatch in the tests below
    (RB 10.0, WR 12.0), so every VOR sign is deterministic:
    wilson (1.1 PPG x 0.9 games) is below replacement AND under the
    games floor; puka (16.0 PPG x 0.8 games) is above replacement but
    under the games floor; negvor (5.0 PPG x 7 games) clears the games
    floor but is below replacement; the rest clear both.
    """
    players = {
        "wilson": {"ros_ppg": 1.1, "ros_total": 1.0, "ros_games": 0.9,
                   "ros_sched_rank": 9, "ros_pos_rank": 3},
        "keeper": {"ros_ppg": 15.0, "ros_total": 105.0, "ros_games": 7,
                   "ros_sched_rank": 4, "ros_pos_rank": 1},
        "negvor": {"ros_ppg": 5.0, "ros_total": 35.0, "ros_games": 7,
                   "ros_sched_rank": 6, "ros_pos_rank": 2},
        "puka": {"ros_ppg": 16.0, "ros_total": 13.1, "ros_games": 0.8,
                 "ros_sched_rank": 3, "ros_pos_rank": 3},
        "alpha": {"ros_ppg": 20.0, "ros_total": 140.0, "ros_games": 7,
                  "ros_sched_rank": 2, "ros_pos_rank": 1},
        "beta": {"ros_ppg": 14.0, "ros_total": 84.0, "ros_games": 6,
                 "ros_sched_rank": 5, "ros_pos_rank": 2},
    }
    table = [
        {"id": "wilson", "name": "Wilson Fillin", "team": "GB", "position": "RB"},
        {"id": "keeper", "name": "Keeper Back", "team": "DET", "position": "RB"},
        {"id": "negvor", "name": "Below Replacement Back", "team": "MIA", "position": "RB"},
        {"id": "puka", "name": "Puka Elite", "team": "LAR", "position": "WR"},
        {"id": "alpha", "name": "Wideout Alpha", "team": "CIN", "position": "WR"},
        {"id": "beta", "name": "Wideout Beta", "team": "DAL", "position": "WR"},
    ]
    return ({"players": players, "window": {"start_week": 11, "end_week": 17},
             "updated": "November 09, 2026"}, table)


def _pin_replacement(monkeypatch):
    monkeypatch.setattr(
        "utils.ros_rankings.ros_replacement_map",
        lambda by_pos, league_size, league_type, **kw: {"RB": 10.0, "WR": 12.0},
    )


def test_public_overall_page_has_no_guest_only_floor(monkeypatch):
    # Audit D3: the overall board used to drop ros_games < 2.0 and
    # VOR < 0 on the guest page only (60 of 140 pool players, Mahomes
    # included), while /players ROS All listed the same pool in full.
    # The guest overall now shows the full eligible pool in VOR order:
    # sub-replacement and short-slate players land at the bottom.
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    _pin_replacement(monkeypatch)
    payload, table = _floor_payload()
    body = build_ros_rankings_body(payload, table, position=None)
    for name in ("Keeper Back", "Wideout Alpha", "Wideout Beta",
                 "Wilson Fillin", "Puka Elite", "Below Replacement Back"):
        assert name in body
    # VOR slots (RB line 10.0, WR line 12.0): alpha +56, keeper +35,
    # beta +12, puka +3.2, wilson -8.0, negvor -35, so the position
    # sequence is WR, RB, WR, WR, RB, RB. This fixture carries no
    # factor scores, so each position fills its slots in ROS-total
    # order (keeper 105, negvor 35, wilson 1.0): negvor takes the
    # second RB slot ahead of wilson despite the lower VOR.
    order = ["Wideout Alpha", "Keeper Back", "Wideout Beta", "Puka Elite",
             "Below Replacement Back", "Wilson Fillin"]
    idx = [body.index(n) for n in order]
    assert idx == sorted(idx)


def _slot_fill_rows():
    """Mixed pool where VOR and the factor model disagree inside RB:
    the Javonte-shaped back owns the higher VOR (38.6 on a higher PPG
    x games), the Chuba-shaped back owns the higher factor score
    (0.670, RB1) on a 0.1-lower VOR."""
    def row(name, pos, vor, score, rank, total, ppg):
        return {"name": name, "position": pos, "_vor": vor,
                "ros_score": score, "ros_pos_rank": rank,
                "ros_total": total, "ros_ppg": ppg}
    return [
        row("Qb One", "QB", 60.0, 0.90, 1, 180.0, 25.0),
        row("Qb Two", "QB", 10.0, 0.80, 2, 150.0, 21.0),
        row("Back Vor", "RB", 38.6, 0.571, 2, 92.6, 17.02),
        row("Back Score", "RB", 38.5, 0.670, 1, 96.5, 16.41),
        row("Back Three", "RB", 5.0, 0.40, 3, 80.0, 13.0),
        row("Wideout One", "WR", 55.0, 0.85, 1, 170.0, 24.0),
        row("Wideout Two", "WR", 20.0, 0.70, 2, 140.0, 20.0),
        row("Tight One", "TE", 30.0, 0.88, 1, 120.0, 17.0),
        row("Tight Two", "TE", 2.0, 0.60, 2, 95.0, 13.5),
    ]


def test_order_overall_slot_fills_factor_order():
    rows = _slot_fill_rows()
    out = order_overall(rows)
    # The slots each position occupies are exactly the pure-VOR slots:
    # the position sequence of the combined list is unchanged.
    pure = sorted(rows, key=lambda r: -r["_vor"])
    assert [r["position"] for r in out] == [r["position"] for r in pure]
    # Inside RB the factor winner fills the first RB slot even though
    # his VOR is 0.1 lower: the All view now agrees with the RB page.
    names = [r["name"] for r in out]
    assert names.index("Back Score") < names.index("Back Vor")


def test_order_overall_matches_position_rank_within_every_position():
    rows = _slot_fill_rows()
    out = order_overall(rows)
    for pos in ("QB", "RB", "WR", "TE"):
        sub = [r for r in out if r["position"] == pos]
        by_rank = sorted((r for r in rows if r["position"] == pos),
                         key=lambda r: r["ros_pos_rank"])
        assert [r["name"] for r in sub] == [r["name"] for r in by_rank]


def test_order_overall_monotonicity_guard_fixes_tiebreak_inversion():
    # The McBride shape: TE2 owns the higher VOR total (and the
    # higher ROS total) on more expected games, TE1 owns the better
    # per-game line. Their factor scores and PPGs are exactly tied,
    # so the rank assignment breaks the tie by player id (TE1 first)
    # while the slot-fill breaks it by name (TE2's name sorts first)
    # — without the guard the combined board prints TE2 above TE1.
    def row(name, pos, vor, score, rank, total, ppg):
        return {"name": name, "position": pos, "_vor": vor,
                "ros_score": score, "ros_pos_rank": rank,
                "ros_total": total, "ros_ppg": ppg}
    rows = [
        row("Qb One", "QB", 60.0, 0.90, 1, 180.0, 25.0),
        row("Aaron Spike", "TE", 35.0, 1.0, 2, 92.5, 17.0),
        row("Back One", "RB", 32.0, 0.80, 1, 150.0, 20.0),
        row("Zed Anchor", "TE", 30.0, 1.0, 1, 88.2, 17.0),
        row("Wideout One", "WR", 20.0, 0.85, 1, 140.0, 19.0),
    ]
    out = order_overall(rows)
    names = [r["name"] for r in out]
    # TE1 sits above TE2 even though TE2 has the higher VOR total.
    assert names.index("Zed Anchor") < names.index("Aaron Spike")
    assert names == ["Qb One", "Zed Anchor", "Back One",
                     "Aaron Spike", "Wideout One"]
    # The guard only permutes inside a position's own slots: the
    # position sequence is still exactly the pure-VOR sequence.
    pure = sorted(rows, key=lambda r: -r["_vor"])
    assert [r["position"] for r in out] == [r["position"] for r in pure]


def test_order_overall_monotonicity_skips_positions_without_ranks():
    # A position whose rows carry no ros_pos_rank has no rank order
    # to enforce: the guard leaves its slot-fill order untouched.
    def row(name, pos, vor, total, ppg):
        return {"name": name, "position": pos, "_vor": vor,
                "ros_total": total, "ros_ppg": ppg}
    rows = [
        row("Back Low", "RB", 40.0, 50.0, 10.0),
        row("Back High", "RB", 30.0, 90.0, 15.0),
        row("Wideout One", "WR", 20.0, 70.0, 12.0),
    ]
    out = order_overall(rows)
    # No factor scores, so the fill is ROS-total order: Back High
    # takes the first RB slot despite Back Low's higher VOR, and the
    # guard does not second-guess it without ranks.
    assert [r["name"] for r in out] == ["Back High", "Back Low",
                                        "Wideout One"]


def test_public_overall_page_slot_fills_like_position_pages(monkeypatch):
    # Same Chuba/Javonte shape through the guest builder: pinned
    # replacement lines (RB 10.0, WR 12.0) make "Back Vor" the VOR
    # winner (84 vs 77) and "Back Score" the factor winner (0.95).
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    _pin_replacement(monkeypatch)
    players = {
        "wideout1": {"ros_ppg": 25.0, "ros_total": 175.0, "ros_games": 7,
                     "ros_sched_rank": 5, "ros_pos_rank": 1, "ros_score": 0.80},
        "backvor": {"ros_ppg": 22.0, "ros_total": 154.0, "ros_games": 7,
                    "ros_sched_rank": 5, "ros_pos_rank": 2, "ros_score": 0.50},
        "backscore": {"ros_ppg": 21.0, "ros_total": 147.0, "ros_games": 7,
                      "ros_sched_rank": 5, "ros_pos_rank": 1, "ros_score": 0.95},
        "wideout2": {"ros_ppg": 15.0, "ros_total": 105.0, "ros_games": 7,
                     "ros_sched_rank": 5, "ros_pos_rank": 2, "ros_score": 0.70},
    }
    table = [
        {"id": "wideout1", "name": "Wideout One", "team": "CIN", "position": "WR"},
        {"id": "backvor", "name": "Back Vor", "team": "DAL", "position": "RB"},
        {"id": "backscore", "name": "Back Score", "team": "CAR", "position": "RB"},
        {"id": "wideout2", "name": "Wideout Two", "team": "DAL", "position": "WR"},
    ]
    payload = {"players": players, "window": {"start_week": 11, "end_week": 17},
               "updated": "November 09, 2026"}
    body = build_ros_rankings_body(payload, table, position=None)
    # Pure VOR order would be One, Vor, Score, Two; the slots stay
    # WR, RB, RB, WR but the RB slots fill in factor order.
    order = ["Wideout One", "Back Score", "Back Vor", "Wideout Two"]
    idx = [body.index(n) for n in order]
    assert idx == sorted(idx)
    # And the RB position page agrees: Back Score is its RB1.
    rb = build_ros_rankings_body(payload, table, position="RB")
    assert rb.index("Back Score") < rb.index("Back Vor")


def test_public_position_pages_keep_full_pool(monkeypatch):
    # Position pages keep the full eligible pool (this fixture carries
    # no model scores, so the fallback ROS-total order applies).
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    _pin_replacement(monkeypatch)
    payload, table = _floor_payload()
    rb = build_ros_rankings_body(payload, table, position="RB")
    assert rb.index("Keeper Back") < rb.index("Below Replacement Back")
    assert rb.index("Below Replacement Back") < rb.index("Wilson Fillin")
    wr = build_ros_rankings_body(payload, table, position="WR")
    assert wr.index("Wideout Alpha") < wr.index("Wideout Beta")
    # The elite sub-2-games receiver is still on his position page, last
    # by ROS total, exactly where the live board had him.
    assert wr.index("Wideout Beta") < wr.index("Puka Elite")


def test_public_pages_label_games_as_expected():
    # The stamped ros_games ARE expected games (scheduled games x
    # availability rate), so fractional values like 7.1 are correct data
    # that read as a typo under a bare "Games" header. The header and the
    # overall page's intro now say so.
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    payload["players"]["rb1"]["ros_games"] = 7.1
    for pos in (None, "RB"):
        body = build_ros_rankings_body(payload, table, position=pos)
        assert '<th class="rnk-val rnk-col-games">Exp. Games</th>' in body
        assert '<th class="rnk-val">Games</th>' not in body
    body = build_ros_rankings_body(payload, table, position=None)
    assert '<td class="rnk-val rnk-col-games">7.1</td>' in body
    assert "expected games over the rest of the season" in body


def test_players_board_ros_games_header_says_expected():
    # The /players board's ROS mode carries the same expected-games field
    # in the column it renames at render time; it gets the same label.
    # (Its dynasty/redraft "Games" surfaces elsewhere are actual games
    # played and keep the bare label.)
    src = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    assert "_totHeadEl.textContent = 'Exp. Games'" in src
    assert "_totHeadEl.textContent = 'Games'" not in src


def test_public_pages_have_unique_editorial_and_no_em_dash():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    bodies = {pos: build_ros_rankings_body(payload, table, position=pos)
              for pos in (None, "QB", "RB", "WR", "TE")}
    intros = set()
    for body in bodies.values():
        assert "\u2014" not in body  # no em dashes in copy
        start = body.index("rnk-analysis-title")
        intros.add(body[start:start + 60])
    assert len(intros) == 5  # every page carries its own editorial


def test_sitemap_registers_ros_pages():
    src = open(os.path.join(_ROOT, "routes", "public_bp.py"), encoding="utf-8").read()
    for path in ("/rankings/rest-of-season", "/rankings/rest-of-season-qb",
                 "/rankings/rest-of-season-rb", "/rankings/rest-of-season-wr",
                 "/rankings/rest-of-season-te"):
        assert f'("{path}", "0.8", "weekly")' in src


def test_changelog_has_one_ros_entry():
    from tests.pin_helpers import changelog_entry

    entry = changelog_entry("Rest of Season mode", tag="new", link="/players")
    assert "stretch run through Week 14" in entry["text"]


# ── Frontend wiring (static guards, the page's established test style) ──────

def test_players_shell_ros_wiring():
    shell = open(os.path.join(
        _ROOT, "dashboard_services", "pages", "players_page.py"),
        encoding="utf-8").read()
    assert 'data-value="ros"' in shell
    assert "prSetScoringType('ros')" in shell
    assert 'id="prRosWindowSection"' in shell
    assert "prSetRosWindow('playoffs')" in shell
    # Plain list (Kaedon, 2026-11-22): the ROS sort pills and hidden
    # sort options that named ROS Pts / ROS PPG are gone, and the
    # shell carries the pr-ros-plain CSS that hides the PPG / Value
    # columns and re-grids the rows in ROS mode.
    assert 'data-sortkey="ros_total"' not in shell
    assert 'data-sortkey="ros_ppg"' not in shell
    assert '<option value="ros_total">' not in shell
    assert '<option value="ros_ppg">' not in shell
    assert ".pr-ros-plain" in shell
    assert ".pr-sched" in shell


def test_players_shell_ros_no_sort_bar_and_settings_button():
    # Kaedon's screenshot fixes (2026-11-22): in ROS mode there is NO
    # sort bar at all (the lone "Pos Rank" pill left a big empty grey
    # bar), the phone ROS rows collapse to rank | player with the
    # player group LEFT-anchored (the approved mock; a centered group
    # was built first and Kaedon rejected it — he only ever meant the
    # row alignment left by the removed Value column), and league
    # settings live behind the one visible Settings button — never
    # inline segments.
    shell = open(os.path.join(
        _ROOT, "dashboard_services", "pages", "players_page.py"),
        encoding="utf-8").read()
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    # No inline league-format segment; format + size are panel sections.
    assert 'id="prFormatSeg"' not in shell
    assert 'id="prFormatSection"' in shell
    assert 'id="prSizeSection"' in shell
    assert 'id="prSettingsBtn"' in shell
    assert 'id="prSettingsPanel"' in shell
    # The Settings row stays visible on phones (it used to be hidden
    # with the whole pinned-right group, which is why league settings
    # could not be found).
    assert ".pr-pinright { display: flex; flex: 1 1 100%;" in shell
    # Phone ROS rows: two tracks, Exp. Games header gone, player group
    # anchored at the start of the player track — never centered.
    assert ("#prTableScroll.pr-ros-plain .pr-name { justify-content: flex-start; }"
            in shell)
    assert ("#prTableScroll.pr-ros-plain .pr-name { justify-content: center; }"
            not in shell)
    assert "#prTableScroll.pr-ros-plain #prTotalHeader { display: none !important; }" in shell
    # The Settings button carries the current-settings summary from the
    # approved mock ("1QB · 12 team · PPR"), filled by
    # updateSettingsIndicator() from the same state the panel edits.
    assert 'id="prSettingsSummary"' in shell
    assert "prSettingsSummary" in js
    assert "' team</b> &middot; <b>'" in js
    # JS hides the whole sort bar in ROS mode and no longer exempts
    # Pos Rank from the ROS sort reset.
    assert "_sortToggleEl.style.display = type === 'ros' ? 'none' : '';" in js
    assert "_sortLabelEl.style.display = type === 'ros' ? 'none' : '';" in js
    assert "prSortKey !== 'pos_rank'" not in js
    # The panel toggles sync from the active league's defaults on load
    # (prSyncHeaderSegs runs at init and on every settings change).
    assert "#prFormatSection .settings-toggle" in js
    assert "#prSizeSection .settings-toggle" in js
    assert "prLeagueType   = (typeof brLeagueType" in js
    assert "prLeagueSize   = (typeof brLeagueSize" in js


def test_rankings_js_ros_mode():
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    assert "function prSetRosWindow" in js
    assert "function prRosVor" in js
    assert "function prRosReplacementMap" in js
    assert "'Rest of Season'" in js
    assert "prTierThresholdsRos" in js
    assert "ros_stretch_total" in js and "ros_playoff_total" in js
    # The full-window position views order and tier by the factor model:
    # the window config carries the score field, the sort comparator and
    # the tier getter both read it.
    assert "score: 'ros_score'" in js
    assert "prRosVal(a, 'score')" in js
    assert "prRosVal(p, 'score')" in js
    # League Size stays visible in ROS mode (only redraft hides it).
    assert "type === 'redraft' ? 'none' : ''" in js


def test_rankings_js_ros_all_view_slot_fills():
    # The /players board mirrors the guest page's slot-fill rule
    # (utils.ros_rankings.order_overall): its All view re-fills the VOR
    # slots in position-view order, in both the displayed list and the
    # overall rank numbers, so the two surfaces cannot disagree inside
    # a position.
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    assert "function prRosSlotFillOrder" in js
    assert "function prRosPosViewCmp" in js
    assert "mirroring order_overall in" in js and "utils/ros_rankings.py" in js
    # The 1QB QB calibration mirrors too (Kaedon, 2026-11-22: QB1 in
    # the 20-30 band, QBs clustered within ~6 spots, max 5 in the top
    # 50; Superflex is never calibrated).
    assert "function prRosApplyQbCalibration" in js
    assert "PR_ROS_QB_CLUSTER_GAP = 6" in js
    assert "prLeagueType !== 'sf'" in js
    # The positional-rank monotonicity guard mirrors too, in both
    # formats: it enforces the printed rank order inside the fill.
    assert "function prRosApplyPositionMonotonicity" in js
    assert "prRosApplyPositionMonotonicity(ordered)" in js
    assert js.count("prRosSlotFillActive(sortBy)") >= 3  # gate + both sort sites
    pages = open(os.path.join(
        _ROOT, "dashboard_services", "pages", "dynasty_pages.py"),
        encoding="utf-8").read()
    assert "rows = order_overall(rows, league_type=league_type)" in pages


def test_rankings_js_ros_all_view_uses_vor():
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    # All-view tiers come from the payload's VOR thresholds (internal
    # ordering only). The board is a plain ranked list in ROS mode:
    # no ROS Value / ROS PPG sort meta, labels or cells remain, and
    # the pr-ros-plain class drives the hidden columns.
    assert "prTierThresholdsRosVor" in js
    assert "tier_thresholds_ros_vor" in js
    assert "ros_value" not in js and "ROS Value" not in js
    assert "valueLabel" not in js and "ppgLabel" not in js
    assert "label: 'ROS PPG'" not in js and "label: 'ROS Pts'" not in js
    assert "pr-ros-plain" in js
    # The JS mirrors the server's replacement map: the 1QB QB
    # streaming replacement (weekly best below the starter line from
    # the stamped weekly series, with the best-of fallback over the
    # six quarterbacks below the line) and the shared flex allocation
    # (the old starter-line index mirror, "prLeagueSize - 1", went
    # away with the definition).
    assert "PR_ROS_QB_STREAM_TIER = 6" in js
    assert "PR_ROS_QB_STREAM_FALLBACK_TOP_N = 2" in js
    assert "qb.slice(prLeagueSize, prLeagueSize + PR_ROS_QB_STREAM_TIER)" in js
    assert "weekly: 'ros_weekly'" in js
    assert "weekly: 'ros_stretch_weekly'" in js
    assert "weekly: 'ros_playoff_weekly'" in js
    assert "p[cfg.weekly]" in js
    assert "leftovers" in js and "flexSpots" in js
    # The pinned PPG-grid line stays byte-for-byte (a pin test guards it).
    assert ("const valueMeta = (sortBy === 'ppg' || sortBy === 'total_pts') "
            "? PR_SORT_META.rank : sortMeta;") in js
    # app.py stamps the VOR tier block onto the payload next to the
    # positional one.
    app_src = open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read()
    assert 'payload["tier_thresholds_ros_vor"]' in app_src


# ── 1QB QB calibration + plain list + settings + precomputed cache ─────────
# Kaedon, 2026-11-22: in 1QB quarterbacks just aren't that valuable
# (one starter per team, the streaming replacement sits close to the
# mid-tier starters), so the Overall list holds QB1 in the 20-30 band,
# clusters the QBs within ~6 spots of each other, and caps them at 5
# in the top 50. Superflex starts ~24 QBs, replacement is far worse,
# and QBs legitimately rise: no calibration there. ROS surfaces are a
# plain ranked list (no ROS value, no ROS PPG displayed), the guest
# pages carry Format x Teams settings that re-derive the order from
# the replacement-level logic, and load_ros_map serves the
# precomputed disk cache instead of computing on read.

def _calibration_rows():
    """60-row ordered pool: QB1 on top by VOR, the rest of the QBs
    scattered deep, exactly the shape the live board showed."""
    rows = []

    def row(name, pos):
        rows.append({"name": name, "position": pos})

    row("Qb One", "QB")
    for i in range(39):
        row(f"Flex {i + 1}", ("RB", "WR", "TE")[i % 3])
    row("Qb Two", "QB")
    for i in range(4):
        row(f"Flex B{i + 1}", "RB")
    row("Qb Three", "QB")
    for i in range(4):
        row(f"Flex C{i + 1}", "WR")
    row("Qb Four", "QB")
    row("Flex D1", "TE")
    row("Qb Five", "QB")
    row("Flex D2", "RB")
    row("Qb Six", "QB")
    row("Flex D3", "WR")
    row("Qb Seven", "QB")
    row("Flex D4", "TE")
    row("Flex D5", "RB")
    row("Flex D6", "WR")
    return rows


def test_qb_calibration_band_cluster_and_cap():
    rows = _calibration_rows()
    assert len(rows) == 60
    out = apply_qb_calibration(rows, "1qb")
    qb_ranks = [i + 1 for i, r in enumerate(out) if r["position"] == "QB"]
    # QB1 in the band, each QB within 6 spots of the previous one,
    # and exactly the capped 5 inside the top 50.
    assert qb_ranks[0] == 24
    assert all(b - a <= 6 for a, b in zip(qb_ranks, qb_ranks[1:]))
    assert sum(1 for r in qb_ranks if r <= 50) == 5
    # Quarterbacks keep their positional order, and so do the rest.
    assert [r["name"] for r in out if r["position"] == "QB"] == [
        "Qb One", "Qb Two", "Qb Three", "Qb Four", "Qb Five",
        "Qb Six", "Qb Seven"]
    assert [r["name"] for r in out if r["position"] != "QB"] == [
        r["name"] for r in rows if r["position"] != "QB"]


def test_qb_calibration_cap_handles_boundary_dense_qbs():
    # Regression: QBs packed across the rank-50 boundary used to
    # loop forever (popping the 6th QB pulled the 7th into the top
    # 50, and the loop swapped them back). The cap is one pass now.
    rows = [{"name": f"Flex {i}", "position": "RB"} for i in range(44)]
    rows += [{"name": f"Qb {i + 1}", "position": "QB"} for i in range(10)]
    rows += [{"name": f"Flex T{i}", "position": "WR"} for i in range(6)]
    assert len(rows) == 60
    out = apply_qb_calibration(rows, "1qb")
    qb_ranks = [i + 1 for i, r in enumerate(out) if r["position"] == "QB"]
    assert sum(1 for r in qb_ranks if r <= 50) == 5
    assert qb_ranks[5] == 51  # the excess QB lands just below the cap
    assert [r["name"] for r in out if r["position"] == "QB"] == [
        f"Qb {i + 1}" for i in range(10)]


def test_qb_calibration_never_touches_sf_or_small_pools():
    rows = _calibration_rows()
    out = apply_qb_calibration(rows, "sf")
    assert [r["name"] for r in out] == [r["name"] for r in rows]
    small = rows[:20]
    out = apply_qb_calibration(small, "1qb")
    assert [r["name"] for r in out] == [r["name"] for r in small]


def _settings_pool():
    """Synthetic 76-player pool for the guest settings shape check:
    20 QBs with a clear #1 (a realistic QB pool depth, so the
    Superflex replacement line has a deep QB2 band to fall to),
    deep RB/WR pools, 6 TEs."""
    players, table = {}, []

    def add(pid, name, pos, ppg, rank):
        players[pid] = {"ros_ppg": ppg, "ros_total": ppg * 6,
                        "ros_games": 6, "ros_sched_rank": 5,
                        "ros_pos_rank": rank,
                        "ros_score": 3.0 - 0.05 * rank}
        table.append({"id": pid, "name": name, "team": "BUF",
                      "position": pos})

    for i in range(20):
        add(f"qb{i}", f"Quarterback {i + 1}", "QB", 28.0 - 0.6 * i, i + 1)
    for i in range(26):
        add(f"rb{i}", f"Back {i + 1}", "RB", 21.0 - 0.35 * i, i + 1)
    for i in range(24):
        add(f"wr{i}", f"Wideout {i + 1}", "WR", 20.0 - 0.35 * i, i + 1)
    for i in range(6):
        add(f"te{i}", f"Tight End {i + 1}", "TE", 16.0 - 0.5 * i, i + 1)
    return ({"players": players,
             "window": {"start_week": 11, "end_week": 17},
             "updated": "November 22, 2026"}, table)


def _overall_names(body):
    import re
    return re.findall(r'rnk-player-link"[^>]*>([^<]+)</a>', body)


def test_guest_settings_1qb_holds_qbs_sf_lets_them_rise():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _settings_pool()
    body_1qb = build_ros_rankings_body(payload, table, position=None)
    names_1qb = _overall_names(body_1qb)
    qb_ranks_1qb = [i + 1 for i, n in enumerate(names_1qb)
                    if n.startswith("Quarterback")]
    assert 20 <= qb_ranks_1qb[0] <= 30
    assert sum(1 for r in qb_ranks_1qb if r <= 50) <= 5
    # The cluster binds the group inside the top 50; the cap exiles
    # the rest below it by design, so their gap to QB5 is the cap
    # working, not a cluster break.
    in50 = [r for r in qb_ranks_1qb if r <= 50]
    assert all(b - a <= 6 for a, b in zip(in50, in50[1:]))

    body_sf = build_ros_rankings_body(
        payload, table, position=None, league_type="sf", league_size=12)
    names_sf = _overall_names(body_sf)
    qb_ranks_sf = [i + 1 for i, n in enumerate(names_sf)
                   if n.startswith("Quarterback")]
    # Superflex: ~24 QBs start, the replacement line drops to the
    # deep QB2 band, and the top quarterbacks rise legitimately.
    assert qb_ranks_sf[0] <= 5
    assert qb_ranks_sf[0] < qb_ranks_1qb[0]


def test_guest_pages_are_plain_lists_with_settings_control():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _settings_pool()
    body = build_ros_rankings_body(payload, table, position=None)
    # No ROS value and no ROS PPG is displayed: no such headers, and
    # none of the pool's PPG / total figures render as cells.
    assert "ROS PPG" not in body and "ROS Pts" not in body
    assert "168.0" not in body  # QB1's ROS total
    assert '<th class="rnk-val rnk-col-games">Exp. Games</th>' in body
    # Settings control: format + teams, default shape on bare URLs,
    # non-default shapes on validated query params.
    assert 'aria-label="League settings"' in body
    assert 'href="/rankings/rest-of-season?fmt=sf"' in body
    assert 'href="/rankings/rest-of-season?size=8"' in body
    assert 'href="/rankings/rest-of-season?fmt=sf&size=8"' not in body
    body_sf8 = build_ros_rankings_body(
        payload, table, position=None, league_type="sf", league_size=8)
    assert 'href="/rankings/rest-of-season?fmt=sf&size=8"' in body_sf8
    # Position pages carry no settings control: format/size cannot
    # move a within-position list, so there is nothing to set there.
    body_rb = build_ros_rankings_body(payload, table, position="RB")
    assert 'aria-label="League settings"' not in body_rb
    assert "ROS PPG" not in body_rb and "ROS Pts" not in body_rb


def test_ros_disk_cache_roundtrip_and_signature_gate(tmp_path, monkeypatch):
    import utils.ros_rankings as rr

    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    key = (2026, 11, (("sleeper_stats_s2026_w10.json", 1.5),
                      ("projections_s2026_w11.json", 2.5)))
    payload = {"players": {"1": {"ros_ppg": 20.0}},
               "window": {"start_week": 11, "end_week": 17},
               "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
               "updated": "November 22, 2026"}
    assert rr._read_ros_disk_cache(2026, key) is None
    rr._write_ros_disk_cache(2026, key, payload)
    assert rr._read_ros_disk_cache(2026, key) == payload
    # A different data signature is a miss, never a stale serve.
    other = (2026, 11, (("sleeper_stats_s2026_w10.json", 9.9),))
    assert rr._read_ros_disk_cache(2026, other) is None


def test_ros_disk_cache_build_version_invalidates_pre_guard_payload(
        tmp_path, monkeypatch):
    # The cache key carries ROS_BUILD_VERSION: a payload persisted by
    # a pre-guard build under the SAME data signature (the old
    # 3-element key shape) must never be served to the guarded build.
    import utils.ros_rankings as rr

    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    sig = (("sleeper_stats_s2026_w10.json", 1.5),)
    payload = {"players": {"1": {"ros_ppg": 20.0}},
               "window": {"start_week": 11, "end_week": 17},
               "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
               "updated": "November 22, 2026"}
    old_key = (2026, 11, sig)
    rr._write_ros_disk_cache(2026, old_key, payload)
    new_key = (2026, 11, rr.ROS_BUILD_VERSION, sig)
    assert rr._read_ros_disk_cache(2026, new_key) is None
    # The versioned key itself round-trips.
    rr._write_ros_disk_cache(2026, new_key, payload)
    assert rr._read_ros_disk_cache(2026, new_key) == payload


def test_load_ros_map_serves_precomputed_disk_cache(tmp_path, monkeypatch):
    import utils.ros_rankings as rr

    monkeypatch.setattr("utils.data_cache.CACHE_DIR", str(tmp_path))
    payload = {"players": {"1": {"ros_ppg": 20.0, "ros_total": 120.0}},
               "window": {"start_week": 11, "end_week": 17},
               "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
               "updated": "November 22, 2026"}
    builds = {"n": 0}

    def fake_build(season, start_week, updated):
        builds["n"] += 1
        return payload

    monkeypatch.setattr(rr, "_build_ros_map", fake_build)
    monkeypatch.setattr(rr, "_files_signature", lambda s, w: (("f.json", 1.0),))
    rr._ROS_CACHE["key"] = None
    rr._ROS_CACHE["payload"] = None
    try:
        first = rr.load_ros_map(2026)
        assert builds["n"] == 1 and first["players"]
        # A cold process (in-memory cache cleared) serves the
        # precomputed disk payload with NO rebuild: the read path is
        # a file read, like the dynasty/redraft value tables.
        rr._ROS_CACHE["key"] = None
        rr._ROS_CACHE["payload"] = None
        second = rr.load_ros_map(2026)
        assert builds["n"] == 1
        assert second == payload
    finally:
        rr._ROS_CACHE["key"] = None
        rr._ROS_CACHE["payload"] = None
