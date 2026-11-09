"""Focused tests for the Rest of Season rankings mode.

Covers the ROS compute (utils/ros_rankings.py): the backtested
position-aware formula (projection-only for RB/WR/TE, 40/30/30 with
form shrinkage for QB), the eligibility gates, weeks-out / bye handling
inside each window, per-window positional ranks, tier thresholds derived
from the ROS totals displayed, the payload whitelist, the public
/rest-of-season page bodies, the sitemap registration, and the
rankings.js / players-shell mode wiring (static guards, the established
style for this page).
"""
import os

import pytest

pytest.importorskip("pandas")

from utils.ros_rankings import (
    assign_ros_ranks,
    compute_player_ros,
    compute_ros_tier_thresholds,
    is_season_out,
    passes_role_gate,
    qb_ros_ppg,
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


# ── Window series math ──────────────────────────────────────────────────────

def test_window_stats_leading_zero_run_is_weeks_out():
    stats = window_projection_stats([0.0, 0.4, 18.0, 20.0])
    assert stats["weeks_out"] == 2
    assert stats["games"] == 2
    assert stats["total"] == pytest.approx(38.0)
    assert stats["fwd_ppg"] == pytest.approx(19.0)


def test_window_stats_absent_weeks():
    # Absent before the return counts as out; absent mid-window is skipped,
    # never scored as a zero game.
    assert window_projection_stats([None, 15.0])["games"] == 1
    stats = window_projection_stats([20.0, None, 18.0])
    assert stats["games"] == 2
    assert stats["fwd_ppg"] == pytest.approx(19.0)


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
    # A backup QB passes only as his team's #1.
    assert not passes_role_gate("QB", is_team_top=False, touches_per_game=None,
                                targets_per_game=None)
    assert passes_role_gate("QB", is_team_top=True, touches_per_game=None,
                            targets_per_game=None)


def test_five_game_floor_and_season_out_exclusion():
    assert _ros("RB", played_games=4) is None
    assert _ros("RB", played_games=5) is not None
    assert _ros("RB", season_out=True) is None
    assert is_season_out("IR") and is_season_out("PUP")
    assert not is_season_out("Out") and not is_season_out("Questionable")


def test_no_projection_no_fields():
    assert _ros("RB", windows={"all": [0.0, 0.0], "stretch": [], "playoffs": []}) is None


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


def test_tier_thresholds_come_from_ros_totals():
    pool, pos = {}, {}
    for i in range(60):
        pid = str(i)
        total = 200.0 - i * 3.0
        pool[pid] = {"ros_total": total, "ros_ppg": total / 7,
                     "ros_stretch_total": total * 0.6, "ros_stretch_ppg": total / 7,
                     "ros_playoff_total": total * 0.4, "ros_playoff_ppg": total / 7}
        pos[pid] = "RB" if i % 2 == 0 else "WR"
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


# ── Payload path ────────────────────────────────────────────────────────────

def test_slim_payload_keeps_ros_fields_and_blocks():
    from dashboard_services.league_players_board import slim_rankings_payload

    payload = {
        "players": [{"id": "1", "name": "X", "position": "RB",
                     "ros_ppg": 20.0, "ros_total": 120.0, "ros_games": 6,
                     "ros_sched_rank": 3, "ros_pos_rank": 2,
                     "ros_stretch_total": 80.0, "ros_playoff_total": 40.0,
                     "internal_only": 1}],
        "ros_window": {"start_week": 11, "end_week": 17},
        "tier_thresholds_ros": {"all": [100.0], "stretch": [60.0], "playoffs": [30.0]},
    }
    out = slim_rankings_payload(payload)
    row = out["players"][0]
    assert row["ros_total"] == 120.0 and row["ros_sched_rank"] == 3
    assert row["ros_stretch_total"] == 80.0 and row["ros_playoff_total"] == 40.0
    assert "internal_only" not in row
    assert out["ros_window"] == {"start_week": 11, "end_week": 17}
    assert out["tier_thresholds_ros"]["playoffs"] == [30.0]


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


def test_public_position_page_orders_by_ros_total():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position="RB")
    assert body.index("Back One") < body.index("Back Two")
    assert "Quarterback One" not in body
    assert "Updated November 09, 2026" in body or "updated November 09, 2026" in body
    assert "Full PPR" in body


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
    assert '<option value="ros_total">ROS Total</option>' in shell
    assert '<option value="ros_ppg">ROS PPG</option>' in shell
    assert ".pr-sched" in shell


def test_rankings_js_ros_mode():
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    assert "function prSetRosWindow" in js
    assert "function prRosVor" in js
    assert "function prRosReplacementMap" in js
    assert "'Rest of Season'" in js
    assert "prTierThresholdsRos" in js
    assert "ros_stretch_total" in js and "ros_playoff_total" in js
    # League Size stays visible in ROS mode (only redraft hides it).
    assert "type === 'redraft' ? 'none' : ''" in js
