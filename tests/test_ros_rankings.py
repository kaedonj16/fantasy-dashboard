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
    availability_rate,
    compute_player_ros,
    compute_ros_tier_thresholds,
    compute_ros_vor_tier_thresholds,
    is_season_out,
    passes_role_gate,
    qb_ros_ppg,
    ros_flex_replacement_ppgs,
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


# ── Cross-position VOR: replacement convention + All-view tiers ────────────

def test_ros_replacement_definitions():
    # Replaced the old index-convention test: the replacement lines are
    # no longer per-position indices. QB 1QB is the streaming level, QB
    # SF keeps the scarcity index, and RB/WR/TE come out of one shared
    # flex-aware allocation (below).
    ladder = [26.0 - 0.5 * i for i in range(20)]
    # 1QB streaming: 0.92 x mean of the top half of the starter pool
    # (QB1..QB5 of the 10 starters): mean(26, 25.5, 25, 24.5, 24) = 25.0.
    assert ros_qb_replacement_ppg(ladder, 10, "1qb") == pytest.approx(23.0)
    # SF: the player at index 2 x 10 x 1.5 = 30 on a deeper ladder.
    deep = [26.0 - 0.5 * i for i in range(40)]
    assert ros_qb_replacement_ppg(deep, 10, "sf") == pytest.approx(11.0)
    assert ros_qb_replacement_ppg([], 10, "1qb") is None
    # The streaming line sits ABOVE the worst starter (ladder[9] = 21.5):
    # that is the point of pricing the streaming fallback, not a fixed QB.
    assert ros_qb_replacement_ppg(ladder, 10, "1qb") > ladder[9]


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
    assert full["QB"] == pytest.approx(23.0)
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
    # 1QB replacement QB is the streaming level: 0.92 x mean of the top
    # half of the starter pool (26, 25.4, 24.8, 24.2, 23.6 -> 24.8), so
    # QB1's VOR is (26 - 22.816) x 8 = 25.472, stamped rounded to 25.5.
    assert vor_1qb["qb0"] == pytest.approx(25.5)
    # SF: idx 30 clamps to this pool's QB12 (19.4): a strictly bigger edge.
    assert vor_sf["qb0"] == pytest.approx((26.0 - 19.4) * 8)
    assert vor_sf["qb0"] > vor_1qb["qb0"]
    # RB replacement is format-independent: identical VOR either way.
    # Flex-aware line: 20 dedicated RBs + 6 of the 10 flex spots (the
    # combined leftovers order RB-heavy on this pool) -> RB idx 26.
    assert vor_1qb["rb0"] == vor_sf["rb0"] == pytest.approx((24.0 - 12.3) * 8)
    # The point of the discount: in 1QB the clustered QB1 no longer
    # out-VORs the top RB; in SF he closes most of the gap back.
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
    # ...but his VOR is mid-pack (25.5) because 1QB QBs cluster and the
    # streaming line prices the fallback honestly, so the All view's VOR
    # thresholds must NOT stamp him T1, while RB1 (the actual VOR leader
    # at 93.6) is T1. One metric, one sequence.
    vor_t = compute_ros_vor_tier_thresholds(pool, pos)["all"]["1qb"]["10"]
    qb1_vor = ros_vor_totals(pool, pos, 10, "1qb")["all"]["qb0"]
    rb1_vor = ros_vor_totals(pool, pos, 10, "1qb")["all"]["rb0"]
    assert _tier_of(vor_t, rb1_vor) == 1
    assert _tier_of(vor_t, qb1_vor) != 1


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


def test_public_position_page_orders_by_ros_total():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position="RB")
    assert body.index("Back One") < body.index("Back Two")
    assert "Quarterback One" not in body
    assert "Updated November 09, 2026" in body or "updated November 09, 2026" in body
    assert "Full PPR" in body


def test_public_overall_page_discounts_qb1_in_1qb():
    # 20 clustered QBs: under the old 1.5x index the replacement QB was
    # QB19 (19.7 PPG) and QB1's VOR (44.1) beat the 20-PPG back's (42.0).
    # At the 1QB streaming level the replacement is 0.92 x the top-half
    # starter mean (25.125) = 23.115 PPG, QB1's VOR drops to ~20.2, and
    # the back orders ahead of him by a wide margin.
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
    # "Back 3" is the 20.0-PPG back: ahead of "Quarterback 1" only under
    # the starter-line replacement.
    assert body.index("Back 3") < body.index("Quarterback 1")
    assert body.index("Back 1") < body.index("Back 3")


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
        assert '<th class="rnk-val">Exp. Games</th>' in body
        assert '<th class="rnk-val">Games</th>' not in body
    body = build_ros_rankings_body(payload, table, position=None)
    assert '<td class="rnk-val">7.1</td>' in body
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


def test_rankings_js_ros_all_view_uses_vor():
    js = open(os.path.join(_ROOT, "static", "rankings.js"), encoding="utf-8").read()
    # All-view tiers come from the payload's VOR thresholds, and the
    # trailing column is the VOR "Value" column the board orders by.
    assert "prTierThresholdsRosVor" in js
    assert "tier_thresholds_ros_vor" in js
    assert "ros_value" in js and "ROS Value" in js
    assert "valueLabel" in js
    # The JS mirrors the server's replacement map: the 1QB QB streaming
    # factor and the shared flex allocation (the old starter-line index
    # mirror, "prLeagueSize - 1", went away with the definition).
    assert "PR_ROS_QB_STREAM_FACTOR = 0.92" in js
    assert "leftovers" in js and "flexSpots" in js
    # The pinned PPG-grid line stays byte-for-byte (a pin test guards it).
    assert ("const valueMeta = (sortBy === 'ppg' || sortBy === 'total_pts') "
            "? PR_SORT_META.rank : sortMeta;") in js
    # app.py stamps the VOR tier block onto the payload next to the
    # positional one.
    app_src = open(os.path.join(_ROOT, "app.py"), encoding="utf-8").read()
    assert 'payload["tier_thresholds_ros_vor"]' in app_src
