"""Regression tests for the mock-fidelity fix pass (PR after #3195).

Covers: (1) catch rate shipped as a 0-1 fraction and rendered "1%"
everywhere, (2) -0% formatting artifacts, (3) the surname-suffix
takeaway bug (JS source level), (4) WRs sprouting an RB gap map,
(8) zone trends built from calendar weeks instead of the player's
own game weeks, and the comparables rebuild (tier gate, full-profile
distance, honest scale, twin == first comp). Follow-ups on the same
PR: (9) the gap map is bars only (no league tick on the lanes, no RB
dot under the formation), (10) zone/lane details open as popovers
with a close button instead of inline panels.
"""
import math
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from dashboard_services import player_roles as pr  # noqa: E402

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"


# ---------------------------------------------------------------------------
# 1. Catch rate is percent units everywhere in the payload
# ---------------------------------------------------------------------------

def _zone_cell(**kw):
    cell = {"primary": 0, "secondary": 0, "yards": 0.0, "td": 0, "int": 0,
            "first_downs": 0, "air": 0.0, "big": 0,
            "primary_last4": 0, "primary_base": 0}
    cell.update(kw)
    return cell


def test_zone_catch_rate_is_percent_not_fraction():
    # Egbuka's real Short Right line: 8 receptions on 12 targets.
    zones = {("Short", "right"): _zone_cell(
        primary=12, secondary=8, yards=84.0, air=60.0,
        primary_last4=6, primary_base=12)}
    league = {("Short", "right"): _zone_cell(primary=100, secondary=74)}
    totals = {"att": 12, "comp": 8, "yards": 84.0, "td": 0, "int": 0,
              "air": 60.0, "deep": 0, "rz": 0, "yac": 30.0, "rec": 8,
              "big_rec": 0}
    zone_list, summary, _td, _big = pr._zone_payload(
        zones, league, False, totals, 1000.0, 60.0)
    z = next(x for x in zone_list
             if x["row"] == "Short" and x["col"] == "right")
    assert z["catch_rate"] == pytest.approx(66.7)
    assert z["league_catch_rate"] == pytest.approx(74.0)
    assert summary["catch_rate"] == pytest.approx(66.7)


# ---------------------------------------------------------------------------
# 2. Negative zero never survives formatting helpers
# ---------------------------------------------------------------------------

def test_r1_normalises_negative_zero():
    assert pr._r1(-0.04) == 0.0
    assert str(pr._r1(-0.04)) == "0.0"
    assert math.copysign(1.0, pr._r1(-0.04)) == 1.0
    # A genuinely negative value is untouched.
    assert pr._r1(-0.2) == -0.2


def test_share_normalises_negative_zero():
    assert pr._share(-1, 100000) == 0.0
    assert str(pr._share(-1, 100000)) == "0.0"


def test_payload_air_yards_share_never_negative_zero():
    zones = {("Short", "right"): _zone_cell(primary=4, secondary=3,
                                            yards=30.0, air=-1.0)}
    totals = {"att": 4, "comp": 3, "yards": 30.0, "td": 0, "int": 0,
              "air": -1.0, "deep": 0, "rz": 0, "yac": 5.0, "rec": 3,
              "big_rec": 0}
    _z, summary, _td, _big = pr._zone_payload(
        zones, {}, False, totals, 100000.0, -1.0)
    assert summary["air_yards_share"] == 0.0
    assert str(summary["air_yards_share"]) == "0.0"


# ---------------------------------------------------------------------------
# 3. Takeaway lines use first names, never a suffix token
# ---------------------------------------------------------------------------

def test_takeaways_use_mock_his_phrasing():
    src = APP_JS.read_text(encoding="utf-8")
    roles_src = src[src.index("// Compare Roles tab:"):]
    roles_src = roles_src.split("function cmpSwitchTab")[0]
    # The old last-token parse produced "III" for James Cook III; it
    # must not survive anywhere in the roles renderers. The mock's
    # takeaway lines use "his" phrasing instead.
    assert "split(' ').slice(-1)[0]" not in roles_src
    assert "his bread and butter" in roles_src
    assert "His best gap:" in roles_src
    assert "His highest-volume zone:" in roles_src


# ---------------------------------------------------------------------------
# 4. Gap map is RB-only in the payload
# ---------------------------------------------------------------------------

def _series_row(week, **kw):
    base = {"week": week, "snap_pct": 50.0, "snaps": 30, "team_snaps": 60,
            "targets": 0, "receptions": 0, "carries": 0, "touches": 0,
            "target_share": None, "carry_share": None, "ppr_pts": 0.0,
            "rec_yards": 0.0, "rush_yards": 0.0, "pass_att": 0,
            "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
            "rz_targets": 0, "rz_carries": 0}
    base.update(kw)
    return base


def _carry_tables(pid):
    lane = {"carries": 6, "yards": 30.0, "td": 1, "first_downs": 2,
            "stuffed": 1, "explosive": 1, "success": 3, "longest": 12,
            "carries_last4": 6, "carries_base": 6}
    return {
        "receiving": {"zones": {}, "league": {}, "totals": {},
                      "team_air": {}, "player_air": {}},
        "passing": {"zones": {}, "league": {}, "totals": {},
                    "team_air": {}, "player_air": {}},
        "rushing": {"lanes": {pid: {"M": dict(lane)}},
                    "league": {"M": dict(lane)},
                    "qb": {pid: {"att": 6, "yards": 30.0, "td": 1,
                                 "scramble": 0}},
                    "direction": {pid: {"left": 2, "middle": 2, "right": 2}}},
        "feats": {},
    }


def test_wr_with_carries_gets_no_gap_map():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "wr1": {"pos": "WR", "team": "TB", "name": "Test Receiver",
                "series": [_series_row(w, snap_pct=85.0, targets=8,
                                       receptions=5, target_share=22.0,
                                       rec_yards=70.0, ppr_pts=12.0)
                           for w in weeks]},
        "wr2": {"pos": "WR", "team": "NYG", "name": "Other Receiver",
                "series": [_series_row(w, snap_pct=70.0, targets=5,
                                       receptions=3, target_share=14.0,
                                       rec_yards=40.0, ppr_pts=7.0)
                           for w in weeks]},
    }
    payload = pr.build_player_payload("wr1", 2026, "season", pool=pool,
                                      tables=_carry_tables("wr1"),
                                      pa_feats={})
    assert payload is not None
    assert payload["gap_map"] is None


def test_rb_with_carries_keeps_gap_map():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "rb1": {"pos": "RB", "team": "BUF", "name": "Test Back",
                "series": [_series_row(w, snap_pct=65.0, carries=14,
                                       carry_share=50.0, rush_yards=65.0,
                                       ppr_pts=13.0, rz_carries=2)
                           for w in weeks]},
        "rb2": {"pos": "RB", "team": "MIA", "name": "Other Back",
                "series": [_series_row(w, snap_pct=45.0, carries=8,
                                       carry_share=30.0, rush_yards=34.0,
                                       ppr_pts=7.0, rz_carries=1)
                           for w in weeks]},
    }
    payload = pr.build_player_payload("rb1", 2026, "season", pool=pool,
                                      tables=_carry_tables("rb1"),
                                      pa_feats={})
    assert payload is not None
    assert payload["gap_map"] is not None
    assert payload["gap_map"]["tendency"]["middle"] is not None


# ---------------------------------------------------------------------------
# 8. Zone trends use the player's own game weeks
# ---------------------------------------------------------------------------

def _pass_row(week, pid, loc, air, complete=1):
    return {"play_type": "pass", "qb_spike": 0, "week": week,
            "receiver_player_id": pid, "passer_player_id": "qb1",
            "pass_location": loc, "air_yards": air,
            "complete_pass": complete, "yards_gained": 7.0,
            "pass_touchdown": 0, "interception": 0, "first_down_pass": 0,
            "yards_after_catch": 2.0, "posteam": "TB", "yardline_100": 50}


def test_trend_last4_uses_game_weeks_not_calendar_weeks():
    rows = []
    # p1 plays weeks 1-4 and 9 only (missed 5-8). Calendar last-4 is
    # {7,8,9,10}: under the old build his trend came from week 9 alone.
    for wk in (1, 2, 3, 4, 9):
        rows.append(_pass_row(wk, "p1", "right", 5))   # Short Right
        rows.append(_pass_row(wk, "p1", "left", 5))    # Short Left
    # A decoy fills the calendar weeks so the dataset spans 1-10.
    for wk in range(1, 11):
        rows.append(_pass_row(wk, "decoy", "middle", 5))
    weeks = set(range(1, 11))
    tables = pr.build_pbp_tables_from_rows(
        rows, {}, last4_weeks={7, 8, 9, 10}, base_weeks=weeks,
        window_weeks=weeks, window="season", player_weeks=True)
    zones = tables["receiving"]["zones"]["p1"]
    sr = zones[("Short", "right")]
    # His last-4 GAME weeks are {2,3,4,9}: one Short Right target each.
    assert sr["primary_last4"] == 4
    assert sr["primary"] == 5


def test_last4_window_uses_game_weeks():
    rows = []
    for wk in (1, 2, 3, 4, 9):
        rows.append(_pass_row(wk, "p1", "right", 5))
    for wk in range(1, 11):
        rows.append(_pass_row(wk, "decoy", "middle", 5))
    weeks = set(range(1, 11))
    tables = pr.build_pbp_tables_from_rows(
        rows, {}, last4_weeks={7, 8, 9, 10}, base_weeks={3, 4, 5, 6},
        window_weeks={7, 8, 9, 10}, window="last4", player_weeks=True)
    zones = tables["receiving"]["zones"]["p1"]
    total = sum(c["primary"] for c in zones.values())
    # Window = his last 4 game weeks {2,3,4,9}, not calendar {7,8,9,10}
    # (which would count only the week-9 target).
    assert total == 4
    sr = zones[("Short", "right")]
    # Prior-4 base = his earlier game weeks: just week 1 here.
    assert sr["primary_base"] == 1


# ---------------------------------------------------------------------------
# Comparables rebuild: tier gate, full profile, honest scale, twin == #1
# ---------------------------------------------------------------------------

_AXES = ["Volume", "Deep Work", "Efficiency", "Explosiveness",
         "Red Zone", "Receiving"]


def _computed(axes_by_pid):
    return {pid: {"axes": axes, "feats": {"games": 8}}
            for pid, axes in axes_by_pid.items()}


def _comp_pool():
    return {
        "hero": {"pos": "WR", "team": "TB", "name": "Hero Receiver"},
        "twin": {"pos": "WR", "team": "NYG", "name": "Twin Receiver"},
        "alpha": {"pos": "WR", "team": "DAL", "name": "Alpha Receiver"},
        "far": {"pos": "WR", "team": "MIA", "name": "Far Receiver"},
    }


def _comp_axes():
    hero = {"Volume": 62, "Deep Work": 55, "Efficiency": 60,
            "Explosiveness": 58, "Red Zone": 50, "Receiving": 66}
    twin = {"Volume": 58, "Deep Work": 60, "Efficiency": 55,
            "Explosiveness": 52, "Red Zone": 45, "Receiving": 60}
    # Similar Volume to the hero, everything else far away.
    alpha = {"Volume": 64, "Deep Work": 95, "Efficiency": 30,
             "Explosiveness": 92, "Red Zone": 90, "Receiving": 35}
    far = {"Volume": 58, "Deep Work": 50, "Efficiency": 55,
           "Explosiveness": 50, "Red Zone": 45, "Receiving": 60}
    return {"hero": hero, "twin": twin, "alpha": alpha, "far": far}


def _comp_distributions():
    # 12-zone share vectors, ZONE_ROWS x ZONE_COLS order.
    hero = [0.02, 0.03, 0.02, 0.06, 0.08, 0.06,
            0.20, 0.28, 0.19, 0.02, 0.02, 0.02]
    twin = [0.03, 0.03, 0.03, 0.07, 0.09, 0.06,
            0.18, 0.26, 0.19, 0.02, 0.02, 0.02]
    alpha = [0.15, 0.14, 0.15, 0.10, 0.10, 0.10,
             0.06, 0.08, 0.06, 0.02, 0.02, 0.02]
    far = [0.10, 0.05, 0.10, 0.05, 0.20, 0.05,
           0.10, 0.10, 0.10, 0.05, 0.05, 0.05]
    return {pid: {"zones": vec} for pid, vec in
            (("hero", hero), ("twin", twin), ("alpha", alpha), ("far", far))}


def test_comparables_tier_gate_and_twin_consistency():
    pool = _comp_pool()
    computed = _computed(_comp_axes())
    jobs = {"hero": "WR2", "twin": "WR2", "alpha": "Alpha WR1",
            "far": "WR3"}
    comps, twin = pr.build_comparables("hero", pool, computed, jobs,
                                       _comp_distributions())
    ids = [c["player_id"] for c in comps]
    # The alpha is two tiers away: gated out even with similar Volume.
    assert "alpha" not in ids
    # Same-tier near-identical profile ranks first, and the twin IS
    # the first comp (the old build picked them from different sorts).
    assert ids[0] == "twin"
    assert twin is not None and twin["player_id"] == comps[0]["player_id"]
    assert twin["similarity"] == comps[0]["similarity"]
    # Honest scale: nobody scores 100; the close twin lands 85-95 and
    # clearly clear of the different-shaped same-tier-ish candidate.
    assert all(c["similarity"] < 100 for c in comps)
    assert 85 <= comps[0]["similarity"] <= 95
    far_sim = next(c["similarity"] for c in comps
                   if c["player_id"] == "far")
    assert comps[0]["similarity"] - far_sim >= 10


def test_comparables_without_distributions_still_gated():
    pool = _comp_pool()
    computed = _computed(_comp_axes())
    jobs = {"hero": "WR2", "twin": "WR2", "alpha": "Alpha WR1",
            "far": "WR3"}
    comps, twin = pr.build_comparables("hero", pool, computed, jobs)
    ids = [c["player_id"] for c in comps]
    assert "alpha" not in ids
    assert ids[0] == "twin"
    assert twin["player_id"] == "twin"
    assert all(c["similarity"] <= 99 for c in comps)


# ---------------------------------------------------------------------------
# 9. Gap map is bars only: no league tick, no RB dot (JS/CSS source level)
# ---------------------------------------------------------------------------

CSS = Path(__file__).resolve().parent.parent / "static" / "dashboard.css"


def _roles_src():
    src = APP_JS.read_text(encoding="utf-8")
    roles_src = src[src.index("// Compare Roles tab:"):]
    return roles_src.split("function cmpSwitchTab")[0]


def test_gap_map_bars_only_no_tick_no_rb_dot():
    roles_src = _roles_src()
    css = CSS.read_text(encoding="utf-8")
    # The league-average tick on each lane bar is gone (the baseline
    # still lives in the tap-a-lane detail, asserted below).
    assert "compare-roles-lane-tick" not in roles_src
    assert "compare-roles-lane-tick" not in css
    assert "Dark tick" not in roles_src
    # The RB dot under the formation row is gone.
    assert "compare-roles-rb-dot" not in roles_src
    assert "compare-roles-rb-dot" not in css
    # The O-line formation row itself (LT LG C RG RT + TE chips) is
    # gone too: the gap map is lane bars, tendency and legend only.
    assert "compare-roles-formation" not in roles_src
    assert "compare-roles-formation" not in css
    assert "compare-roles-ol" not in roles_src
    assert "compare-roles-ol" not in css
    assert '"compare-roles-te"' not in roles_src
    assert ".compare-roles-te {" not in css
    assert ".compare-roles-te," not in css
    # League baseline is still available in the lane detail panel.
    assert "League baseline:" in roles_src
    assert "league_share" in roles_src


# ---------------------------------------------------------------------------
# 10. Zone and lane details are popovers, not inline panels
# ---------------------------------------------------------------------------

def test_role_details_open_as_popovers():
    roles_src = _roles_src()
    css = CSS.read_text(encoding="utf-8")
    # Both details render inside a hidden popover with a close button.
    assert 'class="compare-roles-pop" id="cmpRolePop-' in roles_src
    assert 'class="compare-roles-pop" id="cmpRoleLanePop-' in roles_src
    assert roles_src.count('class="compare-roles-pop-close"') == 2
    assert "function cmpRoleClosePop(" in roles_src
    assert "function _roleOpenPop(" in roles_src
    # The old inline detail panels are no longer emitted.
    assert 'class="compare-roles-detail" id="cmpRoleDetail-' not in roles_src
    assert 'class="compare-roles-detail" id="cmpRoleLaneDetail-' not in roles_src
    # Popover styling: floating overlay anchored to the map/lane box.
    pop_rule = css.split(".compare-roles-pop {")[1].split("}")[0]
    assert "position: absolute" in pop_rule
    assert "z-index: 30" in pop_rule
    assert ".compare-roles-pop-close {" in css
    # Nothing auto-fills a detail on paint any more: the popover opens
    # only from the click handlers.
    paint = roles_src.split("function _cmpRolesPaintMaps")[1]
    paint = paint.split("// Comparables block")[0]
    assert "_roleZoneDetailHTML" not in paint


# ---------------------------------------------------------------------------
# 11. Generated verdict line: payload facts only
# ---------------------------------------------------------------------------

def test_verdict_rising_with_tag_and_driver():
    line = pr.verdict_for("Alpha WR1", ["Target Hog"], "Rising", "Stable",
                          None, 31.6, 24.2)
    assert line == ("Alpha WR1 riding a target-hog role, and the last 4 "
                    "weeks strengthened it: opportunity share 32% over "
                    "the last 4 weeks vs 24% for the season.")


def test_verdict_contingent_names_the_lead():
    line = pr.verdict_for("Committee 1B", [], "Steady", "Contingent",
                          "Kenneth Walker III", None, None)
    assert line == ("Committee 1B whose production is contingent on "
                    "Kenneth Walker III's health.")


def test_verdict_no_change_signal_is_job_and_tag_only():
    line = pr.verdict_for("WR2", ["Possession"], "Steady", "Stable",
                          None, 18.0, 17.5)
    assert line == "WR2 riding a possession role."


def test_verdict_fading_and_security_fallbacks():
    line = pr.verdict_for("Lead 1A", [], "Fading", "Stable", None,
                          30.0, 41.0)
    assert line == ("Lead 1A, but the last 4 weeks cooled it: opportunity "
                    "share 30% over the last 4 weeks vs 41% for the "
                    "season.")
    assert pr.verdict_for("Workhorse", [], "Steady", "Locked", None,
                          None, None) == "Workhorse, with the job locked in."
    assert pr.verdict_for("Depth", [], "Steady", "Volatile", None,
                          None, None) == (
        "Depth, though the week-to-week role has been volatile.")
    # Contingent without a named lead falls through to the plain line.
    assert pr.verdict_for("Committee 1B", [], "Steady", "Contingent",
                          None, None, None) == "Committee 1B."


def test_payload_role_carries_verdict():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "rb1": {"pos": "RB", "team": "BUF", "name": "Test Back",
                "series": [_series_row(w, snap_pct=65.0, carries=14,
                                       carry_share=50.0, rush_yards=65.0,
                                       ppr_pts=13.0, rz_carries=2)
                           for w in weeks]},
        "rb2": {"pos": "RB", "team": "MIA", "name": "Other Back",
                "series": [_series_row(w, snap_pct=45.0, carries=8,
                                       carry_share=30.0, rush_yards=34.0,
                                       ppr_pts=7.0, rz_carries=1)
                           for w in weeks]},
    }
    payload = pr.build_player_payload("rb1", 2026, "season", pool=pool,
                                      tables=_carry_tables("rb1"),
                                      pa_feats={})
    verdict = payload["role"]["verdict"]
    assert verdict.startswith(payload["role"]["job"])
    assert verdict.endswith(".")


# ---------------------------------------------------------------------------
# 12. Thin-volume guard flags in the payload
# ---------------------------------------------------------------------------

def test_gap_map_thin_flag_and_volume():
    payload_tables = _carry_tables("rb1")
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "rb1": {"pos": "RB", "team": "BUF", "name": "Test Back",
                "series": [_series_row(w, snap_pct=65.0, carries=14,
                                       carry_share=50.0, rush_yards=65.0,
                                       ppr_pts=13.0, rz_carries=2)
                           for w in weeks]},
    }
    payload = pr.build_player_payload("rb1", 2026, "season", pool=pool,
                                      tables=payload_tables, pa_feats={})
    # The fixture lane holds 6 carries: under the 15-carry floor.
    assert payload["gap_map"]["volume"] == 6
    assert payload["gap_map"]["thin"] is True


def test_gap_map_not_thin_at_volume():
    lane = {"carries": 22, "yards": 99.0, "td": 1, "first_downs": 5,
            "stuffed": 2, "explosive": 2, "success": 11, "longest": 18,
            "carries_last4": 22, "carries_base": 22}
    tables = _carry_tables("rb1")
    tables["rushing"]["lanes"]["rb1"] = {"M": dict(lane)}
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        "rb1": {"pos": "RB", "team": "BUF", "name": "Test Back",
                "series": [_series_row(w, snap_pct=65.0, carries=14,
                                       carry_share=50.0, rush_yards=65.0,
                                       ppr_pts=13.0, rz_carries=2)
                           for w in weeks]},
    }
    payload = pr.build_player_payload("rb1", 2026, "season", pool=pool,
                                      tables=tables, pa_feats={})
    assert payload["gap_map"]["volume"] == 22
    assert payload["gap_map"]["thin"] is False


def _wr_pool(pid):
    weeks = [1, 2, 3, 4, 5, 6]
    return {
        pid: {"pos": "WR", "team": "TB", "name": "Test Receiver",
              "series": [_series_row(w, snap_pct=85.0, targets=8,
                                     receptions=5, target_share=22.0,
                                     rec_yards=70.0, ppr_pts=12.0)
                         for w in weeks]},
        "wr2": {"pos": "WR", "team": "NYG", "name": "Other Receiver",
                "series": [_series_row(w, snap_pct=70.0, targets=5,
                                       receptions=3, target_share=14.0,
                                       rec_yards=40.0, ppr_pts=7.0)
                           for w in weeks]},
    }


def test_target_map_thin_flag_and_volume():
    rows = [_pass_row(wk, "wr1", "right", 5) for wk in (1, 2, 3, 4, 5)]
    weeks = set(range(1, 11))
    tables = pr.build_pbp_tables_from_rows(
        rows, {}, last4_weeks={7, 8, 9, 10}, base_weeks=weeks,
        window_weeks=weeks, window="season", player_weeks=True)
    payload = pr.build_player_payload("wr1", 2026, "season",
                                      pool=_wr_pool("wr1"), tables=tables,
                                      pa_feats={})
    assert payload["target_map"]["volume"] == 5
    assert payload["target_map"]["thin"] is True


def test_target_map_not_thin_at_volume():
    rows = [_pass_row(wk, "wr1", "right", 5) for wk in range(1, 13)]
    weeks = set(range(1, 13))
    tables = pr.build_pbp_tables_from_rows(
        rows, {}, last4_weeks={9, 10, 11, 12}, base_weeks=weeks,
        window_weeks=weeks, window="season", player_weeks=True)
    payload = pr.build_player_payload("wr1", 2026, "season",
                                      pool=_wr_pool("wr1"), tables=tables,
                                      pa_feats={})
    assert payload["target_map"]["volume"] == 12
    assert payload["target_map"]["thin"] is False


# ---------------------------------------------------------------------------
# 13. Follow-up scope: prefetch, tappable comps, skeleton, styling
# (JS/CSS source level)
# ---------------------------------------------------------------------------

PM_JS = (Path(__file__).resolve().parent.parent / "static"
         / "player_modal.js")


def test_modal_prefetches_role_payload_on_open():
    src = PM_JS.read_text(encoding="utf-8")
    early = src.split("function _pmStartEarlyFetches")[1]
    early = early.split("\nfunction ")[0]
    # The prefetch uses the exact key pmRoleLoad consumes.
    assert "'role:' + playerId + ':' + season + ':season'" in early
    assert "/api/player-role/" in early
    # And the Role tab's loading state is the card skeleton.
    assert "_cmpRoleSkeletonHTML" in src


def test_modal_comp_swap_stays_on_role():
    src = PM_JS.read_text(encoding="utf-8")
    assert "tab: 'metrics', roleSub: true" in src
    assert "opts.roleSub" in src
    assert "pmAdvSetSub('role')" in src


def test_compare_comps_are_tappable_buttons():
    roles_src = _roles_src()
    # Rows and the twin are buttons wired to the slot loader, not the
    # old deep-link anchors.
    assert '<a class="compare-roles-comp" href=' not in roles_src
    assert "function cmpRoleOpenComp(" in roles_src
    assert "compare-roles-twin-btn" in roles_src
    assert "window._cmpSetSlot" in roles_src  # consumed by the handler
    # The compare picker exposes the slot loader.
    src = APP_JS.read_text(encoding="utf-8")
    assert "window._cmpSetSlot = function" in src


def test_skeleton_markup_in_compare_loader():
    roles_src = _roles_src()
    assert "function _cmpRoleSkeletonHTML(" in roles_src
    assert "compare-roles-skel" in roles_src
    assert "skeleton skeleton-line" in roles_src
    assert "compare-roles-skel-map" in roles_src


def test_style_scope_css():
    css = CSS.read_text(encoding="utf-8")
    # Similarity %: navy text colour (var(--text)), tabular, name weight.
    sim_rule = css.split(".compare-roles-comp-sim {")[1].split("}")[0]
    assert "color: var(--text)" in sim_rule
    assert "font-variant-numeric: tabular-nums" in sim_rule
    assert "font-weight: 700" in sim_rule
    assert "#2563eb" not in sim_rule
    # Job headline steps up; evidence values step down to slate.
    job_rule = css.split(".compare-roles-job {")[1].split("}")[0]
    assert "font-size: 26px" in job_rule
    assert "font-weight: 900" in job_rule
    ev_rule = css.split(".compare-roles-ev-val {")[1].split("}")[0]
    assert "font-size: 13px" in ev_rule
    assert "#64748b" in ev_rule
    # Thin maps mute their surfaces.
    assert ".compare-roles-thin canvas.compare-roles-canvas" in css
    assert ".compare-roles-thin .compare-roles-lane-track i" in css
    # Popover: site dropdown shadow, entrance animation with a
    # reduced-motion opt-out, caret only at >=640px.
    pop_rule = css.split(".compare-roles-pop {")[1].split("}")[0]
    assert "box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15)" in pop_rule
    assert "compare-roles-pop-in" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    caret_at = css.index(".compare-roles-pop::before")
    assert "@media (min-width: 640px)" in css[max(0, caret_at - 200):caret_at]
    # Timeline: current segment is the tallest; change labels exist.
    assert ".compare-roles-tl-seg.is-cur" in css
    assert ".compare-roles-tl-change" in css


def test_thin_note_and_verdict_render_and_field_furniture():
    roles_src = _roles_src()
    assert "compare-roles-thinnote" in roles_src
    assert "Thin volume:" in roles_src
    assert "role.verdict" in roles_src
    assert "compare-roles-cardverdict" in roles_src
    assert "compare-roles-tl-change" in roles_src
    # Field furniture (hash columns) is painted before the heat image.
    assert roles_src.index("hash columns") < roles_src.index(
        "drawImage(oc")


# ---------------------------------------------------------------------------
# 14. Modal header meta line carries the role Job (season window only)
# ---------------------------------------------------------------------------

def test_modal_header_job_from_role_payload():
    src = PM_JS.read_text(encoding="utf-8")
    # The open-time prefetch notes the job for the header.
    early = src.split("function _pmStartEarlyFetches")[1]
    early = early.split("\nfunction ")[0]
    assert "_pmRoleNoteHeaderJob(playerId, d)" in early
    # The note helper reads the job and nothing else from the role
    # block: no Security / Trend / Confidence in the header.
    note = src.split("function _pmRoleNoteHeaderJob(")[1]
    note = note.split("\nfunction ")[0]
    assert "d.role.job" in note
    assert "security" not in note
    assert "confidence" not in note
    assert ".trend" not in note
    # The painter keys off the modal's current player (a stale payload
    # for a previous player can never paint) and renders a plain span:
    # no pill/badge classes, no inline styling on the job element.
    paint = src.split("function _pmRolePaintHeaderJob(")[1]
    paint = paint.split("\nfunction ")[0]
    assert "pmPlayerId" in paint
    assert "el.className = 'pm-role-job'" in paint
    assert "el.setAttribute('style'" not in paint
    # A job that landed before the header render is painted by the
    # meta build itself.
    assert "_pmRolePaintHeaderJob();" in src
    # Role-tab loads for non-season windows never touch the header.
    load = src.split("function pmRoleLoad(")[1]
    load = load.split("\nfunction ")[0]
    assert "_pmRoleState.win === 'season'" in load
    assert "_pmRoleNoteHeaderJob(pid, d)" in load


# ---------------------------------------------------------------------------
# 15. Summary strips carry a stat-count class; labels never wrap;
#     popovers grow to full height instead of scrolling internally;
#     the header job never doubles its separator.
# ---------------------------------------------------------------------------

CSS = Path(__file__).resolve().parent.parent / "static" / "dashboard.css"


def _css_rule(css, selector):
    i = css.index(selector)
    return css[i:css.index("}", i)]


def test_summary_strips_emit_count_class():
    src = APP_JS.read_text(encoding="utf-8")
    assert "function _cmpRoleSummaryHTML(stats)" in src
    assert "compare-roles-summary cr-n' + stats.length" in src
    # All three strips (map, gap, QB rushing) route through the helper.
    assert src.count("_cmpRoleSummaryHTML(") >= 4  # helper + 3 call sites
    assert 'class="compare-roles-summary">' not in src


def test_role_labels_nowrap_and_phone_grid():
    css = CSS.read_text(encoding="utf-8")
    assert "white-space: nowrap" in _css_rule(css, ".compare-roles-stat span {")
    assert "white-space: nowrap" in _css_rule(css, ".compare-roles-dkv span {")
    phone = css[css.index("@media (max-width: 640px)"):]
    assert ".compare-roles-summary.cr-n5, .compare-roles-summary.cr-n6" in phone
    assert "grid-template-columns: repeat(3, 1fr)" in phone
    assert ".compare-roles-dgrid { grid-template-columns: repeat(3, 1fr); }" in phone


def test_popover_has_no_internal_scroll_cap():
    css = CSS.read_text(encoding="utf-8")
    rule = _css_rule(css, ".compare-roles-pop {")
    assert "max-height" not in rule
    assert "overflow" not in rule


def test_header_job_separator_is_single():
    src = PM_JS.read_text(encoding="utf-8")
    paint = src.split("function _pmRolePaintHeaderJob(")[1]
    paint = paint.split("\nfunction ")[0]
    # The value-trend pill is gone from the header, and with it the
    # pill-relative insertion branch: the job is simply appended at
    # the end of the meta row behind exactly one fresh separator
    # (the #3200 double-separator fix, preserved without the pill).
    assert "pm-trend-pill" not in paint
    assert "insertBefore" not in paint
    assert "row.appendChild(mkSep()); row.appendChild(el);" in paint


def test_header_meta_has_no_value_trend_pill():
    src = PM_JS.read_text(encoding="utf-8")
    # The value-trend pill ("Recovering", "Declining", ...) never
    # renders into the modal header again: neither the details render
    # nor the deferred patch-in may emit its markup, in any state.
    assert "pm-trend-pill" not in src
    assert "data-trend-tip" not in src
    # The trend state still arrives inside the deferred payload (the
    # trend computation feeds other surfaces); the modal just never
    # renders it in the header.
    assert "data.value_trend = dfd.value_trend || null;" in src
    # And the role Job still paints, at the end of the meta line.
    paint = src.split("function _pmRolePaintHeaderJob(")[1]
    paint = paint.split("\nfunction ")[0]
    assert "el.className = 'pm-role-job'" in paint
    assert "row.appendChild(mkSep()); row.appendChild(el);" in paint


# ---------------------------------------------------------------------------
# 16. Header Job paint ordering: the tab bar is bound to the player
#     when the modal DOM is built, not when the details payload lands.
#     The painter is guarded on dataset.pmPlayerId, and both paint
#     triggers (the role payload landing, the meta render) run inside
#     the same synchronous details callback that used to be the FIRST
#     place the dataset was assigned — so with a warm role endpoint
#     (or the session small-fetch cache resolving on a microtask) the
#     guard failed at both triggers and the Job never painted.
#     No JS harness exercises the modal's async ordering, so this pins
#     the source invariant both payload orders depend on.
# ---------------------------------------------------------------------------

def test_header_job_identity_bound_before_any_paint_trigger():
    src = PM_JS.read_text(encoding="utf-8")
    open_fn = src.split("function openPlayerModal(")[1]
    # The early bind lives in openPlayerModal, on the freshly built
    # tab bar, after the modal is in the document.
    early = "_pmBarEarly.dataset.pmPlayerId = String(playerId);"
    assert early in open_fn
    assert open_fn.index("document.body.appendChild(overlay)") \
        < open_fn.index(early)
    # It precedes every paint trigger: the payload-landing note (fired
    # from the early fetches kicked off in the same function) and the
    # meta render's paint call in the details callback.
    assert open_fn.index(early) < open_fn.index("_pmRolePaintHeaderJob();")
    # It also precedes the details callback's own dataset assignment,
    # which remains (it re-binds the same identity alongside season /
    # name / position).
    assert open_fn.index(early) \
        < open_fn.index("pmTabBar.dataset.pmPlayerId = playerId;")
    # Belt-and-braces: immediately after that dataset block, the
    # painter runs again, covering a payload noted mid-callback.
    block = open_fn.split("pmTabBar.dataset.pmPlayerId = playerId;")[1]
    block = block.split("const _teamPanel")[0]
    assert "_pmRolePaintHeaderJob();" in block
    # The stale-player protection is untouched: the painter still
    # refuses to paint when the noted job belongs to another player.
    paint = src.split("function _pmRolePaintHeaderJob(")[1]
    paint = paint.split("\nfunction ")[0]
    assert "_pmRoleHeaderJob.pid !== String(pid)" in paint
