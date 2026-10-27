"""Player modal Advanced tab TRENDING hot/cold section.

The TRENDING section surfaces only metrics with a significant recent trend at
the top of the Advanced tab, reusing the weekly series and sparkline renderer.
Quiet metrics stay as plain bars below.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_node_required = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available")


def _compute_trending(weeks, position):
    """Run the real pmComputeTrending from player_modal.js under node."""
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    start = js.index("Player modal: TRENDING hot/cold section")
    start = js.rindex("//", 0, start)
    end = js.index("// Fire-and-forget kick", start)
    src = js[start:end]
    driver = (
        src
        + "\nconst WEEKS = " + json.dumps(weeks) + ";\n"
        + "const out = pmComputeTrending(WEEKS, " + json.dumps(position) + ");\n"
        + "process.stdout.write(JSON.stringify(out.map(function(t) { return {\n"
        + "  key: t.key, kind: t.kind, pct: t.pct, baseline: t.baseline,\n"
        + "  recentAvg: t.recentAvg, delta: t.delta, cluster: t.cluster\n"
        + "}; })));\n"
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True,
                         text=True, timeout=20, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _wk(**kw):
    base = {"week": 1, "opponent": "BUF", "snap_pct": 0, "touches": 0,
            "carries": 0, "carry_share": 0, "rush_yards": 0, "targets": 0,
            "receptions": 0, "rec_yards": 0, "rz_targets": 0, "rz_carries": 0,
            "ppr_pts": 0, "target_share": 0, "pass_att": 0}
    base.update(kw)
    return base


def test_trending_placeholder_at_top_of_advanced_metrics():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Placeholder div is rendered before the metric bars (prepended to rankNote).
    assert 'id="pmTrendingSection"' in js
    assert "'<div id=\"pmTrendingSection\"></div>' + rankNote" in js


def test_trending_uses_weekly_series_and_sparkline():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function pmComputeTrending(weeks, position)" in js
    assert "function pmRenderTrendingSection(playerId, season, position)" in js
    # Reuses the existing sparkline renderer and weekly-metrics endpoint.
    assert "pmSparkline(t.series," in js
    assert "/api/player-weekly-metrics/" in js


def test_trending_is_position_aware():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function _pmTrendMetricSets(pos)" in js
    assert "p === 'RB'" in js
    assert "p === 'WR' || p === 'TE'" in js
    assert "p === 'QB'" in js


def test_trending_threshold_and_limit():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Significant = |3-wk avg vs season avg| >= 15% on an established
    # baseline, top 4 after the per-cluster cap.
    assert "_PM_TREND_PCT = 0.15" in js
    assert "if (picked.length >= 4) return;" in js
    # Needs at least 4 weeks of data (recent-3 vs prior weeks).
    assert "_PM_TREND_MIN_WEEKS = 4" in js


def test_trending_hot_cold_badges():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "HEATING UP" in js
    assert "COOLING OFF" in js
    assert "pm-trend-badge" in js
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".pm-trend-row.hot" in css
    assert ".pm-trend-row.cold" in css


def test_trending_kicked_after_metrics_render():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function _pmKickTrending(" in js
    # Only single-season views get a trending section.
    assert "if (isCareer || isMultiSeason || !activeSeason) return;" in js
    assert "_pmKickTrending(playerId, metricsData, activeSeason, isCareer, isMultiSeason)" in js


def test_trending_shares_weekly_data_with_trends_panel():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Prefetch stores the series on the trends wrap so opening Trends later
    # does not refetch.
    assert "wrap._weeklyData = d.weeks;" in js
    assert "wrap._weeklyDataSeason" in js
    # Season change clears the cached series.
    assert "wtWrap._weeklyData = null;" in js


def test_trending_css_single_verdict_card():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    # Option A: one verdict card, full width at every size (no 2x2 card grid).
    assert ".pm-trend-hero" in css
    assert ".pm-trend-verdict-title" in css
    assert ".pm-trend-group-head" in css
    assert ".pm-trend-role" in css
    assert ".pm-trending-grid" not in css


def test_trending_option_a_view_in_js():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "function pmBuildTrendingView(trending, weeks, position)" in js
    assert "function pmComputeRoleAnchor(weeks, position, moverKeys)" in js
    assert "Last 3 vs prior" in js
    assert "Role check" in js
    # The badge/verdict is stated once, in the verdict block, not per card.
    assert "pm-trend-verdict" in js
    # Honest labels: catch rate is not a drops stat, and no mock annotation
    # text ships in the UI.
    assert "Drops creeping in" not in js
    assert "(not /g)" not in js
    assert "Completion rate down" in js


def test_trending_copy_has_no_em_dashes():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    # Callouts use middots, never em dashes. Scope to the trending block.
    start = js.index("Player modal: TRENDING hot/cold section")
    end = js.index("Collapse/expand a section in the player compare view", start)
    block = js[start:end]
    assert "\u2014" not in block


def test_season_week_selectors_have_no_card():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    # The .adv-time-ctl container must not render as a card (no background,
    # border, or padding wrapping the season/week selectors).
    import re
    m = re.search(r"\.adv-time-ctl\s*\{([^}]*)\}", css)
    assert m, ".adv-time-ctl rule missing"
    body = m.group(1)
    assert "background" not in body
    assert "border" not in body or "border-" in body  # allow border-* subprops? no: strict
    assert "border:" not in body
    assert "padding" not in body


def test_season_pills_single_line_rail():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert "flex-wrap: nowrap" in css
    assert "overflow-x: auto" in css
    assert "-webkit-overflow-scrolling: touch" in css


# ── Trend honesty: no percentages off near-zero baselines ────────────────

@_node_required
def test_backup_takeover_is_new_role_not_4405_percent():
    # Vidal-shaped: 4 prior weeks near zero, last 3 as the starter.
    weeks = [
        _wk(snap_pct=8, touches=1, carries=1, carry_share=5, rush_yards=3,
            targets=0, receptions=0, rec_yards=0, ppr_pts=0.4),
        _wk(snap_pct=10, touches=2, carries=1, carry_share=6, rush_yards=4,
            targets=1, receptions=0, rec_yards=2, ppr_pts=0.3),
        _wk(snap_pct=9, touches=1.5, carries=2, carry_share=5, rush_yards=3.5,
            targets=0, receptions=1, rec_yards=0, ppr_pts=0.35),
        _wk(snap_pct=9, touches=1.5, carries=1, carry_share=5.6, rush_yards=3.5,
            targets=1, receptions=0, rec_yards=1, ppr_pts=0.35),
        _wk(snap_pct=62, touches=15, carries=13, carry_share=50, rush_yards=70,
            targets=3, receptions=2, rec_yards=10, rz_carries=1, ppr_pts=14.9),
        _wk(snap_pct=68, touches=17, carries=15, carry_share=54, rush_yards=80,
            targets=4, receptions=3, rec_yards=14, rz_carries=2, ppr_pts=16.2),
        _wk(snap_pct=65, touches=17, carries=14, carry_share=53.8, rush_yards=79,
            targets=3, receptions=2, rec_yards=12, rz_carries=1, ppr_pts=16.3),
    ]
    out = _compute_trending(weeks, "RB")
    by_key = {t["key"]: t for t in out}
    ppr = by_key["ppr_pts"]
    assert ppr["kind"] == "newrole"
    assert ppr["pct"] is None
    assert abs(ppr["baseline"] - 0.35) < 0.01
    assert abs(ppr["recentAvg"] - 15.8) < 0.01
    assert by_key["carry_share"]["kind"] == "newrole"
    # No percentage anywhere in the output may be the near-zero artifact:
    # every pct is either absent or a sane established-baseline move.
    for t in out:
        assert t["pct"] is None or abs(t["pct"]) <= 2.0, t


@_node_required
def test_established_baseline_riser_keeps_percent():
    # Targets 5.75/g prior (4 active weeks) -> 8.33/g last 3: a real trend.
    weeks = [
        _wk(snap_pct=70, targets=6, target_share=20, receptions=4,
            rec_yards=45, touches=6, ppr_pts=10.5),
        _wk(snap_pct=72, targets=5, target_share=19, receptions=3,
            rec_yards=38, touches=5, ppr_pts=9.3),
        _wk(snap_pct=71, targets=6, target_share=20, receptions=4,
            rec_yards=50, touches=6, ppr_pts=11.0),
        _wk(snap_pct=70, targets=6, target_share=21, receptions=4,
            rec_yards=44, touches=6, ppr_pts=10.4),
        _wk(snap_pct=74, targets=8, target_share=27, receptions=5,
            rec_yards=52, touches=8, ppr_pts=11.2),
        _wk(snap_pct=75, targets=9, target_share=29, receptions=6,
            rec_yards=58, touches=9, ppr_pts=12.8),
        _wk(snap_pct=74, targets=8, target_share=28, receptions=5,
            rec_yards=55, touches=8, ppr_pts=11.5),
    ]
    out = _compute_trending(weeks, "WR")
    by_key = {t["key"]: t for t in out}
    tgt = by_key["targets"]
    assert tgt["kind"] == "hot"
    assert tgt["pct"] is not None and abs(tgt["pct"] - 0.449) < 0.03
    assert abs(tgt["delta"] - 2.58) < 0.05


@_node_required
def test_single_active_prior_week_cannot_anchor_a_percent():
    # Baseline PPR avg clears the floor only because ONE prior week was a
    # real game; the other three prior weeks the player never saw the field.
    weeks = [
        _wk(),
        _wk(),
        _wk(),
        _wk(snap_pct=55, touches=10, carries=8, ppr_pts=12),
        _wk(snap_pct=60, touches=12, carries=10, ppr_pts=15),
        _wk(snap_pct=62, touches=13, carries=10, ppr_pts=16),
        _wk(snap_pct=61, touches=12, carries=11, ppr_pts=17),
    ]
    out = _compute_trending(weeks, "RB")
    by_key = {t["key"]: t for t in out}
    assert by_key["ppr_pts"]["kind"] == "newrole"
    assert by_key["ppr_pts"]["pct"] is None


@_node_required
def test_cluster_cap_limits_correlated_volume_metrics():
    # All four volume metrics qualify hot off real baselines, plus two
    # production metrics and carry share: volume may take at most 2 slots.
    weeks = [
        _wk(snap_pct=60, touches=10, carries=8, carry_share=45, rush_yards=45,
            targets=4, receptions=3, rec_yards=25, ppr_pts=11),
        _wk(snap_pct=61, touches=11, carries=8, carry_share=46, rush_yards=48,
            targets=4, receptions=3, rec_yards=28, ppr_pts=11.5),
        _wk(snap_pct=60, touches=10, carries=9, carry_share=44, rush_yards=44,
            targets=5, receptions=4, rec_yards=30, ppr_pts=11),
        _wk(snap_pct=62, touches=11, carries=8, carry_share=45, rush_yards=47,
            targets=4, receptions=3, rec_yards=26, ppr_pts=11.5),
        _wk(snap_pct=63, touches=16, carries=13, carry_share=58, rush_yards=78,
            targets=7, receptions=6, rec_yards=48, ppr_pts=17),
        _wk(snap_pct=64, touches=17, carries=13, carry_share=60, rush_yards=82,
            targets=7, receptions=6, rec_yards=50, ppr_pts=18),
        _wk(snap_pct=63, touches=16, carries=14, carry_share=59, rush_yards=80,
            targets=8, receptions=5, rec_yards=47, ppr_pts=17.5),
    ]
    out = _compute_trending(weeks, "RB")
    assert len(out) == 4
    volume = [t for t in out if t["cluster"] == "volume"]
    assert len(volume) <= 2


@_node_required
def test_quiet_player_yields_no_trending_cards():
    flat = [
        _wk(snap_pct=60, touches=12, carries=9, carry_share=45, rush_yards=50,
            targets=4, receptions=3, rec_yards=28, ppr_pts=11.5)
        for _ in range(7)
    ]
    assert _compute_trending(flat, "RB") == []


def _build_view(weeks, position):
    """Run the real pmBuildTrendingView from player_modal.js under node."""
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    start = js.index("Player modal: TRENDING hot/cold section")
    start = js.rindex("//", 0, start)
    end = js.index("// Fire-and-forget kick", start)
    src = js[start:end]
    driver = (
        src
        + "\nconst WEEKS = " + json.dumps(weeks) + ";\n"
        + "const trending = pmComputeTrending(WEEKS, " + json.dumps(position) + ");\n"
        + "const view = pmBuildTrendingView(trending, WEEKS, " + json.dumps(position) + ");\n"
        + "const byKey = {}; trending.forEach(function(t) { byKey[t.key] = t; });\n"
        + "process.stdout.write(JSON.stringify({\n"
        + "  hero: view.hero.key, title: view.title, sub: view.sub,\n"
        + "  groups: view.groups.map(function(g) { return {label: g.label, keys: g.items.map(function(t) { return t.key; })}; }),\n"
        + "  anchor: view.roleAnchor.map(function(a) { return a.key; }),\n"
        + "  yptDelta: _pmTrendFmtDeltaFull(byKey['yds_per_target']),\n"
        + "  yptBase: _pmTrendFmtBase(byKey['yds_per_target']),\n"
        + "  pprDelta: _pmTrendFmtDeltaFull(byKey['ppr_pts'])\n"
        + "}));\n"
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True,
                         text=True, timeout=20, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@_node_required
def test_option_a_lamb_shape_is_one_efficiency_led_story():
    # Lamb-shaped: flat targets/share/snaps, catch rate and per-target
    # efficiency collapse, yardage and PPR follow.
    prior = dict(snap_pct=90, targets=12, target_share=30, receptions=10,
                 rec_yards=140, touches=12, ppr_pts=28)
    recent = dict(snap_pct=90, targets=12, target_share=30, receptions=6,
                  rec_yards=50, touches=12, ppr_pts=14)
    weeks = [_wk(**prior) for _ in range(4)] + [_wk(**recent) for _ in range(3)]
    view = _build_view(weeks, "WR")
    # PPR is the hero, stated once; efficiency is the cause group and the
    # remaining production mover is the effect, not two more alarms.
    assert view["hero"] == "ppr_pts"
    assert view["title"] == "Cooling off: efficiency-led dip"
    assert view["sub"] == "Efficiency fell first. Production followed."
    labels = {g["label"]: g["keys"] for g in view["groups"]}
    assert set(labels["EFFICIENCY · THE CAUSE"]) == {"catch_pct", "yds_per_target"}
    assert labels["PRODUCTION · THE EFFECT"] == ["rec_yards"]
    # Flat role is shown as an anchor, not hidden.
    assert set(view["anchor"]) == {"targets", "target_share", "snap_pct"}
    # Rates carry no per-game unit; per-game metrics keep theirs.
    assert "/g" not in view["yptDelta"]
    assert "/g" not in view["yptBase"]
    assert "/g" in view["pprDelta"]


def test_trending_new_role_rendering_and_css():
    js = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
    assert "NEW ROLE" in js
    assert "_PM_TREND_FLOORS" in js
    assert "pm-trend-base" in js
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".pm-trend-row.newrole" in css
    assert ".pm-trend-badge.newrole" in css
    assert ".pm-trend-delta.newrole" in css
    assert ".pm-trend-base" in css
