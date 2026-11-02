"""Player modal Overview: Recent Form game-stat tooltips + Overview speed.

- Each Recent Form column carries a tooltip with that game's basic stats,
  built by _pmRecentGameTip(g, oppText) from g.stats / g.adv: header and
  points, then Pass / Rush / Rec segments only when their data exists,
  snap only when adv is present, and TD / INT / fumble segments only
  when greater than zero. No completions are ever shown (there is no
  such field).
- _pmStartEarlyFetches fires the news small-fetch at click with exactly
  the key and URL _pmLoadNews uses, so the loader adopts it.
- The speculative warmer and _pmStartEarlyFetches share the
  _pmRecentFormUrl helper, so their Recent Form URLs cannot drift.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


def _loader_body() -> str:
    start = JS.index("function _pmLoadRecentForm")
    end = JS.index("function pmInjectContextActions")
    return JS[start:end]


def _early_body() -> str:
    start = JS.index("function _pmStartEarlyFetches")
    end = JS.index("// Shared, bounded player-details request layer")
    return JS[start:end]


def _tip_src() -> str:
    start = JS.index("function _pmRecentGameTip")
    end = JS.index("function _pmLoadRecentForm")
    return JS[start:end]


def _url_src() -> str:
    start = JS.index("function _pmRecentFormUrl")
    end = JS.index("function _pmStartEarlyFetches")
    return JS[start:end]


# ── (a) Column markup ────────────────────────────────────────────────────────

def test_recent_form_column_carries_tooltip_tabindex_and_aria_label():
    body = _loader_body()
    assert "const _tip = _pmEsc(_pmRecentGameTip(g, _opp));" in body
    assert (
        '\'<div class="pm-recent-col" tabindex="0" data-tooltip="\' + _tip'
        ' + \'" aria-label="\' + _tip + \'">\''
    ) in body or (
        'class="pm-recent-col" tabindex="0" data-tooltip="' in body
        and 'aria-label="' in body
    )
    # Plain div, not an interactive element: the global tooltip
    # positioner's tap-to-toggle path ignores A/BUTTON/INPUT/etc.
    assert '<button class="pm-recent-col"' not in body
    assert '<a class="pm-recent-col"' not in body


def test_tooltip_is_recent_form_only():
    # Skeleton columns get no tooltip.
    skel = JS[JS.index("function _pmRecentFormSkeleton()"):JS.index("function _pmRecentGameTip")]
    assert "data-tooltip" not in skel
    # Usage bars (Team tab) build .pm-recent-col markup too: no tooltip.
    # Their column return is the only remaining bare .pm-recent-col
    # return in the file (the skeleton builds its columns without a
    # return, and the Recent Form return now carries the tooltip).
    bare = JS.count("return '<div class=\"pm-recent-col\">'")
    assert bare == 1, "only the usage-bars column should stay a bare .pm-recent-col"


# ── (c) Early news fetch ─────────────────────────────────────────────────────

def test_start_early_fetches_fires_news_with_loader_key_and_url():
    body = _early_body()
    assert "_pmSmallFetch('news:' + playerId, '/api/player-news/' + encodeURIComponent(playerId));" in body
    # The loader uses the same key (its URL is the template-literal form
    # of the same string).
    assert "_pmSmallFetch('news:' + playerId" in JS
    assert "/api/player-news/${encodeURIComponent(playerId)}" in JS


# ── (d) Shared Recent Form URL helper ────────────────────────────────────────

def test_recent_form_url_helper_shared_by_early_fetch_and_warmer():
    assert "function _pmRecentFormUrl(playerId, season)" in JS
    # Click-time early fetch uses the helper.
    assert "_pmEarlyResponseFetch(_pmRecentFormUrl(playerId, season));" in _early_body()
    # Warmer pump uses the helper alongside the speculative details load.
    pump = JS[JS.index("function pump(s)"):JS.index("function enqueue")]
    assert "_pmEarlyResponseFetch(_pmRecentFormUrl(x.id" in pump
    assert "window.pmPlayerDetails.load(x.id,x.opts)" in pump
    # Helper builds the loader's exact URL shape.
    src = _url_src()
    assert "'/api/player-recent-form/' + encodeURIComponent(playerId)" in src
    assert "'?season=' + _season + '&league_type=' + _lt + '&league_size=' + _ls" in src
    assert "rfUrl += '&league_id=' + _leagueId + '&platform=' + (pathParts[0] || 'sleeper')" in src
    # Warmer does not warm news speculatively.
    assert "news" not in pump


# ── (b) Tip line builder, executed in Node ───────────────────────────────────

pytestmark_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def _run_tips(cases):
    """cases: list of [game, oppText]; returns list of tip strings."""
    driver = (
        _tip_src()
        + "\nconst cases = " + json.dumps(cases) + ";\n"
        + "process.stdout.write(JSON.stringify(cases.map(c => _pmRecentGameTip(c[0], c[1]))));\n"
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytestmark_node
def test_tip_qb_pass_line_with_td_int_and_snap():
    (tip,) = _run_tips([[
        {"week": 5, "fantasy_pts": 28.4,
         "stats": {"pass_att": 35, "pass_yd": 300, "pass_td": 3, "pass_int": 1,
                   "rush_att": 5, "rush_yd": 42, "rush_td": 1, "fum_lost": 0},
         "adv": {"snap_pct": 98.4}},
        "@ LV",
    ]])
    assert tip == "Week 5 @ LV · 28.4 pts · Pass 35 att, 300 yd, 3 TD, 1 INT · Rush 5 att, 42 yd, 1 TD · Snap 98%"
    assert "comp" not in tip.lower()


@pytestmark_node
def test_tip_zero_td_int_and_rush_td_are_omitted():
    (tip,) = _run_tips([[
        {"week": 6, "fantasy_pts": 8.5,
         "stats": {"pass_att": 20, "pass_yd": 100, "pass_td": 0, "pass_int": 0,
                   "rush_att": 1, "rush_yd": 5, "rush_td": 0},
         "adv": None},
        "vs MIA",
    ]])
    assert tip == "Week 6 vs MIA · 8.5 pts · Pass 20 att, 100 yd · Rush 1 att, 5 yd"
    assert "TD" not in tip and "INT" not in tip and "Snap" not in tip


@pytestmark_node
def test_tip_rb_rush_and_rec_plus_fumble():
    (tip,) = _run_tips([[
        {"week": 3, "fantasy_pts": 18.2,
         "stats": {"rush_att": 18, "rush_yd": 92, "rush_td": 1,
                   "rec": 4, "rec_tgt": 6, "rec_yd": 31, "rec_td": 0, "fum_lost": 1},
         "adv": {"snap_pct": 72}},
        "vs KC",
    ]])
    assert tip == "Week 3 vs KC · 18.2 pts · Rush 18 att, 92 yd, 1 TD · Rec 4 of 6 tgt, 31 yd · Snap 72% · 1 fum lost"


@pytestmark_node
def test_tip_wr_rec_without_targets_and_no_opponent():
    (tip,) = _run_tips([[
        {"week": 4, "fantasy_pts": 12.0,
         "stats": {"rec": 5, "rec_tgt": None, "rec_yd": 68, "rec_td": 1}},
        "",
    ]])
    assert tip == "Week 4 · 12.0 pts · Rec 5, 68 yd, 1 TD"


@pytestmark_node
def test_tip_fallback_is_header_and_points_only():
    (tip,) = _run_tips([[
        {"week": 2, "fantasy_pts": 3.0,
         "stats": {"pass_yd": None, "pass_att": None, "rush_att": None,
                   "rec": None, "rec_tgt": None}},
        "",
    ]])
    assert tip == "Week 2 · 3.0 pts"


@pytestmark_node
def test_recent_form_url_helper_matches_loader_shape():
    driver = (
        "global.window = {location: {pathname: '/sleeper/2026/12345/dashboard'}};\n"
        "global.brLeagueType = () => 'dynasty';\n"
        "global.brLeagueSize = () => 12;\n"
        + _url_src()
        + "\nconst league = _pmRecentFormUrl('4046', 2026);\n"
        + "global.window = {location: {pathname: '/players'}};\n"
        + "const plain = _pmRecentFormUrl('4046', 2026);\n"
        + "process.stdout.write(JSON.stringify({league, plain}));\n"
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["league"] == (
        "/api/player-recent-form/4046?season=2026&league_type=dynasty"
        "&league_size=12&league_id=12345&platform=sleeper"
    )
    assert out["plain"] == (
        "/api/player-recent-form/4046?season=2026&league_type=dynasty&league_size=12"
    )
