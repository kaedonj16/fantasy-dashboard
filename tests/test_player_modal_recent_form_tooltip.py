"""Player modal Overview: Recent Form game-stat card + Overview speed.

- Each Recent Form column keeps the shared tooltip interaction contract
  (tabindex, aria-label, hover / focus / tap-to-toggle) and carries a
  structured card payload built by _pmRecentGameCard(g, oppText, avg)
  from g.stats: header, hero points, then Pass / Rush / Rec rows only
  when that category had attempts, and TD / INT segments only when
  greater than zero. Fumbles render as a small note only when lost.
  The plain _pmRecentGameTip line remains the aria-label / fallback.
  Receiving always uses labeled counts, never a fraction, and neither
  snap nor target share is ever rendered (adv is not read at all).
  No completions are ever shown (there is no such field).
- _pmStartEarlyFetches fires the news small-fetch at click with exactly
  the key and URL _pmLoadNews uses, so the loader adopts it.
- The speculative warmer and _pmStartEarlyFetches share the
  _pmRecentFormUrl helper, so their Recent Form URLs cannot drift.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


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


def _positioner_src() -> str:
    start = APP_JS.index("// [tooltip-positioner:start]")
    end = APP_JS.index("// [tooltip-positioner:end]")
    return APP_JS[start:end]


def _url_src() -> str:
    start = JS.index("function _pmRecentFormUrl")
    end = JS.index("function _pmStartEarlyFetches")
    return JS[start:end]


# ── (a) Column markup ────────────────────────────────────────────────────────

def test_recent_form_column_carries_card_tabindex_and_aria_label():
    body = _loader_body()
    assert "const _tip = _pmEsc(_pmRecentGameTip(g, _opp));" in body
    assert (
        "const _card = _pmEsc(JSON.stringify("
        "_pmRecentGameCard(g, _opp, _recentAvg)));"
    ) in body
    assert (
        '\'<div class="pm-recent-col" tabindex="0" data-tooltip="\' + _tip'
        ' + \'" data-tooltip-card="\' + _card + \'" aria-label="\' + _tip + \'">\''
    ) in body or (
        'class="pm-recent-col" tabindex="0" data-tooltip="' in body
        and 'data-tooltip-card="' in body
        and 'aria-label="' in body
    )
    # Recent avg is the mean of exactly the games in the rendered row:
    # it is computed from the bye-filtered `real` list's last-five slice,
    # after the slice and before the columns are built from it.
    assert "if (!g || g.is_bye || g.is_projection) return false;" in body
    assert (
        body.index("const shown = real.slice(-5);")
        < body.index("const _recentAvg = shown.reduce")
        < body.index("const cols = shown.map")
    )
    assert (
        "const _recentAvg = shown.reduce((sum, g) => sum + "
        "(Number(g.fantasy_pts) || 0), 0) / shown.length;"
    ) in body
    # Plain div, not an interactive element: the global tooltip
    # positioner's tap-to-toggle path ignores A/BUTTON/INPUT/etc.
    assert '<button class="pm-recent-col"' not in body
    assert '<a class="pm-recent-col"' not in body


def test_recent_form_card_css_uses_site_tokens():
    start = CSS.index(".br-tip--card")
    body = CSS[start:CSS.index("html.br-tips-js", start)]
    assert "width: 230px" in body
    assert "background: var(--card)" in body
    assert "color: var(--text)" in body
    assert "border-color: var(--border)" in body
    assert "border-radius: 10px" in body
    assert ".br-tip-card-head" in body
    assert ".br-tip-card-hero" in body
    assert ".br-tip-card-row" in body
    assert ".br-tip-card-foot" in body
    assert "var(--text-muted)" in body
    assert "font-variant-numeric: tabular-nums" in body


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


def _run_cards(cases):
    """cases: list of [game, oppText, recentAvg]; returns card objects."""
    driver = (
        _tip_src()
        + "\nconst cases = " + json.dumps(cases) + ";\n"
        + "process.stdout.write(JSON.stringify(cases.map(c => _pmRecentGameCard(c[0], c[1], c[2]))));\n"
    )
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytestmark_node
def test_tip_qb_pass_line_with_td_and_int_but_no_snap():
    (tip,) = _run_tips([[
        {"week": 5, "fantasy_pts": 28.4,
         "stats": {"pass_att": 35, "pass_yd": 300, "pass_td": 3, "pass_int": 1,
                   "rush_att": 5, "rush_yd": 42, "rush_td": 1, "fum_lost": 0},
         "adv": {"snap_pct": 98.4}},
        "@ LV",
    ]])
    assert tip == "Week 5 @ LV · 28.4 pts · Pass 35 att, 300 yd, 3 TD, 1 INT · Rush 5 att, 42 yd, 1 TD"
    assert "comp" not in tip.lower()
    assert "Snap" not in tip


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
    assert tip == "Week 3 vs KC · 18.2 pts · Rush 18 att, 92 yd, 1 TD · Rec 4 rec, 6 tgt, 31 yd · 1 fum lost"
    assert "Snap" not in tip


@pytestmark_node
def test_tip_zero_activity_segments_are_omitted():
    # The stat dict carries zeroed fields for categories a player never
    # touched (an RB's pass_att 0 / pass_yd 0): those segments must not
    # render as "Pass 0 att, 0 yd" noise. Same for a zeroed Rec line.
    (tip,) = _run_tips([[
        {"week": 8, "fantasy_pts": 24.8,
         "stats": {"pass_att": 0, "pass_yd": 0, "pass_td": 0,
                   "rush_att": 23, "rush_yd": 153, "rush_td": 1,
                   "rec": 2, "rec_tgt": 4, "rec_yd": 15, "rec_td": 0},
         "adv": None},
        "vs MIN",
    ]])
    assert tip == "Week 8 vs MIN · 24.8 pts · Rush 23 att, 153 yd, 1 TD · Rec 2 rec, 4 tgt, 15 yd"
    assert "Pass" not in tip
    (tip2,) = _run_tips([[
        {"week": 9, "fantasy_pts": 11.0,
         "stats": {"rush_att": 10, "rush_yd": 40, "rush_td": 0,
                   "rec": 0, "rec_tgt": 0, "rec_yd": 0},
         "adv": None},
        "vs KC",
    ]])
    assert tip2 == "Week 9 vs KC · 11.0 pts · Rush 10 att, 40 yd"
    assert "Rec" not in tip2


@pytestmark_node
def test_tip_wr_rec_without_targets_and_no_opponent():
    (tip,) = _run_tips([[
        {"week": 4, "fantasy_pts": 12.0,
         "stats": {"rec": 5, "rec_tgt": None, "rec_yd": 68, "rec_td": 1}},
        "",
    ]])
    assert tip == "Week 4 · 12.0 pts · Rec 5 rec, 68 yd, 1 TD"


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
def test_card_rb_rows_use_dots_labeled_counts_and_no_adv_fields():
    (card,) = _run_cards([[
        {"week": 8, "fantasy_pts": 24.8,
         "stats": {"pass_att": 0, "pass_yd": 0, "pass_td": 0,
                   "rush_att": 23, "rush_yd": 153, "rush_td": 1,
                   "rec": 2, "rec_tgt": 4, "rec_yd": 15, "rec_td": 0},
         "adv": {"snap_pct": 68, "target_share": 12}},
        "vs MIN",
        18.75,
    ]])
    assert card == {
        "header": "Week 8 · vs MIN",
        "points": "24.8",
        "pointsUnit": "pts",
        "rows": [
            ["Rush", "23 att · 153 yd · 1 TD"],
            ["Rec", "2 rec · 4 tgt · 15 yd"],
        ],
        "note": "",
        "footer": "Recent avg 18.8",
    }
    rendered = json.dumps(card)
    assert "snap" not in rendered.lower()
    assert "target" not in rendered.lower()
    assert "Pass" not in rendered
    assert not re.search(r"of \d+ tgt|\d+/\d+ tgt", rendered)


@pytestmark_node
def test_card_qb_rows_include_td_and_int_only_when_positive():
    (card,) = _run_cards([[
        {"week": 5, "fantasy_pts": 28.4,
         "stats": {"pass_att": 35, "pass_yd": 300, "pass_td": 3, "pass_int": 1,
                   "rush_att": 5, "rush_yd": 42, "rush_td": 0,
                   "rec": 0, "rec_tgt": 0, "rec_yd": 0},
         "adv": {"snap_pct": 98.4, "target_share": 0}},
        "@ LV",
        20.0,
    ]])
    assert card["header"] == "Week 5 · @ LV"
    assert card["rows"] == [
        ["Pass", "35 att · 300 yd · 3 TD · 1 INT"],
        ["Rush", "5 att · 42 yd"],
    ]
    assert card["footer"] == "Recent avg 20.0"


@pytestmark_node
def test_card_zero_categories_and_fumble_note():
    (card,) = _run_cards([[
        {"week": 3, "fantasy_pts": 18.2,
         "stats": {"pass_att": 0, "rush_att": 18, "rush_yd": 92, "rush_td": 1,
                   "rec": 4, "rec_tgt": 6, "rec_yd": 31, "rec_td": 1,
                   "fum_lost": 2},
         "adv": {"snap_pct": 72}},
        "vs KC",
        18.2,
    ]])
    assert card["rows"] == [
        ["Rush", "18 att · 92 yd · 1 TD"],
        ["Rec", "4 rec · 6 tgt · 31 yd · 1 TD"],
    ]
    assert card["note"] == "Fum lost 2"


@pytestmark_node
def test_positioner_renders_card_for_recent_form_and_plain_text_elsewhere():
    card = {
        "header": "Week 8 · vs MIN",
        "points": "24.8",
        "pointsUnit": "pts",
        "rows": [
            ["Rush", "23 att · 153 yd · 1 TD"],
            ["Rec", "2 rec · 4 tgt · 15 yd"],
        ],
        "note": "",
        "footer": "Recent avg 18.8",
    }
    driver = r"""
class El {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase();
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.className = '';
    this._text = '';
    this.isConnected = true;
    this.offsetWidth = 230;
    this.offsetHeight = 160;
    this.classList = {
      contains: (name) => this.className.split(/\s+/).includes(name),
      add: (name) => { if (!this.classList.contains(name)) this.className = (this.className + ' ' + name).trim(); }
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
  appendChild(child) { this.children.push(child); return child; }
  set textContent(v) { this._text = String(v == null ? '' : v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  closest(sel) { return sel.indexOf('[data-tooltip]') >= 0 && this.getAttribute('data-tooltip') ? this : null; }
  getBoundingClientRect() { return {left: 100, right: 172, top: 320, bottom: 430, width: 72, height: 110}; }
}
const handlers = {};
global.window = global;
global.innerWidth = 900;
global.innerHeight = 800;
global.__coarse = false;
global.matchMedia = (q) => ({matches: q === '(pointer: coarse)' ? global.__coarse : false});
global.requestAnimationFrame = (fn) => { fn(); return 1; };
global.addEventListener = () => {};
global.getComputedStyle = () => ({position: 'static'});
global.document = {
  body: new El('body'),
  documentElement: new El('html'),
  createElement: (tag) => new El(tag),
  querySelector: () => null,
  addEventListener: (type, fn) => { handlers[type] = fn; }
};
""" + _positioner_src() + r"""
function makeAnchor(card) {
  const el = new El('div');
  el.setAttribute('data-tooltip', 'Plain fallback');
  if (card) el.setAttribute('data-tooltip-card', JSON.stringify(card));
  return el;
}
const cardAnchor = makeAnchor(CARD_JSON);
handlers.focusin({target: cardAnchor});
const tip = global.document.body.children[0];
if (!tip || tip.className !== 'br-tip br-tip--card') throw new Error('card class not applied: ' + (tip && tip.className));
if (tip.style.display !== 'block') throw new Error('card not shown');
for (const needle of ['Week 8 · vs MIN', '24.8', 'Rush', '23 att · 153 yd · 1 TD', 'Rec', '2 rec · 4 tgt · 15 yd', 'Recent avg 18.8']) {
  if (!tip.textContent.includes(needle)) throw new Error('missing card text: ' + needle + ' in ' + tip.textContent);
}
handlers.focusout({target: cardAnchor, relatedTarget: null});
if (tip.style.display !== 'none') throw new Error('card not hidden on focusout');
const plainAnchor = makeAnchor(null);
handlers.focusin({target: plainAnchor});
if (tip.className !== 'br-tip' || tip.textContent !== 'Plain fallback') throw new Error('plain tooltip path changed');
handlers.focusout({target: plainAnchor, relatedTarget: null});
global.__coarse = true;
handlers.click({target: cardAnchor});
if (tip.style.display !== 'block' || tip.className !== 'br-tip br-tip--card') throw new Error('tap did not show card');
handlers.click({target: cardAnchor});
if (tip.style.display !== 'none') throw new Error('second tap did not hide card');
process.stdout.write(JSON.stringify({ok: true, text: tip.textContent}));
"""
    driver = driver.replace("CARD_JSON", json.dumps(card))
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["ok"] is True


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
