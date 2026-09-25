"""PRO upsell prompts: dismissal persistence, feature resolution, placements.

Covers the shared brUpsell infra (static/paywall.js), the improved locked PRO
metric paywall copy, the Wrapped finale CTA, and the Breakout locked-card
conversion. Node-driven JS tests follow the test_adp_delta_verdict.py pattern.

Skips cleanly when Node isn't available.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAYWALL_JS = REPO / "static" / "paywall.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


_NODE_PRELUDE = r"""
global.self = global;
global.window = global;
global.window.location = { hash: '', search: '', pathname: '/' };
function __fakeEl() {
  var handlers = {};
  var el = {
    _handlers: handlers, _q: {}, children: [],
    style: {}, dataset: {},
    className: '', innerHTML: '', textContent: '',
    classList: { add: function(){}, remove: function(){}, toggle: function(){}, contains: function(){ return false; } },
    setAttribute: function(){}, removeAttribute: function(){},
    getAttribute: function(){ return null; }, hasAttribute: function(){ return false; },
    appendChild: function(c){ el.children.push(c); c.parentNode = el; return c; },
    removeChild: function(c){ var i = el.children.indexOf(c); if (i >= 0) el.children.splice(i, 1); return c; },
    remove: function(){ if (el.parentNode) el.parentNode.removeChild(el); },
    addEventListener: function(t, fn){ (handlers[t] = handlers[t] || []).push(fn); },
    removeEventListener: function(){},
    querySelector: function(sel){ if (!el._q[sel]) el._q[sel] = __fakeEl(); return el._q[sel]; },
    querySelectorAll: function(){ return []; },
    focus: function(){}, click: function(){},
    parentNode: null, firstElementChild: null
  };
  return el;
}
global.__fakeEl = __fakeEl;
var __lsStore = {};
global.localStorage = {
  getItem: function(k){ return Object.prototype.hasOwnProperty.call(__lsStore, k) ? __lsStore[k] : null; },
  setItem: function(k, v){ __lsStore[k] = String(v); },
  removeItem: function(k){ delete __lsStore[k]; }
};
global.__lsStore = __lsStore;
global.document = {
  readyState: 'complete', activeElement: null,
  createElement: function(){ return __fakeEl(); },
  getElementById: function(){ return null; },
  querySelector: function(){ return null; },
  querySelectorAll: function(){ return []; },
  addEventListener: function(){}, removeEventListener: function(){},
  body: __fakeEl(), documentElement: __fakeEl()
};
require(PAYWALL_PATH);
""".replace("PAYWALL_PATH", json.dumps(str(PAYWALL_JS)))


def _node(script: str):
    driver = _NODE_PRELUDE + script
    res = subprocess.run(["node", "-e", driver], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


# ── brUpsell dismissal persistence ────────────────────────────────────────────

def test_dismiss_persists_across_reload():
    out = _node(r"""
      var out = {};
      out.before = window.brUpsell.dismissed('k-test');
      window.brUpsell.dismiss('k-test');
      out.after = window.brUpsell.dismissed('k-test');
      out.stored = JSON.parse(global.localStorage.getItem('br-upsell-dismissed.v1'));
      process.stdout.write(JSON.stringify(out));
    """)
    assert out["before"] is False
    assert out["after"] is True
    assert "k-test" in out["stored"]


def test_dismiss_survives_fresh_js_state():
    script = (
        "var PAYWALL = " + json.dumps(str(PAYWALL_JS)) + ";\n"
        r"""
      window.brUpsell.dismiss('k-persist');
      delete require.cache[require.resolve(PAYWALL)];
      require(PAYWALL);
      var out = {};
      out.dismissedAfterReload = window.brUpsell.dismissed('k-persist');
      out.nudgeAfterReload = window.brUpsell.nudge(
        global.document.createElement('div'), { key: 'k-persist', message: 'x' });
      out.nudgeFreshKey = window.brUpsell.nudge(
        global.document.createElement('div'), { key: 'k-fresh', message: 'y' });
      process.stdout.write(JSON.stringify(out));
    """
    )
    out = _node(script)
    assert out["dismissedAfterReload"] is True
    assert out["nudgeAfterReload"] is False
    assert out["nudgeFreshKey"] is True


def test_nudge_renders_and_x_persists_dismissal():
    out = _node(r"""
      var container = global.document.createElement('div');
      var shown = window.brUpsell.nudge(container, {
        key: 'k-nudge', message: 'Hello <b>world</b>', ctaLabel: 'Unlock', feature: 'x'
      });
      var html = container.innerHTML;
      var root = container.querySelector('.br-upsell-nudge');
      root.querySelector('.br-upsell-nudge-x')._handlers.click[0]();
      var out = {
        shown: shown,
        escaped: html.indexOf('Hello &lt;b&gt;world&lt;/b&gt;') >= 0,
        hasCta: html.indexOf('>Unlock<') >= 0,
        dismissed: window.brUpsell.dismissed('k-nudge'),
        rootRemoved: container.querySelector('.br-upsell-nudge').parentNode === null ||
                     root.parentNode === null
      };
      process.stdout.write(JSON.stringify(out));
    """)
    assert out["shown"] is True
    assert out["escaped"] is True
    assert out["hasCta"] is True
    assert out["dismissed"] is True


def test_nudge_cta_opens_paywall_for_feature():
    out = _node(r"""
      var realShow = window.showPaywall;
      var calls = [];
      window.showPaywall = function (f, o) { calls.push([f, o && o.source]); };
      var container = global.document.createElement('div');
      window.brUpsell.nudge(container, { key: 'k-cta', feature: 'wrapped-pro' });
      var root = container.querySelector('.br-upsell-nudge');
      root.querySelector('.br-upsell-nudge-cta')._handlers.click[0]();
      window.showPaywall = realShow;
      process.stdout.write(JSON.stringify({ calls: calls }));
    """)
    assert out["calls"] == [["wrapped-pro", "nudge:k-cta"]]


def test_nudge_respects_promo_eligibility():
    out = _node(r"""
      window._brPromoEligible = function () { return false; };
      var blocked = window.brUpsell.nudge(
        global.document.createElement('div'), { key: 'k-elig' });
      window._brPromoEligible = function () { return true; };
      var allowed = window.brUpsell.nudge(
        global.document.createElement('div'), { key: 'k-elig2' });
      delete window._brPromoEligible;
      var fallback = window.brUpsell.nudge(
        global.document.createElement('div'), { key: 'k-elig3' });
      process.stdout.write(JSON.stringify({ blocked: blocked, allowed: allowed, fallback: fallback }));
    """)
    assert out["blocked"] is False
    assert out["allowed"] is True
    assert out["fallback"] is True


def test_nudge_survives_localstorage_failure():
    out = _node(r"""
      global.localStorage = {
        getItem: function(){ throw new Error('denied'); },
        setItem: function(){ throw new Error('denied'); },
        removeItem: function(){}
      };
      var out = { ok: true };
      try {
        out.shown = window.brUpsell.nudge(
          global.document.createElement('div'), { key: 'k-nos' });
        out.dismissed = window.brUpsell.dismissed('k-nos');
        window.brUpsell.dismiss('k-nos');
      } catch (e) { out.ok = false; out.err = String(e); }
      process.stdout.write(JSON.stringify(out));
    """)
    assert out["ok"] is True
    assert out["shown"] is True
    assert out["dismissed"] is False


# ── Feature resolution: locked PRO metrics get name + why-it-matters ─────────

def test_resolve_pro_feature_metric_and_preset():
    out = _node(r"""
      window.__brProMetricInfo = { wopr: { label: 'WOPR', why: 'Why WOPR matters.' } };
      window.__brProPresetInfo = { buy_low_sell_high: { label: 'Buy Low / Sell High', tagline: 'Find mispriced players.' } };
      var out = {
        metric: window.brResolveProFeature('advanced-metrics-metric-wopr'),
        preset: window.brResolveProFeature('advanced-metrics-buy_low_sell_high'),
        unknownMetric: window.brResolveProFeature('advanced-metrics-metric-nope'),
        unknown: window.brResolveProFeature('something-else'),
        movers: window.brResolveProFeature('advanced-metrics-movers'),
        wrapped: window.brResolveProFeature('wrapped-pro')
      };
      process.stdout.write(JSON.stringify(out));
    """)
    assert out["metric"] == {"name": "WOPR", "benefit": "Why WOPR matters."}
    assert out["preset"] == {"name": "Buy Low / Sell High", "benefit": "Find mispriced players."}
    assert out["unknownMetric"] == {"name": "PRO metric", "benefit": "A PRO intelligence metric."}
    assert out["unknown"] is None
    assert out["movers"]["name"] == "Movers: heating up and cooling off"
    assert "full story" in out["wrapped"]["name"]


def test_showpaywall_headline_uses_metric_name_and_why():
    out = _node(r"""
      window.__brProMetricInfo = { wopr: { label: 'WOPR', why: 'Target share plus air-yards share in one number.' } };
      window.__brProPresetInfo = { buy_low_sell_high: { label: 'Buy Low / Sell High', tagline: 'Find mispriced players before your league does.' } };
      window.showPaywall('advanced-metrics-metric-wopr');
      var kids = global.document.body.children;
      var metricHtml = kids[kids.length - 1].innerHTML;
      window.showPaywall('advanced-metrics-buy_low_sell_high');
      var kids2 = global.document.body.children;
      var presetHtml = kids2[kids2.length - 1].innerHTML;
      process.stdout.write(JSON.stringify({ metricHtml: metricHtml, presetHtml: presetHtml }));
    """)
    assert "<h3>WOPR</h3>" in out["metricHtml"]
    assert "Target share plus air-yards share in one number." in out["metricHtml"]
    assert "<h3>Buy Low / Sell High</h3>" in out["presetHtml"]
    assert "Find mispriced players before your league does." in out["presetHtml"]


# ── Advanced metrics page: public-safe info maps ─────────────────────────────

def _am_cfg():
    from data_building.advanced_metrics import LEADERBOARD_METRICS, PRO_METRICS
    from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
    html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    m = re.search(r"const AM_CFG = ", html)
    assert m, "AM_CFG not found in page HTML"
    start = html.index("{", m.end())
    depth = 0
    for i in range(start, len(html)):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:i + 1]), html, PRO_METRICS
    raise AssertionError("unbalanced AM_CFG braces")


def test_pro_metric_info_covers_all_pro_metrics():
    cfg, html, pro_metrics = _am_cfg()
    info = cfg["proMetricInfo"]
    assert set(info.keys()) == set(pro_metrics)
    for key, entry in info.items():
        assert entry["label"], key
        assert entry["why"], key
        assert "—" not in entry["label"], key
        assert "—" not in entry["why"], key
    # No PRO values leak: only label + why, never numbers.
    for entry in info.values():
        assert set(entry.keys()) == {"label", "why"}


def test_pro_preset_info_covers_locked_presets():
    cfg, html, _ = _am_cfg()
    info = cfg["proPresetInfo"]
    assert set(info.keys()) == {"buy_low_sell_high", "waiver_wire", "breakout_check"}
    for key, entry in info.items():
        assert entry["label"], key
        assert "tagline" in entry, key
    # Published to window for the paywall headline resolver; nudge infra
    # waits for deferred paywall.js instead of racing it.
    assert "window.__brProMetricInfo = cfg.proMetricInfo" in html
    assert "window.__brProPresetInfo = cfg.proPresetInfo" in html
    assert "_whenUpsellReady" in html
    assert "'am-movers'" in html


# ── Wrapped finale CTA ───────────────────────────────────────────────────────

def _wrapped_slides():
    return [
        {"kind": "intro", "eyebrow": "SEASON", "big": "T", "num": False, "dp": 0,
         "suffix": "", "label": "Wrapped", "sub": "s"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": "", "label": "Alpha", "sub": "s"},
        {"kind": "lowscore", "eyebrow": "LOW", "big": "90.2", "num": True,
         "dp": 1, "suffix": "", "label": "Beta", "sub": "s"},
    ]


def test_wrapped_overlay_marks_pro_cta_only_when_asked():
    pytest.importorskip("numpy")
    from dashboard_services.pages import history_page as H
    with_cta = H._wrapped_overlay_markup(_wrapped_slides(), None, ns="wrapped",
                                         show_pro_cta=True)
    assert 'data-pro-cta="1"' in with_cta
    without = H._wrapped_overlay_markup(_wrapped_slides(), None, ns="wrapped")
    assert "data-pro-cta" not in without
    weekly = H._wrapped_overlay_markup(_wrapped_slides(), None, ns="weekly-wrapped",
                                       show_pro_cta=True)
    assert 'data-pro-cta="1"' in weekly


def test_wrapped_bootstrap_appends_pro_finale_client_side():
    pytest.importorskip("numpy")
    from dashboard_services.pages import history_page as H
    js = H._wrapped_bootstrap_js("wrapped")
    assert "_maybeAppendProSlide" in js
    assert "Unlock the full story with PRO" in js
    assert "wrapped-pro-cta" in js
    assert 'getAttribute(\'data-pro-cta\')' in js
    # Never on public share decks, and never nagging: dismissal is checked
    # before the slide is appended.
    assert "__wrappedSharePublic" in js
    assert "br-upsell-dismissed.v1" in js
    # CTA opens the paywall in-app, falls back to /pricing without it.
    assert "showPaywall('wrapped-pro')" in js
    assert "window.location.href = '/pricing'" in js
    # No em dashes in the new UI copy (house style).
    block = js[js.index("_maybeAppendProSlide"):js.index("_maybeAppendProSlide") + 4000]
    assert "—" not in block


def test_wrapped_public_bootstrap_still_valid():
    pytest.importorskip("numpy")
    from dashboard_services.pages import history_page as H
    # Runs the internal preamble/launch-handler assertions; raises if they drift.
    js = H._wrapped_public_bootstrap_js("wrapped")
    assert "__wrappedSharePublic" in js


# ── Breakout Engine: locked card becomes one dismissible nudge ───────────────

def test_breakout_locked_card_replaced_by_dismissible_nudge():
    src = (REPO / "app.py").read_text()
    assert "boUpsellNudge" in src
    assert "more candidates locked" not in src
    assert "'bo-locked'" in src
    assert "PRO unlocks the full list with full details." in src
    # The three free preview candidates still render (grid built from the
    # API's preview list before the nudge placeholder).
    assert "renderBreakoutCard(candidate)" in src
