"""Win-probability tween + animated matchup refresh (motion layer).

Drives window.brTweenWinBar and window.brAnimateMatchupRefresh from
static/app.js in Node with a minimal DOM shim. Verifies the bar sweeps the
gradient stop, ticks the pct labels, flips the leader color at 50%, keeps the
aria-label in sync, counts scores up, and snaps under reduced motion.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest

pytest.importorskip("flask")

_ROOT = os.path.join(os.path.dirname(__file__), "..")

_MOTION_START = "// ── Motion layer ──"


def _extract_motion_iife() -> str:
    lines = open(os.path.join(_ROOT, "static", "app.js"), encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if _MOTION_START in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "})();")
    return "".join(lines[start : end + 1])


_NODE_HARNESS = r"""
const fs = require('fs');
const iifeSrc = fs.readFileSync(process.argv[2], 'utf8');
const scenario = JSON.parse(process.argv[3]);

// ---- minimal browser shims ----
let rafQueue = [];
let nowMs = 0;
const window = {
  matchMedia: () => ({ matches: !!scenario.reduce }),
  IntersectionObserver: undefined,
};
global.window = window;
global.performance = { now: () => nowMs };
global.requestAnimationFrame = (cb) => { rafQueue.push(cb); return rafQueue.length; };
const documentStub = {
  readyState: 'complete',
  addEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
  getElementById() { return null; },
  createElement() { return makeTmp(); },
};
global.document = documentStub;
global.sessionStorage = { getItem: () => null, setItem() {} };

function makeTextNode(text) { return { textContent: text, style: {} }; }

function makeBar(lp, label) {
  const pctL = makeTextNode(Math.round(lp) + '%');
  const pctR = makeTextNode(Math.round(100 - lp) + '%');
  const track = { style: {} };
  const bar = {
    _aria: label || ('Win probability: Left ' + Math.round(lp) + ' percent, Right ' + Math.round(100 - lp) + ' percent'),
    _pcts: [pctL, pctR],
    _track: track,
    querySelectorAll(sel) {
      if (sel === '.m-wp-pct') return this._pcts;
      return [];
    },
    querySelector(sel) {
      if (sel === '.m-wp-pct') return this._pcts[0];
      if (sel === '.m-wp-track') return this._track;
      return null;
    },
    getAttribute(name) { return name === 'aria-label' ? this._aria : null; },
    setAttribute(name, v) { if (name === 'aria-label') this._aria = v; },
  };
  return bar;
}

function makeScore(val) {
  // With scenario.nestedScores the score node carries live-mode nested markup
  // (<span class="num"> + <span class="proj">), like the real .m-score-val.
  const nested = typeof scenario !== 'undefined' && !!scenario.nestedScores;
  const num = nested ? {
    textContent: String(val),
    style: {},
    getAttribute() { return null; },
    setAttribute() {},
  } : null;
  const score = {
    _text: String(val),
    _counted: false, // true if brCountUp ever writes textContent on THIS node
    style: {},
    querySelector(sel) { return sel === '.num' ? num : null; },
    getAttribute() { return null; },
    setAttribute() {},
  };
  Object.defineProperty(score, 'textContent', {
    set(v) { this._text = v; this._counted = true; },
    get() { return this._text; },
  });
  return score;
}

// Fake parsed-HTML node: regex-extracts win bars and scores from a markup string.
function makeTmp() {
  const tmp = {
    _html: '',
    _bars: [],
    _scores: [],
    querySelectorAll(sel) {
      if (sel === '.m-win-bar') return this._bars;
      if (sel === '.m-score-val' || sel === '.ls-team-score') return this._scores;
      return [];
    },
  };
  Object.defineProperty(tmp, 'innerHTML', {
    set(html) {
      this._html = html;
      this._bars = [];
      this._scores = [];
      const pctRe = /m-wp-pct[^>]*>(\d+)%</g;
      let m;
      while ((m = pctRe.exec(html))) this._bars.push(makeBar(Number(m[1])));
      const scoreRe = /m-score-val[^>]*>(?:<span class="num">)?([\d.]+)</g;
      while ((m = scoreRe.exec(html))) this._scores.push(makeScore(m[1]));
      const lsRe = /ls-team-score[^>]*>([\d.]+)</g;
      while ((m = lsRe.exec(html))) this._scores.push(makeScore(m[1]));
    },
    get() { return this._html; },
  });
  return tmp;
}

function makeContainer(barLp, scores) {
  const bars = barLp == null ? [] : [makeBar(barLp)];
  const sc = (scores || []).map(makeScore);
  return {
    _bars: bars,
    _scores: sc,
    querySelectorAll(sel) {
      if (sel === '.m-win-bar') return this._bars;
      if (sel === '.m-score-val' || sel === '.ls-team-score') return this._scores;
      return [];
    },
  };
}

function barHtml(lp) {
  const rp = 100 - lp;
  return '<div class="m-win-bar" role="img" aria-label="Win probability: Left ' + lp + ' percent, Right ' + rp + ' percent">'
    + '<span class="m-wp-pct">' + lp + '%</span>'
    + '<div class="m-wp-track" style="background:linear-gradient(to right,#22c55e ' + lp + '%,rgba(148,163,184,0.35) ' + lp + '%)"></div>'
    + '<span class="m-wp-pct">' + rp + '%</span></div>';
}
function scoreHtml(v) {
  if (scenario.nestedScores) {
    return '<div class="m-score-val"><span class="num">' + v + '</span>'
      + '<span class="proj">99.9<span class="mb-trend">&#9650;</span></span></div>';
  }
  return '<div class="m-score-val">' + v + '</div>';
}

// Step the rAF queue forward in 50ms increments up to targetMs.
function stepTo(targetMs) {
  while (nowMs < targetMs) {
    nowMs += 50;
    const q = rafQueue;
    rafQueue = [];
    q.forEach((cb) => cb(nowMs));
    if (q.length === 0 && rafQueue.length === 0) break;
  }
}

eval(iifeSrc);

const out = {};
const mode = scenario.mode;
if (mode === 'tween') {
  const bar = makeBar(scenario.from);
  window.brTweenWinBar(bar, scenario.to);
  stepTo(2000);
  out.pctL = bar._pcts[0].textContent;
  out.pctR = bar._pcts[1].textContent;
  out.bg = bar._track.style.background || '';
  out.colL = bar._pcts[0].style.color || '';
  out.colR = bar._pcts[1].style.color || '';
  out.aria = bar._aria;
} else if (mode === 'supersede') {
  const bar = makeBar(scenario.from);
  window.brTweenWinBar(bar, 90); // superseded almost immediately
  window.brTweenWinBar(bar, scenario.to);
  stepTo(2000);
  out.pctL = bar._pcts[0].textContent;
  out.aria = bar._aria;
} else if (mode === 'refresh') {
  const container = makeContainer(scenario.fromLp, scenario.fromScores);
  const newHtml = barHtml(scenario.toLp) + scenario.toScores.map(scoreHtml).join('');
  window.brAnimateMatchupRefresh(container, newHtml, function () {
    const tmp = makeTmp();
    tmp.innerHTML = newHtml;
    container._bars = tmp._bars;
    container._scores = tmp._scores;
  });
  stepTo(2000);
  const bar = container._bars[0];
  out.pctL = bar._pcts[0].textContent;
  out.aria = bar._aria;
  out.scores = container._scores.map((s) => s.textContent);
  if (scenario.nestedScores) {
    out.numText = container._scores.map((s) => s.querySelector('.num').textContent);
    out.parentCounted = container._scores.map((s) => s._counted);
  }
} else if (mode === 'missing') {
  // Must not throw on missing/degenerate nodes.
  window.brTweenWinBar(null, 70);
  window.brTweenWinBar({ querySelectorAll: () => [], querySelector: () => null }, 70);
  window.brAnimateMatchupRefresh(null, '<div></div>', null);
  out.ok = true;
}
console.log(JSON.stringify(out));
"""


def _run(scenario: dict) -> dict:
    iife = _extract_motion_iife()
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(iife)
        iife_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(_NODE_HARNESS)
        harness_path = f.name
    try:
        proc = subprocess.run(
            ["node", harness_path, iife_path, json.dumps(scenario)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_ROOT,
        )
    finally:
        os.unlink(iife_path)
        os.unlink(harness_path)
    assert proc.returncode == 0, f"node harness failed: {proc.stderr[:2000]}"
    return json.loads(proc.stdout)


def test_tween_sweeps_bar_and_ticks_labels():
    out = _run({"mode": "tween", "from": 54, "to": 71})
    assert out["pctL"] == "71%"
    assert out["pctR"] == "29%"
    assert "71%" in out["bg"]  # gradient hard stop followed the tween
    assert out["colL"] == "#22c55e"
    assert "71 percent" in out["aria"]
    assert "29 percent" in out["aria"]


def test_tween_flips_leader_color_at_50():
    out = _run({"mode": "tween", "from": 46, "to": 58})
    assert out["pctL"] == "58%"
    assert out["colL"] == "#22c55e"  # left takes the lead: green
    assert out["colR"] == "var(--text-muted)"


def test_tween_snaps_under_reduced_motion():
    out = _run({"mode": "tween", "from": 54, "to": 71, "reduce": True})
    assert out["pctL"] == "71%"
    assert "71 percent" in out["aria"]


def test_superseded_tween_is_cancelled():
    out = _run({"mode": "supersede", "from": 54, "to": 60})
    assert out["pctL"] == "60%"  # the first tween (to 90) must not win
    assert "60 percent" in out["aria"]


def test_refresh_tweens_bar_and_counts_scores():
    out = _run({
        "mode": "refresh",
        "fromLp": 54,
        "fromScores": ["112.4", "108.9"],
        "toLp": 63,
        "toScores": ["118.2", "109.4"],
    })
    assert out["pctL"] == "63%"
    assert "63 percent" in out["aria"]
    assert out["scores"] == ["118.2", "109.4"]


def test_refresh_counts_nested_score_markup_without_flattening():
    # Live .m-score-val nodes carry nested markup (actual + projection + trend
    # arrow). The count-up must target the inner .num node so the projection
    # markup survives the refresh.
    out = _run({
        "mode": "refresh",
        "fromLp": 54,
        "fromScores": ["100.0"],
        "toLp": 60,
        "toScores": ["105.0"],
        "nestedScores": True,
    })
    assert out["numText"] == ["105.0"]
    # The wrapper node itself must not have been counted up (that would
    # flatten its nested projection markup to a bare number).
    assert out["parentCounted"] == [False]


def test_refresh_snaps_under_reduced_motion():
    out = _run({
        "mode": "refresh",
        "fromLp": 54,
        "fromScores": ["112.4", "108.9"],
        "toLp": 63,
        "toScores": ["118.2", "109.4"],
        "reduce": True,
    })
    assert out["pctL"] == "63%"
    assert out["scores"] == ["118.2", "109.4"]


def test_helpers_tolerate_missing_nodes():
    out = _run({"mode": "missing"})
    assert out["ok"] is True
