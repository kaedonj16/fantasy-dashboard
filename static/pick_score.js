// Single source of truth for the draft pick-score FORMULA.
//
// Mirrors utils/pick_score.py::compute_pick_score exactly. The draft room
// (browser) and the Teams-page /api/draft-grades (Python) both grade picks with
// this same math, so a pick's grade can't differ between the two surfaces.
// tests/test_pick_score_parity.py runs this (via Node) against the Python copy
// over random inputs and fails CI on any divergence.
//
// This is the pure math only - pick QUALITY, nothing time-sensitive. Callers
// gather the inputs (value, vor, tier, adp, need, ppg, ...). Live-draft timing
// (survival to the next pick, redraft handcuff insurance) lives entirely in the
// draft-room decision layer (draft_board_core.js decisionScore), so the timing
// signal is counted exactly once and never runs through the depth/relabel
// shaping below.
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BRPickScore = api;
})(typeof self !== 'undefined' ? self : this, function () {
  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

  // Rookie momentum down-weighted 0.06 -> 0.03 (freed weight to value/adp) after
  // a 509-team backtest. Rookie & startup youth set to the validated 0.10 winner
  // (600-league multi-year). Rookie 0.16 -> 0.10 (freed mass to tier/ppg);
  // startup 0.05 -> 0.10 (taken from ADP). Need weights are unchanged. Keep in
  // lockstep with PS_WEIGHTS in utils/pick_score.py. Redraft model value is
  // deliberately DB-only (not an ADP blend); its explicit market weight starts
  // at .18 and decays with round depth, with the released weight redistributed
  // exactly. Live timing (survival, handcuff, late-round upside) is NOT here.
  var WEIGHTS = {
    rookie:  { vor: 0.06, value: 0.20, adp: 0.30, tier: 0.18, need: 0.05, youth: 0.10, mom: 0.03, ppg: 0.13 },
    redraft: { vor: 0.15, value: 0.25, adp: 0.18, tier: 0.12, need: 0.09, youth: 0.00, mom: 0.03, ppg: 0.22 },
    startup: { vor: 0.07, value: 0.24, adp: 0.25, tier: 0.15, need: 0.09, youth: 0.10, mom: 0.03, ppg: 0.12 },
  };
  var AGE_PEAKS = { RB: 24, WR: 27, TE: 27, QB: 29 };
  var CORE = { QB: 1, RB: 1, WR: 1, TE: 1 };

  // o: { pos, value, vor, tier, age, rankChange7d, avgPick, pickNo, maxVal,
  //      draftType, isSf, needRaw, qbCount, totalPicks, numTeams, ppgNorm,
  //      ppr, tep, passTd, isTierCliff }
  // Returns an integer 0-100.
  function computePickScore(o) {
    var pos = (o.pos || '').toUpperCase();
    var value = o.value != null ? +o.value : 0;
    var vor = o.vor != null ? +o.vor : null;
    var age = o.age != null ? +o.age : null;
    var rankChange7d = o.rankChange7d != null ? +o.rankChange7d : null;
    var avgPick = o.avgPick != null ? +o.avgPick : null;
    var maxVal = o.maxVal != null ? +o.maxVal : 0;
    var needRaw = o.needRaw != null ? +o.needRaw : 0;
    var totalPicks = o.totalPicks != null ? +o.totalPicks : 0;
    var pickNo = o.pickNo != null ? +o.pickNo : 0;
    var tier = o.tier;
    var ppr = o.ppr != null ? +o.ppr : 1.0;
    var tep = o.tep != null ? +o.tep : 0.0;
    var passTd = o.passTd != null ? +o.passTd : 4.0;

    var dbValueNorm = (maxVal && maxVal > 0) ? clamp01(value / maxVal) : 0;
    // Keep model/database value independent from the explicit market component.
    // A near-zero value also caps selected-only ADP below.
    var adpUntrusted = dbValueNorm < 0.05;
    var valueNorm = dbValueNorm;
    // Scarcity residual: share of this player's own value above replacement.
    // Same-position players with similar value get nearly the same residual.
    var vorNorm;
    if (vor != null && value > 0) vorNorm = clamp01(Math.max(vor, 0) / value);
    else if (vor != null) vorNorm = 0;
    else vorNorm = valueNorm * 0.8;

    var adpVal;
    if (avgPick != null) {
      var rel = (pickNo - avgPick) / Math.max(avgPick, 1.5);
      if (rel >= 0.5) adpVal = 1.0;
      else if (rel >= -0.3) adpVal = 0.5 + rel;
      else adpVal = Math.max(0, 0.2 + rel * 0.25);
      if (avgPick <= 8) adpVal = Math.max(adpVal, clamp01(0.5 + (8 - avgPick) / 16));
      if (adpUntrusted) adpVal = Math.min(adpVal, 0.5);
    } else {
      adpVal = 0.5;
    }

    var tierScore = tier ? clamp01((10 - Math.min(tier, 9)) / 9) : valueNorm;
    if (o.isTierCliff) tierScore = clamp01(tierScore + 0.15);

    // Need ramps in a touch earlier than before (/10 vs /12) so roster
    // construction starts to matter before the mid rounds, not only after.
    var needRamp = clamp01((pickNo - 1) / 10.0);
    var need = (1 - needRamp) * 0.5 + needRamp * needRaw;

    var youth = 0.5;
    if (age != null && CORE[pos]) {
      var peak = AGE_PEAKS[pos] || 27;
      youth = clamp01((peak - age + 4) / 8);
    }
    var mom = clamp01((rankChange7d || 0) / 20 + 0.5);
    var ppgN = o.ppgNorm != null ? +o.ppgNorm : valueNorm;

    var w = WEIGHTS[o.draftType] || WEIGHTS.startup;
    if (o.draftType === 'redraft') {
      var teams0 = Math.max(1, +o.numTeams || 12);
      var round0 = pickNo ? Math.floor((pickNo - 1) / teams0) + 1 : 1;
      var adpFactor = Math.max(0.15, 1 - Math.max(0, round0 - 2) * 0.075);
      var freed = w.adp * (1 - adpFactor), base = w.vor + w.value + w.tier + w.need + w.ppg;
      w = Object.assign({}, w);
      w.adp *= adpFactor;
      ['vor','value','tier','need','ppg'].forEach(function(k){ w[k] += freed * w[k] / base; });
    }
    var s = w.vor * vorNorm + w.value * valueNorm + w.adp * adpVal
          + w.tier * tierScore + w.need * need + w.youth * youth
          + w.mom * mom + (w.ppg || 0) * ppgN;

    if (!o.isSf && pos === 'QB') {
      var teams = o.numTeams ? +o.numTeams : 12;
      var round = pickNo ? Math.floor((pickNo - 1) / Math.max(teams, 1)) + 1 : 1;
      var qc = o.qbCount || 0;
      var pen = 1.0;
      if (qc >= 1) {
        // Overfill: a 2nd+ QB in 1QB only carries real opportunity cost early;
        // by the late rounds a backup QB is a normal pick, so it tapers out.
        pen = round <= 3 ? 0.30 : (round <= 6 ? 0.60 : (round <= 9 ? 0.85 : 1.0));
        if (qc >= 2) pen *= 0.7;
      } else if (o.draftType === 'redraft') {
        // First QB, redraft only: QB is streamable in 1QB, so a startable QB
        // shouldn't leapfrog clear starters / ADP-fallers in the early-mid
        // rounds. Tapers to none by the late rounds (grabbing a QB is fine
        // then). Dynasty/startup keep full QB value - young QBs are long-term
        // assets, not a streamable weekly slot.
        pen = round <= 3 ? 0.82 : (round <= 6 ? 0.90 : (round <= 9 ? 0.97 : 1.0));
      }
      s *= pen;
    }

    // Redundancy: a pick at a skill position already stocked to its realistic
    // depth target (needRaw == 0) is a bench body at a full spot while other
    // starting needs may remain. Penalize it, hardest at a true single-starter
    // slot (1-TE, or starterSlots <= 1) and in the early rounds. Need math is
    // unchanged — only the single-vs-multi classification uses roster slots.
    if (needRaw <= 0 && (pos === 'RB' || pos === 'WR' || pos === 'TE')) {
      var teams2 = o.numTeams ? +o.numTeams : 12;
      var rd2 = pickNo ? Math.floor((pickNo - 1) / Math.max(teams2, 1)) + 1 : 1;
      var single = (o.starterSlots != null && isFinite(+o.starterSlots))
        ? (+o.starterSlots <= 1)
        : (pos === 'TE');
      var rp = rd2 <= 3 ? (single ? 0.55 : 0.82)
             : rd2 <= 6 ? (single ? 0.72 : 0.90)
             : rd2 <= 9 ? (single ? 0.86 : 0.96)
             : 1.0;
      s *= rp;
    }

    if (tep && tep > 0 && pos === 'TE') s *= (1 + 0.12 * tep);
    // Six-point passing TDs raise the weekly ceiling and replacement advantage
    // of quarterbacks. Keep this modest in 1QB and let the existing SF/VOR logic
    // handle format scarcity rather than double-counting it here.
    if (pos === 'QB' && passTd >= 6) s *= 1.06;
    if (pos === 'WR' || pos === 'TE') { if (ppr != null && ppr >= 1) s *= 1.02; }
    else if (pos === 'RB' && ppr != null && ppr <= 0) s *= 1.03;

    if (totalPicks && totalPicks > 1 && pickNo) {
      var depth = Math.min(0.98, (pickNo - 1) / totalPicks);
      var par = Math.max(0.40, 1.0 - depth * 0.44);
      s = s / par;
    }

    // Display relabel (monotonic): everything above is the backtested ranking
    // and is left untouched. This only stretches the near-ceiling band so the
    // best pick's 0-100 number differentiates instead of clustering at ~97.
    // Scores under the knee (~85) are unchanged; above it the curve is steeper
    // than 1:1, so a truly elite pick pulls toward 100 while a merely-good "best
    // available" reads lower. Monotonic => it never changes which pick ranks
    // higher, only the label. Keep identical to utils/pick_score.py.
    var d = clamp01(s);
    var _knee = 0.85;
    if (d > _knee) {
      var _t = (d - _knee) / (1 - _knee);
      d = _knee + _t * _t * (1 - _knee);
    }
    return Math.floor(d * 100 + 0.5);
  }

  // Effective starters per position from a league's roster slot counts, used to
  // anchor VOR and PPG replacement levels. Shared so the draft room and the
  // server derive the SAME replacement index from the SAME roster (the WR/RB
  // gap between them was the main cause of mismatched grades). Splits SF half to
  // QB and FLEX half each to RB/WR, matching the draft room's computeReplacement.
  //   counts: { QB, SF, RB, WR, TE, FLEX }  ->  { QB, RB, WR, TE } (floats)
  function starterCounts(counts) {
    var c = counts || {};
    var n = function (k) { return +c[k] || 0; };
    return {
      QB: n('QB') + n('SF') * 0.5,
      RB: n('RB') + n('FLEX') * 0.5 + n('RB_WR') * 0.5 + n('RB_TE') * 0.5,
      WR: n('WR') + n('FLEX') * 0.5 + n('RB_WR') * 0.5 + n('WR_TE') * 0.5,
      TE: n('TE') + n('WR_TE') * 0.5 + n('RB_TE') * 0.5,
    };
  }

  return { computePickScore: computePickScore, starterCounts: starterCounts };
});
