// Single source of truth for the draft pick-score FORMULA.
//
// Mirrors utils/pick_score.py::compute_pick_score exactly. The draft room
// (browser) and the Teams-page /api/draft-grades (Python) both grade picks with
// this same math, so a pick's grade can't differ between the two surfaces.
// tests/test_pick_score_parity.py runs this (via Node) against the Python copy
// over random inputs and fails CI on any divergence.
//
// This is the pure math only. Callers gather the inputs (value, vor, tier, adp,
// need, ppg, ...); the live draft-room recommendation additionally layers the
// survival/handcuff timing terms via survivalAdj/handcuff, which grading never
// uses (so grades match the server, which never passes them).
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BRPickScore = api;
})(typeof self !== 'undefined' ? self : this, function () {
  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

  var WEIGHTS = {
    rookie:  { vor: 0.06, value: 0.18, adp: 0.29, tier: 0.12, need: 0.05, youth: 0.24, mom: 0.06, ppg: 0.05 },
    redraft: { vor: 0.10, value: 0.24, adp: 0.33, tier: 0.08, need: 0.07, youth: 0.00, mom: 0.03, ppg: 0.18 },
    startup: { vor: 0.07, value: 0.24, adp: 0.30, tier: 0.12, need: 0.09, youth: 0.10, mom: 0.03, ppg: 0.10 },
  };
  var AGE_PEAKS = { RB: 24, WR: 27, TE: 27, QB: 29 };
  var CORE = { QB: 1, RB: 1, WR: 1, TE: 1 };

  // o: { pos, value, vor, tier, age, rankChange7d, avgPick, pickNo, maxVal,
  //      draftType, isSf, needRaw, qbCount, totalPicks, numTeams, ppgNorm,
  //      ppr, tep, isTierCliff, survivalAdj, handcuff }
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

    var dbValueNorm = (maxVal && maxVal > 0) ? clamp01(value / maxVal) : 0;
    var valueNorm;
    if (avgPick != null && totalPicks > 0) {
      var adpQualNorm = clamp01(1 - avgPick / totalPicks);
      valueNorm = dbValueNorm * 0.35 + adpQualNorm * 0.65;
    } else {
      valueNorm = dbValueNorm;
    }
    var vorNorm = (vor != null) ? clamp01(vor / Math.max(maxVal, 1)) : valueNorm * 0.8;

    var adpVal;
    if (avgPick != null) {
      var rel = (pickNo - avgPick) / Math.max(avgPick, 1.5);
      if (rel >= 0.5) adpVal = 1.0;
      else if (rel >= -0.3) adpVal = 0.5 + rel;
      else adpVal = Math.max(0, 0.2 + rel * 0.25);
      if (avgPick <= 8) adpVal = Math.max(adpVal, clamp01(0.5 + (8 - avgPick) / 16));
    } else {
      adpVal = 0.5;
    }

    var tierScore = tier ? clamp01((10 - Math.min(tier, 9)) / 9) : valueNorm;
    if (o.isTierCliff) tierScore = clamp01(tierScore + 0.15);

    var needRamp = clamp01((pickNo - 1) / 12.0);
    var need = (1 - needRamp) * 0.5 + needRamp * needRaw;

    var youth = 0.5;
    if (age != null && CORE[pos]) {
      var peak = AGE_PEAKS[pos] || 27;
      youth = clamp01((peak - age + 4) / 8);
    }
    var mom = clamp01((rankChange7d || 0) / 20 + 0.5);
    var ppgN = o.ppgNorm != null ? +o.ppgNorm : valueNorm;

    var w = WEIGHTS[o.draftType] || WEIGHTS.startup;
    var s = w.vor * vorNorm + w.value * valueNorm + w.adp * adpVal
          + w.tier * tierScore + w.need * need + w.youth * youth
          + w.mom * mom + (w.ppg || 0) * ppgN;

    // Live-draft timing term (survival to next pick). Grading passes 0.
    if (o.survivalAdj) s += o.survivalAdj;

    if (!o.isSf && pos === 'QB' && (o.qbCount || 0) >= 1) {
      var teams = o.numTeams ? +o.numTeams : 12;
      var round = pickNo ? Math.floor((pickNo - 1) / Math.max(teams, 1)) + 1 : 1;
      var pen = round <= 3 ? 0.30 : (round <= 6 ? 0.60 : (round <= 9 ? 0.85 : 1.0));
      if ((o.qbCount || 0) >= 2) pen *= 0.7;
      s *= pen;
    }

    // Live-draft redraft handcuff term. Grading passes false.
    if (o.handcuff) s = Math.min(1, s + 0.15);

    if (tep && tep > 0 && pos === 'TE') s *= (1 + 0.12 * tep);
    if (pos === 'WR' || pos === 'TE') { if (ppr != null && ppr >= 1) s *= 1.02; }
    else if (pos === 'RB' && ppr != null && ppr <= 0) s *= 1.03;

    if (totalPicks && totalPicks > 1 && pickNo) {
      var depth = Math.min(0.98, (pickNo - 1) / totalPicks);
      var par = Math.max(0.40, 1.0 - depth * 0.44);
      s = s / par;
    }

    return Math.floor(clamp01(s) * 100 + 0.5);
  }

  return { computePickScore: computePickScore };
});
