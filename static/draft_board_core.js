// Shared draft-board primitives: the value / ADP / replacement accessors that
// turn an /api/league-players row into the inputs BRPickScore.computePickScore
// needs. Extracted so the Draft Cheat Sheet and the Draft Room derive VOR,
// replacement level and tiers from ONE implementation instead of two copies that
// can drift. Every function is pure and takes its league context explicitly
// (mode 'redraft'|'dynasty', sf bool, teams, rosterPositions), so it has no
// dependence on any page's local state.
//
// These mirror the corresponding helpers in static/draft_room.js exactly
// (redraftVal / valOf / adpOf / computeReplacement / ppg scale / tierOf); keep
// them in lockstep. Replacement and the PPG scale both anchor on
// BRPickScore.starterCounts, the same roster-derived starter index the server
// pick-score uses, which is what lets the two surfaces agree.
(function (root, factory) {
  var api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DraftBoardCore = api;
})(typeof self !== 'undefined' ? self : this, function (root) {
  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
  function PS() { return root.BRPickScore; }

  var SLOT_MAP = {
    QB: 'QB', RB: 'RB', WR: 'WR', TE: 'TE',
    FLEX: 'FLEX', WRRB_FLEX: 'FLEX', REC_FLEX: 'FLEX', WRRBTE_FLEX: 'FLEX',
    SUPER_FLEX: 'SF', SFLEX: 'SF',
  };

  // Raw starter counts {QB,SF,RB,WR,TE,FLEX} from a roster_positions array,
  // reconciled to the QB format (mirror of draft_room.js rosterFromLeague plus
  // the defaultRoster SF reconcile). Falls back to a standard 1QB/SF shape.
  function rosterCounts(rosterPositions, sf) {
    var lg = null;
    if (rosterPositions && rosterPositions.length) {
      lg = { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0 };
      rosterPositions.forEach(function (s) { var k = SLOT_MAP[String(s).toUpperCase()]; if (k) lg[k]++; });
      if (!(lg.QB + lg.RB + lg.WR + lg.TE + lg.FLEX + lg.SF)) lg = null;
    }
    if (!lg) lg = { QB: 1, SF: 0, RB: 2, WR: 3, TE: 1, FLEX: 1 };
    if (sf) { if (!lg.SF) lg.SF = 1; if (!lg.FLEX) lg.FLEX = 1; }
    else { lg.SF = 0; }
    return lg;
  }

  function redraftVal(p, sf) {
    return (sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
               : p.redraft_value_1qb) || 0;
  }
  function dynVal(p, sf) { return (sf ? (p.sf_value || p.value) : p.value) || 0; }
  function valOf(p, mode, sf) { return mode === 'dynasty' ? dynVal(p, sf) : redraftVal(p, sf); }

  function adpOf(p, mode, sf) {
    if (mode === 'dynasty') { var a = sf ? p.sf_avg_pick : p.avg_pick; return a != null ? Number(a) : null; }
    var r = sf ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
    if (r != null) return Number(r);
    return p._radp != null ? Number(p._radp) : null;   // value-derived redraft rank
  }

  // Effective starters {QB,RB,WR,TE} from a roster_positions array.
  function startersFor(rosterPositions, sf) { return PS().starterCounts(rosterCounts(rosterPositions, sf)); }

  // Replacement level = value of the last startable player at a position across
  // the league (teams x starters), from the FULL pool so it is a fixed baseline.
  //
  // Accessor-based so BOTH surfaces share this exact kernel: the caller passes
  // its own value function and its own effective-starters map (from
  // BRPickScore.starterCounts). The draft room passes its state-aware valOf and
  // its user-edited roster; the cheat sheet passes the league roster.
  function computeReplacement(pool, valFn, starters, teams) {
    var byPos = { QB: [], RB: [], WR: [], TE: [] };
    pool.forEach(function (p) { var pos = (p.position || '').toUpperCase(); if (byPos[pos]) byPos[pos].push(valFn(p)); });
    var r = {};
    Object.keys(byPos).forEach(function (pos) {
      var arr = byPos[pos].sort(function (a, b) { return b - a; });
      if (!arr.length) { r[pos] = 0; return; }
      var idx = Math.round((teams || 12) * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      r[pos] = arr[idx];
    });
    return r;
  }

  function ppgOf(p) { return (p && p.proj_ppg != null) ? Number(p.proj_ppg) : ((p && p.ppg != null) ? Number(p.ppg) : null); }

  // Position PPG scale (replacement -> ~0, elite -> ~1) for the production term.
  // Accessor-based for the same reason as computeReplacement.
  function computePpgScale(pool, ppgFn, starters, teams) {
    ppgFn = ppgFn || ppgOf;
    var byPos = { QB: [], RB: [], WR: [], TE: [] };
    pool.forEach(function (p) { var pos = (p.position || '').toUpperCase(); var v = ppgFn(p); if (byPos[pos] && v != null) byPos[pos].push(v); });
    var out = {};
    Object.keys(byPos).forEach(function (pos) {
      var arr = byPos[pos]; if (!arr.length) return;
      arr.sort(function (a, b) { return b - a; });
      var topN = Math.max(1, Math.min(3, arr.length)); var s = 0; for (var i = 0; i < topN; i++) s += arr[i];
      var elite = s / topN;
      var idx = Math.round((teams || 12) * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      out[pos] = { repl: arr[idx], elite: elite };
    });
    return out;
  }
  function ppgNorm(p, scale, ppgFn) {
    ppgFn = ppgFn || ppgOf;
    var pos = (p.position || '').toUpperCase(); var v = ppgFn(p); var sc = scale[pos];
    if (v == null || !sc) return null;
    var span = sc.elite - sc.repl;
    if (span <= 0) return clamp01(v / Math.max(sc.elite, 1));
    return clamp01((v - sc.repl) / span);
  }

  // Depth targets per position from a roster-counts map (mirror of the draft
  // room's posTargets). Suggestions, not hard needs: bench depth is capped and
  // each position is held to a realistic ceiling. Shared so the cheat sheet's
  // roster-need shading uses the same targets the draft room grades against.
  function posTargets(rc, tep) {
    rc = rc || {}; tep = tep || 0;
    var flex = rc.FLEX || 0, sf = rc.SF || 0, bn = rc.BN || 0;
    var benchEff = Math.min(bn, 7);
    var t = {
      QB: (rc.QB || 0) + sf + Math.round(benchEff * 0.10),
      RB: (rc.RB || 0) + flex + Math.round(benchEff * 0.35),
      WR: (rc.WR || 0) + Math.round(benchEff * 0.40),
      TE: (rc.TE || 0) + (tep > 0 ? Math.round(benchEff * 0.15) : 0),
    };
    if (tep > 0) t.TE += 1;
    var cap = { QB: sf ? 3 : 2, RB: 6, WR: 6, TE: tep > 0 ? 3 : 2 };
    Object.keys(cap).forEach(function (k) { if (t[k] > cap[k]) t[k] = cap[k]; });
    if (rc.K) t.K = rc.K;
    if (rc.DEF) t.DEF = rc.DEF;
    return t;
  }

  // Dynasty tier from the server value thresholds; redraft returns null because
  // the tier table is keyed to dynasty value (mirrors draft_room.js tierOf).
  function tierOf(p, mode, sf, teams, tierThresholds) {
    var pos = (p.position || '').toUpperCase();
    if (pos === 'K' || pos === 'DEF') return null;
    if (mode !== 'dynasty') return null;
    var lt = sf ? 'sf' : '1qb', sz = String(teams);
    var tt = tierThresholds || {};
    var tbl = (tt[lt] || {})[sz] || (tt[lt] || {})['12'] || (tt['1qb'] || {})['12'] || (tt['1qb'] || {})['10'] || [];
    if (!tbl.length) return null;
    var v = valOf(p, mode, sf);
    for (var i = 0; i < tbl.length; i++) { if (v >= tbl[i]) return i + 1; }
    return tbl.length + 1;
  }

  function maxVal(pool, mode, sf) {
    var m = 0; pool.forEach(function (p) { var v = valOf(p, mode, sf); if (v > m) m = v; }); return m;
  }

  return {
    rosterCounts: rosterCounts, startersFor: startersFor,
    redraftVal: redraftVal, dynVal: dynVal, valOf: valOf, adpOf: adpOf,
    computeReplacement: computeReplacement, ppgOf: ppgOf,
    computePpgScale: computePpgScale, ppgNorm: ppgNorm,
    tierOf: tierOf, maxVal: maxVal, posTargets: posTargets,
  };
});
