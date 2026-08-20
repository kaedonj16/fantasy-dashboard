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

  // League-relative depth targets.  Single-start, non-flex positions do not get
  // a backup merely because Math.round assigned them a slice of the bench.
  // Instead the bench follows paths into the weekly lineup: RB/WR get the bulk,
  // while SF and TEP explicitly create useful QB/TE depth.
  function posTargets(rc, tep) {
    rc = rc || {}; tep = tep || 0;
    var flex = rc.FLEX || 0, sf = rc.SF || 0, bn = rc.BN || 0;
    var benchEff = Math.min(bn, 8);
    var rbDepth = Math.ceil(benchEff * 0.45);
    var wrDepth = Math.floor(benchEff * 0.45);
    var t = {
      QB: (rc.QB || 0) + sf + (sf && benchEff >= 5 ? 1 : 0),
      RB: (rc.RB || 0) + flex + rbDepth,
      WR: (rc.WR || 0) + wrDepth,
      TE: (rc.TE || 0) + (tep > 0 && benchEff >= 5 ? 1 : 0),
    };
    var cap = { QB: sf ? 4 : Math.max(1, rc.QB || 0), RB: 7, WR: 7,
                TE: tep > 0 ? Math.max(3, rc.TE || 0) : Math.max(1, rc.TE || 0) };
    Object.keys(cap).forEach(function (k) { if (t[k] > cap[k]) t[k] = cap[k]; });
    if (rc.K) t.K = rc.K;
    if (rc.DEF) t.DEF = rc.DEF;
    return t;
  }

  function starterRequirements(rc, sf) {
    rc = rc || {};
    return { QB: (rc.QB || 0) + (sf ? (rc.SF || 0) : 0), RB: rc.RB || 0,
             WR: rc.WR || 0, TE: rc.TE || 0, FLEX: rc.FLEX || 0,
             K: rc.K || 0, DEF: rc.DEF || 0 };
  }

  // Returns the role occupied by the *next* player at pos. FLEX is deliberately
  // allocated only after dedicated RB/WR/TE slots; that is what makes RB3/WR3 a
  // starter while QB2 in 1QB is backup-only.
  function rosterRole(pos, counts, rc, sf) {
    pos = String(pos || '').toUpperCase(); counts = counts || {}; rc = rc || {};
    var have = +counts[pos] || 0, req = starterRequirements(rc, sf);
    if (have < (req[pos] || 0)) return 'starter';
    if ((pos === 'RB' || pos === 'WR' || pos === 'TE') && req.FLEX > 0) {
      var flexUsed = Math.max(0, (+counts.RB || 0) - req.RB)
                   + Math.max(0, (+counts.WR || 0) - req.WR)
                   + Math.max(0, (+counts.TE || 0) - req.TE);
      if (flexUsed < req.FLEX) return 'flex';
    }
    return have === (req[pos] || 0) ? 'bench1' : 'bench2';
  }

  function rosterSlotUtility(pos, counts, rc, opts) {
    opts = opts || {}; var role = rosterRole(pos, counts, rc, !!opts.sf);
    if (role === 'starter') return 1;
    if (role === 'flex') return String(pos || '').toUpperCase() === 'TE'
      ? ((+opts.tep || 0) > 0 ? 0.86 : 0.70) : 0.96;
    var dynasty = opts.draftType === 'startup' || opts.draftType === 'dynasty';
    var tep = +opts.tep || 0, p = String(pos || '').toUpperCase();
    if (p === 'QB') return role === 'bench1' ? (opts.sf ? 0.82 : (dynasty ? 0.70 : 0.42)) : (opts.sf ? 0.55 : 0.18);
    if (p === 'TE') return role === 'bench1' ? (tep > 0 ? 0.72 : (dynasty ? 0.62 : 0.44)) : (tep > 0 ? 0.48 : 0.20);
    if (p === 'RB') return role === 'bench1' ? 0.82 : 0.68;
    if (p === 'WR') return role === 'bench1' ? 0.78 : 0.64;
    return 1;
  }

  function remainingObligations(counts, rc, remainingPicks, sf) {
    counts = counts || {}; rc = rc || {}; var req = starterRequirements(rc, sf);
    var missing = { QB: Math.max(0, req.QB - (+counts.QB || 0)), RB: Math.max(0, req.RB - (+counts.RB || 0)),
      WR: Math.max(0, req.WR - (+counts.WR || 0)), TE: Math.max(0, req.TE - (+counts.TE || 0)),
      K: Math.max(0, req.K - (+counts.K || 0)), DEF: Math.max(0, req.DEF - (+counts.DEF || 0)) };
    var flexUsed = Math.max(0, (+counts.RB || 0) - req.RB) + Math.max(0, (+counts.WR || 0) - req.WR)
                 + Math.max(0, (+counts.TE || 0) - req.TE);
    missing.FLEX = Math.max(0, req.FLEX - flexUsed);
    var required = 0; Object.keys(missing).forEach(function(k){ required += missing[k]; });
    return { missing: missing, required: required, remaining: Math.max(0, +remainingPicks || 0),
             freePicks: Math.max(0, (+remainingPicks || 0) - required) };
  }

  // Pure, testable final layer used only for live recommendations. A great fall
  // can overcome fit, but ordinary backup-only value pays a persistent cost.
  function decisionScore(o) {
    o = o || {}; var base = +o.base || 0, util = o.utility == null ? 1 : +o.utility;
    var score = base + (util - 1) * 38;
    if (o.bench) {
      score += (+o.quality || 0) * 5;
      if ((+o.required || 0) > 0 && (+o.freePicks || 0) <= 1) score -= 7;
      if ((+o.required || 0) > 0 && (+o.freePicks || 0) <= 0) score -= 13;
      if (o.deepBench) score -= 5;
    }
    if ((+o.waitLoss || 0) > 0) score += Math.min(9, (+o.waitLoss || 0) * 0.30) * Math.max(0.35, util);
    return Math.max(1, Math.min(99, Math.round(score)));
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
    starterRequirements: starterRequirements, rosterRole: rosterRole,
    rosterSlotUtility: rosterSlotUtility, remainingObligations: remainingObligations,
    decisionScore: decisionScore,
  };
});
