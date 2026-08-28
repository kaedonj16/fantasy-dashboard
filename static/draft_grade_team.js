// Single source of truth for the startup/redraft team-grade composite.
//
// Mirrors utils/draft_grade.py::dr_team_grade_score (and its dr_optimal_lineup /
// dr_slot_eligible / dr_lineup_score / dr_avg_top_n helpers). The draft room
// (browser) and the Teams-page /api/draft-grades (Python) both compute the team
// letter from this same math; tests/test_team_grade_parity.py runs this via Node
// against the Python copy and fails CI on any drift.
//
// Caps: startup 35/25/40 (value/starters/construction), redraft 20/50/30.
// picks items: { id, pos, ps, pn, val, ppg }. Returns the raw 0-100 composite
// breakdown { total, value, starter, balance, starterIds } or null.
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.BRTeamGrade = api;
})(typeof self !== 'undefined' ? self : this, function () {
  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
  function rnd(x) { return Math.floor(x + 0.5); }  // half-up, matching Python floor(x+0.5)

  function slotEligible(slot, pos) {
    pos = (pos || '').toUpperCase();
    if (slot === 'FLEX') return pos === 'RB' || pos === 'WR' || pos === 'TE';
    if (slot === 'SF') return pos === 'QB' || pos === 'RB' || pos === 'WR' || pos === 'TE';
    return slot === pos;
  }

  function lineupScore(p) {
    if (!p) return -Infinity;
    if (p.ppg != null) return +p.ppg;
    return (p.val != null ? +p.val : 0) / 1000;
  }

  // Fill the most restrictive slots first with the highest-lineupScore eligible
  // player. Returns a set (object) of starter player-id strings.
  function optimalLineupIds(picks, slots) {
    var flex = { SF: 3, FLEX: 2 };
    var order = slots.map(function (s, i) { return { slot: s, i: i }; });
    order.sort(function (a, b) { return (flex[a.slot] || 1) - (flex[b.slot] || 1) || a.i - b.i; });
    var used = {}, ids = {};
    order.forEach(function (o) {
      var best = -1, bestScore = -Infinity;
      for (var j = 0; j < picks.length; j++) {
        if (used[j] || !slotEligible(o.slot, String(picks[j].pos || ''))) continue;
        var sc = lineupScore(picks[j]);
        if (sc > bestScore) { bestScore = sc; best = j; }
      }
      if (best >= 0) { used[best] = true; ids[String(picks[best].id)] = true; }
    });
    return ids;
  }

  function avgTopN(arr, n) {
    if (!arr.length || n <= 0) return 0;
    var s = arr.slice().sort(function (a, b) { return b - a; }).slice(0, n);
    return s.length ? s.reduce(function (a, b) { return a + b; }, 0) / s.length : 0;
  }

  // Build a roster-valid league-wide starting field. A global top-N baseline is
  // position-blind and, for PPG, becomes an imaginary lineup dominated by QBs.
  function leagueLineupAvg(players, slots, numTeams, metric) {
    var eligible = [];
    (players || []).forEach(function (p, i) {
      if (p == null || p[metric] == null) return;
      eligible.push({ id: 'league-' + i, pos: p.pos, ppg: +p[metric] });
    });
    if (!eligible.length || !slots.length) return null;
    var leagueSlots = [];
    for (var n = 0; n < Math.max(+numTeams || 1, 1); n++) leagueSlots = leagueSlots.concat(slots);
    var selected = optimalLineupIds(eligible, leagueSlots);
    var vals = eligible.filter(function (p) { return selected[String(p.id)]; })
      .map(function (p) { return p.ppg; });
    return vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
  }

  // Value / Starters / Construction caps. Must match utils/draft_grade.py
  // DR_SPLIT_* / DR_CONSTRUCTION_* (pinned by tests/test_team_grade_parity.py).
  var SPLIT_STARTUP = [35, 25, 40];
  var SPLIT_REDRAFT = [20, 50, 30];
  var CONSTRUCTION_STARTUP = [0.45, 0.30, 0.25];
  var CONSTRUCTION_REDRAFT = [0.70, 0.20, 0.10];

  function teamGradeComposite(picks, slots, targets, numTeams, draftType, leaguePpg, leagueVal, leaguePlayers, options) {
    if (!picks || !picks.length) return null;
    var redraft = draftType === 'redraft';
    var split = redraft ? SPLIT_REDRAFT : SPLIT_STARTUP;
    var valueWeight = split[0], starterWeight = split[1], balanceWeight = split[2];
    var starterIds = optimalLineupIds(picks, slots);
    var nStartFilled = 0;
    for (var _k in starterIds) if (starterIds.hasOwnProperty(_k)) nStartFilled++;
    var coverage = slots.length ? nStartFilled / slots.length : 0;
    options = options || {};
    var sf = slots.indexOf('SF') >= 0 || !!options.sf, tep = +options.tep || 0;
    var benchByPos = { QB: [], RB: [], WR: [], TE: [] };
    picks.forEach(function(p){
      var pos = String(p.pos || '').toUpperCase();
      if (benchByPos[pos] && !starterIds[String(p.id)]) benchByPos[pos].push(p);
    });
    Object.keys(benchByPos).forEach(function(pos){ benchByPos[pos].sort(function(a,b){ return lineupScore(b)-lineupScore(a); }); });
    function benchUtility(p) {
      var pos = String(p.pos || '').toUpperCase(), idx = benchByPos[pos] ? benchByPos[pos].indexOf(p) : -1;
      if (pos === 'QB') return idx === 0 ? (sf ? 0.78 : 0.32) : (sf ? 0.55 : 0.12);
      if (pos === 'TE') return idx === 0 ? (tep > 0 ? 0.72 : 0.32) : (tep > 0 ? 0.48 : 0.16);
      if (pos === 'RB') return idx === 0 ? 0.82 : 0.68;
      if (pos === 'WR') return idx === 0 ? 0.78 : 0.64;
      return 0;
    }
    var hasFlex = slots.indexOf('FLEX') >= 0;
    function roleOf(p) {
      if (starterIds[String(p.id)]) return 'starter';
      var pos = String(p.pos || '').toUpperCase();
      var arr = benchByPos[pos] || [];
      var idx = arr.indexOf(p);
      // RB3/WR4 (first bench) are primary cover. A second RB/WR is still
      // primary when FLEX exists — that is the injury/bye path, not QB2/TE2.
      if (pos === 'RB' || pos === 'WR') {
        if (idx === 0) return 'primary';
        if (idx === 1 && hasFlex) return 'primary';
        return 'fringe';
      }
      return idx === 0 ? 'primary' : 'fringe';
    }

    // 1) Starter quality: round-weighted (1/round^0.60) avg PS of starters.
    var wSum = 0, wTot = 0;
    var avgPsVals = picks.filter(function (p) { return p.ps != null; }).map(function (p) { return +p.ps; });
    var avgPs = avgPsVals.length ? avgPsVals.reduce(function (a, b) { return a + b; }, 0) / avgPsVals.length : null;
    picks.forEach(function (x) {
      if (x.ps == null || ['K','DEF','DST','D/ST'].indexOf(String(x.pos || '').toUpperCase()) >= 0) return;
      var r = Math.max(1, Math.ceil((x.pn || 1) / Math.max(numTeams, 1)));
      var role = roleOf(x), rw = role === 'starter' ? 1 : role === 'primary' ? 0.55 : 0.18;
      var util = role === 'starter' ? 1 : benchUtility(x);
      var wt = (1 / Math.pow(1 + (r - 1) / 5, 0.85)) * rw * (0.55 + 0.45 * util);
      wSum += (+x.ps) * wt; wTot += wt;
    });
    var starterAvgPs = wTot > 0 ? wSum / wTot : avgPs;
    var valuePts = starterAvgPs != null
      ? rnd(clamp01((starterAvgPs || 0) / 100) * valueWeight)
      : Math.floor(valueWeight / 2);

    // 2) Starting-lineup strength vs a league-average team.
    var starterArr = picks.filter(function (p) { return starterIds[String(p.id)]; });
    var nStart = Math.max(numTeams, 1) * slots.length;
    var myPpgs = starterArr.filter(function (p) { return p.ppg != null; }).map(function (p) { return +p.ppg; });
    var ppgRatio = null;
    if (myPpgs.length >= Math.max(2, Math.floor(starterArr.length * 0.5))) {
      var myPpgAvg = myPpgs.reduce(function (a, b) { return a + b; }, 0) / myPpgs.length;
      var leaguePpgAvg = leagueLineupAvg(leaguePlayers, slots, numTeams, 'ppg');
      if (leaguePpgAvg == null) leaguePpgAvg = avgTopN(leaguePpg || [], nStart);
      if (leaguePpgAvg > 0) {
        ppgRatio = myPpgAvg / leaguePpgAvg;
        // Redraft playoff odds sum every starting slot (empty = 0). Scale the
        // filled-starter average by coverage so a finished stars-and-scrubs
        // roster with holes doesn't outrank a complete one on mean PPG alone.
        // Only apply once the team has had enough picks to fill those slots —
        // mid-draft every roster has holes, and raw coverage (2/8 at the start
        // of round 3) zeros the 50-pt starter term and prints F for the whole
        // league.
        if (redraft && slots.length && picks.length >= slots.length) {
          ppgRatio *= coverage;
        }
      }
    }
    var myValAvg = starterArr.length
      ? starterArr.reduce(function (a, p) { return a + (+p.val || 0); }, 0) / starterArr.length : 0;
    var leagueValAvg = leagueLineupAvg(leaguePlayers, slots, numTeams, 'val');
    if (leagueValAvg == null) leagueValAvg = avgTopN(leagueVal || [], nStart);
    var valueRatio = leagueValAvg > 0 ? myValAvg / leagueValAvg : null;
    var strengthRatio;
    if (redraft) {
      strengthRatio = ppgRatio != null ? ppgRatio : (valueRatio != null ? valueRatio : 0.80);
    } else if (ppgRatio != null && valueRatio != null) {
      strengthRatio = 0.6 * ppgRatio + 0.4 * valueRatio;
    } else {
      strengthRatio = ppgRatio != null ? ppgRatio : (valueRatio != null ? valueRatio : 0.80);
    }
    var starterPts = rnd(clamp01((strengthRatio - 0.80) / 0.40) * starterWeight);

    // 3) Construction: coverage + functional cover + efficient bench use.
    var counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
    picks.forEach(function (p) {
      var pos = String(p.pos || '').toUpperCase();
      if (counts[pos] != null) counts[pos]++;
    });
    var bench = picks.filter(function(p){ return !starterIds[String(p.id)] && ['QB','RB','WR','TE'].indexOf(String(p.pos || '').toUpperCase()) >= 0; });
    var utilVals = bench.map(benchUtility), efficiency = utilVals.length ? utilVals.reduce(function(a,b){return a+b;},0)/utilVals.length : 1;
    var primary = bench.filter(function(p){ return roleOf(p) === 'primary'; });
    var functionalDepth = primary.length ? primary.map(benchUtility).reduce(function(a,b){return a+b;},0)/primary.length : 0;
    var constructionRaw = redraft
      ? clamp01(0.45 * coverage + 0.35 * functionalDepth + 0.20 * efficiency)
      : (function(){
          var bsum=0,useful=0,graded=0; ['QB','RB','WR','TE'].forEach(function(pos){var t=targets[pos]||0;bsum+=t?Math.min(counts[pos],t)/t:0;useful+=Math.min(counts[pos],t+1);graded+=counts[pos];});
          var mix=CONSTRUCTION_STARTUP; return clamp01(mix[0]*coverage+mix[1]*(bsum/4)+mix[2]*(graded?useful/graded:1));
        })();
    var ramp = Math.min(1, picks.length / 8);
    var balancePts = rnd(((1 - ramp) * 0.85 + ramp * constructionRaw) * balanceWeight);

    return {
      total: valuePts + starterPts + balancePts,
      value: valuePts, starter: starterPts, balance: balancePts,
      strengthRatio: strengthRatio, avgPs: avgPs, starterIds: starterIds,
      functionalDepth: functionalDepth, benchEfficiency: efficiency,
    };
  }

  return {
    teamGradeComposite: teamGradeComposite,
    optimalLineupIds: optimalLineupIds,
    leagueLineupAvg: leagueLineupAvg,
    SPLIT_STARTUP: SPLIT_STARTUP,
    SPLIT_REDRAFT: SPLIT_REDRAFT,
  };
});
