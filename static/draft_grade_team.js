// Single source of truth for the startup/redraft team-grade composite.
//
// Mirrors utils/draft_grade.py::dr_team_grade_score (and its dr_optimal_lineup /
// dr_slot_eligible / dr_lineup_score / dr_avg_top_n helpers). The draft room
// (browser) and the Teams-page /api/draft-grades (Python) both compute the team
// letter from this same math; tests/test_team_grade_parity.py runs this via Node
// against the Python copy and fails CI on any drift.
//
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

  function teamGradeComposite(picks, slots, targets, numTeams, draftType, leaguePpg, leagueVal) {
    if (!picks || !picks.length) return null;
    var starterIds = optimalLineupIds(picks, slots);
    var nStartFilled = 0;
    for (var _k in starterIds) if (starterIds.hasOwnProperty(_k)) nStartFilled++;
    var coverage = slots.length ? nStartFilled / slots.length : 0;

    // 1) Starter quality: round-weighted (1/round^0.60) avg PS of starters.
    var wSum = 0, wTot = 0;
    var avgPsVals = picks.filter(function (p) { return p.ps != null; }).map(function (p) { return +p.ps; });
    var avgPs = avgPsVals.length ? avgPsVals.reduce(function (a, b) { return a + b; }, 0) / avgPsVals.length : null;
    picks.forEach(function (x) {
      if (!starterIds[String(x.id)] || x.ps == null) return;
      var r = Math.max(1, Math.ceil((x.pn || 1) / Math.max(numTeams, 1)));
      var wt = 1.0 / Math.pow(r, 0.60);
      wSum += (+x.ps) * wt; wTot += wt;
    });
    var starterAvgPs = wTot > 0 ? wSum / wTot : avgPs;
    var valuePts = starterAvgPs != null ? rnd(clamp01((starterAvgPs || 0) / 100) * 35) : 17;

    // 2) Starting-lineup strength vs a league-average team.
    var starterArr = picks.filter(function (p) { return starterIds[String(p.id)]; });
    var nStart = Math.max(numTeams, 1) * slots.length;
    var myPpgs = starterArr.filter(function (p) { return p.ppg != null; }).map(function (p) { return +p.ppg; });
    var ppgRatio = null;
    if (myPpgs.length >= Math.max(2, Math.floor(starterArr.length * 0.5))) {
      var myPpgAvg = myPpgs.reduce(function (a, b) { return a + b; }, 0) / myPpgs.length;
      var leaguePpgAvg = avgTopN(leaguePpg || [], nStart);
      if (leaguePpgAvg > 0) ppgRatio = myPpgAvg / leaguePpgAvg;
    }
    var myValAvg = starterArr.length
      ? starterArr.reduce(function (a, p) { return a + (+p.val || 0); }, 0) / starterArr.length : 0;
    var leagueValAvg = avgTopN(leagueVal || [], nStart);
    var valueRatio = leagueValAvg > 0 ? myValAvg / leagueValAvg : null;
    var strengthRatio;
    if (draftType === 'redraft') {
      strengthRatio = ppgRatio != null ? ppgRatio : (valueRatio != null ? valueRatio : 0.80);
    } else if (ppgRatio != null && valueRatio != null) {
      strengthRatio = 0.6 * ppgRatio + 0.4 * valueRatio;
    } else {
      strengthRatio = ppgRatio != null ? ppgRatio : (valueRatio != null ? valueRatio : 0.80);
    }
    var starterPts = rnd(clamp01((strengthRatio - 0.80) / 0.40) * 35);

    // 3) Construction: coverage + balance + efficiency.
    var counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
    picks.forEach(function (p) {
      var pos = String(p.pos || '').toUpperCase();
      if (counts[pos] != null) counts[pos]++;
    });
    var bsum = 0, usefulPicks = 0, gradedPicks = 0;
    ['QB', 'RB', 'WR', 'TE'].forEach(function (pos) {
      var t = targets[pos] || 0;
      bsum += t ? Math.min(counts[pos], t) / t : 0;
      usefulPicks += Math.min(counts[pos], t + 1);
      gradedPicks += counts[pos];
    });
    var efficiency = gradedPicks > 0 ? usefulPicks / gradedPicks : 1;
    var constructionRaw = clamp01(0.45 * coverage + 0.30 * (bsum / 4) + 0.25 * efficiency);
    var ramp = Math.min(1, picks.length / 8);
    var balancePts = rnd(((1 - ramp) * 0.85 + ramp * constructionRaw) * 30);

    return {
      total: valuePts + starterPts + balancePts,
      value: valuePts, starter: starterPts, balance: balancePts,
      strengthRatio: strengthRatio, avgPs: avgPs, starterIds: starterIds,
    };
  }

  return {
    teamGradeComposite: teamGradeComposite,
    optimalLineupIds: optimalLineupIds,
  };
});
