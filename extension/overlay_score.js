// Live recommendation ranker for the docked overlay. Uses the same kernels as
// Draft Room: BRPickScore (pick quality) + DraftBoardCore.decisionScore (rec).
(function (root) {
  "use strict";

  function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
  function Core() { return root.DraftBoardCore; }
  function PS() { return root.BRPickScore; }

  function defaultRoster(sf) {
    return { QB: 1, SF: sf ? 1 : 0, RB: 2, WR: 3, TE: 1, FLEX: 1, K: 1, DEF: 1, BN: 6 };
  }

  function rosterHasStarters(rs) {
    if (!rs) return false;
    return (rs.QB || 0) + (rs.RB || 0) + (rs.WR || 0) + (rs.TE || 0) + (rs.FLEX || 0) + (rs.SF || 0) >= 4;
  }

  function rosterOf(ctx) {
    const base = defaultRoster(!!(ctx && ctx.sf));
    if (ctx && rosterHasStarters(ctx.roster)) {
      return Object.assign({}, base, ctx.roster);
    }
    return base;
  }

  function ownerOf(pn, teams) {
    const n = teams || 12;
    const r = Math.ceil(pn / n);
    const i = (pn - 1) % n;
    return (r % 2 === 1) ? (i + 1) : (n - i);
  }

  function isMine(pn, ctx) {
    return ownerOf(pn, ctx.teams) === ctx.mySlot;
  }

  function upcomingOwned(ctx) {
    const tot = (ctx.teams || 12) * (ctx.rounds || 15);
    const out = [];
    for (let pn = ctx.current || 1; pn <= tot; pn++) {
      if (isMine(pn, ctx)) out.push(pn);
    }
    return out;
  }

  function recommendationPickNo(ctx) {
    const cur = ctx.current || 1;
    if (isMine(cur, ctx)) return cur;
    const ups = upcomingOwned(ctx);
    return ups.length ? ups[0] : cur;
  }

  function recWaitPickNo(ctx) {
    const rec = recommendationPickNo(ctx);
    const tot = (ctx.teams || 12) * (ctx.rounds || 15);
    for (let pn = rec + 1; pn <= tot; pn++) {
      if (isMine(pn, ctx)) return pn;
    }
    return null;
  }

  function posOf(p) {
    return String((p && (p.position || p.pos)) || "").toUpperCase();
  }

  function valOf(p) {
    return Number(p && p.val) || 0;
  }

  function adpOf(p) {
    const a = Number(p && p.adp);
    if (!isFinite(a) || a >= 900) return null;
    return a;
  }

  function myPicks(ctx) {
    return (ctx.picks || []).filter(function (x) { return x.slot === ctx.mySlot; });
  }

  function myPosCounts(ctx) {
    const c = { QB: 0, RB: 0, WR: 0, TE: 0 };
    myPicks(ctx).forEach(function (x) {
      const pos = posOf(x.p);
      if (c[pos] != null) c[pos]++;
    });
    return c;
  }

  function simSigma(a) {
    return Math.max(0.5, Math.min(10, 0.35 + 0.055 * a));
  }

  function observedDraftModel(ctx, byId) {
    const residuals = [];
    const made = [];
    (ctx.picks || []).forEach(function (x) {
      const pos = posOf(x.p);
      made.push({ pn: x.pn, pos: pos });
      const full = byId[String(x.p && x.p.id)] || x.p;
      const a = adpOf(full);
      if (a != null) residuals.push(x.pn - a);
    });
    made.sort(function (a, b) { return a.pn - b.pn; });
    const model = { n: residuals.length, bias: 0, std: null, run: {} };
    if (residuals.length >= 1) {
      let sum = 0;
      residuals.forEach(function (r) { sum += r; });
      const mean = sum / residuals.length;
      model.bias = Math.max(-10, Math.min(10, mean));
      if (residuals.length >= 2) {
        let v = 0;
        residuals.forEach(function (r) { v += (r - mean) * (r - mean); });
        model.std = Math.sqrt(v / (residuals.length - 1));
      }
    }
    const teams = ctx.teams || 12;
    if (made.length >= teams) {
      const win = made.slice(-teams);
      const overall = {};
      const recent = {};
      made.forEach(function (m) { overall[m.pos] = (overall[m.pos] || 0) + 1; });
      win.forEach(function (m) { recent[m.pos] = (recent[m.pos] || 0) + 1; });
      Object.keys(recent).forEach(function (pos) {
        const expected = (overall[pos] / made.length) * teams;
        const excess = recent[pos] - expected;
        model.run[pos] = excess > 0 ? Math.min(0.25, (excess / teams) * 1.5) : 0;
      });
    }
    return model;
  }

  function availProb(p, pn, ctx, byId) {
    const C = Core();
    const a = adpOf(p);
    if (a == null) return null;
    let sigma = simSigma(a);
    let center = a;
    const m = observedDraftModel(ctx, byId);
    if (m.n >= 8) {
      const conf = Math.min(1, (m.n - 8) / 20);
      center = a + m.bias * conf;
      if (m.std != null) {
        const obs = Math.max(simSigma(a) * 0.6, Math.min(m.std, 18));
        sigma = sigma * (1 - conf) + obs * conf;
      }
    }
    const runPen = m.run[posOf(p)] || 0;
    if (C && C.availabilityProbability) {
      return C.availabilityProbability({
        center: center, pick: pn, sigma: sigma, runPenalty: runPen,
        draftType: ctx.type || "redraft", sf: !!ctx.sf,
      });
    }
    return null;
  }

  function demandBeforeNext(ctx, nextPick) {
    const C = Core();
    const demand = { QB: 0, RB: 0, WR: 0, TE: 0 };
    if (!nextPick || !C) return demand;
    const rs = rosterOf(ctx);
    const seen = {};
    const countsBySlot = {};
    (ctx.picks || []).forEach(function (x) {
      const s = x.slot;
      if (!countsBySlot[s]) countsBySlot[s] = { QB: 0, RB: 0, WR: 0, TE: 0 };
      const pos = posOf(x.p);
      if (countsBySlot[s][pos] != null) countsBySlot[s][pos]++;
    });
    for (let qn = (ctx.current || 1) + 1; qn < nextPick; qn++) {
      const os = ownerOf(qn, ctx.teams);
      if (seen[os]) continue;
      seen[os] = true;
      const oc = countsBySlot[os] || { QB: 0, RB: 0, WR: 0, TE: 0 };
      Object.keys(demand).forEach(function (pos) {
        const role = C.rosterRole(pos, oc, rs, !!ctx.sf);
        if (role === "starter") demand[pos] += 1;
        else if (role === "flex") demand[pos] += 0.45;
      });
    }
    return demand;
  }

  function byeConflictLevel(p, ctx) {
    const bye = Number(p && (p.bye_week || p.bye)) || 0;
    if (!bye) return 0;
    let n = 0;
    myPicks(ctx).forEach(function (x) {
      const b = Number(x.p && (x.p.bye_week || x.p.bye)) || 0;
      if (b === bye) n++;
    });
    return n >= 2 ? 2 : (n === 1 ? 1 : 0);
  }

  function buildPsCtx(ctx, byId, repl, ppgScale) {
    const C = Core();
    const rs = rosterOf(ctx);
    const counts = myPosCounts(ctx);
    const remaining = upcomingOwned(ctx).length;
    const targets = C && C.posTargets ? C.posTargets(rs, ctx.tep || 0) : { QB: 1, RB: 3, WR: 3, TE: 1 };
    const qualByPos = {};
    const lastPickByPos = {};
    const rosterQualities = [];
    myPicks(ctx).forEach(function (x) {
      const pos = posOf(x.p);
      const full = byId[String(x.p && x.p.id)] || x.p;
      const v = full && full._vor != null ? full._vor : (valOf(full) - (repl[pos] || 0));
      if (v == null || v > 0) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
      const q = C && C.ppgNorm ? C.ppgNorm(full, ppgScale, function (pl) { return Number(pl.ppg) || 0; }) : 0;
      rosterQualities.push({ pos: pos, quality: q != null ? q : 0.35 });
      if (!lastPickByPos[pos] || x.pn > lastPickByPos[pos]) lastPickByPos[pos] = x.pn;
    });
    const obligations = C && C.remainingObligations
      ? C.remainingObligations(counts, rs, remaining, !!ctx.sf, { tep: ctx.tep || 0 })
      : { missing: {}, required: 0, remaining: remaining, freePicks: remaining, lineupHoles: 0 };
    return {
      targets: targets,
      qualByPos: qualByPos,
      rosterQualities: rosterQualities,
      lastPickByPos: lastPickByPos,
      roster: rs,
      remaining: remaining,
      obligations: obligations,
      nextByPos: {},
      demandByPos: {},
    };
  }

  function computePickScore(p, maxVal, counts, ctx, psc, repl, ppgScale, pickNo) {
    const C = Core();
    const Kernel = PS();
    if (!Kernel) return null;
    const pos = posOf(p);
    const teamVal = String(p.team || "").trim().toUpperCase();
    if (!teamVal || teamVal === "FA") return 2;
    if (pos === "K" || pos === "DEF") return null;
    const adp = adpOf(p);
    const t = psc.targets[pos];
    let needRaw = t ? clamp01(Math.max(0, t - (counts[pos] || 0)) / t) : 0;
    const qualNeed = t ? clamp01(Math.max(0, t - (psc.qualByPos[pos] || 0)) / t) : 0;
    needRaw = Math.max(needRaw, qualNeed);
    const ppgN = C && C.ppgNorm ? C.ppgNorm(p, ppgScale, function (pl) { return Number(pl.ppg) || 0; }) : null;
    const vor = p._vor != null ? p._vor : (valOf(p) - (repl[pos] || 0));
    const cliff = isTierCliff(p, pickNo, ctx);
    const starterSlots = pos === "QB"
      ? ((psc.roster.QB || 0) + (psc.roster.SF || 0))
      : (psc.roster[pos] || 0);
    return Kernel.computePickScore({
      pos: pos,
      value: valOf(p),
      vor: vor,
      tier: p.tier,
      age: p.age != null ? Number(p.age) : null,
      rankChange7d: p.rank_change_7d,
      avgPick: adp,
      pickNo: pickNo,
      maxVal: maxVal,
      draftType: ctx.type || "redraft",
      isSf: !!ctx.sf,
      needRaw: needRaw,
      qbCount: counts.QB || 0,
      totalPicks: (ctx.teams || 12) * (ctx.rounds || 15),
      numTeams: ctx.teams || 12,
      ppgNorm: ppgN,
      ppr: ctx.ppr != null ? ctx.ppr : 1,
      tep: ctx.tep || 0,
      passTd: ctx.passTd >= 6 ? 6 : 4,
      isTierCliff: cliff,
      starterSlots: starterSlots,
    });
  }

  var _ptc = {};
  function refreshTiers(pool) {
    _ptc = {};
    (pool || []).forEach(function (p) {
      const t = p.tier;
      if (t == null) return;
      const k = posOf(p) + "|" + t;
      _ptc[k] = (_ptc[k] || 0) + 1;
    });
  }

  function isTierCliff(p, pickNo, ctx) {
    const pn = pickNo != null ? +pickNo : (ctx.current || 1);
    if (pn <= (ctx.teams || 12)) return false;
    const t = p.tier;
    if (t == null) return false;
    return (_ptc[posOf(p) + "|" + t] || 0) <= 2;
  }

  function prepareNextPickValues(pool, ctx, psc, byId) {
    const next = recWaitPickNo(ctx);
    psc.nextByPos = {};
    psc.demandByPos = {};
    if (!next) return;
    psc.demandByPos = demandBeforeNext(ctx, next);
    ["QB", "RB", "WR", "TE"].forEach(function (pos) {
      const rows = pool.filter(function (p) { return posOf(p) === pos && p._ps != null; })
        .sort(function (a, b) { return b._ps - a._ps; }).slice(0, 16);
      let bestExpected = 0;
      rows.forEach(function (p) {
        let prob = availProb(p, next, ctx, byId);
        if (prob == null) prob = 50;
        bestExpected = Math.max(bestExpected, p._ps * (0.35 + 0.65 * prob / 100));
      });
      psc.nextByPos[pos] = bestExpected;
    });
  }

  const LIVE_WAIT_TUNING = { threshold: 50, maxPenalty: 10 };
  const LIVE_WAIT_SUBONSET = 20;
  const LIVE_WAIT_SUBSHARE = 0.15;

  function liveDecisionScore(p, counts, ctx, psc, byId) {
    const C = Core();
    const base = p._ps;
    if (base == null) return null;
    const pos = posOf(p);
    if (pos === "K" || pos === "DEF") return null;
    if (!C || !C.decisionScore) return base;
    const role = C.candidateRosterRole
      ? C.candidateRosterRole(pos, C.ppgNorm ? (C.ppgNorm(p, psc._ppgScale, function (pl) { return Number(pl.ppg) || 0; }) || 0) : 0, psc.rosterQualities, psc.roster, !!ctx.sf)
      : C.rosterRole(pos, counts, psc.roster, !!ctx.sf);
    const util = C.positionNeedUtility
      ? C.positionNeedUtility(pos, counts, psc.roster, { sf: !!ctx.sf, tep: ctx.tep || 0, draftType: ctx.type || "redraft", role: role })
      : C.rosterSlotUtility(pos, counts, psc.roster, { sf: !!ctx.sf, tep: ctx.tep || 0, draftType: ctx.type || "redraft", role: role });
    const expected = psc.nextByPos[pos] || 0;
    const bench = role === "bench1" || role === "bench2";
    let recentPenalty = 0;
    if (bench && (pos === "QB" || pos === "TE") && psc.lastPickByPos[pos]) {
      const roundsSince = ((ctx.current || 1) - psc.lastPickByPos[pos]) / Math.max(1, ctx.teams || 12);
      recentPenalty = Math.max(0, 10 * (1 - roundsSince / 6));
    }
    const adp = adpOf(p);
    let exceptional = 0;
    if (adp != null) exceptional = clamp01(((ctx.current || 1) - adp) / Math.max(12, adp * 0.65));
    const recPn = recommendationPickNo(ctx);
    const advisingFuture = recPn > ((ctx.current || 1));
    const nextPick = recWaitPickNo(ctx);
    const returnProb = nextPick ? availProb(p, nextPick, ctx, byId) : null;
    const demand = (psc.demandByPos && psc.demandByPos[pos]) || 0;
    const demandRisk = Math.min(0.35, demand / Math.max(1, ctx.teams || 12) * 0.7);
    const effectiveReturnProb = returnProb == null ? null : returnProb * (1 - demandRisk);
    const thr = LIVE_WAIT_TUNING.threshold;
    let wpFrac = 0;
    if (effectiveReturnProb == null) wpFrac = 0;
    else if (effectiveReturnProb >= thr) {
      wpFrac = LIVE_WAIT_SUBSHARE + (1 - LIVE_WAIT_SUBSHARE) * clamp01((effectiveReturnProb - thr) / (100 - thr));
    } else {
      wpFrac = LIVE_WAIT_SUBSHARE * clamp01((effectiveReturnProb - LIVE_WAIT_SUBONSET) / (thr - LIVE_WAIT_SUBONSET));
    }
    const waitPenalty = wpFrac * LIVE_WAIT_TUNING.maxPenalty * (1 - exceptional * 0.5);
    let handcuffBonus = 0;
    if ((ctx.type || "redraft") === "redraft" && pos === "RB" && p.team) {
      const myRBTeams = {};
      myPicks(ctx).forEach(function (x) {
        if (posOf(x.p) === "RB" && x.p.team) myRBTeams[x.p.team] = true;
      });
      if (myRBTeams[p.team]) handcuffBonus = 5;
    }
    let upsideBonus = 0;
    if ((ctx.type || "redraft") === "redraft" && C.lateRoundUpsideBonus) {
      const rd = Math.floor(((ctx.current || 1) - 1) / Math.max(1, ctx.teams || 12)) + 1;
      const path = C.lateRoundPathEvidence
        ? C.lateRoundPathEvidence({
            breakoutScore: p.breakout_score, projectedRole: p.projected_role,
            handcuff: handcuffBonus > 0,
          })
        : 0;
      const ppgn = C.ppgNorm ? C.ppgNorm(p, psc._ppgScale, function (pl) { return Number(pl.ppg) || 0; }) : 0;
      const age = p.age != null ? Number(p.age) : null;
      upsideBonus = C.lateRoundUpsideBonus({
        round: rd, totalRounds: ctx.rounds || 15, path: path,
        aboveReplacement: p._vor > 0 ? clamp01(p._vor / Math.max(1, valOf(p))) : 0,
        tierQuality: p.tier ? clamp01((10 - Math.min(9, p.tier)) / 9) : 0,
        ppgQuality: ppgn == null ? 0 : ppgn,
        functionalUtility: util,
        rosterNeedPath: role === "starter" || role === "flex" || role === "bench1" ? 1 : 0.35,
        youngWithPath: path > 0 && age != null && (pos === "RB" || pos === "WR") && age <= (pos === "RB" ? 24 : 25),
      });
    }
    let byePenalty = 0;
    if ((ctx.type || "redraft") === "redraft" && C.byeSeverityPenalty) {
      byePenalty = C.byeSeverityPenalty(byeConflictLevel(p, ctx));
    }
    const missDed = (psc.obligations && psc.obligations.missing && psc.obligations.missing[pos]) || 0;
    const waitLossScale = C.waitLossScaleFor
      ? C.waitLossScaleFor(pos, missDed, { sf: !!ctx.sf, tep: ctx.tep || 0 })
      : (missDed >= 2 ? 1 : (missDed >= 1 ? 0.6 : 0.4));
    let score = C.decisionScore({
      base: base, utility: util,
      bench: bench, deepBench: role === "bench2", recentPenalty: recentPenalty, exceptional: exceptional,
      quality: C.ppgNorm ? (C.ppgNorm(p, psc._ppgScale, function (pl) { return Number(pl.ppg) || 0; }) || 0) : 0,
      required: psc.obligations.required,
      freePicks: psc.obligations.freePicks,
      waitLoss: Math.max(0, base - expected) * (1 + demandRisk), waitLossScale: waitLossScale,
      waitPenalty: waitPenalty, handcuffBonus: handcuffBonus, upsideBonus: upsideBonus,
      byePenalty: byePenalty,
      draftType: ctx.type || "redraft", lineupHoles: psc.obligations.lineupHoles || 0,
    });
    if (advisingFuture && C.futurePickDecisionScore) {
      score = C.futurePickDecisionScore(score, availProb(p, recPn, ctx, byId));
    }
    return score;
  }

  function pickReason(p, counts, ctx, psc, byId) {
    const C = Core();
    const pos = posOf(p);
    const pickNo = ctx.current || 1;
    const recPn = recommendationPickNo(ctx);
    const advisingFuture = recPn > pickNo;
    const t = psc.targets[pos];
    const need = t ? Math.max(0, t - (counts[pos] || 0)) : 0;
    const adp = adpOf(p);
    const fell = adp != null ? Math.round(pickNo - adp) : null;
    const relGap = adp != null ? ((pickNo - adp) / Math.max(adp, 1.5)) : null;
    const tier = p.tier;
    const left = _ptc[pos + "|" + tier] || 0;
    const role = C && C.rosterRole ? C.rosterRole(pos, counts, psc.roster, !!ctx.sf) : "starter";
    if ((role === "bench1" || role === "bench2") && (psc.obligations.lineupHoles || 0) > 0)
      return "Backup-only · starter slots still open";
    if ((role === "bench1" || role === "bench2") && psc.obligations.required > 0 && psc.obligations.freePicks <= 2)
      return "Backup-only · only " + psc.obligations.freePicks + " discretionary picks";
    if (!ctx.sf && pos === "QB" && (counts.QB || 0) >= 1)
      return "QB filled · backup-only value";
    if (pos === "TE" && String(role).indexOf("bench") === 0 && !(ctx.tep > 0))
      return "TE filled · backup-only value";
    if (role === "flex") return "Fills FLEX · weekly lineup value";
    if (isTierCliff(p, recPn, ctx) && tier != null) {
      if (left <= 1) return "Last " + pos + " in Tier " + tier + ". Grab now";
      return "Only " + left + " " + pos + "s left in Tier " + tier;
    }
    if (relGap != null && relGap >= 1.0) return "Elite steal: " + fell + " picks past ADP";
    if (relGap != null && relGap >= 0.5) return "Steal: fell " + fell + " picks past ADP";
    if (advisingFuture) {
      const atRec = availProb(p, recPn, ctx, byId);
      if (atRec != null && atRec < 20) {
        return atRec <= 0 ? ("Gone before #" + recPn) : ("Unlikely to last to #" + recPn);
      }
    }
    if (p._rank === 1) return advisingFuture ? ("Best available at #" + recPn) : "Best available";
    if (need > 0 && recPn > 4) {
      if (tier != null && tier <= 2) return "Tier " + tier + " " + pos + " fills a need";
      return "Fills " + pos + " need (" + need + " more to target)";
    }
    if (fell != null && fell >= 3) return "Good value: " + fell + " past ADP";
    if (tier != null && tier <= 2) return "Elite tier (T" + tier + ") talent";
    if (adp != null && adp <= 12) return "1st-round talent";
    if (adp != null && adp <= 24) return "Early-round talent";
    return "Strong remaining value";
  }

  function psDisplay(ps, poolMax) {
    if (ps == null) return null;
    if (!poolMax || poolMax <= 0) return ps;
    const d = Math.round(97 * ps / poolMax);
    return d > 99 ? 99 : (d < 1 ? 1 : d);
  }

  function rankPool(allPlayers, available, ctx) {
    const C = Core();
    const Kernel = PS();
    if (!C || !Kernel) return available;
    const byId = {};
    (allPlayers || []).forEach(function (p) {
      if (p && p.id) byId[String(p.id)] = p;
      if (p && !p.position) p.position = p.pos;
    });
    (available || []).forEach(function (p) {
      if (p && !p.position) p.position = p.pos;
    });
    const rs = rosterOf(ctx);
    const valFn = valOf;
    const starters = C.effectiveStarters(allPlayers, rs, ctx.teams || 12, valFn);
    const repl = C.computeReplacement(allPlayers, valFn, starters, ctx.teams || 12) || {};
    const ppgFn = function (pl) { return Number(pl.ppg) || 0; };
    const ppgScale = C.computePpgScale(allPlayers, ppgFn, starters, ctx.teams || 12);
    (allPlayers || []).forEach(function (p) {
      p._vor = valOf(p) - (repl[posOf(p)] || 0);
    });
    refreshTiers(available);
    const counts = myPosCounts(ctx);
    const recPn = recommendationPickNo(ctx);
    const psc = buildPsCtx(ctx, byId, repl, ppgScale);
    psc._ppgScale = ppgScale;
    let maxVal = 1;
    (allPlayers || []).forEach(function (p) { if (valOf(p) > maxVal) maxVal = valOf(p); });
    let poolMax = 0;
    (available || []).forEach(function (p) {
      p._ps = computePickScore(p, maxVal, counts, ctx, psc, repl, ppgScale, recPn);
      if (p._ps != null && p._ps > poolMax) poolMax = p._ps;
    });
    prepareNextPickValues(available, ctx, psc, byId);
    (available || []).forEach(function (p) {
      p._ds = liveDecisionScore(p, counts, ctx, psc, byId);
      p._psShow = psDisplay(p._ps, poolMax);
    });
    const ranked = (available || []).filter(function (p) { return p._ds != null; })
      .sort(function (a, b) { return (b._ds - a._ds) || ((b._ps || 0) - (a._ps || 0)) || ((a.adp || 999) - (b.adp || 999)); });
    ranked.forEach(function (p, i) { p._rank = i + 1; });
    ranked._reasonCtx = { ctx: ctx, psc: psc, byId: byId, counts: counts };
    return ranked;
  }

  root.BROverlayScore = {
    rankPool: rankPool,
    pickReason: function (p, ranked) {
      const bag = ranked && ranked._reasonCtx;
      if (!bag) return "";
      return pickReason(p, bag.counts, bag.ctx, bag.psc, bag.byId);
    },
    recommendationPickNo: recommendationPickNo,
    psDisplay: psDisplay,
  };
})(typeof self !== "undefined" ? self : this);
