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

  function ownerOf(pn, ctx) {
    const n = (ctx && ctx.teams) || 12;
    const mapped = Number(ctx && ctx.pickOwners && ctx.pickOwners[pn]);
    if (mapped >= 1 && mapped <= n) return mapped;
    const r = Math.ceil(pn / n);
    const i = (pn - 1) % n;
    return (r % 2 === 1) ? (i + 1) : (n - i);
  }

  function isMine(pn, ctx) {
    return ownerOf(pn, ctx) === ctx.mySlot;
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
    // Live overlay: rank the pick on the clock. Looking ahead to a later
    // owned pick (1.07 on the clock scored as #20) buried players who are
    // available right now.
    return ctx.current || 1;
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
    const raw = String((p && (p.position || p.pos)) || "").toUpperCase();
    if (raw === "PK") return "K";
    if (raw === "DST" || raw === "D/ST" || raw === "D-ST" || raw === "D ST") return "DEF";
    return raw;
  }

  function isKDef(p) {
    const pos = typeof p === "string" ? posOf({ pos: p }) : posOf(p);
    return pos === "K" || pos === "DEF";
  }

  function valOf(p) {
    return Number(p && p.val) || 0;
  }

  function scoringOf(ctx) {
    return {
      ppr: ctx && ctx.ppr != null ? Number(ctx.ppr) : 1,
      tep: ctx && ctx.tep != null ? Number(ctx.tep) : 0,
      passTd: ctx && ctx.passTd >= 6 ? 6 : 4,
    };
  }

  // Same projection the Draft Room grades with: scoring-adjusted Sleeper PPG,
  // never last-season actuals. Compact overlay rows store that as proj_ppg/ppg.
  function ppgOf(p, ctx) {
    const C = Core();
    if (!p) return null;
    if (C && C.ppgOf) {
      const v = C.ppgOf(p, scoringOf(ctx));
      if (v != null && isFinite(Number(v))) return Number(v);
    }
    const n = Number(p.proj_ppg != null ? p.proj_ppg : p.ppg);
    return isFinite(n) && n > 0 ? n : null;
  }

  function ppgFn(ctx) {
    return function (pl) { return ppgOf(pl, ctx); };
  }

  function lineupScore(p, ctx) {
    if (!p) return -Infinity;
    const ppg = ppgOf(p, ctx);
    if (ppg != null) return ppg;
    return (Number(p.val) || 0) / 1000;
  }

  function slotEligible(slot, pos) {
    const p = String(pos || "").toUpperCase();
    const s = String(slot || "").toUpperCase();
    if (s === "FLEX") return p === "RB" || p === "WR" || p === "TE";
    if (s === "SF" || s === "OP") return p === "QB" || p === "RB" || p === "WR" || p === "TE";
    if (s === "RB_WR") return p === "RB" || p === "WR";
    if (s === "WR_TE") return p === "WR" || p === "TE";
    if (s === "RB_TE") return p === "RB" || p === "TE";
    if (s === "DEF") return posOf({ pos: p }) === "DEF";
    return posOf({ pos: p }) === s;
  }

  function slotListOf(ctx) {
    const rs = rosterOf(ctx);
    const C = Core();
    const slotFn = (root.BRDraftSlot && root.BRDraftSlot.slotListFromRoster)
      || (C && C.slotListFromRoster);
    if (slotFn) {
      const list = slotFn(rs);
      if (list && list.length >= 4) return list;
    }
    const slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"];
    if (ctx && ctx.sf && slots.indexOf("SF") < 0) slots.splice(1, 0, "SF");
    return slots;
  }

  // Restrictive-first fill, same order as Draft Room optimalLineup.
  function optimalLineup(list, ctx) {
    const playerList = (list || []).slice();
    const slots = slotListOf(ctx);
    const flex = { SF: 3, FLEX: 2, RB_WR: 1.5, WR_TE: 1.5, RB_TE: 1.5 };
    const order = slots.map(function (s, i) { return { slot: s, i: i }; });
    order.sort(function (a, b) { return (flex[a.slot] || 1) - (flex[b.slot] || 1) || a.i - b.i; });
    const used = {};
    const assign = {};
    order.forEach(function (o) {
      let best = -1;
      let bestScore = -Infinity;
      for (let j = 0; j < playerList.length; j++) {
        if (used[j] || !slotEligible(o.slot, posOf(playerList[j]))) continue;
        const sc = lineupScore(playerList[j], ctx);
        if (sc > bestScore) {
          bestScore = sc;
          best = j;
        }
      }
      if (best >= 0) {
        used[best] = true;
        assign[o.i] = playerList[best];
      }
    });
    const starters = slots.map(function (s, i) { return { slot: s, p: assign[i] || null }; });
    const bench = [];
    for (let k = 0; k < playerList.length; k++) {
      if (!used[k]) bench.push(playerList[k]);
    }
    bench.sort(function (a, b) { return lineupScore(b, ctx) - lineupScore(a, ctx); });
    return { starters: starters, bench: bench };
  }

  function gradeMax(draftType) {
    if (draftType === "rookie") return { value: 100, starters: 0, construction: 0 };
    const TG = root.BRTeamGrade;
    const split = draftType === "redraft"
      ? ((TG && TG.SPLIT_REDRAFT) || [20, 50, 30])
      : ((TG && TG.SPLIT_STARTUP) || [35, 25, 40]);
    return { value: split[0], starters: split[1], construction: split[2] };
  }

  // Same early-pick identity labels as Draft Room teamArchetype().
  function teamArchetype(mine, ctx) {
    if (!mine || mine.length < 3) return null;
    if ((ctx && ctx.type) === "rookie") return null;
    const sorted = mine.slice().sort(function (a, b) { return a.pn - b.pn; });
    const counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
    const firstIdx = { QB: -1, RB: -1, WR: -1, TE: -1 };
    sorted.forEach(function (m, i) {
      const pos = posOf(m.p || m);
      if (counts[pos] != null) {
        counts[pos]++;
        if (firstIdx[pos] < 0) firstIdx[pos] = i;
      }
    });
    const earlyN = Math.min(5, sorted.length);
    const early = { QB: 0, RB: 0, WR: 0, TE: 0 };
    for (let i = 0; i < earlyN; i++) {
      const pos = posOf(sorted[i].p || sorted[i]);
      if (early[pos] != null) early[pos]++;
    }
    let label;
    if (ctx && ctx.sf && early.QB >= 2) label = "Konami Code";
    else if (firstIdx.TE >= 0 && firstIdx.TE <= 1) label = "TE Premium";
    else if (early.RB === 0 && early.WR >= 3) label = "Zero RB";
    else if (early.RB === 1 && firstIdx.RB <= 1 && early.WR >= 2) label = "Hero RB";
    else if (early.RB >= 3) label = "Robust RB";
    else if (early.WR >= 4 || (counts.WR - counts.RB >= 3)) label = "WR Factory";
    else if (early.RB - early.WR >= 2) label = "Ground & Pound";
    else label = "Balanced Build";
    return { label: label };
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
    const c = { QB: 0, RB: 0, WR: 0, TE: 0, K: 0, DEF: 0 };
    myPicks(ctx).forEach(function (x) {
      const pos = posOf(x.p);
      if (c[pos] != null) c[pos]++;
    });
    return c;
  }

  function sortKdef(a, b) {
    const aa = adpOf(a);
    const ba = adpOf(b);
    if (aa != null || ba != null) {
      return (aa != null ? aa : 99999) - (ba != null ? ba : 99999);
    }
    return (Number(b && b.ppg) || 0) - (Number(a && a.ppg) || 0);
  }

  function kdefNeed(ctx, counts) {
    counts = counts || myPosCounts(ctx);
    const rs = rosterOf(ctx);
    const needK = Math.max(0, (rs.K || 0) - (counts.K || 0));
    const needDef = Math.max(0, (rs.DEF || 0) - (counts.DEF || 0));
    if (needK + needDef <= 0) return null;
    const remaining = upcomingOwned(ctx).length;
    const remainRds = (ctx.rounds || 15) - Math.floor(((ctx.current || 1) - 1) / (ctx.teams || 12));
    if (remaining <= (needK + needDef) + 2 || remainRds <= 3) {
      return { needK: needK, needDef: needDef };
    }
    return null;
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
      const os = ownerOf(qn, ctx);
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
    if (isKDef(p)) return 0;
    const bye = Number(p && (p.bye_week || p.bye)) || 0;
    if (!bye) return 0;
    let n = 0;
    myPicks(ctx).forEach(function (x) {
      if (isKDef(x.p)) return;
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
      const q = C && C.ppgNorm ? C.ppgNorm(full, ppgScale, ppgFn(ctx)) : 0;
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

  function computePickScore(p, maxVal, counts, ctx, psc, repl, ppgScale, pickNo, cliffOverride) {
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
    const ppgN = C && C.ppgNorm ? C.ppgNorm(p, ppgScale, ppgFn(ctx)) : null;
    const vor = p._vor != null ? p._vor : (valOf(p) - (repl[pos] || 0));
    const cliff = cliffOverride != null ? !!cliffOverride : isTierCliff(p, pickNo, ctx);
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
      ? C.candidateRosterRole(pos, C.ppgNorm ? (C.ppgNorm(p, psc._ppgScale, ppgFn(ctx)) || 0) : 0, psc.rosterQualities, psc.roster, !!ctx.sf)
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
      const ppgn = C.ppgNorm ? C.ppgNorm(p, psc._ppgScale, ppgFn(ctx)) : 0;
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
    const recRound = Math.floor(((ctx.current || 1) - 1) / Math.max(1, ctx.teams || 12)) + 1;
    const streamableBackup = bench && (ctx.type || "redraft") === "redraft"
      && C.isStreamableSingleSlot
      && C.isStreamableSingleSlot(pos, psc.roster, { sf: !!ctx.sf, tep: ctx.tep || 0 });
    let score = C.decisionScore({
      base: base, utility: util,
      bench: bench, deepBench: role === "bench2", recentPenalty: recentPenalty, exceptional: exceptional,
      quality: C.ppgNorm ? (C.ppgNorm(p, psc._ppgScale, ppgFn(ctx)) || 0) : 0,
      required: psc.obligations.required,
      freePicks: psc.obligations.freePicks,
      waitLoss: Math.max(0, base - expected) * (1 + demandRisk), waitLossScale: waitLossScale,
      waitPenalty: waitPenalty, handcuffBonus: handcuffBonus, upsideBonus: upsideBonus,
      byePenalty: byePenalty,
      draftType: ctx.type || "redraft", lineupHoles: psc.obligations.lineupHoles || 0,
      streamableBackup: streamableBackup, round: recRound, totalRounds: ctx.rounds || 16,
    });
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
    const ppgScale = C.computePpgScale(allPlayers, ppgFn(ctx), starters, ctx.teams || 12);
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

  function gradeRowsForPicks(mine, allPlayers, ctx, cliffByPn) {
    const C = Core();
    const Kernel = PS();
    if (!C || !Kernel || !mine || !mine.length) return [];
    const byId = {};
    (allPlayers || []).forEach(function (p) {
      if (p && p.id) byId[String(p.id)] = p;
    });
    const rs = rosterOf(ctx);
    const valFn = valOf;
    const starters = C.effectiveStarters(allPlayers, rs, ctx.teams || 12, valFn);
    const repl = C.computeReplacement(allPlayers, valFn, starters, ctx.teams || 12) || {};
    const ppgScale = C.computePpgScale(allPlayers, ppgFn(ctx), starters, ctx.teams || 12);
    (allPlayers || []).forEach(function (p) {
      p._vor = valOf(p) - (repl[posOf(p)] || 0);
    });
    refreshTiers(allPlayers);
    const psc = buildPsCtx(Object.assign({}, ctx, { picks: mine }), byId, repl, ppgScale);
    psc.qualByPos = { QB: 0, RB: 0, WR: 0, TE: 0 };
    let maxVal = 1;
    (allPlayers || []).forEach(function (p) { if (valOf(p) > maxVal) maxVal = valOf(p); });
    const counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
    const rows = [];
    mine.forEach(function (m) {
      const full = byId[String(m.p && m.p.id)] || m.p;
      const pos = posOf(full);
      const cliff = cliffByPn && m.pn != null ? cliffByPn[m.pn] : null;
      const ps = computePickScore(full, maxVal, counts, ctx, psc, repl, ppgScale, m.pn, cliff);
      if (counts[pos] != null) counts[pos]++;
      if (psc.qualByPos[pos] != null) {
        const vor = full && full._vor != null ? full._vor : (valOf(full) - (repl[pos] || 0));
        if (vor == null || vor > 0) psc.qualByPos[pos]++;
      }
      rows.push({
        id: full && full.id,
        pos: pos,
        ps: ps,
        pn: m.pn,
        val: valOf(full),
        ppg: ppgOf(full, ctx),
        age: full && full.age != null ? Number(full.age) : null,
      });
    });
    return rows;
  }

  // Walk the draft in pick order so each historical grade sees remaining-at-slot
  // tier counts, matching Draft Room _buildGradeCliffs.
  function buildGradeCliffs(allPlayers, picksBySlot, ctx) {
    const counts = {};
    (allPlayers || []).forEach(function (p) {
      const t = p && p.tier;
      if (t == null) return;
      const k = posOf(p) + "|" + t;
      counts[k] = (counts[k] || 0) + 1;
    });
    const made = [];
    Object.keys(picksBySlot || {}).forEach(function (s) {
      (picksBySlot[s] || []).forEach(function (x) { made.push(x); });
    });
    made.sort(function (a, b) { return a.pn - b.pn; });
    const map = {};
    const teams = (ctx && ctx.teams) || 12;
    const byId = {};
    (allPlayers || []).forEach(function (p) {
      if (p && p.id) byId[String(p.id)] = p;
    });
    made.forEach(function (x) {
      const full = byId[String(x.p && x.p.id)] || x.p;
      const t = full && full.tier;
      const k = posOf(full) + "|" + t;
      const left = t != null ? (counts[k] || 0) : 0;
      map[x.pn] = (x.pn > teams) && t != null && left <= 2;
      if (t != null && counts[k]) counts[k]--;
    });
    return map;
  }

  function competitiveWindow(rows, draftType) {
    if (draftType === "redraft") return null;
    let wSum = 0, aSum = 0;
    (rows || []).forEach(function (x) {
      if (x.age == null || !isFinite(x.age) || x.age <= 0) return;
      const w = Math.max(1, x.val || 1);
      aSum += x.age * w;
      wSum += w;
    });
    if (wSum <= 0) return null;
    const avgAge = aSum / wSum;
    const label = avgAge <= 24.5 ? "Future" : avgAge >= 26.5 ? "Win-Now" : "Balanced";
    return { label: label, avgAge: avgAge };
  }

  function gradeField(allPlayers, picksBySlot, ctx) {
    const TG = root.BRTeamGrade;
    if (!TG || typeof TG.teamGradeComposite !== "function") return null;
    const C = Core();
    const rs = rosterOf(ctx);
    const slotFn = (root.BRDraftSlot && root.BRDraftSlot.slotListFromRoster)
      || (C && C.slotListFromRoster);
    const slots = slotFn
      ? slotFn(rs).filter(function (s) { return s !== "K" && s !== "DEF"; })
      : ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"];
    if (ctx.sf && slots.indexOf("SF") < 0) slots.splice(1, 0, "SF");
    const targets = C && C.posTargets ? C.posTargets(rs, ctx.tep || 0) : { QB: 1, RB: 3, WR: 3, TE: 1 };
    const draftType = ctx.type === "startup" || ctx.type === "dynasty" ? "startup" : "redraft";
    const leaguePpg = [];
    const leagueVal = [];
    const leaguePlayers = [];
    (allPlayers || []).forEach(function (p) {
      const ppg = ppgOf(p, ctx);
      if (ppg != null) leaguePpg.push(ppg);
      leagueVal.push(valOf(p));
      leaguePlayers.push({ pos: posOf(p), ppg: ppg, val: valOf(p) });
    });
    const teams = Number(ctx.teams) || 12;
    const lists = [];
    const order = [];
    const mineSlot = Number(ctx.mySlot) || 0;
    if (mineSlot && picksBySlot[mineSlot] && picksBySlot[mineSlot].length) {
      lists.push(picksBySlot[mineSlot]);
      order.push({ slot: mineSlot, isMe: true });
    }
    for (let s = 1; s <= teams; s++) {
      if (s === mineSlot) continue;
      if (!picksBySlot[s] || !picksBySlot[s].length) continue;
      lists.push(picksBySlot[s]);
      order.push({ slot: s, isMe: false });
    }
    const cliffs = buildGradeCliffs(allPlayers, picksBySlot, ctx);
    const leagueTeams = lists.map(function (mine) {
      return gradeRowsForPicks(mine, allPlayers, ctx, cliffs);
    });
    const out = [];
    order.forEach(function (info, i) {
      const rows = leagueTeams[i];
      const comp = TG.teamGradeComposite(
        rows, slots, targets, teams, draftType,
        leaguePpg, leagueVal, leaguePlayers,
        { sf: !!ctx.sf, tep: Number(ctx.tep) || 0, leagueTeams: leagueTeams }
      );
      if (!comp) return;
      const starterRows = rows.filter(function (x) { return comp.starterIds[String(x.id)]; });
      const psVals = rows.map(function (x) { return x.ps; }).filter(function (v) { return v != null; });
      const avgPs = psVals.length
        ? psVals.reduce(function (a, b) { return a + b; }, 0) / psVals.length
        : null;
      out.push({
        slot: info.slot,
        isMe: info.isMe,
        grade: {
          score: comp.total,
          value: comp.value,
          starters: comp.starter,
          construction: comp.balance,
          provisional: (picksBySlot[info.slot] || []).length < 8,
          window: competitiveWindow(starterRows, draftType),
          avgPs: avgPs != null ? Math.round(avgPs) : null,
        },
        gradeRows: rows,
        archetype: info.isMe ? teamArchetype(picksBySlot[info.slot], ctx) : null,
        picks: picksBySlot[info.slot],
      });
    });
    out.sort(function (a, b) { return b.grade.score - a.grade.score; });
    return out;
  }

  function rosterProjection(mine, allPlayers, ctx) {
    const minePpg = [];
    (mine || []).forEach(function (p) {
      const v = ppgOf(p, ctx);
      if (v != null) minePpg.push(v);
    });
    if (minePpg.length < 2) return null;
    const myAvg = minePpg.reduce(function (a, b) { return a + b; }, 0) / minePpg.length;
    const allProj = [];
    (allPlayers || []).forEach(function (p) {
      const v = ppgOf(p, ctx);
      if (v != null) allProj.push(v);
    });
    allProj.sort(function (a, b) { return b - a; });
    const topSlice = allProj.slice(0, (ctx.teams || 12) * (ctx.rounds || 15));
    const lgAvg = topSlice.length
      ? topSlice.reduce(function (a, b) { return a + b; }, 0) / topSlice.length
      : 0;
    const pct = lgAvg > 0 ? Math.round(myAvg / lgAvg * 100) : 0;
    return { myAvg: myAvg, lgAvg: lgAvg, pct: pct, withProj: minePpg.length, total: mine.length };
  }

  function recapStats(field) {
    const picks = [];
    (field || []).forEach(function (t) {
      const rows = t.gradeRows || [];
      (t.picks || []).forEach(function (pk, i) {
        const row = rows[i];
        const ps = row && row.ps != null ? row.ps : (pk.ps != null ? pk.ps : null);
        if (ps == null) return;
        const pl = pk.p || pk;
        const adp = adpOf(pl);
        const gap = adp != null ? Math.round(pk.pn - adp) : null;
        picks.push({
          name: pl.name,
          pos: posOf(pl),
          team: t.name || ("Team " + t.slot),
          teamSlot: t.slot,
          pn: pk.pn,
          ps: ps,
          gap: gap,
        });
      });
    });
    if (picks.length < 4) return null;
    const withGap = picks.filter(function (p) { return p.gap != null; });
    const useGap = withGap.length >= 4;
    const pool = useGap ? withGap : picks;
    const steals = pool.slice().sort(function (a, b) { return useGap ? (b.gap - a.gap) : (b.ps - a.ps); }).slice(0, 4);
    const reaches = pool.slice().sort(function (a, b) { return useGap ? (a.gap - b.gap) : (a.ps - b.ps); }).slice(0, 4);
    const teamScores = {};
    picks.forEach(function (p) { (teamScores[p.team] = teamScores[p.team] || []).push(p.ps); });
    let valueTeam = "-";
    let valueAvg = -1e9;
    Object.keys(teamScores).forEach(function (tm) {
      const arr = teamScores[tm];
      const avg = arr.reduce(function (a, b) { return a + b; }, 0) / arr.length;
      if (avg > valueAvg) { valueAvg = avg; valueTeam = tm; }
    });
    const posCount = {};
    picks.forEach(function (p) { if (p.pos) posCount[p.pos] = (posCount[p.pos] || 0) + 1; });
    const topPos = Object.keys(posCount).sort(function (a, b) { return posCount[b] - posCount[a]; })[0] || "-";
    return {
      steals: steals,
      reaches: reaches,
      valueTeam: valueTeam,
      topPos: topPos,
      posCount: posCount,
      pickCount: picks.length,
      useGap: useGap,
    };
  }

  root.BROverlayScore = {
    rankPool: rankPool,
    pickReason: function (p, ranked) {
      const bag = ranked && ranked._reasonCtx;
      if (!bag) return "";
      return pickReason(p, bag.counts, bag.ctx, bag.psc, bag.byId);
    },
    recommendationPickNo: recommendationPickNo,
    recWaitPickNo: recWaitPickNo,
    isKDef: isKDef,
    sortKdef: sortKdef,
    kdefNeed: kdefNeed,
    myPosCounts: myPosCounts,
    availProb: function (p, pn, ctx, allPlayers) {
      const byId = {};
      (allPlayers || []).forEach(function (pl) {
        if (pl && pl.id) byId[String(pl.id)] = pl;
      });
      return availProb(p, pn, ctx, byId);
    },
    psDisplay: psDisplay,
    gradeField: gradeField,
    gradeMax: gradeMax,
    teamArchetype: teamArchetype,
    optimalLineup: optimalLineup,
    lineupScore: lineupScore,
    ppgOf: ppgOf,
    rosterProjection: rosterProjection,
    recapStats: recapStats,
  };
})(typeof self !== "undefined" ? self : this);
