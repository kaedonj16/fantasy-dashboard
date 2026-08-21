#!/usr/bin/env node
'use strict';

// Headless, seeded batch benchmark for the shared CPU roster-economics model.
// It deliberately skips the Draft Room DOM/timers and runs DraftBoardCore's
// role, utility, obligation, Decision Score, and candidate-band kernels in a
// tight synchronous loop.

const fs = require('fs');
const path = require('path');
const Core = require('../static/draft_board_core.js');

function args(argv) {
  const out = { drafts: 1000, type: 'redraft', teams: 12, rounds: 15, sf: false,
    qb: 1, rb: 2, wr: 2, te: 1, flex: 2, k: 0, def: 0, seed: 20260821, json: null };
  for (let i = 2; i < argv.length; i++) {
    let key = argv[i].replace(/^--/, '').replace(/-/g, '_');
    if (key === 'sf' || key === 'superflex') { out.sf = true; continue; }
    if (key === 'help') { out.help = true; continue; }
    if (!(key in out) || i + 1 >= argv.length) throw new Error(`Unknown/incomplete option --${key}`);
    const raw = argv[++i]; out[key] = key === 'type' || key === 'json' ? raw : Number(raw);
  }
  return out;
}

function help() {
  console.log(`Usage: node scripts/benchmark_cpu_drafts.js [options]

  --drafts N       drafts to run (default 1000)
  --type TYPE      redraft or startup
  --teams N        teams (default 12)
  --rounds N       rounds (default 15)
  --sf             Superflex format
  --qb/--rb/--wr/--te N   dedicated starters
  --flex N         FLEX starters (default 2)
  --k N --def N    required K/DEF slots
  --seed N         reproducible random seed
  --json PATH      write machine-readable results
`);
}

function rng(seed) {
  let x = (seed >>> 0) || 1;
  return function () { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 4294967296; };
}

function median(xs) {
  if (!xs.length) return null;
  const a = xs.slice().sort((x, y) => x - y), m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : Math.round((a[m - 1] + a[m]) * 20) / 40;
}

function waitBin(probability) {
  if (probability >= 80) return '80+';
  if (probability >= 65) return '65-79';
  if (probability >= 50) return '50-64';
  return '<50';
}

function loadPool(cfg) {
  const root = path.resolve(__dirname, '..');
  const players = JSON.parse(fs.readFileSync(path.join(root, 'cache/players_index.json')));
  const adp = JSON.parse(fs.readFileSync(path.join(root, 'data/sleeper_adp_2026_2026-07-12.json')));
  const projections = JSON.parse(fs.readFileSync(path.join(root, 'cache/fp_projections_2026_ppr.json')));
  const adpKey = cfg.type === 'startup'
    ? (cfg.sf ? 'adp_dynasty_2qb' : 'adp_dynasty_ppr')
    : (cfg.sf ? 'adp_2qb' : 'adp_ppr');
  const pool = [];
  Object.keys(adp).forEach(id => {
    const meta = players[id] || {}, pos0 = String(meta.pos || '').toUpperCase();
    const pos = pos0 === 'PK' ? 'K' : pos0;
    const pick = Number(adp[id] && adp[id][adpKey]);
    if (!['QB', 'RB', 'WR', 'TE', 'K'].includes(pos) || !Number.isFinite(pick) || pick >= 900) return;
    pool.push({ id, position: pos, adp: pick, ppg: Number(projections[id] && projections[id].ppg) || 0 });
  });
  if (cfg.def) {
    ['ARI','ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC',
      'LV','LAC','LAR','MIA','MIN','NE','NO','NYG','NYJ','PHI','PIT','SEA','SF','TB','TEN','WAS']
      .forEach((id, i) => pool.push({ id: `DEF:${id}`, position: 'DEF', adp: cfg.teams * Math.max(1, cfg.rounds - 1) + i / 4, ppg: 0 }));
  }
  if (cfg.k && !pool.some(p => p.position === 'K')) {
    for (let i = 0; i < 32; i++) pool.push({ id: `K:${i}`, position: 'K', adp: cfg.teams * cfg.rounds - 20 + i / 3, ppg: 0 });
  }
  return pool.sort((a, b) => a.adp - b.adp);
}

function runOne(cfg, source, random, aggregate) {
  const available = source.slice(), rosters = Array.from({ length: cfg.teams }, () => []);
  const last = Array.from({ length: cfg.teams }, () => ({}));
  const pendingWait = Array.from({ length: cfg.teams }, () => null);
  const rc = { QB: cfg.qb, RB: cfg.rb, WR: cfg.wr, TE: cfg.te, FLEX: cfg.flex,
    SF: cfg.sf ? 1 : 0, K: cfg.k, DEF: cfg.def, BN: Math.max(0, cfg.rounds - cfg.qb - cfg.rb - cfg.wr - cfg.te - cfg.flex - (cfg.sf ? 1 : 0) - cfg.k - cfg.def) };
  const total = cfg.teams * cfg.rounds;
  for (let pick = 1; pick <= total && available.length; pick++) {
    const round = Math.ceil(pick / cfg.teams), within = (pick - 1) % cfg.teams;
    const slot = round % 2 ? within : cfg.teams - within - 1;
    if (pendingWait[slot]) {
      const observation = pendingWait[slot], bucket = aggregate.waiting[observation.bin];
      bucket.samples++; bucket.predicted += observation.probability;
      if (available.some(p => p.id === observation.playerId)) bucket.returned++;
      pendingWait[slot] = null;
    }
    const roster = rosters[slot], counts = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    roster.forEach(p => { counts[p.position] = (counts[p.position] || 0) + 1; });
    const obligations = Core.remainingObligations(counts, rc, cfg.rounds - round + 1, cfg.sf);
    const candidates = available.slice(0, Math.min(100, available.length)).filter(p => {
      return !['K', 'DEF'].includes(p.position) || (rc[p.position] > 0 && counts[p.position] < rc[p.position]);
    }).map(p => {
      const role = Core.rosterRole(p.position, counts, rc, cfg.sf);
      const utility = Core.rosterSlotUtility(p.position, counts, rc, { role, sf: cfg.sf, draftType: cfg.type });
      const base = Math.max(20, Math.min(98, 86 - Math.abs(pick - p.adp) * 0.48 + Math.max(0, pick - p.adp) * 0.22));
      const bench = role === 'bench1' || role === 'bench2';
      const since = last[slot][p.position] == null ? 99 : round - last[slot][p.position];
      const recentPenalty = bench && ['QB','TE'].includes(p.position) ? Math.max(0, 10 * (1 - since / 6)) : 0;
      const exceptional = Math.max(0, Math.min(1, (pick - p.adp) / Math.max(12, p.adp * 0.65)));
      const ds = Core.decisionScore({ base, utility, bench, deepBench: role === 'bench2',
        quality: Math.min(1, p.ppg / 25), required: obligations.required,
        freePicks: obligations.freePicks, recentPenalty, exceptional, waitLoss: 0 });
      const sigma = Math.max(1, Math.min(10, 0.35 + 0.055 * p.adp));
      const distance = pick - p.adp;
      const adpWeight = distance <= 0 ? Math.exp(-0.5 * Math.pow(distance / sigma, 2))
        : (distance >= sigma * 2 ? 5 : 1 / (1 + 0.12 * distance));
      return { p, ds, weight: adpWeight * Math.pow(Math.max(0.12, ds / 100), 2.2) };
    });
    if (!candidates.length) break;
    const recommendation = candidates.slice().sort((a, b) => b.ds - a.ds || b.weight - a.weight)[0];
    let nextPick = null;
    for (let future = pick + 1; future <= total; future++) {
      const fw = (future - 1) % cfg.teams, fr = Math.ceil(future / cfg.teams);
      const futureSlot = fr % 2 ? fw : cfg.teams - fw - 1;
      if (futureSlot === slot) { nextPick = future; break; }
    }
    if (recommendation && nextPick) {
      const sigma = Math.max(1, Math.min(10, 0.35 + 0.055 * recommendation.p.adp));
      const probability = Core.availabilityProbability({ adp: recommendation.p.adp, center: recommendation.p.adp,
        pick: nextPick, sigma, runPenalty: 0, draftType: cfg.type, sf: cfg.sf });
      pendingWait[slot] = { playerId: recommendation.p.id, probability, bin: waitBin(probability) };
    }
    const chosen = Core.selectDecisionCandidate(candidates, round, 0.8, random);
    if (!chosen) break;
    roster.push(chosen.p); last[slot][chosen.p.position] = round;
    available.splice(available.findIndex(p => p.id === chosen.p.id), 1);
  }
  rosters.forEach(roster => {
    const finalCounts = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    roster.forEach(p => { finalCounts[p.position] = (finalCounts[p.position] || 0) + 1; });
    const finalObligations = Core.remainingObligations(finalCounts, rc, 0, cfg.sf);
    if (finalObligations.required > 0) aggregate.invariants.incompleteRosters++;
    if (roster.length !== cfg.rounds) aggregate.invariants.wrongRosterSize++;
    if (finalCounts.K > cfg.k || finalCounts.DEF > cfg.def) aggregate.invariants.specialTeamsOverfill++;
    ['QB','RB','WR','TE','K','DEF'].forEach(pos => {
      const rounds = roster.map((p, i) => ({ p, round: Math.floor(i / 1) })).filter(x => x.p.position === pos);
      aggregate.counts[pos].push(rounds.length);
    });
    roster.forEach((p, idx) => { // roster insertion order equals that team's round sequence
      const round = idx + 1; aggregate.phase[round <= 4 ? 'early' : round <= 9 ? 'middle' : 'late'][p.position]++;
      aggregate.totalPhase[round <= 4 ? 'early' : round <= 9 ? 'middle' : 'late']++;
    });
    ['QB','TE'].forEach(pos => {
      const rs = roster.map((p, i) => p.position === pos ? i + 1 : null).filter(Boolean);
      for (let n = 1; n <= 3; n++) if (rs[n - 1] != null) aggregate.timing[`${pos}${n}`].push(rs[n - 1]);
    });
  });
}

function main() {
  const cfg = args(process.argv); if (cfg.help) return help();
  if (!['redraft','startup'].includes(cfg.type) || cfg.drafts < 1 || cfg.teams < 2 || cfg.rounds < 1) throw new Error('Invalid configuration');
  const pool = loadPool(cfg), random = rng(cfg.seed);
  if (pool.length < cfg.teams * cfg.rounds) throw new Error(`Player pool has only ${pool.length} players for ${cfg.teams * cfg.rounds} picks`);
  const aggregate = { timing: {}, counts: {}, phase: {}, totalPhase: { early:0, middle:0, late:0 },
    invariants: { incompleteRosters:0, wrongRosterSize:0, specialTeamsOverfill:0 }, waiting: {} };
  ['QB1','QB2','QB3','TE1','TE2','TE3'].forEach(k => aggregate.timing[k] = []);
  ['QB','RB','WR','TE','K','DEF'].forEach(k => aggregate.counts[k] = []);
  ['early','middle','late'].forEach(ph => { aggregate.phase[ph] = { QB:0,RB:0,WR:0,TE:0,K:0,DEF:0 }; });
  ['<50','50-64','65-79','80+'].forEach(bin => { aggregate.waiting[bin] = { samples:0, predicted:0, returned:0 }; });
  const started = Date.now();
  for (let d = 0; d < cfg.drafts; d++) {
    runOne(cfg, pool, random, aggregate);
    if ((d + 1) % Math.max(1, Math.floor(cfg.drafts / 10)) === 0) process.stderr.write(`Completed ${d + 1}/${cfg.drafts}\n`);
  }
  const totalTeams = cfg.drafts * cfg.teams;
  const result = { configuration: cfg, playerPool: pool.length, elapsedSeconds: (Date.now() - started) / 1000,
    model: 'shared-kernel', medianRound: {}, selectionRate: {}, medianFinalCount: {}, maximumFinalCount: {},
    phaseShare: {}, invariants: aggregate.invariants, waitingCalibration: {} };
  Object.keys(aggregate.timing).forEach(k => {
    result.medianRound[k] = median(aggregate.timing[k]);
    result.selectionRate[k] = Math.round(aggregate.timing[k].length * 1000 / totalTeams) / 10;
  });
  Object.keys(aggregate.counts).forEach(k => {
    result.medianFinalCount[k] = median(aggregate.counts[k]);
    result.maximumFinalCount[k] = Math.max(...aggregate.counts[k]);
  });
  Object.keys(aggregate.phase).forEach(ph => { result.phaseShare[ph] = {}; Object.keys(aggregate.phase[ph]).forEach(pos => {
    result.phaseShare[ph][pos] = Math.round(aggregate.phase[ph][pos] * 1000 / Math.max(1, aggregate.totalPhase[ph])) / 10;
  }); });
  Object.keys(aggregate.waiting).forEach(bin => {
    const row = aggregate.waiting[bin];
    result.waitingCalibration[bin] = { samples: row.samples,
      predictedPct: row.samples ? Math.round(row.predicted / row.samples * 10) / 10 : null,
      actualPct: row.samples ? Math.round(row.returned * 1000 / row.samples) / 10 : null };
  });
  console.log(JSON.stringify(result, null, 2));
  if (cfg.json) fs.writeFileSync(cfg.json, JSON.stringify(result, null, 2) + '\n');
}

try { main(); } catch (err) { console.error(err.message); process.exitCode = 1; }
