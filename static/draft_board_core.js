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
// them in lockstep. Replacement and the PPG scale both anchor on the empirical
// starter allocation (effectiveStarters / empiricalSlotAllocation) — the same
// best-available-fills-each-slot index the server grade uses
// (utils.pick_score.empirical_slot_allocation) — which is what lets the two
// surfaces agree. starterCounts remains the fallback when a pool isn't available.
(function (root, factory) {
    var api = factory(root);
    if (typeof module === 'object' && module.exports) module.exports = api;
    root.DraftBoardCore = api;
})(typeof self !== 'undefined' ? self : this, function (root) {
    function clamp01(x) {
        return x < 0 ? 0 : (x > 1 ? 1 : x);
    }

    function PS() {
        return root.BRPickScore;
    }

    var SLOT_MAP = {
        QB: 'QB', RB: 'RB', WR: 'WR', TE: 'TE',
        FLEX: 'FLEX', WRRBTE_FLEX: 'FLEX',
        'RB/WR/TE': 'FLEX', 'WR/RB/TE': 'FLEX', 'W/R/T': 'FLEX',
        WRRB_FLEX: 'RB_WR', RB_WR: 'RB_WR', 'RB/WR': 'RB_WR', 'WR/RB': 'RB_WR', 'W/R': 'RB_WR',
        REC_FLEX: 'WR_TE', WR_TE: 'WR_TE', 'WR/TE': 'WR_TE', 'W/T': 'WR_TE',
        RB_TE: 'RB_TE', 'RB/TE': 'RB_TE', 'R/T': 'RB_TE',
        SUPER_FLEX: 'SF', SFLEX: 'SF',
        'QB/RB/WR/TE': 'SF', 'QB/WR/RB/TE': 'SF',
    };

    // Raw starter counts {QB,SF,RB,WR,TE,FLEX} from a roster_positions array,
    // reconciled to the QB format (mirror of draft_room.js rosterFromLeague plus
    // the defaultRoster SF reconcile). Falls back to a standard 1QB/SF shape.
    function rosterCounts(rosterPositions, sf) {
        var lg = null;
        if (rosterPositions && rosterPositions.length) {
            lg = {QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, RB_WR: 0, WR_TE: 0, RB_TE: 0};
            rosterPositions.forEach(function (s) {
                var k = SLOT_MAP[String(s).toUpperCase()];
                if (k) lg[k]++;
            });
            if (!(lg.QB + lg.RB + lg.WR + lg.TE + lg.FLEX + lg.SF + lg.RB_WR + lg.WR_TE + lg.RB_TE)) lg = null;
        }
        if (!lg) lg = {QB: 1, SF: 0, RB: 2, WR: 3, TE: 1, FLEX: 1, RB_WR: 0, WR_TE: 0, RB_TE: 0};
        if (sf) {
            if (!lg.SF) lg.SF = 1;
            if (!lg.FLEX) lg.FLEX = 1;
        } else {
            lg.SF = 0;
        }
        return lg;
    }

    function redraftVal(p, sf, teams) {
        if (!p) return null;
        // Size-invariant: same 10-team columns the player modal / rankings use.
        // `teams` is kept in the signature so callers stay unchanged.
        var raw = sf
            ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
            : p.redraft_value_1qb;

        if (raw == null || !isFinite(Number(raw))) return null;

        return Number(raw);
    }

    function dynVal(p, sf) {
        if (!p) return null;

        var raw = sf
            ? (p.sf_value != null ? p.sf_value : p.value)
            : p.value;

        if (raw == null || !isFinite(Number(raw))) return null;

        return Number(raw);
    }

    function valOf(p, mode, sf, teams) {
        return mode === 'dynasty'
            ? dynVal(p, sf)
            : redraftVal(p, sf, teams);
    }


// Replacement level = value of the last startable player at a position across
// the league. Missing player values are excluded instead of being interpreted
// as zero-value players.
    function computeReplacement(pool, valFn, starters, teams) {
        var byPos = {
            QB: [],
            RB: [],
            WR: [],
            TE: []
        };

        (pool || []).forEach(function (p) {
            var pos = String((p && p.position) || '').toUpperCase();

            if (!byPos[pos]) return;

            var value = valFn(p);

            // Critical: missing model value != zero model value.
            // Do not allow null/undefined/NaN rows into replacement calculations.
            if (value == null || !isFinite(Number(value))) return;

            byPos[pos].push(Number(value));
        });

        var replacement = {};

        Object.keys(byPos).forEach(function (pos) {
            var arr = byPos[pos].sort(function (a, b) {
                return b - a;
            });

            if (!arr.length) {
                replacement[pos] = null;
                return;
            }

            var starterCount = Number(starters[pos]) || 1;
            var idx = Math.round((Number(teams) || 12) * starterCount) - 1;

            if (idx < 0) idx = 0;
            if (idx >= arr.length) idx = arr.length - 1;

            replacement[pos] = arr[idx];
        });

        return replacement;
    }

    function adpField(mode, sf) {
        if (mode === 'dynasty') return sf ? 'sf_avg_pick' : 'avg_pick';
        return sf ? 'sf_redraft_avg_pick' : 'redraft_avg_pick';
    }

    function adpOf(p, mode, sf) {
        if (mode === 'dynasty') {
            var a = sf ? p.sf_avg_pick : p.avg_pick;
            return a != null ? Number(a) : null;
        }
        var r = sf ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
        if (r != null) return Number(r);
        return p._radp != null ? Number(p._radp) : null;   // value-derived redraft rank
    }

    // One source's ADP on the current dynasty/redraft × 1QB/SF axis — the same
    // fields the Player Rankings ADP columns read from p.adp_by_source.
    function sourceAdpOf(p, source, mode, sf) {
        var by = p && p.adp_by_source && p.adp_by_source[source];
        if (!by) return null;
        var v = by[adpField(mode, sf)];
        if (v == null || !isFinite(Number(v))) return null;
        return Number(v);
    }

    function consensusAdpOf(p, mode, sf) {
        var v = sourceAdpOf(p, 'consensus', mode, sf);
        return v != null ? v : adpOf(p, mode, sf);
    }

    // Canonical ADP resolution for every draft-board surface.  In particular,
    // redraft never reads the dynasty avg_pick axis when its format is missing.
    function resolveAdp(p, selectedSource, mode, sf) {
        var source = selectedSource && selectedSource !== 'auto' ? selectedSource : 'consensus';
        var consensus = sourceAdpOf(p, 'consensus', mode, sf);
        var selected = sourceAdpOf(p, source, mode, sf);
        if (source === 'consensus') selected = consensus;
        if (selected == null) selected = consensus;
        if (selected == null) selected = adpOf(p, mode, sf);
        return {
            selectedAdp: selected, consensusAdp: consensus != null ? consensus : selected,
            selectedSource: source
        };
    }

    // Effective starters {QB,RB,WR,TE} from a roster_positions array.
    function startersFor(rosterPositions, sf) {
        return PS().starterCounts(rosterCounts(rosterPositions, sf));
    }

    // Replacement level = value of the last startable player at a position across
    // the league (teams x starters), from the FULL pool so it is a fixed baseline.
    //
    // Accessor-based so BOTH surfaces share this exact kernel: the caller passes
    // its own value function and its own effective-starters map (from
    // BRPickScore.starterCounts). The draft room passes its state-aware valOf and
    // its user-edited roster; the cheat sheet passes the league roster.

    // Same keys as utils.proj_variant.pick_proj_variant. Draft-room scoring is
    // {ppr, tep, passTd}; Sleeper settings {rec, bonus_rec_te, pass_td} also work.
    function pickProjVariant(scoring) {
        scoring = scoring || {};
        var rec = scoring.ppr != null ? +scoring.ppr
            : (scoring.rec != null ? +scoring.rec : 1);
        var teBonus = scoring.tep != null ? +scoring.tep
            : (scoring.bonus_rec_te != null ? +scoring.bonus_rec_te : 0);
        var passTd = scoring.passTd != null ? +scoring.passTd
            : (scoring.pass_td != null ? +scoring.pass_td : 4);
        var tep = teBonus >= 0.25;
        var six = passTd >= 5.5;
        var base = rec >= 1 ? 'ppr' : rec >= 0.4 ? 'half_ppr' : 'std';
        if (six && tep && base === 'ppr') return '6pt_tep';
        if (six && base === 'ppr') return '6pt_ppr';
        if (six && base === 'half_ppr') return '6pt_half';
        if (tep && base === 'ppr') return 'tep';
        return base;
    }

    // Projected PPG for the given scoring. `proj_ppg` is Sleeper's upcoming-season
    // PPR figure. When scoring is not full PPR / 4-pt passing TD, scale that
    // figure by Sleeper's variant/PPR ratio so half-PPR and 6-pt TD change the
    // number without swapping projection sources. Last-season actuals are never
    // used as a projection stand-in.
    function scoringProjPpg(p, scoring) {
        if (!p) return null;
        var projectionPos = String(p.position || p.pos || '').toUpperCase();

        function unitSafe(v) {
            if (v == null || !isFinite(Number(v))) return null;
            v = Number(v);
            // Corruption detector only: a K/DST value this large is a season total or
            // malformed category, never a credible PPG. Do not clamp it into a score.
            if ((projectionPos === 'K' && v > 30) ||
                ((projectionPos === 'DEF' || projectionPos === 'DST' || projectionPos === 'D/ST') && v > 40)) return null;
            return v;
        }

        var key = pickProjVariant(scoring);
        var canonical = p.projection;
        var canonVar = canonical && canonical.scoring_variant;
        var canonPpg = canonical
            && canonical.projection_type === 'season_average'
            && canonical.unit === 'points_per_game'
            ? unitSafe(canonical.ppg)
            : null;
        // Trust the stamped canonical PPG only when it was scored for this
        // variant (or the stamp predates scoring_variant). A PPR / TEP /
        // pass-TD toggle must not keep showing the previous overlay's number.
        if (canonPpg != null && (!canonVar || canonVar === key)) {
            return canonPpg;
        }

        var base = unitSafe(p.proj_ppg);
        var by = p.proj_ppg_by;
        var variant = unitSafe(by && by[key]);
        var pprVar = unitSafe(by && by.ppr);
        if (variant != null && variant > 0) {
            // Overlay overwrites proj_ppg with the previous scoring's canonical
            // value. Scaling that by variant/PPR would double-adjust, so when
            // the stamp is for a different format just use Sleeper's variant.
            var pprBaseline = !canonVar || canonVar === 'ppr';
            if (pprBaseline && base != null && pprVar != null && pprVar > 0 && key !== 'ppr') {
                return Math.round(base * (variant / pprVar) * 10) / 10;
            }
            if (!pprBaseline || key !== 'ppr' || base == null) {
                return Math.round(variant * 10) / 10;
            }
        }
        return base;
    }

    function scoringProjPts(p, scoring) {
        if (!p) return null;
        var pts = (p.proj_pts != null && isFinite(Number(p.proj_pts))) ? Number(p.proj_pts) : null;
        var base = (p.proj_ppg != null && isFinite(Number(p.proj_ppg))) ? Number(p.proj_ppg) : null;
        var adj = scoringProjPpg(p, scoring);
        if (pts != null && base != null && base > 0 && adj != null) {
            return Math.round(pts * (adj / base) * 10) / 10;
        }
        return pts;
    }

    function ppgOf(p, scoring) {
        var proj = scoring ? scoringProjPpg(p, scoring) : null;
        if (proj == null && p && p.proj_ppg != null) proj = Number(p.proj_ppg);
        if (proj != null && isFinite(proj)) return proj;
        return null;
    }

    // Position PPG scale (replacement -> ~0, elite -> ~1) for the production term.
    // Accessor-based for the same reason as computeReplacement.
    function computePpgScale(pool, ppgFn, starters, teams) {
        ppgFn = ppgFn || ppgOf;
        var byPos = {QB: [], RB: [], WR: [], TE: []};
        pool.forEach(function (p) {
            var pos = (p.position || '').toUpperCase();
            var v = ppgFn(p);
            if (byPos[pos] && v != null) byPos[pos].push(v);
        });
        var out = {};
        Object.keys(byPos).forEach(function (pos) {
            var arr = byPos[pos];
            if (!arr.length) return;
            arr.sort(function (a, b) {
                return b - a;
            });
            var topN = Math.max(1, Math.min(3, arr.length));
            var s = 0;
            for (var i = 0; i < topN; i++) s += arr[i];
            var elite = s / topN;
            var idx = Math.round((teams || 12) * (starters[pos] || 1)) - 1;
            if (idx < 0) idx = 0;
            if (idx >= arr.length) idx = arr.length - 1;
            out[pos] = {repl: arr[idx], elite: elite};
        });
        return out;
    }

    function ppgNorm(p, scale, ppgFn) {
        ppgFn = ppgFn || ppgOf;
        var pos = (p.position || '').toUpperCase();
        var v = ppgFn(p);
        var sc = scale[pos];
        if (v == null || !sc) return null;
        var span = sc.elite - sc.repl;
        if (span <= 0) return clamp01(v / Math.max(sc.elite, 1));
        return clamp01((v - sc.repl) / span);
    }

    // Empirical starter allocation: infer how many starters each position really
    // fields by filling the league's actual starting slots with the best available
    // eligible players, so FLEX/SF shares are outcomes, not a fixed 50/50 guess.
    // Faithful mirror of utils/pick_score.py::empirical_slot_allocation (pinned by
    // tests/test_pick_score_parity.py::test_empirical_slot_allocation_match); the tuple-max tie-break, the
    // eligibility-ascending slot order and the alias table must match the Python
    // exactly or the two surfaces' VOR/PPG replacement levels drift apart.
    function empiricalSlotAllocation(players, slots, numTeams, metric) {
        metric = metric || 'value';
        var aliases = {
            SUPER_FLEX: 'SF', SUPERFLEX: 'SF', SFLEX: 'SF', OP: 'SF',
            QB_RB_WR_TE: 'SF', Q_RB_WR_TE: 'SF',
            'QB/RB/WR/TE': 'SF', 'QB/WR/RB/TE': 'SF',
            WRRBTE_FLEX: 'FLEX', RB_WR_TE: 'FLEX',
            'RB/WR/TE': 'FLEX', 'WR/RB/TE': 'FLEX', 'W/R/T': 'FLEX',
            WRRB_FLEX: 'RB_WR', RB_WR_FLEX: 'RB_WR', RBWR_FLEX: 'RB_WR',
            'RB/WR': 'RB_WR', 'WR/RB': 'RB_WR', 'W/R': 'RB_WR',
            REC_FLEX: 'WR_TE', WRTE_FLEX: 'WR_TE',
            'WR/TE': 'WR_TE', 'W/T': 'WR_TE',
            'RB/TE': 'RB_TE', 'R/T': 'RB_TE',
        };
        var normalized = (slots || []).map(function (s) {
            var u = String(s).toUpperCase();
            return aliases[u] || u;
        });
        if (!normalized.length) normalized = ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX'];
        var eligibility = {
            QB: ['QB'], RB: ['RB'], WR: ['WR'], TE: ['TE'],
            RB_WR: ['RB', 'WR'], WR_TE: ['WR', 'TE'], RB_TE: ['RB', 'TE'],
            FLEX: ['RB', 'WR', 'TE'], SF: ['QB', 'RB', 'WR', 'TE']
        };
        var elig = function (s) {
            return eligibility[s] || [];
        };
        var pool = [];   // [score, index, pos] tuples, mirroring the Python list
        (players || []).forEach(function (player, i) {
            var pos = String((player && (player.position || player.pos)) || '').toUpperCase();
            // Python skips a player whose metric can't be coerced to float; mirror by
            // dropping a present-but-non-numeric value (null/absent coerces to 0).
            var raw = player ? player[metric] : 0;
            if (raw != null && !isFinite(+raw)) return;
            if (pos === 'QB' || pos === 'RB' || pos === 'WR' || pos === 'TE') pool.push([+raw || 0, i, pos]);
        });
        var teams = Math.max(1, Math.trunc(+numTeams) || 1);
        // Dedicated slots (narrowest eligibility) fill first; SF/FLEX last. Stable by
        // original index to match Python's stable sort exactly.
        var order = [];
        for (var t = 0; t < teams; t++) {
            for (var s = 0; s < normalized.length; s++) order.push({slot: normalized[s], idx: order.length});
        }
        order.sort(function (a, b) {
            return (elig(a.slot).length - elig(b.slot).length) || (a.idx - b.idx);
        });
        var used = {}, selected = {QB: 0, RB: 0, WR: 0, TE: 0};
        var tupleGt = function (a, b) {
            if (a[0] !== b[0]) return a[0] > b[0];
            if (a[1] !== b[1]) return a[1] > b[1];
            return a[2] > b[2];
        };
        order.forEach(function (o) {
            var allowed = elig(o.slot);
            if (!allowed.length) return;
            var best = null;
            for (var j = 0; j < pool.length; j++) {
                var item = pool[j];
                if (used[item[1]]) continue;
                if (allowed.indexOf(item[2]) < 0) continue;
                if (best === null || tupleGt(item, best)) best = item;
            }
            if (best) {
                used[best[1]] = true;
                selected[best[2]] += 1;
            }
        });
        return {QB: selected.QB / teams, RB: selected.RB / teams, WR: selected.WR / teams, TE: selected.TE / teams};
    }

    // Effective starters-per-position for a live pool: the empirical allocation
    // when the pool and roster are available, falling back to the fixed
    // starterCounts heuristic otherwise. `counts` is a {QB,SF,RB,WR,TE,FLEX} map;
    // its starting slots are expanded into the slot list the allocator fills.
    // Both the Draft Room and the Cheat Sheet call this so their VOR and PPG
    // replacement levels are anchored the SAME way the server grade is.
    function effectiveStarters(pool, counts, teams, valFn) {
        if (!pool || !pool.length) return PS().starterCounts(counts);
        var slots = [];
        [['QB', 'QB'], ['RB', 'RB'], ['WR', 'WR'], ['TE', 'TE'],
         ['RB_WR', 'RB_WR'], ['WR_TE', 'WR_TE'], ['RB_TE', 'RB_TE'],
         ['FLEX', 'FLEX'], ['SF', 'SF']].forEach(function (pair) {
            var n = Math.max(0, Math.round(+(counts || {})[pair[0]] || 0));
            for (var i = 0; i < n; i++) slots.push(pair[1]);
        });
        var allocPool = pool.map(function (p) {
            return {position: (p.position || '').toUpperCase(), value: valFn(p)};
        });
        return empiricalSlotAllocation(allocPool, slots, teams, 'value');
    }

    // League-relative depth targets.  Single-start, non-flex positions do not get
    // a backup merely because Math.round assigned them a slice of the bench.
    // Instead the bench follows paths into the weekly lineup: RB/WR get the bulk,
    // while SF and TEP explicitly create useful QB/TE depth.
    function posTargets(rc, tep) {
        rc = rc || {};
        tep = tep || 0;
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
        var cap = {
            QB: sf ? 4 : Math.max(1, rc.QB || 0), RB: 7, WR: 7,
            TE: tep > 0 ? Math.max(3, rc.TE || 0) : Math.max(1, rc.TE || 0)
        };
        Object.keys(cap).forEach(function (k) {
            if (t[k] > cap[k]) t[k] = cap[k];
        });
        if (rc.K) t.K = rc.K;
        if (rc.DEF) t.DEF = rc.DEF;
        return t;
    }

    function starterRequirements(rc, sf) {
        rc = rc || {};
        return {
            QB: (rc.QB || 0) + (sf ? (rc.SF || 0) : 0), RB: rc.RB || 0,
            WR: rc.WR || 0, TE: rc.TE || 0, FLEX: rc.FLEX || 0,
            K: rc.K || 0, DEF: rc.DEF || 0
        };
    }

    // Returns the role occupied by the *next* player at pos. FLEX is deliberately
    // allocated only after dedicated RB/WR/TE slots; that is what makes RB3/WR3 a
    // starter while QB2 in 1QB is backup-only.
    function rosterRole(pos, counts, rc, sf) {
        pos = String(pos || '').toUpperCase();
        counts = counts || {};
        rc = rc || {};
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

    // Quality-aware FLEX role: counts tell us whether FLEX is occupied, but not
    // whether this candidate would replace a weak occupant. Dedicated slots are
    // removed position-by-position, then the best remaining RB/WR/TE qualities
    // compete for FLEX. This keeps an excellent WR5 from being mislabeled as mere
    // bench depth when it would actually upgrade a manager's weekly lineup.
    function candidateRosterRole(pos, candidateQuality, rosterQualities, rc, sf) {
        pos = String(pos || '').toUpperCase();
        rc = rc || {};
        var counts = {QB: 0, RB: 0, WR: 0, TE: 0};
        (rosterQualities || []).forEach(function (r) {
            var p = String(r.pos || '').toUpperCase();
            if (counts[p] != null) counts[p]++;
        });
        var basic = rosterRole(pos, counts, rc, sf);
        if (basic === 'starter' || basic === 'flex' || ['RB', 'WR', 'TE'].indexOf(pos) < 0 || !(rc.FLEX || 0)) return basic;
        var req = starterRequirements(rc, sf), by = {RB: [], WR: [], TE: []};
        (rosterQualities || []).forEach(function (r) {
            var p = String(r.pos || '').toUpperCase();
            if (by[p]) by[p].push(+r.quality || 0);
        });
        by[pos].push(+candidateQuality || 0);
        var flexPool = [];
        Object.keys(by).forEach(function (p) {
            by[p].sort(function (a, b) {
                return b - a;
            });
            flexPool = flexPool.concat(by[p].slice(req[p] || 0));
        });
        flexPool.sort(function (a, b) {
            return b - a;
        });
        var cutoff = flexPool[Math.min((req.FLEX || 0), flexPool.length) - 1];
        return cutoff != null && (+candidateQuality || 0) >= cutoff ? 'flex' : basic;
    }

    function rosterSlotUtility(pos, counts, rc, opts) {
        opts = opts || {};
        var role = opts.role || rosterRole(pos, counts, rc, !!opts.sf);
        if (role === 'starter') return 1;
        if (role === 'flex') return String(pos || '').toUpperCase() === 'TE'
            ? ((+opts.tep || 0) > 0 ? 0.86 : 0.70) : 0.96;
        var dynasty = opts.draftType === 'startup' || opts.draftType === 'dynasty';
        var tep = +opts.tep || 0, p = String(pos || '').toUpperCase();
        // K/DEF are lineup obligations, not ordinary bench-depth assets. Keep a
        // tiny non-zero utility rather than a ban (custom leagues may allow a
        // second), but never value K2/DEF2 like another starter.
        if (p === 'K' || p === 'DEF') return 0.06;
        var have = +(counts && counts[p]) || 0;
        if (p === 'QB') {
            if (role === 'bench1') return opts.sf ? 0.78 : (dynasty ? 0.76 : 0.32);
            // Dynasty QB3 in 1QB is a real stash/trade asset. In SF, QB4 is useful
            // insulation, but QB5+ should not inherit the same utility indefinitely.
            if (opts.sf) return have >= 4 ? 0.18 : 0.55;
            if (dynasty) return have >= 3 ? 0.10 : 0.55;
            return 0.12;
        }
        if (p === 'TE') {
            if (role === 'bench1') return tep > 0 ? 0.72 : (dynasty ? 0.62 : 0.32);
            if (tep > 0) return have >= 4 ? 0.24 : 0.48;
            if (dynasty) return have >= 4 ? 0.12 : 0.44;
            return 0.16;
        }
        if (p === 'RB') return role === 'bench1' ? 0.82 : 0.68;
        if (p === 'WR') return role === 'bench1' ? 0.78 : 0.64;
        return 1;
    }

    // 1QB (not Superflex) and 1TE (no premium) are streamable weekly slots.
    // An empty starter is a real hole, but live recs were treating it like a
    // must-start WR/RB and leaping remaining skill-position depth by a full
    // starter-vs-bench gap (~8 Decision Score points). Superflex QB and TEP
    // keep full starter utility.
    function isStreamableSingleSlot(pos, rc, opts) {
        pos = String(pos || '').toUpperCase();
        opts = opts || {};
        rc = rc || {};
        var dedicated = +(starterRequirements(rc, !!opts.sf)[pos] || 0);
        return (pos === 'QB' && !opts.sf && dedicated <= 1)
            || (pos === 'TE' && (+opts.tep || 0) <= 0 && dedicated <= 1);
    }

    // Scarcity-urgency scale for waitLoss. Multi-slot holes keep full (or near-
    // full) cliff weight; a lone streamable QB/TE slot is muted so a shelf drop
    // there cannot leap a similarly-valued WR/RB.
    function waitLossScaleFor(pos, missingDedicated, opts) {
        var miss = Math.max(0, +missingDedicated || 0);
        if (miss >= 2) return 1;
        pos = String(pos || '').toUpperCase();
        opts = opts || {};
        var streamable = (pos === 'QB' && !opts.sf) || (pos === 'TE' && (+opts.tep || 0) <= 0);
        if (miss >= 1) return streamable ? 0.4 : 0.6;
        return 0.4;
    }

    // Live-draft fit pressure for the next player at a position.  The ordinary
    // slot utility intentionally answers only "starter, flex, or bench?"; this
    // adds a small amount of information about how many lineup paths remain.
    // FLEX is one shared obligation: its pressure is divided among positions
    // that can still occupy it rather than credited in full to every candidate.
    function positionNeedUtility(pos, counts, rc, opts) {
        opts = opts || {};
        counts = counts || {};
        rc = rc || {};
        pos = String(pos || '').toUpperCase();
        var role = opts.role || rosterRole(pos, counts, rc, !!opts.sf);
        var base = rosterSlotUtility(pos, counts, rc, Object.assign({}, opts, {role: role}));
        var req = starterRequirements(rc, !!opts.sf);

        // Cap streamable 1QB/1TE starters so they compete with WR4/RB4 depth
        // instead of outranking it. Redraft is the streamable case; dynasty keeps
        // more of the starter premium (young QBs/TEs are long-term assets).
        if (role === 'starter' && isStreamableSingleSlot(pos, rc, opts)) {
            var cap = opts.draftType === 'redraft' ? 0.74 : 0.80;
            if (base > cap) base = cap;
        }

        if (role !== 'starter' && role !== 'flex') return base;

        var missingDedicated = Math.max(0, (+req[pos] || 0) - (+counts[pos] || 0));
        var flexUsed = Math.max(0, (+counts.RB || 0) - req.RB)
            + Math.max(0, (+counts.WR || 0) - req.WR)
            + Math.max(0, (+counts.TE || 0) - req.TE);
        var remainingFlex = Math.max(0, req.FLEX - flexUsed);
        var flexShare = 0;
        if (remainingFlex > 0 && (pos === 'RB' || pos === 'WR' || pos === 'TE')) {
            var weights = {RB: 1, WR: 1, TE: Math.min(1, 0.55 + Math.max(0, +opts.tep || 0) * 0.3)};
            var total = 0;
            ['RB', 'WR', 'TE'].forEach(function (p) {
                // Capacity prevents a position that already covers all dedicated and
                // FLEX paths from claiming a share of the remaining opportunity.
                var capacity = Math.max(0, (+req[p] || 0) + req.FLEX - (+counts[p] || 0));
                total += capacity * weights[p];
                if (p === pos) flexShare = capacity * weights[p];
            });
            flexShare = total > 0 ? flexShare / total : 0;
        }

        // One open dedicated starter is the existing baseline. Extra dedicated
        // paths and the candidate's share of FLEX add at most ~5 Decision Score
        // points because decisionScore converts utility at 38 points per unit.
        var pressure = Math.max(0, missingDedicated - 1) * 0.075
            + remainingFlex * flexShare * 0.035;
        return Math.min(base + 0.13, base + pressure);
    }

    // Realistic hard roster ceilings for positions whose extra depth has sharply
    // diminishing utility. This is deliberately format-aware: TE2 and an unusual
    // late TE3 remain legal in ordinary redraft, while TEP/multi-TE and dynasty
    // retain deeper rooms. It blocks pathological TE4+ accumulation, not value picks.
    function positionRosterLimit(pos, rc, opts) {
        pos = String(pos || '').toUpperCase();
        rc = rc || {};
        opts = opts || {};
        if (pos === 'K' || pos === 'DEF') return +rc[pos] || 0;
        if (pos !== 'TE') return Infinity;
        var starters = +rc.TE || 0;
        var dynasty = opts.draftType === 'startup' || opts.draftType === 'dynasty';
        if (dynasty) return Math.max(5, starters + 4);
        if ((+opts.tep || 0) > 0) return Math.max(4, starters + 3);
        return Math.max(1, starters + 2);
    }

    function remainingObligations(counts, rc, remainingPicks, sf, opts) {
        counts = counts || {};
        rc = rc || {};
        opts = opts || {};
        var req = starterRequirements(rc, sf);
        var missing = {
            QB: Math.max(0, req.QB - (+counts.QB || 0)), RB: Math.max(0, req.RB - (+counts.RB || 0)),
            WR: Math.max(0, req.WR - (+counts.WR || 0)), TE: Math.max(0, req.TE - (+counts.TE || 0)),
            K: Math.max(0, req.K - (+counts.K || 0)), DEF: Math.max(0, req.DEF - (+counts.DEF || 0))
        };
        var flexUsed = Math.max(0, (+counts.RB || 0) - req.RB) + Math.max(0, (+counts.WR || 0) - req.WR)
            + Math.max(0, (+counts.TE || 0) - req.TE);
        missing.FLEX = Math.max(0, req.FLEX - flexUsed);
        var required = 0;
        Object.keys(missing).forEach(function (k) {
            required += missing[k];
        });
        // Lineup holes that should steer redraft recs: dedicated RB/WR/FLEX, plus
        // QB/TE only when they are not weekly-streamable (1QB / non-premium TE).
        // K/DEF stay in `required` so late-round fill math is unchanged.
        var tep = +opts.tep || 0;
        var streamableQb = !sf && (+req.QB || 0) <= 1;
        var streamableTe = tep <= 0 && (+req.TE || 0) <= 1;
        var lineupHoles = missing.RB + missing.WR + missing.FLEX
            + (streamableQb ? 0 : missing.QB)
            + (streamableTe ? 0 : missing.TE);
        return {
            missing: missing, required: required, remaining: Math.max(0, +remainingPicks || 0),
            freePicks: Math.max(0, (+remainingPicks || 0) - required),
            lineupHoles: lineupHoles
        };
    }

    // Pure, testable final layer used only for live recommendations. A great fall
    // can overcome fit, but ordinary backup-only value pays a persistent cost.
    //
    // Redraft recs optimize this-season starting-lineup strength (the same
    // inputs that become points-for / playoff odds), not Pick Score BPA and not
    // a per-pick playoff Monte Carlo. `lineupHoles` is the count of still-open
    // RB/WR/FLEX starters (plus SF QB / premium TE). Empty 1QB/1TE slots are
    // excluded so streamable fills cannot leap skill depth. Pick Score weights
    // stay in the grade kernel; this tax is recommendation-only.
    function redraftBenchHoleTax(holes) {
        var n = Math.max(0, +holes || 0);
        if (n <= 0) return 0;
        return Math.min(11, 3.5 + 2.5 * Math.min(n, 3));
    }

    // 1QB QB2 / 1TE TE2: leftover NFL starters keep 85-99 Pick Scores after the
    // slot is filled because weekly PPG never went away. The 0.32 utility hit
    // (~26 points) is not enough when remaining skill is late-round (PS ~55-70),
    // so the rec list can lead with four "QB filled · backup-only value" rows.
    // Tax that starter inflation. Fades in the last quarter of the draft so a
    // backup QB/TE is still a normal closer. Opt-in via streamableBackup.
    function streamableBackupTax(o) {
        if (!o || !o.streamableBackup || !o.bench) return 0;
        var base = +o.base || 0;
        var inflation = Math.max(0, base - 62);
        var tax = Math.min(22, 8 + inflation * 0.4);
        var rd = +o.round || 0, total = +o.totalRounds || 0;
        if (rd > 0 && total > 0) {
            var fadeStart = total * 0.75;
            if (rd >= fadeStart) {
                var span = Math.max(1, total - fadeStart);
                tax *= Math.max(0, Math.min(1, (total - rd) / span));
            }
        }
        return tax;
    }

    function decisionScore(o) {
        o = o || {};
        var base = +o.base || 0, util = o.utility == null ? 1 : +o.utility;
        var score = base + (util - 1) * 38;
        var redraft = o.draftType === 'redraft';
        var holes = Math.max(0, +o.lineupHoles || 0);
        if (o.bench) {
            score += (+o.quality || 0) * 5;
            if ((+o.required || 0) > 0 && (+o.freePicks || 0) <= 1) score -= 7;
            if ((+o.required || 0) > 0 && (+o.freePicks || 0) <= 0) score -= 13;
            if (o.deepBench) score -= 5;
            // Drafting a backup immediately after filling a single-starter position is
            // especially wasteful: the board has barely changed and the manager has
            // not used intervening picks on more flexible depth. This fades smoothly.
            score -= Math.max(0, +o.recentPenalty || 0);
            // Truly exceptional falls may buy back some fit cost, but ordinary ADP
            // values cannot use the generic late-round score inflation as an escape.
            score += Math.max(0, Math.min(12, (+o.exceptional || 0) * 12));
            // Mid-draft luxury bench while a real starter/flex hole remains. Opt-in
            // via draftType + lineupHoles so existing callers/tests stay unchanged.
            if (redraft) score -= redraftBenchHoleTax(holes);
            score -= streamableBackupTax(o);
        } else if (redraft) {
            // Starter/flex: tilt toward this-year production among similar PS.
            score += Math.max(0, Math.min(4, (+o.quality || 0) * 4));
        }
        if ((+o.waitLoss || 0) > 0) {
            // waitLossScale (default 1) lets the caller damp positional-scarcity urgency
            // at positions where only a single starter is still needed (e.g. TE, or QB
            // in 1QB): the shelf cliff is real, but grabbing the one body you need is not
            // as urgent as filling a multi-slot need, so an elite single-slot player no
            // longer leaps a higher-value pick that fills a deeper need.
            var wls = o.waitLossScale == null ? 1 : Math.max(0, +o.waitLossScale);
            var waitBonus = Math.min(9, (+o.waitLoss || 0) * 0.30) * Math.max(0.35, util);
            // Urgency should separate close candidates, not flatten every excellent
            // option against the 99 ceiling. Shrink positive bonuses as headroom runs
            // out so 96/95/94-quality decisions remain visibly distinct.
            waitBonus = Math.min(waitBonus, Math.max(0, (99 - score) * 0.35));
            // Damp AFTER the headroom cap so a single-slot need's scarcity urgency is
            // genuinely reduced even on a steep cliff (where the raw bonus would
            // otherwise saturate the cap and the scale would be a no-op).
            score += waitBonus * wls;
        }
        // A player who is likely to survive until the manager's next pick consumes
        // scarce current-pick capital without capturing much value. Keep this
        // separate from waitLoss: waitLoss rewards a genuine positional shelf cliff,
        // while waitPenalty discounts this specific player's probability of
        // returning. Both derive from the same survival estimate the base pick
        // score deliberately does NOT touch, so survival is counted exactly once,
        // here on the point scale.
        score -= Math.max(0, Math.min(10, +o.waitPenalty || 0));
        // Redraft handcuff insurance: backing up one of the manager's own RBs
        // protects a starter's workload. A small tilt, applied here rather than in
        // the shared pick-score kernel so it can't be double-counted or warped by
        // the kernel's depth-normalization / display-relabel math.
        score += Math.max(0, Math.min(8, +o.handcuffBonus || 0));
        // Draft phase changes the question gradually: late reserve picks should
        // prefer a credible path to useful workload over tiny projection/ADP edges.
        // The caller supplies an evidence-based utility; age alone is never enough.
        score += Math.max(-3, Math.min(7, +o.upsideBonus || 0));
        // Prospective bye concentration: mild scheduling risk, not a draft grade.
        // Bounded so bye never outweighs real starter need or Pick Score gaps.
        score -= Math.max(0, Math.min(4, +o.byePenalty || 0));
        return Math.max(1, Math.min(99, Math.round(score)));
    }

    // Map byeWeekSeverity level → Decision Score points (caller computes the
    // prospective delta via byeWeekSeverity with/without the candidate).
    function byeSeverityPenalty(level) {
        if (level === 'severe') return 4;
        if (level === 'meaningful') return 2.5;
        if (level === 'mild') return 1;
        return 0;
    }

    function draftPhase(round, totalRounds) {
        var den = Math.max(1, (+totalRounds || 16) - 4);
        var x = Math.max(0, Math.min(1, ((+round || 1) - 4) / den));
        return x * x * (3 - 2 * x);
    }

    // Conservative late-round signal. `path` must describe an actual workload,
    // handcuff, breakout-role, or lineup path. Youth only interacts with that
    // path, so a young player with no role receives no automatic preference.
    function lateRoundUtility(o) {
        o = o || {};
        var path = Math.max(0, Math.min(1, +o.path || 0));
        var above = Math.max(0, Math.min(1, +o.aboveReplacement || 0));
        var tier = Math.max(0, Math.min(1, +o.tierQuality || 0));
        var ppg = Math.max(0, Math.min(1, +o.ppgQuality || 0));
        var quality = 0.55 * above + 0.25 * tier + 0.20 * ppg;
        var ageInteraction = o.youngWithPath ? path * 0.10 : 0;
        var upside = Math.max(0, Math.min(1, 0.55 * path + 0.45 * quality + ageInteraction));
        var utility = Math.max(0, Math.min(1, o.functionalUtility == null ? 1 : +o.functionalUtility));
        var need = Math.max(0, Math.min(1, o.rosterNeedPath == null ? 0.5 : +o.rosterNeedPath));
        return Math.max(0, Math.min(1, upside * utility * (0.65 + 0.35 * need)));
    }

    function lateRoundUpsideBonus(o) {
        o = o || {};
        var phase = draftPhase(o.round, o.totalRounds);
        return Math.max(-3, Math.min(7, phase * 10 * (lateRoundUtility(o) - 0.35)));
    }

    // Role/path evidence only. Projected PPG is quality (ppgQuality), not a path.
    // Youth without a workload, handcuff, or breakout role must stay at 0.
    function lateRoundPathEvidence(o) {
        o = o || {};
        var path = 0;
        if (o.breakoutScore != null && isFinite(+o.breakoutScore)) {
            path = Math.max(path, clamp01(+o.breakoutScore / 100));
        }
        if (o.projectedRole != null && isFinite(+o.projectedRole)) {
            path = Math.max(path, clamp01(+o.projectedRole));
        }
        if (o.handcuff) path = Math.max(path, 0.75);
        return path;
    }

    function summarizeHistoricalAlternatives(rows, selectedId) {
        rows = (rows || []).slice().sort(function (a, b) {
            return (+b.decisionScore || 0) - (+a.decisionScore || 0)
                || (+b.absolutePickScore || 0) - (+a.absolutePickScore || 0);
        });
        var sel = null;
        for (var i = 0; i < rows.length; i++) {
            if (String(rows[i].id) === String(selectedId)) { sel = rows[i]; break; }
        }
        if (!sel || !rows.length) return null;
        var alts = rows.filter(function (r) { return String(r.id) !== String(selectedId); }).slice(0, 5);
        return {
            selectedScore: sel.decisionScore,
            bestAlternativeScore: rows[0].decisionScore,
            bestAlternative: alts[0] || null,
            topAlternatives: alts
        };
    }

    // Players already off the board before pickNo: prior draft picks plus
    // keepers that have not been written into the pick map yet. Never includes
    // picks at or after pickNo (no future leak).
    function takenBeforePick(o) {
        o = o || {};
        var pickNo = Math.max(1, parseInt(o.pickNo, 10) || 1);
        var picks = o.picks || {};
        var taken = {};
        Object.keys(picks).forEach(function (k) {
            var n = parseInt(k, 10);
            if (!isFinite(n) || n >= pickNo) return;
            var pk = picks[k];
            if (pk && pk.id != null) taken[String(pk.id)] = true;
        });
        var onBoard = {};
        Object.keys(picks).forEach(function (k) {
            var pk = picks[k];
            if (pk && pk.id != null) onBoard[String(pk.id)] = true;
        });
        (o.keepers || []).forEach(function (k) {
            if (k && k.id != null && !onBoard[String(k.id)]) taken[String(k.id)] = true;
        });
        return taken;
    }

    // Reconstruct the manager's roster + remaining pool *immediately before*
    // pickNo. Only prior own picks and keepers count toward counts/qualities;
    // later picks must not leak into Decision Score.
    function historicalDecisionContext(o) {
        o = o || {};
        var pickNo = Math.max(1, parseInt(o.pickNo, 10) || 1);
        var picks = o.picks || {};
        var isMyPick = typeof o.isMyPick === 'function' ? o.isMyPick : function () { return false; };
        var playersById = o.playersById || {};
        var qualityOf = typeof o.qualityOf === 'function' ? o.qualityOf : function () { return 0.35; };
        var vorPositive = typeof o.vorPositive === 'function' ? o.vorPositive : function () { return true; };
        var teams = Math.max(1, +o.teams || 12);
        var rounds = Math.max(1, +o.rounds || 16);
        var excludeSt = o.excludeSpecialTeams !== false;
        var taken = o.taken || takenBeforePick({
            pickNo: pickNo, picks: picks, keepers: o.keepers || []
        });
        var pool = o.pool || [];
        var selected = o.selected || null;
        var keepers = o.keepers || [];

        var counts = { QB: 0, RB: 0, WR: 0, TE: 0 };
        var qualities = [];
        var qualByPos = {};
        var myRbTeams = {};
        var absorbedIds = {};

        function absorb(pk) {
            if (!pk || pk.id == null) return;
            var id = String(pk.id);
            if (absorbedIds[id]) return;
            absorbedIds[id] = true;
            var f = playersById[id] || pk;
            var pos = String(f.position || f.pos || '').toUpperCase();
            if (counts[pos] != null) {
                counts[pos]++;
                var q = qualityOf(f);
                if (q == null || !isFinite(+q)) q = 0.35;
                qualities.push({ pos: pos, quality: +q });
                if (vorPositive(f)) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
            }
            if (pos === 'RB' && f.team) myRbTeams[String(f.team)] = true;
        }

        // Keepers sit on the roster before the draft clock starts. Only the
        // viewer's keepers count toward reconstruction — rivals are taken but
        // not on my roster. Skip when viewer roster is unknown.
        var myKeepers = [];
        if (typeof o.myKeepers === 'function') {
            myKeepers = keepers.filter(o.myKeepers);
        } else if (o.viewerRosterId != null && o.viewerRosterId !== '') {
            myKeepers = keepers.filter(function (k) {
                return String(k.rosterId) === String(o.viewerRosterId);
            });
        }
        myKeepers.forEach(absorb);

        Object.keys(picks).forEach(function (k) {
            var n = parseInt(k, 10);
            if (!isFinite(n) || n >= pickNo) return;
            if (!isMyPick(n)) return;
            absorb(picks[k]);
        });

        function isSpecial(pos) {
            return ['K', 'DEF', 'DST', 'D/ST'].indexOf(pos) >= 0;
        }

        var remaining = pool.filter(function (p) {
            if (!p || p.id == null) return false;
            if (taken[String(p.id)]) return false;
            var pos = String(p.position || p.pos || '').toUpperCase();
            if (excludeSt && isSpecial(pos)) return false;
            return true;
        });
        if (selected && selected.id != null &&
            !remaining.some(function (p) { return String(p.id) === String(selected.id); })) {
            remaining = remaining.concat([selected]);
        }

        var remainingMyPicks = 0;
        var tot = teams * rounds;
        for (var hn = pickNo; hn <= tot; hn++) {
            if (isMyPick(hn)) remainingMyPicks++;
        }

        return {
            pickNo: pickNo,
            counts: counts,
            qualities: qualities,
            qualByPos: qualByPos,
            remaining: remaining,
            myRbTeams: myRbTeams,
            round: Math.floor((pickNo - 1) / teams) + 1,
            remainingMyPicks: remainingMyPicks,
            teams: teams,
            rounds: rounds,
            taken: taken
        };
    }

    // Rank the historical remaining pool by Decision Score. Prefer a scoreRow
    // callback (page supplies absolute Pick Score + decisionScore); or pass
    // prebuilt rows. Returns summarizeHistoricalAlternatives output.
    function rankHistoricalAlternatives(o) {
        o = o || {};
        var selectedId = o.selectedId;
        var rows = o.rows;
        if (!rows) {
            var ctx = o.context || historicalDecisionContext(o);
            var scoreRow = typeof o.scoreRow === 'function' ? o.scoreRow : null;
            if (!scoreRow) return null;
            rows = (ctx.remaining || []).map(function (p) {
                var scored = scoreRow(p, ctx);
                if (!scored) return null;
                return {
                    id: p.id,
                    player: p,
                    absolutePickScore: scored.absolutePickScore,
                    decisionScore: scored.decisionScore
                };
            }).filter(Boolean);
        }
        return summarizeHistoricalAlternatives(rows, selectedId);
    }

    // One unused bench cover per starter. Same-position first, then FLEX-eligible
    // (RB/WR/TE). A single reserve cannot cover two starters on the same bye.
    function assignByeCover(players) {
        var used = {};
        (players || []).forEach(function (p) {
            if (!p || p.role !== 'starter') return;
            if (p.coverQuality != null && isFinite(+p.coverQuality)) return;
            var pos = String(p.pos || '').toUpperCase();
            var best = null, bestQ = -1;
            (players || []).forEach(function (c, idx) {
                if (!c || c === p || c.role === 'starter') return;
                var key = String(c.id != null ? c.id : (c.name || idx));
                if (used[key]) return;
                if (+c.bye && +p.bye && +c.bye === +p.bye) return;
                var cpos = String(c.pos || '').toUpperCase();
                var same = cpos === pos;
                var flex = ['RB', 'WR', 'TE'].indexOf(pos) >= 0 && ['RB', 'WR', 'TE'].indexOf(cpos) >= 0;
                if (!same && !flex) return;
                var q = Math.max(0, Math.min(1, +c.quality || 0));
                var score = q + (same ? 0.05 : 0);
                if (score > bestQ) { bestQ = score; best = {c: c, key: key, q: q}; }
            });
            if (best) {
                used[best.key] = true;
                p.coverQuality = best.q;
            } else {
                p.coverQuality = 0;
            }
        });
        return players || [];
    }

    // Gap bands (Decision Score points): none <4, modest <9, material <15, severe ≥15.
    // significantReach also needs outside-market + not BPA + survivePct ≥ ADP_REACH_SURVIVE (20).
    function opportunityCostVerdict(o) {
        o = o || {};
        var gap = Math.max(0, (+o.bestAlternativeScore || 0) - (+o.selectedScore || 0));
        var severity = gap < 4 ? 'none' : gap < 9 ? 'modest' : gap < 15 ? 'material' : 'severe';
        var significantReach = gap >= 9 && !!o.outsideMarketRange && !o.isBpa
            && (o.survivePct == null || +o.survivePct >= ADP_REACH_SURVIVE);
        return {gap: gap, severity: severity, significantReach: significantReach};
    }

    // significantSteal: marketFall ≥ max(8, 0.5×σ) and Board PS ≥ 80.
    function significantSteal(o) {
        o = o || {};
        var threshold = Math.max(8, 0.5 * Math.max(0, +o.adpUncertainty || 0));
        return (+o.marketFall || 0) >= threshold && (+o.boardPickScore || 0) >= 80;
    }

    // Deep Dive / edges copy. Softens when ADP or survivePct inputs are missing.
    function formatOpportunityCostCopy(o) {
        o = o || {};
        var gap = Math.max(0, +o.gap || 0);
        var altName = String(o.altName || '').trim();
        if (!altName || gap < 4) return '';
        var text = 'Preferred ' + altName + ' by ~' + Math.round(gap) + ' Decision Score';
        var low = o.confidence === 'low';
        if (!low && o.confidence !== 'high') {
            if (o.adpMissing || o.survivePctMissing) low = true;
            else if (('adp' in o && o.adp == null) || ('survivePct' in o && o.survivePct == null)) low = true;
        }
        return low ? (text + ' · lower confidence') : text;
    }

    // Impact-based bye concentration. Callers classify roles from the same
    // optimal lineup used by grading and provide each player's best unused cover.
    function byeWeekSeverity(players, opts) {
        opts = opts || {};
        players = assignByeCover(players || []);
        var weeks = {};
        (players || []).forEach(function (p) {
            var week = +p.bye || 0;
            if (!week) return;
            var role = p.role || 'fringe';
            var roleImpact = role === 'starter' ? 1 : role === 'primary' ? 0.30 : 0.08;
            var quality = 0.60 + 0.40 * Math.max(0, Math.min(1, +p.quality || 0));
            var pos = String(p.pos || '').toUpperCase();
            var stream = ((pos === 'QB' && !opts.sf) || (pos === 'TE' && !(+opts.tep > 0))) ? 0.55 : 1;
            var relief = 0.35 * Math.max(0, Math.min(1, +p.coverQuality || 0));
            var impact = Math.max(0, roleImpact * quality * stream - relief);
            if (!weeks[week]) weeks[week] = {week: week, score: 0, starters: 0, players: []};
            weeks[week].score += impact;
            weeks[week].players.push(p);
            if (role === 'starter' && impact >= 0.35) weeks[week].starters++;
        });
        var out = Object.keys(weeks).map(function (k) {
            var w = weeks[k];
            w.score += 0.25 * Math.max(0, w.starters - 2);
            w.level = (w.score >= 2.8 || w.starters >= 4) ? 'severe'
                : w.score >= 1.8 ? 'meaningful' : w.score >= 1 ? 'mild' : 'none';
            return w;
        });
        out.sort(function (a, b) {
            return b.score - a.score;
        });
        return out;
    }

    // When ranking recommendations for a future owned pick (you pick at #9,
    // the clock is at #1), scale the live decision score by the chance that
    // player is still on the board. A small floor keeps 0% names in a stable
    // order among themselves without letting 1.01 talent outrank someone who
    // will actually be there.
    var REC_FUTURE_SURVIVE_FLOOR = 0.08;

    function futurePickDecisionScore(score, survivePct) {
        score = +score || 0;
        if (survivePct == null || !isFinite(Number(survivePct))) return score;
        var p = Math.max(0, Math.min(100, Number(survivePct))) / 100;
        return score * (REC_FUTURE_SURVIVE_FLOOR + (1 - REC_FUTURE_SURVIVE_FLOOR) * p);
    }

    function decisionBand(rows, round, persona) {
        rows = rows || [];
        round = +round || 1;
        persona = +persona || 0.8;
        var best = 0;
        rows.forEach(function (r) {
            if ((+r.ds || 0) > best) best = +r.ds || 0;
        });
        var width = 4 + Math.min(6, Math.max(0, round - 4) * 0.65) + Math.max(0, persona - 0.8) * 2;
        var eligible = rows.filter(function (r) {
            return (+r.weight || 0) > 0 && (+r.ds || 0) >= best - width;
        });
        return eligible.length ? eligible : rows.slice();
    }

    // Shared final CPU chooser: production mocks and the headless benchmark both
    // use the same decision band, top-field cap and weighted sampling. Callers own
    // candidate economics/personality; this kernel owns how close alternatives
    // become an actual pick.
    function selectDecisionCandidate(rows, round, persona, random) {
        var eligible = decisionBand(rows, round, persona).slice(0, 8);
        if (!eligible.length) return null;
        var sum = 0;
        eligible.forEach(function (r) {
            sum += Math.max(0, +r.weight || 0);
        });
        if (sum <= 0) return eligible[0];
        var roll = (typeof random === 'function' ? random() : Math.random()) * sum;
        for (var i = 0; i < eligible.length; i++) {
            roll -= Math.max(0, +eligible[i].weight || 0);
            if (roll <= 0) return eligible[i];
        }
        return eligible[0];
    }

    function normalCdf(z) {
        var t = 1 / (1 + 0.2316419 * Math.abs(z));
        var d = 0.3989423 * Math.exp(-z * z / 2);
        var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return z > 0 ? 1 - p : p;
    }

    // Empirical return-rate curves from 10k-draft matched-format benchmarks.
    // Raw Normal(ADP, sigma) is directionally useful but systematically optimistic
    // in redraft 1QB and dynasty SF, where many managers attack the same shelves.
    // Keep profiles together and interpolate smoothly rather than scattering format
    // multipliers through Draft Room.
    var AVAILABILITY_CALIBRATION = {
        redraft_1qb: [[0, 0], [9, 9], [57, 25], [73, 27], [93, 51], [100, 60]],
        redraft_sf: [[0, 0], [10, 11], [57, 36], [71, 55], [88, 73], [100, 85]],
        startup_1qb: [[0, 0], [11, 23], [57, 67], [71, 74], [84, 81], [100, 90]],
        startup_sf: [[0, 0], [12, 16], [57, 36], [72, 38], [91, 44], [100, 50]],
    };

    function calibrateAvailability(probability, draftType, sf) {
        var type = draftType === 'redraft' ? 'redraft' : 'startup';
        var points = AVAILABILITY_CALIBRATION[type + '_' + (sf ? 'sf' : '1qb')];
        var p = Math.max(0, Math.min(100, +probability || 0));
        for (var i = 1; i < points.length; i++) {
            if (p <= points[i][0]) {
                var left = points[i - 1], right = points[i];
                var ratio = (p - left[0]) / Math.max(1, right[0] - left[0]);
                return Math.round((left[1] + (right[1] - left[1]) * ratio) * 10) / 10;
            }
        }
        return points[points.length - 1][1];
    }

    // Probability a player survives through `pick`. Production Draft Room and
    // the headless calibration benchmark share this exact probability kernel;
    // callers may supply an observed-draft center and positional-run penalty.
    function availabilityProbability(o) {
        o = o || {};
        var center = o.center == null ? +o.adp : +o.center;
        var sigma = Math.max(0.01, +o.sigma || 1), pick = +o.pick || 0;
        var probability = 1 - normalCdf((pick - center) / sigma);
        probability *= 1 - Math.max(0, Math.min(1, +o.runPenalty || 0));
        var raw = Math.max(0, Math.min(100, Math.round(probability * 100)));
        return o.calibrate === false ? raw : calibrateAvailability(raw, o.draftType, !!o.sf);
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
        var v = valOf(p, mode, sf, teams);
        for (var i = 0; i < tbl.length; i++) {
            if (v >= tbl[i]) return i + 1;
        }
        return tbl.length + 1;
    }

    function maxVal(pool, mode, sf, teams) {
        var m = 0;
        pool.forEach(function (p) {
            var v = valOf(p, mode, sf, teams);
            if (v > m) m = v;
        });
        return m;
    }

    // Autodraft extra multiplier on the live decision score.
    //
    // CPU picks are sampled with an ADP likelihood, so a 13-spot reach has
    // near-zero Gaussian weight. Autodraft is argmax of score: the old uncapped
    // 1.35x open-starter boost could take a TE a full round early in a 1TE
    // league even when that player was still on the board at the turn. Keep the
    // backup-reach / overfill guards, pull toward an open starter only near ADP,
    // and wait on single-slot reaches that survive until the next owned pick.
    var AUTO_STARTER_BOOST = 1.35;
    var AUTO_OVERFILL = 0.4;
    var AUTO_WAIT_TURN = 0.35;
    var AUTO_STARTER_REACH_ROUNDS = 0.5;
    var AUTO_WAIT_SURVIVE = 55;   // same "Can wait" threshold the sidebar uses
    var AUTO_RUN_SURVIVE = 40;    // below this, a positional run is underway

    function autoDraftNeedMultiplier(o) {
        o = o || {};
        var pos = String(o.pos || '').toUpperCase();
        if (pos === 'K' || pos === 'DEF') return 1;
        var have = +o.have || 0;
        var target = +o.target || 0;
        var sSlots = +o.starterSlots || 0;
        var adp = o.adp;
        var pickNo = +o.pickNo || 0;
        var teams = Math.max(1, +o.teams || 12);
        var nextPick = o.nextPick == null ? null : +o.nextPick;
        var survive = o.surviveProb == null ? null : +o.surviveProb;
        var tep = +o.tep || 0;
        var sf = !!o.sf;
        var qbStarters = o.qbStarters == null ? (sf ? 2 : 1) : +o.qbStarters;

        if (adp != null && adp < 9000 && pickNo < adp) {
            if ((pos === 'QB' && have >= qbStarters) ||
                (pos === 'TE' && sSlots <= 1 && have >= 1)) return 0;
        }
        if (target > 0 && have >= target) return AUTO_OVERFILL;

        var openStarter = sSlots > 0 && have < sSlots;
        if (!openStarter) return 1;

        var reach = (adp != null && adp < 9000) ? Math.max(0, adp - pickNo) : 0;
        var nearValue = adp == null || adp >= 9000 || reach <= teams * AUTO_STARTER_REACH_ROUNDS;
        // 1TE (no TEP) and 1QB: a body whose ADP is still at/after the next pick,
        // or who the survival model says will be there, is a free wait. A real
        // positional run (survival already crushed) is allowed to take them now.
        var singleSlot = (pos === 'TE' && sSlots <= 1 && tep <= 0) || (pos === 'QB' && !sf);
        var adpAfterTurn = nextPick != null && adp != null && adp < 9000 && adp >= nextPick;
        var likelyReturn = survive != null && survive >= AUTO_WAIT_SURVIVE;
        var runAway = survive != null && survive < AUTO_RUN_SURVIVE;
        if (singleSlot && reach > 0 && !runAway && (adpAfterTurn || likelyReturn)) {
            return AUTO_WAIT_TURN;
        }
        return nearValue ? AUTO_STARTER_BOOST : 1;
    }

    // Deep Dive "reach" vs leftover ADP. Historical ADP is a mean across full
    // drafts, so after picks 1–8 the best remaining player often has ADP 11 at
    // pick 9 — a cluster, not a reach. Remaining-board BPA (and a tight ADP
    // cluster with no clear favorite) plus players who would not last to the
    // next pick are on-market. Raw Steal/Value (fell past ADP) is unchanged.
    var ADP_REACH_CLUSTER = 1.0;   // within this many ADP of the remaining #1 = co-BPA
    var ADP_REACH_SURVIVE = 20;    // below this % chance to last to the next pick = not a reach
    var ADP_REACH_RAW = -5;        // same raw early-by-N threshold the ledger used

    function adpUncertainty(player, round, sourceValues) {
        var vals = (sourceValues || []).map(Number).filter(function (v) {
            return isFinite(v) && v > 0;
        })
            .sort(function (a, b) {
                return a - b;
            });
        var spread = 0;
        if (vals.length >= 3) {
            var med = vals[Math.floor(vals.length / 2)];
            var dev = vals.map(function (v) {
                return Math.abs(v - med);
            }).sort(function (a, b) {
                return a - b;
            });
            spread = 1.4826 * dev[Math.floor(dev.length / 2)];
        }
        var r = Math.max(1, Number(round) || 1);
        var depth = r <= 2 ? 4 : r <= 4 ? 7 : r <= 6 ? 11 : r <= 9 ? 17 : r <= 12 ? 25 : 32 + (r - 13) * 2;
        return Math.max(depth, spread * 1.5);
    }

    function isRemainingAdpBpa(playerAdp, bestRemainingAdp, cluster) {
        if (playerAdp == null || bestRemainingAdp == null) return false;
        var a = Number(playerAdp), b = Number(bestRemainingAdp);
        if (!isFinite(a) || !isFinite(b)) return false;
        var c = cluster == null ? ADP_REACH_CLUSTER : Number(cluster);
        if (!isFinite(c)) c = ADP_REACH_CLUSTER;
        return a <= b + c;
    }

    function bestRemainingAdp(pool, takenIds, adpFn) {
        var best = null;
        if (!pool || !pool.length) return best;
        takenIds = takenIds || {};
        for (var i = 0; i < pool.length; i++) {
            var p = pool[i];
            if (!p || takenIds[String(p.id)]) continue;
            var a = adpFn ? adpFn(p) : p.adp;
            if (a == null || !isFinite(Number(a))) continue;
            a = Number(a);
            if (best == null || a < best) best = a;
        }
        return best;
    }

    function adpDeltaVerdict(o) {
        o = o || {};
        var diff = o.diff;
        if (diff == null || !isFinite(Number(diff))) return {label: '-', cls: 'na'};
        diff = Number(diff);
        if (diff >= 8) return {label: 'Steal', cls: 'steal'};
        if (diff >= 3) return {label: 'Value', cls: 'value'};
        var tolerance = o.tolerance != null ? Math.max(0, Number(o.tolerance)) : Math.abs(ADP_REACH_RAW);
        if (diff > -tolerance) return diff < ADP_REACH_RAW
            ? {label: 'Aggressive', cls: 'aggressive'} : {label: 'Fair', cls: 'fair'};
        if (o.isBpa) return {label: 'Fair', cls: 'fair'};
        var survive = o.survivePct;
        if (survive != null && isFinite(Number(survive)) && Number(survive) < ADP_REACH_SURVIVE) {
            return {label: 'Fair', cls: 'fair'};
        }
        return {label: 'Reach', cls: 'reach'};
    }

    // Timeline / net-ADP delta: leftover-ADP bias on a remaining BPA (or a
    // player who would not last) sits on the line instead of in the reach zone.
    // Steals (positive) and true reaches (skipped better ADP, likely to last)
    // keep their raw pick − ADP.
    function adpBoardDelta(o) {
        o = o || {};
        var diff = o.diff;
        if (diff == null || !isFinite(Number(diff))) return null;
        diff = Number(diff);
        if (diff >= 0) return diff;
        if (o.isBpa) return 0;
        var survive = o.survivePct;
        if (survive != null && isFinite(Number(survive)) && Number(survive) < ADP_REACH_SURVIVE) {
            return 0;
        }
        return diff;
    }

    // Last-resort K/DEF fill: which required special-teams slot to take when a
    // team has no discretionary picks left. Order follows that team's plan so
    // mocks are not a global kicker-then-defense script.
    function specialTeamsFillPos(needK, needDef, plan) {
        needK = Math.max(0, +needK || 0);
        needDef = Math.max(0, +needDef || 0);
        if (needK + needDef <= 0) return null;
        if (needK > 0 && needDef > 0) {
            plan = plan || {};
            var pickPos = (plan.prefer === 'K' || plan.prefer === 'DEF')
                ? plan.prefer
                : ((+plan.order || 0) < 0.5 ? 'K' : 'DEF');
            if (plan.flip) pickPos = pickPos === 'K' ? 'DEF' : 'K';
            return pickPos;
        }
        return needK > 0 ? 'K' : 'DEF';
    }

    // Auction nomination $ guidance (R02.3): normalize BR value onto a share of
    // remaining budget per remaining roster slot. Explicitly labeled guidance —
    // not a predicted clearing price.
    function suggestAuctionBid(opts) {
        opts = opts || {};
        var value = Number(opts.value);
        var maxValue = Number(opts.maxValue);
        var remainingBudget = Number(opts.remainingBudget);
        var slotsLeft = Number(opts.slotsLeft);
        if (!isFinite(value) || value < 0) value = 0;
        if (!isFinite(maxValue) || maxValue <= 0) maxValue = 1;
        if (!isFinite(remainingBudget) || remainingBudget < 0) remainingBudget = 0;
        if (!isFinite(slotsLeft) || slotsLeft < 1) slotsLeft = 1;
        var share = value / maxValue;
        if (share > 1) share = 1;
        var perSlot = remainingBudget / slotsLeft;
        var dollars = Math.round(share * perSlot);
        if (value > 0 && remainingBudget >= 1 && dollars < 1) dollars = 1;
        if (dollars < 0) dollars = 0;
        if (dollars > remainingBudget) dollars = Math.floor(remainingBudget);
        return { dollars: dollars, label: 'guidance' };
    }

    // Custom board (pin / mute / neighbor re-anchor). Shared by the Cheat Sheet
    // and Draft Room so a PRO board follows the manager into the live room.
    function applyCustomBoardOverrides(list, overridesMap) {
        overridesMap = overridesMap || {};
        var custom = false;
        for (var k in overridesMap) {
            if (Object.prototype.hasOwnProperty.call(overridesMap, k)) {
                custom = true;
                break;
            }
        }
        var byId = {};
        (list || []).forEach(function (p, i) {
            p._mr = i;
            byId[p.id] = p;
        });
        (list || []).forEach(function (p) {
            var o = custom ? overridesMap[p.id] : null;
            if (o && o.p) {
                p.bucket = -1;
                p.moved = false;
                p._eff = p._mr;
            } else if (o && o.m) {
                p.bucket = 1;
                p.moved = false;
                p._eff = p._mr;
            } else if (o && o.r != null) {
                p.bucket = 0;
                p.moved = true;
                p._eff = o.r;
            } else {
                p.bucket = 0;
                p.moved = false;
                p._eff = p._mr;
            }
        });
        if (!custom) return list;
        (list || []).filter(function (p) { return p.moved; })
            .sort(function (a, b) {
                var sa = (overridesMap[a.id] && overridesMap[a.id].s) || 0;
                var sb = (overridesMap[b.id] && overridesMap[b.id].s) || 0;
                return sa - sb;
            })
            .forEach(function (p) {
                var o = overridesMap[p.id] || {};
                var aP = o.a ? byId[o.a] : null;
                var bP = o.b ? byId[o.b] : null;
                if (aP && bP) p._eff = (aP._eff + bP._eff) / 2;
                else if (aP) p._eff = aP._eff + 0.5;
                else if (bP) p._eff = bP._eff - 0.5;
            });
        (list || []).sort(function (a, b) {
            if (a.bucket !== b.bucket) return a.bucket - b.bucket;
            return a._eff - b._eff || a._mr - b._mr;
        });
        return list;
    }

    return {
        rosterCounts: rosterCounts, startersFor: startersFor,
        redraftVal: redraftVal, dynVal: dynVal, valOf: valOf, adpOf: adpOf,
        adpField: adpField, sourceAdpOf: sourceAdpOf, consensusAdpOf: consensusAdpOf, resolveAdp: resolveAdp,
        computeReplacement: computeReplacement, ppgOf: ppgOf,
        pickProjVariant: pickProjVariant, scoringProjPpg: scoringProjPpg, scoringProjPts: scoringProjPts,
        computePpgScale: computePpgScale, ppgNorm: ppgNorm,
        empiricalSlotAllocation: empiricalSlotAllocation, effectiveStarters: effectiveStarters,
        tierOf: tierOf, maxVal: maxVal, posTargets: posTargets,
        starterRequirements: starterRequirements, rosterRole: rosterRole, candidateRosterRole: candidateRosterRole,
        rosterSlotUtility: rosterSlotUtility, positionNeedUtility: positionNeedUtility,
        isStreamableSingleSlot: isStreamableSingleSlot, waitLossScaleFor: waitLossScaleFor,
        positionRosterLimit: positionRosterLimit,
        remainingObligations: remainingObligations,
        decisionScore: decisionScore, futurePickDecisionScore: futurePickDecisionScore,
        draftPhase: draftPhase, lateRoundUtility: lateRoundUtility,
        lateRoundUpsideBonus: lateRoundUpsideBonus,
        lateRoundPathEvidence: lateRoundPathEvidence,
        summarizeHistoricalAlternatives: summarizeHistoricalAlternatives,
        takenBeforePick: takenBeforePick,
        historicalDecisionContext: historicalDecisionContext,
        rankHistoricalAlternatives: rankHistoricalAlternatives,
        assignByeCover: assignByeCover,
        opportunityCostVerdict: opportunityCostVerdict, significantSteal: significantSteal,
        formatOpportunityCostCopy: formatOpportunityCostCopy,
        byeWeekSeverity: byeWeekSeverity, byeSeverityPenalty: byeSeverityPenalty,
        REC_FUTURE_SURVIVE_FLOOR: REC_FUTURE_SURVIVE_FLOOR,
        decisionBand: decisionBand, selectDecisionCandidate: selectDecisionCandidate,
        availabilityProbability: availabilityProbability, calibrateAvailability: calibrateAvailability,
        autoDraftNeedMultiplier: autoDraftNeedMultiplier,
        specialTeamsFillPos: specialTeamsFillPos,
        suggestAuctionBid: suggestAuctionBid,
        applyCustomBoardOverrides: applyCustomBoardOverrides,
        ADP_REACH_CLUSTER: ADP_REACH_CLUSTER, ADP_REACH_SURVIVE: ADP_REACH_SURVIVE,
        adpUncertainty: adpUncertainty,
        isRemainingAdpBpa: isRemainingAdpBpa, bestRemainingAdp: bestRemainingAdp,
        adpDeltaVerdict: adpDeltaVerdict, adpBoardDelta: adpBoardDelta,
    };
});
