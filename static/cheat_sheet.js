// Draft Cheat Sheet — a printable, pre-draft view of the SAME board the Draft
// Room ranks on. It ranks the shared /api/league-players pool by BRPickScore
// (the identical, parity-tested engine the draft room's best-available list and
// the server grade use), with inputs assembled through DraftBoardCore (the same
// value / ADP / replacement / tier primitives). So the sheet's order is the draft
// room's order; the room only adds the live pick slot, roster need and survival
// terms once you are on the clock.
//
// Redraft mode ranks on redraft value; Dynasty mode on dynasty value. Both use
// the league-roster replacement level, and Superflex re-prices QBs by moving that
// replacement line, not by a hand bump.
(function () {
    var cfg = window.__cheatCfg || {};
    var C = window.DraftBoardCore;
    var PS = window.BRPickScore;
    var LIMIT = 300;

    // The backend only emits this redraft signal when independent, confidence-
    // weighted market evidence clears its threshold; baseline-only rows stay "—".
    // Authoritative response metadata controls the entire column.  It is updated
    // on every load, so an unavailable signal cannot leave an invisible/stale
    // column (or CSV field) behind and automatically returns after a good refresh.
    var SHOW_MARKET_VS_ADP = false;
    var SHOW_HISTORICAL = false;
    var showMarket = function (dyn) {
        return !dyn && SHOW_MARKET_VS_ADP;
    };
    // Hist column is redraft-only (ADP axis is 1QB redraft). Trends stays
    // available in dynasty as research with an explicit redraft caveat.
    var showHist = function (dyn) {
        return !dyn && SHOW_HISTORICAL;
    };
    var showTrends = function () {
        return SHOW_HISTORICAL;
    };
    var currentTab = 'board';
    var trendsCache = null;
    var trendsPos = '';
    var trendsRequest = 0;
    var trendsTier = 'top_12';
    var trendsPicks = {};
    var trendsCohort = null;
    var trendsCohortRequest = 0;
    var trendsDockOpen = true;
    var TRENDS_SCOUT_PREVIEW = 3;
    var TAB_PANELS = {
        board: 'cs-panel-board',
        pos: 'cs-panel-pos',
        trends: 'cs-panel-trends',
        logic: 'cs-panel-logic'
    };
    // Warehouse P(top-12) at a skill position is ~5–8%. 25%+ is a strong cell.
    // Do not tint market_higher / signed ADP edge on the board: round-1 market
    // hit rates are 60–90%, so Bijan / Gibbs / Chase would all read as -20.
    var HIST_STRONG_PCT = 25;
    var HIST_TIER_SHORT = { top_5: 'top-5', top_12: 'top-12', top_24: 'top-24' };

    // Same {ppr, tep, passTd} contract as Draft Room setup (readScoring / scoringCfg).
    function normalizeScoring(s) {
        s = s || {};
        var ppr = s.ppr != null ? Number(s.ppr) : 1;
        if (!isFinite(ppr)) ppr = 1;
        // Snap to the three setup options so <select> values always match.
        ppr = ppr >= 0.75 ? 1 : (ppr >= 0.25 ? 0.5 : 0);
        var tep = s.tep != null ? Number(s.tep) : 0;
        if (!isFinite(tep)) tep = 0;
        tep = tep >= 0.75 ? 1 : (tep >= 0.25 ? 0.5 : 0);
        var passTd = Number(s.passTd != null ? s.passTd : (s.pass_td != null ? s.pass_td : 4));
        return { ppr: ppr, tep: tep, passTd: passTd >= 6 ? 6 : 4 };
    }
    function scoringCfg() {
        return normalizeScoring(state && state.scoring);
    }
    function scoringLabel(sc) {
        sc = sc || scoringCfg();
        var ppr = sc.ppr === 1 ? 'Full PPR' : (sc.ppr === 0.5 ? 'Half PPR' : 'Standard');
        var bits = [ppr];
        if (sc.tep > 0) bits.push('TEP +' + sc.tep);
        if (sc.passTd >= 6) bits.push('6-pt Pass TD');
        return bits.join(' · ');
    }
    function readScoringFromUi() {
        var pprEl = $('csPpr');
        var tepEl = $('csTep');
        var passTdEl = $('csPassTd');
        return normalizeScoring({
            ppr: pprEl ? parseFloat(pprEl.value) : scoringCfg().ppr,
            tep: tepEl ? parseFloat(tepEl.value) : scoringCfg().tep,
            passTd: passTdEl ? parseFloat(passTdEl.value) : scoringCfg().passTd
        });
    }
    function syncScoringUi() {
        var sc = scoringCfg();
        [['csPpr', String(sc.ppr)], ['csTep', String(sc.tep)], ['csPassTd', String(sc.passTd)]].forEach(function (pair) {
            var el = $(pair[0]);
            if (!el || el.value === pair[1]) return;
            el.value = pair[1];
            // CSD listens for change to refresh the visible trigger label.
            try { el.dispatchEvent(new Event('change', {bubbles: true})); } catch (e) {}
        });
    }
    function scoringProjPpg(p) {
        if (!p) return null;
        if (C && C.scoringProjPpg) return C.scoringProjPpg(p, scoringCfg());
        return (p.proj_ppg != null && isFinite(Number(p.proj_ppg))) ? Number(p.proj_ppg) : null;
    }

    function histExampleHit(finish, fallback) {
        if (fallback && fallback.hit_label) {
            return { tier: fallback.hit_tier || '', label: fallback.hit_label };
        }
        var n = Number(finish);
        if (!isFinite(n) || n < 1) return null;
        if (n <= 5) return { tier: 'top_5', label: 'Top 5' };
        if (n <= 12) return { tier: 'top_12', label: 'Top 12' };
        if (n <= 24) return { tier: 'top_24', label: 'Top 24' };
        return { tier: 'miss', label: 'Outside top 24' };
    }

    function setTrendsNavOffset() {
        var nav = document.querySelector('.top-nav');
        var h = nav ? Math.round(nav.getBoundingClientRect().height) : 0;
        document.documentElement.style.setProperty('--cs-nav-offset', h + 'px');
    }
    setTrendsNavOffset();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setTrendsNavOffset);
    }
    window.addEventListener('resize', setTrendsNavOffset);
    if (typeof ResizeObserver === 'function') {
        var navEl = document.querySelector('.top-nav');
        if (navEl) new ResizeObserver(setTrendsNavOffset).observe(navEl);
    }

    var state = {
        mode: cfg.mode === 'dynasty' ? 'dynasty' : 'redraft',
        sf: !!cfg.isSuperflex,
        scoring: normalizeScoring(cfg.scoring),
        filter: false,
        needsFilter: false,
        hideDrafted: false,
        adpSource: 'auto',
        search: '',
        posFilter: 'ALL',
        done: new Set(),
        pickSlot: 0,           // 1-based snake seat; 0 = no projected-pick lines
        sortKey: 'vor',        // Big Board display sort; model order stays VOR
        sortDir: -1,           // -1 desc, +1 asc. Default = VOR descending.
    };
    var teams = Number(cfg.numTeams) || 12;
    var allPlayers = [];
    var players = [];
    var tierThresholds = {};
    var adpSourceOptions = {};   // {redraft:[{value,label}], startup:[...], rookie:[...]}
    var draftedIds = null;       // Set of live-drafted player ids, or null if none
    var recommendationOrder = null; // Draft Room snapshot: player id -> supplemental REC rank
    var scrollToFirstAvailable = false; // one-shot when opened from an active draft
    var myCounts = null;         // {QB,RB,WR,TE} drafted by the viewer, or null
    var liveDraftId = null;      // id of the live draft being polled, or null
    var pollTimer = null;
    var maxVor = 1;
    var loading = false;
    var loadError = '';
    var playerRequest = 0;      // only the newest mode/source request may update the board
    var playerAbort = null;
    var scheduleRanks = {};     // player id -> full fantasy-season strength-of-schedule rank
    var scheduleRequest = 0;    // stale schedule responses must not repaint a newer player pool

    // ── Custom draft board (pro): per-player overrides on top of the model board.
    // Intent, not absolute positions: {r: fractional rank on the model scale, p:
    // pinned, m: muted}. A moved player's `r` sits between the model ranks of its
    // chosen neighbours, so drag-drop and the arrows place it exactly and stay
    // stable no matter how many others move. Persisted per league + mode + format
    // so a refresh of the model values keeps the user's intent. See
    // docs/custom-draft-board.md.
    var overrides = {};
    var _ovKey = null;
    var _ovPush = null;      // debounce timer for the server save
    var editBoard = false;   // whether per-row edit controls are shown
    var _flashId = null;     // player row to flash after a move
    function boardKey() {
        return state.mode + ':' + (state.sf ? 'sf' : '1qb');
    }

    function ovKey() {
        return 'csboard:' + (cfg.leagueId || 'guest') + ':' + boardKey();
    }

    function loadOverrides() {
        overrides = {};
        if (!cfg.hasPremium) return;
        try {
            overrides = JSON.parse(localStorage.getItem(ovKey()) || '{}') || {};
        } catch (e) {
            overrides = {};
        }
    }

    function ensureOverrides() {
        var k = ovKey();
        if (k !== _ovKey) {
            _ovKey = k;
            loadOverrides();                 // localStorage cache: instant
            syncOverridesFromServer(k);      // durable, cross-device: async
        }
    }

    // Pull the durable copy from the server (source of truth across devices) and
    // adopt it if it differs from the local cache. Ignored if the user has since
    // switched boards.
    function syncOverridesFromServer(forKey) {
        if (!cfg.hasPremium) return;
        var p = ['board_key=' + encodeURIComponent(boardKey())];
        if (cfg.leagueId) p.push('league_id=' + encodeURIComponent(cfg.leagueId));
        if (cfg.platform) p.push('platform=' + encodeURIComponent(cfg.platform));
        if (cfg.season) p.push('season=' + encodeURIComponent(cfg.season));
        fetch('/api/draft-board/overrides?' + p.join('&'), {cache: 'no-store'})
            .then(function (r) {
                return r.json();
            })
            .then(function (resp) {
                if (ovKey() !== forKey) return;   // switched boards; drop stale response
                var srv = (resp && resp.overrides) || {};
                if (JSON.stringify(srv) !== JSON.stringify(overrides)) {
                    overrides = srv;
                    try {
                        localStorage.setItem(ovKey(), JSON.stringify(overrides));
                    } catch (e) { /* ignore */
                    }
                    compute();
                    render();
                }
            })
            .catch(function () { /* offline: keep the local cache */
            });
    }

    function pushOverridesToServer() {
        if (!cfg.hasPremium) return;
        if (_ovPush) clearTimeout(_ovPush);
        var bk = boardKey(), snap = JSON.parse(JSON.stringify(overrides));
        _ovPush = setTimeout(function () {
            fetch('/api/draft-board/overrides', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    platform: cfg.platform,
                    league_id: cfg.leagueId,
                    season: cfg.season,
                    board_key: bk,
                    overrides: snap
                }),
            }).catch(function () { /* best effort; localStorage still holds it */
            });
        }, 600);
    }

    function saveOverrides() {
        Object.keys(overrides).forEach(function (id) {
            var o = overrides[id];
            if (!o || (o.r == null && !o.p && !o.m)) delete overrides[id];
        });
        try {
            localStorage.setItem(ovKey(), JSON.stringify(overrides));
        } catch (e) { /* storage full/blocked */
        }
        pushOverridesToServer();
    }

    function hasOverrides() {
        return cfg.hasPremium && Object.keys(overrides).length > 0;
    }

    // Current display order of the normal (not pinned/muted) bucket.
    function normOrder() {
        return players.filter(function (p) {
            return p.bucket === 0;
        });
    }

    // A monotonic placement sequence so applyOverrides can re-anchor moves in the
    // order they were made (keeps chains of moves stable across a model re-rank).
    function nextSeq() {
        var mx = 0;
        Object.keys(overrides).forEach(function (id) {
            var s = overrides[id] && overrides[id].s;
            if (s > mx) mx = s;
        });
        return mx + 1;
    }

    // Move `id` between two neighbour players (either may be null at the board ends).
    // We remember the neighbours' ids (a/b) so the move re-anchors to the same
    // players after a value refresh, and keep `r` as an immediate/fallback rank.
    function boardPlaceBetween(id, above, below) {
        var o = overrides[id] || {};
        delete o.p;
        delete o.m;
        if (above) o.a = above.id; else delete o.a;
        if (below) o.b = below.id; else delete o.b;
        o.r = (above && below) ? (above._eff + below._eff) / 2 : (above ? above._eff + 0.5 : (below ? below._eff - 0.5 : 0));
        o.s = nextSeq();
        overrides[id] = o;
        saveOverrides();
        compute();
        // A drop that resolved to the player's own model spot (no net displacement)
        // is a no-op: discard it silently rather than storing a marker-less override.
        var pl = null;
        for (var i = 0; i < players.length; i++) {
            if (players[i].id === id) {
                pl = players[i];
                break;
            }
        }
        if (pl && pl.moved && !pl.ov) {
            delete overrides[id];
            saveOverrides();
            compute();
        } else {
            _flashId = id;
        }
        render();
    }

    // Place player `id` at display index `pos` within the full normal bucket.
    function boardMoveTo(id, pos) {
        var others = normOrder().filter(function (p) {
            return p.id !== id;
        });
        pos = Math.max(0, Math.min(others.length, pos));
        boardPlaceBetween(id, pos > 0 ? others[pos - 1] : null, pos < others.length ? others[pos] : null);
    }

    // Arrow nudge: move one row up (dir +1) or down (dir -1) within the normal bucket.
    function boardNudge(id, dir) {
        var norm = normOrder(), idx = -1;
        for (var i = 0; i < norm.length; i++) {
            if (norm[i].id === id) {
                idx = i;
                break;
            }
        }
        if (idx < 0) return;                              // pinned/muted: not in this bucket
        if (dir > 0 && idx === 0) return;                 // already at the top
        if (dir < 0 && idx === norm.length - 1) return;   // already at the bottom
        boardMoveTo(id, dir > 0 ? idx - 1 : idx + 1);
    }

    function boardPin(id) {
        var was = overrides[id] && overrides[id].p;
        var o = overrides[id] || {};
        delete o.r;
        delete o.a;
        delete o.b;
        delete o.s;
        delete o.m;
        if (was) delete o.p; else o.p = true;
        overrides[id] = o;
        _flashId = id;
        saveOverrides();
        compute();
        render();
    }

    function boardMute(id) {
        var was = overrides[id] && overrides[id].m;
        var o = overrides[id] || {};
        delete o.r;
        delete o.a;
        delete o.b;
        delete o.s;
        delete o.p;
        if (was) delete o.m; else o.m = true;
        overrides[id] = o;
        _flashId = id;
        saveOverrides();
        compute();
        render();
    }

    // Revert a single player to its model spot (undo one override).
    function boardRevert(id) {
        if (!overrides[id]) return;
        delete overrides[id];
        _flashId = id;
        saveOverrides();
        compute();
        render();
    }

    function boardReset() {
        overrides = {};
        saveOverrides();
        compute();
        render();
    }

    // Pointer-based drag reorder (mouse + touch). The grip handle starts a drag; a
    // drop line tracks the insertion point among the normal-bucket rows, and the
    // release writes the player's new fractional rank via boardMoveTo.
    function setupDragReorder(panel) {
        var scroll = panel.querySelector('.cs-tbl-scroll') || panel;
        var dragId = null, startY = 0, didMove = false, line = null;

        function normRows() {
            var byId = {};
            players.forEach(function (p) {
                byId[p.id] = p;
            });
            return Array.prototype.slice.call(panel.querySelectorAll('tbody tr.cs-p')).filter(function (tr) {
                var p = byId[tr.getAttribute('data-id')];
                return p && p.bucket === 0 && tr.offsetParent !== null;
            });
        }

        // The neighbour players on either side of the drop point (from the visible
        // rows), plus the content-space Y at which to draw the drop line.
        function dropAt(clientY) {
            var byId = {};
            players.forEach(function (p) {
                byId[p.id] = p;
            });
            var rows = normRows().filter(function (tr) {
                return tr.getAttribute('data-id') !== dragId;
            });
            var pos = rows.length;
            for (var i = 0; i < rows.length; i++) {
                var r = rows[i].getBoundingClientRect();
                if (clientY < r.top + r.height / 2) {
                    pos = i;
                    break;
                }
            }
            var srect = scroll.getBoundingClientRect(), y = 0;
            if (pos < rows.length) y = rows[pos].getBoundingClientRect().top - srect.top + scroll.scrollTop;
            else if (rows.length) y = rows[rows.length - 1].getBoundingClientRect().bottom - srect.top + scroll.scrollTop;
            return {
                above: pos > 0 ? byId[rows[pos - 1].getAttribute('data-id')] : null,
                below: pos < rows.length ? byId[rows[pos].getAttribute('data-id')] : null,
                y: y,
            };
        }

        function drawLine(y) {
            if (!line) {
                line = document.createElement('div');
                line.className = 'cs-drop-line';
                scroll.appendChild(line);
            }
            line.style.top = y + 'px';
        }

        function cleanup() {
            if (line) {
                line.remove();
                line = null;
            }
            var dg = panel.querySelector('tr.cs-dragging');
            if (dg) dg.classList.remove('cs-dragging');
            dragId = null;
            didMove = false;
        }

        panel.addEventListener('pointerdown', function (e) {
            var h = e.target.closest('.cs-drag');
            if (!h) return;
            e.preventDefault();
            e.stopPropagation();
            dragId = h.getAttribute('data-id');
            startY = e.clientY;
            didMove = false;
            try {
                h.setPointerCapture(e.pointerId);
            } catch (_) {
            }
        });
        panel.addEventListener('pointermove', function (e) {
            if (dragId == null) return;
            if (!didMove) {
                if (Math.abs(e.clientY - startY) < 4) return;   // ignore jitter / a plain tap
                didMove = true;
                var row = panel.querySelector('tr.cs-p[data-id="' + (window.CSS && CSS.escape ? CSS.escape(dragId) : dragId) + '"]');
                if (row) row.classList.add('cs-dragging');
            }
            e.preventDefault();
            var r = scroll.getBoundingClientRect(), M = 42;   // auto-scroll near the edges
            if (e.clientY < r.top + M) scroll.scrollTop -= 10; else if (e.clientY > r.bottom - M) scroll.scrollTop += 10;
            drawLine(dropAt(e.clientY).y);
        });
        panel.addEventListener('pointerup', function (e) {
            if (dragId == null) return;
            var id = dragId, moved = didMove, dr = dropAt(e.clientY);
            cleanup();
            if (moved) boardPlaceBetween(id, dr.above, dr.below);
        });
        panel.addEventListener('pointercancel', cleanup);
    }

    // Apply custom overrides on top of the model board. Pinned players float to the
    // top, muted sink to the bottom; a moved player is re-anchored between the
    // neighbours it was dropped between (by id), so the move survives a model
    // re-rank. After sorting we renumber the RK column, stamp each moved row's net
    // move within the normal bucket for the chip, and let it adopt the tier it
    // settled into so cliff headers stay monotonic instead of repeating.
    function applyOverrides() {
        var custom = hasOverrides();
        var byId = {};
        players.forEach(function (p, i) {
            p._mr = i;
            byId[p.id] = p;
        });   // model (VOR) order
        players.forEach(function (p) {
            var o = custom ? overrides[p.id] : null;
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
            }   // provisional
            else {
                p.bucket = 0;
                p.moved = false;
                p._eff = p._mr;
            }
        });
        if (!custom) {
            players.forEach(function (p) {
                p.grp = p.dtier;
                p.grpLabel = 'Tier ' + p.dtier;
                p.ov = null;
                p.ovN = 0;
            });
            return;
        }
        // Re-anchor moved players against their neighbours' *current* effective rank,
        // oldest placement first so a chain of moves resolves consistently.
        players.filter(function (p) {
            return p.moved;
        })
            .sort(function (a, b) {
                return (overrides[a.id].s || 0) - (overrides[b.id].s || 0);
            })
            .forEach(function (p) {
                var o = overrides[p.id], aP = o.a ? byId[o.a] : null, bP = o.b ? byId[o.b] : null;
                if (aP && bP) p._eff = (aP._eff + bP._eff) / 2;
                else if (aP) p._eff = aP._eff + 0.5;
                else if (bP) p._eff = bP._eff - 0.5;
                // both anchors gone from the pool: keep the stored fallback rank (o.r)
            });
        players.sort(function (a, b) {
            if (a.bucket !== b.bucket) return a.bucket - b.bucket;
            return a._eff - b._eff || a._mr - b._mr;
        });
        // Net-move chip is measured within the normal bucket only, so a player's own
        // pins/mutes don't distort another player's displayed ▲/▼ count.
        var modelNormPos = {}, mnp = 0;
        players.slice().sort(function (a, b) {
            return a._mr - b._mr;
        })
            .forEach(function (x) {
                if (x.bucket === 0) modelNormPos[x.id] = mnp++;
            });
        var runTier = 1, np = 0;
        players.forEach(function (x, i) {
            x.rk = i + 1;
            if (x.bucket === -1) {
                x.grp = -1;
                x.grpLabel = 'Pinned';
                x.ov = 'pin';
                x.ovN = 0;
            } else if (x.bucket === 1) {
                x.grp = 1e9;
                x.grpLabel = 'Muted';
                x.ov = 'mute';
                x.ovN = 0;
            } else {
                var d = modelNormPos[x.id] - np;
                np++;
                if (!x.moved) {
                    runTier = x.dtier;
                    x.ov = null;
                    x.ovN = 0;
                } else {
                    x.ov = d > 0 ? 'up' : (d < 0 ? 'down' : null);
                    x.ovN = Math.abs(d);
                }
                x.grp = runTier;
                x.grpLabel = 'Tier ' + runTier;
            }
        });
    }

    function scoringAxisKey() {
        return state.mode === 'dynasty' ? 'startup' : 'redraft';
    }

    // Career window by age, position-aware: RBs peak and fade youngest, QBs latest,
    // TEs are late-blooming and durable. Position-agnostic bands mislabeled, e.g.,
    // a 31-year-old QB (still prime) as "Fading" like a 31-year-old RB.
    function ageBands(pos) {
        switch ((pos || '').toUpperCase()) {
            case 'RB':
                return [23, 26, 28];
            case 'QB':
                return [26, 31, 35];
            case 'TE':
                return [25, 28, 31];
            default:
                return [24, 27, 30];   // WR
        }
    }

    function youthWindow(age, pos) {
        if (age == null) return ['', ''];
        var b = ageBands(pos);
        if (age <= b[0]) return ['Ascending', 'win-asc'];
        if (age <= b[1]) return ['Prime', 'win-prime'];
        if (age <= b[2]) return ['Win-now', 'win-now'];
        return ['Fading', 'win-fade'];
    }

    // ADP source the sheet is showing. "Auto" is Consensus — the same blended
    // column Player Rankings defaults to — not the Sleeper overlay on avg_pick.
    function sheetAdpSource() {
        return (state.adpSource && state.adpSource !== 'auto') ? state.adpSource : 'consensus';
    }

    // Prefer p.adp_by_source[src] (rankings ADP columns) so Consensus 12.4 here
    // is Consensus 12.4 there. Fall back to the top-level field if a source is
    // missing for this player.
    function sheetAdpOf(p, mode, sf) {
        var src = sheetAdpSource();
        var v = C.sourceAdpOf ? C.sourceAdpOf(p, src, mode, sf) : null;
        if (v != null) return v;
        if (src === 'consensus' && C.consensusAdpOf) return C.consensusAdpOf(p, mode, sf);
        return C.adpOf(p, mode, sf);
    }

    // Real market ADP for late-board ordering.
//
// IMPORTANT:
// This intentionally NEVER falls back to _radp.
//
// _radp is derived from the site's redraft-value ordering. Using it as the
// late-round fallback just reproduces the same model/VOR ordering and can
// create positional walls such as 30+ TEs appearing consecutively.
    function marketAdpOf(p, mode, sf) {
        var src = sheetAdpSource();

        // Selected source first.
        var selected = C.sourceAdpOf
            ? C.sourceAdpOf(p, src, mode, sf)
            : null;

        if (
            selected != null &&
            isFinite(Number(selected)) &&
            Number(selected) > 0
        ) {
            return Number(selected);
        }

        // Consensus second.
        var consensus = C.sourceAdpOf
            ? C.sourceAdpOf(p, 'consensus', mode, sf)
            : null;

        if (
            consensus != null &&
            isFinite(Number(consensus)) &&
            Number(consensus) > 0
        ) {
            return Number(consensus);
        }

        // If the precomputed consensus is missing, calculate a median from the
        // actual ADP sources attached to the player.
        var field = mode === 'dynasty'
            ? (sf ? 'sf_avg_pick' : 'avg_pick')
            : (sf ? 'sf_redraft_avg_pick' : 'redraft_avg_pick');

        var vals = [];
        var by = p && p.adp_by_source;

        if (by) {
            Object.keys(by).forEach(function (key) {
                if (key === 'consensus') return;

                var row = by[key];
                var value = row && row[field];

                if (
                    value != null &&
                    isFinite(Number(value)) &&
                    Number(value) > 0
                ) {
                    vals.push(Number(value));
                }
            });
        }

        if (vals.length) {
            vals.sort(function (a, b) {
                return a - b;
            });

            var mid = Math.floor(vals.length / 2);

            return vals.length % 2
                ? vals[mid]
                : (vals[mid - 1] + vals[mid]) / 2;
        }

        return null;
    }

    function fmtAdp(v) {
        return (v != null && isFinite(Number(v))) ? Number(v).toFixed(1) : '';
    }

    // Display-only column sort on the Big Board. compute() still ranks by VOR
    // (tiers, custom-board overrides, By Position). This reorders the table view
    // without rewriting that model board. Default is VOR descending.
    var SORT_DEFAULT_DIR = {
        rk: 1, name: 1, pos: 1, vor: -1, projectedPpg: -1,
        adp: 1, age: 1, value: -1, window: 1, scheduleRank: 1, market: -1, hist: -1
    };
    var POS_SORT = {QB: 0, RB: 1, WR: 2, TE: 3};
    var SORT_LABEL = {
        rk: 'Rk', name: 'Player', pos: 'Pos', vor: 'VOR', projectedPpg: 'Proj PPG',
        adp: 'ADP', age: 'Age', value: 'Value', window: 'Window',
        scheduleRank: 'Sched Rk', market: 'Market vs ADP', hist: 'Hist'
    };

    function isDefaultSort() {
        // VOR desc and Rk asc are the same model-board order (Rk is assigned
        // after the VOR rank + custom overrides). Treat both as the default view
        // so tier cliffs and proj-pick lines stay put.
        return (state.sortKey === 'vor' && state.sortDir === -1)
            || (state.sortKey === 'rk' && state.sortDir === 1);
    }

    function sortVal(x, key) {
        if (key === 'rk') return x.rk;
        if (key === 'name') return (x.name || '').toLowerCase();
        if (key === 'pos') return (POS_SORT[x.pos] != null ? POS_SORT[x.pos] : 9) * 1000 + (x.posRankN || 0);
        if (key === 'vor') {
            return x.vorRaw != null
                ? Number(x.vorRaw)
                : -9999;
        }
        if (key === 'projectedPpg') return x.projectedPpg;
        if (key === 'adp') return x.adp;
        if (key === 'age' || key === 'window') return x.age;
        if (key === 'value') return x.value;
        if (key === 'scheduleRank') return x.scheduleRank;
        if (key === 'market') return x.marketVsAdp;
        if (key === 'hist') return x.histP;
        return null;
    }

    function cmpSort(a, b, dir) {
        var aN = a == null || a === '' || (typeof a === 'number' && !isFinite(a));
        var bN = b == null || b === '' || (typeof b === 'number' && !isFinite(b));
        if (aN && bN) return 0;
        if (aN) return 1;
        if (bN) return -1;
        if (a < b) return -dir;
        if (a > b) return dir;
        return 0;
    }

    function displayPlayers() {
        if (isDefaultSort()) return players;
        var key = state.sortKey, dir = state.sortDir;
        return players.slice().sort(function (a, b) {
            var c = cmpSort(sortVal(a, key), sortVal(b, key), dir);
            return c || ((a.rk || 0) - (b.rk || 0));
        });
    }

    function resetBoardSort() {
        state.sortKey = 'vor';
        state.sortDir = -1;
    }

    function setSort(key) {
        if (!key || !SORT_DEFAULT_DIR[key]) return;
        if (state.sortKey === key) state.sortDir = -state.sortDir;
        else {
            state.sortKey = key;
            state.sortDir = SORT_DEFAULT_DIR[key];
        }
        if (!isDefaultSort() && editBoard) editBoard = false;
    }

    function sortTh(key, label, extraClass, title) {
        var active = state.sortKey === key;
        var aria = active ? (state.sortDir === 1 ? 'ascending' : 'descending') : 'none';
        var cls = 'cs-sort' + (extraClass ? ' ' + extraClass : '')
            + (active ? (state.sortDir === 1 ? ' cs-sort-asc' : ' cs-sort-desc') : '');
        var tip = title || ('Sort by ' + label);
        return '<th class="' + cls + '" data-sort="' + key + '" aria-sort="' + aria + '" title="' + tip + '">'
            + '<button type="button" class="cs-sortbtn">' + label + '</button></th>';
    }

    function compute() {
        // A sheet opened from the Draft Room must mirror that room exactly. Custom
        // pre-draft overrides still apply to the standalone value board, but not on
        // top of a live Recommendation snapshot.
        if (!recommendationOrder) ensureOverrides();
        var mode = state.mode, sf = state.sf;
        // Value-derived redraft ADP fallback (mirrors the draft room).
        allPlayers.slice().sort(function (a, b) {
            return C.redraftVal(b, sf) - C.redraftVal(a, sf);
        })
            .forEach(function (p, i) {
                p._radp = i + 1;
            });

        var pool = allPlayers.filter(function (p) {
            return ['QB', 'RB', 'WR', 'TE'].indexOf((p.position || '').toUpperCase()) >= 0 && C.valOf(p, mode, sf) > 0;
        });
        if (!pool.length) {
            players = [];
            return;
        }

        var valFn = function (p) {
            return C.valOf(p, mode, sf);
        };
        // Empirical starter allocation (best-available fills each starting slot),
        // matching the Draft Room and the server grade, rather than the fixed
        // half-QB/half-RB/half-WR heuristic. Falls back to startersFor if the shared
        // core is an older build without the allocator.
        var starters = C.effectiveStarters
            ? C.effectiveStarters(pool, C.rosterCounts(cfg.rosterPositions, sf), teams, valFn)
            : C.startersFor(cfg.rosterPositions, sf);
        var repl = C.computeReplacement(pool, valFn, starters, teams);
        // Roster-need shading: targets from the league roster, "my" counts from live
        // draft picks that are mine. Only meaningful once a live draft is connected.
        var targets = C.posTargets(C.rosterCounts(cfg.rosterPositions, sf), scoringCfg().tep);
        var needByPos = {};
        ['QB', 'RB', 'WR', 'TE'].forEach(function (pos) {
            var have = (myCounts && myCounts[pos]) || 0;
            needByPos[pos] = {target: targets[pos] || 0, have: have, need: Math.max(0, (targets[pos] || 0) - have)};
        });
        window.__csNeed = needByPos;

        // VOR remains the stable cheat-sheet order. A Draft Room Recommendation
        // snapshot is supplemental context only; it must never re-sort the board.
        var scored = pool.map(function (p) {
            var pos = (p.position || '').toUpperCase();
            var value = C.valOf(p, mode, sf);

            var replacement =
                repl[pos] != null &&
                isFinite(Number(repl[pos]))
                    ? Number(repl[pos])
                    : 0;

            // Keep full precision for ranking.
            // Do NOT round VOR before sorting.
            var vorRaw = Number(value) - replacement;

            return {
                id: String(p.id),
                pos: pos,
                name: p.name || String(p.id),

                age:
                    p.age != null
                        ? Number(p.age)
                        : null,

                // Existing display ADP.
                adp: sheetAdpOf(p, mode, sf),

                // Real market ADP for late-board ordering.
                // This helper should NOT fall back to _radp.
                marketAdp: marketAdpOf(p, mode, sf),

                // Full-precision VOR.
                vorRaw: vorRaw,

                // Keep vor for compatibility with the rest of the file.
                vor: vorRaw,

                projectedPpg: scoringProjPpg(p),

                marketVsAdp:
                    mode === 'redraft' &&
                    p.market_vs_adp != null
                        ? Number(p.market_vs_adp)
                        : null,

                marketExpectedAdp:
                    mode === 'redraft' &&
                    p.market_expected_adp != null
                        ? Number(p.market_expected_adp)
                        : null,

                marketConfidence:
                    mode === 'redraft' &&
                    p.market_confidence != null
                        ? Number(p.market_confidence)
                        : null,

                marketConfidenceLabel:
                    mode === 'redraft'
                        ? (p.market_confidence_label || null)
                        : null,

                marketBasis:
                    mode === 'redraft'
                        ? (p.market_basis || null)
                        : null,

                scheduleRank:
                    scheduleRanks[String(p.id)] || null,

                historical: p.historical || null,
                histP:
                    p.historical && p.historical.p_hit_pct != null
                        ? Number(p.historical.p_hit_pct)
                        : null,
                histEdge:
                    p.historical && p.historical.h_vs_m_pts != null
                        ? Number(p.historical.h_vs_m_pts)
                        : null,
            };
        });

        scored.sort(function (a, b) {
            var aVor = Number(a.vorRaw);
            var bVor = Number(b.vorRaw);

            var aAboveReplacement = aVor > 0;
            var bAboveReplacement = bVor > 0;

            // Above replacement: rank primarily by VOR.
            if (aAboveReplacement && bAboveReplacement) {
                var vorDifference = bVor - aVor;

                if (Math.abs(vorDifference) > 1e-9) {
                    return vorDifference;
                }

                // Real tie -> use actual market ADP.
                var aMarket =
                    a.marketAdp != null
                        ? Number(a.marketAdp)
                        : 9999;

                var bMarket =
                    b.marketAdp != null
                        ? Number(b.marketAdp)
                        : 9999;

                return aMarket - bMarket;
            }

            // Above-replacement players always stay ahead of below-replacement players.
            if (aAboveReplacement !== bAboveReplacement) {
                return aAboveReplacement ? -1 : 1;
            }

            // Both are below replacement:
            // stop comparing cross-position negative VOR and use actual market order.
            var aHasMarket =
                a.marketAdp != null &&
                isFinite(Number(a.marketAdp));

            var bHasMarket =
                b.marketAdp != null &&
                isFinite(Number(b.marketAdp));

            if (aHasMarket && !bHasMarket) {
                return -1;
            }

            if (!aHasMarket && bHasMarket) {
                return 1;
            }

            if (aHasMarket && bHasMarket) {
                var marketDifference =
                    Number(a.marketAdp) -
                    Number(b.marketAdp);

                if (Math.abs(marketDifference) > 1e-9) {
                    return marketDifference;
                }
            }

            // No real market data: fall back to raw VOR.
            var tailVorDifference = bVor - aVor;

            if (Math.abs(tailVorDifference) > 1e-9) {
                return tailVorDifference;
            }

            // Next fallback: projected PPG.
            var aPpg =
                a.projectedPpg != null
                    ? Number(a.projectedPpg)
                    : -9999;

            var bPpg =
                b.projectedPpg != null
                    ? Number(b.projectedPpg)
                    : -9999;

            if (Math.abs(bPpg - aPpg) > 1e-9) {
                return bPpg - aPpg;
            }

            return String(a.name || '')
                .localeCompare(String(b.name || ''));
        });

        scored.forEach(function (x, i) {
            x._mr = i;
        });

        players = scored.slice(0, LIMIT);

        maxVor = players.length
            ? Math.max.apply(
                null,
                players.map(function (x) {
                    return Math.max(
                        1,
                        x.vorRaw != null
                            ? Number(x.vorRaw)
                            : 0
                    );
                })
            )
            : 1;

        var pc = {};
        var availableRank = 0;

        players.forEach(function (x, i) {
            x.drafted = draftedIds
                ? draftedIds.has(x.id)
                : false;

            x.rk = i + 1;

            x.recRank =
                recommendationOrder &&
                recommendationOrder[x.id] != null
                    ? recommendationOrder[x.id] + 1
                    : null;

            x.value =
                x.adp != null &&
                x.rk != null
                    ? Math.round(x.adp - x.rk)
                    : null;

            x.good =
                state.mode === 'dynasty'
                    ? (youthWindow(x.age, x.pos)[1] === 'win-asc' ? 1 : 0)
                    : (x.value != null && x.value >= 5 ? 1 : 0);

            x.posfull =
                myCounts
                    ? (
                        (needByPos[x.pos] && needByPos[x.pos].need) <= 0 &&
                        (needByPos[x.pos] && needByPos[x.pos].have) > 0
                    )
                    : false;

            pc[x.pos] = (pc[x.pos] || 0) + 1;

            x.posRankN = pc[x.pos];
            x.prk = x.pos + x.posRankN;
        });

        assignTiers();
        applyOverrides();
    }

    // Same drop-based tiering the rankings page uses (utils/tier_thresholds.py):
    // boundaries fall on natural value cliffs scored by *local* significance (a gap
    // vs the median of nearby gaps), with two hard rules - no tier spans more than
    // MAX_SPAN, and none is smaller than MIN_SIZE (the elite T1 may be as small as
    // ELITE_MIN). Ported here to run on the VOR the board is sorted by, so redraft
    // (which has no server value-tier table) is covered too and tiers stay
    // contiguous and monotonic with the displayed order.
    function assignTiers() {
        var n = players.length;
        if (!n) return;
        var vals = players.map(function (p, i) {
            if (
                p.vorRaw != null &&
                Number(p.vorRaw) > 0
            ) {
                return 10000 + Number(p.vorRaw);
            }
            return -i;
        });

        var NUM_TIERS = 12, MIN_SIZE = 5, ELITE_MIN = 3, MAX_SPAN = 220, WINDOW = 10, SIG_MIN = 2.0;

        // Too few players to derive meaningful drops: fall back to fixed VOR bands.
        if (n < NUM_TIERS * 3) {
            var mx = maxVor || 1;
            players.forEach(function (x) {
                var r = x.vor / mx;
                x.dtier = r >= 0.72 ? 1 : r >= 0.50 ? 2 : r >= 0.33 ? 3 : r >= 0.16 ? 4 : 5;
            });
            var remap = {}, nx = 0;
            players.forEach(function (x) {
                if (!(x.dtier in remap)) {
                    nx++;
                    remap[x.dtier] = nx;
                }
                x.dtier = remap[x.dtier];
            });
            return;
        }

        // Local significance of each gap: gap size vs the median of nearby gaps.
        var score = [];
        for (var i = 0; i < n - 1; i++) {
            var gap = vals[i] - vals[i + 1];
            var lo = Math.max(0, i - WINDOW), hi = Math.min(n - 1, i + WINDOW);
            var nbrs = [];
            for (var j = lo; j < hi; j++) {
                if (j !== i) nbrs.push(vals[j] - vals[j + 1]);
            }
            nbrs.sort(function (a, b) {
                return a - b;
            });
            var med = nbrs.length ? nbrs[Math.floor(nbrs.length / 2)] : 1.0;
            score[i] = gap / Math.max(med, 0.5);
        }

        var bounds = [];   // boundary index i = split between player i and i+1
        function segment(i) {
            var lower = -1, upper = n - 1;
            for (var k = 0; k < bounds.length; k++) {
                var b = bounds[k];
                if (b < i && b > lower) lower = b;
                if (b > i && b < upper) upper = b;
            }
            return [lower, upper];
        }

        function valid(i) {
            var s = segment(i);
            var top = i - s[0], bot = s[1] - i;
            var tmin = (s[0] === -1) ? ELITE_MIN : MIN_SIZE;
            return top >= tmin && bot >= MIN_SIZE;
        }

        while (bounds.length < NUM_TIERS - 1) {
            // 1) Mandatory: split the worst over-span segment at its biggest gap.
            var prev = -1, worst = null, worstSpan = MAX_SPAN;
            var seq = bounds.slice().sort(function (a, b) {
                return a - b;
            });
            seq.push(n - 1);
            for (var s2 = 0; s2 < seq.length; s2++) {
                var bb = seq[s2], loS = prev + 1, hiS = bb;
                prev = bb;
                var sp = vals[loS] - vals[hiS];
                if (sp > worstSpan) {
                    worstSpan = sp;
                    worst = [loS, hiS];
                }
            }
            var did = false;
            if (worst) {
                var loW = worst[0], hiW = worst[1], bestI = null, bestG = -1;
                for (var jj = loW + MIN_SIZE - 1; jj < hiW - MIN_SIZE + 1; jj++) {
                    var g = vals[jj] - vals[jj + 1];
                    if (g > bestG) {
                        bestG = g;
                        bestI = jj;
                    }
                }
                if (bestI !== null && valid(bestI)) {
                    bounds.push(bestI);
                    did = true;
                }
            }
            if (did) continue;

            // 2) Discretionary: the most locally-significant remaining valid drop.
            var cand = [];
            for (var ii = 0; ii < n - 1; ii++) {
                if (bounds.indexOf(ii) < 0 && score[ii] >= SIG_MIN && valid(ii)) cand.push([score[ii], ii]);
            }
            if (!cand.length) break;
            cand.sort(function (a, b) {
                return b[0] - a[0];
            });
            bounds.push(cand[0][1]);
        }

        // Assign contiguous tiers from the sorted boundary indices.
        bounds.sort(function (a, b) {
            return a - b;
        });
        var tier = 1, bp = 0;
        for (var t = 0; t < n; t++) {
            players[t].dtier = tier;
            if (bp < bounds.length && t === bounds[bp]) {
                tier++;
                bp++;
            }
        }
    }

    // ── render ──────────────────────────────────────────────────────────────────
    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
        });
    }

    function posrk(x) {
        return '<span class="cs-posrk cs-pos-' + x.pos + '">' + x.prk + '</span>';
    }

    function valChip(v) {
        if (v == null) return '';
        return v > 0 ? '<span class="cs-val g">+' + v + '</span>' : (v < 0 ? '<span class="cs-val b">' + v + '</span>' : '<span class="cs-val n">even</span>');
    }

    function histPctClass(h) {
        if (!h) return 'n';
        // Green only for a strong absolute cell or history beating the ADP
        // bucket. Never paint market_higher red — early ADP is a high bar.
        if (h.h_vs_m === 'history_higher') return 'g';
        var pct = h.p_hit_pct;
        if (pct != null && isFinite(Number(pct)) && Number(pct) >= HIST_STRONG_PCT) return 'g';
        return 'n';
    }

    function histCell(x, dyn) {
        if (!showHist(dyn)) return '';
        var h = x.historical || {};
        var pct = h.p_hit_pct;
        var tipBits = [];
        if (pct != null) tipBits.push('Players like this: ' + pct + '% top-12');
        if (h.mkt_pct != null) tipBits.push('that ADP round: ' + h.mkt_pct + '%');
        if (h.h_vs_m_pts != null && isFinite(Number(h.h_vs_m_pts)) && h.h_vs_m && h.h_vs_m !== 'unknown') {
            var pts = Number(h.h_vs_m_pts);
            if (h.h_vs_m === 'history_higher') {
                tipBits.push('history ahead of that ADP round by ' + pts + ' pts');
            } else if (h.h_vs_m === 'aligned') {
                tipBits.push('in line with that ADP round');
            } else if (h.h_vs_m === 'market_higher') {
                tipBits.push('early ADP is a high bar ('
                    + (pts > 0 ? '+' : '') + pts + ' vs that round)');
            }
        }
        if (!tipBits.length) tipBits.push('Historical top-12 chance unknown');
        tipBits.push('Open for the full mix.');
        var cls = histPctClass(h);
        var body = pct == null ? '-' : (pct + '%');
        return '<td class="cs-hist-col"><span class="cs-hist-cell">'
            + '<span class="cs-val ' + cls + '" title="' + esc(tipBits.join('. ')) + '">' + body + '</span>'
            + '<button type="button" class="cs-hist-btn" data-hist-id="' + esc(x.id) + '" data-hist-name="' + esc(x.name) + '" data-hist-adp="' + (x.adp != null && isFinite(Number(x.adp)) ? String(x.adp) : '') + '" data-hist-pos="' + esc(x.pos || '') + '" data-hist-proj="' + (x.projectedPpg != null && isFinite(Number(x.projectedPpg)) ? String(x.projectedPpg) : '') + '" data-hist-proj-rk="' + (h.proj_rk != null ? String(h.proj_rk) : '') + '" data-hist-adp-rk="' + (h.adp_rk != null ? String(h.adp_rk) : '') + '" title="This player\'s historical chance">i</button>'
            + '</span></td>';
    }

    function smallVal(v) {
        if (v == null) return '';
        return v > 0 ? '<span class="cs-pgv cs-val g">+' + v + '</span>' : (v < 0 ? '<span class="cs-pgv cs-val b">' + v + '</span>' : '');
    }

    function winChip(age, pos) {
        var w = youthWindow(age, pos);
        return w[0] ? '<span class="cs-winpill ' + w[1] + '">' + w[0] + '</span>' : '';
    }

    function $(id) {
        return document.getElementById(id);
    }

    // Search + position filters narrow which players are shown (they don't re-rank).
    function visiblePlayer(x) {
        if (state.posFilter !== 'ALL' && x.pos !== state.posFilter) return false;
        if (state.search && x.name.toLowerCase().indexOf(state.search) < 0) return false;
        return true;
    }

    // Snake overall pick numbers for a 1-based seat. Odd rounds run 1→N, even
    // rounds N→1 — the same convention the Draft Room board uses.
    function snakePickNum(round, slot, nTeams) {
        var inRound = (round % 2 === 1) ? slot : (nTeams - slot + 1);
        return (round - 1) * nTeams + inRound;
    }

    function roundPickLabel(pn, nTeams) {
        var rd = Math.ceil(pn / nTeams);
        var pk = pn - (rd - 1) * nTeams;
        return rd + '.' + (pk < 10 ? '0' + pk : String(pk));
    }

    function projPicks() {
        var slot = state.pickSlot;
        if (!slot || slot < 1 || slot > teams || !players.length) return [];
        var out = [], maxPn = players.length;
        for (var r = 1; ; r++) {
            var pn = snakePickNum(r, slot, teams);
            if (pn > maxPn) break;
            out.push({pn: pn, label: roundPickLabel(pn, teams)});
        }
        return out;
    }

    function projPickMap() {
        var m = {};
        projPicks().forEach(function (pk) {
            m[pk.pn] = pk;
        });
        return m;
    }

    function projLineRow(pk, span, taken) {
        return '<tr class="cs-cliff cs-proj' + (taken ? ' cs-proj-taken' : '') + '"><td colspan="' + span + '"><div class="cs-projline">Proj Pick ' + pk.label + ' <span class="cs-proj-ov">#' + pk.pn + '</span></div></td></tr>';
    }

    function slotStoreKey() {
        return 'cspickslot:' + (cfg.leagueId || 'guest');
    }

    function savePickSlot() {
        try {
            if (!state.pickSlot) localStorage.removeItem(slotStoreKey());
            else localStorage.setItem(slotStoreKey(), String(state.pickSlot));
        } catch (e) { /* storage full/blocked */
        }
    }

    function renderPickSlot() {
        var sel = $('csPickSlot');
        if (!sel) return;
        if (sel.options.length !== teams + 1) {
            var html = '<option value="0">Proj pick: Off</option>';
            for (var s = 1; s <= teams; s++) {
                var pad = s < 10 ? '0' + s : String(s);
                html += '<option value="' + s + '">Slot ' + s + ' · 1.' + pad + '</option>';
            }
            sel.innerHTML = html;
        }
        var cur = (state.pickSlot >= 1 && state.pickSlot <= teams) ? state.pickSlot : 0;
        if (state.pickSlot !== cur) state.pickSlot = cur;
        sel.value = String(cur);
        var valEl = sel.parentNode && sel.parentNode.querySelector('.csd-value');
        if (valEl && sel.selectedIndex >= 0) valEl.textContent = sel.options[sel.selectedIndex].textContent.trim();
    }

    function render() {
        var dyn = state.mode === 'dynasty';
        syncHistSurfaces();
        $('csTitle').textContent = dyn ? 'Dynasty Cheat Sheet' : 'Redraft Cheat Sheet';
        $('csSub').textContent = recommendationOrder
            ? 'The stable VOR board, with live Draft Room Recommendation ranks shown as supplemental context.'
            : dyn
                ? 'Ranked by value over replacement on dynasty value, for your league roster. Tiers are cliffs in the value curve. Age and career window replace ADP.'
                : 'Ranked by value over replacement for your league scoring and roster. Tiers are cliffs in the value curve. The Value column flags where the market disagrees.';

        document.querySelectorAll('.cs-board').forEach(function (b) {
            b.classList.toggle('filteron', state.filter);
            b.classList.toggle('hidedrafted', state.hideDrafted);
            b.classList.toggle('needson', state.needsFilter);
        });
        $('csValBtn').textContent = dyn ? 'Ascenders only' : 'Values only';
        var hd = $('csHideDrafted');
        if (hd) {
            hd.style.display = draftedIds ? '' : 'none';
            hd.setAttribute('aria-pressed', String(state.hideDrafted));
        }
        var nb = $('csNeedsBtn');
        if (nb) {
            nb.style.display = myCounts ? '' : 'none';
            nb.setAttribute('aria-pressed', String(state.needsFilter));
        }
        // Show a Clear button once the user has hand-marked players as gone, so they
        // can wipe those marks in one tap. Live/mock drafted ids are not touched.
        var cb = $('csClearBtn');
        if (cb) cb.style.display = state.done.size ? '' : 'none';
        var liveBtn = $('csConnectLive');
        if (liveBtn) {
            liveBtn.style.display = (cfg.leagueId && cfg.platform) ? '' : 'none';
            liveBtn.textContent = liveDraftId ? 'Disconnect live draft' : 'Connect live draft';
        }
        // Custom board (pro): edit toggle always available; reset only with overrides.
        var eb = $('csEditBtn');
        if (eb) {
            eb.style.display = (cfg.hasPremium && !recommendationOrder) ? '' : 'none';
            eb.setAttribute('aria-pressed', String(editBoard));
            eb.textContent = editBoard ? 'Done editing' : 'Edit board';
        }
        var rb = $('csResetBoardBtn');
        if (rb) rb.style.display = (hasOverrides() || state.done.size || draftedIds) ? '' : 'none';
        var bp = $('cs-panel-board');
        if (bp) bp.classList.toggle('editing', editBoard && cfg.hasPremium);
        renderNeedsBar();
        renderPickSlot();

        if (!players.length) {
            var emptyMsg = loading ? 'Loading players…' : (loadError || 'No players for this format yet.');
            if (!loading && window.brEmptyState) {
                var host = $('csBoardBody');
                host.innerHTML = '<tr><td colspan="6"><div id="csBoardEmpty"></div></td></tr>';
                window.brEmptyState('csBoardEmpty', {
                    icon: loadError ? 'error' : 'empty',
                    title: loadError ? 'Couldn’t load' : 'No players yet',
                    message: emptyMsg,
                    compact: true,
                    error: !!loadError
                });
            } else {
                $('csBoardBody').innerHTML = '<tr><td colspan="6" class="cs-empty">' + emptyMsg + '</td></tr>';
            }
            $('csLegend').innerHTML = '';
            return;
        }

        var draftedNote = draftedIds ? '<span class="cs-lg"><span class="cs-taken-dot"></span> already drafted</span>' : '';
        var projNote = state.pickSlot
            ? '<span class="cs-lg"><b>Proj Pick</b> slot ' + state.pickSlot + ' snake windows on this board</span>'
            : '';
        var sortNote = !isDefaultSort()
            ? '<span class="cs-lg"><b>Sorted by ' + (SORT_LABEL[state.sortKey] || state.sortKey) + '</b> '
            + (state.sortDir === 1 ? 'low → high' : 'high → low') + ' · click Rk or VOR to restore the board</span>'
            : '';
        var fmtNote = '<span class="cs-lg" id="csFmtNote">'
            + (dyn ? 'Dynasty ' : '')
            + (state.sf ? 'Superflex' : '1QB')
            + ' &middot; ' + scoringLabel()
            + ' &middot; ' + teams + '-team</span>';
        $('csLegend').innerHTML = recommendationOrder
            ? '<span class="cs-lg"><b>VOR</b> controls the cheat-sheet order</span>'
            + '<span class="cs-lg"><b>REC #</b> current Draft Room rank, shown for context</span>'
            + sortNote
            + projNote
            + draftedNote
            + fmtNote
            : dyn
                ? '<span class="cs-lg"><b>VOR</b> dynasty value over replacement, the ranking</span>'
                + '<span class="cs-lg"><b>Age</b> drives the window</span>'
                + '<span class="cs-lg">' + winChip(23) + ' ascending</span>'
                + sortNote
                + projNote
                + draftedNote
                + fmtNote
                : '<span class="cs-lg"><b>VOR</b> value over replacement, the ranking</span>'
                + '<span class="cs-lg"><span class="cs-val g">+7</span> above ADP, target it</span>'
                + '<span class="cs-lg"><span class="cs-val b">-4</span> going early, let it fall</span>'
                + '<span class="cs-lg"><b>Sched Rk</b> full-season schedule (1 = easiest)</span>'
                + (showHist(dyn) ? '<span class="cs-lg"><b>Hist</b> top-12 chance for this profile</span>' : '')
                + sortNote
                + projNote
                + draftedNote
                + fmtNote;

        renderBoard(dyn);
        renderPos(dyn);
        // An in-draft sheet may open many rounds into the board. Put the first row
        // that is still available at the top of the table instead of making the user
        // manually scroll past crossed-off players. This is intentionally one-shot
        // so filters, live polling and later renders never steal the user's scroll.
        if (scrollToFirstAvailable) {
            scrollToFirstAvailable = false;
            requestAnimationFrame(function () {
                var row = document.querySelector('#csBoardBody tr.cs-p:not(.drafted):not(.done)');
                var scroller = row && row.closest('.cs-tbl-scroll');
                if (row && scroller) scroller.scrollTop = Math.max(0, row.offsetTop - 4);
            });
        }
        // The move-flash is a one-shot: clear the id so later renders don't replay it.
        if (_flashId) setTimeout(function () {
            _flashId = null;
        }, 650);
    }

    // Per-row custom-board controls (pro): drag handle, move up/down, pin, mute, and
    // (when the row is overridden) revert. Shown only in edit mode. Each is a
    // .cs-ovbtn with data-act/data-id handled by a delegated capture listener so it
    // never also toggles the row's crossed-off state; the drag handle is driven by
    // pointer events instead.
    function ovControls(x) {
        if (!cfg.hasPremium || recommendationOrder) return '';

        function b(act, glyph, on, title, extra) {
            return '<button type="button" class="cs-ovbtn' + (on ? ' on' : '') + (extra || '') + '" data-act="' + act + '" data-id="' + esc(x.id) + '" title="' + title + '" aria-label="' + title + '">' + glyph + '</button>';
        }

        // Drag + arrows reorder within the ranked list, so they're hidden on pinned/
        // muted rows (terminal buckets) — those are managed by the pin/mute toggles
        // and revert. Keeps the arrows from looking clickable when they'd no-op.
        var canMove = x.ov !== 'pin' && x.ov !== 'mute';
        var move = canMove
            ? b('drag', '&#8942;', false, 'Drag to reorder', ' cs-drag')
            + b('up', '&#9650;', false, 'Move up a row')
            + b('down', '&#9660;', false, 'Move down a row')
            : '';
        return '<td class="cs-edit-cell"><span class="cs-ovbtns">'
            + move
            + b('pin', '&#9733;', x.ov === 'pin', 'Pin to the top')
            + b('mute', '&times;', x.ov === 'mute', 'Mute to the bottom')
            + (x.ov ? b('revert', '&#8630;', false, 'Reset to model spot', ' cs-revert') : '')
            + '</span></td>';
    }

    // A small chip that shows how a row differs from the model board.
    function ovChip(x) {
        if (x.ov === 'pin') return '<span class="cs-ovchip pin">pinned</span>';
        if (x.ov === 'mute') return '<span class="cs-ovchip mute">muted</span>';
        if (x.ov === 'up' || x.ov === 'down') return '<span class="cs-ovchip bump">' + (x.ov === 'up' ? '&#9650;' : '&#9660;') + x.ovN + '</span>';
        return '';
    }

    function renderBoard(dyn) {
        var col5 = dyn ? 'Age' : 'ADP', col6 = dyn ? 'Window' : 'Value';
        var col5Key = dyn ? 'age' : 'adp', col6Key = dyn ? 'window' : 'value';
        var editable = cfg.hasPremium;
        var editTh = editable ? '<th class="cs-edit-th"></th>' : '';
        var boardSort = isDefaultSort();
        $('csBoardHead').innerHTML = '<tr>'
            + sortTh('rk', 'Rk', 'cs-rk', 'VOR board rank. Click to restore the default order.')
            + sortTh('name', 'Player', 'l cs-player', 'Sort by player name')
            + sortTh('pos', 'Pos', '', 'Sort by position, then positional rank')
            + sortTh('vor', 'VOR', 'cs-vor-col', 'VALUE: Value over replacement — the model ranking')
            + sortTh('projectedPpg', 'Proj PPG', '', 'PROJECTION: Projected fantasy points per game')
            + sortTh(col5Key, col5, '', dyn ? 'Sort by age' : 'MARKET: Sort by ADP')
            + sortTh(col6Key, col6, 'cs-value-col', dyn ? 'Sort by career window (age)' : 'VALUE: Sort by value vs ADP')
            + sortTh('scheduleRank', 'Sched Rk', '', 'PROJECTION: Full fantasy-season strength of schedule rank (1 = easiest)')
            + (showHist(dyn) ? sortTh('hist', 'Hist', 'cs-hist-col', 'HISTORY: Historical top-12 chance for this career and situation. Green when the cell is strong or history beats that ADP round. Early ADP is a high bar, not a miss.') : '')
            + (showMarket(dyn) ? sortTh('market', 'Market vs ADP', 'cs-market-col', 'MARKET: Where market signals imply this player should be drafted vs ADP') : '')
            + editTh + '</tr>';
        var span = (editable ? 9 : 8) + (showMarket(dyn) ? 1 : 0) + (showHist(dyn) ? 1 : 0);
        var lastT = null, html = '', shown = 0;
        var pickAt = boardSort ? projPickMap() : {};
        displayPlayers().forEach(function (x) {
            if (!visiblePlayer(x)) return;
            if (boardSort && !recommendationOrder && x.grp !== lastT) {
                lastT = x.grp;
                html += '<tr class="cs-cliff"><td colspan="' + span + '"><div class="cs-cliffline">' + x.grpLabel + '</div></td></tr>';
            }
            var pk = pickAt[x.rk];
            if (pk) html += projLineRow(pk, span, x.drafted);
            shown++;
            var cls = 'cs-p cs-c-' + x.pos + (state.done.has(x.id) ? ' done' : '') + (x.drafted ? ' drafted' : '') + (x.ov === 'mute' ? ' cs-muted' : '') + (x.ov ? ' cs-ov' : '') + (x.id === _flashId ? ' cs-flash' : '') + (pk ? ' cs-proj-row' : '');
            // Proj pick is marked by the divider line above + cs-proj-row highlight.
            // Do not also inject an inline "Proj …" chip into the sticky name cell —
            // on mobile it crowds the name and paints over it when the board scrolls.
            var projTitle = pk ? ' title="Projected pick ' + pk.label + ' (overall #' + pk.pn + ')"' : '';
            var c5 = dyn ? '<td class="cs-num">' + (x.age != null ? x.age : '') + '</td>' : '<td class="cs-num">' + fmtAdp(x.adp) + '</td>';
            var c6 = dyn ? '<td class="cs-value-col">' + winChip(x.age, x.pos) + '</td>' : '<td class="cs-value-col">' + valChip(x.value) + '</td>';
            var market = '';
            if (showMarket(dyn)) {
                if (x.marketVsAdp == null) market = '<td class="cs-num cs-market-col" title="Not enough independent market data yet.">&ndash;</td>';
                else {
                    var mcls = x.marketVsAdp > 0 ? 'g' : (x.marketVsAdp < 0 ? 'b' : 'n');
                    var basisLabel = x.marketBasis === 'season_props' ? 'season-long player markets' : x.marketBasis === 'rolling_market' ? 'multiple recent weekly player markets' : x.marketBasis === 'team_environment' ? 'team betting environment' : 'a blend of available market signals';
                    var confLabel = x.marketConfidenceLabel || (x.marketConfidence >= .7 ? 'High' : (x.marketConfidence >= .5 ? 'Moderate' : 'Low'));
                    var direction = x.marketVsAdp > 0 ? 'earlier' : (x.marketVsAdp < 0 ? 'later' : 'near its current ADP');
                    var mtip = 'Market context implies this player should be drafted ' + (x.marketVsAdp === 0 ? direction : 'about ' + Math.abs(Math.round(x.marketVsAdp)) + ' picks ' + direction) + '. Expected Pick ' + Math.round(x.marketExpectedAdp) + '; current ADP ' + (x.adp != null ? Number(x.adp).toFixed(1) : '—') + '. Confidence: ' + confLabel + ' (' + Math.round((x.marketConfidence || 0) * 100) + '%). Based primarily on ' + basisLabel + '.';
                    market = '<td class="cs-market-col"><span class="cs-val ' + mcls + '" title="' + esc(mtip) + '">' + (x.marketVsAdp > 0 ? '+' : '') + Math.round(x.marketVsAdp) + '</span></td>';
                }
            }
            var hist = histCell(x, dyn);
            var recChip = x.recRank != null ? '<span class="cs-ovchip bump">REC #' + x.recRank + '</span>' : '';
            var vorDisplay =
                x.vorRaw != null &&
                isFinite(Number(x.vorRaw))
                    ? Number(x.vorRaw).toFixed(1)
                    : '&ndash;';

            var vorBarPct =
                x.vorRaw != null &&
                isFinite(Number(x.vorRaw))
                    ? Math.max(
                        0,
                        Math.round(
                            Math.max(0, Number(x.vorRaw)) /
                            maxVor *
                            100
                        )
                    )
                    : 0;
            html +=
                '<tr class="' + cls + '"' +
                projTitle +
                ' data-good="' + x.good + '"' +
                ' data-posfull="' + (x.posfull ? 1 : 0) + '"' +
                ' data-name="' + esc(x.name) + '"' +
                ' data-id="' + esc(x.id) + '">' +
                '<td class="cs-rk">' +
                (x.rk == null ? '&ndash;' : x.rk) +
                '</td>' +
                '<td class="cs-player">' +
                '<span class="cs-pcell">' +
                '<span class="cs-pname">' +
                esc(x.name) +
                '</span>' +
                recChip +
                ovChip(x) +
                '</span>' +
                '</td>' +
                '<td>' +
                posrk(x) +
                '</td>' +
                '<td class="cs-vor-col">' +
                '<span class="cs-vorwrap">' +
                '<span class="cs-num">' +
                vorDisplay +
                '</span>' +
                '<span class="cs-vorbar">' +
                '<i style="width:' + vorBarPct + '%"></i>' +
                '</span>' +
                '</span>' +
                '</td>' +
                '<td class="cs-num">' +
                (
                    x.projectedPpg != null
                        ? Number(x.projectedPpg).toFixed(1)
                        : '&ndash;'
                ) +
                '</td>' +
                c5 +
                c6 +
                '<td class="cs-num"' +
                ' title="Full fantasy-season strength of schedule; 1 is easiest">' +
                (
                    x.scheduleRank
                        ? '#' + x.scheduleRank
                        : '&ndash;'
                ) +
                '</td>' +
                hist +
                market +
                ovControls(x) +
                '</tr>';
        });
        if (!shown) html = '<tr><td colspan="' + span + '" class="cs-empty">No players match this filter.</td></tr>';
        $('csBoardBody').innerHTML = html;
        var foot = recommendationOrder
            ? 'VOR keeps this board stable; REC # shows the live Draft Room opinion without changing the order. Reopen to refresh ranks.'
            : dyn
                ? 'Ranked by value over replacement (dynasty value), youth-aware via the Window column. Tap a row to cross a player off.'
                : 'Ranked by value over replacement, so a scarce elite TE or QB can still outrank a higher-scoring skill player. Tap a row to cross a player off.';
        if (!boardSort) foot = 'Showing the board sorted by ' + (SORT_LABEL[state.sortKey] || state.sortKey) + '. Rk is still the VOR rank. Click VOR or Rk to restore the default order.';
        if (state.pickSlot && boardSort) foot += ' Proj Pick lines mark slot ' + state.pickSlot + ' snake windows.';
        $('csBoardFoot').textContent = foot;
    }

    function renderPos(dyn) {
        var POS = ['RB', 'WR', 'QB', 'TE'];
        var BAND = Math.max(1, maxVor * 0.045), CAP = 6;
        var groups = [], cur = null;
        // By Position stays the model view: iterate in model (VOR) order even when the
        // Big Board has been custom-reordered, so its tier grouping stays contiguous.
        var list = players.slice().sort(function (a, b) {
            return (a._mr || 0) - (b._mr || 0);
        });
        var pickAt = projPickMap();
        list.forEach(function (x) {
            if (!visiblePlayer(x)) return;
            var tierChanged = !cur || x.dtier !== cur.tier;
            if (!cur || tierChanged || x.vor < cur.lead - BAND || cur.items.length >= CAP) {
                cur = {tier: x.dtier, lead: x.vor, items: [], tierBreak: tierChanged};
                groups.push(cur);
            }
            cur.items.push(x);
        });

        function nameChip(x) {
            var cls = 'cs-pgc cs-c-' + x.pos + (state.done.has(x.id) ? ' done' : '') + (x.drafted ? ' drafted' : '');
            var tail = dyn ? '' : smallVal(x.value);
            var pk = pickAt[(x._mr || 0) + 1];
            var proj = pk ? '<span class="cs-proj-mark" title="Projected pick ' + pk.label + ' (overall #' + pk.pn + ')">Proj ' + pk.label + '</span>' : '';
            return '<span class="' + cls + '" data-good="' + x.good + '" data-posfull="' + (x.posfull ? 1 : 0) + '" data-name="' + esc(x.name) + '" data-id="' + esc(x.id) + '"><span class="cs-pgn">' + esc(x.name) + tail + proj + '</span></span>';
        }

        var out = '<div class="cs-pgrid-head">' + POS.map(function (p) {
            return '<div class="cs-c-' + p + '">' + p + '</div>';
        }).join('') + '</div>';
        var ri = 0;
        groups.forEach(function (g) {
            if (g.tierBreak) {
                var counts = POS.map(function (pos) {
                    var n = players.filter(function (y) {
                        return y.dtier === g.tier && y.pos === pos && !state.done.has(y.id) && !y.drafted;
                    }).length;
                    return n ? pos + ' ' + n : null;
                }).filter(Boolean).join(' &middot; ');
                out += '<div class="cs-pgtier">Tier ' + g.tier + (counts ? '<span class="cs-sc">' + counts + ' left</span>' : '') + '</div>';
            }
            var marks = [], seenPk = {};
            g.items.forEach(function (x) {
                var pk = pickAt[(x._mr || 0) + 1];
                if (pk && !seenPk[pk.pn]) {
                    seenPk[pk.pn] = 1;
                    marks.push(pk);
                }
            });
            if (marks.length) {
                var taken = marks.every(function (pk) {
                    return g.items.some(function (x) {
                        var hit = pickAt[(x._mr || 0) + 1];
                        return hit && hit.pn === pk.pn && x.drafted;
                    });
                });
                out += '<div class="cs-pgtier cs-proj-bar' + (taken ? ' cs-proj-taken' : '') + '">' + marks.map(function (pk) {
                    return 'Proj Pick ' + pk.label;
                }).join(' · ') + '</div>';
            }
            var byPos = {RB: [], WR: [], QB: [], TE: []};
            g.items.forEach(function (x) {
                byPos[x.pos].push(x);
            });
            var alt = (ri % 2) ? ' alt' : '';
            ri++;
            var cells = POS.map(function (pos) {
                return '<div class="cs-pgcell">' + byPos[pos].map(nameChip).join('') + '</div>';
            }).join('');
            out += '<div class="cs-pgrow' + alt + '">' + cells + '</div>';
        });
        $('csPosGrid').innerHTML = out;
        $('csPosFoot').textContent = dyn
            ? 'Read down a column for a position board, across a row for who else goes at that slot. Tap a name to cross it off.'
            : 'Read down a column for a position board, across a row for who else goes at that slot. Green is value over ADP. Tap a name to cross it off.';
    }

    function renderNeedsBar() {
        var bar = $('csNeeds');
        if (!bar) return;
        if (!myCounts) {
            bar.style.display = 'none';
            bar.innerHTML = '';
            return;
        }
        var need = window.__csNeed || {};
        var chips = ['QB', 'RB', 'WR', 'TE'].map(function (pos) {
            var n = need[pos] || {need: 0, have: 0, target: 0};
            if (n.need > 0) return '<span class="cs-need cs-need-open">' + pos + ' +' + n.need + '</span>';
            return '<span class="cs-need cs-need-full">' + pos + ' full</span>';
        }).join('');
        bar.style.display = '';
        bar.innerHTML = '<span class="cs-need-lbl">Your roster</span>' + chips
            + '<span class="cs-need-hint">from your live picks</span>';
    }

    // ── ADP source selector ─────────────────────────────────────────────────────
    function renderAdpSources() {
        var sel = $('csAdpSrc');
        if (!sel) return;
        var opts = adpSourceOptions[scoringAxisKey()] || [];
        if (!opts.length) {
            sel.style.display = 'none';
            return;
        }
        sel.style.display = '';
        var cur = sheetAdpSource();
        if (!opts.some(function (o) {
            return o.value === cur;
        })) cur = (opts[0] && opts[0].value) || cur;
        sel.innerHTML = opts.map(function (o) {
            return '<option value="' + esc(o.value) + '"' + (o.value === cur ? ' selected' : '') + '>ADP: ' + esc(o.label) + '</option>';
        }).join('');
    }

    function leagueParams() {
        var p = [];
        if (cfg.leagueId) p.push('league_id=' + encodeURIComponent(cfg.leagueId));
        if (cfg.platform) p.push('platform=' + encodeURIComponent(cfg.platform));
        return p;
    }

    function loadPlayers() {
        var requestId = ++playerRequest;
        if (playerAbort) playerAbort.abort();
        playerAbort = typeof AbortController !== 'undefined' ? new AbortController() : null;
        loading = true;
        loadError = '';
        var params = ['view=board'];
        params.push('league_type=' + (state.sf ? 'sf' : '1qb'));
        // Projection context matches Draft Room so half-PPR / TE premium / 6-pt
        // pass TD select the same Sleeper variant (and cache key) the room uses.
        var sc = scoringCfg();
        params.push('proj_rec=' + encodeURIComponent(String(sc.ppr)));
        params.push('proj_te_bonus=' + encodeURIComponent(String(sc.tep)));
        params.push('proj_pass_td=' + encodeURIComponent(String(sc.passTd)));
        // Always send league context so Yahoo viewers get the same rebuilt
        // consensus column Player Rankings shows. Do not pass adp_source: the
        // sheet reads p.adp_by_source so every dropdown choice matches the
        // rankings source column (to one decimal) instead of overlaying Sleeper
        // onto avg_pick and rounding it.
        params = params.concat(leagueParams());
        var url = '/api/league-players' + (params.length ? ('?' + params.join('&')) : '');
        var pending = window.__cheatPlayersP;
        var req;
        if (pending && pending.url === url) {
            window.__cheatPlayersP = null;
            req = pending.catch(function () {
                return fetch(url, {cache: 'no-store', signal: playerAbort ? playerAbort.signal : undefined})
                    .then(function (r) {
                        if (!r.ok) throw new Error('Players request failed (' + r.status + ')');
                        return r.json();
                    });
            });
        } else {
            req = fetch(url, {cache: 'no-store', signal: playerAbort ? playerAbort.signal : undefined})
                .then(function (r) {
                    if (!r.ok) throw new Error('Players request failed (' + r.status + ')');
                    return r.json();
                });
        }
        return req
            .then(function (resp) {
                if (requestId !== playerRequest) return;
                var raw = Array.isArray(resp) ? resp : (resp.players || []);
                if (!Array.isArray(resp)) {
                    if (resp.tier_thresholds) tierThresholds = resp.tier_thresholds;
                    if (resp.adp_source_options) adpSourceOptions = resp.adp_source_options;
                    SHOW_MARKET_VS_ADP = resp.market_vs_adp_available === true;
                    SHOW_HISTORICAL = resp.historical_available === true;
                } else {
                    SHOW_MARKET_VS_ADP = false;
                    SHOW_HISTORICAL = false;
                }
                allPlayers = raw.filter(function (p) {
                    return p && p.id != null && ['QB', 'RB', 'WR', 'TE'].indexOf(String(p.position || '').toUpperCase()) >= 0;
                });
                loading = false;
                renderAdpSources();
                compute();
                render();
                // Only the displayed board needs schedule context. Keeping this to the
                // 175-row sheet also avoids an oversized query for the full player index.
                loadScheduleRanks(players);
            })
            .catch(function (err) {
                if (requestId !== playerRequest || (err && err.name === 'AbortError')) return;
                loading = false;
                loadError = 'Could not load players. Refresh to retry.';
                allPlayers = [];
                players = [];
                SHOW_MARKET_VS_ADP = false;
                SHOW_HISTORICAL = false;
                render();
                $('csPosGrid').innerHTML = '<div class="cs-empty">' + loadError + '</div>';
            });
    }

    // Schedule rank is supporting draft context, not an input to the VOR order.
    // Fetch the full fantasy regular season once the player pool is known and
    // merge the API's position-specific SoS rank onto every matching row.
    function loadScheduleRanks(pool) {
        var ids = pool.map(function (p) {
            return String(p.id);
        });
        if (!ids.length) return;
        var requestId = ++scheduleRequest;
        var season = Number(cfg.season) || new Date().getFullYear();
        var url = '/api/schedule?season=' + season + '&week_start=1&week_end=17&pids=' + encodeURIComponent(ids.join(','));
        fetch(url, {cache: 'no-store'})
            .then(function (r) {
                if (!r.ok) throw new Error('Schedule request failed');
                return r.json();
            })
            .then(function (resp) {
                if (requestId !== scheduleRequest) return;
                scheduleRanks = {};
                (resp.players || []).forEach(function (p) {
                    if (p && p.pid != null && p.sos_rank != null) scheduleRanks[String(p.pid)] = Number(p.sos_rank);
                });
                compute();
                render();
            })
            .catch(function () { /* Schedule context degrades to an em dash. */
            });
    }

    // ── live-draft cross-off ────────────────────────────────────────────────────
    function detectLiveDraft() {
        // Live Sleeper draft sync (auto cross-off + real-time board) is free once
        // the viewer has a connected league. Custom board edits stay PRO.
        if (!cfg.leagueId || !cfg.platform) return Promise.resolve(false);
        return fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || '')
            + (String(cfg.platform || '').toLowerCase() === 'espn' ? '&sync=1' : ''))
            .then(function (r) {
                return r.json();
            })
            .then(function (resp) {
                if (!resp || resp.unsupported) return;
                var all = resp.drafts || [];
                // Only connect to a current/upcoming draft. Historical drafts should not
                // unexpectedly replace a clean cheat sheet.
                var pick = all.filter(function (d) {
                        return String(d.status) === 'drafting';
                    })[0]
                    || all.filter(function (d) {
                        return String(d.status) === 'pre_draft';
                    })[0];
                if (!pick || !pick.draft_id) return false;
                liveDraftId = pick.draft_id;
                pollDraft();   // start the live loop
                render();
                return true;
            })
            .catch(function () {
                return false;
            });
    }

    function disconnectLiveDraft() {
        liveDraftId = null;
        if (pollTimer) clearTimeout(pollTimer);
        pollTimer = null;
        draftedIds = null;
        myCounts = null;
    }

    // Poll the live draft so players auto-cross-off and the roster-need bar update
    // in real time. Stops when the draft completes; backs off when the tab is
    // hidden; re-fetches faster while actively drafting.
    function schedulePoll(ms) {
        if (pollTimer) clearTimeout(pollTimer);
        if (liveDraftId) pollTimer = setTimeout(pollDraft, ms);
    }

    function pollDraft() {
        if (!liveDraftId) return;
        if (typeof document !== 'undefined' && document.hidden) {
            schedulePoll(10000);
            return;
        }
        var requestedDraftId = liveDraftId;
        fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(requestedDraftId), {cache: 'no-store'})
            .then(function (r) {
                return r.json();
            })
            .then(function (d) {
                if (liveDraftId !== requestedDraftId) return;
                applyLiveDraft(d);
                var status = d && String(d.status || '');
                if (status === 'complete') {
                    liveDraftId = null;
                    return;
                }   // final state applied; stop
                schedulePoll(status === 'drafting' ? 5000 : 12000);          // slower before it starts
            })
            .catch(function () {
                if (liveDraftId === requestedDraftId) schedulePoll(10000);
            });
    }

    function applyLiveDraft(d) {
        var picks = (d && d.picks) || [];
        var s = new Set();
        var mine = {QB: 0, RB: 0, WR: 0, TE: 0};
        picks.forEach(function (pk) {
            if (!pk || !pk.player_id) return;
            s.add(String(pk.player_id));
            if (cfg.viewerUserId && String(pk.picked_by || '') === String(cfg.viewerUserId)) {
                var pos = String(pk.position || '').toUpperCase();
                if (mine[pos] != null) mine[pos]++;
            }
        });
        // Every poll is an authoritative snapshot. This also clears stale marks if a
        // commissioner rolls a pick back (including rolling the draft back to zero).
        draftedIds = s.size ? s : null;
        myCounts = cfg.viewerUserId ? mine : null;
        compute();
        render();
    }

    // Draft Room overlay pushes pick updates while it stays open so this sheet
    // stays crossed-off without turning on Sleeper live polling.
    function applyDraftRoomContext(payload) {
        if (!payload || payload.type !== 'drCheatContext') return;
        var changed = false;
        // When Connect live draft is polling Sleeper, that feed owns drafted/myCounts.
        // Recommendation order still comes from the room (Sleeper has no REC #).
        if (!liveDraftId && Object.prototype.hasOwnProperty.call(payload, 'drafted')) {
            var list = Array.isArray(payload.drafted) ? payload.drafted : [];
            draftedIds = list.length ? new Set(list.map(String)) : null;
            changed = true;
        }
        if (Object.prototype.hasOwnProperty.call(payload, 'rec_order')) {
            var rec = Array.isArray(payload.rec_order) ? payload.rec_order : [];
            if (!rec.length) {
                recommendationOrder = null;
            } else {
                recommendationOrder = {};
                rec.forEach(function (id, i) {
                    id = String(id);
                    if (recommendationOrder[id] == null) recommendationOrder[id] = i;
                });
            }
            changed = true;
        }
        if (!liveDraftId && payload.myCounts && typeof payload.myCounts === 'object') {
            myCounts = payload.myCounts;
            changed = true;
        }
        var qTeams = parseInt(payload.teams, 10);
        if (qTeams >= 2 && qTeams <= 32) {
            teams = qTeams;
            changed = true;
        }
        var qSlot = parseInt(payload.slot, 10);
        if (qSlot >= 1 && qSlot <= teams) {
            state.pickSlot = qSlot;
            changed = true;
        }
        if (payload.scoring && typeof payload.scoring === 'object') {
            var nextSc = normalizeScoring(payload.scoring);
            var curSc = scoringCfg();
            if (nextSc.ppr !== curSc.ppr || nextSc.tep !== curSc.tep || nextSc.passTd !== curSc.passTd) {
                state.scoring = nextSc;
                syncScoringUi();
                // Scoring changes the projection cache key — refetch the pool.
                loadPlayers();
                return;
            }
        }
        if (changed) {
            compute();
            render();
        }
    }

    // ── CSV export ──────────────────────────────────────────────────────────────
    function exportCsv() {
        if (!players.length) return;
        var dyn = state.mode === 'dynasty';
        var head = ['Rank', 'Player', 'Pos', 'PosRank', 'VOR', 'Proj PPG', (dyn ? 'Age' : 'ADP'), (dyn ? 'Window' : 'Value'), 'Schedule Rank'].concat(showHist(dyn) ? ['Hist P(top-12)'] : []).concat(showMarket(dyn) ? ['Market vs ADP'] : []).concat(['Tier']);
        var rows = displayPlayers().map(function (x) {
            var c5 = dyn ? (x.age != null ? x.age : '') : fmtAdp(x.adp);
            var c6 = dyn ? youthWindow(x.age, x.pos)[0] : (x.value != null ? (x.value > 0 ? '+' + x.value : x.value) : '');
            var ppgCsv = x.projectedPpg != null ? x.projectedPpg.toFixed(1) : '';
            return [x.rk, x.name, x.pos, x.prk, x.vor, ppgCsv, c5, c6, x.scheduleRank || ''].concat(showHist(dyn) ? [x.histP == null ? '' : x.histP] : []).concat(showMarket(dyn) ? [x.marketVsAdp == null ? '' : x.marketVsAdp] : []).concat([x.dtier]);
        });
        var csv = [head].concat(rows).map(function (r) {
            return r.map(function (v) {
                var s = String(v == null ? '' : v);
                return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
            }).join(',');
        }).join('\n');
        var blob = new Blob([csv], {type: 'text/csv;charset=utf-8'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = (dyn ? 'dynasty' : 'redraft') + '-cheat-sheet-' + (state.sf ? 'sf' : '1qb') + '.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function () {
            URL.revokeObjectURL(url);
        }, 1000);
    }

    // ── interactions ────────────────────────────────────────────────────────────
    function wireSeg(id, apply) {
        document.querySelectorAll('#' + id + ' button').forEach(function (b) {
            b.addEventListener('click', function () {
                document.querySelectorAll('#' + id + ' button').forEach(function (x) {
                    x.setAttribute('aria-pressed', String(x === b));
                });
                apply(b);
                renderAdpSources();
                compute();
                render();
            });
        });
    }

    function closeHistPanel() {
        var modal = $('csHistModal');
        if (!modal) return;
        modal.classList.remove('open', 'cs-hist-qb', 'cs-hist-rb', 'cs-hist-wr', 'cs-hist-te');
        var posEl = $('csHistPos');
        if (posEl) {
            posEl.hidden = true;
            posEl.textContent = '';
            posEl.className = 'cs-pos-badge';
        }
    }

    function histPosKey(value) {
        var p = String(value || '').toUpperCase();
        return (p === 'QB' || p === 'RB' || p === 'WR' || p === 'TE') ? p : '';
    }

    function histPosOf(resp, fallback) {
        var found = histPosKey(fallback);
        if (found) return found;
        if (!resp) return '';
        var pre = resp.preseason || {};
        found = histPosKey(pre.position || pre.pos);
        if (found) return found;
        var hist = resp.history || {};
        var key = hist.key_used || {};
        return histPosKey(key.position);
    }

    function applyHistPos(pos) {
        var modal = $('csHistModal');
        var key = histPosKey(pos);
        if (modal) {
            modal.classList.remove('cs-hist-qb', 'cs-hist-rb', 'cs-hist-wr', 'cs-hist-te');
            if (key) modal.classList.add('cs-hist-' + key.toLowerCase());
        }
        var posEl = $('csHistPos');
        if (!posEl) return;
        if (key) {
            posEl.hidden = false;
            posEl.className = 'cs-pos-badge cs-pos-' + key;
            posEl.textContent = key;
        } else {
            posEl.hidden = true;
            posEl.textContent = '';
            posEl.className = 'cs-pos-badge';
        }
    }

    function findSheetPlayer(id) {
        id = String(id || '');
        var i;
        for (i = 0; i < players.length; i++) {
            if (String(players[i].id) === id) return players[i];
        }
        for (i = 0; i < allPlayers.length; i++) {
            if (String(allPlayers[i].id) === id) return allPlayers[i];
        }
        return null;
    }

    function liveHistAdp(id, attrAdp) {
        if (attrAdp != null && attrAdp !== '' && isFinite(Number(attrAdp)) && Number(attrAdp) > 0 && Number(attrAdp) < 999) {
            return Number(attrAdp);
        }
        var row = findSheetPlayer(id);
        if (!row) return null;
        var candidates = [row.adp, row.marketAdp, row.redraft_avg_pick, row.adp_overall];
        if (row.position || row.pos) {
            try { candidates.push(sheetAdpOf(row, state.mode, state.sf)); } catch (e) {}
        }
        var i;
        for (i = 0; i < candidates.length; i++) {
            var v = candidates[i];
            if (v != null && isFinite(Number(v)) && Number(v) > 0 && Number(v) < 999) return Number(v);
        }
        return null;
    }

    function openHistPanel(id, name, adp, pos, projPpg, projRk, adpRk) {
        var modal = $('csHistModal');
        if (!modal || !id) return;
        var row = findSheetPlayer(id);
        var hist = (row && row.historical) || {};
        var liveAdp = liveHistAdp(id, adp);
        var livePos = pos || (row && (row.pos || row.position)) || '';
        var liveProj = (projPpg != null && projPpg !== '' && isFinite(Number(projPpg)))
            ? Number(projPpg)
            : (row && row.projectedPpg != null && isFinite(Number(row.projectedPpg)) ? Number(row.projectedPpg) : null);
        var liveProjRk = (projRk != null && projRk !== '' && isFinite(Number(projRk)))
            ? Number(projRk)
            : (hist.proj_rk != null ? Number(hist.proj_rk) : null);
        var liveAdpRk = (adpRk != null && adpRk !== '' && isFinite(Number(adpRk)))
            ? Number(adpRk)
            : (hist.adp_rk != null ? Number(hist.adp_rk) : null);
        var fallbackMarket = hist.mkt_sentence;
        $('csHistTitle').textContent = name || 'History';
        $('csHistSub').textContent = 'Historical chance for this career and situation.';
        $('csHistBody').innerHTML = '<p class="cs-hist-sub">Loading…</p>';
        applyHistPos(livePos);
        modal.classList.add('open');
        var url = '/api/historical-player/' + encodeURIComponent(id);
        var qs = [];
        if (liveAdp != null) {
            qs.push('adp=' + encodeURIComponent(liveAdp));
            qs.push('redraft_avg_pick=' + encodeURIComponent(liveAdp));
        }
        if (livePos) qs.push('position=' + encodeURIComponent(livePos));
        if (liveProj != null) qs.push('proj_ppg=' + encodeURIComponent(liveProj));
        if (liveProjRk != null && isFinite(liveProjRk)) qs.push('proj_rk=' + encodeURIComponent(liveProjRk));
        if (liveAdpRk != null && isFinite(liveAdpRk)) qs.push('adp_rk=' + encodeURIComponent(liveAdpRk));
        var liveSpot = hist.trend_feats && hist.trend_feats.roster_spot;
        if (liveSpot != null && isFinite(Number(liveSpot))) qs.push('roster_spot=' + encodeURIComponent(liveSpot));
        if (qs.length) url += '?' + qs.join('&');
        fetch(url, {cache: 'no-store'})
            .then(function (r) {
                if (!r.ok) throw new Error('hist ' + r.status);
                return r.json();
            })
            .then(function (resp) {
                if (!modal.classList.contains('open')) return;
                var body = $('csHistBody');
                if (!body) return;
                try {
                    body.innerHTML = renderHistPanel(resp, fallbackMarket, livePos);
                } catch (err) {
                    body.innerHTML = '<p class="cs-hist-sub">Could not load similar-player history.</p>';
                }
            })
            .catch(function () {
                if (!modal.classList.contains('open')) return;
                var body = $('csHistBody');
                if (body) body.innerHTML = '<p class="cs-hist-sub">Could not load similar-player history.</p>';
            });
    }

    function histTrendTitle(row) {
        row = row || {};
        if (row.title) return String(row.title);
        var bucket = String(row.bucket || '').trim();
        var label = String(row.label || '').trim();
        var kind = row.kind || '';
        var qualified = bucket ? trendsQualifyLabel(kind, bucket) : '';
        var generic = {
            age: 1, 'draft capital': 1, 'career stage': 1,
            'last year target share': 1, 'last year snaps': 1,
            'last year adot': 1, 'last year rush yards over expected': 1,
            'last year touches': 1, 'last year carries': 1,
            'last year receptions': 1, 'last year targets': 1,
            'last year games played': 1, 'last year pass attempts': 1
        };
        if (kind === 'draft_capital' && bucket) return 'Drafted NFL ' + bucket + ', any season';
        if (kind === 'top12_as_rookie' && bucket) return 'Drafted NFL ' + bucket + ', year 1';
        if (kind === 'top12_by_year_2' && bucket) return 'Drafted NFL ' + bucket + ', year 2';
        if (kind === 'capital_miss' && bucket) return 'Drafted NFL ' + bucket + ', miss (any season)';
        if (kind === 'career_stage' && bucket) {
            return (String(bucket).toLowerCase() === 'rookie' ? 'Rookie' : bucket) + ' season, any capital';
        }
        if ((kind === 'age' || kind === 'age_exact') && bucket) {
            return (String(bucket).toLowerCase().indexOf('age') === 0 ? bucket : 'Age ' + bucket) + ', any season';
        }
        if (kind === 'offense' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' projected offense';
        }
        if (kind === 'offense_year_1' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' projected offense, year 1';
        }
        if (kind === 'offense_year_2' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' projected offense, year 2';
        }
        if (kind === 'offense_last_year' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' offense last year';
        }
        if (kind === 'offense_last_year_1' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' offense last year, year 1';
        }
        if (kind === 'offense_last_year_2' && bucket) {
            return (String(bucket).toLowerCase() === 'top 10' ? 'Top-10' : bucket) + ' offense last year, year 2';
        }
        if ((kind === 'offense_roster' || kind === 'offense_roster_1' || kind === 'offense_roster_2') && bucket) {
            var bits = String(bucket).split(', ');
            var band = bits[0] || bucket;
            var spot = bits.length > 1 ? bits.slice(1).join(', ') : '';
            var base = (String(band).toLowerCase() === 'top 10' ? 'Top-10' : band) + ' projected offense';
            if (spot) base += ', ' + spot;
            if (kind === 'offense_roster_1') return base + ', year 1';
            if (kind === 'offense_roster_2') return base + ', year 2';
            return base;
        }
        if ((kind === 'capital_roster' || kind === 'capital_roster_1' || kind === 'capital_roster_2') && bucket) {
            var bits = String(bucket).split(', ');
            var cap = bits[0] || bucket;
            var spot = bits.length > 1 ? bits.slice(1).join(', ') : '';
            var base = 'Drafted NFL ' + cap;
            if (spot) base += ', ' + spot;
            if (kind === 'capital_roster_1') return base + ', year 1';
            if (kind === 'capital_roster_2') return base + ', year 2';
            return base;
        }
        if (kind === 'offense_capital' && bucket) {
            var bits = String(bucket).split(', ');
            var band = bits[0] || bucket;
            var cap = bits.length > 1 ? bits.slice(1).join(', ') : '';
            var base = (String(band).toLowerCase() === 'top 10' ? 'Top-10' : band) + ' projected offense';
            if (cap) base += ', ' + (String(cap).toLowerCase() === 'top 10' ? 'NFL Top 10' : cap);
            return base;
        }
        if (kind === 'bounce_roster' && bucket) {
            return 'Outside top 36 last year, ' + bucket;
        }
        if (label && !generic[label.toLowerCase()]) return label;
        return qualified || label || row.sentence || '';
    }

    function trendsBaselineOf(row) {
        if (!row || row.pct == null || !isFinite(Number(row.pct))) return null;
        if (typeof row.vs_baseline !== 'number') return null;
        return Number(row.pct) - Number(row.vs_baseline);
    }

    function histSampleLabel(n) {
        return n == null ? '' : ('Sample: ' + n);
    }

    function histTrendRow(row, barHtml, markCells) {
        row = row || {};
        var meta = [];
        if (row.n != null) meta.push(histSampleLabel(row.n));
        if (row.secondary) meta.push(row.secondary);
        var shown = row.display != null && row.display !== ''
            ? row.display
            : (row.pct != null ? row.pct + '%' : '-');
        var vsShort = (typeof row.vs_baseline === 'number' && row.vs_baseline !== 0)
            ? ((row.vs_baseline > 0 ? '+' : '') + row.vs_baseline)
            : '';
        var isThis = row.role === 'this';
        var roleChip = (markCells && isThis)
            ? '<span class="cs-hist-hit-role">This player</span>'
            : '';
        var roleClass = markCells ? (isThis ? ' is-this' : ' is-analog') : '';
        var names = Array.isArray(row.examples) ? row.examples.filter(Boolean) : [];
        var namesHtml = '';
        if (names.length) {
            var peek = names.slice(0, 2).map(function (ex) {
                return (ex.name || '') + (ex.season ? ' ' + ex.season : '');
            }).filter(Boolean).join(', ');
            namesHtml = '<details class="cs-hist-tile-ex"><summary>'
                + esc(peek || 'Names') + '</summary><ul>';
            names.forEach(function (ex) {
                var hit = histExampleHit(ex.positional_finish, ex);
                namesHtml += '<li' + (hit && hit.tier ? ' class="is-' + esc(hit.tier) + '"' : '') + '>'
                    + '<span>' + esc(ex.name || '')
                    + (ex.season ? ' · ' + esc(String(ex.season)) : '') + '</span>'
                    + (hit ? '<b class="cs-hist-ex-hit">' + esc(hit.label) + '</b>' : '')
                    + '</li>';
            });
            namesHtml += '</ul></details>';
        }
        return '<div class="cs-hist-hit' + (row.polarity === 'miss' ? ' is-miss' : '')
            + roleClass + '"><div class="cs-hist-hit-top">'
            + trendsConfDot(row.confidence_label)
            + '<div><div class="cs-hist-hit-label">' + esc(histTrendTitle(row)) + '</div>'
            + roleChip
            + (meta.length ? '<div class="cs-hist-hit-meta">' + esc(meta.join(' · ')) + '</div>' : '')
            + '</div><div class="cs-hist-hit-pct">' + esc(String(shown))
            + (vsShort ? ' <span>' + esc(String(vsShort)) + '</span>' : '')
            + '</div></div>'
            + (barHtml || '')
            + namesHtml
            + '</div>';
    }

    function trendsHitRow(row, polarity, baselinePct, span, markCells) {
        row = row || {};
        var pol = polarity || row.polarity;
        var base = pol === 'miss' ? null : (baselinePct != null ? baselinePct : trendsBaselineOf(row));
        return histTrendRow(row, trendsRailHtml(row.pct, base, pol, span), markCells);
    }

    function defaultTrendsPos() {
        var p = String(state.posFilter || 'ALL').toUpperCase();
        if (['QB', 'RB', 'WR', 'TE'].indexOf(p) >= 0) return p;
        return 'RB';
    }

    function showSheetTab(tab) {
        if (!TAB_PANELS[tab]) tab = 'board';
        currentTab = tab;
        document.querySelectorAll('.cs-tabs [role=tab]').forEach(function (x) {
            x.setAttribute('aria-selected', String(x.getAttribute('data-tab') === tab));
        });
        Object.keys(TAB_PANELS).forEach(function (k) {
            var el = $(TAB_PANELS[k]);
            if (el) el.classList.toggle('cs-hidden', k !== tab);
        });
        var hideBoardChrome = tab === 'logic' || tab === 'trends';
        var legend = $('csLegend');
        if (legend) legend.style.display = hideBoardChrome ? 'none' : '';
        var fb = $('csFilterbar');
        if (fb) fb.style.display = hideBoardChrome ? 'none' : '';
        if (tab === 'trends') {
            setTrendsNavOffset();
            loadTrends();
        }
    }

    function syncHistSurfaces() {
        var trendsOn = showTrends();
        var tab = document.querySelector('.cs-tabs [data-tab="trends"]');
        if (tab) tab.classList.toggle('cs-hidden', !trendsOn);
        if (!trendsOn && currentTab === 'trends') showSheetTab('board');
    }

    function loadTrends() {
        if (!trendsPos) trendsPos = defaultTrendsPos();
        if (trendsCache) {
            renderTrends();
            return;
        }
        var host = $('csTrends');
        if (host) host.innerHTML = '<p class="cs-hist-sub">Loading historical trends…</p>';
        var req = ++trendsRequest;
        fetch('/api/historical-trends', {cache: 'no-store'})
            .then(function (r) { return r.json(); })
            .then(function (resp) {
                if (req !== trendsRequest) return;
                trendsCache = resp;
                renderTrends();
            })
            .catch(function () {
                if (req !== trendsRequest) return;
                var el = $('csTrends');
                if (el) el.innerHTML = '<p class="cs-hist-sub">Could not load historical trends.</p>';
            });
    }

    var TRENDS_LANE_OF = {
        repeat: 'career', league_winner: 'career', career_stage: 'career',
        repeat_top5: 'career', two_plus: 'career', breakout: 'career',
        first_time_elite: 'career', league_winner_smash: 'career',
        draft_capital: 'capital', top12_as_rookie: 'capital', top12_by_year_2: 'capital',
        capital_miss: 'capital',
        age: 'age', age_exact: 'age', prime: 'age',
        target_share: 'usage', snap_pct: 'usage', adot: 'usage', ryoe: 'usage',
        touches: 'usage', carries: 'usage', receptions: 'usage',
        targets: 'usage', games: 'usage', pass_attempts: 'usage',
        target_share_change: 'usage', snap_pct_change: 'usage',
        workload_change: 'usage',
        offense: 'team', offense_year_1: 'team', offense_year_2: 'team',
        offense_last_year: 'team', offense_last_year_1: 'team', offense_last_year_2: 'team',
        offense_roster: 'team', offense_roster_1: 'team', offense_roster_2: 'team',
        capital_roster: 'capital', capital_roster_1: 'capital', capital_roster_2: 'capital',
        offense_capital: 'team',
        bounce_roster: 'career'
    };
    var TRENDS_LANES = [
        ['all', 'All'], ['career', 'Career'],
        ['capital', 'Capital'], ['age', 'Age'], ['usage', 'Usage'],
        ['team', 'Team']
    ];
    var TRENDS_LABEL_PREFIX = {
        draft_capital: 'NFL',
        capital_miss: 'NFL',
        top12_as_rookie: 'NFL',
        top12_by_year_2: 'NFL',
        target_share: 'Targets',
        snap_pct: 'Snaps',
        adot: 'aDOT',
        ryoe: 'RYOE',
        touches: 'Touches',
        carries: 'Carries',
        receptions: 'Receptions',
        targets: 'Targets',
        games: 'Games',
        pass_attempts: 'Attempts',
        age: 'Age',
        age_exact: 'Age',
        target_share_change: 'Targets',
        snap_pct_change: 'Snaps',
        workload_change: 'Workload',
        offense: 'Offense',
        offense_year_1: 'Offense',
        offense_year_2: 'Offense',
        offense_roster: 'Offense',
        offense_roster_1: 'Offense',
        offense_roster_2: 'Offense',
        capital_roster: 'NFL',
        capital_roster_1: 'NFL',
        capital_roster_2: 'NFL',
        offense_capital: 'Offense',
        bounce_roster: 'Last year'
    };

    function trendsConfKey(label) {
        var t = String(label || '').toLowerCase();
        if (t.indexOf('large') >= 0 || t === 'strong') return 'strong';
        if (t.indexOf('solid') >= 0 || t === 'good') return 'good';
        if (t.indexOf('moderate') >= 0) return 'moderate';
        if (t.indexOf('small') >= 0 || t === 'low') return 'low';
        return '';
    }

    function trendsConfDot(label) {
        var key = trendsConfKey(label);
        if (!key) return '';
        return '<span class="cs-trends-conf cs-trends-conf-' + key + '" title="'
            + esc(label) + '"><i></i></span>';
    }

    function trendsRailSpan(baselinePct, sections, extraPcts) {
        var maxV = baselinePct != null && isFinite(Number(baselinePct)) ? Number(baselinePct) : 1;
        (extraPcts || []).forEach(function (p) {
            if (p != null && isFinite(Number(p)) && Number(p) > maxV) maxV = Number(p);
        });
        (sections || []).forEach(function (sec) {
            (sec.rows || []).forEach(function (row) {
                if (row && row.pct != null && isFinite(Number(row.pct)) && Number(row.pct) > maxV) {
                    maxV = Number(row.pct);
                }
            });
        });
        return Math.min(100, Math.max(maxV * 1.08, 12));
    }

    function trendsRailHtml(pct, baselinePct, polarity, span) {
        if (pct == null || !isFinite(Number(pct))) return '';
        var p = Math.max(0, Math.min(100, Number(pct)));
        var scale = span > 0 ? span : 100;
        var leftP = Math.max(0, Math.min(100, (p / scale) * 100));
        var base = (baselinePct != null && isFinite(Number(baselinePct))) ? Number(baselinePct) : null;
        var html = '<div class="cs-trends-rail' + (polarity === 'miss' ? ' is-miss' : '')
            + (base != null && p + 0.5 < base ? ' is-down' : '') + '" aria-hidden="true">';
        html += '<span class="cs-trends-rail-track"></span>';
        if (base != null) {
            var leftB = Math.max(0, Math.min(100, (base / scale) * 100));
            var fillLeft = Math.min(leftB, leftP);
            var fillW = Math.abs(leftP - leftB);
            html += '<span class="cs-trends-rail-base" style="left:' + leftB.toFixed(2) + '%"></span>';
            if (fillW > 0.4) {
                html += '<span class="cs-trends-rail-fill" style="left:' + fillLeft.toFixed(2)
                    + '%;width:' + fillW.toFixed(2) + '%"></span>';
            }
        } else {
            html += '<span class="cs-trends-rail-fill" style="left:0;width:' + leftP.toFixed(2) + '%"></span>';
        }
        html += '<span class="cs-trends-rail-mark" style="left:' + leftP.toFixed(2) + '%"></span></div>';
        return html;
    }

    function trendsQualifyLabel(sid, label) {
        label = String(label || '');
        var prefix = TRENDS_LABEL_PREFIX[sid];
        if (!prefix) return label;
        if (label.toLowerCase().indexOf(prefix.toLowerCase()) >= 0) return label;
        return prefix + ' ' + label;
    }

    function trendsFinishLabel(tier) {
        return HIST_TIER_SHORT[tier] || String(tier || '').replace('_', '-');
    }

    function trendsPicksFor(pos) {
        var key = String(pos || trendsPos || 'RB');
        if (!trendsPicks[key]) trendsPicks[key] = {};
        return trendsPicks[key];
    }

    function trendsRowView(row, sec, tier) {
        row = row || {};
        var tied = !!(sec && sec.finish_tied);
        var pct = row.pct;
        var vs = row.vs_baseline;
        var vsLabel = row.vs_label;
        if (tied && row.pcts && row.pcts[tier] != null) {
            pct = row.pcts[tier];
            if (row.vs_by_tier && row.vs_by_tier[tier] != null) vs = row.vs_by_tier[tier];
            if (row.vs_label_by_tier && row.vs_label_by_tier[tier]) vsLabel = row.vs_label_by_tier[tier];
        }
        return { pct: pct, vs: vs, vsLabel: vsLabel };
    }

    function trendsPageBaseline(page, tier) {
        var block = ((page && page.baselines) || {})[tier] || {};
        if (block.pct != null && isFinite(Number(block.pct))) {
            return { pct: Number(block.pct), n: block.n };
        }
        if (tier === 'top_12') {
            return { pct: page && page.baseline_pct, n: page && page.baseline_n };
        }
        return { pct: null, n: null };
    }

    function trendsBoardFeaturesPayload(pos) {
        var want = String(pos || trendsPos || '').toUpperCase();
        var featsIndex = (trendsCache && trendsCache.player_features) || {};
        var src = (players && players.length) ? players : allPlayers;
        var out = {};
        (src || []).forEach(function (p) {
            if (!p) return;
            var id = String(p.id || '');
            var ppos = String(p.pos || p.position || '').toUpperCase();
            if (!id || ppos !== want) return;
            var stamped = p.historical && p.historical.trend_feats;
            var feats = stamped || featsIndex[id] || null;
            if (feats && typeof feats === 'object') out[id] = feats;
        });
        return out;
    }

    function trendsSignedPts(pts) {
        if (pts == null || !isFinite(Number(pts))) return '';
        var n = Number(pts);
        if (n > 0) return '+' + n + ' pts';
        if (n < 0) return String(n) + ' pts';
        return '0 pts';
    }

    function trendsFiltersPayload(picks) {
        return Object.keys(picks || {}).map(function (id) {
            var rec = picks[id] || {};
            var spec = rec.match || {};
            var out = {
                group: spec.group || spec.field || rec.sid || id,
                field: spec.field,
                label: rec.label || spec.eq || spec.field
            };
            ['eq', 'in', 'gte', 'lte', 'between', 'null_as', 'all'].forEach(function (k) {
                if (spec[k] !== undefined) out[k] = spec[k];
            });
            return out;
        }).filter(function (f) { return f.field || (f.all && f.all.length); });
    }

    function loadTrendsCohort(picks, done) {
        var filters = trendsFiltersPayload(picks);
        if (!filters.length) {
            trendsCohort = null;
            if (done) done(null);
            return;
        }
        var req = ++trendsCohortRequest;
        fetch('/api/historical-cohort', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            cache: 'no-store',
            body: JSON.stringify({
                position: trendsPos,
                filters: filters,
                tier: trendsTier,
                board_features: trendsBoardFeaturesPayload(trendsPos)
            })
        }).then(function (r) { return r.json(); }).then(function (resp) {
            if (req !== trendsCohortRequest) return;
            trendsCohort = resp;
            if (done) done(resp);
        }).catch(function () {
            if (req !== trendsCohortRequest) return;
            trendsCohort = null;
            if (done) done(null);
        });
    }

    function trendsCohortPct(cohort, key) {
        if (!cohort) return null;
        if (key === 'top_12' && cohort.display_pct != null) return cohort.display_pct;
        var rec = ((cohort.rates || {})[key]) || {};
        return rec.display_pct != null ? rec.display_pct : rec.pct;
    }

    function trendsScoutHits(data, pos, picks) {
        var src = (players && players.length) ? players : allPlayers;
        var byId = {};
        (src || []).forEach(function (p) {
            if (p && p.id != null) byId[String(p.id)] = p;
        });
        var matches = (trendsCohort && Array.isArray(trendsCohort.scout_matches))
            ? trendsCohort.scout_matches : [];
        var hits = [];
        matches.forEach(function (m) {
            if (!m || !m.id) return;
            var p = byId[String(m.id)];
            if (!p) return;
            var ppos = String(p.pos || p.position || '').toUpperCase();
            if (ppos !== pos) return;
            var hist = p.historical || {};
            var mktPct = hist.mkt_pct != null ? Number(hist.mkt_pct) : null;
            var cohortPct = trendsCohortPct(trendsCohort, 'top_12');
            var edge = (cohortPct != null && mktPct != null && isFinite(mktPct))
                ? Number(cohortPct) - Number(mktPct) : null;
            hits.push({
                id: String(m.id),
                name: p.name || String(m.id),
                pos: ppos,
                adp: p.adp != null ? p.adp : p.redraft_avg_pick,
                vor: p.vorRaw != null ? p.vorRaw : p.vor,
                projectedPpg: p.projectedPpg != null ? p.projectedPpg : p.proj_ppg,
                hist: hist,
                drafted: !!(p.drafted || (draftedIds && draftedIds.has(String(m.id))) || (state.done && state.done.has(String(m.id)))),
                why: Array.isArray(m.why) ? m.why : [],
                cohortPct: cohortPct,
                mktPct: mktPct,
                profileEdge: edge
            });
        });
        hits.sort(function (a, b) {
            var av = a.vor != null && isFinite(Number(a.vor)) ? Number(a.vor) : -9999;
            var bv = b.vor != null && isFinite(Number(b.vor)) ? Number(b.vor) : -9999;
            return bv - av;
        });
        return hits;
    }

    function trendsMiniPcts(row, tier) {
        var pcts = (row && row.pcts) || {};
        var keys = ['top_5', 'top_12', 'top_24'].filter(function (k) { return pcts[k] != null; });
        if (keys.length < 2) return '';
        return '<div class="cs-trends-minipcts">' + keys.map(function (k) {
            return '<span' + (k === tier ? ' class="is-on"' : '') + '>'
                + esc(trendsFinishLabel(k)) + ' ' + esc(String(pcts[k])) + '%</span>';
        }).join('') + '</div>';
    }

    function trendsProfileHtml(cohort, picks, pickCount) {
        var html = '<section class="cs-trends-profile' + (pickCount ? '' : ' is-idle') + '">';
        if (!pickCount) {
            html += '<p class="cs-hist-sub">Tap historical buckets to build a profile.</p></section>';
            return html;
        }
        var labels = Object.keys(picks || {}).map(function (id) {
            return picks[id] && picks[id].label;
        }).filter(Boolean);
        var n = cohort && cohort.sample_size;
        var players = cohort && cohort.n_players;
        html += '<div class="cs-trends-sec-head"><div><h3>Selected profile</h3><p>'
            + esc(trendsPos) + (labels.length ? ' · ' + esc(labels.join(' · ')) : '')
            + '</p></div>';
        if (cohort && cohort.available !== false) {
            html += '<div class="cs-trends-profile-n">' + esc(histSampleLabel(n == null ? 0 : n))
                + (players != null ? ' · ' + esc(String(players)) + ' players' : '')
                + '</div>';
        }
        html += '</div>';
        if (!cohort || cohort.available === false) {
            var reason = cohort && cohort.unknown_reason;
            html += '<p class="cs-hist-sub">'
                + (reason === 'cohort_index_missing'
                    ? 'Combined historical rates need a rebuilt observation index.'
                    : (reason === 'empty_cohort'
                        ? 'No historical player-seasons match that mix.'
                        : 'Loading combined historical rate…'))
                + '</p></section>';
            return html;
        }
        var rates = cohort.rates || {};
        html += '<div class="cs-trends-profile-stats">';
        html += '<div class="cs-trends-profile-tiers">';
        ['top_5', 'top_12', 'top_24'].forEach(function (tier) {
            var rec = rates[tier] || {};
            var pct = rec.display_pct;
            if (pct == null && tier === 'top_12') pct = cohort.display_pct;
            var on = tier === (cohort.tier || trendsTier);
            html += '<div class="cs-trends-profile-tier' + (on ? ' is-on' : '') + '">'
                + '<div class="cs-trends-profile-k">' + esc(trendsFinishLabel(tier)) + '</div>'
                + '<div class="cs-trends-profile-v">' + (pct != null ? esc(String(pct)) + '%' : '-') + '</div>'
                + (tier === 'top_12' && rec.ci_low_pct != null && rec.ci_high_pct != null
                    ? '<div class="cs-trends-profile-ci">' + esc(String(rec.ci_low_pct)) + '-'
                        + esc(String(rec.ci_high_pct)) + '%</div>'
                    : (tier === 'top_12' && cohort.ci_low_pct != null && cohort.ci_high_pct != null
                        ? '<div class="cs-trends-profile-ci">' + esc(String(cohort.ci_low_pct)) + '-'
                            + esc(String(cohort.ci_high_pct)) + '%</div>'
                        : ''))
                + '</div>';
        });
        html += '</div>';
        html += '<dl class="cs-trends-profile-dl">';
        if (cohort.baseline_pct != null) {
            html += '<div><dt>Typical ' + esc(trendsPos) + ' ' + esc(trendsFinishLabel(cohort.tier || 'top_12'))
                + '</dt><dd>' + esc(String(cohort.baseline_pct)) + '%</dd></div>';
        }
        if (cohort.adjusted_edge_pts != null) {
            html += '<div><dt>Adjusted edge</dt><dd class="'
                + (Number(cohort.adjusted_edge_pts) > 0 ? 'is-up' : (Number(cohort.adjusted_edge_pts) < 0 ? 'is-down' : ''))
                + '">' + esc(trendsSignedPts(cohort.adjusted_edge_pts)) + '</dd></div>';
        }
        var mkt = cohort.market || {};
        if (mkt.expected_market_pct != null) {
            html += '<div><dt>At historical ADP</dt><dd>' + esc(String(mkt.expected_market_pct)) + '%</dd></div>';
        }
        if (mkt.market_adjusted_edge_pts != null) {
            html += '<div><dt>Vs market</dt><dd class="'
                + (Number(mkt.market_adjusted_edge_pts) > 0 ? 'is-up' : (Number(mkt.market_adjusted_edge_pts) < 0 ? 'is-down' : ''))
                + '">' + esc(trendsSignedPts(mkt.market_adjusted_edge_pts)) + '</dd></div>';
        }
        if (cohort.confidence_short) {
            html += '<div class="is-conf"><dt>Confidence</dt><dd>' + esc(String(cohort.confidence_short)) + '</dd></div>';
        }
        html += '</dl>';
        html += '<p class="cs-hist-note">Actual matching seasons.</p>';
        html += '</div></section>';
        return html;
    }

    function trendsScoutHtml(data, picks, pickCount) {
        var hits = pickCount ? trendsScoutHits(data, trendsPos, picks) : [];
        var premium = !!cfg.hasPremium;
        var shown = premium ? hits : hits.slice(0, TRENDS_SCOUT_PREVIEW);
        var html = '<section class="cs-trends-scout' + (pickCount ? '' : ' is-idle') + '">';
        html += '<div class="cs-trends-sec-head"><h3>';
        if (!pickCount) {
            html += 'Board players who match</h3><p>Tap historical buckets to build a profile.</p></div></section>';
            return html;
        }
        html += esc(String(hits.length)) + ' matching ' + esc(trendsPos)
            + (hits.length === 1 ? '' : 's') + '</h3><p>';
        var labels = Object.keys(picks).map(function (id) { return picks[id].label; }).filter(Boolean);
        html += esc(labels.join(' + '));
        html += '</p></div>';
        html += '<div class="cs-trends-scout-chips">';
        Object.keys(picks).forEach(function (id) {
            html += '<button type="button" class="cs-trends-chip" data-trends-pick="' + esc(id)
                + '">' + esc(picks[id].label || id) + ' ×</button>';
        });
        html += '<button type="button" class="cs-trends-chip is-clear" data-trends-clear="1">Clear</button>';
        html += '</div>';
        if (!hits.length) {
            html += '<p class="cs-hist-sub">No current-board ' + esc(trendsPos)
                + 's match those buckets. Try fewer filters, or load the Big Board first.</p>';
        } else {
            html += '<div class="cs-trends-scout-list">';
            shown.forEach(function (hit) {
                var adpBit = hit.adp != null && isFinite(Number(hit.adp)) ? 'ADP ' + Number(hit.adp).toFixed(1) : '';
                var edgeBits = [];
                if (hit.cohortPct != null) edgeBits.push(hit.cohortPct + '% hist');
                if (hit.mktPct != null && isFinite(Number(hit.mktPct))) {
                    edgeBits.push(Number(hit.mktPct) + '% at ADP');
                }
                if (hit.profileEdge != null && isFinite(Number(hit.profileEdge))) {
                    edgeBits.push(trendsSignedPts(hit.profileEdge));
                }
                html += '<button type="button" class="cs-trends-player' + (hit.drafted ? ' is-drafted' : '')
                    + '" data-trends-player="' + esc(hit.id) + '" data-trends-name="' + esc(hit.name)
                    + '" data-trends-ppos="' + esc(hit.pos) + '" data-trends-adp="'
                    + (hit.adp != null ? esc(String(hit.adp)) : '') + '" data-trends-proj="'
                    + (hit.projectedPpg != null ? esc(String(hit.projectedPpg)) : '') + '">'
                    + '<span class="cs-pos-badge cs-pos-' + esc(hit.pos) + '">' + esc(hit.pos) + '</span>'
                    + '<span class="cs-trends-player-copy">'
                    + '<span class="cs-trends-player-n">' + esc(hit.name) + '</span>'
                    + (adpBit ? '<span class="cs-trends-player-adp">' + esc(adpBit) + '</span>' : '')
                    + (edgeBits.length ? '<span class="cs-trends-player-edge">' + esc(edgeBits.join(' · ')) + '</span>' : '')
                    + '</span></button>';
            });
            html += '</div>';
            if (!premium && hits.length > TRENDS_SCOUT_PREVIEW) {
                html += '<p class="cs-trends-scout-more">' + (hits.length - TRENDS_SCOUT_PREVIEW)
                    + ' more matching ' + esc(trendsPos) + 's. '
                    + '<button type="button" data-trends-unlock="1">Unlock the full list</button></p>';
            }
        }
        html += '</section>';
        return html;
    }

    function trendsPickRecord(id, sections) {
        var found = null;
        (sections || []).forEach(function (sec) {
            (sec.rows || []).forEach(function (row) {
                if (row && row.id === id) found = { row: row, sec: sec };
            });
        });
        if (!found || !found.row || !found.row.match) return null;
        return {
            id: id,
            label: trendsQualifyLabel(found.sec.id, found.row.label || ''),
            match: found.row.match,
            sid: found.sec.id
        };
    }

    function toggleTrendsPick(id, sections) {
        if (!id) return;
        var bag = trendsPicksFor(trendsPos);
        if (bag[id]) delete bag[id];
        else {
            var rec = trendsPickRecord(id, sections);
            if (rec) bag[id] = rec;
        }
        paintTrendsSelection(sections);
    }

    function paintTrendsSelection(sections) {
        var host = $('csTrends');
        if (!host) return;
        var picks = trendsPicksFor(trendsPos);
        host.querySelectorAll('.cs-trends-srow.is-pick').forEach(function (row) {
            var id = row.getAttribute('data-trends-pick');
            row.classList.toggle('is-on', !!(id && picks[id]));
        });
        function swapDock() {
            var pickCount = Object.keys(picks).length;
            var sticky = host.querySelector('.cs-trends-sticky');
            if (sticky) {
                sticky.classList.toggle('is-picked', !!pickCount);
                sticky.classList.toggle('is-collapsed', !trendsDockOpen);
            }
            var wrap = document.createElement('div');
            wrap.innerHTML = trendsProfileHtml(trendsCohort, picks, pickCount)
                + trendsScoutHtml(trendsCache, picks, pickCount);
            var profileHost = host.querySelector('.cs-trends-profile');
            var scoutHost = host.querySelector('.cs-trends-scout');
            var nextProfile = wrap.querySelector('.cs-trends-profile');
            var nextScout = wrap.querySelector('.cs-trends-scout');
            if (profileHost && nextProfile) profileHost.replaceWith(nextProfile);
            if (scoutHost && nextScout) scoutHost.replaceWith(nextScout);
            syncTrendsDockToggle(host);
            bindTrendsDock(host, sections);
        }
        loadTrendsCohort(picks, function () { swapDock(); });
        swapDock();
    }

    function trendsDockToggleLabel() {
        if (trendsDockOpen) return 'Hide';
        var n = Object.keys(trendsPicksFor(trendsPos)).length;
        return n ? ('Show · ' + n) : 'Show';
    }

    function syncTrendsDockToggle(host) {
        var sticky = host && host.querySelector('.cs-trends-sticky');
        if (!sticky) return;
        sticky.classList.toggle('is-collapsed', !trendsDockOpen);
        var btn = sticky.querySelector('[data-trends-dock]');
        if (!btn) return;
        btn.setAttribute('aria-expanded', trendsDockOpen ? 'true' : 'false');
        btn.textContent = trendsDockToggleLabel();
    }

    function setTrendsDockOpen(on) {
        trendsDockOpen = !!on;
        syncTrendsDockToggle($('csTrends'));
    }

    function bindTrendsDock(host, sections) {
        var dock = host && host.querySelector('.cs-trends-scout');
        if (!dock) return;
        dock.querySelectorAll('[data-trends-pick]').forEach(function (b) {
            b.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                toggleTrendsPick(b.getAttribute('data-trends-pick'), sections);
            });
        });
        var clearPicks = dock.querySelector('[data-trends-clear]');
        if (clearPicks) clearPicks.addEventListener('click', function () {
            trendsPicks[trendsPos] = {};
            paintTrendsSelection(sections);
        });
        dock.querySelectorAll('[data-trends-unlock]').forEach(function (b) {
            b.addEventListener('click', function () {
                if (typeof window.showPaywall === 'function') window.showPaywall('draft-trends-scout');
            });
        });
        dock.querySelectorAll('[data-trends-player]').forEach(function (b) {
            b.addEventListener('click', function () {
                var id = b.getAttribute('data-trends-player');
                var name = b.getAttribute('data-trends-name') || '';
                var pos = b.getAttribute('data-trends-ppos') || trendsPos;
                var adp = b.getAttribute('data-trends-adp') || '';
                var proj = b.getAttribute('data-trends-proj') || '';
                if (id) openHistPanel(id, name, adp, pos, proj, '', '');
            });
        });
    }

    function trendsRowEdge(row, sec, tier) {
        row = row || {};
        var by = row.ranking_edge_by_tier || {};
        if (by[tier] != null && isFinite(Number(by[tier]))) return Number(by[tier]);
        if (tier === 'top_12' && typeof row.ranking_edge === 'number') return row.ranking_edge;
        if (typeof row.adjusted_edge === 'number') return row.adjusted_edge;
        var view = trendsRowView(row, sec, tier || 'top_12');
        return view.vs;
    }

    function trendsTopEdges(sections, limit, tier) {
        var scored = [];
        (sections || []).forEach(function (sec) {
            if (String(sec.polarity || '') === 'miss') return;
            (sec.rows || []).forEach(function (row) {
                var view = trendsRowView(row, sec, tier || 'top_12');
                var edge = trendsRowEdge(row, sec, tier || 'top_12');
                if (typeof edge !== 'number' || edge <= 0) return;
                scored.push({
                    sid: sec.id || '',
                    lane: TRENDS_LANE_OF[sec.id] || '',
                    section: sec.heading || '',
                    label: trendsQualifyLabel(sec.id, row.label || ''),
                    pct: view.pct,
                    vs_baseline: view.vs,
                    ranking_edge: edge,
                    vs_label: view.vsLabel,
                    n: row.n,
                    confidence_label: row.confidence_label
                });
            });
        });
        scored.sort(function (a, b) {
            return (Number(b.ranking_edge || 0) - Number(a.ranking_edge || 0))
                || (Number(b.pct || 0) - Number(a.pct || 0));
        });
        return scored.slice(0, limit || 10);
    }

    function trendsRedFlags(sections, limit, tier) {
        var scored = [];
        (sections || []).forEach(function (sec) {
            if (String(sec.polarity || '') === 'miss') return;
            (sec.rows || []).forEach(function (row) {
                var view = trendsRowView(row, sec, tier || 'top_12');
                var edge = trendsRowEdge(row, sec, tier || 'top_12');
                if (typeof edge !== 'number' || edge >= 0) return;
                scored.push({
                    sid: sec.id || '',
                    label: trendsQualifyLabel(sec.id, row.label || ''),
                    pct: view.pct,
                    ranking_edge: edge,
                    vs_label: view.vsLabel,
                    n: row.n,
                    confidence_label: row.confidence_label
                });
            });
        });
        scored.sort(function (a, b) {
            return Number(a.ranking_edge || 0) - Number(b.ranking_edge || 0);
        });
        return scored.slice(0, limit || 6);
    }

    function trendsSectionRow(row, polarity, baselinePct, span, opts) {
        row = row || {};
        opts = opts || {};
        var view = trendsRowView(row, opts.sec, opts.tier || 'top_12');
        var vs = view.vsLabel;
        var selectable = !!(row.match && row.id);
        var on = selectable && opts.selected && opts.selected[row.id];
        var meta = [];
        if (row.n != null) meta.push(histSampleLabel(row.n));
        if (vs) meta.push(vs);
        if (row.secondary && (opts.tier || 'top_12') === 'top_12') meta.push(row.secondary);
        if (on && row.ci_low != null && row.ci_high != null) {
            meta.push('95% CI ' + row.ci_low + '% to ' + row.ci_high + '%');
        }
        var shown = view.pct != null ? view.pct + '%' : '-';
        var cls = 'cs-trends-srow'
            + (polarity === 'miss' ? ' is-miss' : '')
            + (selectable ? ' is-pick' : '')
            + (on ? ' is-on' : '');
        var tag = selectable ? 'button type="button"' : 'div';
        var close = selectable ? 'button' : 'div';
        var pickAttr = selectable ? ' data-trends-pick="' + esc(row.id) + '"' : '';
        return '<' + tag + ' class="' + cls + '"' + pickAttr + '>'
            + '<div class="cs-trends-srow-top">'
            + trendsConfDot(row.confidence_label)
            + '<span class="cs-trends-srow-label">' + esc(row.label || '') + '</span>'
            + '<span class="cs-trends-srow-pct">' + esc(shown) + '</span></div>'
            + trendsRailHtml(view.pct, polarity === 'miss' ? null : baselinePct, polarity, span)
            + trendsMiniPcts(row, opts.tier || 'top_12')
            + (meta.length ? '<div class="cs-trends-srow-meta">' + esc(meta.join(' · ')) + '</div>' : '')
            + '</' + close + '>';
    }

    function renderTrends(opts) {
        var host = $('csTrends');
        if (!host) return;
        var data = trendsCache;
        if (!data || data.available === false) {
            host.innerHTML = '<p class="cs-hist-sub">Historical trends are not available yet.</p>';
            return;
        }
        var positions = data.positions || ['QB', 'RB', 'WR', 'TE'];
        if (!trendsPos || positions.indexOf(trendsPos) < 0) trendsPos = positions[0] || 'RB';
        var page = (data.by_position || {})[trendsPos] || {};
        var sections = (page.sections || []).filter(function (sec) {
            var id = sec && sec.id;
            return id !== 'adp' && id !== 'adp_positional';
        });
        var finishTiers = data.finish_tiers || page.finish_tiers || ['top_5', 'top_12', 'top_24'];
        if (finishTiers.indexOf(trendsTier) < 0) trendsTier = 'top_12';
        var baseline = trendsPageBaseline(page, trendsTier);
        var baselinePct = baseline.pct;
        var baselineN = baseline.n != null ? baseline.n : page.baseline_n;
        var finishName = trendsFinishLabel(trendsTier);
        var picks = trendsPicksFor(trendsPos);
        var pickCount = Object.keys(picks).length;
        var lane = host.getAttribute('data-trends-lane') || 'all';
        var present = {};
        sections.forEach(function (sec) {
            var id = TRENDS_LANE_OF[sec.id] || 'all';
            present[id] = true;
        });
        if (lane !== 'all' && !present[lane]) {
            lane = 'all';
            host.setAttribute('data-trends-lane', 'all');
        }
        var edges = trendsTopEdges(sections, 10, trendsTier);
        var flags = trendsRedFlags(sections, 6, trendsTier);
        var span = trendsRailSpan(baselinePct, sections, edges.map(function (e) { return e.pct; }).concat(flags.map(function (e) { return e.pct; })));
        var compact = false;
        try { compact = window.matchMedia('(max-width: 720px)').matches; } catch (e) { compact = false; }
        host.className = 'cs-trends cs-trends-' + String(trendsPos || '').toLowerCase();
        var html = '<p class="cs-trends-lede">' + esc(data.note || data.headline || '') + '</p>';
        if (state.mode === 'dynasty') {
            html += '<p class="cs-trends-lede cs-trends-dynasty-note">'
                + '1QB redraft history — research only, not dynasty ranking.</p>';
        }
        html += '<div class="cs-trends-pos" role="group" aria-label="Trends position">';
        positions.forEach(function (pos) {
            html += '<button type="button" data-trends-pos="' + esc(pos) + '" aria-pressed="'
                + String(pos === trendsPos) + '">' + esc(pos) + '</button>';
        });
        html += '</div>';
        html += '<div class="cs-trends-tiers" role="group" aria-label="Finish line">';
        finishTiers.forEach(function (tier) {
            html += '<button type="button" data-trends-tier="' + esc(tier) + '" aria-pressed="'
                + String(tier === trendsTier) + '">' + esc(trendsFinishLabel(tier)) + '</button>';
        });
        var finishCopy = page.finish_tier_copy || data.finish_tier_copy
            || 'Top 5 is the league-winner line. Top 12 is a starter. Top 24 is the next skill-position line.';
        html += '<span class="cs-trends-tier-n">' + esc(finishCopy) + '</span>';
        html += '</div>';
        html += '<div class="cs-trends-summary">';
        html += '<div class="cs-trends-base-pct">'
            + (baselinePct != null ? esc(String(baselinePct)) + '<sup>%</sup>' : '-') + '</div>';
        html += '<div class="cs-trends-base-copy"><div class="cs-trends-base-k">Typical '
            + esc(trendsPos) + ' ' + esc(finishName) + '</div><div class="cs-trends-base-v">';
        var bits = [];
        bits.push(esc(trendsPos) + 's finished ' + esc(finishName) + ' in '
            + (baselinePct != null ? esc(String(baselinePct)) + '%' : 'an unknown share')
            + ' of player-seasons'
            + (baselineN != null ? ' (' + esc(histSampleLabel(baselineN)) + ')' : '') + '.');
        if (page.prime_window) bits.push('Prime window is ages ' + esc(page.prime_window) + '.');
        html += bits.join(' ') + '</div></div></div>';
        if (edges.length) {
            var mid = Math.ceil(edges.length / 2);
            html += '<section class="cs-trends-board"><div class="cs-trends-sec-head">'
                + '<h3>Top 10 edges vs typical</h3>'
                + '<p>Ranked by shrinkage-adjusted lift versus the position ' + esc(finishName)
                + ' baseline, so a tiny sample cannot automatically outrank a large one. The tick is typical; the marker is this bucket.</p>'
                + '</div><div class="cs-trends-callouts">';
            [edges.slice(0, mid), edges.slice(mid)].forEach(function (col, colIdx) {
                html += '<div class="cs-trends-callout-col">';
                col.forEach(function (h, i) {
                    var idx = colIdx === 0 ? i : i + mid;
                    var edgeShow = (typeof h.ranking_edge === 'number') ? h.ranking_edge : h.vs_baseline;
                    var vsShort = (typeof edgeShow === 'number' && edgeShow > 0)
                        ? '+' + edgeShow : (h.vs_label || '');
                    html += '<div class="cs-trends-callout">'
                        + '<div class="cs-trends-callout-copy">'
                        + '<span class="cs-trends-rk">' + (idx + 1) + '</span>'
                        + '<div class="cs-trends-callout-v">' + esc(h.label || '') + '</div></div>'
                        + trendsRailHtml(h.pct, baselinePct, null, span)
                        + '<div class="cs-trends-callout-pct">'
                        + (h.pct != null ? esc(String(h.pct)) + '%' : '-')
                        + (vsShort ? ' <span>' + esc(String(vsShort)) + '</span>' : '') + '</div></div>';
                });
                html += '</div>';
            });
            html += '</div></section>';
        }
        if (flags.length) {
            html += '<section class="cs-trends-board cs-trends-redflags"><div class="cs-trends-sec-head">'
                + '<h3>Historical red flags</h3>'
                + '<p>Strongest negative adjusted edges versus a typical ' + esc(trendsPos)
                + ' player-season. Only patterns present in the tables, not invented heuristics.</p>'
                + '</div><div class="cs-trends-callouts">';
            flags.forEach(function (h, i) {
                html += '<div class="cs-trends-callout is-down">'
                    + '<div class="cs-trends-callout-copy">'
                    + '<span class="cs-trends-rk">' + (i + 1) + '</span>'
                    + '<div class="cs-trends-callout-v">' + esc(h.label || '') + '</div></div>'
                    + trendsRailHtml(h.pct, baselinePct, 'miss', span)
                    + '<div class="cs-trends-callout-pct">'
                    + (h.pct != null ? esc(String(h.pct)) + '%' : '-')
                    + (typeof h.ranking_edge === 'number' ? ' <span>' + esc(String(h.ranking_edge)) + '</span>' : '')
                    + '</div></div>';
            });
            html += '</div></section>';
        }
        var curve = ((page.age_curve_by_tier || {})[trendsTier]) || (trendsTier === 'top_12' ? (page.age_curve || []) : []);
        if (curve.length) {
            var maxP = 1;
            curve.forEach(function (pt) {
                if (pt && pt.pct != null && Number(pt.pct) > maxP) maxP = Number(pt.pct);
            });
            var prime = {};
            (page.prime_ages || []).forEach(function (a) { prime[String(a)] = true; });
            var baseLine = '';
            if (baselinePct != null && isFinite(Number(baselinePct)) && maxP > 0) {
                var baseH = Math.max(0, Math.min(100, (Number(baselinePct) / maxP) * 100));
                baseLine = '<span class="cs-trends-ages-base" style="bottom:' + baseH.toFixed(2)
                    + '%" title="Typical ' + esc(String(baselinePct)) + '%"></span>';
            }
            html += '<section class="cs-trends-agewrap"><div class="cs-trends-sec-head">'
                + '<h3>Age curve</h3>'
                + '<p>' + esc(finishName.charAt(0).toUpperCase() + finishName.slice(1))
                + ' rate by integer age. Highlighted columns are the prime window'
                + (baselinePct != null ? '; the line is typical (' + esc(String(baselinePct)) + '%).' : '.')
                + ' Hover or tap a bar for the sample size.</p></div>';
            html += '<div class="cs-trends-ages">';
            html += '<div class="cs-trends-ages-plot" role="img" aria-label="'
                + esc(finishName) + ' rate by age">' + baseLine;
            curve.forEach(function (pt) {
                var h = Math.max(8, Math.round(100 * Number(pt.pct || 0) / maxP));
                var cls = prime[String(pt.age)] ? ' is-prime' : '';
                var nBit = pt.n != null ? ' · ' + pt.n + ' player-seasons' : '';
                var tip = 'Age ' + pt.age + ': ' + pt.pct + '%' + nBit;
                html += '<button type="button" class="cs-trends-age' + cls + '" data-age-tip="1" aria-label="'
                    + esc(tip) + '"><span class="cs-trends-age-bar" style="height:' + h
                    + '%"></span><span class="cs-trends-age-tip">' + esc('Age ' + pt.age + ' · ' + pt.pct + '%'
                    + (pt.n != null ? ' · ' + histSampleLabel(pt.n) : '')) + '</span></button>';
            });
            html += '</div><div class="cs-trends-ages-axis">';
            curve.forEach(function (pt) {
                html += '<span>' + esc(String(pt.age)) + '</span>';
            });
            html += '</div></div></section>';
        }
        html += '<div class="cs-trends-sticky'
            + (pickCount ? ' is-picked' : '')
            + (trendsDockOpen ? '' : ' is-collapsed') + '">';
        html += '<div class="cs-trends-lanes" role="group" aria-label="Trends lane">';
        TRENDS_LANES.forEach(function (pair) {
            if (pair[0] !== 'all' && !present[pair[0]]) return;
            html += '<button type="button" data-trends-lane="' + pair[0] + '" aria-pressed="'
                + String(pair[0] === lane) + '">' + pair[1] + '</button>';
        });
        var shown = sections.filter(function (sec) {
            return lane === 'all' || TRENDS_LANE_OF[sec.id] === lane;
        }).map(function (sec) {
            if (!sec.finish_tied || trendsTier === 'top_12') return sec;
            var rows = (sec.rows || []).filter(function (row) {
                return row.pcts && row.pcts[trendsTier] != null;
            });
            return rows.length ? { id: sec.id, heading: sec.heading, note: sec.note, polarity: sec.polarity, finish_tied: sec.finish_tied, rows: rows } : null;
        }).filter(Boolean);
        html += '<span class="cs-trends-lane-n">' + shown.length + ' table'
            + (shown.length === 1 ? '' : 's')
            + (compact ? '. Open one, or pick a lane.' : '')
            + ' Tap a bucket to list matching players.</span>';
        html += '<button type="button" class="cs-trends-sticky-toggle" data-trends-dock="1"'
            + ' aria-expanded="' + (trendsDockOpen ? 'true' : 'false') + '"'
            + ' aria-controls="cs-trends-sticky-body">'
            + esc(trendsDockToggleLabel()) + '</button>';
        html += '</div>';
        html += '<div class="cs-trends-sticky-body" id="cs-trends-sticky-body">';
        html += trendsProfileHtml(trendsCohort, picks, pickCount);
        html += trendsScoutHtml(data, picks, pickCount);
        html += '</div></div>';
        html += '<div class="cs-trends-grid">';
        shown.forEach(function (sec) {
            var laneId = TRENDS_LANE_OF[sec.id] || 'all';
            var peekRow = null;
            var peekView = null;
            (sec.rows || []).forEach(function (row) {
                var view = trendsRowView(row, sec, trendsTier);
                if (!peekRow) { peekRow = row; peekView = view; }
                else if ((view.vs || 0) > (peekView.vs || 0)) { peekRow = row; peekView = view; }
            });
            var peek = peekRow && peekView && peekView.pct != null
                ? trendsQualifyLabel(sec.id, peekRow.label || '') + ' ' + peekView.pct + '%'
                : ((sec.rows || []).length + ' buckets');
            html += '<details class="cs-trends-card" data-lane="' + esc(laneId) + '"'
                + (compact ? '' : ' open') + '>'
                + '<summary><h3>' + esc(sec.heading || '') + '</h3>'
                + '<span class="cs-trends-card-peek">' + esc(peek) + '</span></summary>';
            if (sec.note) html += '<p class="cs-hist-note">' + esc(sec.note) + '</p>';
            html += '<div class="cs-trends-card-rows">';
            (sec.rows || []).forEach(function (row) {
                html += trendsSectionRow(row, sec.polarity, baselinePct, span, {
                    sec: sec, tier: trendsTier, selected: picks
                });
            });
            html += '</div></details>';
        });
        html += '</div>';
        host.innerHTML = html;
        host.querySelectorAll('[data-trends-pos]').forEach(function (b) {
            b.addEventListener('click', function () {
                trendsPos = b.getAttribute('data-trends-pos') || 'RB';
                renderTrends();
            });
        });
        host.querySelectorAll('[data-trends-tier]').forEach(function (b) {
            b.addEventListener('click', function () {
                trendsTier = b.getAttribute('data-trends-tier') || 'top_12';
                renderTrends();
            });
        });
        host.querySelectorAll('[data-trends-lane]').forEach(function (b) {
            b.addEventListener('click', function () {
                host.setAttribute('data-trends-lane', b.getAttribute('data-trends-lane') || 'all');
                renderTrends();
            });
        });
        host.querySelectorAll('[data-trends-dock]').forEach(function (b) {
            b.addEventListener('click', function () {
                setTrendsDockOpen(!trendsDockOpen);
            });
        });
        host.querySelectorAll('.cs-trends-srow.is-pick').forEach(function (b) {
            b.addEventListener('click', function (e) {
                e.preventDefault();
                e.stopPropagation();
                toggleTrendsPick(b.getAttribute('data-trends-pick'), sections);
            });
        });
        bindTrendsDock(host, sections);
        if (pickCount) {
            loadTrendsCohort(picks, function () { paintTrendsSelection(sections); });
        } else {
            trendsCohort = null;
        }
        host.querySelectorAll('.cs-trends-card > summary').forEach(function (s) {
            s.addEventListener('click', function (e) {
                var wide = false;
                try { wide = window.matchMedia('(min-width: 721px)').matches; } catch (err) { wide = false; }
                if (wide) e.preventDefault();
            });
        });
        host.querySelectorAll('[data-age-tip]').forEach(function (b) {
            b.addEventListener('click', function () {
                var on = b.classList.contains('is-open');
                host.querySelectorAll('[data-age-tip].is-open').forEach(function (x) {
                    x.classList.remove('is-open');
                });
                if (!on) b.classList.add('is-open');
            });
        });
    }

    function renderHistPanel(resp, fallbackMarket, posHint) {
        if (!resp || resp.available === false) {
            return '<p class="cs-hist-sub">No historical profile for this player yet.</p>';
        }
        applyHistPos(histPosOf(resp, posHint));
        var copy = resp.copy || {};
        var html = '';

        // Verdict first: lead with this player's historical top-12 chance given
        // career and situation, with other tiers as supporting stats below.
        var hits = Array.isArray(copy.hit_rates) ? copy.hit_rates : [];
        var lead = null, i;
        for (i = 0; i < hits.length; i++) {
            if (hits[i] && hits[i].tier === 'top_12') { lead = hits[i]; break; }
        }
        if (!lead && hits.length) lead = hits[0];
        if (lead) {
            var confBits = [];
            if (lead.confidence_label) confBits.push(lead.confidence_label);
            if (lead.n != null) confBits.push(histSampleLabel(lead.n));
            if (lead.ci_low != null && lead.ci_high != null) {
                confBits.push('95% CI ' + lead.ci_low + '% to ' + lead.ci_high + '%');
            }
            html += '<div class="cs-hist-verdict">'
                + '<div class="cs-hist-hero">'
                + '<div class="cs-hist-big">' + (lead.pct != null ? lead.pct : '-') + '<sup>%</sup></div>'
                + '<div class="cs-hist-hero-cap">'
                + '<div class="cs-hist-hero-lead">finished top-12</div>'
                + '<div class="cs-hist-hero-sub">this player\'s historical chance</div>'
                + (confBits.length ? '<span class="cs-hist-conf"><i></i>' + esc(confBits.join(' · ')) + '</span>' : '')
                + '</div></div>';
            html += '<div class="cs-hist-tiers">';
            hits.forEach(function (row) {
                if (!row) return;
                var short = HIST_TIER_SHORT[row.tier] || (row.label || '').replace('Then finished ', '');
                html += '<div class="cs-hist-tier' + (row.tier === 'top_12' ? ' lead' : '') + '">'
                    + '<div class="cs-hist-tier-k">' + esc(short) + '</div>'
                    + '<div class="cs-hist-tier-v">' + (row.pct != null ? row.pct + '%' : '-') + '</div></div>';
            });
            html += '</div>';
            var histPct = copy.history_pct != null ? copy.history_pct : lead.pct;
            var mktPct = copy.market_pct;
            if (histPct != null || mktPct != null) {
                html += '<div class="cs-hist-market">';
                html += '<div class="cs-hist-compare-h">'
                    + esc(copy.market_compare_heading || 'Two groups, not one chance')
                    + '</div>';
                html += '<div class="cs-hist-compare">';
                html += '<div class="cs-hist-compare-col">'
                    + '<div class="cs-hist-compare-k">'
                    + esc(copy.history_group_label || 'Players like this') + '</div>'
                    + '<div class="cs-hist-compare-v">'
                    + (histPct != null ? histPct + '%' : '-') + '</div>'
                    + '<div class="cs-hist-compare-s">'
                    + esc(copy.history_group_hint || 'this career and situation')
                    + '</div></div>';
                html += '<div class="cs-hist-compare-col">'
                    + '<div class="cs-hist-compare-k">'
                    + esc(copy.market_group_label || 'That ADP round') + '</div>'
                    + '<div class="cs-hist-compare-v">'
                    + (mktPct != null ? mktPct + '%' : 'need ADP') + '</div>'
                    + '<div class="cs-hist-compare-s">'
                    + esc(copy.market_group_hint || 'anyone taken in that fantasy round')
                    + '</div></div>';
                html += '</div>';
                var gapNote = copy.gap_note;
                if (!gapNote && mktPct == null) {
                    gapNote = 'Need live ADP to show the other group.';
                }
                if (gapNote) html += '<p class="cs-hist-gap">' + esc(gapNote) + '</p>';
                html += '</div>';
            }
            if (copy.headline) html += '<p class="cs-hist-cohort">' + esc(copy.headline) + '</p>';
            if (copy.sample_prior_note) html += '<p class="cs-hist-note">' + esc(copy.sample_prior_note) + '</p>';
            html += '</div>';
        }
        var examples = (resp.history && Array.isArray(resp.history.closest_examples) && resp.history.closest_examples.length)
            ? resp.history.closest_examples
            : ((resp.history && Array.isArray(resp.history.examples)) ? resp.history.examples : []);
        if (examples.length) {
            var sum = copy.examples_summary || (resp.history && resp.history.closest_summary) || {};
            html += '<details class="cs-hist-sec cs-hist-closest"><summary><h3>'
                + esc(copy.examples_heading || 'Closest historical examples') + '</h3>'
                + (sum.label ? '<span class="cs-hist-ex-peek">' + esc(sum.label) + '</span>' : '')
                + '</summary><div class="cs-hist-closest-body">';
            if (copy.examples_note) html += '<p class="cs-hist-note">' + esc(copy.examples_note) + '</p>';
            if (copy.examples_vs_cohort_note) {
                html += '<p class="cs-hist-note">' + esc(copy.examples_vs_cohort_note) + '</p>';
            }
            if (sum.label) html += '<p class="cs-hist-ex-sum">' + esc(sum.label) + '</p>';
            html += '<ul class="cs-hist-ex">';
            examples.forEach(function (ex) {
                if (!ex) return;
                var left = esc(ex.name || ex.sleeper_id || '') + (ex.season ? ' · ' + esc(String(ex.season)) : '');
                var right = [];
                if (ex.adp != null && isFinite(Number(ex.adp))) right.push('ADP ' + Number(ex.adp).toFixed(1));
                if (ex.positional_finish != null) right.push('#' + ex.positional_finish);
                if (ex.ppr_points != null) right.push(ex.ppr_points + ' pts');
                var hit = histExampleHit(ex.positional_finish, ex);
                var traits = Array.isArray(ex.traits)
                    ? ex.traits.filter(Boolean).map(function (t) { return esc(String(t)); }).join(' · ')
                    : '';
                html += '<li' + (hit && hit.tier ? ' class="is-' + esc(hit.tier) + '"' : '') + '><span>' + left
                    + (traits ? '<small>' + traits + '</small>' : '')
                    + '</span><span class="cs-hist-ex-right">'
                    + (right.length ? '<span class="cs-hist-ex-meta">' + esc(right.join(' · ')) + '</span>' : '')
                    + (hit ? '<b class="cs-hist-ex-hit">' + esc(hit.label) + '</b>' : '')
                    + '</span></li>';
            });
            html += '</ul></div></details>';
        }
        var trends = (Array.isArray(copy.trends) ? copy.trends : []).filter(function (row) {
            var kind = row && row.kind;
            return kind !== 'adp' && kind !== 'adp_positional';
        });
        if (trends.length) {
            var histBaseline = null;
            trends.forEach(function (row) {
                var inferred = trendsBaselineOf(row);
                if (histBaseline == null && inferred != null) histBaseline = inferred;
            });
            var histSpan = trendsRailSpan(histBaseline, [], trends.map(function (row) {
                return row && row.pct;
            }));
            html += '<section class="cs-hist-sec"><h3>' + esc(copy.trends_heading || 'Trends for this player\'s buckets') + '</h3>';
            if (copy.trends_note) html += '<p class="cs-hist-note">' + esc(copy.trends_note) + '</p>';
            var groups = (Array.isArray(copy.trend_groups) && copy.trend_groups.length)
                ? copy.trend_groups
                : [{ id: 'all', heading: '', rows: trends }];
            groups.forEach(function (sec) {
                if (!sec || !sec.rows || !sec.rows.length) return;
                var markCells = sec.rows.some(function (r) { return r && r.role === 'analog'; });
                html += '<div class="cs-hist-sec">';
                if (sec.heading) html += '<h3>' + esc(sec.heading) + '</h3>';
                html += '<div class="cs-hist-hits">';
                sec.rows.forEach(function (row) {
                    html += trendsHitRow(row, row && row.polarity, histBaseline, histSpan, markCells);
                });
                html += '</div></div>';
            });
            html += '</section>';
        }

        // Progressive disclosure: dropped filters and the full profile.
        var detail = '';
        if (copy.relaxed && copy.relaxed.length) {
            detail += '<div class="cs-hist-sec"><h3>' + esc(copy.relaxed_heading || 'Dropped to grow the sample') + '</h3>';
            if (copy.relaxed_note) detail += '<p class="cs-hist-note">' + esc(copy.relaxed_note) + '</p>';
            detail += '<p class="cs-hist-note">' + copy.relaxed.map(function (row) { return esc((row && row.label) || ''); }).join(' · ') + '</p></div>';
        }
        var profile = copy.profile || [];
        if (profile.length) {
            detail += '<div class="cs-hist-sec"><h3>' + esc(copy.profile_heading || 'This pre-season profile') + '</h3><div class="cs-hist-profile">';
            profile.forEach(function (row) {
                if (!row) return;
                detail += '<span class="cs-hist-chip"><span class="cs-hist-chip-k">' + esc(row.label || '') + '</span><span class="cs-hist-chip-v">' + esc(row.value || '') + '</span></span>';
            });
            detail += '</div></div>';
        }
        if (detail) {
            html += '<details class="cs-hist-more"><summary>Profile &amp; detail</summary><div class="cs-hist-more-inner">' + detail + '</div></details>';
        }

        return html || '<p class="cs-hist-sub">No historical profile for this player yet.</p>';
    }

    function init() {
        var back = $('csBack');
        if (back && cfg.draftUrl) back.href = cfg.draftUrl;
        // A Draft Room mock/live board can pass its current snapshot. drafted = ids
        // to cross off; mode / sf = that draft's format.
        try {
            var qp = new URLSearchParams(location.search);
            var qMode = qp.get('mode');
            if (qMode === 'redraft' || qMode === 'dynasty') state.mode = qMode;
            var qSf = qp.get('sf');
            if (qSf === '1' || qSf === '0') state.sf = qSf === '1';
            // Scoring from Draft Room URL snapshot (same keys as setup: ppr/tep/passTd).
            var qPpr = qp.get('ppr');
            var qTep = qp.get('tep');
            var qPassTd = qp.get('passTd') || qp.get('pass_td');
            if (qPpr != null || qTep != null || qPassTd != null) {
                state.scoring = normalizeScoring({
                    ppr: qPpr != null ? qPpr : scoringCfg().ppr,
                    tep: qTep != null ? qTep : scoringCfg().tep,
                    passTd: qPassTd != null ? qPassTd : scoringCfg().passTd
                });
            }
            var qDrafted = qp.get('drafted');
            if (qDrafted) {
                var draftedList = qDrafted.split(',').map(function (s) {
                    return s.trim();
                }).filter(Boolean);
                draftedIds = new Set(draftedList);
                scrollToFirstAvailable = draftedIds.size > 0;
            }
            var qRecommendations = qp.get('rec_order');
            if (qRecommendations) {
                recommendationOrder = {};
                qRecommendations.split(',').map(function (s) {
                    return s.trim();
                }).filter(Boolean)
                    .forEach(function (id, i) {
                        if (recommendationOrder[id] == null) recommendationOrder[id] = i;
                    });
            }
            var qTeams = parseInt(qp.get('teams'), 10);
            if (qTeams >= 2 && qTeams <= 32) teams = qTeams;
            var qSlot = parseInt(qp.get('slot'), 10);
            if (qSlot >= 1 && qSlot <= teams) state.pickSlot = qSlot;
        } catch (e) { /* no URL state */
        }
        syncScoringUi();
        if (!state.pickSlot) {
            try {
                var storedSlot = parseInt(localStorage.getItem(slotStoreKey()), 10);
                if (storedSlot >= 1 && storedSlot <= teams) state.pickSlot = storedSlot;
            } catch (e) { /* storage blocked */
            }
        }
        // Mode switch changes the scoring axis (redraft <-> dynasty), so a source
        // that's only valid on the old axis (e.g. Yahoo, redraft-only) must not carry
        // over. Reset to the default source and refetch cleanly for the new axis.
        document.querySelectorAll('#csMode button').forEach(function (b) {
            b.addEventListener('click', function () {
                recommendationOrder = null;
                document.querySelectorAll('#csMode button').forEach(function (x) {
                    x.setAttribute('aria-pressed', String(x === b));
                });
                state.mode = b.getAttribute('data-mode');
                state.adpSource = 'auto';
                resetBoardSort();
                renderAdpSources();
                compute();
                render();
            });
        });
        wireSeg('csQb', function (b) {
            recommendationOrder = null;
            state.sf = b.getAttribute('data-qb') === 'SF';
            resetBoardSort();
        });

        // Scoring selects: same three Draft Room Format controls. Changing them
        // refetches the projection-aware player pool and rebuilds TE targets.
        function onScoringChange() {
            recommendationOrder = null;
            state.scoring = readScoringFromUi();
            resetBoardSort();
            loadPlayers();
        }
        ['csPpr', 'csTep', 'csPassTd'].forEach(function (id) {
            var el = $(id);
            if (el) el.addEventListener('change', onScoringChange);
        });

        $('csValBtn').addEventListener('click', function () {
            state.filter = !state.filter;
            this.setAttribute('aria-pressed', String(state.filter));
            document.querySelectorAll('.cs-board').forEach(function (b) {
                b.classList.toggle('filteron', state.filter);
            });
        });
        var hd = $('csHideDrafted');
        if (hd) hd.addEventListener('click', function () {
            state.hideDrafted = !state.hideDrafted;
            this.setAttribute('aria-pressed', String(state.hideDrafted));
            document.querySelectorAll('.cs-board').forEach(function (b) {
                b.classList.toggle('hidedrafted', state.hideDrafted);
            });
        });
        var nb = $('csNeedsBtn');
        if (nb) nb.addEventListener('click', function () {
            state.needsFilter = !state.needsFilter;
            this.setAttribute('aria-pressed', String(state.needsFilter));
            document.querySelectorAll('.cs-board').forEach(function (b) {
                b.classList.toggle('needson', state.needsFilter);
            });
        });
        var clearBtn = $('csClearBtn');
        if (clearBtn) clearBtn.addEventListener('click', function () {
            if (!state.done.size) return;
            // Live/mock drafted players are authoritative draft state; this button only
            // clears the viewer's hand marks. Hide drafted controls draft visibility.
            state.done.clear();
            render();
        });
        var connectLiveBtn = $('csConnectLive');
        if (connectLiveBtn) connectLiveBtn.addEventListener('click', function () {
            if (liveDraftId) {
                disconnectLiveDraft();
                render();
                return;
            }
            connectLiveBtn.disabled = true;
            connectLiveBtn.textContent = 'Connecting…';
            detectLiveDraft().then(function (connected) {
                connectLiveBtn.disabled = false;
                if (!connected) connectLiveBtn.textContent = 'No current draft found';
                else render();
            });
        });
        // Custom board (pro): toggle edit mode, reset the whole board, and the
        // per-row bump / pin / mute controls (captured so they never cross a row off).
        var editBtn = $('csEditBtn');
        if (editBtn) editBtn.addEventListener('click', function () {
            if (!cfg.hasPremium) {
                if (typeof window.showPaywall === 'function') window.showPaywall('draft-cheat-sheet');
                return;
            }
            editBoard = !editBoard;
            if (editBoard) resetBoardSort();
            render();
        });
        var resetBoardBtn = $('csResetBoardBtn');
        if (resetBoardBtn) resetBoardBtn.addEventListener('click', function () {
            disconnectLiveDraft();
            state.done.clear();
            state.hideDrafted = false;
            state.needsFilter = false;
            if (hasOverrides()) boardReset(); else render();
        });
        var boardPanel = $('cs-panel-board');
        if (boardPanel) boardPanel.addEventListener('click', function (e) {
            var b = e.target.closest('.cs-ovbtn');
            if (!b) return;
            e.stopPropagation();
            e.preventDefault();
            var id = b.getAttribute('data-id'), act = b.getAttribute('data-act');
            if (act === 'up') boardNudge(id, 1);
            else if (act === 'down') boardNudge(id, -1);
            else if (act === 'pin') boardPin(id);
            else if (act === 'mute') boardMute(id);
            else if (act === 'revert') boardRevert(id);
            // 'drag' is handled by the pointer-drag reorder below.
        }, true);   // capture: run before the document row-click (cross-off) handler
        if (boardPanel) setupDragReorder(boardPanel);
        if (boardPanel) boardPanel.addEventListener('click', function (e) {
            var th = e.target.closest('thead th[data-sort]');
            if (!th) return;
            e.preventDefault();
            setSort(th.getAttribute('data-sort'));
            render();
        });

        // CSV export is free — it dumps the currently visible board order.
        var csvBtn = $('csCsvBtn');
        if (csvBtn) csvBtn.addEventListener('click', function () {
            exportCsv();
        });
        $('csPrintBtn').addEventListener('click', function () {
            window.print();
        });
        var srcSel = $('csAdpSrc');
        if (srcSel) srcSel.addEventListener('change', function () {
            state.adpSource = this.value;
            compute();
            render();
        });
        var slotSel = $('csPickSlot');
        if (slotSel) slotSel.addEventListener('change', function () {
            var v = parseInt(this.value, 10);
            state.pickSlot = (v >= 1 && v <= teams) ? v : 0;
            savePickSlot();
            render();
        });

        var searchEl = $('csSearch');
        if (searchEl) searchEl.addEventListener('input', function () {
            state.search = this.value.toLowerCase().trim();
            render();
        });
        document.querySelectorAll('#csPosF button').forEach(function (b) {
            b.addEventListener('click', function () {
                document.querySelectorAll('#csPosF button').forEach(function (x) {
                    x.setAttribute('aria-pressed', String(x === b));
                });
                state.posFilter = b.getAttribute('data-pos');
                render();
            });
        });

        document.addEventListener('click', function (e) {
            var histBtn = e.target.closest('.cs-hist-btn');
            if (histBtn) {
                e.preventDefault();
                e.stopPropagation();
                openHistPanel(
                    histBtn.getAttribute('data-hist-id'),
                    histBtn.getAttribute('data-hist-name'),
                    histBtn.getAttribute('data-hist-adp'),
                    histBtn.getAttribute('data-hist-pos'),
                    histBtn.getAttribute('data-hist-proj'),
                    histBtn.getAttribute('data-hist-proj-rk'),
                    histBtn.getAttribute('data-hist-adp-rk')
                );
                return;
            }
            var el = e.target.closest('[data-name]');
            if (!el || !e.target.closest('#cs-panel-board, #cs-panel-pos')) return;
            var playerId = el.getAttribute('data-id');
            if (!playerId) return;
            if (state.done.has(playerId)) state.done.delete(playerId); else state.done.add(playerId);
            // Re-render so the By Position tier "N left" counts reflect the change
            // (a crossed-off player drops out of the count). Preserve scroll position.
            var sc = document.querySelector('#cs-panel-board:not(.cs-hidden) .cs-tbl-scroll, #cs-panel-pos:not(.cs-hidden) .cs-pgrid-scroll');
            var top = sc ? sc.scrollTop : 0;
            render();
            var sc2 = document.querySelector('#cs-panel-board:not(.cs-hidden) .cs-tbl-scroll, #cs-panel-pos:not(.cs-hidden) .cs-pgrid-scroll');
            if (sc2) sc2.scrollTop = top;
        });

        var tabs = document.querySelectorAll('.cs-tabs [role=tab]');
        tabs.forEach(function (t) {
            t.addEventListener('click', function () {
                var tab = t.getAttribute('data-tab');
                if (tab === 'trends' && !showHist(state.mode === 'dynasty')) return;
                showSheetTab(tab);
            });
        });
        syncHistSurfaces();

        document.querySelectorAll('#csMode button').forEach(function (b) {
            b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === state.mode));
        });
        document.querySelectorAll('#csQb button').forEach(function (b) {
            b.setAttribute('aria-pressed', String((b.getAttribute('data-qb') === 'SF') === state.sf));
        });

        // Re-sync the live draft immediately when the tab regains focus (the poll
        // backs off to 10s while hidden, so this avoids a stale board on return).
        document.addEventListener('visibilitychange', function () {
            if (!document.hidden && liveDraftId) {
                if (pollTimer) clearTimeout(pollTimer);
                pollDraft();
            }
        });

        // Live sync is intentionally opt-in through "Connect live draft". A sheet
        // opened from an active Draft Room still starts with that board's snapshot
        // and then receives pick updates via postMessage while the overlay is open.
        window.addEventListener('message', function (e) {
            if (e.origin !== window.location.origin) return;
            applyDraftRoomContext(e.data);
        });
        loadPlayers();
        if (window.initCustomSelects) window.initCustomSelects(document.querySelector('.cs-wrap') || document);
        var histClose = $('csHistClose');
        if (histClose) histClose.addEventListener('click', closeHistPanel);
        var histModal = $('csHistModal');
        if (histModal) histModal.addEventListener('click', function (e) {
            if (e.target === histModal) closeHistPanel();
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') closeHistPanel();
        });
        try {
            if (window.parent && window.parent !== window) {
                window.parent.postMessage({type: 'drCheatReady'}, window.location.origin);
            }
        } catch (e) { /* not embedded */
        }
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
