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
  var LIMIT = 175;

  // The backend only emits this redraft signal when independent, confidence-
  // weighted market evidence clears its threshold; baseline-only rows stay "—".
  // Authoritative response metadata controls the entire column.  It is updated
  // on every load, so an unavailable signal cannot leave an invisible/stale
  // column (or CSV field) behind and automatically returns after a good refresh.
  var SHOW_MARKET_VS_ADP = false;
  var showMarket = function (dyn) { return !dyn && SHOW_MARKET_VS_ADP; };

  var state = {
    mode: cfg.mode === 'dynasty' ? 'dynasty' : 'redraft',
    sf: !!cfg.isSuperflex,
    filter: false,
    needsFilter: false,
    hideDrafted: false,
    adpSource: 'auto',
    search: '',
    posFilter: 'ALL',
    done: new Set(),
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
  function boardKey() { return state.mode + ':' + (state.sf ? 'sf' : '1qb'); }
  function ovKey() { return 'csboard:' + (cfg.leagueId || 'guest') + ':' + boardKey(); }
  function loadOverrides() {
    overrides = {};
    if (!cfg.hasPremium) return;
    try { overrides = JSON.parse(localStorage.getItem(ovKey()) || '{}') || {}; } catch (e) { overrides = {}; }
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
    fetch('/api/draft-board/overrides?' + p.join('&'), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (ovKey() !== forKey) return;   // switched boards; drop stale response
        var srv = (resp && resp.overrides) || {};
        if (JSON.stringify(srv) !== JSON.stringify(overrides)) {
          overrides = srv;
          try { localStorage.setItem(ovKey(), JSON.stringify(overrides)); } catch (e) { /* ignore */ }
          compute(); render();
        }
      })
      .catch(function () { /* offline: keep the local cache */ });
  }
  function pushOverridesToServer() {
    if (!cfg.hasPremium) return;
    if (_ovPush) clearTimeout(_ovPush);
    var bk = boardKey(), snap = JSON.parse(JSON.stringify(overrides));
    _ovPush = setTimeout(function () {
      fetch('/api/draft-board/overrides', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ platform: cfg.platform, league_id: cfg.leagueId, season: cfg.season, board_key: bk, overrides: snap }),
      }).catch(function () { /* best effort; localStorage still holds it */ });
    }, 600);
  }
  function saveOverrides() {
    Object.keys(overrides).forEach(function (id) { var o = overrides[id]; if (!o || (o.r == null && !o.p && !o.m)) delete overrides[id]; });
    try { localStorage.setItem(ovKey(), JSON.stringify(overrides)); } catch (e) { /* storage full/blocked */ }
    pushOverridesToServer();
  }
  function hasOverrides() { return cfg.hasPremium && Object.keys(overrides).length > 0; }
  // Current display order of the normal (not pinned/muted) bucket.
  function normOrder() { return players.filter(function (p) { return p.bucket === 0; }); }
  // A monotonic placement sequence so applyOverrides can re-anchor moves in the
  // order they were made (keeps chains of moves stable across a model re-rank).
  function nextSeq() { var mx = 0; Object.keys(overrides).forEach(function (id) { var s = overrides[id] && overrides[id].s; if (s > mx) mx = s; }); return mx + 1; }
  // Move `id` between two neighbour players (either may be null at the board ends).
  // We remember the neighbours' ids (a/b) so the move re-anchors to the same
  // players after a value refresh, and keep `r` as an immediate/fallback rank.
  function boardPlaceBetween(id, above, below) {
    var o = overrides[id] || {}; delete o.p; delete o.m;
    if (above) o.a = above.id; else delete o.a;
    if (below) o.b = below.id; else delete o.b;
    o.r = (above && below) ? (above._eff + below._eff) / 2 : (above ? above._eff + 0.5 : (below ? below._eff - 0.5 : 0));
    o.s = nextSeq();
    overrides[id] = o; saveOverrides(); compute();
    // A drop that resolved to the player's own model spot (no net displacement)
    // is a no-op: discard it silently rather than storing a marker-less override.
    var pl = null; for (var i = 0; i < players.length; i++) { if (players[i].id === id) { pl = players[i]; break; } }
    if (pl && pl.moved && !pl.ov) { delete overrides[id]; saveOverrides(); compute(); }
    else { _flashId = id; }
    render();
  }
  // Place player `id` at display index `pos` within the full normal bucket.
  function boardMoveTo(id, pos) {
    var others = normOrder().filter(function (p) { return p.id !== id; });
    pos = Math.max(0, Math.min(others.length, pos));
    boardPlaceBetween(id, pos > 0 ? others[pos - 1] : null, pos < others.length ? others[pos] : null);
  }
  // Arrow nudge: move one row up (dir +1) or down (dir -1) within the normal bucket.
  function boardNudge(id, dir) {
    var norm = normOrder(), idx = -1;
    for (var i = 0; i < norm.length; i++) { if (norm[i].id === id) { idx = i; break; } }
    if (idx < 0) return;                              // pinned/muted: not in this bucket
    if (dir > 0 && idx === 0) return;                 // already at the top
    if (dir < 0 && idx === norm.length - 1) return;   // already at the bottom
    boardMoveTo(id, dir > 0 ? idx - 1 : idx + 1);
  }
  function boardPin(id) {
    var was = overrides[id] && overrides[id].p; var o = overrides[id] || {};
    delete o.r; delete o.a; delete o.b; delete o.s; delete o.m; if (was) delete o.p; else o.p = true;
    overrides[id] = o; _flashId = id; saveOverrides(); compute(); render();
  }
  function boardMute(id) {
    var was = overrides[id] && overrides[id].m; var o = overrides[id] || {};
    delete o.r; delete o.a; delete o.b; delete o.s; delete o.p; if (was) delete o.m; else o.m = true;
    overrides[id] = o; _flashId = id; saveOverrides(); compute(); render();
  }
  // Revert a single player to its model spot (undo one override).
  function boardRevert(id) { if (!overrides[id]) return; delete overrides[id]; _flashId = id; saveOverrides(); compute(); render(); }
  function boardReset() { overrides = {}; saveOverrides(); compute(); render(); }

  // Pointer-based drag reorder (mouse + touch). The grip handle starts a drag; a
  // drop line tracks the insertion point among the normal-bucket rows, and the
  // release writes the player's new fractional rank via boardMoveTo.
  function setupDragReorder(panel) {
    var scroll = panel.querySelector('.cs-tbl-scroll') || panel;
    var dragId = null, startY = 0, didMove = false, line = null;
    function normRows() {
      var byId = {}; players.forEach(function (p) { byId[p.id] = p; });
      return Array.prototype.slice.call(panel.querySelectorAll('tbody tr.cs-p')).filter(function (tr) {
        var p = byId[tr.getAttribute('data-id')];
        return p && p.bucket === 0 && tr.offsetParent !== null;
      });
    }
    // The neighbour players on either side of the drop point (from the visible
    // rows), plus the content-space Y at which to draw the drop line.
    function dropAt(clientY) {
      var byId = {}; players.forEach(function (p) { byId[p.id] = p; });
      var rows = normRows().filter(function (tr) { return tr.getAttribute('data-id') !== dragId; });
      var pos = rows.length;
      for (var i = 0; i < rows.length; i++) {
        var r = rows[i].getBoundingClientRect();
        if (clientY < r.top + r.height / 2) { pos = i; break; }
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
      if (!line) { line = document.createElement('div'); line.className = 'cs-drop-line'; scroll.appendChild(line); }
      line.style.top = y + 'px';
    }
    function cleanup() {
      if (line) { line.remove(); line = null; }
      var dg = panel.querySelector('tr.cs-dragging'); if (dg) dg.classList.remove('cs-dragging');
      dragId = null; didMove = false;
    }
    panel.addEventListener('pointerdown', function (e) {
      var h = e.target.closest('.cs-drag'); if (!h) return;
      e.preventDefault(); e.stopPropagation();
      dragId = h.getAttribute('data-id'); startY = e.clientY; didMove = false;
      try { h.setPointerCapture(e.pointerId); } catch (_) {}
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
    players.forEach(function (p, i) { p._mr = i; byId[p.id] = p; });   // model (VOR) order
    players.forEach(function (p) {
      var o = custom ? overrides[p.id] : null;
      if (o && o.p)             { p.bucket = -1; p.moved = false; p._eff = p._mr; }
      else if (o && o.m)        { p.bucket = 1;  p.moved = false; p._eff = p._mr; }
      else if (o && o.r != null){ p.bucket = 0;  p.moved = true;  p._eff = o.r; }   // provisional
      else                      { p.bucket = 0;  p.moved = false; p._eff = p._mr; }
    });
    if (!custom) {
      players.forEach(function (p) { p.grp = p.dtier; p.grpLabel = 'Tier ' + p.dtier; p.ov = null; p.ovN = 0; });
      return;
    }
    // Re-anchor moved players against their neighbours' *current* effective rank,
    // oldest placement first so a chain of moves resolves consistently.
    players.filter(function (p) { return p.moved; })
      .sort(function (a, b) { return (overrides[a.id].s || 0) - (overrides[b.id].s || 0); })
      .forEach(function (p) {
        var o = overrides[p.id], aP = o.a ? byId[o.a] : null, bP = o.b ? byId[o.b] : null;
        if (aP && bP)  p._eff = (aP._eff + bP._eff) / 2;
        else if (aP)   p._eff = aP._eff + 0.5;
        else if (bP)   p._eff = bP._eff - 0.5;
        // both anchors gone from the pool: keep the stored fallback rank (o.r)
      });
    players.sort(function (a, b) {
      if (a.bucket !== b.bucket) return a.bucket - b.bucket;
      return a._eff - b._eff || a._mr - b._mr;
    });
    // Net-move chip is measured within the normal bucket only, so a player's own
    // pins/mutes don't distort another player's displayed ▲/▼ count.
    var modelNormPos = {}, mnp = 0;
    players.slice().sort(function (a, b) { return a._mr - b._mr; })
      .forEach(function (x) { if (x.bucket === 0) modelNormPos[x.id] = mnp++; });
    var runTier = 1, np = 0;
    players.forEach(function (x, i) {
      x.rk = i + 1;
      if (x.bucket === -1)      { x.grp = -1;  x.grpLabel = 'Pinned'; x.ov = 'pin';  x.ovN = 0; }
      else if (x.bucket === 1)  { x.grp = 1e9; x.grpLabel = 'Muted';  x.ov = 'mute'; x.ovN = 0; }
      else {
        var d = modelNormPos[x.id] - np; np++;
        if (!x.moved) { runTier = x.dtier; x.ov = null; x.ovN = 0; }
        else { x.ov = d > 0 ? 'up' : (d < 0 ? 'down' : null); x.ovN = Math.abs(d); }
        x.grp = runTier; x.grpLabel = 'Tier ' + runTier;
      }
    });
  }

  function scoringAxisKey() { return state.mode === 'dynasty' ? 'startup' : 'redraft'; }

  // Career window by age, position-aware: RBs peak and fade youngest, QBs latest,
  // TEs are late-blooming and durable. Position-agnostic bands mislabeled, e.g.,
  // a 31-year-old QB (still prime) as "Fading" like a 31-year-old RB.
  function ageBands(pos) {
    switch ((pos || '').toUpperCase()) {
      case 'RB': return [23, 26, 28];
      case 'QB': return [26, 31, 35];
      case 'TE': return [25, 28, 31];
      default:   return [24, 27, 30];   // WR
    }
  }
  function youthWindow(age, pos) {
    if (age == null) return ['', ''];
    var b = ageBands(pos);
    if (age <= b[0]) return ['Ascending', 'win-asc']; if (age <= b[1]) return ['Prime', 'win-prime'];
    if (age <= b[2]) return ['Win-now', 'win-now']; return ['Fading', 'win-fade'];
  }

  function compute() {
    // A sheet opened from the Draft Room must mirror that room exactly. Custom
    // pre-draft overrides still apply to the standalone value board, but not on
    // top of a live Recommendation snapshot.
    if (!recommendationOrder) ensureOverrides();
    var mode = state.mode, sf = state.sf;
    // Value-derived redraft ADP fallback (mirrors the draft room).
    allPlayers.slice().sort(function (a, b) { return C.redraftVal(b, sf) - C.redraftVal(a, sf); })
      .forEach(function (p, i) { p._radp = i + 1; });

    var pool = allPlayers.filter(function (p) {
      return ['QB', 'RB', 'WR', 'TE'].indexOf((p.position || '').toUpperCase()) >= 0 && C.valOf(p, mode, sf) > 0;
    });
    if (!pool.length) { players = []; return; }

    var valFn = function (p) { return C.valOf(p, mode, sf); };
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
    var targets = C.posTargets(C.rosterCounts(cfg.rosterPositions, sf), 0);
    var needByPos = {};
    ['QB', 'RB', 'WR', 'TE'].forEach(function (pos) {
      var have = (myCounts && myCounts[pos]) || 0;
      needByPos[pos] = { target: targets[pos] || 0, have: have, need: Math.max(0, (targets[pos] || 0) - have) };
    });
    window.__csNeed = needByPos;

    // VOR remains the stable cheat-sheet order. A Draft Room Recommendation
    // snapshot is supplemental context only; it must never re-sort the board.
    var scored = pool.map(function (p) {
      var pos = (p.position || '').toUpperCase();
      var value = C.valOf(p, mode, sf);
      return {
        id: String(p.id), pos: pos, name: p.name || String(p.id),
        age: (p.age != null ? Number(p.age) : null),
        adp: C.adpOf(p, mode, sf), vor: Math.round(value - (repl[pos] || 0)),
        projectedPpg: p.proj_ppg != null && isFinite(Number(p.proj_ppg)) ? Number(p.proj_ppg) : null,
        marketVsAdp: mode === 'redraft' && p.market_vs_adp != null ? Number(p.market_vs_adp) : null,
        marketExpectedAdp: mode === 'redraft' && p.market_expected_adp != null ? Number(p.market_expected_adp) : null,
        marketConfidence: mode === 'redraft' && p.market_confidence != null ? Number(p.market_confidence) : null,
        marketConfidenceLabel: mode === 'redraft' ? (p.market_confidence_label || null) : null,
        marketBasis: mode === 'redraft' ? (p.market_basis || null) : null,
        scheduleRank: scheduleRanks[String(p.id)] || null,
      };
    });
    scored.sort(function (a, b) { return b.vor - a.vor || ((a.adp || 9999) - (b.adp || 9999)); });
    scored.forEach(function (x, i) { x._mr = i; });
    players = scored.slice(0, LIMIT);

    maxVor = players.length ? Math.max.apply(null, players.map(function (x) { return Math.max(1, x.vor); })) : 1;
    var pc = {};
    var availableRank = 0;
    players.forEach(function (x, i) {
      x.drafted = draftedIds ? draftedIds.has(x.id) : false;
      x.rk = i + 1;
      x.recRank = recommendationOrder && recommendationOrder[x.id] != null
        ? recommendationOrder[x.id] + 1 : null;
      x.value = (x.adp != null && x.rk != null) ? Math.round(x.adp - x.rk) : null;
      x.good = state.mode === 'dynasty' ? (youthWindow(x.age, x.pos)[1] === 'win-asc' ? 1 : 0) : (x.value != null && x.value >= 5 ? 1 : 0);
      x.posfull = myCounts ? ((needByPos[x.pos] && needByPos[x.pos].need) <= 0 && (needByPos[x.pos] && needByPos[x.pos].have) > 0) : false;
      pc[x.pos] = (pc[x.pos] || 0) + 1; x.prk = x.pos + pc[x.pos];
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
    var vals = players.map(function (p) { return p.vor; });   // already sorted desc

    var NUM_TIERS = 12, MIN_SIZE = 5, ELITE_MIN = 3, MAX_SPAN = 220, WINDOW = 10, SIG_MIN = 2.0;

    // Too few players to derive meaningful drops: fall back to fixed VOR bands.
    if (n < NUM_TIERS * 3) {
      var mx = maxVor || 1;
      players.forEach(function (x) {
        var r = x.vor / mx;
        x.dtier = r >= 0.72 ? 1 : r >= 0.50 ? 2 : r >= 0.33 ? 3 : r >= 0.16 ? 4 : 5;
      });
      var remap = {}, nx = 0;
      players.forEach(function (x) { if (!(x.dtier in remap)) { nx++; remap[x.dtier] = nx; } x.dtier = remap[x.dtier]; });
      return;
    }

    // Local significance of each gap: gap size vs the median of nearby gaps.
    var score = [];
    for (var i = 0; i < n - 1; i++) {
      var gap = vals[i] - vals[i + 1];
      var lo = Math.max(0, i - WINDOW), hi = Math.min(n - 1, i + WINDOW);
      var nbrs = [];
      for (var j = lo; j < hi; j++) { if (j !== i) nbrs.push(vals[j] - vals[j + 1]); }
      nbrs.sort(function (a, b) { return a - b; });
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
      var seq = bounds.slice().sort(function (a, b) { return a - b; }); seq.push(n - 1);
      for (var s2 = 0; s2 < seq.length; s2++) {
        var bb = seq[s2], loS = prev + 1, hiS = bb; prev = bb;
        var sp = vals[loS] - vals[hiS];
        if (sp > worstSpan) { worstSpan = sp; worst = [loS, hiS]; }
      }
      var did = false;
      if (worst) {
        var loW = worst[0], hiW = worst[1], bestI = null, bestG = -1;
        for (var jj = loW + MIN_SIZE - 1; jj < hiW - MIN_SIZE + 1; jj++) {
          var g = vals[jj] - vals[jj + 1];
          if (g > bestG) { bestG = g; bestI = jj; }
        }
        if (bestI !== null && valid(bestI)) { bounds.push(bestI); did = true; }
      }
      if (did) continue;

      // 2) Discretionary: the most locally-significant remaining valid drop.
      var cand = [];
      for (var ii = 0; ii < n - 1; ii++) {
        if (bounds.indexOf(ii) < 0 && score[ii] >= SIG_MIN && valid(ii)) cand.push([score[ii], ii]);
      }
      if (!cand.length) break;
      cand.sort(function (a, b) { return b[0] - a[0]; });
      bounds.push(cand[0][1]);
    }

    // Assign contiguous tiers from the sorted boundary indices.
    bounds.sort(function (a, b) { return a - b; });
    var tier = 1, bp = 0;
    for (var t = 0; t < n; t++) {
      players[t].dtier = tier;
      if (bp < bounds.length && t === bounds[bp]) { tier++; bp++; }
    }
  }

  // ── render ──────────────────────────────────────────────────────────────────
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function badge(pos) { return '<span class="cs-pos-badge cs-pos-' + pos + '">' + pos + '</span>'; }
  function posrk(x) { return '<span class="cs-posrk cs-pos-' + x.pos + '">' + x.prk + '</span>'; }
  function valChip(v) { if (v == null) return ''; return v > 0 ? '<span class="cs-val g">+' + v + '</span>' : (v < 0 ? '<span class="cs-val b">' + v + '</span>' : '<span class="cs-val n">even</span>'); }
  function smallVal(v) { if (v == null) return ''; return v > 0 ? '<span class="cs-pgv cs-val g">+' + v + '</span>' : (v < 0 ? '<span class="cs-pgv cs-val b">' + v + '</span>' : ''); }
  function winChip(age, pos) { var w = youthWindow(age, pos); return w[0] ? '<span class="cs-winpill ' + w[1] + '">' + w[0] + '</span>' : ''; }
  function $(id) { return document.getElementById(id); }
  // Search + position filters narrow which players are shown (they don't re-rank).
  function visiblePlayer(x) {
    if (state.posFilter !== 'ALL' && x.pos !== state.posFilter) return false;
    if (state.search && x.name.toLowerCase().indexOf(state.search) < 0) return false;
    return true;
  }

  function render() {
    var dyn = state.mode === 'dynasty';
    $('csTitle').textContent = dyn ? 'Dynasty Cheat Sheet' : 'Redraft Cheat Sheet';
    $('csSub').textContent = recommendationOrder
      ? 'The stable VOR board, with live Draft Room Recommendation ranks shown as supplemental context.'
      : dyn
      ? 'Ranked by value over replacement on dynasty value, for your league roster. Tiers are cliffs in the value curve. Age and career window replace ADP.'
      : 'Ranked by value over replacement for your league scoring and roster. Tiers are cliffs in the value curve. The Value column flags where the market disagrees.';

    document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); b.classList.toggle('hidedrafted', state.hideDrafted); b.classList.toggle('needson', state.needsFilter); });
    $('csValBtn').textContent = dyn ? 'Ascenders only' : 'Values only';
    var hd = $('csHideDrafted'); if (hd) { hd.style.display = draftedIds ? '' : 'none'; hd.setAttribute('aria-pressed', String(state.hideDrafted)); }
    var nb = $('csNeedsBtn'); if (nb) { nb.style.display = myCounts ? '' : 'none'; nb.setAttribute('aria-pressed', String(state.needsFilter)); }
    // Show a Clear button once the user has hand-marked players as gone, so they
    // can wipe those marks in one tap. Live/mock drafted ids are not touched.
    var cb = $('csClearBtn'); if (cb) cb.style.display = state.done.size ? '' : 'none';
    var liveBtn = $('csConnectLive');
    if (liveBtn) {
      liveBtn.style.display = (cfg.hasPremium && cfg.leagueId && cfg.platform) ? '' : 'none';
      liveBtn.textContent = liveDraftId ? 'Disconnect live draft' : 'Connect live draft';
    }
    // Custom board (pro): edit toggle always available; reset only with overrides.
    var eb = $('csEditBtn');
    if (eb) { eb.style.display = (cfg.hasPremium && !recommendationOrder) ? '' : 'none'; eb.setAttribute('aria-pressed', String(editBoard)); eb.textContent = editBoard ? 'Done editing' : 'Edit board'; }
    var rb = $('csResetBoardBtn');
    if (rb) rb.style.display = (hasOverrides() || state.done.size || draftedIds) ? '' : 'none';
    var bp = $('cs-panel-board'); if (bp) bp.classList.toggle('editing', editBoard && cfg.hasPremium);
    renderNeedsBar();

    if (!players.length) {
      $('csBoardBody').innerHTML = '<tr><td colspan="6" class="cs-empty">' + (loading ? 'Loading players…' : (loadError || 'No players for this format yet.')) + '</td></tr>';
      $('csLegend').innerHTML = '';
      return;
    }

    var draftedNote = draftedIds ? '<span class="cs-lg"><span class="cs-taken-dot"></span> already drafted</span>' : '';
    $('csLegend').innerHTML = recommendationOrder
      ? '<span class="cs-lg"><b>VOR</b> controls the cheat-sheet order</span>'
        + '<span class="cs-lg"><b>REC #</b> current Draft Room rank, shown for context</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">' + (dyn ? 'Dynasty ' : '') + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>'
      : dyn
      ? '<span class="cs-lg"><b>VOR</b> dynasty value over replacement, the ranking</span>'
        + '<span class="cs-lg"><b>Age</b> drives the window</span>'
        + '<span class="cs-lg">' + winChip(23) + ' ascending</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">Dynasty ' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>'
      : '<span class="cs-lg"><b>VOR</b> value over replacement, the ranking</span>'
        + '<span class="cs-lg"><span class="cs-val g">+7</span> above ADP, target it</span>'
        + '<span class="cs-lg"><span class="cs-val b">-4</span> going early, let it fall</span>'
        + '<span class="cs-lg"><b>Sched Rk</b> full-season schedule (1 = easiest)</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>';

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
    if (_flashId) setTimeout(function () { _flashId = null; }, 650);
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
    if (x.ov === 'pin')  return '<span class="cs-ovchip pin">pinned</span>';
    if (x.ov === 'mute') return '<span class="cs-ovchip mute">muted</span>';
    if (x.ov === 'up' || x.ov === 'down') return '<span class="cs-ovchip bump">' + (x.ov === 'up' ? '&#9650;' : '&#9660;') + x.ovN + '</span>';
    return '';
  }

  function renderBoard(dyn) {
    var col5 = dyn ? 'Age' : 'ADP', col6 = dyn ? 'Window' : 'Value';
    var editable = cfg.hasPremium;
    var editTh = editable ? '<th class="cs-edit-th"></th>' : '';
    $('csBoardHead').innerHTML =
      '<tr><th>Rk</th><th class="l">Player</th><th>Pos</th><th class="cs-vor-col">VOR</th><th title="Projected fantasy points per game">Proj PPG</th><th>' + col5 + '</th><th class="cs-value-col">' + col6 + '</th><th title="Full fantasy-season strength of schedule rank (1 = easiest)">Sched Rk</th>' + (showMarket(dyn) ? '<th class="cs-market-col">Market vs ADP</th>' : '') + editTh + '</tr>';
    var span = (editable ? 9 : 8) + (showMarket(dyn) ? 1 : 0);
    var lastT = null, html = '', shown = 0;
    players.forEach(function (x) {
      if (!visiblePlayer(x)) return;
      if (!recommendationOrder && x.grp !== lastT) { lastT = x.grp; html += '<tr class="cs-cliff"><td colspan="' + span + '"><div class="cs-cliffline">' + x.grpLabel + '</div></td></tr>'; }
      shown++;
      var cls = 'cs-p' + (state.done.has(x.id) ? ' done' : '') + (x.drafted ? ' drafted' : '') + (x.ov === 'mute' ? ' cs-muted' : '') + (x.ov ? ' cs-ov' : '') + (x.id === _flashId ? ' cs-flash' : '');
      var c5 = dyn ? '<td class="cs-num">' + (x.age != null ? x.age : '') + '</td>' : '<td class="cs-num">' + (x.adp != null ? Math.round(x.adp) : '') + '</td>';
      var c6 = dyn ? '<td class="cs-value-col">' + winChip(x.age, x.pos) + '</td>' : '<td class="cs-value-col">' + valChip(x.value) + '</td>';
      var market = '';
      if (showMarket(dyn)) {
        if (x.marketVsAdp == null) market = '<td class="cs-num cs-market-col" title="Not enough independent market data yet.">&ndash;</td>';
        else {
          var mcls = x.marketVsAdp > 0 ? 'g' : (x.marketVsAdp < 0 ? 'b' : 'n');
          var basisLabel = x.marketBasis === 'season_props' ? 'season-long player markets' : x.marketBasis === 'rolling_market' ? 'multiple recent weekly player markets' : x.marketBasis === 'team_environment' ? 'team betting environment' : 'a blend of available market signals';
          var confLabel = x.marketConfidenceLabel || (x.marketConfidence >= .7 ? 'High' : (x.marketConfidence >= .5 ? 'Moderate' : 'Low'));
          var direction = x.marketVsAdp > 0 ? 'earlier' : (x.marketVsAdp < 0 ? 'later' : 'near its current ADP');
          var mtip = 'Market context implies this player should be drafted ' + (x.marketVsAdp === 0 ? direction : 'about ' + Math.abs(Math.round(x.marketVsAdp)) + ' picks ' + direction) + '. Expected Pick ' + Math.round(x.marketExpectedAdp) + '; current ADP ' + Math.round(x.adp) + '. Confidence: ' + confLabel + ' (' + Math.round((x.marketConfidence || 0) * 100) + '%). Based primarily on ' + basisLabel + '.';
          market = '<td class="cs-market-col"><span class="cs-val ' + mcls + '" title="' + esc(mtip) + '">' + (x.marketVsAdp > 0 ? '+' : '') + Math.round(x.marketVsAdp) + '</span></td>';
        }
      }
      var recChip = x.recRank != null ? '<span class="cs-ovchip bump">REC #' + x.recRank + '</span>' : '';
      html += '<tr class="' + cls + '" data-good="' + x.good + '" data-posfull="' + (x.posfull ? 1 : 0) + '" data-name="' + esc(x.name) + '" data-id="' + esc(x.id) + '">'
        + '<td class="cs-rk">' + (x.rk == null ? '&ndash;' : x.rk) + '</td>'
        + '<td><span class="cs-pcell">' + badge(x.pos) + '<span class="cs-pname">' + esc(x.name) + '</span>' + recChip + ovChip(x) + '</span></td>'
        + '<td>' + posrk(x) + '</td>'
        + '<td class="cs-vor-col"><span class="cs-vorwrap"><span class="cs-num">' + x.vor + '</span><span class="cs-vorbar"><i style="width:' + Math.max(0, Math.round(x.vor / maxVor * 100)) + '%"></i></span></span></td>'
        + '<td class="cs-num">' + (x.projectedPpg != null ? x.projectedPpg.toFixed(1) : '&ndash;') + '</td>'
        + c5 + c6 + '<td class="cs-num" title="Full fantasy-season strength of schedule; 1 is easiest">' + (x.scheduleRank ? '#' + x.scheduleRank : '&ndash;') + '</td>' + market + ovControls(x) + '</tr>';
    });
    if (!shown) html = '<tr><td colspan="' + span + '" class="cs-empty">No players match this filter.</td></tr>';
    $('csBoardBody').innerHTML = html;
    $('csBoardFoot').textContent = recommendationOrder
      ? 'VOR keeps this board stable; REC # shows the live Draft Room opinion without changing the order. Reopen to refresh ranks.'
      : dyn
      ? 'Ranked by value over replacement (dynasty value), youth-aware via the Window column. Tap a row to cross a player off.'
      : 'Ranked by value over replacement, so a scarce elite TE or QB can still outrank a higher-scoring skill player. Tap a row to cross a player off.';
  }

  function renderPos(dyn) {
    var POS = ['RB', 'WR', 'QB', 'TE'];
    var BAND = Math.max(1, maxVor * 0.045), CAP = 6;
    var groups = [], cur = null;
    // By Position stays the model view: iterate in model (VOR) order even when the
    // Big Board has been custom-reordered, so its tier grouping stays contiguous.
    var list = players.slice().sort(function (a, b) { return (a._mr || 0) - (b._mr || 0); });
    list.forEach(function (x) {
      if (!visiblePlayer(x)) return;
      var tierChanged = !cur || x.dtier !== cur.tier;
      if (!cur || tierChanged || x.vor < cur.lead - BAND || cur.items.length >= CAP) {
        cur = { tier: x.dtier, lead: x.vor, items: [], tierBreak: tierChanged };
        groups.push(cur);
      }
      cur.items.push(x);
    });
    function nameChip(x) {
      var cls = 'cs-pgc cs-c-' + x.pos + (state.done.has(x.id) ? ' done' : '') + (x.drafted ? ' drafted' : '');
      var tail = dyn ? '' : smallVal(x.value);
      return '<span class="' + cls + '" data-good="' + x.good + '" data-posfull="' + (x.posfull ? 1 : 0) + '" data-name="' + esc(x.name) + '" data-id="' + esc(x.id) + '"><span class="cs-pgn">' + esc(x.name) + tail + '</span></span>';
    }
    var out = '<div class="cs-pgrid-head">' + POS.map(function (p) { return '<div>' + p + '</div>'; }).join('') + '</div>';
    var ri = 0;
    groups.forEach(function (g) {
      if (g.tierBreak) {
        var counts = POS.map(function (pos) { var n = players.filter(function (y) { return y.dtier === g.tier && y.pos === pos && !state.done.has(y.id) && !y.drafted; }).length; return n ? pos + ' ' + n : null; }).filter(Boolean).join(' &middot; ');
        out += '<div class="cs-pgtier">Tier ' + g.tier + (counts ? '<span class="cs-sc">' + counts + ' left</span>' : '') + '</div>';
      }
      var byPos = { RB: [], WR: [], QB: [], TE: [] };
      g.items.forEach(function (x) { byPos[x.pos].push(x); });
      var alt = (ri % 2) ? ' alt' : ''; ri++;
      var cells = POS.map(function (pos) { return '<div class="cs-pgcell">' + byPos[pos].map(nameChip).join('') + '</div>'; }).join('');
      out += '<div class="cs-pgrow' + alt + '">' + cells + '</div>';
    });
    $('csPosGrid').innerHTML = out;
    $('csPosFoot').textContent = dyn
      ? 'Read down a column for a position board, across a row for who else goes at that slot. Tap a name to cross it off.'
      : 'Read down a column for a position board, across a row for who else goes at that slot. Green is value over ADP. Tap a name to cross it off.';
  }

  function renderNeedsBar() {
    var bar = $('csNeeds'); if (!bar) return;
    if (!myCounts) { bar.style.display = 'none'; bar.innerHTML = ''; return; }
    var need = window.__csNeed || {};
    var chips = ['QB', 'RB', 'WR', 'TE'].map(function (pos) {
      var n = need[pos] || { need: 0, have: 0, target: 0 };
      if (n.need > 0) return '<span class="cs-need cs-need-open">' + pos + ' +' + n.need + '</span>';
      return '<span class="cs-need cs-need-full">' + pos + ' full</span>';
    }).join('');
    bar.style.display = '';
    bar.innerHTML = '<span class="cs-need-lbl">Your roster</span>' + chips
      + '<span class="cs-need-hint">from your live picks</span>';
  }

  // ── ADP source selector ─────────────────────────────────────────────────────
  function renderAdpSources() {
    var sel = $('csAdpSrc'); if (!sel) return;
    var opts = adpSourceOptions[scoringAxisKey()] || [];
    if (!opts.length) { sel.style.display = 'none'; return; }
    sel.style.display = '';
    var cur = state.adpSource === 'auto' ? (opts[0] && opts[0].value) : state.adpSource;
    sel.innerHTML = opts.map(function (o) { return '<option value="' + esc(o.value) + '"' + (o.value === cur ? ' selected' : '') + '>ADP: ' + esc(o.label) + '</option>'; }).join('');
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
    var params = [];
    params.push('league_type=' + (state.sf ? 'sf' : '1qb'));
    if (state.adpSource && state.adpSource !== 'auto') {
      params.push('adp_source=' + encodeURIComponent(state.adpSource));
      params = params.concat(leagueParams());
    }
    var url = '/api/league-players' + (params.length ? ('?' + params.join('&')) : '');
    return fetch(url, { cache: 'no-store', signal: playerAbort ? playerAbort.signal : undefined })
      .then(function (r) {
        if (!r.ok) throw new Error('Players request failed (' + r.status + ')');
        return r.json();
      })
      .then(function (resp) {
        if (requestId !== playerRequest) return;
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        if (!Array.isArray(resp)) {
          if (resp.tier_thresholds) tierThresholds = resp.tier_thresholds;
          if (resp.adp_source_options) adpSourceOptions = resp.adp_source_options;
          SHOW_MARKET_VS_ADP = resp.market_vs_adp_available === true;
        } else {
          SHOW_MARKET_VS_ADP = false;
        }
        allPlayers = raw.filter(function (p) { return p && p.id != null && ['QB', 'RB', 'WR', 'TE'].indexOf(String(p.position || '').toUpperCase()) >= 0; });
        loading = false;
        renderAdpSources();
        compute(); render();
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
        render();
        $('csPosGrid').innerHTML = '<div class="cs-empty">' + loadError + '</div>';
      });
  }

  // Schedule rank is supporting draft context, not an input to the VOR order.
  // Fetch the full fantasy regular season once the player pool is known and
  // merge the API's position-specific SoS rank onto every matching row.
  function loadScheduleRanks(pool) {
    var ids = pool.map(function (p) { return String(p.id); });
    if (!ids.length) return;
    var requestId = ++scheduleRequest;
    var season = Number(cfg.season) || new Date().getFullYear();
    var url = '/api/schedule?season=' + season + '&week_start=1&week_end=17&pids=' + encodeURIComponent(ids.join(','));
    fetch(url, { cache: 'no-store' })
      .then(function (r) { if (!r.ok) throw new Error('Schedule request failed'); return r.json(); })
      .then(function (resp) {
        if (requestId !== scheduleRequest) return;
        scheduleRanks = {};
        (resp.players || []).forEach(function (p) {
          if (p && p.pid != null && p.sos_rank != null) scheduleRanks[String(p.pid)] = Number(p.sos_rank);
        });
        compute(); render();
      })
      .catch(function () { /* Schedule context degrades to an em dash. */ });
  }

  // ── live-draft cross-off ────────────────────────────────────────────────────
  function detectLiveDraft() {
    // Live Sleeper draft sync (auto cross-off + real-time board) is a pro feature.
    // Non-premium users keep the free static board (and any static mock snapshot).
    if (!cfg.hasPremium || !cfg.leagueId || !cfg.platform) return Promise.resolve(false);
    return fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (!resp || resp.unsupported) return;
        var all = resp.drafts || [];
        // Only connect to a current/upcoming draft. Historical drafts should not
        // unexpectedly replace a clean cheat sheet.
        var pick = all.filter(function (d) { return String(d.status) === 'drafting'; })[0]
          || all.filter(function (d) { return String(d.status) === 'pre_draft'; })[0];
        if (!pick || !pick.draft_id) return false;
        liveDraftId = pick.draft_id;
        pollDraft();   // start the live loop
        render();
        return true;
      })
      .catch(function () { return false; });
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
    if (typeof document !== 'undefined' && document.hidden) { schedulePoll(10000); return; }
    var requestedDraftId = liveDraftId;
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(requestedDraftId), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (liveDraftId !== requestedDraftId) return;
        applyLiveDraft(d);
        var status = d && String(d.status || '');
        if (status === 'complete') { liveDraftId = null; return; }   // final state applied; stop
        schedulePoll(status === 'drafting' ? 5000 : 12000);          // slower before it starts
      })
      .catch(function () { if (liveDraftId === requestedDraftId) schedulePoll(10000); });
  }
  function applyLiveDraft(d) {
    var picks = (d && d.picks) || [];
    var s = new Set();
    var mine = { QB: 0, RB: 0, WR: 0, TE: 0 };
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
    compute(); render();
  }

  // ── CSV export ──────────────────────────────────────────────────────────────
  function exportCsv() {
    if (!players.length) return;
    var dyn = state.mode === 'dynasty';
    var head = ['Rank', 'Player', 'Pos', 'PosRank', 'VOR', 'Proj PPG', (dyn ? 'Age' : 'ADP'), (dyn ? 'Window' : 'Value'), 'Schedule Rank'].concat(showMarket(dyn) ? ['Market vs ADP'] : []).concat(['Tier']);
    var rows = players.map(function (x) {
      var c5 = dyn ? (x.age != null ? x.age : '') : (x.adp != null ? Math.round(x.adp) : '');
      var c6 = dyn ? youthWindow(x.age, x.pos)[0] : (x.value != null ? (x.value > 0 ? '+' + x.value : x.value) : '');
      return [x.rk, x.name, x.pos, x.prk, x.vor, x.projectedPpg == null ? '' : x.projectedPpg.toFixed(1), c5, c6, x.scheduleRank || ''].concat(showMarket(dyn) ? [x.marketVsAdp == null ? '' : x.marketVsAdp] : []).concat([x.dtier]);
    });
    var csv = [head].concat(rows).map(function (r) {
      return r.map(function (v) { var s = String(v == null ? '' : v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }).join(',');
    }).join('\n');
    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = (dyn ? 'dynasty' : 'redraft') + '-cheat-sheet-' + (state.sf ? 'sf' : '1qb') + '.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ── interactions ────────────────────────────────────────────────────────────
  function wireSeg(id, apply) {
    document.querySelectorAll('#' + id + ' button').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('#' + id + ' button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        apply(b); renderAdpSources(); compute(); render();
      });
    });
  }

  function init() {
    var back = $('csBack'); if (back && cfg.draftUrl) back.href = cfg.draftUrl;
    // A Draft Room mock/live board can pass its current snapshot. drafted = ids
    // to cross off; mode / sf = that draft's format.
    try {
      var qp = new URLSearchParams(location.search);
      var qMode = qp.get('mode'); if (qMode === 'redraft' || qMode === 'dynasty') state.mode = qMode;
      var qSf = qp.get('sf'); if (qSf === '1' || qSf === '0') state.sf = qSf === '1';
      var qDrafted = qp.get('drafted');
      if (qDrafted) {
        var draftedList = qDrafted.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
        draftedIds = new Set(draftedList);
        scrollToFirstAvailable = draftedIds.size > 0;
      }
      var qRecommendations = qp.get('rec_order');
      if (qRecommendations) {
        recommendationOrder = {};
        qRecommendations.split(',').map(function (s) { return s.trim(); }).filter(Boolean)
          .forEach(function (id, i) { if (recommendationOrder[id] == null) recommendationOrder[id] = i; });
      }
    } catch (e) { /* no URL state */ }
    // Mode switch changes the scoring axis (redraft <-> dynasty), so a source
    // that's only valid on the old axis (e.g. Yahoo, redraft-only) must not carry
    // over. Reset to the default source and refetch cleanly for the new axis.
    document.querySelectorAll('#csMode button').forEach(function (b) {
      b.addEventListener('click', function () {
        recommendationOrder = null;
        document.querySelectorAll('#csMode button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        state.mode = b.getAttribute('data-mode');
        if (state.adpSource !== 'auto') { state.adpSource = 'auto'; loadPlayers(); }
        else { renderAdpSources(); compute(); render(); }
      });
    });
    wireSeg('csQb', function (b) { recommendationOrder = null; state.sf = b.getAttribute('data-qb') === 'SF'; });

    $('csValBtn').addEventListener('click', function () {
      state.filter = !state.filter; this.setAttribute('aria-pressed', String(state.filter));
      document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); });
    });
    var hd = $('csHideDrafted');
    if (hd) hd.addEventListener('click', function () {
      state.hideDrafted = !state.hideDrafted; this.setAttribute('aria-pressed', String(state.hideDrafted));
      document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('hidedrafted', state.hideDrafted); });
    });
    var nb = $('csNeedsBtn');
    if (nb) nb.addEventListener('click', function () {
      state.needsFilter = !state.needsFilter; this.setAttribute('aria-pressed', String(state.needsFilter));
      document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('needson', state.needsFilter); });
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
      if (!cfg.hasPremium) { if (typeof window.showPaywall === 'function') window.showPaywall('draft-cheat-sheet'); return; }
      editBoard = !editBoard; render();
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
      var b = e.target.closest('.cs-ovbtn'); if (!b) return;
      e.stopPropagation(); e.preventDefault();
      var id = b.getAttribute('data-id'), act = b.getAttribute('data-act');
      if (act === 'up') boardNudge(id, 1);
      else if (act === 'down') boardNudge(id, -1);
      else if (act === 'pin') boardPin(id);
      else if (act === 'mute') boardMute(id);
      else if (act === 'revert') boardRevert(id);
      // 'drag' is handled by the pointer-drag reorder below.
    }, true);   // capture: run before the document row-click (cross-off) handler
    if (boardPanel) setupDragReorder(boardPanel);

    // CSV export is a pro feature; non-premium users get the upgrade prompt.
    var csvBtn = $('csCsvBtn');
    if (csvBtn && !cfg.hasPremium) csvBtn.textContent = 'CSV (Pro)';
    if (csvBtn) csvBtn.addEventListener('click', function () {
      if (!cfg.hasPremium) {
        if (typeof window.showPaywall === 'function') window.showPaywall('draft-cheat-sheet');
        return;
      }
      exportCsv();
    });
    $('csPrintBtn').addEventListener('click', function () { window.print(); });
    var srcSel = $('csAdpSrc');
    if (srcSel) srcSel.addEventListener('change', function () { state.adpSource = this.value; loadPlayers(); });

    var searchEl = $('csSearch');
    if (searchEl) searchEl.addEventListener('input', function () { state.search = this.value.toLowerCase().trim(); render(); });
    document.querySelectorAll('#csPosF button').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('#csPosF button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        state.posFilter = b.getAttribute('data-pos'); render();
      });
    });

    document.addEventListener('click', function (e) {
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
    var panels = { board: 'cs-panel-board', pos: 'cs-panel-pos', logic: 'cs-panel-logic' };
    tabs.forEach(function (t) {
      t.addEventListener('click', function () {
        tabs.forEach(function (x) { x.setAttribute('aria-selected', String(x === t)); });
        Object.keys(panels).forEach(function (k) { $(panels[k]).classList.toggle('cs-hidden', k !== t.getAttribute('data-tab')); });
        var onLogic = t.getAttribute('data-tab') === 'logic';
        $('csLegend').style.display = onLogic ? 'none' : '';
        var fb = $('csFilterbar'); if (fb) fb.style.display = onLogic ? 'none' : '';
      });
    });

    document.querySelectorAll('#csMode button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === state.mode)); });
    document.querySelectorAll('#csQb button').forEach(function (b) { b.setAttribute('aria-pressed', String((b.getAttribute('data-qb') === 'SF') === state.sf)); });

    // Re-sync the live draft immediately when the tab regains focus (the poll
    // backs off to 10s while hidden, so this avoids a stale board on return).
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden && liveDraftId) { if (pollTimer) clearTimeout(pollTimer); pollDraft(); }
    });

    // Live sync is intentionally opt-in through "Connect live draft". A sheet
    // opened from an active Draft Room may still start with that board's snapshot.
    loadPlayers();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
