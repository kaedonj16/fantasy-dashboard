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
  var myCounts = null;         // {QB,RB,WR,TE} drafted by the viewer, or null
  var liveDraftId = null;      // id of the live draft being polled, or null
  var pollTimer = null;
  var maxVor = 1;
  var loading = false;

  function scoringAxisKey() { return state.mode === 'dynasty' ? 'startup' : 'redraft'; }

  function youthWindow(age) {
    if (age == null) return ['', ''];
    if (age <= 24) return ['Ascending', 'win-asc']; if (age <= 27) return ['Prime', 'win-prime'];
    if (age <= 30) return ['Win-now', 'win-now']; return ['Fading', 'win-fade'];
  }

  function compute() {
    var mode = state.mode, sf = state.sf;
    // Value-derived redraft ADP fallback (mirrors the draft room).
    allPlayers.slice().sort(function (a, b) { return C.redraftVal(b, sf) - C.redraftVal(a, sf); })
      .forEach(function (p, i) { p._radp = i + 1; });

    var pool = allPlayers.filter(function (p) {
      return ['QB', 'RB', 'WR', 'TE'].indexOf((p.position || '').toUpperCase()) >= 0 && C.valOf(p, mode, sf) > 0;
    });
    if (!pool.length) { players = []; return; }

    var starters = C.startersFor(cfg.rosterPositions, sf);
    var valFn = function (p) { return C.valOf(p, mode, sf); };
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

    // Rank by VOR — cross-positional value over replacement. A cheat sheet is a
    // value board, not a live-pick recommender. Ranking by the pick-independent
    // Pick Score neutralized its ADP/need terms and let position-relative
    // production float scarce TEs above much higher-value WRs; VOR is the honest
    // cross-position value the board should sort on. The draft room's live Pick
    // Score still owns the on-the-clock recommendation (with your roster + slot).
    var scored = pool.map(function (p) {
      var pos = (p.position || '').toUpperCase();
      var value = C.valOf(p, mode, sf);
      return {
        id: String(p.id), pos: pos, name: p.name || String(p.id),
        age: (p.age != null ? Number(p.age) : null),
        proj: (p.proj_pts != null ? Number(p.proj_pts) : (p.proj_ppg != null ? Number(p.proj_ppg) * 17 : null)),
        adp: C.adpOf(p, mode, sf), vor: Math.round(value - (repl[pos] || 0)),
      };
    });
    scored.sort(function (a, b) { return b.vor - a.vor || ((a.adp || 9999) - (b.adp || 9999)); });
    players = scored.slice(0, LIMIT);

    maxVor = players.length ? Math.max(1, players[0].vor) : 1;
    var pc = {};
    players.forEach(function (x, i) {
      x.rk = i + 1;
      x.value = (x.adp != null) ? Math.round(x.adp - x.rk) : null;
      x.good = state.mode === 'dynasty' ? (x.age != null && x.age <= 24 ? 1 : 0) : (x.value != null && x.value >= 5 ? 1 : 0);
      x.drafted = draftedIds ? draftedIds.has(x.id) : false;
      x.posfull = myCounts ? ((needByPos[x.pos] && needByPos[x.pos].need) <= 0 && (needByPos[x.pos] && needByPos[x.pos].have) > 0) : false;
      pc[x.pos] = (pc[x.pos] || 0) + 1; x.prk = x.pos + pc[x.pos];
    });
    assignTiers();
    var seen = {};
    for (var i = players.length - 1; i >= 0; i--) {
      var k = players[i].pos + '|' + players[i].dtier;
      if (!seen[k]) { seen[k] = 1; players[i].lastInTier = true; }
    }
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
  function winChip(age) { var w = youthWindow(age); return w[0] ? '<span class="cs-winpill ' + w[1] + '">' + w[0] + '</span>' : ''; }
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
    $('csSub').textContent = dyn
      ? 'Ranked by value over replacement on dynasty value, for your league roster. Tiers are cliffs in the value curve. Age and career window replace ADP.'
      : 'Ranked by value over replacement for your league scoring and roster. Tiers are cliffs in the value curve. The Value column flags where the market disagrees.';

    document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); b.classList.toggle('hidedrafted', state.hideDrafted); b.classList.toggle('needson', state.needsFilter); });
    $('csValBtn').textContent = dyn ? 'Ascenders only' : 'Values only';
    var hd = $('csHideDrafted'); if (hd) { hd.style.display = draftedIds ? '' : 'none'; hd.setAttribute('aria-pressed', String(state.hideDrafted)); }
    var nb = $('csNeedsBtn'); if (nb) { nb.style.display = myCounts ? '' : 'none'; nb.setAttribute('aria-pressed', String(state.needsFilter)); }
    // Show a Clear button once the user has hand-marked players as gone, so they
    // can wipe those marks in one tap. Live/mock drafted ids are not touched.
    var cb = $('csClearBtn'); if (cb) cb.style.display = state.done.size ? '' : 'none';
    renderNeedsBar();

    if (!players.length) {
      $('csBoardBody').innerHTML = '<tr><td colspan="6" class="cs-empty">' + (loading ? 'Loading players…' : 'No players for this format yet.') + '</td></tr>';
      $('csLegend').innerHTML = '';
      return;
    }

    var draftedNote = draftedIds ? '<span class="cs-lg"><span class="cs-taken-dot"></span> already drafted</span>' : '';
    $('csLegend').innerHTML = dyn
      ? '<span class="cs-lg"><b>VOR</b> dynasty value over replacement, the ranking</span>'
        + '<span class="cs-lg"><b>Age</b> drives the window</span>'
        + '<span class="cs-lg">' + winChip(23) + ' ascending</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">Dynasty ' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>'
      : '<span class="cs-lg"><b>VOR</b> value over replacement, the ranking</span>'
        + '<span class="cs-lg"><span class="cs-val g">+7</span> above ADP, target it</span>'
        + '<span class="cs-lg"><span class="cs-val b">-4</span> going early, let it fall</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>';

    renderBoard(dyn);
    renderPos(dyn);
    renderDraft(dyn);
  }

  function renderBoard(dyn) {
    var col5 = dyn ? 'Age' : 'ADP', col6 = dyn ? 'Window' : 'Value';
    $('csBoardHead').innerHTML =
      '<tr><th>Rk</th><th class="l">Player</th><th>Pos</th><th>VOR</th><th>' + col5 + '</th><th>' + col6 + '</th></tr>';
    var lastT = 0, html = '', shown = 0;
    players.forEach(function (x) {
      if (!visiblePlayer(x)) return;
      if (x.dtier !== lastT) { lastT = x.dtier; html += '<tr class="cs-cliff"><td colspan="6"><div class="cs-cliffline">Tier ' + x.dtier + '</div></td></tr>'; }
      shown++;
      var cls = 'cs-p' + (state.done.has(x.name) ? ' done' : '') + (x.drafted ? ' drafted' : '');
      var c5 = dyn ? '<td class="cs-num">' + (x.age != null ? x.age : '') + '</td>' : '<td class="cs-num">' + (x.adp != null ? Math.round(x.adp) : '') + '</td>';
      var c6 = dyn ? '<td>' + winChip(x.age) + '</td>' : '<td>' + valChip(x.value) + '</td>';
      html += '<tr class="' + cls + '" data-good="' + x.good + '" data-posfull="' + (x.posfull ? 1 : 0) + '" data-name="' + esc(x.name) + '">'
        + '<td class="cs-rk">' + x.rk + '</td>'
        + '<td><span class="cs-pcell">' + badge(x.pos) + '<span class="cs-pname">' + esc(x.name) + '</span></span></td>'
        + '<td>' + posrk(x) + '</td>'
        + '<td><span class="cs-vorwrap"><span class="cs-num">' + x.vor + '</span><span class="cs-vorbar"><i style="width:' + Math.max(0, Math.round(x.vor / maxVor * 100)) + '%"></i></span></span></td>'
        + c5 + c6 + '</tr>';
    });
    if (!shown) html = '<tr><td colspan="6" class="cs-empty">No players match this filter.</td></tr>';
    $('csBoardBody').innerHTML = html;
    $('csBoardFoot').textContent = dyn
      ? 'Ranked by value over replacement (dynasty value), youth-aware via the Window column. Tap a row to cross a player off.'
      : 'Ranked by value over replacement, so a scarce elite TE or QB can still outrank a higher-scoring skill player. Tap a row to cross a player off.';
  }

  function renderPos(dyn) {
    var POS = ['RB', 'WR', 'QB', 'TE'];
    var BAND = Math.max(1, maxVor * 0.045), CAP = 6;
    var groups = [], cur = null;
    players.forEach(function (x) {
      if (!visiblePlayer(x)) return;
      var tierChanged = !cur || x.dtier !== cur.tier;
      if (!cur || tierChanged || x.vor < cur.lead - BAND || cur.items.length >= CAP) {
        cur = { tier: x.dtier, lead: x.vor, items: [], tierBreak: tierChanged };
        groups.push(cur);
      }
      cur.items.push(x);
    });
    function nameChip(x) {
      var cls = 'cs-pgc cs-c-' + x.pos + (state.done.has(x.name) ? ' done' : '') + (x.drafted ? ' drafted' : '');
      var tail = dyn ? '' : smallVal(x.value);
      return '<span class="' + cls + '" data-good="' + x.good + '" data-posfull="' + (x.posfull ? 1 : 0) + '" data-name="' + esc(x.name) + '"><span class="cs-pgn">' + esc(x.name) + tail + '</span></span>';
    }
    var out = '<div class="cs-pgrid-head">' + POS.map(function (p) { return '<div>' + p + '</div>'; }).join('') + '</div>';
    var ri = 0;
    groups.forEach(function (g) {
      if (g.tierBreak) {
        var counts = POS.map(function (pos) { var n = players.filter(function (y) { return y.dtier === g.tier && y.pos === pos && !state.done.has(y.name) && !y.drafted; }).length; return n ? pos + ' ' + n : null; }).filter(Boolean).join(' &middot; ');
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

  function renderDraft(dyn) {
    var live = players.filter(function (x) { return !x.drafted && visiblePlayer(x); });
    var dh = live.map(function (x) {
      var run = x.lastInTier;
      // Overall rank, name, then a clean meta cluster: position rank, tier, window.
      return '<div class="cs-drow' + (run ? ' run' : '') + '">'
        + '<span class="cs-drk">' + x.rk + '</span>'
        + '<span class="cs-dname">' + esc(x.name) + '</span>'
        + (run ? '<span class="cs-runflag">last ' + x.pos + '</span>' : '')
        + '<span class="cs-dmeta">'
        + posrk(x)
        + '<span class="cs-dtier">Tier ' + x.dtier + '</span>'
        + (dyn ? winChip(x.age) : '')
        + '</span>'
        + '</div>';
    }).join('');
    $('csDboard').innerHTML = dh || '<div class="cs-empty" style="padding:22px;">Board is empty.</div>';
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
    loading = true;
    var params = [];
    if (state.adpSource && state.adpSource !== 'auto') {
      params.push('adp_source=' + encodeURIComponent(state.adpSource));
      params = params.concat(leagueParams());
    }
    var url = '/api/league-players' + (params.length ? ('?' + params.join('&')) : '');
    return fetch(url, { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        if (!Array.isArray(resp)) {
          if (resp.tier_thresholds) tierThresholds = resp.tier_thresholds;
          if (resp.adp_source_options) adpSourceOptions = resp.adp_source_options;
        }
        allPlayers = raw.filter(function (p) { return p && p.id != null && ['QB', 'RB', 'WR', 'TE'].indexOf(String(p.position || '').toUpperCase()) >= 0; });
        loading = false;
        renderAdpSources();
        compute(); render();
      })
      .catch(function () {
        loading = false;
        $('csBoardBody').innerHTML = '<tr><td colspan="6" class="cs-empty">Could not load players. Refresh to retry.</td></tr>';
      });
  }

  // ── live-draft cross-off ────────────────────────────────────────────────────
  function detectLiveDraft() {
    // Live Sleeper draft sync (auto cross-off + real-time board) is a pro feature.
    // Non-premium users keep the free static board (and any static mock snapshot).
    if (!cfg.hasPremium) return;
    if (!cfg.leagueId || !cfg.platform) return;
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (!resp || resp.unsupported) return;
        var all = resp.drafts || [];
        // Prefer a live (drafting) draft, then a pre-draft one about to start, then
        // the most recent completed one (still tells us who is gone).
        var pick = all.filter(function (d) { return String(d.status) === 'drafting'; })[0]
          || all.filter(function (d) { return String(d.status) === 'pre_draft'; })[0]
          || all[0];
        if (!pick || !pick.draft_id) return;
        liveDraftId = pick.draft_id;
        pollDraft();   // start the live loop
      })
      .catch(function () { /* no live draft; sheet stays static */ });
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
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(liveDraftId), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        applyLiveDraft(d);
        var status = d && String(d.status || '');
        if (status === 'complete') { liveDraftId = null; return; }   // final state applied; stop
        schedulePoll(status === 'drafting' ? 5000 : 12000);          // slower before it starts
      })
      .catch(function () { schedulePoll(10000); });
  }
  function applyLiveDraft(d) {
    var picks = (d && d.picks) || [];
    if (!picks.length) return;   // pre-draft: nothing to cross off yet
    var s = new Set();
    var mine = { QB: 0, RB: 0, WR: 0, TE: 0 };
    var haveMine = false;
    picks.forEach(function (pk) {
      if (!pk || !pk.player_id) return;
      s.add(String(pk.player_id));
      if (cfg.viewerUserId && String(pk.picked_by || '') === String(cfg.viewerUserId)) {
        var pos = String(pk.position || '').toUpperCase();
        if (mine[pos] != null) { mine[pos]++; haveMine = true; }
      }
    });
    draftedIds = s;
    if (haveMine) myCounts = mine;
    compute(); render();
  }

  // ── CSV export ──────────────────────────────────────────────────────────────
  function exportCsv() {
    if (!players.length) return;
    var dyn = state.mode === 'dynasty';
    var head = ['Rank', 'Player', 'Pos', 'PosRank', 'VOR', (dyn ? 'Age' : 'ADP'), (dyn ? 'Window' : 'Value'), 'Tier'];
    var rows = players.map(function (x) {
      var c5 = dyn ? (x.age != null ? x.age : '') : (x.adp != null ? Math.round(x.adp) : '');
      var c6 = dyn ? youthWindow(x.age)[0] : (x.value != null ? (x.value > 0 ? '+' + x.value : x.value) : '');
      return [x.rk, x.name, x.pos, x.prk, x.vor, c5, c6, x.dtier];
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

  var _embeddedMock = false;   // opened from the Draft Room with a mock's state
  function init() {
    var back = $('csBack'); if (back && cfg.draftUrl) back.href = cfg.draftUrl;
    // Draft Room overlay can pass a local (mock/manual) draft's state so the sheet
    // reflects THAT draft (no live feed exists for a mock). drafted = ids to cross
    // off; mode / sf = the draft's format.
    try {
      var qp = new URLSearchParams(location.search);
      var qMode = qp.get('mode'); if (qMode === 'redraft' || qMode === 'dynasty') state.mode = qMode;
      var qSf = qp.get('sf'); if (qSf === '1' || qSf === '0') state.sf = qSf === '1';
      var qDrafted = qp.get('drafted');
      if (qDrafted) {
        draftedIds = new Set(qDrafted.split(',').map(function (s) { return s.trim(); }).filter(Boolean));
        // A mock has no live feed, so freeze on this snapshot. A live draft passes
        // live=1: seed the board now, but let live detection keep it current.
        if (qp.get('live') !== '1') _embeddedMock = true;
      }
    } catch (e) { /* no URL state */ }
    // Mode switch changes the scoring axis (redraft <-> dynasty), so a source
    // that's only valid on the old axis (e.g. Yahoo, redraft-only) must not carry
    // over. Reset to the default source and refetch cleanly for the new axis.
    document.querySelectorAll('#csMode button').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('#csMode button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        state.mode = b.getAttribute('data-mode');
        if (state.adpSource !== 'auto') { state.adpSource = 'auto'; loadPlayers(); }
        else { renderAdpSources(); compute(); render(); }
      });
    });
    wireSeg('csQb', function (b) { state.sf = b.getAttribute('data-qb') === 'SF'; });

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
      state.done.clear();
      render();
    });
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
      var nm = el.getAttribute('data-name');
      if (state.done.has(nm)) state.done.delete(nm); else state.done.add(nm);
      // Re-render so the By Position tier "N left" counts reflect the change
      // (a crossed-off player drops out of the count). Preserve scroll position.
      var sc = document.querySelector('#cs-panel-board:not(.cs-hidden) .cs-tbl-scroll, #cs-panel-pos:not(.cs-hidden) .cs-pgrid-scroll');
      var top = sc ? sc.scrollTop : 0;
      render();
      var sc2 = document.querySelector('#cs-panel-board:not(.cs-hidden) .cs-tbl-scroll, #cs-panel-pos:not(.cs-hidden) .cs-pgrid-scroll');
      if (sc2) sc2.scrollTop = top;
    });

    var tabs = document.querySelectorAll('.cs-tabs [role=tab]');
    var panels = { board: 'cs-panel-board', pos: 'cs-panel-pos', draft: 'cs-panel-draft', logic: 'cs-panel-logic' };
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

    // A mock's drafted set was passed in; don't let live-draft detection override it.
    loadPlayers().then(function () { if (!_embeddedMock) detectLiveDraft(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
