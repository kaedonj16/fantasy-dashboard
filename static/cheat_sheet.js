// Draft Cheat Sheet — a printable, pre-draft view of the SAME board the Draft
// Room ranks on. It fetches the shared /api/league-players pool and computes
// value-over-replacement with BRPickScore.starterCounts (the exact roster-derived
// replacement index the draft room and the server pick-score use), so a player's
// cheat-sheet rank can never contradict their in-draft Pick Score. The draft room
// simply layers the live pick slot, roster need and ADP-timing terms on top.
//
// Redraft mode ranks on the site's redraft value; Dynasty mode on dynasty value.
// Both subtract the same league-roster replacement level, and Superflex re-prices
// QBs by moving that replacement line (up to ~24 QBs start), not by a hand bump.
(function () {
  var cfg = window.__cheatCfg || {};
  var PS = window.BRPickScore;
  var LIMIT = 160;                      // players shown on the big board

  var state = {
    mode: cfg.mode === 'dynasty' ? 'dynasty' : 'redraft',
    sf: !!cfg.isSuperflex,
    filter: false,
    tab: 'board',
    done: new Set(),
  };
  var teams = Number(cfg.numTeams) || 12;
  var allPlayers = [];
  var players = [];
  var maxEff = 1;

  // ── roster → starter counts (mirror of draft_room.js rosterFromLeague) ──────
  function rosterFromLeague() {
    var rp = cfg.rosterPositions;
    if (!rp || !rp.length) return null;
    var r = { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0 };
    var map = {
      QB: 'QB', RB: 'RB', WR: 'WR', TE: 'TE',
      FLEX: 'FLEX', WRRB_FLEX: 'FLEX', REC_FLEX: 'FLEX', WRRBTE_FLEX: 'FLEX',
      SUPER_FLEX: 'SF', SFLEX: 'SF',
    };
    rp.forEach(function (s) { var k = map[String(s).toUpperCase()]; if (k) r[k]++; });
    if (!(r.QB + r.RB + r.WR + r.TE + r.FLEX + r.SF)) return null;
    return r;
  }
  function rosterCounts() {
    var lg = rosterFromLeague() || { QB: 1, SF: 0, RB: 2, WR: 3, TE: 1, FLEX: 1 };
    if (state.sf) { if (!lg.SF) lg.SF = 1; if (!lg.FLEX) lg.FLEX = 1; }
    else { lg.SF = 0; }
    return lg;
  }

  // ── value / adp / projection accessors (same fields the draft room reads) ───
  function redraftVal(p) {
    return (state.sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
                     : p.redraft_value_1qb) || 0;
  }
  function dynVal(p) { return (state.sf ? (p.sf_value || p.value) : p.value) || 0; }
  function scoreOf(p) { return state.mode === 'dynasty' ? dynVal(p) : redraftVal(p); }
  function projOf(p) {
    if (p.proj_pts != null) return Number(p.proj_pts);
    if (p.proj_ppg != null) return Number(p.proj_ppg) * 17;
    return null;
  }
  function adpOf(p) {
    if (state.mode === 'dynasty') { var a = state.sf ? p.sf_avg_pick : p.avg_pick; return a != null ? Number(a) : null; }
    var r = state.sf ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
    if (r != null) return Number(r);
    return p._radp != null ? Number(p._radp) : null;
  }

  // Replacement level on the current scoring scale, from the FULL pool so it is a
  // fixed preseason baseline (matches computeReplacement in the draft room).
  function computeReplacement(pool) {
    var starters = PS.starterCounts(rosterCounts());
    var byPos = { QB: [], RB: [], WR: [], TE: [] };
    pool.forEach(function (p) { var pos = (p.position || '').toUpperCase(); if (byPos[pos]) byPos[pos].push(scoreOf(p)); });
    var r = {};
    Object.keys(byPos).forEach(function (pos) {
      var arr = byPos[pos].sort(function (a, b) { return b - a; });
      if (!arr.length) { r[pos] = 0; return; }
      var idx = Math.round(teams * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      r[pos] = arr[idx];
    });
    return r;
  }

  function youthFactor(age) {
    if (age <= 22) return 0.40; if (age === 23) return 0.28; if (age === 24) return 0.18;
    if (age === 25) return 0.08; if (age === 26) return 0.02; if (age === 27) return -0.05;
    if (age === 28) return -0.13; if (age === 29) return -0.22; if (age === 30) return -0.32;
    return -0.44;
  }
  function windowFor(age) {
    if (age == null) return ['', ''];
    if (age <= 24) return ['Ascending', 'win-asc']; if (age <= 27) return ['Prime', 'win-prime'];
    if (age <= 30) return ['Win-now', 'win-now']; return ['Fading', 'win-fade'];
  }

  function compute() {
    // Stable redraft ADP fallback when there is no Sleeper redraft feed.
    allPlayers.slice().sort(function (a, b) { return redraftVal(b) - redraftVal(a); })
      .forEach(function (p, i) { p._radp = i + 1; });

    var repl = computeReplacement(allPlayers);
    var scored = [];
    allPlayers.forEach(function (p) {
      var pos = (p.position || '').toUpperCase();
      if (['QB', 'RB', 'WR', 'TE'].indexOf(pos) < 0) return;
      var s = scoreOf(p);
      if (s <= 0) return;                       // not draftable in this mode
      scored.push({
        p: p, pos: pos, name: p.name || String(p.id),
        age: (p.age != null ? Number(p.age) : null),
        proj: projOf(p), adp: adpOf(p),
        vor: Math.round(s - (repl[pos] || 0)),
      });
    });
    scored.sort(function (a, b) { return b.vor - a.vor || ((a.adp || 9999) - (b.adp || 9999)); });
    players = scored.slice(0, LIMIT);

    maxEff = players.length ? Math.max(1, players[0].vor) : 1;
    var pc = {};
    players.forEach(function (x, i) {
      x.rk = i + 1;
      x.value = (x.adp != null) ? Math.round(x.adp - x.rk) : null;
      x.good = state.mode === 'dynasty'
        ? (x.age != null && x.age <= 24 ? 1 : 0)
        : (x.value != null && x.value >= 5 ? 1 : 0);
      var r = x.vor / maxEff;
      x.oTier = r >= 0.72 ? 1 : r >= 0.50 ? 2 : r >= 0.33 ? 3 : r >= 0.16 ? 4 : 5;
      pc[x.pos] = (pc[x.pos] || 0) + 1; x.prk = x.pos + pc[x.pos];
    });
    var seen = {};
    for (var i = players.length - 1; i >= 0; i--) {
      var k = players[i].pos + '|' + players[i].oTier;
      if (!seen[k]) { seen[k] = 1; players[i].lastInTier = true; }
    }
  }

  // ── rendering helpers ───────────────────────────────────────────────────────
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function badge(pos) { return '<span class="cs-pos-badge cs-pos-' + pos + '">' + pos + '</span>'; }
  function posrk(x) { return '<span class="cs-posrk cs-pos-' + x.pos + '">' + x.prk + '</span>'; }
  function valChip(v) {
    if (v == null) return '';
    return v > 0 ? '<span class="cs-val g">+' + v + '</span>'
      : (v < 0 ? '<span class="cs-val b">' + v + '</span>' : '<span class="cs-val n">even</span>');
  }
  function smallVal(v) {
    if (v == null) return '';
    return v > 0 ? '<span class="cs-pgv cs-val g">+' + v + '</span>'
      : (v < 0 ? '<span class="cs-pgv cs-val b">' + v + '</span>' : '');
  }
  function winChip(age) { var w = windowFor(age); return w[0] ? '<span class="cs-winpill ' + w[1] + '">' + w[0] + '</span>' : ''; }
  function $(id) { return document.getElementById(id); }

  function render() {
    var dyn = state.mode === 'dynasty';
    $('csTitle').textContent = dyn ? 'Dynasty Cheat Sheet' : 'Redraft Cheat Sheet';
    $('csSub').textContent = dyn
      ? 'Ranked by dynasty value over replacement, from your league roster. The same value the Draft Room and Pick Score run on. Age and career window replace ADP.'
      : 'Ranked by value over replacement for your league scoring and roster. The same board the Draft Room ranks on, so the sheet never fights your live Pick Score.';

    if (!players.length) {
      $('csBoardBody').innerHTML = '<tr><td colspan="7" class="cs-empty">No players for this format yet.</td></tr>';
      return;
    }

    // legend
    $('csLegend').innerHTML = dyn
      ? '<span class="cs-lg"><b>VOR</b> dynasty value over replacement</span>'
        + '<span class="cs-lg"><b>Age</b> drives the window</span>'
        + '<span class="cs-lg">' + winChip(23) + ' ascending</span>'
        + '<span class="cs-lg">' + winChip(29) + ' win-now</span>'
        + '<span class="cs-lg" id="csFmtNote">Dynasty ' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>'
      : '<span class="cs-lg"><b>VOR</b> value over replacement, the ranking</span>'
        + '<span class="cs-lg"><b>Proj</b> projected season points</span>'
        + '<span class="cs-lg"><span class="cs-val g">+7</span> ranked above ADP, target it</span>'
        + '<span class="cs-lg"><span class="cs-val b">-4</span> going early, let it fall</span>'
        + '<span class="cs-lg" id="csFmtNote">' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>';

    renderBoard(dyn);
    renderPos(dyn);
    renderDraft(dyn);

    document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); });
    $('csValBtn').textContent = dyn ? 'Ascenders only' : 'Values only';
  }

  function renderBoard(dyn) {
    var col6 = dyn ? 'Age' : 'ADP', col7 = dyn ? 'Window' : 'Value';
    $('csBoardHead').innerHTML =
      '<tr><th>Rk</th><th class="l">Player</th><th>Pos</th><th>Proj</th><th>VOR</th><th>' + col6 + '</th><th>' + col7 + '</th></tr>';
    var lastT = 0, html = '';
    players.forEach(function (x) {
      if (x.oTier !== lastT) { lastT = x.oTier; html += '<tr class="cs-cliff"><td colspan="7"><div class="cs-cliffline">Tier ' + x.oTier + '</div></td></tr>'; }
      var done = state.done.has(x.name) ? ' done' : '';
      var c6 = dyn ? '<td class="cs-num">' + (x.age != null ? x.age : '') + '</td>' : '<td class="cs-num">' + (x.adp != null ? Math.round(x.adp) : '') + '</td>';
      var c7 = dyn ? '<td>' + winChip(x.age) + '</td>' : '<td>' + valChip(x.value) + '</td>';
      html += '<tr class="cs-p' + done + '" data-good="' + x.good + '" data-name="' + esc(x.name) + '">'
        + '<td class="cs-rk">' + x.rk + '</td>'
        + '<td><span class="cs-pcell">' + badge(x.pos) + '<span class="cs-pname">' + esc(x.name) + '</span></span></td>'
        + '<td>' + posrk(x) + '</td>'
        + '<td class="cs-num">' + (x.proj != null ? Math.round(x.proj) : '') + '</td>'
        + '<td><span class="cs-vorwrap"><span class="cs-num">' + x.vor + '</span><span class="cs-vorbar"><i style="width:' + Math.max(0, Math.round(x.vor / maxEff * 100)) + '%"></i></span></span></td>'
        + c6 + c7 + '</tr>';
    });
    $('csBoardBody').innerHTML = html;
    $('csBoardFoot').textContent = dyn
      ? 'Ranked by youth-aware dynasty VOR. Tap a row to cross a player off.'
      : 'Ranked by VOR, so a scarce elite TE or QB can outrank a higher-scoring skill player. Tap a row to cross a player off.';
  }

  function renderPos(dyn) {
    var POS = ['RB', 'WR', 'QB', 'TE'];
    var BAND = Math.max(1, maxEff * 0.045), CAP = 6;
    var groups = [], cur = null;
    players.forEach(function (x) {
      var tierChanged = !cur || x.oTier !== cur.tier;
      if (!cur || tierChanged || x.vor < cur.lead - BAND || cur.items.length >= CAP) {
        cur = { tier: x.oTier, lead: x.vor, items: [], tierBreak: tierChanged };
        groups.push(cur);
      }
      cur.items.push(x);
    });
    function nameChip(x) {
      var done = state.done.has(x.name) ? ' done' : '';
      var tail = dyn ? '' : smallVal(x.value);
      return '<span class="cs-pgc cs-c-' + x.pos + done + '" data-good="' + x.good + '" data-name="' + esc(x.name) + '"><span class="cs-pgn">' + esc(x.name) + tail + '</span></span>';
    }
    var out = '<div class="cs-pgrid-head">' + POS.map(function (p) { return '<div>' + p + '</div>'; }).join('') + '</div>';
    var ri = 0;
    groups.forEach(function (g) {
      if (g.tierBreak) {
        var counts = POS.map(function (pos) { var n = players.filter(function (y) { return y.oTier === g.tier && y.pos === pos; }).length; return n ? pos + ' ' + n : null; }).filter(Boolean).join(' &middot; ');
        out += '<div class="cs-pgtier">Tier ' + g.tier + '<span class="cs-sc">' + counts + ' left</span></div>';
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

  function renderDraft(dyn) {
    var n = Math.min(teams || 12, players.length);
    var dh = players.slice(0, n).map(function (x) {
      var round = Math.ceil(x.rk / (teams || 12));
      var run = x.lastInTier;
      return '<div class="cs-drow' + (run ? ' run' : '') + '">'
        + '<span class="cs-pick">#' + x.rk + '<small>RD ' + round + '</small></span>'
        + '<span class="cs-dtiers"><span class="cs-tchip cs-pos-' + x.pos + '">' + x.prk + '<span class="cs-ex">Tier ' + x.oTier + (dyn ? ' &middot; ' + windowFor(x.age)[0] : '') + '</span></span>'
        + '<span class="cs-pname" style="font-size:13px">' + esc(x.name) + '</span>'
        + (run ? '<span class="cs-runflag">last ' + x.pos + ' in Tier ' + x.oTier + '</span>' : '') + '</span>'
        + '</div>';
    }).join('');
    $('csDboard').innerHTML = dh;
  }

  // ── interactions ────────────────────────────────────────────────────────────
  function wireSeg(id, key, transform) {
    document.querySelectorAll('#' + id + ' button').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('#' + id + ' button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
        state[key] = transform(b);
        compute(); render();
      });
    });
  }

  function init() {
    var back = $('csBack'); if (back && cfg.draftUrl) back.href = cfg.draftUrl;
    wireSeg('csMode', 'mode', function (b) { return b.getAttribute('data-mode'); });
    wireSeg('csQb', 'sf', function (b) { return b.getAttribute('data-qb') === 'SF'; });

    $('csValBtn').addEventListener('click', function () {
      state.filter = !state.filter; this.setAttribute('aria-pressed', String(state.filter));
      document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); });
    });
    $('csPrintBtn').addEventListener('click', function () { window.print(); });

    document.addEventListener('click', function (e) {
      var el = e.target.closest('[data-name]');
      if (!el || !e.target.closest('#cs-panel-board, #cs-panel-pos')) return;
      var nm = el.getAttribute('data-name');
      if (state.done.has(nm)) state.done.delete(nm); else state.done.add(nm);
      document.querySelectorAll('[data-name="' + (window.CSS && CSS.escape ? CSS.escape(nm) : nm.replace(/"/g, '\\"')) + '"]').forEach(function (x) { x.classList.toggle('done', state.done.has(nm)); });
    });

    var tabs = document.querySelectorAll('.cs-tabs [role=tab]');
    var panels = { board: 'cs-panel-board', pos: 'cs-panel-pos', draft: 'cs-panel-draft', logic: 'cs-panel-logic' };
    tabs.forEach(function (t) {
      t.addEventListener('click', function () {
        tabs.forEach(function (x) { x.setAttribute('aria-selected', String(x === t)); });
        Object.keys(panels).forEach(function (k) { $(panels[k]).classList.toggle('hidden', k !== t.getAttribute('data-tab')); });
        $('csLegend').style.display = (t.getAttribute('data-tab') === 'logic') ? 'none' : '';
      });
    });

    // reflect initial format on the toggles
    document.querySelectorAll('#csMode button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === state.mode)); });
    document.querySelectorAll('#csQb button').forEach(function (b) { b.setAttribute('aria-pressed', String((b.getAttribute('data-qb') === 'SF') === state.sf)); });

    fetch('/api/league-players', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        allPlayers = raw.filter(function (p) { return p && p.id != null && ['QB', 'RB', 'WR', 'TE'].indexOf(String(p.position || '').toUpperCase()) >= 0; });
        compute(); render();
      })
      .catch(function () {
        $('csBoardBody').innerHTML = '<tr><td colspan="7" class="cs-empty">Could not load players. Refresh to retry.</td></tr>';
      });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
