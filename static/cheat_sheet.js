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
    hideDrafted: false,
    adpSource: 'auto',
    done: new Set(),
  };
  var teams = Number(cfg.numTeams) || 12;
  var allPlayers = [];
  var players = [];
  var tierThresholds = {};
  var adpSourceOptions = {};   // {redraft:[{value,label}], startup:[...], rookie:[...]}
  var draftedIds = null;       // Set of live-drafted player ids, or null if none
  var maxScore = 1, maxVor = 1;
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

    var repl = C.computeReplacement(pool, mode, sf, teams, cfg.rosterPositions);
    var scale = C.computePpgScale(pool, teams, cfg.rosterPositions, sf);
    var mv = C.maxVal(pool, mode, sf);
    var rounds = Number(cfg.rounds) || (mode === 'dynasty' ? 20 : 15);
    var totalPicks = teams * rounds;

    // Tier-bucket counts for the tier-cliff term (dynasty only; redraft tier=null).
    var tcount = {};
    pool.forEach(function (p) {
      var t = C.tierOf(p, mode, sf, teams, tierThresholds); if (t == null) return;
      var k = (p.position || '').toUpperCase() + '|' + t; tcount[k] = (tcount[k] || 0) + 1;
    });

    var scored = pool.map(function (p) {
      var pos = (p.position || '').toUpperCase();
      var value = C.valOf(p, mode, sf);
      var vor = Math.round(value - (repl[pos] || 0));
      var tier = C.tierOf(p, mode, sf, teams, tierThresholds);
      var adp = C.adpOf(p, mode, sf);
      var pickNo = adp != null ? Math.round(adp) : totalPicks;   // grade each player at his own ADP
      var cliff = tier != null && (tcount[pos + '|' + tier] || 0) <= 2;
      // needRaw 0.5 + empty roster = a team-neutral board grade: no roster-need
      // bias, no redundancy/overfill penalties (those need a filled roster). What
      // survives is exactly the intrinsic Pick Score: value, VOR, production,
      // tier, youth, and the QB streamability taper. This IS the draft room's
      // grade of a pick with an empty roster.
      var score = PS.computePickScore({
        pos: pos, value: value, vor: vor, tier: tier,
        age: (p.age != null ? Number(p.age) : null), rankChange7d: p.rank_change_7d,
        avgPick: adp, pickNo: pickNo, maxVal: mv,
        draftType: mode === 'dynasty' ? 'startup' : 'redraft', isSf: sf,
        needRaw: 0.5, qbCount: 0, totalPicks: totalPicks, numTeams: teams,
        ppgNorm: C.ppgNorm(p, scale), ppr: 1.0, tep: 0, isTierCliff: cliff,
        survivalAdj: 0, handcuff: false,
      });
      return {
        id: String(p.id), pos: pos, name: p.name || String(p.id),
        age: (p.age != null ? Number(p.age) : null),
        proj: (p.proj_pts != null ? Number(p.proj_pts) : (p.proj_ppg != null ? Number(p.proj_ppg) * 17 : null)),
        adp: adp, vor: vor, tier: tier, score: score,
      };
    });
    scored.sort(function (a, b) { return b.score - a.score || b.vor - a.vor || ((a.adp || 9999) - (b.adp || 9999)); });
    players = scored.slice(0, LIMIT);

    maxScore = players.length ? Math.max(1, players[0].score) : 1;
    maxVor = players.length ? Math.max(1, players[0].vor) : 1;
    var pc = {};
    players.forEach(function (x, i) {
      x.rk = i + 1;
      x.value = (x.adp != null) ? Math.round(x.adp - x.rk) : null;
      x.good = state.mode === 'dynasty' ? (x.age != null && x.age <= 24 ? 1 : 0) : (x.value != null && x.value >= 5 ? 1 : 0);
      x.drafted = draftedIds ? draftedIds.has(x.id) : false;
      pc[x.pos] = (pc[x.pos] || 0) + 1; x.prk = x.pos + pc[x.pos];
      // Display tier: dynasty uses the shared value tier (matches the room);
      // redraft has no value tier there, so the sheet shows its own VOR cliffs.
      if (state.mode === 'dynasty') { x.dtier = x.tier || 1; }
      else { var r = x.vor / maxVor; x.dtier = r >= 0.72 ? 1 : r >= 0.50 ? 2 : r >= 0.33 ? 3 : r >= 0.16 ? 4 : 5; }
    });
    var seen = {};
    for (var i = players.length - 1; i >= 0; i--) {
      var k = players[i].pos + '|' + players[i].dtier;
      if (!seen[k]) { seen[k] = 1; players[i].lastInTier = true; }
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

  function render() {
    var dyn = state.mode === 'dynasty';
    $('csTitle').textContent = dyn ? 'Dynasty Cheat Sheet' : 'Redraft Cheat Sheet';
    $('csSub').textContent = dyn
      ? 'Ranked by the same Pick Score the Draft Room uses, on dynasty value over replacement for your league roster. Age and career window replace ADP.'
      : 'Ranked by the same Pick Score the Draft Room uses, on value over replacement for your league scoring and roster. The board and the room agree on who is better.';

    document.querySelectorAll('.cs-board').forEach(function (b) { b.classList.toggle('filteron', state.filter); b.classList.toggle('hidedrafted', state.hideDrafted); });
    $('csValBtn').textContent = dyn ? 'Ascenders only' : 'Values only';
    var hd = $('csHideDrafted'); if (hd) { hd.style.display = draftedIds ? '' : 'none'; hd.setAttribute('aria-pressed', String(state.hideDrafted)); }

    if (!players.length) {
      $('csBoardBody').innerHTML = '<tr><td colspan="7" class="cs-empty">' + (loading ? 'Loading players…' : 'No players for this format yet.') + '</td></tr>';
      $('csLegend').innerHTML = '';
      return;
    }

    var draftedNote = draftedIds ? '<span class="cs-lg"><span class="cs-taken-dot"></span> already drafted</span>' : '';
    $('csLegend').innerHTML = dyn
      ? '<span class="cs-lg"><b>Score</b> Pick Score, the ranking</span>'
        + '<span class="cs-lg"><b>VOR</b> dynasty value over replacement</span>'
        + '<span class="cs-lg">' + winChip(23) + ' ascending</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">Dynasty ' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>'
      : '<span class="cs-lg"><b>Score</b> Pick Score, the ranking</span>'
        + '<span class="cs-lg"><b>VOR</b> value over replacement</span>'
        + '<span class="cs-lg"><span class="cs-val g">+7</span> above ADP, target it</span>'
        + draftedNote
        + '<span class="cs-lg" id="csFmtNote">' + (state.sf ? 'Superflex' : '1QB') + ' &middot; ' + teams + '-team</span>';

    renderBoard(dyn);
    renderPos(dyn);
    renderDraft(dyn);
  }

  function renderBoard(dyn) {
    var col6 = dyn ? 'Age' : 'ADP', col7 = dyn ? 'Window' : 'Value';
    $('csBoardHead').innerHTML =
      '<tr><th>Rk</th><th class="l">Player</th><th>Pos</th><th>Score</th><th>VOR</th><th>' + col6 + '</th><th>' + col7 + '</th></tr>';
    var lastT = 0, html = '';
    players.forEach(function (x) {
      if (x.dtier !== lastT) { lastT = x.dtier; html += '<tr class="cs-cliff"><td colspan="7"><div class="cs-cliffline">Tier ' + x.dtier + '</div></td></tr>'; }
      var cls = 'cs-p' + (state.done.has(x.name) ? ' done' : '') + (x.drafted ? ' drafted' : '');
      var c6 = dyn ? '<td class="cs-num">' + (x.age != null ? x.age : '') + '</td>' : '<td class="cs-num">' + (x.adp != null ? Math.round(x.adp) : '') + '</td>';
      var c7 = dyn ? '<td>' + winChip(x.age) + '</td>' : '<td>' + valChip(x.value) + '</td>';
      html += '<tr class="' + cls + '" data-good="' + x.good + '" data-name="' + esc(x.name) + '">'
        + '<td class="cs-rk">' + x.rk + '</td>'
        + '<td><span class="cs-pcell">' + badge(x.pos) + '<span class="cs-pname">' + esc(x.name) + '</span></span></td>'
        + '<td>' + posrk(x) + '</td>'
        + '<td><span class="cs-vorwrap"><span class="cs-num">' + x.score + '</span><span class="cs-vorbar"><i style="width:' + Math.max(0, Math.round(x.score / maxScore * 100)) + '%"></i></span></span></td>'
        + '<td class="cs-num">' + x.vor + '</td>'
        + c6 + c7 + '</tr>';
    });
    $('csBoardBody').innerHTML = html;
    $('csBoardFoot').textContent = dyn
      ? 'Ranked by Pick Score on youth-aware dynasty value. Tap a row to cross a player off.'
      : 'Ranked by Pick Score, so a scarce elite TE or QB can outrank a higher-scoring skill player. Tap a row to cross a player off.';
  }

  function renderPos(dyn) {
    var POS = ['RB', 'WR', 'QB', 'TE'];
    var BAND = Math.max(1, maxVor * 0.045), CAP = 6;
    var groups = [], cur = null;
    players.forEach(function (x) {
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
      return '<span class="' + cls + '" data-good="' + x.good + '" data-name="' + esc(x.name) + '"><span class="cs-pgn">' + esc(x.name) + tail + '</span></span>';
    }
    var out = '<div class="cs-pgrid-head">' + POS.map(function (p) { return '<div>' + p + '</div>'; }).join('') + '</div>';
    var ri = 0;
    groups.forEach(function (g) {
      if (g.tierBreak) {
        var counts = POS.map(function (pos) { var n = players.filter(function (y) { return y.dtier === g.tier && y.pos === pos; }).length; return n ? pos + ' ' + n : null; }).filter(Boolean).join(' &middot; ');
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
    var live = players.filter(function (x) { return !x.drafted; });
    var n = Math.min(teams || 12, live.length);
    var dh = live.slice(0, n).map(function (x, i) {
      var round = Math.ceil((i + 1) / (teams || 12));
      var run = x.lastInTier;
      return '<div class="cs-drow' + (run ? ' run' : '') + '">'
        + '<span class="cs-pick">#' + x.rk + '<small>RD ' + round + '</small></span>'
        + '<span class="cs-dtiers"><span class="cs-tchip cs-pos-' + x.pos + '">' + x.prk + '<span class="cs-ex">Tier ' + x.dtier + (dyn ? ' &middot; ' + youthWindow(x.age)[0] : '') + '</span></span>'
        + '<span class="cs-pname" style="font-size:13px">' + esc(x.name) + '</span>'
        + (run ? '<span class="cs-runflag">last ' + x.pos + ' in Tier ' + x.dtier + '</span>' : '') + '</span>'
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
        $('csBoardBody').innerHTML = '<tr><td colspan="7" class="cs-empty">Could not load players. Refresh to retry.</td></tr>';
      });
  }

  // ── live-draft cross-off ────────────────────────────────────────────────────
  function detectLiveDraft() {
    if (!cfg.leagueId || !cfg.platform) return;
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function (r) { return r.json(); })
      .then(function (resp) {
        if (!resp || resp.unsupported) return;
        var all = resp.drafts || [];
        // Prefer a live draft, then the most recent; a completed draft still tells
        // us who is gone (useful if you open the sheet mid- or post-draft).
        var live = all.filter(function (d) { return String(d.status) === 'drafting'; });
        var pick = live[0] || all.filter(function (d) { return String(d.status) !== 'pre_draft'; })[0];
        if (!pick || !pick.draft_id) return;
        loadDraftPicks(pick.draft_id);
      })
      .catch(function () { /* no live draft; sheet stays static */ });
  }
  function loadDraftPicks(draftId) {
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(draftId), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var picks = (d && d.picks) || [];
        var s = new Set();
        picks.forEach(function (pk) { if (pk && pk.player_id) s.add(String(pk.player_id)); });
        if (!s.size) return;
        draftedIds = s;
        compute(); render();
      })
      .catch(function () { /* ignore */ });
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
    wireSeg('csMode', function (b) { state.mode = b.getAttribute('data-mode'); });
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
    $('csPrintBtn').addEventListener('click', function () { window.print(); });
    var srcSel = $('csAdpSrc');
    if (srcSel) srcSel.addEventListener('change', function () { state.adpSource = this.value; loadPlayers(); });

    document.addEventListener('click', function (e) {
      var el = e.target.closest('[data-name]');
      if (!el || !e.target.closest('#cs-panel-board, #cs-panel-pos')) return;
      var nm = el.getAttribute('data-name');
      if (state.done.has(nm)) state.done.delete(nm); else state.done.add(nm);
      var q = (window.CSS && CSS.escape) ? CSS.escape(nm) : nm.replace(/"/g, '\\"');
      document.querySelectorAll('[data-name="' + q + '"]').forEach(function (x) { x.classList.toggle('done', state.done.has(nm)); });
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

    document.querySelectorAll('#csMode button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-mode') === state.mode)); });
    document.querySelectorAll('#csQb button').forEach(function (b) { b.setAttribute('aria-pressed', String((b.getAttribute('data-qb') === 'SF') === state.sf)); });

    loadPlayers().then(detectLiveDraft);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
