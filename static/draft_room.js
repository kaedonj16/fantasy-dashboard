(function(){
  var cfg = window.__draftCfg || {};
  // Match the site-wide position palette (see .nav-search-pos-* in dashboard.css).
  var POS_COLOR = { QB:'#3b82f6', RB:'#22c55e', WR:'#f59e0b', TE:'#8b5cf6', K:'#c92c68', DEF:'#475569', FLEX:'#14b8a6', SF:'#a78bfa', BN:'#64748b', IR:'#94a3b8', TAXI:'#64748b', IDP:'#0f766e' };
  var posColor = function(p){ return POS_COLOR[(p||'').toUpperCase()] || '#94a3b8'; };
  var hsUrl = function(id){ return 'https://sleepercdn.com/content/nfl/players/' + id + '.jpg'; };
  // DEF players: prefer locally cached logo (after running download_team_logos.py),
  // fall back to ESPN CDN which the browser fetches directly.
  var playerImgUrl = function(p){
    var pos = String(p.position || '').toUpperCase();
    if (pos === 'DEF' && p.team){
      var t = p.team.toUpperCase();
      // Check local first; if the file doesn't exist the browser's onerror swaps to ESPN CDN.
      return '/static/images/team_logos/' + t + '.png';
    }
    return hsUrl(p.id);
  };
  // Inline onerror for DEF logo <img> tags: try ESPN CDN, then hide.
  var _defImgErr = function(img){
    var t = img.getAttribute('data-team');
    if (t && !img._espnFallback){
      img._espnFallback = true;
      img.src = 'https://a.espncdn.com/i/teamlogos/nfl/500/' + t.toLowerCase() + '.png';
    } else {
      img.style.visibility = 'hidden';
    }
  };

  var sessKey = 'dr_' + location.pathname;
  var state = null;        // { type, teams, rounds, sf, slot, order, picks:{}, current }
  var players = [];        // best-available pool
  var drafted = {};        // id -> true
  var posFilter = {};      // multi-select set of positions ({} = All)
  function _posIsAll(){ for (var _k in posFilter){ if (posFilter.hasOwnProperty(_k)) return false; } return true; }
  function _posMatches(pos){ return _posIsAll() || !!posFilter[String(pos || '').toUpperCase()]; }
  // Reflect the posFilter set on the pills: each selected position gets .active,
  // and "All" is active only when nothing is selected.
  function _syncPosPills(){
    var all = _posIsAll();
    var host = document.getElementById('drPosFilters'); if (!host) return;
    host.querySelectorAll('.dr-pos').forEach(function(x){
      var p = x.getAttribute('data-pos');
      x.classList.toggle('active', p === 'ALL' ? all : !!posFilter[String(p).toUpperCase()]);
    });
  }
  var justPick = null;     // pick # filled this render (for the pop-in animation)
  var playersById = {};    // id -> player (value lookup for live picks)
  var lastLivePicks = null;// last picks payload from the live feed
  var saveTimer = null;    // debounce for DB autosave
  var pollTimer = null;    // live-draft poll: next setTimeout handle
  var pollTickTimer = null;// 1s ticker that refreshes the freshness indicator
  var _pollInFlight = false;
  var _pollCount = 0;      // polls since connect (drives periodic full refresh)
  var _pollLastAt = 0;     // ms timestamp of the last successful poll
  var _pollNextAt = 0;     // ms timestamp the next poll is scheduled for
  var _liveSig = null;     // signature of the last applied live state (skip no-op renders)
  var _pickLagMsg = null;  // diagnostic: "pick +Xs late" shown briefly after each new pick
  var POLL_MS = 4000;      // base cadence (just above the 3s picks cache TTL)
  var POLL_FULL_EVERY = 60;// every N light polls, do a full refresh to catch trades/slot names
  var sim = false;         // mock-draft simulation active
  var simTimer = null;
  var simSpeed = 700;      // ms between CPU picks
  var simPaused = false;
  var simStarted = false;  // CPU picks only run once the user hits Start Draft
  var simAutoDraft = false; // when true, auto-commit best pick on my turn
  var sideTab = 'best';    // best | rec | needs | runs
  var _setupRoster = null; // roster config built on setup page
  var _rosterMode  = 'auto'; // 'auto' = use league/defaults locked; 'custom' = editable steppers
  var _rosterPreset = null; // selected built-in platform preset, if any
  var _setupOwned = null;  // claimed picks (pickNumber -> true) built on setup page
  var keeperSet = [];      // [{id,name,pos,rosterId,costRound,projected}] from cfg.keepers
  var keepersOn = false;   // whether league keepers are removed from the board
  var _setupOwnedSig = ''; // staleness signature for _setupOwned
  var _capAddRound = null; // round whose inline slot picker is open (or null)
  var _capLateOpen = false;// whether the combined late-rounds section is expanded
  var tierThresholds = {}; // {leagueType:{size:[...]}} from /api/league-players
  var adpSources = {};     // {startup|rookie|redraft: 'Sleeper'|'none'} from /api/league-players
  var adpSourceOptions = {}; // {startup|rookie|redraft: [{value,label}]} from payload
  var adpSource = 'brfantasy'; // currently selected ADP source. Draft Room defaults
                             // to BR Fantasy (our own crawl of real draft picks);
                             // 'auto' = server default (Sleeper), any real source
                             // overlays via the resolver. Valid on every draft axis
                             // (redraft/dynasty/rookie), so the default always resolves.
  var _boardSig = null;    // board structure signature (rebuild only when it changes)
  var _summaryShown = false; // auto-open summary only once per draft
  var compareIds = [];     // 0-2 player IDs staged for comparison
  var _chipsCollapsed = false; // best-at-pos strip collapsed state
  var _cellShowPs = false; // board cell corner: false=dynasty value, true=pick score
  var _timerInterval = null;   // pick countdown setInterval handle
  var _timerPickStart = null;  // Date.now() when current pick slot opened
  var _timerPickNo = null;     // which pick number the timer is counting for

  // ── Pick-order helper (snake / linear / 3rr) ───────────────────────────────
  function pickDir(r, order){            // true = forward (slot 1 → N)
    if (order === 'linear') return true;
    if (order === '3rr') {
      if (r === 1) return true;
      if (r === 2 || r === 3) return false;
      return (r % 2 === 0);
    }
    return (r % 2 === 1);                 // snake
  }
  function pickNum(r, slot, teams, order){
    var inRound = pickDir(r, order) ? slot : (teams - slot + 1);
    return (r - 1) * teams + inRound;
  }
  function slotOnClock(pickNo, teams, order){
    var r = Math.ceil(pickNo / teams);
    var posInRound = pickNo - (r - 1) * teams;
    return pickDir(r, order) ? posInRound : (teams - posInRound + 1);
  }
  window.draftPickOrder = { pickDir: pickDir, pickNum: pickNum, slotOnClock: slotOnClock };

  // ── Pick ownership ─────────────────────────────────────────────────────────
  // The user can own any set of picks (multiple firsts via trades, etc.).
  // state.owned maps pickNumber -> true. When absent we fall back to the home
  // slot so older saved sessions keep working.
  function isMyPick(pn){
    if (state && state.owned) return !!state.owned[pn];
    return !!(state && state.slot && slotOnClock(pn, state.teams, state.order) === state.slot);
  }
  function defaultOwned(){
    var o = {};
    if (!state || !state.slot) return o;
    var tot = state.teams * state.rounds;
    for (var pn = 1; pn <= tot; pn++){
      if (slotOnClock(pn, state.teams, state.order) === state.slot) o[pn] = true;
    }
    return o;
  }
  function ensureOwned(){ if (state && !state.owned) state.owned = defaultOwned(); }
  function ownedPicks(){
    ensureOwned();
    if (!state || !state.owned) return [];
    return Object.keys(state.owned).filter(function(k){ return state.owned[k]; })
      .map(Number).sort(function(a, b){ return a - b; });
  }
  function hasOwned(){ return ownedPicks().length > 0; }
  // ── Column (seat) ownership ─────────────────────────────────────────────────
  // A board column maps to a draft seat. Ownership is per-pick (trades can give
  // you picks in any seat's column), so "is this column mine?" must be derived
  // from actual pick ownership, never from the original home slot.
  function picksInColumn(slot){
    var out = [];
    var total = (state.teams || 0) * (state.rounds || 0);
    for (var pn = 1; pn <= total; pn++){
      if (slotOnClock(pn, state.teams, state.order) === slot) out.push(pn);
    }
    return out;
  }
  function ownsAnyInColumn(slot){
    return picksInColumn(slot).some(function(pn){ return isMyPick(pn); });
  }
  function ownsAllInColumn(slot){
    var pcs = picksInColumn(slot);
    return pcs.length > 0 && pcs.every(function(pn){ return isMyPick(pn); });
  }
  // The next owned pick strictly after the current selection (for wait/value-now signals).
  function nextOwnedAfterCurrent(){
    var ups = upcomingOwnedPicks();
    for (var i = 0; i < ups.length; i++){ if (ups[i] > state.current) return ups[i]; }
    return null;
  }
  // Next owned pick number after `pn`, whether or not that pick is already
  // filled. Deep Dive uses this as "your next turn" when grading a historical
  // pick (unlike nextOwnedAfterCurrent, which only looks at unfilled picks
  // from the live clock).
  function nextOwnedPickAfter(pn){
    var owned = ownedPicks();
    for (var i = 0; i < owned.length; i++){ if (owned[i] > pn) return owned[i]; }
    return null;
  }
  // Upcoming owned picks that have not been made yet (current pick included).
  function upcomingOwnedPicks(){
    return ownedPicks().filter(function(pn){ return pn >= state.current && !state.picks[pn]; });
  }
  function toggleOwned(pn){
    ensureOwned();
    if (state.owned[pn]) delete state.owned[pn]; else state.owned[pn] = true;
    save();
  }

  // ── Persistence ────────────────────────────────────────────────────────────
  function save(){ try { sessionStorage.setItem(sessKey, JSON.stringify(state)); } catch(e){} }
  function load(){ try { return JSON.parse(sessionStorage.getItem(sessKey) || 'null'); } catch(e){ return null; } }

  function fillSlotOptions(teams){
    var sel = document.getElementById('drSlot');
    sel.innerHTML = '';
    for (var i = 1; i <= teams; i++){
      var o = document.createElement('option');
      o.value = i; o.textContent = 'Pick ' + i;
      sel.appendChild(o);
    }
  }

  // ── Setup ────────────────────────────────────────────────────────────────
  function _defaultRounds(typeVal){
    if (typeVal === 'rookie')  return String(cfg.numRoundsRookie  || 3);
    if (typeVal === 'startup') return String(cfg.numRoundsStartup || 15);
    return '15';
  }
  // Drafted ids the overlay should cross off — same set that drops players out
  // of best-available (includes keepers), not only cells already on the board.
  function cheatDraftedIds(){
    var ids = [];
    Object.keys(drafted).forEach(function(id){ if (drafted[id]) ids.push(String(id)); });
    return ids;
  }
  function cheatRecommendationOrder(){
    if (!state || !players.length) return [];
    refreshPsPool();
    return rankedRecommendationPool()
      .slice(0, 175).map(function(p){ return String(p.id); });
  }
  function cheatContextPayload(){
    var counts = null;
    try { counts = myPosCounts(); } catch (e) { counts = null; }
    return {
      type: 'drCheatContext',
      drafted: cheatDraftedIds(),
      rec_order: cheatRecommendationOrder(),
      teams: state && state.teams ? state.teams : null,
      slot: state && state.slot ? state.slot : null,
      myCounts: counts
    };
  }
  var _cheatCtxSig = null;
  function pushCheatSheetContext(){
    var overlay = document.getElementById('drCheatSheet');
    var frame = document.getElementById('drCheatFrame');
    if (!overlay || overlay.style.display !== 'flex' || !frame || !frame.contentWindow) return;
    if (!state || (state.mode !== 'mock' && state.mode !== 'live')) return;
    var payload = cheatContextPayload();
    var sig = payload.drafted.join(',') + '#' + payload.rec_order.join(',')
      + '#' + String(payload.slot || '') + '#' + String(payload.teams || '');
    if (sig === _cheatCtxSig) return;
    _cheatCtxSig = sig;
    try { frame.contentWindow.postMessage(payload, window.location.origin); } catch (e) {}
  }
  function applyCfgDefaults(){
    // Point the hero's Draft History link at the league-scoped page when available.
    var _hl = document.getElementById('drToHistory');
    if (_hl && cfg.historyUrl) _hl.setAttribute('href', cfg.historyUrl);
    var _cs = document.getElementById('drToCheatSheet');
    // In-draft cheat sheet: the options-menu link opens the sheet in an overlay
    // (iframe of the chrome-less embed) so you never leave the draft. Cmd/Ctrl/
    // middle-click still opens the full page in a new tab.
    var _cs2 = document.getElementById('drOptsCheatSheet');
    var _csPop = document.getElementById('drCheatPop');
    var _csOverlay = document.getElementById('drCheatSheet');
    var _csFrame = document.getElementById('drCheatFrame');

    // Carry this draft's context to the cheat sheet so it opens with who's already
    // gone crossed off and the exact Recommendation order currently shown in the
    // room. The first paint is a snapshot in the URL; while the overlay stays
    // open, render() pushes pick updates via postMessage so cross-off and REC #
    // stay in sync. Live Sleeper polling remains an explicit choice on the sheet.
    function cheatCtxQuery(){
      // A restored draft can still exist in memory while Edit Setup is open. Do
      // not leak that stale board into the setup-page Cheat Sheet link; context
      // belongs only to a board the user is currently viewing.
      var main = document.getElementById('drMain');
      if (!(state && state.picks && (state.mode === 'mock' || state.mode === 'live')
            && main && main.style.display !== 'none')) return '';
      var ids = cheatDraftedIds();
      var isLocal = state.mode === 'mock';
      var q = [];
      if (isLocal){
        // A local mock uses its own setup, not the league's format.
        q.push('sf=' + (state.sf ? '1' : '0'));
        q.push('mode=' + (state.type === 'redraft' ? 'redraft' : 'dynasty'));
      }
      if (ids.length) q.push('drafted=' + encodeURIComponent(ids.join(',')));
      var recOrder = cheatRecommendationOrder();
      if (recOrder.length) q.push('rec_order=' + encodeURIComponent(recOrder.join(',')));
      if (state.teams) q.push('teams=' + encodeURIComponent(String(state.teams)));
      if (state.slot) q.push('slot=' + encodeURIComponent(String(state.slot)));
      return q.join('&');
    }
    function cheatSheetFullUrl(){
      var url = cfg.cheatSheetUrl || '/draft/cheat-sheet';
      var q = cheatCtxQuery();
      if (q) url += (url.indexOf('?') >= 0 ? '&' : '?') + q;
      return url;
    }
    // Refresh the new-tab links just before the browser follows them (mousedown
    // fires for left, Cmd/Ctrl and middle clicks), so the tab opens with the
    // current picks rather than a stale snapshot from page load.
    function refreshCheatHrefs(){
      var u = cheatSheetFullUrl();
      [_cs, _cs2, _csPop].forEach(function(a){ if (a) a.setAttribute('href', u); });
    }
    refreshCheatHrefs();
    [_cs, _cs2, _csPop].forEach(function(a){ if (a) a.addEventListener('mousedown', refreshCheatHrefs); });
    function openCheatSheet(){
      if (!_csOverlay || !_csFrame) return;
      var url = cfg.cheatSheetEmbedUrl || '/draft/cheat-sheet/embed';
      // First paint is a URL snapshot. Continued Sleeper polling is still
      // opt-in from Connect live draft; pick updates from THIS room are pushed
      // into the iframe via postMessage while the overlay stays open.
      var main = document.getElementById('drMain');
      if (state && state.picks && (state.mode === 'mock' || state.mode === 'live')
          && main && main.style.display !== 'none'){
        var ids = cheatDraftedIds();
        var q = ['sf=' + (state.sf ? '1' : '0'), 'mode=' + (state.type === 'redraft' ? 'redraft' : 'dynasty')];
        if (ids.length) q.push('drafted=' + encodeURIComponent(ids.join(',')));
        var recOrder = cheatRecommendationOrder();
        if (recOrder.length) q.push('rec_order=' + encodeURIComponent(recOrder.join(',')));
        if (state.teams) q.push('teams=' + encodeURIComponent(String(state.teams)));
        if (state.slot) q.push('slot=' + encodeURIComponent(String(state.slot)));
        url += (url.indexOf('?') >= 0 ? '&' : '?') + q.join('&');
      }
      _cheatCtxSig = null;
      _csFrame.src = url;   // (re)load -> re-syncs
      _csOverlay.style.display = 'flex';
      document.body.classList.add('dr-cheat-open');
      var close = document.getElementById('drCheatClose');
      if (close) close.focus();
    }
    function closeCheatSheet(){
      if (!_csOverlay) return;
      _csOverlay.style.display = 'none';
      document.body.classList.remove('dr-cheat-open');
      _cheatCtxSig = null;
      if (_csFrame) _csFrame.src = 'about:blank';   // stop the embed's poll loop
    }
    if (_cs2) _cs2.addEventListener('click', function(e){
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;  // let modified clicks open a tab
      e.preventDefault();
      // Live sync inside the in-draft overlay is free. Custom board edits on
      // the standalone cheat sheet remain PRO.
      openCheatSheet();
    });
    var _csClose = document.getElementById('drCheatClose');
    if (_csClose) _csClose.addEventListener('click', closeCheatSheet);
    if (_csOverlay) _csOverlay.addEventListener('click', function(e){ if (e.target === _csOverlay) closeCheatSheet(); });
    document.addEventListener('keydown', function(e){
      if (e.key === 'Escape' && _csOverlay && _csOverlay.style.display === 'flex') closeCheatSheet();
    });
    if (_csFrame) _csFrame.addEventListener('load', function(){
      if (!_csFrame.src || _csFrame.src === 'about:blank') return;
      _cheatCtxSig = null;
      pushCheatSheetContext();
    });
    window.addEventListener('message', function(e){
      if (e.origin !== window.location.origin) return;
      if (e.data && e.data.type === 'drCheatReady') {
        _cheatCtxSig = null;
        pushCheatSheetContext();
      }
    });
    if (cfg.numTeams) {
      var t = document.getElementById('drTeams');
      var want = String(Math.min(14, Math.max(8, cfg.numTeams)));
      for (var i=0;i<t.options.length;i++){ if (t.options[i].value === want || t.options[i].text === want){ t.selectedIndex = i; break; } }
    }
    if (cfg.isSuperflex) document.getElementById('drSf').value = '1';
    if (cfg.scoring) {
      var cfgPpr = document.getElementById('drPpr'); if (cfgPpr) cfgPpr.value = String(cfg.scoring.ppr != null ? cfg.scoring.ppr : 1);
      var cfgTep = document.getElementById('drTep'); if (cfgTep) cfgTep.value = String(cfg.scoring.tep != null ? cfg.scoring.tep : 0);
      var cfgPassTd = document.getElementById('drPassTd'); if (cfgPassTd) cfgPassTd.value = String(cfg.scoring.passTd >= 6 ? 6 : 4);
    }
    var typeVal = document.getElementById('drType').value;
    var isRookie = typeVal === 'rookie';
    var rf = document.getElementById('drRoundsField');
    if (rf) rf.style.display = isRookie ? '' : 'none';
    if (isRookie) {
      document.getElementById('drRounds').value = String(cfg.numRoundsRookie || 3);
    }
    // For non-rookie: rounds will be synced from roster after renderSetupRoster() runs.
    fillSlotOptions(parseInt(document.getElementById('drTeams').value, 10));
  }

  document.getElementById('drTeams').addEventListener('change', function(){
    fillSlotOptions(parseInt(this.value, 10));
  });
  // Map a Sleeper-style roster_positions list into our slot counts.
  // IR / taxi / IDP are counted so dynasty round depth is right; they are not
  // starters, so they don't inflate RB/WR need — they add stash/round capacity.
  var ROSTER_SLOT_MAP = {
    QB:'QB', RB:'RB', WR:'WR', TE:'TE',
    FLEX:'FLEX', WRRB_FLEX:'FLEX', REC_FLEX:'FLEX', WRRBTE_FLEX:'FLEX',
    SUPER_FLEX:'SF', SFLEX:'SF',
    K:'K', DEF:'DEF', DST:'DEF',
    BN:'BN', BE:'BN', BENCH:'BN',
    IR:'IR', RESERVE:'IR',
    TAXI:'TAXI',
    DL:'IDP', DE:'IDP', DT:'IDP', LB:'IDP', DB:'IDP', CB:'IDP', S:'IDP',
    IDP:'IDP', IDP_FLEX:'IDP'
  };
  function rosterSlotKey(s){
    return ROSTER_SLOT_MAP[String(s || '').toUpperCase()] || null;
  }
  function rosterFromLeague(){
    var rp = cfg.rosterPositions;
    if (!rp || !rp.length) return null;
    var r = { QB:0, SF:0, RB:0, WR:0, TE:0, FLEX:0, K:0, DEF:0, BN:0, IR:0, TAXI:0, IDP:0 };
    rp.forEach(function(s){
      var key = rosterSlotKey(s);
      if (key) r[key]++;
    });
    if (!(r.QB+r.RB+r.WR+r.TE+r.FLEX+r.SF)) return null;  // no usable starters
    return r;
  }
  function defaultRoster(sf, rd){
    if (sf === undefined) sf = state && state.sf;
    if (rd === undefined) rd = state && state.type === 'redraft';
    // Prefer the connected league's actual roster shape when available.
    var lg = rosterFromLeague();
    if (lg){
      // Respect the league's K/DEF roster spots in every format - some dynasty
      // and startup leagues do roster a kicker/defense. Only the no-league
      // fallback below assumes a skill-position-only board.
      // Reconcile with the chosen QB format: a Superflex draft always carries an
      // SF slot and a regular FLEX (superflex is generally just one extra spot),
      // and a 1QB draft drops the SF slot. The league shape only seeds defaults.
      if (sf){ if (!lg.SF) lg.SF = 1; if (!lg.FLEX) lg.FLEX = 1; }
      else   { lg.SF = 0; }
      return lg;
    }
    var starters = 1 + (sf?1:0) + 2 + 3 + 1 + 1 + (rd?1:0) + (rd?1:0);
    var bench = rd ? 6 : Math.max(0, 25 - starters);
    return { QB:1, SF:sf?1:0, RB:2, WR:3, TE:1, FLEX:1, K:rd?1:0, DEF:rd?1:0, BN:bench, IR:0, TAXI:0, IDP:0 };
  }

  // Helper: reconcile a raw roster map for the chosen QB format and return a
  // fresh copy (does not mutate the input).
  function _reconcileRoster(r, sf, rd){
    var out = {};
    ['QB','SF','RB','WR','TE','FLEX','K','DEF','BN','IR','TAXI','IDP'].forEach(function(k){ out[k] = r[k] || 0; });
    if (sf){ if (!out.SF) out.SF = 1; if (!out.FLEX) out.FLEX = 1; }
    else   { out.SF = 0; }
    // K/DEF are kept as-is across formats: if the league (or the user) rosters
    // them, they stay draftable regardless of dynasty/startup/redraft.
    return out;
  }

  // Familiar platform baselines. These are starting points, not claims that
  // every league on a platform uses the same settings; steppers remain editable.
  // Each preset carries the full format it implies - draft type, Superflex,
  // and PPR - so applying one lines the whole setup up, not just the slots.
  var ROSTER_PRESETS = {
    espn:      { label:'ESPN',       type:'redraft', ppr:0.5, QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:1,K:1,DEF:1,BN:8 },
    sleeper:   { label:'Sleeper',    type:'redraft', ppr:0.5, QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:2,K:0,DEF:0,BN:7 },
    yahoo:     { label:'Yahoo',      type:'redraft', ppr:0.5, QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:1,K:1,DEF:1,BN:6 },
    standard:  { label:'Standard',   type:'redraft', ppr:0,   QB:1,SF:0,RB:2,WR:2,TE:1,FLEX:1,K:1,DEF:1,BN:6 },
    sfredraft: { label:'Superflex',  type:'redraft', ppr:0.5, QB:1,SF:1,RB:2,WR:2,TE:1,FLEX:1,K:0,DEF:0,BN:7 },
    bestball:  { label:'Best Ball',  type:'redraft', ppr:0.5, QB:1,SF:0,RB:2,WR:3,TE:1,FLEX:1,K:0,DEF:0,BN:12 },
    dynasty:   { label:'Dynasty SF', type:'startup', ppr:1,   QB:1,SF:1,RB:2,WR:3,TE:1,FLEX:2,K:0,DEF:0,BN:15 },
    dynasty1q: { label:'Dynasty 1QB',type:'startup', ppr:1,   QB:1,SF:0,RB:2,WR:3,TE:1,FLEX:2,K:0,DEF:0,BN:15 }
  };

  function renderSetupRoster(){
    var sf = document.getElementById('drSf').value === '1';
    var rd = document.getElementById('drType').value === 'redraft';
    var rk = document.getElementById('drType').value === 'rookie';
    var leagueRaw = rosterFromLeague();   // null when no league connected
    var hasLeague = !!leagueRaw;

    // In auto mode always re-seed from league (or built-in defaults) so the
    // roster reflects format changes (SF toggle, type change).
    if (_rosterMode === 'auto' || !_setupRoster){
      var seed = hasLeague ? _reconcileRoster(leagueRaw, sf, rd) : defaultRoster(sf, rd);
      seed._sf = sf; seed._rd = rd;
      _setupRoster = seed;
    } else if (_setupRoster._sf !== sf || _setupRoster._rd !== rd){
      // Format changed while in custom mode: reconcile the existing custom config.
      var rec = _reconcileRoster(_setupRoster, sf, rd);
      rec._sf = sf; rec._rd = rd;
      _setupRoster = rec;
    }

    var locked = hasLeague && _rosterMode === 'auto';

    var rows = [
      { key:'QB',   label:'QB' },
      // Superflex is always shown and editable like any other slot; the drSf
      // format toggle just stays in sync with whatever the SF count becomes.
      { key:'SF',   label:'Superflex' },
      { key:'RB',   label:'RB' },
      { key:'WR',   label:'WR' },
      { key:'TE',   label:'TE' },
      { key:'FLEX', label:'FLEX' },
      { key:'K',    label:'K',    hide: rk },
      { key:'DEF',  label:'DEF',  hide: rk },
      { key:'BN',   label:'Bench' },
      { key:'IR',   label:'IR',   hide: !((_setupRoster && _setupRoster.IR) || (leagueRaw && leagueRaw.IR)) },
      { key:'TAXI', label:'Taxi', hide: !((_setupRoster && _setupRoster.TAXI) || (leagueRaw && leagueRaw.TAXI)) },
      { key:'IDP',  label:'IDP',  hide: !((_setupRoster && _setupRoster.IDP) || (leagueRaw && leagueRaw.IDP)) }
    ];

    var presetHtml = '<div class="dr-roster-presets"><span class="dr-roster-presets-label">Presets</span>';
    Object.keys(ROSTER_PRESETS).forEach(function(key){
      var preset = ROSTER_PRESETS[key];
      presetHtml += '<button type="button" class="dr-roster-preset' + (_rosterPreset === key ? ' is-active' : '')
        + '" data-roster-preset="' + key + '">' + preset.label + '</button>';
    });
    presetHtml += '</div>';

    // Source badge + mode-toggle sits outside and immediately above the grid.
    var srcHtml = '';
    if (_rosterPreset && ROSTER_PRESETS[_rosterPreset]){
      srcHtml = '<div class="dr-roster-src">'
        + '<span class="dr-roster-src-tag dr-roster-src-custom">' + ROSTER_PRESETS[_rosterPreset].label + ' preset</span>'
        + '<button type="button" class="dr-roster-src-btn" id="drRosterReset">Reset</button>'
        + '</div>';
    } else if (hasLeague){
      if (locked){
        srcHtml = '<div class="dr-roster-src">'
          + '<span class="dr-roster-src-tag">League settings</span>'
          + '<button type="button" class="dr-roster-src-btn" id="drRosterCustomize">Customize</button>'
          + '</div>';
      } else {
        srcHtml = '<div class="dr-roster-src">'
          + '<span class="dr-roster-src-tag dr-roster-src-custom">Custom</span>'
          + '<button type="button" class="dr-roster-src-btn" id="drRosterReset">Reset to league</button>'
          + '</div>';
      }
    }

    var html = presetHtml + srcHtml + '<div class="dr-setup-roster">';
    rows.forEach(function(r){
      if (r.hide) return;
      var val = _setupRoster[r.key] || 0;
      if (locked){
        // Read-only: just show the value, no steppers.
        html += '<div class="dr-srow">'
          + '<span class="dr-srow-label">' + r.label + '</span>'
          + '<span class="dr-step-val dr-step-val-ro">' + val + '</span>'
          + '</div>';
      } else {
        html += '<div class="dr-srow">'
          + '<span class="dr-srow-label">' + r.label + '</span>'
          + '<div class="dr-stepper">'
          + '<button type="button" class="dr-step-btn" data-key="' + r.key + '" data-d="-1">&#8722;</button>'
          + '<span class="dr-step-val">' + val + '</span>'
          + '<button type="button" class="dr-step-btn" data-key="' + r.key + '" data-d="1">+</button>'
          + '</div></div>';
      }
    });
    html += '</div>';
    document.getElementById('drRosterSection').innerHTML = html;

    // Keep drRounds in sync with roster for non-rookie drafts.
    var _typeEl = document.getElementById('drType');
    if (_typeEl && _typeEl.value !== 'rookie') _syncRoundsFromRoster();

    // Wire the mode-toggle buttons (rendered into innerHTML so must re-attach).
    var cb = document.getElementById('drRosterCustomize');
    if (cb) cb.addEventListener('click', function(){
      _rosterMode = 'custom'; renderSetupRoster();
    });
    var rb = document.getElementById('drRosterReset');
    if (rb) rb.addEventListener('click', function(){
      _rosterMode = 'auto'; _rosterPreset = null; _setupRoster = null; renderSetupRoster();
    });
    Array.prototype.forEach.call(document.querySelectorAll('[data-roster-preset]'), function(btn){
      btn.addEventListener('click', function(){
        var key = btn.getAttribute('data-roster-preset'), preset = ROSTER_PRESETS[key];
        if (!preset) return;
        // A preset defines a whole format, not just slot counts. Apply the draft
        // type first (dispatching change so the rookie/keeper fields, rounds, and
        // capital list all react) before laying in the preset's exact roster.
        if (preset.type){
          var typeEl = document.getElementById('drType');
          if (typeEl && typeEl.value !== preset.type){
            typeEl.value = preset.type;
            typeEl.dispatchEvent(new Event('change'));
          }
        }
        document.getElementById('drSf').value = preset.SF ? '1' : '0';
        if (preset.ppr != null){
          var pprEl = document.getElementById('drPpr');
          if (pprEl) pprEl.value = String(preset.ppr);
        }
        _rosterPreset = key; _rosterMode = 'custom';
        _setupRoster = {};
        ['QB','SF','RB','WR','TE','FLEX','K','DEF','BN'].forEach(function(k){ _setupRoster[k] = preset[k] || 0; });
        _setupRoster.IR = 0; _setupRoster.TAXI = 0; _setupRoster.IDP = 0;
        _setupRoster._sf = !!preset.SF; _setupRoster._rd = preset.type === 'redraft';
        renderSetupRoster(); renderSetupCapital();
      });
    });
  }

  function _stashSlots(rs){
    rs = rs || {};
    return (rs.IR || 0) + (rs.TAXI || 0) + (rs.IDP || 0);
  }
  // Sum of all non-bench roster slots - used to keep rounds and bench in sync.
  function _totalStarterSlots(rs){
    rs = rs || _setupRoster || defaultRoster();
    return (rs.QB||0) + (rs.SF||0) + (rs.RB||0) + (rs.WR||0) + (rs.TE||0) + (rs.FLEX||0) + (rs.K||0) + (rs.DEF||0);
  }
  // Sync the hidden drRounds field from the current roster (starters + bench + stash).
  function _syncRoundsFromRoster(){
    if (!_setupRoster) return;
    var r = _totalStarterSlots(_setupRoster) + (_setupRoster.BN || 0) + _stashSlots(_setupRoster);
    document.getElementById('drRounds').value = Math.max(1, Math.min(40, r));
  }

  // ── Setup: draft capital (claimed picks) ────────────────────────────────────
  function setupCtl(){
    return {
      teams:  parseInt(document.getElementById('drTeams').value, 10) || 12,
      rounds: Math.max(1, Math.min(40, parseInt(document.getElementById('drRounds').value, 10) || 15)),
      order:  document.getElementById('drOrder').value,
      slot:   parseInt(document.getElementById('drSlot').value, 10) || 1
    };
  }
  function defaultSetupOwned(c){
    var o = {};
    for (var r = 1; r <= c.rounds; r++) o[pickNum(r, c.slot, c.teams, c.order)] = true;
    return o;
  }
  // Picks owned in a given round, sorted by overall pick number.
  function roundPicks(r, c){
    var out = [];
    for (var sl = 1; sl <= c.teams; sl++){
      var pn = pickNum(r, sl, c.teams, c.order);
      if (_setupOwned[pn]) out.push(pn);
    }
    return out.sort(function(a, b){ return a - b; });
  }
  // One pick pill (#pn) with a hover remove control.
  function capPill(pn, c){
    var sl = slotOnClock(pn, c.teams, c.order);
    var home = (sl === c.slot);
    return '<span class="dr-cap-pill' + (home ? '' : ' dr-cap-pill-traded') + '" data-rm="' + pn + '" title="' + (home ? 'Your pick' : 'Traded-in pick') + ' &middot; click to remove">'
      + '#' + pn + '<i class="dr-cap-pill-x">&times;</i></span>';
  }
  // Inline slot picker shown under a round when its + button is active.
  function capSlotPicker(r, c){
    var owned = {};
    roundPicks(r, c).forEach(function(pn){ owned[slotOnClock(pn, c.teams, c.order)] = true; });
    var cells = '';
    for (var sl = 1; sl <= c.teams; sl++){
      var pn = pickNum(r, sl, c.teams, c.order);
      var on = !!owned[sl], home = (sl === c.slot);
      cells += '<button type="button" class="dr-cap-slot' + (on ? ' on' : '') + (home ? ' home' : '') + '" data-add="' + pn + '">' + sl + '</button>';
    }
    return '<div class="dr-cap-picker"><span class="dr-cap-picker-lbl">Pick a slot</span><div class="dr-cap-slots">' + cells + '</div></div>';
  }
  // A single round row: label, owned pills, inline add toggle, optional picker.
  function capRow(r, c){
    var pills = roundPicks(r, c).map(function(pn){ return capPill(pn, c); }).join('');
    var open = (_capAddRound === r);
    var h = '<div class="dr-cap-row' + (open ? ' is-open' : '') + '">'
      + '<span class="dr-cap-rlabel">R' + r + '</span>'
      + '<div class="dr-cap-rpicks">' + (pills || '<span class="dr-cap-none">No picks</span>') + '</div>'
      + '<button type="button" class="dr-cap-addbtn" data-addround="' + r + '" aria-label="Add pick to round ' + r + '">' + (open ? '&times;' : '+') + '</button>'
      + '</div>';
    if (open) h += capSlotPicker(r, c);
    return h;
  }
  function renderSetupCapital(){
    var c = setupCtl();
    var sig = [c.teams, c.rounds, c.order, c.slot].join('|');
    if (!_setupOwned || _setupOwnedSig !== sig){
      _setupOwned = defaultSetupOwned(c);   // reset to the slot's natural picks
      _setupOwnedSig = sig;
      _capAddRound = null; _capLateOpen = false;
    }
    var total = Object.keys(_setupOwned).filter(function(k){ return _setupOwned[k]; }).length;
    // Always show R1-R20 individually. Everything from R21 onwards collapses into one section.
    var splitAt = Math.min(20, c.rounds);
    var rows = '';
    for (var r = 1; r <= splitAt; r++) rows += capRow(r, c);
    var late = '';
    if (c.rounds > splitAt){
      var lateCount = 0;
      for (var lr = splitAt + 1; lr <= c.rounds; lr++) lateCount += roundPicks(lr, c).length;
      var lateBody = '';
      if (_capLateOpen){ for (var lr2 = splitAt + 1; lr2 <= c.rounds; lr2++) lateBody += capRow(lr2, c); }
      late = '<div class="dr-cap-late' + (_capLateOpen ? ' is-open' : '') + '">'
        + '<button type="button" class="dr-cap-latehead" id="drCapLateToggle">'
        + '<span class="dr-cap-rlabel">R' + (splitAt + 1) + '-R' + c.rounds + '</span>'
        + '<span class="dr-cap-latecount">' + lateCount + ' pick' + (lateCount === 1 ? '' : 's') + '</span>'
        + '<i class="dr-cap-chev">' + (_capLateOpen ? '&#9650;' : '&#9660;') + '</i></button>'
        + (_capLateOpen ? ('<div class="dr-cap-latebody">' + lateBody + '</div>') : '')
        + '</div>';
    }
    var html = '<div class="dr-cap-head">'
      + '<span class="dr-cap-count">' + total + ' pick' + (total === 1 ? '' : 's') + ' owned</span>'
      + '</div>'
      + '<div class="dr-cap-list">' + rows + late + '</div>';
    document.getElementById('drCapitalSection').innerHTML = html;
  }

  function readSetup(){
    var teams = parseInt(document.getElementById('drTeams').value, 10);
    var sf = document.getElementById('drSf').value === '1';
    // A keeper draft IS a redraft that starts with some picks already spent, so
    // it runs as type 'redraft' (same values, pool, K/DEF rules and pick-score
    // weights) with a keeper flag driving the extra behavior. Giving it its own
    // type would silently fall through to startup weights when grading.
    var rawType = document.getElementById('drType').value;
    var keeper = rawType === 'keeper';
    var rd = keeper || rawType === 'redraft';
    var kcEl = document.getElementById('drKeeperCount');
    var ksEl = document.getElementById('drKeeperSource');
    return {
      type:   keeper ? 'redraft' : rawType,
      keeper: keeper,
      keeperCount:  keeper ? Math.max(0, Math.min(10, parseInt(kcEl && kcEl.value, 10) || 0)) : 0,
      keeperSource: keeper ? ((ksEl && ksEl.value) || 'assistant') : null,
      teams:  teams,
      rounds: Math.max(1, Math.min(40, parseInt(document.getElementById('drRounds').value, 10) || 15)),
      sf:     sf,
      slot:   Math.min(teams, Math.max(1, parseInt(document.getElementById('drSlot').value, 10) || 1)),
      order:  document.getElementById('drOrder').value,
      roster: _setupRoster || defaultRoster(sf, rd),
      scoring: readScoring(),
      picks:  {},
      current: 1,
      queue:  []
    };
  }
  // Scoring settings from setup. These shift the
  // recommended roster build rather than recomputing raw player values.
  function readScoring(){
    var pprEl = document.getElementById('drPpr');
    var tepEl = document.getElementById('drTep');
    var passTdEl = document.getElementById('drPassTd');
    var ppr = pprEl ? parseFloat(pprEl.value) : 1.0;
    var tep = tepEl ? parseFloat(tepEl.value) : 0;
    var passTd = passTdEl ? parseFloat(passTdEl.value) : 4;
    return { ppr: isNaN(ppr) ? 1.0 : ppr, tep: isNaN(tep) ? 0 : tep,
      passTd: passTd >= 6 ? 6 : 4 };
  }
  function scoringCfg(){
    var s = (state && state.scoring) || {};
    return { ppr: s.ppr != null ? s.ppr : 1.0, tep: s.tep != null ? s.tep : 0,
      passTd: s.passTd >= 6 ? 6 : 4 };
  }
  // Convert a Sleeper-style roster_positions array to the {QB:1, RB:2, ...} map
  // used by state.roster. Uses the same normalization as rosterFromLeague so the
  // live/connected path recognizes K/DEF (incl. DST) and FLEX variants identically.
  function _parseRosterPositions(arr){
    if (!Array.isArray(arr) || !arr.length) return null;
    var map = {};
    arr.forEach(function(pos){
      var key = rosterSlotKey(pos);
      if (key) map[key] = (map[key] || 0) + 1;
    });
    return Object.keys(map).length ? map : null;
  }
  // K/DEF enter the pool only when the roster actually carries those slots
  // (always true for redraft; opt-in for startup via the setup steppers).
  function wantsKDef(){
    if (!state) return false;
    if (state.type === 'rookie') return false;   // rookie pools have no K/DEF
    if (state.type === 'redraft') return true;
    // Startup/dynasty: show K/DEF whenever the roster (league or custom) has a slot.
    var rs = state.roster || {};
    return (rs.K || 0) > 0 || (rs.DEF || 0) > 0;
  }

  function startDraft(){
    var prev = state;
    _resetTransient();
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    // Editing the setup mid-draft (e.g. fixing the round count) shouldn't wipe
    // the board. When the pick numbering is unchanged (same teams / order /
    // slot), carry over the picks already made that still fit the new board and
    // resume at the first empty slot. A Reset nulls state first, so a genuine
    // fresh start still begins empty; a change that renumbers picks also starts
    // fresh (carrying by pick number would misplace them).
    if (prev && prev.picks && prev.teams === state.teams &&
        prev.order === state.order && prev.slot === state.slot) {
      var tot = (state.teams || 0) * (state.rounds || 0);
      var carried = {};
      Object.keys(prev.picks).forEach(function(pn){
        var n = parseInt(pn, 10);
        var pk = prev.picks[pn];
        if (n >= 1 && n <= tot && pk) {
          carried[n] = pk;
          if (pk.id) drafted[String(pk.id)] = true;
        }
      });
      if (Object.keys(carried).length) {
        state.picks = carried;
        var next = tot + 1;
        for (var i = 1; i <= tot; i++){ if (!carried[i]){ next = i; break; } }
        state.current = next;
      }
    }
    save();
    resetSideTabs();   // clear any leftover completed-draft sidebar state
    showMain();
    loadPlayers();
  }

  function showMain(){
    _boardSig = null;   // always force a full board rebuild when entering the draft view
    closeEditSetup();
    document.getElementById('drSetup').style.display = 'none';
    var hero = document.getElementById('drHero'); if (hero) hero.style.display = 'none';
    var isLive = !!(state && state.mode === 'live');
    // Practice Mock is only relevant when connected to an upcoming league draft.
    // Edit Setup is hidden during live drafts (settings are locked to the real draft).
    var pm = document.getElementById('drPractice');
    if (pm) pm.style.display = isLive ? '' : 'none';
    var ed = document.getElementById('drEdit');
    if (ed) ed.style.display = isLive ? 'none' : '';
    // Reset becomes "Exit Board" in live mode (no danger color - it's just navigation).
    var rst = document.getElementById('drReset');
    if (rst){
      rst.textContent = isLive ? 'Exit Board' : 'Reset';
      rst.className = isLive ? 'dr-btn dr-btn-ghost' : 'dr-btn dr-btn-ghost dr-btn-danger';
    }
    document.getElementById('drBoard').innerHTML = '';
    document.getElementById('drBaList').innerHTML = loadingNote('Loading players…');
    document.getElementById('drMain').style.display = '';
  }
  function _setElHidden(id, hidden){
    var el = document.getElementById(id);
    if (el) el.hidden = !!hidden;
  }
  // Page setup vs in-draft edit modal: same form, different chrome.
  function setSetupChrome(isModal){
    var setup = document.getElementById('drSetup');
    var hero = document.getElementById('drHero');
    if (setup){
      setup.classList.toggle('dr-setup-is-modal', !!isModal);
      if (isModal){
        setup.setAttribute('role', 'dialog');
        setup.setAttribute('aria-modal', 'true');
        setup.setAttribute('aria-labelledby', 'drEditTitle');
      } else {
        setup.removeAttribute('role');
        setup.removeAttribute('aria-modal');
        setup.removeAttribute('aria-labelledby');
      }
    }
    _setElHidden('drSetupModalHead', !isModal);
    _setElHidden('drEditNote', !isModal);
    _setElHidden('drSetupStartCta', !!isModal);
    _setElHidden('drSetupEditCta', !isModal);
    // Opening the modal hides the hero; closing it does not restore it — the
    // board may still be up. showSetup is what brings the hero back.
    if (isModal && hero) hero.style.display = 'none';
    document.body.classList.toggle('dr-edit-open', !!isModal);
  }
  // Reflect a started draft's settings in the setup controls so Edit (and a
  // later Apply with no changes) reads back the same teams/order/slot and the
  // picks can carry over (see startMock / startDraft). No-op before a draft.
  function hydrateSetupFromState(){
    if (!(state && state.teams)) return;
    var tEl = document.getElementById('drType');
    if (tEl) tEl.value = state.keeper ? 'keeper' : (state.type || tEl.value);
    var sfEl = document.getElementById('drSf'); if (sfEl && state.sf != null) sfEl.value = state.sf ? '1' : '0';
    var teamsEl = document.getElementById('drTeams'), wt = String(state.teams);
    if (teamsEl){
      for (var ti = 0; ti < teamsEl.options.length; ti++){ if (teamsEl.options[ti].value === wt || teamsEl.options[ti].text === wt){ teamsEl.selectedIndex = ti; break; } }
    }
    var ordEl = document.getElementById('drOrder'); if (ordEl && state.order) ordEl.value = state.order;
    var rEl = document.getElementById('drRounds'); if (rEl && state.rounds) rEl.value = String(state.rounds);
    fillSlotOptions(state.teams);   // slot options depend on team count
    var slotEl = document.getElementById('drSlot'); if (slotEl && state.slot) slotEl.value = String(state.slot);
    if (state.scoring){
      var pprEl = document.getElementById('drPpr'); if (pprEl) pprEl.value = String(state.scoring.ppr != null ? state.scoring.ppr : 1);
      var tepEl = document.getElementById('drTep'); if (tepEl) tepEl.value = String(state.scoring.tep != null ? state.scoring.tep : 0);
      var passTdEl = document.getElementById('drPassTd'); if (passTdEl) passTdEl.value = String(state.scoring.passTd >= 6 ? 6 : 4);
    }
    if (state.keeper){
      var kcEl = document.getElementById('drKeeperCount'); if (kcEl && state.keeperCount != null){ kcEl.value = String(state.keeperCount); kcEl.dataset.touched = '1'; }
      var ksEl = document.getElementById('drKeeperSource'); if (ksEl && state.keeperSource) ksEl.value = state.keeperSource;
    }
    // Roster editor: seed from the active draft in custom mode so it is not
    // re-seeded from league/defaults. Match the sf/rd markers to the controls
    // just set so renderSetupRoster does not reconcile it.
    if (state.roster){
      var rr = {}; Object.keys(state.roster).forEach(function(k){ rr[k] = state.roster[k]; });
      rr._sf = document.getElementById('drSf').value === '1';
      rr._rd = document.getElementById('drType').value === 'redraft';
      _setupRoster = rr; _rosterMode = 'custom';
    }
    // Draft-capital editor: seed owned picks and match the signature so it is
    // not reset to the slot's natural picks.
    if (state.owned){
      var ow = {}; Object.keys(state.owned).forEach(function(k){ if (state.owned[k]) ow[k] = true; });
      _setupOwned = ow;
      var _c = setupCtl(); _setupOwnedSig = [_c.teams, _c.rounds, _c.order, _c.slot].join('|');
    }
  }
  function refreshSetupEditors(){
    var typeEl = document.getElementById('drType');
    var typeVal = typeEl ? typeEl.value : '';
    syncKeeperSetupFields(typeVal === 'keeper');
    var _rf = document.getElementById('drRoundsField');
    if (_rf) _rf.style.display = (typeVal === 'rookie') ? '' : 'none';
    renderSetupRoster();
    renderSetupCapital();
  }
  function closeEditSetup(){
    var setup = document.getElementById('drSetup');
    if (!setup || !setup.classList.contains('dr-setup-is-modal')){
      document.body.classList.remove('dr-edit-open');
      return;
    }
    setup.style.display = 'none';
    setSetupChrome(false);
  }
  function openEditSetup(){
    if (!state || !state.teams || state.mode === 'live') return;
    endSim();
    hydrateSetupFromState();
    refreshSetupEditors();
    setSetupChrome(true);
    document.getElementById('drSetup').style.display = '';
    var closeBtn = document.getElementById('drEditClose');
    if (closeBtn) closeBtn.focus();
  }
  function applyEditedSetup(){
    if (!state) return;
    var wasMock = state.mode === 'mock';
    var wasStarted = !!(state.simStarted || (sim && simStarted));
    var prevTeams = state.teams, prevOrder = state.order, prevSlot = state.slot;
    var nextTeams = parseInt(document.getElementById('drTeams').value, 10);
    var nextOrder = document.getElementById('drOrder').value;
    var nextSlot = parseInt(document.getElementById('drSlot').value, 10) || 1;
    var hasPicks = !!(state.picks && Object.keys(state.picks).some(function(k){ return !!state.picks[k]; }));
    var willWipe = hasPicks && (prevTeams !== nextTeams || prevOrder !== nextOrder || prevSlot !== nextSlot);
    function go(){
      closeEditSetup();
      if (wasMock) startMock();
      else startDraft();
      var same = state && state.teams === prevTeams && state.order === prevOrder && state.slot === prevSlot;
      if (wasMock && wasStarted && same){
        simStarted = true;
        state.simStarted = true;
        syncSimControls();
      }
    }
    if (willWipe){
      drConfirm('Changing teams, pick order, or your slot restarts the board and clears picks.', 'Apply', go);
    } else {
      go();
    }
  }
  function showSetup(){
    endSim();
    closeEditSetup();
    document.getElementById('drMain').style.display = 'none';
    document.getElementById('drSetup').style.display = '';
    setSetupChrome(false);
    var hero = document.getElementById('drHero'); if (hero) hero.style.display = '';
    hydrateSetupFromState();
    refreshSetupEditors();
  }

  // ── Data ─────────────────────────────────────────────────────────────────
  function finiteVal(v){
    var n = Number(v);
    return isFinite(n) ? n : 0;
  }
  function redraftVal(p){
    if (!p) return 0;
    var v = (state.sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
                      : p.redraft_value_1qb);
    return finiteVal(v);
  }
  function valOf(p){
    if (!p) return 0;
    if (state.type === 'redraft') return redraftVal(p);
    // p.val is the stripped pick-object field (stored by commitPick/applyLivePicks);
    // p.value is the full player-object field from /api/league-players.
    var v = state.sf ? (p.sf_value || p.value || p.val || 0) : (p.value || p.val || 0);
    return finiteVal(v);
  }
  function adpOf(p){
    // Sleeper community ADP (server-side, aggregated from real Sleeper drafts).
    // Redraft has no Sleeper feed, so it falls back to a value-derived rank.
    if (state.type === 'rookie') return state.sf ? p.sf_rookie_avg_pick : p.rookie_avg_pick;
    if (state.type === 'redraft'){
      var ra = state.sf ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
      return (ra != null) ? ra : (p._radp != null ? p._radp : null);
    }
    return state.sf ? p.sf_avg_pick : p.avg_pick;
  }
  // Consensus ADP from the per-source payload (blended Sleeper/BR/ESPN/MFL/Yahoo),
  // independent of the ADP-source dropdown. Used by Deep Dive's Value vs ADP
  // chart so a BR Fantasy or Sleeper-only board still plots against the market.
  // Null when consensus isn't on the player (rookie axis, historical overlay,
  // or a source-less payload) — callers fall back to adpOf().
  function consensusAdpOf(p){
    if (!p || !state) return null;
    if (state.type === 'rookie') return null;
    if (state.mode === 'live' && state.isComplete && state.season && cfg.season
        && Number(state.season) !== Number(cfg.season)) return null;
    var by = p.adp_by_source && p.adp_by_source.consensus;
    if (!by) return null;
    var field = state.type === 'redraft'
      ? (state.sf ? 'sf_redraft_avg_pick' : 'redraft_avg_pick')
      : (state.sf ? 'sf_avg_pick' : 'avg_pick');
    var v = by[field];
    if (v == null || !isFinite(Number(v))) return null;
    return Number(v);
  }

  function loadPlayers(){
    var params = wantsKDef() ? ['kdef=1'] : [];
    // Historical/synced completed draft: grade against THAT season's ADP. The
    // server overlays that season's Sleeper ADP and no-ops for the current one.
    if (state && state.mode === 'live' && state.isComplete && state.season){
      params.push('season=' + encodeURIComponent(state.season));
    }
    // Explicit ADP source (from the source selector). "auto" keeps the server
    // default; any real source (incl. consensus) overlays via the resolver.
    // Carry league context so Yahoo/consensus can resolve a league token.
    if (adpSource && adpSource !== 'auto'){
      params.push('adp_source=' + encodeURIComponent(adpSource));
      if (cfg.leagueId) params.push('league_id=' + encodeURIComponent(cfg.leagueId));
      if (cfg.platform) params.push('platform=' + encodeURIComponent(cfg.platform));
    }
    var url = '/api/league-players' + (params.length ? ('?' + params.join('&')) : '');
    var _loadPlayerPayload = function(attempt){
      return fetch(url, { cache: 'no-store' }).then(function(r){
        return r.text().then(function(body){
          var payload;
          try { payload = JSON.parse(body); }
          catch (_jsonErr) { throw new Error('Player API returned non-JSON (HTTP ' + r.status + ')'); }
          if (!r.ok) {
            var detail = payload && (payload.error || payload.message);
            throw new Error('Player API HTTP ' + r.status + (detail ? ': ' + detail : ''));
          }
          return payload;
        });
      }).catch(function(err){
        // Render can briefly return 502/503 while a worker is restarting. One
        // bounded retry fixes that transient case without creating a request loop.
        if (attempt < 1) return new Promise(function(resolve){
          setTimeout(function(){ resolve(_loadPlayerPayload(attempt + 1)); }, 600);
        });
        throw err;
      });
    };
    _loadPlayerPayload(0)
      .then(function(resp){
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        if (!Array.isArray(raw) || !raw.length) {
          throw new Error((resp && (resp.error || resp.message)) || 'Player API returned an empty player pool');
        }
        tierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};
        adpSources = (!Array.isArray(resp) && resp.adp_sources) ? resp.adp_sources : {};
        if (!Array.isArray(resp) && resp.adp_source_options) adpSourceOptions = resp.adp_source_options;
        players = raw.filter(function(p){
          if (!p || p.id == null) return false;
          var pos = String(p.position || '').toUpperCase();
          if (pos === 'PICK') return false;
          if (state.type === 'rookie') return !!p.is_rookie;
          if (pos === 'K' || pos === 'DEF') return wantsKDef();
          if (state.type === 'redraft') return redraftVal(p) > 0;
          return ['QB','RB','WR','TE'].indexOf(pos) >= 0 || p.is_rookie;
        });
        // Derive a stable redraft ADP rank (1 = top redraft value).
        if (state.type === 'redraft'){
          players.slice().sort(function(a, b){ return redraftVal(b) - redraftVal(a); })
            .forEach(function(p, i){ p._radp = i + 1; });
        }
        playersById = {};
        players.forEach(function(p){ playersById[String(p.id)] = p; });
        // Live mode: re-apply picks now that values are available; else rebuild
        // the drafted set from saved picks.
        if (state.mode === 'live' && lastLivePicks){
          applyLivePicks(lastLivePicks);
        } else {
          drafted = {};
          Object.keys(state.picks).forEach(function(k){ var pp = state.picks[k]; if (pp) drafted[String(pp.id)] = true; });
        }
        applyKeepers();   // no-op unless league keepers are enabled
        // Keeper draft: spend each keeper's cost-round pick before play starts.
        seedKeeperPicks();
        // Re-render the banner now that playersById exists, so keepers that
        // arrived as bare ids from the handoff can show a real name/position.
        renderKeeperBanner();
        render();
        if (sim) scheduleSim();   // begin CPU picks once players are loaded
      })
      .catch(function(err){
        console.error('[draft-room] loadPlayers failed', err);
        var host = document.getElementById('drBaList');
        if (!host) return;
        host.innerHTML = emptyNote('Couldn’t load players', ((err && err.message) || 'Something went wrong. Tap Retry to try again.'));
        var retry = document.createElement('button'); retry.type = 'button';
        retry.className = 'dr-btn dr-btn-sm'; retry.textContent = 'Retry';
        retry.style.marginTop = '10px'; retry.addEventListener('click', loadPlayers);
        var note = host.querySelector('.dr-empty-note');
        if (note) note.appendChild(retry);
      });
  }

  // ── Keepers ──────────────────────────────────────────────────────────────
  // League keepers (yours + projections for other teams) come from the keeper
  // tool via __draftCfg.keepers. Applying them just marks those players as
  // already-drafted, so they drop out of the best-available pool. Everything
  // here is a no-op when there are no keepers, so the draft room is unchanged
  // for non-keeper leagues.
  // Both keeper sources are kept so the setup's "Keepers" control can switch
  // between them: the assistant's league-wide projection, and the picks you
  // actually chose on the keeper page (handed off in sessionStorage).
  var keeperProjected = [];
  var keeperOverride = null;   // {rosterId, ids} for this league, if handed off

  function initKeepers(){
    var kp = cfg.keepers;
    if (!kp) return;   // not a keeper league and not arrived from the keeper tool
    // The projection may legitimately be empty (e.g. no team has a
    // positive-surplus keeper); your own picks stand on their own.
    keeperProjected = Array.isArray(kp.kept) ? kp.kept.slice() : [];
    try {
      var ovRaw = sessionStorage.getItem('brKeeperOverride');
      if (ovRaw){
        var ov = JSON.parse(ovRaw);
        // Fall back to the roster stashed with the handoff: when the session
        // has no viewer_roster_id the server sends viewerRoster null, which
        // used to discard the user's actual keeper picks and leave only the
        // league-wide projection.
        var vr = String((kp.viewerRoster != null ? kp.viewerRoster : ov && ov.rosterId) || '');
        if (ov && String(ov.leagueId) === String(cfg.leagueId) && vr){
          // players[] carries each keeper's resolved cost round (escalation +
          // collision bumps) from the keeper page; keyed by id for lookup.
          var detail = {};
          (ov.players || []).forEach(function(p){ if (p && p.id != null) detail[String(p.id)] = p; });
          keeperOverride = { rosterId: vr, ids: (ov.ids || []).map(String), detail: detail };
        }
      }
    } catch (e) { /* ignore malformed override */ }
    keeperSet = computeKeeperSet();
    if (!keeperSet.length) return;
    keepersOn = true;
    renderKeeperBanner();
  }

  // Effective keepers for the current setup. "Pick my own" uses the selections
  // handed off from the keeper page for your team; "Use Keeper Assistant" uses
  // the optimizer's projection for every team including yours. Rival teams are
  // always projected, capped at the league's keepers-per-team.
  function computeKeeperSet(){
    var out = keeperProjected.slice();
    var source = (state && state.keeperSource) || 'manual';
    if (source !== 'assistant' && keeperOverride){
      var vr = keeperOverride.rosterId;
      var meta = {};
      out.forEach(function(k){ if (String(k.rosterId) === vr) meta[String(k.id)] = k; });
      var detail = keeperOverride.detail || {};
      var mine = keeperOverride.ids.map(function(id){
        var base = meta[String(id)] || { id: String(id), rosterId: vr, projected: false };
        var d = detail[String(id)];
        if (d){
          // Prefer the keeper page's resolved cost/name/pos (it knows the
          // per-player years-kept the server projection can't).
          base = { id: String(id), rosterId: vr, projected: false,
                   costRound: d.costRound != null ? d.costRound : base.costRound,
                   name: d.name != null ? d.name : base.name,
                   pos: d.pos != null ? d.pos : base.pos };
        }
        return base;
      });
      mine.forEach(function(m){ m.projected = false; });
      out = out.filter(function(k){ return String(k.rosterId) !== vr; }).concat(mine);
    }
    // Cap each rival team at the league's keepers-per-team. Your own count is
    // whatever you actually chose.
    var cap = state && state.keeper ? state.keeperCount : null;
    if (cap != null && cap >= 0){
      var myR = keeperOverride && keeperOverride.rosterId;
      var seen = {};
      out = out.filter(function(k){
        var rid = String(k.rosterId);
        if (myR && rid === myR) return true;
        seen[rid] = (seen[rid] || 0) + 1;
        return seen[rid] <= cap;
      });
    }
    return out;
  }

  function applyKeepers(){
    if (!keepersOn) return;
    keeperSet.forEach(function(k){ if (k && k.id != null) drafted[String(k.id)] = true; });
  }

  // ── Keeper drafts ────────────────────────────────────────────────────────
  // In a keeper draft each kept player costs his team the pick at his keeper
  // round, so those picks are spent before the draft starts. Seeding them onto
  // the board (rather than only hiding the players) makes the pick economy real:
  // teams draft fewer times, the rounds line up, and because a keeper occupies a
  // genuine pick slot it flows into the draft grade exactly like any other pick
  // - which is the point of a keeper, a stud held at a late round grades great.

  // Map a keeper's roster to a draft seat. The viewer's own seat is known; rival
  // rosters get a stable, deterministic seat so each team loses picks in the
  // right rounds. Seat identity is approximate in a mock, the pick economy is not.
  function keeperSlotMap(){
    var map = {};
    var vr = cfg.keepers && cfg.keepers.viewerRoster;
    var mySlot = (state && state.slot) || 1;
    if (vr != null && vr !== '') map[String(vr)] = mySlot;
    var rosters = Object.keys((cfg.keepers && cfg.keepers.byTeam) || {})
      .filter(function(rid){ return String(rid) !== String(vr); })
      .sort(function(a, b){ return String(a).localeCompare(String(b), undefined, { numeric: true }); });
    var teams = (state && state.teams) || 12;
    var free = [];
    for (var s = 1; s <= teams; s++) if (s !== mySlot) free.push(s);
    rosters.forEach(function(rid, i){ if (i < free.length) map[String(rid)] = free[i]; });
    return map;
  }

  function seedKeeperPicks(){
    if (!state || !state.keeper || !keepersOn) return;
    // Re-derive now that the draft's keeper source and per-team cap are known
    // (initKeepers runs at page load, before any draft is configured).
    keeperSet = computeKeeperSet();
    keeperSet.forEach(function(k){ if (k && k.id != null) drafted[String(k.id)] = true; });
    if (!keeperSet.length) return;
    var teams = state.teams, rounds = state.rounds, order = state.order;
    var slotBy = keeperSlotMap();
    var used = {};
    keeperSet.forEach(function(k){
      if (!k || k.id == null) return;
      var slot = slotBy[String(k.rosterId)];
      if (!slot) return;                                  // unknown seat: player just stays off the board
      var rnd = parseInt(k.costRound, 10);
      if (!rnd || rnd < 1 || rnd > rounds) return;        // cost outside this draft
      // Two keepers can't spend the same pick; bump to the next open round for
      // that seat, mirroring how leagues resolve a cost collision.
      var pn = null;
      for (var r = rnd; r <= rounds; r++){
        var cand = pickNum(r, slot, teams, order);
        if (!used[cand] && !state.picks[cand]){ pn = cand; break; }
      }
      if (pn == null) return;
      used[pn] = true;
      var p = playersById[String(k.id)] || {};
      state.picks[pn] = {
        id: k.id,
        name: k.name || p.name || ('Player ' + k.id),
        position: k.pos || p.position || '',
        team: p.team || '',
        val: Math.round(p.id != null ? valOf(p) : 0),
        ps: (p.id != null ? pickScoreFor(p, pn) : null),
        reason: 'Keeper (R' + rnd + ')',
        keeper: true
      };
      drafted[String(k.id)] = true;
    });
    skipFilledPicks();
  }

  // Keeper picks are already on the board, so the clock must step over them.
  function skipFilledPicks(){
    if (!state) return;
    var total = state.teams * state.rounds;
    while (state.current <= total && state.picks[state.current]) state.current++;
  }

  function setKeepersOn(on){
    keepersOn = on;
    if (on){ applyKeepers(); }
    else { keeperSet.forEach(function(k){ if (k && k.id != null) delete drafted[String(k.id)]; }); }
    render();
    renderKeeperBanner();
  }

  function renderKeeperBanner(){
    if (!keeperSet.length) return;
    var wrap = document.querySelector('.dr-wrap');
    if (!wrap) return;
    var el = document.getElementById('drKeeperBanner');
    if (!el){
      el = document.createElement('div');
      el.id = 'drKeeperBanner';
      el.className = 'dr-keeper-banner';
      wrap.insertBefore(el, wrap.firstChild);
    }
    var mine = keeperSet.filter(function(k){ return !k.projected; }).length;
    var proj = keeperSet.length - mine;
    var esc = function(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); };
    // Your own handed-off picks carry only an id when the server's projection
    // didn't include them for your team, so fill name/position from the loaded
    // player pool rather than rendering a raw "Player 8134".
    var rows = keeperSet.slice().sort(function(a,b){ return (a.projected?1:0) - (b.projected?1:0); }).map(function(k){
      var meta = playersById[String(k.id)] || {};
      var nm  = k.name || meta.name;
      var pos = k.pos  || meta.position;
      var tag = k.projected
        ? '<span class="dr-keeper-tag proj">projected</span>'
        : '<span class="dr-keeper-tag mine">your keeper</span>';
      var cost = k.costRound ? (' · R' + k.costRound) : '';
      return '<div class="dr-keeper-item"><span>' + esc(nm || ('Player ' + k.id)) +
        (pos ? ' <span class="dr-keeper-pos">' + esc(pos) + '</span>' : '') +
        cost + '</span>' + tag + '</div>';
    }).join('');
    // Keep the details panel open across re-renders (the toggle rebuilds this
    // markup, which would otherwise collapse the list the user just opened).
    var _wasOpen = (function(){ var l = document.getElementById('drKeeperList'); return l && !l.hidden; })();
    el.innerHTML =
      '<div class="dr-keeper-head">' +
        '<b>Keepers ' + (keepersOn ? 'applied' : 'off') + '</b>' +
        '<span class="dr-keeper-sub">' + keeperSet.length + ' off the board · ' +
          mine + ' yours, ' + proj + ' projected</span>' +
        '<button type="button" id="drKeeperView" class="dr-keeper-btn">Details</button>' +
        '<button type="button" id="drKeeperToggle" class="dr-keeper-btn">' +
          (keepersOn ? 'Turn off' : 'Apply') + '</button>' +
      '</div>' +
      '<div id="drKeeperList" class="dr-keeper-list"' + (_wasOpen ? '' : ' hidden') + '>' + rows +
        '<div class="dr-keeper-note">Other teams’ keepers are projected from the same surplus model. They are estimates, not their declared keepers.</div>' +
      '</div>';
    var vbtn = document.getElementById('drKeeperView');
    var tbtn = document.getElementById('drKeeperToggle');
    var list = document.getElementById('drKeeperList');
    if (vbtn && list) vbtn.addEventListener('click', function(){ list.hidden = !list.hidden; });
    if (tbtn) tbtn.addEventListener('click', function(){ setKeepersOn(!keepersOn); });
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render(){
    // No draft yet (setup screen): there is nothing to draw, and every renderer
    // below dereferences state. Bailing here keeps callers that can fire before
    // a draft exists - e.g. the keeper banner's Turn off / Apply toggle - from
    // throwing and silently aborting the rest of their work.
    if (!state) return;
    if (!state.queue) state.queue = [];
    renderStatus(); renderBoard(); renderSide(); justPick = null; save();
    pushCheatSheetContext();
    var _tot = state.teams * state.rounds;
    // Draft is over once current passes the last pick - open the summary regardless
    // of sim state (when the user makes the final pick, sim is still true here).
    if (state.current > _tot && !_summaryShown && hasOwned()){
      _summaryShown = true;
      setTimeout(openSummary, 500);
    }
  }

  // ── Simulation (mock draft) ─────────────────────────────────────────────────
  function simAdp(p){
    var a = adpOf(p);
    // In SF, if sf_avg_pick is missing for a QB but standard avg_pick exists, use
    // a deflated version (QBs are ~30% more valuable in SF so their pick comes earlier).
    if (a == null && state.sf && (p.position || '').toUpperCase() === 'QB' && p.avg_pick != null){
      a = Math.max(1, p.avg_pick * 0.70);
    }
    if (a != null) return a;
    var pos = (p.position || '').toUpperCase();
    // K/DEF with no real ADP: synthesize one in the last few rounds, spread by
    // quality so they are not a single last-two-round clump. Defenses fan out
    // earlier (elite D/ST can go 4-6 rounds out); kickers stay later.
    if (pos === 'K' || pos === 'DEF'){
      var tot = (state.teams || 12) * (state.rounds || 16);
      var teamsN = state.teams || 12;
      var sc = _ppgScale[pos], v = ppgOf(p);
      var n = (sc && v != null && sc.elite > sc.repl) ? clamp01((v - sc.repl) / (sc.elite - sc.repl)) : 0.4;
      var span = pos === 'DEF' ? 5.5 : 3.8;
      var jitter = (_rand01(String(p.id) + ':kdadp') - 0.5) * teamsN * 0.9;
      var slotAdp = tot - Math.round(teamsN * span * (0.12 + 0.88 * n)) + jitter;
      return Math.max(tot - teamsN * (span + 1), Math.min(tot, slotAdp));
    }
    return 10000 - (valOf(p) / 100);  // other ADP-less players sort after, by value
  }
  function teamCounts(slot){
    var c = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    Object.keys(state.picks).forEach(function(k){
      if (slotOnClock(parseInt(k,10), state.teams, state.order) === slot){
        var pos = (state.picks[k].position||'').toUpperCase(); if (c[pos] != null) c[pos]++;
      }
    });
    return c;
  }
  // Spread of a player's real draft slot around their ADP. Tight at the very top
  // of the board (consensus picks barely move) and widens deeper, where ADP is
  // noisier. Drives how far a player realistically slides from their ADP.
  function simSigma(a){ return Math.max(0.5, Math.min(10, 0.35 + 0.055 * a)); }
  // Each CPU team gets a persistent draft personality that scales how hard it
  // leans into roster need/scarcity vs pure best-available value, so they don't
  // all reach the same way. Assigned once per draft and saved with state, so a
  // team behaves consistently within a draft but the mix differs between drafts.
  //   ~0.2 = value-first (best player available)   ~0.85 = balanced   ~1.5 = need-first
  function _simPersona(slot){
    if (!state.simPersonas) state.simPersonas = {};
    if (state.simPersonas[slot] == null){
      var r = Math.random(), f;
      if (r < 0.30)      f = 0.15 + Math.random() * 0.30;   // ~30% BPA / value-first
      else if (r < 0.75) f = 0.60 + Math.random() * 0.50;   // ~45% balanced
      else               f = 1.15 + Math.random() * 0.65;   // ~25% need / roster-builder
      state.simPersonas[slot] = Math.round(f * 100) / 100;
    }
    return state.simPersonas[slot];
  }
  // Per-draft random seed so each mock plays out differently while staying stable
  // within a single draft (saved with state, regenerated on a fresh mock).
  function _simSeed(){ if (state.simSeed == null) state.simSeed = Math.floor(Math.random() * 2147483647); return state.simSeed; }
  // Deterministic [0,1) hash from the draft seed + an arbitrary key, so the same
  // key gives the same value all draft long but differs between drafts.
  function _rand01(key){
    var h = _simSeed() | 0, s = String(key);
    for (var i = 0; i < s.length; i++){ h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0; }
    h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995); h ^= h >>> 15;
    return ((h >>> 0) % 1000000) / 1000000;
  }
  // Per-draft ADP shift: nudges each player's effective ADP a little so the board
  // order isn't identical every draft (some players rise, some slide). Bounded and
  // scaled by ADP so elite picks barely move and deeper players vary more. Roughly
  // Gaussian (sum of three uniforms) and never applied to ADP-less sentinels.
  function _adpNoise(p, adp){
    if (adp == null || adp >= 9000) return 0;
    var id = p.id;
    var g = (_rand01(id + ':a') + _rand01(id + ':b') + _rand01(id + ':c') - 1.5) * 2;  // ~N(0,1)
    var sd = Math.max(0.5, Math.min((state.teams || 12) * 1.0, 0.12 * adp));
    return g * sd;
  }
  // Per-team positional lean: each CPU team slightly favors some positions over
  // others (0.85-1.15x), giving teams distinct character without overriding value.
  function _simBias(slot){
    if (!state.simBias) state.simBias = {};
    if (!state.simBias[slot]){
      var b = {};
      ['QB','RB','WR','TE'].forEach(function(pos){
        b[pos] = Math.round((0.85 + _rand01(slot + ':bias:' + pos) * 0.30) * 100) / 100;
      });
      state.simBias[slot] = b;
    }
    return state.simBias[slot];
  }
  // Named draft strategies: a persistent early-round game plan per CPU team on
  // top of the softer persona/bias. Assigned once per draft from the seeded RNG
  // and saved with state, so a team executes its plan consistently all draft.
  // 'bpa' teams have no plan; the rest shape the first rounds then fade, and
  // every existing sanity guard (overcrowding, backup-reach, K/DEF fill) still
  // applies on top so a plan can never produce a broken roster.
  function _simStrategy(slot){
    if (!state.simStrats) state.simStrats = {};
    if (!state.simStrats[slot]){
      var r = _rand01(slot + ':strat');
      var s;
      if (state.type === 'rookie'){
        // Rookie drafts are short and value-driven: only the simple
        // RB-first/WR-first leans make sense.
        s = r < 0.55 ? 'bpa' : (r < 0.78 ? 'rb_heavy' : 'wr_heavy');
      } else if (r < 0.35) s = 'bpa';
      else if (r < 0.48) s = 'rb_heavy';
      else if (r < 0.61) s = 'wr_heavy';
      else if (r < 0.74) s = 'zero_rb';
      else if (r < 0.83) s = 'hero_rb';
      else if (r < 0.92) s = 'elite_te';
      else               s = 'early_qb';
      state.simStrats[slot] = s;
    }
    return state.simStrats[slot];
  }
  function stratLabel(s){
    return { rb_heavy: 'RB heavy', wr_heavy: 'WR heavy', zero_rb: 'Zero RB',
             hero_rb: 'Hero RB', elite_te: 'Elite TE', early_qb: 'Early QB' }[s] || '';
  }
  // Dynasty age lean: a second, independent axis for startup drafts. A win-now
  // team pays up for proven vets and fades rookies; a youth team punts the
  // short term and hoards early-20s upside. Orthogonal to the positional plan,
  // so "Zero RB + win now" and "RB heavy + youth" both occur. Startup only:
  // redraft has no age dimension and rookie classes are uniformly young.
  function _simAgeLean(slot){
    if (state.type === 'redraft' || state.type === 'rookie') return 'neutral';
    if (!state.simAgeLeans) state.simAgeLeans = {};
    if (!state.simAgeLeans[slot]){
      var r = _rand01(slot + ':agelean');
      state.simAgeLeans[slot] = r < 0.55 ? 'neutral' : (r < 0.78 ? 'win_now' : 'youth');
    }
    return state.simAgeLeans[slot];
  }
  // Age-lean intensity: like the strategy params, each team commits to its
  // rebuild/win-now identity to a different degree per draft.
  function _ageLeanIntensity(slot){
    if (!state.simAgeInt) state.simAgeInt = {};
    if (state.simAgeInt[slot] == null){
      state.simAgeInt[slot] = Math.round((0.70 + _rand01(slot + ':ageint') * 0.60) * 100) / 100;
    }
    return state.simAgeInt[slot];
  }
  function _ageLeanMult(lean, pos, age, I){
    if (!lean || lean === 'neutral') return 1;
    if (pos === 'K' || pos === 'DEF') return 1;
    if (age == null) return 1;
    I = I || 1;
    if (lean === 'win_now'){
      if (age >= 27) return _scaleMult(1.3, I);
      if (age >= 25) return _scaleMult(1.15, I);
      if (age <= 23) return _scaleMult(0.7, I);
      return 1;
    }
    // youth: hoard early-20s upside, punt the aging core entirely.
    if (age <= 23) return _scaleMult(1.5, I);
    if (age <= 25) return _scaleMult(1.15, I);
    if (age >= 30) return _scaleMult(0.35, I);
    if (age >= 27) return _scaleMult(0.55, I);
    return 1;
  }
  // Per-team execution style for the strategy: no two teams (or drafts) run
  // the same plan identically. Intensity scales how hard the multipliers pull
  // (0.7 = half-hearted, 1.35 = doctrinaire) and shift moves the plan's round
  // windows by -1/0/+1. Seeded per slot per draft, saved with state.
  function _stratParams(slot){
    if (!state.simStratParams) state.simStratParams = {};
    if (!state.simStratParams[slot]){
      state.simStratParams[slot] = {
        intensity: Math.round((0.70 + _rand01(slot + ':strint') * 0.65) * 100) / 100,
        shift: Math.floor(_rand01(slot + ':strwin') * 3) - 1,
      };
    }
    return state.simStratParams[slot];
  }
  // Per-team special-teams plan so CPU mocks do not all take K then DEF with
  // the last two picks. Window is rounds-from-the-end when they start looking;
  // prefer/flip choose K vs DEF order; split leaves a skill pick in between.
  function _simKDefPlan(slot){
    if (!state.simKDefPlans) state.simKDefPlans = {};
    if (!state.simKDefPlans[slot]){
      var r = _rand01(slot + ':kdef:pref');
      var prefer = r < 0.40 ? 'DEF' : (r < 0.72 ? 'K' : 'mix');
      var w = _rand01(slot + ':kdef:win');
      var winRds;
      if (w < 0.06) winRds = 7;
      else if (w < 0.16) winRds = 6;
      else if (w < 0.32) winRds = 5;
      else if (w < 0.55) winRds = 4;
      else if (w < 0.82) winRds = 3;
      else winRds = 2;
      state.simKDefPlans[slot] = {
        prefer: prefer,
        window: winRds,
        split: _rand01(slot + ':kdef:split') < 0.58,
        intensity: Math.round((0.55 + _rand01(slot + ':kdef:int') * 0.95) * 100) / 100,
        flip: _rand01(slot + ':kdef:flip') < 0.22,
        order: _rand01(slot + ':kdef:order'),
      };
    }
    return state.simKDefPlans[slot];
  }
  // Scale a multiplier's deviation from 1 by the team's intensity, floored so
  // an aggressive fade can't hit zero.
  function _scaleMult(base, intensity){
    return Math.max(0.1, 1 + (base - 1) * intensity);
  }
  // Strategy freedom scales with the price of executing it. A boost on a player
  // already at/near pick value applies in full (at 1.01 a Hero RB team can take
  // whichever elite RB it wants; they're all at value there), while a boost that
  // requires reaching past ADP fades linearly and is gone by about a round of
  // reach - so a plan never forces taking the ADP-14 RB at pick 7 just to stay
  // on script. Fades (mult < 1) pass through: passing on a player costs nothing.
  function _stratReachDamp(mult, adpEff, pn){
    if (mult <= 1 || adpEff == null || adpEff >= 9000) return mult;
    var teams = state.teams || 12;
    var freeR = teams * 0.25, spanR = teams * 0.75;
    var reach = Math.max(0, adpEff - pn);
    if (reach <= freeR) return mult;
    var damp = Math.max(0, 1 - (reach - freeR) / spanR);
    return 1 + (mult - 1) * damp;
  }
  // Weight multiplier a strategy applies to a candidate: pos, how many of that
  // position the team already has, the current round, and the team's execution
  // params (intensity + window shift).
  function _stratMult(strat, pos, have, round, prm){
    if (!strat || strat === 'bpa') return 1;
    var I = prm ? prm.intensity : 1;
    var S = prm ? prm.shift : 0;
    if (strat === 'rb_heavy'){
      if (round <= Math.max(1, 3 + S)){
        if (pos === 'RB') return _scaleMult(1.6, I);
        if (pos === 'WR') return _scaleMult(0.8, I);
      }
      return 1;
    }
    if (strat === 'wr_heavy'){
      // WR-first without the structural RB fade of zero_rb: load up on
      // receivers early but still take a value RB when one falls.
      if (round <= Math.max(1, 3 + S)){
        if (pos === 'WR') return _scaleMult(1.6, I);
        if (pos === 'RB') return _scaleMult(0.8, I);
      }
      return 1;
    }
    if (strat === 'zero_rb'){
      // Fade RB early, hammer WR; then a catch-up window for RB volume.
      var zEnd = Math.max(2, 5 + S);
      if (round <= zEnd){
        if (pos === 'RB') return _scaleMult(0.35, I);
        if (pos === 'WR') return _scaleMult(1.5, I);
        if (pos === 'TE') return _scaleMult(1.15, I);
        return 1;
      }
      if (round <= 10 + S && pos === 'RB') return _scaleMult(1.35, I);
      return 1;
    }
    if (strat === 'hero_rb'){
      // One anchor RB in the first rounds, then WRs while fading RB depth.
      var hEnd = Math.max(1, 2 + S);
      if (round <= hEnd && pos === 'RB') return have === 0 ? _scaleMult(1.8, I) : _scaleMult(0.35, I);
      if (round <= Math.max(3, 6 + S)){
        if (pos === 'RB') return _scaleMult(0.45, I);
        if (pos === 'WR') return _scaleMult(1.3, I);
      }
      return 1;
    }
    if (strat === 'elite_te'){
      if (round <= Math.max(1, 3 + S) && pos === 'TE' && have === 0) return _scaleMult(1.9, I);
      return 1;
    }
    if (strat === 'early_qb'){
      if (round <= Math.max(2, 4 + S) && pos === 'QB') return _scaleMult(1.6, I);
      return 1;
    }
    return 1;
  }

  // Build a CPU team's scoring context (its own above-replacement counts, its own
  // picks, and its own next pick) so pickScore judges need for THAT team, not the
  // viewer's. Mirrors psCtx()/nextOwnedAfterCurrent() but scoped to one draft slot.
  function _cpuCtx(slot){
    var qualByPos = {}, picksList = [], lastPickByPos = {}, rosterQualities = [];
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (slotOnClock(pn, state.teams, state.order) !== slot) return;
      var mp = state.picks[k];
      picksList.push(mp);
      var pos = (mp.position || '').toUpperCase();
      if (!lastPickByPos[pos] || pn > lastPickByPos[pos]) lastPickByPos[pos] = pn;
      var full = playersById[String(mp.id)];
      var v = full ? vorOf(full) : null;
      if (v == null || v > 0) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
      var rq = full ? ppgNormOf(full) : null;
      rosterQualities.push({ pos: pos, quality: rq != null ? rq : 0.35 });
    });
    var nextOwned = null, tot = state.teams * state.rounds;
    for (var pn2 = state.current + 1; pn2 <= tot; pn2++){
      if (slotOnClock(pn2, state.teams, state.order) === slot){ nextOwned = pn2; break; }
    }
    var remaining = 0;
    for (var pn3 = state.current; pn3 <= tot; pn3++){
      if (!state.picks[pn3] && slotOnClock(pn3, state.teams, state.order) === slot) remaining++;
    }
    var counts = teamCounts(slot), rs = (state && state.roster) || defaultRoster();
    var obligations = window.DraftBoardCore
      ? DraftBoardCore.remainingObligations(counts, rs, remaining, !!state.sf)
      : { required: 0, freePicks: remaining };
    return { qualByPos: qualByPos, picksList: picksList, rosterQualities: rosterQualities, nextOwned: nextOwned,
             lastPickByPos: lastPickByPos, remaining: remaining, obligations: obligations };
  }

  // Estimate how many picks before this team's next turn belong to teams for
  // whom each position is still a starter/FLEX need. It is deliberately a
  // bounded roster inspection, not clairvoyance: CPUs know visible rosters but
  // not which random player another team will select.
  function _demandBeforeNext(nextPick){
    var demand = { QB:0, RB:0, WR:0, TE:0 };
    if (!nextPick || !window.DraftBoardCore) return demand;
    var rs = (state && state.roster) || defaultRoster(), seen = {};
    for (var qn = state.current + 1; qn < nextPick; qn++){
      var os = slotOnClock(qn, state.teams, state.order);
      if (seen[os]) continue; seen[os] = true;
      var oc = teamCounts(os);
      Object.keys(demand).forEach(function(pos){
        var role = DraftBoardCore.rosterRole(pos, oc, rs, !!state.sf);
        if (role === 'starter') demand[pos] += 1;
        else if (role === 'flex') demand[pos] += 0.45;
      });
    }
    return demand;
  }
  // CPU-scoped version of the user's auto-draft completion guard. The late-round
  // weighting normally lets managers choose when to take K/DEF, but once this
  // team's remaining selections equal its unfilled K/DEF slots there is no
  // discretionary pick left: fill a required special-teams slot now. Position
  // order follows that team's plan (not a global kicker-then-defense script).
  function _cpuKDefMustFill(pool, counts, remaining){
    var rs = (state && state.roster) || defaultRoster();
    var needK = Math.max(0, (rs.K || 0) - (counts.K || 0));
    var needDef = Math.max(0, (rs.DEF || 0) - (counts.DEF || 0));
    if (needK + needDef <= 0 || remaining > needK + needDef) return null;
    var slot = slotOnClock(state.current, state.teams, state.order);
    var plan = _simKDefPlan(slot);
    var pickPos = window.DraftBoardCore && DraftBoardCore.specialTeamsFillPos
      ? DraftBoardCore.specialTeamsFillPos(needK, needDef, plan)
      : (needK > 0 && needDef <= 0 ? 'K' : (needDef > 0 && needK <= 0 ? 'DEF' : (plan.prefer === 'DEF' ? 'DEF' : 'K')));
    function bestAt(pos){
      var cands = pool.filter(function(p){ return String(p.position || '').toUpperCase() === pos; });
      cands.sort(function(a,b){ return lineupScore(b) - lineupScore(a); });
      return cands[0] || null;
    }
    var picked = bestAt(pickPos);
    if (picked) return picked;
    if (pickPos === 'K' && needDef > 0) return bestAt('DEF');
    if (pickPos === 'DEF' && needK > 0) return bestAt('K');
    return null;
  }
  function simPick(){
    var pool = availablePool();
    if (!pool.length) return null;
    // Model each available player's draft slot as a draw from Normal(ADP, sigma)
    // and weight them by how likely they are to be taken at THIS exact pick. An
    // ADP 1.1 player has a tight curve so he goes ~1 nearly every time, while an
    // ADP 2.8 player (wider curve, already past pick 1) splits across 2, 3 and 4.
    // After a player slides PAST their ADP, the weight uses inverse-linear decay
    // instead of Gaussian so it never bottoms out - urgency grows, not vanishes.
    var pn = state.current;
    var slot = slotOnClock(pn, state.teams, state.order);
    var counts = teamCounts(slot), targets = posTargets();
    // This CPU's draft personality scales how hard it leans into need/scarcity vs
    // pure best-available value, so teams don't all reach the same way; its bias
    // gives it slight positional preferences.
    var persona = _simPersona(slot);
    var _bias = _simBias(slot);
    var _strat = _simStrategy(slot);
    var _stratPrm = _stratParams(slot);
    var _ageLean = _simAgeLean(slot);
    var _ageInt = _ageLeanIntensity(slot);
    var _kdPlan = _simKDefPlan(slot);
    // Score every candidate from THIS CPU team's perspective (its own roster,
    // depth, and next pick) so need is judged for the right team, not the viewer.
    var cpuCtx = _cpuCtx(slot);
    var mustFillKDef = _cpuKDefMustFill(pool, counts, cpuCtx.remaining);
    if (mustFillKDef) return mustFillKDef;
    var _maxVal = 0; pool.forEach(function(q){ var v = valOf(q); if (v > _maxVal) _maxVal = v; });
    // Starter-slot map: actual lineup spots only (no bench).
    var _rs = (state && state.roster) || defaultRoster();
    var _sfSlots = state.sf ? ((_rs.SF != null ? _rs.SF : 1)) : 0;
    var _stSlots = { QB: (_rs.QB||0) + _sfSlots, RB: (_rs.RB||0) + (_rs.FLEX||0), WR: _rs.WR||0, TE: _rs.TE||0, K: _rs.K||0, DEF: _rs.DEF||0 };
    var _remainRds = state.rounds - Math.floor((pn - 1) / state.teams);
    // Backup-QB timing: a QB beyond the starting slots (a 2nd QB in 1QB, a 3rd QB
    // in SF where two QBs start) only becomes a need a handful of rounds AFTER the
    // team completed its starting QB room - so a manager who locked up their QBs
    // early waits longer for the backup than one who finished the room late. Anchor
    // on the pick that filled the LAST starting QB slot; gate on a per-team gap with
    // a late-draft fallback so the backup still gets rostered by the end.
    var _curRound = Math.ceil(pn / state.teams);
    var _qbStarters = _stSlots.QB || 1;   // 1 in 1QB, 2 in SF
    var _qbPicks = [];
    Object.keys(state.picks).forEach(function(k){
      var _kp = parseInt(k, 10);
      if (slotOnClock(_kp, state.teams, state.order) !== slot) return;
      if ((state.picks[k].position || '').toUpperCase() === 'QB') _qbPicks.push(_kp);
    });
    _qbPicks.sort(function(x, y){ return x - y; });
    // Pick that filled the last starting-QB slot (QB1 in 1QB, QB2 in SF).
    var _qbFullPick = _qbPicks.length >= _qbStarters ? _qbPicks[_qbStarters - 1] : null;
    var _qbFullRound = _qbFullPick ? Math.ceil(_qbFullPick / state.teams) : null;
    // Earlier the QB room was completed => longer wait for the backup. The gap
    // shrinks ~0.5 rounds per round of delay (room full in round 1 -> ~9-11 round
    // gap; round 7 -> ~6-8; round 12 -> ~3-5), plus a little per-team jitter.
    var _qbGap = Math.max(3, Math.round(9.5 - (_qbFullRound || 1) * 0.5)) + Math.round(_rand01(slot + ':qbgap') * 2);
    var _backupQBWanted = _qbFullRound != null &&
      ((_curRound - _qbFullRound) >= _qbGap || _curRound >= (state.rounds || 20) * 0.8);
    // CPU realism inputs (read this team's roster once): stack a QB with his
    // pass-catchers, handcuff a backup to my own RB, and avoid piling startable
    // players onto a single bye week. Small tie-breakers, not reach-forcing.
    var _myQbTeams = {}, _myPassTeams = {}, _myRbTeams = {}, _myByes = {};
    (cpuCtx.picksList || []).forEach(function(mp){
      var _mf = playersById[String(mp.id)] || mp;
      var _mpos = (_mf.position || '').toUpperCase();
      var _tm = (_mf.team || '').toUpperCase();
      if (_tm){
        if (_mpos === 'QB') _myQbTeams[_tm] = true;
        else if (_mpos === 'WR' || _mpos === 'TE') _myPassTeams[_tm] = true;
        else if (_mpos === 'RB') _myRbTeams[_tm] = true;
      }
      var _bw = Number(_mf.bye_week);
      if (_bw) _myByes[_bw] = (_myByes[_bw] || 0) + 1;
    });
    var _stackOn = (state.type === 'redraft' || state.type === 'startup');
    var cands = [];
    var bestPv = 0;   // highest pick score available (the "best player on the board")
    var posQualLeft = {};  // count of remaining startable-quality players per position
    pool.forEach(function(p){
      var a = simAdp(p);
      // Per-draft effective ADP so the board plays out differently each mock.
      var aEff = a + _adpNoise(p, a);
      if (aEff < 0.5) aEff = 0.5;
      var sigma = simSigma(aEff);
      var diff = pn - aEff;
      var w;
      if (diff <= 0) {
        var z = diff / sigma;
        w = Math.exp(-0.5 * z * z);               // peak when the pick reaches the ADP
      } else {
        // Past ADP: urgency grows so players don't slide indefinitely.
        // Hard cap: once a player is maxSlide picks overdue they dominate the pick.
        var maxSlide = Math.max(1, Math.round(sigma * 2));
        if (diff >= maxSlide) {
          w = 5.0;                                 // overdue - near-certain next pick
        } else {
          w = 1.0 / (1.0 + 0.12 * diff);          // inverse-linear ramp up to cap
        }
      }
      // CPU-perspective pick score (same value+need model the app shows). Null for
      // K/DEF, which have no pick score and are handled by the late-round boost.
      var pv = pickScore(p, _maxVal, counts, cpuCtx);
      if (pv != null && pv > bestPv) bestPv = pv;
      // Track how many genuinely startable players remain at each position so the
      // CPU can sense a thinning position (a run on QBs/RBs) and act before the
      // cupboard is bare. 55 is a "startable" pick-score floor.
      if (pv != null && pv >= 55){ var _pp = (p.position||'').toUpperCase(); posQualLeft[_pp] = (posQualLeft[_pp] || 0) + 1; }
      cands.push({ p: p, w: w, a: a, pv: pv, ae: aEff });
    });
    var _nextShelf = {}, _demand = _demandBeforeNext(cpuCtx.nextOwned);
    ['QB','RB','WR','TE'].forEach(function(pos){
      var bestExpected = 0;
      cands.filter(function(c){ return (c.p.position || '').toUpperCase() === pos && c.pv != null; })
        .sort(function(a, b){ return b.pv - a.pv; }).slice(0, 16).forEach(function(c){
          var prob = cpuCtx.nextOwned ? availProb(c.p, cpuCtx.nextOwned) : 0;
          if (prob == null) prob = 50;
          bestExpected = Math.max(bestExpected, c.pv * (0.35 + 0.65 * prob / 100));
        });
      _nextShelf[pos] = bestExpected;
    });
    var _econOpts = { sf: !!state.sf, tep: scoringCfg().tep, draftType: state.type };
    cands.forEach(function(c){
      var p = c.p, w = c.w, a = c.a, pv = c.pv;
      var pos = (p.position||'').toUpperCase();
      var t = targets[pos] || 0, have = counts[pos] || 0;
      var rosterLimit = window.DraftBoardCore && DraftBoardCore.positionRosterLimit
        ? DraftBoardCore.positionRosterLimit(pos, _rs, { draftType:state.type, tep:scoringCfg().tep })
        : Infinity;
      if (have >= rosterLimit){ c.w = 0; c.ds = -1; return; }
      // Kicker and defense are required lineup slots, not ordinary bench depth.
      // Once this team has filled the configured number, remove every additional
      // K/DEF from its candidate set entirely. A soft overfill multiplier is not
      // sufficient because synthetic late ADP can otherwise make several overdue
      // special-teams players beat depleted skill-position options.
      if ((pos === 'K' || pos === 'DEF') && (t <= 0 || have >= t)){
        c.w = 0; c.ds = -1; return;
      }
      // Shared roster economics run before personality. This is the common
      // baseline used by human recommendations and prevents CPU strategy/noise
      // from rescuing a structurally bad QB3/TE3 or an immediate backup pick.
      if (pv != null && window.DraftBoardCore){
        var _role = DraftBoardCore.candidateRosterRole
          ? DraftBoardCore.candidateRosterRole(pos, ppgNormOf(p) || 0, cpuCtx.rosterQualities, _rs, !!state.sf)
          : DraftBoardCore.rosterRole(pos, counts, _rs, !!state.sf);
        _econOpts.role = _role;
        var _util = DraftBoardCore.rosterSlotUtility(pos, counts, _rs, _econOpts);
        var _bench = _role === 'bench1' || _role === 'bench2';
        var _recent = 0;
        if (_bench && (pos === 'QB' || pos === 'TE') && cpuCtx.lastPickByPos[pos]){
          var _since = (pn - cpuCtx.lastPickByPos[pos]) / Math.max(1, state.teams || 12);
          _recent = Math.max(0, 10 * (1 - _since / 6));
        }
        var _exceptional = a < 9000 ? clamp01((pn - a) / Math.max(12, a * 0.65)) : 0;
        var _waitLoss = Math.max(0, pv - (_nextShelf[pos] || 0));
        // Visible needy teams between turns make a real shelf loss more urgent;
        // cap it so scarcity does not double-count VOR/tier effects excessively.
        _waitLoss *= 1 + Math.min(0.35, (_demand[pos] || 0) / Math.max(1, state.teams) * 0.7);
        c.ds = DraftBoardCore.decisionScore({ base: pv, utility: _util,
          bench: _bench, deepBench: _role === 'bench2', quality: ppgNormOf(p) || 0,
          required: cpuCtx.obligations.required, freePicks: cpuCtx.obligations.freePicks,
          recentPenalty: _recent, exceptional: _exceptional, waitLoss: _waitLoss });
        // Decision quality gates the ADP likelihood but does not replace it. A
        // persona may choose among close values; it cannot turn poor roster fit
        // into a favorite merely by stacking several heuristic multipliers.
        w *= Math.pow(Math.max(0.12, c.ds / 100), 2.2);
      } else c.ds = pv || 0;
      var need = t ? Math.max(0, t - have) / t : 0;
      // A QB beyond the starting slots (2nd in 1QB, 3rd in SF) carries no need
      // pull until enough rounds after the QB room was completed (_backupQBWanted).
      if (pos === 'QB' && have >= _qbStarters && !_backupQBWanted) need = 0;
      var over = (t && have >= t) ? (have - t + 1) : 0;
      // Overcrowding penalty: exponential once past depth target (3.5x per excess pick).
      // over=1 -> weight/4.5; over=2 -> weight/13; over=3 -> weight/43.
      var overFactor = over > 0 ? Math.pow(3.5, over) : 1;
      // Early-round starter-slot penalty: prevent CPU from stacking single-slot
      // positions (TE in 1TE no-TEP, QB in 1QB) before filling other positional needs.
      var sSlots = _stSlots[pos] || 0;
      if (sSlots > 0 && have >= sSlots) {
        // Penalty fades from strong in rounds 1-6, gone by round 12+
        var _rnd = Math.ceil(pn / state.teams);
        var earlyMult = Math.max(0, 1 - _rnd / 12);
        // A backup to a thin starter slot (the lone QB/TE in 1QB) is far less
        // valuable than extra depth at a multi-starter position - you only ever
        // start one. Penalize stacking a single-slot position much harder so a
        // CPU never spends an early pick on a 2nd QB while real starting needs
        // remain; depth positions (2-3 starters) keep a milder penalty.
        var slotScarcityMult = sSlots <= 1 ? 3.0 : (sSlots === 2 ? 1.5 : 1.0);
        overFactor *= (1 + 3.5 * slotScarcityMult * earlyMult * (have - sSlots + 1));
      }
      // Depth nudge (base need-awareness): SF QB gets a stronger factor so the CPU
      // targets a 2nd QB; the zero-QB SF case stays urgent once the early rounds pass.
      var needFactor = (pos === 'QB' && state.sf) ? 0.65 : 0.25;
      if (pos === 'QB' && state.sf && have === 0 && pn > state.teams * 2) needFactor = 1.5;
      // All discretionary need/scarcity pulls scale by this team's personality:
      // a value-first (BPA) team barely reaches, a roster-builder reaches hard.
      var needBoost = 1 + persona * needFactor * need;
      // Value-aware starter need: a player who fills an OPEN starting slot and is
      // close in pick score to the best player on the board should be preferred -
      // drafting like a smart manager (e.g. a 2nd QB in Superflex when a near-best
      // one is available). Quadratic in closeness so only genuinely good values
      // trigger the strong pull; a mediocre fit gets only a mild bump.
      var starterNeed = sSlots > 0 ? clamp01((sSlots - have) / sSlots) : 0;
      if (starterNeed > 0 && pv != null && bestPv > 0){
        var closeness = clamp01(pv / bestPv);
        needBoost += persona * 3.0 * starterNeed * closeness * closeness;
        // Scarcity: if the startable pool at this needed position is drying up,
        // grab one now rather than chase a falling-ADP player elsewhere - exactly
        // how a manager reaches when "there aren't many good QBs/RBs left." Urgency
        // ramps as the remaining quality count drops below roughly one-per-team.
        var qualLeft = posQualLeft[pos] || 0;
        var scarce = clamp01((state.teams - qualLeft) / state.teams);
        needBoost += persona * 2.0 * starterNeed * scarce;
      }
      // Gentle per-team positional lean (skill positions only).
      var biasMult = _bias[pos] || 1;
      // Named strategy plan (RB heavy / Zero RB / Hero RB / Elite TE / Early QB).
      // Early-QB only pulls while the starting QB room is unfilled.
      var stratMult = _stratMult(_strat, pos, have, _curRound, _stratPrm);
      if (_strat === 'early_qb' && pos === 'QB' && have >= _qbStarters) stratMult = 1;
      // Executing the plan on this player must not force a big reach: the boost
      // fades with how far past ADP the player still is at this pick.
      stratMult = _stratReachDamp(stratMult, c.ae != null ? c.ae : a, pn);
      // Plan flexibility, part 1: a real value fall breaks through a fade. A
      // Zero RB team still takes the RB who slid a quarter round past ADP -
      // the fade relaxes with the slide and is fully neutral by 3/4 of a round.
      if (stratMult < 1){
        var _slide = Math.max(0, pn - (c.ae != null ? c.ae : a));
        var _slideFree = (state.teams || 12) * 0.25;
        var _slideSpan = (state.teams || 12) * 0.5;
        if (_slide > _slideFree){
          var _rel = Math.min(1, (_slide - _slideFree) / _slideSpan);
          stratMult = stratMult + (1 - stratMult) * _rel;
        }
      }
      // Plan flexibility, part 2: a plan never forces a clearly inferior
      // player. The boost scales away as this candidate's pick score falls
      // off the board's best - when the plan position's shelf is bare, the
      // team quietly pivots to value instead of forcing a bad fit.
      if (stratMult > 1 && pv != null && bestPv > 0){
        var _q = Math.max(0, Math.min(1, (pv / bestPv) / 0.75));
        stratMult = 1 + (stratMult - 1) * _q;
      }
      // Dynasty age lean (startup only): win-now pays for vets, youth punts them.
      var ageMult = _ageLeanMult(_ageLean, pos, p.age != null ? Number(p.age) : null, _ageInt);
      w *= needBoost * biasMult * stratMult * ageMult / overFactor;
      // CPU realism (weekly-lineup formats): stack the team's QB with his
      // pass-catchers, insure an owned RB with its handcuff, and shade away from
      // stacking startable players on a shared bye. Kept small so they break
      // ties near value rather than forcing a reach.
      var _tmU = (p.team || '').toUpperCase();
      if (_stackOn && _tmU){
        if ((pos === 'WR' || pos === 'TE') && _myQbTeams[_tmU]) w *= 1.08;
        else if (pos === 'QB' && _myPassTeams[_tmU]) w *= 1.06;
        if (state.type === 'redraft' && pos === 'RB' && _myRbTeams[_tmU]) w *= 1.05;
      }
      if (state.type === 'redraft' && p.bye_week && (_myByes[Number(p.bye_week)] || 0) >= 2){
        w *= 0.92;
      }
      // Hard guard: a backup at a starter-filled slot has little marginal value,
      // so the CPU must never REACH for one - only take it if its real ADP has
      // fallen to this pick or later. Covers a 2nd QB in 1QB, a 3rd QB in SF (both
      // starting QBs already in hand), and a 2nd TE in 1TE. Stops the classic
      // bad-CPU move of grabbing a backup a round ahead of ADP.
      var _backupReach = a < 9000 && pn < a && (
        (pos === 'QB' && have >= _qbStarters) ||
        (pos === 'TE' && sSlots > 0 && sSlots <= 1 && have >= sSlots)
      );
      if (_backupReach) w = 0;
      // K/DEF: ungraded (no pick score), so they would never enter the decision
      // band until the last-two-pick must-fill. Give each team its own window,
      // order, and intensity so some grab an early DEF, some split ST around a
      // skill pick, and some wait until the end — not K-then-DEF every time.
      if ((pos === 'K' || pos === 'DEF') && (t > 0) && (have < t)){
        var _otherHave = pos === 'K' ? (counts.DEF || 0) : (counts.K || 0);
        var _otherT = pos === 'K' ? (_rs.DEF || 0) : (_rs.K || 0);
        var _alreadyHasOther = _otherT > 0 && _otherHave > 0;
        var _inWindow = _remainRds <= _kdPlan.window;
        var _isPref = _kdPlan.prefer === 'mix' || _kdPlan.prefer === pos;
        var _delayOther = _kdPlan.split && _alreadyHasOther && cpuCtx.remaining > 1;
        if (_inWindow && !_delayOther){
          var _urg = Math.max(0, Math.min(1, (_kdPlan.window - _remainRds + 1) / Math.max(1, _kdPlan.window)));
          var _prefM = _isPref ? 1 : 0.62;
          var _boost = (2.8 + 4.2 * _kdPlan.intensity) * _prefM;
          w = Math.max(w * _boost, 1.4 * _kdPlan.intensity * _prefM);
          c.ds = Math.round(50 + 18 * _urg + (_isPref ? 7 : 0) + 4 * (_kdPlan.intensity - 1)
            + (_rand01(slot + ':kds:' + pn + ':' + String(p.id)) - 0.5) * 10);
        } else if (_inWindow && _delayOther){
          c.ds = 28;
          w *= 0.35;
        } else if (_remainRds <= _kdPlan.window + 2 && _isPref && a < 9000
            && (pn + (state.teams || 12) * 0.5 >= a)){
          c.ds = Math.round(46 + 4 * _kdPlan.intensity);
          w *= 1.25 + 0.6 * _kdPlan.intensity;
        }
      }
      // Every CPU team must draft at least one QB. When a team has none and time is
      // running out, strongly boost any available QB so it cracks the top-8 sample
      // even if its ADP sentinel is large (all named QBs already taken).
      if (pos === 'QB' && have === 0 && t > 0 && _remainRds <= Math.ceil(state.rounds * 0.4)){
        w = Math.max(w * 6, 1e-6);
      }
      // ADP-less players (a huge sentinel) get a tiny value-based floor so they
      // can still fill in late rounds once the ranked board is exhausted.
      if (a >= 9000) w = Math.max(w, 1e-9 * valOf(p));
      c.w = w;
    });
    // Strategy freedom at value: a team drafting for its plan may take ITS guy
    // instead of strictly the board's guy - but only among players whose ADP is
    // basically the same. Anchor on the boosted position's lowest-ADP candidate
    // near this pick; candidates within ~one sigma of the anchor's ADP flatten
    // toward its weight (near-coin-flips, e.g. RB1 at 1.2 vs RB2 at 1.9 for the
    // 1.01 Hero RB team). Anyone meaningfully later in ADP keeps ADP order, so
    // this adds preference among equals without ever skipping down the board.
    if (_strat && _strat !== 'bpa'){
      var _freePick = (state.teams || 12) * 0.25;
      var _boostedPos = {};
      ['QB','RB','WR','TE'].forEach(function(bp){
        var bHave = counts[bp] || 0;
        var bm = _stratMult(_strat, bp, bHave, _curRound, _stratPrm);
        if (_strat === 'early_qb' && bp === 'QB' && bHave >= _qbStarters) bm = 1;
        if (bm > 1) _boostedPos[bp] = true;
      });
      var _posAnchor = {};  // pos -> {ae, w} of the lowest-ADP in-window candidate
      cands.forEach(function(c){
        var bp = (c.p.position || '').toUpperCase();
        if (!_boostedPos[bp] || c.w <= 0) return;
        var ae = c.ae != null ? c.ae : c.a;
        if (ae >= 9000 || Math.max(0, ae - pn) > _freePick) return;
        if (!_posAnchor[bp] || ae < _posAnchor[bp].ae) _posAnchor[bp] = { ae: ae, w: c.w };
      });
      cands.forEach(function(c){
        var bp = (c.p.position || '').toUpperCase();
        var an = _posAnchor[bp];
        if (!an || c.w <= 0) return;
        var ae = c.ae != null ? c.ae : c.a;
        if (ae >= 9000 || Math.max(0, ae - pn) > _freePick) return;
        var eps = Math.max(0.75, simSigma(an.ae));
        if (ae - an.ae > eps) return;  // not basically the same - ADP order stands
        c.w = Math.max(c.w, an.w * 0.65);
      });
    }
    // Restrict to the realistic field, then sample proportionally to weight so
    // the favorite usually wins but upsets happen at the documented rate.
    cands = cands.filter(function(c){ return c.w > 0; });
    if (!cands.length) return null;
    cands.sort(function(x, y){ return y.w - x.w; });
    // Human-like randomness only among defensible alternatives. Early picks use
    // a narrow decision band; later rounds and adventurous personas widen it.
    if (window.DraftBoardCore && DraftBoardCore.selectDecisionCandidate){
      var selected = DraftBoardCore.selectDecisionCandidate(
        cands.map(function(c){ return { ref:c, ds:c.ds, weight:c.w }; }), _curRound, persona, Math.random);
      return selected && selected.ref ? selected.ref.p : cands[0].p;
    }
    var eligible = cands;
    var top = eligible.slice(0, Math.min(eligible.length, 8));
    var sum = 0; top.forEach(function(c){ sum += c.w; });
    if (sum <= 0) return top[0].p;
    var roll = Math.random() * sum;
    for (var i = 0; i < top.length; i++){ roll -= top[i].w; if (roll <= 0) return top[i].p; }
    return top[0].p;
  }
  // Pick the highest-scored available player for my roster (used by auto-draft).
  // Highest available player in the user's queue that still fits an open roster
  // slot. The queue is an explicit target list, so auto-draft honors it first:
  // if you queued a player, auto-draft takes them (in queue order) before falling
  // back to the scored best-available. Drafted or roster-capped entries are
  // skipped so a stale queue never blocks a legal pick.
  function _queuedAutoPick(pool){
    if (!state.queue || !state.queue.length) return null;
    var avail = {};
    (pool || []).forEach(function(p){ avail[String(p.id)] = p; });
    var rs = (state && state.roster) || defaultRoster();
    var counts = myPosCounts();
    for (var i = 0; i < state.queue.length; i++){
      var p = avail[String(state.queue[i])];
      if (!p) continue;   // already drafted or not in the pool
      var pos = String(p.position || '').toUpperCase();
      var limit = window.DraftBoardCore && DraftBoardCore.positionRosterLimit
        ? DraftBoardCore.positionRosterLimit(pos, rs, { draftType: state.type, tep: scoringCfg().tep })
        : Infinity;
      if ((counts[pos] || 0) < limit) return p;   // first legal queued target wins
    }
    return null;
  }
  function autoPick(){
    var pool = availablePool();
    if (!pool.length) return null;
    // Your queue is your auto-draft priority list: take the top available target
    // that still fits a roster slot before any scoring/plan logic runs.
    var _q = _queuedAutoPick(pool);
    if (_q) return _q;
    // Roster-need guard: K/DEF have no pickScore (null -> 0), so they'd never be
    // auto-drafted and your team would finish without a kicker/defense while every
    // CPU team fills theirs. Mirror the CPU's late-round behavior: when you still
    // have unfilled required K/DEF slots and your remaining picks are running out,
    // grab the best available one first so the slot doesn't go empty.
    var _kd = _autoKDefNeed(pool);
    if (_kd) return _kd;
    // Use the exact live recommendation baseline before applying the user's
    // chosen plan. This keeps auto-draft aligned with the sidebar on roster
    // utility, waiting value, recent investment and remaining obligations.
    refreshPsPool();
    // User-chosen auto-draft plan: apply the same strategy/age-lean shaping the
    // CPU teams use, at full intensity with no window jitter (you picked the
    // plan deliberately, so it executes straight).
    var strat = state.myStrat || '';
    var lean = state.myAgeLean || '';
    // Always track counts/targets so the auto-draft respects roster needs the way
    // the CPU does (it used to only when a strategy/lean was set, which let it
    // stack a saturated position while a starter slot sat empty).
    var counts = myPosCounts();
    var targets = posTargets();
    var rs = (state && state.roster) || {};
    var round = Math.ceil(state.current / (state.teams || 12));
    var prm = { intensity: 1, shift: 0 };
    var qbStarters = state.sf ? 2 : 1;
    var nextPn = nextOwnedAfterCurrent();
    var tep = scoringCfg().tep;
    var scored = pool.map(function(p){
      var s = (p._ds != null ? p._ds : pickScoreFor(p)) || 0;
      var pos = (p.position || '').toUpperCase();
      var have = counts[pos] || 0;
      if (strat || lean){
        var sm = _stratMult(strat, pos, have, round, prm);
        if (strat === 'early_qb' && pos === 'QB' && have >= qbStarters) sm = 1;
        // Hard reach cap: pick scores sit close together, so even a damped
        // boost in score space can force a multi-pick reach (RB heavy at 1.07
        // taking the ADP-14 RB). The plan boost applies only at or within a
        // quarter round of value; beyond that the plan waits for its spot.
        if (sm > 1){
          var _apk = adpOf(p);
          if (_apk != null && _apk < 9000
              && (_apk - state.current) > (state.teams || 12) * 0.25){
            sm = 1;
          }
        }
        var am = _ageLeanMult(lean, pos, p.age != null ? Number(p.age) : null, 1);
        s = s * sm * am;
      }
      // Roster-need shaping (mirrors the CPU's simPick guards). The CPU can lean
      // on ADP likelihood so a 13-spot reach almost never wins; autodraft is
      // argmax of score, so an uncapped 1.35x empty-starter boost will take a TE
      // a round early in a 1TE league even when that player is still there at
      // the turn. Shared kernel: backup-reach / overfill guards, starter pull
      // only near ADP, and wait on single-slot reaches that survive to the turn.
      if (s > 0 && pos !== 'K' && pos !== 'DEF' && window.DraftBoardCore
          && DraftBoardCore.autoDraftNeedMultiplier){
        s *= DraftBoardCore.autoDraftNeedMultiplier({
          pos: pos, have: have, target: targets[pos] || 0,
          starterSlots: rs[pos] || 0, adp: adpOf(p), pickNo: state.current,
          teams: state.teams || 12, nextPick: nextPn,
          surviveProb: nextPn ? availProb(p, nextPn) : null,
          tep: tep, sf: !!state.sf, qbStarters: qbStarters,
        });
      }
      return { p: p, s: s };
    });
    scored.sort(function(a, b){ return b.s - a.s; });
    return scored[0].p;
  }
  // Returns the best available K/DEF to draft now if an unfilled required slot
  // would otherwise go empty given the picks you have left; else null.
  function _autoKDefNeed(pool){
    var rs = (state && state.roster) || {};
    var needK   = Math.max(0, (rs.K   || 0) - (myPosCounts().K   || 0));
    var needDef = Math.max(0, (rs.DEF || 0) - (myPosCounts().DEF || 0));
    if (needK + needDef <= 0) return null;
    // How many of your own picks remain (this one included)?
    var remaining = ownedPicks().filter(function(pn){ return pn >= state.current && !state.picks[pn]; }).length;
    if (remaining <= 0) return null;
    // Take K/DEF when you can't afford to wait (picks left <= slots still to fill)
    // or you're in the final stretch where the CPU is filling these too.
    var _remainRds = state.rounds - Math.floor((state.current - 1) / state.teams);
    var mustFill = remaining <= (needK + needDef) || _remainRds <= 3;
    if (!mustFill) return null;
    // Prefer whichever position you still need; rank by projected PPG (lineupScore).
    var want = {};
    if (needK > 0)   want.K = true;
    if (needDef > 0) want.DEF = true;
    var cands = pool.filter(function(p){ return want[(p.position || '').toUpperCase()]; });
    if (!cands.length) return null;
    cands.sort(function(a, b){ return lineupScore(b) - lineupScore(a); });
    return cands[0];
  }
  // Surface an unexpected error in the mock-sim pick path instead of letting the
  // draft freeze silently (a throw inside a setTimeout callback is otherwise
  // swallowed and the sim just stops making picks). Logs the full stack for
  // debugging and shows a dismissible banner so the failure is reportable.
  function _simError(where, e){
    try { console.error('[draft-room] ' + where + ' failed:', e); } catch (_e){}
    var msg = (e && e.message) ? e.message : String(e);
    var box = document.getElementById('drSimError');
    if (!box){
      box = document.createElement('div');
      box.id = 'drSimError';
      box.className = 'dr-sim-error';
      var host = document.getElementById('drMain') || document.body;
      host.insertBefore(box, host.firstChild);
    }
    box.innerHTML = '<b>Mock draft hit an error</b> (' + esc(where) + '): ' + esc(msg)
      + ' &mdash; please screenshot this so it can be fixed.'
      + ' <button type="button" class="dr-sim-error-x" aria-label="Dismiss">&times;</button>';
    box.style.display = '';
    var x = box.querySelector('.dr-sim-error-x');
    if (x) x.addEventListener('click', function(){ box.style.display = 'none'; });
  }
  // Last-resort progress guard for a depleted late board. Normal CPU/auto-draft
  // scoring may legitimately produce no weighted candidate after roster caps,
  // ADP reach guards and strategy filters interact. Choose the best remaining
  // legal player instead of silently stopping the timer with picks unfinished.
  function _fallbackLegalPick(pool, counts){
    var rs = (state && state.roster) || defaultRoster();
    var legal = (pool || []).filter(function(p){
      var pos = String(p.position || '').toUpperCase();
      var limit = window.DraftBoardCore && DraftBoardCore.positionRosterLimit
        ? DraftBoardCore.positionRosterLimit(pos, rs, { draftType:state.type, tep:scoringCfg().tep })
        : Infinity;
      return (counts[pos] || 0) < limit;
    });
    legal.sort(function(a,b){
      var av = valOf(a), bv = valOf(b);
      if (bv !== av) return bv - av;
      var aa = adpOf(a), ba = adpOf(b);
      return (aa == null ? 99999 : aa) - (ba == null ? 99999 : ba);
    });
    return legal[0] || null;
  }
  function _doAutoPick(){
    if (!sim || simPaused || !simStarted) return;
    if (_simTabHidden()) return;
    try {
      var ap = autoPick();
      if (!ap) ap = _fallbackLegalPick(availablePool(), myPosCounts());
      if (!ap){ _simError('auto pick', new Error('No legal players remain for the open roster spot')); endSim(); render(); return; }
      commitPick(ap); render(); scheduleSim();
    } catch (e){ _simError('auto pick', e); endSim(); render(); }
  }
  function _simTabHidden(){
    return typeof document !== 'undefined' && document.hidden;
  }
  function scheduleSim(){
    if (!sim || simPaused || !simStarted) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); return; }
    // Always drop a pending CPU/auto tick before queuing the next one. A timer
    // left over from the previous seat can fire after it becomes your pick and
    // steal the clock, or keep running after you background the tab.
    clearTimeout(simTimer);
    simTimer = null;
    if (_simTabHidden()) return;
    if (isMyPick(state.current)){
      if (simAutoDraft) simTimer = setTimeout(_doAutoPick, simSpeed);
      return;
    }
    simTimer = setTimeout(simStep, simSpeed);
  }
  function simStep(){
    if (!sim || simPaused || !simStarted) return;
    if (_simTabHidden()) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); render(); return; }
    if (isMyPick(state.current)){
      if (simAutoDraft){ clearTimeout(simTimer); simTimer = setTimeout(_doAutoPick, simSpeed); return; }
      render(); return; // your turn - wait for manual pick
    }
    var p;
    try {
      p = simPick();
    } catch (e){
      // A bad scoring pass shouldn't freeze the draft: log it, then fall back to
      // plain best-available-by-value so the CPU still makes a pick.
      _simError('CPU pick', e);
      var _pool = availablePool();
      if (_pool.length){ _pool.sort(function(a, b){ return valOf(b) - valOf(a); }); p = _pool[0]; }
    }
    if (!p) p = _fallbackLegalPick(availablePool(), teamCounts(slotOnClock(state.current, state.teams, state.order)));
    if (!p){ _simError('CPU pick', new Error('No legal players remain before the draft is complete')); endSim(); render(); return; }
    try {
      commitPick(p); render();
    } catch (e){ _simError('commit pick', e); endSim(); render(); return; }
    scheduleSim();
  }
  function endSim(){
    sim = false; clearTimeout(simTimer);
    syncSimControls();
  }
  function _resetTransient(){
    endSim();
    simPaused = false; simStarted = false; simAutoDraft = false;
    players = []; drafted = {};
    posFilter = {}; _syncPosPills();
    justPick = null;
    _liveSig = null; _pickLagMsg = null;
    _pollCount = 0; _pollLastAt = 0; _pollNextAt = 0;
    _boardSig = null;
    _summaryShown = false;
    lastLivePicks = null;
    _poServer = null; _poServerSig = null; _poFetching = false; _poFailedSig = null;   // drop cached playoff odds
    _poMcCache = null; _poMcSig = null;
    _relCache = { sig: null, map: {} };   // drop reconstructed pool-relative scores
  }
  function toggleSim(){
    simPaused = !simPaused;
    document.getElementById('drSimToggle').textContent = simPaused ? 'Resume' : 'Pause';
    if (simPaused){ clearTimeout(simTimer); simTimer = null; }
    else scheduleSim();
  }
  function _onSimVisibility(){
    if (!sim || !simStarted) return;
    if (_simTabHidden()){
      clearTimeout(simTimer);
      simTimer = null;
      return;
    }
    if (!simPaused) scheduleSim();
  }
  // Reflect the current mock state on the status-bar controls.
  function syncSimControls(){
    var start = document.getElementById('drSimStart');
    var tg = document.getElementById('drSimToggle');
    var sp = document.getElementById('drSimSpeed');
    var ab = document.getElementById('drAutoBtn');
    var ready = sim && !simStarted;
    var running = sim && simStarted;
    // The whole Auto-draft settings block (in the gear menu) only applies to a
    // mock; hide it entirely for a live draft so the menu stays lean.
    var autoWrap = document.getElementById('drAutoSettings');
    if (autoWrap) autoWrap.style.display = (ready || running) ? '' : 'none';
    start.style.display = ready ? '' : 'none';
    tg.style.display = running ? '' : 'none';
    sp.style.display = (ready || running) ? '' : 'none';
    if (running){ tg.textContent = simPaused ? 'Resume' : 'Pause'; }
    if (ab){
      ab.style.display = running ? '' : 'none';
      ab.textContent = simAutoDraft ? 'Manual' : 'Auto Draft';
      ab.className = 'dr-btn ' + (simAutoDraft ? 'dr-btn-primary' : 'dr-btn-ghost');
    }
    // Auto-draft plan selectors: strategy whenever a mock is up; age lean only
    // for startup drafts (redraft has no age axis, rookie classes are young).
    var ms = document.getElementById('drMyStrat');
    var ml = document.getElementById('drMyAgeLean');
    if (ms){
      ms.style.display = (ready || running) ? '' : 'none';
      // Rookie drafts only support the simple RB-first/WR-first leans (same
      // restriction as the CPU teams): structural plans like Zero RB would
      // fade a position for essentially the whole short draft.
      var _rk = state && state.type === 'rookie';
      var _rkAllowed = { '': 1, rb_heavy: 1, wr_heavy: 1 };
      for (var _oi = 0; _oi < ms.options.length; _oi++){
        var _ov = ms.options[_oi].value;
        ms.options[_oi].hidden = _rk && !_rkAllowed[_ov];
      }
      if (_rk && state.myStrat && !_rkAllowed[state.myStrat]){
        state.myStrat = '';
        save();
      }
      ms.value = (state && state.myStrat) || '';
    }
    if (ml){
      var startup = state && state.type !== 'redraft' && state.type !== 'rookie';
      ml.style.display = ((ready || running) && startup) ? '' : 'none';
      ml.value = (state && state.myAgeLean) || '';
    }
  }
  // User hit Start Draft: kick off the CPU picks.
  function beginSim(){
    if (!sim || simStarted) return;
    simStarted = true; simPaused = false;
    if (state) state.simStarted = true;
    save();
    syncSimControls();
    try {
      renderSide();
      scheduleSim();
    } catch (e){ _simError('start draft', e); }
  }
  function startMock(){
    var prev = state;
    _resetTransient();
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    // Editing a mock's setup should not wipe the board. When the pick numbering is
    // unchanged (same teams / order / slot), carry over the picks already made
    // that still fit the board and resume at the first empty slot - the same
    // behavior manual mode already has. A Reset nulls state first, so a genuine
    // fresh mock still begins empty.
    if (prev && prev.picks && prev.teams === state.teams &&
        prev.order === state.order && prev.slot === state.slot) {
      var tot = (state.teams || 0) * (state.rounds || 0);
      var carried = {};
      Object.keys(prev.picks).forEach(function(pn){
        var n = parseInt(pn, 10);
        var pk = prev.picks[pn];
        if (n >= 1 && n <= tot && pk) {
          carried[n] = pk;
          if (pk.id) drafted[String(pk.id)] = true;
        }
      });
      if (Object.keys(carried).length) {
        state.picks = carried;
        var next = tot + 1;
        for (var i = 1; i <= tot; i++){ if (!carried[i]){ next = i; break; } }
        state.current = next;
      }
    }
    state.mode = 'mock';
    state.simStarted = false;
    sim = true; simPaused = false; simStarted = false;
    var sp = document.getElementById('drSimSpeed');
    simSpeed = parseInt(sp.value, 10) || 700;
    syncSimControls();
    _setUpcomingMode(false);
    save();
    resetSideTabs();   // clear any leftover completed-draft sidebar state
    showMain();
    loadPlayers();
  }
  // Spin up a CPU mock that mirrors the synced live/upcoming draft's settings
  // (teams, rounds, scoring, order, type, and your owned picks) so you can
  // rehearse the exact same draft against ADP-driven bots.
  function startPracticeMock(){
    if (!state) return;
    drConfirm('Start a practice mock using these draft settings? Your live sync will stop until you reconnect.', 'Start Mock', function(){
      stopPolling(); stopPickTimer();
      var prev = state;
      var ownedCopy = {};
      if (prev.owned){ Object.keys(prev.owned).forEach(function(k){ if (prev.owned[k]) ownedCopy[k] = true; }); }
      // Carry the connected league's real team names into the mock so the draft
      // summary maps each seat to its actual owner instead of a random CPU name.
      // (Shallow-copied so the mock never mutates the live league's name map.)
      var nameCopy = {};
      if (prev.slotNames){ Object.keys(prev.slotNames).forEach(function(k){ nameCopy[k] = prev.slotNames[k]; }); }
      // Carry the per-pick owner map so a mock of a connected draft credits traded
      // picks to the team that traded for them, exactly like the live board.
      var ownerCopy = {};
      if (prev.pickOwners){ Object.keys(prev.pickOwners).forEach(function(k){ ownerCopy[k] = prev.pickOwners[k]; }); }
      _resetTransient();
      state = {
        type: prev.type, teams: prev.teams, rounds: prev.rounds, sf: !!prev.sf,
        // Carry the keeper setup so a practice mock of a keeper league still
        // spends the same picks on keepers.
        keeper: !!prev.keeper, keeperCount: prev.keeperCount || 0,
        keeperSource: prev.keeperSource || null,
        slot: prev.slot, order: prev.order,
        roster: prev.roster || defaultRoster(!!prev.sf, prev.type === 'redraft'),
        scoring: prev.scoring || scoringCfg(),
        picks: {}, current: 1, queue: [],
        owned: Object.keys(ownedCopy).length ? ownedCopy : defaultOwned(),
        slotNames: nameCopy,
        pickOwners: Object.keys(ownerCopy).length ? ownerCopy : null,
        mode: 'mock', simStarted: false
      };
      sim = true; simPaused = false; simStarted = false;
      var sp = document.getElementById('drSimSpeed');
      simSpeed = parseInt(sp.value, 10) || 700;
      document.getElementById('drLiveBadge').style.display = 'none';
      document.getElementById('drUpcomingBadge').style.display = 'none';
      document.getElementById('drSide').style.display = '';
      syncSimControls();
      _setUpcomingMode(false);
      save();
      resetSideTabs();   // clear any leftover completed-draft sidebar state
      showMain();
      loadPlayers();
    });
  }

  // ── Command-center panels ───────────────────────────────────────────────────
  function availablePool(){ return players.filter(function(p){ return !drafted[String(p.id)]; }); }
  function myPicksList(){
    var out = [];
    if (!hasOwned()) return out;
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (isMyPick(pn)) out.push(state.picks[k]);
    });
    return out;
  }
  function posTargets(){
    var rs = (state && state.roster) || defaultRoster();
    if (window.DraftBoardCore) return DraftBoardCore.posTargets(rs, scoringCfg().tep);
    var flex = rs.FLEX||0, sf = rs.SF||0, bn = rs.BN||0;
    // Targets are depth SUGGESTIONS, not hard needs. A deep startup bench (drafts
    // run 20+ rounds) shouldn't imply you "need" 8 RBs or 10 WRs, so the bench
    // contribution is capped and each position is held to a realistic ceiling.
    var benchEff = Math.min(bn, 8);
    var tep = scoringCfg().tep;
    var t = {
      QB: (rs.QB||0) + sf        + (sf && benchEff >= 5 ? 1 : 0),
      RB: (rs.RB||0) + flex      + Math.ceil(benchEff * 0.45),
      WR: (rs.WR||0)             + Math.floor(benchEff * 0.45),
      // Single-start TE with no TE-premium: target just the starter, so a second
      // TE reads as low-priority depth and the redundancy penalty suppresses it
      // until the very late rounds (where a backup TE is normal). Padding the TE
      // target with a bench share made it 2, which kept "TE need" alive after the
      // starter was filled and let a 2nd TE grade as a top pick. TE Premium
      // restores backup-TE depth via the bench share plus the +1 below.
      TE: (rs.TE||0)             + (tep > 0 && benchEff >= 5 ? 1 : 0)
    };
    // Sane ceilings so the assistant never frames an absurd amount of depth as a need.
    // (1QB backup-QB timing is handled per-team in the sim, relative to when each
    // team drafted its starter - see simPick - not by a blanket round cap here.)
    var cap = { QB: sf ? 4 : Math.max(1, rs.QB||0), RB: 7, WR: 7,
                TE: tep > 0 ? Math.max(3, rs.TE||0) : Math.max(1, rs.TE||0) };
    Object.keys(cap).forEach(function(k){ if (t[k] > cap[k]) t[k] = cap[k]; });
    if (rs.K)   t.K   = rs.K;
    if (rs.DEF) t.DEF = rs.DEF;
    return t;
  }
  function myPosCounts(){
    var c = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    myPicksList().forEach(function(p){ var pos = (p.position||'').toUpperCase(); if (c[pos] != null) c[pos]++; });
    return c;
  }
  function listInto(html){ document.getElementById('drBaList').innerHTML = html; }

  // Compact empty copy for the draft side panel (queue / best / needs / league).
  // Draft Room does not load app.js, so this mirrors the shared empty-state look.
  var _DR_EMPTY_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5 5.2 4.6A2 2 0 0 1 7 3.6h10a2 2 0 0 1 1.8 1L21 8.5"/><path d="M3 8.5h5l1.2 2.2h5.6L16 8.5h5v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>';
  var _DR_SEARCH_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.7-4.7"/></svg>';
  function emptyNote(title, message, iconSvg){
    return '<div class="dr-empty-note">'
      + '<span class="dr-empty-note-icon" aria-hidden="true">' + (iconSvg || _DR_EMPTY_ICON) + '</span>'
      + (title ? '<p class="dr-empty-note-title">' + title + '</p>' : '')
      + (message ? '<p class="dr-empty-note-msg">' + message + '</p>' : '')
      + '</div>';
  }
  function loadingNote(message){
    return '<div class="dr-loading-msg"><div class="loading-spinner" aria-hidden="true"></div><span>'
      + (message || 'Loading…') + '</span></div>';
  }

  // ── Tiers + cliffs ──────────────────────────────────────────────────────────
  // Mirrors assign_tier() in value_translation.py: maps a 0-100 prospect grade
  // to its rookie-class tier (1 = elite). Fallback when prospect_tier is absent.
  function prospectTier(score){
    if (score >= 85) return 1;
    if (score >= 72) return 2;
    if (score >= 60) return 3;
    if (score >= 44) return 4;
    if (score >= 33) return 5;
    return 6;
  }
  function tierOf(p){
    var _tp = (p && p.position || '').toUpperCase();
    if (_tp === 'K' || _tp === 'DEF') return null;   // K/DEF aren't tiered
    if (state.type === 'redraft') return null;   // tiers are keyed to dynasty value
    // Rookie drafts: use the prospect grade's tier from the prospects page
    // (keyed to the rookie class), not all-player dynasty value tiers.
    if (state.type === 'rookie'){
      if (p && p.prospect_tier != null) return Number(p.prospect_tier);
      if (p && p.prospect_score != null) return prospectTier(Number(p.prospect_score));
      return null;
    }
    var lt = state.sf ? 'sf' : '1qb';
    var sz = String(state.teams);
    var tbl = (tierThresholds[lt] || {})[sz]
           || (tierThresholds[lt] || {})['12']
           || (tierThresholds['1qb'] || {})['12']
           || (tierThresholds['1qb'] || {})['10']
           || [];
    if (!tbl.length) return null;
    var v = valOf(p);
    for (var i = 0; i < tbl.length; i++){ if (v >= tbl[i]) return i + 1; }
    return tbl.length + 1;
  }
  // Count of still-available players per (position|tier) — drives cliff alerts.
  function posTierCounts(pool){
    pool = pool || availablePool();
    var m = {};
    pool.forEach(function(p){
      var t = tierOf(p); if (t == null) return;
      var k = (p.position || '').toUpperCase() + '|' + t;
      m[k] = (m[k] || 0) + 1;
    });
    return m;
  }
  var _ptc = {};   // refreshed each render
  function isTierCliff(p, pickNo){
    // A naturally small elite tier is not a "cliff" before the room has made a
    // full pass. At 1.01 (and throughout Round 1) every position is still at its
    // baseline, so scarcity copy such as "only 2 left" is noise, not urgency.
    var pn = pickNo != null ? +pickNo : ((state && state.current) || 1);
    if (pn <= ((state && state.teams) || 12)) return false;
    var t = tierOf(p); if (t == null) return false;
    var k = (p.position || '').toUpperCase() + '|' + t;
    return (_ptc[k] || 0) <= 2;     // tier is drying up
  }
  // Top-tier (T1+T2) players still available at a position - scarcity signal.
  function posTopRemaining(pos){
    pos = (pos || '').toUpperCase();
    return (_ptc[pos + '|1'] || 0) + (_ptc[pos + '|2'] || 0);
  }

  // ── Value Over Replacement (VOR) ────────────────────────────────────────────
  // Replacement level = value of the last startable player at a position across
  // the league (teams x starters-per-team). VOR(p) = value(p) - replacement.
  var _repl = {};   // refreshed each render
  // Replacement level is computed from the TOTAL player pool (not just what's
  // still available) so it stays a fixed, preseason-style baseline. Using only
  // available players would make the baseline drop as starters get drafted,
  // handing late-round leftovers an inflated VOR.
  function computeReplacement(pool){
    pool = pool || players;
    var rs = (state && state.roster) || defaultRoster();
    var teams = state.teams || 12;
    // Empirical starter allocation (best-available fills each starting slot),
    // the SAME index the server grade uses (utils.pick_score.empirical_slot_
    // allocation), so VOR replacement matches across surfaces instead of the old
    // fixed half-QB/half-RB/half-WR guess. Falls back to the starterCounts
    // heuristic only if the shared core (or its allocator) isn't loaded.
    var starters = (window.DraftBoardCore && DraftBoardCore.effectiveStarters)
      ? DraftBoardCore.effectiveStarters(pool, rs, teams, valOf)
      : BRPickScore.starterCounts(rs);
    // Shared kernel (static/draft_board_core.js): same value fn, same starters,
    // same indexing as the fallback below — one implementation for the Draft Room
    // and the Cheat Sheet. Fallback kept in case the core script fails to load.
    if (window.DraftBoardCore) return DraftBoardCore.computeReplacement(pool, valOf, starters, teams);
    var byPos = { QB: [], RB: [], WR: [], TE: [] };
    pool.forEach(function(p){
      var pos = (p.position || '').toUpperCase();
      if (byPos[pos]) byPos[pos].push(valOf(p));
    });
    var r = {};
    Object.keys(byPos).forEach(function(pos){
      var arr = byPos[pos]; arr.sort(function(a, b){ return b - a; });
      if (!arr.length){ r[pos] = 0; return; }
      var idx = Math.round(teams * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      r[pos] = arr[idx];
    });
    return r;
  }
  function vorOf(p){
    var pos = (p.position || '').toUpperCase();
    if (_repl[pos] == null) return null;
    return Math.round(valOf(p) - _repl[pos]);
  }

  // ── PPG normalization (production) ──────────────────────────────────────────
  // A VOR-style scale for fantasy points so current/projected production matters
  // in the pick score. Replacement-level PPG maps to ~0, elite (top-of-position)
  // PPG maps to ~1, putting QB/RB/WR/TE on a common 0-1 axis despite very
  // different raw point levels (a 22-PPG QB and a 13-PPG TE can both be "elite").
  var _ppgScale = {};   // refreshed each render: { POS: { repl, elite } }
  // Same rationale as computeReplacement: anchor the PPG scale to the total pool
  // so late-round leftovers don't read as elite once the board thins.
  function computePpgScale(pool){
    pool = pool || players;
    var rs = (state && state.roster) || defaultRoster();
    var teams = state.teams || 12;
    // Anchor the PPG replacement index to the SAME empirical starter allocation
    // as VOR and the server grade; K/DEF (which the pick score never grades) keep
    // their raw slot counts. Falls back to the fixed half-split heuristic when the
    // shared core's allocator isn't loaded.
    var emp = (window.DraftBoardCore && DraftBoardCore.effectiveStarters)
      ? DraftBoardCore.effectiveStarters(pool, rs, teams, valOf) : null;
    var flex = rs.FLEX || 0, sf = rs.SF || 0;
    var starters = {
      QB: emp ? emp.QB : (rs.QB || 0) + sf * 0.5,
      RB: emp ? emp.RB : (rs.RB || 0) + flex * 0.5,
      WR: emp ? emp.WR : (rs.WR || 0) + flex * 0.5,
      TE: emp ? emp.TE : (rs.TE || 0),
      K:  (rs.K || 0),
      DEF:(rs.DEF || 0)
    };
    var byPos = { QB: [], RB: [], WR: [], TE: [], K: [], DEF: [] };
    pool.forEach(function(p){
      var pos = (p.position || '').toUpperCase();
      var v = ppgOf(p);
      if (byPos[pos] && v != null) byPos[pos].push(v);
    });
    var out = {};
    Object.keys(byPos).forEach(function(pos){
      var arr = byPos[pos]; if (!arr.length) return;
      arr.sort(function(a, b){ return b - a; });
      // Elite anchor: mean of the top few (robust against a single outlier).
      var topN = Math.max(1, Math.min(3, arr.length));
      var eliteSum = 0; for (var i = 0; i < topN; i++) eliteSum += arr[i];
      var elite = eliteSum / topN;
      // Replacement anchor: PPG at the last startable slot leaguewide.
      var idx = Math.round(teams * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      out[pos] = { repl: arr[idx], elite: elite };
    });
    return out;
  }
  function ppgNormOf(p){
    if (window.DraftBoardCore) return DraftBoardCore.ppgNorm(p, _ppgScale, ppgOf);
    var pos = (p.position || '').toUpperCase();
    var v = ppgOf(p);
    var sc = _ppgScale[pos];
    if (v == null || !sc) return null;
    var span = sc.elite - sc.repl;
    if (span <= 0) return clamp01(v / Math.max(sc.elite, 1));
    return clamp01((v - sc.repl) / span);
  }

  // ── Board Pick Score display scale ──────────────────────────────────────────
  // The board shows Pick Score RELATIVE to the best pick currently AVAILABLE, not
  // on the whole-draft absolute scale. So a strong pick late in the draft reads
  // well (the best remaining option anchors near the top) instead of being buried
  // just because the board is picked over. This is display-only: the report-card
  // grade keeps the absolute, round-weighted score (it recomputes with
  // grading:true), so grades stay accurate and comparable across the draft.
  var _psPoolMax = 0;   // best raw pick score currently available; refreshed each render
  // Recompute p._ps for every available player (one shared pass) and return the
  // highest, which anchors the display scale.
  function refreshPsPool(){
    var pool = availablePool();
    var counts = myPosCounts();
    var maxV = 0; pool.forEach(function(p){ var v = valOf(p); if (v > maxV) maxV = v; });
    var mx = 0;
    pool.forEach(function(p){
      var s = pickScore(p, maxV, counts);
      p._ps = s;
      if (s != null && s > mx) mx = s;
    });
    prepareNextPickValues(pool);
    pool.forEach(function(p){ p._ds = liveDecisionScore(p, counts); });
    _psPoolMax = mx;
    return mx;
  }
  // Canonical all-position Recommendation order. Position/search filters are
  // views into this list, not new rankings; RB #7 remains REC #7 when the user
  // switches from All to RB instead of being relabeled REC #1.
  function rankedRecommendationPool(){
    return availablePool().filter(function(p){ return p._ds != null; })
      .sort(function(a,b){ return b._ds - a._ds || (b._ps || 0) - (a._ps || 0); });
  }
  // Map a raw pick score onto the pool-relative display scale. Live board,
  // sidebar, compare modal, and player preview all use this so a strong pick
  // on a depleted board still reads well. Report-card / Deep Dive “Board PS”
  // uses the same scale at the historical slot (via relPS). Recommendation
  // rows still carry Decision rank separately (see playerRowHtml).
  function psDisplay(ps){
    if (ps == null) return null;
    if (!_psPoolMax || _psPoolMax <= 0) return ps;
    var d = Math.round(97 * ps / _psPoolMax);
    return d > 99 ? 99 : (d < 1 ? 1 : d);
  }

  // Pool-relative Pick Score for live surfaces (sidebar, compare, preview).
  // Same 0-100 chip playerRowHtml shows, so those modals never disagree with
  // the board. Report-card grades still recompute with grading:true on the
  // absolute kernel.
  function psRelLive(p){
    return psDisplay(p._ps != null ? p._ps : pickScoreFor(p));
  }

  // ── Pool-relative score for ALREADY-MADE picks (report card) ────────────────
  // Mock/manual picks capture psRel at commit time. Synced (live) picks don't go
  // through commitPick, so reconstruct each pick's "vs best available then" score
  // from the draft order: for each pick, rebuild the pool as it stood (minus every
  // player taken before it) and rank the pick's raw score against the best pick
  // available at that slot. Computed once and cached per pick-set; anchored to the
  // top players by value so it stays cheap on a full board.
  var _relCache = { sig: null, map: {} };
  function _relSig(){
    var n = 0; Object.keys(state.picks).forEach(function(k){ if (state.picks[k]) n++; });
    return n + '@' + state.current + '@' + (state.mode || '') + '@' + players.length;
  }
  function _pnOf(pl){
    if (!pl || pl.id == null) return 0;
    var found = 0;
    Object.keys(state.picks).forEach(function(k){
      if (state.picks[k] && String(state.picks[k].id) === String(pl.id)) found = parseInt(k, 10);
    });
    return found;
  }
  function _ensureRelScores(){
    var sig = _relSig();
    if (_relCache.sig === sig) return _relCache.map;
    var map = {};
    if (players.length){
      var K = 60;   // anchor candidates: the best pick score is among top-value names
      var byVal = players.slice().sort(function(a, b){ return valOf(b) - valOf(a); });
      var order = Object.keys(state.picks).filter(function(k){ return state.picks[k]; })
        .map(function(k){ return parseInt(k, 10); }).sort(function(a, b){ return a - b; });
      var taken = {};
      order.forEach(function(pn){
        var pk = state.picks[pn];
        var pkFull = playersById[String(pk.id)];
        var cand = [], maxV = 0;
        for (var i = 0; i < byVal.length && cand.length < K; i++){
          var q = byVal[i]; if (taken[String(q.id)]) continue;
          if (!cand.length) maxV = valOf(q);
          cand.push(q);
        }
        if (pkFull && !cand.some(function(c){ return String(c.id) === String(pk.id); })) cand.push(pkFull);
        var best = 0, mine = 0;
        for (var j = 0; j < cand.length; j++){
          var s = pickScore(cand[j], maxV, {}, { grading: true, pickNo: pn });
          if (s != null){ if (s > best) best = s; if (String(cand[j].id) === String(pk.id)) mine = s; }
        }
        if (best > 0 && mine > 0){ var d = Math.round(97 * mine / best); map[pn] = d > 99 ? 99 : (d < 1 ? 1 : d); }
        else if (pk.psRel != null){ map[pn] = pk.psRel; }
        taken[String(pk.id)] = true;
      });
    }
    _relCache = { sig: sig, map: map };
    return map;
  }
  // Pool-relative score for a made pick: the commit-time capture (mock), else the
  // reconstruction (synced), else the absolute score as a last resort.
  function relPS(pl, pn){
    if (pl && pl.psRel != null) return pl.psRel;
    if (!pn) pn = _pnOf(pl);
    var m = _ensureRelScores();
    if (m[pn] != null) return m[pn];
    return pl ? storedPickScore(pn, pl) : null;
  }

  // Per-render pickScore context: posTargets() and my above-replacement counts by
  // position are identical for every player scored in a pass, so compute them once
  // instead of re-running posTargets() + a full myPicksList() scan inside pickScore
  // for every player in the pool (was O(pool x myPicks) per render). Invalidated at
  // the top of each renderSide; rebuilt lazily on first use.
  var _psCtxCache = null;
  function psCtxInvalidate(){ _psCtxCache = null; }
  function psCtx(){
    if (_psCtxCache) return _psCtxCache;
    var targets = posTargets();
    var qualByPos = {}, lastPickByPos = {}, rosterQualities = [];
    myPicksList().forEach(function(mp){
      var pos = (mp.position || '').toUpperCase();
      var full = playersById[String(mp.id)];
      var v = full ? vorOf(full) : null;
      if (v == null || v > 0) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
      var q = full ? ppgNormOf(full) : null;
      rosterQualities.push({ pos: pos, quality: q != null ? q : 0.35 });
    });
    Object.keys(state.picks || {}).forEach(function(k){
      var pn = parseInt(k, 10), mp = state.picks[k];
      if (!mp || !isMyPick(pn)) return;
      var pos = (mp.position || '').toUpperCase();
      if (!lastPickByPos[pos] || pn > lastPickByPos[pos]) lastPickByPos[pos] = pn;
    });
    var rs = (state && state.roster) || defaultRoster();
    var remaining = hasOwned() ? upcomingOwnedPicks().length : 0;
    var obligations = window.DraftBoardCore
      ? DraftBoardCore.remainingObligations(myPosCounts(), rs, remaining, !!state.sf)
      : { missing: {}, required: 0, remaining: remaining, freePicks: remaining };
    _psCtxCache = { targets: targets, qualByPos: qualByPos, rosterQualities: rosterQualities,
                    lastPickByPos: lastPickByPos, roster: rs,
                    remaining: remaining, obligations: obligations, nextByPos: {} };
    return _psCtxCache;
  }

  // ── Availability probability ────────────────────────────────────────────────
  // Model a player's actual draft slot as Normal(ADP, sigma); the chance they
  // survive to overall pick `pn` is P(slot >= pn) = 1 - CDF((pn - ADP)/sigma).
  function _normCdf(z){
    var t = 1 / (1 + 0.2316419 * Math.abs(z));
    var d = 0.3989423 * Math.exp(-z * z / 2);
    var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z >= 0 ? 1 - p : p;
  }
  // Learned draft tendencies from the picks already made. Lets the survival odds
  // adapt to how THIS draft is actually going - a room that reaches pulls every
  // player's expected slot earlier; a room that lets players slide pushes it
  // later; a noisy room widens the spread; and a position going on a run is less
  // likely to last. In a CPU mock the picks track ADP, so the bias is ~0 and this
  // stays neutral; in a live draft it adapts to real humans.
  var _draftModelCache = null;
  function draftModelInvalidate(){ _draftModelCache = null; }
  function observedDraftModel(){
    if (_draftModelCache) return _draftModelCache;
    var residuals = [];        // pick# - ADP for each completed pick (>0 = slid, <0 = reach)
    var made = [];             // {pn, pos} sorted by pn, for positional-run detection
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      var pk = state.picks[k]; if (!pk) return;
      made.push({ pn: pn, pos: (pk.position || '').toUpperCase() });
      var full = playersById[String(pk.id)];
      var a = full ? adpOf(full) : null;
      if (a != null) residuals.push(pn - a);
    });
    made.sort(function(x, y){ return x.pn - y.pn; });
    var n = residuals.length;
    var model = { n: n, bias: 0, std: null, run: {} };
    if (n >= 1){
      var sum = 0; residuals.forEach(function(r){ sum += r; });
      var mean = sum / n;
      model.bias = Math.max(-10, Math.min(10, mean));   // clamp so a few wild picks can't dominate
      if (n >= 2){
        var v = 0; residuals.forEach(function(r){ v += (r - mean) * (r - mean); });
        model.std = Math.sqrt(v / (n - 1));
      }
    }
    // Positional run: compare each position's rate in the last round to its rate
    // across the whole draft, so naturally-popular positions (WR) aren't flagged
    // unless they're going FASTER than their own established pace.
    var teams = state.teams || 12;
    var totalMade = made.length;
    if (totalMade >= teams){
      var win = made.slice(-teams);
      var overall = {}, recent = {};
      made.forEach(function(m){ overall[m.pos] = (overall[m.pos] || 0) + 1; });
      win.forEach(function(m){ recent[m.pos] = (recent[m.pos] || 0) + 1; });
      Object.keys(recent).forEach(function(pos){
        var expected = (overall[pos] / totalMade) * teams;
        var excess = recent[pos] - expected;
        model.run[pos] = excess > 0 ? Math.min(0.25, (excess / teams) * 1.5) : 0;
      });
    }
    _draftModelCache = model;
    return model;
  }
  function availProb(p, pn){
    var a = adpOf(p);
    if (a == null) return null;
    // Baseline: model the slot as Normal(ADP, simSigma) - the same spread the CPU
    // sim draws from, so mock odds match on-board behavior.
    var sigma = simSigma(a);
    var center = a;
    var m = observedDraftModel();
    // Once a meaningful chunk of the board has gone, fold in how this specific
    // draft is behaving. Confidence ramps from 8 picks to full by ~28.
    if (m.n >= 8){
      var conf = Math.min(1, (m.n - 8) / 20);
      center = a + m.bias * conf;                       // reach pulls earlier, slide pushes later
      if (m.std != null){
        var obs = Math.max(simSigma(a) * 0.6, Math.min(m.std, 18));
        sigma = sigma * (1 - conf) + obs * conf;        // blend toward observed unpredictability
      }
    }
    // A position going on a run is less likely to make it back to you.
    var runPen = m.run[(p.position || '').toUpperCase()] || 0;
    if (window.DraftBoardCore && DraftBoardCore.availabilityProbability){
      return DraftBoardCore.availabilityProbability({ center:center, pick:pn, sigma:sigma, runPenalty:runPen,
        draftType:state.type, sf:!!state.sf });
    }
    var prob = 1 - _normCdf((pn - center) / sigma);
    if (runPen > 0) prob *= (1 - runPen);
    return Math.round(prob * 100);
  }
  function availColor(pct){ return pct >= 65 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'; }

  function recRankOf(p){
    if (!p) return null;
    var pool = rankedRecommendationPool();
    for (var i = 0; i < pool.length; i++){
      if (String(pool[i].id) === String(p.id)) return i + 1;
    }
    return null;
  }
  function expLabel(p){
    if (!p) return '';
    if (p.is_rookie) return 'Rookie';
    var ye = p.years_exp;
    if (ye == null || ye === '') return '';
    ye = Number(ye);
    if (!isFinite(ye) || ye < 0) return '';
    if (ye === 0) return 'Rookie';
    return ye + ' yr';
  }
  // Shared snapshot of everything the player preview and compare modal show so
  // the two surfaces cannot drift (and so we only compute ADP/VOR/survival once).
  function draftPlayerFacts(p){
    var pos = (p.position || '').toUpperCase();
    var adp = adpOf(p);
    var adpN = (state.type === 'rookie') ? p.rookie_adp_n
      : (state.sf ? p.sf_adp_n : p.adp_n);
    var adpGap = (adp != null && state && state.current) ? (state.current - adp) : null;
    var proj = (p.proj_ppg != null && isFinite(Number(p.proj_ppg))) ? Number(p.proj_ppg) : null;
    var last = (p.ppg != null && isFinite(Number(p.ppg))) ? Number(p.ppg) : null;
    var nextOwned = nextOwnedAfterCurrent();
    var survive = nextOwned ? availProb(p, nextOwned) : null;
    var vorp = p.vorp != null ? Number(p.vorp) : vorOf(p);
    return {
      pos: pos,
      adp: adp,
      adpN: adpN != null && isFinite(Number(adpN)) ? Number(adpN) : null,
      vsAdp: adpGap,
      vor: vorp,
      vorLbl: p.vorp != null ? 'VORP' : 'VOR',
      value: valOf(p),
      tier: tierOf(p),
      age: p.age != null ? Number(p.age) : null,
      projPpg: proj,
      lastPpg: last,
      ppgSeason: p.ppg_season,
      ppgRank: p.ppg_rank,
      posRank: state && state.sf ? (p.sf_pos_rank_label || '') : (p.pos_rank_label || ''),
      posRankN: state && state.sf ? p.sf_pos_rank : p.pos_rank,
      bye: p.bye_week != null ? Number(p.bye_week) : null,
      rec: recRankOf(p),
      projPts: p.proj_pts != null ? Number(p.proj_pts) : null,
      scarce: posTopRemaining(pos),
      survive: survive,
      survivePn: nextOwned,
      market: p.market_vs_adp != null ? Number(p.market_vs_adp) : null,
      exp: expLabel(p),
      injury: p.injury || ''
    };
  }
  function fmtSigned(n, digits){
    if (n == null || !isFinite(Number(n))) return '—';
    var x = Number(n);
    var s = digits != null ? x.toFixed(digits) : (Number.isInteger(x) ? String(x) : x.toFixed(1));
    if (Number(s) === 0) return digits != null ? Number(0).toFixed(digits) : '0';
    return (Number(s) > 0 ? '+' : '') + s;
  }

  // ── Best-at-position chips + scarcity bar ───────────────────────────────────
  function renderBestChips(){
    var el = document.getElementById('drBestChips');
    if (!el) return;
    if (sideTab !== 'best'){ el.style.display = 'none'; return; }
    var pool = availablePool();
    if (!pool.length){ el.style.display = 'none'; return; }
    var isDynasty = (state.type !== 'redraft');
    var positions = state.type === 'redraft' ? ['QB','RB','WR','TE','K','DEF'] : ['QB','RB','WR','TE'];
    // T1-2 counts: remaining top-tier players per position (dynasty only)
    var scarHtml = '';
    if (isDynasty){
      scarHtml = '<div class="dr-bchips-section-title">T1-2 Counts</div><div class="dr-scarcity">';
      positions.forEach(function(pos){
        var n = posTopRemaining(pos);
        var col = n <= 3 ? '#ef4444' : n <= 6 ? '#f59e0b' : '#22c55e';
        scarHtml += '<div class="dr-scar-pos" data-scarpos="' + pos + '" title="' + n + ' T1-T2 ' + pos + 's available">'
          + '<span class="dr-scar-count" style="color:' + col + '">' + n + '</span>'
          + '<span class="dr-scar-label">' + pos + '</span>'
          + '</div>';
      });
      scarHtml += '</div>';
    }
    // Best-at-pos: lowest ADP (or highest value) per position
    var byPos = {};
    pool.forEach(function(p){
      var pos = (p.position || '').toUpperCase();
      if (positions.indexOf(pos) < 0) return;
      if (!byPos[pos]){
        byPos[pos] = p;
      } else {
        var a = adpOf(p), ab = adpOf(byPos[pos]);
        if (a != null && ab != null){ if (a < ab) byPos[pos] = p; }
        else if (a != null){ byPos[pos] = p; }
        else if (a == null && ab == null && valOf(p) > valOf(byPos[pos])){ byPos[pos] = p; }
      }
    });
    var hasAny = false;
    var bchipsInner = '';
    positions.forEach(function(pos){
      var p = byPos[pos]; if (!p) return;
      hasAny = true;
      var adp = adpOf(p);
      var sub = adp != null ? 'ADP ' + Number(adp).toFixed(1) : 'Val ' + Math.round(valOf(p));
      var lastName = p.name.split(' ').slice(1).join(' ') || p.name;
      bchipsInner += '<div class="dr-bchip" data-bchip="' + esc(String(p.id)) + '">'
        + '<img class="dr-bchip-img" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div class="dr-bchip-body">'
        + '<div class="dr-bchip-name"><span class="dr-posbadge" style="background:' + posColor(pos) + ';font-size:8px;">' + pos + '</span> ' + esc(lastName) + '</div>'
        + '<div class="dr-bchip-adp">' + sub + '</div>'
        + '</div></div>';
    });
    var bchipsHtml = '<div class="dr-bchips-section-title">Best Available</div><div class="dr-bchips">' + bchipsInner + '</div>';
    if (!hasAny){ el.style.display = 'none'; return; }
    // Single outer toggle collapses both T1-2 Counts and Best Available together
    var toggleHtml = '<div class="dr-bchips-header" id="drBestChipsToggle">'
      + '<span class="dr-bchips-label">Player Tools</span>'
      + '<span class="dr-bchips-hint">' + (_chipsCollapsed ? '&#9654; Show' : '&#9660; Hide') + '</span>'
      + '</div>';
    var contentHtml = _chipsCollapsed ? '' : (scarHtml + bchipsHtml);
    el.innerHTML = toggleHtml + contentHtml;
    el.style.display = '';
  }

  // ── Balance alert ───────────────────────────────────────────────────────────
  // Fires late in drafts when a critical position is severely underfilled.
  function balanceAlert(){
    // Rookie drafts: managers often own only a few scattered picks, so a
    // "picks remaining" balance banner is noise. Skip it entirely.
    if (state.type === 'rookie') return '';
    if (!hasOwned()) return '';
    var remaining = upcomingOwnedPicks().length;
    if (remaining <= 0 || remaining > 8) return '';
    var counts = myPosCounts();
    var targets = posTargets();
    var msgs = [];
    ['QB','RB','WR','TE'].forEach(function(pos){
      var t = targets[pos] || 0;
      var have = counts[pos] || 0;
      var need = Math.max(0, t - have);
      if (!need) return;
      if (need >= remaining || (have === 0 && remaining <= 4)){
        msgs.push(need + ' ' + pos + (need > 1 ? 's' : '') + ' needed');
      }
    });
    if (!msgs.length) return '';
    var picks = remaining === 1 ? '1 pick' : remaining + ' picks';
    return '<div class="dr-bal-alert"><b>' + picks + ' remaining</b>: ' + msgs.join(', ') + '</div>';
  }

  // ── Bye week conflict ───────────────────────────────────────────────────────
  // Returns how many already-owned players share this player's bye week.
  function byeConflict(p){
    if (state.type !== 'redraft' || !p.bye_week) return 0;
    var bw = Number(p.bye_week);
    var count = 0;
    myPicksList().forEach(function(mp){
      var full = playersById[String(mp.id)];
      if (full && Number(full.bye_week) === bw) count++;
    });
    return count;
  }

  // ── Player comparison ───────────────────────────────────────────────────────
  function toggleCompare(id){
    id = String(id);
    var idx = compareIds.indexOf(id);
    if (idx >= 0){
      compareIds.splice(idx, 1);
      renderSide();
    } else if (compareIds.length >= 2){
      compareIds = [id];
      renderSide();
    } else {
      compareIds.push(id);
      if (compareIds.length === 2) openCompare();
      else renderSide();
    }
  }
  function closeCompare(){
    document.getElementById('drCompare').style.display = 'none';
    compareIds = [];
    renderSide();
  }
  function openCompare(){
    var p1 = playersById[String(compareIds[0])];
    var p2 = playersById[String(compareIds[1])];
    if (!p1 || !p2) return;
    // Compare is a live surface: reuse the sidebar's pool-relative Pick Score
    // so the modal matches the chips next to each name. refreshPsPool is the
    // same guard renderBA uses for any path that didn't just renderSide.
    if (_psPoolMax <= 0) refreshPsPool();
    function cmpCol(p, other){
      var f = draftPlayerFacts(p);
      var o = draftPlayerFacts(other);
      var ps = psRelLive(p);
      var vorLbl = (p.vorp != null || other.vorp != null) ? 'VORP' : 'VOR';
      var ppgRowLbl = 'Proj PPG';
      var ppg = f.projPpg;
      var oppg = o.projPpg;
      function statRow(lbl, val, oval, higherBetter, fmtFn){
        if (val == null && oval == null) return '';
        var vStr = fmtFn ? fmtFn(val) : (val != null ? String(val) : '-');
        var win = val != null && oval != null && (higherBetter ? val > oval : val < oval);
        return '<div class="dr-cmp-stat' + (win ? ' win' : '') + '">'
          + '<span class="dr-cmp-stat-lbl">' + lbl + '</span>'
          + '<span class="dr-cmp-stat-val">' + vStr + '</span></div>';
      }
      var sc = ps != null ? psColor(ps) : 'var(--text-muted)';
      var metaBits = [p.team || '', f.exp, (f.age ? 'Age ' + f.age.toFixed(0) : ''), (f.injury ? f.injury : '')].filter(Boolean);
      return '<div class="dr-cmp-player">'
        + '<div class="dr-cmp-top"><img class="dr-cmp-hs" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div><div class="dr-cmp-name"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span> ' + esc(p.name) + '</div>'
        + '<div class="dr-cmp-meta">' + esc(metaBits.join(' · ')) + '</div>'
        + '</div></div>'
        + '<div class="dr-cmp-ps" style="color:' + sc + '">' + (ps != null ? ps : '&ndash;') + '</div>'
        + '<div class="dr-cmp-ps-lbl">Pick Score</div>'
        + '<div class="dr-cmp-stats">'
        + statRow('Value', f.value, o.value, true, function(x){ return x != null ? String(Math.round(x)) : '-'; })
        + statRow(ppgRowLbl, ppg, oppg, true, function(x){ return x != null ? x.toFixed(1) : 'N/A'; })
        + (f.lastPpg != null || o.lastPpg != null ? statRow((f.ppgSeason || 'Last') + ' PPG', f.lastPpg, o.lastPpg, true, function(x){ return x != null ? x.toFixed(1) : '-'; }) : '')
        + statRow(vorLbl, f.vor, o.vor, true, function(x){ return x != null ? fmtSigned(x, Number.isInteger(x) ? 0 : 1) : '-'; })
        + statRow('ADP', f.adp, o.adp, false, function(x){ return x != null ? Number(x).toFixed(1) : 'N/A'; })
        + statRow('vs ADP', f.vsAdp, o.vsAdp, true, function(x){ return fmtSigned(Math.round(x), 0); })
        + (f.posRank || o.posRank ? statRow('Pos Rank', f.posRankN, o.posRankN, false, function(x){
            if (x == null) return '-';
            if (f.posRankN === x && f.posRank) return f.posRank;
            if (o.posRankN === x && o.posRank) return o.posRank;
            return String(x);
          }) : '')
        + (state.type !== 'redraft' ? statRow('Tier', f.tier, o.tier, false, function(x){ return x != null ? 'T' + x : '-'; }) : '')
        + statRow('Age', f.age, o.age, false, function(x){ return x != null ? x.toFixed(0) : '-'; })
        + (f.bye != null || o.bye != null ? statRow('Bye', f.bye, o.bye, false, function(x){ return x != null ? String(x) : '-'; }) : '')
        + (f.rec != null || o.rec != null ? statRow('REC', f.rec, o.rec, false, function(x){ return x != null ? '#' + x : '-'; }) : '')
        + (f.survive != null || o.survive != null ? statRow('Survive', f.survive, o.survive, true, function(x){ return x != null ? x + '%' : '-'; }) : '')
        + (f.projPts != null || o.projPts != null ? statRow('Proj Pts', f.projPts, o.projPts, true, function(x){ return x != null ? String(Math.round(x)) : '-'; }) : '')
        + (f.market != null || o.market != null ? statRow('Mkt vs ADP', f.market, o.market, true, function(x){ return fmtSigned(Math.round(x), 0); }) : '')
        + '</div></div>';
    }
    var draftBtns = (state && state.mode !== 'live' && (isYourTurn() || !sim) && (!sim || simStarted))
      ? '<button class="dr-btn dr-btn-primary" data-cmp-draft="' + esc(String(p1.id)) + '">Draft ' + esc(p1.name.split(' ').pop()) + '</button>'
        + '<button class="dr-btn dr-btn-primary" data-cmp-draft="' + esc(String(p2.id)) + '">Draft ' + esc(p2.name.split(' ').pop()) + '</button>'
      : '';
    var h = '<button class="dr-cmp-close" id="drCmpClose" aria-label="Close">&times;</button>'
      + '<div class="dr-cmp-title">Compare Players</div>'
      + '<div class="dr-cmp-cols">' + cmpCol(p1, p2) + cmpCol(p2, p1) + '</div>'
      + (draftBtns ? '<div class="dr-cmp-actions">' + draftBtns + '</div>' : '');
    var card = document.getElementById('drCompareCard');
    card.innerHTML = h;
    document.getElementById('drCompare').style.display = '';
  }

  // ── Pick Score (composite) ──────────────────────────────────────────────────
  // Fuses VOR, raw value, ADP value, tier, quality-adjusted need, position-peak
  // age, momentum, and opportunity cost into one 0-100 score per player.
  // Weights branch by draft type so rookie/redraft/startup each prioritize correctly.
  function clamp01(x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }
  // opts (optional) overrides the "my team" context so the same scoring can be
  // run from a CPU team's perspective during simulation: { qualByPos, pickNo }.
  // (nextOwned/picksList are no longer read here now that the timing terms live
  // in the decision layer; callers may still pass them harmlessly.) When
  // omitted, pickScore scores for the viewer's own team exactly as before.
  function pickScore(p, maxVal, counts, opts){
    var pos = (p.position || '').toUpperCase();
    // Free agents have no current team and no real draft value for any format.
    var teamVal = (p.team || '').trim().toUpperCase();
    if (!teamVal || teamVal === 'FA') return 2;
    // K/DEF aren't graded - they're streamed/last-round picks with no meaningful
    // pick score. Return null so no PS chip renders and they're excluded from
    // grade math; they still appear in the pool (ranked by projected PPG) and the
    // sim drafts them late via their synthesized ADP.
    if (pos === 'K' || pos === 'DEF') return null;
    var adp = adpOf(p);

    // All grade math lives in static/pick_score.js (BRPickScore), shared with the
    // Python server grade (utils/pick_score.compute_pick_score) and pinned by a
    // parity test - so the Draft Room and Teams page can never grade a pick
    // differently. This wrapper only gathers inputs. The kernel is pure pick
    // QUALITY: live-draft timing (survival, handcuff) is applied later in
    // liveDecisionScore, NOT here, so the board's Pick Score chip IS the grade
    // (the `grading` opt no longer changes the formula, only the input context
    // callers pass). Nothing time-sensitive leaks into the historical grade.
    // Pick this score is being computed AT. Defaults to the clock, but a
    // keeper is scored at the pick his keeper round consumed.
    var _pn = (opts && opts.pickNo) || state.current;

    // Quality-adjusted need: two below-replacement RBs still leave a real need.
    var _ctx = psCtx();
    var _qualByPos = (opts && opts.qualByPos) || _ctx.qualByPos;
    var _t = _ctx.targets[pos];
    var needRaw = _t ? clamp01(Math.max(0, _t - (counts[pos] || 0)) / _t) : 0;
    var _qualNeed = _t ? clamp01(Math.max(0, _t - (_qualByPos[pos] || 0)) / _t) : 0;
    needRaw = Math.max(needRaw, _qualNeed);

    // Position-normalized PPG (null -> the formula falls back to value).
    var ppgN = ppgNormOf(p);

    var _sc = scoringCfg();
    var _cliff;
    if (opts && opts.isTierCliff != null) _cliff = !!opts.isTierCliff;
    else if (opts && opts.grading && _gradeCliffByPn && _pn && _gradeCliffByPn[_pn] != null)
      _cliff = !!_gradeCliffByPn[_pn];
    else _cliff = isTierCliff(p, _pn);
    return BRPickScore.computePickScore({
      pos: pos, value: valOf(p), vor: vorOf(p), tier: tierOf(p),
      age: (p.age != null ? Number(p.age) : null), rankChange7d: p.rank_change_7d,
      avgPick: adp, pickNo: _pn, maxVal: maxVal,
      draftType: state.type, isSf: state.sf, needRaw: needRaw,
      qbCount: counts['QB'] || 0, totalPicks: (state.teams || 12) * (state.rounds || 16),
      numTeams: state.teams || 12, ppgNorm: ppgN,
      ppr: _sc.ppr, tep: _sc.tep, passTd: _sc.passTd, isTierCliff: _cliff,
    });
  }

  // Live recommendations answer a different question from the historical Pick
  // Score: how much does this player help THIS roster before the next owned pick?
  // Keep these tunings together and outside the shared grade kernel so completed
  // draft grades remain stable and JS/Python parity is untouched.
  function rosterRoleFor(p, counts){
    var pos = String(p && p.position || p || '').toUpperCase();
    var c = psCtx(), opts = { sf: !!state.sf, tep: scoringCfg().tep, draftType: state.type };
    if (!window.DraftBoardCore) return 'bench1';
    if (p && p.position && DraftBoardCore.candidateRosterRole){
      return DraftBoardCore.candidateRosterRole(pos, ppgNormOf(p) || 0, c.rosterQualities, c.roster, opts.sf);
    }
    return DraftBoardCore.rosterRole(pos, counts, c.roster, opts.sf);
  }
  function rosterUtilityFor(p, counts, role){
    var pos = String(p && p.position || p || '').toUpperCase();
    var c = psCtx(), opts = { sf: !!state.sf, tep: scoringCfg().tep, draftType: state.type };
    if (!window.DraftBoardCore) return 1;
    opts.role = role;
    if (DraftBoardCore.positionNeedUtility)
      return DraftBoardCore.positionNeedUtility(pos, counts, c.roster, opts);
    return DraftBoardCore.rosterSlotUtility(pos, counts, c.roster, opts);
  }

  // One bounded pass per render finds the best plausible same-position option at
  // our next pick. This captures the cost of waiting without a simulation or an
  // O(players²) rescore: deep QB shelves produce a small urgency signal while a
  // thinning WR/RB shelf produces a larger one.
  function prepareNextPickValues(pool){
    var c = psCtx(), next = nextOwnedAfterCurrent(); c.nextByPos = {}; c.demandByPos = {};
    if (!next) return;
    c.demandByPos = _demandBeforeNext(next);
    ['QB','RB','WR','TE'].forEach(function(pos){
      var rows = pool.filter(function(p){ return String(p.position || '').toUpperCase() === pos && p._ps != null; })
        .sort(function(a, b){ return b._ps - a._ps; }).slice(0, 16);
      var bestExpected = 0;
      rows.forEach(function(p){
        var prob = availProb(p, next); if (prob == null) prob = 50;
        // A likely survivor contributes nearly all its quality; a long shot is
        // discounted. The best expected survivor is our inexpensive next-pick proxy.
        bestExpected = Math.max(bestExpected, p._ps * (0.35 + 0.65 * prob / 100));
      });
      c.nextByPos[pos] = bestExpected;
    });
  }

  var LIVE_WAIT_TUNING = { threshold: 50, maxPenalty: 10 };
  // Sub-threshold survival discount: below the 50% "more likely than not to
  // return" line the old ramp gave a flat zero, so a player with a real chance of
  // falling back to you (e.g. 25%) was treated exactly like one who is certain to
  // be gone (0%). A small continuous discount from LIVE_WAIT_SUBONSET% up to the
  // threshold restores that signal without disturbing the >=50% band materially
  // (it reaches only LIVE_WAIT_SUBSHARE of the max penalty at the threshold).
  var LIVE_WAIT_SUBONSET = 20;   // % return prob where the sub-threshold discount begins
  var LIVE_WAIT_SUBSHARE = 0.15; // fraction of maxPenalty reached at the threshold
  function liveDecisionScore(p, counts){
    var base = p._ps != null ? p._ps : 0;
    if (base == null) return null;
    var pos = String(p.position || '').toUpperCase();
    if (pos === 'K' || pos === 'DEF') return null;
    var c = psCtx(), role = rosterRoleFor(p, counts), util = rosterUtilityFor(p, counts, role);
    var expected = c.nextByPos[pos] || 0;
    if (!window.DraftBoardCore || !DraftBoardCore.decisionScore) return base;
    var bench = role === 'bench1' || role === 'bench2';
    var recentPenalty = 0;
    if (bench && (pos === 'QB' || pos === 'TE') && c.lastPickByPos[pos]){
      var roundsSince = (state.current - c.lastPickByPos[pos]) / Math.max(1, state.teams || 12);
      recentPenalty = Math.max(0, 10 * (1 - roundsSince / 6));
    }
    var adp = adpOf(p), exceptional = 0;
    if (adp != null) exceptional = clamp01(((state.current || 1) - adp) / Math.max(12, adp * 0.65));
    var nextPick = nextOwnedAfterCurrent();
    var returnProb = nextPick ? availProb(p, nextPick) : null;
    // Do not spend this pick on a player who is likely to return unless his
    // quality/fit advantage is large enough to overcome a bounded opportunity
    // cost. A true positional cliff still earns waitLoss above, and an extreme
    // ADP fall keeps half of this discount from becoming a disguised hard ban.
    // Visible position-needy opponents before our next turn reduce the nominal
    // ADP survival probability. This uses only public roster state—no knowledge
    // of future random CPU selections.
    var demand = (c.demandByPos && c.demandByPos[pos]) || 0;
    var demandRisk = Math.min(0.35, demand / Math.max(1, state.teams || 12) * 0.7);
    var effectiveReturnProb = returnProb == null ? null : returnProb * (1 - demandRisk);
    var _thr = LIVE_WAIT_TUNING.threshold;
    var _wpFrac;
    if (effectiveReturnProb == null) _wpFrac = 0;
    else if (effectiveReturnProb >= _thr)
      // Original >=50% ramp, floored at the small sub-threshold share so the two
      // segments meet continuously at the threshold.
      _wpFrac = LIVE_WAIT_SUBSHARE + (1 - LIVE_WAIT_SUBSHARE)
        * clamp01((effectiveReturnProb - _thr) / (100 - _thr));
    else
      _wpFrac = LIVE_WAIT_SUBSHARE
        * clamp01((effectiveReturnProb - LIVE_WAIT_SUBONSET) / (_thr - LIVE_WAIT_SUBONSET));
    var waitPenalty = _wpFrac * LIVE_WAIT_TUNING.maxPenalty * (1 - exceptional * 0.5);
    // Redraft handcuff insurance: a small point tilt toward the backup of one of
    // my own RBs. Formerly a term inside the pick-score kernel, it now lives here
    // on the decision scale so it is applied once and undistorted (the CPU sim
    // has always applied its own handcuff nudge separately, at line ~1704).
    var handcuffBonus = 0;
    if (state.type === 'redraft' && pos === 'RB' && p.team){
      var myRBTeams = {};
      myPicksList().forEach(function(mp){
        if ((mp.position || '').toUpperCase() === 'RB' && mp.team) myRBTeams[mp.team] = true;
      });
      if (myRBTeams[p.team]) handcuffBonus = 5;
    }
    // Positional-scarcity urgency scales with how many dedicated STARTERS are
    // still open at this position, not just whether the next one starts. A single
    // remaining slot (TE, or QB in 1QB) produces a real but muted cliff; a
    // multi-slot need (WR/RB you still need several of) keeps the full shelf-cliff
    // urgency. This stops an elite single-slot player from leaping a higher-value
    // pick that fills a deeper roster need on scarcity alone.
    var missDed = (c.obligations && c.obligations.missing && c.obligations.missing[pos]) || 0;
    var waitLossScale = missDed >= 2 ? 1 : (missDed >= 1 ? 0.6 : 0.4);
    return DraftBoardCore.decisionScore({ base: base, utility: util,
      bench: bench, deepBench: role === 'bench2', recentPenalty: recentPenalty, exceptional: exceptional,
      quality: ppgNormOf(p) || 0, required: c.obligations.required,
      freePicks: c.obligations.freePicks,
      waitLoss: Math.max(0, base - expected) * (1 + demandRisk), waitLossScale: waitLossScale,
      waitPenalty: waitPenalty, handcuffBonus: handcuffBonus });
  }
  // How many players remain in this player's (position|tier) bucket.
  function tierRemaining(p){
    var t = tierOf(p); if (t == null) return null;
    return _ptc[(p.position || '').toUpperCase() + '|' + t] || 0;
  }
  function pickReason(p, counts){
    var pos = (p.position || '').toUpperCase();
    var pickNo = (state && state.current) || 1;
    var t = psCtx().targets[pos];
    var need = t ? Math.max(0, t - (counts[pos] || 0)) : 0;
    var adp = adpOf(p);
    var fell = (adp != null) ? Math.round(pickNo - adp) : null;
    var relGap = (adp != null) ? ((pickNo - adp) / Math.max(adp, 1.5)) : null;
    var tier = tierOf(p);
    var left = tierRemaining(p);
    var role = rosterRoleFor(p, counts), _pc = psCtx();
    if ((role === 'bench1' || role === 'bench2') && _pc.obligations.required > 0 && _pc.obligations.freePicks <= 2)
      return 'Backup-only · only ' + _pc.obligations.freePicks + ' discretionary picks';
    if (!state.sf && pos === 'QB' && (counts['QB'] || 0) >= 1)
      return 'QB filled · backup-only value';
    if (pos === 'TE' && role.indexOf('bench') === 0 && scoringCfg().tep <= 0)
      return 'TE filled · backup-only value';
    if (role === 'flex') return 'Fills FLEX · weekly lineup value';
    // Tier cliff: urgent positional scarcity.
    if (isTierCliff(p) && tier != null){
      if (left <= 1) return 'Last ' + pos + ' in Tier ' + tier + '. Grab now';
      return 'Only ' + left + ' ' + pos + 's left in Tier ' + tier;
    }
    // Relative steal: fell significantly relative to their own ADP tier.
    if (relGap != null && relGap >= 1.0) return 'Elite steal: ' + fell + ' picks past ADP';
    if (relGap != null && relGap >= 0.5) return 'Steal: fell ' + fell + ' picks past ADP';
    // Roster need after early picks.
    if (need > 0 && state.current > 4){
      if (tier != null && tier <= 2) return 'Tier ' + tier + ' ' + pos + ' fills a need';
      return 'Fills ' + pos + ' need (' + need + ' more to target)';
    }
    if (fell != null && fell >= 3) return 'Good value: ' + fell + ' past ADP';
    if (tier != null && tier <= 2) return 'Elite tier (T' + tier + ') talent';
    return 'Best available';
  }

  function tierBadge(p){
    var t = tierOf(p); if (t == null) return '';
    var cliff = isTierCliff(p) ? ' dr-tier-cliff' : '';
    return '<span class="dr-tier' + cliff + '">T' + t + '</span>';
  }

  // ── Queue / targets ─────────────────────────────────────────────────────────
  function isQueued(id){ return !!(state.queue && state.queue.indexOf(String(id)) >= 0); }
  function toggleQueue(id){
    id = String(id);
    if (!state.queue) state.queue = [];
    var i = state.queue.indexOf(id);
    if (i >= 0) state.queue.splice(i, 1); else state.queue.push(id);
    save(); renderSide();
  }

  // opts: { reason, sub, wait, availAt: {pn, prob} }
  function playerRowHtml(p, opts){
    opts = opts || {};
    var adp = adpOf(p);
    // Sidebar Pick Score is POOL-RELATIVE: psRelLive anchors the best pick still
    // available near the top so a strong pick on a depleted board still reads well
    // instead of being buried just because the board is picked over. Ranking stays
    // owned by the live Decision Score (the REC chip), so ordering is unchanged.
    var ps = psRelLive(p);
    var sub = (adp != null ? 'ADP ' + Number(adp).toFixed(1) : '')
      + (!opts.showPickScore && p._ds != null && ps != null ? ' · PS ' + ps : '');
    var reasonLine = '';
    if (opts.reason || (opts.rank && p._ds != null)) {
      reasonLine = '<div class="dr-ba-reason">'
        + (opts.reason ? '<span>' + esc(opts.reason) + '</span>' : '') + '</div>';
    }
    var waitLine = opts.wait
      ? '<div class="dr-ba-wait">Can wait: ' + opts.wait.prob + '% there at #' + opts.wait.pn + '</div>'
      : '';
    // Recommendation is an ordering, not a historical grade. Showing its raw
    // internal utility as 99 early and 18 late made the same sound decision look
    // wildly inconsistent. Surface the rank for Recommendation and reserve the
    // numeric 0-100 chip for the actual Pick Score.
    var _isRec = opts.rank && p._ds != null;
    var psChip = _isRec
      ? '<div class="dr-ba-pschip dr-ba-recchip">#' + opts.rank + '<small>REC</small></div>'
      : (ps != null ? '<div class="dr-ba-pschip" style="color:' + psColor(ps) + ';background:' + psColor(ps) + '1a;">' + ps + '<small>PS</small></div>' : '');
    var availClass = '';
    var availLine = '';
    if (opts.availAt){
      var ap = opts.availAt.prob;
      var ac = availColor(ap);
      availClass = ap >= 65 ? ' dr-avail-hi' : (ap >= 40 ? ' dr-avail-md' : ' dr-avail-lo');
      availLine = '<div class="dr-ba-avail" style="color:' + ac + '">'
        + (ap >= 65 ? '&#10003; ' : '&#8226; ') + ap + '% at #' + opts.availAt.pn + '</div>';
    }
    // Bye week conflict flag (redraft only)
    var byeFlag = '';
    var bc = byeConflict(p);
    if (bc >= 2) byeFlag = '<span class="dr-bye-flag">Bye ' + p.bye_week + ' clash</span>';
    // Projected PPG (Sleeper upcoming-season only). Last-season actual is a
    // separate stat, never a projection stand-in.
    var ppgNum = p.proj_ppg != null ? Number(p.proj_ppg) : null;
    var ppgPart = ppgNum != null ? ' · ' + ppgNum.toFixed(1) + ' proj' : '';
    // Compare button state
    var onCmp = compareIds.indexOf(String(p.id)) >= 0;
    var _isDef = String(p.position || '').toUpperCase() === 'DEF';
    return '<div class="dr-ba-row' + availClass + '" data-id="' + esc(String(p.id)) + '">'
      + '<img class="dr-ba-hs" src="' + playerImgUrl(p) + '" alt=""'
      + (_isDef ? ' data-team="' + esc(p.team || '') + '" onerror="_defImgErr(this)"' : ' onerror="this.style.visibility=\'hidden\'"')
      + '>'
      + '<div class="dr-ba-body"><div class="dr-ba-name">' + esc(p.name) + '</div>'
      + '<div class="dr-ba-meta"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span>' + esc(p.team || '') + tierBadge(p) + ppgPart + byeFlag + '</div>'
      + reasonLine + waitLine + availLine + '</div>'
      + '<div class="dr-ba-right-col">'
      + '<div class="dr-ba-metrics">'
      + '<div class="dr-ba-right"><div class="dr-ba-val">' + Math.round(valOf(p)) + '</div><div class="dr-ba-sub">' + sub + '</div></div>'
      + psChip
      + '</div>'
      + '<div class="dr-ba-actions">'
      + '<button class="dr-cmp-btn' + (onCmp ? ' on' : '') + '" data-cmp="' + esc(String(p.id)) + '" title="Compare">vs</button>'
      + '<button class="dr-star' + (isQueued(p.id) ? ' on' : '') + '" data-star="' + esc(String(p.id)) + '" title="Queue" aria-label="Queue">' + (isQueued(p.id) ? '★' : '☆') + '</button>'
      + (state && state.mode !== 'live' && (isYourTurn() || !sim) && (!sim || simStarted) ? '<button class="dr-ba-draft" data-draft="' + esc(String(p.id)) + '" title="Draft now">Draft</button>' : '')
      + '</div>'
      + '</div>'
      + '</div>';
  }

  function renderQueue(){
    var q = (state.queue || []).map(function(id){ return playersById[String(id)]; })
      .filter(function(p){ return p && !drafted[String(p.id)]; });
    if (!q.length){ listInto(emptyNote('Queue is empty', 'Tap the ★ on any player to add a target.')); return; }
    // Survival odds on every queued target: the whole point of a queue is
    // deciding who can wait until your next pick, so show the number.
    var nextPick = nextOwnedAfterCurrent();
    var html = alertBanners();
    q.forEach(function(p){
      var opts = {};
      if (nextPick){
        var wp = availProb(p, nextPick);
        if (wp != null){
          if (wp >= 55) opts.wait = { pn: nextPick, prob: wp };
          else opts.availAt = { pn: nextPick, prob: wp };
        }
      }
      html += playerRowHtml(p, opts);
    });
    listInto(html);
  }

  function renderSide(){
    // Tier counts reflect who's still available (drives cliffs); VOR/PPG scales
    // use the total pool so the baseline stays fixed. Invalidate the pickScore
    // context so it rebuilds against fresh repl/ppg.
    _ptc = posTierCounts(availablePool());
    _repl = computeReplacement(players);
    _ppgScale = computePpgScale(players);
    psCtxInvalidate();
    draftModelInvalidate();   // re-learn reach/slide/run tendencies from the latest board
    refreshPsPool();          // anchor the pool-relative Pick Score display scale
    var kdef = wantsKDef();
    var kbtns = document.querySelectorAll('.dr-pos-kdef');
    for (var i = 0; i < kbtns.length; i++){ kbtns[i].style.display = kdef ? '' : 'none'; }
    var bc = document.getElementById('drBestControls');
    if (bc) bc.style.display = (sideTab === 'best') ? '' : 'none';
    if (sideTab === 'queue')  return renderQueue();
    if (sideTab === 'needs')  return renderNeeds();
    if (sideTab === 'league') return renderLeague();
    return renderBA();
  }

  function showCompleteSidebar(){
    var side = document.getElementById('drSide');
    side.style.display = '';
    // Show only the Team tab for a finished draft - players/recs/queue are irrelevant.
    var tabs = document.querySelectorAll('#drSideTabs .otc-main-tab');
    tabs.forEach(function(b){
      var stab = b.getAttribute('data-stab');
      if (stab === 'needs'){ b.classList.add('is-active'); b.style.display = ''; }
      else if (stab === 'league'){ b.classList.remove('is-active'); b.style.display = ''; }
      else { b.classList.remove('is-active'); b.style.display = 'none'; }
    });
    sideTab = 'needs';
    document.getElementById('drCompleteBar').style.display = '';
    renderSide();
    // Prefetch standings-engine odds so League / Summary / Deep Dive paint the
    // final number on first open (no interim JS estimate, no visible jump).
    try { refreshServerPlayoffOdds(gradeAllTeams()); } catch (e){ /* grades may not be ready yet */ }
  }

  // Undo showCompleteSidebar(): a fresh mock/manual draft started after viewing a
  // finished draft must get the full tab set back (Players/Queue were hidden and
  // the Team tab pinned), the complete bar hidden, and the default tab restored.
  function resetSideTabs(){
    var tabs = document.querySelectorAll('#drSideTabs .otc-main-tab');
    for (var i = 0; i < tabs.length; i++){
      tabs[i].style.display = '';
      tabs[i].classList.toggle('is-active', tabs[i].getAttribute('data-stab') === 'best');
    }
    sideTab = 'best';
    var cbar = document.getElementById('drCompleteBar');
    if (cbar) cbar.style.display = 'none';
    var side = document.getElementById('drSide');
    if (side) side.style.display = '';
  }

  // Positional-run alert banner (folded into Recs): fires when 3+ of the last 5
  // picks share a position. Returns '' when there's no active run.
  function runBanner(){
    var last5 = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 }, n = 0;
    for (var pn = state.current - 1; pn >= 1 && n < 5; pn--){
      var p = state.picks[pn]; if (!p) continue;
      var pos = (p.position||'').toUpperCase(); if (last5[pos] != null) last5[pos]++; n++;
    }
    var hot = '';
    ['RB','WR','QB','TE'].forEach(function(pos){ if (!hot && last5[pos] >= 3) hot = pos; });
    if (!hot) return '';
    return '<div class="dr-run-banner"><i class="fa-solid fa-fire"></i> <b>' + hot + ' run</b>: ' + last5[hot]
      + ' of the last 5 picks. Weigh your ' + hot + ' need before the tier dries up.</div>';
  }

  // Tier-cliff banner: a position whose top-tier (T1-2) shelf is about to
  // empty. Suppressed in round 1 so naturally-thin positions (TE) don't cry
  // wolf before the draft has shape; after that, 1-2 left is a real cliff.
  function cliffBanner(){
    if (state.type === 'redraft') return '';
    if ((state.current || 1) <= (state.teams || 12)) return '';
    var out = '';
    ['QB','RB','WR','TE'].forEach(function(pos){
      var n = posTopRemaining(pos);
      if (n === 1){
        out += '<div class="dr-run-banner dr-cliff-banner"><i class="fa-solid fa-triangle-exclamation"></i> '
          + '<b>Last T1-2 ' + pos + '</b> on the board.</div>';
      } else if (n === 2){
        out += '<div class="dr-run-banner dr-cliff-banner"><i class="fa-solid fa-triangle-exclamation"></i> '
          + '<b>Only 2 T1-2 ' + pos + 's</b> left.</div>';
      }
    });
    return out;
  }
  function alertBanners(){ return runBanner() + cliffBanner(); }

  function renderRec(){
    if (!hasOwned()){ listInto(emptyNote('Set your pick slot', 'Choose your draft slot to get personalized recommendations.')); return; }
    var counts = myPosCounts();
    var pool = availablePool().slice();
    if (!pool.length){ listInto(emptyNote('No players available', 'Everyone matching this filter has already been drafted.', _DR_SEARCH_ICON)); return; }
    var maxVal = 0; pool.forEach(function(p){ var v = valOf(p); if (v > maxVal) maxVal = v; });
    pool.forEach(function(p){ p._ps = pickScore(p, maxVal, counts); });
    prepareNextPickValues(pool);
    pool.forEach(function(p){ p._ds = liveDecisionScore(p, counts); });
    pool.sort(function(a, b){ return (b._ds || 0) - (a._ds || 0) || (b._ps || 0) - (a._ps || 0); });
    var html = balanceAlert() + alertBanners();
    // Assistant looks across your whole draft capital: a player you can likely
    // get at a later owned pick is flagged so you can spend this pick elsewhere.
    var nextPick = nextOwnedAfterCurrent();
    for (var i = 0; i < Math.min(pool.length, 50); i++){
      var p = pool[i];
      var opts = { reason: pickReason(p, counts), rank: i + 1 };
      if (nextPick){
        var wp = availProb(p, nextPick);
        if (wp != null){
          // Always surface the odds this player is still there at your next pick.
          // A strong likelihood gets the explicit "can wait" nudge; otherwise just
          // show the survival % (avoids printing the same number twice).
          if (wp >= 55) opts.wait = { pn: nextPick, prob: wp };
          else opts.availAt = { pn: nextPick, prob: wp };
        }
      }
      html += playerRowHtml(p, opts);
    }
    listInto(html);
  }

  // ── My Team (roster slots) ──────────────────────────────────────────────────
  function lineupSlots(){
    var rs = (state && state.roster) || defaultRoster();
    var slots = [];
    ['QB','SF','RB','WR','TE','FLEX','K','DEF'].forEach(function(s){
      var n = rs[s] || 0;
      for (var i = 0; i < n; i++) slots.push(s);
    });
    return slots;
  }
  function slotEligible(slot, pos){
    pos = (pos || '').toUpperCase();
    if (slot === 'FLEX') return pos === 'RB' || pos === 'WR' || pos === 'TE';
    if (slot === 'SF')   return pos === 'QB' || pos === 'RB' || pos === 'WR' || pos === 'TE';
    return slot === pos;
  }
  // Score used to rank players for the optimal starting lineup: projected points
  // (Sleeper projected PPG). When a player has no projection
  // (e.g. some rookies) fall back to dynasty/redraft value scaled into a ppg-like
  // range so they still slot in a sensible order instead of always benching.
  function lineupScore(p){
    if (!p) return -Infinity;
    var ppg = ppgOf(p);
    if (ppg == null){ var full = playersById[String(p.id)]; if (full) ppg = ppgOf(full); }
    if (ppg != null) return Number(ppg);
    var v = (p.val != null) ? Number(p.val) : null;
    if (v == null){ var f2 = playersById[String(p.id)]; if (f2) v = valOf(f2); }
    return (v || 0) / 1000;
  }
  // Build the highest-projected legal starting lineup. Slot eligibility is laminar
  // (a dedicated slot's position ⊂ FLEX ⊂ SF), so filling the most restrictive
  // slots first - each with the best remaining eligible player by lineupScore -
  // yields the optimal total points. This is why a late high-projection QB will
  // claim the SF slot and push the lower scorer it displaces back to the bench.
  // Returns starters in roster-slot order (null p for unfilled slots) + bench.
  function optimalLineup(playerList, slots){
    slots = slots || lineupSlots();
    function posOf(p){ return String((p && (p.position || p.pos)) || '').toUpperCase(); }
    var flex = { SF: 3, FLEX: 2 };  // higher = more flexible, filled later
    var order = slots.map(function(s, i){ return { slot: s, i: i }; });
    order.sort(function(a, b){ return (flex[a.slot] || 1) - (flex[b.slot] || 1) || a.i - b.i; });
    var used = {}, assign = {};
    order.forEach(function(o){
      var best = -1, bestScore = -Infinity;
      for (var j = 0; j < playerList.length; j++){
        if (used[j] || !slotEligible(o.slot, posOf(playerList[j]))) continue;
        var sc = lineupScore(playerList[j]);
        if (sc > bestScore){ bestScore = sc; best = j; }
      }
      if (best >= 0){ used[best] = true; assign[o.i] = playerList[best]; }
    });
    var starters = slots.map(function(s, i){ return { slot: s, p: assign[i] || null }; });
    var bench = [];
    for (var k = 0; k < playerList.length; k++){ if (!used[k]) bench.push(playerList[k]); }
    bench.sort(function(a, b){ return lineupScore(b) - lineupScore(a); });
    var starterIds = {};
    starters.forEach(function(s){ if (s.p) starterIds[String(s.p.id)] = true; });
    return { starters: starters, bench: bench, starterIds: starterIds };
  }
  function slotColor(slot){
    if (slot === 'FLEX') return '#14b8a6';
    if (slot === 'SF')   return '#a78bfa';
    if (slot === 'BN')   return '#64748b';
    return posColor(slot);
  }
  function pickNoStr(p){
    if (!p || !p.id || !state) return '';
    var found = 0;
    Object.keys(state.picks).forEach(function(k){
      if (state.picks[k] && state.picks[k].id === p.id) found = parseInt(k, 10);
    });
    if (!found) return '';
    var rd = Math.ceil(found / state.teams);
    var pk = found - (rd - 1) * state.teams;
    return 'Pick ' + rd + '.' + (pk < 10 ? '0' + pk : String(pk));
  }
  // Short "1.04" form (no "Pick " prefix) for compact contexts like the share card.
  function pickNoShort(p){
    if (!p || !p.id || !state) return '';
    var found = 0;
    Object.keys(state.picks).forEach(function(k){
      if (state.picks[k] && state.picks[k].id === p.id) found = parseInt(k, 10);
    });
    if (!found) return '';
    var rd = Math.ceil(found / state.teams);
    var pk = found - (rd - 1) * state.teams;
    return rd + '.' + (pk < 10 ? '0' + pk : String(pk));
  }
  function slotRow(slot, p){
    if (p){
      var _rsps = relPS(p);
      var psBadge = (_rsps != null) ? '<span class="dr-rslot-ps" style="color:' + psColor(_rsps) + '">' + _rsps + '</span>' : '';
      var pickLbl = pickNoStr(p);
      var _isDefSlot = String(p.position || '').toUpperCase() === 'DEF';
      return '<div class="dr-rslot">'
        + '<span class="dr-rslot-pos" style="background:' + slotColor(slot) + '">' + slot + '</span>'
        + '<img class="dr-rslot-hs" src="' + playerImgUrl(p) + '" alt=""'
        + (_isDefSlot ? ' data-team="' + esc(p.team || '') + '" onerror="_defImgErr(this)"' : ' onerror="this.style.visibility=\'hidden\'"')
        + '>'
        + '<div class="dr-rslot-body"><div class="dr-rslot-name">' + esc(p.name) + '</div>'
        + '<div class="dr-rslot-meta">' + esc(p.position) + ' &middot; ' + esc(p.team || '') + (pickLbl ? ' &middot; <span style="color:var(--accent)">' + pickLbl + '</span>' : '') + '</div></div>'
        + psBadge
        + '<div class="dr-rslot-val">' + (p.val != null ? Math.round(p.val) : '') + '</div>'
        + '</div>';
    }
    return '<div class="dr-rslot dr-rslot-open">'
      + '<span class="dr-rslot-pos" style="background:' + slotColor(slot) + '">' + slot + '</span>'
      + '<span class="dr-rslot-empty">open</span></div>';
  }
  // ── Draft grade / roster strength ───────────────────────────────────────────
  // Projected PPG from Sleeper only (upcoming season). Last-season actuals
  // are a separate stat and are never used as a projection.
  function ppgOf(p){ if (window.DraftBoardCore) return DraftBoardCore.ppgOf(p); return (p && p.proj_ppg != null) ? Number(p.proj_ppg) : null; }

  // ── Projected playoff odds (completed draft only) ───────────────────────────
  // Once every team has a full roster we can project each team's season from its
  // drafted strength. Team strength = projected points of its optimal starting
  // lineup; a light Monte Carlo plays random weekly matchups (score ~ Normal(
  // strength, week-to-week sigma)), ranks by record (points break ties), and
  // counts how often each team lands in a playoff seed. It's a rough projection,
  // not a real season sim (there's no schedule in a draft), so it's labeled as
  // odds and only shown when the draft is over.
  function _gauss(){
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function _teamStrengthPPG(picksList){
    var ol = optimalLineup(picksList);
    var s = 0;
    ol.starters.forEach(function(x){ if (x.p){ var v = lineupScore(x.p); if (isFinite(v) && v > 0) s += v; } });
    return s;
  }
  var _poMcCache = null, _poMcSig = null;
  // allTeams: gradeAllTeams() output; each entry has .slot and .picks[{pn,p}].
  // Returns { slot: oddsPct }.
  function playoffOddsBySlot(allTeams){
    var sig = allTeams.map(function(t){ return t.slot + ':' + (t.picks ? t.picks.length : 0); }).join('|') + '@' + state.current;
    if (_poMcCache && _poMcSig === sig) return _poMcCache;
    var teams = allTeams.map(function(t){
      var picks = (t.picks || []).map(function(x){ return x.p; }).filter(Boolean);
      return { slot: t.slot, S: _teamStrengthPPG(picks) };
    });
    var n = teams.length;
    var odds = {};
    teams.forEach(function(t){ odds[t.slot] = 0; });
    if (n >= 2){
      var spots = n <= 8 ? 4 : 6;
      if (spots >= n) spots = Math.max(1, n - 1);
      var W = 14, N = 2500, sigma = 27;
      for (var s = 0; s < N; s++){
        var wins = [], pts = [];
        for (var t = 0; t < n; t++){ wins[t] = 0; pts[t] = 0; }
        for (var w = 0; w < W; w++){
          var idx = []; for (var q = 0; q < n; q++) idx[q] = q;
          for (var i = idx.length - 1; i > 0; i--){ var j = (Math.random() * (i + 1)) | 0; var tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; }
          for (var k = 0; k + 1 < idx.length; k += 2){
            var a = idx[k], b = idx[k + 1];
            var sa = teams[a].S + _gauss() * sigma;
            var sb = teams[b].S + _gauss() * sigma;
            pts[a] += sa; pts[b] += sb;
            if (sa >= sb) wins[a]++; else wins[b]++;
          }
        }
        var ord = []; for (var o = 0; o < n; o++) ord[o] = o;
        ord.sort(function(x, y){ return (wins[y] - wins[x]) || (pts[y] - pts[x]); });
        for (var r = 0; r < spots; r++){ odds[teams[ord[r]].slot] += 1; }
      }
      teams.forEach(function(t){ odds[t.slot] = Math.round(odds[t.slot] / N * 100); });
    }
    _poMcCache = odds; _poMcSig = sig;
    return odds;
  }
  function _poColor(po){ return po >= 60 ? '#22c55e' : po >= 35 ? '#f59e0b' : '#ef4444'; }
  function _draftComplete(){ return !!state && (!!state.isComplete || state.current > (state.teams || 12) * (state.rounds || 0)); }

  // Server-computed playoff odds - the SAME engine the standings page uses
  // (simulate_playoff_odds, preseason mode). For a completed draft we ONLY show
  // these numbers (or a loading placeholder) — never the JS Monte Carlo first —
  // so the user never sees a percentage jump a moment later. The JS estimate is
  // used mid-draft and as a one-shot fallback if the server fetch fails.
  var _poServer = null, _poServerSig = null, _poFetching = false;
  var _poFailedSig = null;
  function _poSig(allTeams){ return allTeams.map(function(t){ return t.slot + ':' + (t.picks ? t.picks.length : 0); }).join('|') + '@' + state.current; }
  function _repaintPlayoffOdds(){
    if (sideTab === 'league') renderSide();
    // Summary / Deep Dive capture odds at open; rebuild once when the final
    // source lands so the first painted number is the only number.
    var sum = document.getElementById('drSummary');
    if (sum && sum.style.display !== 'none' && typeof openSummary === 'function'){
      try { openSummary(); } catch (e){ /* openSummary may not be ready */ }
    }
    var ddOv = document.getElementById('drDeepDive');
    if (ddOv && ddOv.style.display !== 'none') openDeepDive();
  }
  function refreshServerPlayoffOdds(allTeams){
    if (!_draftComplete() || !allTeams || allTeams.length < 2) return;
    var sig = _poSig(allTeams);
    if (_poFetching || (_poServer && _poServerSig === sig)) return;
    if (_poFailedSig === sig) return;
    _poFetching = true;
    var _sc = scoringCfg();
    var payload = {
      season: state.season || 0,
      ppr: (_sc && _sc.ppr != null) ? _sc.ppr : 1,
      roster: (state && state.roster) || defaultRoster(),
      playoff_teams: (state.teams && state.teams <= 8) ? 4 : 6,
      teams: allTeams.map(function(t){
        return { slot: t.slot, name: t.name,
          players: (t.picks || []).map(function(x){ return (x.p && x.p.id != null) ? String(x.p.id) : null; }).filter(Boolean) };
      })
    };
    fetch('/api/draft-playoff-odds', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        _poFetching = false;
        if (resp && resp.odds && resp.odds.length){
          var m = {};
          resp.odds.forEach(function(o){ if (o.slot != null) m[o.slot] = Math.round(o.playoff_pct); });
          _poServer = m; _poServerSig = sig; _poFailedSig = null;
          _repaintPlayoffOdds();
        } else {
          _poFailedSig = sig;
          _repaintPlayoffOdds();
        }
      })
      .catch(function(){ _poFetching = false; _poFailedSig = sig; _repaintPlayoffOdds(); });
  }
  // Odds for display. Completed drafts wait for the standings engine (empty
  // object = pending). Mid-draft and failed fetches use the JS estimate.
  function playoffOddsSource(allTeams){
    refreshServerPlayoffOdds(allTeams);
    var sig = _poSig(allTeams);
    if (_poServer && _poServerSig === sig) return _poServer;
    if (!_draftComplete()) return playoffOddsBySlot(allTeams);
    if (_poFailedSig === sig) return playoffOddsBySlot(allTeams);
    return {};
  }
  function playoffOddsPending(allTeams){
    if (!_draftComplete() || !allTeams || allTeams.length < 2) return false;
    var sig = _poSig(allTeams);
    return !(_poServer && _poServerSig === sig) && _poFailedSig !== sig;
  }
  function gradeTeam(){
    if (!hasOwned()) return null;
    // Pull "your" grade from the full field so the Team / League / Deep Dive
    // surfaces share one gradeAllTeams() pass (absolute composite — no field curve).
    var field = gradeAllTeams();
    for (var i = 0; i < field.length; i++){ if (field[i].isMe) return field[i].grade; }
    var mine = [];
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (isMyPick(pn)) mine.push({ pn: pn, p: state.picks[k] });
    });
    if (!mine.length) return null;
    mine.sort(function(a, b){ return a.pn - b.pn; }); // process in pick order for need context
    return gradePicks(mine);
  }
  // Competitive window from a starter set: young core => Future (ascending),
  // veteran core => Win-Now (compete before decline), in between => Balanced.
  // Value-weighted so your best players define the window more than depth. Only
  // meaningful for dynasty/startup (redraft is always "now").
  function _competitiveWindow(starterArr){
    if (state.type === 'redraft') return null;
    var wSum = 0, aSum = 0;
    starterArr.forEach(function(x){
      var full = playersById[String(x.id)];
      var age = (full && full.age != null) ? Number(full.age) : null;
      if (age == null) return;
      var w = Math.max(1, x.val || 1);
      aSum += age * w; wSum += w;
    });
    if (wSum <= 0) return null;
    var avgAge = aSum / wSum;
    var label = avgAge <= 24.5 ? 'Future' : (avgAge >= 26.5 ? 'Win-Now' : 'Balanced');
    return { label: label, avgAge: avgAge };
  }
  // Grade a single rookie pick with the BPA/ADP-diff letter system.
  // Mirrors pick_grade() in app.py used by the Teams-tab draft grader.
  function rookiePickGrade(adpDiff, need, isBpa, bpaGap, pos, qbCount, numTeams){
    if (adpDiff == null) return 'N/A';
    var bigReach = -(numTeams * 1.1);
    var score;
    if      (adpDiff >= 4)        score = 4;
    else if (adpDiff >= 2)        score = 3;
    else if (adpDiff >= -3)       score = 2;
    else if (adpDiff >= bigReach) score = 1;
    else                          score = 0;
    if (isBpa) {
      score += (adpDiff < -3) ? 1 : 2;
    } else if (bpaGap != null && bpaGap >= 5) {
      score = Math.max(score - 1, 0);
    }
    if (need) {
      score += 1;
    } else if (pos === 'QB' && !state.sf && qbCount >= 2) {
      score = Math.max(score - 2, 0);
    } else if (pos === 'QB' && !state.sf && qbCount >= 1) {
      score = Math.max(score - 1, 0);
    }
    if (adpDiff >= -3) score = Math.max(score, 1);
    if (need && adpDiff >= -4) score = Math.max(score, 2);
    var map = {5:'A+', 4:'A', 3:'B', 2:'C', 1:'D', 0:'F'};
    return map[Math.min(score, 5)] || 'F';
  }
  // Average letter grades into a team letter. Mirrors team_grade() in app.py.
  function teamLetterFromPicks(letters){
    if (!letters.length) return 'N/A';
    var gv = {'A+':5,'A':4,'B':3,'C':2,'D':1,'F':0,'N/A':2};
    var avg = letters.reduce(function(s,g){ return s + (gv[g] != null ? gv[g] : 2); }, 0) / letters.length;
    if (avg >= 4.5) return 'A+';
    if (avg >= 3.5) return 'A';
    if (avg >= 2.5) return 'B';
    if (avg >= 1.5) return 'C';
    if (avg >= 0.5) return 'D';
    return 'F';
  }
  // Map team letter to a canonical 0-100 score for gradeLetter() and sorting.
  // Chosen so gradeLetter(letterToScore(L)) === L for each letter band.
  function letterToScore(letter){
    return {'A+':92,'A':87,'B':70,'C':55,'D':43,'F':20,'N/A':55}[letter] || 55;
  }

  // Grade any team's picks. `mine` = sorted [{pn, p}] for one team.
  function gradePicks(mine){
    if (!mine || !mine.length) return null;
    var counts = { QB:0, RB:0, WR:0, TE:0 };
    // Pre-compute maxVal for pickScore (matches what pickScore callers do)
    var _gmaxVal = 0; players.forEach(function(q){ var v = valOf(q); if (v > _gmaxVal) _gmaxVal = v; });
    // Progressive need context for THIS team only. Do not fall back to psCtx()
    // (viewer roster) — that leaked the viewer's quality counts into every other
    // team's grade and skewed the league board / Deep Dive ranks.
    var countsSoFar = { QB: 0, RB: 0, WR: 0, TE: 0 };
    var qualSoFar = { QB: 0, RB: 0, WR: 0, TE: 0 };
    var picks = []; // { id, pos, ps, val, ppg }
    mine.forEach(function(m){
      var pos = (m.p.position || '').toUpperCase();
      // Grade score: recompute at the pick number so it measures pick quality
      // and matches the server's compute_pick_score exactly. The kernel carries
      // no timing terms, so this equals the board's Pick Score for the same
      // inputs. Falls back to the stored score only when the player is no longer
      // resolvable.
      var full = playersById[String(m.p.id)];
      var ps = null;
      if (players.length > 0 && _gmaxVal > 0 && full){
        ps = pickScore(full, _gmaxVal, countsSoFar, {
          grading: true, pickNo: m.pn, qualByPos: qualSoFar
        });
      }
      if (ps == null) ps = m.p.ps;
      if (countsSoFar[pos] != null) countsSoFar[pos]++;
      if (counts[pos] != null) counts[pos]++;
      if (qualSoFar[pos] != null){
        var _qv = full ? vorOf(full) : null;
        if (_qv == null || _qv > 0) qualSoFar[pos]++;
      }
      picks.push({ id: m.p.id, pos: pos, ps: ps, pn: m.pn,
        val: full ? valOf(full) : (m.p.val || 0), ppg: full ? ppgOf(full) : null });
    });
    var psVals = picks.map(function(x){ return x.ps; }).filter(function(v){ return v != null; });
    var avgPs = psVals.length ? psVals.reduce(function(a, b){ return a + b; }, 0) / psVals.length : null;

    if (state.type === 'rookie'){
      // Overall letter grade uses the BPA/ADP-diff system (same as the Teams-tab draft
      // grade). Per-pick 0-100 pick-score chips are unchanged.
      var _numTeams = state.teams || 12;
      // Reconstruct which players were taken before each pick number so we can compute
      // BPA (best player available) for each of this team's picks.
      var _allPicks = [];
      Object.keys(state.picks).forEach(function(k){
        var pn = parseInt(k, 10);
        if (state.picks[k]) _allPicks.push({ pn: pn, id: String(state.picks[k].id) });
      });
      _allPicks.sort(function(a, b){ return a.pn - b.pn; });
      var _takenBefore = {}, _cumTaken = {};
      _allPicks.forEach(function(ap){
        _takenBefore[ap.pn] = Object.assign({}, _cumTaken);
        _cumTaken[ap.id] = true;
      });
      var _rCounts = { QB:0, RB:0, WR:0, TE:0 };
      var _letters = [];
      picks.forEach(function(x){
        var full = playersById[String(x.id)];
        var myAdp = full ? adpOf(full) : null;
        var adpDiff = (myAdp != null) ? (x.pn - myAdp) : null;
        var pos = x.pos;
        var need = (posTargets()[pos] || 0) > (_rCounts[pos] || 0);
        var qbCount = _rCounts['QB'] || 0;
        // BPA: find players with a better ADP still on the board at this pick.
        var _taken = _takenBefore[x.pn] || {};
        var _betterAdps = [];
        if (myAdp != null){
          players.forEach(function(q){
            if (_taken[String(q.id)]) return;
            var qa = adpOf(q);
            if (qa != null && qa < myAdp) _betterAdps.push(qa);
          });
          _betterAdps.sort(function(a, b){ return a - b; });
        }
        var isBpa = _betterAdps.length === 0;
        var bpaGap = (_betterAdps.length > 0 && myAdp != null) ? (myAdp - _betterAdps[0]) : null;
        var letter = rookiePickGrade(adpDiff, need, isBpa, bpaGap, pos, qbCount, _numTeams);
        if (letter !== 'N/A') _letters.push(letter);
        if (_rCounts[pos] != null) _rCounts[pos]++;
      });
      // Smooth 0-100: MEAN of each pick's canonical letter score, not the coarse
      // team-letter bucket (mirrors utils.draft_grade.dr_rookie_team_score; keep
      // the two in lock-step). An [A, B] class -> 78.5 (B+), not a rounded-up A.
      // All-N/A (no ADP) is ungradeable — Python returns None, not N/A→55→C.
      var _rk = _letters.filter(function(L){ return L && L !== 'N/A'; })
                        .map(function(L){ return letterToScore(L); });
      if (!_rk.length) return null;
      var rv = _rk.reduce(function(a, b){ return a + b; }, 0) / _rk.length;
      return { score: rv, value: avgPs != null ? Math.round(avgPs) : 50,
        balance: 0, tier: 0, count: mine.length,
        avgPs: avgPs ? Math.round(avgPs) : null, window: null,
        provisional: gradeIsProvisional(mine.length) };
    }

    // Startup / redraft composite lives in static/draft_grade_team.js
    // (BRTeamGrade), shared with the Python server grade and pinned by a parity
    // test. Startup is Value 35 / Starters 25 / Construction 40; redraft is
    // 20/50/30 so lineup strength (what playoff odds rank) leads. K/DEF are
    // excluded from grading.
    var _slots = lineupSlots().filter(function(s){ return s !== 'K' && s !== 'DEF'; });
    var _leaguePpg = [];
    players.forEach(function(q){ var v = ppgOf(q); if (v != null) _leaguePpg.push(v); });
    var _leagueVal = players.map(function(q){ return valOf(q); });
    var _leaguePlayers = players.map(function(q){
      return { pos: String(q.position || '').toUpperCase(), ppg: ppgOf(q), val: valOf(q) };
    });
    var _tg = window.BRTeamGrade;
    if (!_tg || typeof _tg.teamGradeComposite !== 'function') return null;
    var _comp = _tg.teamGradeComposite(
      picks, _slots, posTargets(), state.teams || 12, state.type,
      _leaguePpg, _leagueVal, _leaguePlayers
    );
    // Missing composite is ungradeable — do not invent score 0 (letter F).
    if (!_comp) return null;
    var _starterArr = picks.filter(function(x){ return _comp.starterIds[String(x.id)]; });
    return { score: _comp.total, value: _comp.value, balance: _comp.balance, tier: _comp.starter,
      count: mine.length, avgPs: avgPs != null ? Math.round(avgPs) : null,
      strength: Math.round(_comp.strengthRatio * 100),
      window: _competitiveWindow(_starterArr),
      provisional: gradeIsProvisional(mine.length) };
  }
  // At-pick tier-cliff map for grading: walk the full draft in order starting from
  // the full-pool tier counts, so each historical pick sees remaining-at-that-slot
  // (matches live isTierCliff / the Teams API), not post-draft leftovers.
  var _gradeCliffByPn = null;
  function _buildGradeCliffs(){
    var counts = {}, map = {};
    players.forEach(function(p){
      var t = tierOf(p); if (t == null) return;
      var k = String(p.position || '').toUpperCase() + '|' + t;
      counts[k] = (counts[k] || 0) + 1;
    });
    var teams = (state && state.teams) || 12;
    Object.keys(state.picks || {}).filter(function(k){ return !!state.picks[k]; })
      .map(function(k){ return parseInt(k, 10); }).sort(function(a, b){ return a - b; })
      .forEach(function(pn){
        var pl = state.picks[pn];
        var full = playersById[String(pl.id)] || pl;
        var t = tierOf(full);
        var pos = String(pl.position || '').toUpperCase();
        var k = pos + '|' + t;
        var left = (t != null) ? (counts[k] || 0) : 0;
        map[pn] = (pn > teams) && t != null && left <= 2;
        if (t != null && counts[k]) counts[k]--;
      });
    return map;
  }
  // Grade every team in the draft, sorted best-first. Picks are attributed by
  // OWNERSHIP, not by the board column they sit in: every pick the user owns (a
  // 1.01 traded in, a pick in another seat, etc.) belongs to the single "You"
  // team, and each seat shows only the picks it still owns. Otherwise a user who
  // owns picks across two seats would show up as "You" twice.
  function gradeAllTeams(){
    if (!state) return [];
    _gradeCliffByPn = _buildGradeCliffs();
    var teams = state.teams || 12;
    var mine = [];        // every pick the user owns, regardless of seat
    var bySlot = {};      // remaining picks grouped by the seat that owns them
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (!state.picks[k]) return;
      var entry = { pn: pn, p: state.picks[k] };
      if (isMyPick(pn)){ mine.push(entry); return; }
      // Credit the pick to the team that actually owns it (traded picks map to
      // their new owner), falling back to the board seat when no map is available.
      var slot = (state.pickOwners && state.pickOwners[pn] != null) ? state.pickOwners[pn] : slotOnClock(pn, teams, state.order);
      if (!bySlot[slot]) bySlot[slot] = [];
      bySlot[slot].push(entry);
    });
    var out = [];
    // The user's team: all owned picks consolidated into one entry. Slot 0 is a
    // sentinel that never collides with the real seats (1..teams), so the League
    // tab's per-row detail lookup stays unambiguous.
    if (mine.length){
      mine.sort(function(a, b){ return a.pn - b.pn; });
      var gm = gradePicks(mine);
      if (gm) out.push({ slot: 0, name: 'You', isMe: true, grade: gm, picks: mine });
    }
    for (var s = 1; s <= teams; s++){
      var picks = bySlot[s];
      if (!picks || !picks.length) continue;
      picks.sort(function(a, b){ return a.pn - b.pn; });
      var g = gradePicks(picks);
      if (!g) continue;
      out.push({ slot: s, name: teamName(s), isMe: false, grade: g, picks: picks });
    }
    _applyFieldCurve(out);
    out.sort(function(a, b){ return b.grade.score - a.grade.score; });
    return out;
  }
  // Absolute grading: the team letter reflects the team's OWN composite (starter
  // quality + lineup strength vs league + construction), not its rank within the
  // field - so a genuinely elite draft earns an A even when the whole room drafted
  // well, and a weak one earns a C even in a weak room. Field-curve helpers
  // (draft_grade_curve.js / dr_apply_field_curve) remain for backtests only.
  // rawScore is kept in sync for any consumer that reads it.
  function _applyFieldCurve(out){
    if (!out) return;
    out.forEach(function(t){ if (t && t.grade) t.grade.rawScore = t.grade.score; });
  }
  // Classify a startup/redraft build into a recognizable draft archetype based on
  // positional emphasis, weighting the early picks where strategy is actually set.
  function teamArchetype(){
    if (!state || state.type === 'rookie') return null;
    var mine = [];
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (isMyPick(pn)) mine.push({ pn: pn, p: state.picks[k] });
    });
    if (mine.length < 3) return null;
    mine.sort(function(a, b){ return a.pn - b.pn; });

    var counts = { QB:0, RB:0, WR:0, TE:0 };
    var firstIdx = { QB:-1, RB:-1, WR:-1, TE:-1 };
    mine.forEach(function(m, i){
      var pos = (m.p.position || '').toUpperCase();
      if (counts[pos] != null){ counts[pos]++; if (firstIdx[pos] < 0) firstIdx[pos] = i; }
    });
    // "Early" = first 5 picks (or all, if fewer) - that's where build identity lives.
    var earlyN = Math.min(5, mine.length);
    var early = { QB:0, RB:0, WR:0, TE:0 };
    for (var i = 0; i < earlyN; i++){
      var pos = (mine[i].p.position || '').toUpperCase();
      if (early[pos] != null) early[pos]++;
    }

    var label;
    if (state.sf && early.QB >= 2){
      label = 'Konami Code';
    } else if (firstIdx.TE >= 0 && firstIdx.TE <= 1){
      label = 'TE Premium';
    } else if (early.RB === 0 && early.WR >= 3){
      label = 'Zero RB';
    } else if (early.RB === 1 && firstIdx.RB <= 1 && early.WR >= 2){
      label = 'Hero RB';
    } else if (early.RB >= 3){
      label = 'Robust RB';
    } else if (early.WR >= 4 || (counts.WR - counts.RB >= 3)){
      label = 'WR Factory';
    } else if (early.RB - early.WR >= 2){
      label = 'Ground & Pound';
    } else {
      label = 'Balanced Build';
    }
    return { label: label };
  }
  function gradeLetter(s){
    if (s>=90) return 'A+'; if (s>=85) return 'A'; if (s>=80) return 'A-';
    if (s>=75) return 'B+'; if (s>=70) return 'B'; if (s>=65) return 'B-';
    if (s>=60) return 'C+'; if (s>=55) return 'C'; if (s>=50) return 'C-';
    if (s>=40) return 'D';  return 'F';
  }
  // Construction ramps with min(1, picks/8). Show the real letter from pick 1
  // (including two-pick / start-of-round-3 boards — that F-grade bug is fixed)
  // but mark it Early until the sample is large enough to trust construction.
  function gradeIsProvisional(count){
    if ((state && state.type) === 'rookie') return count < 3;
    return count < 8;
  }
  function gradeEarlySuffix(g){
    return (g && g.provisional) ? ' · Early' : '';
  }
  function gradeBar(label, val, max, tip){
    var pct = max ? Math.round(val / max * 100) : 0;
    var col = pct >= 80 ? '#22c55e' : pct >= 60 ? '#38bdf8' : pct >= 40 ? '#f59e0b' : '#ef4444';
    return '<div class="dr-gbar-row"><span class="dr-gbar-lbl">' + label + (tip ? infoIcon(tip) : '') + '</span>'
      + '<div class="dr-gbar"><div class="dr-gbar-fill" style="width:' + pct + '%;background:' + col + '"></div></div>'
      + '<span class="dr-gbar-pct" style="color:' + col + '">' + pct + '</span>'
      + '</div>';
  }
  // Per-component max points. Rookie grade is value-only (avg pick score);
  // startup/redraft weights pick value, starting-lineup strength, and construction.
  // Caps must match BRTeamGrade / utils.draft_grade splits.
  function gradeMax(){
    if (state.type === 'rookie') return { value:100, balance:0, tier:0 };
    var split = (state.type === 'redraft')
      ? ((window.BRTeamGrade && window.BRTeamGrade.SPLIT_REDRAFT) || [20, 50, 30])
      : ((window.BRTeamGrade && window.BRTeamGrade.SPLIT_STARTUP) || [35, 25, 40]);
    return { value: split[0], balance: split[2], tier: split[1] };
  }
  function gradeBars(g){
    var m = gradeMax();
    if (state.type === 'rookie') return gradeBar('Avg Pick Score', g.value, 100, 'The bar shows average 0-100 pick score (same chips shown on each pick). The letter grade uses the BPA/ADP system: did you reach, and was a better player available?');
    // g.tier holds the starting-lineup strength component.
    var starterTip = state.type === 'redraft'
      ? 'Projected starting-lineup PPG versus a league-average team — the same strength playoff odds use. This is the largest slice of a redraft grade.'
      : 'How good your projected starting lineup is versus a league-average team.';
    var consTip = state.type === 'redraft'
      ? 'Mostly whether you’ve filled starting slots. Extra bench depth is not a penalty; empty starters are.'
      : 'How well you’ve filled your starting slots and balanced positions.';
    return gradeBar('Value', g.value, m.value, 'How strong your picks are by pick score, weighted toward the earlier rounds.')
      + gradeBar('Starters', g.tier, m.tier, starterTip)
      + gradeBar('Construction', g.balance, m.balance, consTip);
  }

  function renderNeeds(){
    if (!hasOwned()){ listInto(emptyNote('Set your pick slot', 'Choose your draft slot to see your team build.')); return; }
    var mine = myPicksList().slice().sort(function(a, b){ return (b.val || 0) - (a.val || 0); });
    var html = '';
    var g = gradeTeam();
    if (g){
      // The rookie card shows "Avg Pick Score" as a labeled bar below, so don't
      // repeat it as the subtitle - use the team archetype label instead.
      var gSub = '';
      var _ga = teamArchetype(); if (_ga) gSub = _ga.label;
      var _gwn = g.window;
      var gAgeSub = _gwn ? (esc(_gwn.label) + ' \xb7 Avg age ' + _gwn.avgAge.toFixed(1)) : '';
      html += '<div class="dr-grade-card"><div class="dr-grade-mark"><div class="dr-grade-letter">' + gradeLetter(g.score) + '</div>'
        + (g.provisional ? '<div class="dr-grade-early">Early</div>' : '')
        + '</div>'
        + '<div class="dr-grade-meta">' + (gSub ? '<div class="dr-grade-pace">' + gSub + '</div>' : '')
        + (gAgeSub ? '<div class="dr-grade-pace" style="font-size:11px;color:var(--text-muted)">' + gAgeSub + '</div>' : '')
        + gradeBars(g)
        + '</div></div>';
    }
    html += '<div class="dr-roster">';
    // Highest-projected legal lineup (projection-first, value fallback), so the
    // strongest scorer fills each slot - a high-proj QB takes SF over a weaker flex.
    var _olN = optimalLineup(mine, lineupSlots());
    _olN.starters.forEach(function(s){ html += slotRow(s.slot, s.p); });
    var bench = _olN.bench;
    html += '<div class="dr-roster-div">Bench</div>';
    if (bench.length){ bench.forEach(function(p){ html += slotRow('BN', p); }); }
    else { html += slotRow('BN', null); }
    html += '</div>';
    // Roster projection: Sleeper upcoming-season proj_ppg only (including 0).
    function _pPpg(p){ return p.proj_ppg != null ? Number(p.proj_ppg) : null; }
    var projPlayers = mine.filter(function(p){ return _pPpg(p) != null; });
    if (projPlayers.length >= 2){
      var myProjTotal = 0;
      projPlayers.forEach(function(p){ myProjTotal += _pPpg(p); });
      var myAvg = myProjTotal / projPlayers.length;
      // Compare per-player avg to avoid partial-roster vs full-team distortion
      var allProj = [];
      players.forEach(function(p){ var v = _pPpg(p); if (v != null) allProj.push(v); });
      allProj.sort(function(a, b){ return b - a; });
      var numTeams = state.teams || 12, numRds = state.rounds || 15;
      var topSlice = allProj.slice(0, numTeams * numRds);
      var leagueAvgPerPlayer = topSlice.length ? topSlice.reduce(function(a, b){ return a + b; }, 0) / topSlice.length : 0;
      var projPct = leagueAvgPerPlayer > 0 ? Math.round(myAvg / leagueAvgPerPlayer * 100) : 0;
      var projColor = projPct >= 108 ? '#22c55e' : projPct >= 92 ? '#f59e0b' : '#ef4444';
      html += '<div class="dr-proj-card">'
        + '<div class="dr-proj-title">Roster Projection</div>'
        + '<div class="dr-proj-stats">'
        + '<div class="dr-proj-stat"><div class="dr-proj-val">' + myAvg.toFixed(1) + '</div><div class="dr-proj-lbl">My Avg PPG</div></div>'
        + (leagueAvgPerPlayer > 0 ? '<div class="dr-proj-stat"><div class="dr-proj-val">' + leagueAvgPerPlayer.toFixed(1) + '</div><div class="dr-proj-lbl">Avg Player</div></div>' : '')
        + (leagueAvgPerPlayer > 0 ? '<div class="dr-proj-stat"><div class="dr-proj-val" style="color:' + projColor + '">' + projPct + '%</div><div class="dr-proj-lbl">vs League</div></div>' : '')
        + '</div>'
        + (leagueAvgPerPlayer > 0 ? '<div class="dr-proj-bar-wrap"><div class="dr-proj-bar-bg"><div class="dr-proj-bar-fill" style="width:' + Math.min(100, projPct) + '%;background:' + projColor + '"></div></div>'
          + '<div class="dr-proj-bar-lbl">' + projPlayers.length + ' of ' + mine.length + ' picks have projection data</div></div>' : '')
        + '</div>';
    }
    listInto(html);
  }

  // Monochrome inline icons for the recap headers (inherit the header color).
  var _RECAP_IC = {
    gem:    '<svg class="dr-recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M6 3h12l3 6-9 12L3 9z"/><path d="M3 9h18"/><path d="M9 3 7.5 9 12 21l4.5-12L15 3"/></svg>',
    down:   '<svg class="dr-recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 8 9 12 13 9 21 16"/><polyline points="21 11 21 16 16 16"/></svg>',
    bars:   '<svg class="dr-recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M5 20V10M12 20V4M19 20v-7"/></svg>',
    trophy: '<svg class="dr-recap-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M7 6H4.5A1.5 1.5 0 0 0 3 7.5 3.5 3.5 0 0 0 6.5 11M17 6h2.5A1.5 1.5 0 0 1 21 7.5 3.5 3.5 0 0 1 17.5 11M9.5 18h5M8.5 21h7M12 14v4"/></svg>'
  };

  // Coin-style rank medals for the draft-grade board — a client-side port of
  // rank_mark()/_medal_svg() in dashboard_services/rank_medals.py so the board's
  // top 3 match the standings/power medals exactly. Gradient ids are prefixed
  // "dr" to avoid colliding with any server-rendered medals on the page.
  var _DR_METALS = {
    gold:   { face: ['#fff7d6','#f6d375','#d9a531','#a9781f'], rim: ['#ffe89a','#c99a2e','#7d5a12'], edge: '#7d5a12', num: '#8a6410' },
    silver: { face: ['#ffffff','#dde3ea','#a7b2be','#727d89'], rim: ['#eef2f6','#aab4bf','#66707b'], edge: '#66707b', num: '#5c6670' },
    bronze: { face: ['#ffe6cc','#e0a56a','#bd7539','#7f4a22'], rim: ['#f0c39a','#b87a44','#7a4620'], edge: '#7a4620', num: '#7a4620' }
  };
  var _DR_CROWN =
    '<path d="M-7 5 L7 5 L5.4 -3 L1.8 0.3 L0 -5 L-1.8 0.3 L-5.4 -3 Z" fill="url(#drmCrown)" stroke="#8a6410" stroke-width="0.6" stroke-linejoin="round"/>' +
    '<circle cx="-7" cy="-3.4" r="1.4" fill="#f6d375" stroke="#8a6410" stroke-width="0.5"/>' +
    '<circle cx="7" cy="-3.4" r="1.4" fill="#f6d375" stroke="#8a6410" stroke-width="0.5"/>' +
    '<circle cx="0" cy="-6.4" r="1.5" fill="#fff7d6" stroke="#8a6410" stroke-width="0.5"/>';
  function _drMedalSvg(metal, label, size, crown) {
    var m = _DR_METALS[metal], f = m.face, r = m.rim, uid = metal;
    var crownSvg = crown ? '<g transform="translate(32 7)">' + _DR_CROWN + '</g>' : '';
    return '<svg viewBox="0 0 64 66" width="' + size + '" height="' + Math.floor(size * 66 / 64) + '" role="img" ' +
      'aria-label="Rank ' + label + '" style="overflow:visible;flex:none;filter:drop-shadow(0 2px 3px rgba(0,0,0,.28))">' +
      '<defs>' +
      '<radialGradient id="drmFace-' + uid + '" cx="38%" cy="30%" r="78%">' +
      '<stop offset="0%" stop-color="' + f[0] + '"/><stop offset="42%" stop-color="' + f[1] + '"/>' +
      '<stop offset="80%" stop-color="' + f[2] + '"/><stop offset="100%" stop-color="' + f[3] + '"/></radialGradient>' +
      '<linearGradient id="drmRim-' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="' + r[0] + '"/><stop offset="52%" stop-color="' + r[1] + '"/>' +
      '<stop offset="100%" stop-color="' + r[2] + '"/></linearGradient>' +
      '<radialGradient id="drmCrown" cx="40%" cy="30%" r="80%"><stop offset="0%" stop-color="#fff7d6"/>' +
      '<stop offset="60%" stop-color="#f4cf6a"/><stop offset="100%" stop-color="#d59f2e"/></radialGradient>' +
      '</defs>' + crownSvg +
      '<circle cx="32" cy="32" r="22" fill="url(#drmRim-' + uid + ')"/>' +
      '<circle cx="32" cy="32" r="20.5" fill="none" stroke="' + m.edge + '" stroke-width="2.4" stroke-dasharray="1.7 2.2" opacity="0.45"/>' +
      '<circle cx="32" cy="32" r="16.5" fill="url(#drmFace-' + uid + ')" stroke="' + m.edge + '" stroke-width="1"/>' +
      '<circle cx="32" cy="32" r="16.5" fill="none" stroke="#ffffff" stroke-width="1" opacity="0.30"/>' +
      '<path d="M20 24 A16 16 0 0 1 44 22" fill="none" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" opacity="0.45"/>' +
      '<text x="32" y="33" text-anchor="middle" dominant-baseline="central" style="font:800 18px system-ui,sans-serif" fill="#ffffff" opacity="0.5">' + label + '</text>' +
      '<text x="32" y="32" text-anchor="middle" dominant-baseline="central" style="font:800 18px system-ui,sans-serif" fill="' + m.num + '">' + label + '</text>' +
      '</svg>';
  }
  var _DR_RANK_METAL = { 1: 'gold', 2: 'silver', 3: 'bronze' };
  function _drRankMedal(rank, size) {
    var metal = _DR_RANK_METAL[rank];
    if (!metal) return null;
    return _drMedalSvg(metal, String(rank), rank === 1 ? size + 4 : size, rank === 1);
  }

  // Whole-draft recap: the biggest values and reaches vs. ADP across every team.
  // Each pick's gap is its overall pick number minus the player's ADP: a positive
  // gap means he fell (a steal), negative means he was reached for. Falls back to
  // the per-pick grade score when ADP data isn't available. Returns '' until there
  // are enough graded picks to be meaningful.
  function _draftRecapHtml(allTeams){
    var picks = [];
    allTeams.forEach(function(t){
      (t.picks || []).forEach(function(pk){
        if (!pk || !pk.p) return;
        var ps = storedPickScore(pk.pn, pk.p);
        if (ps == null) return;
        var full = playersById[String(pk.p.id)] || pk.p;
        var adp = adpOf(full);
        var gap = (adp != null) ? Math.round(pk.pn - adp) : null;
        picks.push({ name: pk.p.name, pos: (pk.p.position || '').toUpperCase(), team: t.name, pn: pk.pn, ps: ps, gap: gap });
      });
    });
    if (picks.length < 4) return '';
    var withGap = picks.filter(function(p){ return p.gap != null; });
    var useGap = withGap.length >= 4;   // prefer ADP gap; fall back to grade score
    var pool = useGap ? withGap : picks;

    function rx(pn){ var teams = state.teams || 12; var rd = Math.ceil(pn / teams); var pp = pn - (rd - 1) * teams; return rd + '.' + (pp < 10 ? '0' + pp : pp); }
    function valTxt(x){ return (useGap && x.gap != null) ? ((x.gap > 0 ? '+' : '') + x.gap) : String(x.ps); }
    function valCol(x){
      if (useGap && x.gap != null) return x.gap > 0 ? '#22c55e' : (x.gap < 0 ? '#ef4444' : '#94a3b8');
      return psColor(x.ps);
    }
    function rows(list){
      return list.map(function(x){
        return '<div class="dr-recap-row">'
          + '<span class="dr-recap-pos" style="background:' + slotColor(x.pos) + '">' + esc(x.pos || '-') + '</span>'
          + '<span class="dr-recap-main"><span class="dr-recap-name">' + esc(x.name) + '</span>'
          + '<span class="dr-recap-sub">' + esc(x.team) + ' &middot; ' + rx(x.pn) + '</span></span>'
          + '<span class="dr-recap-ps" style="color:' + valCol(x) + '" title="pick vs ADP">' + valTxt(x) + '</span>'
          + '</div>';
      }).join('');
    }
    var steals = pool.slice().sort(function(a, b){ return useGap ? (b.gap - a.gap) : (b.ps - a.ps); }).slice(0, 4);
    var reaches = pool.slice().sort(function(a, b){ return useGap ? (a.gap - b.gap) : (a.ps - b.ps); }).slice(0, 4);

    // ── By the numbers: a few whole-draft superlatives ──────────────────
    var teamScores = {};
    picks.forEach(function(p){ (teamScores[p.team] = teamScores[p.team] || []).push(p.ps); });
    var valueTeam = '-', valueAvg = -1e9;
    Object.keys(teamScores).forEach(function(tm){
      var arr = teamScores[tm];
      var avg = arr.reduce(function(a, b){ return a + b; }, 0) / arr.length;
      if (avg > valueAvg){ valueAvg = avg; valueTeam = tm; }
    });
    var posCount = {};
    picks.forEach(function(p){ if (p.pos) posCount[p.pos] = (posCount[p.pos] || 0) + 1; });
    var topPos = Object.keys(posCount).sort(function(a, b){ return posCount[b] - posCount[a]; })[0] || '-';

    function tile(label, big, sub){
      return '<div class="dr-recap-tile"><div class="dr-recap-tlbl">' + label + '</div>'
        + '<div class="dr-recap-tbig">' + esc(big) + '</div>'
        + '<div class="dr-recap-tsub">' + esc(sub) + '</div></div>';
    }
    var numsHtml = '<div class="dr-recap-nums">'
      + tile('Steal of the draft', steals[0].name, steals[0].team + ' · ' + rx(steals[0].pn))
      + tile('Biggest reach', reaches[0].name, reaches[0].team + ' · ' + rx(reaches[0].pn))
      + tile('Best value drafter', valueTeam, 'Highest average pick grade')
      + tile('Most drafted', topPos + (posCount[topPos] ? ' (' + posCount[topPos] + ')' : ''), picks.length + ' picks total')
      + '</div>';

    return '<div class="dr-recap">'
      + '<div class="dr-recap-sec"><p class="dr-recap-h">' + _RECAP_IC.gem + 'Biggest steals</p>' + rows(steals) + '</div>'
      + '<div class="dr-recap-sec"><p class="dr-recap-h">' + _RECAP_IC.down + 'Biggest reaches</p>' + rows(reaches) + '</div>'
      + '</div>'
      + '<p class="dr-recap-h dr-recap-nums-h">' + _RECAP_IC.bars + 'By the numbers</p>' + numsHtml;
  }

  function renderLeague(){
    var allTeams = gradeAllTeams();
    if (!allTeams.length){
      listInto(emptyNote('No picks yet', 'Grades will appear as teams draft.'));
      return;
    }
    if (allTeams.length < 2){
      listInto(emptyNote('Waiting on more teams', 'Grades appear once at least 2 teams have drafted.'));
      return;
    }
    var _rc = ['gold','silver','bronze'];
    // Projected playoff odds per team - only once the draft is complete.
    var _leagueDone = _draftComplete();
    var _poOdds = _leagueDone ? playoffOddsSource(allTeams) : {};
    var html = '<div class="dr-league-body">' + _draftRecapHtml(allTeams)
      + '<p class="dr-recap-h dr-recap-grades-h">' + _RECAP_IC.trophy + 'Draft grades</p><div class="dr-sum-league">';
    allTeams.forEach(function(t, i){
      var w = t.grade.window;
      var winTag = w ? '<span class="dr-sum-lwin dr-win-' + w.label.toLowerCase().replace('-','') + '">' + esc(w.label) + '</span>' : '';
      var tCol = t.grade.score >= 75 ? '#22c55e' : t.grade.score >= 60 ? '#38bdf8' : t.grade.score >= 45 ? '#f59e0b' : '#ef4444';
      var rCls = i < 3 ? (' ' + _rc[i]) : '';
      var _medal = _drRankMedal(i + 1, 22);
      var rankCell = _medal
        ? '<span class="dr-sum-lrank has-medal rank-mark">' + _medal + '</span>'
        : '<span class="dr-sum-lrank' + rCls + '">' + (i + 1) + '</span>';
      // CPU plans are hole cards: hidden while the draft is live (reading the
      // room from picks is the skill a mock trains) and revealed once it ends,
      // so you can check your inferences. The age lean is never shown: the
      // window chip already reflects age posture from the actual roster.
      var stratTag = '';
      var _draftDone = state.current > (state.teams || 12) * (state.rounds || 0);
      if (sim && _draftDone && !t.isMe && state.simStrats){
        var _sl = stratLabel(state.simStrats[t.slot]);
        if (_sl) stratTag = '<span class="dr-strat-tag">' + _sl + '</span>';
      }
      var poTag = '';
      if (_leagueDone){
        if (playoffOddsPending(allTeams)){
          poTag = '<span class="dr-sum-lpo dr-sum-lpo-pending" title="Calculating playoff odds">…</span>';
        } else if (_poOdds[t.slot] != null){
          poTag = '<span class="dr-sum-lpo" style="color:' + _poColor(_poOdds[t.slot]) + '" title="Projected playoff odds">'
            + _poOdds[t.slot] + '%</span>';
        }
      }
      html += '<div class="dr-sum-lrow' + (t.isMe ? ' is-me' : '') + '" data-legslot="' + t.slot + '">'
        + rankCell
        + '<span class="dr-sum-lname">' + esc(t.name) + stratTag + '</span>'
        + winTag
        + poTag
        + '<span class="dr-sum-lgrade" style="color:' + tCol + '">' + gradeLetter(t.grade.score)
        + (t.grade.provisional ? '<span class="dr-grade-early-inline"> Early</span>' : '') + '</span>'
        + '<span class="dr-sum-lchev">&#9660;</span>'
        + '</div>'
        + '<div class="dr-sum-ldtl" id="drLegLdtl' + t.slot + '"></div>';
    });
    html += '</div></div>';
    listInto(html);
    document.querySelectorAll('#drBaList [data-legslot]').forEach(function(row){
      row.addEventListener('click', function(){
        var slot = parseInt(row.getAttribute('data-legslot'), 10);
        var dtl = document.getElementById('drLegLdtl' + slot);
        if (!dtl) return;
        var isOpen = row.classList.toggle('is-open');
        dtl.classList.toggle('is-open', isOpen);
        if (isOpen && !dtl.innerHTML){
          var team = allTeams.filter(function(t){ return t.slot === slot; })[0];
          if (!team || !team.picks) return;
          var _sp = team.picks.slice().map(function(x){ return x.p; }).filter(Boolean);
          var _tst = optimalLineup(_sp).starters.filter(function(x){ return x.p; });
          // Starters fill the lineup slots; everything else is bench. Show the
          // whole board, not just starters, so a team's full set of picks is visible.
          var _starterIds = {};
          _tst.forEach(function(x){ if (x.p && x.p.id != null) _starterIds[x.p.id] = true; });
          var _bench = team.picks.slice()
            .filter(function(pk){ return pk.p && !_starterIds[pk.p.id]; })
            .sort(function(a, b){ return (a.pn || 0) - (b.pn || 0); });
          function _pickRx(pn){
            if (!pn) return '';
            var rd = Math.ceil(pn / state.teams); var pp = pn - (rd - 1) * state.teams;
            return rd + '.' + (pp < 10 ? '0' + pp : pp);
          }
          function _ldtlRow(slotLabel, p, pn){
            var pickRx = _pickRx(pn);
            var _ps = relPS(p, pn);
            var psRx = _ps != null ? '<span class="dr-sum-ldtl-ps" style="color:' + psColor(_ps) + '">' + _ps + '</span>' : '';
            return '<div class="dr-sum-ldtl-row">'
              + '<span class="dr-sum-ldtl-slot" style="background:' + slotColor(slotLabel) + '">' + esc(slotLabel) + '</span>'
              + '<span class="dr-sum-ldtl-name">' + esc(p.name) + '</span>'
              + (pickRx ? '<span class="dr-sum-ldtl-pick">' + pickRx + '</span>' : '')
              + psRx + '</div>';
          }
          var dtlHtml = '';
          _tst.forEach(function(x){
            var _pnx = (team.picks.filter(function(pk){ return pk.p && pk.p.id === x.p.id; })[0] || {}).pn || 0;
            dtlHtml += _ldtlRow(x.slot, x.p, _pnx);
          });
          _bench.forEach(function(pk){ dtlHtml += _ldtlRow('BN', pk.p, pk.pn || 0); });
          dtl.innerHTML = dtlHtml || '<span style="font-size:10px;color:var(--text-muted);padding:4px 0;display:block">No picks found</span>';
        }
      });
    });
  }

  // ── Live draft (P5, Sleeper) ────────────────────────────────────────────────
  function valLookup(id){ var p = playersById[String(id)]; return (p && state) ? Math.round(valOf(p)) : null; }
  function applyLivePicks(picks){
    lastLivePicks = picks;
    state.picks = {}; drafted = {};
    var latestPickedAt = 0;
    picks.forEach(function(p){
      if (p.pick_no == null) return;
      state.picks[p.pick_no] = { id: p.player_id, name: p.name, position: p.position, team: p.team, val: valLookup(p.player_id) };
      if (p.player_id) drafted[String(p.player_id)] = true;
      if (p.picked_at && p.picked_at > latestPickedAt) latestPickedAt = p.picked_at;
    });
    state.lastPickedAt = latestPickedAt || state.lastPickedAt || 0;
    var _tot = (state.teams || 12) * (state.rounds || 15), _next = _tot + 1;
    for (var _pn = 1; _pn <= _tot; _pn++){ if (!state.picks[_pn]){ _next = _pn; break; } }
    state.current = _next;
    _boardSig = null;   // force a full board rebuild on the next render
  }
  // Friendly label for a Sleeper draft status (raw values are snake_case).
  function liveStatusLabel(s){
    s = String(s || '');
    if (s === 'drafting') return 'Live';
    if (s === 'pre_draft') return 'Pre-Draft';
    if (s === 'complete') return 'Complete';
    if (s === 'paused') return 'Paused';
    return s.replace(/_/g, ' ');
  }
  // Update the secondary status badge (Upcoming / Paused / etc.) based on current draft status.
  function _setStatusBadge(isDrafting, isComplete, rawStatus){
    var el = document.getElementById('drUpcomingBadge');
    if (!el) return;
    if (isDrafting || isComplete){ el.style.display = 'none'; return; }
    var isPre = rawStatus === 'pre_draft';
    var isPaused = rawStatus === 'paused';
    el.className = 'dr-pill ' + (isPaused ? 'dr-pill-paused' : 'dr-pill-upcoming');
    el.textContent = isPaused ? 'Paused' : (isPre ? 'Pre-Draft' : liveStatusLabel(rawStatus));
    el.style.display = '';
  }
  function detectLive(){
    if (cfg.isGuest || !cfg.leagueId){
      drAlert('Live draft sync requires opening the Draft Room from your league.');
      return;
    }
    var box = document.getElementById('drLiveList');
    box.style.display = ''; box.innerHTML = '<div class="dr-live-head">Detecting drafts…</div>';
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){ box.innerHTML = '<div class="dr-live-head">Live sync currently supports Sleeper leagues.</div>'; return; }
        var all = resp.drafts || [];
        if (!all.length){ box.innerHTML = '<div class="dr-live-head">No drafts found for this league yet.</div>'; return; }
        // Prefer live/upcoming drafts; only if there are none do we fall back to
        // showing completed ones (so connect never dead-ends with "no drafts").
        var active = all.filter(function(d){ return String(d.status) !== 'complete'; });
        var ds = active.length ? active : all;
        // Soonest first: live, then by scheduled start time.
        ds.sort(function(a, b){
          var ar = a.status === 'drafting' ? 0 : 1, br = b.status === 'drafting' ? 0 : 1;
          if (ar !== br) return ar - br;
          return (Number(a.start_time) || 9e15) - (Number(b.start_time) || 9e15);
        });
        var html = '<div class="dr-live-head">' + (ds.length > 1 ? 'Pick a draft to connect' : 'Your draft') + '</div>';
        ds.forEach(function(d){
          var when = '';
          if (d.start_time){
            try {
              var dt = new Date(Number(d.start_time));
              var today = new Date();
              var sameDay = dt.toDateString() === today.toDateString();
              var t = dt.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
              when = ' · ' + (sameDay ? t : (dt.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + t));
            } catch(e){}
          }
          html += '<button class="dr-live-item" data-id="' + esc(d.draft_id) + '">'
            + '<span class="dr-live-status dr-ls-' + esc(d.status || '') + '">' + esc(liveStatusLabel(d.status)) + '</span>'
            + esc((d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds' + when) + '</button>';
        });
        box.innerHTML = html;
      })
      .catch(function(){ box.innerHTML = '<div class="dr-live-head">Could not detect drafts.</div>'; });
  }
  // Build the owned-pick map from a /api/draft/live response.
  // Completed picks: trust picked_by (user_id). Future picks: apply traded_picks
  // (roster_id-based) to override the default home-slot ownership.
  function buildOwnedFromResponse(d, teams, rounds, order, slot){
    var owned = {};
    var madePickNos = {};
    (d.picks || []).forEach(function(p){
      if (p.pick_no == null) return;
      madePickNos[p.pick_no] = true;
      if (cfg.viewerUserId && p.picked_by === cfg.viewerUserId) owned[p.pick_no] = true;
    });
    if (String(d.status) !== 'complete'){
      // Sleeper traded_picks uses roster_id (original owner) not a slot field.
      // Build slot -> roster_id via draft_order (uid->slot) + user_roster_map (uid->roster_id).
      var slotToRosterId = {};
      if (d.draft_order && d.user_roster_map){
        Object.keys(d.draft_order).forEach(function(uid){
          var sl = d.draft_order[uid];
          var rid = d.user_roster_map[uid];
          if (sl != null && rid != null) slotToRosterId[sl] = rid;
        });
      }
      // tradedPickMap: "originalRosterId:round" -> currentOwnerRosterId
      var tradedPickMap = {};
      (d.traded_picks || []).forEach(function(tp){
        if (tp.roster_id != null && tp.round != null) tradedPickMap[tp.roster_id + ':' + tp.round] = tp.owner_id;
      });
      var viewerRosterId = (cfg.viewerUserId && d.user_roster_map) ? (d.user_roster_map[cfg.viewerUserId] || null) : null;
      for (var pn2 = 1; pn2 <= teams * rounds; pn2++){
        if (madePickNos[pn2]) continue;
        var pn2Slot = slotOnClock(pn2, teams, order);
        var pn2Round = Math.ceil(pn2 / teams);
        var origRid = slotToRosterId[pn2Slot];
        if (origRid != null){
          var tk = origRid + ':' + pn2Round;
          if (tradedPickMap.hasOwnProperty(tk)){
            // Pick has been traded - viewer owns it only if they are the current owner.
            if (viewerRosterId !== null && tradedPickMap[tk] === viewerRosterId) owned[pn2] = true;
          } else {
            // Not traded - original roster still owns it.
            if (viewerRosterId !== null && origRid === viewerRosterId) owned[pn2] = true;
          }
        } else {
          // slot->roster mapping unavailable - fall back to home slot.
          if (slot && pn2Slot === slot) owned[pn2] = true;
        }
      }
    }
    return Object.keys(owned).length ? owned : null;
  }
  // Build pickNo -> owning board slot for EVERY team, so the draft summary credits
  // a traded pick to the team that actually owns it, not the seat it sits in.
  // Completed picks use roster_id; remaining picks apply traded_picks (original
  // roster + round -> current owner_id) over the home-slot owner, mapping roster
  // ids back to board slots via draft_order + user_roster_map. Returns null when
  // the league provides no roster mapping, so callers fall back to home seats.
  function buildPickOwnersFromResponse(d, teams, rounds, order){
    var slotToRosterId = {}, rosterToSlot = {};
    if (d.draft_order && d.user_roster_map){
      Object.keys(d.draft_order).forEach(function(uid){
        var sl = d.draft_order[uid], rid = d.user_roster_map[uid];
        if (sl != null && rid != null){ slotToRosterId[sl] = rid; rosterToSlot[rid] = parseInt(sl, 10); }
      });
    }
    if (!Object.keys(rosterToSlot).length) return null;
    var owners = {};
    (d.picks || []).forEach(function(p){
      if (p.pick_no == null) return;
      if (p.roster_id != null && rosterToSlot[p.roster_id] != null) owners[p.pick_no] = rosterToSlot[p.roster_id];
    });
    var tradedPickMap = {};
    (d.traded_picks || []).forEach(function(tp){
      if (tp.roster_id != null && tp.round != null) tradedPickMap[tp.roster_id + ':' + tp.round] = tp.owner_id;
    });
    var tot = teams * rounds;
    for (var pn = 1; pn <= tot; pn++){
      if (owners[pn] != null) continue;
      var sl = slotOnClock(pn, teams, order);
      var rnd = Math.ceil(pn / teams);
      var origRid = slotToRosterId[sl];
      if (origRid != null){
        var tk = origRid + ':' + rnd;
        var ownRid = tradedPickMap.hasOwnProperty(tk) ? tradedPickMap[tk] : origRid;
        owners[pn] = (rosterToSlot[ownRid] != null) ? rosterToSlot[ownRid] : sl;
      } else {
        owners[pn] = sl;
      }
    }
    return owners;
  }

  function connectLive(draftId){
    stopPolling(); stopPickTimer();
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(draftId), { cache: 'no-store' })
      .then(function(r){
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(d){
        if (!d || d.error){ drAlert('Could not load that draft.'); return; }
        try { _connectLiveApply(d, draftId); }
        catch(err){
          if (window.console) console.error('[draft] connect processing error', err);
          drAlert('Connected, but failed to load the draft board. Refresh to retry.');
        }
      })
      .catch(function(err){
        if (window.console) console.error('[draft] connect fetch error', err);
        drAlert('Could not connect to the live draft.');
      });
  }
  function _connectLiveApply(d, draftId){
    {
        var teams = parseInt(d.teams || 0, 10) || (cfg.numTeams || 12);
        var rounds = parseInt(d.rounds || 0, 10) || 15;
        var order = d.order || 'snake';
        var slot = 0;
        if (cfg.viewerUserId && d.draft_order && d.draft_order[cfg.viewerUserId]) {
          slot = parseInt(d.draft_order[cfg.viewerUserId], 10) || 0;
        }
        var isComplete = String(d.status) === 'complete';
        var isDrafting = String(d.status) === 'drafting';
        var draftType = d.draft_type || (parseInt(d.rounds) <= 5 ? 'rookie' : 'startup');
        state = {
          type: draftType, teams: teams, rounds: rounds, sf: !!cfg.isSuperflex,
          slot: slot, order: order, picks: {}, current: 1,
          owned: buildOwnedFromResponse(d, teams, rounds, order, slot),
          mode: 'live', isComplete: isComplete, isDrafting: isDrafting, sourceDraftId: draftId,
          season: parseInt(d.season, 10) || 0,   // draft's season, for point-in-time ADP grading
          status: String(d.status || ''),
          pickTimer: parseInt(d.pick_timer) || 0,
          startTime: parseInt(d.start_time) || 0,
          slotNames: d.slot_names || {}, queue: [],
          pickOwners: buildPickOwnersFromResponse(d, teams, rounds, order),
          roster: _parseRosterPositions(d.roster_positions),
          scoring: cfg.scoring || readScoring()
        };
        applyLivePicks(d.picks || []);
        // Completed draft: reload the pool against that season's ADP so grades
        // reflect ADP at draft time, not today. No-op cost for the current season
        // (server returns the shared current-season pool); overlays for past ones.
        if (isComplete && state.season){ loadPlayers(); }
        // Seed the poll signature so the first poll doesn't redundantly rebuild.
        _liveSig = liveSig(d); _pollLastAt = Date.now();
        showMain();
        updateDraftBanner();
        document.getElementById('drUndo').style.display = 'none';
        document.getElementById('drLiveBadge').style.display = isDrafting ? '' : 'none';
        _setStatusBadge(isDrafting, isComplete, String(d.status));
        if (isComplete){
          showCompleteSidebar();
        } else {
          document.getElementById('drSide').style.display = '';
          _setUpcomingMode(!isDrafting);
          startPolling();
          _liveSig = liveSig(d);  // re-seed after startPolling resets it so first auto-poll skips a redundant rebuild
          if (isDrafting) startPickTimer();
        }
        loadPlayers();
    }
  }
  // Signature of the live state so a poll that brought nothing new skips the
  // (expensive) board rebuild entirely.
  function liveSig(d){
    var ps = d.picks || [];
    var last = ps.length ? ps[ps.length - 1] : null;
    return String(d.status || '') + '|' + ps.length + '|' + (last && last.pick_no != null ? last.pick_no : 0);
  }
  function _fmtAgo(ms){
    if (!ms) return '';
    var s = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (s < 2) return 'just now';
    if (s < 60) return s + 's ago';
    return Math.floor(s / 60) + 'm ago';
  }
  // Freshness indicator: "Updated 3s ago · next in 1s", or "Syncing…" in flight.
  function updatePollStatus(){
    var el = document.getElementById('drPollStatus');
    if (!el) return;
    if (!state || state.mode !== 'live' || state.isComplete){ el.style.display = 'none'; return; }
    el.style.display = '';
    if (_pollInFlight){
      el.classList.add('is-syncing');
      el.innerHTML = '<span class="dr-poll-dot"></span>Syncing&hellip;';
      return;
    }
    el.classList.remove('is-syncing');
    var lagSuffix = _pickLagMsg ? (' <span style="color:#f59e0b">' + _pickLagMsg + '</span>') : '';
    el.innerHTML = '<span class="dr-poll-dot"></span>' + (_fmtAgo(_pollLastAt) || '-') + lagSuffix;
  }
  // In-page draft banner. Two states: a countdown when a connected draft is within
  // 15 min of its scheduled start, and a "live now" bar while it's drafting. Both
  // carry an "Open in Sleeper" button (you're already in the BR draft room).
  var _START_WINDOW_MS = 15 * 60 * 1000;
  function _fmtCountdown(ms){
    var t = Math.max(0, Math.floor(ms / 1000));
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    return (h > 0 ? h + ':' + (m < 10 ? '0' : '') : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }
  function sleeperDraftUrl(){
    if (cfg.platform === 'sleeper' && state && state.sourceDraftId)
      return 'https://sleeper.com/draft/nfl/' + encodeURIComponent(state.sourceDraftId);
    return null;
  }
  function updateDraftBanner(){
    var el = document.getElementById('drStartBanner');
    if (!el) return;
    // Determine which banner (if any) applies.
    var mode = 'none';
    if (state && state.mode === 'live' && !state.isComplete && !state.isDrafting){
      var st = state.startTime || 0, ms0 = st ? st - Date.now() : 0;
      if (st && ms0 > 0 && ms0 <= _START_WINDOW_MS) mode = 'upcoming';
    }
    if (mode === 'none'){ el.style.display = 'none'; el.removeAttribute('data-bk'); return; }
    if (el.getAttribute('data-bk') !== mode){
      el.setAttribute('data-bk', mode);
      var url = sleeperDraftUrl();
      var joinBtn = url ? '<a class="dr-banner-join" href="' + url + '" target="_blank" rel="noopener">Open in Sleeper <i class="fa-solid fa-arrow-right-long"></i></a>' : '';
      el.className = 'dr-start-banner';
      el.innerHTML = '<span class="dr-banner-ic"><i class="fa-solid fa-calendar-days"></i></span>'
        + '<div class="dr-banner-txt"><b>Your draft starts in <span class="dr-start-cd"></span></b>'
        + '<span>Get your board ready - the pick board goes live automatically.</span></div>' + joinBtn;
    }
    el.style.display = '';
    // Per-tick: refresh only the countdown number (don't clobber the button).
    if (mode === 'upcoming'){
      var cdEl = el.querySelector('.dr-start-cd');
      if (cdEl) cdEl.textContent = _fmtCountdown(state.startTime - Date.now());
    }
  }
  function startPolling(){
    stopPolling();
    _pollCount = 0; _liveSig = null;
    pollTickTimer = setInterval(function(){ updatePollStatus(); updateDraftBanner(); }, 1000);
    pollOnce();
  }
  function schedulePoll(){
    var ms = POLL_MS;
    // Poll faster as a scheduled start approaches (or just passed) so a connected
    // pre-draft flips to live within a couple seconds of the draft actually
    // starting, instead of waiting out the normal cadence.
    if (state && state.mode === 'live' && !state.isComplete){
      if (state.isDrafting){
        ms = 2000;  // active draft: poll every 2s so picks surface quickly
      } else if (state.startTime){
        // Clean gradient: poll harder as the scheduled start nears so the board
        // flips to live promptly, but keep checking often enough far out that a
        // pushed-back start time (moved back 15 min / 1 hr) is caught within ~15s.
        var toStart = state.startTime - Date.now();
        if (toStart <= 60000 && toStart > -900000) ms = 5000;    // 1 min before -> 15 min after
        else if (toStart <= 300000) ms = 10000;                  // 1-5 min out
        else ms = 15000;                                         // far out: catch reschedules
      }
    }
    _pollNextAt = Date.now() + ms;
    pollTimer = setTimeout(pollOnce, ms);
    updatePollStatus();
  }
  // One poll. Chained via setTimeout (never setInterval) so a slow response can't
  // stack overlapping requests. Most polls are "light" (status + picks only); a
  // periodic full poll refreshes slot names and trade-based ownership.
  function pollOnce(){
    if (!state || state.mode !== 'live'){ stopPolling(); return; }
    _pollCount++;
    var full = (_pollCount === 1) || (_pollCount % POLL_FULL_EVERY === 0);
    var url = '/api/draft/live?platform=' + encodeURIComponent(cfg.platform)
            + '&draft_id=' + encodeURIComponent(state.sourceDraftId)
            + (full ? '' : '&light=1');
    _pollInFlight = true; updatePollStatus();
    var ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    var to = setTimeout(function(){ if (ctrl) ctrl.abort(); }, 8000);  // don't let a hung request stall the cadence
    fetch(url, ctrl ? { signal: ctrl.signal, cache: 'no-store' } : { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(d){
        clearTimeout(to); _pollInFlight = false;
        // Guard: if mode changed while this fetch was in-flight (e.g. user started
        // a Practice Mock), discard the response so it can't re-show the LIVE badge.
        if (!state || state.mode !== 'live'){ return; }
        if (!d || !d.picks){ _pollLastAt = Date.now(); return; }
        if (d.start_time != null) state.startTime = parseInt(d.start_time) || 0;
        if (d.pick_timer != null) state.pickTimer = parseInt(d.pick_timer) || 0;
        var sig = liveSig(d);
        if (sig !== _liveSig){
          _liveSig = sig;
          var prevCurrent = state.current, prevDrafting = state.isDrafting;
          // Diagnostic: measure Sleeper REST lag (picked_at to detection time).
          // picked_at is epoch ms from Sleeper; if missing we show "no ts" so we
          // know whether the field is being returned at all.
          var _newPickedAt = 0;
          (d.picks || []).forEach(function(pk){ if (pk.picked_at && pk.picked_at > _newPickedAt) _newPickedAt = pk.picked_at; });
          if (_newPickedAt){
            var lagS = Math.round((Date.now() - _newPickedAt) / 1000);
            _pickLagMsg = 'pick +' + Math.max(0, lagS) + 's';
          } else {
            _pickLagMsg = 'pick (no ts)';
          }
          setTimeout(function(){ _pickLagMsg = null; }, 15000);
          if (full){
            // Only the full payload carries trades + roster map for ownership.
            state.owned = buildOwnedFromResponse(d, state.teams, state.rounds, state.order, state.slot);
            state.pickOwners = buildPickOwnersFromResponse(d, state.teams, state.rounds, state.order) || state.pickOwners;
            if (d.slot_names) state.slotNames = d.slot_names;
          }
          applyLivePicks(d.picks); render();
          var isDrafting = String(d.status) === 'drafting';
          state.isDrafting = isDrafting;
          state.status = String(d.status || '');
          document.getElementById('drLiveBadge').style.display = isDrafting ? '' : 'none';
          _setStatusBadge(isDrafting, String(d.status) === 'complete', String(d.status));
          _setUpcomingMode(!isDrafting && String(d.status) !== 'complete');
          if (isDrafting && (!prevDrafting || state.current !== prevCurrent)) startPickTimer();
          if (String(d.status) === 'complete'){
            stopPolling(); stopPickTimer(); state.isComplete = true; save();
            showCompleteSidebar();
            return;
          }
        }
        _pollLastAt = Date.now();
      })
      .catch(function(){ clearTimeout(to); _pollInFlight = false; })
      .then(function(){
        // Schedule the next poll once this one settles (success or failure),
        // unless polling was torn down (e.g. draft completed).
        if (pollTickTimer && state && state.mode === 'live' && !state.isComplete) schedulePoll();
      });
  }
  function stopPolling(){
    if (pollTimer){ clearTimeout(pollTimer); pollTimer = null; }
    if (pollTickTimer){ clearInterval(pollTickTimer); pollTickTimer = null; }
    _pollInFlight = false;
    var el = document.getElementById('drPollStatus'); if (el) el.style.display = 'none';
    var sb = document.getElementById('drStartBanner'); if (sb) sb.style.display = 'none';
  }

  function _setUpcomingMode(upcoming){
    // Queue tab stays visible in pre-draft so users can build their target list.
  }

  // ── Pick countdown timer ────────────────────────────────────────────────────
  // Counts down from state.pickTimer seconds. Clock starts fresh whenever the
  // current pick number changes (detected by the poll loop above).
  function startPickTimer(){
    stopPickTimer();
    if (!state || !state.pickTimer) return;
    _timerPickNo = state.current;
    _timerPickStart = Date.now();
    var el = document.getElementById('drPickTimer');
    if (el) el.style.display = '';
    function tick(){
      if (!state || state.pickTimer <= 0){ stopPickTimer(); return; }
      var elapsed = Math.round((Date.now() - _timerPickStart) / 1000);
      var remaining = Math.max(0, state.pickTimer - elapsed);
      var m = Math.floor(remaining / 60), s = remaining % 60;
      var txt = m > 0 ? (m + ':' + (s < 10 ? '0' : '') + s) : (remaining + 's');
      var el2 = document.getElementById('drPickTimer');
      if (el2){
        el2.textContent = txt;
        el2.className = 'dr-pick-timer' + (remaining <= 30 ? ' urgent' : '');
      }
      if (remaining === 0) stopPickTimer();
    }
    tick();
    _timerInterval = setInterval(tick, 1000);
  }
  function stopPickTimer(){
    if (_timerInterval){ clearInterval(_timerInterval); _timerInterval = null; }
    var el = document.getElementById('drPickTimer');
    if (el){ el.style.display = 'none'; el.textContent = ''; }
  }


  function userNextPick(){
    var ups = upcomingOwnedPicks();
    return ups.length ? ups[0] : null;
  }

  function leagueMetaParts(){
    if (!state) return [];
    var typeLbl = state.keeper ? 'Keeper'
      : (state.type === 'rookie' ? 'Rookie' : (state.type === 'redraft' ? 'Redraft' : 'Startup'));
    var sc = scoringCfg();
    var pprLbl = sc.ppr === 1 ? 'Full PPR' : (sc.ppr === 0.5 ? 'Half PPR' : (sc.ppr === 0 ? 'Standard' : (sc.ppr + ' PPR')));
    var orderLbl = state.order === 'linear' ? 'Linear' : (state.order === '3rr' ? '3RR' : 'Snake');
    var parts = [
      typeLbl,
      state.teams + '-team ' + (state.sf ? 'SF' : '1QB'),
      pprLbl
    ];
    if (sc.tep) parts.push('TEP +' + sc.tep);
    if (sc.passTd >= 6) parts.push('6-pt Pass TD');
    parts.push(orderLbl);
    if (state.rounds) parts.push(state.rounds + ' rnd');
    return parts;
  }
  function renderLeagueMeta(){
    var el = document.getElementById('drLeagueMeta');
    if (!el || !state) return;
    var parts = leagueMetaParts();
    el.innerHTML = parts.map(function(p){ return '<span class="dr-lm-chip">' + p + '</span>'; }).join('');
    el.hidden = !parts.length;
    var canEdit = state.mode !== 'live';
    el.classList.toggle('is-editable', canEdit);
    el.disabled = !canEdit;
    el.title = (canEdit ? 'Edit setup — ' : '') + parts.join(' · ');
    el.setAttribute('aria-label', (canEdit ? 'Edit setup: ' : 'League settings: ') + parts.join(', '));
  }

  function renderStatus(){
    var total = state.teams * state.rounds;
    var done = state.current > total;
    var r = Math.ceil(state.current / state.teams);
    var pickInRound = ((state.current - 1) % state.teams) + 1;
    document.getElementById('drPickPill').textContent = done ? 'Done' : ('Pick: ' + r + '.' + (pickInRound < 10 ? '0' : '') + pickInRound);
    renderLeagueMeta();
    var oc = document.getElementById('drOnClock');
    var ocWrap = document.getElementById('drOnClockWrap');
    var ocLabel = ocWrap ? ocWrap.querySelector('.dr-onclock-label') : null;
    var mineNow = false;
    // A paused draft HAS started - show who's on the clock (the "Paused" badge
    // already conveys the paused state). Only a true pre_draft is "not started".
    var notStarted = state.mode === 'live' && !state.isDrafting && state.status !== 'paused' && !done;
    if (done) { oc.textContent = 'Draft complete'; if (ocLabel) ocLabel.style.display = 'none'; }
    else if (notStarted) { oc.textContent = 'Draft hasn\'t started'; if (ocLabel) ocLabel.style.display = 'none'; }
    else if (sim && !simStarted) { oc.textContent = 'Ready to draft'; if (ocLabel){ ocLabel.style.display = ''; ocLabel.textContent = 'Claim picks, then Start'; } }
    else {
      var slot = slotOnClock(state.current, state.teams, state.order);
      mineNow = isMyPick(state.current);
      oc.textContent = mineNow ? ('You (Team ' + slot + ')') : teamName(slot);
      if (ocLabel){ ocLabel.style.display = ''; ocLabel.textContent = 'On the clock'; }
    }
    if (ocWrap) ocWrap.classList.toggle('dr-onclock-you', mineNow);
    var nextPill = document.getElementById('drNextPill');
    var np = done ? null : userNextPick();
    if (np){
      var npRound = Math.ceil(np / state.teams);
      var npInRound = ((np - 1) % state.teams) + 1;
      nextPill.style.display = '';
      nextPill.textContent = 'Next: ' + npRound + '.' + (npInRound < 10 ? '0' : '') + npInRound;
    }
    else { nextPill.style.display = 'none'; }
    document.getElementById('drProgress').textContent = Math.min(state.current - 1, total) + ' / ' + total + ' picks';
    var gp = document.getElementById('drGradePill');
    var g = gradeTeam();
    if (g){ gp.style.display = ''; gp.textContent = 'Grade ' + gradeLetter(g.score) + gradeEarlySuffix(g); } else { gp.style.display = 'none'; }
    // The pick-trade evaluator only makes sense while picks are still to be made;
    // hide it once the draft is done/complete (nothing left to trade for).
    var _ptBtn = document.getElementById('drPickTradeBtn');
    if (_ptBtn) _ptBtn.style.display = (done || (state && state.isComplete)) ? 'none' : '';
    // The report card grades every team in the league, so make it reachable any
    // time there are picks on the board - not just when the draft is finished -
    // so you can compare other teams' grades mid-draft.
    var _anyPicks = !!(state && state.picks && Object.keys(state.picks).some(function(k){ return !!state.picks[k]; }));
    document.getElementById('drSummaryBtn').style.display = _anyPicks ? '' : 'none';
  }

  // ── Board rendering (incremental) ───────────────────────────────────────────
  // Pool of fantasy team names so CPU opponents in a mock have some character
  // instead of "Team 7". Connected/live leagues supply their own names via
  // slot_names and never use these. Kept apostrophe-free for clean embedding.
  var _CPU_TEAM_NAMES = [
    'The Audibles', 'Gridiron Gang', 'End Zone Elite', 'Hail Mary Heroes',
    'Blitz Brigade', 'Pigskin Pirates', 'Fourth and Long', 'Red Zone Rebels',
    'Touchdown Titans', 'Field Goal Frenzy', 'Purple People Eaters', 'Victory Formation',
    'The Hurry Up', 'Pocket Presence', 'Play Action Heroes', 'The Pick Six',
    'Goal Line Stand', 'Two Minute Drill', 'The Blind Side', 'Shotgun Formation',
    'Cover Two Crew', 'The Zone Read', 'Screen Pass Squad', 'Nickel Defense',
    'The Flea Flickers', 'Wildcat Offense', 'Backfield Bandits', 'The Gunslingers',
    'Moss Boss', 'The Brady Bunch', 'The Replacements', 'Comeback Kids',
    'Sack Religious', 'Captain Checkdown', 'Cooked Lamb', 'Order of the Pick'
  ];
  // Assign a stable, unique random name to every non-user slot once per mock.
  // Seeded by the draft seed so each mock differs but stays consistent within it.
  function _ensureSlotNames(){
    if (state.mode === 'live') return;           // real leagues supply names
    if (!state.slotNames) state.slotNames = {};
    var teams = state.teams || 12, filled = 0;
    for (var s = 1; s <= teams; s++){ if (state.slotNames[s]) filled++; }
    if (filled >= teams) return;                 // already assigned
    // Deterministic Fisher-Yates shuffle of the pool using the draft seed.
    var pool = _CPU_TEAM_NAMES.slice();
    for (var i = pool.length - 1; i > 0; i--){
      var j = Math.floor(_rand01('teamname:' + i) * (i + 1));
      var tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }
    var k = 0;
    for (var s2 = 1; s2 <= teams; s2++){
      if (state.slotNames[s2]) continue;
      var base = pool[k % pool.length];
      // If teams exceed the pool, suffix a number so names stay unique.
      state.slotNames[s2] = k >= pool.length ? base + ' ' + (Math.floor(k / pool.length) + 1) : base;
      k++;
    }
    save();
  }
  // Seat label only. "You" identity is decided by pick ownership at the call
  // site (ownsAllInColumn / isMyPick), not by the original home slot.
  function teamName(slot){
    if (state.slotNames && state.slotNames[slot]) return state.slotNames[slot];
    if (state.mode !== 'live'){
      _ensureSlotNames();
      if (state.slotNames && state.slotNames[slot]) return state.slotNames[slot];
    }
    return 'Team ' + slot;
  }
  function cellClass(pn){
    var slot = slotOnClock(pn, state.teams, state.order);
    var pl = state.picks[pn];
    var mine = isMyPick(pn);
    // A claimed pick that is not your home slot is a traded-in pick.
    var traded = mine && state.slot && slot !== state.slot;
    return 'dr-cell' + (pl ? ' dr-cell-filled' : ' dr-cell-empty')
      + (mine ? ' dr-cell-mine' : '') + (traded ? ' dr-cell-claimed' : '')
      + ((canClaim(pn)) ? ' dr-cell-claimable' : '')
      + (pl && pl.keeper ? ' dr-cell-keeper' : '')
      + (pn === justPick ? ' dr-cell-just' : '');
  }
  // Future, uncommitted picks can be claimed/unclaimed in mock/manual mode.
  function canClaim(pn){
    return state.mode !== 'live' && pn >= state.current && !state.picks[pn];
  }
  // Label for a pick that has been traded to another team (shown on the seat the
  // pick originally belonged to). Empty for untraded picks and the viewer's own
  // picks (those use the YOU flag / accent styling instead).
  function tradedOwnerLabel(pn){
    if (!state.pickOwners) return '';
    var owner = state.pickOwners[pn];
    if (owner == null) return '';
    if (owner === slotOnClock(pn, state.teams, state.order)) return '';  // not traded
    if (isMyPick(pn)) return '';
    return teamName(owner);
  }
  // Pick score for a stored board pick. Mock picks store it at draft time; live/
  // synced picks don't, so compute it retroactively at the historical pick slot
  // (same approach gradePicks uses) and cache it back onto the pick object.
  function storedPickScore(pn, pl){
    if (!pl) return null;
    // Display the GRADE score so every per-pick chip matches the Teams-page
    // grade. The kernel carries no timing terms, so this equals the board's
    // Pick Score for the same inputs; memoized separately from the live pl.ps.
    if (pl.gps != null) return pl.gps;
    var full = playersById[String(pl.id)];
    if (!full || !players.length) return null;
    var maxVal = 0; players.forEach(function(q){ var v = valOf(q); if (v > maxVal) maxVal = v; });
    if (maxVal <= 0) return null;
    // Owning team's positional + quality counts from this team's earlier picks
    // (same progressive context gradePicks / the server use — never the viewer's).
    var owner = (state.pickOwners && state.pickOwners[pn] != null)
      ? state.pickOwners[pn] : slotOnClock(pn, state.teams, state.order);
    var counts = { QB:0, RB:0, WR:0, TE:0 };
    var qualByPos = { QB:0, RB:0, WR:0, TE:0 };
    Object.keys(state.picks).forEach(function(k){
      var kp = parseInt(k, 10);
      if (kp >= pn || !state.picks[k]) return;
      var o2 = (state.pickOwners && state.pickOwners[kp] != null)
        ? state.pickOwners[kp] : slotOnClock(kp, state.teams, state.order);
      if (o2 !== owner) return;
      var prev = state.picks[k];
      var pos2 = (prev.position || '').toUpperCase();
      if (counts[pos2] != null) counts[pos2]++;
      if (qualByPos[pos2] != null){
        var prevFull = playersById[String(prev.id)];
        var v = prevFull ? vorOf(prevFull) : null;
        if (v == null || v > 0) qualByPos[pos2]++;
      }
    });
    var ps = pickScore(full, maxVal, counts, {
      grading: true, pickNo: pn, qualByPos: qualByPos
    });
    pl.gps = ps;   // memoize the grade score (matches gradePicks / the server)
    return ps;
  }
  // "3.06" round.pick label for a given overall pick number.
  function roundPickStr(pn){
    var teams = state.teams || 12;
    var rd = Math.ceil(pn / teams);
    var pk = pn - (rd - 1) * teams;
    return rd + '.' + (pk < 10 ? '0' + pk : String(pk));
  }
  function cellInner(pn){
    var pl = state.picks[pn];
    // Empty cell: show the round.pick centered so the grid reads as a real board
    // (keeper/YOU/traded flags still render); filled cell keeps the overall # top-left.
    if (!pl){
      var eh = '';
      if (isMyPick(pn)) eh += '<span class="dr-cell-mineflag">YOU</span>';
      var _eown = tradedOwnerLabel(pn);
      if (_eown) eh += '<span class="dr-cell-owner">' + esc(_eown) + '</span>';
      eh += '<span class="dr-cell-rp">' + roundPickStr(pn) + '<small class="dr-cell-rp-ov">' + pn + '</small></span>';
      return eh;
    }
    var h = '<span class="dr-cell-num">' + pn + '</span>';
    if (pl && pl.keeper) h += '<span class="dr-cell-keepflag">KEEP</span>';
    else if (isMyPick(pn)) h += '<span class="dr-cell-mineflag">YOU</span>';
    var _own = tradedOwnerLabel(pn);
    if (_own) h += '<span class="dr-cell-owner">' + esc(_own) + '</span>';
    if (pl){
      if (_cellShowPs) {
        var _cvps = relPS(pl, pn);
        if (_cvps != null) h += '<span class="dr-cell-val" style="color:' + psColor(_cvps) + '">' + _cvps + '</span>';
      } else {
        if (pl.val != null) h += '<span class="dr-cell-val">' + Math.round(pl.val) + '</span>';
      }
      h += '<img class="dr-hs" src="' + playerImgUrl(pl) + '" alt="" onerror="this.style.visibility=\'hidden\'">';
      h += '<div class="dr-cell-body"><div class="dr-cell-name">' + esc(pl.name) + '</div>'
        + '<div class="dr-cell-meta"><span class="dr-posbadge" style="background:' + posColor(pl.position) + '">' + esc(pl.position) + '</span> ' + esc(pl.team || '') + '</div></div>';
    }
    return h;
  }
  function buildBoard(){
    var board = document.getElementById('drBoard');
    var teams = state.teams, rounds = state.rounds;
    board.style.gridTemplateColumns = '30px repeat(' + teams + ', minmax(108px, 1fr))';
    var html = '<div class="dr-colhead dr-rowhead dr-corner"></div>';
    for (var s = 1; s <= teams; s++){
      // Highlight columns by actual ownership: full column you own = "You",
      // a column where you only hold traded-in pick(s) keeps its seat name but
      // still gets the star + accent so you can spot your picks.
      var ownsAny = ownsAnyInColumn(s);
      var you = ownsAny ? ' dr-colhead-you' : '';
      var label = ownsAllInColumn(s) ? 'You' : teamName(s);
      html += '<div class="dr-colhead' + you + '" data-slot="' + s + '" style="cursor:default;">' + esc(label) + (ownsAny ? ' ★' : '') + '</div>';
    }
    for (var rnd = 1; rnd <= rounds; rnd++){
      html += '<div class="dr-colhead dr-rowhead">R' + rnd + '</div>';
      for (var slot = 1; slot <= teams; slot++){
        var pn = pickNum(rnd, slot, teams, state.order);
        html += '<div class="' + cellClass(pn) + '" id="dc' + pn + '" data-pn="' + pn + '" style="' + cellPosVar(pn) + '">' + cellInner(pn) + '</div>';
      }
    }
    board.innerHTML = html;
    _boardSig = boardSig();
    refreshCurrent();
  }
  // Per-cell CSS var carrying the pick's position colour, so .dr-cell-filled can
  // tint the whole cell by position instead of a uniform accent wash.
  function cellPosVar(pn){
    var pl = state.picks[pn];
    if (!pl) return '';
    var c = posColor(pl.position);
    return c ? ('--pos:' + c) : '';
  }
  function paintCell(pn){
    var el = document.getElementById('dc' + pn);
    if (!el) return;
    el.className = cellClass(pn);
    el.style.cssText = cellPosVar(pn);
    el.innerHTML = cellInner(pn);
  }
  function refreshCurrent(){
    var prev = document.querySelector('.dr-cell-current');
    if (prev) prev.classList.remove('dr-cell-current');
    var cur = document.getElementById('dc' + state.current);
    if (cur){
      cur.classList.add('dr-cell-current');
      requestAnimationFrame(function(){
        try { cur.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' }); }
        catch (e) { try { cur.scrollIntoView(); } catch (_) {} }
      });
    }
  }
  function boardSig(){ return [state.teams, state.rounds, state.order, state.slot, state.mode || 'm', ownedPicks().join(',')].join('|'); }
  function renderBoard(){
    // Full rebuild only when the board structure changes; otherwise picks are
    // painted incrementally by commitPick/undo for smooth (Instant) sims.
    if (_boardSig !== boardSig()) buildBoard();
  }

  // Build/refresh the ADP source dropdown inside #drAdpSrc for the current
  // draft mode (state.type). Options come from the payload and already exclude
  // sources invalid for the mode (Yahoo redraft-only, BR Fantasy dyn/rookie).
  function syncAdpSourceSelector(){
    var host = document.getElementById('drAdpSrc');
    if (!host) return;
    var serverOpts = adpSourceOptions[state.type] || [];
    if (!serverOpts.length){
      host.textContent = 'ADP source: ' + (adpSources[state.type] || 'unavailable');
      return;
    }
    // "Auto" is the server default the pool loads with; label it with the
    // source the server actually used so the dropdown never misstates it.
    var usedLabel = adpSources[state.type];
    var autoLabel = (usedLabel && usedLabel !== 'none') ? ('Auto (' + usedLabel + ')') : 'Auto';
    var opts = [{ value: 'auto', label: autoLabel }].concat(serverOpts);

    // Preserve the current selection across refreshes (the option set changes
    // with the draft mode); default to the first option ("auto").
    var want = adpSource;
    if (!opts.some(function(o){ return o.value === want; })) want = opts[0] ? opts[0].value : 'auto';
    adpSource = want;

    // Custom dropdown (reuses the .dr-sortsel styles) instead of a native
    // <select>, whose option popup can't be themed to match the app.
    var ui = document.getElementById('drAdpSrcUI');
    if (!ui){
      host.innerHTML = '<label class="dr-adp-src-label" for="drAdpSrcBtn">ADP source</label>'
        + '<div class="dr-sortsel" id="drAdpSrcUI">'
        +   '<button type="button" class="dr-sortsel-btn" id="drAdpSrcBtn" aria-haspopup="listbox" aria-expanded="false">'
        +     '<span id="drAdpSrcLbl"></span>'
        +     '<svg class="dr-sortsel-caret" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>'
        +   '</button>'
        +   '<div class="dr-sortsel-menu" id="drAdpSrcMenu" role="listbox" hidden></div>'
        + '</div>';
      ui = document.getElementById('drAdpSrcUI');
      var btn = document.getElementById('drAdpSrcBtn');
      var menu = document.getElementById('drAdpSrcMenu');
      btn.addEventListener('click', function(e){
        e.stopPropagation();
        var willOpen = menu.hidden;
        menu.hidden = !willOpen;
        btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
      });
      menu.addEventListener('click', function(e){
        var opt = e.target.closest('.dr-sortsel-opt'); if (!opt) return;
        e.stopPropagation();
        menu.hidden = true;
        btn.setAttribute('aria-expanded', 'false');
        var v = opt.getAttribute('data-val');
        if (v !== adpSource){
          adpSource = v;
          syncAdpSourceSelector();   // refresh label + active state
          loadPlayers();             // re-fetch the pool scored by the chosen source
        }
      });
      // One persistent outside-click handler (survives menu rebuilds).
      if (!window.__drAdpSrcDocBound){
        window.__drAdpSrcDocBound = true;
        document.addEventListener('click', function(e){
          var m = document.getElementById('drAdpSrcMenu');
          var u = document.getElementById('drAdpSrcUI');
          var b = document.getElementById('drAdpSrcBtn');
          if (m && u && !m.hidden && !u.contains(e.target)){
            m.hidden = true; if (b) b.setAttribute('aria-expanded', 'false');
          }
        });
      }
    }

    // Refresh the button label and option list every call.
    var cur = opts.filter(function(o){ return o.value === want; })[0] || opts[0];
    document.getElementById('drAdpSrcLbl').textContent = cur ? cur.label : 'Auto';
    document.getElementById('drAdpSrcMenu').innerHTML = opts.map(function(o){
      return '<button type="button" class="dr-sortsel-opt' + (o.value === want ? ' is-active' : '')
        + '" role="option" data-val="' + esc(o.value) + '">' + esc(o.label) + '</button>';
    }).join('');
  }

  function renderBA(){
    syncAdpSourceSelector();
    var _sortEl = document.getElementById('drBaSortBtn');
    var sortBy = (_sortEl && _sortEl.getAttribute('data-val')) || 'ps';
    var q = (document.getElementById('drSearch').value || '').trim().toLowerCase();
    var pool = availablePool().filter(function(p){
      if (!_posMatches(p.position)) return false;
      if (q && String(p.name||'').toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    // p._ps + the pool-relative scale are refreshed in renderSide; ensure they
    // exist for any path that reaches renderBA directly (search/sort handlers).
    if (_psPoolMax <= 0) refreshPsPool();
    var recommendationRanks = {};
    if (sortBy === 'ps') rankedRecommendationPool().forEach(function(p, i){
      recommendationRanks[String(p.id)] = i + 1;
    });
    pool.sort(function(a, b){
      // K/DEF: order them among themselves by ADP (the server now attaches real
      // Sleeper D/ST + kicker ADP), so the defense/kicker managers actually draft
      // first surfaces first instead of alphabetically. Fall back to projected PPG
      // only when neither has an ADP (e.g. an unpriced kicker).
      var aKd = (String(a.position||'').toUpperCase() === 'K' || String(a.position||'').toUpperCase() === 'DEF');
      var bKd = (String(b.position||'').toUpperCase() === 'K' || String(b.position||'').toUpperCase() === 'DEF');
      if (aKd && bKd){
        var akAdp = adpOf(a), bkAdp = adpOf(b);
        if (akAdp != null || bkAdp != null){
          return (akAdp != null ? akAdp : 99999) - (bkAdp != null ? bkAdp : 99999);
        }
        return (ppgOf(b) || 0) - (ppgOf(a) || 0);
      }
      if (sortBy === 'adp'){
        var aa = adpOf(a), ba = adpOf(b);
        return (aa != null ? aa : 99999) - (ba != null ? ba : 99999);
      }
      if (sortBy === 'pickscore'){ return (b._ps || 0) - (a._ps || 0); }
      if (sortBy === 'ps'){ return (b._ds || 0) - (a._ds || 0) || (b._ps || 0) - (a._ps || 0); }
      if (sortBy === 'ppg'){ return (ppgOf(b) || 0) - (ppgOf(a) || 0); }
      return valOf(b) - valOf(a);
    });
    if (!pool.length){ listInto(emptyNote('No players match', 'Try another position filter or clear your search.', _DR_SEARCH_ICON)); return; }
    var nextPick = hasOwned() ? nextOwnedAfterCurrent() : null;
    var html = balanceAlert() + alertBanners();
    // K/DEF have no startup ADP so they sort to the very end and fall past the
    // 200-player cap. Separate them out so they always render after skill players.
    var _isKD = function(p){ var pos = String(p.position||'').toUpperCase(); return pos === 'K' || pos === 'DEF'; };
    var mainPool = (_posIsAll() && wantsKDef()) ? pool.filter(function(p){ return !_isKD(p); }) : pool;
    var kdPool  = (_posIsAll() && wantsKDef()) ? pool.filter(_isKD) : [];
    // Late-round K/DEF nudge: K/DEF are ungraded and normally sit at the very
    // bottom, so they'd never read as a suggestion. Once a required K/DEF slot
    // must be filled soon (few picks left, or the last few rounds), surface the
    // best available one at the TOP with a reason so it actually gets drafted.
    var _promoted = [];
    if (_posIsAll() && wantsKDef() && kdPool.length && hasOwned()){
      var _rs = (state && state.roster) || defaultRoster();
      var _mc = myPosCounts();
      var _needK = Math.max(0, (_rs.K || 0) - (_mc.K || 0));
      var _needDef = Math.max(0, (_rs.DEF || 0) - (_mc.DEF || 0));
      var _remainPicks = upcomingOwnedPicks().length;
      var _remainRds = (state.rounds || 0) - Math.floor((state.current - 1) / (state.teams || 12));
      var _kdefTime = (_needK + _needDef) > 0 && (_remainPicks <= (_needK + _needDef) + 2 || _remainRds <= 3);
      if (_kdefTime){
        if (_needK > 0){ var _bk = kdPool.filter(function(p){ return String(p.position).toUpperCase() === 'K'; })[0]; if (_bk) _promoted.push(_bk); }
        if (_needDef > 0){ var _bd = kdPool.filter(function(p){ return String(p.position).toUpperCase() === 'DEF'; })[0]; if (_bd) _promoted.push(_bd); }
        kdPool = kdPool.filter(function(p){ return _promoted.indexOf(p) < 0; });
      }
    }
    _promoted.forEach(function(p){
      html += playerRowHtml(p, { reason: 'Fill your ' + String(p.position || '').toUpperCase() + ' slot before the draft ends' });
    });
    var _reasonCounts = sortBy === 'ps' ? myPosCounts() : null;
    for (var i = 0; i < Math.min(mainPool.length, 200); i++){
      var p = mainPool[i];
      var opts = sortBy === 'ps' ? { reason: pickReason(p, _reasonCounts), rank: recommendationRanks[String(p.id)] }
        : { showPickScore: sortBy === 'pickscore' };
      if (nextPick){
        var prob = availProb(p, nextPick);
        // Show the survival % for every player so elite names that likely go
        // before your pick still display their (low) odds, not a blank.
        if (prob != null) opts.availAt = { pn: nextPick, prob: prob };
      }
      html += playerRowHtml(p, opts);
    }
    kdPool.forEach(function(p){ html += playerRowHtml(p, {}); });
    listInto(html);
  }

  // Pick Score for a single player (computes the pool max + your roster counts).
  function pickScoreFor(p, pickNo){
    var pool = availablePool();
    var maxVal = 0; pool.forEach(function(x){ var v = valOf(x); if (v > maxVal) maxVal = v; });
    return pickScore(p, maxVal, myPosCounts(), pickNo ? { pickNo: pickNo } : undefined);
  }

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  // ── Glossary / inline term explainers ───────────────────────────────────────
  // Single source of truth so the inline ⓘ tooltips and the help popover agree.
  var _GLOSSARY = [
    { term: 'Recommendation', def: 'The live, roster-aware order for this pick. It starts with Pick Score, then accounts for whether the player fills a starter or FLEX spot, backup and overfill cost, required slots and picks remaining, positional depth, expected availability at your next pick, and recent investment at QB or TE. A major value fall can still overcome imperfect roster fit. Recommendation is shown as a rank rather than a grade because its internal utility naturally changes as the board is depleted.' },
    { term: 'Pick Score (PS)', def: 'A 0-100 grade of pick quality. On the live board, sidebar, compare modal, and player preview it is scaled relative to the best player still available (so a strong late pick still reads well). Made-pick chips on the report card / Deep Dive “Board PS” use the same relative scale at that historical slot. Your letter grade’s Value bar uses the absolute, round-weighted kernel score — those two numbers can differ. Kickers and defenses aren’t scored.' },
    { term: 'Value', def: 'The player’s trade value as an asset on a 0-999 scale - dynasty value for startup/rookie drafts, redraft value for redraft.' },
    { term: 'VOR / VORP', def: 'Value Over Replacement: how much better a player is than a replacement-level starter at their position (a fixed, preseason-style baseline). VORP uses projected season fantasy points; VOR uses dynasty or redraft trade value. Last season\'s injury-shortened totals are not used.' },
    { term: 'ADP', def: 'Average Draft Position - the typical overall pick a player goes at in real drafts. If it’s below your current pick, they’ve fallen and may be a value. When a sample size (n=) is shown, a small n means the ADP is noisy.' },
    { term: 'Tier', def: 'Players grouped by talent gaps (Tier 1 = elite). A tier “cliff” means only a couple of players remain before a real drop-off at that position.' },
    { term: 'PPG', def: 'Points per game - projected for the upcoming season, or last season’s actual when that’s shown.' },
    { term: 'Survival %', def: 'The chance a player is still on the board at your next pick. Starts from consensus ADP, then adapts to how your draft is actually going - if the room is reaching, letting players slide, drafting unpredictably, or running on a position, the odds shift to match (kicks in after the first several picks).' },
    { term: 'Grade · Value', def: 'How strong your picks are by pick score, weighted toward the earlier rounds where it matters most.' },
    { term: 'Grade · Starters', def: 'How good your projected starting lineup is versus a league-average team. 100% is a league-average lineup; the rank is among teams in this draft. Snake drafts are close to zero-sum, so a lineup near 100% of average can still rank 1st or 2nd.' },
    { term: 'Grade · Construction', def: 'How well you’ve filled your starting slots and balanced your positions.' },
    { term: 'Grade · Early', def: 'Shown until your team has 8 picks (3 in a rookie draft). The letter is real — including at two picks / the start of round 3 — but construction is still ramping and the sample is small.' }
  ];
  // Inline info icon: data-tip drives a CSS hover/focus bubble. tabindex makes it
  // tap- and keyboard-accessible.
  function infoIcon(tip){
    return '<span class="dr-info" tabindex="0" role="button" aria-label="' + esc(tip) + '" data-tip="' + esc(tip) + '">i</span>';
  }
  function openGlossary(){
    var body = _GLOSSARY.map(function(g){
      return '<div class="dr-gloss-item"><div class="dr-gloss-term">' + esc(g.term) + '</div>'
        + '<div class="dr-gloss-def">' + esc(g.def) + '</div></div>';
    }).join('');
    document.getElementById('drGlossBody').innerHTML = body;
    document.getElementById('drGloss').style.display = 'flex';
  }
  function closeGlossary(){ document.getElementById('drGloss').style.display = 'none'; }

  // ── Summary overlay ─────────────────────────────────────────────────────────
  function openSummary(){
    var hasSlot = hasOwned();
    var hasPicks = state && Object.keys(state.picks || {}).some(function(k){ return !!state.picks[k]; });
    if (!state || (!hasSlot && !hasPicks)) return;

    var g = hasSlot ? gradeTeam() : null;
    var gradeCol = g ? (g.score >= 75 ? '#22c55e' : g.score >= 60 ? '#38bdf8' : g.score >= 45 ? '#f59e0b' : '#ef4444') : null;

    // Build starters / bench for my team
    var mine = [], starters = [], bench = [];
    if (hasSlot){
      mine = myPicksList().slice();
      var _olS = optimalLineup(mine);
      starters = _olS.starters;
      bench = _olS.bench;
      // A synced/completed draft's picks never got a live pick score, so their
      // PS chips (and the Avg/Starter PS stats) would be blank. Recompute the
      // grade score - memoized, same value the Teams page shows - so the report
      // card is populated regardless of how the draft was run.
      var _idToPn = {};
      Object.keys(state.picks || {}).forEach(function(k){
        var _pp = state.picks[k]; if (_pp) _idToPn[String(_pp.id)] = parseInt(k, 10);
      });
      mine.forEach(function(p){
        if (p && p.ps == null){
          var _g = storedPickScore(_idToPn[String(p.id)] || 0, p);
          if (_g != null) p.ps = _g;
        }
      });
    }

    // Grade ring + component bars
    var gradeHtml = g
      ? ('<div class="dr-sum-grade-wrap">'
         + '<div class="dr-sum-grade-ring" style="border-color:' + gradeCol + ';color:' + gradeCol + '">'
         + '<span class="dr-sum-grade">' + gradeLetter(g.score) + '</span></div>'
         + '<div class="dr-sum-grade-bars">'
         + (g.provisional ? '<div class="dr-grade-early">Early — still forming</div>' : '')
         + gradeBars(g) + '</div>'
         + '</div>')
      : '';

    // Stats strip
    var statsHtml = '';
    if (hasSlot && mine.length){
      var sumProjTotal = 0, sumProjCount = 0, sumT12 = 0;
      var sumAllPsTotal = 0, sumAllPsCount = 0, sumStarterPsTotal = 0, sumStarterPsCount = 0;
      var _ssSet = {};
      starters.forEach(function(s){ if (s.p) _ssSet[String(s.p.id)] = true; });
      mine.forEach(function(p){
        var _ppgv = p.proj_ppg != null ? Number(p.proj_ppg) : null;
        if (_ppgv != null){ sumProjTotal += _ppgv; sumProjCount++; }
        var _fp = playersById[String(p.id)] || p;
        var _t = tierOf(_fp); if (_t != null && _t <= 2) sumT12++;
        var _psShown = relPS(p);
        if (_psShown != null){ sumAllPsTotal += _psShown; sumAllPsCount++; }
        if (_ssSet[String(p.id)] && _psShown != null){ sumStarterPsTotal += _psShown; sumStarterPsCount++; }
      });
      var _sits = [];
      if (sumProjCount >= 2) _sits.push({ v: sumProjTotal.toFixed(1), l: 'Proj PPG' });
      if (state.type !== 'redraft') _sits.push({ v: sumT12, l: 'T1-2 Picks' });
      if (state.type === 'rookie'){
        if (sumAllPsCount >= 1) _sits.push({ v: Math.round(sumAllPsTotal / sumAllPsCount), l: 'Avg PS' });
      } else {
        if (sumStarterPsCount >= 2) _sits.push({ v: Math.round(sumStarterPsTotal / sumStarterPsCount), l: 'Starter PS' });
      }
      // Projected playoff odds for this team - only once the draft is complete.
      // Wait for the standings engine so the tile never flashes a different %.
      if (_draftComplete()){
        var _allT = gradeAllTeams(), _meT = null;
        for (var _ti = 0; _ti < _allT.length; _ti++){ if (_allT[_ti].isMe){ _meT = _allT[_ti]; break; } }
        if (_meT){
          if (playoffOddsPending(_allT)){
            _sits.push({ v: '…', l: 'Playoff Odds' });
          } else {
            var _myOdds = playoffOddsSource(_allT)[_meT.slot];
            if (_myOdds != null) _sits.push({ v: _myOdds + '%', l: 'Playoff Odds' });
          }
        }
      }
      if (_sits.length){
        statsHtml = '<div class="dr-sum-stats">';
        _sits.forEach(function(s){ statsHtml += '<div class="dr-sum-stat"><div class="dr-sum-stat-v">' + s.v + '</div><div class="dr-sum-stat-l">' + s.l + '</div></div>'; });
        statsHtml += '</div>';
      }
    }

    // Archetype + competitive window strip
    var archHtml = '';
    if (hasSlot){
      var _arch = teamArchetype(), _win = g && g.window;
      if (_arch || _win){
        var _aSections = [];
        // Combined descriptive profile, e.g. "Hero RB · Win-Now" (archetype +
        // competitive window). Purely descriptive of the draft's shape.
        var _profileLabel = _arch ? (_arch.label + (_win ? ' \xb7 ' + _win.label : ''))
                                  : (_win ? _win.label : '');
        if (_profileLabel){
          var _winCls = _win ? (' dr-win-' + _win.label.toLowerCase().replace('-','')) : '';
          _aSections.push('<div class="dr-sum-arch-item"><div class="dr-sum-arch-tag">Draft Profile</div><div class="dr-sum-arch-label' + _winCls + '">' + esc(_profileLabel) + '</div></div>');
        }
        if (_win){
          _aSections.push('<div class="dr-sum-arch-item"><div class="dr-sum-arch-tag">Avg Age</div><div class="dr-sum-arch-label" style="color:var(--text)">' + _win.avgAge.toFixed(1) + '</div></div>');
        }
        archHtml = '<div class="dr-sum-arch">' + _aSections.join('<div class="dr-sum-arch-div"></div>') + '</div>';
      }
    }

    // Player row builder
    function sumRow(slot, p){
      if (!p) return '<div class="dr-sum-row"><span class="dr-sum-slot-badge" style="background:' + slotColor(slot) + '">' + slot + '</span><span class="dr-sum-empty">open</span></div>';
      var _pn = (Object.keys(state.picks).filter(function(k){ return state.picks[k] && state.picks[k].id === p.id; }).map(function(k){ return parseInt(k,10); })[0]) || 0;
      var pickStr = _pn ? (function(){ var _rd = Math.ceil(_pn/state.teams); var _pp = _pn - (_rd-1)*state.teams; return 'Pick ' + _rd + '.' + (_pp < 10 ? '0'+_pp : String(_pp)); })() : '';
      var _rowps = relPS(p, _pn);
      var psStr = (_rowps != null) ? '<span class="dr-sum-ps" style="color:' + psColor(_rowps) + '">' + _rowps + '</span>' : '';
      return '<div class="dr-sum-row">'
        + '<span class="dr-sum-slot-badge" style="background:' + slotColor(slot) + '">' + slot + '</span>'
        + '<img class="dr-sum-hs" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div class="dr-sum-body"><div class="dr-sum-name">' + esc(p.name) + '</div>'
        + '<div class="dr-sum-meta">' + esc(p.position) + (p.team ? ' \xb7 ' + esc(p.team) : '') + (pickStr ? ' \xb7 ' + pickStr : '') + '</div>'
        + (p.reason ? '<div class="dr-sum-reason">' + esc(p.reason) + '</div>' : '')
        + '</div>' + psStr + '</div>';
    }

    // Starters + bench HTML
    var starterBenchHtml = '';
    if (hasSlot){
      starterBenchHtml = '<div class="dr-sum-section">Starters</div>';
      starters.forEach(function(s){ starterBenchHtml += sumRow(s.slot, s.p); });
      starterBenchHtml += '<div class="dr-sum-section">Bench</div>';
      if (bench.length){ bench.forEach(function(p){ starterBenchHtml += sumRow('BN', p); }); }
      else { starterBenchHtml += sumRow('BN', null); }
    }

    var html = '<button class="dr-prev-close" id="drSumClose" aria-label="Close">&times;</button>'
      + '<div class="dr-sum-header"><div class="dr-sum-title">Draft Report Card</div>' + gradeHtml + '</div>'
      + statsHtml + archHtml
      + '<div class="dr-sum-body-wrap">' + starterBenchHtml + '</div>'
      + '<div class="dr-sum-footer">'
      + '<button class="dr-btn dr-btn-primary" id="drSumDeepDive">Deep Dive'
      +   (cfg.hasPremium ? '' : ' <span class="dr-sum-prolock">PRO</span>') + '</button>'
      + (hasSlot ? '<button class="dr-btn" id="drSumShare">Share</button>' : '')
      + '<button class="dr-btn" id="drSumCloseBtn">Close</button>'
      + '</div>';

    var card = document.getElementById('drSummaryCard');
    card.innerHTML = html;
    document.getElementById('drSummary').style.display = '';
    document.getElementById('drSumClose').addEventListener('click', closeSummary);
    document.getElementById('drSumCloseBtn').addEventListener('click', closeSummary);
    var _ddBtn = document.getElementById('drSumDeepDive');
    if (_ddBtn) _ddBtn.addEventListener('click', function(){ closeSummary(); openDeepDive(); });
    var _shareBtn = document.getElementById('drSumShare');
    if (_shareBtn) _shareBtn.addEventListener('click', function(){ closeSummary(); shareDraft(); });
  }
  function closeSummary(){ document.getElementById('drSummary').style.display = 'none'; }

  // ── Deep Dive analyzer (Pro) ─────────────────────────────────────────────────
  // A richer post-draft view layered on the SAME grade/odds engine the report card
  // and the Teams page use, so every number here matches those surfaces exactly:
  //   • gradeAllTeams() -> overall + Value/Starters/Construction, and league ranks
  //   • playoffOddsSource() -> the standings engine's playoff odds per team
  //   • adpOf() with the app's pn - adp convention (positive = fell to you = value)
  // Gated behind cfg.hasPremium; available for every mock and live/synced draft.
  function ddGradeCol(s){ return s >= 75 ? '#22c55e' : s >= 60 ? '#38bdf8' : s >= 45 ? '#f59e0b' : '#ef4444'; }
  // Verdict from the market delta (pn - adp), with remaining-board BPA and
  // survival exemptions so leftover-ADP (best remaining at 11.0, pick 9) is
  // Fair rather than Reach. Shared kernel in DraftBoardCore.adpDeltaVerdict.
  function ddVerdict(p, useCons){
    var Core = window.DraftBoardCore;
    if (p == null || typeof p === 'number'){
      return Core && Core.adpDeltaVerdict
        ? Core.adpDeltaVerdict({ diff: p })
        : { label:'—', cls:'na' };
    }
    var diff = (useCons && p.consDiff != null) ? p.consDiff : p.diff;
    var bpa = useCons ? !!p.consIsBpa : !!p.isBpa;
    if (Core && Core.adpDeltaVerdict){
      return Core.adpDeltaVerdict({ diff: diff, isBpa: bpa, survivePct: p.survivePct });
    }
    if (diff == null) return { label:'—', cls:'na' };
    if (diff >= 8)  return { label:'Steal', cls:'steal' };
    if (diff >= 3)  return { label:'Value', cls:'value' };
    if (diff > -5)  return { label:'Fair',  cls:'fair'  };
    return { label:'Reach', cls:'reach' };
  }
  // Players taken before `pn` (plus keepers that never landed on a pick slot).
  function ddTakenBefore(pn){
    var taken = {};
    Object.keys(state.picks).forEach(function(k){
      if (parseInt(k, 10) < pn && state.picks[k]) taken[String(state.picks[k].id)] = true;
    });
    if (keepersOn && keeperSet && keeperSet.length){
      var onBoard = {};
      Object.keys(state.picks).forEach(function(k){
        if (state.picks[k]) onBoard[String(state.picks[k].id)] = true;
      });
      keeperSet.forEach(function(k){
        if (k && k.id != null && !onBoard[String(k.id)]) taken[String(k.id)] = true;
      });
    }
    return taken;
  }
  // Chance this player lasts to `nextPn` from ADP alone (no future-board
  // leakage from the live observedDraftModel). No later pick → 0 (can't wait).
  function ddSurvivePct(full, nextPn){
    if (nextPn == null) return 0;
    var a = adpOf(full);
    if (a == null) return null;
    if (window.DraftBoardCore && DraftBoardCore.availabilityProbability){
      return DraftBoardCore.availabilityProbability({
        center: a, pick: nextPn, sigma: simSigma(a),
        draftType: state.type, sf: !!state.sf
      });
    }
    return null;
  }
  // Signed ADP delta for display (pick number minus ADP). Keep full precision
  // on the raw `diff` for sorting; round here so the ledger doesn't print
  // IEEE leftovers like +82.0560271646859.
  function fmtAdpDelta(n){
    if (n == null || !isFinite(Number(n))) return '—';
    var s = Number(n).toFixed(1);
    if (Number(s) === 0) return '0.0';
    return (Number(s) > 0 ? '+' : '') + s;
  }
  // My picks in draft order, each carrying the market delta + pool-relative pick
  // score (relPS = the exact number the report-card rows show).
  function ddMyPicks(){
    var rows = [];
    if (!hasOwned()) return rows;
    var Core = window.DraftBoardCore;
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (!isMyPick(pn) || !state.picks[k]) return;
      var pl = state.picks[k];
      var full = playersById[String(pl.id)] || pl;
      var adp = adpOf(full);
      var diff = (adp != null) ? (pn - adp) : null;
      var consAdp = consensusAdpOf(full);
      var consDiff = (consAdp != null) ? (pn - consAdp) : null;
      var taken = ddTakenBefore(pn);
      var remPool = players;
      if (full && full.id != null && !playersById[String(full.id)]) remPool = players.concat([full]);
      var bestAdp = Core && Core.bestRemainingAdp ? Core.bestRemainingAdp(remPool, taken, adpOf) : adp;
      if (bestAdp == null && adp != null) bestAdp = adp;
      var consAdpFn = function(p){ var c = consensusAdpOf(p); return c != null ? c : adpOf(p); };
      var bestCons = Core && Core.bestRemainingAdp ? Core.bestRemainingAdp(remPool, taken, consAdpFn) : (consAdp != null ? consAdp : adp);
      if (bestCons == null && (consAdp != null || adp != null)) bestCons = consAdp != null ? consAdp : adp;
      var isBpa = Core && Core.isRemainingAdpBpa
        ? Core.isRemainingAdpBpa(adp, bestAdp)
        : (adp != null && bestAdp != null && adp <= bestAdp + 1);
      var consIsBpa = Core && Core.isRemainingAdpBpa
        ? Core.isRemainingAdpBpa(consAdp != null ? consAdp : adp, bestCons)
        : isBpa;
      var survivePct = ddSurvivePct(full, nextOwnedPickAfter(pn));
      var boardDiff = Core && Core.adpBoardDelta
        ? Core.adpBoardDelta({ diff: diff, isBpa: isBpa, survivePct: survivePct })
        : diff;
      var consBoardDiff = Core && Core.adpBoardDelta
        ? Core.adpBoardDelta({
            diff: consDiff != null ? consDiff : diff,
            isBpa: consIsBpa,
            survivePct: survivePct
          })
        : (consDiff != null ? consDiff : diff);
      rows.push({ pn: pn, pl: pl, full: full, pos: String(pl.position || '').toUpperCase(),
        adp: adp, diff: diff, consAdp: consAdp, consDiff: consDiff,
        isBpa: isBpa, consIsBpa: consIsBpa, survivePct: survivePct,
        boardDiff: boardDiff, consBoardDiff: consBoardDiff,
        ps: relPS(pl, pn), tier: tierOf(full) });
    });
    rows.sort(function(a, b){ return a.pn - b.pn; });
    return rows;
  }
  // League rank (1 = best) for a grade component across the whole field.
  // Tied values share a rank (1, 1, 3) so equal lineups don't look ordered.
  function ddRankBy(field, keyFn){
    var arr = field.map(function(t){ return { slot: t.slot, v: keyFn(t.grade) || 0 }; })
      .sort(function(a, b){ return b.v - a.v; });
    var rank = {}, lastV, lastRank = 0, seen = false;
    arr.forEach(function(x, i){
      if (!seen || x.v !== lastV){ lastRank = i + 1; lastV = x.v; seen = true; }
      rank[x.slot] = lastRank;
    });
    return rank;
  }
  function ddRankPill(rank, n){
    if (rank == null) return '';
    var cls = rank <= Math.ceil(n * 0.28) ? 'dd-rk-top' : rank <= Math.ceil(n * 0.62) ? 'dd-rk-mid' : 'dd-rk-low';
    return '<span class="dd-rankpill ' + cls + '">' + rank + ordinalSuffix(rank) + '</span>';
  }
  function ordinalSuffix(n){
    var t = n % 100; if (t >= 11 && t <= 13) return 'th';
    return { 1:'st', 2:'nd', 3:'rd' }[n % 10] || 'th';
  }

  // Positional rank of a drafted player by projected PPG among every player at that
  // position taken across the whole league (drives the "RB · 2nd of 41" chips).
  function ddPosRankIndex(){
    var byPos = {};
    Object.keys(state.picks).forEach(function(k){
      var pl = state.picks[k]; if (!pl) return;
      var pos = String(pl.position || '').toUpperCase();
      var full = playersById[String(pl.id)] || pl;
      (byPos[pos] = byPos[pos] || []).push({ id: String(pl.id), ppg: ppgOf(full) || 0, val: valOf(full) || 0 });
    });
    Object.keys(byPos).forEach(function(pos){
      byPos[pos].sort(function(a, b){ return (b.ppg - a.ppg) || (b.val - a.val); });
    });
    return byPos;
  }

  function openDeepDive(){
    var hasPicks = state && Object.keys(state.picks || {}).some(function(k){ return !!state.picks[k]; });
    if (!state || !hasPicks) return;
    if (!cfg.hasPremium){
      if (typeof window.showPaywall === 'function') window.showPaywall('draft-analyzer');
      return;
    }
    var field = gradeAllTeams();
    var n = field.length || 1;
    var me = null, myRank = null;
    for (var i = 0; i < field.length; i++){ if (field[i].isMe){ me = field[i]; myRank = i + 1; break; } }
    var odds = {};
    try { odds = playoffOddsSource(field) || {}; } catch (e){ odds = {}; }

    var picks = ddMyPicks();
    var withAdp = picks.filter(function(p){ return p.diff != null; });
    // Cap each pick's contribution so one late-round freefall doesn't dominate
    // the "net ADP value" tile (a +40 slide in round 14 ≠ forty early-round steals).
    // Use the remaining-board delta so leftover-ADP BPA picks don't look like
    // systematic reaches.
    var netValue = withAdp.reduce(function(s, p){
      var d = p.consBoardDiff != null ? p.consBoardDiff : p.boardDiff;
      if (d == null) d = p.diff;
      return s + Math.max(-12, Math.min(12, d));
    }, 0);
    netValue = Math.round(netValue * 10) / 10;
    var nValues = withAdp.filter(function(p){ return p.diff >= 3; }).length;
    var nReaches = withAdp.filter(function(p){ return ddVerdict(p).cls === 'reach'; }).length;

    var html = '<button class="dr-prev-close" id="drDdClose" aria-label="Close">&times;</button>';
    html += '<div class="dd-head"><div class="dd-kicker">Draft Report · Deep Dive'
      + '<span class="dd-pro">PRO</span></div>'
      + '<div class="dd-sub">' + (state.teams || 12) + '-team · ' + ddScoringLabel()
      + ' · ' + (state.rounds || myPicksList().length) + ' rounds · '
      + (state.mode === 'live' ? 'Connected league' : 'Mock draft') + '</div></div>';

    html += '<div class="dd-scroll">';
    html += ddOverviewHtml(me, myRank, n, field, netValue, nValues, nReaches);
    if (me) html += ddTimelineHtml(picks);
    if (me) html += ddLedgerHtml(picks);
    html += ddLeagueHtml(field, odds, n);
    if (me) html += ddConstructionHtml(me, field);
    if (me) html += ddEdgesHtml(picks, me);
    html += '</div>';
    html += '<div class="dd-foot"><button class="dr-btn" id="drDdCloseBtn">Close</button></div>';

    var card = document.getElementById('drDeepDiveCard');
    card.innerHTML = html;
    document.getElementById('drDeepDive').style.display = '';
    document.getElementById('drDdClose').addEventListener('click', closeDeepDive);
    document.getElementById('drDdCloseBtn').addEventListener('click', closeDeepDive);
    if (me) ddDrawTimeline(picks);
    if (me) ddWireLedger(picks);
  }
  function closeDeepDive(){
    var o = document.getElementById('drDeepDive'); if (o) o.style.display = 'none';
    var t = document.getElementById('drDdTip'); if (t) t.classList.remove('show');
  }
  function ddScoringLabel(){
    var sc = scoringCfg();
    if (state.type === 'rookie') return 'Rookie';
    if (state.sf) return 'Superflex';
    var ppr = sc && sc.ppr != null ? sc.ppr : 1;
    return ppr >= 1 ? 'PPR' : ppr > 0 ? 'Half-PPR' : 'Standard';
  }

  // ── Overview: grade ring, component meters w/ league rank, stat tiles ────────
  function ddOverviewHtml(me, myRank, n, field, netValue, nValues, nReaches){
    if (!me){
      return '<div class="dd-card dd-note">Set your pick slot to unlock the personalized breakdown. '
        + 'The league board and playoff odds below are available for every team.</div>';
    }
    var g = me.grade, col = ddGradeCol(g.score);
    var m = gradeMax();
    var vRank = ddRankBy(field, function(x){ return x.value; })[me.slot];
    // Rank starters by the vs-league ratio we display, not the 0-25 grade
    // slice. That 80–120% → 0–100 mapping puts a slightly-above-average
    // lineup at ~52/100, which reads like a C next to a "2nd" badge.
    var sRank = ddRankBy(field, function(x){ return x.strength != null ? x.strength : x.tier; })[me.slot];
    var cRank = ddRankBy(field, function(x){ return x.balance; })[me.slot];
    function meter(lbl, sub, val, max, rank, opts){
      opts = opts || {};
      var pct = max ? Math.round(val / max * 100) : 0;
      var fill = Math.max(0, Math.min(100, pct));
      var c = fill >= 80 ? '#22c55e' : fill >= 60 ? '#38bdf8' : fill >= 40 ? '#f59e0b' : '#ef4444';
      var shown = opts.shown != null ? opts.shown : pct;
      var unit = opts.unit != null ? opts.unit : '/100';
      // vsAvg: `val` is already % of a league-average lineup (100 = avg).
      // Keep the bar's 80–120 mapping so average sits at the midpoint, but
      // color around 100 so "slightly above average, 2nd" doesn't render orange.
      if (opts.vsAvg){
        fill = Math.max(0, Math.min(100, Math.round((val - 80) / 0.40)));
        c = val >= 108 ? '#22c55e' : val >= 100 ? '#38bdf8' : val >= 94 ? '#f59e0b' : '#ef4444';
        shown = val;
      }
      return '<div class="dd-meter"><div class="dd-meter-lab">' + lbl + '<small>' + sub + '</small></div>'
        + '<div class="dd-track"><i style="width:' + fill + '%;background:' + c + '"></i></div>'
        + '<div class="dd-meter-val">' + shown + '<span>' + unit + '</span> ' + ddRankPill(rank, n) + '</div></div>';
    }
    var starterPct = (g.strength != null) ? g.strength
      : (m.tier ? Math.round(80 + (g.tier / m.tier) * 40) : 100);
    var arch = null; try { arch = teamArchetype(); } catch (e){ arch = null; }
    var verdict = ddOverviewVerdict(g, myRank, n, netValue, arch);
    var meters = (state.type === 'rookie')
      ? meter('Avg Pick Score', 'BPA / ADP letter system', g.value, 100, vRank)
      : meter('Value', 'round-weighted pick score', g.value, m.value, vRank)
        + meter('Starters', '100% = league-average lineup', starterPct, 100, sRank, { unit: '% of avg', vsAvg: true })
        + meter('Construction', 'slot coverage & balance', g.balance, m.balance, cRank);

    var tiles = '';
    var tileDefs = [
      { v: fmtAdpDelta(netValue), l: 'Net ADP value (capped)', cls: netValue >= 0 ? 'good' : 'bad' },
      { v: nValues, l: 'Values (fell 3+ to you)', cls: 'good' },
      { v: nReaches, l: 'Reaches (early 5+, could wait)', cls: nReaches ? 'bad' : '' },
      { v: g.avgPs != null ? g.avgPs : '—', l: 'Avg pick score' }
    ];
    tileDefs.forEach(function(t){
      tiles += '<div class="dd-tile ' + (t.cls || '') + '"><div class="dd-tile-v">' + t.v + '</div><div class="dd-tile-l">' + t.l + '</div></div>';
    });

    return '<div class="dd-card dd-overview">'
      + '<div class="dd-ov-top">'
      + '<div class="dd-ring" style="--pct:' + Math.max(0, Math.min(100, Math.round(g.score))) + ';--gc:' + col + '">'
      + '<b style="color:' + col + '">' + gradeLetter(g.score) + '<small>' + Math.round(g.score)
      + (g.provisional ? ' · Early' : '') + '</small></b></div>'
      + '<div class="dd-ov-txt"><h3>' + verdict.title + '</h3>'
      + '<div class="dd-rankline">Ranked <b>' + myRank + ordinalSuffix(myRank) + ' of ' + n + '</b>'
      + (arch ? ' · ' + esc(arch.label) : '') + '</div>'
      + '<div class="dd-say">' + verdict.say + '</div></div>'
      + '<div class="dd-meters">' + meters + '</div>'
      + '</div>'
      + '<div class="dd-tiles">' + tiles + '</div>'
      + '</div>';
  }
  function ddOverviewVerdict(g, myRank, n, netValue, arch){
    var strong = [], weak = [];
    var m = gradeMax();
    // Rookie grades are value-only (no starters/construction component), so only
    // score the components that carry weight for this draft type.
    var comps = (state.type === 'rookie')
      ? [{ k: 'pick value', pct: (m.value ? g.value / m.value : 0) }]
      : [
          { k: 'value tier', pct: (m.value ? g.value / m.value : 0) },
          { k: 'starting lineup', pct: (m.tier ? g.tier / m.tier : 0) },
          { k: 'roster construction', pct: (m.balance ? g.balance / m.balance : 0) }
        ];
    comps.forEach(function(c){ if (c.pct >= 0.72) strong.push(c.k); else if (c.pct <= 0.5) weak.push(c.k); });
    var title = g.score >= 80 ? 'Elite draft' : g.score >= 70 ? 'Strong, well-rounded board'
      : g.score >= 58 ? 'Solid with a soft spot' : g.score >= 45 ? 'Playable but uneven' : 'Rebuild from the wire';
    var parts = [];
    parts.push('You banked <b>' + fmtAdpDelta(netValue) + ' picks of ADP value</b>');
    if (strong.length) parts.push('your <b>' + strong[0] + '</b> is a league strength');
    if (weak.length) parts.push('but <b>' + weak[0] + '</b> is where you can lose');
    else parts.push('with no glaring hole');
    return { title: title, say: parts.join(', ') + '.' };
  }

  // ── Value-vs-ADP timeline (SVG) ──────────────────────────────────────────────
  // Plot against consensus ADP when the payload has it; otherwise the selected
  // source (adpOf). The heading subscript only appears when consensus is in use.
  function ddTlDelta(p){
    if (p.consBoardDiff != null) return p.consBoardDiff;
    if (p.boardDiff != null) return p.boardDiff;
    return p.consDiff != null ? p.consDiff : p.diff;
  }
  function ddTlAdp(p){ return p.consAdp != null ? p.consAdp : p.adp; }
  function ddTimelineHtml(picks){
    var hasCons = picks.some(function(p){ return p.consAdp != null; });
    var sub = hasCons ? '<small class="dd-h-sub">Consensus ADP</small>' : '';
    var blurb = hasCons
      ? 'Each pick against consensus ADP. Above the line it fell to you (value). Best remaining ADP and players under 20% to last to your next pick sit on the line even if historical ADP is a couple of spots later; below is a reach past someone who was likely to last.'
      : 'Each pick against where the market had it. Above the line it fell to you (value). Best remaining ADP and players under 20% to last to your next pick sit on the line even if historical ADP is a couple of spots later; below is a reach past someone who was likely to last.';
    return '<div class="dd-card">'
      + '<div class="dd-sec"><h4>Value vs ADP timeline' + sub + '</h4>'
      + '<p>' + blurb + '</p></div>'
      + '<div class="dd-legend">'
      + ['QB','RB','WR','TE'].map(function(p){ return '<span><i class="dd-dot" style="background:' + posColor(p) + '"></i>' + p + '</span>'; }).join('')
      + '<span style="margin-left:auto"><i class="dd-sq" style="background:color-mix(in srgb,#22c55e 22%,transparent);border:1px solid #22c55e"></i>value</span>'
      + '<span><i class="dd-sq" style="background:color-mix(in srgb,#94a3b8 22%,transparent);border:1px solid #94a3b8"></i>on board</span>'
      + '<span><i class="dd-sq" style="background:color-mix(in srgb,#ef4444 22%,transparent);border:1px solid #ef4444"></i>reach</span>'
      + '</div>'
      + '<div class="dd-chartscroll"><svg id="drDdTl" width="900" height="340" viewBox="0 0 900 340" role="img" aria-label="Value versus consensus ADP by pick"></svg></div>'
      + '</div>';
  }
  function ddDrawTimeline(picks){
    var svg = document.getElementById('drDdTl'); if (!svg) return;
    var pts = picks.filter(function(p){ return ddTlDelta(p) != null; });
    if (!pts.length){ svg.parentNode.parentNode.style.display = 'none'; return; }
    var NS = 'http://www.w3.org/2000/svg';
    function el(nm, a){ var e = document.createElementNS(NS, nm); for (var k in a) e.setAttribute(k, a[k]); return e; }
    var W = 900, H = 340, mr = { l: 42, r: 14, t: 16, b: 30 };
    var iw = W - mr.l - mr.r, ih = H - mr.t - mr.b;
    var maxD = 2, minD = -2;
    pts.forEach(function(p){ var d = ddTlDelta(p); if (d > maxD) maxD = d; if (d < minD) minD = d; });
    maxD = Math.ceil(maxD / 5) * 5 + 2; minD = Math.floor(minD / 5) * 5 - 2;
    var x = function(i){ return mr.l + (pts.length === 1 ? iw / 2 : (i / (pts.length - 1)) * iw); };
    var y = function(d){ return mr.t + (maxD - d) / (maxD - minD) * ih; };
    var y0 = y(0);
    svg.appendChild(el('rect', { x: mr.l, y: mr.t, width: iw, height: y0 - mr.t, fill: 'color-mix(in srgb,#22c55e 7%,transparent)' }));
    svg.appendChild(el('rect', { x: mr.l, y: y0, width: iw, height: mr.t + ih - y0, fill: 'color-mix(in srgb,#ef4444 7%,transparent)' }));
    var step = (maxD - minD) > 40 ? 15 : (maxD - minD) > 20 ? 10 : 5;
    for (var d = Math.ceil(minD / step) * step; d <= maxD; d += step){
      svg.appendChild(el('line', { x1: mr.l, y1: y(d), x2: mr.l + iw, y2: y(d), stroke: 'var(--border)', 'stroke-width': d === 0 ? 1.4 : 1, opacity: d === 0 ? 1 : 0.6 }));
      var tx = el('text', { x: mr.l - 7, y: y(d) + 4, 'text-anchor': 'end', 'font-size': 10.5, fill: 'var(--text-muted)' });
      tx.textContent = (d > 0 ? '+' : '') + d; svg.appendChild(tx);
    }
    // cumulative value line
    var cum = 0, cmax = 1; var cpts = [];
    pts.forEach(function(p){ cum += ddTlDelta(p); cpts.push(cum); if (Math.abs(cum) > cmax) cmax = Math.abs(cum); });
    var cy = function(v){ return mr.t + ih / 2 - (v / cmax) * (ih / 2) * 0.9; };
    var dpath = cpts.map(function(v, i){ return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + cy(v).toFixed(1); }).join(' ');
    svg.appendChild(el('path', { d: dpath, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3', opacity: 0.55 }));
    pts.forEach(function(p, i){
      var px = x(i), py = y(ddTlDelta(p)), c = posColor(p.pos);
      svg.appendChild(el('line', { x1: px, y1: y0, x2: px, y2: py, stroke: c, 'stroke-width': 1.3, opacity: 0.32 }));
      var r = p.ps == null ? 5 : Math.max(4, 4 + (p.ps - 40) / 60 * 6);
      var dot = el('circle', { cx: px, cy: py, r: r, fill: c, 'fill-opacity': 0.9, stroke: 'var(--card)', 'stroke-width': 1.5, class: 'dd-tl-dot', style: 'cursor:pointer' });
      dot.addEventListener('mousemove', function(ev){ ddTip(ev, p); });
      dot.addEventListener('mouseleave', ddTipHide);
      svg.appendChild(dot);
      var rl = el('text', { x: px, y: H - mr.b + 17, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--text-subtle,var(--text-muted))' });
      rl.textContent = roundPickStr(p.pn); svg.appendChild(rl);
    });
  }
  function ddTip(ev, p){
    var tip = document.getElementById('drDdTip');
    if (!tip){ tip = document.createElement('div'); tip.id = 'drDdTip'; tip.className = 'dd-tip'; document.body.appendChild(tip); }
    var dlt = p.consDiff != null ? p.consDiff : p.diff, adp = ddTlAdp(p);
    var vd = ddVerdict(p, true);
    var adpLbl = p.consAdp != null ? 'Consensus ADP' : 'ADP';
    var why = '';
    if (vd.cls !== 'reach' && dlt != null && dlt < 0){
      if (p.consIsBpa || p.isBpa) why = 'Best remaining ADP at the pick.';
      else if (p.survivePct != null && p.survivePct < 20) why = 'Under 20% to last to your next pick.';
    }
    tip.innerHTML = '<b>' + esc(p.pl.name) + '</b> <span style="color:var(--text-muted)">' + p.pos + (p.pl.team ? ' · ' + esc(p.pl.team) : '') + '</span>'
      + '<div class="dd-tip-r">Pick <b>' + roundPickStr(p.pn) + '</b></div>'
      + (adp != null ? '<div class="dd-tip-r">' + adpLbl + ' <b>' + Number(adp).toFixed(1) + '</b></div>' : '')
      + (dlt != null ? '<div class="dd-tip-r">± vs ADP <b style="color:' + (dlt >= 0 ? '#22c55e' : '#ef4444') + '">' + fmtAdpDelta(dlt) + '</b></div>' : '')
      + (p.ps != null ? '<div class="dd-tip-r">Board PS <b style="color:' + psColor(p.ps) + '">' + p.ps + '</b> <span style="color:var(--text-muted)">(vs best avail)</span></div>' : '')
      + '<div class="dd-tip-r">Verdict <b>' + vd.label + '</b></div>'
      + (why ? '<div class="dd-tip-r" style="color:var(--text-muted)">' + why + '</div>' : '');
    tip.classList.add('show');
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var lx = ev.clientX + 14, ty = ev.clientY - th - 8;
    if (lx + tw > window.innerWidth - 8) lx = ev.clientX - tw - 14;
    if (ty < 8) ty = ev.clientY + 16;
    tip.style.left = lx + 'px'; tip.style.top = ty + 'px';
  }
  function ddTipHide(){ var t = document.getElementById('drDdTip'); if (t) t.classList.remove('show'); }

  // ── Pick ledger (sortable) ───────────────────────────────────────────────────
  function ddLedgerHtml(picks){
    return '<div class="dd-card">'
      + '<div class="dd-sec"><h4>Pick ledger</h4><p>Every selection with market delta, board pick score (vs best available then), tier, and verdict. Reach means you skipped a better remaining ADP and the player was likely to last to your next pick. Click a header to sort.</p></div>'
      + '<div class="dd-tablescroll"><table class="dd-ledger" id="drDdLedger">'
      + '<thead><tr>'
      + '<th data-k="pn" data-t="n">Pick</th><th data-k="name" data-t="s">Player</th><th data-k="pos" data-t="s">Pos</th>'
      + '<th data-k="adp" data-t="n" class="r">ADP</th><th data-k="diff" data-t="n" class="r dd-sorted">± ADP</th>'
      + '<th data-k="ps" data-t="n" class="r" title="Board pick score vs best available at that slot">Board PS</th><th data-k="tier" data-t="n" class="r">Tier</th>'
      + '<th data-k="vord" data-t="s">Verdict</th>'
      + '</tr></thead><tbody id="drDdLedgerBody"></tbody></table></div></div>';
  }
  function ddLedgerRows(list){
    return list.map(function(p){
      var vd = ddVerdict(p);
      var dcl = p.diff == null ? 'z' : p.diff > 0 ? 'p' : p.diff < 0 ? 'n' : 'z';
      var dtxt = fmtAdpDelta(p.diff);
      return '<tr>'
        + '<td class="num" style="color:var(--text-muted)">' + roundPickStr(p.pn) + '</td>'
        + '<td class="dd-plname">' + esc(p.pl.name) + ' <span style="color:var(--text-subtle,var(--text-muted));font-size:11px">' + esc(p.pl.team || '') + '</span></td>'
        + '<td><span class="dd-posbadge" style="background:' + posColor(p.pos) + '">' + p.pos + '</span></td>'
        + '<td class="r num">' + (p.adp != null ? Number(p.adp).toFixed(1) : '—') + '</td>'
        + '<td class="r num"><span class="dd-diff ' + dcl + '">' + dtxt + '</span></td>'
        + '<td class="r">' + (p.ps != null ? '<span class="num" style="font-weight:700;color:' + psColor(p.ps) + '">' + p.ps + '</span>' : '<span style="color:var(--text-subtle,var(--text-muted))">—</span>') + '</td>'
        + '<td class="r"><span style="color:var(--text-muted);font-size:12px">' + (p.tier != null ? 'T' + p.tier : '—') + '</span></td>'
        + '<td><span class="dd-verd dd-v-' + vd.cls + '">' + vd.label + '</span></td>'
        + '</tr>';
    }).join('');
  }
  function ddWireLedger(picks){
    var body = document.getElementById('drDdLedgerBody'); if (!body) return;
    var st = { k: 'diff', dir: -1 };
    body.innerHTML = ddLedgerRows(picks.slice().sort(function(a, b){ return (b.diff == null ? -999 : b.diff) - (a.diff == null ? -999 : a.diff); }));
    var ths = document.querySelectorAll('#drDdLedger thead th');
    ths.forEach(function(th){
      th.addEventListener('click', function(){
        var k = th.getAttribute('data-k'), t = th.getAttribute('data-t');
        st.dir = (st.k === k) ? -st.dir : (t === 'n' ? -1 : 1); st.k = k;
        var list = picks.slice().sort(function(a, b){
          var av, bv;
          if (k === 'vord'){ av = ddVerdict(a).label; bv = ddVerdict(b).label; return st.dir * String(av).localeCompare(String(bv)); }
          if (k === 'name'){ return st.dir * String(a.pl.name).localeCompare(String(b.pl.name)); }
          if (k === 'pos'){ return st.dir * String(a.pos).localeCompare(String(b.pos)); }
          av = a[k] == null ? -999 : a[k]; bv = b[k] == null ? -999 : b[k];
          return st.dir * (av - bv);
        });
        body.innerHTML = ddLedgerRows(list);
        ths.forEach(function(o){ o.classList.toggle('dd-sorted', o === th); });
      });
    });
  }

  // ── League board: grades + playoff odds for every team ───────────────────────
  function ddLeagueHtml(field, odds, n){
    if (!field.length) return '';
    var pending = playoffOddsPending(field);
    var rows = field.map(function(t, i){
      var col = ddGradeCol(t.grade.score);
      var od = (!pending && odds && odds[t.slot] != null) ? odds[t.slot] : null;
      var odBar = pending
        ? '<span class="dd-odds-pending">Calculating…</span>'
        : (od != null
          ? '<div class="dd-odds"><div class="dd-odds-track"><i style="width:' + Math.max(2, od) + '%;background:' + (od >= 60 ? '#22c55e' : od >= 35 ? '#38bdf8' : '#f59e0b') + '"></i></div><span class="num">' + od + '%</span></div>'
          : '<span style="color:var(--text-subtle,var(--text-muted));font-size:12px">—</span>');
      return '<tr class="' + (t.isMe ? 'dd-me' : '') + '">'
        + '<td class="num" style="color:var(--text-muted)">' + (i + 1) + '</td>'
        + '<td class="dd-plname">' + esc(t.name) + (t.isMe ? ' <span class="dd-youtag">YOU</span>' : '') + '</td>'
        + '<td class="r"><span class="dd-gletter" style="color:' + col + '">' + gradeLetter(t.grade.score)
        + (t.grade.provisional ? '<span class="dr-grade-early-inline"> Early</span>' : '') + '</span></td>'
        + '<td class="r num" style="color:var(--text-muted)">' + Math.round(t.grade.score) + '</td>'
        + '<td>' + odBar + '</td>'
        + '</tr>';
    }).join('');
    var note = !_draftComplete()
      ? 'Live estimate — odds sharpen to the full simulation once the draft completes.'
      : (pending
        ? 'Running the standings simulation engine…'
        : 'Playoff odds from the standings simulation engine (preseason mode).');
    return '<div class="dd-card">'
      + '<div class="dd-sec"><h4>League board &amp; playoff odds</h4><p>' + note + '</p></div>'
      + '<div class="dd-tablescroll"><table class="dd-ledger dd-league">'
      + '<thead><tr><th>#</th><th>Team</th><th class="r">Grade</th><th class="r">Score</th><th>Playoff odds</th></tr></thead>'
      + '<tbody>' + rows + '</tbody></table></div></div>';
  }

  // ── Construction: draft capital by position + starters vs league ─────────────
  function ddConstructionHtml(me, field){
    var POSes = ['QB','RB','WR','TE'];
    var myByPos = { QB:0, RB:0, WR:0, TE:0 }, myTot = 0;
    var lgByPos = { QB:0, RB:0, WR:0, TE:0 }, lgTot = 0;
    var myCount = { QB:0, RB:0, WR:0, TE:0 }, myN = 0;
    var lgCount = { QB:0, RB:0, WR:0, TE:0 }, lgN = 0;
    field.forEach(function(t){
      (t.picks || []).forEach(function(x){
        var pl = (x && x.p) ? x.p : x;
        if (!pl) return;
        var full = playersById[String(pl.id)] || pl;
        var pos = String(full.position || pl.position || '').toUpperCase();
        if (pos === 'DST' || pos === 'D/ST') pos = 'DEF';
        if (lgByPos[pos] == null) return;
        // Always add a finite number. String values (e.g. "184.2" from JSON)
        // used to concatenate via += and turn every share into NaN%.
        var v = valOf(full);
        if (!v) v = finiteVal(pl.val);
        lgByPos[pos] += v; lgTot += v;
        lgCount[pos]++; lgN++;
        if (t.isMe){ myByPos[pos] += v; myTot += v; myCount[pos]++; myN++; }
      });
    });
    // No resolvable trade value (common for K/DEF-heavy or unresolved live ids):
    // share by pick count so the bars still show how the draft was spent.
    if (!lgTot){
      lgByPos = lgCount; lgTot = lgN;
      myByPos = myCount; myTot = myN;
    }
    function capPct(part, tot){
      if (!tot) return 0;
      var n = Math.round(part / tot * 100);
      return isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
    }
    var capBars = POSes.map(function(pos){
      var mine = capPct(myByPos[pos], myTot);
      var lg = capPct(lgByPos[pos], lgTot);
      return '<div class="dd-cap-row"><div class="dd-cap-pos" style="color:' + posColor(pos) + '">' + pos + '</div>'
        + '<div class="dd-cap-track"><i style="width:' + mine + '%;background:' + posColor(pos) + '"></i>'
        + '<span class="dd-cap-lg" style="left:' + lg + '%" title="league avg ' + lg + '%"></span></div>'
        + '<div class="dd-cap-val num">' + mine + '%<small>lg ' + lg + '%</small></div></div>';
    }).join('');

    // Starters vs league: per-starter positional rank + team strength ratio.
    var mine = myPicksList().slice();
    var ol = optimalLineup(mine);
    var posIdx = ddPosRankIndex();
    var starterRows = ol.starters.filter(function(s){ return s.p && s.slot !== 'K' && s.slot !== 'DEF'; }).map(function(s){
      var full = playersById[String(s.p.id)] || s.p;
      var pos = String(s.p.position || '').toUpperCase();
      var ppg = ppgOf(full);
      var list = posIdx[pos] || [];
      var rank = 0; for (var i = 0; i < list.length; i++){ if (list[i].id === String(s.p.id)){ rank = i + 1; break; } }
      return '<div class="dd-st-row">'
        + '<span class="dd-slotbadge" style="background:color-mix(in srgb,' + slotColor(s.slot) + ' 16%,var(--card));border-color:color-mix(in srgb,' + slotColor(s.slot) + ' 40%,var(--border));color:' + slotColor(s.slot) + '">' + s.slot + '</span>'
        + '<span class="dd-st-name">' + esc(s.p.name) + '</span>'
        + '<span class="dd-st-ppg num">' + (ppg != null ? ppg.toFixed(1) : '—') + '<small>ppg</small></span>'
        + '<span class="dd-st-rank">' + (rank ? pos + ' <b>' + rank + ordinalSuffix(rank) + '</b> of ' + list.length : '') + '</span>'
        + '</div>';
    }).join('');
    var strength = (me.grade.strength != null) ? me.grade.strength : null;

    return '<div class="dd-card"><div class="dd-two">'
      + '<div><div class="dd-sec"><h4>Draft capital</h4><p>Share of your value spent per position vs the league average (tick).</p></div>' + capBars + '</div>'
      + '<div><div class="dd-sec"><h4>Starters vs league</h4><p>'
      + (strength != null ? 'Your starters project <b>' + strength + '%</b> of a league-average lineup.' : 'Your starting lineup, ranked by position.')
      + '</p></div>' + starterRows + '</div>'
      + '</div></div>';
  }

  // ── Edges & risks ────────────────────────────────────────────────────────────
  function ddEdgesHtml(picks, me){
    var withAdp = picks.filter(function(p){ return p.diff != null; });
    var edges = '';
    if (withAdp.length){
      var steal = withAdp.slice().sort(function(a, b){ return b.diff - a.diff; })[0];
      var reach = withAdp.filter(function(p){ return ddVerdict(p).cls === 'reach'; })
        .sort(function(a, b){ return a.diff - b.diff; })[0];
      var best = picks.filter(function(p){ return p.ps != null; }).sort(function(a, b){ return b.ps - a.ps; })[0];
      function edge(kind, cls, p, extra){
        return '<div class="dd-edge ' + cls + '"><div class="dd-edge-k">' + kind + '</div>'
          + '<div class="dd-edge-pl">' + esc(p.pl.name) + '</div>'
          + '<div class="dd-edge-sub">' + p.pos + ' · ' + roundPickStr(p.pn) + (p.adp != null ? ' · ADP ' + Number(p.adp).toFixed(0) : '') + '</div>'
          + '<div class="dd-edge-say">' + extra + '</div></div>';
      }
      var parts = [];
      // Only label Steal/Reach when the market delta clears the same thresholds
      // the ledger uses — otherwise a "Fair" pick was being sold as an edge.
      if (steal && steal.diff >= 3){
        parts.push(edge('Biggest steal', 'win', steal, 'Fell <b>' + Math.abs(steal.diff).toFixed(1) + '</b> picks past ADP' + (steal.ps != null ? ' — a ' + steal.ps + ' pick score.' : '.')));
      }
      if (best && (!steal || best !== steal || steal.diff < 3)){
        parts.push(edge('Best pick', 'winb', best, 'Your highest pick score at <b>' + best.ps + '</b>.'));
      }
      if (reach){
        parts.push(edge('Biggest reach', 'bad', reach, 'Taken <b>' + Math.abs(reach.diff).toFixed(1) + '</b> picks before ADP with a better option still on the board.'));
      }
      if (parts.length) edges = '<div class="dd-edges">' + parts.join('') + '</div>';
    }
    // Risk flags
    var flags = [];
    if (state.type === 'redraft'){
      var byeMap = {};
      picks.forEach(function(p){
        var bw = p.full && p.full.bye_week ? Number(p.full.bye_week) : null;
        if (bw){ (byeMap[bw] = byeMap[bw] || []).push(p.pl.name); }
      });
      var worst = null;
      Object.keys(byeMap).forEach(function(w){ if (!worst || byeMap[w].length > byeMap[worst].length) worst = w; });
      if (worst && byeMap[worst].length >= 3){
        flags.push({ cls: 'crit', ttl: 'Week ' + worst + ' bye cluster — ' + byeMap[worst].length + ' players out',
          ds: byeMap[worst].join(', ') + ' all sit on the same week. Plan a stopgap before then.' });
      }
    }
    // Thin position: rostered count at or below starter demand, including FLEX/SF
    // shares so a 2-RB + FLEX roster with only 2 RBs still flags as thin.
    var counts = { QB:0, RB:0, WR:0, TE:0 };
    picks.forEach(function(p){ if (counts[p.pos] != null) counts[p.pos]++; });
    var rs = (state && state.roster) || defaultRoster();
    var flex = rs.FLEX || 0, sf = rs.SF || 0;
    var starterNeed = {
      QB: (rs.QB || 0) + sf,
      RB: (rs.RB || 0) + Math.ceil(flex * 0.5),
      WR: (rs.WR || 0) + Math.floor(flex * 0.5),
      TE: (rs.TE || 0)
    };
    ['RB','WR','TE','QB'].forEach(function(pos){
      var need = starterNeed[pos] || 0;
      if (need > 0 && counts[pos] <= need){
        flags.push({ cls: 'warn', ttl: 'Thin at ' + pos + ' — ' + counts[pos] + ' rostered',
          ds: 'You have no margin behind your ' + pos + ' starters. Prioritize depth on the waiver wire.' });
      }
    });
    var flagsHtml = flags.length ? '<div class="dd-flags">' + flags.map(function(f){
      return '<div class="dd-flag dd-flag-' + f.cls + '"><div class="dd-flag-ic">' + (f.cls === 'crit' ? '!' : '▾') + '</div>'
        + '<div><div class="dd-flag-ttl">' + f.ttl + '</div><div class="dd-flag-ds">' + esc(f.ds) + '</div></div></div>';
    }).join('') + '</div>' : '';

    if (!edges && !flagsHtml) return '';
    return '<div class="dd-card"><div class="dd-sec"><h4>Edges &amp; risks</h4>'
      + '<p>The picks that define your team and the exposures to plan for.</p></div>' + edges + flagsHtml + '</div>';
  }

  // ── Custom modal (replaces native confirm/alert) ─────────────────────────────
  function drAlert(msg, cb){
    var m = document.getElementById('drModal');
    document.getElementById('drModalMsg').textContent = msg;
    var btns = document.getElementById('drModalBtns');
    btns.innerHTML = '';
    var ok = document.createElement('button');
    ok.className = 'dr-btn dr-btn-primary'; ok.textContent = 'OK';
    ok.addEventListener('click', function(){ m.style.display = 'none'; if (cb) cb(); });
    btns.appendChild(ok);
    m.style.display = 'flex';
  }
  // ── In-draft pick trade evaluator ───────────────────────────────────────────
  // Values an overall pick number as the player likely on the board there: the
  // k-th best remaining player by ADP, where k is how many picks away it is.
  // Board-derived, so it prices THIS draft (a thin board makes late picks cheap).
  function pickNumValue(pn){
    var cur = state.current || 1;
    var k = Math.max(1, Math.round(pn) - cur + 1);
    var pool = availablePool().slice().sort(function(a, b){
      var aa = adpOf(a), ab = adpOf(b);
      if (aa == null && ab == null) return valOf(b) - valOf(a);
      if (aa == null) return 1;
      if (ab == null) return -1;
      return aa - ab;
    });
    if (!pool.length) return null;
    var p = pool[Math.min(k, pool.length) - 1];
    return { value: Math.max(0, Math.round(valOf(p))), proxy: p, pos: (p.position || '').toUpperCase() };
  }
  // Round.pick label for an overall pick number (e.g. 22 -> "2.10").
  function pickLabel(pn){
    var teams = state.teams || 12;
    var r = Math.floor((pn - 1) / teams) + 1, p = (pn - 1) % teams + 1;
    return r + '.' + (p < 10 ? '0' : '') + p;
  }
  function _parsePickNums(s){
    // Accepts overall pick numbers ("22") and round.pick ("2.10"), separated
    // by commas, spaces, or slashes - the iOS numeric keypad has no comma, so
    // any reasonable separator must work.
    var teams = state.teams || 12;
    return String(s || '').split(/[,\s\/;]+/)
      .map(function(t){
        t = t.trim();
        if (!t) return NaN;
        var m = t.match(/^(\d{1,2})\.(\d{1,2})$/);
        if (m){
          var r = parseInt(m[1], 10), p = parseInt(m[2], 10);
          if (r >= 1 && p >= 1 && p <= teams) return (r - 1) * teams + p;
          return NaN;
        }
        return parseInt(t, 10);
      })
      .filter(function(n){ return n >= 1 && n <= 600; });
  }
  function drPickTradeOpen(){
    var m = document.getElementById('drModal');
    var msg = document.getElementById('drModalMsg');
    var teams = state.teams || 12;
    var rounds = Math.max(1, state.rounds || 15);
    var give = [], get = [];   // overall pick numbers, built via the selectors
    var mine = upcomingOwnedPicks();

    function pickerHtml(id){
      var ro = ''; for (var r = 1; r <= rounds; r++) ro += '<option value="' + r + '">Round ' + r + '</option>';
      var po = ''; for (var p = 1; p <= teams; p++) po += '<option value="' + p + '">Pick ' + p + '</option>';
      return '<div class="dr-pt-picker">'
        + '<select class="dr-pt-sel" id="' + id + 'Rd">' + ro + '</select>'
        + '<select class="dr-pt-sel" id="' + id + 'Pk">' + po + '</select>'
        + '<button type="button" class="dr-btn dr-btn-primary dr-pt-add" data-target="' + id + '">Add</button>'
        + '</div>';
    }
    var quick = mine.length
      ? '<div class="dr-pt-chips"><span class="dr-pt-chips-lbl">Quick add your picks</span>'
        + mine.slice(0, 14).map(function(pn){ return '<button type="button" class="dr-pt-chip" data-pn="' + pn + '">' + pickLabel(pn) + '</button>'; }).join('')
        + '</div>'
      : '';
    msg.innerHTML = '<div class="dr-pt-title">Pick trade evaluator</div>'
      + '<div class="dr-pt-sub">Pick a round and pick, then Add. Each pick is priced as the player likely on the board there, by ADP on this draft’s remaining pool.</div>'
      + '<label class="dr-pt-lbl">You give</label>'
      + '<div class="dr-pt-chiprow" id="drPtGiveChips"></div>'
      + pickerHtml('drPtG')
      + quick
      + '<label class="dr-pt-lbl">You get</label>'
      + '<div class="dr-pt-chiprow" id="drPtGetChips"></div>'
      + pickerHtml('drPtR')
      + '<div id="drPtResult" class="dr-pt-result"></div>';
    var btns = document.getElementById('drModalBtns');
    btns.innerHTML = '';
    var close = document.createElement('button');
    close.className = 'dr-btn'; close.textContent = 'Close';
    close.addEventListener('click', function(){ m.style.display = 'none'; });
    btns.appendChild(close);

    function renderChips(){
      [['give', 'drPtGiveChips'], ['get', 'drPtGetChips']].forEach(function(pair){
        var list = pair[0] === 'give' ? give : get;
        var el = document.getElementById(pair[1]);
        el.innerHTML = list.length
          ? list.map(function(pn, i){ return '<span class="dr-pt-tok">' + pickLabel(pn) + '<button type="button" class="dr-pt-tokx" data-side="' + pair[0] + '" data-i="' + i + '" aria-label="Remove">&times;</button></span>'; }).join('')
          : '<span class="dr-pt-empty">No picks yet</span>';
      });
    }
    function sideRows(list){
      var tot = 0;
      var rows = list.map(function(pn){
        var v = pickNumValue(pn), lbl = pickLabel(pn);
        if (!v) return '<div class="dr-pt-row"><span class="dr-pt-pk">' + lbl + '</span><span class="dr-pt-nm dr-pt-empty">board empty</span></div>';
        tot += v.value;
        return '<div class="dr-pt-row"><span class="dr-pt-pk">' + lbl + '</span>'
          + (v.pos ? '<span class="dr-pt-pos dr-pt-pos-' + v.pos + '">' + esc(v.pos) + '</span>' : '')
          + '<span class="dr-pt-nm">' + esc(v.proxy.name) + '</span>'
          + '<b class="dr-pt-val">' + v.value + '</b></div>';
      }).join('');
      return { html: rows, tot: tot };
    }
    function bar(gt, rt){
      var t = gt + rt; if (t <= 0) return '';
      var gp = Math.round(gt / t * 100);
      return '<div class="dr-pt-bar" title="Value split"><span class="dr-pt-bar-g" style="width:' + gp + '%"></span><span class="dr-pt-bar-r" style="width:' + (100 - gp) + '%"></span></div>';
    }
    function evalNow(){
      var out = document.getElementById('drPtResult');
      if (!give.length && !get.length){ out.innerHTML = ''; return; }
      var g = sideRows(give), r = sideRows(get);
      var diff = r.tot - g.tot;
      var base = Math.max(1, g.tot, r.tot);
      var apct = Math.abs(diff) / base;
      var label, col;
      if (apct < 0.05){ label = 'Fair trade'; col = 'var(--text-muted)'; }
      else {
        var who = diff > 0 ? 'you' : 'them';
        label = (apct < 0.15 ? 'Slight edge to ' : (apct < 0.30 ? 'Good value for ' : 'Clear win for ')) + who;
        col = diff > 0 ? '#22c55e' : '#ef4444';
      }
      var detail = diff === 0 ? '' : ' <span class="dr-pt-vpct">' + (diff > 0 ? '+' : '-') + Math.round(apct * 100) + '%, ' + (diff > 0 ? '+' : '-') + Math.abs(diff) + ' value</span>';
      out.innerHTML = '<div class="dr-pt-cols">'
        + '<div><div class="dr-pt-side-h">You give (' + g.tot + ')</div>' + (g.html || '<div class="dr-pt-row dr-pt-empty">none</div>') + '</div>'
        + '<div><div class="dr-pt-side-h">You get (' + r.tot + ')</div>' + (r.html || '<div class="dr-pt-row dr-pt-empty">none</div>') + '</div>'
        + '</div>'
        + bar(g.tot, r.tot)
        + '<div class="dr-pt-verdict" style="color:' + col + '">' + label + detail + '</div>';
    }
    function refresh(){ renderChips(); evalNow(); }

    // Add a pick from a round/pick selector to its side.
    msg.querySelectorAll('.dr-pt-add').forEach(function(btn){
      btn.addEventListener('click', function(){
        var id = this.getAttribute('data-target');
        var rd = parseInt(document.getElementById(id + 'Rd').value, 10) || 1;
        var pk = parseInt(document.getElementById(id + 'Pk').value, 10) || 1;
        (id === 'drPtG' ? give : get).push((rd - 1) * teams + pk);
        refresh();
      });
    });
    // Quick-add your own picks to the give side.
    msg.querySelectorAll('.dr-pt-chip').forEach(function(c){
      c.addEventListener('click', function(){ give.push(parseInt(this.getAttribute('data-pn'), 10)); refresh(); });
    });
    // Remove a token (delegated on each freshly-created chip row).
    [['give', 'drPtGiveChips'], ['get', 'drPtGetChips']].forEach(function(pair){
      document.getElementById(pair[1]).addEventListener('click', function(e){
        var x = e.target.closest && e.target.closest('.dr-pt-tokx');
        if (!x) return;
        (pair[0] === 'give' ? give : get).splice(parseInt(x.getAttribute('data-i'), 10), 1);
        refresh();
      });
    });

    refresh();
    m.style.display = 'flex';
  }
  (function(){
    var b = document.getElementById('drPickTradeBtn');
    if (b) b.addEventListener('click', drPickTradeOpen);
  })();

  function drConfirm(msg, okLabel, cb){
    if (typeof okLabel === 'function'){ cb = okLabel; okLabel = 'Confirm'; }
    var m = document.getElementById('drModal');
    document.getElementById('drModalMsg').textContent = msg;
    var btns = document.getElementById('drModalBtns');
    btns.innerHTML = '';
    var cancel = document.createElement('button');
    cancel.className = 'dr-btn'; cancel.textContent = 'Cancel';
    cancel.addEventListener('click', function(){ m.style.display = 'none'; });
    var ok = document.createElement('button');
    ok.className = 'dr-btn dr-btn-primary'; ok.textContent = okLabel;
    ok.addEventListener('click', function(){ m.style.display = 'none'; cb(); });
    btns.appendChild(cancel); btns.appendChild(ok);
    m.style.display = 'flex';
  }

  // ── Actions ──────────────────────────────────────────────────────────────
  function commitPick(p){
    var pn = state.current;
    // Pick scoring leans on the external BRPickScore module + a lot of board
    // context; if any of that throws, the pick itself must still commit so the
    // draft never freezes. Score/reason are cosmetic and default to empty.
    var ps = null;
    try { ps = pickScoreFor(p); } catch (e){ _simError('score pick', e); }
    // Pool-relative score captured at the moment of the pick (the pool then = what
    // was still on the board), so the report card can show each pick "vs the best
    // still available" - the same scale as the live board. Absolute ps is kept for
    // the round-weighted grade.
    var psRel = null;
    try { psRel = psDisplay(ps); } catch (e){ psRel = null; }
    var reason = '';
    try { reason = pickReason(p, myPosCounts()); } catch (e){ reason = ''; }
    state.picks[pn] = { id: p.id, name: p.name, position: p.position, team: p.team, val: Math.round(valOf(p)), ps: ps, psRel: psRel, reason: reason };
    drafted[String(p.id)] = true;
    // Drop the just-drafted player from the queue so it stays a live target list
    // (and auto-draft never re-considers a taken player).
    if (state.queue){ var _qi = state.queue.indexOf(String(p.id)); if (_qi >= 0) state.queue.splice(_qi, 1); }
    justPick = pn;
    state.current++;
    if (window.brHaptic) window.brHaptic(14);   // tactile confirm on a pick
    skipFilledPicks();    // step over picks already spent on keepers
    paintCell(pn);        // fill just-picked cell (incremental)
    // Reveal the just-made pick with a pop + accent ring (CPU picks are ~700ms
    // apart, so these land one at a time). CSS disables it under reduced motion.
    var _revEl = document.getElementById('dc' + pn);
    if (_revEl){
      _revEl.classList.add('dr-cell-reveal');
      _revEl.addEventListener('animationend', function(){ _revEl.classList.remove('dr-cell-reveal'); }, { once: true });
    }
    refreshCurrent();     // move the on-the-clock highlight + auto-scroll
  }
  function draftPlayer(id){
    if (state.mode === 'live') return;   // live board is driven by the platform
    var total = state.teams * state.rounds;
    if (state.current > total) return;
    if (sim && !isMyPick(state.current)) return; // not your turn
    var p = playersById[String(id)];
    if (!p || drafted[String(id)]) return;
    commitPick(p);
    render();
    if (sim) scheduleSim();   // resume CPU picks after your selection
  }
  function undo(){
    if (state.current <= 1 || state.mode === 'live') return;
    var wasDone = state.current > state.teams * state.rounds;
    state.current--;
    var p = state.picks[state.current];
    if (p) { delete drafted[String(p.id)]; delete state.picks[state.current]; }
    paintCell(state.current);
    refreshCurrent();
    if (wasDone && !sim){ sim = true; simStarted = true; _summaryShown = false; syncSimControls(); }
    render();
  }
  function resetDraft(){
    var isLive = !!(state && state.mode === 'live');
    function doReset(){
      stopPolling(); stopPickTimer();
      _resetTransient();
      try { sessionStorage.removeItem(sessKey); } catch(e){}
      // Strip ?connect / ?live so a reload or Back doesn't auto-reconnect to the
      // draft we just exited (that's what re-loaded the board after Exit Board).
      try {
        var _u = new URL(location.href);
        if (_u.searchParams.has('connect') || _u.searchParams.has('live')){
          _u.searchParams.delete('connect'); _u.searchParams.delete('live');
          history.replaceState(null, '', _u.pathname + _u.search + _u.hash);
        }
      } catch(e){}
      state = null;
      showSetup();
    }
    if (isLive){ doReset(); } else { drConfirm('Reset the draft board? This wipes every pick and returns to setup.', 'Reset', doReset); }
  }

  // ── Share a draft image ─────────────────────────────────────────────────────
  var _shareDataUrls = { dark: null, light: null };
  var _shareTheme = 'dark';

  function _readThemeVars(dark){
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme');
    var want = dark ? 'dark' : null;
    if (want !== cur){ want ? root.setAttribute('data-theme', want) : root.removeAttribute('data-theme'); }
    var cs = getComputedStyle(root);
    function g(v){ return cs.getPropertyValue(v).trim(); }
    var vars = {
      bg:     g('--card'),
      header: g('--card-soft'),
      accent: g('--accent'),
      text:   g('--text'),
      sub:    g('--text-muted'),
      empty:  g('--text-subtle'),
      border: g('--border'),
      win:    g('--win'),
      info:   g('--info'),
    };
    if (want !== cur){ cur ? root.setAttribute('data-theme', cur) : root.removeAttribute('data-theme'); }
    return vars;
  }

  function _buildShareCanvas(dark){
    var mine = myPicksList().slice();
    var _olShare = optimalLineup(mine);
    var rows = [];
    _olShare.starters.forEach(function(s){ rows.push({ slot: s.slot, p: s.p }); });
    _olShare.bench.forEach(function(p){ rows.push({ slot: 'BN', p: p }); });
    var clr = _readThemeVars(dark);
    var POSC = POS_COLOR;  // shared palette defined at top of the IIFE
    var W = 720, pad = 30, lineH = 44, headerH = 130;
    var H = headerH + rows.length * lineH + pad;
    var c = document.createElement('canvas'); c.width = W; c.height = H;
    var ctx = c.getContext('2d');
    // Background
    ctx.fillStyle = clr.bg; ctx.fillRect(0, 0, W, H);
    // Header band
    ctx.fillStyle = clr.header; ctx.fillRect(0, 0, W, headerH - 10);
    // Title
    ctx.fillStyle = clr.accent; ctx.font = 'bold 28px system-ui,Arial,sans-serif';
    ctx.fillText('My ' + (state.type.charAt(0).toUpperCase() + state.type.slice(1)) + ' Draft', pad, pad + 28);
    // Subtitle
    ctx.fillStyle = clr.sub; ctx.font = '14px system-ui,Arial,sans-serif';
    ctx.fillText((state.sf ? 'Superflex' : '1QB') + ' \xb7 ' + state.teams + ' teams \xb7 BR Fantasy', pad, pad + 52);
    // Grade
    var g = gradeTeam();
    if (g){
      var gl = gradeLetter(g.score);
      var gp = (state.type === 'rookie' && g.avgPs != null) ? ('Avg pick score ' + g.avgPs) : null;
      if (!gp){ var _sa = teamArchetype(); if (_sa) gp = _sa.label; }
      ctx.fillStyle = clr.win; ctx.font = 'bold 15px system-ui,Arial,sans-serif';
      ctx.fillText('Grade ' + gl + (g.provisional ? ' \xb7 Early' : '') + (gp ? ('  \xb7  ' + gp) : ''), pad, pad + 76);
    }
    // Divider below header
    ctx.fillStyle = clr.border; ctx.fillRect(0, headerH - 10, W, 1);
    // Rows
    var y = headerH;
    rows.forEach(function(r, ri){
      // Alternating row tint using card-soft
      if (ri % 2 === 0){ ctx.fillStyle = clr.header; ctx.fillRect(0, y, W, lineH); }
      // Position badge (rounded rect)
      var posClr = POSC[r.slot] || clr.sub;
      ctx.fillStyle = posClr;
      ctx.beginPath();
      var bx = pad, by = y + (lineH - 22) / 2, bw = 38, bh = 22, br = 5;
      ctx.moveTo(bx + br, by); ctx.lineTo(bx + bw - br, by);
      ctx.arcTo(bx + bw, by, bx + bw, by + br, br);
      ctx.lineTo(bx + bw, by + bh - br);
      ctx.arcTo(bx + bw, by + bh, bx + bw - br, by + bh, br);
      ctx.lineTo(bx + br, by + bh);
      ctx.arcTo(bx, by + bh, bx, by + bh - br, br);
      ctx.lineTo(bx, by + br);
      ctx.arcTo(bx, by, bx + br, by, br);
      ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 11px system-ui,Arial,sans-serif';
      ctx.textAlign = 'center'; ctx.fillText(r.slot, bx + bw / 2, by + 15); ctx.textAlign = 'left';
      // Pick location badge (e.g. 1.04), just right of the position badge
      var pickShort = r.p ? pickNoShort(r.p) : '';
      var nameX = pad + 52;
      if (pickShort){
        ctx.fillStyle = clr.accent; ctx.font = 'bold 12px system-ui,Arial,sans-serif';
        ctx.fillText(pickShort, nameX, y + lineH / 2 + 6);
        nameX += ctx.measureText(pickShort).width + 12;
      }
      // Player name
      ctx.fillStyle = r.p ? clr.text : clr.empty;
      ctx.font = r.p ? 'bold 15px system-ui,Arial,sans-serif' : 'italic 14px system-ui,Arial,sans-serif';
      ctx.fillText(r.p ? r.p.name : 'open', nameX, y + lineH / 2 + 6);
      // Position + team (right-aligned)
      if (r.p){
        ctx.fillStyle = clr.sub; ctx.font = '13px system-ui,Arial,sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((r.p.position || '') + (r.p.team ? '  ' + r.p.team : ''), W - pad, y + lineH / 2 + 6);
        ctx.textAlign = 'left';
      }
      // Row divider
      ctx.fillStyle = clr.border; ctx.fillRect(pad, y + lineH - 1, W - pad * 2, 1);
      y += lineH;
    });
    return c;
  }

  function shareDraft(){
    if (!state || !hasOwned()){ drAlert('Pick a draft slot to build and share your team.'); return; }
    // Build both themes
    _shareDataUrls.dark  = _buildShareCanvas(true).toDataURL('image/png');
    _shareDataUrls.light = _buildShareCanvas(false).toDataURL('image/png');
    // Default to current UI theme
    _shareTheme = (document.documentElement.getAttribute('data-theme') === 'dark') ? 'dark' : 'light';
    var tabs = document.querySelectorAll('#drShareViewTabs .dr-shareview-tab');
    tabs.forEach(function(b){ b.classList.toggle('is-active', b.getAttribute('data-sv') === _shareTheme); });
    document.getElementById('drShareViewImg').src = _shareDataUrls[_shareTheme];
    document.getElementById('drShareView').style.display = 'flex';
  }

  function _doShareOrDownload(download){
    var url = _shareDataUrls[_shareTheme];
    if (!url) return;
    if (!download){
      // Try native share first
      try {
        var bstr = atob(url.split(',')[1]);
        var arr = new Uint8Array(bstr.length);
        for (var i = 0; i < bstr.length; i++) arr[i] = bstr.charCodeAt(i);
        var blob = new Blob([arr], { type: 'image/png' });
        var file = new File([blob], 'br-draft.png', { type: 'image/png' });
        if (navigator.share && navigator.canShare && navigator.canShare({ files: [file] })){
          navigator.share({ files: [file], title: 'My BR Fantasy Draft' }).catch(function(){});
          return;
        }
      } catch(e){}
    }
    var a = document.createElement('a'); a.href = url; a.download = 'br-draft.png'; a.click();
  }

  // ── Player preview / draft confirm ──────────────────────────────────────────
  function statBox(label, val, sub, tip){
    return '<div class="dr-prev-stat"><div class="dr-prev-stat-v">' + val + '</div>'
      + '<div class="dr-prev-stat-l">' + label + (tip ? infoIcon(tip) : '') + '</div>'
      + (sub ? '<div class="dr-prev-stat-sub">' + sub + '</div>' : '')
      + '</div>';
  }
  function isYourTurn(){
    if (state.mode === 'live') return false;
    if (state.current > state.teams * state.rounds) return false;
    if (sim && !isMyPick(state.current)) return false;
    return true;
  }
  function psColor(ps){ return ps >= 90 ? '#22c55e' : ps >= 75 ? '#38bdf8' : ps >= 60 ? '#f59e0b' : '#ef4444'; }
  // Client-side slug matching player_page.slugify on the server.
  function playerSlug(name){
    return String(name || '').toLowerCase().replace(/&/g, ' and ')
      .replace(/['‘’]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  }
  function openPreview(id){
    var p = playersById[String(id)]; if (!p) return;
    if (_psPoolMax <= 0) refreshPsPool();
    var f = draftPlayerFacts(p);
    var t = f.tier, ps = psRelLive(p);
    var vorStr = f.vor != null ? fmtSigned(f.vor, Number.isInteger(f.vor) ? 0 : 1) : '-';
    var pos = f.pos;
    var vsAdp = f.vsAdp != null ? fmtSigned(Math.round(f.vsAdp), 0) : '-';
    var sc = ps != null ? psColor(ps) : 'var(--text-muted)';
    var pc = posColor(p.position);
    var c = document.getElementById('drPreviewCard');
    // Position-colored top accent
    c.style.boxShadow = '0 16px 50px rgba(0,0,0,.3), inset 0 3px 0 ' + pc;
    var metaBits = [p.team || '', f.posRank, f.exp, (f.age != null ? 'Age ' + f.age.toFixed(0) : ''), (f.injury ? f.injury : '')].filter(Boolean);
    var h = '<button class="dr-prev-close" id="drPrevClose" aria-label="Close">&times;</button>'
      // Player identity row
      + '<div class="dr-prev-top">'
      + '<img class="dr-prev-hs" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
      + '<div class="dr-prev-id"><div class="dr-prev-name">' + esc(p.name) + (t ? (' <span class="dr-tier' + (isTierCliff(p) ? ' dr-tier-cliff' : '') + '">T' + t + '</span>') : '') + '</div>'
      + '<div class="dr-prev-meta"><span class="dr-posbadge" style="background:' + pc + '">' + esc(p.position) + '</span> ' + esc(metaBits.join(' · ')) + '</div>'
      + '</div></div>'
      // Pick Score hero
      + '<div class="dr-prev-score-hero" style="border-color:' + sc + ';background:' + sc + '1a;">'
      + '<div class="dr-prev-score-num" style="color:' + sc + '">' + (ps != null ? ps : '&ndash;') + '</div>'
      + '<div class="dr-prev-score-lbl">Pick Score' + infoIcon('A 0-100 grade of this pick at this slot: value, fall vs ADP, tier, your needs, age, and projected points. Higher is better.') + '</div>'
      + '<div class="dr-prev-score-reason">' + esc(ps != null ? pickReason(p, myPosCounts()) : 'Streamer / last-round pick') + '</div>'
      + '</div>'
      // Stats grid
      + '<div class="dr-prev-stats">'
      + statBox('Value', Math.round(f.value), null, 'Trade value as an asset on a 0-999 scale (dynasty value, or redraft value in redraft).')
      + statBox(f.vorLbl, vorStr, null, 'Value Over Replacement: how much better than a freely-available starter at this position. ' + (f.vorLbl === 'VORP' ? 'Based on projected season fantasy points, not last year\'s injury-shortened totals.' : 'Based on dynasty or redraft trade value.'))
      + statBox('ADP', f.adp != null ? (Number(f.adp).toFixed(1) + (f.adpN ? ' <span class="dr-adp-n">n=' + f.adpN + '</span>' : '')) : '-', null, 'Average Draft Position - the typical overall pick this player goes at in real drafts. n is how many real drafts the ADP is based on.')
      + statBox('vs ADP', vsAdp, null, 'How far this player has fallen past their ADP at the current pick. Positive = a value.')
      + (f.projPpg != null ? statBox('Proj PPG', f.projPpg.toFixed(1), 'projected', 'Points per game, projected for the upcoming season.') : '')
      + (f.lastPpg != null ? statBox((f.ppgSeason ? f.ppgSeason + ' PPG' : 'PPG'), f.lastPpg.toFixed(1), f.ppgRank != null ? (pos + f.ppgRank) : 'last season', 'Points per game last season.') : '')
      + (f.posRank ? statBox('Pos Rank', f.posRank, null, 'Rank at this position by current value.') : '')
      + (f.rec != null ? statBox('REC', '#' + f.rec, null, 'Live recommendation rank for this pick — roster-aware order, not a grade.') : '')
      + (f.bye != null ? statBox('Bye', f.bye, null, 'NFL bye week. Stacking several players on the same bye can leave a hole.') : '')
      + (f.projPts != null ? statBox('Proj Pts', Math.round(f.projPts), 'season', 'Projected fantasy points for the full upcoming season.') : '')
      + (f.market != null ? statBox('Mkt vs ADP', fmtSigned(Math.round(f.market), 0), null, 'How much earlier (positive) or later (negative) betting markets imply this player should go versus ADP.') : '')
      + (state.type !== 'redraft' && f.age != null ? statBox('Age', f.age.toFixed(0)) : '')
      + statBox(pos + ' T1-2 left', f.scarce, null, 'How many Tier 1-2 (elite) players remain available at this position - a scarcity signal.')
      + '</div>';
    // Survival probability at the user's next upcoming pick
    if (f.survivePn && f.survive != null){
      var col = availColor(f.survive);
      h += '<div class="dr-prev-avail-track">'
        + '<div class="dr-prev-avail-label">Survival at your next pick (#' + f.survivePn + ')</div>'
        + '<div class="dr-prev-avail-picks"><div class="dr-prev-avail-pick" style="background:' + col + '14;border:1px solid ' + col + '44;">'
        + '<span style="color:' + col + ';font-size:18px;font-weight:900;">' + f.survive + '%</span>'
        + '<span class="dr-prev-avail-pn">' + (f.survive >= 65 ? 'Likely available' : f.survive >= 40 ? 'Might be there' : 'Unlikely to last') + '</span>'
        + '</div></div></div>';
    }
    h += '<div class="dr-prev-btns">';
    if (state.mode === 'live' && !state.isDrafting){
      h += '<div class="dr-prev-note">Draft hasn\'t started yet. Picks will come from the platform once it begins.</div>';
    } else if (state.mode === 'live'){
      h += '<div class="dr-prev-note">Live draft. Picks come from the platform.</div>';
    } else if (isYourTurn() || !sim){
      h += '<button class="dr-btn dr-btn-primary dr-btn-lg dr-prev-draft" data-id="' + esc(String(p.id)) + '">Draft ' + esc(p.name) + '</button>';
    } else {
      h += '<div class="dr-prev-note">Not your pick yet. A CPU team is on the clock.</div>';
    }
    // Full profile: modal when logged in, external link for guests.
    if (pos !== 'PICK'){
      if (!cfg.isGuest && typeof openPlayerModal === 'function'){
        h += '<button class="dr-btn dr-prev-profile" data-profile="' + esc(String(p.id)) + '" data-profile-name="' + esc(p.name) + '">View full profile</button>';
      } else {
        var slug = playerSlug(p.name);
        if (slug) h += '<a class="dr-btn dr-prev-profile" href="/player/' + encodeURIComponent(slug) + '/trade-value" target="_blank" rel="noopener">View full player profile &#8599;</a>';
      }
    }
    h += '</div>';
    c.innerHTML = h;
    document.getElementById('drPreview').style.display = '';
  }
  function closePreview(){ document.getElementById('drPreview').style.display = 'none'; }

  // ── Wire up ──────────────────────────────────────────────────────────────
  document.getElementById('drStart').addEventListener('click', startDraft);
  document.getElementById('drStartSim').addEventListener('click', startMock);
  document.getElementById('drSimStart').addEventListener('click', beginSim);
  document.getElementById('drSimToggle').addEventListener('click', toggleSim);
  document.getElementById('drAutoBtn').addEventListener('click', function(){
    simAutoDraft = !simAutoDraft;
    syncSimControls();
    // If it's currently my pick and auto-draft just turned on, kick it off
    if (simAutoDraft && sim && simStarted && !simPaused && isMyPick(state.current)){
      clearTimeout(simTimer); simTimer = setTimeout(_doAutoPick, simSpeed);
    }
  });
  document.getElementById('drSimSpeed').addEventListener('change', function(){
    simSpeed = parseInt(this.value, 10) || 700;
    if (sim && simStarted && !simPaused) scheduleSim();
  });
  var _myStratSel = document.getElementById('drMyStrat');
  if (_myStratSel) _myStratSel.addEventListener('change', function(){
    if (state){ state.myStrat = this.value || ''; save(); }
  });
  var _myLeanSel = document.getElementById('drMyAgeLean');
  if (_myLeanSel) _myLeanSel.addEventListener('change', function(){
    if (state){ state.myAgeLean = this.value || ''; save(); }
  });
  // Collapsible Auto-draft settings group in the gear menu.
  var _autoTog = document.getElementById('drAutoSettingsToggle');
  if (_autoTog) _autoTog.addEventListener('click', function(){
    var body = document.getElementById('drAutoSettingsBody');
    var open = this.getAttribute('aria-expanded') === 'true';
    this.setAttribute('aria-expanded', String(!open));
    if (body) body.hidden = open;
  });
  document.getElementById('drSideTabs').addEventListener('click', function(e){
    var b = e.target.closest('.otc-main-tab'); if (!b) return;
    sideTab = b.getAttribute('data-stab');
    this.querySelectorAll('.otc-main-tab').forEach(function(x){ x.classList.toggle('is-active', x === b); });
    renderSide();
  });
  document.getElementById('drBoard').addEventListener('click', function(e){
    var cell = e.target.closest('[data-pn]'); if (!cell) return;
    var pn = parseInt(cell.getAttribute('data-pn'), 10);
    if (!canClaim(pn)) return;          // only future, uncommitted picks (mock/manual)
    toggleOwned(pn);
    _boardSig = null;                   // ownership changed: force board rebuild
    render();
    if (sim) scheduleSim();             // if you released the current pick, let the CPU run
  });
  document.getElementById('drConnect').addEventListener('click', detectLive);
  document.getElementById('drLiveList').addEventListener('click', function(e){
    var b = e.target.closest('.dr-live-item'); if (b) connectLive(b.getAttribute('data-id'));
  });
  document.getElementById('drCellToggle').addEventListener('click', function(e){
    var opt = e.target.closest('.dr-ct-opt'); if (!opt) return;
    var mode = opt.getAttribute('data-mode');
    _cellShowPs = (mode === 'ps');
    this.querySelectorAll('.dr-ct-opt').forEach(function(o){ o.classList.toggle('is-active', o.getAttribute('data-mode') === mode); });
    // Repaint all filled cells so the corner stat updates immediately.
    if (state) Object.keys(state.picks).forEach(function(k){ if (state.picks[k]) paintCell(parseInt(k, 10)); });
  });
  document.getElementById('drUndo').addEventListener('click', undo);
  document.getElementById('drReset').addEventListener('click', resetDraft);
  document.getElementById('drEdit').addEventListener('click', openEditSetup);
  document.getElementById('drEditApply').addEventListener('click', applyEditedSetup);
  document.getElementById('drEditCancel').addEventListener('click', closeEditSetup);
  document.getElementById('drEditClose').addEventListener('click', closeEditSetup);
  document.getElementById('drEditReset').addEventListener('click', resetDraft);
  document.getElementById('drLeagueMeta').addEventListener('click', function(){
    if (this.disabled || (state && state.mode === 'live')) return;
    openEditSetup();
  });
  document.getElementById('drSetup').addEventListener('click', function(e){
    if (this.classList.contains('dr-setup-is-modal') && e.target === this) closeEditSetup();
  });
  document.addEventListener('keydown', function(e){
    if (e.key !== 'Escape') return;
    var confirm = document.getElementById('drModal');
    if (confirm && confirm.style.display === 'flex') return;
    var setup = document.getElementById('drSetup');
    if (setup && setup.classList.contains('dr-setup-is-modal')) closeEditSetup();
  });
  document.getElementById('drPractice').addEventListener('click', startPracticeMock);
  // Header Settings dropdown (gear). Opens below the gear; closes on an outside
  // tap or the gear again. The outside-close listener is attached on the NEXT
  // tick after opening, so the very tap that opened it can't also close it —
  // that race (touch + the synthesized click on mobile) is what made it
  // insta-close. The listener is removed again on close.
  (function initOptsDropdown(){
    var wrap = document.querySelector('.dr-side-opts');
    var panel = document.getElementById('drOptsPanel');
    var btn = document.getElementById('drOptsBtn');
    if (!wrap || !panel || !btn) return;
    var mq = window.matchMedia('(max-width: 900px)');
    var statusRight = document.querySelector('.dr-status-right');
    var tabs = document.getElementById('drSideTabs');
    var panelHome = panel.parentNode, panelNext = panel.nextSibling;

    // Gear lives in the header on desktop, beside the side-panel tabs on mobile.
    function placeWrap(){
      var dest = mq.matches ? tabs : statusRight;
      if (dest && wrap.parentNode !== dest) dest.appendChild(wrap);
    }
    function isOpen(){ return btn.getAttribute('aria-expanded') === 'true'; }

    function position(){
      if (!mq.matches){
        // Desktop: CSS-anchored dropdown below the gear (in the header).
        panel.style.position = ''; panel.style.top = ''; panel.style.bottom = '';
        panel.style.left = ''; panel.style.right = '';
        return;
      }
      // Mobile: fixed popover so it escapes the sheet (transformed + overflow:hidden).
      // Opens UPWARD when the gear sits in the lower half of the screen (sheet at the
      // mid snap), DOWNWARD when it's in the upper half (sheet at the top snap).
      var r = btn.getBoundingClientRect();
      panel.style.position = 'fixed';
      panel.style.left = 'auto';
      panel.style.right = Math.max(8, window.innerWidth - r.right) + 'px';
      if (r.top > window.innerHeight * 0.5){
        panel.style.bottom = (window.innerHeight - r.top + 6) + 'px';
        panel.style.top = 'auto';
      } else {
        panel.style.top = (r.bottom + 6) + 'px';
        panel.style.bottom = 'auto';
      }
    }
    function onDoc(e){
      if (btn.contains(e.target) || panel.contains(e.target)) return;  // gear / panel taps
      close();
    }
    function open(){
      if (mq.matches && panel.parentNode !== document.body) document.body.appendChild(panel);
      panel.style.display = 'flex';
      position();
      btn.setAttribute('aria-expanded', 'true');
      setTimeout(function(){
        document.addEventListener('click', onDoc);
        document.addEventListener('touchstart', onDoc, { passive: true });
        window.addEventListener('resize', position);
      }, 0);
    }
    function close(){
      panel.style.display = 'none';
      btn.setAttribute('aria-expanded', 'false');
      if (panel.parentNode === document.body){
        panel.style.position = ''; panel.style.top = ''; panel.style.bottom = '';
        panel.style.left = ''; panel.style.right = '';
        panelHome.insertBefore(panel, panelNext);   // back beside the gear
      }
      document.removeEventListener('click', onDoc);
      document.removeEventListener('touchstart', onDoc);
      window.removeEventListener('resize', position);
    }
    btn.addEventListener('click', function(e){
      e.stopPropagation();
      if (isOpen()) close(); else open();
    });
    // Tapping an action in the menu dismisses it (the speed <select> doesn't).
    panel.addEventListener('click', function(e){ if (e.target.closest('.dr-btn')) close(); });
    if (mq.addEventListener) mq.addEventListener('change', function(){ if (isOpen()) close(); placeWrap(); });
    else if (mq.addListener) mq.addListener(placeWrap);
    placeWrap();
  })();
  // Custom sort dropdown — the only sort control (the native <select> popup
  // mis-anchors inside the transformed mobile sheet). The current sort lives in
  // the button's data-val, which renderBA reads.
  (function initSortSelect(){
    var ui = document.getElementById('drBaSortUI');
    var btn = document.getElementById('drBaSortBtn');
    var menu = document.getElementById('drBaSortMenu');
    var lbl = document.getElementById('drBaSortLbl');
    if (!ui || !btn || !menu || !lbl) return;
    var LABELS = { value: 'Value', adp: 'ADP', pickscore: 'Pick Score', ps: 'Recommendation', ppg: 'Proj PPG' };
    var opts = menu.querySelectorAll('.dr-sortsel-opt');
    var cur = btn.getAttribute('data-val') || 'ps';
    function apply(v){
      cur = v;
      btn.setAttribute('data-val', v);
      lbl.textContent = LABELS[v] || v;
      for (var i = 0; i < opts.length; i++){ opts[i].classList.toggle('is-active', opts[i].getAttribute('data-val') === v); }
    }
    function open(){ menu.hidden = false; btn.setAttribute('aria-expanded', 'true'); }
    function close(){ menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); }
    function isOpen(){ return !menu.hidden; }
    btn.addEventListener('click', function(e){ e.stopPropagation(); if (isOpen()) close(); else open(); });
    menu.addEventListener('click', function(e){
      var opt = e.target.closest('.dr-sortsel-opt'); if (!opt) return;
      e.stopPropagation();
      var v = opt.getAttribute('data-val');
      if (v !== cur){ apply(v); renderBA(); }
      close();
    });
    document.addEventListener('click', function(e){ if (isOpen() && !ui.contains(e.target)) close(); });
    apply(cur);
  })();
  document.getElementById('drSearch').addEventListener('input', renderBA);
  document.getElementById('drBaList').addEventListener('click', function(e){
    var cmp = e.target.closest('[data-cmp]');
    if (cmp){ e.stopPropagation(); toggleCompare(cmp.getAttribute('data-cmp')); return; }
    var star = e.target.closest('[data-star]');
    if (star){ e.stopPropagation(); toggleQueue(star.getAttribute('data-star')); return; }
    var draft = e.target.closest('[data-draft]');
    if (draft){ e.stopPropagation(); draftPlayer(draft.getAttribute('data-draft')); return; }
    var row = e.target.closest('.dr-ba-row');
    if (row) openPreview(row.getAttribute('data-id'));
  });
  // Best-at-pos chips: collapse toggle, chip preview, scarcity filter
  document.getElementById('drBestChips').addEventListener('click', function(e){
    if (e.target.closest('#drBestChipsToggle')){
      _chipsCollapsed = !_chipsCollapsed;
      renderBestChips();
      return;
    }
    var chip = e.target.closest('[data-bchip]');
    if (chip){ openPreview(chip.getAttribute('data-bchip')); return; }
    var sp = e.target.closest('[data-scarpos]');
    if (sp){
      var pos = sp.getAttribute('data-scarpos');
      posFilter = {}; posFilter[String(pos).toUpperCase()] = true;   // focus this position
      _syncPosPills();
      renderBA();
    }
  });
  // Compare overlay: close, draft from compare
  document.getElementById('drCompare').addEventListener('click', function(e){
    if (e.target === this || e.target.closest('#drCmpClose')){ closeCompare(); return; }
    var d = e.target.closest('[data-cmp-draft]');
    if (d){ var id = d.getAttribute('data-cmp-draft'); closeCompare(); draftPlayer(id); }
  });
  document.getElementById('drSummaryBtn').addEventListener('click', openSummary);
  // The status-bar grade pill doubles as a shortcut into the league report card.
  (function(){ var gp = document.getElementById('drGradePill'); if (gp) gp.addEventListener('click', openSummary); })();
  document.getElementById('drSummary').addEventListener('click', function(e){
    if (e.target === this) closeSummary();
  });
  (function(){
    var ddc = document.getElementById('drCompleteDeepDiveBtn');
    if (ddc){
      ddc.addEventListener('click', openDeepDive);
      // Premium users don't need the PRO chip on the button.
      if (cfg.hasPremium){ var _chip = ddc.querySelector('.dr-dd-prochip'); if (_chip) _chip.remove(); }
    }
    var ddOv = document.getElementById('drDeepDive');
    if (ddOv) ddOv.addEventListener('click', function(e){ if (e.target === this) closeDeepDive(); });
    document.addEventListener('keydown', function(e){
      if (e.key === 'Escape' && ddOv && ddOv.style.display !== 'none') closeDeepDive();
    });
  })();
  document.getElementById('drHelpBtn').addEventListener('click', openGlossary);
  document.addEventListener('visibilitychange', _onSimVisibility);
  document.getElementById('drGlossClose').addEventListener('click', closeGlossary);
  document.getElementById('drGloss').addEventListener('click', function(e){ if (e.target === this) closeGlossary(); });
  document.getElementById('drShare').addEventListener('click', shareDraft);
  document.getElementById('drCompleteSummaryBtn').addEventListener('click', openSummary);
  document.getElementById('drCompleteShareBtn').addEventListener('click', shareDraft);
  document.getElementById('drShareViewClose').addEventListener('click', function(){ document.getElementById('drShareView').style.display = 'none'; });
  document.getElementById('drShareView').addEventListener('click', function(e){ if (e.target === this) this.style.display = 'none'; });
  document.getElementById('drShareViewTabs').addEventListener('click', function(e){
    var b = e.target.closest('.dr-shareview-tab'); if (!b) return;
    _shareTheme = b.getAttribute('data-sv');
    this.querySelectorAll('.dr-shareview-tab').forEach(function(x){ x.classList.toggle('is-active', x === b); });
    document.getElementById('drShareViewImg').src = _shareDataUrls[_shareTheme];
  });
  document.getElementById('drShareViewShare').addEventListener('click', function(){ _doShareOrDownload(false); });
  document.getElementById('drShareViewDl').addEventListener('click', function(){ _doShareOrDownload(true); });
  document.getElementById('drPreview').addEventListener('click', function(e){
    if (e.target === this || e.target.closest('#drPrevClose')){ closePreview(); return; }
    var d = e.target.closest('.dr-prev-draft');
    if (d){ var id = d.getAttribute('data-id'); closePreview(); draftPlayer(id); return; }
    var prof = e.target.closest('[data-profile]');
    if (prof && typeof openPlayerModal === 'function'){
      e.preventDefault();
      closePreview();
      openPlayerModal(prof.getAttribute('data-profile'), prof.getAttribute('data-profile-name') || '');
    }
  });
  document.getElementById('drPosFilters').addEventListener('click', function(e){
    var b = e.target.closest('.dr-pos'); if (!b) return;
    var pos = b.getAttribute('data-pos');
    if (pos === 'ALL'){
      posFilter = {};                       // clear the set -> show everything
    } else {
      var key = String(pos).toUpperCase();
      if (posFilter[key]) delete posFilter[key]; else posFilter[key] = true;  // toggle
    }
    _syncPosPills();
    renderBA();
  });

  applyCfgDefaults();
  // Dynasty / non-keeper leagues: drop the Keeper draft type and its fields
  // entirely so keepers never appear where they don't apply.
  if (cfg.showKeeper === false) {
    var _kOpt = document.querySelector('#drType option[value="keeper"]');
    if (_kOpt) _kOpt.remove();
    Array.prototype.forEach.call(document.querySelectorAll('.dr-keeper-only'), function (el) { el.remove(); });
  }
  renderSetupRoster();
  renderSetupCapital();
  document.getElementById('drSf').addEventListener('change', function(){ _rosterMode = 'auto'; _rosterPreset = null; _setupRoster = null; renderSetupRoster(); });
  document.getElementById('drType').addEventListener('change', function(){
    // Reset roster to defaults for the new type, then re-render.
    _rosterMode = 'auto'; _rosterPreset = null; _setupRoster = null; renderSetupRoster();
    // Show/hide rookie rounds field; for non-rookie, rounds auto-sync from roster.
    var isRookie = this.value === 'rookie';
    var rf = document.getElementById('drRoundsField');
    if (rf) rf.style.display = isRookie ? '' : 'none';
    if (isRookie) document.getElementById('drRounds').value = String(cfg.numRoundsRookie || 3);
    syncKeeperSetupFields(this.value === 'keeper');
    renderSetupCapital();   // refresh claimed-pick list after rounds change
  });

  // Keeper-only setup fields. Defaults come from the league's own keeper payload
  // so the count matches what the league actually allows.
  function syncKeeperSetupFields(on){
    Array.prototype.forEach.call(document.querySelectorAll('.dr-keeper-only'), function(el){
      el.style.display = on ? '' : 'none';
    });
    if (!on) return;
    var cEl = document.getElementById('drKeeperCount');
    if (cEl && !cEl.dataset.touched){
      var lim = cfg.keepers && cfg.keepers.limit;
      cEl.value = String(lim != null ? lim : 2);
    }
    var sEl = document.getElementById('drKeeperSource');
    if (sEl && !(cfg.keepers && (cfg.keepers.kept || []).length)){
      // Nothing from the assistant for this league - default to picking your own.
      sEl.value = 'manual';
    }
  }
  (function(){
    var cEl = document.getElementById('drKeeperCount');
    if (cEl) cEl.addEventListener('input', function(){ this.dataset.touched = '1'; });
  })();
  // Any control that changes the pick map resets claimed picks to the slot default.
  ['drTeams','drRounds','drOrder','drSlot'].forEach(function(idn){
    document.getElementById(idn).addEventListener('change', renderSetupCapital);
  });
  // Rounds <-> bench two-way sync: rounds = starters + bench.
  document.getElementById('drRounds').addEventListener('change', function(){
    var rounds = Math.max(1, Math.min(40, parseInt(this.value, 10) || 15));
    this.value = rounds;
    if (!_setupRoster) _setupRoster = defaultRoster();
    _rosterMode = 'custom'; _rosterPreset = null;
    _setupRoster.BN = Math.max(0, rounds - _totalStarterSlots(_setupRoster) - _stashSlots(_setupRoster));
    renderSetupRoster();
  });
  document.getElementById('drRosterSection').addEventListener('click', function(e){
    var step = e.target.closest('.dr-step-btn');
    if (!step) return;
    e.stopPropagation();
    var key = step.getAttribute('data-key');
    var d = parseInt(step.getAttribute('data-d'), 10);
    if (!_setupRoster) _setupRoster = defaultRoster();
    _setupRoster[key] = Math.max(0, (_setupRoster[key] || 0) + d);
    _rosterMode = 'custom'; _rosterPreset = null;
    // The Superflex slot is now editable like any other position. Keep the drSf
    // format toggle (and the roster's own _sf marker) in sync with the count so
    // downstream recommendations and grading match the roster the user built.
    if (key === 'SF'){
      var _isSf = (_setupRoster.SF || 0) > 0;
      document.getElementById('drSf').value = _isSf ? '1' : '0';
      _setupRoster._sf = _isSf;
    }
    // Keep rounds = starters + bench in sync for every slot change.
    // Bench change -> update rounds. Starter change -> update rounds (bench stays).
    var _newRounds = Math.max(1, Math.min(40, _totalStarterSlots(_setupRoster) + (_setupRoster.BN || 0) + _stashSlots(_setupRoster)));
    document.getElementById('drRounds').value = _newRounds;
    renderSetupCapital();
    renderSetupRoster();
  });
  document.getElementById('drCapitalSection').addEventListener('click', function(e){
    if (!_setupOwned) _setupOwned = {};
    // Remove a pick by clicking its pill.
    var rm = e.target.closest('[data-rm]');
    if (rm){ delete _setupOwned[rm.getAttribute('data-rm')]; renderSetupCapital(); return; }
    // Toggle a pick on/off from the inline slot picker.
    var add = e.target.closest('[data-add]');
    if (add){
      var pn = add.getAttribute('data-add');
      if (_setupOwned[pn]) delete _setupOwned[pn]; else _setupOwned[pn] = true;
      renderSetupCapital();
      return;
    }
    // Open/close the inline slot picker for a round.
    var ar = e.target.closest('[data-addround]');
    if (ar){
      var r = parseInt(ar.getAttribute('data-addround'), 10);
      _capAddRound = (_capAddRound === r) ? null : r;
      renderSetupCapital();
      return;
    }
    // Expand/collapse the combined late-rounds section.
    if (e.target.closest('#drCapLateToggle')){
      _capLateOpen = !_capLateOpen;
      renderSetupCapital();
    }
  });

  function resumeFromSession(){
    var saved = load();
    if (saved && saved.teams && saved.picks){
      state = saved;
      if (!state.roster) state.roster = defaultRoster();
      if (state.mode !== 'live' && !state.owned) state.owned = defaultOwned();
      if (state.mode === 'live'){
        document.getElementById('drUndo').style.display = 'none';
        if (state.isComplete){
          showCompleteSidebar();
          document.getElementById('drLiveBadge').style.display = 'none';
          document.getElementById('drUpcomingBadge').style.display = 'none';
        } else {
          // Badges are refreshed on the first poll; hide both until confirmed
          document.getElementById('drLiveBadge').style.display = 'none';
          document.getElementById('drUpcomingBadge').style.display = 'none';
        }
      } else if (state.mode === 'mock'){
        // Restore the mock: a not-yet-started draft comes back to the Start
        // Draft (ready) state; an in-progress one resumes running.
        var done = state.current > state.teams * state.rounds;
        if (!done){ sim = true; simStarted = !!state.simStarted; simPaused = false; }
        syncSimControls();
        _setUpcomingMode(false);
      }
      showMain();
      loadPlayers();
      if (state.mode === 'live' && state.sourceDraftId && !state.isComplete) startPolling();
    }
  }

  // ── Board hover: team needs tooltip ─────────────────────────────────────────
  function buildTeamTip(slot){
    var teams = state.teams || 12;
    var isMe = ownsAllInColumn(slot);
    var total = teams * state.rounds;
    var nameLabel = isMe ? 'You (Team ' + slot + ')' : teamName(slot);
    // Resolve the seat that OWNS a pick, following traded picks (a team can own
    // picks sitting in another seat's column) - so the tip reflects the team's
    // real haul, not just what fell in this column. Falls back to snake order.
    function ownerOf(pn){
      return (state.pickOwners && state.pickOwners[pn] != null)
        ? state.pickOwners[pn] : slotOnClock(pn, teams, state.order);
    }
    var nextPick = null;
    for (var pn = state.current; pn <= total; pn++){
      if (ownerOf(pn) === slot && !state.picks[pn]){ nextPick = pn; break; }
    }
    var nextHtml = nextPick ? '<div class="dr-team-tip-next">Next pick: #' + nextPick + '</div>' : '';

    // Collect this team's selections (in pick order) once, shared by both layouts.
    var seatPicks = [];
    Object.keys(state.picks).map(Number).sort(function(a, b){ return a - b; }).forEach(function(pn){
      var pick = state.picks[pn]; if (!pick) return;
      if (ownerOf(pn) !== slot) return;
      seatPicks.push(pick);
    });

    // ── Rookie drafts: roster needs are noise (you're adding to a full team).
    // Show what they actually drafted plus how many elite (T1-2) rookies landed.
    if (state.type === 'rookie'){
      var elite = 0;
      seatPicks.forEach(function(p){ var t = tierOf(playersById[String(p.id)] || p); if (t != null && t <= 2) elite++; });
      var statsHtml = '<div class="dr-team-tip-stats">'
        + '<div class="dr-team-tip-stat"><div class="dr-team-tip-stat-v">' + seatPicks.length + '</div><div class="dr-team-tip-stat-l">Picks</div></div>'
        + '<div class="dr-team-tip-stat"><div class="dr-team-tip-stat-v">' + elite + '</div><div class="dr-team-tip-stat-l">Elite T1-2</div></div>'
        + '</div>';
      var picksHtml;
      if (seatPicks.length){
        picksHtml = '<div class="dr-team-tip-picks">';
        seatPicks.forEach(function(p){
          var pos = (p.position || '').toUpperCase();
          var col = posColor(pos);
          var tier = tierOf(playersById[String(p.id)] || p);
          picksHtml += '<div class="dr-team-tip-pick">'
            + '<span class="dr-team-tip-pick-pos" style="background:' + col + '22;color:' + col + '">' + esc(pos) + '</span>'
            + '<span class="dr-team-tip-pick-nm">' + esc(p.name) + '</span>'
            + (tier != null ? '<span class="dr-team-tip-pick-tier">T' + tier + '</span>' : '')
            + '</div>';
        });
        picksHtml += '</div>';
      } else {
        picksHtml = '<div class="dr-team-tip-empty">No picks yet.</div>';
      }
      return '<div class="dr-team-tip"><div class="dr-team-tip-name">' + esc(nameLabel) + '</div>'
        + statsHtml + picksHtml + nextHtml + '</div>';
    }

    // ── Startup / redraft: positional needs vs targets.
    var positions = state.type === 'redraft' ? ['QB','RB','WR','TE','K','DEF'] : ['QB','RB','WR','TE'];
    var targets = posTargets();
    var counts = {}; positions.forEach(function(p){ counts[p] = 0; });
    seatPicks.forEach(function(pick){
      var pos = (pick.position || '').toUpperCase();
      if (counts[pos] != null) counts[pos]++;
    });
    var html = '<div class="dr-team-tip"><div class="dr-team-tip-name">' + esc(nameLabel) + '</div>'
      + '<div class="dr-team-tip-pos-row">';
    positions.forEach(function(pos){
      var t = targets[pos] || 0; if (!t) return;
      var have = counts[pos] || 0;
      var filled = have >= t;
      var col = filled ? '#22c55e' : (have > 0 ? '#f59e0b' : '#ef4444');
      html += '<div class="dr-team-tip-pos" style="border-color:' + col + '44;background:' + col + '18;">'
        + '<span class="dr-team-tip-pos-lbl" style="color:' + col + '">' + pos + '</span>'
        + '<span class="dr-team-tip-pos-cnt" style="color:' + col + '">' + have + '/' + t + '</span>'
        + '</div>';
    });
    html += '</div>' + nextHtml + '</div>';
    return html;
  }
  (function initBoardTip(){
    var board = document.getElementById('drBoard');
    var tip = document.getElementById('drTeamTip');
    if (!board || !tip) return;
    var _tipSlot = null;
    board.addEventListener('mouseover', function(e){
      if (window.matchMedia('(max-width: 900px)').matches) { tip.style.display = 'none'; return; }
      var head = e.target.closest('[data-slot]');
      if (!head){ tip.style.display = 'none'; _tipSlot = null; return; }
      var slot = parseInt(head.getAttribute('data-slot'), 10);
      if (slot === _tipSlot) return;
      _tipSlot = slot;
      tip.innerHTML = buildTeamTip(slot);
      tip.style.display = '';
    });
    board.addEventListener('mousemove', function(e){
      if (tip.style.display === 'none') return;
      var x = e.clientX + 16, y = e.clientY + 24;
      if (x + 200 > window.innerWidth) x = e.clientX - 210;
      if (y + 180 > window.innerHeight) y = e.clientY - 180;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    });
    board.addEventListener('mouseleave', function(){
      tip.style.display = 'none'; _tipSlot = null;
    });
  })();

  // ── Mobile bottom-sheet drag behavior ───────────────────────────────────────
  // Below 900px the side panel is a draggable sheet with three snap points:
  // peek (~14vh), mid (~44vh, default), and full (~92vh). Drag the grip handle
  // up/down; on release it snaps to the nearest point.
  (function initSheet(){
    var sheet = document.getElementById('drSide');
    var handle = document.getElementById('drSheetHandle');
    if (!sheet || !handle) return;
    var mq = window.matchMedia('(max-width: 900px)');
    var dragging = false, startY = 0, startT = 0, curT = 0, snapIdx = 1;
    function ih(){ return window.innerHeight; }
    // translateY offsets (px): full (whole 85vh sheet shows, top stops below the
    // header + status bar), mid (~49vh visible), peek (~19vh visible - the handle,
    // tabs, and a couple of rows). Peek accounts for the 85vh sheet sitting ~15vh down.
    function snaps(){ return [0, ih() * 0.36, ih() * 0.66]; }
    function applyT(t){ curT = t; sheet.style.transform = 'translateY(' + t + 'px)'; }
    function snapTo(idx){
      var pts = snaps();
      snapIdx = Math.max(0, Math.min(pts.length - 1, idx));
      sheet.classList.remove('dragging');
      // Fully-expanded sheet covers the global mobile tab bar; every other snap
      // (and desktop) leaves it visible below the sheet.
      document.body.classList.toggle('dr-sheet-expanded', mq.matches && snapIdx === 0);
      applyT(pts[snapIdx]);
    }
    function pointY(e){ return e.touches ? e.touches[0].clientY : e.clientY; }
    function onDown(e){
      if (!mq.matches) return;
      dragging = true; startY = pointY(e); startT = curT;
      sheet.classList.add('dragging');
      e.preventDefault();
    }
    function onMove(e){
      if (!dragging) return;
      var dy = pointY(e) - startY;
      var t = Math.max(0, Math.min(ih() * 0.74, startT + dy));
      applyT(t);
      if (e.cancelable) e.preventDefault();
    }
    function onUp(){
      if (!dragging) return;
      dragging = false;
      var pts = snaps(), best = 0, bd = Infinity;
      for (var i = 0; i < pts.length; i++){ var d = Math.abs(pts[i] - curT); if (d < bd){ bd = d; best = i; } }
      snapTo(best);
    }
    handle.addEventListener('touchstart', onDown, { passive: false });
    handle.addEventListener('mousedown', onDown);
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchend', onUp);
    window.addEventListener('mouseup', onUp);
    // Don't interfere with options panel interactions
    var optsPanel = document.getElementById('drOptsPanel');
    var optsBtn = document.getElementById('drOptsBtn');
    if (optsPanel && optsBtn){
      optsPanel.addEventListener('touchstart', function(e){ e.stopPropagation(); }, { passive: false });
      optsBtn.addEventListener('touchstart', function(e){ e.stopPropagation(); }, { passive: false });
    }
    // Tapping a tab while peeking lifts the sheet to mid so the content shows.
    document.getElementById('drSideTabs').addEventListener('click', function(){
      if (mq.matches && snapIdx === 2) snapTo(1);
    });
    function applyMode(){
      if (mq.matches){ snapTo(snapIdx); }
      else { sheet.style.transform = ''; sheet.classList.remove('dragging'); document.body.classList.remove('dr-sheet-expanded'); }
    }
    // Safety: never leave the tab bar hidden if the page is navigated away from.
    window.addEventListener('pagehide', function(){ document.body.classList.remove('dr-sheet-expanded'); });
    if (mq.addEventListener) mq.addEventListener('change', applyMode); else mq.addListener(applyMode);
    window.addEventListener('resize', function(){ if (mq.matches) snapTo(snapIdx); });
    applyMode();
  })();

  // Open a specific league draft directly: ?connect=<id> (from the site-wide
  // "Join Draft Room" banner) or ?live=<id> (from Draft History). Otherwise
  // resume the in-progress session draft.
  initKeepers();   // seed league keepers from the keeper tool (no-op if none)

  var _qs = new URLSearchParams(location.search);
  var urlLive = _qs.get('connect') || _qs.get('live');
  if (urlLive){
    connectLive(urlLive);
  } else {
    resumeFromSession();
  }
})();
