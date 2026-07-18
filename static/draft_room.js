(function(){
  var cfg = window.__draftCfg || {};
  var POS_COLOR = { QB:'#f59e0b', RB:'#22c55e', WR:'#3b82f6', TE:'#8b5cf6', K:'#94a3b8', DEF:'#64748b' };
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
  var posFilter = 'ALL';
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
  var _setupOwned = null;  // claimed picks (pickNumber -> true) built on setup page
  var _setupOwnedSig = ''; // staleness signature for _setupOwned
  var _capAddRound = null; // round whose inline slot picker is open (or null)
  var _capLateOpen = false;// whether the combined late-rounds section is expanded
  var tierThresholds = {}; // {leagueType:{size:[...]}} from /api/league-players
  var adpSources = {};     // {startup|rookie|redraft: 'Sleeper'|'none'} from /api/league-players
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
  function applyCfgDefaults(){
    if (cfg.numTeams) {
      var t = document.getElementById('drTeams');
      var want = String(Math.min(14, Math.max(8, cfg.numTeams)));
      for (var i=0;i<t.options.length;i++){ if (t.options[i].value === want || t.options[i].text === want){ t.selectedIndex = i; break; } }
    }
    if (cfg.isSuperflex) document.getElementById('drSf').value = '1';
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
  function rosterFromLeague(){
    var rp = cfg.rosterPositions;
    if (!rp || !rp.length) return null;
    var r = { QB:0, SF:0, RB:0, WR:0, TE:0, FLEX:0, K:0, DEF:0, BN:0 };
    var map = {
      QB:'QB', RB:'RB', WR:'WR', TE:'TE',
      FLEX:'FLEX', WRRB_FLEX:'FLEX', REC_FLEX:'FLEX', WRRBTE_FLEX:'FLEX',
      SUPER_FLEX:'SF', SFLEX:'SF',
      K:'K', DEF:'DEF', DST:'DEF', BN:'BN'
    };
    rp.forEach(function(s){
      var key = map[String(s).toUpperCase()];
      if (key) r[key]++;            // IDP/TAXI/IR positions are ignored
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
    return { QB:1, SF:sf?1:0, RB:2, WR:3, TE:1, FLEX:1, K:rd?1:0, DEF:rd?1:0, BN:bench };
  }

  // Helper: reconcile a raw roster map for the chosen QB format and return a
  // fresh copy (does not mutate the input).
  function _reconcileRoster(r, sf, rd){
    var out = {};
    ['QB','SF','RB','WR','TE','FLEX','K','DEF','BN'].forEach(function(k){ out[k] = r[k] || 0; });
    if (sf){ if (!out.SF) out.SF = 1; if (!out.FLEX) out.FLEX = 1; }
    else   { out.SF = 0; }
    // K/DEF are kept as-is across formats: if the league (or the user) rosters
    // them, they stay draftable regardless of dynasty/startup/redraft.
    return out;
  }

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
      { key:'SF',   label:'Superflex', hide: !sf },
      { key:'RB',   label:'RB' },
      { key:'WR',   label:'WR' },
      { key:'TE',   label:'TE' },
      { key:'FLEX', label:'FLEX' },
      { key:'K',    label:'K',    hide: rk },
      { key:'DEF',  label:'DEF',  hide: rk },
      { key:'BN',   label:'Bench' }
    ];

    // Source badge + mode-toggle button at the top.
    var srcHtml = '';
    if (hasLeague){
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

    var html = '<div class="dr-setup-roster">' + srcHtml;
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
      _rosterMode = 'auto'; _setupRoster = null; renderSetupRoster();
    });
  }

  // Sum of all non-bench roster slots - used to keep rounds and bench in sync.
  function _totalStarterSlots(rs){
    rs = rs || _setupRoster || defaultRoster();
    return (rs.QB||0) + (rs.SF||0) + (rs.RB||0) + (rs.WR||0) + (rs.TE||0) + (rs.FLEX||0) + (rs.K||0) + (rs.DEF||0);
  }
  // Sync the hidden drRounds field from the current roster (starters + bench).
  function _syncRoundsFromRoster(){
    if (!_setupRoster) return;
    var r = _totalStarterSlots(_setupRoster) + (_setupRoster.BN || 0);
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
    var rd = document.getElementById('drType').value === 'redraft';
    return {
      type:   document.getElementById('drType').value,
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
  // Scoring settings from setup (PPR weight + TE premium). These shift the
  // recommended roster build rather than recomputing raw player values.
  function readScoring(){
    var pprEl = document.getElementById('drPpr');
    var tepEl = document.getElementById('drTep');
    var ppr = pprEl ? parseFloat(pprEl.value) : 1.0;
    var tep = tepEl ? parseFloat(tepEl.value) : 0;
    return { ppr: isNaN(ppr) ? 1.0 : ppr, tep: isNaN(tep) ? 0 : tep };
  }
  function scoringCfg(){
    var s = (state && state.scoring) || {};
    return { ppr: s.ppr != null ? s.ppr : 1.0, tep: s.tep != null ? s.tep : 0 };
  }
  // Convert a Sleeper-style roster_positions array to the {QB:1, RB:2, ...} map
  // used by state.roster. Uses the same normalization as rosterFromLeague so the
  // live/connected path recognizes K/DEF (incl. DST) and FLEX variants identically.
  function _parseRosterPositions(arr){
    if (!Array.isArray(arr) || !arr.length) return null;
    var nmap = {
      QB:'QB', RB:'RB', WR:'WR', TE:'TE',
      FLEX:'FLEX', WRRB_FLEX:'FLEX', REC_FLEX:'FLEX', WRRBTE_FLEX:'FLEX',
      SUPER_FLEX:'SF', SFLEX:'SF',
      K:'K', DEF:'DEF', DST:'DEF', BN:'BN'
    };
    var map = {};
    arr.forEach(function(pos){
      var key = nmap[String(pos || '').toUpperCase()];
      if (key) map[key] = (map[key] || 0) + 1;   // IDP/TAXI/IR positions are ignored
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
    _resetTransient();
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    save();
    showMain();
    loadPlayers();
  }

  function showMain(){
    _boardSig = null;   // always force a full board rebuild when entering the draft view
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
    document.getElementById('drBaList').innerHTML = '<div class="dr-loading">Loading players…</div>';
    document.getElementById('drMain').style.display = '';
  }
  function showSetup(){
    endSim();
    document.getElementById('drMain').style.display = 'none';
    document.getElementById('drSetup').style.display = '';
    var hero = document.getElementById('drHero'); if (hero) hero.style.display = '';
    // Reflect the active draft's scoring in the setup controls when editing.
    if (state && state.scoring){
      var pprEl = document.getElementById('drPpr'); if (pprEl) pprEl.value = String(state.scoring.ppr != null ? state.scoring.ppr : 1);
      var tepEl = document.getElementById('drTep'); if (tepEl) tepEl.value = String(state.scoring.tep != null ? state.scoring.tep : 0);
    }
    renderSetupRoster();
    renderSetupCapital();
  }

  // ── Data ─────────────────────────────────────────────────────────────────
  function redraftVal(p){
    return (state.sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
                     : p.redraft_value_1qb) || 0;
  }
  function valOf(p){
    if (state.type === 'redraft') return redraftVal(p);
    // p.val is the stripped pick-object field (stored by commitPick/applyLivePicks);
    // p.value is the full player-object field from /api/league-players.
    return state.sf ? (p.sf_value || p.value || p.val || 0) : (p.value || p.val || 0);
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

  function loadPlayers(){
    var params = wantsKDef() ? ['kdef=1'] : [];
    // Historical/synced completed draft: grade against THAT season's ADP. The
    // server overlays that season's Sleeper ADP and no-ops for the current one.
    if (state && state.mode === 'live' && state.isComplete && state.season){
      params.push('season=' + encodeURIComponent(state.season));
    }
    var url = '/api/league-players' + (params.length ? ('?' + params.join('&')) : '');
    fetch(url, { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        tierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};
        adpSources = (!Array.isArray(resp) && resp.adp_sources) ? resp.adp_sources : {};
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
        render();
        if (sim) scheduleSim();   // begin CPU picks once players are loaded
      })
      .catch(function(){
        document.getElementById('drBaList').innerHTML =
          '<div class="dr-loading">Could not load players. Refresh to retry.</div>';
      });
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render(){
    if (state && !state.queue) state.queue = [];
    renderStatus(); renderBoard(); renderSide(); justPick = null; save();
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
    // K/DEF have no ADP: synthesize one in the last ~3 rounds so CPUs draft them
    // late, best (by projected PPG) first, instead of dumping them all at the end.
    if (pos === 'K' || pos === 'DEF'){
      var tot = (state.teams || 12) * (state.rounds || 16);
      var sc = _ppgScale[pos], v = ppgOf(p);
      var n = (sc && v != null && sc.elite > sc.repl) ? clamp01((v - sc.repl) / (sc.elite - sc.repl)) : 0.4;
      return tot - Math.round(n * (state.teams || 12) * 2.0);
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
    var qualByPos = {}, picksList = [];
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (slotOnClock(pn, state.teams, state.order) !== slot) return;
      var mp = state.picks[k];
      picksList.push(mp);
      var pos = (mp.position || '').toUpperCase();
      var full = playersById[String(mp.id)];
      var v = full ? vorOf(full) : null;
      if (v == null || v > 0) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
    });
    var nextOwned = null, tot = state.teams * state.rounds;
    for (var pn2 = state.current + 1; pn2 <= tot; pn2++){
      if (slotOnClock(pn2, state.teams, state.order) === slot){ nextOwned = pn2; break; }
    }
    return { qualByPos: qualByPos, picksList: picksList, nextOwned: nextOwned };
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
    // Score every candidate from THIS CPU team's perspective (its own roster,
    // depth, and next pick) so need is judged for the right team, not the viewer.
    var cpuCtx = _cpuCtx(slot);
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
    cands.forEach(function(c){
      var p = c.p, w = c.w, a = c.a, pv = c.pv;
      var pos = (p.position||'').toUpperCase();
      var t = targets[pos] || 0, have = counts[pos] || 0;
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
      // K/DEF: in the final 3 rounds, teams MUST fill these slots or they go empty.
      // Boost weight strongly so K/DEF crack the top-8 candidate sample and actually
      // get drafted, rather than losing out to late-sliding skill players every pick.
      if ((pos === 'K' || pos === 'DEF') && (t > 0) && (have < t) && _remainRds <= 3){
        w *= 8;
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
    cands.sort(function(x, y){ return y.w - x.w; });
    var top = cands.slice(0, Math.min(cands.length, 8));
    var sum = 0; top.forEach(function(c){ sum += c.w; });
    if (sum <= 0) return top[0].p;
    var roll = Math.random() * sum;
    for (var i = 0; i < top.length; i++){ roll -= top[i].w; if (roll <= 0) return top[i].p; }
    return top[0].p;
  }
  // Pick the highest-scored available player for my roster (used by auto-draft).
  function autoPick(){
    var pool = availablePool();
    if (!pool.length) return null;
    // Roster-need guard: K/DEF have no pickScore (null -> 0), so they'd never be
    // auto-drafted and your team would finish without a kicker/defense while every
    // CPU team fills theirs. Mirror the CPU's late-round behavior: when you still
    // have unfilled required K/DEF slots and your remaining picks are running out,
    // grab the best available one first so the slot doesn't go empty.
    var _kd = _autoKDefNeed(pool);
    if (_kd) return _kd;
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
    var scored = pool.map(function(p){
      var s = pickScoreFor(p) || 0;
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
      // Roster-need shaping (mirrors the CPU's simPick guards): never reach ahead
      // of ADP for a backup at a single-start position (a 2nd TE in 1TE, a 2nd QB
      // in 1QB), deprioritize a position once you're past a realistic depth, and
      // pull toward an OPEN dedicated starter slot (e.g. a 2nd starting RB).
      if (s > 0 && pos !== 'K' && pos !== 'DEF'){
        var target = targets[pos] || 0;
        var sSlots = rs[pos] || 0;             // dedicated starter slots (RB=2, TE=1, ...)
        var adp = adpOf(p);
        var backupReach = adp != null && adp < 9000 && state.current < adp && (
          (pos === 'QB' && have >= qbStarters) ||
          (pos === 'TE' && (rs.TE || 0) <= 1 && have >= 1)
        );
        if (backupReach) s = 0;
        else if (target > 0 && have >= target) s *= 0.4;   // past realistic depth
        else if (sSlots > 0 && have < sSlots) s *= 1.35;    // open dedicated starter
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
  function _doAutoPick(){
    var ap = autoPick();
    if (ap){ commitPick(ap); render(); scheduleSim(); }
  }
  function scheduleSim(){
    if (!sim || simPaused || !simStarted) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); return; }
    if (isMyPick(state.current)){
      if (simAutoDraft){ clearTimeout(simTimer); simTimer = setTimeout(_doAutoPick, simSpeed); }
      return;
    }
    clearTimeout(simTimer);
    simTimer = setTimeout(simStep, simSpeed);
  }
  function simStep(){
    if (!sim || simPaused || !simStarted) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); render(); return; }
    if (isMyPick(state.current)){
      if (simAutoDraft){ clearTimeout(simTimer); simTimer = setTimeout(_doAutoPick, simSpeed); return; }
      render(); return; // your turn - wait for manual pick
    }
    var p = simPick();
    if (!p){ endSim(); render(); return; } // pool exhausted - stop, don't spin forever
    commitPick(p); render();
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
    posFilter = 'ALL';
    justPick = null;
    _liveSig = null; _pickLagMsg = null;
    _pollCount = 0; _pollLastAt = 0; _pollNextAt = 0;
    _boardSig = null;
    _summaryShown = false;
    lastLivePicks = null;
  }
  function toggleSim(){
    simPaused = !simPaused;
    document.getElementById('drSimToggle').textContent = simPaused ? 'Resume' : 'Pause';
    if (simPaused) clearTimeout(simTimer); else scheduleSim();
  }
  // Reflect the current mock state on the status-bar controls.
  function syncSimControls(){
    var start = document.getElementById('drSimStart');
    var tg = document.getElementById('drSimToggle');
    var sp = document.getElementById('drSimSpeed');
    var ab = document.getElementById('drAutoBtn');
    var ready = sim && !simStarted;
    var running = sim && simStarted;
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
    renderSide();
    scheduleSim();
  }
  function startMock(){
    _resetTransient();
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    state.mode = 'mock';
    state.simStarted = false;
    sim = true; simPaused = false; simStarted = false;
    var sp = document.getElementById('drSimSpeed');
    simSpeed = parseInt(sp.value, 10) || 700;
    syncSimControls();
    _setUpcomingMode(false);
    save();
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
    var flex = rs.FLEX||0, sf = rs.SF||0, bn = rs.BN||0;
    // Targets are depth SUGGESTIONS, not hard needs. A deep startup bench (drafts
    // run 20+ rounds) shouldn't imply you "need" 8 RBs or 10 WRs, so the bench
    // contribution is capped and each position is held to a realistic ceiling.
    var benchEff = Math.min(bn, 7);
    var t = {
      QB: (rs.QB||0) + sf        + Math.round(benchEff * 0.10),
      RB: (rs.RB||0) + flex      + Math.round(benchEff * 0.35),
      WR: (rs.WR||0)             + Math.round(benchEff * 0.40),
      TE: (rs.TE||0)             + Math.round(benchEff * 0.15)
    };
    // TE Premium scoring nudges the build toward a second startable TE.
    var tep = scoringCfg().tep;
    if (tep > 0) t.TE += 1;
    // Sane ceilings so the assistant never frames an absurd amount of depth as a need.
    // (1QB backup-QB timing is handled per-team in the sim, relative to when each
    // team drafted its starter - see simPick - not by a blanket round cap here.)
    var cap = { QB: sf ? 3 : 2, RB: 6, WR: 6, TE: tep > 0 ? 3 : 2 };
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
  function isTierCliff(p){
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
    // Roster-derived starter counts, shared with the server grade so both anchor
    // replacement to the same numbers (a hardcoded server guess was the main
    // cause of WR/RB grade mismatches).
    var starters = BRPickScore.starterCounts(rs);
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
    var flex = rs.FLEX || 0, sf = rs.SF || 0;
    var starters = {
      QB: (rs.QB || 0) + sf * 0.5,
      RB: (rs.RB || 0) + flex * 0.5,
      WR: (rs.WR || 0) + flex * 0.5,
      TE: (rs.TE || 0),
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
    var pos = (p.position || '').toUpperCase();
    var v = ppgOf(p);
    var sc = _ppgScale[pos];
    if (v == null || !sc) return null;
    var span = sc.elite - sc.repl;
    if (span <= 0) return clamp01(v / Math.max(sc.elite, 1));
    return clamp01((v - sc.repl) / span);
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
    var qualByPos = {};
    myPicksList().forEach(function(mp){
      var pos = (mp.position || '').toUpperCase();
      var full = playersById[String(mp.id)];
      var v = full ? vorOf(full) : null;
      if (v == null || v > 0) qualByPos[pos] = (qualByPos[pos] || 0) + 1;
    });
    _psCtxCache = { targets: targets, qualByPos: qualByPos };
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
    var z = (pn - center) / sigma;
    var prob = 1 - _normCdf(z);
    // A position going on a run is less likely to make it back to you.
    var runPen = m.run[(p.position || '').toUpperCase()] || 0;
    if (runPen > 0) prob *= (1 - runPen);
    return Math.round(prob * 100);
  }
  function availColor(pct){ return pct >= 65 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'; }

  // ── Best-at-position chips + scarcity bar ───────────────────────────────────
  function renderBestChips(){
    var el = document.getElementById('drBestChips');
    if (!el) return;
    if (sideTab !== 'best' && sideTab !== 'rec'){ el.style.display = 'none'; return; }
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
    function cmpCol(p, other){
      var pos = (p.position || '').toUpperCase();
      var adp = adpOf(p), oadp = adpOf(other);
      var vor = p.vorp != null ? Number(p.vorp) : vorOf(p);
      var ovor = other.vorp != null ? Number(other.vorp) : vorOf(other);
      var vorLbl = (p.vorp != null || other.vorp != null) ? 'VORP' : 'VOR';
      var ps = pickScoreFor(p), ops = pickScoreFor(other);
      var v = valOf(p), ov = valOf(other);
      var t = tierOf(p), ot = tierOf(other);
      var ppg = p.proj_ppg != null ? Number(p.proj_ppg) : (p.ppg != null ? Number(p.ppg) : null);
      var oppg = other.proj_ppg != null ? Number(other.proj_ppg) : (other.ppg != null ? Number(other.ppg) : null);
      var ppgRowLbl = (p.proj_ppg != null || other.proj_ppg != null) ? 'Proj PPG' : 'PPG';
      var age = p.age != null ? Number(p.age) : null;
      var oage = other.age != null ? Number(other.age) : null;
      function statRow(lbl, val, oval, higherBetter, fmtFn){
        if (val == null && oval == null) return '';
        var vStr = fmtFn ? fmtFn(val) : (val != null ? String(val) : '-');
        var win = val != null && oval != null && (higherBetter ? val > oval : val < oval);
        return '<div class="dr-cmp-stat' + (win ? ' win' : '') + '">'
          + '<span class="dr-cmp-stat-lbl">' + lbl + '</span>'
          + '<span class="dr-cmp-stat-val">' + vStr + '</span></div>';
      }
      var sc = ps != null ? psColor(ps) : 'var(--text-muted)';
      return '<div class="dr-cmp-player">'
        + '<div class="dr-cmp-top"><img class="dr-cmp-hs" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div><div class="dr-cmp-name"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span> ' + esc(p.name) + '</div>'
        + '<div class="dr-cmp-meta">' + esc(p.team || '') + (p.age ? ' &middot; Age ' + Number(p.age).toFixed(0) : '') + '</div>'
        + '</div></div>'
        + '<div class="dr-cmp-ps" style="color:' + sc + '">' + (ps != null ? ps : '&ndash;') + '</div>'
        + '<div class="dr-cmp-ps-lbl">Pick Score</div>'
        + '<div class="dr-cmp-stats">'
        + statRow('Value', v, ov, true, function(x){ return x != null ? Math.round(x) : '-'; })
        + statRow(ppgRowLbl, ppg, oppg, true, function(x){ return x != null ? x.toFixed(1) : 'N/A'; })
        + statRow(vorLbl, vor, ovor, true, function(x){ return x != null ? (x >= 0 ? '+' + (Number.isInteger(x) ? x : x.toFixed(1)) : (x.toFixed ? x.toFixed(1) : String(x))) : '-'; })
        + statRow('ADP', adp, oadp, false, function(x){ return x != null ? Number(x).toFixed(1) : 'N/A'; })
        + (state.type !== 'redraft' ? statRow('Tier', t, ot, false, function(x){ return x != null ? 'T' + x : '-'; }) : '')
        + statRow('Age', age, oage, false, function(x){ return x != null ? x.toFixed(0) : '-'; })
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
  // run from a CPU team's perspective during simulation: { qualByPos, nextOwned,
  // picksList }. When omitted, pickScore scores for the viewer's own team exactly
  // as before.
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
    // differently. This wrapper only gathers inputs.
    var grading = !!(opts && opts.grading);

    // Quality-adjusted need: two below-replacement RBs still leave a real need.
    var _ctx = psCtx();
    var _qualByPos = (opts && opts.qualByPos) || _ctx.qualByPos;
    var _t = _ctx.targets[pos];
    var needRaw = _t ? clamp01(Math.max(0, _t - (counts[pos] || 0)) / _t) : 0;
    var _qualNeed = _t ? clamp01(Math.max(0, _t - (_qualByPos[pos] || 0)) / _t) : 0;
    needRaw = Math.max(needRaw, _qualNeed);

    // Position-normalized PPG (null -> the formula falls back to value).
    var ppgN = ppgNormOf(p);

    // Live-draft-only timing terms; grading passes neither, matching the server.
    var survivalAdj = 0;
    if (!grading){
      var nextOwned = (opts && opts.nextOwned !== undefined) ? opts.nextOwned : nextOwnedAfterCurrent();
      if (nextOwned){
        var survProb = availProb(p, nextOwned);
        if (survProb != null) survivalAdj = 0.05 - survProb / 100 * 0.08;
      }
    }
    var handcuff = false;
    if (!grading && state.type === 'redraft' && pos === 'RB'){
      var myRBTeams = {};
      ((opts && opts.picksList) || myPicksList()).forEach(function(mp){
        if ((mp.position || '').toUpperCase() === 'RB' && mp.team) myRBTeams[mp.team] = true;
      });
      if (p.team && myRBTeams[p.team]) handcuff = true;
    }

    var _sc = scoringCfg();
    return BRPickScore.computePickScore({
      pos: pos, value: valOf(p), vor: vorOf(p), tier: tierOf(p),
      age: (p.age != null ? Number(p.age) : null), rankChange7d: p.rank_change_7d,
      avgPick: adp, pickNo: state.current, maxVal: maxVal,
      draftType: state.type, isSf: state.sf, needRaw: needRaw,
      qbCount: counts['QB'] || 0, totalPicks: (state.teams || 12) * (state.rounds || 16),
      numTeams: state.teams || 12, ppgNorm: ppgN,
      ppr: _sc.ppr, tep: _sc.tep, isTierCliff: isTierCliff(p),
      survivalAdj: survivalAdj, handcuff: handcuff,
    });
  }
  // How many players remain in this player's (position|tier) bucket.
  function tierRemaining(p){
    var t = tierOf(p); if (t == null) return null;
    return _ptc[(p.position || '').toUpperCase() + '|' + t] || 0;
  }
  function pickReason(p, counts){
    var pos = (p.position || '').toUpperCase();
    var t = psCtx().targets[pos];
    var need = t ? Math.max(0, t - (counts[pos] || 0)) : 0;
    var adp = adpOf(p);
    var fell = (adp != null) ? Math.round(state.current - adp) : null;
    var relGap = (adp != null) ? ((state.current - adp) / Math.max(adp, 1.5)) : null;
    var tier = tierOf(p);
    var left = tierRemaining(p);
    // QB overfill: only a warning early, when a skill player is the better use of
    // the pick. Late rounds: a backup QB is fine, so fall through to a normal reason.
    if (!state.sf && pos === 'QB' && (counts['QB'] || 0) >= 1
        && Math.ceil(state.current / (state.teams || 12)) <= 6){
      return 'Starting QB already filled, this pick could be a skill player';
    }
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
    var ps = pickScoreFor(p);
    var sub = adp != null ? 'ADP ' + Number(adp).toFixed(1) : '';
    var reasonLine = opts.reason ? '<div class="dr-ba-reason">' + esc(opts.reason) + '</div>' : '';
    var waitLine = opts.wait
      ? '<div class="dr-ba-wait">Can wait: ' + opts.wait.prob + '% there at #' + opts.wait.pn + '</div>'
      : '';
    var psChip = (ps != null)
      ? '<div class="dr-ba-pschip" style="color:' + psColor(ps) + ';background:' + psColor(ps) + '1a;">' + ps + '<small>PS</small></div>'
      : '';
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
    // Projected PPG for meta line (proj_ppg = upcoming season, ppg = last season fallback)
    var ppgNum = p.proj_ppg != null ? Number(p.proj_ppg) : (p.ppg != null ? Number(p.ppg) : null);
    var ppgPart = ppgNum != null ? ' · ' + ppgNum.toFixed(1) + ' Proj PPG' : '';
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
    if (!q.length){ listInto('<div class="dr-empty-note">Your queue is empty. Tap the ☆ on any player to add a target.</div>'); return; }
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
    var kdef = wantsKDef();
    var kbtns = document.querySelectorAll('.dr-pos-kdef');
    for (var i = 0; i < kbtns.length; i++){ kbtns[i].style.display = kdef ? '' : 'none'; }
    var bc = document.getElementById('drBestControls');
    if (bc) bc.style.display = (sideTab === 'best') ? '' : 'none';
    if (sideTab === 'rec')    return renderRec();
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
    if (!hasOwned()){ listInto('<div class="dr-empty-note">Set your pick slot to get personalized recommendations.</div>'); return; }
    var counts = myPosCounts();
    var pool = availablePool().slice();
    if (!pool.length){ listInto('<div class="dr-empty-note">No players available.</div>'); return; }
    var maxVal = 0; pool.forEach(function(p){ var v = valOf(p); if (v > maxVal) maxVal = v; });
    pool.forEach(function(p){ p._ps = pickScore(p, maxVal, counts); });
    pool.sort(function(a, b){ return b._ps - a._ps; });
    var html = balanceAlert() + alertBanners();
    // Assistant looks across your whole draft capital: a player you can likely
    // get at a later owned pick is flagged so you can spend this pick elsewhere.
    var nextPick = nextOwnedAfterCurrent();
    for (var i = 0; i < Math.min(pool.length, 50); i++){
      var p = pool[i];
      var opts = { reason: pickReason(p, counts) };
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
  // (Sleeper PPG, else FantasyPros projection). When a player has no projection
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
      var psBadge = (p.ps != null) ? '<span class="dr-rslot-ps" style="color:' + psColor(p.ps) + '">' + p.ps + '</span>' : '';
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
  // Projected PPG (upcoming season) preferred; last-season actual as fallback.
  function ppgOf(p){ return (p && p.proj_ppg != null) ? Number(p.proj_ppg) : ((p && p.ppg != null) ? Number(p.ppg) : null); }
  function gradeTeam(){
    if (!hasOwned()) return null;
    // Pull "your" grade from the full field so the relative-to-league curve matches
    // what the League tab shows. Before there are enough teams on the board to curve
    // against, gradeAllTeams returns the raw (uncurved) grades, which is also correct.
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
    var countsSoFar = { QB: 0, RB: 0, WR: 0, TE: 0 };
    var picks = []; // { id, pos, ps, val, ppg }
    mine.forEach(function(m){
      var pos = (m.p.position || '').toUpperCase();
      // Grade score: recompute at the pick number in grading mode (no live
      // survival/handcuff timing terms) so it measures pick quality and matches
      // the server's compute_pick_score exactly. Falls back to the stored live
      // score only when the player is no longer resolvable.
      var full = playersById[String(m.p.id)];
      var ps = null;
      if (players.length > 0 && _gmaxVal > 0 && full){
        var _saved = state.current;
        state.current = m.pn;
        ps = pickScore(full, _gmaxVal, countsSoFar, { grading: true });
        state.current = _saved;
      }
      if (ps == null) ps = m.p.ps;
      if (countsSoFar[pos] != null) countsSoFar[pos]++;
      if (counts[pos] != null) counts[pos]++;
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
      var teamLetter = teamLetterFromPicks(_letters);
      var rv = letterToScore(teamLetter);
      return { score: rv, value: avgPs != null ? Math.round(avgPs) : 50,
        balance: 0, tier: 0, count: mine.length,
        avgPs: avgPs ? Math.round(avgPs) : null, window: null };
    }

    // Startup / redraft composite (Value 35 / Starters 35 / Construction 30)
    // lives in static/draft_grade_team.js (BRTeamGrade), shared with the Python
    // server grade and pinned by a parity test. K/DEF are excluded from grading.
    var _slots = lineupSlots().filter(function(s){ return s !== 'K' && s !== 'DEF'; });
    var _leaguePpg = [];
    players.forEach(function(q){ var v = ppgOf(q); if (v != null) _leaguePpg.push(v); });
    var _leagueVal = players.map(function(q){ return valOf(q); });
    var _comp = BRTeamGrade.teamGradeComposite(
      picks, _slots, posTargets(), state.teams || 12, state.type, _leaguePpg, _leagueVal
    );
    if (!_comp){
      return { score: 0, value: 0, balance: 0, tier: 0, count: mine.length,
        avgPs: avgPs != null ? Math.round(avgPs) : null, window: null };
    }
    var _starterArr = picks.filter(function(x){ return _comp.starterIds[String(x.id)]; });
    return { score: _comp.total, value: _comp.value, balance: _comp.balance, tier: _comp.starter,
      count: mine.length, avgPs: avgPs != null ? Math.round(avgPs) : null,
      strength: Math.round(_comp.strengthRatio * 100),
      window: _competitiveWindow(_starterArr) };
  }
  // Grade every team in the draft, sorted best-first. Picks are attributed by
  // OWNERSHIP, not by the board column they sit in: every pick the user owns (a
  // 1.01 traded in, a pick in another seat, etc.) belongs to the single "You"
  // team, and each seat shows only the picks it still owns. Otherwise a user who
  // owns picks across two seats would show up as "You" twice.
  function gradeAllTeams(){
    if (!state) return [];
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
  // Curve grades relative to the actual field so real differences between teams span
  // a meaningful range instead of all clustering in one letter band. This does NOT
  // fabricate gaps: each team's adjustment is proportional to how far its real
  // composite score sits from the field mean, normalized by the field's own spread.
  // A genuinely even field (low variance) stays tight; a field with real separation
  // spreads out. The raw score is preserved on grade.rawScore.
  function _applyFieldCurve(out){
    if (!state || state.type === 'rookie') return;  // rookie grades are absolute (Teams-page letters), not curved
    if (!out || out.length < 3) return;             // need a real field to curve against
    // Curve math lives in static/draft_grade_curve.js (shared with the Python
    // server grade via a parity test) so both surfaces stay identical.
    var _teams = state.teams || 12;
    var roundsDone = Math.floor(((state.current || 1) - 1) / _teams);
    var raws = out.map(function(t){ return t.grade.score; });
    var curved = BRDraftGrade.curveFieldScores(raws, roundsDone);
    out.forEach(function(t, i){
      t.grade.rawScore = t.grade.score;
      t.grade.score = curved[i];
    });
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
  function gradeMax(){
    return (state.type === 'rookie') ? { value:100, balance:0, tier:0 }
                                     : { value:35, balance:30, tier:35 };
  }
  function gradeBars(g){
    var m = gradeMax();
    if (state.type === 'rookie') return gradeBar('Avg Pick Score', g.value, 100, 'The bar shows average 0-100 pick score (same chips shown on each pick). The letter grade uses the BPA/ADP system: did you reach, and was a better player available?');
    // g.tier holds the starting-lineup strength component.
    return gradeBar('Value', g.value, m.value, 'How strong your picks are by pick score, weighted toward the earlier rounds.')
      + gradeBar('Starters', g.tier, m.tier, 'How good your projected starting lineup is versus a league-average team.')
      + gradeBar('Construction', g.balance, m.balance, 'How well you’ve filled your starting slots and balanced positions.');
  }

  function renderNeeds(){
    if (!hasOwned()){ listInto('<div class="dr-empty-note">Set your pick slot to see your team build.</div>'); return; }
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
      html += '<div class="dr-grade-card"><div class="dr-grade-letter">' + gradeLetter(g.score) + '</div>'
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
    // Roster projection: use Sleeper ppg (preferred) or proj_ppg fallback
    function _pPpg(p){ return p.ppg != null ? Number(p.ppg) : (p.proj_ppg != null ? Number(p.proj_ppg) : null); }
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
      listInto('<div class="dr-empty-note">No picks yet - grades will appear as teams draft.</div>');
      return;
    }
    if (allTeams.length < 2){
      listInto('<div class="dr-empty-note">Grades appear once at least 2 teams have drafted.</div>');
      return;
    }
    var _rc = ['gold','silver','bronze'];
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
      html += '<div class="dr-sum-lrow' + (t.isMe ? ' is-me' : '') + '" data-legslot="' + t.slot + '">'
        + rankCell
        + '<span class="dr-sum-lname">' + esc(t.name) + stratTag + '</span>'
        + winTag
        + '<span class="dr-sum-lgrade" style="color:' + tCol + '">' + gradeLetter(t.grade.score) + '</span>'
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
            var _ps = storedPickScore(pn, p);
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
          roster: _parseRosterPositions(d.roster_positions)
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
    el.innerHTML = '<span class="dr-poll-dot"></span>' + (_fmtAgo(_pollLastAt) || '&mdash;') + lagSuffix;
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

  function renderStatus(){
    var total = state.teams * state.rounds;
    var done = state.current > total;
    var r = Math.ceil(state.current / state.teams);
    var pickInRound = ((state.current - 1) % state.teams) + 1;
    document.getElementById('drPickPill').textContent = done ? 'Done' : ('Pick: ' + r + '.' + (pickInRound < 10 ? '0' : '') + pickInRound);
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
    if (g){ gp.style.display = ''; gp.textContent = 'Grade ' + gradeLetter(g.score); } else { gp.style.display = 'none'; }
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
    // Display the GRADE score (no live survival/handcuff timing terms) so every
    // per-pick chip matches the Teams-page grade. Memoized separately from the
    // live pl.ps (which the sim stores with timing terms for its own use).
    if (pl.gps != null) return pl.gps;
    var full = playersById[String(pl.id)];
    if (!full || !players.length) return null;
    var maxVal = 0; players.forEach(function(q){ var v = valOf(q); if (v > maxVal) maxVal = v; });
    if (maxVal <= 0) return null;
    // Owning team's positional counts from this team's earlier picks.
    var owner = (state.pickOwners && state.pickOwners[pn] != null)
      ? state.pickOwners[pn] : slotOnClock(pn, state.teams, state.order);
    var counts = { QB:0, RB:0, WR:0, TE:0 };
    Object.keys(state.picks).forEach(function(k){
      var kp = parseInt(k, 10);
      if (kp >= pn || !state.picks[k]) return;
      var o2 = (state.pickOwners && state.pickOwners[kp] != null)
        ? state.pickOwners[kp] : slotOnClock(kp, state.teams, state.order);
      if (o2 !== owner) return;
      var pos2 = (state.picks[k].position || '').toUpperCase();
      if (counts[pos2] != null) counts[pos2]++;
    });
    var saved = state.current;
    state.current = pn;
    var ps = pickScore(full, maxVal, counts, { grading: true });
    state.current = saved;
    pl.gps = ps;   // memoize the grade score (matches gradePicks / the server)
    return ps;
  }
  function cellInner(pn){
    var pl = state.picks[pn];
    var h = '<span class="dr-cell-num">' + pn + '</span>';
    if (isMyPick(pn)) h += '<span class="dr-cell-mineflag">YOU</span>';
    var _own = tradedOwnerLabel(pn);
    if (_own) h += '<span class="dr-cell-owner">' + esc(_own) + '</span>';
    if (pl){
      if (_cellShowPs) {
        var _cvps = storedPickScore(pn, pl);
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
        html += '<div class="' + cellClass(pn) + '" id="dc' + pn + '" data-pn="' + pn + '">' + cellInner(pn) + '</div>';
      }
    }
    board.innerHTML = html;
    _boardSig = boardSig();
    refreshCurrent();
  }
  function paintCell(pn){
    var el = document.getElementById('dc' + pn);
    if (!el) return;
    el.className = cellClass(pn);
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

  function renderBA(){
    var srcEl = document.getElementById('drAdpSrc');
    if (srcEl){ srcEl.textContent = 'ADP source: ' + (adpSources[state.type] || 'unavailable'); }
    var sortBy = document.getElementById('drBaSort').value;
    var q = (document.getElementById('drSearch').value || '').trim().toLowerCase();
    var pool = availablePool().filter(function(p){
      if (posFilter !== 'ALL' && String(p.position||'').toUpperCase() !== posFilter) return false;
      if (q && String(p.name||'').toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    function steal(p){ var a = adpOf(p); return (a != null) ? (state.current - a) : -99999; }
    if (sortBy === 'ps'){
      var _pcounts = myPosCounts();
      var _pmaxV = 0; pool.forEach(function(p){ var v = valOf(p); if (v > _pmaxV) _pmaxV = v; });
      pool.forEach(function(p){ p._ps = pickScore(p, _pmaxV, _pcounts); });
    }
    pool.sort(function(a, b){
      // K/DEF have no value/ADP/PS - order them among themselves by projected PPG
      // so the best kicker/defense still surfaces first regardless of sort mode.
      var aKd = (String(a.position||'').toUpperCase() === 'K' || String(a.position||'').toUpperCase() === 'DEF');
      var bKd = (String(b.position||'').toUpperCase() === 'K' || String(b.position||'').toUpperCase() === 'DEF');
      if (aKd && bKd) return (ppgOf(b) || 0) - (ppgOf(a) || 0);
      if (sortBy === 'adp'){
        var aa = adpOf(a), ba = adpOf(b);
        return (aa != null ? aa : 99999) - (ba != null ? ba : 99999);
      }
      if (sortBy === 'steals'){ return steal(b) - steal(a); }
      if (sortBy === 'ps'){ return (b._ps || 0) - (a._ps || 0); }
      return valOf(b) - valOf(a);
    });
    if (!pool.length){ listInto('<div class="dr-empty-note">No players match.</div>'); return; }
    var nextPick = hasOwned() ? nextOwnedAfterCurrent() : null;
    var html = balanceAlert() + alertBanners();
    // K/DEF have no startup ADP so they sort to the very end and fall past the
    // 200-player cap. Separate them out so they always render after skill players.
    var _isKD = function(p){ var pos = String(p.position||'').toUpperCase(); return pos === 'K' || pos === 'DEF'; };
    var mainPool = (posFilter === 'ALL' && wantsKDef()) ? pool.filter(function(p){ return !_isKD(p); }) : pool;
    var kdPool  = (posFilter === 'ALL' && wantsKDef()) ? pool.filter(_isKD) : [];
    for (var i = 0; i < Math.min(mainPool.length, 200); i++){
      var p = mainPool[i];
      var opts = {};
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
  function pickScoreFor(p){
    var pool = availablePool();
    var maxVal = 0; pool.forEach(function(x){ var v = valOf(x); if (v > maxVal) maxVal = v; });
    return pickScore(p, maxVal, myPosCounts());
  }

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  // ── Glossary / inline term explainers ───────────────────────────────────────
  // Single source of truth so the inline ⓘ tooltips and the help popover agree.
  var _GLOSSARY = [
    { term: 'Pick Score (PS)', def: 'A 0-100 grade of how good this pick is at this exact slot. Blends the player’s value, how far they fell vs ADP, positional tier, your roster needs, age, and projected points. Late-round scores are re-scaled so a strong slider isn’t buried. Kickers and defenses aren’t scored.' },
    { term: 'Value', def: 'The player’s trade value as an asset on a 0-999 scale - dynasty value for startup/rookie drafts, redraft value for redraft.' },
    { term: 'VOR / VORP', def: 'Value Over Replacement: how much better a player is than a replacement-level starter at their position (a fixed, preseason-style baseline). VORP uses real fantasy points; VOR uses dynasty value.' },
    { term: 'ADP', def: 'Average Draft Position - the typical overall pick a player goes at in real drafts. If it’s below your current pick, they’ve fallen and may be a value.' },
    { term: 'Tier', def: 'Players grouped by talent gaps (Tier 1 = elite). A tier “cliff” means only a couple of players remain before a real drop-off at that position.' },
    { term: 'Steals (sort)', def: 'Orders the board by who has fallen the furthest past their ADP - the biggest available bargains.' },
    { term: 'PPG', def: 'Points per game - projected for the upcoming season, or last season’s actual when that’s shown.' },
    { term: 'Survival %', def: 'The chance a player is still on the board at your next pick. Starts from consensus ADP, then adapts to how your draft is actually going - if the room is reaching, letting players slide, drafting unpredictably, or running on a position, the odds shift to match (kicks in after the first several picks).' },
    { term: 'Grade · Value', def: 'How strong your picks are by pick score, weighted toward the earlier rounds where it matters most.' },
    { term: 'Grade · Starters', def: 'How good your projected starting lineup is versus a league-average team.' },
    { term: 'Grade · Construction', def: 'How well you’ve filled your starting slots and balanced your positions.' }
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
         + '<div class="dr-sum-grade-bars">' + gradeBars(g) + '</div>'
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
        var _ppgv = p.ppg != null ? Number(p.ppg) : (p.proj_ppg != null ? Number(p.proj_ppg) : null);
        if (_ppgv != null){ sumProjTotal += _ppgv; sumProjCount++; }
        var _fp = playersById[String(p.id)] || p;
        var _t = tierOf(_fp); if (_t != null && _t <= 2) sumT12++;
        if (p.ps != null){ sumAllPsTotal += p.ps; sumAllPsCount++; }
        if (_ssSet[String(p.id)] && p.ps != null){ sumStarterPsTotal += p.ps; sumStarterPsCount++; }
      });
      var _sits = [];
      if (sumProjCount >= 2) _sits.push({ v: sumProjTotal.toFixed(1), l: 'Proj PPG' });
      if (state.type !== 'redraft') _sits.push({ v: sumT12, l: 'T1-2 Picks' });
      if (state.type === 'rookie'){
        if (sumAllPsCount >= 1) _sits.push({ v: Math.round(sumAllPsTotal / sumAllPsCount), l: 'Avg PS' });
      } else {
        if (sumStarterPsCount >= 2) _sits.push({ v: Math.round(sumStarterPsTotal / sumStarterPsCount), l: 'Starter PS' });
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
        if (_arch) _aSections.push('<div class="dr-sum-arch-item"><div class="dr-sum-arch-tag">Archetype</div><div class="dr-sum-arch-label">' + esc(_arch.label) + '</div></div>');
        if (_win){
          _aSections.push('<div class="dr-sum-arch-item"><div class="dr-sum-arch-tag">Window</div><span class="dr-sum-win dr-win-' + _win.label.toLowerCase().replace('-','') + '">' + esc(_win.label) + '</span></div>');
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
      var psStr = (p.ps != null) ? '<span class="dr-sum-ps" style="color:' + psColor(p.ps) + '">' + p.ps + '</span>' : '';
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
      + (hasSlot ? '<button class="dr-btn dr-btn-primary" id="drSumShare">Share</button>' : '')
      + '<button class="dr-btn" id="drSumCloseBtn">Close</button>'
      + '</div>';

    var card = document.getElementById('drSummaryCard');
    card.innerHTML = html;
    document.getElementById('drSummary').style.display = '';
    document.getElementById('drSumClose').addEventListener('click', closeSummary);
    document.getElementById('drSumCloseBtn').addEventListener('click', closeSummary);
    var _shareBtn = document.getElementById('drSumShare');
    if (_shareBtn) _shareBtn.addEventListener('click', function(){ closeSummary(); shareDraft(); });
  }
  function closeSummary(){ document.getElementById('drSummary').style.display = 'none'; }

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
    return { value: Math.round(valOf(p)), proxy: p };
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
    msg.innerHTML = '<div class="dr-pt-title">Pick trade evaluator</div>'
      + '<div class="dr-pt-sub">Overall picks ("22") or round.pick ("2.10"), separated by commas or spaces. Each pick is priced as the player likely on the board there (by ADP on this draft’s remaining pool).</div>'
      + '<label class="dr-pt-lbl">You give</label>'
      + '<input id="drPtGive" class="dr-pt-input" autocomplete="off" placeholder="e.g. 14 or 1.07">'
      + '<label class="dr-pt-lbl">You get</label>'
      + '<input id="drPtGet" class="dr-pt-input" autocomplete="off" placeholder="e.g. 22, 27">'
      + '<div id="drPtResult" class="dr-pt-result"></div>';
    var btns = document.getElementById('drModalBtns');
    btns.innerHTML = '';
    var close = document.createElement('button');
    close.className = 'dr-btn'; close.textContent = 'Close';
    close.addEventListener('click', function(){ m.style.display = 'none'; });
    btns.appendChild(close);
    function sideHtml(nums){
      var tot = 0;
      var rows = nums.map(function(pn){
        var v = pickNumValue(pn);
        if (!v) return '<div class="dr-pt-row">#' + pn + ' — n/a</div>';
        tot += v.value;
        return '<div class="dr-pt-row">#' + pn + ' &asymp; <b>' + v.value + '</b>'
          + ' <span class="dr-pt-proxy">(' + esc(v.proxy.name) + ')</span></div>';
      }).join('');
      return { html: rows, tot: tot };
    }
    function evalNow(){
      var give = _parsePickNums(document.getElementById('drPtGive').value);
      var get  = _parsePickNums(document.getElementById('drPtGet').value);
      var out = document.getElementById('drPtResult');
      if (!give.length || !get.length){ out.innerHTML = ''; return; }
      var g = sideHtml(give), r = sideHtml(get);
      var diff = r.tot - g.tot;
      var even = Math.abs(diff) <= Math.max(20, g.tot * 0.05);
      var verdict = even ? 'Even trade'
        : (diff > 0 ? 'You win by ~' + diff : 'You lose by ~' + (-diff));
      var col = even ? 'var(--text-muted)' : (diff > 0 ? '#22c55e' : '#ef4444');
      out.innerHTML = '<div class="dr-pt-cols">'
        + '<div><div class="dr-pt-side-h">Give (' + g.tot + ')</div>' + g.html + '</div>'
        + '<div><div class="dr-pt-side-h">Get (' + r.tot + ')</div>' + r.html + '</div>'
        + '</div>'
        + '<div class="dr-pt-verdict" style="color:' + col + '">' + verdict + '</div>';
    }
    msg.querySelector('#drPtGive').addEventListener('input', evalNow);
    msg.querySelector('#drPtGet').addEventListener('input', evalNow);
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
    var ps = pickScoreFor(p);
    var reason = pickReason(p, myPosCounts());
    state.picks[pn] = { id: p.id, name: p.name, position: p.position, team: p.team, val: Math.round(valOf(p)), ps: ps, reason: reason };
    drafted[String(p.id)] = true;
    justPick = pn;
    state.current++;
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
      state = null;
      showSetup();
    }
    if (isLive){ doReset(); } else { drConfirm('Reset the draft board?', 'Reset', doReset); }
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
    var POSC = { QB:'#f59e0b', RB:'#22c55e', WR:'#3b82f6', TE:'#8b5cf6', K:'#94a3b8', DEF:'#64748b', FLEX:'#14b8a6', SF:'#a78bfa', BN:'#64748b' };
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
      ctx.fillText('Grade ' + gl + (gp ? ('  \xb7  ' + gp) : ''), pad, pad + 76);
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
    var adp = adpOf(p), t = tierOf(p), ps = pickScoreFor(p);
    var posRank = state.sf ? (p.sf_pos_rank_label || '') : (p.pos_rank_label || '');
    var adpGap = (adp != null) ? (state.current - adp) : null;
    var vsAdp = adpGap != null ? (adpGap >= 0 ? ('+' + Math.round(adpGap)) : String(Math.round(adpGap))) : '-';
    // Use real PPR-based VORP from API when available; fall back to dynasty-value VOR
    var vorpVal = (p.vorp != null) ? Number(p.vorp) : vorOf(p);
    var vorpLbl = (p.vorp != null) ? 'VORP' : 'VOR';
    var vorStr = (vorpVal != null) ? (vorpVal >= 0 ? '+' + (Number.isInteger(vorpVal) ? vorpVal : vorpVal.toFixed(1)) : String(vorpVal.toFixed ? vorpVal.toFixed(1) : vorpVal)) : '-';
    var pos = (p.position || '').toUpperCase();
    var scarce = posTopRemaining(pos);
    // Prefer forward-looking projected PPG; fall back to last season actual
    var ppg = null, ppgLbl = 'PPG', ppgSub = '';
    if (p.ppg != null){ ppg = Number(p.ppg); ppgLbl = 'PPG';
      ppgSub = p.ppg_rank != null ? (pos + p.ppg_rank) : (p.ppg_season ? String(p.ppg_season) : ''); }
    else if (p.proj_ppg != null){ ppg = Number(p.proj_ppg); ppgLbl = 'Proj PPG'; ppgSub = 'projected'; }
    var sc = ps != null ? psColor(ps) : 'var(--text-muted)';
    var pc = posColor(p.position);
    var c = document.getElementById('drPreviewCard');
    // Position-colored top accent
    c.style.boxShadow = '0 16px 50px rgba(0,0,0,.3), inset 0 3px 0 ' + pc;
    var agePart = (p.age != null) ? (' &middot; Age ' + Number(p.age).toFixed(0)) : '';
    var h = '<button class="dr-prev-close" id="drPrevClose" aria-label="Close">&times;</button>'
      // Player identity row
      + '<div class="dr-prev-top">'
      + '<img class="dr-prev-hs" src="' + playerImgUrl(p) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
      + '<div class="dr-prev-id"><div class="dr-prev-name">' + esc(p.name) + (t ? (' <span class="dr-tier' + (isTierCliff(p) ? ' dr-tier-cliff' : '') + '">T' + t + '</span>') : '') + '</div>'
      + '<div class="dr-prev-meta"><span class="dr-posbadge" style="background:' + pc + '">' + esc(p.position) + '</span> ' + esc(p.team || '') + (posRank ? (' &middot; ' + esc(posRank)) : '') + agePart + '</div>'
      + '</div></div>'
      // Pick Score hero
      + '<div class="dr-prev-score-hero" style="border-color:' + sc + ';background:' + sc + '1a;">'
      + '<div class="dr-prev-score-num" style="color:' + sc + '">' + (ps != null ? ps : '&ndash;') + '</div>'
      + '<div class="dr-prev-score-lbl">Pick Score' + infoIcon('A 0-100 grade of this pick at this slot: value, fall vs ADP, tier, your needs, age, and projected points. Higher is better.') + '</div>'
      + '<div class="dr-prev-score-reason">' + esc(ps != null ? pickReason(p, myPosCounts()) : 'Streamer / last-round pick') + '</div>'
      + '</div>'
      // Stats grid
      + '<div class="dr-prev-stats">'
      + statBox('Value', Math.round(valOf(p)), null, 'Trade value as an asset on a 0-999 scale (dynasty value, or redraft value in redraft).')
      + statBox(vorpLbl, vorStr, null, 'Value Over Replacement: how much better than a freely-available starter at this position. ' + (vorpLbl === 'VORP' ? 'Based on real fantasy points.' : 'Based on dynasty value.'))
      + statBox('ADP', adp != null ? Number(adp).toFixed(1) : '-', null, 'Average Draft Position - the typical overall pick this player goes at in real drafts.')
      + statBox('vs ADP', vsAdp, null, 'How far this player has fallen past their ADP at the current pick. Positive = a value.')
      + (ppg != null ? statBox(ppgLbl, ppg.toFixed(1), ppgSub, 'Points per game' + (ppgLbl === 'Proj PPG' ? ', projected for the upcoming season.' : ', last season actual.')) : statBox('Pos Rank', posRank || '-'))
      + statBox(pos + ' T1-2 left', scarce, null, 'How many Tier 1-2 (elite) players remain available at this position - a scarcity signal.')
      + '</div>';
    // Survival probability at the user's next upcoming pick
    var nextOwnedPick = nextOwnedAfterCurrent();
    if (nextOwnedPick){
      var prob = availProb(p, nextOwnedPick);
      if (prob != null){
        var col = availColor(prob);
        h += '<div class="dr-prev-avail-track">'
          + '<div class="dr-prev-avail-label">Survival at your next pick (#' + nextOwnedPick + ')</div>'
          + '<div class="dr-prev-avail-picks"><div class="dr-prev-avail-pick" style="background:' + col + '14;border:1px solid ' + col + '44;">'
          + '<span style="color:' + col + ';font-size:18px;font-weight:900;">' + prob + '%</span>'
          + '<span class="dr-prev-avail-pn">' + (prob >= 65 ? 'Likely available' : prob >= 40 ? 'Might be there' : 'Unlikely to last') + '</span>'
          + '</div></div></div>';
      }
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
  });
  var _myStratSel = document.getElementById('drMyStrat');
  if (_myStratSel) _myStratSel.addEventListener('change', function(){
    if (state){ state.myStrat = this.value || ''; save(); }
  });
  var _myLeanSel = document.getElementById('drMyAgeLean');
  if (_myLeanSel) _myLeanSel.addEventListener('change', function(){
    if (state){ state.myAgeLean = this.value || ''; save(); }
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
  document.getElementById('drEdit').addEventListener('click', showSetup);
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
  document.getElementById('drBaSort').addEventListener('change', renderBA);
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
      posFilter = pos;
      var btns = document.querySelectorAll('#drPosFilters .dr-pos');
      btns.forEach(function(b){ b.classList.toggle('active', b.getAttribute('data-pos') === pos); });
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
  document.getElementById('drHelpBtn').addEventListener('click', openGlossary);
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
    posFilter = b.getAttribute('data-pos');
    this.querySelectorAll('.dr-pos').forEach(function(x){ x.classList.toggle('active', x === b); });
    renderBA();
  });

  applyCfgDefaults();
  renderSetupRoster();
  renderSetupCapital();
  document.getElementById('drSf').addEventListener('change', function(){ _rosterMode = 'auto'; _setupRoster = null; renderSetupRoster(); });
  document.getElementById('drType').addEventListener('change', function(){
    // Reset roster to defaults for the new type, then re-render.
    _rosterMode = 'auto'; _setupRoster = null; renderSetupRoster();
    // Show/hide rookie rounds field; for non-rookie, rounds auto-sync from roster.
    var isRookie = this.value === 'rookie';
    var rf = document.getElementById('drRoundsField');
    if (rf) rf.style.display = isRookie ? '' : 'none';
    if (isRookie) document.getElementById('drRounds').value = String(cfg.numRoundsRookie || 3);
    renderSetupCapital();   // refresh claimed-pick list after rounds change
  });
  // Any control that changes the pick map resets claimed picks to the slot default.
  ['drTeams','drRounds','drOrder','drSlot'].forEach(function(idn){
    document.getElementById(idn).addEventListener('change', renderSetupCapital);
  });
  // Rounds <-> bench two-way sync: rounds = starters + bench.
  document.getElementById('drRounds').addEventListener('change', function(){
    var rounds = Math.max(1, Math.min(40, parseInt(this.value, 10) || 15));
    this.value = rounds;
    if (!_setupRoster) _setupRoster = defaultRoster();
    _rosterMode = 'custom';
    _setupRoster.BN = Math.max(0, rounds - _totalStarterSlots(_setupRoster));
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
    _rosterMode = 'custom';
    // Keep rounds = starters + bench in sync for every slot change.
    // Bench change -> update rounds. Starter change -> update rounds (bench stays).
    var _newRounds = Math.max(1, Math.min(40, _totalStarterSlots(_setupRoster) + (_setupRoster.BN || 0)));
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
  // peek (~14vh), mid (~38vh, default), and full (~92vh). Drag the grip handle
  // up/down; on release it snaps to the nearest point.
  (function initSheet(){
    var sheet = document.getElementById('drSide');
    var handle = document.getElementById('drSheetHandle');
    if (!sheet || !handle) return;
    var mq = window.matchMedia('(max-width: 900px)');
    var dragging = false, startY = 0, startT = 0, curT = 0, snapIdx = 1;
    function ih(){ return window.innerHeight; }
    // translateY offsets (px): full (whole 92vh sheet shows), mid (~36vh), peek (~12vh)
    function snaps(){ return [0, ih() * 0.42, ih() * 0.80]; }
    function applyT(t){ curT = t; sheet.style.transform = 'translateY(' + t + 'px)'; }
    function snapTo(idx){
      var pts = snaps();
      snapIdx = Math.max(0, Math.min(pts.length - 1, idx));
      sheet.classList.remove('dragging');
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
      var t = Math.max(0, Math.min(ih() * 0.86, startT + dy));
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
      else { sheet.style.transform = ''; sheet.classList.remove('dragging'); }
    }
    if (mq.addEventListener) mq.addEventListener('change', applyMode); else mq.addListener(applyMode);
    window.addEventListener('resize', function(){ if (mq.matches) snapTo(snapIdx); });
    applyMode();
  })();

  // Open a specific league draft directly: ?connect=<id> (from the site-wide
  // "Join Draft Room" banner) or ?live=<id> (from Draft History). Otherwise
  // resume the in-progress session draft.
  var _qs = new URLSearchParams(location.search);
  var urlLive = _qs.get('connect') || _qs.get('live');
  if (urlLive){
    connectLive(urlLive);
  } else {
    resumeFromSession();
  }
})();
