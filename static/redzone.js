// BR Redzone live-scoreboard module -- extracted from app.js so it only loads
// on the Redzone page (gated server-side via page_redzone). Runs deferred,
// after app.js, so the shared helpers it relies on (window.openPlayerModal,
// window._rzBuildLiveHtml, window._rzSyncTabLive) and window.__rz__ are ready.
// The IIFE still self-guards on #rz-root, so including it elsewhere is a no-op.

// ── BR Redzone ────────────────────────────────────────────────────────────────
(function () {
  var root = document.getElementById('rz-root');
  if (!root) return;

  var _state    = window.__rz__ || {};
  var _feed     = [];
  var _shownFeedIds = new Set();
  var _prevStats = {};
  var _prevPts   = {};
  var _countdown = 15;
  var _timer     = null;
  var _playerCache = {};
  var _isDemo   = !!_state.is_demo;
  var _demoT    = parseFloat(_state.demo_t || 150);
  var _scope    = _state.scope || 'league';
  var _filters  = { nfl: 'all', pos: 'all', stat: 'all' };
  var _filterOpen  = false;
  var _myTeamOnly  = false;
  var _bigPlaysOnly = false; // TD or >=4 fantasy pts
  var _heroMid    = null;
  var _heroTouched = false; // true once the viewer explicitly picks/clears the hero matchup
  var _seenPlayIds = new Set(); // Tank01 / demo play ids already in the feed
  var _seenContributions = new Set(); // contribution keys (play + pid) for deduping
  var _pbpGames = {}; // game_id → true once we have real PBP for that game
  var _pbpHistory = []; // full PBP contribution history (not capped)
  var _playGroupsByKey = {}; // canonical play groups: nflPlayKey → {contributionsByKey, primaryEvent, playState}
  var _contributionsByKey = {}; // global contribution storage: contribKey → contribution (for revisions)
  var _alertsArmed = false; // suppress TD beep/push until the first hydration seeds the feed
  var _slideDir   = 'none';
  var _feedPage   = 0;
  var _prevMatchupPts = {};
  var _flashRids  = new Set();
  var _lastPollFailed = false;
  var _hadInteraction = false;
  var _notifDismissed = !!(localStorage && localStorage.getItem('rz-notif-dismissed'));
  var _notifHistory = (function() {
    try { return JSON.parse(localStorage.getItem('rz-notif-history') || '[]'); } catch (_) { return []; }
  }());
  var _historyOpen = false;
  var _unreadCount = 0;
  var _milestonesSeen = {};
  var _blowoutSeen = {};
  var _prevInjury = {};
  var _prevLeader = {}; // matchup_id → leading roster_id (for lead-change events)
  var _scoreDelta = { me: 0, opp: 0 }; // pts gained since last poll (for hero card)
  var _loadingScope = false; // true while awaiting the first fetch after a scope switch
  var _mlNames  = [];        // My Leagues: league display names by portfolio index
  var _mlLoaded = null;      // My Leagues: Set of portfolio indices whose card has arrived (null = not streaming)
  var _mlFailed = null;      // My Leagues: Set of portfolio indices that failed to load
  var _streaming = false;    // true while a progressive My Leagues stream is in flight
  var _streamGen = 0;        // bumped on every scope switch so a stale stream/poll can abort
  // Last-good payload per scope so My Leagues → This League never paints
  // portfolio (cross-league) data under the league-scoped chrome.
  var _scopeCache = { league: null, user: null };
  if (_state && Object.keys(_state).length) {
    _scopeCache[_state.scope || 'league'] = _state;
  }

  function _prefsKey(kind) {
    var plat = _state.platform || 'x';
    var lid = _state.league_id || 'x';
    return 'rz-' + kind + ':' + plat + ':' + lid + ':' + _scope;
  }
  function _loadPrefs() {
    if (_isDemo) return;
    try {
      var h = localStorage.getItem(_prefsKey('hero-mid'));
      if (h !== null && h !== '') { _heroMid = h; _heroTouched = true; }
      var mt = localStorage.getItem(_prefsKey('my-team'));
      if (mt === '1') _myTeamOnly = true;
      if (mt === '0') _myTeamOnly = false;
      var bp = localStorage.getItem(_prefsKey('big-only'));
      if (bp === '1') _bigPlaysOnly = true;
      if (bp === '0') _bigPlaysOnly = false;
    } catch (_) {}
  }
  function _savePrefs() {
    if (_isDemo) return;
    try {
      if (_heroTouched) {
        if (_heroMid) localStorage.setItem(_prefsKey('hero-mid'), String(_heroMid));
        else localStorage.removeItem(_prefsKey('hero-mid'));
      }
      localStorage.setItem(_prefsKey('my-team'), _myTeamOnly ? '1' : '0');
      localStorage.setItem(_prefsKey('big-only'), _bigPlaysOnly ? '1' : '0');
    } catch (_) {}
  }

  document.addEventListener('click', function() { _hadInteraction = true; }, { once: true });

  // Keep player-modal wiring in one delegated handler. Redzone replaces most of
  // its DOM on every render; attaching another listener to each rendered row as
  // well as this delegate opens two overlays from a single click (most visibly
  // on the Top Scorers rows and position-leader tiles).
  root.addEventListener('click', function(e) {
    var target = e.target;
    // Walk up to find element with data-pid
    while (target && target !== root) {
      if (target.dataset && target.dataset.pid) {
        // Skip if it's a player-pts element (just displays score)
        if (target.classList.contains('rz-player-pts')) {
          target = target.parentElement;
          continue;
        }
        var pid = target.dataset.pid;
        if (pid && pid !== '0' && window.openPlayerModal) {
          window.openPlayerModal(pid, _name(pid), { tab: 'live' });
          e.stopPropagation();
          return;
        }
      }
      target = target.parentElement;
    }
  });

  function _playTDBeep() {
    if (!_hadInteraction) return;
    try {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = 'sine';
      gain.gain.setValueAtTime(0.18, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.35);
      // second note
      setTimeout(function() {
        try {
          var o2 = ctx.createOscillator(), g2 = ctx.createGain();
          o2.connect(g2); g2.connect(ctx.destination);
          o2.frequency.value = 1100; o2.type = 'sine';
          g2.gain.setValueAtTime(0.12, ctx.currentTime);
          g2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
          o2.start(ctx.currentTime); o2.stop(ctx.currentTime + 0.25);
        } catch (_) {}
      }, 160);
    } catch (_) {}
  }

  function _myRidSet(data) {
    var ids = (data.viewer_roster_ids && data.viewer_roster_ids.length)
      ? data.viewer_roster_ids
      : (data.viewer_roster_id ? [data.viewer_roster_id]
         : (window._viewerRid ? [window._viewerRid] : []));
    return new Set(ids.map(String));
  }
  var _myRids = _myRidSet(_state);

  function _heroMatchupPids() {
    if (!_heroMid) return null;
    var pids = new Set();
    if (_scope === 'league') {
      (_state.matchups || []).forEach(function(m) {
        if (String(m.matchup_id) === _heroMid) {
          (m.starters || []).forEach(function(p) { pids.add(p); });
          (m.players  || []).forEach(function(p) { pids.add(p); });
        }
      });
    } else {
      var myM = (_state.matchups || []).find(function(m) { return String(m.roster_id) === _heroMid; });
      if (myM) {
        var mid = String(myM.matchup_id);
        (_state.matchups || []).forEach(function(m) {
          if (String(m.matchup_id) === mid) {
            (m.starters || []).forEach(function(p) { pids.add(p); });
            (m.players  || []).forEach(function(p) { pids.add(p); });
          }
        });
      }
    }
    return pids.size ? pids : null;
  }

  function _anyLive() {
    var live = false;
    (_state.matchups || []).forEach(function(m) {
      (m.starters || []).forEach(function(pid) { if (_gameStatus(pid).type === 'live') live = true; });
    });
    return live;
  }

  function _matchupIsLive(matchups) {
    var live = false;
    (matchups || []).forEach(function(m) {
      if (!m) return;
      (m.starters || []).forEach(function(pid) { if (_gameStatus(pid).type === 'live') live = true; });
    });
    return live;
  }

  // Resolve the two sides of the focused hero card. This League heroes can be
  // any matchup; My Leagues heroes are a viewer roster id.
  function _focusedPair() {
    var mine = _myMatchups();
    if (!_heroMid) {
      var m0 = mine[0];
      return m0 ? { mine: m0, opp: _oppOf(m0), isMine: true } : null;
    }
    if (_scope === 'user') {
      var focused = mine.find(function(m) { return String(m.roster_id) === _heroMid; });
      if (!focused) return mine[0] ? { mine: mine[0], opp: _oppOf(mine[0]), isMine: true } : null;
      return { mine: focused, opp: _oppOf(focused), isMine: true };
    }
    var pair = (_state.matchups || []).filter(function(m) { return String(m.matchup_id) === _heroMid; });
    if (!pair.length) {
      var m1 = mine[0];
      return m1 ? { mine: m1, opp: _oppOf(m1), isMine: true } : null;
    }
    var mineSide = pair.find(function(m) { return _isMyRid(m.roster_id); }) || pair[0];
    var oppSide = pair.find(function(m) { return String(m.roster_id) !== String(mineSide.roster_id); }) || null;
    return { mine: mineSide, opp: oppSide, isMine: !!_isMyRid(mineSide.roster_id) };
  }

  function _syncScopeUrl() {
    try {
      var url = new URL(window.location.href);
      if (_scope === 'user') url.searchParams.set('scope', 'user');
      else url.searchParams.delete('scope');
      window.history.replaceState({}, '', url.pathname + url.search + url.hash);
    } catch (_) {}
  }

  // True when a not-yet-started game in this view kicks off within `mins`
  // minutes. player_info is already scoped to the fetched matchups, so this
  // only counts games that involve players on screen.
  function _kickoffWithin(mins) {
    var now = Date.now() / 1000;
    var horizon = now + mins * 60;
    return Object.keys(_state.player_info || {}).some(function(pid) {
      var p = _state.player_info[pid] || {};
      if (String(p.game_code || '0') !== '0') return false; // upcoming games only
      var ep = parseFloat(p.game_time_epoch || 0);
      return ep > 0 && ep <= horizon;
    });
  }

  // The window where Redzone presents its "live" look: a game actually in
  // progress, or the hour before the next kickoff. Used for the page-level
  // status chip (per-game badges stay accurate -- PRE until their own kickoff).
  function _liveWindow() {
    if (_anyLive()) return { on: true, live: true };
    if (_isDemo) return { on: true, live: true };
    if (_kickoffWithin(60)) return { on: true, live: false };
    return { on: false, live: false };
  }

  // Header status chip: "LIVE" once a game is in progress, "PREGAME" in the
  // hour before kickoff, nothing otherwise. Same pulsing dot in both states.
  function _statusChipHtml() {
    var w = _liveWindow();
    if (!w.on) return '';
    var cls = w.live ? 'rz-live-chip' : 'rz-live-chip rz-pregame-chip';
    return '<span class="' + cls + '"><span class="rz-nav-dot"></span>' + (w.live ? 'LIVE' : 'PREGAME') + '</span>';
  }

  function _fmtKickoff(ep) {
    var d = new Date(ep * 1000);
    if (isNaN(d.getTime())) return 'Upcoming';
    var day = d.toLocaleDateString([], { weekday: 'short' });
    var time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    return day + ' ' + time;
  }

  function _fmt(n) {
    if (n == null || n === '') return '0.0';
    return parseFloat(n).toFixed(1);
  }
  // Redzone-specific fantasy-point formatters that preserve precision
  // Preserve meaningful hundredths, strip unnecessary trailing zeros
  // Examples: 0.04→"0.04", 1.04→"1.04", 1.1→"1.1", 6→"6", 8.44→"8.44"
  function _fmtFantasyPrecise(n) {
    if (n == null || n === '') return '0';
    var val = parseFloat(n);
    if (isNaN(val)) return '0';
    // Round to hundredths to avoid floating-point artifacts
    var rounded = Math.round(val * 100) / 100;
    // Format with 2 decimals, then strip unnecessary trailing zeros
    var s = rounded.toFixed(2);
    // Remove trailing zeros after decimal: "1.00"→"1", "1.10"→"1.1", "1.04"→"1.04"
    s = s.replace(/\.?0+$/, '');
    return s;
  }
  function _fmtFantasyDelta(n) {
    return _fmtFantasyPrecise(n);
  }
  function _fmtFantasyTotal(n) {
    return _fmtFantasyPrecise(n);
  }
  function _fmtTimer(n) {
    n = Math.max(0, Math.round(n));
    return n >= 120 ? Math.round(n / 60) + 'm' : n + 's';
  }
  function _name(pid) { return ((_state.player_info || {})[pid] || {}).name || pid; }
  function _pos(pid)  { return ((_state.player_info || {})[pid] || {}).pos  || ''; }
  function _team(pid) { return ((_state.player_info || {})[pid] || {}).team || ''; }
  function _statLine(pid) { return ((_state.player_info || {})[pid] || {}).stat_line || null; }

  // Normalized NFL game status resolver. `g` is an _nflGameInfo() row keyed by
  // game_id -- the authoritative game-level snapshot. The server pre-normalizes
  // `g.status` (utils.redzone_pbp.normalize_nfl_game_status); we prefer it and
  // only re-derive from the raw code/text when an older payload omits it.
  //
  // game_code is the game-level source of truth (Tank01 gameStatusCode, with
  // ESPN mapped onto the same scale): 0 pregame, 1 live, 2 final. A game is
  // FINAL only when that code says 2 (or, lacking a code, the text is
  // explicitly final). A blank/unknown code is 'unknown' -- never final -- so a
  // bye, a provider gap, or a stale player record can't fake a completed game.
  // Returns 'pregame'|'live'|'halftime'|'final'|'delayed'|'unknown'.
  function _normGameStatus(g) {
    if (!g) return 'unknown';
    if (g.status) return String(g.status);
    var code = String(g.game_code || '');
    var txt  = String(g.game_status || '').toLowerCase();
    if (code === '1') return txt.indexOf('half') >= 0 ? 'halftime' : 'live';
    if (code === '2') return 'final';
    if (code === '0') return 'pregame';
    if (txt.indexOf('final') >= 0) return 'final';
    if (txt.indexOf('half') >= 0) return 'halftime';
    if (/postpon|delay|suspend|cancel/.test(txt)) return 'delayed';
    if (/progress|quarter|qtr|q[1-4]/.test(txt)) return 'live';
    return txt ? 'pregame' : 'unknown';
  }

  // Coarse per-player state for matchup math, resolved from the player's NFL
  // game (game-level authority), NOT the player's own possibly-stale record.
  //   'final' | 'live' | 'upcoming' | 'bye' | 'unknown' | 'empty'
  // 'live' folds in halftime; 'upcoming' folds in delayed (game not complete).
  function _playerGameState(pid) {
    if (pid === '0') return { type: 'empty', label: '', norm: 'empty' };
    var p = (_state.player_info || {})[pid] || {};
    var gid = p.game_id || '';
    var g = gid ? _nflGameInfo(gid) : null;
    if (!g) {
      // Rostered player whose team has no game this week = bye (not "to play").
      // No team at all = unknown. Neither is ever final.
      return p.team
        ? { type: 'bye', label: 'BYE', norm: 'bye' }
        : { type: 'unknown', label: '', norm: 'unknown' };
    }
    var norm = _normGameStatus(g);
    var coarse = norm === 'final' ? 'final'
               : (norm === 'live' || norm === 'halftime') ? 'live'
               : (norm === 'pregame' || norm === 'delayed') ? 'upcoming'
               : 'unknown';
    var label = coarse === 'final' ? 'FINAL'
              : coarse === 'live' ? (norm === 'halftime' ? 'HALF' : 'LIVE')
              : coarse === 'upcoming' ? (g.game_status || 'Upcoming')
              : '';
    return { type: coarse, label: label, norm: norm, game: g };
  }

  // Backward-compatible player status used throughout the render helpers.
  // type ∈ 'live' | 'final' | 'pre'. Derives from the game-level resolver so a
  // stale player_info record can never win over the authoritative game object.
  function _gameStatus(pid) {
    var s = _playerGameState(pid);
    if (s.type === 'final') return { label: 'FINAL', type: 'final' };
    if (s.type === 'live')  return { label: s.label || 'LIVE', type: 'live' };
    return { label: s.label || '', type: 'pre' };
  }

  // ── Fantasy matchup state ──────────────────────────────────────────────────
  // Per-side player counts by NFL game state. Bye/empty players are excluded
  // from `total` (they are not "relevant" to still-to-play), never counted as
  // upcoming, and never stuck as such. Counts derive from the normalized game
  // status only -- never from fantasy points, a 0.0 score, or a shown clock.
  function _sideCounts(matchup) {
    var c = { total: 0, final: 0, live: 0, upcoming: 0, unknown: 0, bye: 0 };
    if (!matchup) return c;
    (matchup.starters || []).forEach(function(pid) {
      if (pid === '0') return;
      var s = _playerGameState(pid).type;
      if (s === 'empty') return;
      if (s === 'bye') { c.bye++; return; }
      c.total++;
      if (s === 'final') c.final++;
      else if (s === 'live') c.live++;
      else if (s === 'upcoming') c.upcoming++;
      else c.unknown++;
    });
    return c;
  }

  // Fantasy matchup state resolver (§15). A matchup is FINAL only when every
  // relevant starter's NFL game is actually complete -- never because a single
  // player's game ended, and never from the fantasy score. LIVE takes
  // precedence over TO PLAY; unknown-only falls back to a neutral state.
  function _matchupState(a, b) {
    var ca = _sideCounts(a), cb = _sideCounts(b);
    var live = ca.live + cb.live;
    var upcoming = ca.upcoming + cb.upcoming;
    var unknown = ca.unknown + cb.unknown;
    var total = ca.total + cb.total;
    var state;
    if (live > 0) state = 'live';
    else if (upcoming > 0) state = 'toplay';
    else if (total > 0 && unknown === 0) state = 'final';
    else state = 'unknown';
    return {
      state: state, a: ca, b: cb,
      toPlayA: ca.upcoming, toPlayB: cb.upcoming,
      liveA: ca.live, liveB: cb.live
    };
  }

  // Center status column shared by every fantasy matchup card / row.
  function _matchupCenterHtml(a, b) {
    var ms = _matchupState(a, b);
    if (ms.state === 'live') {
      return '<div class="rz-mc-state live">LIVE</div>'
        + '<div class="rz-mc-playing-counts">' + ms.liveA
        + '<span class="rz-mc-to-play-divider">|</span>' + ms.liveB + '</div>'
        + '<div class="rz-mc-substate">PLAYING</div>';
    }
    if (ms.state === 'toplay') {
      return '<div class="rz-mc-state toplay">TO PLAY</div>'
        + '<div class="rz-mc-to-play-counts">' + ms.toPlayA
        + '<span class="rz-mc-to-play-divider">|</span>' + ms.toPlayB + '</div>';
    }
    if (ms.state === 'final') return '<div class="rz-mc-state final">FINAL</div>';
    return '<div class="rz-mc-state neutral">–</div>';
  }

  // Minimal HTML-attribute escape for aria-label text.
  function _esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Screen-reader summary for a fantasy matchup card (§21).
  function _matchupAria(nameA, ptsA, nameB, ptsB, ms) {
    var s = nameA + ' ' + _fmt(ptsA) + ', ' + nameB + ' ' + _fmt(ptsB);
    if (ms.state === 'toplay') {
      s += ', ' + ms.toPlayA + ' players to play versus ' + ms.toPlayB + ' players to play';
    } else if (ms.state === 'live') {
      s += ', live: ' + ms.liveA + ' versus ' + ms.liveB + ' players in play';
    } else if (ms.state === 'final') {
      s += ', final';
    }
    return s;
  }
  function _gameLine(pid) {
    var p = (_state.player_info || {})[pid] || {};
    if (!p.home || !p.away) return '';
    var a = p.away_pts, h = p.home_pts;
    if (a === '' && h === '') return p.away + ' @ ' + p.home;
    return p.away + ' ' + (a || '0') + ' @ ' + p.home + ' ' + (h || '0');
  }
  function _quarterNum(q) {
    var m = String(q || '').match(/(\d)/);
    return m ? parseInt(m[1], 10) : 0;
  }
  var _INJ_RANK = { '': 0, 'Q': 1, 'D': 2, 'O': 3, 'IR': 3 };
  function _injLabel(code) {
    return code === 'O' ? 'Out' : code === 'IR' ? 'IR' : code === 'D' ? 'Doubtful'
         : code === 'Q' ? 'Questionable' : code ? code : 'Active';
  }
  function _rosterOf(rid) {
    var s = String(rid);
    return (_state.rosters || []).find(function(r) { return String(r.roster_id) === s; });
  }
  function _ownerName(rid) {
    var roster = _rosterOf(rid);
    if (!roster) return '';
    var user = (_state.users || []).find(function(u) { return u.user_id === roster.owner_id; });
    return (user && user.display_name) || '';
  }
  function _leagueOfRid(rid) {
    var s = String(rid);
    var m = (_state.matchups || []).find(function(x) { return String(x.roster_id) === s; });
    return (m && m.league_name) || '';
  }
  function _isMyRid(rid) { return _myRids.has(String(rid)); }

  // ── Scoring math ───────────────────────────────────────────────────────────────
  function _n(x) { return parseFloat(x || 0) || 0; }
  // Per-made-FG rate for a league. Leagues that score by distance define
  // buckets (Sleeper fgm_0_19..fgm_50p; ESPN fgm_0_39/fgm_40_49/fgm_50p); pick
  // the tightest bucket the league actually defines, else the flat fgm/fg rate.
  function _fgRate(yds, s) {
    yds = _n(yds);
    var order;
    if (yds >= 60) order = ['fgm_60p', 'fgm_50p', 'fgm_40_49', 'fgm_0_39', 'fgm', 'fg'];
    else if (yds >= 50) order = ['fgm_50p', 'fgm_50_59', 'fgm_40_49', 'fgm_0_39', 'fgm', 'fg'];
    else if (yds >= 40) order = ['fgm_40_49', 'fgm_0_39', 'fgm', 'fg'];
    else if (yds >= 30) order = ['fgm_30_39', 'fgm_0_39', 'fgm', 'fg'];
    else if (yds >= 20) order = ['fgm_20_29', 'fgm_0_39', 'fgm', 'fg'];
    else if (yds > 0)   order = ['fgm_0_19', 'fgm_0_39', 'fgm', 'fg'];
    else order = ['fgm', 'fg']; // unknown distance (e.g. cumulative box line)
    for (var i = 0; i < order.length; i++) {
      if (s[order[i]] != null) return _n(s[order[i]]);
    }
    return 0;
  }
  function _lineToPts(L, s, pos) {
    if (!L) return 0;
    s = s || _state.scoring || {};
    var pts = _n(L.pass_yds) * _n(s.pass_yd) + _n(L.pass_td) * _n(s.pass_td) + _n(L.int) * _n(s.pass_int)
      + _n(L.rush_yds) * _n(s.rush_yd) + _n(L.rush_td) * _n(s.rush_td)
      + _n(L.rec) * _n(s.rec) + _n(L.rec_yds) * _n(s.rec_yd) + _n(L.rec_td) * _n(s.rec_td);
    // TE reception premium (bonus_rec_te) when the league runs one.
    if (String(pos || '').toUpperCase() === 'TE') pts += _n(L.rec) * _n(s.bonus_rec_te);
    // Kicker (distance-aware) + defense (Sleeper-style keys with common aliases)
    pts += _n(L.fgm) * _fgRate(L.fg_yds, s) + _n(L.xpm) * _n(s.xpm || s.xp);
    var sacks = _n(L.sacks != null ? L.sacks : L.sack);
    pts += sacks * _n(s.sack) + _n(L.def_int) * _n(s.int || s.def_int)
      + _n(L.fum_rec) * _n(s.fum_rec) + _n(L.def_td) * _n(s.def_td || s.td);
    return pts;
  }

  function _scoringForPid(pid, matchup) {
    var sbl = _state.scoring_by_league || {};
    var lid = (matchup && (matchup.league_id || matchup.leagueId)) || (_state.pid_league || {})[pid];
    if (lid && sbl[lid]) return sbl[lid];
    return _state.scoring || {};
  }
  // Platform players_points can lag behind Tank01 boxscores (Yahoo often sends
  // {}). Prefer max(platform, boxscore) while the NFL game is live/final so
  // My Teams doesn't show 0.0 for active players.
  function _playerPts(pid, matchup) {
    var pp = (matchup && matchup.players_points) || {};
    var platformN = (pp[pid] != null && pp[pid] !== '') ? parseFloat(pp[pid]) : NaN;
    var live = _lineToPts(_statLine(pid), _scoringForPid(pid, matchup));
    var gs = _gameStatus(pid);
    if (gs.type === 'live' || gs.type === 'final') {
      if (isNaN(platformN)) return live;
      return Math.max(platformN, live);
    }
    if (!isNaN(platformN)) return platformN;
    return live || 0;
  }
  function _totalPtsForPid(pid, scoring, newData) {
    var infoSrc = (newData && newData.player_info) || _state.player_info || {};
    var fromLine = _lineToPts((infoSrc[pid] || {}).stat_line || null, scoring || _scoringForPid(pid));
    var fromPlatform = NaN;
    var matchups = (newData && newData.matchups) || _state.matchups || [];
    for (var i = 0; i < matchups.length; i++) {
      var pp = matchups[i].players_points || {};
      if (pp[pid] != null && pp[pid] !== '') {
        var n = parseFloat(pp[pid]);
        if (!isNaN(n)) fromPlatform = isNaN(fromPlatform) ? n : Math.max(fromPlatform, n);
      }
    }
    if (isNaN(fromPlatform)) return fromLine;
    return Math.max(fromPlatform, fromLine);
  }

  function _calcBreakdown(pos, bd, scoring) {
    var rows = [], total = 0;
    function row(label, val, key) {
      var rate = parseFloat(scoring[key] || 0);
      if (!val || !rate) return;
      var pts = parseFloat((val * rate).toFixed(2));
      total += pts;
      rows.push({ label: label, val: val, pts: pts });
    }
    if (pos === 'QB') {
      row('Pass Yds', bd.pass_yds || 0, 'pass_yd');
      row('Pass TDs', bd.pass_tds || 0, 'pass_td');
      row('INTs',     bd.ints     || 0, 'pass_int');
      row('Rush Yds', bd.rush_yds || 0, 'rush_yd');
      row('Rush TDs', bd.rush_tds || 0, 'rush_td');
    } else {
      row('Rush Yds', bd.rush_yds   || 0, 'rush_yd');
      row('Rush TDs', bd.rush_tds   || 0, 'rush_td');
      row('Rec',      bd.receptions || 0, 'rec');
      row('Rec Yds',  bd.rec_yds    || 0, 'rec_yd');
      row('Rec TDs',  bd.rec_tds    || 0, 'rec_td');
    }
    return { rows: rows, total: parseFloat(total.toFixed(2)) };
  }

  // ── Seed snapshots ───────────────────────────────────────────────────────────
  function _seedPrevStats(data) {
    Object.keys(data.player_info || {}).forEach(function(pid) {
      var sl = data.player_info[pid].stat_line;
      if (sl) _prevStats[pid] = Object.assign({}, sl);
    });
    (data.matchups || []).forEach(function(m) {
      var pp = m.players_points || {};
      Object.keys(pp).forEach(function(pid) { _prevPts[pid] = pp[pid]; });
    });
  }

  // Clear poll-diff snapshots so a scope switch can rehydrate Plays from the
  // new payload instead of silently diffing against the other scope's lines.
  function _resetFeedSnapshots() {
    _prevStats = {};
    _prevPts = {};
    _milestonesSeen = {};
    _blowoutSeen = {};
    _prevInjury = {};
    _prevLeader = {};
    _prevMatchupPts = {};
    _scoreDelta = { me: 0, opp: 0 };
    _flashRids = new Set();
    _seenPlayIds = new Set();
    _seenContributions = new Set();
    _pbpGames = {};
    _pbpHistory = [];
    _playGroupsByKey = {};
    // A scope switch rehydrates from scratch -- re-arm alerts only after it does.
    _alertsArmed = false;
  }

  // Same cold-boot order as page load: suppress milestone/injury/lead noise,
  // then turn current box scores into Plays against empty prevStats.
  function _hydrateFeed(data) {
    _seedMilestones(data);
    _seedInjuries(data);
    _seedLeaders(data);
    (data.matchups || []).forEach(function(m) {
      _prevMatchupPts[String(m.roster_id)] = parseFloat(m.points || 0);
    });
    _detectChanges(data);
    _seedPrevStats(data);
    // The first pass ingests a whole game of history at once; only alert on
    // TDs discovered by subsequent live polls, never on this initial backfill.
    _alertsArmed = true;
  }
  function _seedMilestones(data) {
    var _MS_THRS = [
      { key: 'rush_yds_100', field: 'rush_yds', thr: 100 },
      { key: 'rush_yds_150', field: 'rush_yds', thr: 150 },
      { key: 'pass_yds_300', field: 'pass_yds', thr: 300 },
      { key: 'pass_yds_400', field: 'pass_yds', thr: 400 },
      { key: 'rec_yds_100', field: 'rec_yds', thr: 100 },
      { key: 'td_2', field: '__tds', thr: 2 },
      { key: 'td_3', field: '__tds', thr: 3 },
    ];
    Object.keys(data.player_info || {}).forEach(function(pid) {
      var sl = (data.player_info[pid] || {}).stat_line;
      if (!sl) return;
      var seen = _milestonesSeen[pid] || {};
      var tds = (sl.rush_td||0) + (sl.rec_td||0) + (sl.pass_td||0);
      _MS_THRS.forEach(function(ms) {
        var val = ms.field === '__tds' ? tds : (sl[ms.field] || 0);
        if (val >= ms.thr) seen[ms.key] = true;
      });
      _milestonesSeen[pid] = seen;
    });
  }

  function _seedInjuries(data) {
    Object.keys(data.player_info || {}).forEach(function(pid) {
      _prevInjury[pid] = (data.player_info[pid] || {}).injury_status || '';
    });
  }

  function _seedLeaders(data) {
    var groups = {};
    (data.matchups || []).forEach(function(m) {
      var mid = String(m.matchup_id);
      (groups[mid] = groups[mid] || []).push(m);
    });
    Object.keys(groups).forEach(function(mid) {
      var pair = groups[mid];
      if (pair.length < 2) return;
      var a = pair[0], b = pair[1];
      var ptsA = parseFloat(a.points || 0), ptsB = parseFloat(b.points || 0);
      _prevLeader[mid] = (ptsA <= 0 && ptsB <= 0) ? null
        : (ptsA >= ptsB ? String(a.roster_id) : String(b.roster_id));
    });
  }

  // ── Play detection ─────────────────────────────────────────────────────────────
  function _rosterTags(data) {
    var myRosters = new Set(_myRids), oppRosters = new Set();
    (data.matchups || []).forEach(function(m) {
      if (!_isMyRid(m.roster_id)) return;
      var mid = String(m.matchup_id);
      (data.matchups || []).forEach(function(o) {
        if (String(o.matchup_id) === mid && !_isMyRid(o.roster_id)) oppRosters.add(String(o.roster_id));
      });
    });
    
    // When a hero matchup is active, also include all rosters in that matchup
    // so filtered matchups display correct mine/opp flags even when viewer isn't in them
    if (_heroMid && _scope === 'league') {
      (data.matchups || []).forEach(function(m) {
        if (String(m.matchup_id) === _heroMid) {
          var rid = String(m.roster_id);
          if (_isMyRid(rid)) myRosters.add(rid);
          else oppRosters.add(rid);
        }
      });
    }
    
    var pidToRoster = {};
    (data.matchups || []).forEach(function(m) {
      (m.players || m.starters || []).forEach(function(pid) {
        // Prefer the viewer's roster when the same pid appears in multiple
        // My Leagues slices; otherwise first wins.
        var rid = String(m.roster_id);
        if (!(pid in pidToRoster)) pidToRoster[pid] = rid;
        else if (_isMyRid(rid) && !_isMyRid(pidToRoster[pid])) pidToRoster[pid] = rid;
      });
    });
    return { my: myRosters, opp: oppRosters, pidToRoster: pidToRoster };
  }


  function _isBigPlay(ev) {
    return ev.kind === 'td' || (ev.pts || 0) >= 4;
  }
  // "MM:SS" game clock → seconds remaining in the quarter (null if unparsable).
  function _clockSecs(clk) {
    var m = String(clk == null ? '' : clk).match(/(\d+):(\d+)/);
    if (!m) return null;
    return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
  }
  // Approximate wall-clock a play occurred at, so the feed reads in real
  // chronological order (newest first): game kickoff epoch + elapsed game
  // seconds. Live-only events (milestones, bulk) fall back to detection time.
  function _chronoKey(ev) {
    // Reconstruct elapsed game time first. Unlike a provider sequence number,
    // this remains comparable when plays from simultaneous games are merged.
    var q = parseInt(ev.gameQuarter, 10);
    if (q > 0) {
      var per = 900; // 15:00 quarters (OT still monotonic under this model)
      var cs = _clockSecs(ev.gameClock);
      var inQ = (cs == null) ? 0 : Math.max(0, per - cs);
      var elapsed = (q - 1) * per + inQ;
      // Try to get gameId from event or player_info
      var gid = ev.gameId || ((_state.player_info || {})[ev.pid] || {}).game_id || '';
      var g = (_state.games || {})[gid] || {};
      var kickoff = parseFloat(g.game_time_epoch || 0) || 0;
      if (kickoff) return kickoff + elapsed;
    }
    // A provider sequence is still useful within one game when clock data is
    // absent, but it must not override the cross-game wall-clock estimate.
    if (ev.gameId && ev.seq != null) {
      var seqGame = (_state.games || {})[ev.gameId] || {};
      var seqKickoff = parseFloat(seqGame.game_time_epoch || 0) || 0;
      if (seqKickoff) return seqKickoff + (ev.seq * 0.001);
    }
    return (ev.ts || 0) / 1000;
  }
  // Newest first. Modern JavaScript's stable sort preserves ingestion order
  // for exact ties instead of quietly reintroducing the removed "For You"
  // ranking for simultaneous plays.
  function _chronoSort(list) {
    return list.slice().sort(function(a, b) {
      var ka = _chronoKey(a), kb = _chronoKey(b);
      if (ka !== kb) return kb - ka;
      return 0;
    });
  }
  // "4" → "Q4", "5"+ → "OT"; pass through non-numeric labels ("OT", "Half").
  function _fmtQuarter(q) {
    var s = String(q == null ? '' : q).trim();
    if (!s) return '';
    if (/^\d+$/.test(s)) { var n = parseInt(s, 10); return n >= 5 ? 'OT' : ('Q' + n); }
    return s;
  }
  function _downDist(ev) {
    var d = ev.down, dist = ev.distance;
    if (!d && !dist) return '';
    var ord = ({'1':'1st','2':'2nd','3':'3rd','4':'4th'})[String(d)] || (d ? (d + 'th') : '');
    if (ord && dist) return ord + ' & ' + dist;
    if (ord) return ord + ' down';
    return dist ? ('& ' + dist) : '';
  }
  // Sleeper-style running stat line for this player *through this play*
  // (e.g. "2/3 CMP, 13 YD, 1 TD"). Built from the server's cumulative snapshot.
  function _cumeLine(pos, c) {
    if (!c) return '';
    var p = [];
    var yd = function(n) { return _n(n) + ' YD'; };
    if (_n(c.pass_att)) {
      p.push(_n(c.pass_cmp) + '/' + _n(c.pass_att) + ' CMP');
      p.push(yd(c.pass_yds));
      if (_n(c.pass_td)) p.push(_n(c.pass_td) + ' TD');
      if (_n(c.int)) p.push(_n(c.int) + ' INT');
      if (_n(c.carries)) p.push(_n(c.rush_yds) + ' RUSH');
    } else if (_n(c.carries)) {
      p.push(_n(c.carries) + ' CAR');
      p.push(yd(c.rush_yds));
      if (_n(c.rush_td)) p.push(_n(c.rush_td) + ' TD');
      if (_n(c.rec)) { p.push(_n(c.rec) + ' REC'); p.push(yd(c.rec_yds)); }
      if (_n(c.rec_td)) p.push(_n(c.rec_td) + ' REC TD');
    } else if (_n(c.rec) || _n(c.targets)) {
      p.push(_n(c.rec) + '/' + _n(c.targets) + ' REC');
      p.push(yd(c.rec_yds));
      if (_n(c.rec_td)) p.push(_n(c.rec_td) + ' TD');
    } else if (_n(c.fgm) || _n(c.xpm)) {
      if (_n(c.fgm)) p.push(_n(c.fgm) + ' FG');
      if (_n(c.xpm)) p.push(_n(c.xpm) + ' XP');
    }
    return p.join(', ');
  }
  // Red zone = the ball is inside the *opponent's* 20. yardLine reads like
  // "SEA 16"; it's the red zone only when that team is not the offense's own.
  function _isRedZone(yardLine, offenseTeam) {
    var m = String(yardLine || '').match(/([A-Za-z]{2,3})\s*(\d{1,2})\b/);
    if (!m) return false;
    var n = parseInt(m[2], 10);
    if (!(n >= 1 && n <= 20)) return false;
    var off = String(offenseTeam || '').toUpperCase();
    if (!off) return false; // can't tell own 20 from opponent's -- don't guess
    return m[1].toUpperCase() !== off;
  }
  function _impactLine(ev) {
    var bits = [];
    if (ev.pts) bits.push((ev.pts > 0 ? '+' : '') + _fmt(ev.pts) + ' pts');
    if (ev.mine && ev.kind === 'td') bits.push('your roster');
    else if (ev.opp && ev.kind === 'td') bits.push('vs your matchup');
    else if (ev.mine) bits.push('your starter');
    return bits.join(' · ');
  }
  function _describe(d, pos) {
    // Prefer play-by-play wording over bulk box-score shorthand
    // (e.g. "Throws a 66-yard touchdown pass" vs "66 yd TD pass").
    function _yds(n) {
      n = Math.round(n || 0);
      return n === 1 ? '1 yard' : n + ' yards';
    }

    // DEF: only sacks / INT / fumbles / defensive TDs
    if (pos === 'DEF') {
      var segs = [];
      var st = [];
      if (d.sacks   > 0) { segs.push(d.sacks   === 1 ? 'Records a sack' : 'Records ' + d.sacks + ' sacks'); st.push('sack'); }
      if (d.def_int > 0) { segs.push(d.def_int  === 1 ? 'Picks off a pass' : 'Picks off ' + d.def_int + ' passes'); st.push('int'); }
      if (d.fum_rec > 0) { segs.push(d.fum_rec  === 1 ? 'Recovers a fumble' : 'Recovers ' + d.fum_rec + ' fumbles'); st.push('fumble'); }
      if (d.def_td  > 0) { segs.push(d.def_td   === 1 ? 'Scores a defensive touchdown' : 'Scores ' + d.def_td + ' defensive touchdowns'); st.push('td'); }
      if (!segs.length) return null;
      return { desc: segs.join(' · '), kind: d.def_td > 0 ? 'td' : 'gain', stats: st };
    }

    // K: FG or PAT
    if (pos === 'K') {
      if (d.fgm > 0) {
        var dist = Math.round(d.fg_long || 0);
        return {
          desc: dist > 0 ? ('Drills a ' + dist + '-yard field goal') : 'Drills a field goal',
          kind: 'gain', stats: ['kick']
        };
      }
      if (d.xpm > 0) return { desc: 'Knocks through the extra point', kind: 'gain', stats: ['kick'] };
      return null;
    }

    // QB / RB / WR / TE
    var ry = Math.round(d.rec_yds), uy = Math.round(d.rush_yds), py = Math.round(d.pass_yds);
    var tdc = d.rec_td + d.rush_td + d.pass_td;
    var stats = [];
    if (d.rec      >= 1) stats.push('reception');
    if (d.carries  >= 1) stats.push('carry');
    if (d.pass_yds  > 0 || d.pass_td > 0) stats.push('pass');
    if (tdc         > 0) stats.push('td');
    if (d.int       > 0) stats.push('int');
    if (d.targets   > 0 && d.rec < 1) stats.push('target');

    var kind = tdc > 0 ? 'td' : (d.int > 0 ? 'neg'
             : ((d.rec >= 1 || d.carries >= 1 || d.pass_yds > 0) ? 'gain' : 'target'));

    // Single-play TD shapes -- read like a call from the booth.
    if (tdc === 1 && d.rec_td === 1 && d.rec === 1 && d.carries < 1) {
      return { desc: 'Hauls in a ' + ry + '-yard touchdown catch', kind: kind, stats: stats };
    }
    if (tdc === 1 && d.rush_td === 1 && d.carries === 1 && d.rec < 1) {
      return { desc: uy > 0 ? ('Breaks a ' + uy + '-yard touchdown run') : 'Punches it in for a touchdown', kind: kind, stats: stats };
    }
    if (tdc === 1 && d.pass_td === 1 && d.rec < 1 && d.carries < 1) {
      return { desc: 'Throws a ' + py + '-yard touchdown pass', kind: kind, stats: stats };
    }

    var segs = [];
    if (d.pass_td > 0) {
      segs.push(d.pass_td === 1 ? 'Throws a touchdown pass' : ('Throws ' + d.pass_td + ' touchdown passes'));
    } else if (d.pass_yds > 0) {
      segs.push(d.pass_yds === py && py > 0 && !(d.rec >= 1 || d.carries >= 1)
        ? ('Completes a pass for ' + _yds(py))
        : ('Moves the chains for ' + _yds(py) + ' through the air'));
    }
    if (d.rec_td > 0) {
      segs.push(d.rec_td === 1 ? 'Hauls in a touchdown catch' : ('Hauls in ' + d.rec_td + ' touchdown catches'));
    } else if (d.rec >= 1) {
      if (d.rec === 1) segs.push('Catches a pass for ' + _yds(ry));
      else segs.push('Hauls in ' + d.rec + ' catches for ' + _yds(ry));
    }
    if (d.rush_td > 0) {
      segs.push(d.rush_td === 1 ? 'Runs it in for a touchdown' : ('Runs in ' + d.rush_td + ' touchdowns'));
    } else if (d.carries >= 1) {
      if (d.carries === 1) segs.push(uy >= 0 ? ('Runs for ' + _yds(uy)) : ('Is stuffed for a loss of ' + _yds(-uy)));
      else segs.push('Rushes ' + d.carries + ' times for ' + _yds(uy));
    }
    if (!segs.length && d.targets > 0) segs.push('Targeted -- pass incomplete');
    if (d.int > 0) segs.push(d.int === 1 ? 'Throws an interception' : ('Throws ' + d.int + ' interceptions'));
    if (!segs.length) return null;
    return { desc: segs.join(' · '), kind: kind, stats: stats };
  }

  // Big-play FX for a freshly-arrived feed event: touchdowns flash (with a
  // confetti burst from the top of the feed when it's the viewer's own team),
  // and explosive non-TD plays (4+ fantasy points on one play) get a lighter
  // pulse. `container` is the scrollable feed list the burst is anchored to.
  function _bigPlayFx(node, ev, container, live) {
    if (ev.kind === 'td') {
      node.classList.add('rz-td-new');
      if (ev.mine) {
        node.classList.add('rz-td-mine');
        if (live && window.brConfetti) {
          try {
            window.brConfetti(container || node, {
              palette: ['#f59e0b', '#fbbf24', '#22c55e', '#ffffff'],
              y: 42, count: 30,
            });
          } catch (e) { /* confetti is decorative */ }
        }
      }
    } else if ((ev.pts || 0) >= 4) {
      node.classList.add('rz-bigplay');
    }
  }

  function _playsFromDiff(pid, oldL, newL, tags, scoring) {
    var pos = _pos(pid);
    var rid = tags.pidToRoster[pid] || '';
    var _pi = (_state.player_info || {})[pid] || {};
    var base = {
      pid: pid, name: _name(pid), pos: pos, nflTeam: _team(pid),
      rosterId: rid, owner: _ownerName(rid), league: _leagueOfRid(rid),
      mine: tags.my.has(rid), opp: tags.opp.has(rid),
      line: _gameLine(pid), ts: Date.now(),
      gameQuarter: _pi.game_quarter || '',
      gameClock:   _pi.game_clock   || '',
    };
    function mkEv(d, oldLine, newLine) {
      var info = _describe(d, pos);
      if (!info) return null;
      var earned = parseFloat((_lineToPts(newLine, scoring) - _lineToPts(oldLine, scoring)).toFixed(2));
      // Cumulative total after this play (boxscore + platform, whichever is higher).
      var totalPts = parseFloat(_totalPtsForPid(pid, scoring).toFixed(2));
      return Object.assign({}, base, {
        desc: info.desc, kind: info.kind, stats: info.stats, pts: earned,
        totalPts: totalPts, ts: Date.now() + Math.random()
      });
    }

    // DEF
    if (pos === 'DEF') {
      var d = {
        sacks:   (newL.sacks  ||0) - (oldL.sacks  ||0),
        def_int: (newL.def_int||0) - (oldL.def_int||0),
        fum_rec: (newL.fum_rec||0) - (oldL.fum_rec||0),
        def_td:  (newL.def_td ||0) - (oldL.def_td ||0)
      };
      if (d.sacks < 0.001 && d.def_int < 0.001 && d.fum_rec < 0.001 && d.def_td < 0.001) return [];
      var ev = mkEv(d, oldL, newL);
      return ev ? [ev] : [];
    }

    // K
    if (pos === 'K') {
      var d = {
        fgm:     (newL.fgm    ||0) - (oldL.fgm    ||0),
        fg_long: newL.fg_long ||0,
        xpm:     (newL.xpm    ||0) - (oldL.xpm    ||0)
      };
      if (d.fgm < 0.001 && d.xpm < 0.001) return [];
      var ev = mkEv(d, oldL, newL);
      return ev ? [ev] : [];
    }

    // Offensive: split into separate receiving, rushing, passing events
    var results = [];
    var Z = { pass_yds:0, pass_td:0, int:0, carries:0, rush_yds:0, rush_td:0, rec:0, rec_yds:0, rec_td:0, targets:0 };

    // Receiving / target
    var recD = Object.assign({}, Z, {
      rec:     (newL.rec    ||0) - (oldL.rec    ||0),
      rec_yds: (newL.rec_yds||0) - (oldL.rec_yds||0),
      rec_td:  (newL.rec_td ||0) - (oldL.rec_td ||0),
      targets: (newL.targets||0) - (oldL.targets||0)
    });
    if (recD.rec > 0.001 || recD.targets > 0.001) {
      var oldRec = Object.assign({}, Z, { rec: oldL.rec||0, rec_yds: oldL.rec_yds||0, rec_td: oldL.rec_td||0, targets: oldL.targets||0 });
      var newRec = Object.assign({}, Z, { rec: newL.rec||0, rec_yds: newL.rec_yds||0, rec_td: newL.rec_td||0, targets: newL.targets||0 });
      var ev = mkEv(recD, oldRec, newRec);
      if (ev) results.push(ev);
    }

    // Rushing
    var rushD = Object.assign({}, Z, {
      carries:  (newL.carries ||0) - (oldL.carries ||0),
      rush_yds: (newL.rush_yds||0) - (oldL.rush_yds||0),
      rush_td:  (newL.rush_td ||0) - (oldL.rush_td ||0)
    });
    if (rushD.carries > 0.001) {
      var oldRush = Object.assign({}, Z, { carries: oldL.carries||0, rush_yds: oldL.rush_yds||0, rush_td: oldL.rush_td||0 });
      var newRush = Object.assign({}, Z, { carries: newL.carries||0, rush_yds: newL.rush_yds||0, rush_td: newL.rush_td||0 });
      var ev = mkEv(rushD, oldRush, newRush);
      if (ev) results.push(ev);
    }

    // Passing
    var passD = Object.assign({}, Z, {
      pass_yds: (newL.pass_yds||0) - (oldL.pass_yds||0),
      pass_td:  (newL.pass_td ||0) - (oldL.pass_td ||0),
      int:      (newL.int     ||0) - (oldL.int     ||0)
    });
    if (passD.pass_yds > 0.001 || passD.pass_td > 0.001 || passD.int > 0.001) {
      var oldPass = Object.assign({}, Z, { pass_yds: oldL.pass_yds||0, pass_td: oldL.pass_td||0, int: oldL.int||0 });
      var newPass = Object.assign({}, Z, { pass_yds: newL.pass_yds||0, pass_td: newL.pass_td||0, int: newL.int||0 });
      var ev = mkEv(passD, oldPass, newPass);
      if (ev) results.push(ev);
    }

    return results;
  }


  function _pidFromPlayName(play, newData) {
    // CRITICAL: Validate explicit play.pid against canonical player_info before trusting it.
    // A backend/provider PID must exist in the canonical index to be used.
    // This prevents wrong-namespace, stale, or malformed PIDs from being blindly accepted.
    var info = (newData && newData.player_info) || _state.player_info || {};
    var pid = play.pid || '';
    if (pid && pid !== '0' && Object.prototype.hasOwnProperty.call(info, String(pid))) {
      return String(pid);
    }
    
    // Explicit PID was either missing or not in canonical index - resolve by name
    var want = String(play.name || '').toLowerCase().trim();
    if (!want) return '';
    
    // Resolution order:
    // 1. Exact normalized full-name match
    // 2. Initial + surname match (already normalized by backend)
    // 3. Team-scoped surname match (handled by backend)
    var keys = Object.keys(info);
    for (var i = 0; i < keys.length; i++) {
      var row = info[keys[i]] || {};
      var nm = String(row.name || '').toLowerCase().trim();
      if (nm && nm === want) return keys[i];
      // DEF rows sometimes arrive as "KC DEF"
      if (row.pos === 'DEF' && row.team) {
        var defLabel = String(row.team).toLowerCase() + ' def';
        if (want === defLabel || want === String(row.team).toLowerCase()) return keys[i];
      }
    }
    return '';
  }

  // NFL play identity: game_id + play_id (no pid)
  // MUST include gid even when play_id exists to prevent cross-game collisions
  function _nflPlayKey(play, gid) {
    var playId = play.play_id || ('seq:' + (play.seq || 0));
    return gid + ':' + playId;
  }
  
  // Fantasy contribution identity: NFL play + pid
  function _contributionKey(play, gid, pid) {
    return _nflPlayKey(play, gid) + ':' + pid;
  }
  
  // Check if contribution data changed (for detecting revisions)
  function _contributionDataChanged(existingContrib, newPlay, isInvalid) {
    if (!existingContrib) return true;
    
    // If validity changed, it's a revision
    if (!!existingContrib.isInvalid !== !!isInvalid) return true;
    
    // Compare stat lines
    var oldLine = existingContrib.line || {};
    var newLine = newPlay.stat_line || {};
    
    // Key stats to compare
    var keys = ['rec', 'rec_yds', 'rec_td', 'targets', 'carries', 'rush_yds', 'rush_td',
                'pass_yds', 'pass_td', 'int', 'fgm', 'xpm', 'sacks', 'def_int', 'fum_rec', 'def_td'];
    
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      var oldVal = parseFloat(oldLine[k] || 0);
      var newVal = parseFloat(newLine[k] || 0);
      if (Math.abs(oldVal - newVal) > 0.001) return true;
    }
    
    // Compare play text (for corrections)
    var oldText = (existingContrib.rawPlayText || '').toLowerCase();
    var newText = (newPlay.play_text || '').toLowerCase();
    if (oldText !== newText) {
      // Text changed - check if it's meaningful (not just formatting)
      if (oldText.replace(/\s+/g, ' ') !== newText.replace(/\s+/g, ' ')) {
        return true;
      }
    }
    
    return false;
  }
  
  // Get description for nullified plays
  function _getNullifiedDesc(playState, rawPlayText) {
    if (playState === 'OVERTURNED') {
      return 'Play overturned by replay review';
    }
    if (playState === 'NULLIFIED') {
      return 'Play nullified by penalty';
    }
    if (playState === 'NO_PLAY') {
      // Check for specific pre-snap penalties
      var text = (rawPlayText || '').toLowerCase();
      if (text.indexOf('false start') >= 0) return 'False start · No Play';
      if (text.indexOf('delay of game') >= 0) return 'Delay of game · No Play';
      if (text.indexOf('encroachment') >= 0) return 'Encroachment · No Play';
      if (text.indexOf('offsides') >= 0) return 'Offsides · No Play';
      return 'No Play';
    }
    if (playState === 'CORRECTED') {
      return 'Stat correction';
    }
    return 'No Play';
  }

  // Select primary actor for a grouped NFL play based on fantasy relevance hierarchy
  function _selectPrimaryActor(contributions) {
    if (!contributions || !contributions.length) return null;
    if (contributions.length === 1) return contributions[0];
    // Hierarchy: receiver > rusher > QB > kicker > DEF
    var recTD = contributions.find(function(c) { return c.line.rec_td > 0; });
    if (recTD) return recTD;
    var rec = contributions.find(function(c) { return c.line.rec > 0; });
    if (rec) return rec;
    var target = contributions.find(function(c) { return c.line.targets > 0 && !c.line.rec; });
    if (target) return target;
    var rushTD = contributions.find(function(c) { return c.line.rush_td > 0; });
    if (rushTD) return rushTD;
    var rush = contributions.find(function(c) { return c.line.carries > 0; });
    if (rush) return rush;
    var passInt = contributions.find(function(c) { return c.line.int > 0; });
    if (passInt) return passInt;
    var passTD = contributions.find(function(c) { return c.line.pass_td > 0; });
    if (passTD) return passTD;
    var pass = contributions.find(function(c) { return c.line.pass_yds > 0; });
    if (pass) return pass;
    var defTD = contributions.find(function(c) { return c.line.def_td > 0; });
    if (defTD) return defTD;
    var defPlay = contributions.find(function(c) { return c.line.sacks || c.line.def_int || c.line.fum_rec; });
    if (defPlay) return defPlay;
    var kick = contributions.find(function(c) { return c.line.fgm || c.line.xpm; });
    if (kick) return kick;
    return contributions[0];
  }

  // Improve play description with fantasy-focused copy
  // Helper to check if play is nullified
  function _isPlayNullified(event) {
    return event && (event.isNullified || event.playState !== 'VALID');
  }
  
  function _improvePlayDesc(primary, contributions, rawText) {
    var line = primary.line || {};
    var pos = primary.pos;
    var yds = Math.round(line.rec_yds || line.rush_yds || line.pass_yds || 0);
    // Receiving TD
    if (line.rec_td > 0 && line.rec > 0) {
      var qb = contributions.find(function(c) { return c.line.pass_td > 0 && c.pid !== primary.pid; });
      var qbName = qb ? qb.name : '';
      return yds > 0
        ? yds + '-yard TD reception' + (qbName ? ' from ' + qbName : '')
        : 'TD reception' + (qbName ? ' from ' + qbName : '');
    }
    // Completed pass (receiver primary)
    if (line.rec > 0) {
      var qb2 = contributions.find(function(c) { return (c.line.pass_yds > 0 || c.line.pass_td > 0) && c.pid !== primary.pid; });
      var qbName2 = qb2 ? qb2.name : '';
      // Handle negative yardage
      if (yds < 0) {
        return Math.abs(yds) + '-yard loss on reception' + (qbName2 ? ' from ' + qbName2 : '');
      }
      return yds > 0
        ? yds + '-yard reception' + (qbName2 ? ' from ' + qbName2 : '')
        : 'Reception' + (qbName2 ? ' from ' + qbName2 : '');
    }
    // Incomplete target
    if (line.targets > 0 && !line.rec) {
      var qb3 = contributions.find(function(c) { return c.pos === 'QB' && c.pid !== primary.pid; });
      var qbName3 = qb3 ? qb3.name : '';
      return 'Target' + (qbName3 ? ' from ' + qbName3 : '') + ' · incomplete';
    }
    // Rushing TD
    if (line.rush_td > 0 && line.carries > 0) {
      return yds > 0 ? yds + '-yard rushing TD' : 'Rushing TD';
    }
    // Rush
    if (line.carries > 0) {
      if (pos === 'QB') return yds > 0 ? yds + '-yard scramble' : 'Scramble';
      return yds > 0 ? yds + '-yard run' : 'Run';
    }
    // Interception
    if (line.int > 0) return 'Pass intercepted';
    // Passing TD (QB primary - rare, only when no receiver mapped)
    if (line.pass_td > 0) return yds > 0 ? yds + '-yard TD pass' : 'TD pass';
    // Kicker
    if (line.fgm > 0) {
      var dist = Math.round(line.fg_long || line.fg_yds || 0);
      return dist > 0 ? dist + '-yard field goal' : 'Field goal';
    }
    if (line.xpm > 0) return 'Extra point';
    // DEF - improved sack description
    if (line.sacks > 0 || line.sack > 0) {
      var sackCount = line.sacks || line.sack || 0;
      // Try to find QB being sacked
      var qb = contributions.find(function(c) { return c.pos === 'QB' && c.pid !== primary.pid; });
      var qbName = qb ? qb.name : '';
      if (sackCount === 1) {
        return qbName ? 'Sack of ' + qbName : 'Sack';
      }
      return sackCount + ' sacks';
    }
    if (line.def_int > 0) return 'Interception';
    if (line.fum_rec > 0) return 'Fumble recovery';
    if (line.def_td > 0) return 'Defensive TD';
    // Fallback to raw or generic
    if (rawText) return rawText;
    var info = _describe(line, pos);
    return info ? info.desc : 'Play';
  }
  
  // Rebuild cumulative stats from valid PBP contributions
  function _rebuildCumulativeStats(pid, upToSeq, gameId) {
    if (!pid || !gameId) return null;
    
    var validContribs = _pbpHistory.filter(function(c) {
      return c.pid === pid && 
             c.gameId === gameId && 
             !c.isInvalid && 
             c.seq <= upToSeq;
    });
    
    if (validContribs.length === 0) return null;
    
    // Sort by seq
    validContribs.sort(function(a, b) { return a.seq - b.seq; });
    
    // Sum up stats
    var cume = {
      rec: 0, rec_yds: 0, rec_td: 0, targets: 0,
      carries: 0, rush_yds: 0, rush_td: 0,
      pass_yds: 0, pass_td: 0, pass_att: 0, pass_cmp: 0, int: 0,
      fgm: 0, fga: 0, xpm: 0, xpa: 0,
      sacks: 0, def_int: 0, fum_rec: 0, def_td: 0
    };
    
    validContribs.forEach(function(c) {
      var line = c.line || {};
      Object.keys(cume).forEach(function(k) {
        cume[k] += parseFloat(line[k] || 0);
      });
    });
    
    return cume;
  }

  // Calculate post-play cumulative fantasy total from play.cume stats
  function _cumeToFantasyPts(cume, scoring, pos) {
    if (!cume || typeof cume !== 'object') return null;
    return parseFloat(_lineToPts(cume, scoring, pos).toFixed(2));
  }
  
  // Recalculate fantasy totals after play revision
  function _recalculateTotals(pid, gameId, newData, scoring) {
    if (!pid || !gameId) return null;
    
    // Find latest seq for this player in this game
    var latestSeq = -1;
    _pbpHistory.forEach(function(c) {
      if (c.pid === pid && c.gameId === gameId && c.seq > latestSeq) {
        latestSeq = c.seq;
      }
    });
    
    if (latestSeq < 0) return null;
    
    // Rebuild cumulative stats
    var cume = _rebuildCumulativeStats(pid, latestSeq, gameId);
    if (!cume) return null;
    
    var pos = _pos(pid);
    return _cumeToFantasyPts(cume, scoring, pos);
  }

  function _eventsFromPbp(newData, tags, scFor) {
    var byGame = newData.pbp_by_game || {};
    var newContributions = [];
    var revisedPlayKeys = new Set(); // Track plays that need updates
    
    // Process new contributions
    Object.keys(byGame).forEach(function(gid) {
      _pbpGames[gid] = true;
      (byGame[gid] || []).forEach(function(play) {
        var pid = _pidFromPlayName(play, newData);
        if (!pid || pid === '0') return;
        
        // Roster ownership is OPTIONAL metadata - do not gate on it
        var rid = tags.pidToRoster[pid] || '';
        
        var contribKey = _contributionKey(play, gid, pid);
        var playKey = _nflPlayKey(play, gid);
        var playState = play.play_state || 'VALID';
        var isInvalid = playState !== 'VALID';
        
        // Check if this is a revision of an existing contribution
        var existingContrib = _contributionsByKey[contribKey];
        var isRevision = !!existingContrib;
        
        // For invalid plays (NO_PLAY, NULLIFIED, OVERTURNED), create zeroed contribution
        if (isInvalid) {
          // If we've seen this contribution before and it was valid, this is a revision
          if (existingContrib && !existingContrib.isInvalid) {
            isRevision = true;
          }
          // Skip creating new invalid contributions unless it's a revision
          if (!isRevision) return;
        }
        
        // Check if contribution data changed (for revisions)
        if (isRevision) {
          var dataChanged = _contributionDataChanged(existingContrib, play, isInvalid);
          if (!dataChanged) return; // No change, skip
        }
        
        // Mark as seen after validation
        _seenContributions.add(contribKey);
        var line = play.stat_line || {};
        var scoring = scFor(pid);
        var pos = _pos(pid);
        var pts = parseFloat(_lineToPts(line, scoring, pos).toFixed(2));
        // Post-play total: use cume if available, else fall back to current total
        var cumePts = _cumeToFantasyPts(play.cume, scoring, pos);
        var totalPts = cumePts !== null ? cumePts : parseFloat(_totalPtsForPid(pid, scoring, newData).toFixed(2));
        var kind = play.is_td ? 'td' : (line.int > 0 ? 'neg'
                 : ((line.rec || line.carries || line.pass_yds || line.fgm || line.sacks || line.sack
                     || line.def_td || line.def_int || line.fum_rec) ? 'gain' : 'target'));
        var stats = [];
        if (line.rec) stats.push('reception');
        if (line.carries) stats.push('carry');
        if (line.pass_yds || line.pass_td) stats.push('pass');
        if (play.is_td || line.pass_td || line.rush_td || line.rec_td || line.def_td) stats.push('td');
        if (line.int || line.def_int) stats.push('int');
        if (line.targets && !line.rec) stats.push('target');
        if (line.fgm || line.xpm) stats.push('kick');
        if (line.sacks || line.sack) stats.push('sack');
        var contrib = {
          pid: pid, name: _name(pid), pos: pos, nflTeam: _team(pid),
          rosterId: rid || '', owner: rid ? _ownerName(rid) : '', league: rid ? _leagueOfRid(rid) : '',
          mine: rid ? tags.my.has(rid) : false, opp: rid ? tags.opp.has(rid) : false,
          line: line, pts: pts, kind: kind, stats: stats,
          playKey: playKey,
          contribKey: contribKey,
          rawPlayText: play.play_text || '',
          quarter: play.quarter || ((newData.player_info || {})[pid] || {}).game_quarter || '',
          clock: play.clock || ((newData.player_info || {})[pid] || {}).game_clock || '',
          down: play.down || '',
          distance: play.distance || '',
          yardLine: play.yard_line || '',
          seq: play.seq != null ? play.seq : 0,
          gameId: gid,
          cume: play.cume || null,
          cumeStatLine: play.cume || null,
          totalPts: totalPts,
          scoring: scoring,
          playState: playState,
          isInvalid: isInvalid,
          isRevision: isRevision
        };
        
        // Store in global contribution map
        _contributionsByKey[contribKey] = contrib;
        
        newContributions.push(contrib);
        
        // Add to history only if not a revision
        if (!isRevision) {
          _pbpHistory.push(contrib);
        }
        
        // Mark play for update
        revisedPlayKeys.add(playKey);
      });
    });
    // Merge new contributions into canonical play groups
    newContributions.forEach(function(c) {
      var group = _playGroupsByKey[c.playKey];
      if (!group) {
        var contribsByKey = {};
        contribsByKey[c.contribKey] = c;
        _playGroupsByKey[c.playKey] = {
          contributionsByKey: contribsByKey,
          needsUpdate: true,
          gameId: c.gameId,
          seq: c.seq,
          playState: c.playState
        };
      } else {
        // Update or add contribution
        if (!group.contributionsByKey) {
          // Migrate old structure
          group.contributionsByKey = {};
          (group.contributions || []).forEach(function(old) {
            group.contributionsByKey[old.contribKey] = old;
          });
        }
        group.contributionsByKey[c.contribKey] = c;
        group.needsUpdate = true;
        // Update gameId/seq/playState from latest contribution
        if (c.gameId) group.gameId = c.gameId;
        if (c.seq != null) group.seq = c.seq;
        if (c.playState) group.playState = c.playState;
      }
    });
    
    // Mark revised plays for update
    revisedPlayKeys.forEach(function(playKey) {
      var group = _playGroupsByKey[playKey];
      if (group) group.needsUpdate = true;
    });
    // Generate/update events for plays that need it
    var events = [];
    Object.keys(_playGroupsByKey).forEach(function(playKey) {
      var group = _playGroupsByKey[playKey];
      if (!group.needsUpdate) return;
      group.needsUpdate = false;
      
      // Derive contributions array from contributionsByKey
      var contribs = group.contributionsByKey 
        ? Object.keys(group.contributionsByKey).map(function(k) { return group.contributionsByKey[k]; })
        : (group.contributions || []);
      
      // Filter out invalid contributions for primary selection
      var validContribs = contribs.filter(function(c) { return !c.isInvalid; });
      
      // If all contributions are invalid, mark play as nullified
      var playState = group.playState || 'VALID';
      var isNullified = playState !== 'VALID' || validContribs.length === 0;
      
      if (isNullified) {
        // Create nullified event
        var firstContrib = contribs[0];
        if (!firstContrib) return;
        
        var isNewPlay = !_seenPlayIds.has(playKey);
        if (isNewPlay) _seenPlayIds.add(playKey);
        
        // Recalculate totals after nullification
        var recalcTotal = _recalculateTotals(
          firstContrib.pid, 
          firstContrib.gameId, 
          newData, 
          firstContrib.scoring
        );
        
        var nullifiedEvent = {
          pid: firstContrib.pid,
          name: firstContrib.name,
          pos: firstContrib.pos,
          nflTeam: firstContrib.nflTeam,
          rosterId: firstContrib.rosterId,
          owner: firstContrib.owner,
          league: firstContrib.league,
          mine: firstContrib.mine,
          opp: firstContrib.opp,
          line: {},
          ts: Date.now() + firstContrib.seq * 0.001,
          gameQuarter: firstContrib.quarter,
          gameClock: firstContrib.clock,
          down: firstContrib.down,
          distance: firstContrib.distance,
          yardLine: firstContrib.yardLine,
          desc: _getNullifiedDesc(playState, firstContrib.rawPlayText),
          kind: 'nullified',
          stats: [],
          pts: 0,
          statLine: {},
          cume: null,
          cumeStatLine: null,
          totalPts: recalcTotal !== null ? recalcTotal : 0,
          playId: playKey,
          fromPbp: true,
          contributions: contribs,
          isUpdate: !isNewPlay,
          isNullified: true,
          playState: playState,
          impact: '',
          gameId: group.gameId || firstContrib.gameId,
          seq: group.seq != null ? group.seq : firstContrib.seq
        };
        group.primaryEvent = nullifiedEvent;
        events.push(nullifiedEvent);
        return;
      }
      
      // Normal play processing with valid contributions
      var primary = _selectPrimaryActor(validContribs);
      if (!primary) return;
      
      var desc = _improvePlayDesc(primary, validContribs, primary.rawPlayText);
      var allMine = validContribs.some(function(c) { return c.mine; });
      var allOpp = validContribs.some(function(c) { return c.opp; });
      var isNewPlay = !_seenPlayIds.has(playKey);
      if (isNewPlay) _seenPlayIds.add(playKey);
      
      var event = {
        pid: primary.pid,
        name: primary.name,
        pos: primary.pos,
        nflTeam: primary.nflTeam,
        rosterId: primary.rosterId,
        owner: primary.owner,
        league: primary.league,
        mine: allMine,
        opp: allOpp,
        line: _gameLine(primary.pid),
        ts: Date.now() + primary.seq * 0.001,
        gameQuarter: primary.quarter,
        gameClock: primary.clock,
        down: primary.down,
        distance: primary.distance,
        yardLine: primary.yardLine,
        desc: desc,
        kind: primary.kind,
        stats: primary.stats,
        pts: primary.pts,
        statLine: primary.line,
        cume: primary.cume,
        cumeStatLine: primary.cumeStatLine,
        totalPts: primary.totalPts,
        playId: playKey,
        fromPbp: true,
        contributions: validContribs,
        isUpdate: !isNewPlay,
        isNullified: false,
        playState: playState,
        impact: '',
        gameId: group.gameId || primary.gameId,
        seq: group.seq != null ? group.seq : primary.seq
      };
      group.primaryEvent = event;
      events.push(event);
    });
    return events;
  }

  function _detectChanges(newData) {
    var tags = _rosterTags(newData);
    // Resolve scoring per player by league (user scope spans multiple leagues);
    // fall back to the single top-level scoring.
    var _sbl = newData.scoring_by_league, _pidLg = newData.pid_league || {};
    var _scFor = function(pid) {
      if (_sbl) {
        // Prefer the league of the roster this pid is tagged to (viewer-first),
        // so shared players don't inherit another league's last-write scoring.
        var rid = tags.pidToRoster[pid];
        var lid = null;
        if (rid) {
          var mm = (newData.matchups || []).find(function(x) { return String(x.roster_id) === String(rid); });
          if (mm && mm.league_id) lid = mm.league_id;
        }
        if (!lid) lid = _pidLg[pid];
        if (lid && _sbl[lid]) return _sbl[lid];
      }
      return newData.scoring || {};
    };
    var allEvents = [], handled = {};
    // Real play-by-play first (Tank01 / demo). Mark those games so we don't
    // also invent bulk-diff blurbs for the same snaps.
    var pbpEvents = _eventsFromPbp(newData, tags, _scFor);
    // No arbitrary limits - full PBP history retained
    // Separate new plays from updates
    var newPlays = [];
    var updates = [];
    pbpEvents.forEach(function(ev) {
      if (ev.isUpdate) {
        updates.push(ev);
      } else {
        newPlays.push(ev);
        allEvents.push(ev);
      }
      handled[ev.pid] = true;
    });
    // Update existing feed entries for plays that got new contributions
    updates.forEach(function(upd) {
      for (var i = 0; i < _feed.length; i++) {
        if (_feed[i].playId === upd.playId) {
          _feed[i] = upd;
          break;
        }
      }
    });
    Object.keys(newData.player_info || {}).forEach(function(pid) {
      var pi = newData.player_info[pid] || {};
      var newL = pi.stat_line;
      if (!newL) return;
      if (handled[pid]) return;
      var code = String(pi.game_code || '');
      var gid = pi.game_id || '';
      // Live/final: real Tank01 play-by-play lines only -- never invent
      // boxscore-diff narratives or bulk "Scored X pts" cards.
      if (code === '1' || code === '2' || (gid && _pbpGames[gid])) {
        handled[pid] = true;
        return;
      }
      handled[pid] = true;
      var evs = _playsFromDiff(pid, _prevStats[pid] || {}, newL, tags, _scFor(pid));
      evs.forEach(function(ev) { allEvents.push(ev); });
    });
    (newData.matchups || []).forEach(function(m) {
      var pp = m.players_points || {};
      Object.keys(pp).forEach(function(pid) {
        if (handled[pid] || pid === '0') return;
        var info = newData.player_info[pid] || {};
        var gid = info.game_id || '';
        var code = String(info.game_code || '');
        // Live/final or any game with a PBP attempt: no bulk point dumps.
        if (code === '1' || code === '2' || (gid && _pbpGames[gid])) {
          handled[pid] = true;
          return;
        }
        var delta = parseFloat((parseFloat(pp[pid] || 0) - parseFloat(_prevPts[pid] || 0)).toFixed(2));
        if (delta <= 0.05) return;
        var rid = tags.pidToRoster[pid] || String(m.roster_id);
        allEvents.push({
          pid: pid, name: info.name || pid, pos: info.pos || '', nflTeam: info.team || '',
          rosterId: rid, owner: _ownerName(rid), league: _leagueOfRid(rid),
          desc: 'Scored ' + delta.toFixed(1) + ' pts', kind: 'gain', stats: ['pts'], pts: delta,
          totalPts: parseFloat(_totalPtsForPid(pid, _scFor(pid), newData).toFixed(2)),
          mine: tags.my.has(rid), opp: tags.opp.has(rid), line: '', ts: Date.now()
        });
      });
    });
    // Chronological within this batch (newest first), then prepend.
    allEvents = _chronoSort(allEvents);
    // Prepend in reverse so the sorted order survives unshift.
    for (var i = allEvents.length - 1; i >= 0; i--) _feed.unshift(allEvents[i]);
    // Keep the feed chronological overall (newest first) for first-page paint.
    _feed = _chronoSort(_feed);
    // No arbitrary cap - full history retained, rendering controlled by pagination

    // Push notification + audio chime for my TDs + log to history. Only for TDs
    // found by a live poll -- never the initial backfill of already-played snaps.
    // Dedupe by playId so grouped plays (QB+receiver) only trigger ONE alert.
    var myTDs = _alertsArmed
      ? allEvents.filter(function(ev) { return ev.kind === 'td' && ev.mine && !ev.isUpdate; })
      : [];
    // Dedupe by playId - one alert per NFL play regardless of contributors
    var seenTdPlays = new Set();
    myTDs = myTDs.filter(function(ev) {
      var key = ev.playId || (ev.pid + ':' + ev.ts);
      if (seenTdPlays.has(key)) return false;
      seenTdPlays.add(key);
      return true;
    });
    if (myTDs.length) {
      myTDs.forEach(function(ev) {
        _notifHistory.unshift({ ts: Date.now(), name: ev.name, desc: ev.desc, pts: ev.pts, kind: ev.kind });
      });
      if (_notifHistory.length > 50) _notifHistory = _notifHistory.slice(0, 50);
      try { localStorage.setItem('rz-notif-history', JSON.stringify(_notifHistory)); } catch (_) {}
      _playTDBeep();
      try { if (navigator.vibrate) navigator.vibrate([100, 50, 200]); } catch (_) {}
      try {
        if (navigator.serviceWorker && navigator.serviceWorker.ready) {
          navigator.serviceWorker.ready.then(function(sw) {
            myTDs.forEach(function(ev) {
              var p = sw.showNotification('TD: ' + ev.name, {
                body: ev.desc + (ev.pts > 0 ? '  +' + _fmtFantasyPrecise(ev.pts) + ' pts' : ''),
                icon: '/static/BR_Mark.png?v=f4228e0e', tag: 'rz-td-' + (ev.playId || ev.pid)
              });
              if (p && p.catch) p.catch(function() {});
            });
          }).catch(function() {});
        }
      } catch (_) {}
    }

    var _specialCount = 0;

    // Stat milestones: fire a feed event when a player crosses a threshold for first time
    var _MS_DEFS = [
      { key: 'rush_yds_100', field: 'rush_yds', thr: 100, desc: '100 rush yds' },
      { key: 'rush_yds_150', field: 'rush_yds', thr: 150, desc: '150 rush yds' },
      { key: 'pass_yds_300', field: 'pass_yds', thr: 300, desc: '300 pass yds' },
      { key: 'pass_yds_400', field: 'pass_yds', thr: 400, desc: '400 pass yds' },
      { key: 'rec_yds_100', field: 'rec_yds',  thr: 100, desc: '100 rec yds' },
      { key: 'td_2',        field: '__tds',     thr: 2,   desc: '2 TDs' },
      { key: 'td_3',        field: '__tds',     thr: 3,   desc: '3 TDs' },
    ];
    Object.keys(newData.player_info || {}).forEach(function(pid) {
      var sl = (newData.player_info[pid] || {}).stat_line;
      if (!sl) return;
      var seen = _milestonesSeen[pid] || {};
      var tds = (sl.rush_td||0) + (sl.rec_td||0) + (sl.pass_td||0);
      var rid = tags.pidToRoster[pid] || '';
      _MS_DEFS.forEach(function(ms) {
        if (seen[ms.key]) return;
        var val = ms.field === '__tds' ? tds : (sl[ms.field] || 0);
        if (val < ms.thr) return;
        seen[ms.key] = true;
        _specialCount++;
        _feed.unshift({
          pid: pid, name: _name(pid), pos: _pos(pid), nflTeam: _team(pid),
          rosterId: rid, owner: _ownerName(rid), league: _leagueOfRid(rid),
          mine: tags.my.has(rid), opp: tags.opp.has(rid),
          desc: ms.desc + '!', kind: 'milestone', stats: ['milestone'],
          pts: 0, ts: Date.now() + Math.random(),
          line: '', gameQuarter: (newData.player_info[pid] || {}).game_quarter || '',
          gameClock: (newData.player_info[pid] || {}).game_clock || ''
        });
      });
      _milestonesSeen[pid] = seen;
    });

    // Injury-status changes: fire a feed event when a rostered player's status worsens mid-game
    Object.keys(newData.player_info || {}).forEach(function(pid) {
      var info = newData.player_info[pid] || {};
      var now = info.injury_status || '';
      var was = _prevInjury[pid];
      _prevInjury[pid] = now;
      if (was === undefined || now === was) return;
      if ((_INJ_RANK[now] || 0) <= (_INJ_RANK[was] || 0)) return; // only surface worsening
      var rid = tags.pidToRoster[pid] || '';
      // Only for players in a viewable matchup (mine or opp), to keep the feed relevant
      if (!tags.my.has(rid) && !tags.opp.has(rid)) return;
      _specialCount++;
      _feed.unshift({
        pid: pid, name: _name(pid), pos: _pos(pid), nflTeam: _team(pid),
        rosterId: rid, owner: _ownerName(rid), league: _leagueOfRid(rid),
        mine: tags.my.has(rid), opp: tags.opp.has(rid),
        desc: 'Injury: now ' + _injLabel(now), kind: 'neg', stats: ['injury'],
        pts: 0, ts: Date.now() + Math.random(),
        line: '', gameQuarter: info.game_quarter || '', gameClock: info.game_clock || ''
      });
    });

    // Blowout warnings: one-time alert when an NFL game is 21+ apart in Q3/Q4
    var _games = {};
    Object.keys(newData.player_info || {}).forEach(function(pid) {
      var info = newData.player_info[pid] || {};
      var gid = info.game_id || '';
      if (!gid || String(info.game_code || '') !== '1') return; // live games only
      if (_games[gid]) {
        if (_quarterNum(info.game_quarter) > _games[gid].qn) _games[gid].qn = _quarterNum(info.game_quarter);
        return;
      }
      _games[gid] = {
        home: info.home, away: info.away,
        hp: parseFloat(info.home_pts || 0), ap: parseFloat(info.away_pts || 0),
        qn: _quarterNum(info.game_quarter), qLabel: info.game_quarter || '', clock: info.game_clock || ''
      };
    });
    Object.keys(_games).forEach(function(gid) {
      if (_blowoutSeen[gid]) return;
      var g = _games[gid];
      var spread = Math.abs(g.hp - g.ap);
      if (g.qn < 3 || spread < 21) return;
      _blowoutSeen[gid] = true;
      _specialCount++;
      var leader = g.hp >= g.ap ? g.home : g.away;
      var trailer = g.hp >= g.ap ? g.away : g.home;
      _feed.unshift({
        pid: '0', name: 'Blowout Alert', pos: '', nflTeam: leader,
        rosterId: '', owner: '', league: '',
        mine: false, opp: false,
        desc: leader + ' leading ' + trailer + ' by ' + spread + ', watch for reduced volume',
        kind: 'neg', stats: ['blowout'], pts: 0, ts: Date.now() + Math.random(),
        line: g.away + ' ' + g.ap + ' @ ' + g.home + ' ' + g.hp,
        gameQuarter: g.qLabel, gameClock: g.clock
      });
    });

    // Lead change alerts: fire once when the leading side flips in a matchup
    var _lcGroups = {};
    (newData.matchups || []).forEach(function(m) {
      var mid = String(m.matchup_id);
      (_lcGroups[mid] = _lcGroups[mid] || []).push(m);
    });
    Object.keys(_lcGroups).forEach(function(mid) {
      var pair = _lcGroups[mid];
      if (pair.length < 2) return;
      var a = pair[0], b = pair[1];
      var ptsA = parseFloat(a.points || 0), ptsB = parseFloat(b.points || 0);
      if (ptsA <= 0 && ptsB <= 0) return;
      var newLdr = ptsA >= ptsB ? String(a.roster_id) : String(b.roster_id);
      var prevLdr = _prevLeader[mid];
      _prevLeader[mid] = newLdr;
      if (prevLdr === undefined || prevLdr === null || prevLdr === newLdr) return;
      // Leader has flipped
      var isMyMid = _isMyRid(a.roster_id) || _isMyRid(b.roster_id);
      _specialCount++;
      var trailRid = newLdr === String(a.roster_id) ? String(b.roster_id) : String(a.roster_id);
      var leadPts = Math.max(ptsA, ptsB), trailPts = Math.min(ptsA, ptsB);
      _feed.unshift({
        pid: '0', name: 'Lead Change', pos: '', nflTeam: '',
        rosterId: newLdr, owner: _ownerName(newLdr) || 'Team',
        league: _leagueOfRid(newLdr),
        mine: isMyMid && _isMyRid(newLdr), opp: isMyMid && _isMyRid(trailRid),
        desc: (_ownerName(newLdr) || 'Team') + ' takes the lead (' + _fmt(leadPts) + ' – ' + _fmt(trailPts) + ')',
        kind: 'gain', stats: ['lead_change'],
        pts: 0, ts: Date.now() + Math.random(),
        line: '', gameQuarter: '', gameClock: ''
      });
    });

    // Increment unread count when user isn't on Plays tab
    if (_activeTab !== 'plays' && (allEvents.length + _specialCount)) _unreadCount += (allEvents.length + _specialCount);

    // Track which rosters had point changes (for score flash) + capture score delta.
    // When a hero card is focused, only accumulate deltas for that matchup so the
    // "+N this update" strip doesn't mix other My Leagues games.
    _scoreDelta = { me: 0, opp: 0 };
    var _focusMids = new Set();
    var _focusMeRid = null;
    if (_heroMid) {
      if (_scope === 'user') {
        var _fm = (newData.matchups || []).find(function(m) { return String(m.roster_id) === _heroMid; });
        if (_fm) { _focusMids.add(String(_fm.matchup_id)); _focusMeRid = String(_fm.roster_id); }
      } else {
        _focusMids.add(String(_heroMid));
        var _fpair = (newData.matchups || []).filter(function(m) { return String(m.matchup_id) === _heroMid; });
        var _fmine = _fpair.find(function(m) { return _isMyRid(m.roster_id); }) || _fpair[0];
        if (_fmine) _focusMeRid = String(_fmine.roster_id);
      }
    }
    if (!_focusMids.size) {
      (newData.matchups || []).forEach(function(m) {
        if (_isMyRid(m.roster_id)) _focusMids.add(String(m.matchup_id));
      });
    }
    (newData.matchups || []).forEach(function(m) {
      var rid = String(m.roster_id);
      var newPts = parseFloat(m.points || 0);
      var oldPts = _prevMatchupPts[rid];
      if (oldPts !== undefined && Math.abs(newPts - oldPts) > 0.01) _flashRids.add(rid);
      if (oldPts !== undefined && _focusMids.has(String(m.matchup_id))) {
        var delta = parseFloat((newPts - oldPts).toFixed(1));
        if (_focusMeRid ? rid === _focusMeRid : _isMyRid(rid)) _scoreDelta.me += delta;
        else _scoreDelta.opp += delta;
      }
      _prevMatchupPts[rid] = newPts;
    });
  }

  // ── Filters ────────────────────────────────────────────────────────────────────
  function _nflMatchupOptions() {
    // Unique NFL games (away @ home) keyed by game_id. Prefer the authoritative
    // `_state.games` collection; fall back to deriving from player_info so an
    // older payload still yields a usable list.
    var byId = {};
    var addRow = function(gid, away, home, code, epoch) {
      if (!gid || !away || !home || byId[gid]) return;
      byId[gid] = {
        id: gid,
        label: away + ' @ ' + home,
        away: away,
        home: home,
        code: String(code || '0'),
        epoch: parseFloat(epoch || 0) || 0
      };
    };
    var games = _state.games || {};
    Object.keys(games).forEach(function(gid) {
      var g = games[gid] || {};
      addRow(gid, g.away || '', g.home || '', g.game_code, g.game_time_epoch);
    });
    Object.keys(_state.player_info || {}).forEach(function(pid) {
      var p = _state.player_info[pid] || {};
      addRow(p.game_id || '', p.away || '', p.home || '', p.game_code, p.game_time_epoch);
    });
    // Deterministic, stable slate order (§7): live → upcoming → final, then by
    // kickoff time, then label. Sorting by kickoff (not mutable score/clock)
    // keeps the row from reshuffling on every poll.
    var rank = function(c) { return c === '1' ? 0 : c === '0' ? 1 : 2; };
    return Object.keys(byId).map(function(k) { return byId[k]; }).sort(function(a, b) {
      var rd = rank(a.code) - rank(b.code);
      if (rd) return rd;
      if (a.epoch !== b.epoch) return a.epoch - b.epoch;
      return a.label.localeCompare(b.label);
    });
  }
  // Cleared at the top of each render/refresh pass so repeated per-player
  // lookups within one pass don't re-sort PBP for the same game.
  var _giCache = {};
  function _resetGameCache() { _giCache = {}; }
  function _nflGameInfo(gid) {
    if (!gid || gid === 'all') return null;
    if (Object.prototype.hasOwnProperty.call(_giCache, gid)) return _giCache[gid];
    var _row = _nflGameInfoUncached(gid);
    _giCache[gid] = _row;
    return _row;
  }
  function _nflGameInfoUncached(gid) {
    if (!gid || gid === 'all') return null;
    var games = _state.games || {};
    // Start with the server scoreboard when available, then enrich missing
    // situation fields from the PBP already loaded by the client. The server
    // row can legitimately arrive before the first situation snapshot.
    var info = _state.player_info || {};
    var row = games[gid] ? Object.assign({}, games[gid]) : null;
    if (!row) {
      Object.keys(info).some(function(pid) {
        var p = info[pid];
        if ((p.game_id || '') !== gid) return false;
        row = {
          game_id: gid,
          away: p.away || '', home: p.home || '',
          away_pts: p.away_pts || '', home_pts: p.home_pts || '',
          game_status: p.game_status || '', game_code: String(p.game_code || ''),
          game_clock: p.game_clock || '', game_quarter: p.game_quarter || '',
          game_time_epoch: p.game_time_epoch || 0,
          possession: '', down: '', distance: '', yard_line: ''
        };
        return true;
      });
    }
    if (!row) return null;
    var plays = (_state.pbp_by_game || {})[gid] || [];
    var playTeam = function(play) {
      if (play.team) return play.team;
      var pid = String(play.pid || '');
      return pid && info[pid] ? (info[pid].team || '') : '';
    };
    // Provider arrays are not guaranteed to be ordered. Pick the greatest
    // sequence carrying usable field context rather than the last array row.
    var ordered = plays.slice().sort(function(a, b) {
      var as = parseFloat((a || {}).seq), bs = parseFloat((b || {}).seq);
      if (!isFinite(as)) as = -1;
      if (!isFinite(bs)) bs = -1;
      return bs - as;
    });
    var best = ordered.find(function(pl) {
      pl = pl || {};
      return !!(playTeam(pl) || pl.down || pl.distance || pl.yard_line || pl.clock || pl.quarter);
    }) || null;
    if (best) {
      row.possession = playTeam(best) || row.possession || '';
      row.down = best.down || row.down || '';
      row.distance = best.distance || row.distance || '';
      row.yard_line = best.yard_line || row.yard_line || '';
      row.game_clock = best.clock || row.game_clock || '';
      row.game_quarter = best.quarter || row.game_quarter || '';
    }
    return row;
  }
  var _POS_LIST  = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF'];
  var _STAT_LIST = [['td','TD'], ['reception','Reception'], ['carry','Carry'],
                    ['pass','Pass'], ['target','Target'], ['int','INT'], ['milestone','Milestone'], ['lead_change','Lead']];

  function _eventMatches(ev) {
    // Ownership filters inspect ALL contributions (QB + receiver both count)
    if (_myTeamOnly && !ev.mine) return false;
    if (_bigPlaysOnly && !_isBigPlay(ev)) return false;
    if (_filters.nfl !== 'all') {
      var gid = ((_state.player_info || {})[ev.pid] || {}).game_id || '';
      if (gid !== _filters.nfl) return false;
    }
    // Position filter checks primary actor only
    if (_filters.pos !== 'all' && ev.pos !== _filters.pos) return false;
    if (_filters.stat !== 'all' && (ev.stats || []).indexOf(_filters.stat) < 0) return false;
    // Hero matchup: check if ANY contribution involves hero matchup players
    if (_heroMid) {
      var hp = _heroMatchupPids();
      if (hp) {
        var hasHero = hp.has(ev.pid);
        if (!hasHero && ev.contributions) {
          hasHero = ev.contributions.some(function(c) { return hp.has(c.pid); });
        }
        if (!hasHero) return false;
      }
    }
    return true;
  }
  function _topMatches(pid, rid) {
    if (_filters.nfl !== 'all') {
      var gid = ((_state.player_info || {})[pid] || {}).game_id || '';
      if (gid !== _filters.nfl) return false;
    }
    if (_filters.pos !== 'all' && _pos(pid) !== _filters.pos) return false;
    if (_heroMid) { var hp2 = _heroMatchupPids(); if (hp2 && !hp2.has(pid)) return false; }
    return true;
  }

  function _heroLabel() {
    if (!_heroMid) return '';
    if (_scope === 'league') {
      var groups = {};
      (_state.matchups || []).forEach(function(m) {
        var mid = String(m.matchup_id);
        (groups[mid] = groups[mid] || []).push(m);
      });
      var pair = groups[_heroMid] || [];
      if (pair.length >= 2) return (_ownerName(pair[0].roster_id) || 'Team') + ' vs ' + (_ownerName(pair[1].roster_id) || 'Team');
      return 'Matchup ' + _heroMid;
    }
    var m = (_state.matchups || []).find(function(x) { return String(x.roster_id) === _heroMid; });
    return (m && m.league_name) || 'Selected League';
  }

  function _nflFilterLabel(gid) {
    var opts = _nflMatchupOptions();
    for (var i = 0; i < opts.length; i++) {
      if (opts[i].id === gid) return opts[i].label;
    }
    var g = _nflGameInfo(gid);
    if (g && g.away && g.home) return g.away + ' @ ' + g.home;
    return gid;
  }

  function _teamLogoSrc(abv) {
    if (!abv) return '';
    if (window.brTeamLogoLocal) return window.brTeamLogoLocal(abv);
    var t = String(abv).toUpperCase();
    if (t === 'WSH') t = 'WAS';
    return '/static/images/team_logos/' + t + '.png';
  }
  // Live down-and-distance line. Returns '' unless the game is actually live,
  // so a final/pregame board never carries stale D&D (§2).
  function _nflBoardSitLine(g, norm) {
    if (!g || norm !== 'live') return '';
    var dd = _downDist({ down: g.down, distance: g.distance });
    var bits = [];
    if (dd) bits.push(dd);
    if (g.yard_line) bits.push(g.yard_line);
    return bits.join(' · ');
  }
  // Centered status line. FINAL / HALFTIME / kickoff / "Q3 · 7:42" -- driven by
  // the normalized game status, never a raw code guess.
  function _nflBoardClockLine(g, norm) {
    if (!g) return '';
    if (norm === 'final') return 'FINAL';
    if (norm === 'halftime') return 'HALFTIME';
    if (norm === 'pregame' || norm === 'delayed') {
      if (norm === 'delayed') return g.game_status || 'Delayed';
      var ep = parseFloat(g.game_time_epoch || 0);
      if (ep) return _fmtKickoff(ep);
      return g.game_status || 'Upcoming';
    }
    if (norm === 'live') {
      var q = _fmtQuarter(g.game_quarter || '');
      var clk = g.game_clock || '';
      var mid = [q, clk].filter(Boolean).join(' · ');
      return mid || 'LIVE';
    }
    return g.game_status || '';
  }
  function _renderNflBoard() {
    if (_filters.nfl === 'all') return '';
    var g = _nflGameInfo(_filters.nfl);
    if (!g || (!g.away && !g.home)) return '';
    var norm = _normGameStatus(g);
    var live = norm === 'live' || norm === 'halftime';
    var isFinal = norm === 'final';
    var away = g.away || '--', home = g.home || '--';
    var aPts = (g.away_pts === '' || g.away_pts == null) ? '–' : g.away_pts;
    var hPts = (g.home_pts === '' || g.home_pts == null) ? '–' : g.home_pts;
    var poss = String(g.possession || '').toUpperCase();
    // Possession is a *current* marker -- only meaningful for a live game (§2).
    var awayPoss = norm === 'live' && poss && poss === String(away).toUpperCase();
    var homePoss = norm === 'live' && poss && poss === String(home).toUpperCase();
    var sit = _nflBoardSitLine(g, norm);
    var clock = _nflBoardClockLine(g, norm);
    var stateCls = ' is-' + norm;
    var logo = function(abv) {
      var src = _teamLogoSrc(abv);
      if (!src) return '<span class="rz-nfl-abv-only">' + abv + '</span>';
      return '<img class="rz-nfl-logo" src="' + src + '" alt="" data-team="' + abv + '"'
        + ' onerror="var t=this.getAttribute(\'data-team\');if(t&&!this._espnFallback){this._espnFallback=1;this.src=(window.brTeamLogoEspn?window.brTeamLogoEspn(t):\'\');}else{this.style.display=\'none\';}">';
    };
    var side = function(abv, pts, hasBall, align) {
      var ball = hasBall
        ? '<span class="rz-nfl-ball" title="Possession" aria-label="Has possession"></span>'
        : '<span class="rz-nfl-ball-slot" aria-hidden="true"></span>';
      var meta = '<div class="rz-nfl-side-meta">'
        + '<span class="rz-nfl-abv">' + abv + '</span>'
        + '<span class="rz-nfl-pts">' + pts + '</span>'
        + '</div>';
      // Mirror the teams: the home logo is the outermost item on the right.
      var contents = align === 'home'
        ? ball + meta + logo(abv)
        : ball + logo(abv) + meta;
      return '<div class="rz-nfl-side rz-nfl-' + align + (hasBall ? ' has-ball' : '') + '">'
        + contents + '</div>';
    };
    var liveDot = live && norm === 'live' ? '<span class="rz-nfl-live-dot" aria-hidden="true"></span>' : '';
    return '<div class="rz-nfl-board' + stateCls + (live ? ' is-live' : '') + '" id="rz-nfl-board">'
      + side(away, aPts, awayPoss, 'away')
      + '<div class="rz-nfl-mid">'
      + '<div class="rz-nfl-clock">' + liveDot + clock + '</div>'
      + (sit ? '<div class="rz-nfl-sit">' + sit + '</div>' : '')
      + '</div>'
      + side(home, hPts, homePoss, 'home')
      + '</div>';
  }

  // Compact centered status for an NFL game pill.
  function _gamePillStatus(g, norm) {
    if (norm === 'final') return 'FINAL';
    if (norm === 'halftime') return 'HALF';
    if (norm === 'delayed') return 'DELAYED';
    if (norm === 'live') {
      var mid = [_fmtQuarter(g.game_quarter || ''), g.game_clock || ''].filter(Boolean).join(' ');
      return mid || 'LIVE';
    }
    var ep = parseFloat((g && g.game_time_epoch) || 0);
    if (ep) {
      var d = new Date(ep * 1000);
      if (!isNaN(d.getTime())) return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    }
    return (g && g.game_status) || 'PRE';
  }

  // ESPN-style horizontal NFL game selector (§4). One pill per NFL game plus an
  // "All" pill, all bound to the single canonical filter (`_filters.nfl`) so it
  // stays in lock-step with the Filter panel's Matchup control -- no second
  // game-selection state. Hidden entirely when there are no NFL games (§20).
  function _renderGameStrip() {
    var opts = _nflMatchupOptions();
    if (!opts.length) return '';
    var selectedAll = _filters.nfl === 'all';
    var pills = '<button type="button" class="rz-game-pill rz-game-pill-all'
      + (selectedAll ? ' is-selected' : '') + '" data-nfl-gid="all"'
      + ' aria-pressed="' + (selectedAll ? 'true' : 'false') + '"'
      + ' aria-label="Show all NFL games">'
      + '<span class="rz-gp-all">ALL</span></button>';
    var logo = function(abv) {
      var src = _teamLogoSrc(abv);
      if (!src) return '<span class="rz-gp-logo rz-gp-logo-txt">' + abv + '</span>';
      return '<img class="rz-gp-logo" src="' + src + '" alt="" data-team="' + abv + '"'
        + ' onerror="var t=this.getAttribute(\'data-team\');if(t&&!this._espnFallback){this._espnFallback=1;this.src=(window.brTeamLogoEspn?window.brTeamLogoEspn(t):\'\');}else{this.style.display=\'none\';}">';
    };
    pills += opts.map(function(o) {
      var g = _nflGameInfo(o.id) || o;
      var norm = _normGameStatus(g);
      var selected = _filters.nfl === o.id;
      var aPts = (g.away_pts === '' || g.away_pts == null) ? '' : g.away_pts;
      var hPts = (g.home_pts === '' || g.home_pts == null) ? '' : g.home_pts;
      var status = _gamePillStatus(g, norm);
      var aria = o.away + ' ' + (aPts || '') + ' at ' + o.home + ' ' + (hPts || '') + ', ' + status;
      var teamRow = function(abv, pts, side) {
        return '<span class="rz-gp-team ' + side + '">'
          + logo(abv)
          + '<span class="rz-gp-abv">' + abv + '</span>'
          + '<span class="rz-gp-score">' + (pts === '' ? '' : pts) + '</span>'
          + '</span>';
      };
      return '<button type="button" class="rz-game-pill is-' + norm
        + (selected ? ' is-selected' : '') + '" data-nfl-gid="' + o.id + '"'
        + ' aria-pressed="' + (selected ? 'true' : 'false') + '"'
        + ' aria-label="' + _esc(aria) + '">'
        + teamRow(o.away, aPts, 'away')
        + '<span class="rz-gp-status">' + status + '</span>'
        + teamRow(o.home, hPts, 'home')
        + '</button>';
    }).join('');
    return '<div class="rz-game-strip">'
      + '<div class="rz-game-strip-scroll" role="group" aria-label="NFL games">'
      + pills + '</div></div>';
  }

  function _renderFilterChips() {
    var activeCount = ['nfl','pos','stat'].filter(function(k) { return _filters[k] !== 'all'; }).length;
    var chips = '';
    if (_heroMid) chips += '<span class="rz-active-chip rz-hero-chip" data-clear-hero="1">&#9654; ' + _heroLabel() + ' ×</span>';
    if (_myTeamOnly) chips += '<span class="rz-active-chip" data-clear-myteam="1">My Team ×</span>';
    if (_bigPlaysOnly) chips += '<span class="rz-active-chip" data-clear-big="1">Big Plays ×</span>';
    if (_filters.nfl  !== 'all') chips += '<span class="rz-active-chip" data-clear="nfl">'  + _nflFilterLabel(_filters.nfl)  + ' ×</span>';
    if (_filters.pos  !== 'all') chips += '<span class="rz-active-chip" data-clear="pos">'  + _filters.pos  + ' ×</span>';
    if (_filters.stat !== 'all') {
      var sl = _STAT_LIST.find(function(x) { return x[0] === _filters.stat; });
      chips += '<span class="rz-active-chip" data-clear="stat">' + (sl ? sl[1] : _filters.stat) + ' ×</span>';
    }
    var panel = '';
    if (_filterOpen) {
      var fpRow = function(label, key, opts) {
        var btns = opts.map(function(o) {
          var val = Array.isArray(o) ? o[0] : o, lbl = Array.isArray(o) ? o[1] : o;
          return '<button class="otc-day-filter rz-fp-opt' + (_filters[key] === val ? ' active' : '') + '" data-fk="' + key + '" data-fv="' + val + '">' + lbl + '</button>';
        }).join('');
        return '<div class="rz-fp-row"><span class="rz-fp-label">' + label + '</span><div class="otc-day-filters rz-fp-opts">' + btns + '</div></div>';
      }
      var nOpts = [['all','All']].concat(_nflMatchupOptions().map(function(g) {
        return [g.id, g.label];
      }));
      var pOpts = [['all','All']].concat(_POS_LIST.map(function(p) { return [p, p]; }));
      var sOpts = [['all','All']].concat(_STAT_LIST);
      panel = '<div class="rz-filter-panel">'
        + fpRow('Matchup', 'nfl', nOpts)
        + fpRow('Pos',  'pos',  pOpts)
        + fpRow('Type', 'stat', sOpts)
        + '</div>';
    }
    var myTeamBtn = _myRids.size
      ? '<button class="rz-myteam-btn' + (_myTeamOnly ? ' active' : '') + '" id="rz-myteam-btn">My Team</button>'
      : '';
    var bigBtn = '<button class="rz-bigplays-btn' + (_bigPlaysOnly ? ' active' : '') + '" id="rz-bigplays-btn">Big Plays</button>';
    var histCount = _notifHistory.length;
    var histBtn = histCount > 0
      ? '<button class="rz-hist-btn" id="rz-hist-btn">Alerts <span class="rz-hist-count">' + histCount + '</span></button>'
      : '';
    return '<div class="rz-chip-bar">'
      + myTeamBtn
      + bigBtn
      + '<button class="rz-filter-toggle' + (_filterOpen ? ' open' : '') + '" id="rz-filter-btn">'
      + (activeCount ? '⊞ Filter (' + activeCount + ')' : '⊞ Filter') + '</button>'
      + histBtn
      + '<span class="br-chip-pop" style="display:contents">' + chips + '</span>'
      + '</div>'
      + panel;
  }

  // ── Render helpers ────────────────────────────────────────────────────────────
  function _posHtml(pos) {
    var safe = (pos || '').replace(/[^A-Z_]/g, '');
    return '<span class="rz-pos-badge rz-pos-' + safe + '">' + (pos || '?') + '</span>';
  }
  function _injuryDot(pid) {
    var inj = ((_state.player_info || {})[pid] || {}).injury_status || '';
    if (!inj) return '';
    var cls = inj === 'O' || inj === 'IR' ? 'out' : inj === 'D' ? 'doubtful' : 'questionable';
    return '<span class="rz-inj-dot ' + cls + '" title="' + inj + '"></span>';
  }

  function _playerRowHtml(pid, pts, isBench) {
    var gs = _gameStatus(pid), line = _gameLine(pid), isLive = gs.type === 'live';
    var dot = isLive ? '<span class="rz-live-dot-sm"></span>' : '';
    var meta = '<span class="rz-meta-game ' + gs.type + '">' + (line || gs.label || '') + '</span>';
    return (
      '<div class="rz-player-row' + (isBench ? ' bench-row' : '') + '" data-pid="' + pid + '">'
      + _posHtml(_pos(pid))
      + '<div class="rz-player-info"><div class="rz-player-name">' + _name(pid) + _injuryDot(pid) + '</div>'
      + '<div class="rz-player-meta">' + dot + meta + '</div></div>'
      + '<div class="rz-player-pts' + (isLive ? ' live-pts' : '') + '" data-pid="' + pid + '">'
      + (pts != null ? _fmt(pts) : '0') + '</div></div>'
    );
  }
  function _rosterCard(matchup) {
    if (!matchup) return '<div class="rz-feed-empty">No lineup data.</div>';
    var starters = matchup.starters || [];
    var bench = (matchup.players || []).filter(function(pid) { return pid !== '0' && !starters.includes(pid); });
    var rows = starters.map(function(pid) {
      if (pid === '0') return '<div class="rz-player-row"><span class="rz-pos-badge rz-pos-" style="opacity:.25"></span><div class="rz-player-info"><div class="rz-player-name" style="color:var(--rz-muted)">Empty slot</div></div><div class="rz-player-pts">0</div></div>';
      return _playerRowHtml(pid, _playerPts(pid, matchup), false);
    }).join('');
    var benchRows = bench.slice(0, 6).map(function(pid) { return _playerRowHtml(pid, _playerPts(pid, matchup), true); }).join('');
    return '<div class="rz-roster-card">' + rows
      + (bench.length ? '<div class="rz-section-label">Bench</div>' + benchRows : '') + '</div>';
  }

  function _myMatchups() {
    return (_state.matchups || []).filter(function(m) { return _isMyRid(m.roster_id); });
  }
  function _oppOf(myM) {
    var mid = String(myM.matchup_id);
    return (_state.matchups || []).find(function(m) { return String(m.matchup_id) === mid && !_isMyRid(m.roster_id); });
  }

  // Previously auto-focused the viewer's own matchup in This League. That
  // filtered Plays down to one game by default; we now land on the full feed
  // and only filter when the viewer taps a hero card.
  function _defaultHeroMid() {
    return null;
  }

  // No-op kept so call sites stay stable. Hero focus is opt-in via tap.
  function _applyDefaultHero() {
    return;
  }

  // Starters whose NFL game is not yet complete (live OR upcoming). Byes and
  // empty slots don't count. Uses the authoritative game-level state, not the
  // player's raw game_code, so a stale "final" record can't drop the count.
  function _playersLeft(matchup) {
    if (!matchup) return 0;
    var c = _sideCounts(matchup);
    return c.live + c.upcoming;
  }
  // Starters whose NFL game has not started yet ("to play"). Live players are
  // deliberately excluded (§14). Byes/empty never count.
  function _playersToPlay(matchup) {
    if (!matchup) return 0;
    return _sideCounts(matchup).upcoming;
  }

  function _renderHero() {
    var mine = _myMatchups();
    if (!mine.length) {
      return '<div class="rz-no-matchup">Log in to a league to track your matchup live.<br><br>'
        + '<a href="?demo=1" style="color:var(--rz-red);font-weight:700;font-size:13px;">View Demo</a></div>';
    }
    var myMatchup = mine[0], oppMatchup = _oppOf(myMatchup);
    var myPts = parseFloat(myMatchup.points || 0), oppPts = parseFloat(oppMatchup ? oppMatchup.points || 0 : 0);
    var myProj = myMatchup.projected_pts != null ? parseFloat(myMatchup.projected_pts) : null;
    var oppProj = oppMatchup && oppMatchup.projected_pts != null ? parseFloat(oppMatchup.projected_pts) : null;
    var total = myPts + oppPts, winning = myPts >= oppPts, diff = Math.abs(myPts - oppPts).toFixed(1);
    var myName = _ownerName(myMatchup.roster_id) || 'My Team';
    var oppName = oppMatchup ? (_ownerName(oppMatchup.roster_id) || 'Opponent') : 'Opponent';
    var fillPct = total > 0 ? Math.min(Math.max((myPts / total) * 100, 5), 95) : 50;
    var liveCnt = 0;
    [myMatchup, oppMatchup].forEach(function(m) {
      if (!m) return;
      (m.starters || []).forEach(function(pid) { if (_gameStatus(pid).type === 'live') liveCnt++; });
    });
    var myLeft = _playersLeft(myMatchup), oppLeft = _playersLeft(oppMatchup);
    var isClose = liveCnt > 0 && parseFloat(diff) < 5;
    var meta = [];
    if (winning && diff > 0) meta.push('<span class="accent win">+' + diff + ' lead</span>');
    else if (!winning && diff > 0) meta.push('<span class="accent lose">Trailing ' + diff + '</span>');
    else meta.push('<span class="accent">Tied</span>');
    if (liveCnt > 0) meta.push(liveCnt + ' live');
    return (
      '<div class="rz-hero' + (isClose ? ' rz-close-game' : '') + '">'
      + '<div class="rz-hero-label">Your Matchup  •  Week ' + (_state.week || '') + (isClose ? '  •  <span class="rz-close-label">Close game</span>' : '') + '</div>'
      + '<div class="rz-hero-scores">'
      + '<div class="rz-hero-side left' + (!winning ? ' losing' : '') + '">'
      +   '<div class="rz-hero-tname">' + myName + '</div>'
      +   '<div class="rz-hero-pts">' + _fmt(myPts) + '</div>'
      +   (myProj != null ? '<div class="rz-proj-score">Proj ' + _fmt(myProj) + '</div>' : '')
      +   (myLeft > 0 ? '<div class="rz-players-left">' + myLeft + ' left</div>' : '')
      + '</div>'
      + '<div class="rz-hero-vs">vs</div>'
      + '<div class="rz-hero-side right">'
      +   '<div class="rz-hero-tname" style="text-align:right">' + oppName + '</div>'
      +   '<div class="rz-hero-pts" style="text-align:right">' + _fmt(oppPts) + '</div>'
      +   (oppProj != null ? '<div class="rz-proj-score" style="text-align:right">Proj ' + _fmt(oppProj) + '</div>' : '')
      +   (oppLeft > 0 ? '<div class="rz-players-left" style="text-align:right">' + oppLeft + ' left</div>' : '')
      + '</div>'
      + '</div>'
      + '<div class="rz-adv-wrap"><div class="rz-adv-bar"><div class="rz-adv-fill ' + (winning ? 'winning' : 'losing') + '" style="width:' + fillPct + '%"></div></div></div>'
      + '<div class="rz-hero-meta">' + meta.join('  •  ') + '</div>'
      + '</div>'
    );
  }

  function _renderLeagueOthers() {
    // Every OTHER matchup in this league (viewer's own is shown as the hero).
    var groups = {};
    (_state.matchups || []).forEach(function(m) {
      var mid = String(m.matchup_id);
      (groups[mid] = groups[mid] || []).push(m);
    });
    var mine = _myMatchups();
    var myMid = mine[0] ? String(mine[0].matchup_id) : null;
    var rows = '';
    Object.keys(groups).sort().forEach(function(mid) {
      if (mid === myMid) return;
      var pair = groups[mid], a = pair[0], b = pair[1];
      if (!b) return;
      var ptsA = parseFloat(a.points || 0), ptsB = parseFloat(b.points || 0), aLead = ptsA >= ptsB;
      var ms = _matchupState(a, b);
      var anyLive = ms.state === 'live';
      var isClose = anyLive && Math.abs(ptsA - ptsB) < 5;
      var lbMid = ms.state === 'live'
        ? '<span class="rz-lb-live">LIVE</span>'
        : ms.state === 'toplay'
          ? '<span class="rz-lb-toplay">' + ms.toPlayA + '<span class="rz-mc-to-play-divider">|</span>' + ms.toPlayB + '</span>'
          : ms.state === 'final'
            ? '<span class="rz-lb-final">FINAL</span>'
            : '<span class="rz-lb-final">–</span>';
      rows += (
        '<div class="rz-lb-row' + (isClose ? ' close' : '') + '">'
        + '<div class="rz-lb-team' + (aLead ? ' lead' : '') + '">'
        +   '<span class="rz-lb-name">' + (_ownerName(a.roster_id) || 'Team') + '</span>'
        +   '<span class="rz-lb-score">' + _fmt(ptsA) + '</span>'
        + '</div>'
        + '<div class="rz-lb-mid">' + lbMid + '</div>'
        + '<div class="rz-lb-team right' + (!aLead ? ' lead' : '') + '">'
        +   '<span class="rz-lb-score">' + _fmt(ptsB) + '</span>'
        +   '<span class="rz-lb-name">' + (_ownerName(b.roster_id) || 'Team') + '</span>'
        + '</div>'
        + '</div>'
      );
    });
    if (!rows) return '';
    return '<div class="rz-league-board"><div class="rz-lb-title">Around the League</div>' + rows + '</div>';
  }

  // Wrap the matchup cards in a horizontal scroller with prev/next arrows so
  // the strip is navigable on desktop (mouse, no h-scroll gesture) as well as
  // touch. Arrows/fades are toggled by _updateHeroArrows based on overflow.
  function _heroCardsWrap(deltaHtml, cardsHtml) {
    return '<div class="rz-hero-cards">' + deltaHtml
      + '<div class="rz-hero-scroller">'
      +   '<button type="button" class="rz-hero-arrow left rz-arrow-off" data-hero-arrow="-1" aria-label="Scroll to earlier matchups">&#8249;</button>'
      +   '<div class="rz-hero-cards-row">' + cardsHtml + '</div>'
      +   '<button type="button" class="rz-hero-arrow right rz-arrow-off" data-hero-arrow="1" aria-label="Scroll to more matchups">&#8250;</button>'
      + '</div>'
      + '</div>';
  }

  // Placeholder matchup cards shown while a scope switch (e.g. "My Leagues")
  // fetches its data, which can take a while when it spans many leagues.
  function _renderSkeletonHero() {
    var one =
      '<div class="rz-mc-hero rz-mc-skel" aria-hidden="true">'
      + '<div class="rz-skel-line rz-skel-league"></div>'
      + '<div class="rz-mch-matchup">'
      +   '<div class="rz-mch-side"><div class="rz-skel-line rz-skel-name"></div><div class="rz-skel-num"></div></div>'
      +   '<div class="rz-mch-vs"><div class="rz-skel-badge"></div></div>'
      +   '<div class="rz-mch-side right"><div class="rz-skel-line rz-skel-name"></div><div class="rz-skel-num"></div></div>'
      + '</div>'
      + '</div>';
    var cards = '';
    for (var i = 0; i < 4; i++) cards += one;
    return '<div class="rz-hero-cards"><div class="rz-hero-scroller"><div class="rz-hero-cards-row">'
      + cards + '</div></div></div>';
  }

  // One placeholder My Leagues card, labeled with the league name, shown while
  // that league's data is still streaming in.
  function _mlPlaceholderCard(name) {
    return '<div class="rz-mc-hero rz-mc-skel rz-mc-loading" aria-hidden="true">'
      + '<div class="rz-mch-league">' + (name || 'League') + '</div>'
      + '<div class="rz-mch-matchup">'
      +   '<div class="rz-mch-side"><div class="rz-skel-line rz-skel-name"></div><div class="rz-skel-num"></div></div>'
      +   '<div class="rz-mch-vs"><div class="rz-skel-badge"></div></div>'
      +   '<div class="rz-mch-side right"><div class="rz-skel-line rz-skel-name"></div><div class="rz-skel-num"></div></div>'
      + '</div>'
      + '</div>';
  }

  function _mlFailedCard(name) {
    return '<div class="rz-mc-hero rz-mc-failed" title="Could not load this league">'
      + '<div class="rz-mch-league">' + (name || 'League') + '</div>'
      + '<div class="rz-mch-matchup">'
      +   '<div class="rz-mch-side"><div class="rz-mch-owner">Unavailable</div><div class="rz-mch-score">--</div></div>'
      +   '<div class="rz-mch-vs"><span class="rz-mch-pre">ERR</span></div>'
      +   '<div class="rz-mch-side right"><div class="rz-mch-owner">--</div><div class="rz-mch-score">--</div></div>'
      + '</div>'
      + '</div>';
  }

  function _renderHeroCards() {
    // Score delta badge ("+N this update"): reflects the change from the most
    // recent poll. _detectChanges recomputes (and resets) _scoreDelta every
    // poll, so the badge updates each cycle and clears when nothing changed --
    // we only read it here (don't consume) so it survives per-second partial
    // re-renders between polls.
    var _dMe = _scoreDelta.me, _dOpp = _scoreDelta.opp;
    var _deltaHtml = '';
    if (_dMe > 0.05 || _dOpp > 0.05) {
      var _gain = _dMe - _dOpp;
      var _gainStr = (_gain >= 0 ? '+' : '') + _gain.toFixed(1);
      _deltaHtml = '<div class="rz-delta-strip"><span class="rz-delta ' + (_gain >= 0 ? 'pos' : 'neg') + '">' + _gainStr + ' this update</span></div>';
    }
    if (_scope === 'user') {
      var mine = _myMatchups();
      // While streaming, show a named placeholder for every league whose card
      // hasn't arrived yet (loaded-but-teamless leagues are in _mlLoaded and get
      // no placeholder). Rendered after the real cards.
      var pending = '';
      if (_mlLoaded || _mlFailed) {
        var showingCache = _streaming && _scopeCache.user && (_scopeCache.user.matchups || []).length
          && _state === _scopeCache.user;
        for (var _pi = 0; _pi < _mlNames.length; _pi++) {
          if (_mlFailed && _mlFailed.has(_pi)) pending += _mlFailedCard(_mlNames[_pi]);
          else if (!showingCache && _mlLoaded && !_mlLoaded.has(_pi)) pending += _mlPlaceholderCard(_mlNames[_pi]);
        }
      }
      if (!mine.length && !pending) {
        return '<div class="rz-no-matchup">No leagues found for your account.<br><a href="?demo=1" class="rz-demo-link">View Demo</a></div>';
      }
      var cards = mine.map(function(m) {
        var opp = _oppOf(m);
        var myPts = parseFloat(m.points || 0), oppPts = parseFloat(opp ? opp.points || 0 : 0);
        var win = myPts >= oppPts;
        var ms = _matchupState(m, opp);
        var rid = String(m.roster_id);
        var selected = _heroMid === rid;
        var oppName = opp ? (_ownerName(opp.roster_id) || 'Opp') : 'Opp';
        return '<div class="rz-mc-hero' + (selected ? ' selected' : '') + (ms.state === 'live' ? ' is-live' : '') + '" data-heromid="' + rid + '"'
          + ' role="button" tabindex="0" aria-pressed="' + (selected ? 'true' : 'false') + '"'
          + ' aria-label="' + _esc(_matchupAria('You', myPts, oppName, oppPts, ms)) + '">'
          + '<div class="rz-mch-league">' + (m.league_name || 'League') + '</div>'
          + '<div class="rz-mch-matchup">'
          +   '<div class="rz-mch-side rz-mc-team-left">'
          +     '<div class="rz-mch-owner viewer">Me</div>'
          +     '<div class="rz-mch-score' + (win ? ' lead' : '') + '" data-score-rid="' + String(m.roster_id) + '">' + _fmt(myPts) + '</div>'
          +   '</div>'
          +   '<div class="rz-mch-vs rz-mc-center">' + _matchupCenterHtml(m, opp) + '</div>'
          +   '<div class="rz-mch-side right rz-mc-team-right">'
          +     '<div class="rz-mch-owner">' + oppName + '</div>'
          +     '<div class="rz-mch-score' + (!win ? ' lead' : '') + '" data-score-rid="' + (opp ? String(opp.roster_id) : '') + '">' + _fmt(oppPts) + '</div>'
          +   '</div>'
          + '</div>'
          + '</div>';
      }).join('');
      return _heroCardsWrap(_deltaHtml, cards + pending);
    }

    // This League mode: one card per matchup, viewer's first
    var groups = {};
    (_state.matchups || []).forEach(function(m) {
      var mid = String(m.matchup_id);
      (groups[mid] = groups[mid] || []).push(m);
    });
    if (!Object.keys(groups).length) {
      return '<div class="rz-no-matchup">No matchup data yet.<br><a href="?demo=1" class="rz-demo-link">View Demo</a></div>';
    }
    var myMid = null;
    var mine2 = _myMatchups();
    if (mine2[0]) myMid = String(mine2[0].matchup_id);
    var mids = Object.keys(groups).sort(function(a, b) {
      if (a === myMid) return -1;
      if (b === myMid) return 1;
      return parseInt(a) - parseInt(b);
    });
    var cards2 = mids.map(function(mid) {
      var pair = groups[mid], a = pair[0], b = pair[1];
      if (!b) return '';
      var ptsA = parseFloat(a.points || 0), ptsB = parseFloat(b.points || 0), aLead = ptsA >= ptsB;
      var ms = _matchupState(a, b);
      var selected = _heroMid === mid;
      var isViewer = mid === myMid;
      var nameA = _ownerName(a.roster_id) || 'Team';
      var nameB = _ownerName(b.roster_id) || 'Team';
      var vA = _isMyRid(a.roster_id), vB = _isMyRid(b.roster_id);
      return '<div class="rz-mc-hero' + (selected ? ' selected' : '') + (ms.state === 'live' ? ' is-live' : '') + (isViewer ? ' viewer-matchup' : '') + '" data-heromid="' + mid + '"'
        + ' role="button" tabindex="0" aria-pressed="' + (selected ? 'true' : 'false') + '"'
        + ' aria-label="' + _esc(_matchupAria(nameA, ptsA, nameB, ptsB, ms)) + '">'
        + '<div class="rz-mch-matchup">'
        +   '<div class="rz-mch-side rz-mc-team-left">'
        +     '<div class="rz-mch-owner' + (vA ? ' viewer' : '') + '">' + nameA + '</div>'
        +     '<div class="rz-mch-score' + (aLead ? ' lead' : '') + '" data-score-rid="' + String(a.roster_id) + '">' + _fmt(ptsA) + '</div>'
        +   '</div>'
        +   '<div class="rz-mch-vs rz-mc-center">' + _matchupCenterHtml(a, b) + '</div>'
        +   '<div class="rz-mch-side right rz-mc-team-right">'
        +     '<div class="rz-mch-owner' + (vB ? ' viewer' : '') + '">' + nameB + '</div>'
        +     '<div class="rz-mch-score' + (!aLead ? ' lead' : '') + '" data-score-rid="' + String(b.roster_id) + '">' + _fmt(ptsB) + '</div>'
        +   '</div>'
        + '</div>'
        + '</div>';
    }).filter(Boolean).join('');
    return _heroCardsWrap(_deltaHtml, cards2);
  }

  function _renderLeaguesSummary() {
    var mine = _myMatchups();
    if (!mine.length) return '<div class="rz-no-matchup">No leagues found for your account.</div>';
    var winning = 0;
    var cards = mine.map(function(m) {
      var opp = _oppOf(m);
      var myPts = parseFloat(m.points || 0), oppPts = parseFloat(opp ? opp.points || 0 : 0);
      var win = myPts >= oppPts;
      var diff = Math.abs(myPts - oppPts);
      var isClose = diff < 5;
      if (win) winning++;
      var oppName = opp ? (_ownerName(opp.roster_id) || 'Opponent') : 'Opponent';
      var myLeft = _playersLeft(m), oppLeft = _playersLeft(opp);
      return (
        '<div class="rz-lg-row' + (isClose ? ' rz-close-row' : '') + '">'
        + '<div class="rz-lg-name">' + (m.league_name || 'League') + '<span>vs ' + oppName + (myLeft > 0 ? ' · ' + myLeft + ' left' : '') + '</span></div>'
        + '<div class="rz-lg-score ' + (win ? 'win' : 'lose') + '">' + _fmt(myPts) + '</div>'
        + '<div class="rz-lg-sep">-</div>'
        + '<div class="rz-lg-score opp">' + _fmt(oppPts) + '</div>'
        + (isClose ? '<span class="rz-close-pip" title="Close game">!</span>' : '')
        + '<span class="rz-lg-pill ' + (win ? 'win' : 'lose') + '">' + (win ? 'W' : 'L') + '</span>'
        + '</div>'
      );
    }).join('');
    return (
      '<div class="rz-hero">'
      + '<div class="rz-hero-label">My Leagues  •  Week ' + (_state.week || '') + '  •  ' + winning + '-' + (mine.length - winning) + '</div>'
      + '<div class="rz-lg-list">' + cards + '</div>'
      + '</div>'
    );
  }

  function _mtRowHtml(pid, matchup) {
    if (pid === '0') {
      return '<div class="rz-mt-row is-empty"><span class="rz-pos-badge" style="opacity:.25">--</span>'
        + '<span class="rz-mt-name" style="color:var(--rz-muted)">Empty</span>'
        + '<span class="rz-mt-pts">0.0</span></div>';
    }
    var gs = _gameStatus(pid);
    var pts = _playerPts(pid, matchup);
    var live = gs.type === 'live';
    var status = live ? '<span class="rz-mt-live">LIVE</span>'
               : gs.type === 'final' ? '<span class="rz-mt-final">FINAL</span>'
               : '<span class="rz-mt-pre">' + (gs.label || '') + '</span>';
    // Last name only keeps the compact list scannable across many leagues.
    var full = _name(pid) || pid;
    var short = full.indexOf(' ') >= 0 ? full.split(' ').slice(-1)[0] : full;
    return (
      '<div class="rz-mt-row' + (live ? ' is-live' : '') + '" data-pid="' + pid + '">'
      + _posHtml(_pos(pid))
      + '<span class="rz-mt-name" title="' + full + '">' + short + _injuryDot(pid) + '</span>'
      + status
      + '<span class="rz-mt-pts' + (live ? ' live-pts' : '') + '">' + _fmt(pts) + '</span>'
      + '</div>'
    );
  }

  function _renderMyTeams() {
    var mine = _myMatchups();
    if (!mine.length) return '<div class="rz-feed-empty">No teams found.</div>';
    return '<div class="rz-mt-list">' + mine.map(function(m) {
      var opp = _oppOf(m);
      var myPts = parseFloat(m.points || 0);
      var oppPts = opp ? parseFloat(opp.points || 0) : 0;
      var starters = m.starters || [];
      var anyLive = starters.some(function(pid) { return pid !== '0' && _gameStatus(pid).type === 'live'; });
      var openAttr = anyLive ? ' open' : '';
      var score = _fmt(myPts) + ' – ' + _fmt(oppPts);
      var liveBadge = anyLive ? '<span class="rz-mt-sum-live">LIVE</span>' : '';
      var rows = starters.map(function(pid) { return _mtRowHtml(pid, m); }).join('');
      var bench = (m.players || []).filter(function(pid) { return pid !== '0' && starters.indexOf(pid) < 0; });
      var benchHtml = '';
      if (bench.length) {
        benchHtml = '<details class="rz-mt-bench"><summary class="rz-mt-bench-sum">Bench (' + bench.length + ')</summary>'
          + bench.slice(0, 8).map(function(pid) { return _mtRowHtml(pid, m); }).join('')
          + '</details>';
      }
      return (
        '<details class="rz-mt-league"' + openAttr + '>'
        + '<summary class="rz-mt-sum">'
        + '<span class="rz-mt-lg">' + (m.league_name || 'League') + '</span>'
        + '<span class="rz-mt-score">' + score + '</span>'
        + liveBadge
        + '</summary>'
        + '<div class="rz-mt-starters">' + rows + '</div>'
        + benchHtml
        + '</details>'
      );
    }).join('') + '</div>';
  }

  function _renderScoreboard() {
    var groups = {};
    (_state.matchups || []).forEach(function(m) {
      var mid = String(m.matchup_id);
      (groups[mid] = groups[mid] || []).push(m);
    });
    var html = '';
    Object.keys(groups).sort().forEach(function(mid) {
      var pair = groups[mid], a = pair[0], b = pair[1];
      if (!b) return;
      var ptsA = parseFloat(a.points || 0), ptsB = parseFloat(b.points || 0), aLead = ptsA >= ptsB;
      var projA = a.projected_pts != null ? parseFloat(a.projected_pts) : null;
      var projB = b.projected_pts != null ? parseFloat(b.projected_pts) : null;
      var ms = _matchupState(a, b);
      var isClose = ms.state === 'live' && Math.abs(ptsA - ptsB) < 5;
      var cls = ms.state === 'live' ? 'live' : ms.state === 'final' ? 'final' : ms.state === 'toplay' ? 'toplay' : 'pre';
      var statusLabel = ms.state === 'live' ? 'LIVE'
        : ms.state === 'toplay' ? 'TO PLAY ' + ms.toPlayA + ' | ' + ms.toPlayB
        : ms.state === 'final' ? 'FINAL' : '';
      var vA = _isMyRid(a.roster_id), vB = _isMyRid(b.roster_id);
      var lgHdr = a.league_name ? '<span class="rz-mc-league">' + a.league_name + '</span>' : '';
      var leftA = _playersLeft(a), leftB = _playersLeft(b);
      html += '<div class="rz-matchup-card' + (isClose ? ' rz-close-card' : '') + '">'
        + '<div class="rz-mc-header">' + lgHdr + (statusLabel ? '<span class="rz-mc-status ' + cls + '">' + statusLabel + '</span>' : '') + (isClose ? '<span class="rz-close-label">Close</span>' : '') + '</div>'
        + '<div class="rz-mc-row">'
        + '<div class="rz-mc-name' + (vA ? ' viewer' : '') + '">' + (_ownerName(a.roster_id) || 'Team') + (leftA > 0 ? '<span class="rz-mc-left"> ' + leftA + ' left</span>' : '') + '</div>'
        + '<div class="rz-mc-pts-col">'
        +   '<div class="rz-mc-score' + (aLead ? ' leader' : '') + '">' + _fmt(ptsA) + '</div>'
        +   (projA != null ? '<div class="rz-mc-proj">Proj ' + _fmt(projA) + '</div>' : '')
        + '</div>'
        + '<div class="rz-mc-sep">-</div>'
        + '<div class="rz-mc-pts-col">'
        +   '<div class="rz-mc-score' + (!aLead ? ' leader' : '') + '">' + _fmt(ptsB) + '</div>'
        +   (projB != null ? '<div class="rz-mc-proj">Proj ' + _fmt(projB) + '</div>' : '')
        + '</div>'
        + '<div class="rz-mc-name' + (vB ? ' viewer' : '') + '" style="text-align:right">' + (_ownerName(b.roster_id) || 'Team') + (leftB > 0 ? '<span class="rz-mc-left"> ' + leftB + ' left</span>' : '') + '</div>'
        + '</div></div>';
    });
    return html || '<div class="rz-feed-empty">No matchup data yet.</div>';
  }

  function _renderPosLeaders(pidMap, myStarters) {
    var positions = ['QB', 'RB', 'WR', 'TE'];
    var leaders = {};
    Object.keys(pidMap).forEach(function(pid) {
      var p = (_state.player_info || {})[pid] || {};
      var pos = p.pos || '';
      if (positions.indexOf(pos) === -1) return;
      var pts = pidMap[pid].pts;
      if (pts <= 0) return;
      if (!leaders[pos] || pts > leaders[pos].pts) leaders[pos] = { pid: pid, pts: pts, roster_id: pidMap[pid].roster_id };
    });
    var tiles = positions.map(function(pos) {
      var ldr = leaders[pos];
      if (!ldr) return '<div class="rz-pl-tile rz-pl-empty"><div class="rz-pl-pos">' + pos + '</div><div class="rz-pl-name">-</div><div class="rz-pl-pts">-</div></div>';
      var p = (_state.player_info || {})[ldr.pid] || {};
      var mine = myStarters.has(ldr.pid) || _isMyRid(ldr.roster_id);
      return (
        '<div class="rz-pl-tile' + (mine ? ' mine' : '') + '" data-pid="' + ldr.pid + '">'
        + '<div class="rz-pl-pos">' + pos + '</div>'
        + '<div class="rz-pl-name">' + (p.name || ldr.pid).split(' ').pop() + '</div>'
        + '<div class="rz-pl-pts">' + _fmt(ldr.pts) + '</div>'
        + '</div>'
      );
    }).join('');
    return '<div class="rz-pos-leaders">' + tiles + '</div>';
  }

  function _renderTopPerformers() {
    var myStarters = new Set();
    _myMatchups().forEach(function(m) { (m.starters || []).forEach(function(p) { myStarters.add(p); }); });
    var pidMap = {};
    (_state.matchups || []).forEach(function(m) {
      var seen = {};
      var pp = m.players_points || {};
      Object.keys(pp).forEach(function(pid) { seen[pid] = true; });
      (m.starters || []).forEach(function(pid) { if (pid && pid !== '0') seen[pid] = true; });
      Object.keys(seen).forEach(function(pid) {
        var pts = parseFloat(_playerPts(pid, m) || 0);
        if (!pidMap[pid] || pts > pidMap[pid].pts) pidMap[pid] = { pts: pts, roster_id: m.roster_id };
      });
    });
    // Filtered map (respects active team/pos/owner filters) -- used for both
    // the leaderboard list and the position leaders strip so they stay consistent.
    var filteredMap = {};
    Object.keys(pidMap).forEach(function(pid) {
      if (pidMap[pid].pts > 0 && _topMatches(pid, pidMap[pid].roster_id)) filteredMap[pid] = pidMap[pid];
    });
    var sorted = Object.keys(filteredMap)
      .sort(function(a, b) { return filteredMap[b].pts - filteredMap[a].pts; })
      .slice(0, 25);
    if (!sorted.length) return _renderPosLeaders(filteredMap, myStarters) + '<div class="rz-feed-empty">No players match these filters yet.</div>';
    return _renderPosLeaders(filteredMap, myStarters) + sorted.map(function(pid, i) {
      var d = pidMap[pid], rank = i + 1;
      var rc = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
      var mine = myStarters.has(pid) || _isMyRid(d.roster_id);
      var p = (_state.player_info || {})[pid] || {};
      var gs = _gameStatus(pid);
      var live = gs.type === 'live' ? ' • <span style="color:#fca5a5">LIVE</span>' : '';
      var ctx = _scope === 'user' ? _leagueOfRid(d.roster_id) : _ownerName(d.roster_id);
      return (
        '<div class="rz-top-row" data-pid="' + pid + '">'
        + '<div class="rz-top-rank ' + rc + '">#' + rank + '</div>'
        + _posHtml(p.pos || '?')
        + '<div class="rz-top-info"><strong>' + (p.name || pid) + '</strong><span>' + (p.team || '') + live + '</span></div>'
        + '<div class="rz-top-owner' + (mine ? ' mine' : '') + '">' + ctx + '</div>'
        + '<div class="rz-top-pts">' + _fmt(d.pts) + '</div>'
        + '</div>'
      );
    }).join('');
  }

  var _FEED_ICON = { td: '🏈', gain: '🟢', neg: '⚠️', target: '🎯', milestone: '⭐' };

  function _eid(ev) { return ev.playId || (ev.pid + ':' + (ev.ts || ev.desc)); }

  function _eventHtml(ev, animate) {
    var tagLabel = ev.mine ? (_scope === 'user' && ev.league ? ev.league : 'MY TEAM')
                 : ev.opp  ? (_scope === 'user' && ev.league ? ('OPP · ' + ev.league) : 'OPP')
                 : (_scope === 'user' && ev.league ? ev.league : '');
    var tagCls = ev.mine ? 'mine' : 'opp';
    var tag = tagLabel ? '<span class="rz-event-tag ' + tagCls + '">' + tagLabel + '</span>' : '';
    var totalStr = (ev.totalPts != null && !isNaN(ev.totalPts)) ? _fmtFantasyTotal(ev.totalPts) : '';
    // Headline the points THIS play earned (like Sleeper's game log), with the
    // player's running total underneath. A zero-value play (sack, incompletion,
    // stat we don't model) shows a muted "0" -- never a green "+0" that reads
    // like a score, and never the running total masquerading as the play's pts.
    var d = _n(ev.pts);
    var deltaPrimary = (d > 0.0001 ? '+' : (d < -0.0001 ? '' : '')) + _fmtFantasyDelta(d);
    var deltaSecondary = totalStr;
    var deltaCls = d > 0.0001 ? 'pos' : (d < -0.0001 ? 'neg' : 'zero');
    var posKey = (ev.pos || 'x').toLowerCase().replace(/[^a-z]/g, '');
    var initials = (ev.name || '?').trim().split(/\s+/).map(function(w) { return w[0] || ''; }).join('').slice(0, 2).toUpperCase();
    // ── Sleeper-style situation strip ──────────────────────────────────────
    // Left: down & distance @ field spot, with a red-zone flag. Right: quarter
    // + clock over a compact score. (No reactions / replies.)
    var clockStr = [_fmtQuarter(ev.gameQuarter), ev.gameClock].filter(Boolean).join(' ');
    var dd = _downDist(ev);
    var situation = [dd, ev.yardLine].filter(Boolean).join(' @ ');
    var rzBadge = _isRedZone(ev.yardLine, ev.nflTeam)
      ? '<span class="rz-event-rz">RZ</span>' : '';
    var pi = (_state.player_info || {})[ev.pid] || {};
    var scoreStr = '';
    if (pi.away && pi.home && !(pi.away_pts === '' && pi.home_pts === '')) {
      scoreStr = pi.away + ' ' + (pi.away_pts || '0') + '–' + (pi.home_pts || '0') + ' ' + pi.home;
    }
    var situationHtml = (situation || rzBadge)
      ? '<div class="rz-event-meta">'
        + '<span class="rz-event-situation">' + situation + rzBadge + '</span>'
        + '</div>'
      : '';
    // Yardage this player gained on the play (Sleeper's "+12 YD" chip).
    var sl = ev.statLine || {};
    var ydVal = null;
    if (_n(sl.carries)) ydVal = _n(sl.rush_yds);
    else if (_n(sl.rec)) ydVal = _n(sl.rec_yds);
    else if ('pass_yds' in sl && !_n(sl.int)) ydVal = _n(sl.pass_yds);
    var ydChip = (ydVal != null)
      ? ' <span class="rz-event-yd ' + (ydVal > 0 ? 'pos' : (ydVal < 0 ? 'neg' : 'zero')) + '">'
        + (ydVal > 0 ? '+' : '') + ydVal + ' YD</span>'
      : '';
    // Running per-player stat line through this play (Sleeper-style context).
    var cumeStr = _cumeLine(ev.pos, ev.cume);
    var cumeHtml = cumeStr
      ? '<div class="rz-event-cume">' + ev.pos + ' · ' + cumeStr + '</div>'
      : '';
    var isDef = String(ev.pos || '').toUpperCase() === 'DEF';
    var defTeam = ev.nflTeam || (isDef ? ev.pid : '') || '';
    var avSrc, avOnErr;
    if (isDef && defTeam && window.brTeamLogoLocal) {
      avSrc = window.brTeamLogoLocal(defTeam);
      avOnErr = 'brDefImgOnError(this)';
    } else if (isDef && defTeam) {
      var _dt = String(defTeam).toUpperCase();
      if (_dt === 'WSH') _dt = 'WAS';
      avSrc = '/static/images/team_logos/' + _dt + '.png';
      avOnErr = "var t=this.getAttribute('data-team');if(t&&!this._espnFallback){this._espnFallback=1;this.src='https://a.espncdn.com/i/teamlogos/nfl/500/'+(String(t).toUpperCase()==='WAS'?'wsh':String(t).toLowerCase())+'.png';}else{this.parentNode.classList.add('img-err');}";
    } else {
      avSrc = 'https://sleepercdn.com/content/nfl/players/thumb/' + ev.pid + '.jpg';
      avOnErr = "this.parentNode.classList.add('img-err')";
    }
    return (
      '<div class="rz-event ' + ev.kind + (ev.mine ? ' is-mine' : '') + (_isBigPlay(ev) ? ' is-big' : '') + (animate ? '' : ' rz-event-old') + '" data-pid="' + ev.pid + '">'
      + '<div class="rz-event-avatar rz-av-' + posKey + '" data-init="' + initials + '">'
      + '<img class="rz-headshot' + (isDef ? ' rz-team-logo' : '') + '" src="' + avSrc + '" alt=""'
      + (isDef && defTeam ? ' data-team="' + String(defTeam).replace(/"/g, '') + '"' : '')
      + ' onerror="' + avOnErr + '">'
      + '</div>'
      + '<div class="rz-event-body">'
      + situationHtml
      + '<div class="rz-event-main"><span class="rz-event-name">' + ev.name + '</span>' + tag + '</div>'
      + '<div class="rz-event-desc">' + ev.desc + ydChip + '</div>'
      + cumeHtml
      + '</div>'
      + '<div class="rz-event-delta ' + deltaCls + '">'
      + ((scoreStr || clockStr) ? '<div class="rz-event-delta-game">'
        + (scoreStr ? '<div class="rz-event-score">' + scoreStr + '</div>' : '')
        + (clockStr ? '<div class="rz-event-clock">' + clockStr + '</div>' : '')
        + '</div>' : '')
      + '<div class="rz-event-delta-pts">' + (deltaPrimary || '') + '</div>'
      + (deltaSecondary ? '<div class="rz-event-total"><span>' + deltaSecondary + '</span> total</div>' : '')
      + '</div>'
      + '</div>'
    );
  }

  var _PAGE_SIZE = 20;

  // Expose live data to the global player modal (injected as "Live" tab)
  // Override the global stub with the live Redzone state + event feed
  window.__rzGetPlayerLive = function(pid) {
    return window._rzBuildLiveHtml(pid, _state, _feed);
  };

  // "On deck": games kicking off within the next 90 min that include my players.
  function _onDeckHtml() {
    var info = _state.player_info || {};
    var myPids = new Set();
    _myMatchups().forEach(function(m) {
      (m.starters || []).forEach(function(pid) { myPids.add(pid); });
    });
    if (!myPids.size) return '';
    var now = Date.now() / 1000;
    var WINDOW = 90 * 60;
    var games = {};
    Object.keys(info).forEach(function(pid) {
      if (!myPids.has(pid)) return;
      var p = info[pid];
      if (String(p.game_code || '0') !== '0') return;        // upcoming only
      var ep = parseFloat(p.game_time_epoch || 0);
      if (!ep || ep < now || ep - now > WINDOW) return;       // within next 90 min
      var gid = p.game_id || (p.away + '@' + p.home);
      if (!games[gid]) games[gid] = { away: p.away, home: p.home, ep: ep, names: [] };
      games[gid].names.push((p.name || pid));
    });
    var gids = Object.keys(games).sort(function(a, b) { return games[a].ep - games[b].ep; });
    if (!gids.length) return '';
    var rows = gids.map(function(gid) {
      var g = games[gid];
      var mins = Math.max(1, Math.round((g.ep - now) / 60));
      var nameStr = g.names.slice(0, 3).join(', ') + (g.names.length > 3 ? ' +' + (g.names.length - 3) : '');
      return '<div class="rz-ondeck-item">'
        + '<span class="rz-ondeck-clock">▶ ' + mins + 'm</span>'
        + '<span class="rz-ondeck-game">' + g.away + ' @ ' + g.home + '</span>'
        + '<span class="rz-ondeck-players">' + nameStr + '</span>'
        + '</div>';
    }).join('');
    return '<div class="rz-ondeck-bar"><span class="rz-ondeck-label">On deck</span>' + rows + '</div>';
  }

  function _pregameScheduleHtml() {
    var info = _state.player_info || {};
    // When a matchup card is selected (hero), scope the schedule to that
    // matchup's games so its upcoming kickoffs stay visible; otherwise show
    // every game. focusPids is the selected matchup's player set (or null).
    var focusPids = _heroMid ? _heroMatchupPids() : null;
    var myPids = new Set();
    _myMatchups().forEach(function(m) {
      (m.players || []).forEach(function(pid) { myPids.add(pid); });
    });
    // Players to spotlight on each game card: the focused matchup's when one is
    // selected, otherwise the viewer's own.
    var spotPids = focusPids || myPids;
    var heroIsMine = !!_heroMid && _myMatchups().some(function(m) {
      return String(m.matchup_id) === _heroMid || String(m.roster_id) === _heroMid;
    });
    var spotLabel = (!focusPids || heroIsMine) ? 'My Players' : 'In Matchup';

    // Group scheduled/live players by game_id
    var gameMap = {};
    Object.keys(info).forEach(function(pid) {
      var p = info[pid];
      var gid = p.game_id || '';
      if (!gid || !p.home || !p.away) return;
      var code = String(p.game_code || '0');
      if (code === '2') return; // skip final games
      if (!gameMap[gid]) gameMap[gid] = { home: p.home, away: p.away, status: p.game_status || '', code: code, kickoff: parseFloat(p.game_time_epoch || 0) || 0, spot: [], hasFocus: false };
      if (focusPids && focusPids.has(pid)) gameMap[gid].hasFocus = true;
      if (spotPids.has(pid)) gameMap[gid].spot.push({ name: p.name || pid, pos: p.pos || '' });
    });

    var gameIds = Object.keys(gameMap);
    // Focused matchup: keep only games that include one of its players.
    if (focusPids) gameIds = gameIds.filter(function(gid) { return gameMap[gid].hasFocus; });
    if (!gameIds.length) {
      return '<div class="rz-pregame-empty-hint">Plays appear here as games unfold -- targets, catches, carries and touchdowns with live fantasy points.'
      + (!_isDemo ? '<div class="rz-demo-cta"><a href="?demo=1" class="rz-demo-link">Try the Redzone demo</a></div>' : '')
      + '</div>';
    }

    // Sort: games with spotlighted players first
    gameIds.sort(function(a, b) {
      return (gameMap[b].spot.length > 0 ? 1 : 0) - (gameMap[a].spot.length > 0 ? 1 : 0);
    });

    var html = '<div class="rz-pregame-wrap">'
      + '<div class="rz-pregame-label">' + (gameIds.some(function(g) { return gameMap[g].code === '1'; }) ? 'Games in Progress' : 'Upcoming Games') + '</div>';

    gameIds.forEach(function(gid) {
      var g = gameMap[gid];
      var statusText = g.code === '1'
        ? 'LIVE · ' + (g.status || '')
        : (g.kickoff ? _fmtKickoff(g.kickoff) : (g.status || 'Upcoming'));
      var playerChip = '';
      if (g.spot.length) {
        var rows = g.spot.slice(0, 3).map(function(pl) {
          return '<span class="rz-pregame-player">' + _posHtml(pl.pos)
            + '<span class="rz-pregame-pname">' + pl.name + '</span></span>';
        }).join('');
        var moreN = g.spot.length - 3;
        var more = moreN > 0 ? '<span class="rz-pregame-more">+' + moreN + ' more</span>' : '';
        playerChip = '<div class="rz-pregame-players"><strong>' + spotLabel + '</strong>' + rows + more + '</div>';
      }
      html += '<div class="rz-pregame-game">'
        + '<div class="rz-pregame-teams">'
        +   g.away + ' @ ' + g.home
        +   '<div class="rz-pregame-time">' + statusText + '</div>'
        + '</div>'
        + playerChip
        + '</div>';
    });

    html += '</div>';
    return html;
  }

  function _syncFeed() {
    _resetGameCache();
    var container = document.getElementById('rz-feed-list');
    if (!container) return;

    if (_loadingScope) {
      container.innerHTML = '<div class="rz-feed-loading">'
        + '<span class="rz-feed-spinner"></span>Loading your leagues…</div>';
      _renderPagination(1);
      return;
    }

    // The feed is always reverse chronological: newest plays belong first.
    var filtered = _feed.filter(_eventMatches);
    var list = _chronoSort(filtered);
    // Hero focus alone should not force the "no matching" empty when the feed
    // itself is empty -- the pregame schedule already respects hero focus.
    var hardFilter = _filters.nfl !== 'all' || _filters.pos !== 'all' || _filters.stat !== 'all' || _myTeamOnly || _bigPlaysOnly
      || (_heroMid && _feed.length > 0);
    var totalPages = Math.max(1, Math.ceil(list.length / _PAGE_SIZE));
    if (_feedPage >= totalPages) _feedPage = totalPages - 1;

    if (!list.length) {
      if (hardFilter) {
        if (window.brEmptyState) {
          window.brEmptyState(container, {
            icon: 'search',
            title: 'No matching plays',
            message: 'Try clearing a filter to see more of the feed.',
            compact: true
          });
        } else {
          container.innerHTML = '<div class="rz-feed-empty">No plays match these filters yet.</div>';
        }
      } else {
        // Live/final with a PBP attempt but no lines yet -- honest empty, not
        // boxscore / "Scored X pts" fiction.
        var pbpAttempted = Object.keys(_state.pbp_by_game || {}).length > 0
          || Object.keys(_pbpGames).length > 0;
        var liveOrFinal = Object.keys(_state.player_info || {}).some(function(pid) {
          var c = String(((_state.player_info || {})[pid] || {}).game_code || '');
          return c === '1' || c === '2';
        });
        if (pbpAttempted && liveOrFinal) {
          container.innerHTML = '<div class="rz-feed-empty">Play-by-play lines aren’t available for these games yet. We only show real PBP -- not box-score summaries.</div>';
        } else {
          container.innerHTML = _pregameScheduleHtml();
        }
      }
      _renderPagination(totalPages);
      return;
    }

    // Clear empty-state placeholder if present
    var empty = container.querySelector('.rz-feed-empty');
    if (empty) empty.remove();

    if (_feedPage > 0) {
      // Static page: full rebuild from slice
      var pageItems = list.slice(_feedPage * _PAGE_SIZE, (_feedPage + 1) * _PAGE_SIZE);
      var frag2 = document.createDocumentFragment();
      pageItems.forEach(function(ev) {
        var wrap = document.createElement('div');
        wrap.innerHTML = _eventHtml(ev, false);
        var node = wrap.firstChild;
        node.dataset.eid = _eid(ev);
        frag2.appendChild(node);
      });
      container.innerHTML = '';
      container.appendChild(frag2);
      _renderPagination(totalPages);
      // Click handlers now managed by root event delegation
      return;
    }

    // Page 0: live DOM-patching + FLIP
    var page0Items = list.slice(0, _PAGE_SIZE);
    var page0Eids = new Set(page0Items.map(function(ev) { return _eid(ev); }));

    function _orderFeedDom(target, orderedItems) {
      var byId = {};
      target.querySelectorAll('[data-eid]').forEach(function(el) { byId[el.dataset.eid] = el; });
      orderedItems.forEach(function(ev) {
        var node = byId[_eid(ev)];
        if (node) target.appendChild(node);
      });
    }

    // Remove events that have fallen off page 0
    container.querySelectorAll('[data-eid]').forEach(function(el) {
      if (!page0Eids.has(el.dataset.eid)) el.remove();
    });

    var inDom = new Set();
    container.querySelectorAll('[data-eid]').forEach(function(el) { inDom.add(el.dataset.eid); });
    var toAdd = page0Items.filter(function(ev) { return !inDom.has(_eid(ev)); });

    if (toAdd.length) {
      var isInitialLoad = _shownFeedIds.size === 0;
      var newCount = toAdd.filter(function(ev) { return !_shownFeedIds.has(_eid(ev)); }).length;
      // Sequential stagger: insert new plays one at a time during live polling
      var liveStagger = !isInitialLoad && newCount > 1;

      var existingEls = [], existingTops = [];
      if (!isInitialLoad && !liveStagger && toAdd.length <= 4) {
        existingEls = Array.from(container.querySelectorAll('[data-eid]')).slice(0, 12);
        existingTops = existingEls.map(function(n) { return n.getBoundingClientRect().top; });
      }

      var newIdx = 0;
      var insertDelay = 0;
      var frag = document.createDocumentFragment();
      toAdd.forEach(function(ev) {
        var id = _eid(ev);
        var isNew = !_shownFeedIds.has(id);
        var wrap = document.createElement('div');
        wrap.innerHTML = _eventHtml(ev, isNew);
        var node = wrap.firstChild;
        node.dataset.eid = id;
        _shownFeedIds.add(id);

        if (isNew && liveStagger) {
          // Insert each new play into the DOM individually, one at a time
          (function(n, e, delay) {
            setTimeout(function() {
              _bigPlayFx(n, e, container, true);
              container.insertBefore(n, container.firstChild);
              _orderFeedDom(container, page0Items);
              // Click handlers now managed by root event delegation
            }, delay);
          })(node, ev, insertDelay);
          insertDelay += 420;
          newIdx++;
        } else {
          if (isNew && newCount > 1) {
            // Initial load: quick cascade so the list doesn't appear all at once
            node.style.animationDelay = (newIdx * 60) + 'ms';
          }
          if (isNew) {
            newIdx++;
            _bigPlayFx(node, ev, container, false);
          }
          frag.appendChild(node);
        }
      });
      container.insertBefore(frag, container.firstChild);
      _orderFeedDom(container, page0Items);

      // Auto-scroll to top if user was already near top (don't interrupt mid-scroll)
      if (!isInitialLoad && !liveStagger) {
        var feedTop = container.getBoundingClientRect().top;
        if (feedTop > -80) {
          var first = container.firstChild;
          if (first && first.scrollIntoView) {
            first.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
      }

      if (existingEls.length) {
        requestAnimationFrame(function() {
          existingEls.forEach(function(n, i) {
            if (!n.parentNode) return;
            var dy = n.getBoundingClientRect().top - existingTops[i];
            if (Math.abs(dy) > 0.5) {
              n.style.transition = 'none';
              n.style.transform = 'translateY(' + (-dy) + 'px)';
              requestAnimationFrame(function() {
                n.style.transition = 'transform .28s cubic-bezier(.22,.68,0,1.15)';
                n.style.transform = '';
                setTimeout(function() { if (n.style) n.style.transition = ''; }, 320);
              });
            }
          });
        });
      }
    }

    // Reconcile even when every ID already existed: filter changes and
    // corrected provider ordering must still be reflected by the DOM.
    _orderFeedDom(container, page0Items);

    // Prune to page size
    var items = container.querySelectorAll('[data-eid]');
    for (var i = _PAGE_SIZE; i < items.length; i++) items[i].remove();

    _renderPagination(totalPages);

    // Live feed header
    var hdr = document.getElementById('rz-feed-hdr');
    if (hdr) {
      var totalEvts = list.length;
      var liveNow = _anyLive();
      var statusText = totalEvts
        ? (liveNow
          ? '<span class="rz-fh-dot"></span><span class="rz-fh-text">Live · <b>' + totalEvts + '</b> ' + (totalEvts === 1 ? 'play' : 'plays') + '</span>'
          : '<span class="rz-fh-text"><b>' + totalEvts + '</b> ' + (totalEvts === 1 ? 'play' : 'plays') + ' · Final</span>')
        : '';
      hdr.innerHTML = '<div class="rz-feed-hdr-left">' + statusText + '</div>';
    }

    // Click handlers now managed by root event delegation
  }

  function _renderPagination(totalPages) {
    var el = document.getElementById('rz-feed-pagination');
    if (!el) return;
    if (totalPages <= 1) { el.innerHTML = ''; return; }
    var prevDis = _feedPage <= 0;
    var nextDis  = _feedPage >= totalPages - 1;
    el.innerHTML =
      '<button class="rz-page-btn' + (prevDis ? ' disabled' : '') + '" id="rz-page-prev"' + (prevDis ? ' disabled' : '') + '>← Prev</button>'
      + '<span class="rz-page-info">' + (_feedPage + 1) + ' / ' + totalPages + '</span>'
      + '<button class="rz-page-btn' + (nextDis ? ' disabled' : '') + '" id="rz-page-next"' + (nextDis ? ' disabled' : '') + '>Next →</button>';
    var prevBtn = el.querySelector('#rz-page-prev');
    var nextBtn = el.querySelector('#rz-page-next');
    if (prevBtn && !prevDis) prevBtn.addEventListener('click', function() { _feedPage--; _syncFeed(); });
    if (nextBtn && !nextDis) nextBtn.addEventListener('click', function() { _feedPage++; _syncFeed(); });
  }

  // Wire the NFL game pills to the single canonical filter (`_filters.nfl`).
  // A full _render() repaints the Filter panel from the same state, so both
  // controls stay synchronized with no second selection state (§5, §23).
  function _wireGameStrip(scrollToSelected) {
    root.querySelectorAll('[data-nfl-gid]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var next = btn.dataset.nflGid || 'all';
        if (_filters.nfl === next) return; // re-tapping the selection is a no-op
        _filters.nfl = next;
        _filterOpen = false;
        _feedPage = 0;
        _render();
      });
    });
    if (scrollToSelected) {
      var sel = root.querySelector('.rz-game-pill.is-selected:not(.rz-game-pill-all)');
      if (sel && sel.scrollIntoView) {
        try { sel.scrollIntoView({ inline: 'nearest', block: 'nearest' }); }
        catch (_) { /* older browsers: leave scroll as-is */ }
      }
    }
  }

  function _wireHeroCards() {
    root.querySelectorAll('[data-heromid]').forEach(function(el) {
      // Keyboard activation for the (role=button) matchup cards.
      el.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
          e.preventDefault();
          el.click();
        }
      });
      el.addEventListener('click', function() {
        var mid = el.dataset.heromid;
        var prevMid = _heroMid;
        var order = [];
        root.querySelectorAll('[data-heromid]').forEach(function(c) { order.push(c.dataset.heromid); });
        var oldIdx = order.indexOf(prevMid || '');
        var newIdx = order.indexOf(mid);
        if (prevMid === null) {
          _slideDir = 'from-right';
        } else if (prevMid === mid) {
          _slideDir = 'from-left';
        } else if (oldIdx >= 0 && newIdx >= 0) {
          _slideDir = newIdx > oldIdx ? 'from-right' : 'from-left';
        } else {
          _slideDir = 'from-right';
        }
        _heroMid = prevMid === mid ? null : mid;
        _heroTouched = true;
        _savePrefs();
        _feedPage = 0;
        _render();
      });
    });
  }

  // Toggle the hero-strip arrows + edge fades based on current scroll position.
  // Re-queries the live DOM each call so it is safe to bind to window resize
  // once (the strip node is replaced on every render).
  function _updateHeroArrows() {
    var scroller = root.querySelector('.rz-hero-scroller');
    if (!scroller) return;
    var row = scroller.querySelector('.rz-hero-cards-row');
    if (!row) return;
    var maxScroll = row.scrollWidth - row.clientWidth;
    var overflow = maxScroll > 4;
    var x = row.scrollLeft;
    var leftBtn = scroller.querySelector('[data-hero-arrow="-1"]');
    var rightBtn = scroller.querySelector('[data-hero-arrow="1"]');
    if (leftBtn) leftBtn.classList.toggle('rz-arrow-off', !overflow || x <= 2);
    if (rightBtn) rightBtn.classList.toggle('rz-arrow-off', !overflow || x >= maxScroll - 2);
    scroller.classList.toggle('rz-fade-left', overflow && x > 2);
    scroller.classList.toggle('rz-fade-right', overflow && x < maxScroll - 2);
  }

  function _wireHeroScroll() {
    var scroller = root.querySelector('.rz-hero-scroller');
    if (!scroller) return;
    var row = scroller.querySelector('.rz-hero-cards-row');
    if (!row) return;

    function scrollByDir(dir) {
      var amt = Math.max(row.clientWidth * 0.8, 160);
      row.scrollBy({ left: dir * amt, behavior: 'smooth' });
    }
    scroller.querySelectorAll('[data-hero-arrow]').forEach(function(btn) {
      btn.addEventListener('click', function() { scrollByDir(parseInt(btn.dataset.heroArrow, 10)); });
    });

    // Vertical mouse-wheel → horizontal scroll (desktop mice have no h-scroll).
    row.addEventListener('wheel', function(e) {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return; // let native h-scroll pass
      var maxScroll = row.scrollWidth - row.clientWidth;
      if (maxScroll <= 4) return;
      if ((e.deltaY < 0 && row.scrollLeft <= 0) || (e.deltaY > 0 && row.scrollLeft >= maxScroll)) return;
      e.preventDefault();
      row.scrollLeft += e.deltaY;
      _updateHeroArrows();
    }, { passive: false });

    row.addEventListener('scroll', _updateHeroArrows, { passive: true });
    _updateHeroArrows();
  }

  function _renderScopeToggle() {
    var canUser = _isDemo || (_state.viewer_roster_ids && _state.viewer_roster_ids.length) || window._isSignedIn;
    if (!canUser) return '';
    return (
      '<div class="rz-scope-toggle">'
      + '<button class="rz-scope-btn' + (_scope === 'league' ? ' active' : '') + '" data-scope="league">This League</button>'
      + '<button class="rz-scope-btn' + (_scope === 'user'   ? ' active' : '') + '" data-scope="user">My Leagues</button>'
      + '</div>'
    );
  }

  var _activeTab = 'plays';

  function _partialUpdate() {
    _resetGameCache();
    // Update timer text
    var timerEl = document.getElementById('rz-timer');
    if (timerEl) { timerEl.textContent = _fmtTimer(_countdown); timerEl.classList.remove('rz-timer-refreshing'); }

    // Update live chip in header
    var liveChipEl = root.querySelector('.rz-live-chip');
    var headerRight = root.querySelector('.rz-header-right');
    if (headerRight) {
      var liveChipHtml = _statusChipHtml();
      headerRight.innerHTML = liveChipHtml + '<button class="rz-refresh-timer" id="rz-timer">' + _fmtTimer(_countdown) + '</button>';
    }

    // Replace hero cards in-place and re-wire. Preserve the strip's horizontal
    // scroll position so a live poll doesn't yank the user back to the start.
    var heroWrap = root.querySelector('.rz-hero-cards, .rz-no-matchup');
    if (heroWrap) {
      var prevRow = heroWrap.querySelector('.rz-hero-cards-row');
      var prevScrollLeft = prevRow ? prevRow.scrollLeft : 0;
      var tempDiv = document.createElement('div');
      tempDiv.innerHTML = _renderHeroCards();
      var newHero = tempDiv.firstChild;
      if (newHero) {
        heroWrap.parentNode.replaceChild(newHero, heroWrap);
        var newRow = newHero.querySelector && newHero.querySelector('.rz-hero-cards-row');
        if (newRow && prevScrollLeft) newRow.scrollLeft = prevScrollLeft;
      }
    }
    _wireHeroCards();
    _wireHeroScroll();

    // Update filter chips (hero chip may change)
    var showFilters = (_activeTab === 'plays' || _activeTab === 'top');
    var chipBar = root.querySelector('.rz-chip-bar');
    if (chipBar && showFilters) {
      var tempDiv2 = document.createElement('div');
      tempDiv2.innerHTML = _renderFilterChips();
      var newChips = tempDiv2.firstChild;
      if (newChips) chipBar.parentNode.replaceChild(newChips, chipBar);
      // Re-wire chip clear handlers
      var myTeamToggle2 = root.querySelector('#rz-myteam-btn');
      if (myTeamToggle2) {
        myTeamToggle2.addEventListener('click', function() { _myTeamOnly = !_myTeamOnly; _savePrefs(); _feedPage = 0; _render(); });
      }
      var bigToggle2 = root.querySelector('#rz-bigplays-btn');
      if (bigToggle2) {
        bigToggle2.addEventListener('click', function() { _bigPlaysOnly = !_bigPlaysOnly; _savePrefs(); _feedPage = 0; _render(); });
      }
      root.querySelectorAll('[data-clear]').forEach(function(btn) {
        btn.addEventListener('click', function() { _filters[btn.dataset.clear] = 'all'; _feedPage = 0; _render(); });
      });
      root.querySelectorAll('[data-clear-hero]').forEach(function(el) {
        el.addEventListener('click', function() { _heroMid = null; _heroTouched = true; _savePrefs(); _feedPage = 0; _render(); });
      });
      root.querySelectorAll('[data-fk]').forEach(function(btn) {
        btn.addEventListener('click', function() { _filters[btn.dataset.fk] = btn.dataset.fv; _filterOpen = false; _feedPage = 0; _render(); });
      });
      var filterBtn = root.querySelector('#rz-filter-btn');
      if (filterBtn) filterBtn.addEventListener('click', function() { _filterOpen = !_filterOpen; _render(); });
    }

    // Refresh the NFL game pill strip in place (scores/status update) without
    // moving its horizontal scroll position or yanking the page.
    var stripEl = root.querySelector('.rz-game-strip');
    if (stripEl && showFilters) {
      var prevScroll = 0;
      var prevScrollEl = stripEl.querySelector('.rz-game-strip-scroll');
      if (prevScrollEl) prevScroll = prevScrollEl.scrollLeft;
      var stripWrap = document.createElement('div');
      stripWrap.innerHTML = _renderGameStrip();
      var newStrip = stripWrap.firstChild;
      if (newStrip) {
        stripEl.parentNode.replaceChild(newStrip, stripEl);
        var newScrollEl = newStrip.querySelector && newStrip.querySelector('.rz-game-strip-scroll');
        if (newScrollEl && prevScroll) newScrollEl.scrollLeft = prevScroll;
        _wireGameStrip(false);
      } else {
        stripEl.remove();
      }
    }

    // Refresh NFL matchup scoreboard (score / clock / d&d / possession)
    var boardEl = root.querySelector('#rz-nfl-board');
    var boardHtml = showFilters ? _renderNflBoard() : '';
    if (boardHtml) {
      var boardWrap = document.createElement('div');
      boardWrap.innerHTML = boardHtml;
      var newBoard = boardWrap.firstChild;
      if (boardEl && newBoard) boardEl.parentNode.replaceChild(newBoard, boardEl);
      else if (newBoard) {
        var chipBar2 = root.querySelector('.rz-chip-bar');
        var anchor = chipBar2 && (chipBar2.nextElementSibling && chipBar2.nextElementSibling.classList.contains('rz-filter-panel')
          ? chipBar2.nextElementSibling : chipBar2);
        var panelsHost = root.querySelector('.rz-main-card');
        if (anchor && anchor.parentNode) anchor.parentNode.insertBefore(newBoard, anchor.nextSibling);
        else if (panelsHost) panelsHost.insertBefore(newBoard, panelsHost.firstChild);
      }
    } else if (boardEl) {
      boardEl.remove();
    }

    // Sync feed (live-patches page 0)
    _syncFeed();
  }

  function _tabsFor() {
    return _scope === 'user'
      ? [{ key: 'plays', label: 'Plays' }, { key: 'mine', label: 'My Teams' }, { key: 'top', label: 'Top' }]
      : [{ key: 'plays', label: 'Plays' }, { key: 'mine', label: 'My Team' }, { key: 'opp', label: 'Opp' }, { key: 'top', label: 'Top' }];
  }

  function _historyPanelHtml() {
    if (!_notifHistory.length) return '';
    var rows = _notifHistory.map(function(h) {
      var d = new Date(h.ts);
      var timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      var ptsStr = h.pts > 0 ? '+' + _fmt(h.pts) + ' pts' : '';
      var kindCls = h.kind === 'td' ? ' td' : h.kind === 'score' ? ' score' : '';
      return '<div class="rz-hist-row' + kindCls + '">'
        + '<span class="rz-hist-time">' + timeStr + '</span>'
        + '<span class="rz-hist-desc">' + h.name + ' – ' + h.desc + '</span>'
        + (ptsStr ? '<span class="rz-hist-pts">' + ptsStr + '</span>' : '')
        + '</div>';
    }).join('');
    return '<div class="rz-hist-overlay" id="rz-hist-overlay">'
      + '<div class="rz-hist-panel">'
      + '<div class="rz-hist-hdr"><span>TD Alert History</span>'
      + '<button class="rz-hist-clear" id="rz-hist-clear">Clear</button>'
      + '<button class="rz-hist-close" id="rz-hist-close">✕</button>'
      + '</div>'
      + '<div class="rz-hist-body">' + rows + '</div>'
      + '</div>'
      + '</div>';
  }

  // ── Full render ───────────────────────────────────────────────────────────────
  function _render() {
    _resetGameCache();
    var TABS = _tabsFor();
    if (!TABS.some(function(t) { return t.key === _activeTab; })) _activeTab = 'plays';

    var live = _anyLive();
    var idle = !_isDemo && !live && !_isGameDay();  // offseason / no games today
    var liveChip = _statusChipHtml();
    var demoPill = _isDemo ? '<span class="rz-demo-pill">DEMO</span>' : '';
    var showFilters = (_activeTab === 'plays' || _activeTab === 'top');

    var pair = _focusedPair();
    var myMatchup = pair ? pair.mine : null;
    var oppMatchup = pair ? pair.opp : null;

    var summary = _loadingScope ? _renderSkeletonHero() : _renderHeroCards();

    var tabBar = '<div class="rz-tab-bar">' + TABS.map(function(t) {
      var badge = (t.key === 'plays' && _unreadCount > 0 && _activeTab !== 'plays')
        ? '<span class="rz-tab-badge">' + (_unreadCount > 99 ? '99+' : _unreadCount) + '</span>' : '';
      return '<button class="rz-tab-btn' + (_activeTab === t.key ? ' active' : '') + '" data-tab="' + t.key + '">' + t.label + badge + '</button>';
    }).join('') + '</div>';

    var minePanel = _scope === 'user' ? _renderMyTeams() : _rosterCard(myMatchup);

    // Pinned score bar for the Plays tab
    var playsScoreBar = '';
    if (myMatchup && !_loadingScope) {
      var _mm = myMatchup, _om = oppMatchup;
      var _myP = parseFloat(_mm.points || 0), _opP = parseFloat(_om ? _om.points || 0 : 0);
      var _win = _myP >= _opP, _diff = Math.abs(_myP - _opP).toFixed(1);
      var _liveBar = _matchupIsLive([_mm, _om]) || (!_heroMid && _anyLive());
      var _meLabel = (pair && pair.isMine) ? 'Me' : (_ownerName(_mm.roster_id) || 'Team');
      var _oppName = _om ? (_ownerName(_om.roster_id) || 'Opp') : 'Opp';
      playsScoreBar = '<div class="rz-plays-scorebar">'
        + '<span class="rz-psb-me' + (_win ? ' lead' : '') + '">' + _meLabel + '  ' + _fmt(_myP) + '</span>'
        + '<span class="rz-psb-sep">' + (_liveBar ? '<span class="rz-psb-live-dot"></span>' : '') + 'vs</span>'
        + '<span class="rz-psb-opp' + (!_win ? ' lead' : '') + '">' + _oppName + '  ' + _fmt(_opP) + '</span>'
        + (_liveBar ? '<span class="rz-psb-spread">' + (_win ? '+' : '-') + _diff + '</span>' : '')
        + '</div>';
    }

    var panels =
        '<div class="rz-panel' + (_activeTab === 'plays'  ? ' active' : '') + '" id="rz-panel-plays"><div class="rz-feed-hdr" id="rz-feed-hdr"></div><div id="rz-feed-list"></div><div id="rz-feed-pagination"></div>' + playsScoreBar + '</div>'
      + '<div class="rz-panel' + (_activeTab === 'mine'   ? ' active' : '') + '" id="rz-panel-mine">'   + minePanel              + '</div>'
      + (_scope === 'user' ? '' :
        '<div class="rz-panel' + (_activeTab === 'opp'    ? ' active' : '') + '" id="rz-panel-opp">'    + _rosterCard(oppMatchup) + '</div>')
      + '<div class="rz-panel' + (_activeTab === 'top'    ? ' active' : '') + '" id="rz-panel-top">'    + _renderTopPerformers()  + '</div>';

    var exitBtn  = _isDemo ? '<button class="rz-demo-exit" id="rz-demo-exit">Exit Demo</button>' : '';
    var staleChip = _lastPollFailed ? '<span class="rz-stale-badge">⚠ Stale</span>' : '';
    var timerLabel = _lastPollFailed ? '?' : (idle ? '-' : _fmtTimer(_countdown));
    var notifCta = (!_notifDismissed && 'Notification' in window && Notification.permission === 'default')
      ? '<div class="rz-notif-cta" id="rz-notif-cta"><span>Enable TD alerts</span><button class="rz-notif-cta-btn" id="rz-notif-enable">Enable</button><button class="rz-notif-cta-x" id="rz-notif-dismiss">✕</button></div>'
      : '';
    root.innerHTML =
      notifCta
      + '<div class="rz-header">'
      + '<div class="rz-brand"><div class="rz-brand-dot' + (live ? ' is-live' : '') + '"></div><span class="rz-brand-name">BR Redzone</span><span class="rz-brand-week">Wk ' + (_state.week || '') + '</span>' + demoPill + '</div>'
      + '<div class="rz-header-right">' + exitBtn + staleChip + liveChip + '<button class="rz-refresh-timer" id="rz-timer">' + timerLabel + '</button></div>'
      + '</div>'
      + '<div class="rz-content">'
      + _renderScopeToggle()
      + summary
      + _onDeckHtml()
      + '<div class="rz-main-card">'
      + tabBar
      + (showFilters ? _renderFilterChips() : '')
      + (showFilters ? _renderGameStrip() : '')
      + (showFilters ? _renderNflBoard() : '')
      + panels
      + '</div>'
      + '</div>'
      + '<p class="rz-source-note" style="margin:12px 16px 0;font-size:11px;color:var(--text-muted);">Live scores and player lines use Sleeper ids plus Tank01 box scores. Cross-league “My Leagues” follows your signed-in portfolio.</p>'
      + (_historyOpen ? _historyPanelHtml() : '');

    _syncFeed();

    // Sync sticky header top to actual nav height (prevents overlap on scroll)
    var topNav = document.querySelector('.top-nav');
    if (topNav) {
      var navH = Math.ceil(topNav.getBoundingClientRect().height);
      if (navH > 0) document.documentElement.style.setProperty('--rz-nav-h', navH + 'px');
    }

    // Slide animation when hero card selection changes
    if (_slideDir !== 'none') {
      var mc = root.querySelector('.rz-main-card');
      if (mc) {
        mc.classList.add('rz-slide-' + _slideDir);
        var _sd = _slideDir;
        setTimeout(function() { if (mc) mc.classList.remove('rz-slide-' + _sd); }, 300);
      }
      _slideDir = 'none';
    }

    var myTeamToggle = root.querySelector('#rz-myteam-btn');
    if (myTeamToggle) {
      myTeamToggle.addEventListener('click', function() {
        _myTeamOnly = !_myTeamOnly;
        _savePrefs();
        _feedPage = 0;
        _render();
      });
    }
    var bigToggle = root.querySelector('#rz-bigplays-btn');
    if (bigToggle) {
      bigToggle.addEventListener('click', function() {
        _bigPlaysOnly = !_bigPlaysOnly;
        _savePrefs();
        _feedPage = 0;
        _render();
      });
    }

    root.querySelectorAll('.rz-scope-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        if (btn.dataset.scope === _scope) return;
        _streamGen++; // abort any in-flight My Leagues stream / poll from a prior switch
        _streaming = false;
        _mlNames = []; _mlLoaded = null; _mlFailed = null;
        _scope = btn.dataset.scope;
        _filters = { nfl: 'all', pos: 'all', stat: 'all' };
        _feed = [];
        _shownFeedIds = new Set();
        _resetFeedSnapshots(); // don't diff the new scope against the old one's lines
        _filterOpen = false;
        _myTeamOnly = false;
        _bigPlaysOnly = false;
        _heroMid = null;
        _heroTouched = false; // restore prefs for the newly selected scope
        _loadPrefs();
        _feedPage = 0;
        _countdown = 1;
        // Prefer the last-good payload for this scope so a late My Leagues
        // response cannot flash ESPN/portfolio names under This League.
        var cached = _scopeCache[_scope];
        if (cached) {
          _state = cached;
          _myRids = _myRidSet(cached);
          _loadingScope = false;
          // Restore cards + Plays immediately from cache, then refresh in background.
          _hydrateFeed(cached);
          _applyDefaultHero();
        } else {
          // Show skeleton cards until this scope's (often multi-league) data lands.
          _loadingScope = true;
        }
        _syncScopeUrl();
        _render();
        // My Leagues streams a card at a time; This League is a single fetch.
        if (_scope === 'user') _refreshUserStream();
        else _refresh();
      });
    });
    root.querySelectorAll('.rz-tab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        _activeTab = btn.dataset.tab;
        if (_activeTab === 'plays') _unreadCount = 0;
        _render();
      });
    });
    var filterBtn = root.querySelector('#rz-filter-btn');
    if (filterBtn) filterBtn.addEventListener('click', function() {
      _filterOpen = !_filterOpen;
      _render();
    });
    root.querySelectorAll('[data-fk]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        _filters[btn.dataset.fk] = btn.dataset.fv;
        _filterOpen = false;
        _feedPage = 0;
        _render();
      });
    });
    root.querySelectorAll('[data-clear]').forEach(function(chip) {
      chip.addEventListener('click', function() {
        _filters[chip.dataset.clear] = 'all';
        _feedPage = 0;
        _render();
      });
    });
    root.querySelectorAll('[data-clear-hero]').forEach(function(el) {
      el.addEventListener('click', function() { _heroMid = null; _heroTouched = true; _savePrefs(); _feedPage = 0; _render(); });
    });
    root.querySelectorAll('[data-clear-myteam]').forEach(function(el) {
      el.addEventListener('click', function() { _myTeamOnly = false; _savePrefs(); _feedPage = 0; _render(); });
    root.querySelectorAll('[data-clear-big]').forEach(function(el) {
      el.addEventListener('click', function() { _bigPlaysOnly = false; _savePrefs(); _feedPage = 0; _render(); });
    });
    });
    var exitDemo = root.querySelector('#rz-demo-exit');
    if (exitDemo) exitDemo.addEventListener('click', function() { window.location.href = window.location.pathname; });

    var notifEnable = root.querySelector('#rz-notif-enable');
    if (notifEnable) notifEnable.addEventListener('click', function() {
      Notification.requestPermission().then(function() { _notifDismissed = true; _render(); });
    });
    var notifDismiss = root.querySelector('#rz-notif-dismiss');
    if (notifDismiss) notifDismiss.addEventListener('click', function() {
      _notifDismissed = true;
      try { localStorage && localStorage.setItem('rz-notif-dismissed', '1'); } catch (_) {}
      _render();
    });

    var histBtn = root.querySelector('#rz-hist-btn');
    if (histBtn) histBtn.addEventListener('click', function() {
      _historyOpen = true;
      _render();
    });
    var histClose = document.getElementById('rz-hist-close');
    if (histClose) histClose.addEventListener('click', function() {
      _historyOpen = false;
      _render();
    });
    var histOverlay = document.getElementById('rz-hist-overlay');
    if (histOverlay) histOverlay.addEventListener('click', function(e) {
      if (e.target === histOverlay) { _historyOpen = false; _render(); }
    });
    var histClear = document.getElementById('rz-hist-clear');
    if (histClear) histClear.addEventListener('click', function() {
      _notifHistory = [];
      try { localStorage && localStorage.removeItem('rz-notif-history'); } catch (_) {}
      _historyOpen = false;
      _render();
    });

    _wireHeroCards();
    _wireHeroScroll();
    _wireGameStrip(true);
  }

  // ── Polling ───────────────────────────────────────────────────────────────────
  // Restore last-good data for the active scope after a failed scope-switch
  // fetch. Never clear the skeleton onto a foreign scope's payload (that is
  // how My Leagues ESPN names used to appear under This League).
  function _recoverScopeLoad(myGen, myScope) {
    if (myGen !== _streamGen || myScope !== _scope) return;
    _lastPollFailed = true;
    if (!_loadingScope) return;
    var cached = _scopeCache[myScope];
    if (cached) {
      _state = cached;
      _myRids = _myRidSet(cached);
      _loadingScope = false;
      _render();
    }
    // else keep the skeleton -- do not paint the other scope's state
  }

  async function _refresh() {
    if (_streaming) return; // a progressive My Leagues stream owns the screen
    // Capture at start so a late My Leagues poll cannot overwrite This League
    // (or vice versa) after the user flips the scope tabs.
    var myGen = _streamGen;
    var myScope = _scope;
    try {
      var parts = window.location.pathname.split('/');
      var apiBase = '/api/' + parts[1] + '/' + parts[2] + '/' + parts[3];
      var url = apiBase + '/redzone-data?_cb=' + Date.now() + '&scope=' + myScope;
      if (_isDemo) { _demoT += 15; url += '&demo=1&t=' + _demoT; }
      var resp = await fetch(url, { cache: 'no-store' });
      if (myGen !== _streamGen || myScope !== _scope) return;
      if (!resp.ok) {
        _recoverScopeLoad(myGen, myScope);
        if (myGen === _streamGen && myScope === _scope) _render();
        return;
      }
      var newData = await resp.json();
      if (myGen !== _streamGen || myScope !== _scope) return;
      // Server stamps scope; reject a mismatched payload even if gen lined up.
      if (newData && newData.scope && newData.scope !== myScope) return;
      _lastPollFailed = false;
      _loadingScope = false;
      _myRids = _myRidSet(newData);
      // Apply state before detect so owner/league labels read the new payload.
      _state = newData;
      _scopeCache[myScope] = newData;
      _detectChanges(newData);
      _seedPrevStats(newData);
      // After a scope switch the first fetch backfills history silently; arm
      // alerts so only subsequent live polls beep/notify.
      _alertsArmed = true;
      _applyDefaultHero(); // no-op: Plays start unfiltered; hero focus is opt-in
      _countdown = _pollInterval();

      _render();

      // Auto-refresh Live tab in player modal if it's currently visible
      var livePanelEl = document.getElementById('pm-panel-live');
      if (livePanelEl && livePanelEl.classList.contains('pm-panel-active') && window.__rzGetPlayerLive) {
        var pmBar = document.getElementById('pmTabBar');
        var pmPid = pmBar ? pmBar.dataset.pmPlayerId : null;
        if (pmPid) {
          livePanelEl.innerHTML = window.__rzGetPlayerLive(pmPid);
          window._rzSyncTabLive(livePanelEl);
        }
      }

      // Flash scores that changed
      if (_flashRids.size) {
        root.querySelectorAll('[data-score-rid]').forEach(function(el) {
          if (_flashRids.has(el.dataset.scoreRid)) {
            el.classList.remove('rz-score-flash');
            void el.offsetWidth; // reflow to restart animation
            el.classList.add('rz-score-flash');
          }
        });
        _flashRids.clear();
      }
    } catch (_) {
      _recoverScopeLoad(myGen, myScope);
      if (myGen === _streamGen && myScope === _scope) _render();
    }
  }

  // ── Progressive My Leagues load ────────────────────────────────────────────
  // Stream the viewer's leagues one card at a time (NDJSON from &stream=1) so
  // each populates as it lands instead of waiting for the whole portfolio.
  // Degrades to the aggregate _refresh() if streaming is unavailable, the
  // response isn't a stream (server fell back), or the stream errors midway.
  function _emptyUserState() {
    return {
      scope: 'user', week: _state.week, season: _state.season,
      platform: _state.platform, league_id: _state.league_id,
      games_today: _state.games_today,
      matchups: [], rosters: [], users: [], leagues: [],
      player_info: {}, scoring: {}, scoring_by_league: {}, pid_league: {},
      viewer_roster_id: '', viewer_roster_ids: []
    };
  }

  function _mergeLeagueSlice(base, s) {
    (s.matchups || []).forEach(function(m) { base.matchups.push(m); });
    (s.rosters  || []).forEach(function(r) { base.rosters.push(r); });
    (s.users    || []).forEach(function(u) {
      if (!base.users.some(function(x) { return x.user_id === u.user_id; })) base.users.push(u);
    });
    (s.leagues  || []).forEach(function(l) { base.leagues.push(l); });
    Object.assign(base.player_info, s.player_info || {});
    Object.assign(base.scoring_by_league, s.scoring_by_league || {});
    Object.assign(base.pid_league, s.pid_league || {});
    if (!base.pbp_by_game) base.pbp_by_game = {};
    Object.keys(s.pbp_by_game || {}).forEach(function(gid) {
      var arr = base.pbp_by_game[gid] || (base.pbp_by_game[gid] = []);
      (s.pbp_by_game[gid] || []).forEach(function(p) { arr.push(p); });
    });
    if (!base.scoring || !Object.keys(base.scoring).length) base.scoring = s.scoring || base.scoring;
    if (s.viewer_roster_id) {
      base.viewer_roster_ids.push(s.viewer_roster_id);
      if (!base.viewer_roster_id) base.viewer_roster_id = s.viewer_roster_id;
    }
  }

  async function _refreshUserStream() {
    var myGen = ++_streamGen; // this stream owns the screen until the next scope switch
    _streaming = true;
    _mlFailed = new Set();
    // Keep cached Plays on screen while streaming; rebuild from the full
    // portfolio once at stream end.
    var parts = window.location.pathname.split('/');
    var apiBase = '/api/' + parts[1] + '/' + parts[2] + '/' + parts[3];
    var url = apiBase + '/redzone-data?_cb=' + Date.now() + '&scope=user&stream=1';
    var resp;
    try { resp = await fetch(url, { cache: 'no-store' }); } catch (_) {
      _streaming = false;
      if (myGen !== _streamGen) return;
      return _refresh();
    }
    if (myGen !== _streamGen) { _streaming = false; return; } // superseded by a newer switch
    var ctype = (resp.headers.get('content-type') || '');
    // Server fell back to aggregate JSON (no portfolio / error) → use it directly.
    if (!resp.ok || !resp.body || ctype.indexOf('ndjson') < 0) {
      _streaming = false;
      if (myGen !== _streamGen) return;
      return _refresh();
    }

    var base = _emptyUserState();
    _mlNames = []; _mlLoaded = new Set();
    var reader = resp.body.getReader();
    var decoder = new TextDecoder();
    var buf = '', gotLeague = false;

    function _handle(obj) {
      if (myGen !== _streamGen) return; // a newer scope switch owns the screen now
      if (obj.type === 'meta') {
        base.week = obj.week; base.season = obj.season;
        if (obj.games_today != null) base.games_today = obj.games_today;
        _mlNames = (obj.leagues || []).map(function(l) { return l.name || 'League'; });
        _loadingScope = false;
        // Keep last-good portfolio cards on screen until the first slice lands.
        // Never write an empty shell into _scopeCache.user (that poisoned switches).
        if (!(_state.matchups && _state.matchups.length)) {
          _state = base; _myRids = new Set();
        }
        _render();
      } else if (obj.type === 'league') {
        if (!obj.empty) { _mergeLeagueSlice(base, obj); gotLeague = true; }
        else if (obj.index != null) { (_mlFailed = _mlFailed || new Set()).add(obj.index); }
        if (obj.index != null) _mlLoaded.add(obj.index);
        // If a prior portfolio is already on screen, keep it until stream end so
        // cards don't collapse to the first arriving league.
        var keepCache = _scopeCache.user && (_scopeCache.user.matchups || []).length;
        if (!keepCache) {
          _state = base;
          _myRids = _myRidSet(base);
          _seedMilestones(base); _seedInjuries(base); _seedLeaders(base);
          (base.matchups || []).forEach(function(m) { _prevMatchupPts[String(m.roster_id)] = parseFloat(m.points || 0); });
        }
        _render();
      }
    }

    try {
      while (true) {
        var chunk = await reader.read();
        if (myGen !== _streamGen) { try { reader.cancel(); } catch (_) {} _streaming = false; return; }
        if (chunk.done) break;
        buf += decoder.decode(chunk.value, { stream: true });
        var lines = buf.split('\n');
        buf = lines.pop();
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i].trim();
          if (!line) continue;
          var obj; try { obj = JSON.parse(line); } catch (_) { continue; }
          _handle(obj);
        }
      }
    } catch (_) {
      _mlNames = []; _mlLoaded = null; _streaming = false;
      if (myGen === _streamGen) return _refresh();
      return;
    }

    _mlLoaded = _mlLoaded; // keep for failed-card rendering until next switch
    _streaming = false;
    if (myGen !== _streamGen) return;
    if (!gotLeague) { return _refresh(); }
    _state = base;
    _scopeCache.user = base;
    _myRids = _myRidSet(base);
    // Rebuild Plays from the full portfolio (clear first to avoid duplicates
    // from the cached hydrate shown during the stream).
    _feed = [];
    _shownFeedIds = new Set();
    _resetFeedSnapshots();
    _hydrateFeed(base);
    _loadingScope = false;
    _countdown = _pollInterval();
    _render();
  }

  function _isGameDay() {
    if (_isDemo) return true;
    if (_anyLive()) return true; // already in progress -- always poll regardless of day/time
    // Server checks the week's schedule file for a game dated today.
    if (_state.games_today) return true;
    // Fallback: any player has a kickoff later today
    var now = Date.now() / 1000;
    var todayEnd = now - (now % 86400) + 86400; // midnight tonight UTC
    return Object.values(_state.player_info || {}).some(function(p) {
      var ep = parseFloat(p.game_time_epoch || 0);
      return ep > 0 && ep < todayEnd;
    });
  }

  function _nextGameEpoch() {
    var now = Date.now() / 1000;
    var earliest = Infinity;
    Object.values(_state.player_info || {}).forEach(function(p) {
      if (String(p.game_code || '0') !== '0') return;
      var ep = parseFloat(p.game_time_epoch || 0);
      if (ep > now && ep < earliest) earliest = ep;
    });
    return earliest === Infinity ? null : earliest;
  }

  function _pollInterval() {
    if (_isDemo) return 15;
    if (_anyLive()) return 15;
    var next = _nextGameEpoch();
    if (next) {
      var minsUntil = (next - Date.now() / 1000) / 60;
      if (minsUntil < 30) return 60;   // game kicking off soon
      if (minsUntil < 120) return 180; // within 2 hours → poll every 3 min
    }
    return 300; // no live games, nothing imminent → poll every 5 min
  }

  function _tick() {
    if (_streaming) return; // hold the countdown/poll while a stream is loading cards
    // Offseason / non-game day: stay idle -- no countdown, no polling, no live
    // look. Checked first so the timer never ticks down when nothing is on.
    if (!_isDemo && !_isGameDay()) {
      _countdown = 3600;
      var elIdle = document.getElementById('rz-timer');
      if (elIdle && elIdle.textContent !== '-') elIdle.textContent = '-';
      return;
    }
    _countdown--;
    var el = document.getElementById('rz-timer');
    if (el) el.textContent = _fmtTimer(_countdown);
    if (_countdown <= 0) {
      _countdown = _pollInterval();
      _refresh();
    }
  }

  document.addEventListener('click', function(e) {
    if (e.target && e.target.id === 'rz-timer') {
      var el = document.getElementById('rz-timer');
      if (el) { el.textContent = '↻'; el.classList.add('rz-timer-refreshing'); }
      _refresh();
    }
  });

  // Seed initial matchup points so first refresh doesn't trigger flash
  (_state.matchups || []).forEach(function(m) {
    _prevMatchupPts[String(m.roster_id)] = parseFloat(m.points || 0);
  });

  _loadPrefs();              // restore hero / My Team / Big Plays for this league+scope
  _seedMilestones(_state);   // pre-mark already-crossed milestones (no retroactive events)
  _seedInjuries(_state);     // snapshot injuries so only changes fire later
  _seedLeaders(_state);      // snapshot leading rosters so lead-change events don't fire on load
  _detectChanges(_state);    // populate initial feed from empty _prevStats
  _seedPrevStats(_state);    // snapshot stat lines for the next poll diff
  _alertsArmed = true;       // initial feed is backfill; only live polls alert after this
  _applyDefaultHero();       // no-op unless prefs restored a hero; focus stays opt-in by default

  _render();
  if (_isDemo) setTimeout(_refresh, 300);
  _timer = setInterval(_tick, 1000);
  // Re-evaluate hero-strip arrows when the viewport width changes.
  window.addEventListener('resize', _updateHeroArrows);
})();
