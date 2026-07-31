// BR Redzone live-scoreboard module — extracted from app.js so it only loads
// on the Redzone page (gated server-side via page_redzone). Runs deferred,
// after app.js, so the shared helpers it relies on (openPlayerModal,
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
  var _filters  = { team: 'all', nfl: 'all', pos: 'all', stat: 'all' };
  var _filterOpen  = false;
  var _myTeamOnly  = false;
  var _heroMid    = null;
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

  document.addEventListener('click', function() { _hadInteraction = true; }, { once: true });

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

  function _fmt(n) {
    if (n == null || n === '') return '0.0';
    return parseFloat(n).toFixed(1);
  }
  function _fmtTimer(n) {
    n = Math.max(0, Math.round(n));
    return n >= 120 ? Math.round(n / 60) + 'm' : n + 's';
  }
  function _name(pid) { return ((_state.player_info || {})[pid] || {}).name || pid; }
  function _pos(pid)  { return ((_state.player_info || {})[pid] || {}).pos  || ''; }
  function _team(pid) { return ((_state.player_info || {})[pid] || {}).team || ''; }
  function _statLine(pid) { return ((_state.player_info || {})[pid] || {}).stat_line || null; }

  function _gameStatus(pid) {
    var p = (_state.player_info || {})[pid] || {};
    var code = String(p.game_code || '');
    var st   = (p.game_status || '').toLowerCase();
    if (code === '2' || st.includes('final')) return { label: 'FINAL', type: 'final' };
    if (code === '1' || st.includes('progress') || st.includes('live')) return { label: 'LIVE', type: 'live' };
    return { label: p.team ? (p.game_status || '') : '', type: 'pre' };
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
  function _lineToPts(L, s) {
    if (!L) return 0;
    s = s || _state.scoring || {};
    return _n(L.pass_yds) * _n(s.pass_yd) + _n(L.pass_td) * _n(s.pass_td) + _n(L.int) * _n(s.pass_int)
      + _n(L.rush_yds) * _n(s.rush_yd) + _n(L.rush_td) * _n(s.rush_td)
      + _n(L.rec) * _n(s.rec) + _n(L.rec_yds) * _n(s.rec_yd) + _n(L.rec_td) * _n(s.rec_td);
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
    var pidToRoster = {};
    (data.matchups || []).forEach(function(m) {
      (m.players || []).forEach(function(pid) {
        // prefer a roster the viewer can see; first wins
        if (!(pid in pidToRoster)) pidToRoster[pid] = String(m.roster_id);
      });
    });
    return { my: myRosters, opp: oppRosters, pidToRoster: pidToRoster };
  }

  function _describe(d, pos) {
    // DEF: only sacks / INT / fumbles / defensive TDs
    if (pos === 'DEF') {
      var segs = [];
      var st = [];
      if (d.sacks   > 0) { segs.push(d.sacks   === 1 ? '1 sack'   : d.sacks   + ' sacks');   st.push('sack'); }
      if (d.def_int > 0) { segs.push(d.def_int  === 1 ? '1 INT'    : d.def_int  + ' INTs');   st.push('int'); }
      if (d.fum_rec > 0) { segs.push(d.fum_rec  === 1 ? '1 fumble' : d.fum_rec  + ' fumbles'); st.push('fumble'); }
      if (d.def_td  > 0) { segs.push(d.def_td   === 1 ? '1 DEF TD' : d.def_td   + ' DEF TDs'); st.push('td'); }
      if (!segs.length) return null;
      return { desc: segs.join(' · '), kind: d.def_td > 0 ? 'td' : 'gain', stats: st };
    }

    // K: FG or PAT
    if (pos === 'K') {
      if (d.fgm > 0) {
        var dist = Math.round(d.fg_long || 0);
        return { desc: (dist > 0 ? dist + ' yd FG' : 'FG'), kind: 'gain', stats: ['kick'] };
      }
      if (d.xpm > 0) return { desc: 'PAT', kind: 'gain', stats: ['kick'] };
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

    if (tdc === 1 && d.rec_td === 1 && d.rec === 1 && d.carries < 1) return { desc: ry + ' yd TD catch', kind: kind, stats: stats };
    if (tdc === 1 && d.rush_td === 1 && d.carries === 1 && d.rec < 1) return { desc: uy + ' yd TD run', kind: kind, stats: stats };
    if (tdc === 1 && d.pass_td === 1 && d.rec < 1 && d.carries < 1) return { desc: py + ' yd TD pass', kind: kind, stats: stats };

    var segs = [];
    if (d.rec >= 1)     segs.push(d.rec === 1 ? ('1 catch ' + ry + ' yds') : (d.rec + ' catches ' + ry + ' yds'));
    if (d.carries >= 1) segs.push(d.carries === 1 ? (uy + ' yd carry') : (d.carries + ' carries ' + uy + ' yds'));
    if (d.pass_yds > 0) segs.push(py + ' pass yds');
    if (!segs.length && d.targets > 0) segs.push('Targeted (incomplete)');
    var tail = '';
    if (tdc > 0) tail += ' · ' + tdc + ' TD' + (tdc > 1 ? 's' : '');
    if (d.int > 0) tail += ' · INT';
    return { desc: (segs.join(', ') || 'Active') + tail, kind: kind, stats: stats };
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
      return Object.assign({}, base, { desc: info.desc, kind: info.kind, stats: info.stats, pts: earned, ts: Date.now() + Math.random() });
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

  function _detectChanges(newData) {
    var tags = _rosterTags(newData);
    // Resolve scoring per player by league (user scope spans multiple leagues);
    // fall back to the single top-level scoring.
    var _sbl = newData.scoring_by_league, _pidLg = newData.pid_league || {};
    var _scFor = function(pid) {
      if (_sbl) { var lid = _pidLg[pid]; if (lid && _sbl[lid]) return _sbl[lid]; }
      return newData.scoring || {};
    };
    var allEvents = [], handled = {};
    Object.keys(newData.player_info || {}).forEach(function(pid) {
      var newL = newData.player_info[pid].stat_line;
      if (!newL) return;
      handled[pid] = true;
      var evs = _playsFromDiff(pid, _prevStats[pid] || {}, newL, tags, _scFor(pid));
      evs.forEach(function(ev) { allEvents.push(ev); });
    });
    (newData.matchups || []).forEach(function(m) {
      var pp = m.players_points || {};
      Object.keys(pp).forEach(function(pid) {
        if (handled[pid] || pid === '0') return;
        var delta = parseFloat((parseFloat(pp[pid] || 0) - parseFloat(_prevPts[pid] || 0)).toFixed(2));
        if (delta <= 0.05) return;
        var rid = tags.pidToRoster[pid] || String(m.roster_id);
        var info = newData.player_info[pid] || {};
        allEvents.push({
          pid: pid, name: info.name || pid, pos: info.pos || '', nflTeam: info.team || '',
          rosterId: rid, owner: _ownerName(rid), league: _leagueOfRid(rid),
          desc: 'Scored ' + delta.toFixed(1) + ' pts', kind: 'gain', stats: ['pts'], pts: delta,
          mine: tags.my.has(rid), opp: tags.opp.has(rid), line: '', ts: Date.now()
        });
      });
    });
    // TDs first, then chronological
    allEvents.sort(function(a, b) { return (b.kind === 'td') - (a.kind === 'td'); });
    allEvents.forEach(function(ev) { _feed.unshift(ev); });
    if (_feed.length > 200) _feed = _feed.slice(0, 200);

    // Push notification + audio chime for my TDs + log to history
    var myTDs = allEvents.filter(function(ev) { return ev.kind === 'td' && ev.mine; });
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
                body: ev.desc + (ev.pts > 0 ? '  +' + _fmt(ev.pts) + ' pts' : ''),
                icon: '/static/BR_Logo.png', tag: 'rz-td-' + ev.pid
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

    // Track which rosters had point changes (for score flash) + capture score delta
    _scoreDelta = { me: 0, opp: 0 };
    var _myMidSet = new Set();
    (newData.matchups || []).forEach(function(m) {
      if (_isMyRid(m.roster_id)) _myMidSet.add(String(m.matchup_id));
    });
    (newData.matchups || []).forEach(function(m) {
      var rid = String(m.roster_id);
      var newPts = parseFloat(m.points || 0);
      var oldPts = _prevMatchupPts[rid];
      if (oldPts !== undefined && Math.abs(newPts - oldPts) > 0.01) _flashRids.add(rid);
      if (oldPts !== undefined && _myMidSet.has(String(m.matchup_id))) {
        var delta = parseFloat((newPts - oldPts).toFixed(1));
        if (_isMyRid(rid)) _scoreDelta.me += delta;
        else _scoreDelta.opp += delta;
      }
      _prevMatchupPts[rid] = newPts;
    });
  }

  // ── Filters ────────────────────────────────────────────────────────────────────
  function _teamOptions() {
    if (_scope === 'user') {
      return (_state.leagues || []).map(function(l) { return l.name; }).filter(Boolean);
    }
    var seen = {};
    (_state.rosters || []).forEach(function(r) {
      var nm = _ownerName(r.roster_id);
      if (nm) seen[nm] = 1;
    });
    return Object.keys(seen);
  }
  function _nflOptions() {
    var seen = {};
    (_state.matchups || []).forEach(function(m) {
      (m.players || []).forEach(function(pid) {
        var t = _team(pid);
        if (t) seen[t] = 1;
      });
    });
    return Object.keys(seen).sort();
  }
  var _POS_LIST  = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF'];
  var _STAT_LIST = [['td','TD'], ['reception','Reception'], ['carry','Carry'],
                    ['pass','Pass'], ['target','Target'], ['int','INT'], ['milestone','Milestone'], ['lead_change','Lead']];

  function _passTeam(ev) {
    if (_filters.team === 'all') return true;
    return (_scope === 'user' ? ev.league : ev.owner) === _filters.team;
  }
  function _eventMatches(ev) {
    if (_myTeamOnly && !ev.mine) return false;
    if (!_passTeam(ev)) return false;
    if (_filters.nfl !== 'all' && ev.nflTeam !== _filters.nfl) return false;
    if (_filters.pos !== 'all' && ev.pos !== _filters.pos) return false;
    if (_filters.stat !== 'all' && (ev.stats || []).indexOf(_filters.stat) < 0) return false;
    if (_heroMid) { var hp = _heroMatchupPids(); if (hp && !hp.has(ev.pid)) return false; }
    return true;
  }
  function _topMatches(pid, rid) {
    if (_filters.team !== 'all') {
      var key = _scope === 'user' ? _leagueOfRid(rid) : _ownerName(rid);
      if (key !== _filters.team) return false;
    }
    if (_filters.nfl !== 'all' && _team(pid) !== _filters.nfl) return false;
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

  function _renderFilterChips() {
    var activeCount = ['team','nfl','pos','stat'].filter(function(k) { return _filters[k] !== 'all'; }).length;
    var chips = '';
    if (_heroMid) chips += '<span class="rz-active-chip rz-hero-chip" data-clear-hero="1">&#9654; ' + _heroLabel() + ' ×</span>';
    if (_filters.team !== 'all') chips += '<span class="rz-active-chip" data-clear="team">' + _filters.team + ' ×</span>';
    if (_filters.nfl  !== 'all') chips += '<span class="rz-active-chip" data-clear="nfl">'  + _filters.nfl  + ' ×</span>';
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
          return '<button class="rz-fp-opt' + (_filters[key] === val ? ' active' : '') + '" data-fk="' + key + '" data-fv="' + val + '">' + lbl + '</button>';
        }).join('');
        return '<div class="rz-fp-row"><span class="rz-fp-label">' + label + '</span><div class="rz-fp-opts">' + btns + '</div></div>';
      }
      var tOpts = [['all','All']].concat(_teamOptions().map(function(t) { return [t, t.length > 12 ? t.slice(0,11) + '…' : t]; }));
      var nOpts = [['all','All']].concat(_nflOptions().map(function(t) { return [t, t]; }));
      var pOpts = [['all','All']].concat(_POS_LIST.map(function(p) { return [p, p]; }));
      var sOpts = [['all','All']].concat(_STAT_LIST);
      panel = '<div class="rz-filter-panel">'
        + fpRow('Team', 'team', tOpts)
        + fpRow('NFL',  'nfl',  nOpts)
        + fpRow('Pos',  'pos',  pOpts)
        + fpRow('Type', 'stat', sOpts)
        + '</div>';
    }
    var myTeamBtn = _myRids.size
      ? '<button class="rz-myteam-btn' + (_myTeamOnly ? ' active' : '') + '" id="rz-myteam-btn">My Team</button>'
      : '';
    var histCount = _notifHistory.length;
    var histBtn = histCount > 0
      ? '<button class="rz-hist-btn" id="rz-hist-btn">Alerts <span class="rz-hist-count">' + histCount + '</span></button>'
      : '';
    return '<div class="rz-chip-bar">'
      + myTeamBtn
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
    var pp = matchup.players_points || {}, starters = matchup.starters || [];
    var bench = (matchup.players || []).filter(function(pid) { return pid !== '0' && !starters.includes(pid); });
    var rows = starters.map(function(pid) {
      if (pid === '0') return '<div class="rz-player-row"><span class="rz-pos-badge rz-pos-" style="opacity:.25"></span><div class="rz-player-info"><div class="rz-player-name" style="color:var(--rz-muted)">Empty slot</div></div><div class="rz-player-pts">0</div></div>';
      return _playerRowHtml(pid, pp[pid], false);
    }).join('');
    var benchRows = bench.slice(0, 6).map(function(pid) { return _playerRowHtml(pid, pp[pid], true); }).join('');
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

  function _playersLeft(matchup) {
    if (!matchup) return 0;
    return (matchup.starters || []).filter(function(pid) {
      if (pid === '0') return false;
      var code = ((_state.player_info || {})[pid] || {}).game_code || '0';
      return code !== '2'; // not final
    }).length;
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
      var anyLive = false;
      [a, b].forEach(function(m) { (m.starters || []).forEach(function(pid) { if (_gameStatus(pid).type === 'live') anyLive = true; }); });
      var isClose = anyLive && Math.abs(ptsA - ptsB) < 5;
      rows += (
        '<div class="rz-lb-row' + (isClose ? ' close' : '') + '">'
        + '<div class="rz-lb-team' + (aLead ? ' lead' : '') + '">'
        +   '<span class="rz-lb-name">' + (_ownerName(a.roster_id) || 'Team') + '</span>'
        +   '<span class="rz-lb-score">' + _fmt(ptsA) + '</span>'
        + '</div>'
        + '<div class="rz-lb-mid">' + (anyLive ? '<span class="rz-lb-live">LIVE</span>' : '<span class="rz-lb-final">FINAL</span>') + '</div>'
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

  function _renderHeroCards() {
    // Score delta badge ("+N this update"): reflects the change from the most
    // recent poll. _detectChanges recomputes (and resets) _scoreDelta every
    // poll, so the badge updates each cycle and clears when nothing changed —
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
      if (!mine.length) {
        return '<div class="rz-no-matchup">No leagues found for your account.<br><a href="?demo=1" class="rz-demo-link">View Demo</a></div>';
      }
      var cards = mine.map(function(m) {
        var opp = _oppOf(m);
        var myPts = parseFloat(m.points || 0), oppPts = parseFloat(opp ? opp.points || 0 : 0);
        var win = myPts >= oppPts;
        var anyLive = false, anyFinal = false;
        [m, opp].forEach(function(r) {
          if (!r) return;
          (r.starters || []).forEach(function(pid) {
            var gs = _gameStatus(pid);
            if (gs.type === 'live') anyLive = true;
            if (gs.type === 'final') anyFinal = true;
          });
        });
        var rid = String(m.roster_id);
        var selected = _heroMid === rid;
        var badge = anyLive ? '<span class="rz-mch-live">LIVE</span>' : anyFinal ? '<span class="rz-mch-final">FINAL</span>' : '<span class="rz-mch-pre">PRE</span>';
        var oppName = opp ? (_ownerName(opp.roster_id) || 'Opp') : 'Opp';
        return '<div class="rz-mc-hero' + (selected ? ' selected' : '') + (anyLive ? ' is-live' : '') + '" data-heromid="' + rid + '">'
          + '<div class="rz-mch-league">' + (m.league_name || 'League') + '</div>'
          + '<div class="rz-mch-matchup">'
          +   '<div class="rz-mch-side">'
          +     '<div class="rz-mch-owner viewer">Me</div>'
          +     '<div class="rz-mch-score' + (win ? ' lead' : '') + '" data-score-rid="' + String(m.roster_id) + '">' + _fmt(myPts) + '</div>'
          +   '</div>'
          +   '<div class="rz-mch-vs">' + badge + '</div>'
          +   '<div class="rz-mch-side right">'
          +     '<div class="rz-mch-owner">' + oppName + '</div>'
          +     '<div class="rz-mch-score' + (!win ? ' lead' : '') + '" data-score-rid="' + (opp ? String(opp.roster_id) : '') + '">' + _fmt(oppPts) + '</div>'
          +   '</div>'
          + '</div>'
          + '</div>';
      }).join('');
      return '<div class="rz-hero-cards">' + _deltaHtml + '<div class="rz-hero-cards-row">' + cards + '</div></div>';
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
      var anyLive = false, anyFinal = false;
      [a, b].forEach(function(m) { (m.starters || []).forEach(function(pid) {
        var gs = _gameStatus(pid);
        if (gs.type === 'live') anyLive = true;
        if (gs.type === 'final') anyFinal = true;
      }); });
      var selected = _heroMid === mid;
      var isViewer = mid === myMid;
      var badge = anyLive ? '<span class="rz-mch-live">LIVE</span>' : anyFinal ? '<span class="rz-mch-final">FINAL</span>' : '<span class="rz-mch-pre">PRE</span>';
      var nameA = _ownerName(a.roster_id) || 'Team';
      var nameB = _ownerName(b.roster_id) || 'Team';
      var vA = _isMyRid(a.roster_id), vB = _isMyRid(b.roster_id);
      return '<div class="rz-mc-hero' + (selected ? ' selected' : '') + (anyLive ? ' is-live' : '') + (isViewer ? ' viewer-matchup' : '') + '" data-heromid="' + mid + '">'
        + '<div class="rz-mch-matchup">'
        +   '<div class="rz-mch-side">'
        +     '<div class="rz-mch-owner' + (vA ? ' viewer' : '') + '">' + nameA + '</div>'
        +     '<div class="rz-mch-score' + (aLead ? ' lead' : '') + '" data-score-rid="' + String(a.roster_id) + '">' + _fmt(ptsA) + '</div>'
        +   '</div>'
        +   '<div class="rz-mch-vs">' + badge + '</div>'
        +   '<div class="rz-mch-side right">'
        +     '<div class="rz-mch-owner' + (vB ? ' viewer' : '') + '">' + nameB + '</div>'
        +     '<div class="rz-mch-score' + (!aLead ? ' lead' : '') + '" data-score-rid="' + String(b.roster_id) + '">' + _fmt(ptsB) + '</div>'
        +   '</div>'
        + '</div>'
        + '</div>';
    }).filter(Boolean).join('');
    return '<div class="rz-hero-cards">' + _deltaHtml + '<div class="rz-hero-cards-row">' + cards2 + '</div></div>';
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

  function _renderMyTeams() {
    var mine = _myMatchups();
    if (!mine.length) return '<div class="rz-feed-empty">No teams found.</div>';
    return mine.map(function(m) {
      return '<div class="rz-section-label" style="opacity:1;color:var(--rz-text);font-size:11px;">' + (m.league_name || 'League') + '</div>' + _rosterCard(m);
    }).join('');
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
      var anyLive = false, anyFinal = false;
      [a, b].forEach(function(m) { (m.starters || []).forEach(function(pid) { var gs = _gameStatus(pid); if (gs.type === 'live') anyLive = true; if (gs.type === 'final') anyFinal = true; }); });
      var isClose = anyLive && Math.abs(ptsA - ptsB) < 5;
      var cls = anyLive ? 'live' : anyFinal ? 'final' : 'pre';
      var statusLabel = anyLive ? 'LIVE' : anyFinal ? 'FINAL' : '';
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
      var pp = m.players_points || {};
      Object.keys(pp).forEach(function(pid) {
        if (pid === '0') return;
        var pts = parseFloat(pp[pid] || 0);
        if (!pidMap[pid] || pts > pidMap[pid].pts) pidMap[pid] = { pts: pts, roster_id: m.roster_id };
      });
    });
    // Filtered map (respects active team/pos/owner filters) — used for both
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

  function _eid(ev) { return ev.pid + ':' + (ev.ts || ev.desc); }

  function _eventHtml(ev, animate) {
    var tagLabel = ev.mine ? (_scope === 'user' && ev.league ? ev.league : 'MY TEAM')
                 : ev.opp  ? 'OPP'
                 : (_scope === 'user' && ev.league ? ev.league : '');
    var tagCls = ev.mine ? 'mine' : 'opp';
    var tag = tagLabel ? '<span class="rz-event-tag ' + tagCls + '">' + tagLabel + '</span>' : '';
    var ptStr = ev.pts > 0 ? '+' + _fmt(ev.pts) : _fmt(ev.pts);
    var posKey = (ev.pos || 'x').toLowerCase().replace(/[^a-z]/g, '');
    var initials = (ev.name || '?').trim().split(/\s+/).map(function(w) { return w[0] || ''; }).join('').slice(0, 2).toUpperCase();
    // Game score line (already includes both teams) — bold the player's own
    // team within it instead of a separate, duplicated team prefix.
    var subInner = ev.line || ev.nflTeam || '';
    if (ev.line && ev.nflTeam) {
      var _tm = String(ev.nflTeam).replace(/[^A-Za-z0-9]/g, '');
      if (_tm) subInner = ev.line.replace(new RegExp('\\b' + _tm + '\\b'),
        '<strong class="rz-event-myteam">' + ev.nflTeam + '</strong>');
    }
    var clockStr = [ev.gameQuarter, ev.gameClock].filter(Boolean).join(' ');
    var clockHtml = clockStr ? '<span class="rz-event-clock">' + clockStr + '</span>' : '';
    var subRow = (subInner || clockHtml)
      ? '<div class="rz-event-sub">' + subInner + (subInner && clockHtml ? '  ·  ' : '') + clockHtml + '</div>'
      : '';
    return (
      '<div class="rz-event ' + ev.kind + (ev.mine ? ' is-mine' : '') + (animate ? '' : ' rz-event-old') + '" data-pid="' + ev.pid + '">'
      + '<div class="rz-event-avatar rz-av-' + posKey + '" data-init="' + initials + '">'
      + '<img class="rz-headshot" src="https://sleepercdn.com/content/nfl/players/thumb/' + ev.pid + '.jpg" alt="" onerror="this.parentNode.classList.add(\'img-err\')">'
      + '</div>'
      + '<div class="rz-event-body">'
      + '<div class="rz-event-main"><span class="rz-event-name">' + ev.name + '</span>' + tag + '</div>'
      + '<div class="rz-event-desc">' + ev.desc + '</div>'
      + subRow
      + '</div>'
      + '<div class="rz-event-delta ' + (ev.pts >= 0 ? 'pos' : 'neg') + '">' + ptStr + '</div>'
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
    var myPids = new Set();
    _myMatchups().forEach(function(m) {
      (m.players || []).forEach(function(pid) { myPids.add(pid); });
    });

    // Group scheduled/live players by game_id
    var gameMap = {};
    Object.keys(info).forEach(function(pid) {
      var p = info[pid];
      var gid = p.game_id || '';
      if (!gid || !p.home || !p.away) return;
      var code = String(p.game_code || '0');
      if (code === '2') return; // skip final games
      if (!gameMap[gid]) gameMap[gid] = { home: p.home, away: p.away, status: p.game_status || '', code: code, mine: [], other: [] };
      if (myPids.has(pid)) gameMap[gid].mine.push(p.name || pid);
      else gameMap[gid].other.push(pid);
    });

    var gameIds = Object.keys(gameMap);
    if (!gameIds.length) {
      return '<div class="rz-pregame-empty-hint">Plays appear here as games unfold, targets, catches, carries and touchdowns with live fantasy points.</div>';
    }

    // Sort: games with my players first
    gameIds.sort(function(a, b) {
      return (gameMap[b].mine.length > 0 ? 1 : 0) - (gameMap[a].mine.length > 0 ? 1 : 0);
    });

    var hasMyGames = gameIds.some(function(gid) { return gameMap[gid].mine.length > 0; });
    var html = '<div class="rz-pregame-wrap">'
      + '<div class="rz-pregame-label">' + (gameIds.some(function(g) { return gameMap[g].code === '1'; }) ? 'Games in Progress' : 'Upcoming Games') + '</div>';

    gameIds.forEach(function(gid) {
      var g = gameMap[gid];
      var statusText = g.code === '1' ? 'LIVE · ' + (g.status || '') : (g.status || 'Upcoming');
      var playerChip = '';
      if (g.mine.length) {
        var names = g.mine.slice(0, 3).join(', ') + (g.mine.length > 3 ? ' +' + (g.mine.length - 3) : '');
        playerChip = '<div class="rz-pregame-players"><strong>My Players</strong>' + names + '</div>';
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
    var container = document.getElementById('rz-feed-list');
    if (!container) return;

    var list = _feed.filter(_eventMatches);
    var anyFilter = _filters.team !== 'all' || _filters.nfl !== 'all' || _filters.pos !== 'all' || _filters.stat !== 'all' || !!_heroMid;
    var totalPages = Math.max(1, Math.ceil(list.length / _PAGE_SIZE));
    if (_feedPage >= totalPages) _feedPage = totalPages - 1;

    if (!list.length) {
      if (anyFilter || _myTeamOnly) {
        container.innerHTML = '<div class="rz-feed-empty">No plays match these filters yet.</div>';
      } else {
        container.innerHTML = _pregameScheduleHtml();
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
      container.querySelectorAll('[data-pid]').forEach(function(el) {
        if (!el.dataset.pid || el.dataset.pid === '0') return;
        el.onclick = function() { openPlayerModal(el.dataset.pid, _name(el.dataset.pid), { tab: 'live' }); };
      });
      return;
    }

    // Page 0: live DOM-patching + FLIP
    var page0Items = list.slice(0, _PAGE_SIZE);
    var page0Eids = new Set(page0Items.map(function(ev) { return _eid(ev); }));

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
              n.querySelectorAll('[data-pid]').forEach(function(el) {
                if (!el.dataset.pid || el.dataset.pid === '0') return;
                el.onclick = function() { openPlayerModal(el.dataset.pid, _name(el.dataset.pid), { tab: 'live' }); };
              });
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

    // Prune to page size
    var items = container.querySelectorAll('[data-eid]');
    for (var i = _PAGE_SIZE; i < items.length; i++) items[i].remove();

    _renderPagination(totalPages);

    // Live feed header
    var hdr = document.getElementById('rz-feed-hdr');
    if (hdr) {
      var totalEvts = list.length;
      var liveNow = _anyLive();
      hdr.innerHTML = totalEvts
        ? (liveNow
          ? '<span class="rz-fh-dot"></span><span class="rz-fh-text">Live · <b>' + totalEvts + '</b> ' + (totalEvts === 1 ? 'play' : 'plays') + '</span>'
          : '<span class="rz-fh-text"><b>' + totalEvts + '</b> ' + (totalEvts === 1 ? 'play' : 'plays') + ' · Final</span>')
        : '';
    }

    container.querySelectorAll('[data-pid]').forEach(function(el) {
      if (el.classList.contains('rz-player-pts')) return;
      if (!el.dataset.pid || el.dataset.pid === '0') return;
      el.onclick = function() { openPlayerModal(el.dataset.pid, _name(el.dataset.pid), { tab: 'live' }); };
    });
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

  function _wireHeroCards() {
    root.querySelectorAll('[data-heromid]').forEach(function(el) {
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
        _feedPage = 0;
        _render();
      });
    });
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
    // Update timer text
    var timerEl = document.getElementById('rz-timer');
    if (timerEl) { timerEl.textContent = _fmtTimer(_countdown); timerEl.classList.remove('rz-timer-refreshing'); }

    // Update live chip in header
    var liveChipEl = root.querySelector('.rz-live-chip');
    var headerRight = root.querySelector('.rz-header-right');
    if (headerRight) {
      var live = _anyLive();
      var liveChipHtml = live ? '<span class="rz-live-chip"><span class="rz-nav-dot"></span>LIVE</span>' : '';
      var demoLink = !_isDemo ? '<a href="?demo=1" class="rz-demo-btn">Demo</a>' : '';
      headerRight.innerHTML = demoLink + liveChipHtml + '<button class="rz-refresh-timer" id="rz-timer">' + _fmtTimer(_countdown) + '</button>';
    }

    // Replace hero cards in-place and re-wire
    var heroWrap = root.querySelector('.rz-hero-cards, .rz-no-matchup');
    if (heroWrap) {
      var tempDiv = document.createElement('div');
      tempDiv.innerHTML = _renderHeroCards();
      var newHero = tempDiv.firstChild;
      if (newHero) heroWrap.parentNode.replaceChild(newHero, heroWrap);
    }
    _wireHeroCards();

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
        myTeamToggle2.addEventListener('click', function() { _myTeamOnly = !_myTeamOnly; _feedPage = 0; _render(); });
      }
      root.querySelectorAll('[data-clear]').forEach(function(btn) {
        btn.addEventListener('click', function() { _filters[btn.dataset.clear] = 'all'; _feedPage = 0; _render(); });
      });
      root.querySelectorAll('[data-clear-hero]').forEach(function(el) {
        el.addEventListener('click', function() { _heroMid = null; _feedPage = 0; _render(); });
      });
      root.querySelectorAll('[data-fk]').forEach(function(btn) {
        btn.addEventListener('click', function() { _filters[btn.dataset.fk] = btn.dataset.fv; _filterOpen = false; _feedPage = 0; _render(); });
      });
      var filterBtn = root.querySelector('#rz-filter-btn');
      if (filterBtn) filterBtn.addEventListener('click', function() { _filterOpen = !_filterOpen; _render(); });
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
    var TABS = _tabsFor();
    if (!TABS.some(function(t) { return t.key === _activeTab; })) _activeTab = 'plays';

    var live = _anyLive();
    var idle = !_isDemo && !live && !_isGameDay();  // offseason / no games today
    var liveChip = live ? '<span class="rz-live-chip"><span class="rz-nav-dot"></span>LIVE</span>' : '';
    var demoPill = _isDemo ? '<span class="rz-demo-pill">DEMO</span>' : '';
    var showFilters = (_activeTab === 'plays' || _activeTab === 'top');

    var mine = _myMatchups();
    var myMatchup = mine[0], oppMatchup = myMatchup ? _oppOf(myMatchup) : null;

    var summary = _renderHeroCards();

    var tabBar = '<div class="rz-tab-bar">' + TABS.map(function(t) {
      var badge = (t.key === 'plays' && _unreadCount > 0 && _activeTab !== 'plays')
        ? '<span class="rz-tab-badge">' + (_unreadCount > 99 ? '99+' : _unreadCount) + '</span>' : '';
      return '<button class="rz-tab-btn' + (_activeTab === t.key ? ' active' : '') + '" data-tab="' + t.key + '">' + t.label + badge + '</button>';
    }).join('') + '</div>';

    var minePanel = _scope === 'user' ? _renderMyTeams() : _rosterCard(myMatchup);

    // Pinned score bar for the Plays tab
    var playsScoreBar = '';
    if (myMatchup) {
      var _mm = myMatchup, _om = oppMatchup;
      var _myP = parseFloat(_mm.points || 0), _opP = parseFloat(_om ? _om.points || 0 : 0);
      var _win = _myP >= _opP, _diff = Math.abs(_myP - _opP).toFixed(1);
      var _liveBar = _anyLive();
      var _oppName = _om ? (_ownerName(_om.roster_id) || 'Opp') : 'Opp';
      playsScoreBar = '<div class="rz-plays-scorebar">'
        + '<span class="rz-psb-me' + (_win ? ' lead' : '') + '">Me  ' + _fmt(_myP) + '</span>'
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

    var demoLink = !_isDemo ? '<a href="?demo=1" class="rz-demo-btn">Demo</a>' : '';
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
      + '<div class="rz-header-right">' + demoLink + exitBtn + staleChip + liveChip + '<button class="rz-refresh-timer" id="rz-timer">' + timerLabel + '</button></div>'
      + '</div>'
      + '<div class="rz-content">'
      + _renderScopeToggle()
      + summary
      + _onDeckHtml()
      + '<div class="rz-main-card">'
      + tabBar
      + (showFilters ? _renderFilterChips() : '')
      + panels
      + '</div>'
      + '</div>'
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
        _feedPage = 0;
        _render();
      });
    }

    root.querySelectorAll('.rz-scope-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        if (btn.dataset.scope === _scope) return;
        _scope = btn.dataset.scope;
        _filters = { team: 'all', nfl: 'all', pos: 'all', stat: 'all' };
        _feed = [];
        _shownFeedIds = new Set();
        _filterOpen = false;
        _myTeamOnly = false;
        _heroMid = null;
        _feedPage = 0;
        _countdown = 1;
        _render();
        _refresh();
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
      el.addEventListener('click', function() { _heroMid = null; _feedPage = 0; _render(); });
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
    root.querySelectorAll('[data-pid]').forEach(function(el) {
      if (el.classList.contains('rz-player-pts')) return;
      // Feed events are wired by _syncFeed (el.onclick) — skip them here so a
      // click doesn't fire openPlayerModal twice (two stacked modals).
      if (el.classList.contains('rz-event')) return;
      if (!el.dataset.pid || el.dataset.pid === '0') return;
      el.addEventListener('click', function() { openPlayerModal(el.dataset.pid, _name(el.dataset.pid), { tab: 'live' }); });
    });
  }

  // ── Polling ───────────────────────────────────────────────────────────────────
  async function _refresh() {
    try {
      var parts = window.location.pathname.split('/');
      var apiBase = '/api/' + parts[1] + '/' + parts[2] + '/' + parts[3];
      var url = apiBase + '/redzone-data?_cb=' + Date.now() + '&scope=' + _scope;
      if (_isDemo) { _demoT += 15; url += '&demo=1&t=' + _demoT; }
      var resp = await fetch(url);
      if (!resp.ok) { _lastPollFailed = true; return; }
      var newData = await resp.json();
      _lastPollFailed = false;
      _myRids = _myRidSet(newData);
      _detectChanges(newData);
      _state = newData;
      _seedPrevStats(newData);
      _countdown = _pollInterval();

      var savedFeedHtml = null;
      var oldFeedEl = root.querySelector('#rz-feed-list');
      if (oldFeedEl && oldFeedEl.children.length > 0) savedFeedHtml = oldFeedEl.innerHTML;

      _render();

      if (savedFeedHtml !== null) {
        var newFeedEl = root.querySelector('#rz-feed-list');
        if (newFeedEl) { newFeedEl.innerHTML = savedFeedHtml; _syncFeed(); }
      }

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
    } catch (_) { _lastPollFailed = true; }
  }

  function _isGameDay() {
    if (_isDemo) return true;
    if (_anyLive()) return true; // already in progress — always poll regardless of day/time
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
    // Offseason / non-game day: stay idle — no countdown, no polling, no live
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

  _seedMilestones(_state);   // pre-mark already-crossed milestones (no retroactive events)
  _seedInjuries(_state);     // snapshot injuries so only changes fire later
  _seedLeaders(_state);      // snapshot leading rosters so lead-change events don't fire on load
  _detectChanges(_state);    // populate initial feed from empty _prevStats
  _seedPrevStats(_state);    // snapshot stat lines for the next poll diff

  _render();
  if (_isDemo) setTimeout(_refresh, 300);
  _timer = setInterval(_tick, 1000);
})();
