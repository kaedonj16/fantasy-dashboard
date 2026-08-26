"""Schedule assistant HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_schedule_body(ctx):
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _team_bye_map, datetime, get_players_index_global, logger,
    )

    import json as _json
    season = int(ctx.get("season") or datetime.now().year)
    league_id = ctx.get("league_id") or ""
    platform = ctx.get("platform") or "sleeper"
    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    rosters = ctx.get("rosters") or []
    players_idx = get_players_index_global() or {}

    current_week = int(ctx.get("current_week") or 0)
    max_week = 18
    start_week = current_week if current_week >= 1 else 1
    def_start = start_week
    def_end = min(start_week + 3, max_week)

    viewer_roster = next((r for r in rosters if str(r.get("roster_id")) == viewer_rid), None)
    roster_pids = [str(p) for p in (viewer_roster.get("players") if viewer_roster else []) or []]
    roster_positions = ctx.get("roster_positions") or []
    _idp_slots = any(
        str(p).upper() in ("DL", "LB", "DB", "IDP", "IDP_FLEX", "DE", "DT", "CB", "S")
        for p in roster_positions
    )
    _skill_pos = {"QB", "RB", "WR", "TE", "K", "DEF"}
    if _idp_slots:
        _skill_pos |= {"DL", "LB", "DB", "IDP", "DE", "DT", "CB", "S"}
    init_pids = [
        pid for pid in roster_pids
        if ((players_idx.get(pid) or {}).get("pos") or "").upper() in _skill_pos
    ]

    cfg = _json.dumps({
        "season": season, "leagueId": league_id, "platform": platform,
        "startWeek": start_week, "maxWeek": max_week,
        "defStart": def_start, "defEnd": def_end,
        "initPids": init_pids,
    })

    shell = """
    <div class="card central schedule-card">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
          <h2>Schedule Assistant</h2>
          <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
            Matchup difficulty by week. Ratings rebuild on Wednesdays in-season. Add or remove players and pick a single week or a range.
          </div>
        </div>
        <div class="sched-view-toggle" id="schedViewToggle">
          <button class="sched-view-btn active" data-view="my-players">My Players</button>
          <button class="sched-view-btn" data-view="rankings">Schedule Rankings</button>
        </div>
      </div>

      __BYE_BANNER__

      <!-- Shared controls: week range always; player search vs. rank controls flip per tab -->
      <div class="sched-controls sched-shared-controls">
        <div class="sched-week-range">
          <span class="sched-ctrl-label">Weeks</span>
          <select id="schedWkStart" class="sched-select"></select>
          <span class="sched-ctrl-sep">to</span>
          <select id="schedWkEnd" class="sched-select"></select>
          <button type="button" class="sched-preset-btn" id="schedFullPreset"
            title="Show every remaining week">Full Season</button>
          <button type="button" class="sched-preset-btn" id="schedPlayoffPreset"
            title="Jump to the fantasy playoff weeks (15-17)">Playoffs</button>
        </div>

        <!-- My Players: player search -->
        <div class="sched-add" id="schedAddWrap">
          <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:13px;color:var(--text-muted);pointer-events:none;"><i class="fa-solid fa-magnifying-glass"></i></span>
          <input id="schedAddInput" type="text" placeholder="Add a player..." autocomplete="off"
            style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                   border:1px solid var(--border);background:var(--card-bg);
                   color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
          <button id="schedAddClear" type="button"
            style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                   background:none;border:none;cursor:pointer;color:var(--text-muted);
                   font-size:16px;line-height:1;padding:2px;" aria-label="Clear search">&#x2715;</button>
          <div id="schedAddResults" class="sched-add-results" style="display:none;"></div>
        </div>

        <!-- Schedule Rankings: position pills + sort -->
        <div class="sched-rank-controls" id="schedRankControls" style="display:none;">
          <div class="otc-day-filters sched-pos-pills" id="schedRankPosPills">
            <button class="otc-day-filter sched-rank-pos active" data-pos="QB">QB</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="RB">RB</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="WR">WR</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="TE">TE</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="K">K</button>
          </div>
          <button class="sched-sort-btn" id="schedRankSort">
            Easiest First <i class="fa-solid fa-arrow-up-short-wide" aria-hidden="true"></i>
          </button>
        </div>
      </div>

      <!-- My Players view -->
      <div id="schedMyPlayersSection">
        <div class="sched-legend">
          <span><span class="sched-chip" style="background:#22c55e;"></span>Elite (top 25%)</span>
          <span><span class="sched-chip" style="background:#84cc16;"></span>Good</span>
          <span><span class="sched-chip" style="background:#f59e0b;"></span>Tough</span>
          <span><span class="sched-chip" style="background:#ef4444;"></span>Brutal (bottom 25%)</span>
          <span class="sched-legend-note">Rank = fantasy pts allowed per game at that position (PPR). #1 = easiest matchup.</span>
        </div>

        <div id="schedGrid" class="sched-grid-wrap">
          <div class="sched-empty">Loading schedule&#8230;</div>
        </div>
      </div>

      <!-- Schedule Rankings view -->
      <div id="schedRankingsSection" style="display:none;">
        <div class="sched-legend" style="padding-top:0;">
          <span><span class="sched-chip" style="background:#22c55e;"></span>Elite</span>
          <span><span class="sched-chip" style="background:#84cc16;"></span>Good</span>
          <span><span class="sched-chip" style="background:#f59e0b;"></span>Tough</span>
          <span><span class="sched-chip" style="background:#ef4444;"></span>Brutal</span>
          <span class="sched-legend-note">Ease score: 100 = easiest schedule, 0 = hardest. Rank = fpts allowed rank vs. that position.</span>
        </div>
        <div id="schedRankingsGrid" class="sched-grid-wrap">
          <div class="sched-empty">Loading&#8230;</div>
        </div>
      </div>
    </div>
    """

    script = """
    <script>
    (function() {
      var CFG = __CFG__;
      var LS_PIDS = 'sched_pids_' + CFG.leagueId;
      var LS_WKS  = 'sched_wks_'  + CFG.leagueId;

      var selPids = [];
      try { selPids = JSON.parse(localStorage.getItem(LS_PIDS) || 'null'); } catch (e) {}
      if (!Array.isArray(selPids) || !selPids.length) selPids = CFG.initPids.slice();

      var wkStart = CFG.defStart, wkEnd = CFG.defEnd;
      try {
        var saved = JSON.parse(localStorage.getItem(LS_WKS) || 'null');
        if (saved && saved.s) { wkStart = saved.s; wkEnd = saved.e; }
      } catch (e) {}
      if (wkStart < CFG.startWeek) wkStart = CFG.startWeek;
      if (wkEnd > CFG.maxWeek) wkEnd = CFG.maxWeek;
      if (wkEnd < wkStart) wkEnd = wkStart;

      var currentView  = 'my-players';
      var rankPos      = 'QB';
      var rankHardFirst = false;
      var rankingsCache = null;   // last fetched rankings data
      var _myKey   = null;        // week range My Players last rendered at
      var _rankKey = null;        // week range Rankings last rendered at

      var pool      = [];
      var poolReady = false;

      var startSel   = document.getElementById('schedWkStart');
      var endSel     = document.getElementById('schedWkEnd');
      var addInput   = document.getElementById('schedAddInput');
      var addResults = document.getElementById('schedAddResults');
      var addClear   = document.getElementById('schedAddClear');
      var gridEl     = document.getElementById('schedGrid');

      function fillWeekSelects() {
        var optsS = '', optsE = '';
        for (var w = CFG.startWeek; w <= CFG.maxWeek; w++) {
          optsS += '<option value="' + w + '"' + (w === wkStart ? ' selected' : '') + '>Week ' + w + '</option>';
          optsE += '<option value="' + w + '"' + (w === wkEnd   ? ' selected' : '') + '>Week ' + w + '</option>';
        }
        startSel.innerHTML = optsS;
        endSel.innerHTML   = optsE;
      }

      function persist() {
        try { localStorage.setItem(LS_PIDS, JSON.stringify(selPids)); } catch (e) {}
        try { localStorage.setItem(LS_WKS,  JSON.stringify({s: wkStart, e: wkEnd})); } catch (e) {}
      }

      function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, function(c) {
          return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
        });
      }

      // ── My Players view ─────────────────────────────────────────────────────
      function renderGrid() {
        _myKey = wkStart + '-' + wkEnd;
        if (!selPids.length) {
          gridEl.innerHTML = '<div class="sched-empty"><strong style="color:var(--text);display:block;margin-bottom:4px;">No players yet</strong>Use the search above to add players.</div>';
          return;
        }
        gridEl.innerHTML = '<div class="loading-state-msg"><div class="loading-spinner" aria-hidden="true"></div><span>Loading schedule…</span></div>';
        var url = '/api/schedule?season=' + CFG.season +
                  '&week_start=' + wkStart + '&week_end=' + wkEnd +
                  '&pids=' + encodeURIComponent(selPids.join(','));
        fetch(url).then(function(r) { return r.json(); }).then(function(data) {
          var weeks   = data.weeks   || [];
          var players = data.players || [];
          if (!players.length) {
            gridEl.innerHTML = '<div class="sched-empty"><strong style="color:var(--text);display:block;margin-bottom:4px;">No matchup data</strong>Nothing to show for this selection.</div>';
            return;
          }
          var head = '<th class="sched-th sched-th-player">Player</th>' +
                     '<th class="sched-th sched-th-sos" title="Strength of schedule rank for this position over the selected weeks (1 = easiest)">SoS</th>';
          for (var i = 0; i < weeks.length; i++) head += '<th class="sched-th">WK ' + weeks[i] + '</th>';
          var rows = '';
          players.forEach(function(p) {
            var cells = '';
            (p.cells || []).forEach(function(c) {
              if (c.bye) { cells += '<td class="sched-td sched-bye">BYE</td>'; return; }
              var rankLabel = c.rank ? ('#' + c.rank) : '–';
              var fptsLabel = c.fpts ? (c.fpts + ' pts') : '';
              cells += '<td class="sched-td" style="background:' + c.bg + ';">' +
                         '<div class="sched-opp">'  + esc(c.at + c.opp) + '</div>' +
                         '<div class="sched-rank" style="color:' + c.txt + ';">' + rankLabel + '</div>' +
                         '<div class="sched-fpts">' + esc(fptsLabel) + '</div>' +
                       '</td>';
            });
            var sosCell;
            if (p.sos_rank) {
              var sosFrac = p.sos_rank / (p.sos_total || 32);
              var sc = sosFrac <= 0.25 ? '#22c55e' : sosFrac <= 0.50 ? '#84cc16' : sosFrac <= 0.75 ? '#f59e0b' : '#ef4444';
              sosCell = '<td class="sched-td sched-sos-td" style="color:' + sc + ';">#' + p.sos_rank +
                        '<span class="sched-sos-total">/' + (p.sos_total || 32) + '</span></td>';
            } else {
              sosCell = '<td class="sched-td sched-sos-td">–</td>';
            }
            rows += '<tr>' +
              '<td class="sched-td sched-td-player">' +
                '<button class="sched-remove" data-pid="' + esc(p.pid) + '" title="Remove">&times;</button>' +
                '<span class="sched-pos" style="background:' + p.color + '22;color:' + p.color + ';">' + esc(p.pos) + '</span>' +
                '<span class="player-clickable sched-pname" data-player-id="' + esc(p.pid) + '">' + esc(p.name) + '</span>' +
                '<span class="sched-nfl">' + esc(p.nfl) + '</span>' +
              '</td>' + sosCell + cells + '</tr>';
          });
          gridEl.innerHTML =
            '<table class="sched-table"><thead><tr>' + head + '</tr></thead><tbody>' + rows + '</tbody></table>';
        }).catch(function() {
          gridEl.innerHTML = '<div class="sched-empty" style="color:var(--loss);"><strong style="display:block;margin-bottom:4px;">Couldn’t load schedule</strong>Refresh and try again.</div>';
        });
      }

      // ── Schedule Rankings view ────────────────────────────────────────────────────
      function switchView(v) {
        currentView = v;
        document.getElementById('schedMyPlayersSection').style.display  = v === 'my-players' ? '' : 'none';
        document.getElementById('schedRankingsSection').style.display   = v === 'rankings'   ? '' : 'none';
        // Flip the shared controls to match the active tab
        document.getElementById('schedAddWrap').style.display      = v === 'my-players' ? '' : 'none';
        document.getElementById('schedRankControls').style.display = v === 'rankings'   ? '' : 'none';
        document.querySelectorAll('.sched-view-btn').forEach(function(b) {
          b.classList.toggle('active', b.getAttribute('data-view') === v);
        });
        // Re-render the target view if the (shared) week range changed since it
        // last rendered, so switching tabs always reflects the current weeks.
        var key = wkStart + '-' + wkEnd;
        if (v === 'rankings') {
          if (_rankKey !== key) { rankPage = 0; rankingsCache = null; renderRankings(); }
        } else {
          if (_myKey !== key) renderGrid();
        }
      }

      function renderRankings() {
        _rankKey = wkStart + '-' + wkEnd;
        rankingsCache = null;
        var rankGrid = document.getElementById('schedRankingsGrid');
        rankGrid.innerHTML = '<div class="loading-state-msg"><div class="loading-spinner" aria-hidden="true"></div><span>Loading rankings…</span></div>';
        var url = '/api/schedule-rankings?season=' + CFG.season +
                  '&week_start=' + wkStart + '&week_end=' + wkEnd +
                  '&position=' + rankPos +
                  '&league_id=' + encodeURIComponent(CFG.leagueId) +
                  '&platform='  + encodeURIComponent(CFG.platform);
        fetch(url).then(function(r) { return r.json(); }).then(function(data) {
          rankingsCache = data;
          buildRankingsTable(data);
        }).catch(function() {
          rankGrid.innerHTML = '<div class="sched-empty" style="color:var(--loss);"><strong style="display:block;margin-bottom:4px;">Couldn’t load rankings</strong>Refresh and try again.</div>';
        });
      }

      var rankPage = 0;

      function buildRankingsTable(data) {
        var rankGrid  = document.getElementById('schedRankingsGrid');
        var weeks     = data.weeks    || [];
        var total     = data.total_teams || 32;
        var rankings  = (data.rankings || []).slice();
        if (rankHardFirst) rankings.reverse();

        if (!rankings.length) {
          rankGrid.innerHTML = '<div class="sched-empty"><strong style="color:var(--text);display:block;margin-bottom:4px;">No data</strong>Nothing for this position or week range.</div>';
          return;
        }

        // Group players from the same NFL team into one row - their schedule,
        // avg rank and ease are identical, so separate rows just repeat values.
        var groups = [];
        var byTeam = {};
        rankings.forEach(function(p) {
          var g = byTeam[p.team];
          if (!g) { g = { team: p.team, rep: p, players: [] }; byTeam[p.team] = g; groups.push(g); }
          g.players.push(p);
        });

        var head = '<th class="sched-th sched-th-player">Team</th>' +
                   '<th class="sched-th sched-th-sos">Avg</th>';
        for (var i = 0; i < weeks.length; i++) head += '<th class="sched-th">WK ' + weeks[i] + '</th>';
        head += '<th class="sched-th sched-th-ease" style="min-width:90px;">Ease</th>';

        var rows = '';
        groups.forEach(function(g, idx) {
          var p = g.rep;
          var rank = idx + 1;
          var medalHtml;
          if (rank === 1)      medalHtml = '<span class="sched-rank-medal sched-rank-medal-1">1</span>';
          else if (rank === 2) medalHtml = '<span class="sched-rank-medal sched-rank-medal-2">2</span>';
          else if (rank === 3) medalHtml = '<span class="sched-rank-medal sched-rank-medal-3">3</span>';
          else                 medalHtml = '<span class="sched-rank-num">' + rank + '</span>';

          var namesHtml = g.players.map(function(pl) {
            var badge = pl.owner
              ? '<span class="sched-roster-badge sched-roster-badge--owner">' + esc(pl.owner) + '</span>'
              : (pl.on_roster ? '<span class="sched-roster-badge">Rostered</span>' : '');
            return '<div class="sched-rank-player-line">' +
                     '<span class="player-clickable sched-pname" data-player-id="' + esc(pl.pid) + '">' + esc(pl.name) + '</span>' +
                     badge +
                   '</div>';
          }).join('');

          var cells = '';
          (p.cells || []).forEach(function(c) {
            if (c.bye) { cells += '<td class="sched-td sched-bye">BYE</td>'; return; }
            var rankLabel = c.rank ? ('#' + c.rank) : '–';
            cells += '<td class="sched-td" style="background:' + c.bg + ';">' +
                       '<div class="sched-opp">'  + esc((c.at || '') + c.opp) + '</div>' +
                       '<div class="sched-rank" style="color:' + c.txt + ';">' + rankLabel + '</div>' +
                     '</td>';
          });

          var ar = p.avg_rank;
          var avgTxt   = ar < 900 ? ('#' + ar) : '–';
          var avgColor = ar <= total * 0.25 ? '#22c55e'
                       : ar <= total * 0.50 ? '#84cc16'
                       : ar <= total * 0.75 ? '#f59e0b' : '#ef4444';

          var ease      = p.ease_score || 0;
          var easeColor = ease >= 75 ? '#22c55e'
                        : ease >= 50 ? '#84cc16'
                        : ease >= 25 ? '#f59e0b' : '#ef4444';
          var easeBar = '<div class="sched-ease-wrap">' +
                          '<div class="sched-ease-bar" style="width:' + Math.round(ease) + '%;background:' + easeColor + ';"></div>' +
                        '</div>' +
                        '<div class="sched-ease-num" style="color:' + easeColor + ';">' + Math.round(ease) + '</div>';

          rows += '<tr>' +
            '<td class="sched-td sched-td-player">' +
              '<div class="sched-rank-player-cell sched-rank-player-cell--group">' +
                medalHtml +
                '<span class="sched-pos" style="background:' + p.color + '22;color:' + p.color + ';">' + esc(p.pos) + '</span>' +
                '<div class="sched-rank-name-wrap">' + namesHtml + '</div>' +
                '<span class="sched-nfl">' + esc(g.team) + '</span>' +
              '</div>' +
            '</td>' +
            '<td class="sched-td sched-sos-td" style="color:' + avgColor + ';">' + avgTxt + '</td>' +
            cells +
            '<td class="sched-td sched-ease-td">' + easeBar + '</td>' +
          '</tr>';
        });

        rankGrid.innerHTML =
          '<table class="sched-table"><thead><tr>' + head + '</tr></thead><tbody>' + rows + '</tbody></table>';
      }

      // ── Player search / pool ────────────────────────────────────────────────────────────────
      function loadPool() {
        fetch('/api/league-players').then(function(r) { return r.json(); }).then(function(resp) {
          var arr = Array.isArray(resp) ? resp : (resp.players || []);
          pool = arr.filter(function(p) {
            var pos = String(p.position || '').toUpperCase();
            return ['QB','RB','WR','TE','K','DEF'].indexOf(pos) !== -1 && p.team && p.team !== 'FA';
          }).map(function(p) {
            return {id: String(p.id), name: p.name || '', pos: String(p.position || '').toUpperCase(), team: p.team || ''};
          });
          poolReady = true;
        }).catch(function() {});
      }

      function showAddResults(q) {
        if (!poolReady || !q) { addResults.style.display = 'none'; return; }
        var ql  = q.toLowerCase();
        var sel = {};
        selPids.forEach(function(id) { sel[id] = 1; });
        var matches = pool.filter(function(p) {
          return !sel[p.id] && p.name.toLowerCase().indexOf(ql) !== -1;
        }).slice(0, 8);
        if (!matches.length) { addResults.style.display = 'none'; return; }
        addResults.innerHTML = matches.map(function(p) {
          var col = ({QB:'#3b82f6',RB:'#22c55e',WR:'#f59e0b',TE:'#8b5cf6'})[p.pos] || '#6b7280';
          return '<div class="sched-add-row" data-pid="' + esc(p.id) + '">' +
                   '<span class="sched-pos" style="background:' + col + '22;color:' + col + ';">' + esc(p.pos) + '</span>' +
                   '<span>' + esc(p.name) + '</span>' +
                   '<span class="sched-nfl">' + esc(p.team) + '</span>' +
                 '</div>';
        }).join('');
        addResults.style.display = 'block';
      }

      // ── Wire up controls ──────────────────────────────────────────────────────────────────
      fillWeekSelects();

      startSel.addEventListener('change', function() {
        wkStart = parseInt(this.value, 10);
        if (wkEnd < wkStart) wkEnd = wkStart;
        fillWeekSelects(); persist(); syncPresetBtns();
        if (currentView === 'my-players') renderGrid(); else { rankPage = 0; rankingsCache = null; renderRankings(); }
      });
      endSel.addEventListener('change', function() {
        wkEnd = parseInt(this.value, 10);
        if (wkStart > wkEnd) wkStart = wkEnd;
        fillWeekSelects(); persist(); syncPresetBtns();
        if (currentView === 'my-players') renderGrid(); else { rankPage = 0; rankingsCache = null; renderRankings(); }
      });

      // One-click range presets: Full Season (every remaining week) and
      // Playoffs (weeks 15-17), each clamped to what's still selectable.
      var fullBtn    = document.getElementById('schedFullPreset');
      var playoffBtn = document.getElementById('schedPlayoffPreset');
      function syncPresetBtns() {
        var ps = Math.max(CFG.startWeek, 15), pe = Math.min(CFG.maxWeek, 17);
        if (fullBtn) fullBtn.classList.toggle('active', wkStart === CFG.startWeek && wkEnd === CFG.maxWeek);
        if (playoffBtn) {
          playoffBtn.classList.toggle('active', wkStart === ps && wkEnd === pe);
          playoffBtn.disabled = ps > pe;
        }
      }
      function applyPreset(s, e) {
        wkStart = s; wkEnd = e;
        if (wkEnd < wkStart) wkEnd = wkStart;
        fillWeekSelects(); persist(); syncPresetBtns();
        if (currentView === 'my-players') renderGrid(); else { rankPage = 0; rankingsCache = null; renderRankings(); }
      }
      if (fullBtn) {
        fullBtn.addEventListener('click', function() { applyPreset(CFG.startWeek, CFG.maxWeek); });
      }
      if (playoffBtn) {
        playoffBtn.addEventListener('click', function() {
          applyPreset(Math.max(CFG.startWeek, 15), Math.min(CFG.maxWeek, 17));
        });
      }
      syncPresetBtns();

      addInput.addEventListener('input', function() {
        showAddResults(this.value.trim());
        if (addClear) addClear.style.display = this.value ? '' : 'none';
      });
      addInput.addEventListener('focus', function() { showAddResults(this.value.trim()); });
      if (addClear) {
        addClear.addEventListener('click', function() {
          addInput.value = '';
          addResults.style.display = 'none';
          addClear.style.display = 'none';
          addInput.focus();
        });
      }

      document.addEventListener('click', function(e) {
        var row = e.target.closest ? e.target.closest('.sched-add-row') : null;
        if (row && addResults.contains(row)) {
          var pid = row.getAttribute('data-pid');
          if (pid && selPids.indexOf(pid) === -1) { selPids.push(pid); persist(); renderGrid(); }
          addInput.value = '';
          addResults.style.display = 'none';
          if (addClear) addClear.style.display = 'none';
          return;
        }
        if (!addResults.contains(e.target) && e.target !== addInput) {
          addResults.style.display = 'none';
        }
      });

      gridEl.addEventListener('click', function(e) {
        var btn = e.target.closest ? e.target.closest('.sched-remove') : null;
        if (btn) {
          e.stopPropagation();
          var pid = btn.getAttribute('data-pid');
          selPids = selPids.filter(function(id) { return id !== pid; });
          persist(); renderGrid();
        }
      });

      // View toggle
      document.getElementById('schedViewToggle').addEventListener('click', function(e) {
        var btn = e.target.closest ? e.target.closest('.sched-view-btn') : null;
        if (btn) switchView(btn.getAttribute('data-view'));
      });

      // Rankings position filter
      document.getElementById('schedRankPosPills').addEventListener('click', function(e) {
        var btn = e.target.closest ? e.target.closest('.sched-rank-pos') : null;
        if (!btn) return;
        rankPos = btn.getAttribute('data-pos');
        rankPage = 0;
        document.querySelectorAll('.sched-rank-pos').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        renderRankings();
      });

      // Sort toggle
      document.getElementById('schedRankSort').addEventListener('click', function() {
        rankHardFirst = !rankHardFirst;
        this.innerHTML = rankHardFirst
          ? 'Hardest First <i class="fa-solid fa-arrow-down-short-wide" aria-hidden="true"></i>'
          : 'Easiest First <i class="fa-solid fa-arrow-up-short-wide" aria-hidden="true"></i>';
        rankPage = 0;
        if (rankingsCache) buildRankingsTable(rankingsCache);
      });

      loadPool();
      renderGrid();
    })();
    </script>
    """.replace("__CFG__", cfg)

    # ── Bye-week outlook strip: upcoming weeks where the viewer's roster has
    # multiple players sharing a bye, so waiver moves can be planned ahead. ──
    bye_banner = ""
    try:
        from utils.bye_outlook import build_bye_outlook as _bbo
        _byes = _team_bye_map(season)
        if _byes and roster_pids:
            _reqs = dict(ctx.get("lineup_requirements") or {})
            if not _reqs:
                for _slot in (ctx.get("roster_positions") or []):
                    _s = str(_slot).upper()
                    if _s in {"QB", "RB", "WR", "TE", "K", "DEF"}:
                        _reqs[_s] = _reqs.get(_s, 0) + 1
            _roster = [
                {"pos": (players_idx.get(pid) or {}).get("pos"),
                 "team": (players_idx.get(pid) or {}).get("team")}
                for pid in roster_pids
            ]
            _from = current_week if current_week and current_week >= 1 else 1
            _outlook = _bbo(_byes, _roster, _reqs, from_week=_from)
            if _outlook:
                def _short(bp):
                    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
                    items = sorted(bp.items(), key=lambda kv: (-kv[1], order.get(kv[0], 9)))
                    return " ".join(
                        f'<span class="sbo-tag">{n} {p}</span>' for p, n in items
                    )

                _warn = ('<i class="fa-solid fa-triangle-exclamation sbo-warn" '
                         'aria-hidden="true"></i>')

                def _chip(w):
                    cr = w["crunch"]
                    return (
                        f'<div class="sbo-week{" is-crunch" if cr else ""}">'
                        f'<span class="sbo-wk">{_warn if cr else ""}Wk {w["week"]}</span>'
                        f'<span class="sbo-pos">{_short(w["by_pos"])}</span>'
                        '</div>'
                    )

                _chips = "".join(_chip(w) for w in _outlook[:8])
                _has_crunch = any(w["crunch"] for w in _outlook)
                _note = ("Amber = multiple starters share a bye"
                         if _has_crunch else "Upcoming byes on your roster")
                bye_banner = (
                    '<div class="sched-bye-outlook" role="group" aria-label="Bye week outlook">'
                    '<div class="sbo-header">'
                    '<span class="sbo-label">Bye Outlook</span>'
                    f'<span class="sbo-note">{_note}</span>'
                    '</div>'
                    f'<div class="sbo-weeks">{_chips}</div>'
                    '</div>'
                )
    except Exception:
        logger.debug("[schedule] bye outlook skipped", exc_info=True)

    shell = shell.replace("__BYE_BANNER__", bye_banner)
    return shell + script

