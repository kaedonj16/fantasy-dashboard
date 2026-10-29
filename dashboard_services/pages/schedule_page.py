"""Schedule rankings HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_schedule_body(ctx):
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        datetime, logger,
    )

    import json as _json
    season = int(ctx.get("season") or datetime.now().year)
    league_id = ctx.get("league_id") or ""
    platform = ctx.get("platform") or "sleeper"
    viewer_rid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    rosters = ctx.get("rosters") or []

    current_week = int(ctx.get("current_week") or 0)
    max_week = 18
    start_week = current_week if current_week >= 1 else 1
    def_start = start_week
    playoff_start = int((ctx.get("league_settings") or {}).get("playoff_week_start") or 15)
    playoff_end = min(max_week, playoff_start + 2)
    def_end = playoff_end
    from utils.defensive_matchup_ratings import season_weights
    from utils.season_qualification import qualification_policy
    active_season = int(ctx.get("current_season") or season)
    completed_week = max(qualification_policy(season).completed_weeks, default=0) if season == active_season else (18 if season < active_season else 0)
    previous_weight, current_weight = season_weights(completed_week, blend=season >= active_season)

    viewer_roster = next((r for r in rosters if str(r.get("roster_id")) == viewer_rid), None)
    roster_pids = [str(p) for p in (viewer_roster.get("players") if viewer_roster else []) or []]

    cfg = _json.dumps({
        "season": season, "leagueId": league_id, "platform": platform,
        "startWeek": start_week, "maxWeek": max_week,
        "defStart": def_start, "defEnd": def_end,
        "playoffStart": playoff_start, "playoffEnd": playoff_end,
        "completedWeek": completed_week, "activeSeason": active_season,
        "previousWeight": previous_weight, "currentWeight": current_weight,
        "myPids": roster_pids,
    })

    shell = """
    <div class="card central schedule-card">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
          <h2>Schedule Rankings</h2>
          <div style="font-size:15px;color:var(--text-muted);margin-top:4px;">
            Remaining schedules ranked easiest to hardest, adjusted for opponent strength. Ratings rebuild on Wednesdays in-season.
          </div>
          <div class="srk-scope-note" id="srkScopeNote">Every NFL team at the position, with the player who holds the schedule in this league.</div>
        </div>
        <div class="sched-view-toggle" id="schedViewToggle">
          <button class="sched-view-btn active" data-view="league">League</button>
          <button class="sched-view-btn" data-view="myteam">My Team</button>
        </div>
      </div>

      <!-- Shared controls: week range, position pills and sort order -->
      <div class="sched-controls sched-shared-controls">
        <div class="sched-week-range">
          <span class="sched-ctrl-label">Weeks</span>
          <select id="schedWkStart" class="sched-select"></select>
          <span class="sched-ctrl-sep">to</span>
          <select id="schedWkEnd" class="sched-select"></select>
          <div class="sched-presets" role="group" aria-label="Schedule range presets">
            <button type="button" class="sched-preset-btn" id="schedRosPreset"
              title="Current NFL week through the last playoff week">ROS</button>
            <button type="button" class="sched-preset-btn" id="schedFullPreset"
              title="Show Weeks 1 through 18">Full Season</button>
            <button type="button" class="sched-preset-btn" id="schedPlayoffPreset"
              title="Jump to this league's fantasy playoff weeks">Playoffs</button>
          </div>
        </div>

        <div class="sched-rank-controls" id="schedRankControls">
          <div class="otc-day-filters sched-pos-pills" id="schedRankPosPills">
            <button class="otc-day-filter sched-rank-pos" data-pos="all">All</button>
            <button class="otc-day-filter sched-rank-pos active" data-pos="QB">QB</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="RB">RB</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="WR">WR</button>
            <button class="otc-day-filter sched-rank-pos" data-pos="TE">TE</button>
          </div>
          <button class="sched-sort-btn" id="schedRankSort">
            Easiest First <i class="fa-solid fa-arrow-up-short-wide" aria-hidden="true"></i>
          </button>
        </div>
      </div>

      <div class="sched-legend">
        <span><span class="sched-chip" style="background:#22c55e;"></span>Elite (top 25%)</span>
        <span><span class="sched-chip" style="background:#84cc16;"></span>Good</span>
        <span><span class="sched-chip" style="background:#f59e0b;"></span>Tough</span>
        <span><span class="sched-chip" style="background:#ef4444;"></span>Brutal (bottom 25%)</span>
        <span class="sched-legend-note">Ease is the 0 to 100 score for the selected range, byes excluded. Each week cell is that defense's adjusted % for the week.</span>
        <button type="button" class="sched-info" aria-label="About ease" title="Ease ranks the selected schedule using opponent-adjusted defensive performance. #1 is easiest. Bye weeks are excluded."><i class="fa-solid fa-circle-info" aria-hidden="true"></i></button>
        <button type="button" class="sched-info" aria-label="Seasonal rating data source" title="__SOURCE_COPY__"><i class="fa-solid fa-circle-info" aria-hidden="true"></i></button>
      </div>

      <div id="schedRankingsGrid" class="sched-grid-wrap">
        <div class="sched-empty">Loading schedule&#8230;</div>
      </div>
    </div>
    """

    script = """
    <script>
    (function() {
      var CFG = __CFG__;
      var LS_WKS   = 'sched_wks_'   + CFG.leagueId + '_' + CFG.season;
      var LS_ADDED = 'sched_added_' + CFG.leagueId;
      var POSITIONS = ['QB', 'RB', 'WR', 'TE'];

      var mySet = {};
      (CFG.myPids || []).forEach(function(p) { mySet[String(p)] = 1; });

      // Players staged onto this page (deep links, assistant links) live in
      // their own saved list, separate from the viewer's roster.
      var addedPids = [];
      try {
        var savedAdded = JSON.parse(localStorage.getItem(LS_ADDED) || '[]');
        if (Array.isArray(savedAdded)) {
          savedAdded.forEach(function(p) {
            p = String(p);
            if (!mySet[p] && addedPids.indexOf(p) === -1) addedPids.push(p);
          });
        }
      } catch (e) {}
      function persistAdded() {
        try { localStorage.setItem(LS_ADDED, JSON.stringify(addedPids)); } catch (e) {}
      }
      var addedSet = {};
      function syncAddedSet() {
        addedSet = {};
        addedPids.forEach(function(p) { addedSet[p] = 1; });
      }
      syncAddedSet();

      var hasAddParam = false;
      try {
        var addPid = new URLSearchParams(window.location.search).get('add');
        if (addPid) {
          addPid = String(addPid);
          hasAddParam = true;
          if (!mySet[addPid] && addedPids.indexOf(addPid) === -1) {
            addedPids.unshift(addPid);
            // A URL-staged add must land in the same saved list the remove
            // button writes, or it silently vanishes on the next visit
            // without the ?add= param.
            persistAdded();
            syncAddedSet();
          }
        }
      } catch (e) {}

      var wkStart = CFG.defStart, wkEnd = CFG.defEnd;
      try {
        var saved = JSON.parse(localStorage.getItem(LS_WKS) || 'null');
        if (saved && saved.s) { wkStart = saved.s; wkEnd = saved.e; }
      } catch (e) {}
      if (wkStart < 1) wkStart = 1;
      if (wkEnd > CFG.maxWeek) wkEnd = CFG.maxWeek;
      if (wkEnd < wkStart) wkEnd = wkStart;

      var currentView = hasAddParam ? 'myteam' : 'league';
      var rankPos     = 'QB';
      var hardFirst   = false;

      // Fetch promises for the selected range, keyed by position; dropped
      // when the range changes.
      var mainCache = {};

      var startSel = document.getElementById('schedWkStart');
      var endSel   = document.getElementById('schedWkEnd');
      var gridEl   = document.getElementById('schedRankingsGrid');
      var sortBtn  = document.getElementById('schedRankSort');

      function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"]/g, function(c) {
          return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
        });
      }

      function fmtPct(v) {
        if (v == null || isNaN(v)) return 'N/A';
        return (v > 0 ? '+' : '') + Number(v).toFixed(1) + '%';
      }

      function tierColor(rank, total) {
        if (rank == null) return '#94a3b8';
        var f = rank / (total || 32);
        return f <= 0.25 ? '#22c55e' : f <= 0.50 ? '#84cc16' : f <= 0.75 ? '#f59e0b' : '#ef4444';
      }

      // ── Data layer ──────────────────────────────────────────────────────
      function fetchRange(cache, pos, s, e) {
        if (!cache[pos]) {
          var url = '/api/schedule-rankings?season=' + CFG.season +
                    '&week_start=' + s + '&week_end=' + e +
                    '&position=' + pos +
                    '&league_id=' + encodeURIComponent(CFG.leagueId) +
                    '&platform='  + encodeURIComponent(CFG.platform);
          cache[pos] = fetch(url).then(function(r) { return r.json(); });
          // A failed fetch must not poison the cache for the next render.
          cache[pos].catch(function() { delete cache[pos]; });
        }
        return cache[pos];
      }

      function posData(pos) {
        return fetchRange(mainCache, pos, wkStart, wkEnd).then(function(data) {
          return { main: data };
        });
      }

      // ── Row models ──────────────────────────────────────────────────────
      function makeModel(r, pd) {
        var pid = String(r.pid);
        return { r: r, pid: pid, added: !!addedSet[pid] };
      }

      function sortModels(models) {
        var dir = hardFirst ? -1 : 1;
        return models.slice().sort(function(a, b) {
          var ra = a.r.sos_rank == null ? 999 : a.r.sos_rank;
          var rb = b.r.sos_rank == null ? 999 : b.r.sos_rank;
          return (ra - rb) * dir;
        });
      }

      // League scope: one row per NFL team. Payload order is ease order, so
      // the first row seen for a team is its representative.
      function leagueModels(pd) {
        var seen = {}, out = [];
        sortModels((pd.main.rankings || []).map(function(r) { return makeModel(r, pd); }))
          .forEach(function(m) {
            if (!seen[m.r.team]) { seen[m.r.team] = 1; out.push(m); }
          });
        return out;
      }

      // My Team scope for one position: staged players pin above roster rows.
      function myModels(pd) {
        var added = [], mine = [];
        (pd.main.rankings || []).forEach(function(r) {
          var pid = String(r.pid);
          if (addedSet[pid]) added.push(makeModel(r, pd));
          else if (mySet[pid]) mine.push(makeModel(r, pd));
        });
        var order = {};
        addedPids.forEach(function(p, i) { order[p] = i; });
        added.sort(function(a, b) { return order[a.pid] - order[b.pid]; });
        return { added: added, mine: sortModels(mine) };
      }

      // ── Shared cell fragments ───────────────────────────────────────────
      function badgeHtml(m) {
        if (m.added) return '<span class="sched-roster-badge srk-badge-added">Added</span>';
        if (mySet[m.pid]) return '<span class="sched-roster-badge srk-badge-yours">Yours</span>';
        if (m.r.on_roster && m.r.owner) return '<span class="sched-roster-badge sched-roster-badge--owner">' + esc(m.r.owner) + '</span>';
        if (m.r.on_roster) return '<span class="sched-roster-badge">Rostered</span>';
        return '<span class="sched-roster-badge srk-badge-fa">Available</span>';
      }

      function removeHtml(m) {
        return m.added
          ? '<button class="sched-remove srk-remove" data-remove-pid="' + esc(m.pid) + '" title="Remove">&times;</button>'
          : '';
      }

      function playerInner(m) {
        var r = m.r;
        return '<span class="srk-pcell">' +
               '<span class="sched-pos" style="background:' + r.color + '22;color:' + r.color + ';">' + esc(r.pos) + '</span>' +
               '<span class="srk-names"><span class="player-clickable sched-pname" data-player-id="' + esc(m.pid) + '">' + esc(r.name) + '</span></span>' +
               '<span class="sched-nfl">' + esc(r.team) + '</span>' + badgeHtml(m) + removeHtml(m) +
               '</span>';
      }

      // ── Desktop table ───────────────────────────────────────────────────
      function poColIndex(weeks) {
        for (var i = 0; i < weeks.length; i++) if (weeks[i] >= CFG.playoffStart) return i;
        return -1;
      }

      function bandRow(weeks) {
        var reg = 0, po = 0;
        weeks.forEach(function(w) { if (w >= CFG.playoffStart) po++; else reg++; });
        // The spacer is two cells, not one colspan: the Rank+Player spacer
        // stays pinned over the sticky columns, and the Ease spacer hides
        // together with the Ease column on phones. A single colspan spacer
        // would shift the Playoffs band one column when Ease hides.
        var h = '<tr class="srk-band"><th colspan="2" class="srk-band-pin"></th><th class="srk-band-ease"></th>';
        if (reg) h += '<th colspan="' + reg + '">Regular season</th>';
        if (po) h += '<th colspan="' + po + '" class="srk-band-po">Playoffs</th>';
        return h + '</tr>';
      }

      function tableRow(m, poIdx) {
        var r = m.r;
        var tc = tierColor(r.sos_rank, r.sos_total);
        var rankHtml = r.sos_rank != null
          ? '<span style="color:' + tc + '">#' + r.sos_rank + '</span>'
          : '<span class="srk-rank-na">n/a</span>';
        var easeHtml;
        if (r.ease_score != null) {
          var ease = Math.round(r.ease_score);
          easeHtml = '<span class="srk-ease-bar"><span style="width:' + ease + '%;background:' + tc + '"></span></span>' +
                     '<span class="srk-ease-num" style="color:' + tc + '">' + ease + '</span>';
        } else {
          easeHtml = '<span class="srk-rank-na">n/a</span>';
        }
        var cells = '';
        (r.cells || []).forEach(function(c, i) {
          var cls = 'sched-td' + (i === poIdx && i > 0 ? ' srk-po-start' : '');
          if (c.bye) { cells += '<td class="' + cls + ' sched-bye">BYE</td>'; return; }
          var effect = c.adjusted_percent != null ? fmtPct(c.adjusted_percent) : 'N/A';
          var meaning = c.opp + ' is #' + (c.rank || 'N/A') + ' of ' + (c.total || 32) + ' vs ' + r.pos + ', ' + effect + ' adjusted. This is not ' + r.name + '’s projected score.';
          cells += '<td class="' + cls + '" title="' + esc(meaning) + '" style="background:' + (c.bg || 'transparent') + '">' +
                     '<div class="sched-opp">' + esc((c.at || '') + (c.opp || '')) + '</div>' +
                     '<div class="sched-fpts" style="color:' + (c.txt || 'inherit') + ';">' + effect + '</div>' +
                   '</td>';
        });

        return '<tr>' +
          '<td class="sched-td srk-td-rank">' + rankHtml + '</td>' +
          '<td class="sched-td sched-td-player">' + playerInner(m) + '</td>' +
          '<td class="sched-td srk-ease-cell">' + easeHtml + '</td>' +
          cells + '</tr>';
      }

      function buildTable(sections, weeks) {
        var poIdx = poColIndex(weeks);
        var head = '<th class="sched-th srk-th-rank">Rank</th>' +
                   '<th class="sched-th sched-th-player">Player</th>' +
                   '<th class="sched-th srk-th-ease">Ease</th>';
        weeks.forEach(function(w, i) {
          head += '<th class="sched-th' + (i === poIdx && i > 0 ? ' srk-po-start' : '') + '">Wk ' + w +
                    (w === CFG.startWeek ? '<span class="srk-now">Now</span>' : '') + '</th>';
        });
        var totalCols = 3 + weeks.length;
        var body = '';
        sections.forEach(function(sec) {
          if (sec.label) {
            body += '<tr class="srk-group"><td colspan="' + totalCols + '">' +
                      '<span class="srk-group-label">' + esc(sec.label) + '</span>' +
                      '<span class="srk-group-note">' + esc(sec.note) + '</span></td></tr>';
          }
          sec.models.forEach(function(m) { body += tableRow(m, poIdx); });
        });
        return '<table class="sched-table"><thead>' + bandRow(weeks) + '<tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>';
      }

      // ── View assembly ───────────────────────────────────────────────────
      function buildView(results) {
        var weeks = (results[0].pd.main.weeks) || [];
        var sections = [];
        if (currentView === 'league') {
          sections = [{ label: null, note: '', models: leagueModels(results[0].pd) }];
        } else if (rankPos === 'all') {
          var order = {};
          addedPids.forEach(function(p, i) { order[p] = i; });
          var addedAll = [];
          results.forEach(function(res) { addedAll = addedAll.concat(myModels(res.pd).added); });
          addedAll.sort(function(a, b) { return order[a.pid] - order[b.pid]; });
          if (addedAll.length) {
            sections.push({ label: 'Added', note: addedAll.length + (addedAll.length === 1 ? ' player' : ' players'), models: addedAll });
          }
          results.forEach(function(res) {
            var mine = myModels(res.pd).mine;
            if (mine.length) {
              sections.push({ label: res.pos, note: mine.length + (mine.length === 1 ? ' player' : ' players') + ', in ranking order', models: mine });
            }
          });
        } else {
          var mm = myModels(results[0].pd);
          sections = [{ label: null, note: '', models: mm.added.concat(mm.mine) }];
        }
        var total = 0;
        sections.forEach(function(sec) { total += sec.models.length; });
        if (!total) {
          gridEl.innerHTML = currentView === 'league'
            ? '<div class="sched-empty"><strong style="color:var(--text);display:block;margin-bottom:4px;">No data</strong>Nothing for this position or week range.</div>'
            : '<div class="sched-empty"><strong style="color:var(--text);display:block;margin-bottom:4px;">No players to show</strong>No one on your roster lands in this view.</div>';
          return;
        }
        gridEl.innerHTML = buildTable(sections, weeks);
      }

      var renderSeq = 0;
      function render() {
        syncChrome();
        var seq = ++renderSeq;
        var loading = '<div class="loading-state-msg"><div class="loading-spinner" aria-hidden="true"></div><span>Loading rankings…</span></div>';
        gridEl.innerHTML = loading;
        var positions = (currentView === 'myteam' && rankPos === 'all')
          ? POSITIONS
          : [rankPos === 'all' ? 'QB' : rankPos];
        Promise.all(positions.map(function(pos) {
          return posData(pos).then(function(pd) { return { pos: pos, pd: pd }; });
        })).then(function(results) {
          if (seq !== renderSeq) return;
          buildView(results);
        }).catch(function() {
          if (seq !== renderSeq) return;
          var err = '<div class="sched-empty" style="color:var(--loss);"><strong style="display:block;margin-bottom:4px;">Couldn’t load rankings</strong>Refresh and try again.</div>';
          gridEl.innerHTML = err;
        });
      }

      // ── Controls ────────────────────────────────────────────────────────
      function fillWeekSelects() {
        var optsS = '', optsE = '';
        for (var w = 1; w <= CFG.maxWeek; w++) {
          optsS += '<option value="' + w + '"' + (w === wkStart ? ' selected' : '') + '>Week ' + w + '</option>';
          optsE += '<option value="' + w + '"' + (w === wkEnd   ? ' selected' : '') + '>Week ' + w + '</option>';
        }
        startSel.innerHTML = optsS;
        endSel.innerHTML   = optsE;
      }

      function persistWks() {
        try { localStorage.setItem(LS_WKS, JSON.stringify({s: wkStart, e: wkEnd})); } catch (e) {}
      }

      function syncChrome() {
        document.querySelectorAll('.sched-view-btn').forEach(function(b) {
          b.classList.toggle('active', b.getAttribute('data-view') === currentView);
        });
        document.querySelectorAll('.sched-rank-pos').forEach(function(b) {
          b.classList.toggle('active', b.getAttribute('data-pos') === rankPos);
        });
        sortBtn.innerHTML = hardFirst
          ? 'Hardest First <i class="fa-solid fa-arrow-down-short-wide" aria-hidden="true"></i>'
          : 'Easiest First <i class="fa-solid fa-arrow-up-short-wide" aria-hidden="true"></i>';
        var note = document.getElementById('srkScopeNote');
        if (note) note.textContent = currentView === 'league'
          ? 'Every NFL team at the position, with the player who holds the schedule in this league.'
          : (rankPos === 'all'
              ? 'Your full team at once, grouped by position, each group in ranking order.'
              : 'Your players only, same ranking order.');
      }

      var rosBtn     = document.getElementById('schedRosPreset');
      var fullBtn    = document.getElementById('schedFullPreset');
      var playoffBtn = document.getElementById('schedPlayoffPreset');
      function syncPresetBtns() {
        var ps = CFG.playoffStart, pe = CFG.playoffEnd;
        if (rosBtn) rosBtn.classList.toggle('active', wkStart === CFG.startWeek && wkEnd === CFG.playoffEnd);
        if (fullBtn) fullBtn.classList.toggle('active', wkStart === 1 && wkEnd === CFG.maxWeek);
        if (playoffBtn) {
          playoffBtn.classList.toggle('active', wkStart === ps && wkEnd === pe);
          playoffBtn.disabled = ps > pe;
        }
      }
      function applyPreset(s, e) {
        wkStart = s; wkEnd = e;
        if (wkEnd < wkStart) wkEnd = wkStart;
        onRangeChange();
      }
      function onRangeChange() {
        fillWeekSelects(); persistWks(); syncPresetBtns();
        mainCache = {};
        render();
      }
      if (rosBtn) rosBtn.addEventListener('click', function() { applyPreset(CFG.startWeek, CFG.playoffEnd); });
      if (fullBtn) fullBtn.addEventListener('click', function() { applyPreset(1, CFG.maxWeek); });
      if (playoffBtn) playoffBtn.addEventListener('click', function() { applyPreset(CFG.playoffStart, CFG.playoffEnd); });

      startSel.addEventListener('change', function() {
        wkStart = parseInt(this.value, 10);
        if (wkEnd < wkStart) wkEnd = wkStart;
        onRangeChange();
      });
      endSel.addEventListener('change', function() {
        wkEnd = parseInt(this.value, 10);
        if (wkStart > wkEnd) wkStart = wkEnd;
        onRangeChange();
      });

      // View toggle: League resets an All selection back to a single position.
      document.getElementById('schedViewToggle').addEventListener('click', function(e) {
        var btn = e.target.closest ? e.target.closest('.sched-view-btn') : null;
        if (!btn) return;
        currentView = btn.getAttribute('data-view');
        if (currentView === 'league' && rankPos === 'all') rankPos = 'QB';
        render();
      });

      // Position pills: All only exists for My Team, so it flips scope too.
      document.getElementById('schedRankPosPills').addEventListener('click', function(e) {
        var btn = e.target.closest ? e.target.closest('.sched-rank-pos') : null;
        if (!btn) return;
        rankPos = btn.getAttribute('data-pos');
        if (rankPos === 'all') currentView = 'myteam';
        render();
      });

      // Sort toggle re-renders from the cached fetch promises (no refetch).
      sortBtn.addEventListener('click', function() {
        hardFirst = !hardFirst;
        render();
      });

      // Remove a staged player from its table cell.
      function onRemoveClick(e) {
        var btn = e.target.closest ? e.target.closest('[data-remove-pid]') : null;
        if (!btn) return;
        e.stopPropagation();
        var pid = btn.getAttribute('data-remove-pid');
        addedPids = addedPids.filter(function(id) { return id !== pid; });
        syncAddedSet();
        persistAdded(); render();
      }
      gridEl.addEventListener('click', onRemoveClick);

      fillWeekSelects();
      syncPresetBtns();
      render();
    })();
    </script>
    """.replace("__CFG__", cfg)

    if season < active_season:
        source_copy = f"Ratings use final completed {season} games."
    elif current_weight >= 1:
        source_copy = f"Ratings use completed {season} games."
    elif completed_week == 0:
        source_copy = f"{season} ratings use {season - 1} defensive results until {season} games are completed."
    else:
        source_copy = (f"Early-season ratings blend {season - 1} defensive results with completed {season} games. "
                       f"Through Week {completed_week}: {current_weight:.0%} {season} data and {previous_weight:.0%} {season - 1} data.")
    shell = shell.replace("__SOURCE_COPY__", source_copy)

    return shell + script
