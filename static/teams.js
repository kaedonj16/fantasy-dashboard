// Teams-page analytics module — extracted from the inline <script> in
// build_teams_body (app.py) so it's cached and minified instead of re-sent
// in the HTML on every teams navigation. Config arrives via window.__teamsCfg
// (set inline just before this loads). Deferred, so app.js globals are ready.

    (function() {
      var _cfg             = window.__teamsCfg || {};
      var _platform        = _cfg.platform;
      var _leagueId        = _cfg.leagueId;
      var _season          = _cfg.season;
      var _leagueType      = _cfg.leagueType;
      var _leagueSize      = _cfg.leagueSize;
      var _viewerRosterId  = _cfg.viewerRosterId;
      var _offseasonMode   = _cfg.offseasonMode;
      var _draftEnded      = _cfg.draftEnded;
      var _loaded          = {};

      // A small right-aligned "view the full page" link for a tab panel.
      function _fullPageLink(label, path) {
        if (!_platform || !_leagueId || !_season) return '';
        var href = '/' + _platform + '/' + _season + '/' + _leagueId + path;
        return '<div style="display:flex;justify-content:flex-end;margin-bottom:10px;">' +
          '<a href="' + href + '" style="font-size:12px;font-weight:600;color:#3b82f6;' +
          'text-decoration:none;white-space:nowrap;">' + label + ' →</a></div>';
      }


      function loadBtm() {
        if (_loaded.btm) return;
        _loaded.btm = true;
        var panel = document.getElementById('btmPanel');
        if (!panel) return;
        // Slim (narrow) rendering only in the desktop sidebar. On mobile the
        // analytics panel spans the full width of the tabbed card, so render
        // the rich full layout (with top movers) there.
        var slim = !!panel.closest('.teams-sidebar') &&
                   window.matchMedia('(min-width: 1181px)').matches;

        function fmtDate(isoStr) {
          if (!isoStr) return '';
          var d = new Date(isoStr + 'T00:00:00');
          return d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
        }

        function renderBtm(data, days) {
          if (data.error) {
            panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>';
            return;
          }
          var rows = data.rosters || [];
          var avgDelta = data.league_avg_delta || 0;
          var avgSign = avgDelta >= 0 ? '+' : '';
          var avgFmt = avgSign + Math.round(avgDelta).toLocaleString();
          var html = '';

          // Header: title + window pills (full mode only)
          if (!slim) {
            html += '<div class="btm-header">' +
              '<div class="btm-header-text">' +
                '<span class="btm-title">Value Tracker</span>' +
                '<span class="btm-subtitle">Which rosters gained the most dynasty value?</span>' +
              '</div>' +
              '<div class="btm-window-pills">' +
                '<button class="btm-pill' + (days === 7  ? ' active' : '') + '" data-days="7">7d</button>' +
                '<button class="btm-pill' + (days === 14 ? ' active' : '') + '" data-days="14">14d</button>' +
                '<button class="btm-pill' + (days === 30 ? ' active' : '') + '" data-days="30">30d</button>' +
                '<button class="btm-pill' + (days === 60 ? ' active' : '') + '" data-days="60">60d</button>' +
              '</div>' +
            '</div>';
          }

          // Meta: date range + league avg
          html += '<div class="btm-meta">' +
            '<span class="btm-date-range">' + fmtDate(data.baseline_date) + ' – ' + fmtDate(data.latest_date) + '</span>' +
            '<span class="btm-league-avg">League Avg: <strong>' + avgFmt + '</strong></span>' +
          '</div>';

          // Column header
          html += '<div class="btm-col-header' + (slim ? ' btm-slim' : '') + '">' +
            '<span></span>' +
            '<span>Team</span>' +
            '<span style="text-align:right;">' + days + 'd</span>' +
            '<span style="text-align:right;">vs Avg</span>' +
          '</div>';

          // Rows
          html += '<div class="btm-rows">';
          rows.forEach(function(r, idx) {
            var pos    = r.vs_avg >= 0;
            var cls    = pos ? 'btm-pos' : 'btm-neg';
            var pdSign = r.total_delta >= 0 ? '+' : '';
            var vsSign = pos ? '+' : '';

            var rankHtml;
            if      (idx === 0) rankHtml = '<span class="btm-rank-badge rk-gold">1</span>';
            else if (idx === 1) rankHtml = '<span class="btm-rank-badge rk-silver">2</span>';
            else if (idx === 2) rankHtml = '<span class="btm-rank-badge rk-bronze">3</span>';
            else                rankHtml = '<span class="btm-rank-num">' + (idx + 1) + '</span>';

            var moversHtml = '';
            if (!slim && r.top_movers && r.top_movers.length) {
              moversHtml = '<div class="btm-movers-row">';
              r.top_movers.slice(0, 4).forEach(function(m) {
                var mc       = m.delta >= 0 ? 'btm-mover-pos' : 'btm-mover-neg';
                var arrow    = m.delta >= 0 ? '↑' : '↓';
                var lastName = m.name.split(' ').slice(-1)[0];
                var dFmt     = (m.delta >= 0 ? '+' : '') + Math.round(m.delta);
                moversHtml  += '<span class="btm-mover ' + mc + '" title="' + m.name + ' · ' + m.position + '">' +
                  arrow + ' <strong>' + lastName + '</strong>&nbsp;' + dFmt +
                '</span>';
              });
              moversHtml += '</div>';
            }

            html += '<div class="btm-row ' + cls + (slim ? ' btm-slim' : '') + '">' +
              '<div class="btm-rank-cell">' + rankHtml + '</div>' +
              '<div class="btm-team-cell">' +
                '<div class="btm-team-name">' + r.team_name + '</div>' +
                moversHtml +
              '</div>' +
              '<div class="btm-change-cell">' +
                '<div class="btm-change-num ' + cls + '">' + pdSign + Math.round(r.total_delta).toLocaleString() + '</div>' +
              '</div>' +
              '<div class="btm-vsavg-cell">' +
                '<span class="btm-vsavg-badge ' + cls + '">' + vsSign + Math.round(r.vs_avg).toLocaleString() + '</span>' +
              '</div>' +
            '</div>';
          });
          html += '</div>';

          panel.innerHTML = html;

          panel.querySelectorAll('.btm-pill').forEach(function(btn) {
            btn.addEventListener('click', function() {
              fetchBtm(parseInt(this.getAttribute('data-days')));
            });
          });
        }

        function fetchBtm(days) {
          panel.innerHTML = '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div><div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w60" style="margin-top:10px"></div></div>';
          fetch('/api/beat-the-market?platform=' + _platform +
                '&league_id=' + _leagueId + '&season=' + _season +
                '&league_type=' + _leagueType + '&league_size=' + _leagueSize + '&days=' + days)
            .then(function(r) { return r.json(); })
            .then(function(data) { renderBtm(data, days); })
            .catch(function() { panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; });
        }

        fetchBtm(30);
      }

      function loadSos() {
        if (_loaded.sos) return;
        _loaded.sos = true;
        var panel = document.getElementById('sosPanel');
        if (!panel) return;
        fetch('/api/schedule-strength?platform=' + _platform +
              '&league_id=' + _leagueId + '&season=' + _season)
          .then(r => r.json())
          .then(data => {
            if (data.error) { panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }
            var teams = data.teams || [];
            if (!teams.length) { panel.innerHTML = '<p class="analytics-empty">No schedule data available.</p>'; return; }
            var usingPR = data.using_power_rankings;
            var maxOpp = Math.max(...teams.map(t => t.avg_opp_points), 1);
            var wr = data.weeks_remaining || 0;
            var sortLabel = usingPR ? 'Based on roster strength (no games played yet)' : 'Sorted by avg opponent score (hardest first)';
            var html = '<div class="analytics-btm-header"><span class="analytics-date-label">Weeks remaining: ' + wr +
              '</span><span class="analytics-avg-label">' + sortLabel + '</span></div>';
            if (usingPR) {
              html += '<p class="analytics-empty" style="margin:4px 0 8px;font-size:12px;color:var(--text-muted)">No games played - opponent strength estimated from roster values.</p>';
            }
            html += '<div class="analytics-bar-list">';
            teams.forEach(function(t) {
              var pct = Math.min(100, Math.round(t.avg_opp_points / maxOpp * 100));
              var cls = t.avg_opp_points >= maxOpp * 0.75 ? 'analytics-bar-neg' :
                        t.avg_opp_points <= maxOpp * 0.5  ? 'analytics-bar-pos' : 'analytics-bar-mid';
              var valLabel = usingPR ? '' : t.avg_opp_points.toFixed(1);
              html += '<div class="analytics-bar-row">' +
                '<span class="analytics-bar-name">' + t.team_name + '</span>' +
                '<div class="analytics-bar-track">' +
                  '<div class="analytics-bar-fill ' + cls + '" style="width:' + pct + '%"></div>' +
                '</div>' +
                '<span class="analytics-bar-val">' + valLabel + '</span>' +
              '</div>';
            });
            html += '</div>';
            panel.innerHTML = html;
          })
          .catch(function() { panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; });
      }

      // For startup drafts, the server-side FC fetch is blocked (403 from server IPs).
      // This function fetches FC dynasty rankings directly from the browser and
      // re-maps each pick's avg_pick, adp_diff, grade, and pos_rank to FC values.
      async function _applyFcStartupAdp(data) {
        var numQbs = _leagueType === 'sf' ? 2 : 1;
        var url = 'https://fantasycalc.com/api/values/current?numQbs=' + numQbs + '&ppr=0.5';
        try {
          var resp = await fetch(url);
          if (!resp.ok) return null;
          var fcData = await resp.json();
          if (!fcData || !fcData.length) return null;

          var fcMap = {};
          var posCounts = {};
          fcData
            .filter(function(e) { return e.overallRank && e.player && e.player.sleeperId; })
            .sort(function(a, b) { return a.overallRank - b.overallRank; })
            .forEach(function(entry) {
              var p   = entry.player;
              var sid = String(p.sleeperId);
              var pos = (p.position || '').toUpperCase();
              posCounts[pos] = (posCounts[pos] || 0) + 1;
              fcMap[sid] = { avg_pick: entry.overallRank, position: pos, pos_rank: posCounts[pos] };
            });

          if (!Object.keys(fcMap).length) return null;

          var nt       = data.num_teams || 10;
          var bigReach = -(nt * 1.1);
          function _regrade(diff) {
            if (diff === null || diff === undefined) return 'N/A';
            if (diff >= 4)        return 'A+';
            if (diff >= 2)        return 'A';
            if (diff >= -3)       return 'B';
            if (diff >= bigReach) return 'C';
            return 'D';
          }
          var gradeVals = {'A+':5,'A':4,'B':3,'C':2,'D':1,'F':0,'N/A':2};

          (data.teams || []).forEach(function(team) {
            (team.picks || []).forEach(function(pick) {
              var fc = fcMap[pick.player_id];
              if (!fc) return;
              pick.avg_pick   = fc.avg_pick;
              pick.adp_diff   = pick.pick_no - fc.avg_pick;
              pick.pos_rank   = fc.pos_rank;
              pick.grade      = _regrade(pick.adp_diff);
              pick.could_wait = pick.adp_diff < -2 && fc.avg_pick > pick.pick_no + nt;
            });
            var gs = (team.picks || []).map(function(p) {
              return gradeVals[p.grade] !== undefined ? gradeVals[p.grade] : 2;
            });
            if (!gs.length) return;
            var avg = gs.reduce(function(a,b) { return a+b; }, 0) / gs.length;
            team.grade = avg>=4.5?'A+':avg>=3.5?'A':avg>=2.5?'B':avg>=1.5?'C':avg>=0.5?'D':'F';
          });

          data.adp_source = 'fantasycalc';
          return data;
        } catch(e) { return null; }
      }

      function loadDraft() {
        if (_loaded.draft) return;
        _loaded.draft = true;
        var panel = document.getElementById('draftPanel');
        if (!panel) return;
        panel.innerHTML = '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div><div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w60" style="margin-top:10px"></div></div>';
        fetch('/api/draft-grades?platform=' + _platform +
              '&league_id=' + _leagueId + '&season=' + _season + '&league_type=' + _leagueType)
          .then(function(r) { return r.json(); })
          .then(async function(data) {
            if (data.error) { panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }

            // For startup drafts, overlay FC dynasty rankings from the browser
            // (server-side fetch is blocked by Cloudflare; browser requests are not)
            if (data.draft_type === 'startup') {
              var upgraded = await _applyFcStartupAdp(data);
              if (upgraded) data = upgraded;
            }

            var teams = data.teams || [];
            if (!teams.length) { panel.innerHTML = '<p class="analytics-empty">No draft data available.</p>'; return; }

            var numTeams    = data.num_teams || 10;
            var totalRounds = data.total_rounds || 1;
            var isStartup   = data.draft_type === 'startup';

            // Build team name lookup: roster_id -> team_name
            var teamNames = {};
            teams.forEach(function(t) { teamNames[t.roster_id] = t.team_name; });

            // Flatten all picks across all teams (for By Round view)
            var allPicks = [];
            teams.forEach(function(t) {
              t.picks.forEach(function(p) {
                allPicks.push(Object.assign({}, p, { _team_name: t.team_name }));
              });
            });
            allPicks.sort(function(a,b) { return a.pick_no - b.pick_no; });

            var chevronSvg = '<svg class="draft-acc-chevron" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6l4 4 4-4"/></svg>';

            // Pick-score color thresholds match psColor() in the Draft Room.
            function psColor(ps) { return ps >= 90 ? '#22c55e' : ps >= 75 ? '#38bdf8' : ps >= 60 ? '#f59e0b' : '#ef4444'; }
            // Team-grade color band — matches the Draft Room League tab (tCol).
            function gradeCol(s) { return s >= 75 ? '#22c55e' : s >= 60 ? '#38bdf8' : s >= 45 ? '#f59e0b' : '#ef4444'; }
            // Numeric pick-score chip, falling back to the letter grade when a
            // pick has no score (e.g. no value data for that player).
            function gradeChip(p) {
              if (p.pick_score != null) {
                var c = psColor(p.pick_score);
                return '<span class="analytics-pick-grade analytics-pick-ps" style="color:' + c + ';border-color:' + c + '66;">' + p.pick_score + '</span>';
              }
              return '<span class="analytics-pick-grade dg-' + p.grade.replace('+', 'plus') + '">' + p.grade + '</span>';
            }

            // ── Shared pick row renderer ─────────────────────────────────────
            function renderPickRow(p, showTeamName) {
              var adpLine = '';
              if (p.avg_pick != null) {
                var diff = p.adp_diff;
                var diffHtml = diff > 1
                  ? '<span class="adp-value">+Steal (' + diff.toFixed(1) + ' vs ADP)</span>'
                  : diff < -1
                    ? '<span class="adp-reach">Reach (' + diff.toFixed(1) + ' vs ADP)</span>'
                    : '<span class="adp-neutral">on ADP</span>';
                var posTag = p.pos_rank != null ? ' · ' + p.position + p.pos_rank : '';
                var waitTag = p.could_wait ? ' <span class="adp-wait">Reach</span>' : '';
                var rankAdp = isStartup || data.draft_type === 'redraft';
                var adpStr = rankAdp ? ('#' + Math.round(p.avg_pick)) : p.avg_pick.toFixed(2);
                adpLine = '<div class="analytics-pick-adp-line">ADP ' + adpStr + posTag + ' ' + diffHtml + waitTag + '</div>';
              }

              var bpaLine = '';
              if (p.bpa && p.bpa.length) {
                var bpaNames = p.bpa.map(function(b) {
                  var posRank = b.pos_rank != null ? b.pos_rank : '';
                  // Show first-initial + last name so "Isaiah Likely" renders as "I. Likely"
                  var parts = (b.name || '').split(' ');
                  var suffixRe = /^(jr\.?|sr\.?|ii|iii|iv|v)$/i;
                  var suffix = parts.length > 1 && suffixRe.test(parts[parts.length - 1]) ? ' ' + parts[parts.length - 1] : '';
                  var coreParts = suffix ? parts.slice(0, -1) : parts;
                  var displayName = coreParts.length > 1
                    ? coreParts[0][0] + '. ' + coreParts[coreParts.length - 1] + suffix
                    : b.name;
                  return '<span class="bpa-name pos-' + (b.position || '').toLowerCase() + '">' +
                    displayName + ' (' + (b.position || '') + posRank + ')</span>';
                }).join(' ');
                bpaLine = '<div class="analytics-pick-bpa">Available: ' + bpaNames + '</div>';
              }

              var needBadge = p.need ? ' <span class="draft-need-badge">Need</span>' : '';
              var teamTag = showTeamName
                ? '<span class="draft-pick-team-tag">' + (p._team_name || '') + '</span>'
                : '';

              return '<div class="analytics-pick-row">' +
                '<span class="analytics-pick-num">#' + p.pick_no + '</span>' +
                gradeChip(p) +
                '<div class="analytics-pick-info">' +
                  '<div class="analytics-pick-name">' + p.name +
                    ' <span class="analytics-pick-pos pos-' + (p.position || '').toLowerCase() + '">' + (p.position || '') + '</span>' +
                    needBadge + teamTag +
                  '</div>' +
                  adpLine +
                  bpaLine +
                '</div>' +
              '</div>';
            }

            // ── Build "By Team" accordion HTML ───────────────────────────────
            function buildByTeamHtml() {
              var html = '<div class="draft-accordion">';
              teams.forEach(function(t, idx) {
                var teamChip;
                if (t.team_grade_letter != null && t.team_grade_score != null) {
                  // Draft-Room-aligned letter grade (same composite + field curve
                  // the Draft Room League tab shows), colored by its score band.
                  var gc = gradeCol(t.team_grade_score);
                  var psTitle = (t.avg_pick_score != null) ? (' · avg pick score ' + t.avg_pick_score) : '';
                  teamChip = '<span class="draft-acc-grade dg-' + t.team_grade_letter.replace('+', 'plus').replace('-', 'minus') + '" style="color:' + gc + ';border-color:' + gc + '66;" title="Draft grade ' + t.team_grade_letter + ' (' + t.team_grade_score + '/100)' + psTitle + '">' + t.team_grade_letter + '</span>';
                } else if (t.avg_pick_score != null) {
                  var tc = psColor(t.avg_pick_score);
                  teamChip = '<span class="draft-acc-grade analytics-pick-ps" style="color:' + tc + ';border-color:' + tc + '66;">' + t.avg_pick_score + '</span>';
                } else {
                  teamChip = '<span class="draft-acc-grade dg-' + t.grade.replace('+', 'plus') + '">' + t.grade + '</span>';
                }
                html += '<div class="draft-acc-item' + (idx === 0 ? ' open' : '') + '">' +
                  '<button class="draft-acc-header" type="button">' +
                    '<span class="draft-acc-name">' + t.team_name + '</span>' +
                    '<div class="draft-acc-right">' +
                      teamChip +
                      chevronSvg +
                    '</div>' +
                  '</button>' +
                  '<div class="draft-acc-body"><div class="draft-acc-picks">';
                t.picks.forEach(function(p) { html += renderPickRow(p, false); });
                html += '</div></div></div>';
              });
              html += '</div>';
              return html;
            }

            // ── By Round state & renderer ────────────────────────────────────
            var currentRound = 1;
            var roundContainerId = 'draftRoundView_' + Date.now();

            function buildByRoundHtml(round) {
              var roundPicks = allPicks.filter(function(p) { return p.round === round; });
              var ordinals = ['','1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
                              '11th','12th','13th','14th','15th'];
              var label = (ordinals[round] || (round + 'th')) + ' Round';

              var prevDis = round <= 1 ? ' disabled' : '';
              var nextDis = round >= totalRounds ? ' disabled' : '';

              var html = '<div class="draft-round-nav">' +
                '<button class="pagination-btn"' + prevDis + ' id="draftRoundPrev">&#8592; Prev</button>' +
                '<span class="draft-round-label">' + label + '</span>' +
                '<button class="pagination-btn"' + nextDis + ' id="draftRoundNext">Next &#8594;</button>' +
              '</div>' +
              '<div class="draft-acc-picks">';

              if (!roundPicks.length) {
                html += '<p class="analytics-empty" style="padding:12px;">No picks recorded for this round yet.</p>';
              } else {
                roundPicks.forEach(function(p) { html += renderPickRow(p, true); });
              }
              html += '</div>';
              return html;
            }

            function renderRoundView(container, round) {
              container.innerHTML = buildByRoundHtml(round);
              var prev = container.querySelector('#draftRoundPrev');
              var next = container.querySelector('#draftRoundNext');
              if (prev) prev.addEventListener('click', function() {
                if (currentRound > 1) { currentRound--; renderRoundView(container, currentRound); }
              });
              if (next) next.addEventListener('click', function() {
                if (currentRound < totalRounds) { currentRound++; renderRoundView(container, currentRound); }
              });
            }

            // ── Wire up tabs and render ──────────────────────────────────────
            var tabsHtml =
              '<div class="draft-view-tabs br-chip-pop">' +
                '<button class="draft-view-tab active" data-view="team">By Team</button>' +
                '<button class="draft-view-tab" data-view="round">By Round</button>' +
              '</div>' +
              '<div id="draftTeamView">' + buildByTeamHtml() + '</div>' +
              '<div id="draftRoundView" style="display:none;"></div>';

            panel.innerHTML = _fullPageLink('Draft history', '/draft/history') + tabsHtml;

            // Accordion toggle for By Team view
            panel.querySelectorAll('.draft-acc-header').forEach(function(btn) {
              btn.addEventListener('click', function() {
                var item = this.closest('.draft-acc-item');
                var wasOpen = item.classList.contains('open');
                panel.querySelectorAll('.draft-acc-item').forEach(function(el) {
                  el.classList.remove('open');
                });
                if (!wasOpen) item.classList.add('open');
              });
            });

            // Tab switching
            var roundViewEl = panel.querySelector('#draftRoundView');
            var teamViewEl  = panel.querySelector('#draftTeamView');
            var roundRendered = false;
            panel.querySelectorAll('.draft-view-tab').forEach(function(tab) {
              tab.addEventListener('click', function() {
                panel.querySelectorAll('.draft-view-tab').forEach(function(t) { t.classList.remove('active'); });
                this.classList.add('active');
                if (this.dataset.view === 'round') {
                  teamViewEl.style.display = 'none';
                  roundViewEl.style.display = '';
                  if (!roundRendered) {
                    roundRendered = true;
                    renderRoundView(roundViewEl, currentRound);
                  }
                } else {
                  roundViewEl.style.display = 'none';
                  teamViewEl.style.display = '';
                }
              });
            });
          })
          .catch(function() { panel.innerHTML = '<p class="analytics-empty">Could not load data.</p>'; });
      }

      function loadRosterIntel() {
        if (_loaded.rosterIntel) return;
        _loaded.rosterIntel = true;
        var panel = document.getElementById('rosterIntelPanel');
        if (!panel) return;

        // Fetch FC dynasty ADP from the browser (server IP is blocked by FC).
        // Gracefully degrade to empty dict if FC is unavailable.
        var numQbs = _leagueType === 'sf' ? 2 : 1;
        var fcPromise = fetch(
          'https://fantasycalc.com/api/values/current?numQbs=' + numQbs + '&ppr=0.5',
          {credentials: 'omit'}
        )
          .then(function(r) { return r.ok ? r.json() : []; })
          .catch(function() { return []; });

        fcPromise.then(function(fcRaw) {
          // Transform FC array into {sleeperId: {pos_rank, adp_rank, position}}
          var fcAdp = {};
          if (Array.isArray(fcRaw)) {
            var posCounters = {};
            var sorted = fcRaw.filter(function(e) { return e && e.overallRank; })
                              .sort(function(a, b) { return a.overallRank - b.overallRank; });
            sorted.forEach(function(entry) {
              var p = entry.player || {};
              var sid = String(p.sleeperId || '');
              if (!sid || sid === 'null' || sid === 'undefined') return;
              var pos = String(p.position || '').toUpperCase();
              posCounters[pos] = (posCounters[pos] || 0) + 1;
              fcAdp[sid] = {
                adp_rank: entry.overallRank,
                pos_rank:  posCounters[pos],
                position:  pos,
              };
            });
          }

          return fetch('/api/roster-intel', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              platform:         _platform,
              league_id:        _leagueId,
              season:           _season,
              league_type:      _leagueType,
              viewer_roster_id: _viewerRosterId || '',
              fc_adp:           fcAdp,
            }),
          }).then(function(r) { return r.json(); });
        })
          .then(function(data) {
            if (data.error) { panel.innerHTML = '<p class="analytics-empty">' + data.error + '</p>'; return; }
            var teams = data.teams || [];
            if (!teams.length) { panel.innerHTML = '<p class="analytics-empty">No roster data available.</p>'; return; }

            var sigColor = {
              'Core':      '#22c55e',
              'Sell High': '#ef4444',
              'Breakout':  '#8b5cf6',
              'Sleeper':   '#06b6d4',
              'Monitor':   '#f59e0b',
              'Stash':     '#0d9488',
              'Hold':      'var(--text-muted)',
              'Cut':       '#94a3b8',
            };
            var sigDesc = {
              'Core':      'Elite, in-prime asset — a keeper.',
              'Sell High': 'Aging or market-hyped — sell while the value is high.',
              'Breakout':  'On the Breakout Engine board — hold for the leap.',
              'Sleeper':   'Valued above the dynasty market — buy or hold.',
              'Monitor':   'Sharp recent drop — watch before value erodes.',
              'Stash':     'Young/rookie upside below rosterable depth — hold for later.',
              'Hold':      'No action needed right now.',
              'Cut':       'Below rosterable depth or past prime — drop candidate.',
            };
            var healthColor = {
              'Strong':  '#22c55e',
              'Average': 'var(--text-muted)',
              'Thin':    '#f59e0b',
              'Aging':   '#ef4444',
            };
            var healthDesc = {
              'Strong':  'Top third of the league at this position.',
              'Average': 'Middle of the pack for the league.',
              'Thin':    'Fewer than two starter-grade assets.',
              'Aging':   'Most of this group is past prime.',
            };
            var posColor = { QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6' };
            var POS_ORDER = ['QB', 'RB', 'WR', 'TE'];
            var esc = function(s) { return (s || '').replace(/"/g, '&quot;'); };
            var names = function(arr) { return arr.map(function(p) { return p.name; }).join(', '); };

            var html = '';
            teams.forEach(function(t) {
              var positions = t.positions || {};

              // ── Action summary: pull the flagged players + thin/aging spots to the top ──
              var buckets = { 'Sell High': [], 'Cut': [], 'Breakout': [], 'Sleeper': [], 'Stash': [], 'Monitor': [] };
              var needs = [];
              POS_ORDER.forEach(function(pos) {
                var pd = positions[pos];
                if (!pd || !pd.players.length) return;
                pd.players.forEach(function(p) { if (buckets[p.signal]) buckets[p.signal].push(p); });
                if (pd.health === 'Thin' || pd.health === 'Aging') needs.push({ pos: pos, health: pd.health });
              });
              var moves = [];
              var addMove = function(tag, color, body) {
                moves.push('<div class="ri-move"><span class="ri-move-tag" style="background:' + color + '">' + tag +
                  '</span><span class="ri-move-body">' + body + '</span></div>');
              };
              if (buckets['Sell High'].length) addMove('Sell high', sigColor['Sell High'], '<b>' + names(buckets['Sell High']) + '</b> <span class="why">— sell while the value is high.</span>');
              if (buckets['Cut'].length)       addMove('Cut', '#64748b', '<b>' + names(buckets['Cut']) + '</b> <span class="why">— low value; free the bench spot' + (buckets['Cut'].length > 1 ? 's' : '') + '.</span>');
              if (buckets['Breakout'].length)  addMove('Breakout', sigColor['Breakout'], '<b>' + names(buckets['Breakout']) + '</b> <span class="why">— breakout upside; hold for the leap.</span>');
              if (buckets['Sleeper'].length)   addMove('Buy / hold', sigColor['Sleeper'], '<b>' + names(buckets['Sleeper']) + '</b> <span class="why">— valued above the market.</span>');
              if (buckets['Stash'].length)     addMove('Stash', sigColor['Stash'], '<b>' + names(buckets['Stash']) + '</b> <span class="why">— young upside; stash for later.</span>');
              if (buckets['Monitor'].length)   addMove('Monitor', sigColor['Monitor'], '<b>' + names(buckets['Monitor']) + '</b> <span class="why">— slipping; watch closely.</span>');
              if (needs.length) {
                var needStr = needs.map(function(n) { return '<b>' + n.pos + '</b> is ' + n.health.toLowerCase(); }).join(', ');
                addMove('Target', healthColor['Thin'], needStr + ' <span class="why">— shop for an upgrade.</span>');
              }
              html += '<div class="ri-summary"><div class="ri-summary-eyebrow">Suggested moves</div>' +
                (moves.length ? moves.join('') : '<div class="ri-summary-stable">Roster looks stable — no moves flagged.</div>') +
                '</div>';

              // ── Legend for the signal chips ──
              html += '<div class="ri-legend">' + ['Core', 'Sell High', 'Breakout', 'Sleeper', 'Stash', 'Monitor', 'Cut'].map(function(k) {
                return '<span class="ri-legend-item" title="' + esc(sigDesc[k]) + '"><span class="ri-legend-dot" style="background:' + sigColor[k] + '"></span>' + k + '</span>';
              }).join('') + '</div>';

              POS_ORDER.forEach(function(pos) {
                var pd = positions[pos];
                if (!pd || !pd.players.length) return;

                var rankStr = pd.league_rank ? (pd.league_rank + '/' + pd.num_teams) : '';
                var hc = healthColor[pd.health] || 'var(--text-muted)';
                var maxVal = pd.players.reduce(function(m, p) { return Math.max(m, p.value || 0); }, 0) || 1;
                var pc = posColor[pos] || 'var(--text-muted)';

                html += '<div class="ri-pos-section">' +
                  '<div class="ri-pos-header">' +
                    '<span class="ri-pos-label">' + pos + '</span>' +
                    '<div class="ri-pos-stats">' +
                      '<span>' + pd.player_count + ' player' + (pd.player_count !== 1 ? 's' : '') + '</span>' +
                      (pd.avg_age ? '<span>Avg ' + pd.avg_age + ' yrs</span>' : '') +
                      (rankStr ? '<span>Rank ' + rankStr + '</span>' : '') +
                    '</div>' +
                    '<span class="ri-health-badge" style="color:' + hc + ';" title="' + esc(healthDesc[pd.health] || '') + '">' + pd.health + '</span>' +
                  '</div>';

                pd.players.forEach(function(p) {
                  // Market context note: show FC ADP divergence if notable
                  var mktNote = '';
                  if (p.mkt_gap !== null && p.mkt_gap !== undefined && Math.abs(p.mkt_gap) >= 4 && p.fc_pos_rank) {
                    var mktDir = p.mkt_gap > 0 ? 'mkt ↑' : 'mkt ↓';
                    var mktCol = p.mkt_gap > 0 ? '#ef4444' : '#06b6d4';
                    mktNote = ' <span style="font-size:10px;color:' + mktCol + ';margin-left:4px;">' + mktDir + '</span>';
                  }
                  var sc = sigColor[p.signal] || 'var(--text-muted)';
                  var metaParts = [];
                  if (p.pos_rank_label) metaParts.push(p.pos_rank_label);
                  if (p.age) metaParts.push('Age ' + parseFloat(p.age).toFixed(1));
                  if (p.fc_pos_rank) metaParts.push('FC ' + pos + p.fc_pos_rank);
                  var safeName = esc(p.name);
                  var barPct = Math.max(4, Math.round((p.value || 0) / maxVal * 100));
                  html += '<div class="ri-player-row">' +
                    '<div class="ri-player-info">' +
                      '<span class="ri-player-name player-clickable" style="cursor:pointer;" data-player-id="' + (p.player_id || '') + '" data-player-name="' + safeName + '">' + p.name + mktNote + '</span>' +
                      '<span class="ri-player-meta">' + metaParts.join(' · ') + '</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">' +
                      '<span class="ri-signal" style="color:' + sc + ';background:color-mix(in srgb,' + sc + ' 15%,transparent);" title="' + esc(sigDesc[p.signal] || '') + '">' + p.signal + '</span>' +
                      '<span class="ri-val"><span class="ri-val-bar" style="width:' + barPct + '%;background:' + pc + ';"></span><span class="ri-val-num">' + (p.value || 0) + '</span></span>' +
                    '</div>' +
                  '</div>';
                });

                html += '</div>';
              });
            });

            panel.innerHTML = html || '<p class="analytics-empty">Roster looks stable - no actions flagged.</p>';
          })
          .catch(function(err) {
            console.warn('[roster-intel]', err);
            panel.innerHTML = '<p class="analytics-empty">Could not load roster intel.</p>';
          });
      }

      function loadPowerRankings() {
        if (_loaded.powerRankings) return;
        _loaded.powerRankings = true;
        var panel = document.getElementById('powerRankingsPanel');
        if (!panel) return;
        fetch('/api/power-rankings', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({platform: _platform, league_id: _leagueId, season: _season})
        })
          .then(r => r.json())
          .then(data => {
            if (!data.success) { panel.innerHTML = '<p class="analytics-empty">' + (data.error || 'Failed to load.') + '</p>'; return; }
            panel.innerHTML = _fullPageLink('Full standings', '/standings') +
              (data.html || '<p class="analytics-empty">No rankings available.</p>');
          })
          .catch(function() { panel.innerHTML = '<p class="analytics-empty">Could not load power rankings.</p>'; });
      }

      // Show the Schedule tab only in-season (Draft + Power Rankings tabs removed).
      (function() {
        var sosBtn = document.getElementById('sosTabBtn');
        // Schedule: visible when not in pure offseason (in-season or preseason)
        if (sosBtn && !_offseasonMode) sosBtn.style.display = '';
      })();

      // Wire data-loading onto the tab buttons; the active-panel toggling itself
      // is handled by initCardTabs (app.js). We also mirror the active tab onto
      // the page container's data-active-tab so the mobile CSS knows whether to
      // show the team grid ("teams") or an analytics panel.
      function _setActiveTabAttr(tab) {
        var layout = document.getElementById('teamsPageLayout');
        if (layout) layout.dataset.activeTab = tab;
      }

      // Directly toggle the active tab/panel (mirrors initCardTabs) plus the
      // data-active-tab attr and lazy load. Used for the viewport default so it
      // doesn't depend on initCardTabs having bound yet (soft-nav ordering).
      function _activateTab(tab, load) {
        var strip = document.getElementById('teamsAnalyticsTabs');
        var card  = document.getElementById('teamsAnalyticsCard');
        if (strip) strip.querySelectorAll('.tab-btn').forEach(function(b) {
          b.classList.toggle('active', b.dataset.tab === tab);
        });
        if (card) card.querySelectorAll('.tab-panel').forEach(function(p) {
          p.classList.toggle('active', p.dataset.tab === tab);
        });
        _setActiveTabAttr(tab);
        if (load) {
          if (tab === 'btm')          loadBtm();
          if (tab === 'roster-intel') loadRosterIntel();
          if (tab === 'sos')          loadSos();
        }
      }

      function wireAnalyticsTabs() {
        var tabs = document.querySelectorAll('#teamsAnalyticsTabs > .tab-btn');
        tabs.forEach(function(btn) {
          btn.addEventListener('click', function() {
            var tab = btn.dataset.tab;
            _setActiveTabAttr(tab);
            if (tab === 'btm')          loadBtm();
            if (tab === 'roster-intel') loadRosterIntel();
            if (tab === 'sos')          loadSos();
          });
        });

        // Default tab depends on viewport. On desktop the team grid is always
        // visible in the main column, so the sidebar defaults to "Value". On
        // mobile the single tabbed card opens on "Teams" (the grid). The server
        // renders with "Teams" active for the mobile-first default; promote to
        // Value on desktop here.
        if (window.matchMedia('(min-width: 1181px)').matches) {
          _activateTab('btm', true);
        }
      }

      // Auto-open a specific tab when navigated with a hash (e.g. #power-rankings from nav)
      function _activateTabFromHash() {
        var hash = window.location.hash.replace('#', '');
        if (!hash) return;
        var btn = document.querySelector('#teamsAnalyticsTabs > .tab-btn[data-tab="' + hash + '"]');
        if (btn) btn.click();
      }

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { wireAnalyticsTabs(); _activateTabFromHash(); });
      } else {
        wireAnalyticsTabs();
        _activateTabFromHash();
      }
    })();
