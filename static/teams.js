// Teams-page analytics module — extracted from the inline <script> in
// build_teams_body (app.py) so it's cached and minified instead of re-sent
// in the HTML on every teams navigation. Config arrives via window.__teamsCfg
// (set inline just before this loads). Deferred, so app.js globals are ready.

(function () {
    var _cfg = window.__teamsCfg || {};
    var _platform = _cfg.platform;
    var _leagueId = _cfg.leagueId;
    var _season = _cfg.season;
    var _leagueType = _cfg.leagueType;
    var _leagueSize = _cfg.leagueSize;
    var _viewerRosterId = _cfg.viewerRosterId;
    var _offseasonMode = _cfg.offseasonMode;
    var _draftEnded = _cfg.draftEnded;
    var _loaded = {};

    // Soft-nav / league switch can leave this module bound to the previous
    // room. Re-read the inline cfg and drop lazy-load flags when the league
    // identity changes so each panel fetches that league's data.
    function _syncLeagueCfg() {
        var next = window.__teamsCfg || {};
        var same = String(next.leagueId || '') === String(_leagueId || '')
            && String(next.platform || '') === String(_platform || '')
            && String(next.season || '') === String(_season || '');
        if (same) return;
        _cfg = next;
        _platform = next.platform;
        _leagueId = next.leagueId;
        _season = next.season;
        _leagueType = next.leagueType;
        _leagueSize = next.leagueSize;
        _viewerRosterId = next.viewerRosterId;
        _offseasonMode = next.offseasonMode;
        _draftEnded = next.draftEnded;
        _loaded = {};
    }

    function _panelEmpty(panel, title, message, opts) {
        opts = opts || {};
        if (window.brEmptyState) {
            window.brEmptyState(panel, {
                icon: opts.icon || 'empty',
                title: title,
                message: message,
                compact: true,
                error: !!opts.error,
                retry: opts.retry
            });
            return;
        }
        panel.innerHTML = '<div class="analytics-empty">' + (message || title) + '</div>';
    }

    function _panelError(panel, message, retry) {
        if (window.brErrorState) {
            window.brErrorState(panel, message, retry, {compact: true});
            return;
        }
        panel.innerHTML = '<div class="analytics-empty">' + (message || 'Could not load data.') + '</div>';
    }

    function loadBtm() {
        _syncLeagueCfg();
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
                _panelEmpty(panel, 'Couldn’t load', data.error, {error: true, icon: 'error'});
                return;
            }
            var rows = data.rosters || [];
            var avgDelta = data.league_avg_delta || 0;
            var avgSign = avgDelta >= 0 ? '+' : '';
            var avgFmt = avgSign + Math.round(avgDelta).toLocaleString();
            // Largest deviation from the league average — scales the diverging bars.
            var maxAbsVs = rows.reduce(function (m, r) {
                return Math.max(m, Math.abs(r.vs_avg || 0));
            }, 0) || 1;
            var html = '';

            // Header: title + window pills (full mode only)
            if (!slim) {
                html += '<div class="btm-header">' +
                    '<div class="btm-header-text">' +
                    '<span class="btm-title">Value Tracker</span>' +
                    '<span class="btm-subtitle">Which rosters gained the most dynasty value?</span>' +
                    '</div>' +
                    '<div class="btm-window-pills">' +
                    '<button class="btm-pill' + (days === 7 ? ' active' : '') + '" data-days="7">7d</button>' +
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
            rows.forEach(function (r, idx) {
                var pos = r.vs_avg >= 0;
                var cls = pos ? 'btm-pos' : 'btm-neg';
                var pdSign = r.total_delta >= 0 ? '+' : '';
                var vsSign = pos ? '+' : '';

                var rankHtml;
                if (idx === 0) rankHtml = '<span class="btm-rank-badge rk-gold">1</span>';
                else if (idx === 1) rankHtml = '<span class="btm-rank-badge rk-silver">2</span>';
                else if (idx === 2) rankHtml = '<span class="btm-rank-badge rk-bronze">3</span>';
                else rankHtml = '<span class="btm-rank-num">' + (idx + 1) + '</span>';

                var moversHtml = '';
                if (!slim && r.top_movers && r.top_movers.length) {
                    moversHtml = '<div class="btm-movers-row">';
                    r.top_movers.slice(0, 4).forEach(function (m) {
                        var mc = m.delta >= 0 ? 'btm-mover-pos' : 'btm-mover-neg';
                        var arrow = m.delta >= 0 ? '↑' : '↓';
                        var lastName = m.name.split(' ').slice(-1)[0];
                        var dFmt = (m.delta >= 0 ? '+' : '') + Math.round(m.delta);
                        moversHtml += '<span class="btm-mover ' + mc + '" title="' + m.name + ' · ' + m.position + '">' +
                            arrow + ' <strong>' + lastName + '</strong>&nbsp;' + dFmt +
                            '</span>';
                    });
                    moversHtml += '</div>';
                }

                // Diverging bar: fills from the center (league avg) outward — right/green
                // for above-average rosters, left/red for below — scaled to the widest gap.
                var barPct = Math.min(50, Math.round(Math.abs(r.vs_avg || 0) / maxAbsVs * 50));
                var barStyle = (pos ? 'left:50%;' : 'right:50%;') + 'width:' + barPct + '%;';

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
                    '<div class="btm-bar-track"><span class="btm-bar-fill ' + cls + '" style="' + barStyle + '"></span></div>' +
                    '</div>';
            });
            html += '</div>';

            panel.innerHTML = html;

            panel.querySelectorAll('.btm-pill').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    fetchBtm(parseInt(this.getAttribute('data-days')));
                });
            });
        }

        function fetchBtm(days) {
            panel.innerHTML = '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div><div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w60" style="margin-top:10px"></div></div>';
            fetch('/api/beat-the-market?platform=' + _platform +
                '&league_id=' + _leagueId + '&season=' + _season +
                '&league_type=' + _leagueType + '&league_size=' + _leagueSize + '&days=' + days)
                .then(function (r) {
                    return r.json();
                })
                .then(function (data) {
                    renderBtm(data, days);
                })
                .catch(function () {
                    _panelError(panel, 'Could not load data.');
                });
        }

        fetchBtm(30);
    }

    function _sosEsc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
        });
    }

    // Rank 0 = hardest remaining schedule. Spread-based bars so clustered
    // opponent values (typical before week 1) still read as a ranking, not
    // a stack of identical full-width red pills.
    function _sosTier(idx, n, even) {
        if (even) return {key: 'even', label: 'Even'};
        var q = n <= 1 ? 0.5 : idx / (n - 1);
        if (q <= 0.2) return {key: 'hard', label: idx === 0 ? 'Hardest' : 'Hard'};
        if (q <= 0.45) return {key: 'hard', label: 'Hard'};
        if (q <= 0.7) return {key: 'mid', label: 'Average'};
        if (q < 1) return {key: 'easy', label: 'Easy'};
        return {key: 'easy', label: 'Easiest'};
    }

    function _sosBarPct(val, minOpp, spread) {
        if (spread < 0.05) return 48;
        return Math.round(22 + ((val - minOpp) / spread) * 78);
    }

    function renderSos(panel, data) {
        if (data.error) {
            _panelEmpty(panel, 'Couldn’t load', data.error, {error: true, icon: 'error'});
            return;
        }
        var teams = data.teams || [];
        if (!teams.length) {
            _panelEmpty(panel, 'No schedule data', 'Schedule strength will appear once games are available.');
            return;
        }

        var usingPR = !!data.using_power_rankings;
        var usingProj = !!data.using_projections;
        var usingBlend = !!data.using_blend;
        var wr = data.weeks_remaining || 0;
        var values = teams.map(function (t) {
            return Number(t.avg_opp_points) || 0;
        });
        var maxOpp = Math.max.apply(null, values);
        var minOpp = Math.min.apply(null, values);
        var spread = maxOpp - minOpp;
        var even = spread < 0.05;
        var avgOpp = values.reduce(function (a, b) {
            return a + b;
        }, 0) / values.length;
        var n = teams.length;
        var viewerId = _viewerRosterId != null && _viewerRosterId !== '' ? String(_viewerRosterId) : '';
        var schedHref = (_platform && _leagueId && _season)
            ? '/' + _platform + '/' + _season + '/' + _leagueId + '/schedule'
            : '';

        var html = '<div class="sos-panel">';
        html += '<div class="sos-header">' +
            '<div class="sos-header-text">' +
            '<span class="sos-title">Remaining schedule</span>' +
            '</div>';
        html += '</div>';

        html += '<div class="sos-meta">' +
            '<span class="sos-weeks"><strong>' + wr + '</strong> week' + (wr === 1 ? '' : 's') + ' left</span>' +
            '<span class="sos-sort-note">' + (even ? 'Schedules look even' : 'Hardest first') + '</span>' +
            '</div>';

        if (usingProj) {
            html += '<div class="sos-note" role="note">No games played yet — remaining opponents ranked by projected starter scoring.</div>';
        } else if (usingBlend) {
            html += '<div class="sos-note" role="note">Early results are mixed with preseason projections so one week doesn’t flip SOS.</div>';
        } else if (usingPR) {
            html += '<div class="sos-note" role="note">No games played yet — SOS needs scoring and win rate, so remaining schedules look even.</div>';
        }

        html += '<div class="sos-legend" aria-hidden="true">' +
            '<span class="sos-leg sos-leg-hard"><i></i>Hard</span>' +
            '<span class="sos-leg sos-leg-mid"><i></i>Average</span>' +
            '<span class="sos-leg sos-leg-easy"><i></i>Easy</span>' +
            '</div>';

        html += '<div class="sos-cols" aria-hidden="true"><span>#</span><span>Team</span><span>vs Avg</span></div>';
        html += '<div class="sos-list">';

        teams.forEach(function (t, idx) {
            var val = Number(t.avg_opp_points) || 0;
            var tier = _sosTier(idx, n, even);
            var vs = val - avgOpp;
            var vsLbl = (vs >= 0 ? '+' : '') + vs.toFixed(1);
            var mine = viewerId && String(t.roster_id) === viewerId;
            var pct = _sosBarPct(val, minOpp, spread);
            var rankHtml;
            if (idx === 0) rankHtml = '<span class="btm-rank-badge rk-gold">1</span>';
            else if (idx === 1) rankHtml = '<span class="btm-rank-badge rk-silver">2</span>';
            else if (idx === 2) rankHtml = '<span class="btm-rank-badge rk-bronze">3</span>';
            else rankHtml = '<span class="sos-rank-num">' + (idx + 1) + '</span>';

            var tipParts = [t.team_name || ''];
            if ((usingProj || usingBlend || !usingPR) && val) tipParts.push('SOS ' + val.toFixed(1));
            if (t.games_remaining) tipParts.push(t.games_remaining + ' games left');

            html += '<div class="sos-row sos-' + tier.key + (mine ? ' sos-mine' : '') + '"' +
                ' title="' + _sosEsc(tipParts.join(' · ')) + '">' +
                '<div class="sos-rank">' + rankHtml + '</div>' +
                '<div class="sos-body">' +
                '<div class="sos-top">' +
                '<span class="sos-name">' +
                '<span class="sos-name-text">' + _sosEsc(t.team_name) + '</span>' +
                (mine ? '<span class="sos-you">YOU</span>' : '') +
                '</span>' +
                '<span class="sos-diff sos-diff-' + tier.key + '">' + tier.label + '</span>' +
                '</div>' +
                '<div class="sos-track">' +
                '<div class="sos-fill sos-fill-' + tier.key + '" style="width:' + pct + '%"></div>' +
                '</div>' +
                '</div>' +
                '<span class="sos-val">' + vsLbl + '</span>' +
                '</div>';
        });

        html += '</div></div>';
        panel.innerHTML = html;
    }

    function loadSos() {
        _syncLeagueCfg();
        if (_loaded.sos) return;
        _loaded.sos = true;
        var panel = document.getElementById('sosPanel');
        if (!panel) return;
        panel.innerHTML = '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div><div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div><div class="sk-shimmer sk-line sk-line--w60" style="margin-top:10px"></div></div>';
        fetch('/api/schedule-strength?platform=' + _platform +
            '&league_id=' + _leagueId + '&season=' + _season)
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                renderSos(panel, data);
            })
            .catch(function () {
                _panelError(panel, 'Could not load data.');
            });
    }

    function loadRosterIntel() {
        _syncLeagueCfg();
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
            .then(function (r) {
                return r.ok ? r.json() : [];
            })
            .catch(function () {
                return [];
            });

        fcPromise.then(function (fcRaw) {
            // Transform FC array into {sleeperId: {pos_rank, adp_rank, position}}
            var fcAdp = {};
            if (Array.isArray(fcRaw)) {
                var posCounters = {};
                var sorted = fcRaw.filter(function (e) {
                    return e && e.overallRank;
                })
                    .sort(function (a, b) {
                        return a.overallRank - b.overallRank;
                    });
                sorted.forEach(function (entry) {
                    var p = entry.player || {};
                    var sid = String(p.sleeperId || '');
                    if (!sid || sid === 'null' || sid === 'undefined') return;
                    var pos = String(p.position || '').toUpperCase();
                    posCounters[pos] = (posCounters[pos] || 0) + 1;
                    fcAdp[sid] = {
                        adp_rank: entry.overallRank,
                        pos_rank: posCounters[pos],
                        position: pos,
                    };
                });
            }

            return fetch('/api/roster-intel', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    platform: _platform,
                    league_id: _leagueId,
                    season: _season,
                    league_type: _leagueType,
                    viewer_roster_id: _viewerRosterId || '',
                    fc_adp: fcAdp,
                }),
            }).then(function (r) {
                return r.json();
            });
        })
            .then(function (data) {
                if (data.error) {
                    _panelEmpty(panel, 'Couldn’t load', data.error, {error: true, icon: 'error'});
                    return;
                }
                var teams = data.teams || [];
                if (!teams.length) {
                    _panelEmpty(panel, 'No roster data', 'Roster breakdown will appear once teams are loaded.');
                    return;
                }

                var sigColor = {
                    'Core': '#22c55e',
                    'Sell High': '#ef4444',
                    'Breakout': '#8b5cf6',
                    'Sleeper': '#06b6d4',
                    'Monitor': '#f59e0b',
                    'Stash': '#0d9488',
                    'Hold': 'var(--text-muted)',
                    'Cut': '#94a3b8',
                };
                var sigDesc = {
                    'Core': 'Elite, in-prime asset: a keeper.',
                    'Sell High': 'Aging or market-hyped: sell while the value is high.',
                    'Breakout': 'On the Breakout Engine board: hold for the leap.',
                    'Sleeper': 'Valued above the dynasty market: buy or hold.',
                    'Monitor': 'Sharp recent drop: watch before value erodes.',
                    'Stash': 'Young/rookie upside below rosterable depth: hold for later.',
                    'Hold': 'No action needed right now.',
                    'Cut': 'Below rosterable depth or past prime: drop candidate.',
                };
                var healthColor = {
                    'Strong': '#22c55e',
                    'Average': 'var(--text-muted)',
                    'Thin': '#f59e0b',
                    'Aging': '#ef4444',
                };
                var healthDesc = {
                    'Strong': 'Top third of the league at this position.',
                    'Average': 'Middle of the pack for the league.',
                    'Thin': 'Fewer than two starter-grade assets.',
                    'Aging': 'Most of this group is past prime.',
                };
                var posColor = {QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6'};
                var POS_ORDER = ['QB', 'RB', 'WR', 'TE'];
                var esc = function (s) {
                    return (s || '').replace(/"/g, '&quot;');
                };
                var names = function (arr) {
                    return arr.map(function (p) {
                        return p.name;
                    }).join(', ');
                };

                var html = '';
                teams.forEach(function (t) {
                    var positions = t.positions || {};

                    // ── Action summary: pull the flagged players + thin/aging spots to the top ──
                    var buckets = {
                        'Sell High': [],
                        'Cut': [],
                        'Breakout': [],
                        'Sleeper': [],
                        'Stash': [],
                        'Monitor': []
                    };
                    var needs = [];
                    POS_ORDER.forEach(function (pos) {
                        var pd = positions[pos];
                        if (!pd || !pd.players.length) return;
                        pd.players.forEach(function (p) {
                            if (buckets[p.signal]) buckets[p.signal].push(p);
                        });
                        if (pd.health === 'Thin' || pd.health === 'Aging') {
                            // A thin spot that already has a strong anchor (a Core/keeper, or
                            // a top-third league rank at the position) needs depth behind it,
                            // not an upgrade. Only a thin spot with no anchor wants an upgrade.
                            var anchored = (pd.players || []).some(function (p) {
                                    return p.signal === 'Core';
                                })
                                || (!!pd.league_rank && pd.league_rank <= Math.max(1, Math.round((pd.num_teams || 10) * 0.3)));
                            needs.push({pos: pos, health: pd.health, anchored: anchored});
                        }
                    });
                    var moves = [];
                    var addMove = function (tag, color, body) {
                        moves.push('<div class="ri-move"><span class="ri-move-tag" style="background:' + color + '">' + tag +
                            '</span><span class="ri-move-body">' + body + '</span></div>');
                    };
                    if (buckets['Sell High'].length) addMove('Sell high', sigColor['Sell High'], '<b>' + names(buckets['Sell High']) + '</b><span class="why">: sell while the value is high.</span>');
                    if (buckets['Cut'].length) addMove('Cut', '#64748b', '<b>' + names(buckets['Cut']) + '</b><span class="why">: low value; free the bench spot' + (buckets['Cut'].length > 1 ? 's' : '') + '.</span>');
                    if (buckets['Breakout'].length) addMove('Breakout', sigColor['Breakout'], '<b>' + names(buckets['Breakout']) + '</b><span class="why">: breakout upside; hold for the leap.</span>');
                    if (buckets['Sleeper'].length) addMove('Buy / hold', sigColor['Sleeper'], '<b>' + names(buckets['Sleeper']) + '</b><span class="why">: valued above the market.</span>');
                    if (buckets['Stash'].length) addMove('Stash', sigColor['Stash'], '<b>' + names(buckets['Stash']) + '</b><span class="why">: young upside; stash for later.</span>');
                    if (buckets['Monitor'].length) addMove('Monitor', sigColor['Monitor'], '<b>' + names(buckets['Monitor']) + '</b><span class="why">: slipping; watch closely.</span>');
                    if (needs.length) {
                        // Aging -> get younger; thin with an anchor -> add a backup; thin
                        // with no anchor -> shop for an upgrade.
                        var needStr = needs.map(function (n) {
                            var advice = n.health === 'Aging' ? 'get younger'
                                : (n.anchored ? 'add a backup' : 'shop for an upgrade');
                            return '<b>' + n.pos + '</b> is ' + n.health.toLowerCase() + '<span class="why">, ' + advice + '</span>';
                        }).join('; ');
                        addMove('Target', healthColor['Thin'], needStr + '<span class="why">.</span>');
                    }
                    html += '<div class="ri-summary"><div class="ri-summary-eyebrow">Suggested moves</div>' +
                        (moves.length ? moves.join('') : '<div class="ri-summary-stable">Roster looks stable, no moves flagged.</div>') +
                        '</div>';

                    // ── Legend for the signal chips ──
                    html += '<div class="ri-legend">' + ['Core', 'Sell High', 'Breakout', 'Sleeper', 'Stash', 'Monitor', 'Cut'].map(function (k) {
                        return '<span class="ri-legend-item" title="' + esc(sigDesc[k]) + '"><span class="ri-legend-dot" style="background:' + sigColor[k] + '"></span>' + k + '</span>';
                    }).join('') + '</div>';

                    POS_ORDER.forEach(function (pos) {
                        var pd = positions[pos];
                        if (!pd || !pd.players.length) return;

                        var rankStr = pd.league_rank ? (pd.league_rank + '/' + pd.num_teams) : '';
                        var hc = healthColor[pd.health] || 'var(--text-muted)';
                        var maxVal = pd.players.reduce(function (m, p) {
                            return Math.max(m, p.value || 0);
                        }, 0) || 1;
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

                        pd.players.forEach(function (p) {
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

                panel.innerHTML = html || '';
                if (!html) {
                    _panelEmpty(panel, 'Roster looks stable', 'No actions flagged right now.');
                }
            })
            .catch(function (err) {
                console.warn('[roster-intel]', err);
                _panelError(panel, 'Could not load roster intel.');
            });
    }

    // Show the Schedule tab only in-season (Draft + Power Rankings tabs removed).
    (function () {
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
        var card = document.getElementById('teamsAnalyticsCard');
        if (strip) strip.querySelectorAll('.tab-btn').forEach(function (b) {
            b.classList.toggle('active', b.dataset.tab === tab);
        });
        if (card) card.querySelectorAll('.tab-panel').forEach(function (p) {
            p.classList.toggle('active', p.dataset.tab === tab);
        });
        _setActiveTabAttr(tab);
        if (load) {
            if (tab === 'btm') loadBtm();
            if (tab === 'roster-intel') loadRosterIntel();
            if (tab === 'sos') loadSos();
        }
    }

    function wireAnalyticsTabs() {
        var tabs = document.querySelectorAll('#teamsAnalyticsTabs > .tab-btn');
        tabs.forEach(function (btn) {
            btn.addEventListener('click', function () {
                var tab = btn.dataset.tab;
                _setActiveTabAttr(tab);
                if (tab === 'btm') loadBtm();
                if (tab === 'roster-intel') loadRosterIntel();
                if (tab === 'sos') loadSos();
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

    // Auto-open a specific tab when navigated with a hash (e.g. #btm / #roster-intel / #sos)
    function _activateTabFromHash() {
        var hash = window.location.hash.replace('#', '');
        if (!hash) return;
        var btn = document.querySelector('#teamsAnalyticsTabs > .tab-btn[data-tab="' + hash + '"]');
        if (btn) btn.click();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            wireAnalyticsTabs();
            _activateTabFromHash();
        });
    } else {
        wireAnalyticsTabs();
        _activateTabFromHash();
    }
})();
