// Teams-page analytics module -- extracted from the inline <script> in
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
            // Largest deviation from the league average -- scales the diverging bars.
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

            // Meta: date range + league avg (Mock 4 ranked-list header)
            var avgCls = avgDelta >= 0 ? 'pos-chg' : 'neg-chg';
            html += '<div class="rl-head">' +
                '<span class="rng">' + fmtDate(data.baseline_date) + ' – ' + fmtDate(data.latest_date) + '</span>' +
                '<span class="avg">League Avg: <b class="' + avgCls + '">' + avgFmt + '</b></span>' +
                '</div>';

            // Column header
            html += '<div class="rl-cols" aria-hidden="true">' +
                '<span class="c-team">TEAM</span>' +
                '<span class="c-30d">' + days + 'D</span>' +
                '<span class="c-vs">VS AVG</span>' +
                '</div>';

            // Rows
            html += '<div class="rl-rows' + (slim ? ' rl-slim' : '') + '">';
            rows.forEach(function (r, idx) {
                var pos = r.vs_avg >= 0;
                var chgCls = r.total_delta >= 0 ? 'pos-chg' : 'neg-chg';
                var pdSign = r.total_delta >= 0 ? '+' : '';
                var vsSign = pos ? '+' : '';
                var rkCls = idx === 0 ? 't1' : (idx === 1 ? 't2' : (idx === 2 ? 't3' : ''));
                var mine = _viewerRosterId && String(r.roster_id) === String(_viewerRosterId);

                var moversHtml = '';
                if (!slim && r.top_movers && r.top_movers.length) {
                    moversHtml = '<div class="btm-movers-row">';
                    r.top_movers.slice(0, 4).forEach(function (m) {
                        var mc = m.delta >= 0 ? 'btm-mover-pos' : 'btm-mover-neg';
                        var arrow = m.delta >= 0 ? '↑' : '↓';
                        var lastName = m.name.split(' ').slice(-1)[0];
                        var dFmt = (m.delta >= 0 ? '+' : '') + Math.round(m.delta);
                        moversHtml += '<span class="btm-mover ' + mc + '" title="' + _sosEsc(m.name) + ' · ' + _sosEsc(m.position) + '">' +
                            arrow + ' <strong>' + _sosEsc(lastName) + '</strong>&nbsp;' + dFmt +
                            '</span>';
                    });
                    moversHtml += '</div>';
                }

                // Diverging bar under the team name: fills from the center
                // (league avg) outward -- right/green for above-average
                // rosters, left/red for below -- scaled to the widest gap.
                var barPct = Math.min(50, Math.round(Math.abs(r.vs_avg || 0) / maxAbsVs * 50));
                var barStyle = (pos ? 'left:50%;' : 'right:50%;') + 'width:' + barPct + '%;';

                html += '<div class="rl-row' + (mine ? ' rl-mine' : '') + '">' +
                    '<div class="rk ' + rkCls + '">' + (idx + 1) + '</div>' +
                    '<div class="nm"><span class="rl-nm-text">' + _sosEsc(r.team_name) + '</span>' +
                    (mine ? '<span class="rl-you">YOU</span>' : '') +
                    moversHtml +
                    '<div class="vbar"><span class="vbar-mid"></span><em class="' + (pos ? 'gpos' : 'gneg') + '" style="' + barStyle + '"></em></div></div>' +
                    '<div class="chg ' + chgCls + '">' + pdSign + Math.round(r.total_delta).toLocaleString() + '</div>' +
                    '<div class="vs"><span class="' + (pos ? 'pill-pos' : 'pill-neg') + '">' + vsSign + Math.round(r.vs_avg).toLocaleString() + '</span></div>' +
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

    // Difficulty chip classes keyed off the tier: HARDEST/HARD/AVERAGE/EASY/EASIEST.
    function _sosDiffCls(tier, idx, n) {
        if (tier.key === 'even') return 'd-avg';
        if (tier.key === 'hard') return idx === 0 ? 'd-hardest' : 'd-hard';
        if (tier.key === 'easy') return idx === n - 1 ? 'd-easiest' : 'd-easy';
        return 'd-avg';
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

        var html = '<div class="rl-head">' +
            '<span class="rng"><b>' + wr + '</b> week' + (wr === 1 ? '' : 's') + ' left</span>' +
            '<span class="avg">' + (even ? 'Schedules look even' : 'Hardest first') + '</span>' +
            '</div>';

        if (usingProj) {
            html += '<div class="sos-note" role="note">No games played yet -- remaining opponents ranked by projected starter scoring.</div>';
        } else if (usingBlend) {
            html += '<div class="sos-note" role="note">Early results are mixed with preseason projections so one week doesn’t flip SOS.</div>';
        } else if (usingPR) {
            html += '<div class="sos-note" role="note">No games played yet -- SOS needs scoring and win rate, so remaining schedules look even.</div>';
        }

        html += '<div class="slegend" aria-hidden="true">' +
            '<span><i class="sleg-hard"></i>Hard</span>' +
            '<span><i class="sleg-avg"></i>Average</span>' +
            '<span><i class="sleg-easy"></i>Easy</span>' +
            '</div>';

        html += '<div class="rl-rows">';
        teams.forEach(function (t, idx) {
            var val = Number(t.avg_opp_points) || 0;
            var tier = _sosTier(idx, n, even);
            // Above-average opponent scoring = a harder remaining schedule.
            var hard = (val - avgOpp) >= 0;
            var vsLbl = (hard ? '+' : '') + (val - avgOpp).toFixed(1);
            var mine = viewerId && String(t.roster_id) === viewerId;
            var pct = _sosBarPct(val, minOpp, spread);
            var rkCls = idx === 0 ? 't1' : (idx === 1 ? 't2' : (idx === 2 ? 't3' : ''));

            var tipParts = [t.team_name || ''];
            if ((usingProj || usingBlend || !usingPR) && val) tipParts.push('SOS ' + val.toFixed(1));
            if (t.games_remaining) tipParts.push(t.games_remaining + ' games left');

            html += '<div class="rl-row' + (mine ? ' rl-mine' : '') + '"' +
                ' title="' + _sosEsc(tipParts.join(' · ')) + '">' +
                '<div class="rk ' + rkCls + '">' + (idx + 1) + '</div>' +
                '<div class="nm"><span class="rl-nm-text">' + _sosEsc(t.team_name) + '</span>' +
                (mine ? '<span class="rl-you">YOU</span>' : '') +
                '<div class="vbar"><em class="' + (hard ? 'gneg' : 'gpos') + '" style="left:0;width:' + pct + '%;"></em></div></div>' +
                '<div class="vs sos-vs"><span class="diff ' + _sosDiffCls(tier, idx, n) + '">' + tier.label.toUpperCase() + '</span>' +
                '<b class="' + (hard ? 'neg-chg' : 'pos-chg') + '">' + vsLbl + '</b></div>' +
                '</div>';
        });
        html += '</div>';
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

    // ── Roster-intel shared builders (sidebar tab + team drawer) ──
    var _RI_SIG_COLOR = {
        'Core': '#22c55e',
        'Sell High': '#ef4444',
        'Breakout': '#8b5cf6',
        'Sleeper': '#06b6d4',
        'Monitor': '#f59e0b',
        'Stash': '#0d9488',
        'Hold': 'var(--text-muted)',
        'Cut': '#94a3b8',
    };
    var _RI_HEALTH_COLOR = {
        'Strong': '#22c55e',
        'Average': 'var(--text-muted)',
        'Thin': '#f59e0b',
        'Aging': '#ef4444',
    };
    var _RI_POS_ORDER = ['QB', 'RB', 'WR', 'TE'];

    // Cached roster-intel fetch shared by the sidebar tab and the team drawer.
    // Keyed on the league identity so a league switch re-fetches.
    var _riDataPromise = null;
    var _riDataKey = '';
    function _riFetchData() {
        _syncLeagueCfg();
        var key = [_platform, _leagueId, _season, _leagueType, _viewerRosterId].join('|');
        if (!_riDataPromise || _riDataKey !== key) {
            _riDataKey = key;
            // Fetch FC dynasty ADP from the browser (server IP is blocked by FC).
            // Gracefully degrade to empty dict if FC is unavailable.
            var numQbs = _leagueType === 'sf' ? 2 : 1;
            _riDataPromise = fetch(
                'https://fantasycalc.com/api/values/current?numQbs=' + numQbs + '&ppr=0.5',
                {credentials: 'omit'}
            )
                .then(function (r) {
                    return r.ok ? r.json() : [];
                })
                .catch(function () {
                    return [];
                })
                .then(function (fcRaw) {
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
                            var pl = entry.player || {};
                            var sid = String(pl.sleeperId || '');
                            if (!sid || sid === 'null' || sid === 'undefined') return;
                            var pos = String(pl.position || '').toUpperCase();
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
                });
        }
        return _riDataPromise;
    }

    // Suggested-moves summary for one intel team (same math as the sidebar).
    function _riSuggestedMoves(t) {
        var positions = t.positions || {};
        var buckets = {
            'Sell High': [],
            'Cut': [],
            'Breakout': [],
            'Sleeper': [],
            'Stash': [],
            'Monitor': []
        };
        var needs = [];
        var names = function (arr) {
            return arr.map(function (pl) {
                return pl.name;
            }).join(', ');
        };
        _RI_POS_ORDER.forEach(function (pos) {
            var pd = positions[pos];
            if (!pd || !pd.players.length) return;
            pd.players.forEach(function (pl) {
                if (buckets[pl.signal]) buckets[pl.signal].push(pl);
            });
            if (pd.health === 'Thin' || pd.health === 'Aging') {
                // A thin spot that already has a strong anchor (a Core/keeper, or
                // a top-third league rank at the position) needs depth behind it,
                // not an upgrade. Only a thin spot with no anchor wants an upgrade.
                var anchored = (pd.players || []).some(function (pl) {
                        return pl.signal === 'Core';
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
        if (buckets['Sell High'].length) addMove('Sell high', _RI_SIG_COLOR['Sell High'], '<b>' + names(buckets['Sell High']) + '</b><span class="why">: sell while the value is high.</span>');
        if (buckets['Cut'].length) addMove('Cut', '#64748b', '<b>' + names(buckets['Cut']) + '</b><span class="why">: low value; free the bench spot' + (buckets['Cut'].length > 1 ? 's' : '') + '.</span>');
        if (buckets['Breakout'].length) addMove('Breakout', _RI_SIG_COLOR['Breakout'], '<b>' + names(buckets['Breakout']) + '</b><span class="why">: breakout upside; hold for the leap.</span>');
        if (buckets['Sleeper'].length) addMove('Buy / hold', _RI_SIG_COLOR['Sleeper'], '<b>' + names(buckets['Sleeper']) + '</b><span class="why">: valued above the market.</span>');
        if (buckets['Stash'].length) addMove('Stash', _RI_SIG_COLOR['Stash'], '<b>' + names(buckets['Stash']) + '</b><span class="why">: young upside; stash for later.</span>');
        if (buckets['Monitor'].length) addMove('Monitor', _RI_SIG_COLOR['Monitor'], '<b>' + names(buckets['Monitor']) + '</b><span class="why">: slipping; watch closely.</span>');
        if (needs.length) {
            var needStr = needs.map(function (n) {
                var advice = n.health === 'Aging' ? 'get younger'
                    : (n.anchored ? 'add a backup' : 'shop for an upgrade');
                return '<b>' + n.pos + '</b> is ' + n.health.toLowerCase() + '<span class="why">, ' + advice + '</span>';
            }).join('; ');
            addMove('Target', _RI_HEALTH_COLOR['Thin'], needStr + '<span class="why">.</span>');
        }
        return '<div class="ri-summary"><div class="ri-summary-eyebrow">Suggested moves</div>' +
            (moves.length ? moves.join('') : '<div class="ri-summary-stable">Roster looks stable, no moves flagged.</div>') +
            '</div>';
    }

    // Signal -> drawer tag class (Mock 4 roster-intel tags).
    function _riTagClass(signal) {
        return {
            'Core': 'td-t-core',
            'Hold': 'td-t-hold',
            'Sell High': 'td-t-sell',
            'Stash': 'td-t-stash',
            'Sleeper': 'td-t-sleeper',
            'Breakout': 'td-t-breakout',
            'Monitor': 'td-t-monitor',
            'Cut': 'td-t-cut',
        }[signal] || 'td-t-hold';
    }

    // ── Team detail drawer (Mock 4) ──
    var _drawerDataCache = null;
    var _drawerDataEl = null;
    function _drawerPayload() {
        // Re-parse when soft-nav swaps in a new payload element (league switch).
        var el = document.getElementById('teamsDrawerData');
        if (el && el !== _drawerDataEl) {
            _drawerDataEl = el;
            try {
                _drawerDataCache = JSON.parse(el.textContent || '{}');
            } catch (e) {
                _drawerDataCache = {};
            }
        }
        return _drawerDataCache || {};
    }

    function _tdEsc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
        });
    }

    function _tdAvatar(t) {
        if (t.avatar) {
            return '<span class="td-ava"><img src="' + _tdEsc(t.avatar) + '" alt="" loading="lazy" decoding="async" onerror="this.style.visibility=\'hidden\'"></span>';
        }
        return '<span class="td-ava td-ava-mono">' + _tdEsc(t.initials || '?') + '</span>';
    }

    function _tdHeadHtml(t) {
        var status = t.window ? _tdEsc(t.window) : 'Unranked';
        var bits = status;
        if (t.grade && t.grade !== '?') bits += ' &middot; ' + _tdEsc(t.grade);
        bits += ' &middot; Pos index ' + _tdEsc(t.pos_index);
        return '<div class="td-head-row">' + _tdAvatar(t) +
            '<div class="td-head-id">' +
            '<div class="td-name">' + _tdEsc(t.name) + (t.is_viewer ? '<span class="tsc-you">YOU</span>' : '') + '</div>' +
            '<div class="td-status"><span class="td-dot" style="background:' + _tdEsc(t.win_color || '#94a3b8') + ';"></span>' + bits + '</div>' +
            '</div>' +
            '<button class="td-x" id="teamDrawerClose" type="button" aria-label="Close team details">&#10005;</button></div>';
    }

    var _TD_POSC = {QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6'};
    function _tdBodyHtml(t) {
        var html = '<div class="td-sec">POSITIONAL BREAKDOWN</div>';
        (t.positions || []).forEach(function (pos, i) {
            var pills = '<span class="td-pill">PLAYERS <b>' + pos.count + '</b></span>';
            if (pos.age) pills += '<span class="td-pill">AVG AGE <b>' + _tdEsc(pos.age) + ' yrs</b></span>';
            if (pos.rank) pills += '<span class="td-pill">RANK <b>#' + pos.rank + '/' + (pos.num_teams || '') + '</b></span>';
            if (pos.strength) pills += '<span class="td-pill">STRENGTH <b>' + _tdEsc(pos.strength) + '</b></span>';
            html += '<details class="td-pos"' + (i === 0 ? ' open' : '') + '>' +
                '<summary><span class="td-chip" style="background:' + _TD_POSC[pos.pos] + ';">' + _tdEsc(pos.pos) + '</span>' +
                '<span class="td-pval">' + Number(pos.total).toFixed(1) + '</span>' +
                (pos.rank ? '<span class="td-rank">#' + pos.rank + '</span>' : '') +
                '</summary>' +
                '<div class="td-pdet"><div class="td-pills">' + pills + '</div>' +
                (pos.players_html || '<div class="td-note">No players at this position.</div>') +
                '</div></details>';
        });
        html += '<div class="td-sec">ROSTER INTEL</div>' +
            '<div class="td-intel" id="teamDrawerIntel">' +
            '<div class="analytics-skeleton"><div class="sk-shimmer sk-line" style="width:60%"></div>' +
            '<div class="sk-shimmer sk-line sk-line--w75" style="margin-top:10px"></div>' +
            '<div class="sk-shimmer sk-line sk-line--w50" style="margin-top:10px"></div></div></div>';
        return html;
    }

    function _tdInjectIntelTags(vt) {
        var byName = {};
        _RI_POS_ORDER.forEach(function (pos) {
            var pd = (vt.positions || {})[pos];
            if (!pd) return;
            (pd.players || []).forEach(function (pl) {
                if (pl && pl.name) byName[String(pl.name).toLowerCase()] = pl.signal;
            });
        });
        var body = document.getElementById('teamDrawerBody');
        if (!body) return;
        Array.prototype.forEach.call(body.querySelectorAll('.player-clickable[data-player-name]'), function (el) {
            var sig = byName[String(el.getAttribute('data-player-name') || '').toLowerCase()];
            if (!sig) return;
            var tag = document.createElement('span');
            tag.className = 'td-tag ' + _riTagClass(sig);
            tag.textContent = String(sig).toUpperCase();
            el.parentNode.insertBefore(tag, el);
        });
    }

    function _tdLoadIntel(rid, t) {
        var box = document.getElementById('teamDrawerIntel');
        if (!box) return;
        // The roster-intel API only computes signals for the viewer's team.
        if (!t.is_viewer) {
            box.innerHTML = '<div class="td-note">Roster intel is only computed for your team, so this drawer shows the roster breakdown without move suggestions.</div>';
            return;
        }
        _riFetchData()
            .then(function (data) {
                var b = document.getElementById('teamDrawerIntel');
                if (!b) return;
                var teams = (data && data.teams) || [];
                var vt = null;
                teams.forEach(function (x) {
                    if (String(x.roster_id) === String(rid)) vt = x;
                });
                if (!vt) {
                    b.innerHTML = '<div class="td-note">Could not load roster intel.</div>';
                    return;
                }
                b.innerHTML = _riSuggestedMoves(vt);
                _tdInjectIntelTags(vt);
            })
            .catch(function () {
                var b2 = document.getElementById('teamDrawerIntel');
                if (b2) b2.innerHTML = '<div class="td-note">Could not load roster intel.</div>';
            });
    }

    function openTeamDrawer(rid) {
        var payload = _drawerPayload();
        var t = payload[String(rid)];
        if (!t) return;
        var drawer = document.getElementById('teamDrawer');
        var scrim = document.getElementById('teamDrawerScrim');
        if (!drawer || !scrim) return;
        document.getElementById('teamDrawerHead').innerHTML = _tdHeadHtml(t);
        document.getElementById('teamDrawerBody').innerHTML = _tdBodyHtml(t);
        drawer.classList.add('open');
        scrim.classList.add('open');
        document.body.classList.add('td-lock');
        var x = document.getElementById('teamDrawerClose');
        if (x) x.addEventListener('click', closeTeamDrawer);
        _tdLoadIntel(rid, t);
    }

    function closeTeamDrawer() {
        var drawer = document.getElementById('teamDrawer');
        var scrim = document.getElementById('teamDrawerScrim');
        if (drawer) drawer.classList.remove('open');
        if (scrim) scrim.classList.remove('open');
        document.body.classList.remove('td-lock');
    }

    function wireTeamDrawer() {
        var scrim = document.getElementById('teamDrawerScrim');
        if (scrim) scrim.addEventListener('click', closeTeamDrawer);
        Array.prototype.forEach.call(document.querySelectorAll('.team-strength-card'), function (card) {
            card.addEventListener('click', function (e) {
                if (e.target.closest('a')) return;
                openTeamDrawer(card.getAttribute('data-roster-id'));
            });
        });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' || e.key === 'Esc') closeTeamDrawer();
        });
    }

    function loadRosterIntel() {
        _syncLeagueCfg();
        if (_loaded.rosterIntel) return;
        _loaded.rosterIntel = true;
        var panel = document.getElementById('rosterIntelPanel');
        if (!panel) return;

        _riFetchData()
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

                    html += _riSuggestedMoves(t);

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
            wireTeamDrawer();
            _activateTabFromHash();
        });
    } else {
        wireAnalyticsTabs();
        wireTeamDrawer();
        _activateTabFromHash();
    }
})();
