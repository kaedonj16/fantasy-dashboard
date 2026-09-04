// ============================================================
// player_modal.js — player overlay (openPlayerModal + closePlayerModal)
// Extracted from app.js. Watchlist / comparison stay in app.js.
//
// Full pages: load AFTER app.js.
// Lite/public.js pages: do NOT attach this file as a blocking script — that
// would overwrite the openPlayerModal stub and skip lazy-loading features.
// _ensure_features_js concatenates this file into app-features.js instead.
// ============================================================

function openPlayerModal(playerId, playerName, opts) {
  opts = opts || {};

  // Guests: a player-name click goes to the public player page (SEO landing),
  // not the in-app modal. Signed-in users keep the modal. opts.force bypasses
  // this (used when auto-opening from ?player= after sign-in).
  if (!opts.force && !window._isSignedIn) {
    const _guestSlug = pmSlugify(playerName);
    if (_guestSlug) {
      window.location.href = '/player/' + _guestSlug + '/trade-value';
      return;
    }
  }

  const _ppSlug = pmSlugify(playerName);

  // Extract league context from URL path: /<platform>/<season>/<league_id>/<page>
  // For non-league pages (portfolio, home, etc.) fall back to query params.
  const pathParts = window.location.pathname.split('/').filter(p => p);
  const urlParams = new URLSearchParams(window.location.search);
  const _isLeaguePath = pathParts.length >= 3 && !isNaN(parseInt(pathParts[1]));
  // Explicit opts win (used by the player page's "view in your league" flow),
  // then a league URL path, then query params.
  const platform = opts.platform || (_isLeaguePath ? pathParts[0] : (urlParams.get('platform') || 'sleeper'));
  const season = opts.season || (_isLeaguePath ? pathParts[1] : (urlParams.get('season') || new Date().getFullYear()));
  const leagueId = opts.leagueId || (_isLeaguePath ? pathParts[2] : (urlParams.get('from_league') || null));

  // Use page-level league settings when available (set for logged-in users)
  const modalLt = brLeagueType();
  const modalLs = brLeagueSize();
  const leagueParams = `league_type=${encodeURIComponent(modalLt)}&league_size=${encodeURIComponent(modalLs)}`;

  // Build API URL with league context if available
  const apiUrl = leagueId
    ? `/api/player-details/${playerId}?league_id=${leagueId}&platform=${platform}&season=${season}&${leagueParams}`
    : `/api/player-details/${playerId}?${leagueParams}`;
  
  // Create modal overlay
  const overlay = document.createElement('div');
  overlay.className = 'player-modal-overlay';
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      closePlayerModal();
    }
  });

  // Create modal
  const modal = document.createElement('div');
  modal.className = 'player-modal';
  modal.id = 'playerModal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-label', (playerName ? playerName + ': player details' : 'Player details'));

  modal.innerHTML = `
    <div class="player-modal-header">
        <div class="player-modal-headshot-container">
          <img class="player-modal-headshot" id="playerModalHeadshot" src="" alt="${playerName || 'Player'}" />
        </div>
      <div class="player-modal-title-section">
        <div class="player-modal-title-text">
          <h2 class="player-modal-name">${playerName || 'Loading...'}</h2>
          <div class="player-modal-meta" id="playerModalMeta">
            <span class="skeleton skeleton-line" style="display:inline-block;width:150px;height:12px;margin:2px 0 0;border-radius:5px;"></span>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
        <button class="player-modal-watchlist-btn" id="playerModalWatchlistBtn" title="Add to watchlist" aria-pressed="false" style="display: none;"><span class="wl-star-glyph" aria-hidden="true">☆</span></button>
        ${_ppSlug ? `<a class="player-modal-page-btn" href="/player/${_ppSlug}/trade-value" title="View full player page">Player Page</a>` : ''}
        <button class="player-modal-close" onclick="closePlayerModal()" aria-label="Close">×</button>
      </div>
    </div>
    <div class="pm-tab-bar" id="pmTabBar" role="tablist" aria-label="Player details" style="display:none">
      <button class="pm-tab active" role="tab" aria-selected="true" data-tab="overview" onclick="pmSwitchTab('overview')">Overview</button>
      <button class="pm-tab" role="tab" aria-selected="false" data-tab="stats" onclick="pmSwitchTab('stats')">Stats</button>
      <button class="pm-tab" role="tab" aria-selected="false" id="pmTabTeam" data-tab="team" onclick="pmSwitchTab('team')" style="display:none">Team</button>
      <button class="pm-tab" role="tab" aria-selected="false" id="pmTabMetrics" data-tab="metrics" onclick="pmSwitchTab('metrics')" style="display:none">Adv Metrics</button>
      <button class="pm-tab" role="tab" aria-selected="false" id="pmTabProspect" data-tab="prospect" onclick="pmSwitchTab('prospect')" style="display:none">Prospect</button>
      <button class="pm-tab" role="tab" aria-selected="false" id="pmTabBreakout" data-tab="breakout" onclick="pmSwitchTab('breakout')" style="display:none">Breakout</button>
      <button class="pm-tab" role="tab" aria-selected="false" data-tab="trades" onclick="pmSwitchTab('trades')">Trades</button>
    </div>
    <div class="player-modal-body" id="playerModalBody">
      <div class="pm-skel" style="padding:16px 18px;">
        <div class="pm-hero-row">
          <div class="pm-hero-stat pm-hero-primary">
            <div class="skeleton skeleton-line" style="width:64%;height:9px;margin:0 auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:50%;height:22px;margin:2px auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:80%;height:8px;margin:0 auto;"></div>
          </div>
          <div class="pm-hero-stat">
            <div class="skeleton skeleton-line" style="width:64%;height:9px;margin:0 auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:50%;height:22px;margin:2px auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:80%;height:8px;margin:0 auto;"></div>
          </div>
          <div class="pm-hero-stat">
            <div class="skeleton skeleton-line" style="width:64%;height:9px;margin:0 auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:50%;height:22px;margin:2px auto 9px;"></div>
            <div class="skeleton skeleton-line" style="width:80%;height:8px;margin:0 auto;"></div>
          </div>
        </div>
        <hr class="pm-section-divider">
        <div class="skeleton skeleton-line" style="width:96px;height:10px;margin:14px 0 12px;"></div>
        <div class="skeleton" style="height:200px;border-radius:12px;"></div>
      </div>
    </div>
  `;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);
  document.body.style.overflow = 'hidden';

  // ── Accessibility: focus management + focus trap ──────────────────────────
  // Remember what had focus so closePlayerModal can restore it, then move focus
  // into the dialog so keyboard and screen-reader users land inside it instead
  // of tabbing through the page behind the overlay.
  overlay._pmReturnFocus = (document.activeElement instanceof HTMLElement) ? document.activeElement : null;
  const _pmCloseBtn = modal.querySelector('.player-modal-close');
  if (_pmCloseBtn) { try { _pmCloseBtn.focus(); } catch (_) {} }
  // Keep Tab / Shift+Tab cycling within the dialog. Focusables are queried at
  // key time so tabs/buttons added after the async data load are included.
  overlay.addEventListener('keydown', function (e) {
    if (e.key !== 'Tab') return;
    const focusables = modal.querySelectorAll(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    const visible = Array.prototype.filter.call(focusables, el => el.getClientRects().length > 0);
    if (!visible.length) return;
    const first = visible[0], last = visible[visible.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });

  // Wire the watchlist star (add/remove this player from the device watchlist).
  try {
    const _wlBtn = document.getElementById('playerModalWatchlistBtn');
    if (_wlBtn && typeof _toggleWatchlist === 'function') {
      _wlBtn.style.display = '';
      _updateWatchlistBtn(_wlBtn, playerId);
      _wlBtn.onclick = function () {
        _toggleWatchlist({ player_id: playerId, name: playerName || '', position: (opts.position || '') });
        _updateWatchlistBtn(_wlBtn, playerId);
      };
    }
  } catch (_) {}

  // Fetch player data (with 5-min localStorage cache to speed up re-opens)
  // Contract version prevents pre-canonical projection/scoring payloads from
  // surviving a deploy. apiUrl already carries platform/league/season context.
  const _cacheKey = 'pm_cache_v3_' + apiUrl;
  const _cacheTTL = 5 * 60 * 1000;
  let _cachedRaw = null;
  try {
    const _entry = JSON.parse(localStorage.getItem(_cacheKey) || 'null');
    if (_entry && Date.now() - _entry.ts < _cacheTTL) _cachedRaw = _entry.data;
  } catch (_) {}

  const _fetchPromise = _cachedRaw
    ? Promise.resolve(_cachedRaw)
    : fetch(apiUrl)
        .then(res => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
        .then(data => {
          try { localStorage.setItem(_cacheKey, JSON.stringify({ ts: Date.now(), data })); } catch (_) {}
          return data;
        });

  _fetchPromise
    .then(data => {

      const modalBody = document.getElementById('playerModalBody');
      if (!modalBody) return; // modal was closed before fetch completed

      if (data.error) {
        if (window.brErrorState) {
          window.brErrorState(modalBody, data.error, function () {
            closePlayerModal();
            openPlayerModal(playerId, playerName, opts);
          });
        } else {
          modalBody.innerHTML = `
            <div class="player-modal-loading">
              <div style="color: var(--loss); font-weight: 500;">Error loading player data</div>
              <div style="font-size: 13px;">${data.error}</div>
            </div>
          `;
        }
        return;
      }

      // Check if data has expected structure
      if (!data.name) {
        modalBody.innerHTML = `
          <div class="player-modal-loading">
            <div style="color: #f59e0b; font-weight: 500;">Player data incomplete</div>
            <div style="font-size: 13px;">Player ID: ${playerId}</div>
          </div>
        `;
        return;
      }

      // Determine badges using playerIndicators (same source as all other badge displays)
      let badges = '';
      const yearsExp = data.stats?.years_exp;
      const pid = String(data.player_id || playerId);

      // Check if player has no game logs (indicating a rookie without NFL stats).
      // Game logs themselves are lazy-loaded by the Stats tab; player-details now
      // ships a cheap has_game_logs boolean instead of the full per-week payload.
      const hasGameLogs = (typeof data.has_game_logs === 'boolean')
        ? data.has_game_logs
        : (data.game_logs_by_year && Object.keys(data.game_logs_by_year).length > 0);
      const isRookieWithoutGameLogs = !hasGameLogs && data.prospect_data && data.prospect_data.prospect_score != null;

      if (isElite(pid)) {
        badges += '<span class="player-badge player-badge-elite"><i class="fa-solid fa-star-solid" aria-hidden="true"></i> ELITE</span>';
      }
      // Rookie (drafted / year-0) gets the ROOKIE mark; a pre-draft prospect
      // gets the PROSPECT seedling instead.
      if ((yearsExp != null && yearsExp === 0) || isRookieWithoutGameLogs) {
        badges += '<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>';
      } else if (isProspect(pid)) {
        badges += '<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> PROSPECT</span>';
      }
      if (isBreakout(pid)) {
        badges += '<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>';
      }
      // Injury designation (from the full Sleeper feed). Severity by color.
      if (data.injury && data.injury.status) {
        const _u = String(data.injury.status).toUpperCase();
        let _icls = 'player-badge-inj-q';
        if (['IR', 'OUT', 'PUP', 'SUSP', 'NFI'].includes(_u)) _icls = 'player-badge-inj-out';
        else if (['DOUBTFUL', 'D'].includes(_u)) _icls = 'player-badge-inj-d';
        const _tip = [data.injury.body_part, data.injury.notes].filter(Boolean).join(' · ') || _u;
        const _lbl = _u.length > 14 ? _u.slice(0, 14) : _u;
        badges += `<span class="player-badge ${_icls}" title="${String(_tip).replace(/"/g, '&quot;')}"><i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> ${_lbl}</span>`;
        const plan = data.injury.return_plan;
        if (plan && plan.verdict) {
          const wk = plan.weeks_label || '';
          const src = plan.source === 'espn' ? 'ESPN approx' : 'approx';
          const tip = String(plan.reason || 'Approximate return guidance, not medical advice.')
            .replace(/"/g, '&quot;');
          badges += `<span class="player-badge player-badge-inj-q" title="${tip}"><i class="fa-solid fa-clock-rotate-left" aria-hidden="true"></i> ${escapeHtml(plan.verdict)}${wk ? ' · ' + escapeHtml(wk) : ''} <span style="opacity:.7;font-weight:500;">(${escapeHtml(src)})</span></span>`;
        }
      }

      // Name with inline badges
      const nameEl = document.querySelector('.player-modal-name');
      if (!nameEl) return;
      nameEl.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;';
      nameEl.innerHTML = `<span>${escapeHtml(playerName || 'Unknown Player')}</span>${badges}`;

      // Meta with dots separator
      const metaParts = [];
      if (data.position && data.pos_rank) metaParts.push(`<span style="font-weight:600;color:var(--text);">${data.position}${data.pos_rank}</span>`);
      if (data.team) metaParts.push(`<span>${data.team}</span>`);
      const ageNum = parseFloat(data.age);
      if (!isNaN(ageNum)) metaParts.push(`<span>${ageNum.toFixed(1)} yrs</span>`);
      
      // ── Value trend classification (small meta pill) ──────────────────────
      const vt = data.value_trend || {};
      const vtClass = vt.class || 'unknown';
      const vtIcons = { rising:'↑', declining:'↓', stable:'→', volatile:'↕', peaked:'↘', recovering:'↗', unknown:'' };
      const vtIcon = vtIcons[vtClass] || '';
      
      if (vtClass && vtClass !== 'unknown' && vtIcon) {
        const _slopeTxt = vt.slope_pct_month != null
          ? ' · ' + (vt.slope_pct_month >= 0 ? '+' : '') + vt.slope_pct_month.toFixed(1) + '%/mo'
          : '';
        const _tipTxt = (vt.description || vt.label) + _slopeTxt;
        metaParts.push(`<span class="pm-trend-pill" data-trend-tip="${_tipTxt}" style="padding:1px 6px;border-radius:4px;background:${vt.color}18;border:1px solid ${vt.color}40;color:${vt.color};font-size:10px;font-weight:700;cursor:help;">${vtIcon} ${vt.label}</span>`);
      }

      const metaEl = document.getElementById('playerModalMeta');
      let metaHTML = `<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;">${metaParts.join('<span style="opacity:.35;margin:0 3px;">·</span>')}</div>`;
      if (data.fantasy_team) {
        const _ownerStr = data.fantasy_team_owner ? ` · <span style="opacity:.65;">@${escapeHtml(data.fantasy_team_owner)}</span>` : '';
        metaHTML += `<div style="font-size:11px;font-weight:600;color:var(--accent);margin-top:3px;opacity:.9;">${escapeHtml(data.fantasy_team)}${_ownerStr}</div>`;
      }
      metaEl.innerHTML = metaHTML;

      // Update headshot
      const headshotEl = document.getElementById('playerModalHeadshot');
      if (headshotEl && data.espnHeadshot) {
        headshotEl.dataset.raw = data.espnHeadshot;
        headshotEl.onerror = function () {
          if (this.dataset.raw && this.src !== this.dataset.raw) { this.src = this.dataset.raw; }
          else { this.style.visibility = 'hidden'; }
        };
        headshotEl.src = _hiResHeadshot(data.espnHeadshot, 360);
      }

      // Extract player position
      const pos = data.position;

      // ── Hero row ─────────────────────────────────────────────────────────
      const val1qb = data.stats?.value || 0;
      const valsf  = data.stats?.sf_value || 0;
      const posRankLabel = data.stats?.pos_rank_label || (data.stats?.pos_rank ? `${pos}${data.stats.pos_rank}` : '');
      const expLabel = data.stats?.years_exp === 0 ? 'Rookie'
        : data.stats?.years_exp != null ? `${data.stats.years_exp} yr${data.stats.years_exp !== 1 ? 's' : ''}`
        : '-';

      // Dynasty vs Redraft values for the hero toggle. Default from the league
      // format (ESPN / settings.type 0|1 → redraft); on non-league pages, follow
      // the trade-calc / rankings scoring control when present.
      // Position multipliers MUST match SCORING_MULTS in utils/trade_value.py /
      // static/app.js trade calc.
      const PM_SCORING_MULTS = {
        ppr:  { QB: 1.00, RB: 1.00, WR: 1.00, TE: 1.00 },
        half: { QB: 1.00, RB: 1.06, WR: 0.97, TE: 0.94 },
        std:  { QB: 1.00, RB: 1.13, WR: 0.93, TE: 0.87 },
      };
      const _heroDyn = {
        v1: Number(val1qb) || 0,
        vs: Number(valsf) || 0,
        p1: data.stats?.pos_rank,
        o1: data.stats?.value_ovr_rank,
        ps: data.stats?.sf_pos_rank,
        os: data.stats?.sf_value_ovr_rank,
      };
      const _heroRd = {
        v1: Number(data.stats?.redraft_value_1qb) || 0,
        vs: Number(data.stats?.redraft_value_sf) || 0,
        p1: data.stats?.redraft_pos_rank,
        o1: data.stats?.redraft_value_ovr_rank,
        ps: data.stats?.redraft_sf_pos_rank,
        os: data.stats?.redraft_sf_value_ovr_rank,
      };
      const _fmtRanks = data.scoring_format_ranks || {};
      let pmScoringType = (data.default_scoring_type === 'redraft') ? 'redraft' : 'dynasty';
      let pmScoringFormat = (['ppr', 'half', 'std'].includes(data.default_scoring_format)
        ? data.default_scoring_format : 'ppr');
      if (!leagueId) {
        const _pageType = document.querySelector('#scoringTypeSelect')?.value
          || (typeof prScoringType !== 'undefined' ? prScoringType : null);
        if (_pageType === 'redraft' || _pageType === 'dynasty') {
          pmScoringType = _pageType;
        }
        const _pageFmt = document.querySelector('#scoringFormatSelect')?.value
          || (typeof getScoringFormat === 'function' ? getScoringFormat() : null);
        if (_pageFmt === 'ppr' || _pageFmt === 'half' || _pageFmt === 'std') {
          pmScoringFormat = _pageFmt;
        }
      }
      const _heroFmt = (v) => (v > 0 ? v : '-');
      const _heroSub = (posR, ovrR) => (posR ? `POS : ${posR} · OVR : ${ovrR ?? '–'}` : '-');
      const _heroActive = () => {
        const base = pmScoringType === 'redraft' ? _heroRd : _heroDyn;
        const mults = PM_SCORING_MULTS[pmScoringFormat] || PM_SCORING_MULTS.ppr;
        const mult = mults[(pos || '').toUpperCase()] ?? 1;
        const scale = (v) => {
          const n = Number(v) || 0;
          if (n <= 0) return 0;
          return Math.floor(n * mult * 10 + 0.5) / 10;
        };
        let p1 = base.p1, o1 = base.o1, ps = base.ps, os = base.os;
        if (pmScoringFormat !== 'ppr') {
          const fr = _fmtRanks[pmScoringFormat] || {};
          if (pmScoringType === 'redraft') {
            p1 = fr.redraft_pos_rank;
            o1 = fr.redraft_value_ovr_rank;
            ps = fr.redraft_sf_pos_rank;
            os = fr.redraft_sf_value_ovr_rank;
          } else {
            p1 = fr.pos_rank;
            o1 = fr.value_ovr_rank;
            ps = fr.sf_pos_rank;
            os = fr.sf_value_ovr_rank;
          }
        }
        return { v1: scale(base.v1), vs: scale(base.vs), p1, o1, ps, os };
      };

      const _draftYrVal = data.draft_year ? String(data.draft_year) : '';
      const thirdValueCard = data.stats?.pos_rank
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">Dynasty</div>
            <div class="pm-hero-val">${posRankLabel || data.stats.pos_rank}</div>
          </div>`
        : `<div class="pm-hero-stat" style="position:relative;">
            <div class="pm-hero-label" style="display:flex;align-items:center;gap:4px;">
              Experience
              <button onclick="pmEditDraftYear('${pid}')" title="Set draft year"
                style="background:none;border:none;cursor:pointer;padding:0;line-height:1;color:var(--text-muted);font-size:11px;opacity:.55;" aria-label="Edit draft year">✏</button>
            </div>
            <div class="pm-hero-val" id="pmExpLabel">${expLabel}</div>
            <div id="pmDraftYrEdit" style="display:none;margin-top:6px;gap:4px;align-items:center;flex-wrap:wrap;">
              <input id="pmDraftYrInput" type="number" min="2000" max="2030" value="${_draftYrVal}"
                placeholder="e.g. 2024"
                style="width:72px;padding:3px 6px;border:1px solid var(--border);border-radius:6px;font-size:12px;background:var(--bg);color:var(--text);"/>
              <button onclick="pmSaveDraftYear('${pid}')"
                style="padding:3px 10px;border-radius:6px;background:var(--accent);color:#fff;border:none;cursor:pointer;font-size:12px;">Save</button>
            </div>
          </div>`;

      const ppgVal       = data.stats?.ppg;
      const ppgRank      = data.stats?.ppg_rank;
      const ppgOvrRank   = data.stats?.ppg_ovr_rank;
      const ppgSeason    = data.stats?.ppg_season;
      const totalPts     = data.stats?.total_pts;
      const totalPtsRank = data.stats?.total_pts_rank;
      const totalPtsOvrRank = data.stats?.total_pts_ovr_rank;
      const seasonLabel  = ppgSeason ? ` · ${ppgSeason}` : '';
      const ppgCard = ppgVal != null
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">PPG${seasonLabel}</div>
            <div class="pm-hero-val">${ppgVal}</div>
            <div class="pm-hero-sub">${ppgRank ? `POS : ${ppgRank} · OVR : ${ppgOvrRank ?? '–'}` : '-'}</div>
          </div>`
        : '';
      const totalCard = totalPts != null
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">Total Pts${seasonLabel}</div>
            <div class="pm-hero-val">${fmtPts(totalPts)}</div>
            <div class="pm-hero-sub">${totalPtsRank ? `POS : ${totalPtsRank} · OVR : ${totalPtsOvrRank ?? '–'}` : '-'}</div>
          </div>`
        : '';

      // ── Prospect Profile tab ─────────────────────────────────────────────────
      const pd = data.prospect_data;
      const hasProspectData = pd && pd.prospect_score != null;
      // isRookieWithProspectData controls the ROOKIE badge and no-stats overview layout
      const isRookieWithProspectData = !hasGameLogs && hasProspectData;
      let pdColHTML = '';
      if (hasProspectData) {
        const pdConf  = parseFloat(pd.confidence_score || 0);

        // Draft info + ADP row
        const pdAdp1qb = pd.avg_pick != null ? parseFloat(pd.avg_pick).toFixed(1) : null;
        const pdAdpSf  = pd.sf_avg_pick != null ? parseFloat(pd.sf_avg_pick).toFixed(1) : null;
        const pdDraftCap = pd.draft_capital_label || (pd.projected_pick ? `Pick #${pd.projected_pick}` : null);
        const pdDraftRow = `
          <div style="display:flex;justify-content:space-between;align-items:center;
                      border:1px solid var(--border);border-radius:10px;padding:13px 16px;
                      margin-bottom:14px;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:baseline;gap:8px;">
              <span style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;">Draft</span>
              <span style="font-size:15px;font-weight:700;color:var(--text);">${pdDraftCap || 'TBD'}</span>
              ${pd.num_mocks_used ? `<span style="font-size:11px;color:var(--text-muted);">(${pd.num_mocks_used} mocks)</span>` : ''}
            </div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;">
              ${pdAdp1qb ? `<span style="font-size:12px;color:var(--text-muted);">1QB ADP: <strong style="color:var(--text);font-size:13px;">${pdAdp1qb}</strong></span>` : ''}
              ${pdAdpSf  ? `<span style="font-size:12px;color:var(--text-muted);">SF ADP: <strong style="color:var(--text);font-size:13px;">${pdAdpSf}</strong></span>` : ''}
            </div>
          </div>`;

        // Measurables row
        const pdHt = pd.height_inches;
        const pdHeightStr = pdHt ? `${Math.floor(pdHt/12)}'${pdHt%12}"` : '-';
        const pdWeightStr = pd.weight_lbs ? `${pd.weight_lbs} lbs` : '-';
        const pdFortyStr  = pd.forty_yard  ? `${pd.forty_yard}s`  : '-';
        const pdRasStr    = pd.ras_score   ? `${parseFloat(pd.ras_score).toFixed(1)}` : '-';
        const pdMeasurables = [
          {label:'Height',  val: pdHeightStr},
          {label:'Weight',  val: pdWeightStr},
          {label:'40 Dash', val: pdFortyStr},
          {label:'RAS',     val: pdRasStr},
        ];
        const pdMeasRow = `
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px;">
            ${pdMeasurables.map(m => `
              <div style="border:1px solid var(--border);border-radius:10px;padding:11px 8px;text-align:center;">
                <div style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;">${m.label}</div>
                <div style="font-size:14px;font-weight:700;color:var(--text);">${m.val}</div>
              </div>`).join('')}
          </div>`;

        // Component scores
        const pdComponents = [
          {label:'Production',  val: pd.production_score,              color:'#10b981'},
          {label:'Efficiency',  val: pd.efficiency_score,              color:'#3b82f6'},
          {label:'Age',         val: pd.age_score,                     color:'#8b5cf6'},
          {label:'Breakout',    val: pd.breakout_profile_score,        color:'#f59e0b'},
          {label:'Athleticism', val: pd.athleticism_score,             color:'#ef4444'},
          {label:'Competition', val: pd.competition_score,             color:'#06b6d4'},
          {label:'Draft Cap.',  val: pd.projected_draft_capital_score, color:'#f97316'},
        ];
        const pdCompsHtml = pdComponents.map(c => {
          const v = parseFloat(c.val || 0);
          return `<div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;">
            <div style="width:88px;flex-shrink:0;font-size:13px;color:var(--text);">${c.label}</div>
            <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;">
              <div style="height:100%;width:${Math.round(v)}%;background:${c.color};border-radius:3px;transition:width .3s;"></div>
            </div>
            <div style="width:28px;text-align:right;font-size:13px;font-weight:700;color:${c.color};">${v.toFixed(0)}</div>
          </div>`;
        }).join('');

        // Scouting notes (strip leading bullet chars stored in key_reasons)
        const pdReasons = (pd.key_reasons || '').split('\n')
          .map(l => l.replace(/^[•·\-\*]\s*/, '').trim())
          .filter(l => l);
        const pdScoutingHtml = pdReasons.length ? `
          <div style="margin-top:20px;">
            <div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">Scouting Notes</div>
            <ul style="margin:0;padding:0;list-style:none;">
              ${pdReasons.map(r => `<li style="font-size:13px;color:var(--text-muted);padding:3px 0 3px 14px;position:relative;line-height:1.5;"><span style="position:absolute;left:0;color:var(--accent);">·</span>${r}</li>`).join('')}
            </ul>
          </div>` : '';

        const pdScore = parseFloat(pd.prospect_score || 0);
        const pdTier  = pd.tier;
        const pdTierClass  = pdTier ? `rk-tier-${pdTier}` : '';
        const pdRankStr    = pd.overall_rank ? `#${pd.overall_rank} Overall` : '';
        const pdPosRankStr = pd.position_rank ? `${pos}${pd.position_rank}` : '';

        const pdHeroSection = `
          <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:10px;margin-bottom:14px;">
            <div style="background:var(--accent-soft);border-radius:12px;padding:14px 16px;text-align:center;">
              <div style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Prospect Score</div>
              <div style="font-size:28px;font-weight:700;color:var(--accent);line-height:1;">${pdScore.toFixed(1)}</div>
              <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">${pd.tier_label || ''}</div>
            </div>
            <div class="${pdTierClass}" style="border-radius:12px;padding:14px 12px;text-align:center;display:flex;align-items:center;justify-content:center;">
              ${pdTier ? `<div style="font-size:22px;font-weight:700;">Tier ${pdTier}</div>` : '<div style="font-size:18px;font-weight:700;color:var(--text-muted);">-</div>'}
            </div>
            <div style="background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:14px 12px;text-align:center;">
              <div style="font-size:10px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Rank</div>
              <div style="font-size:18px;font-weight:700;color:var(--text);line-height:1;">${pdRankStr || '-'}</div>
              ${pdPosRankStr ? `<div style="font-size:11px;color:var(--text-muted);margin-top:4px;">${pdPosRankStr}</div>` : ''}
            </div>
          </div>`;

        pdColHTML = `
          ${pdHeroSection}
          ${pdDraftRow}
          ${pdMeasRow}
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px;">
            <span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;">Component Scores</span>
            <span style="font-size:12px;color:var(--text-muted);">Data confidence: <strong style="color:var(--text);">${pdConf.toFixed(0)}</strong></span>
          </div>
          ${pdCompsHtml}
          ${pdScoutingHtml}
          <div id="pmProspectComparables" style="margin-top:20px;">
            <div class="rk-section-divider"></div>
            <div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">Historical Comparables</div>
            <div id="pmComparablesBody" style="font-size:13px;color:var(--text-muted);">
              <div style="display:flex;align-items:center;gap:8px;"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading…</div>
            </div>
          </div>
        `;
      }

      // ── Advanced Metrics / Prospect Profile + Value History flags ──
      const hasMetrics = !hasProspectData && pos && pos !== 'K' && pos !== 'DEF';
      const hasChart   = data.value_history && data.value_history.length > 0;

      const vtTrendBadge = '';

      // ── Build Overview panel HTML ─────────────────────────────────────────
      const valPosRank   = data.stats?.pos_rank;
      const valPosLabel  = data.stats?.pos_rank_label;
      const valOvrRank   = data.stats?.value_ovr_rank;
      const sfPosRank    = data.stats?.sf_pos_rank;
      const sfPosLabel   = data.stats?.sf_pos_rank_label;
      const sfOvrRank    = data.stats?.sf_value_ovr_rank;

      // TE-premium pill: shown beside the value titles when THIS league applies
      // a TE premium and the player is a tight end. Half premium → "TE+",
      // full (1pt) premium → "TE++". Mirrors the value scaling done server-side.
      const _tep = (pos === 'TE') ? (Number(data.te_premium) || 0) : 0;
      const tepPill = _tep >= 0.75
        ? '<span class="pm-tep-pill" title="Full TE premium (+20%)">TE++</span>'
        : _tep >= 0.25
        ? '<span class="pm-tep-pill" title="TE premium (+10%)">TE+</span>'
        : '';

      // Sleeper ADP (same feed as the rankings page): dynasty + redraft, 1QB + SF,
      // grouped into two format cards. The value matching the viewer's league
      // type is highlighted.
      const _adp = data.stats?.adp;
      const _adpIsSf = brLeagueType() === 'sf';
      const _adpV = v => (v != null ? v : '<span class="pm-adp-na">–</span>');
      // Multi-source ADP (Sleeper / BR Fantasy / ESPN / Yahoo / MFL / Consensus).
      // The Sleeper source arrives inline; the market sources are lazy-loaded from
      // /api/player-adp and merged in after the modal opens (ESPN/Yahoo/MFL are
      // redraft-only globals, so they only fill the Redraft card). Falls back to
      // the old flat single-source shape for backward compatibility.
      let _adpSources = (_adp && Array.isArray(_adp.sources)) ? _adp.sources.slice()
        : (_adp ? [{ label: 'Sleeper', vals: _adp }] : []);
      // Highlight the value matching the viewer's league type.
      const _c1 = _adpIsSf ? '' : ' pm-adp-cur';
      const _cS = _adpIsSf ? ' pm-adp-cur' : '';
      // Market-range ADP: one card per format (Dynasty / Redraft); inside, the
      // 1QB and SF ranges sit side by side. Each source is a colored dot on a
      // shared draft-pick scale, with a spread band and a consensus marker, so
      // the market reads at a glance. A source with no value for the format is
      // hidden entirely — no dot, no legend entry.
      const _adpColors = {
        'Sleeper': 'var(--adp-c-sleeper)', 'BR Fantasy': 'var(--adp-c-brf)',
        'BR Fantasy Live (7d)': 'var(--adp-c-brf-live)',
        'ESPN': 'var(--adp-c-espn)', 'Yahoo': 'var(--adp-c-yahoo)', 'MFL': 'var(--adp-c-mfl)',
      };
      const _adpEsc = s => String(s).replace(/[&<>"]/g,
        c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
      const _adpNum = v => (Math.round(v * 10) / 10).toFixed(1);
      // One range track for a (format, axis): dots + spread band + consensus mark
      // on an auto-scaled pick axis. Cons is the mean of the plotted dots
      // (BR Fantasy's 1..N rank included), so it sits among them — e.g.
      // (2.0 + 4.3) / 2 → 3.2. A minimum span keeps a tight cluster looking
      // tight instead of stretching two near-equal picks across the whole track.
      const _adpRangeTrack = (pts, cons) => {
        const all = pts.map(p => p.v).concat(cons != null ? [cons] : []);
        if (!all.length) {
          return '<div class="pm-adp-scale pm-adp-scale-empty"><span class="pm-adp-na">–</span></div>'
               + '<div class="pm-adp-ends"></div>';
        }
        const lo = Math.min(...all), hi = Math.max(...all);
        const base = cons != null ? cons : lo;
        const span = Math.max(hi - lo, Math.max(2, base * 0.25));
        const mid = (lo + hi) / 2, pad = span * 0.12;
        const sLo = mid - span / 2 - pad, sHi = mid + span / 2 + pad;
        const pos = v => Math.max(0, Math.min(100, (v - sLo) / (sHi - sLo) * 100));
        const dots = pts.map(p => {
          const tip = `${_adpEsc(p.label)} ADP: ${_adpNum(p.v)}`;
          return `<span class="pm-adp-dot" style="left:${pos(p.v).toFixed(1)}%;background:${p.color}" tabindex="0" role="img" aria-label="${tip}" data-tooltip="${tip}"></span>`;
        }).join('');
        const cmk = cons != null
          ? `<span class="pm-adp-cmk" style="left:${pos(cons).toFixed(1)}%"></span>` : '';
        return `
          <div class="pm-adp-scale">
            <span class="pm-adp-line"></span>
            <span class="pm-adp-band" style="left:${pos(lo).toFixed(1)}%;right:${(100 - pos(hi)).toFixed(1)}%"></span>
            ${cmk}${dots}
          </div>
          <div class="pm-adp-ends"><span>${_adpNum(lo)}</span><span>${_adpNum(hi)}</span></div>`;
      };
      const _adpRangeBlock = (fmtKey, axisKey, axisLabel, isCur, sources) => {
        const k = fmtKey + '_' + axisKey;
        const pts = sources
          .filter(s => s.vals[k] != null)
          .map(s => ({ label: s.label, color: _adpColors[s.label] || 'var(--text-muted)', v: Number(s.vals[k]) }));
        // Mean of the dots on this axis (including BR Fantasy's ordinal rank).
        const cons = pts.length ? pts.reduce((sum, p) => sum + p.v, 0) / pts.length : null;
        return `
          <div class="pm-adp-range${isCur ? ' pm-adp-range-cur' : ''}">
            <div class="pm-adp-range-hd">
              <span class="pm-adp-range-ax">${axisLabel}</span>
              <span class="pm-adp-range-cons"${cons != null ? ' title="Average of the source values shown on this axis"' : ''}>${cons != null ? 'Cons <b>' + _adpNum(cons) + '</b>' : ''}</span>
            </div>
            ${_adpRangeTrack(pts, cons)}
          </div>`;
      };
      // One card per format, its 1QB + SF ranges side by side, and a legend of
      // just the sources that actually have data for that format.
      const _adpFmtCard = (sources, fmtKey, fmtLabel) => {
        const present = sources.filter(s =>
          s.vals[fmtKey + '_1qb'] != null || s.vals[fmtKey + '_sf'] != null);
        if (!present.length) return '';
        const legend = present.map(s =>
          `<span class="pm-adp-lg"><i class="pm-adp-dot-sm" style="background:${_adpColors[s.label] || 'var(--text-muted)'}"></i>${_adpEsc(s.label)}</span>`).join('');
        return `
          <div class="pm-adp-card" data-adp-fmt="${fmtKey}">
            <div class="pm-adp-card-h">${fmtLabel}</div>
            <div class="pm-adp-ranges">
              ${_adpRangeBlock(fmtKey, '1qb', '1QB', !_adpIsSf, present)}
              ${_adpRangeBlock(fmtKey, 'sf', 'SF', _adpIsSf, present)}
            </div>
            <div class="pm-adp-legend">${legend}</div>
          </div>`;
      };
      // Inner grid for the ADP block. Backend Consensus is a raw-ADP mean and
      // is dropped — Cons on the range is the mean of the remaining dots
      // (BR Fantasy already ranked 1..N), so the marker sits among them.
      const _adpGridHTML = (sources) => {
        const srcs = sources.filter(s => s.label !== 'Consensus');
        const dyn = _adpFmtCard(srcs, 'dynasty', 'Dynasty');
        const rdr = _adpFmtCard(srcs, 'redraft', 'Redraft');
        if (!dyn && !rdr) return '';
        if (!dyn || !rdr) return dyn + rdr;   // one format only → no tabs needed
        // Both formats present: a Dynasty/Redraft segmented control so mobile
        // shows one card at a time (see .pm-adp-tabbed). Opens on the format the
        // modal is already set to; wider screens ignore the tabs and show both.
        const active = pmScoringType === 'redraft' ? 'redraft' : 'dynasty';
        const _tab = (key, label) =>
          `<button type="button" class="pm-adp-tab${key === active ? ' active' : ''}" data-adp-fmt="${key}" role="tab" aria-selected="${key === active}">${label}</button>`;
        return `<div class="pm-adp-tabs" role="tablist" aria-label="ADP format">${_tab('dynasty', 'Dynasty')}${_tab('redraft', 'Redraft')}</div>${dyn}${rdr}`;
      };
      // Wire the ADP format tabs after the grid renders: clicking a tab shows its
      // card and hides the other (mobile); on wide screens CSS shows both and
      // hides the tabs, so this state is simply ignored there.
      const _wireAdpTabs = (root) => {
        const tabs = Array.from(root.querySelectorAll('.pm-adp-tab'));
        if (!tabs.length) return;
        const cards = Array.from(root.querySelectorAll('.pm-adp-card[data-adp-fmt]'));
        const show = (fmt) => {
          cards.forEach(c => { c.hidden = c.getAttribute('data-adp-fmt') !== fmt; });
          tabs.forEach(b => {
            const on = b.getAttribute('data-adp-fmt') === fmt;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
          });
        };
        tabs.forEach(b => b.addEventListener('click', () => show(b.getAttribute('data-adp-fmt'))));
        show(pmScoringType === 'redraft' ? 'redraft' : 'dynasty');
      };
      // Skeleton shown while the market sources load, so all sources appear
      // together rather than Sleeper first.
      const _adpSkelCard = (fmtLabel) => `
        <div class="pm-adp-card">
          <div class="pm-adp-card-h">${fmtLabel}</div>
          <div class="pm-adp-ranges">
            ${[0, 1].map(() => `
              <div class="pm-adp-range">
                <div class="pm-adp-range-hd">
                  <span class="skeleton skeleton-line" style="width:26px;height:9px;"></span>
                  <span class="skeleton skeleton-line" style="width:42px;height:9px;"></span>
                </div>
                <div class="pm-adp-scale"><span class="skeleton skeleton-line" style="width:100%;height:6px;margin-top:8px;"></span></div>
                <div class="pm-adp-ends"></div>
              </div>`).join('')}
          </div>
        </div>`;
      const _adpSkeletonGrid = () => _adpSkelCard('Dynasty') + _adpSkelCard('Redraft');
      // Render the block with a skeleton up front; the real grid replaces it once
      // the market fetch settles (or falls back to what we have on timeout/error).
      const adpRow = `
        <div class="pm-adp-block" id="pmAdpBlock" data-pid="${playerId}">
          <div class="pm-adp-head">ADP</div>
          <div class="pm-adp-grid" id="pmAdpGrid">${_adpSkeletonGrid()}</div>
        </div>`;

      const _h0 = _heroActive();
      const _heroCardCount = 2 + (ppgCard ? 1 : 0) + (totalCard ? 1 : 0);
      const heroGridStyle = `style="grid-template-columns:repeat(${_heroCardCount},1fr);"`;
      const scoringToggles = `
        <div class="pm-scoring-toggles">
          <div class="pm-trades-toggle pm-scoring-toggle" id="pmScoringTypeToggle" role="tablist" aria-label="Value format">
            <button type="button" class="pm-trades-toggle-btn${pmScoringType === 'dynasty' ? ' active' : ''}" data-scoring="dynasty" role="tab" aria-selected="${pmScoringType === 'dynasty'}">Dynasty</button>
            <button type="button" class="pm-trades-toggle-btn${pmScoringType === 'redraft' ? ' active' : ''}" data-scoring="redraft" role="tab" aria-selected="${pmScoringType === 'redraft'}">Redraft</button>
          </div>
          <div class="pm-trades-toggle pm-scoring-toggle" id="pmScoringFormatToggle" role="tablist" aria-label="Scoring format">
            <button type="button" class="pm-trades-toggle-btn${pmScoringFormat === 'ppr' ? ' active' : ''}" data-format="ppr" role="tab" aria-selected="${pmScoringFormat === 'ppr'}">PPR</button>
            <button type="button" class="pm-trades-toggle-btn${pmScoringFormat === 'half' ? ' active' : ''}" data-format="half" role="tab" aria-selected="${pmScoringFormat === 'half'}">Half</button>
            <button type="button" class="pm-trades-toggle-btn${pmScoringFormat === 'std' ? ' active' : ''}" data-format="std" role="tab" aria-selected="${pmScoringFormat === 'std'}">STD</button>
          </div>
        </div>`;
      let overviewHTML = `
        ${scoringToggles}
        <div class="pm-hero-row" ${heroGridStyle}>
          <div class="pm-hero-stat pm-hero-primary">
            <div class="pm-hero-label">1QB Value${tepPill}</div>
            <div class="pm-hero-val" id="pmHero1qbVal" style="color:#3b82f6;">${_heroFmt(_h0.v1)}</div>
            <div class="pm-hero-sub" id="pmHero1qbSub">${_heroSub(_h0.p1, _h0.o1)}</div>
          </div>
          <div class="pm-hero-stat">
            <div class="pm-hero-label">SF Value${tepPill}</div>
            <div class="pm-hero-val" id="pmHeroSfVal">${_heroFmt(_h0.vs)}</div>
            <div class="pm-hero-sub" id="pmHeroSfSub">${_heroSub(_h0.ps, _h0.os)}</div>
          </div>
          ${ppgCard}
          ${totalCard}
        </div>
        ${adpRow}
      `;

      if (hasChart) {
        overviewHTML += `
          <hr class="pm-section-divider">
          <div class="pm-section-header"><span class="pm-section-label" id="pmValueHistoryLabel">${pmScoringType === 'redraft' ? 'Dynasty Value History' : 'Value History'}</span></div>
          <div class="pm-vh-summary" id="pmVhSummary" aria-live="polite"></div>
          <div class="player-modal-chart-container" id="playerValueChart" style="min-height:200px;"></div>
        `;
      }

      if (data.position && data.position !== 'PICK') {
        overviewHTML += `
          <hr class="pm-section-divider">
          <div class="pm-news-section" id="pmNewsSection">
            <div class="pm-section-header"><span class="pm-section-label">Recent News</span></div>
            <div id="pmNewsBody" style="padding:8px 0;font-size:13px;color:var(--text-muted);max-height:300px;overflow-y:auto;">
              <div class="loading-spinner" style="width:14px;height:14px;flex-shrink:0;"></div>Loading…
            </div>
          </div>
        `;
      }

      // ── Build Adv Metrics panel HTML ──────────────────────────────────────
      const _metricsBase = leagueId
        ? `/${platform}/${season}/${leagueId}/metrics`
        : '/metrics';
      const _metricsPos  = pos && pos !== 'PICK' ? pos : '';
      const _srch = '&search=' + encodeURIComponent(playerName || '');
      const _posSets = {
        QB: { label: 'Passing',   preset: 'Passing'   },
        RB: { label: 'Rushing',   preset: 'Rushing'   },
        WR: { label: 'Receiving', preset: 'Receiving' },
        TE: { label: 'Receiving', preset: 'Receiving' },
      };
      const _posSet = _posSets[_metricsPos];
      // The section title itself links to the filtered Adv Metrics leaderboard
      // (no separate "View in Adv Metrics" button). The info icon stays a
      // separate control - it can't be nested inside the title's link.
      const _advTitle = _posSet
        ? `<a href="${_metricsBase}?pos=${encodeURIComponent(_metricsPos)}&preset=${_posSet.preset}${_srch}" class="pm-section-label-link" title="Open ${_posSet.label} leaderboard">Advanced Metrics</a>`
        : 'Advanced Metrics';
      const metricsHTML = hasMetrics ? `
        <div id="advancedMetricsSection">
          <div class="pm-section-header">
            <span class="pm-section-label">${_advTitle} <span id="advMetricsSeasonLabel" style="font-size:12px;opacity:.6;"></span><span class="adv-info-icon" onclick="advShowInfoTip(event)" aria-label="About metric tooltips">ⓘ</span></span>
          </div>
          <div id="advMetricsPills"></div>
          <div id="advancedMetricsContent">
            <div style="padding:12px 0;display:flex;align-items:center;gap:10px;">
              <div class="loading-spinner" style="width:16px;height:16px;"></div>
              <span style="font-size:13px;color:var(--text-muted);">Loading...</span>
            </div>
          </div>
          <div id="pmWeeklyTrendsWrap" data-position="${pos || ''}" data-pid="${playerId}">
            <button type="button" id="pmWeeklyTrendsBtn" class="pm-weekly-toggle"
              onclick="pmToggleWeeklyTrends('${playerId}')">Trends &#9662;</button>
            <div id="pmWeeklyTrendsBody" style="display:none;"></div>
          </div>
        </div>
      ` : '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">Advanced metrics not available for this player.</div></div>';

      // ── Build Breakout panel HTML (lazy-loaded) ───────────────────────────
      const breakoutHTML = `
        <div style="padding:32px 0;display:flex;align-items:center;justify-content:center;gap:10px;">
          <div class="loading-spinner" style="width:16px;height:16px;"></div>
          <span style="font-size:13px;color:var(--text-muted);">Loading breakout analysis…</span>
        </div>
      `;

      // ── Build Trades panel HTML (lazy-loaded) ─────────────────────────────
      const tradesHTML = `
        <div style="padding:32px 0;display:flex;align-items:center;justify-content:center;gap:10px;">
          <div class="loading-spinner" style="width:16px;height:16px;"></div>
          <span style="font-size:13px;color:var(--text-muted);">Loading trade history…</span>
        </div>
      `;

      // ── Build Prospect panel HTML ─────────────────────────────────────────
      const prospectPanelHTML = isRookieWithProspectData ? pdColHTML
        : '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">No prospect data available.</div></div>';

      // ── Assemble panels into modal body ───────────────────────────────────
      modalBody.innerHTML = `
        <div class="pm-panel pm-panel-active" id="pm-panel-overview">${overviewHTML}</div>
        <div class="pm-panel" id="pm-panel-stats">
          <div class="player-modal-loading" style="padding:40px 0;">
            <div class="loading-spinner"></div>
            <div style="font-size:13px;margin-top:8px;color:var(--text-muted);">Loading stats…</div>
          </div>
        </div>
        <div class="pm-panel" id="pm-panel-team">
          <div class="player-modal-loading" style="padding:40px 0;">
            <div class="loading-spinner"></div>
            <div style="font-size:13px;margin-top:8px;color:var(--text-muted);">Loading team…</div>
          </div>
        </div>
        <div class="pm-panel" id="pm-panel-metrics">${metricsHTML}</div>
        <div class="pm-panel" id="pm-panel-prospect">${prospectPanelHTML}</div>
        <div class="pm-panel" id="pm-panel-breakout">${breakoutHTML}</div>
        <div class="pm-panel" id="pm-panel-trades">${tradesHTML}</div>
        <div class="pm-panel" id="pm-panel-live"></div>
      `;

      // Dynasty/Redraft + PPR/Half/STD toggles: swap hero 1QB/SF values + ranks.
      // Value history stays dynasty PPR (DB), so the chart label clarifies when
      // redraft is selected.
      (function _wireScoringToggle() {
        const typeToggle = document.getElementById('pmScoringTypeToggle');
        const fmtToggle = document.getElementById('pmScoringFormatToggle');
        const applyHero = () => {
          const b = _heroActive();
          const v1 = document.getElementById('pmHero1qbVal');
          const s1 = document.getElementById('pmHero1qbSub');
          const vs = document.getElementById('pmHeroSfVal');
          const ss = document.getElementById('pmHeroSfSub');
          if (v1) v1.textContent = _heroFmt(b.v1);
          if (s1) s1.textContent = _heroSub(b.p1, b.o1);
          if (vs) vs.textContent = _heroFmt(b.vs);
          if (ss) ss.textContent = _heroSub(b.ps, b.os);
          const histLbl = document.getElementById('pmValueHistoryLabel');
          if (histLbl) {
            histLbl.textContent = pmScoringType === 'redraft'
              ? 'Dynasty Value History' : 'Value History';
          }
          if (typeToggle) {
            typeToggle.querySelectorAll('.pm-trades-toggle-btn').forEach(btn => {
              const on = btn.getAttribute('data-scoring') === pmScoringType;
              btn.classList.toggle('active', on);
              btn.setAttribute('aria-selected', on ? 'true' : 'false');
            });
          }
          if (fmtToggle) {
            fmtToggle.querySelectorAll('.pm-trades-toggle-btn').forEach(btn => {
              const on = btn.getAttribute('data-format') === pmScoringFormat;
              btn.classList.toggle('active', on);
              btn.setAttribute('aria-selected', on ? 'true' : 'false');
            });
          }
        };
        if (typeToggle) {
          typeToggle.querySelectorAll('.pm-trades-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              pmScoringType = btn.getAttribute('data-scoring') === 'redraft' ? 'redraft' : 'dynasty';
              applyHero();
            });
          });
        }
        if (fmtToggle) {
          fmtToggle.querySelectorAll('.pm-trades-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              const f = btn.getAttribute('data-format');
              pmScoringFormat = (f === 'half' || f === 'std') ? f : 'ppr';
              applyHero();
            });
          });
        }
      })();

      // ── Load market ADP (BR Fantasy + Consensus), then reveal all at once ──
      // These need per-player draft-crawler DB queries, so they're fetched after
      // the modal opens. The block shows a skeleton until this settles, then
      // Sleeper + market render together (rather than Sleeper appearing first).
      (function _loadMarketAdp() {
        let _done = false;
        const _reveal = (extra) => {
          if (_done) return;
          _done = true;
          const block = document.getElementById('pmAdpBlock');
          const grid = document.getElementById('pmAdpGrid');
          // Bail if the modal was closed or a different player is now shown.
          if (!block || !grid || block.dataset.pid !== String(playerId)) return;
          const inner = _adpGridHTML(_adpSources.concat(extra || []));
          if (inner) {
            grid.innerHTML = inner;
            grid.classList.toggle('pm-adp-tabbed', !!grid.querySelector('.pm-adp-tabs'));
            _wireAdpTabs(grid);
            block.style.display = '';
          } else { block.style.display = 'none'; }   // no ADP anywhere → drop it
        };
        // Safety net: if the request hangs, reveal what we have so the skeleton
        // never sticks.
        const _t = setTimeout(() => _reveal([]), 8000);
        fetch(`/api/player-adp/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}`)
          .then(r => r.ok ? r.json() : null)
          .then(j => { clearTimeout(_t); _reveal(j && Array.isArray(j.sources) ? j.sources : []); })
          .catch(() => { clearTimeout(_t); _reveal([]); });
      })();

      // ── Show tab bar and configure it ─────────────────────────────────────
      const pmTabBar = document.getElementById('pmTabBar');

      // Inject Live/Redzone tab. Shown whenever the season is active (same gate
      // as the Redzone nav item), and always on the Redzone page/Demo itself
      // (#rz-root present). Hidden in the offseason. __seasonActive !== false
      // treats an unknown flag as active, preserving prior behavior.
      const _seasonActive = (window.__seasonActive !== false) || !!document.getElementById('rz-root');
      const _existLive = pmTabBar ? pmTabBar.querySelector('.pm-tab[data-tab="live"]') : null;
      if (_existLive) _existLive.remove();
      if (window.__rzGetPlayerLive && _seasonActive) {
        const _liveBtn = document.createElement('button');
        _liveBtn.className = 'pm-tab pm-tab-live';
        _liveBtn.dataset.tab = 'live';
        _liveBtn.onclick = function() { pmSwitchTab('live'); };
        _liveBtn.innerHTML = '<span class="pm-live-dot"></span>Redzone';
        if (pmTabBar) pmTabBar.appendChild(_liveBtn);
      }
      pmTabBar.style.display = '';
      pmTabBar.dataset.pmPlayerId = playerId;
      pmTabBar.dataset.pmSeason = season;
      pmTabBar.dataset.pmPlayerName = data.name || playerName || '';
      pmTabBar.dataset.pmPosition = data.position || '';

      // Show/hide conditional tabs
      const tabMetrics = document.getElementById('pmTabMetrics');
      if (tabMetrics) tabMetrics.style.display = hasMetrics ? '' : 'none';
      const tabTeam = document.getElementById('pmTabTeam');
      const _teamPos = String(data.position || '').toUpperCase();
      const _showTeamTab = !!(data.team && ['QB', 'RB', 'WR', 'TE'].includes(_teamPos));
      if (tabTeam) tabTeam.style.display = _showTeamTab ? '' : 'none';
      if (pmTabBar) pmTabBar.dataset.pmHasTeam = _showTeamTab ? '1' : '';
      const tabProspect = document.getElementById('pmTabProspect');
      // Prospect tab: only for players with no NFL game logs drafted in the current season
      const _currentNFLYear = new Date().getFullYear();
      const _isCurrentYearProspect = hasProspectData && !hasGameLogs
        && String(pd.draft_class_year) === String(_currentNFLYear);
      if (tabProspect) tabProspect.style.display = _isCurrentYearProspect ? '' : 'none';
      // Breakout tab: only for players flagged as breakout candidates on the board
      // (same set as the BREAKOUT badge via /api/player-indicators).
      const tabBreakout = document.getElementById('pmTabBreakout');
      if (tabBreakout) tabBreakout.style.display = isBreakout(pid) ? '' : 'none';

      // Must be set before pmSwitchTab is called so the metrics lazy-load check works
      if (pmTabBar) pmTabBar.dataset.pmHasMetrics = hasMetrics ? '1' : '';

      // Switch to requested tab, or Overview by default
      const _initialTab = (opts && opts.tab) || 'overview';
      document.querySelectorAll('.pm-tab').forEach(t => t.classList.remove('active'));
      const _initTabBtn = document.querySelector(`.pm-tab[data-tab="${_initialTab}"]`);
      if (_initTabBtn && _initTabBtn.style.display !== 'none') {
        _initTabBtn.classList.add('active');
        pmSwitchTab(_initialTab);
      } else {
        const overviewTabBtn = document.querySelector('.pm-tab[data-tab="overview"]');
        if (overviewTabBtn) overviewTabBtn.classList.add('active');
      }

      // Sliding underline under the active player-modal tab. Init here (not at
      // modal creation) because the bar is display:none until now and its
      // conditional tabs have just been resolved.
      if (window.brSlideTabs) {
        window._pmSlideTabs = window.brSlideTabs(pmTabBar, {
          tabSelector: '.pm-tab', activeClass: 'active', underline: true
        });
      }

      // Warm the other tabs in the background so switching to them is instant.
      pmPrefetchTabs();

      // ── Lazy-load prospect comparables for rookies ─────────────────────────
      if (isRookieWithProspectData && pd.player_id) {
        fetch(`/api/prospects/comparables/${encodeURIComponent(pd.player_id)}`)
          .then(r => r.json())
          .then(cd => {
            const cb = document.getElementById('pmComparablesBody');
            if (!cb) return;
            const comps = cd.comparables || [];
            if (!comps.length) {
              cb.innerHTML = '<span style="color:var(--text-muted);">No close historical comps found.</span>';
              return;
            }
            const tierColors = ['','#10b981','#22d3ee','#3b82f6','#8b5cf6','#a855f7','#f59e0b','#f97316','#94a3b8','#64748b'];
            cb.innerHTML = comps.map(c => {
              const tc = tierColors[c.tier] || '#9ca3af';
              const pickStr = c.actual_pick ? ` · Pick ${c.actual_pick}` : '';
              return `<div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);">
                <div>
                  <span style="font-weight:600;color:var(--text);font-size:13px;">${c.name}</span>
                  <span style="color:var(--text-muted);font-size:12px;margin-left:6px;">${c.draft_class_year}${pickStr}</span>
                  ${c.school ? `<span style="color:var(--text-muted);font-size:12px;"> · ${c.school}</span>` : ''}
                </div>
                <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                  <span style="font-size:12px;color:var(--text-muted);">${parseFloat(c.prospect_score).toFixed(1)}</span>
                  <span style="padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;background:${tc}22;color:${tc};border:1px solid ${tc}44;">T${c.tier}</span>
                </div>
              </div>`;
            }).join('');
          })
          .catch(() => {
            const cb = document.getElementById('pmComparablesBody');
            if (cb) cb.innerHTML = '';
          });
      }

      // ── Lazy-load news into Overview panel ────────────────────────────────
      if (data.position && data.position !== 'PICK') {
        fetch(`/api/player-news/${encodeURIComponent(playerId)}`)
          .then(r => r.json())
          .then(nd => {
            const nb = document.getElementById('pmNewsBody');
            if (!nb) return;
            const items = nd.news || [];
            if (!items.length) {
              nb.innerHTML = '<span style="color:var(--text-muted);font-size:13px;">No recent news found.</span>';
              return;
            }
            nb.innerHTML = items.map(n => `
              <div class="pm-news-item">
                <div class="pm-news-headline">
                  ${n.url
                    ? `<a href="${n.url}" target="_blank" rel="noopener" class="pm-news-link">${n.headline}</a>`
                    : `<span>${n.headline}</span>`}
                </div>
                ${n.description ? `<div class="pm-news-desc">${n.description}</div>` : ''}
                <div class="pm-news-meta">${[n.source, n.age].filter(Boolean).join(' · ')}</div>
              </div>
            `).join('');
          })
          .catch(() => {
            const nb = document.getElementById('pmNewsBody');
            if (nb) nb.innerHTML = '';
          });
      }

      pmInjectContextActions(playerId, playerName, data, leagueId, platform, season);

      // The "vs Avg <pos><tier>" benchmark is reachable from Actions → Compare
      // (it offers the positional-tier averages as pickable opponents),
      // so it is no longer surfaced as a standalone header chip.

      // ── Render value history chart in Overview panel ───────────────────────
      if (data.value_history && data.value_history.length > 0) {
        const chartDiv = document.getElementById('playerValueChart');
        if (chartDiv) {
          const fullHistory = data.value_history;
          const formatDateLabel = (dateStr) => {
            if (!dateStr) return '';
            const m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})/);
            if (!m) return '';
            const [, year, month, day] = m;
            const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            return `${monthNames[parseInt(month, 10) - 1]} ${parseInt(day, 10)}`;
          };
          const parseHistDate = (s) => {
            const m = String(s || '').match(/^(\d{4})-(\d{2})-(\d{2})/);
            return m ? new Date(+m[1], +m[2] - 1, +m[3]) : null;
          };
          const latestD   = parseHistDate(fullHistory[fullHistory.length - 1].as_of_date);
          const earliestD = parseHistDate(fullHistory[0].as_of_date);
          const spanDays  = (latestD && earliestD) ? (latestD - earliestD) / 86400000 : 0;

          // ── Direction 1 "momentum": range-aware summary above the chart ──────
          // Trend is derived from the visible window, so it works even when the
          // backend value_trend field is absent (it currently is). Delta / peak /
          // floor track the selected range, so switching 1 Mo / 3 Mo / All
          // re-answers "where is this value going" rather than just redrawing.
          const _fmtVal = (v) => (typeof fmtInt === 'function')
            ? fmtInt(Math.round(v)) : Math.round(v).toLocaleString('en-US');
          const TREND_META = {
            rising:     { icon: '↑', label: 'Rising',      color: 'var(--win)' },
            recovering: { icon: '↗', label: 'Recovering',  color: '#3b82f6' },
            peaked:     { icon: '↘', label: 'Cooling off', color: '#f59e0b' },
            declining:  { icon: '↓', label: 'Declining',   color: 'var(--loss)' },
            stable:     { icon: '→', label: 'Stable',      color: 'var(--text-muted)' },
          };
          function classifyTrend(vals) {
            const n = vals.length;
            if (n < 2) return null;
            const first = vals[0], last = vals[n - 1];
            const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
            const pct = first ? (last - first) / first : 0;
            if (Math.abs(pct) < 0.03) return TREND_META.stable;
            if (pct > 0) {
              // Climbed back up from a meaningful dip → recovering, else rising.
              return (lo < last * 0.92 && last < hi * 0.995)
                ? TREND_META.recovering : TREND_META.rising;
            }
            // Net down: off a recent high but holding above the low → cooling off.
            return (hi > last * 1.08 && last > lo * 1.02)
              ? TREND_META.peaked : TREND_META.declining;
          }
          function updateSummary(history) {
            const el = document.getElementById('pmVhSummary');
            if (!el) return;
            const vals = history.map(d => Number(d.value_1qb ?? d.value)).filter(v => !isNaN(v));
            if (vals.length < 2) { el.style.display = 'none'; el.innerHTML = ''; return; }
            el.style.display = '';
            const ysf2 = history.map(d => Number(d.value_sf ?? d.value));
            const dual = vals.some((v, i) => Math.abs(v - ysf2[i]) > 1);
            const first = vals[0], last = vals[vals.length - 1];
            const peak = Math.max.apply(null, vals), floor = Math.min.apply(null, vals);
            const diff = last - first;
            const pct = first ? diff / first * 100 : 0;
            const dir = pct > 0.5 ? 'up' : pct < -0.5 ? 'down' : 'flat';
            const arrow = dir === 'up' ? '▲' : dir === 'down' ? '▼' : '→';
            const sign = diff > 0 ? '+' : diff < 0 ? '−' : '';
            const activeBtn = document.querySelector('.pvc-range-bar .pvc-range-btn.is-active');
            const rangeLbl = activeBtn ? activeBtn.textContent.trim() : '';
            const t = classifyTrend(vals);
            const trendPill = t
              ? `<span class="pm-vh-trend" style="color:${t.color};background:color-mix(in srgb, ${t.color} 14%, transparent);">${t.icon} ${t.label}</span>`
              : '';
            el.innerHTML = `
              <div class="pm-vh-now">
                <span class="pm-vh-now-lbl">${dual ? '1QB value' : 'Value'}</span>
                <span class="pm-vh-now-val">${_fmtVal(last)}</span>
              </div>
              <div class="pm-vh-move">
                <div class="pm-vh-move-row">
                  <span class="pm-vh-delta ${dir}">${arrow} ${sign}${_fmtVal(Math.abs(diff))} · ${sign}${Math.abs(pct).toFixed(1)}%</span>
                  ${trendPill}
                </div>
                ${rangeLbl ? `<span class="pm-vh-range">over ${rangeLbl}</span>` : ''}
              </div>
              <div class="pm-vh-extremes">
                <div class="pm-vh-ext"><span class="pm-vh-ext-k">Peak</span><span class="pm-vh-ext-v">${_fmtVal(peak)}</span></div>
                <div class="pm-vh-ext"><span class="pm-vh-ext-k">Floor</span><span class="pm-vh-ext-v">${_fmtVal(floor)}</span></div>
              </div>`;
          }

          function renderChart(history) {
            updateSummary(history);
            const n = history.length;
            const xIdx  = history.map((_, i) => i);            // numeric x for clean tick control
            const dates = history.map(d => formatDateLabel(d.as_of_date));

            const y1qb = history.map(d => Number(d.value_1qb ?? d.value));
            const ysf  = history.map(d => Number(d.value_sf  ?? d.value));

            // Show both 1QB and Superflex lines whenever they genuinely diverge.
            // Both columns are EMA-smoothed at the data layer (smooth_value_history.py),
            // so the SF series is already clean; for players where SF tracks 1QB the
            // two coincide and we fall back to a single "Value" line.
            const hasDualSeries = y1qb.some((v, i) => Math.abs(v - ysf[i]) > 1);

            const allY = hasDualSeries ? [...y1qb, ...ysf] : y1qb;
            const yMin = Math.min(...allY);
            const yMax = Math.max(...allY);
            const yRange = yMax - yMin;
            const yPad = Math.max(yRange * 0.15, 20);
            // Floor: no tighter than (currentValue - 200), so small wiggles don't look huge
            const currentVal = allY[allY.length - 1] ?? yMax;
            const yFloor = Math.max(0, currentVal - 200);

            // 3-4 evenly spaced date ticks (deduped) so labels never collide.
            const tickCount = Math.max(2, Math.min(4, n));
            const tickvals = [];
            if (n <= 1) { tickvals.push(0); }
            else {
              for (let i = 0; i < tickCount; i++) {
                const idx = Math.round(i * (n - 1) / (tickCount - 1));
                if (!tickvals.includes(idx)) tickvals.push(idx);
              }
            }
            const ticktext = tickvals.map(i => dates[i]);

            const rootStyle = getComputedStyle(document.documentElement);
            const mutedColor = rootStyle.getPropertyValue('--text-muted').trim() || '#6b7280';
            const gridColor = rootStyle.getPropertyValue('--border').trim() || 'rgba(127,127,127,.18)';
            // Matches the .pm-adp-dot border (source-card surface) so the chart's
            // end markers read like the ADP dots.
            const rowColor = rootStyle.getPropertyValue('--row').trim() || '#ffffff';

            const hover1qb = dates.map((date, i) => `<b>${date}</b><br>1QB: ${y1qb[i]?.toFixed(1) || ''}`);
            const hoverSF  = dates.map((date, i) => `<b>${date}</b><br>SF: ${ysf[i]?.toFixed(1) || ''}`);

            const trace1qb = {
              x: xIdx, y: y1qb,
              type: 'scatter', mode: 'lines', name: '1QB',
              line: { color: '#3b82f6', width: 2.5, shape: 'spline', smoothing: 1.3 },
              fill: hasDualSeries ? 'none' : 'tozeroy',
              fillcolor: 'rgba(59, 130, 246, 0.08)',
              hovertemplate: '%{text}<extra></extra>',
              text: hover1qb,
            };
            const traceSF = {
              x: xIdx, y: ysf,
              type: 'scatter', mode: 'lines', name: 'SF',
              line: { color: '#f59e0b', width: 2.5, shape: 'spline', smoothing: 1.3 },
              fill: 'none',
              hovertemplate: '%{text}<extra></extra>',
              text: hoverSF,
            };

            const lineTraces = hasDualSeries ? [trace1qb, traceSF] : [
              { ...trace1qb, name: 'Value', text: dates.map((date, i) => `<b>${date}</b><br>Value: ${y1qb[i]?.toFixed(1) || ''}`) }
            ];
            // A single marker at the latest point of each line — echoes the
            // ADP source dots (colored fill, surface-colored ring) so the two
            // blocks share a visual language.
            const endDot = (color, yArr) => ({
              x: [xIdx[n - 1]], y: [yArr[n - 1]],
              type: 'scatter', mode: 'markers', showlegend: false, hoverinfo: 'skip',
              marker: { color, size: 8, line: { color: rowColor, width: 2 } },
            });
            // Peak & floor markers on the primary (1QB) series — Direction 1
            // "momentum": make the high-water mark and the trough self-evident.
            // Skipped when they land on the current point (the end dot already
            // marks it) or when the series is flat / too short.
            const winC  = rootStyle.getPropertyValue('--win').trim()  || '#16a34a';
            const lossC = rootStyle.getPropertyValue('--loss').trim() || '#ef4444';
            const peakIdx  = y1qb.indexOf(Math.max.apply(null, y1qb));
            const floorIdx = y1qb.indexOf(Math.min.apply(null, y1qb));
            const extremeDot = (idx, color) => ({
              x: [xIdx[idx]], y: [y1qb[idx]],
              type: 'scatter', mode: 'markers', showlegend: false, hoverinfo: 'skip',
              marker: { color, size: 7, line: { color: rowColor, width: 2 } },
            });
            const extremeMarkers = [];
            if (n > 2 && peakIdx !== floorIdx) {
              if (peakIdx  !== n - 1) extremeMarkers.push(extremeDot(peakIdx, winC));
              if (floorIdx !== n - 1) extremeMarkers.push(extremeDot(floorIdx, lossC));
            }

            const traces = lineTraces.concat(
              hasDualSeries
                ? [endDot('#3b82f6', y1qb), endDot('#f59e0b', ysf)]
                : [endDot('#3b82f6', y1qb)]
            ).concat(extremeMarkers);

            const isMobile = window.innerWidth <= 768;
            const chartHeight = isMobile ? 200 : 250;
            const xPad = Math.max(n * 0.02, 0.5);

            const layout = {
              // r needs to fit half of the last (right-most) date label, which is
              // centered on the final data point at the plot's right edge — too
              // small and the last date clips off.
              margin: { l: 36, r: 34, t: 10, b: 26 },
              height: chartHeight,
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              showlegend: hasDualSeries,
              legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.14, font: { size: 11, color: mutedColor } },
              xaxis: {
                showgrid: false,
                zeroline: false,
                tickmode: 'array',
                tickvals: tickvals,
                ticktext: ticktext,
                tickangle: 0,
                tickfont: { size: 11, color: mutedColor },
                fixedrange: true,
                range: [-xPad, (n - 1) + xPad],
              },
              yaxis: {
                showgrid: true,
                gridcolor: gridColor,
                griddash: 'dot',
                gridwidth: 1,
                zeroline: false,
                showticklabels: true,
                range: [Math.min(yMin - yPad, yFloor), yMax + yPad],
                tickfont: { size: 11, color: mutedColor },
                nticks: 4,
              },
              hovermode: 'closest',
            };

            if (window.ensurePlotly) window.ensurePlotly().then(function () {
              Plotly.newPlot('playerValueChart', traces, layout, {
                displayModeBar: false,
                responsive: true
              });
            }).catch(function () {});
          }

          // Time-range filter — only show ranges the data actually spans (e.g. a
          // player with 2 weeks of history gets no "3 Mo"/"1 Yr" buttons).
          const RANGES = [{ label: '1 Mo', days: 30 }, { label: '3 Mo', days: 90 }, { label: '1 Yr', days: 365 }];
          const applicable = RANGES.filter(r => spanDays > r.days);
          if (applicable.length && latestD) {
            const subsetFor = (days) => {
              if (!isFinite(days)) return fullHistory;
              const cutoff = latestD.getTime() - days * 86400000;
              const sub = fullHistory.filter(d => { const dd = parseHistDate(d.as_of_date); return dd && dd.getTime() >= cutoff; });
              return sub.length ? sub : fullHistory;
            };
            const ranges = applicable.concat([{ label: 'All', days: Infinity }]);
            const bar = document.createElement('div');
            bar.className = 'otc-day-filters pvc-range-bar';
            bar.innerHTML = ranges.map(r =>
              `<button type="button" class="otc-day-filter pvc-range-btn${r.days === Infinity ? ' is-active' : ''}" data-days="${r.days}">${r.label}</button>`
            ).join('');
            chartDiv.parentNode.insertBefore(bar, chartDiv);
            bar.addEventListener('click', (e) => {
              const b = e.target.closest('.pvc-range-btn');
              if (!b) return;
              bar.querySelectorAll('.pvc-range-btn').forEach(x => x.classList.toggle('is-active', x === b));
              renderChart(subsetFor(parseFloat(b.getAttribute('data-days'))));
            });
          }

          renderChart(fullHistory);
        }
      }

    })
    .catch(err => {
      console.error('Error loading player data:', err);
      const b = document.getElementById('playerModalBody');
      if (!b) return;
      if (window.brErrorState) {
        window.brErrorState(b, 'Please try again.', function () {
          closePlayerModal();
          openPlayerModal(playerId, playerName, opts);
        });
      } else {
        b.innerHTML = `
          <div class="player-modal-loading">
            <div style="color: var(--loss); font-weight: 500;">Error loading player data</div>
            <div style="font-size: 13px;">Please try again</div>
          </div>
        `;
      }
    });
}

// ── Draft Year Edit (player modal) ───────────────────────────────────────────
function pmEditDraftYear(playerId) {
  const editEl = document.getElementById('pmDraftYrEdit');
  if (!editEl) return;
  const showing = editEl.style.display && editEl.style.display !== 'none';
  editEl.style.display = showing ? 'none' : 'flex';
}

function pmSaveDraftYear(playerId) {
  const input = document.getElementById('pmDraftYrInput');
  if (!input) return;
  const val = parseInt(input.value, 10);
  if (!val || val < 2000 || val > 2030) { input.style.borderColor = '#ef4444'; return; }
  input.style.borderColor = '';
  fetch('/api/player-index/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({player_id: playerId, draft_year: val}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        const currentYear = new Date().getFullYear();
        const yrs = Math.max(0, currentYear - val);
        const label = yrs === 0 ? 'Rookie' : `${yrs} yr${yrs !== 1 ? 's' : ''}`;
        const expEl = document.getElementById('pmExpLabel');
        if (expEl) expEl.textContent = label;
        const editEl = document.getElementById('pmDraftYrEdit');
        if (editEl) editEl.style.display = 'none';
      }
    })
    .catch(() => {});
}

function pmLeaguePath(suffix) {
  const parts = window.location.pathname.split('/').filter(Boolean);
  if (parts.length >= 3 && !isNaN(parseInt(parts[1], 10))) {
    return '/' + parts.slice(0, 3).join('/') + suffix;
  }
  const c = window.__brctx || {};
  if (c.leagueId && c.platform && c.season) {
    return '/' + c.platform + '/' + c.season + '/' + c.leagueId + suffix;
  }
  return suffix;
}

function pmCloseActionsMenu(wrap) {
  wrap = wrap || document.getElementById('pmContextActions');
  if (!wrap) return;
  const menu = wrap.querySelector('.pm-actions-dropdown');
  const trigger = wrap.querySelector('.pm-actions-trigger');
  if (menu) menu.hidden = true;
  if (trigger) trigger.setAttribute('aria-expanded', 'false');
  wrap.classList.remove('open');
}

function pmToggleActionsMenu(wrap) {
  if (!wrap) return;
  const menu = wrap.querySelector('.pm-actions-dropdown');
  const trigger = wrap.querySelector('.pm-actions-trigger');
  if (!menu || !trigger) return;
  if (menu.hidden) {
    menu.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
    wrap.classList.add('open');
  } else {
    pmCloseActionsMenu(wrap);
  }
}

function pmInjectContextActions(playerId, playerName, data, leagueId, platform, season) {
  const modal = document.getElementById('playerModal');
  if (!modal) return;

  const slug = pmSlugify(playerName);
  const actions = [
    {
      label: 'Compare',
      run: function () {
        if (typeof openCompareSearch === 'function') openCompareSearch(data);
      },
    },
  ];
  if (leagueId) {
    actions.push({ label: 'Trade For', href: pmLeaguePath('/trade') + '?add=' + encodeURIComponent(playerId) });
    actions.push({ label: 'Recent Trades', run: function () { pmSwitchTab('trades'); } });
  }
  if (slug) {
    actions.push({ label: 'Full Analysis', href: '/player/' + slug + '/trade-value' });
  }

  let wrap = document.getElementById('pmContextActions');
  if (!wrap) {
    wrap = document.createElement('div');
    wrap.id = 'pmContextActions';
    wrap.className = 'pm-actions-menu';
    const closeBtn = modal.querySelector('.player-modal-close');
    if (closeBtn && closeBtn.parentNode) {
      closeBtn.parentNode.insertBefore(wrap, closeBtn);
    } else {
      const tabBar = document.getElementById('pmTabBar');
      if (tabBar) tabBar.parentNode.insertBefore(wrap, tabBar);
      else modal.querySelector('.player-modal-header').after(wrap);
    }
  }

  wrap.textContent = '';
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'player-modal-page-btn pm-actions-trigger';
  trigger.id = 'pmActionsTrigger';
  trigger.setAttribute('aria-haspopup', 'menu');
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-controls', 'pmActionsMenu');
  trigger.setAttribute('aria-label', 'Player actions');
  trigger.title = 'Player actions';
  trigger.appendChild(document.createTextNode('Actions'));
  const chevron = document.createElement('span');
  chevron.className = 'pm-actions-chevron';
  chevron.setAttribute('aria-hidden', 'true');
  trigger.appendChild(chevron);

  const menu = document.createElement('div');
  menu.id = 'pmActionsMenu';
  menu.className = 'pm-actions-dropdown';
  menu.setAttribute('role', 'menu');
  menu.hidden = true;

  actions.forEach(function (a) {
    let el;
    if (a.href) {
      el = document.createElement('a');
      el.href = a.href;
    } else {
      el = document.createElement('button');
      el.type = 'button';
    }
    el.className = 'pm-actions-item';
    el.setAttribute('role', 'menuitem');
    el.textContent = a.label;
    el.addEventListener('click', function () {
      pmCloseActionsMenu(wrap);
      if (typeof a.run === 'function') a.run();
    });
    menu.appendChild(el);
  });

  wrap.appendChild(trigger);
  wrap.appendChild(menu);

  if (!wrap._pmWired) {
    wrap._pmWired = true;
    wrap.addEventListener('click', function (e) {
      if (!e.target.closest('.pm-actions-trigger')) return;
      e.preventDefault();
      e.stopPropagation();
      pmToggleActionsMenu(wrap);
    });
    const overlay = modal.closest('.player-modal-overlay') || modal;
    overlay.addEventListener('click', function (e) {
      if (!wrap.contains(e.target)) pmCloseActionsMenu(wrap);
    });
    overlay.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape') return;
      const openMenu = wrap.querySelector('.pm-actions-dropdown');
      if (!openMenu || openMenu.hidden) return;
      e.preventDefault();
      e.stopPropagation();
      pmCloseActionsMenu(wrap);
      const t = wrap.querySelector('.pm-actions-trigger');
      if (t) t.focus();
    });
  }
}

// ── Player Modal Tab Switching (global) ──────────────────────────────────────
function pmSwitchTab(tab) {
  document.querySelectorAll('.pm-panel').forEach(p => p.classList.remove('pm-panel-active'));
  document.querySelectorAll('.pm-tab').forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
  });
  const panel = document.getElementById('pm-panel-' + tab);
  const btn = document.querySelector('.pm-tab[data-tab="' + tab + '"]');
  if (panel) panel.classList.add('pm-panel-active');
  if (btn) {
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
  }
  // The Team tab manages its own edge-to-edge section padding, so drop the
  // modal body's inset while it's active.
  const _pmBodyEl = document.getElementById('playerModalBody');
  if (_pmBodyEl) _pmBodyEl.classList.toggle('pm-body-flush', tab === 'team');
  if (window._pmSlideTabs) window._pmSlideTabs.sync(true);

  const pmTabBar = document.getElementById('pmTabBar');
  if (!pmTabBar) return;
  const playerId = pmTabBar.dataset.pmPlayerId;
  const season = pmTabBar.dataset.pmSeason;

  // ── Lazy-load Adv Metrics tab ────────────────────────────────────────────
  if (tab === 'metrics' && panel && !panel.dataset.loaded && pmTabBar && pmTabBar.dataset.pmHasMetrics) {
    panel.dataset.loaded = '1';
    const path = window.location.pathname;
    const match = path.match(/\/(sleeper|espn|yahoo|mfl)\/(\d+)\/([^\/]+)/);
    const leagueIdForMetrics = match ? match[3] : null;
    loadAdvancedMetrics(playerId, leagueIdForMetrics, 'auto');
  }

  // ── Lazy-load Breakout tab ───────────────────────────────────────────────
  if (tab === 'breakout' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    const _isPremium = document.getElementById('page-root')?.dataset.premium === 'true';
    if (!_isPremium) {
      panel.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:48px 24px;text-align:center;">
          <i class="fa-solid fa-lock" style="font-size:28px;color:var(--text-muted);opacity:0.5;"></i>
          <div style="font-size:15px;font-weight:700;color:var(--text);">Breakout Analysis is a PRO feature</div>
          <div style="font-size:13px;color:var(--text-muted);max-width:280px;line-height:1.5;">
            See breakout scores, opportunity drivers, hit probability, and PPG projections for every candidate.
          </div>
          <button onclick="showPaywall('breakout-candidates')"
                  style="margin-top:4px;padding:9px 20px;background:linear-gradient(135deg,#122d4b,#2563eb);
                         color:#fff;border:none;border-radius:10px;font-size:13px;font-weight:700;cursor:pointer;">
            Upgrade to PRO
          </button>
        </div>`;
      return;
    }
    const _boMatch = window.location.pathname.match(/\/(sleeper|espn|yahoo|mfl)\/(\d+)\/([^\/]+)/);
    const _boLeague = _boMatch ? _boMatch[3] : '';
    const _boPlatform = _boMatch ? _boMatch[1] : 'sleeper';
    fetch(`/api/breakout/player/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}&league_id=${encodeURIComponent(_boLeague)}&platform=${encodeURIComponent(_boPlatform)}`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (!panel.isConnected) return;
        if (!data || data.available === false || (data.breakout_opportunity_score == null && !data.breakout_blend)) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">Not in this week’s board.</div></div>';
          return;
        }
        const score = parseFloat(data.breakout_opportunity_score || 0);
        let scoreColor = '#10b981';
        if (score < 50) scoreColor = '#3b82f6';
        if (score < 40) scoreColor = '#f59e0b';
        if (score < 30) scoreColor = '#6b7280';
        panel.innerHTML = _buildBkTabHTML(data, scoreColor);
      })
      .catch(() => {
        if (panel.isConnected) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">Breakout analysis not available.</div></div>';
        }
      });
  }

  // ── Live tab (Redzone context) ───────────────────────────────────────────
  if (tab === 'live' && panel && window.__rzGetPlayerLive) {
    panel.innerHTML = window.__rzGetPlayerLive(playerId);
    if (window._rzSyncTabLive) window._rzSyncTabLive(panel);
  }

  // ── Lazy-load Team tab (10-min localStorage cache for instant re-opens) ───
  if (tab === 'team' && panel && !panel.dataset.loaded && pmTabBar && pmTabBar.dataset.pmHasTeam) {
    panel.dataset.loaded = '1';
    const _teamUrl = `/api/player-team/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}`;
    const _teamKey = 'pm_team_v1_' + _teamUrl;
    const _teamTTL = 10 * 60 * 1000;
    const _renderTeam = (data) => {
      if (!panel.isConnected) return;
      if (!data || data.available === false) {
        window.brEmptyState(panel, { icon: 'search', title: 'No team data', message: 'Team context is not available for this player.', compact: true });
        return;
      }
      panel.innerHTML = _pmBuildTeamHTML(data);
      _pmWireTeamPanel(panel);
    };
    let _teamCached = null;
    try {
      const _e = JSON.parse(localStorage.getItem(_teamKey) || 'null');
      if (_e && Date.now() - _e.ts < _teamTTL) _teamCached = _e.data;
    } catch (_) {}
    if (_teamCached) {
      _renderTeam(_teamCached);
    } else {
      fetch(_teamUrl)
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(data => {
          try { localStorage.setItem(_teamKey, JSON.stringify({ ts: Date.now(), data })); } catch (_) {}
          _renderTeam(data);
        })
        .catch(() => {
          if (panel.isConnected) {
            window.brErrorState(panel, 'Could not load team.', () => { panel.dataset.loaded = ''; pmSwitchTab(tab); }, { compact: true });
          }
        });
    }
  }

  // ── Lazy-load Stats tab ──────────────────────────────────────────────────
  if (tab === 'stats' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    const pathParts2 = window.location.pathname.split('/').filter(p => p);
    const _platform = pathParts2[0] || 'sleeper';
    const _season   = pathParts2[1] || new Date().getFullYear();
    const _leagueId = pathParts2[2] || null;
    const _lt = brLeagueType();
    const _ls = brLeagueSize();
    let logsUrl = `/api/player-game-logs/${encodeURIComponent(playerId)}?season=${_season}&league_type=${_lt}&league_size=${_ls}`;
    if (_leagueId) logsUrl += `&league_id=${_leagueId}&platform=${_platform}`;
    fetch(logsUrl)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (!panel.isConnected) return;
        const logsByYear = data.game_logs_by_year || {};
        if (!Object.keys(logsByYear).length) {
          window.brEmptyState(panel, { icon: 'search', title: 'No game logs', message: 'No game-by-game data is available for this player yet.', compact: true });
          return;
        }
        panel.innerHTML = _buildStatsHTML(logsByYear, false, (pmTabBar && pmTabBar.dataset.pmPosition) || '');
      })
      .catch(() => {
        if (panel.isConnected) {
          window.brErrorState(panel, 'Could not load stats.', () => { panel.dataset.loaded = ''; pmSwitchTab(tab); }, { compact: true });
        }
      });
  }

  // ── Lazy-load Trades tab (This League ↔ Trade DB toggle) ────────────────
  if (tab === 'trades' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    pmLoadTradesTab(panel, playerId, season, pmTabBar);
  }
}

function _pmTradePathCtx() {
  const pathParts = window.location.pathname.split('/').filter(p => p);
  const platform = pathParts[0];
  const season = pathParts[1];
  const leagueId = pathParts[2];
  const isLeague = !!(platform && season && leagueId &&
    !['players','breakouts','prospects','trade-database','trade-intel','rankings','compare','guides','glossary','pricing','portfolio','watchlist'].includes(platform));
  return { platform, season, leagueId, isLeague };
}

function _pmRenderTradeAssets(assets, playerId) {
  if (!assets || !assets.length) {
    return '<span style="font-size:12px;color:var(--text-muted);">-</span>';
  }
  return assets.map(a => {
    const isPick = a.type === 'pick' || a.is_pick ||
      (a.name || '').toLowerCase().includes('pick') ||
      (a.name || '').toLowerCase().includes('round');
    const isFocus = String(a.player_id || '') === String(playerId) || a.is_focus;
    const cls = isPick ? 'pm-trade-asset pm-pick' : (isFocus ? 'pm-trade-asset pm-focus' : 'pm-trade-asset');
    let label = a.name || a.player_name || '?';
    if (a.drafted_player && a.drafted_player.name && isPick && !String(label).includes('→')) {
      label = `${label} → ${a.drafted_player.name}`;
    }
    const pos = (!isPick && a.position) ? `<span class="pm-trade-pos">${a.position}</span>` : '';
    return `<div class="${cls}">${label}${pos}</div>`;
  }).join('');
}

function _pmNormalizeTradeSides(t) {
  // League API: { team_name, assets[] }. Trade DB: assets[] on side_a/side_b.
  const norm = (side) => {
    if (!side) return { team_name: '', assets: [] };
    if (Array.isArray(side)) return { team_name: '', assets: side };
    return {
      team_name: side.team_name || '',
      assets: side.assets || [],
    };
  };
  return { a: norm(t.side_a), b: norm(t.side_b) };
}

function _pmRenderTradeCards(trades, playerId, { showTeams } = {}) {
  return trades.map(t => {
    const dateStr = t.date
      ? (String(t.date).includes('/') ? t.date
        : new Date(t.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }))
      : '-';
    const seasonBit = t.season ? `<span class="pm-trade-season">${t.season}</span>` : '';
    const sfBadge = (t.is_superflex === true || t.league_type === 'sf' || t.league_type === 'superflex')
      ? '<span class="pm-trade-badge pm-trade-badge-sf">SF</span>'
      : (t.is_superflex === false
        ? '<span class="pm-trade-badge pm-trade-badge-1qb">1QB</span>'
        : '');
    const sides = _pmNormalizeTradeSides(t);
    const teamA = (showTeams && sides.a.team_name)
      ? `<div class="pm-trade-team">${sides.a.team_name}</div>` : '';
    const teamB = (showTeams && sides.b.team_name)
      ? `<div class="pm-trade-team">${sides.b.team_name}</div>` : '';
    return `<div class="pm-trade-card">
      <div class="pm-trade-head">
        <span class="pm-trade-date">${dateStr}${seasonBit ? ' · ' + seasonBit : ''}</span>
        <div style="display:flex;gap:5px;">${sfBadge}</div>
      </div>
      <div class="pm-trade-body">
        <div class="pm-trade-col">${teamA}${_pmRenderTradeAssets(sides.a.assets, playerId)}</div>
        <div class="pm-trade-swap">⇄</div>
        <div class="pm-trade-col">${teamB}${_pmRenderTradeAssets(sides.b.assets, playerId)}</div>
      </div>
    </div>`;
  }).join('');
}

function pmLoadTradesTab(panel, playerId, season, pmTabBar) {
  const playerName = (pmTabBar && pmTabBar.dataset.pmPlayerName) || '';
  const ctx = _pmTradePathCtx();
  const tdbBase = ctx.isLeague
    ? `/${ctx.platform}/${ctx.season}/${ctx.leagueId}/trade-database`
    : '/trade-database';
  const tdbLink = playerName
    ? `${tdbBase}?q=${encodeURIComponent(playerName)}`
    : tdbBase;

  const defaultScope = ctx.isLeague ? 'league' : 'db';
  const saved = panel.dataset.pmTradeScope || defaultScope;
  const scope = (saved === 'league' && !ctx.isLeague) ? 'db' : saved;
  panel.dataset.pmTradeScope = scope;

  const toggleHTML = ctx.isLeague
    ? `<div class="pm-trades-toggle" role="tablist" aria-label="Trade source">
        <button type="button" class="pm-trades-toggle-btn${scope === 'league' ? ' active' : ''}" data-scope="league">This League</button>
        <button type="button" class="pm-trades-toggle-btn${scope === 'db' ? ' active' : ''}" data-scope="db">Trade DB</button>
      </div>`
    : `<div class="pm-trades-toggle-note">Showing trades from the Trade Database</div>`;

  const linkHTML = `<div class="pm-trades-footer"><a href="${tdbLink}">Search all trades in Trade Database →</a></div>`;
  const bodyId = 'pm-trades-body';
  panel.innerHTML = `${toggleHTML}<div id="${bodyId}" class="pm-trades-body"><div class="player-modal-loading" style="padding:28px 0;"><div class="loading-spinner"></div><div style="color:var(--text-muted);font-size:13px;margin-top:8px;">Loading trade history…</div></div></div>${linkHTML}`;

  panel.querySelectorAll('.pm-trades-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const next = btn.dataset.scope;
      if (!next || next === panel.dataset.pmTradeScope) return;
      panel.dataset.pmTradeScope = next;
      panel.querySelectorAll('.pm-trades-toggle-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.scope === next);
      });
      _pmFetchTradesInto(panel, playerId, season, ctx);
    });
  });

  _pmFetchTradesInto(panel, playerId, season, ctx);
}

function _pmFetchTradesInto(panel, playerId, season, ctx) {
  const body = panel.querySelector('.pm-trades-body');
  if (!body) return;
  const scope = panel.dataset.pmTradeScope || 'db';
  body.innerHTML = '<div class="player-modal-loading" style="padding:28px 0;"><div class="loading-spinner"></div><div style="color:var(--text-muted);font-size:13px;margin-top:8px;">Loading trade history…</div></div>';

  let url;
  if (scope === 'league' && ctx.isLeague) {
    url = `/api/player-league-trades/${encodeURIComponent(playerId)}?platform=${encodeURIComponent(ctx.platform)}&league_id=${encodeURIComponent(ctx.leagueId)}&season=${encodeURIComponent(ctx.season || season)}&limit=50`;
  } else {
    url = `/api/trade-intel/player-trades/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}&limit=20`;
    if (ctx.platform) url += `&platform=${encodeURIComponent(ctx.platform)}`;
    if (ctx.leagueId) url += `&league_id=${encodeURIComponent(ctx.leagueId)}`;
  }

  fetch(url)
    .then(r => r.json().then(d => ({ status: r.status, ok: r.ok, d: d || {} })).catch(() => ({ status: r.status, ok: r.ok, d: {} })))
    .then(res => {
      if (!panel.isConnected || !body.isConnected) return;
      if (res.status === 403 && res.d.paywall) {
        body.innerHTML = '<div style="padding:18px 0;text-align:center;">'
          + '<div style="font-size:13px;color:var(--text-muted);margin-bottom:10px;">Market trade history is a PRO feature.</div>'
          + '<button type="button" class="login-gate-btn" id="pmTradesPaywallBtn">Upgrade</button></div>';
        var btn = body.querySelector('#pmTradesPaywallBtn');
        if (btn) btn.addEventListener('click', function () {
          if (typeof showPaywall === 'function') showPaywall('trade-history');
        });
        return;
      }
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = res.d;
      const trades = data.trades || [];
      if (!trades.length) {
        const emptyMsg = scope === 'league'
          ? 'No trades for this player in this league yet.'
          : 'No recent trades found for this player.';
        body.innerHTML = `<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">${emptyMsg}</div></div>`;
        return;
      }
      body.innerHTML = `<div style="padding:4px 0;">${_pmRenderTradeCards(trades, playerId, { showTeams: scope === 'league' })}</div>`;
    })
    .catch(() => {
      if (panel.isConnected && body.isConnected) {
        window.brErrorState(body, 'Could not load trade history.', () => {
          _pmFetchTradesInto(panel, playerId, season, ctx);
        }, { compact: true });
      }
    });
}

// Prefetch the lazy tabs (Stats / Trades / Adv Metrics) once the modal's
// Overview has loaded, so clicking a tab shows already-rendered content instead
// of a spinner. We reuse pmSwitchTab's exact load path by briefly activating
// each un-loaded tab and restoring the current one — all synchronously in one
// idle callback, so no intermediate tab state is ever painted.
// ── Team tab (player modal) ───────────────────────────────────────────────────
let _pmTeamAdvOpen = false;

// Tier color for a plain "higher rank = better" stat (Scoring, Pace), as a
// theme token so hero rank labels track light/dark like the profile bars.
function _pmTeamTierColor(rank, total) {
  if (rank == null || !total) return 'var(--text)';
  const third = Math.ceil(total / 3);
  if (rank <= third) return 'var(--win)';
  if (rank <= third * 2) return 'var(--warning)';
  return 'var(--loss)';
}

function _pmFmtTeamVal(key, val) {
  if (val == null || val === '') return '—';
  const n = Number(val);
  if (Number.isNaN(n)) return String(val);
  if (key === 'pass_rate') return (n * 100).toFixed(1) + '%';
  if (key === 'points' || key === 'plays_pg') return n.toFixed(1);
  if (key === 'pass_yds' || key === 'rush_yds' || key === 'total_yds' || key === 'pass_att' || key === 'rush_att') {
    return Math.round(n).toLocaleString();
  }
  if (key === 'pass_tds' || key === 'rush_tds') return String(Math.round(n));
  return String(n);
}

function _pmTeamInjBadge(injury) {
  const injRaw = String(injury || '').trim();
  if (!injRaw) return '';
  const u = injRaw.toUpperCase();
  let icls = 'player-badge-inj-q';
  if (['IR', 'OUT', 'O', 'PUP', 'SUSP', 'SUS', 'SUSPENDED', 'NFI', 'DNR', 'COV'].includes(u)) {
    icls = 'player-badge-inj-out';
  } else if (['DOUBTFUL', 'D'].includes(u)) {
    icls = 'player-badge-inj-d';
  }
  const code = u === 'QUESTIONABLE' ? 'Q'
    : u === 'DOUBTFUL' ? 'D'
    : (u === 'OUT' || u === 'O') ? 'OUT'
    : (u === 'SUSP' || u === 'SUS' || u === 'SUSPENDED') ? 'SUS'
    : (u.length > 4 ? u.slice(0, 4) : u);
  return `<span class="player-badge ${icls}" title="${injRaw.replace(/"/g, '&quot;')}"><i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> ${code}</span>`;
}

function _pmTeamOrd(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
function _pmTeamOrdSup(n) {
  return _pmTeamOrd(n).replace(/(st|nd|rd|th)$/, '<sup>$1</sup>');
}

// Plain rank tiers for offense-profile dots: top third green (good), middle
// yellow (mid), bottom red (bad). Same scale as Scoring/Pace hero ranks —
// not inverted by whether the tendency "helps" the player's position.
function _pmTeamMetricColor(pos, key, rank, total) {
  return _pmTeamTierColor(rank, total);
}

function _pmTeamProfileAxis() {
  return '<div class="pm-tp-axis"><span></span><span class="pm-tp-ends"><span class="l">32ND</span><span class="m">AVG</span><span class="r">1ST</span></span><span></span></div>';
}
function _pmTeamProfileRow(pos, label, key, entry) {
  if (!entry || entry.rank == null) return '';
  const total = entry.total || 32, rank = entry.rank;
  const x = Math.max(3, Math.min(97, ((total - rank) / Math.max(1, total - 1)) * 100));
  const c = _pmTeamMetricColor(pos, key, rank, total);
  const val = _pmFmtTeamVal(key, entry.value);
  const tip = `${label}: ${_pmTeamOrd(rank)} of ${total} · ${val}`.replace(/"/g, '&quot;');
  // Whole row is hoverable/focusable (not just the 13px dot) so the detail is
  // reachable by pointer and keyboard. Uses the shared themed tooltip engine
  // (advEnterMetricDef/advShowMetricDef) rather than a native `title`: the
  // modal body clips overflow, so this fixed-position bubble actually shows
  // (and matches the app theme) where a native tooltip did not.
  return `<div class="pm-tp-row" data-def="${tip}" onmouseenter="advEnterMetricDef(event)" onmouseleave="advLeaveMetricDef(event)" onclick="advShowMetricDef(event)" tabindex="0" aria-label="${tip}">
    <span class="pm-tp-label">${label}</span>
    <span class="pm-tp-track"><span class="pm-tp-base"></span><span class="pm-tp-mid"></span>
      <span class="pm-tp-fill" style="width:${x}%;background:${c}"></span>
      <span class="pm-tp-dot" style="left:${x}%;background:${c}"></span></span>
    <span class="pm-tp-end"><span class="pm-tp-ord" style="color:${c}">${_pmTeamOrdSup(rank)}</span><span class="pm-tp-val">${val}</span></span>
  </div>`;
}

// Target-share bar (WR/TE only — the endpoint carries target share, not carries).
function _pmTeamShareBar(data) {
  const pos = String(data.position || '').toUpperCase();
  if (pos !== 'WR' && pos !== 'TE') return '';
  const room = (data.depth_chart && data.depth_chart[pos]) || [];
  const segs = room.filter(p => p.tgt_share != null && p.tgt_share > 0)
    .map(p => ({ name: p.name, pct: p.tgt_share, me: p.is_focus }));
  if (!segs.length) return '';
  const rest = Math.max(0, 100 - segs.reduce((a, s) => a + s.pct, 0));
  const me = segs.find(s => s.me);
  // Theme-adaptive neutral ramp (subtle text blended toward the card) so the
  // teammate segments read correctly in both light and dark themes.
  const grays = [
    'color-mix(in srgb, var(--text-subtle) 85%, var(--card))',
    'color-mix(in srgb, var(--text-subtle) 66%, var(--card))',
    'color-mix(in srgb, var(--text-subtle) 50%, var(--card))',
    'color-mix(in srgb, var(--text-subtle) 37%, var(--card))',
    'color-mix(in srgb, var(--text-subtle) 27%, var(--card))',
  ];
  let gi = 0;
  const bars = segs.map(s => {
    const col = s.me ? 'var(--accent)' : grays[Math.min(gi++, grays.length - 1)];
    const tip = `${s.name}: ${s.pct}% of team targets`;
    return `<i class="${s.me ? 'me' : ''}" style="width:${s.pct}%;background:${col}" title="${tip.replace(/"/g, '&quot;')}">${s.pct >= 9 ? '<span>' + s.pct + '%</span>' : ''}</i>`;
  }).join('');
  const restBar = rest > 3 ? `<i style="width:${rest}%;background:var(--border)" title="Rest of offense: ${rest}%"></i>` : '';
  const last = String(data.player_name || '').split(' ').slice(-1)[0];
  return `<div class="pm-tshare-cap"><span>Team target share</span><span><b>${me ? me.pct + '%' : '—'}</b>${me ? ' to ' + last : ''}</span></div>
    <div class="pm-tshare-bar">${bars}${restBar}</div>`;
}

function _pmTeamRoomRow(row, i) {
  const clickable = !row.is_focus && !!row.id;
  const cls = 'pm-troom-row' + (row.is_focus ? ' pm-troom-focus' : (clickable ? ' pm-troom-click' : '')) + (row.order === 1 ? ' pm-troom-starter' : '');
  const attrs = clickable
    ? ` data-pid="${row.id}" data-pname="${String(row.name || '').replace(/"/g, '&quot;')}" role="button" tabindex="0"`
    : '';
  const inj = _pmTeamInjBadge(row.injury);
  const snap = row.snap_pct != null
    ? `<span class="pm-troom-snap"><span class="pm-troom-bar"><span style="width:${Math.min(100, row.snap_pct)}%"></span></span><b>${row.snap_pct}%${row.snap_pct_source === 'derived' ? '<span class="pm-snap-est">est.</span>' : ''}</b></span>`
    : '<span class="pm-troom-num mut">—</span>';
  return `<div class="${cls}"${attrs}>
    <span class="pm-troom-slot">${i + 1}</span>
    <span class="pm-troom-name">${row.name || '—'}${inj}</span>
    ${snap}
    <span class="pm-troom-num${row.tgt_share == null ? ' mut' : ''}">${row.tgt_share != null ? row.tgt_share + '%' : '—'}</span>
    <span class="pm-troom-num${row.ppg == null ? ' mut' : ''}">${row.ppg != null ? row.ppg : '—'}</span>
  </div>`;
}

function _pmTeamMiniItem(row, i) {
  const clickable = !row.is_focus && !!row.id;
  const cls = 'pm-team-depth-row pm-mini-item' + (row.is_focus ? ' pm-mini-focus' : (clickable ? ' pm-mini-click' : ''));
  const attrs = clickable
    ? ` data-pid="${row.id}" data-pname="${String(row.name || '').replace(/"/g, '&quot;')}" role="button" tabindex="0"`
    : '';
  const inj = _pmTeamInjBadge(row.injury);
  return `<div class="${cls}"${attrs}><span class="pm-mini-ord">${i + 1}</span><span class="pm-mini-name">${row.name || '—'}</span>${inj}</div>`;
}

function _pmBuildTeamHTML(data) {
  const team = data.team || '';
  const pos = String(data.position || '').toUpperCase();
  const crestAbbr = team.slice(0, 2);
  const logoImg = data.logo
    ? `<img class="pm-team-logo" src="${data.logo}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="pm-crest" style="display:none" aria-hidden="true">${crestAbbr}</div>`
    : `<div class="pm-crest" aria-hidden="true">${crestAbbr}</div>`;
  const wm = data.logo ? `<img class="pm-team-wm" src="${data.logo}" alt="" aria-hidden="true">` : '';
  const bye = data.bye_week != null ? `Bye ${data.bye_week}` : '';
  const posLine = [data.position, team, bye].filter(Boolean).join(' · ');

  const ranks = data.ranks || {};
  const rm = data.ranks_more || {};
  const pr = (rm.pass_rate && rm.pass_rate.value != null) ? Math.round(Number(rm.pass_rate.value) * 100) : null;
  const heroStats = [
    ranks.points ? `<div class="pm-hero-stat"><div class="pm-hero-label">Scoring</div><div class="pm-hero-val" style="color:${_pmTeamTierColor(ranks.points.rank, ranks.points.total)}">${_pmTeamOrdSup(ranks.points.rank)}</div></div>` : '',
    rm.plays_pg ? `<div class="pm-hero-stat"><div class="pm-hero-label">Pace</div><div class="pm-hero-val" style="color:${_pmTeamTierColor(rm.plays_pg.rank, rm.plays_pg.total)}">${_pmTeamOrdSup(rm.plays_pg.rank)}</div></div>` : '',
    pr != null ? `<div class="pm-hero-stat pm-hero-split"><div class="pm-hero-label">Pass / Run</div>
      <div class="pm-hero-splitbar"><i style="width:${pr}%;background:var(--accent)"></i><i style="width:${100 - pr}%;background:var(--border)"></i></div>
      <div class="pm-hero-splitlbl"><span>${pr}% Pass</span><span>${100 - pr}% Run</span></div></div>` : '',
  ].join('');

  const profile = [['Pass Yards', 'pass_yds'], ['Pass Attempts', 'pass_att'], ['Rush Yards', 'rush_yds'], ['Rush Attempts', 'rush_att']]
    .map(function (m) { return _pmTeamProfileRow(pos, m[0], m[1], ranks[m[1]]); }).join('');
  const moreProfile = [['Total Yards', 'total_yds'], ['Pass TDs', 'pass_tds'], ['Rush TDs', 'rush_tds'], ['Pass Rate', 'pass_rate'], ['Plays / Game', 'plays_pg']]
    .map(function (m) { return _pmTeamProfileRow(pos, m[0], m[1], rm[m[1]]); }).join('');

  const room = (data.depth_chart && data.depth_chart[pos]) || [];
  const roomRows = room.length ? room.map(_pmTeamRoomRow).join('') : '<div class="pm-team-depth-empty">—</div>';
  const shareBar = _pmTeamShareBar(data);

  const otherPos = ['QB', 'RB', 'WR', 'TE'].filter(function (p) { return p !== pos; });
  const miniCols = otherPos.map(function (p) {
    const rows = (data.depth_chart && data.depth_chart[p]) || [];
    const body = rows.length ? rows.slice(0, 5).map(_pmTeamMiniItem).join('') : '<div class="pm-team-depth-empty">—</div>';
    return `<div class="pm-mini-col"><h5>${p}</h5>${body}</div>`;
  }).join('');

  const roleName = String(data.player_name || '').split(' ').slice(-1)[0] || pos;
  const advOpen = _pmTeamAdvOpen;
  const advChev = advOpen ? '&#9662;' : '&#9656;';
  const advHint = advOpen ? 'click to collapse' : 'click to expand';

  return `<div class="pm-team-wrap">
    <div class="pm-team-header">
      ${wm}
      <div class="pm-team-headtop">
        ${logoImg}
        <div class="pm-team-header-text">
          <div class="pm-team-name">${data.team_name || team}</div>
          <div class="pm-team-meta">${posLine}</div>
        </div>
        <span class="pm-team-season">${data.stats_season} stats</span>
      </div>
      ${heroStats ? '<div class="pm-team-herostats">' + heroStats + '</div>' : ''}
    </div>
    <div class="pm-team-sec">
      <div class="pm-section-header"><span class="pm-section-label">Offense Profile</span><span class="pm-team-secnote">${data.stats_season} · rank of 32</span></div>
      ${_pmTeamProfileAxis()}${profile}
      <div class="pm-team-note">Dot = team rank (right = 1st). Color = rank tier: <b style="color:var(--win)">green good</b>, <b style="color:var(--warning)">yellow mid</b>, <b style="color:var(--loss)">red bad</b>.</div>
      <div class="pm-section-header pm-section-collapsible pm-team-adv-toggle" role="button" tabindex="0" aria-expanded="${advOpen ? 'true' : 'false'}" aria-controls="pmTeamAdvBody">
        <span class="pm-collapse-chevron" aria-hidden="true">${advChev}</span>
        <span class="pm-section-label">More team ranks</span>
        <span class="pm-collapse-hint">${advHint}</span>
      </div>
      <div class="pm-team-adv-body" id="pmTeamAdvBody"${advOpen ? '' : ' hidden'}>
        ${_pmTeamProfileAxis()}${moreProfile}
      </div>
    </div>
    <div class="pm-team-sec">
      <div class="pm-section-header"><span class="pm-section-label">${roleName}&#39;s Role</span><span class="pm-team-secnote">${pos} room</span></div>
      ${shareBar}
      <div class="pm-team-usage">
        <div class="pm-troom-row pm-troom-head"><span></span><span>${pos} Room</span><span>Snap %</span><span>Tgt %</span><span>PPG</span></div>
        ${roomRows}
      </div>
      <div class="pm-team-note">Depth order + injuries from Sleeper. Tap any teammate to open their card.</div>
    </div>
    <div class="pm-team-sec">
      <div class="pm-section-header"><span class="pm-section-label">Rest of Depth Chart</span><span class="pm-team-secnote">Sleeper order</span></div>
      <div class="pm-team-depth"><div class="pm-mini-grid">${miniCols}</div></div>
    </div>
  </div>`;
}


function _pmWireTeamPanel(panel) {
  if (!panel) return;
  // Any clickable teammate (depth chart or usage table) carries data-pid.
  const _pmOpenTeammate = (row) => {
    const pid = row.dataset.pid;
    const pname = row.dataset.pname;
    if (!pid || typeof openPlayerModal !== 'function') return;
    // Replace the current modal rather than stacking a second overlay on top.
    const ov = document.querySelector('.player-modal-overlay');
    if (ov) { document.body.style.overflow = ''; ov.remove(); }
    openPlayerModal(pid, pname, { force: true });
  };
  panel.querySelectorAll('[data-pid]').forEach(row => {
    row.addEventListener('click', () => _pmOpenTeammate(row));
    row.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        _pmOpenTeammate(row);
      }
    });
  });
  const toggle = panel.querySelector('.pm-team-adv-toggle');
  const body = panel.querySelector('.pm-team-adv-body');
  if (toggle && body) {
    const flipAdv = () => {
      _pmTeamAdvOpen = !_pmTeamAdvOpen;
      toggle.setAttribute('aria-expanded', _pmTeamAdvOpen ? 'true' : 'false');
      body.hidden = !_pmTeamAdvOpen;
      const chev = toggle.querySelector('.pm-collapse-chevron');
      const hint = toggle.querySelector('.pm-collapse-hint');
      if (chev) chev.innerHTML = _pmTeamAdvOpen ? '&#9662;' : '&#9656;';
      if (hint) hint.textContent = _pmTeamAdvOpen ? 'click to collapse' : 'click to expand';
      if (hint) hint.style.opacity = _pmTeamAdvOpen ? '0.8' : '';
    };
    toggle.addEventListener('click', flipAdv);
    toggle.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        flipAdv();
      }
    });
  }
}

function pmPrefetchTabs() {
  const bar = document.getElementById('pmTabBar');
  if (!bar || bar.dataset.pmPrefetched) return;
  bar.dataset.pmPrefetched = '1';
  const run = function () {
    if (!document.getElementById('pmTabBar')) return; // modal closed meanwhile
    const activeBtn = document.querySelector('.pm-tab.active');
    const activeTab = (activeBtn && activeBtn.dataset.tab) || 'overview';
    const tabs = ['stats', 'trades'];
    if (bar.dataset.pmHasTeam) tabs.push('team');
    if (bar.dataset.pmHasMetrics) tabs.push('metrics');
    tabs.forEach(function (t) {
      const panel = document.getElementById('pm-panel-' + t);
      const btn = document.querySelector('.pm-tab[data-tab="' + t + '"]');
      // Only warm tabs that exist, are visible, and haven't loaded yet.
      if (panel && !panel.dataset.loaded && btn && btn.style.display !== 'none') {
        pmSwitchTab(t);
      }
    });
    pmSwitchTab(activeTab); // restore — net-zero visual change
  };
  if (window.requestIdleCallback) window.requestIdleCallback(run, { timeout: 1500 });
  else setTimeout(run, 400);
}

// ── Breakout tab HTML builder (returns HTML string, no DOM side effects) ─────
function _buildBkTabHTML(data, scoreColor) {
  // Headline = the 0-100 breakout SCORE: a blend of the opportunity-weighted
  // aggregate score and the model's hit probability (server-computed breakout_blend).
  // The fitted model can't weight opportunity (collinear with readiness + tiny
  // samples), so ranking on probability alone buried opportunity-rich players; the
  // blend puts opportunity back in. The model's raw hit chance is shown below it.
  const prob    = data.hit_probability != null ? parseFloat(data.hit_probability) : null;
  const probPct = prob != null ? Math.round(prob * 100) : null;
  const blend   = data.breakout_blend != null ? parseFloat(data.breakout_blend)
                : (prob != null ? prob : null);           // fallback: probability
  const score   = blend != null ? Math.round(blend * 100) : null;   // 0-100 headline
  // Tier + color from the blended score.
  let tier = 'Low', probColor = '#6b7280';
  if (score != null) {
    if (score >= 60)      { tier = 'Elite';    probColor = '#10b981'; }
    else if (score >= 45) { tier = 'High';     probColor = '#3b82f6'; }
    else if (score >= 30) { tier = 'Moderate'; probColor = '#f59e0b'; }
    else                  { tier = 'Low';      probColor = '#6b7280'; }
  }
  // Drive the whole modal's accent off the headline so the PPG tile, score and bars
  // agree. Fall back to the passed color when there's no score.
  scoreColor = score != null ? probColor : (scoreColor || '#3b82f6');

  const reasons = (data.key_reasons || '').split('\n')
    .map(r => r.replace(/^[•\-]\s*/, '').trim())
    .filter(r => r.length > 0)
    // Role fit is rendered as a dedicated chip below - drop the text duplicate.
    .filter(r => !/role fit for vacated targets/i.test(r));

  const txnSummary    = data.vacated_usage_summary || '';
  const addedCompSumm = data.added_competition_summary || '';

  // ── Role / archetype fit factor (context only - does not affect the score) ──
  let cd = data.component_details;
  if (typeof cd === 'string') { try { cd = JSON.parse(cd); } catch (e) { cd = {}; } }
  const aFit = (cd && cd.opportunity_opened && cd.opportunity_opened.archetype_fit) || null;
  let roleFitItem = '';
  if (aFit && aFit.label) {
    const fitColor = aFit.label === 'high' ? '#10b981'
                   : aFit.label === 'medium' ? '#f59e0b' : '#6b7280';
    roleFitItem = `
      <div title="How well this player's receiving role matches the vacated targets. Context only - it does not change the score."
           style="font-size:13px;display:flex;gap:12px;align-items:flex-start;padding:6px 0;border-bottom:1px solid var(--surface-2,rgba(255,255,255,0.06));margin-bottom:2px;">
        <span style="color:${fitColor};font-weight:700;flex-shrink:0;"><i class="fa-solid fa-bullseye" aria-hidden="true"></i></span>
        <span>
          <span style="font-weight:600;color:${fitColor};text-transform:capitalize;">${aFit.label} Role Fit</span>
          <span style="display:block;color:var(--text-muted);margin-top:2px;">Vacated Role: ${aFit.vacated_role}</span>
          <span style="display:block;color:var(--text-muted);">His Role: ${aFit.candidate_role}</span>
        </span>
      </div>`;
  }


  // ── PPG range computation ──────────────────────────────────────────────────
  const s1      = parseFloat(data.season1_ppr || 0);
  const prevPpg = parseFloat(data.prev_ppr_ppg || 0);
  let ppgRange = null;
  if (s1 > 0) {
    const modelPpg = s1 / 17;
    // Confidence-adjusted range: high confidence → tight band, low → wide
    const conf = Math.min(100, Math.max(30, parseFloat(data.confidence_score || 70)));
    // halfSpread: ±4% at conf=90, ±7% at conf=70, ±12% at conf=30
    const halfSpread = 0.04 + (0.12 - 0.04) * (90 - conf) / 60;
    // Upside cap: a sanity ceiling on the projection vs last season. The old 1.25x
    // capped the range BELOW the breakout definition itself (>=1.4x), so it hid the
    // very jump this board exists to show. Allow up to +85% and let a genuinely
    // projected breakout read through; still clips absurd outputs.
    const UPSIDE_CAP = 1.85;
    const highRaw  = modelPpg * (1 + halfSpread);
    const high     = (prevPpg > 0 && highRaw > prevPpg * UPSIDE_CAP) ? prevPpg * UPSIDE_CAP : highRaw;
    const isCapped = prevPpg > 0 && highRaw > prevPpg * UPSIDE_CAP;
    const rawLow   = isCapped ? high * (1 - halfSpread) : modelPpg * (1 - halfSpread * 0.8);
    const lowFloor = (prevPpg > 0 && rawLow < prevPpg) ? prevPpg : rawLow;
    const low      = Math.min(lowFloor, high);  // never let low exceed high
    const midPpg   = (low + high) / 2;
    ppgRange = {
      lowStr:  (Math.round(low  * 10) / 10).toFixed(1),
      highStr: (Math.round(high * 10) / 10).toFixed(1),
      prevStr: prevPpg > 0 ? (Math.round(prevPpg * 10) / 10).toFixed(1) : null,
      delta:   prevPpg > 0 ? Math.round((midPpg - prevPpg) * 10) / 10 : null,
    };
  }

  const hitProb = probPct != null ? probPct + '%' : null;
  // Breakout rank among this season's candidates: "TE #2 · #14 overall".
  const rank = data.breakout_rank || null;

  // ── Hero: 2-column layout - PPG Range left, Probability + Rank stacked right ─
  // Fixed 2-col avoids the orphaned-score problem on mobile (3-item grids wrap).
  let html = `<div style="display:grid;grid-template-columns:1.5fr 1fr;gap:8px;margin-bottom:6px;align-items:stretch;">`;

  if (ppgRange) {
    const deltaHtml = ppgRange.delta !== null
      ? `<span style="font-weight:700;color:${ppgRange.delta >= 0 ? '#10b981' : '#f59e0b'};">
           ${ppgRange.delta >= 0 ? '↑ +' : '↓ '}${Math.abs(ppgRange.delta).toFixed(1)}
         </span>`
      : '';
    html += `
      <div class="pm-hero-stat" style="background:${scoreColor}1a;border-color:${scoreColor}33;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;">
        <div class="pm-hero-label" style="color:${scoreColor};">Projected PPG</div>
        <div style="font-size:22px;font-weight:800;color:${scoreColor};line-height:1.1;margin:4px 0;">
          ${ppgRange.lowStr}–${ppgRange.highStr}
        </div>
        ${ppgRange.prevStr ? `
        <div style="font-size:11px;color:var(--text-muted);display:flex;align-items:center;justify-content:center;gap:5px;margin-top:2px;">
          <span>vs ${ppgRange.prevStr} last season</span>${deltaHtml}
        </div>` : ''}
      </div>`;
  } else {
    html += `
      <div class="pm-hero-stat" style="background:${scoreColor}1a;border-color:${scoreColor}33;">
        <div class="pm-hero-label" style="color:${scoreColor};">Breakout Score</div>
        <div class="pm-hero-val" style="color:${scoreColor};">${score != null ? score : '—'}</div>
      </div>`;
  }

  // Right column: Breakout Score headline (opportunity + model blend), with the
  // model's raw hit chance below it, then Rank.
  html += `<div style="display:flex;flex-direction:column;gap:8px;">
    <div class="pm-hero-stat">
      <div class="pm-hero-label">Breakout Score</div>
      <div class="pm-hero-val" style="color:${scoreColor};">${score != null ? score : '—'}</div>
      ${score != null ? `<div style="font-size:11px;font-weight:700;color:${scoreColor};text-transform:uppercase;letter-spacing:0.03em;margin-top:1px;">${tier}${hitProb ? ` · ${hitProb} hit` : ''}</div>` : ''}
    </div>`;
  if (rank && rank.overall) {
    const posLabel = rank.position && rank.pos ? `${rank.position} #${rank.pos}` : null;
    html += `
    <div class="pm-hero-stat">
      <div class="pm-hero-label">Breakout Rank</div>
      <div class="pm-hero-val" style="color:${scoreColor};">${posLabel || ('#' + rank.overall)}</div>
      <div style="font-size:11px;color:var(--text-muted);margin-top:1px;">#${rank.overall} of ${rank.overall_total} overall</div>
    </div>`;
  }
  html += `</div></div>`;

  // ── What Changed ───────────────────────────────────────────────────────────
  const hasContext = (txnSummary && txnSummary !== 'No departures') ||
                     (addedCompSumm && addedCompSumm !== 'No new competition added');
  if (hasContext) {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">What Changed</span></div>
      <div style="display:flex;flex-direction:column;gap:8px;">
    `;
    if (txnSummary && txnSummary !== 'No departures') {
      html += `
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <span style="font-size:14px;margin-top:1px;">&#8599;</span>
          <div style="font-size:13px;color:var(--text-muted);line-height:1.5;">${txnSummary}</div>
        </div>`;
    }
    if (addedCompSumm && addedCompSumm !== 'No new competition added') {
      html += `
        <div style="display:flex;gap:10px;align-items:flex-start;">
          <span style="font-size:14px;margin-top:1px;color:#ef4444;">&#8601;</span>
          <div style="font-size:13px;color:var(--text-muted);line-height:1.5;">${addedCompSumm}</div>
        </div>`;
    }
    html += `</div>`;
  }

  // ── Component breakdown ────────────────────────────────────────────────────
  const components = [
    { label: 'Opportunity',     val: data.opportunity_opened_score,  color: '#10b981' },
    { label: 'Competition',     val: data.competition_removed_score, color: '#3b82f6' },
    { label: 'Team Env.',       val: data.team_environment_score,    color: null      },
    { label: 'Readiness',       val: data.player_readiness_score,    color: '#8b5cf6' },
    { label: 'Role Trajectory', val: data.role_trajectory_score,     color: null      },
    { label: 'Confidence',      val: data.confidence_score,          color: '#6b7280' },
  ];
  // Always show the raw component breakdown (consistent across every position).
  // The old "what's driving it" contribution view reflected the fitted model's
  // coefficients — but the headline Breakout Score is now an opportunity-weighted
  // blend, and the model's opportunity coefficient is 0, so the contribution view
  // contradicted the headline (and only existed for WR/RB, not curve QB/TE). The
  // raw 0-100 component scores are consistent with the blended score.
  const contribs = null;

  html += `<div class='pm-two-column'>`;

  // ── Component breakdown (left on desktop, below on mobile) ─────────────────
  html += `<div class='pm-left-column pm-bk-comp-col'>`;
  html += `<hr class="pm-section-divider">`;
  // Plain, uniform bars — magnitude only, one accent color, no per-strength coloring.
  html += `<div class="pm-section-header"><span class="pm-section-label">Component Breakdown</span></div>`;
  html += '<div class="pm-comp-list-bo">';
  components.forEach(c => {
    const v    = parseFloat(c.val || 0);
    const fill = Math.min(100, Math.max(0, v));
    const disp = c.suffix ? v.toFixed(0) + c.suffix : v.toFixed(1);
    html += `
      <div class="pm-comp-row">
        <span class="pm-comp-label">${c.label}</span>
        <div class="pm-comp-bar-wrap"><div class="pm-comp-bar" style="width:${fill.toFixed(1)}%;background:${scoreColor};"></div></div>
        <span class="pm-comp-val" style="color:var(--text-muted);">${disp}</span>
      </div>`;
  });
  html += '</div></div>';

  // ── Key factors (right on desktop, above on mobile) ────────────────────────
  if (reasons.length || roleFitItem) {
    html += `<div class='pm-right-column pm-bk-reasons-col'>`;
    html += `<hr class="pm-section-divider">`;
    html += `<div class="pm-section-header"><span class="pm-section-label">Key Factors</span></div>`;
    html += `<div style="display:flex;flex-direction:column;gap:6px;">`;
    html += roleFitItem;
    reasons.forEach(r => {
      html += `<div style="font-size:13px;color:var(--text-muted);display:flex;gap:15px;align-items:flex-start;">
        <span style="color:${scoreColor};font-weight:700;flex-shrink:0;">•</span><span>${r}</span>
      </div>`;
    });
    html += `</div>`;
    if (data.peer_comparison) {
      html += `<div style="margin-top:10px;padding:8px 10px;background:var(--surface-2,rgba(255,255,255,0.04));border-radius:6px;border-left:2px solid ${scoreColor}44;">
        <div style="font-size:11px;color:var(--text-muted);line-height:1.5;">${data.peer_comparison}</div>
      </div>`;
    }
    html += `</div>`;
  }

  html += `</div>`;

  return html;
}

// ── Stats tab HTML builder (returns HTML string, no DOM side effects) ─────────
function _buildStatsHTML(game_logs_by_year, skipHeader, positionHint) {
  let statsHTML = '';
  if (game_logs_by_year && Object.keys(game_logs_by_year).length > 0) {
    statsHTML += `
      <div class="player-modal-section">
        ${skipHeader ? '' : '<div class="pm-section-header"><span class="pm-section-label">Game Logs</span></div>'}
    `;

    // Sort years in descending order (most recent first)
    const years = Object.keys(game_logs_by_year).sort((a, b) => b - a);

    // Show only the stat groups this player actually has, so a WR isn't padded
    // with empty passing/rushing columns (which pushed the real columns off-screen
    // - worse in the side-by-side compare modal). Computed once across all years
    // so the columns stay consistent between year sections. Falls back to the
    // position default for projection-only players with no real stats yet.
    // Each column: [statKey, header, roundFlag].
    const _PASS = [['pass_yd','Pass Yd',1],['pass_td','Pass TD',0],['pass_int','INT',0]];
    const _RUSH = [['rush_att','Rush Att',0],['rush_yd','Rush Yd',1],['rush_td','Rush TD',0]];
    const _REC  = [['rec_tgt','Tgt',0],['rec','Rec',0],['rec_yd','Rec Yd',1],['rec_td','Rec TD',0]];
    let _anyPass = false, _anyRush = false, _anyRec = false;
    years.forEach(y => (game_logs_by_year[y] || []).forEach(g => {
      const s = g.stats || {};
      if (s.pass_yd || s.pass_td || s.pass_int) _anyPass = true;
      if (s.rush_att || s.rush_yd || s.rush_td) _anyRush = true;
      if (s.rec_tgt || s.rec || s.rec_yd || s.rec_td) _anyRec = true;
    }));
    let statCols = [];
    if (_anyPass) statCols = statCols.concat(_PASS);
    if (_anyRush) statCols = statCols.concat(_RUSH);
    if (_anyRec)  statCols = statCols.concat(_REC);
    if (!statCols.length) {
      const P = (positionHint || '').toUpperCase();
      statCols = P === 'QB' ? _PASS.concat(_RUSH)
               : P === 'RB' ? _RUSH.concat(_REC)
               : _REC;
    }
    const _statTh = statCols.map(c => `<th>${c[1]}</th>`).join('');
    const _statCell = (s) => statCols.map(c => {
      const v = s[c[0]];
      const disp = (v != null && v > 0) ? (c[2] ? Math.round(v) : v) : '-';
      return `<td>${disp}</td>`;
    }).join('');

    // Matchup-difficulty chip: grades each game by how the opponent defense
    // ranks vs this position (same SoS-adjusted table as the Schedule Assistant,
    // #1 = easiest). Colors match sched_rank_color's 4-tier scale.
    const _mPosWord = ({QB:'QBs',RB:'RBs',WR:'WRs',TE:'TEs'})[(positionHint||'').toUpperCase()] || 'this position';
    const _matchupChip = (g) => {
      const rk = g.opp_rank, tot = g.opp_total;
      if (!rk || !tot) return '';
      const pct = rk / tot;
      const tier = pct <= 0.25 ? 1 : pct <= 0.50 ? 2 : pct <= 0.75 ? 3 : 4;
      const opp = (g.opponent || '').replace('@', '');
      const tip = `${opp} vs ${_mPosWord}: matchup rank #${rk} of ${tot} (#1 = easiest)`;
      return `<span class="game-log-matchup mt${tier}" title="${tip}">#${rk}</span>`;
    };
    const _oppCell = (g, dash) => {
      const code = g.opponent || dash;
      const chip = _matchupChip(g);
      return chip
        ? `<span class="opp-stack"><span class="opp-code">${code}</span>${chip}</span>`
        : code;
    };

    years.forEach((year, index) => {
      const gameLogs = game_logs_by_year[year];
      const isFirstYear = index === 0;
      const hasRealGames = gameLogs.some(g => !g.is_projection && !g.is_bye && g.fantasy_pts != null);
      const hasProjGames = gameLogs.some(g => g.is_projection);
      const isProjection = !hasRealGames && hasProjGames;   // ALL entries are projected
      const isMixed      = hasRealGames && hasProjGames;    // active season mid-way

      // Accumulate real completed games for the header (never mix in projections)
      let totalFantasyPts = 0;
      let totalPassYd = 0, totalPassTd = 0, totalPassInt = 0;
      let totalRushAtt = 0, totalRushYd = 0, totalRushTd = 0;
      let totalRecTgt = 0, totalRec = 0, totalRecYd = 0, totalRecTd = 0;
      let totalFumLost = 0;
      let gamesPlayed = 0;
      // Projected totals tracked separately for the tfoot footnote
      let projTotalPts = 0, projGames = 0;

      gameLogs.forEach(game => {
        if (game.is_bye) return;
        if (game.is_projection) {
          if (game.fantasy_pts != null) { projTotalPts += game.fantasy_pts; projGames++; }
          return;
        }
        const s = game.stats || {};
        const playedThisGame = game.stats != null && (
          s.pass_yd != null || s.rush_att != null || s.rec != null || s.rec_tgt != null
        );
        if (playedThisGame) gamesPlayed++;
        totalFantasyPts += game.fantasy_pts || 0;
        totalPassYd  += s.pass_yd  || 0;
        totalPassTd  += s.pass_td  || 0;
        totalPassInt += s.pass_int || 0;
        totalRushAtt += s.rush_att || 0;
        totalRushYd  += s.rush_yd  || 0;
        totalRushTd  += s.rush_td  || 0;
        totalRecTgt  += s.rec_tgt  || 0;
        totalRec     += s.rec      || 0;
        totalRecYd   += s.rec_yd   || 0;
        totalRecTd   += s.rec_td   || 0;
        totalFumLost += s.fum_lost || 0;
      });

      // Header summary - always based on real completed games
      const ppg = gamesPlayed > 0 ? (totalFantasyPts / gamesPlayed).toFixed(1) : '0.0';
      let summaryHTML;
      if (isProjection) {
        const projPpg = projGames > 0 ? (projTotalPts / projGames).toFixed(1) : '0.0';
        summaryHTML = `<span class="game-log-year-summary">~${projPpg} ppg &nbsp;<span style="opacity:0.65;"></span></span>`;
      } else {
        summaryHTML = `<span class="game-log-year-summary">${gamesPlayed}g &nbsp;·&nbsp; ${ppg} ppg &nbsp;·&nbsp; ${fmtPts(totalFantasyPts)} pts</span>`;
      }

      statsHTML += `
        <div class="game-log-year-section">
          <div class="game-log-year-header" onclick="toggleGameLogYear(this)">
            <div class="game-log-year-header-main">
              <span class="game-log-year-toggle ${isFirstYear ? '' : 'collapsed'}" id="toggle-${year}">▼</span>
              <span class="game-log-year-title">${year} Season</span>
              ${isProjection ? '<span class="game-log-proj-badge">Projected</span>' : ''}
            </div>
            ${summaryHTML}
          </div>
          <div class="game-log-year-content ${isFirstYear ? 'expanded' : ''}" id="year-${year}">
            <table class="game-log-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th class="${isProjection ? 'game-log-proj-th' : ''}">Pts${(isProjection || isMixed) ? ' *' : ''}</th>
                  ${_statTh}
                </tr>
              </thead>
              <tbody>
      `;

      gameLogs.forEach(game => {
        // Projection row
        if (game.is_projection) {
          const projVal = game.fantasy_pts != null ? fmtPts(game.fantasy_pts) : '–';
          let projDate = game.date || '';
          if (projDate.length === 8) {
            projDate = `${parseInt(projDate.substring(4,6))}/${parseInt(projDate.substring(6,8))}`;
          }
          statsHTML += `
            <tr class="game-log-table-row game-log-proj-row">
              <td>${projDate || `Wk ${game.week}`}</td>
              <td class="game-log-table-opp">${_oppCell(game, '–')}</td>
              <td class="game-log-table-pts game-log-proj-pts">${projVal}</td>
              ${statCols.map(() => '<td>–</td>').join('')}
            </tr>
          `;
          return;
        }

        const stats = game.stats || null;

        // Format date: 20240908 -> 9/8
        let dateStr = game.date || '';
        if (dateStr.length === 8) {
          const month = parseInt(dateStr.substring(4, 6));
          const day = parseInt(dateStr.substring(6, 8));
          dateStr = `${month}/${day}`;
        }

        // Check if player has any stats at all
        const isBye = game.is_bye === true;
        const hasAnyStats = !isBye && stats != null && (
          stats.pass_yd != null || stats.rush_att != null ||
          stats.rec != null || stats.rec_tgt != null);

        const val = (v) => v != null && v > 0 ? v : '-';
        const rowClass = isBye ? 'game-log-table-row game-log-bye' : hasAnyStats ? 'game-log-table-row' : 'game-log-table-row game-log-no-stats';
        const s = stats || {};

        const ptsCell = isBye
          ? '-'
          : hasAnyStats ? (game.fantasy_pts != null ? fmtPts(game.fantasy_pts) : '-') : '<span style="color:#9ca3af;">DNP</span>';

        statsHTML += `
          <tr class="${rowClass}">
            <td>${dateStr}</td>
            <td class="game-log-table-opp">${_oppCell(game, '-')}</td>
            <td class="game-log-table-pts">${ptsCell}</td>
            ${_statCell(s)}
          </tr>
        `;
      });

      const valTotal = (v) => v != null && v > 0 ? v : '-';

      // Tfoot: projected season shows PPG; completed shows full totals
      if (isProjection) {
        statsHTML += `
              </tbody>
              <tfoot>
                <tr class="game-log-table-total game-log-proj-row">
                  <td><strong>Total</strong></td>
                  <td><strong>${projGames}G</strong></td>
                  <td class="game-log-table-pts game-log-proj-pts"><strong>${fmtPts(projTotalPts)}</strong></td>
                  <td colspan="${statCols.length}" style="text-align:left;font-size:11px;color:var(--text-muted);padding-left:8px;">* Projected - actuals update when games are played</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
        `;
      } else {
        statsHTML += `
              </tbody>
              <tfoot>
                <tr class="game-log-table-total">
                  <td><strong>Total</strong></td>
                  <td><strong>${gamesPlayed}G</strong></td>
                  <td class="game-log-table-pts"><strong>${fmtPts(totalFantasyPts)}</strong></td>
                  ${statCols.map(c => {
                    const totMap = {pass_yd:totalPassYd,pass_td:totalPassTd,pass_int:totalPassInt,rush_att:totalRushAtt,rush_yd:totalRushYd,rush_td:totalRushTd,rec_tgt:totalRecTgt,rec:totalRec,rec_yd:totalRecYd,rec_td:totalRecTd};
                    const v = totMap[c[0]];
                    const disp = (v != null && v > 0) ? (c[2] ? Math.round(v) : v) : '-';
                    return `<td><strong>${disp}</strong></td>`;
                  }).join('')}
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
        `;
      }
    });

    statsHTML += `</div>`;
  }
  return statsHTML || '<div class="player-modal-loading" style="padding:40px 0;"><div style="color:var(--text-muted);font-size:13px;">No game log data available.</div></div>';
}

function getRoleGrade(roleScore) {
  // Calibrated for role_score v2 (absolute "% of an elite role"): only true
  // alphas/bellcows reach Elite, solid weekly starters land Good/Great, and the
  // long tail of depth players is Limited.
  if (roleScore >= 90) return 'Elite';      // true alpha / bellcow
  if (roleScore >= 75) return 'Great';      // clear feature role
  if (roleScore >= 60) return 'Good';       // solid starter
  if (roleScore >= 45) return 'Average';    // rotational starter / flex
  if (roleScore >= 30) return 'Below Avg';  // committee / depth
  return 'Limited';
}

const _advMetricsCache = new Map();
const _advRanksCache = new Map(); // session cache for player-metric-ranks responses
let _advMetricsToken = 0; // incremented on each loadAdvancedMetrics call; guards stale callbacks

// Fetch with a hard timeout so a hung request (slow cold server, dropped
// connection that never errors) can't leave the Advanced Metrics tab spinning
// forever — it aborts and rejects, which the caller turns into a Retry.
// @public-js:include-start  (shared fetch helper used by core/public-page code)
function _advFetch(url, ms, init) {
  const ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  const t = ctl ? setTimeout(function() { ctl.abort(); }, ms || 12000) : null;
  const opts = Object.assign({}, init || {}, ctl ? { signal: ctl.signal } : {});
  return fetch(url, opts)
    .finally(function() { if (t) clearTimeout(t); });
}
// @public-js:include-end

// ── Advanced-metrics config cache ────────────────────────────────────────────
// Fetches LEADERBOARD_METRICS in frontend format once; cached for the session.
let _advMetricsCfg = null;
function _ensureAdvMetricsCfg() {
  if (_advMetricsCfg) return Promise.resolve(_advMetricsCfg);
  return _advFetch('/api/advanced-metrics/config', 12000)
    .then(function(r) { return r.ok ? r.json() : {}; })
    .then(function(d) { _advMetricsCfg = d.metrics || {}; return _advMetricsCfg; })
    .catch(function() { return {}; });
}

// ── Shared week-bar range selector (Custom Slider 3 style) ──────────────────
// Dark track bar with labeled week ticks, a bordered selection window, and
// grip handles on each side. Used in the player modal, compare modal, and
// the advanced-metrics leaderboard.

// Build HTML for the bar. min/max are integers; ws/we are the current selection
// (both null means full range selected). Returns an empty string if max < min.
function _wkBarBuild(id, min, max, ws, we) {
  if (max < min) return '';
  const n = max - min + 1;
  const loW = (ws != null) ? Math.max(min, Math.min(max, ws)) : min;
  const hiW = (we != null) ? Math.max(min, Math.min(max, we)) : max;
  let ticks = '';
  for (let w = min; w <= max; w++) {
    ticks += '<span class="wk-tick' + (w >= loW && w <= hiW ? ' wk-tick-in' : '') + '">W' + w + '</span>';
  }
  const pctL = ((loW - min) / n * 100).toFixed(2);
  const pctR = ((max - hiW) / n * 100).toFixed(2);
  // With many weeks the per-tick cells get too narrow for "W18"-width labels on
  // phones; flag dense bars so CSS can thin the labels (every other, endpoints
  // kept) on small screens. All weeks stay draggable regardless.
  const dense = n > 10 ? ' wk-bar-dense' : '';
  // Ticks live BELOW the track (not inside it) so grip handles can reach the
  // edge without overlapping any week labels.
  return '<div class="wk-bar' + dense + '" id="' + id + '" data-min="' + min + '" data-max="' + max
    + '" data-ws="' + loW + '" data-we="' + hiW + '">'
    + '<div class="wk-bar-track">'
    + '<div class="wk-bar-bg"></div>'
    + '<div class="wk-bar-sel" style="left:' + pctL + '%;right:' + pctR + '%">'
    + '<div class="wk-bar-grip wk-bar-grip-l" role="slider" aria-label="Start week"'
    + ' aria-valuemin="' + min + '" aria-valuemax="' + max + '" aria-valuenow="' + loW + '" tabindex="0"><span></span><span></span></div>'
    + '<div class="wk-bar-grip wk-bar-grip-r" role="slider" aria-label="End week"'
    + ' aria-valuemin="' + min + '" aria-valuemax="' + max + '" aria-valuenow="' + hiW + '" tabindex="0"><span></span><span></span></div>'
    + '</div>'
    + '</div>'
    + '<div class="wk-bar-ticks">' + ticks + '</div>'
    + '</div>';
}

// Wire drag interaction on a rendered bar. onChange(ws, we) fires on release.
function _wkBarInit(id, onChange) {
  const root = document.getElementById(id);
  if (!root) return;
  const track = root.querySelector('.wk-bar-track');
  const sel   = root.querySelector('.wk-bar-sel');
  const gripL = root.querySelector('.wk-bar-grip-l');
  const gripR = root.querySelector('.wk-bar-grip-r');
  const tickEls = Array.from(root.querySelectorAll('.wk-tick'));
  if (!track || !sel || !gripL || !gripR) return;

  const min = Number(root.dataset.min);
  const max = Number(root.dataset.max);
  const n   = max - min + 1;
  let ws = Number(root.dataset.ws);
  let we = Number(root.dataset.we);

  function weekFromX(clientX) {
    const rect = track.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(1 - 1e-9, (clientX - rect.left) / rect.width));
    return min + Math.floor(pct * n);
  }
  function paint() {
    sel.style.left  = ((ws - min) / n * 100).toFixed(2) + '%';
    sel.style.right = ((max - we) / n * 100).toFixed(2) + '%';
    tickEls.forEach((t, i) => {
      const w = min + i;
      t.classList.toggle('wk-tick-in', w >= ws && w <= we);
    });
    gripL.setAttribute('aria-valuenow', String(ws));
    gripR.setAttribute('aria-valuenow', String(we));
  }

  function startDrag(e, mode) {
    e.preventDefault();
    const startX     = e.touches ? e.touches[0].clientX : e.clientX;
    const startWs    = ws, startWe = we;
    const startWkFrom = weekFromX(startX);

    function onMove(ev) {
      if (ev.cancelable) ev.preventDefault();
      const cx = ev.touches ? ev.touches[0].clientX : ev.clientX;
      const w  = weekFromX(cx);
      if (mode === 'lo') {
        ws = Math.max(min, Math.min(we, w));
      } else if (mode === 'hi') {
        we = Math.max(ws, Math.min(max, w));
      } else {
        const delta = w - startWkFrom;
        const span  = startWe - startWs;
        ws = Math.max(min, Math.min(max - span, startWs + delta));
        we = ws + span;
      }
      paint();
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend',  onUp);
      onChange(ws, we);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend',  onUp);
  }

  gripL.addEventListener('mousedown', e => { e.stopPropagation(); startDrag(e, 'lo'); });
  gripR.addEventListener('mousedown', e => { e.stopPropagation(); startDrag(e, 'hi'); });
  gripL.addEventListener('touchstart', e => { e.stopPropagation(); startDrag(e, 'lo'); }, { passive: false });
  gripR.addEventListener('touchstart', e => { e.stopPropagation(); startDrag(e, 'hi'); }, { passive: false });

  // Keyboard support: arrow keys adjust the focused handle one week at a time.
  function _onGripKey(e, mode) {
    const dec = e.key === 'ArrowLeft' || e.key === 'ArrowDown';
    const inc = e.key === 'ArrowRight' || e.key === 'ArrowUp';
    if (!dec && !inc) return;
    e.preventDefault();
    if (mode === 'lo') { ws = Math.max(min, Math.min(we, ws + (inc ? 1 : -1))); }
    else               { we = Math.max(ws, Math.min(max, we + (inc ? 1 : -1))); }
    paint();
    onChange(ws, we);
  }
  gripL.addEventListener('keydown', e => _onGripKey(e, 'lo'));
  gripR.addEventListener('keydown', e => _onGripKey(e, 'hi'));

  sel.addEventListener('mousedown', e => {
    if (!e.target.closest('.wk-bar-grip')) startDrag(e, 'move');
  });
  sel.addEventListener('touchstart', e => {
    if (!e.target.closest('.wk-bar-grip')) startDrag(e, 'move');
  }, { passive: false });

  track.addEventListener('click', e => {
    if (e.target.closest('.wk-bar-sel')) return;
    const w = weekFromX(e.clientX);
    ws = w; we = w;
    paint();
    onChange(ws, we);
  });
}

function _advParseSeasons(raw) {
  return String(raw || '').split(',').map(function(s) { return s.trim(); })
    .filter(function(s) { return /^\d{4}$/.test(s); })
    .map(Number);
}

window.advPickSeason = function(playerId, leagueId, yr) {
  const careerBtn = document.querySelector('.adv-season-pill[data-year="career"]');
  const yearPills = document.querySelectorAll('.adv-season-pill[data-year]:not([data-year="career"])');
  const careerOn = !!(careerBtn && careerBtn.classList.contains('active'));
  const selected = [];
  yearPills.forEach(function(p) {
    if (p.classList.contains('active')) selected.push(Number(p.dataset.year));
  });
  if (yr === 'career') {
    loadAdvancedMetrics(playerId, leagueId, 'career');
    return;
  }
  const year = Number(yr);
  if (careerOn || !selected.length) {
    loadAdvancedMetrics(playerId, leagueId, year);
    return;
  }
  if (selected.indexOf(year) >= 0) {
    const next = selected.filter(function(s) { return s !== year; });
    if (!next.length) return;
    next.sort(function(a, b) { return b - a; });
    loadAdvancedMetrics(playerId, leagueId, next.length === 1 ? next[0] : next.join(','));
    return;
  }
  const next = selected.concat([year]);
  next.sort(function(a, b) { return b - a; });
  loadAdvancedMetrics(playerId, leagueId, next.join(','));
};

function loadAdvancedMetrics(playerId, leagueId, season, weekStart, weekEnd) {
  const token = ++_advMetricsToken;
  const contentEl = document.getElementById('advancedMetricsContent');
  if (!contentEl) return;

  const isAuto = season === 'auto';
  const selectedYears = _advParseSeasons(season);
  const isMultiSeason = selectedYears.length > 1;
  const hasExplicitSeason = season != null && season !== 'career' && season !== 'auto';
  const realSeason = hasExplicitSeason && !isMultiSeason;
  const hasWeekRange = realSeason && weekStart != null && weekEnd != null;

  const leagueParam = leagueId ? `&league_id=${encodeURIComponent(leagueId)}` : '';
  const seasonParam = hasExplicitSeason ? `&season=${encodeURIComponent(season)}` : '';
  const weekParam = hasWeekRange ? `&week_start=${weekStart}&week_end=${weekEnd}` : '';
  const url = `/api/player-advanced-metrics/${encodeURIComponent(playerId)}?_=1${leagueParam}${seasonParam}${weekParam}`;

  // When season is explicitly known and no week range, pre-fetch ranks in parallel
  // with the metrics request so we can render once with both instead of two renders.
  let _earlyRanksPromise = null;
  if (realSeason && !hasWeekRange) {
    let _rUrl = `/api/player-metric-ranks/${encodeURIComponent(playerId)}?season=${season}`;
    if (leagueId) _rUrl += `&league_id=${encodeURIComponent(leagueId)}`;
    const _rCached = _advRanksCache.get(_rUrl);
    _earlyRanksPromise = _rCached
      ? Promise.resolve(_rCached)
      : _advFetch(_rUrl, 12000).then(r => r.ok ? r.json() : null).catch(() => null).then(d => {
          if (d && d.ranks) {
            _advRanksCache.set(_rUrl, d);
            if (_advRanksCache.size > 20) _advRanksCache.delete(_advRanksCache.keys().next().value);
          }
          return d;
        });
  }

  const _cached = _advMetricsCache.get(url);
  if (!_cached) {
    contentEl.innerHTML = `
      <div style="padding:12px 0;display:flex;align-items:center;gap:10px;">
        <div class="loading-spinner" style="width:16px;height:16px;"></div>
        <span style="font-size:13px;color:var(--text-muted);">Loading...</span>
      </div>
    `;
  }

  (_cached ? Promise.resolve(_cached) : _advFetch(url, 12000)
    .then(res => {
      // 404 = no stored metrics for this player. Surface that as an empty
      // payload instead of throwing — the old `!res.ok` throw made a missing
      // row look like a network failure ("Retry").
      if (res.status === 404) {
        return res.json().catch(function() {
          return { error: 'No metrics available for this player' };
        });
      }
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(data => {
      if (!data.error && !data.premium_required) {
        _advMetricsCache.set(url, data);
        if (_advMetricsCache.size > 8) _advMetricsCache.delete(_advMetricsCache.keys().next().value);
      }
      return data;
    }))
    .then(metricsData => {
      if (token !== _advMetricsToken) return; // superseded by a newer call
      if (metricsData.error || metricsData.premium_required) {
        contentEl.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;">'
          + '<div style="color:var(--text-muted);font-size:13px;">Advanced metrics not available for this player.</div></div>';
        return;
      }

      const availableSeasons = metricsData.available_seasons || [];

      // Auto mode: redirect to most recent season without rendering career view
      if (isAuto && availableSeasons.length > 0) {
        loadAdvancedMetrics(playerId, leagueId, availableSeasons[0]);
        return;
      }

      const activeSeason = metricsData.season;
      const selectedFromResp = (metricsData.selected_seasons || []).map(Number).filter(Boolean);
      const combinedYears = isMultiSeason
        ? (selectedFromResp.length ? selectedFromResp : selectedYears)
        : [];
      const isCareer = season === 'career' || (activeSeason == null && !isMultiSeason);

      // Update year label in section header
      const seasonLabelEl = document.getElementById('advMetricsSeasonLabel');
      if (seasonLabelEl) {
        let _wk = '';
        if (metricsData.week_start != null) {
          _wk = (metricsData.week_start === metricsData.week_end)
            ? ` · W${metricsData.week_start}`
            : ` · W${metricsData.week_start}–W${metricsData.week_end}`;
        }
        if (isMultiSeason) {
          seasonLabelEl.textContent = combinedYears.join(' + ');
        } else {
          seasonLabelEl.textContent = isCareer ? 'Career' : ((activeSeason || '') + _wk);
        }
      }

      const activeWS = metricsData.week_start != null ? Number(metricsData.week_start) : null;
      const activeWE = metricsData.week_end != null ? Number(metricsData.week_end) : null;
      const weekActive = activeWS != null;
      const availableWeeks = (metricsData.available_weeks || []).map(Number);

      // Season pills above the layout - always show when there's at least 1 season
      const pillsEl = document.getElementById('advMetricsPills');
      if (pillsEl && availableSeasons.length >= 1) {
        const lidExpr = leagueId ? `'${leagueId}'` : 'null';
        const activeYears = isMultiSeason
          ? combinedYears
          : ((!isCareer && activeSeason) ? [Number(activeSeason)] : []);
        const lidPick = leagueId ? `'${leagueId}'` : 'null';
        let pillsHTML = '<div class="adv-metrics-season-pills">';
        pillsHTML += `<button type="button" class="adv-season-pill${isCareer ? ' active' : ''}" data-year="career" onclick="advPickSeason('${playerId}', ${lidPick}, 'career')">Career</button>`;
        availableSeasons.forEach(yr => {
          const activeClass = (!isCareer && activeYears.indexOf(Number(yr)) >= 0) ? ' active' : '';
          pillsHTML += `<button type="button" class="adv-season-pill${activeClass}" data-year="${yr}" onclick="advPickSeason('${playerId}', ${lidPick}, ${yr})">${yr}</button>`;
        });
        pillsHTML += '</div>';
        if (availableSeasons.length >= 2) {
          pillsHTML += '<div class="adv-season-hint">Tap more years to combine · only seasons with data are listed</div>';
        }
        // Week-bar: only show when the player has per-week data for this season.
        if (!isCareer && !isMultiSeason && activeSeason && availableWeeks.length > 0) {
          const wkMin = Math.min(...availableWeeks);
          const wkMax = Math.max(...availableWeeks);
          const barWS = activeWS != null ? activeWS : (weekStart != null ? weekStart : null);
          const barWE = activeWE != null ? activeWE : (weekEnd != null ? weekEnd : null);
          const isFullRange = (barWS == null);
          const lidExpr2 = leagueId ? ("'" + String(leagueId) + "'") : 'null';
          pillsHTML += '<div class="adv-week-bar-row">'
            + '<button class="adv-week-full-btn' + (isFullRange ? ' active' : '') + '" onclick="loadAdvancedMetrics(\'' + playerId + '\',' + lidExpr2 + ',' + activeSeason + ')">Season</button>'
            + _wkBarBuild('advWkBar', wkMin, wkMax, barWS, barWE)
            + '</div>';
        }
        pillsEl.innerHTML = pillsHTML;
        if (!isCareer && activeSeason && availableWeeks.length > 0) {
          const _wkPid = playerId, _wkLid = leagueId, _wkSeas = activeSeason;
          _wkBarInit('advWkBar', function(ws, we) {
            loadAdvancedMetrics(_wkPid, _wkLid, _wkSeas, ws, we);
          });
        }
      }

      // Populate bars — fetch metric config and ranks, then render once with both.
      // When season was known upfront (_earlyRanksPromise), the ranks fetch ran in
      // parallel with the metrics fetch so both are likely already resolved here.
      // For week-range views or auto-season, we fall back to a sequential fetch.
      _ensureAdvMetricsCfg().then(function(cfg) {
        if (token !== _advMetricsToken) return;

        if (!isCareer && activeSeason) {
          // Determine the ranks promise to use.
          let _ranksPromise;
          if (_earlyRanksPromise && !weekActive) {
            // Already in-flight or resolved from the parallel pre-fetch above.
            _ranksPromise = _earlyRanksPromise;
          } else {
            // Week-range or auto-season: fetch ranks now (season resolved from response).
            let rankUrl = `/api/player-metric-ranks/${encodeURIComponent(playerId)}?season=${activeSeason}`;
            if (leagueId) rankUrl += `&league_id=${encodeURIComponent(leagueId)}`;
            if (weekActive) rankUrl += `&week_start=${activeWS}&week_end=${activeWE}`;
            const _rCached2 = _advRanksCache.get(rankUrl);
            _ranksPromise = _rCached2
              ? Promise.resolve(_rCached2)
              : _advFetch(rankUrl, 12000).then(r => r.ok ? r.json() : null).catch(() => null).then(d => {
                  if (d && d.ranks) {
                    _advRanksCache.set(rankUrl, d);
                    if (_advRanksCache.size > 20) _advRanksCache.delete(_advRanksCache.keys().next().value);
                  }
                  return d;
                });
          }

          _ranksPromise.then(function(ranksData) {
            if (token !== _advMetricsToken) return;
            const ranks = (ranksData && ranksData.ranks && Object.keys(ranksData.ranks).length)
              ? ranksData.ranks : null;
            const counts = (ranksData && ranksData.counts) ? ranksData.counts : null;
            const bounds = (ranksData && ranksData.bounds) ? ranksData.bounds : null;
            contentEl.innerHTML = buildAdvancedMetricsHTML(metricsData, ranks, cfg, weekActive, counts, bounds);
          }).catch(function() {
            contentEl.innerHTML = buildAdvancedMetricsHTML(metricsData, null, cfg, weekActive, null, null);
          });
        } else {
          contentEl.innerHTML = buildAdvancedMetricsHTML(metricsData, null, cfg, weekActive, null, null);
        }
      });

      // Update weekly-trends panel with current season and week range.
      const wtWrap = document.getElementById('pmWeeklyTrendsWrap');
      if (wtWrap) {
        // Store active week range so pmWtRender can filter accordingly.
        wtWrap.dataset.wkStart = activeWS != null ? String(activeWS) : '';
        wtWrap.dataset.wkEnd   = activeWE != null ? String(activeWE) : '';

        const s = (!isCareer && activeSeason) ? String(activeSeason) : '';
        if (wtWrap.dataset.season !== s) {
          // Season changed — reset panel so it refetches.
          wtWrap.dataset.season = s;
          wtWrap.dataset.loaded = '';
          const wtBody = document.getElementById('pmWeeklyTrendsBody');
          if (wtBody) { wtBody.style.display = 'none'; wtBody.innerHTML = ''; }
          const wtBtn = document.getElementById('pmWeeklyTrendsBtn');
          if (wtBtn) wtBtn.innerHTML = 'Trends &#9662;';
        } else if (wtWrap.dataset.loaded) {
          // Panel already loaded — re-render with updated week range filter.
          const wtBody = document.getElementById('pmWeeklyTrendsBody');
          if (wtBody && wtBody.style.display !== 'none') {
            pmWtRender(wtWrap, wtWrap.dataset.position || '');
          }
        }
        // Auto-open on first load (body still hidden after DOM insertion).
        const wtBodyEl = document.getElementById('pmWeeklyTrendsBody');
        if (wtBodyEl && wtBodyEl.style.display === 'none') {
          pmToggleWeeklyTrends(playerId);
        }
      }

      // Tell the Trends panel whether a season-over-season view is available
      // (2+ seasons of data) so it can offer the Weekly/Season toggle.
      const trendsWrap = document.getElementById('pmWeeklyTrendsWrap');
      if (trendsWrap) {
        trendsWrap.dataset.multiseason = (availableSeasons.length >= 2) ? '1' : '';
        trendsWrap.dataset.pid = playerId;
      }
    })
    .catch(err => {
      console.error('Error loading advanced metrics:', err);
      if (token !== _advMetricsToken) return;  // superseded by a newer open
      // A transient network error (e.g. ERR_NETWORK_CHANGED on a Wi-Fi switch)
      // used to hide the whole section, so it appeared to "never load" with no
      // way to recover. Keep it visible and offer a one-tap retry instead.
      contentEl.innerHTML = '<div style="padding:14px 0;font-size:13px;color:var(--text-muted);">'
        + 'Couldn’t load advanced metrics, network hiccup. '
        + '<button type="button" class="adv-retry-btn" style="margin-left:6px;padding:4px 12px;'
        + 'border:1px solid var(--border,#334155);border-radius:6px;background:transparent;'
        + 'color:var(--accent,#3b82f6);cursor:pointer;font-weight:700;">Retry</button>'
        + '</div>';
      const _btn = contentEl.querySelector('.adv-retry-btn');
      if (_btn) _btn.addEventListener('click', function() {
        loadAdvancedMetrics(playerId, leagueId, season, weekStart, weekEnd);
      });
    });
}

// ── Player modal: season-over-season metric trend ────────────────────────────
var _pmSeasonTrendCache = {};   // key: playerId + '|' + metric  → API payload

// Switch the Trends panel between the Weekly sparklines and the Season line
// charts. Season loads lazily on first switch.
function pmTrendsSetMode(playerId, mode) {
  var weeklyEl = document.getElementById('pmWtWeekly');
  var seasonEl = document.getElementById('pmWtSeason');
  if (!weeklyEl || !seasonEl) return;
  if (mode === 'season') {
    weeklyEl.style.display = 'none';
    seasonEl.style.display = '';
    if (!seasonEl.dataset.loaded) pmLoadSeasonAll(playerId, seasonEl);
  } else {
    seasonEl.style.display = 'none';
    weeklyEl.style.display = '';
  }
}

// Season mode: render a line chart for EVERY available metric at once (no
// picker), mirroring how weekly trends shows all stats.
function pmLoadSeasonAll(playerId, host) {
  host.innerHTML = '<div style="padding:10px 0;font-size:12px;color:var(--text-muted);">Loading season trends…</div>';
  var fail = function() { host.innerHTML = '<div style="padding:10px 0;font-size:12px;color:var(--text-muted);">Couldn’t load season trends.</div>'; };
  // First call (no metrics) returns the full list of available metrics; then
  // request them all in one shot.
  pmSeasonTrendFetch(playerId, null).then(function(meta) {
    if (!meta || !meta.options || !meta.options.length) {
      host.innerHTML = '<div style="padding:10px 0;font-size:12px;color:var(--text-muted);">No multi-season data for this player.</div>';
      return;
    }
    host.dataset.loaded = '1';
    var keys = meta.options.map(function(o) { return o.key; });
    pmSeasonTrendFetch(playerId, keys.join(',')).then(function(data) {
      if (!data || !data.series) { fail(); return; }
      host.innerHTML = buildSeasonTrendRows(meta.options, data.series, data.position);
    }).catch(fail);
  }).catch(fail);
}

// Season trends in the SAME slim two-column row format as the weekly trends:
// label + season sparkline + latest value / first→latest delta / current rank.
function buildSeasonTrendRows(options, series, position) {
  var pos = (position || '').toUpperCase();
  var accent = ((getComputedStyle(document.documentElement).getPropertyValue('--accent') || '').trim()) || '#3b82f6';
  var rows = '', minYr = null, maxYr = null;
  (options || []).forEach(function(o) {
    var pts = (series || {})[o.key];
    if (!pts) return;
    var withVal = pts.filter(function(p) { return p.value != null; });
    if (withVal.length < 2) return;   // need 2+ seasons for a trend
    var vals = withVal.map(function(p) { return p.value; });
    var first = withVal[0], last = withVal[withVal.length - 1];
    if (minYr == null || first.season < minYr) minYr = first.season;
    if (maxYr == null || last.season > maxYr) maxYr = last.season;
    var improved = (last.value !== first.value)
      ? (o.lower_better ? last.value < first.value : last.value > first.value) : null;
    var color = improved === true ? '#22c55e' : (improved === false ? '#ef4444' : accent);
    var deltaTxt = pmStFmt(Math.abs(last.value - first.value), o);
    var deltaHtml = '';
    if (improved === true) deltaHtml = '<span class="pm-wt-delta" style="color:#10b981">&#9650; +' + deltaTxt + '</span>';
    else if (improved === false) deltaHtml = '<span class="pm-wt-delta" style="color:#ef4444">&#9660; -' + deltaTxt + '</span>';
    var tips = withVal.map(function(p) {
      return p.season + ' · ' + pmStFmt(p.value, o) + (p.rank != null ? ' (' + pos + p.rank + ')' : '');
    });
    var rankTxt = (last.rank != null) ? (pos + last.rank) : '';
    rows += '<div class="pm-wt-row">'
      + '<div class="pm-wt-label">' + o.label + '</div>'
      + pmSparkline(vals, color, tips)
      + '<div class="pm-wt-stats">'
      + '<div class="pm-wt-stats-top">'
      + '<span class="pm-wt-last">' + pmStFmt(last.value, o) + '</span>'
      + deltaHtml
      + '</div>'
      + (rankTxt ? '<span class="pm-wt-avg">' + rankTxt + '</span>' : '')
      + '</div></div>';
  });
  if (!rows) return '<div style="padding:10px 0;color:var(--text-muted);font-size:12px;">Not enough multi-season data.</div>';
  return '<div class="pm-wt-grid pm-st-rows">' + rows + '</div>'
    + '<div class="pm-wt-footer">' + (minYr && maxYr ? (minYr + '&ndash;' + maxYr + ' &middot; ') : '')
    + '&#9650;&#9660; = first&rarr;latest season</div>';
}

function pmSeasonTrendFetch(playerId, metric) {
  var ck = playerId + '|' + (metric || '');
  if (_pmSeasonTrendCache[ck]) return Promise.resolve(_pmSeasonTrendCache[ck]);
  var url = '/api/player-advanced-metrics-trend/' + encodeURIComponent(playerId);
  if (metric) url += '?metrics=' + encodeURIComponent(metric);
  return fetch(url).then(function(r) { return r.ok ? r.json() : null; }).then(function(d) {
    if (d) _pmSeasonTrendCache[ck] = d;
    return d;
  });
}

function pmStFmt(v, opt) {
  if (v == null) return '—';
  opt = opt || {};
  if (opt.pct) { var p = opt.pct_frac ? v * 100 : v; return (Math.round(p * 10) / 10) + '%'; }
  if (opt.integer) return Math.round(v).toLocaleString();
  return String(Math.round(v * 100) / 100);
}

// Line chart of one metric across seasons. Value drives the line; each point is
// annotated with the season and that season's positional rank, and the line is
// colored by whether the metric improved (respecting lower-is-better metrics).
function pmSeasonTrendChartHTML(points, opt, position) {
  opt = opt || {};
  var pos = (position || '').toUpperCase();
  var withVal = points.filter(function(p) { return p.value != null; });
  if (!withVal.length) return '<div style="padding:10px 0;font-size:12px;color:var(--text-muted);">No data for this metric.</div>';

  var vals = withVal.map(function(p) { return p.value; });
  var vmin = Math.min.apply(null, vals), vmax = Math.max.apply(null, vals);
  var span = (vmax - vmin) || Math.abs(vmax) || 1;
  var padV = span * 0.15;
  var lo = vmin - padV, hi = vmax + padV;

  var W = 320, H = 152, padL = 22, padR = 22, padTop = 24, padBot = 40;
  var innerW = W - padL - padR, innerH = H - padTop - padBot;
  var n = points.length;
  function xAt(i) { return n === 1 ? padL + innerW / 2 : padL + (i / (n - 1)) * innerW; }
  function yAt(v) { return padTop + innerH - ((v - lo) / ((hi - lo) || 1)) * innerH; }
  // Anchor edge labels inward so the first/last season + rank text can't clip
  // against the SVG bounds (which was chopping "RB33/95" → "B33/95", etc.).
  function anchorAt(i) { return (n > 1 && i === 0) ? 'start' : (n > 1 && i === n - 1) ? 'end' : 'middle'; }

  var first = withVal[0], last = withVal[withVal.length - 1];
  var lastIdx = -1, maxIdx = -1;
  points.forEach(function(p, i) {
    if (p.value == null) return;
    lastIdx = i;                                                   // most-recent point with data
    if (maxIdx < 0 || p.value > points[maxIdx].value) maxIdx = i;  // season high
  });
  var improved = null;
  if (withVal.length >= 2 && last.value !== first.value) {
    improved = opt.lower_better ? (last.value < first.value) : (last.value > first.value);
  }
  var accent = ((getComputedStyle(document.documentElement).getPropertyValue('--accent') || '').trim()) || '#3b82f6';
  var col = improved === true ? '#22c55e' : (improved === false ? '#ef4444' : accent);

  var linePts = [];
  points.forEach(function(p, i) { if (p.value != null) linePts.push(xAt(i).toFixed(1) + ',' + yAt(p.value).toFixed(1)); });

  var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="pm-st-svg" preserveAspectRatio="xMidYMid meet">';
  if (linePts.length >= 2) {
    var fx = linePts[0].split(',')[0], lx = linePts[linePts.length - 1].split(',')[0];
    var baseY = (padTop + innerH).toFixed(1);
    svg += '<path d="M' + fx + ',' + baseY + ' L' + linePts.join(' L') + ' L' + lx + ',' + baseY + ' Z" fill="' + col + '" opacity="0.10"/>';
    svg += '<polyline fill="none" stroke="' + col + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="' + linePts.join(' ') + '"/>';
  }
  points.forEach(function(p, i) {
    var x = xAt(i);
    var anchor = anchorAt(i);
    svg += '<text x="' + x.toFixed(1) + '" y="' + (H - 24) + '" text-anchor="' + anchor + '" class="pm-st-x">' + p.season + '</text>';
    var rankTxt = (p.rank != null) ? (pos + p.rank + (p.count ? '/' + p.count : '')) : '—';
    svg += '<text x="' + x.toFixed(1) + '" y="' + (H - 10) + '" text-anchor="' + anchor + '" class="pm-st-rank">' + rankTxt + '</text>';
    if (p.value == null) return;
    var y = yAt(p.value);
    var isLast = (i === lastIdx);
    var isMax  = (i === maxIdx);
    // Emphasize the most-recent point AND the season high (haloed, larger dot).
    if (isLast || isMax) {
      svg += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="5.5" fill="' + col + '" opacity="0.20"/>';
      svg += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.8" fill="' + col + '"/>';
    } else {
      svg += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="3.0" fill="' + col + '"/>';
    }
    // Season high: mark the value with a ▲ in the line color so it reads even
    // when it coincides with the most-recent point.
    var valTxt = pmStFmt(p.value, opt) + (isMax ? ' ▲' : '');
    var valStyle = isMax ? ' style="fill:' + col + ';font-weight:900;"' : '';
    svg += '<text x="' + x.toFixed(1) + '" y="' + (y - 9).toFixed(1) + '" text-anchor="' + anchor + '" class="pm-st-val' + (isLast ? ' pm-st-val-last' : '') + '"' + valStyle + '>' + valTxt + '</text>';
  });
  svg += '</svg>';

  var cap;
  if (withVal.length >= 2) {
    var arrow = improved === true ? '▲' : (improved === false ? '▼' : '▬');
    var word  = improved === true ? 'improving' : (improved === false ? 'declining' : 'flat');
    cap = '<div class="pm-st-caption"><b>' + pmStFmt(first.value, opt) + '</b> → <b>' + pmStFmt(last.value, opt) + '</b> '
        + '<span style="color:' + col + ';font-weight:700;">' + arrow + ' ' + word + '</span>';
    if (first.rank != null && last.rank != null) cap += ' · rank ' + pos + first.rank + ' → ' + pos + last.rank;
    cap += '</div>';
  } else {
    cap = '<div class="pm-st-caption">Only one season of data.</div>';
  }
  if (opt.lower_better) cap += '<div class="pm-st-note">Lower is better — the line is green when it drops.</div>';
  return cap + svg;
}

// ── Player modal: weekly usage trends ────────────────────────────────────────
var _pmSparkId = 0;
function pmSparkline(series, color, tips) {
  if (!series || series.length < 2) return '<div class="pm-wt-spark"></div>';
  var W = 240, H = 40, padX = 8, padTop = 8, padBot = 8;
  var iw = W - padX * 2, ih = H - padTop - padBot;
  var lo = Math.min.apply(null, series), hi = Math.max.apply(null, series);
  var span = (hi - lo) || Math.abs(hi) || 1, pad = span * 0.18;
  lo -= pad; hi += pad;
  var n = series.length;
  var x = function(i) { return padX + (i / (n - 1)) * iw; };
  var y = function(v) { return padTop + ih - ((v - lo) / ((hi - lo) || 1)) * ih; };
  var id = 'pmspk' + (++_pmSparkId);
  var pts = series.map(function(v, i) { return x(i).toFixed(1) + ',' + y(v).toFixed(1); });
  var baseY = (padTop + ih).toFixed(1);
  var avg = series.reduce(function(s, v) { return s + v; }, 0) / n;
  var s = '<div class="pm-wt-spark"><svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" style="width:100%;height:auto;display:block;">';
  s += '<defs><linearGradient id="' + id + '" x1="0" x2="0" y1="0" y2="1">'
    + '<stop offset="0" stop-color="' + color + '" stop-opacity="0.24"/>'
    + '<stop offset="1" stop-color="' + color + '" stop-opacity="0.02"/></linearGradient></defs>';
  s += '<path d="M' + pts[0].split(',')[0] + ',' + baseY + ' L' + pts.join(' L') + ' L'
    + pts[pts.length - 1].split(',')[0] + ',' + baseY + ' Z" fill="url(#' + id + ')"/>';
  s += '<line x1="' + padX + '" x2="' + (W - padX) + '" y1="' + y(avg).toFixed(1) + '" y2="' + y(avg).toFixed(1)
    + '" stroke="' + color + '" stroke-width="1" stroke-dasharray="3 3" opacity="0.35"/>';
  s += '<polyline fill="none" stroke="' + color + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="' + pts.join(' ') + '"/>';
  // solid dot on every point (emphasize the latest); + invisible hover target
  series.forEach(function(v, i) {
    var last = i === n - 1, cx = x(i).toFixed(1), cy = y(v).toFixed(1);
    if (last) s += '<circle cx="' + cx + '" cy="' + cy + '" r="4.5" fill="' + color + '" opacity="0.20"/>';
    s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + (last ? 3 : 2.2) + '" fill="' + color + '" stroke="var(--card,#fff)" stroke-width="1"/>';
    if (tips && tips[i]) {
      s += '<circle class="wk-dot" cx="' + cx + '" cy="' + cy + '" r="9" fill="transparent" '
        + 'style="pointer-events:all;cursor:pointer" data-tip="' + String(tips[i]).replace(/"/g, '&quot;') + '"/>';
    }
  });
  s += '</svg></div>';
  return s;
}

// Weekly datapoint hover tooltip: shows "value + opponent" for the hovered
// point (delegated once, works for every .wk-dot the sparklines render).
(function () {
  if (typeof document === 'undefined') return;
  var tip = null;
  function ensureTip() {
    if (!tip) { tip = document.createElement('div'); tip.className = 'wk-tip'; document.body.appendChild(tip); }
    return tip;
  }
  function place(e) { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }
  function show(d, e) { ensureTip(); tip.textContent = d.getAttribute('data-tip') || ''; place(e); tip.classList.add('show'); }
  document.addEventListener('pointerover', function (e) {
    if (e.pointerType && e.pointerType !== 'mouse') return;   // touch handled on pointerdown
    var d = e.target && e.target.closest && e.target.closest('.wk-dot');
    if (d) show(d, e);
  });
  document.addEventListener('pointermove', function (e) {
    if (tip && tip.classList.contains('show') && (!e.pointerType || e.pointerType === 'mouse')) place(e);
  });
  document.addEventListener('pointerout', function (e) {
    if (e.pointerType && e.pointerType !== 'mouse') return;   // don't yank the tip on touch-end
    var d = e.target && e.target.closest && e.target.closest('.wk-dot');
    if (d && tip) tip.classList.remove('show');
  });
  // Touch: tap a point to show its data; it stays until you tap elsewhere (a
  // plain pointerout on touch-end would flash it away instantly).
  document.addEventListener('pointerdown', function (e) {
    var d = e.target && e.target.closest && e.target.closest('.wk-dot');
    if (d) { show(d, e); e.preventDefault(); }
    else if (tip) { tip.classList.remove('show'); }
  }, true);
})();

// Shared renderer: sparkline rows for a player's weekly usage series.
function buildWeeklyTrendRows(weeks, position) {
  if (!weeks || weeks.length < 2) {
    return '<div style="padding:10px 0;color:var(--text-muted);font-size:12px;">Not enough weekly data for this season.</div>';
  }
  var pos = (position || '').toUpperCase();
  // Short unit word for the hover tooltip ("78 yds vs NYG"). Rates and % rows
  // carry no unit word (the % suffix / row label already says it).
  function _unitFor(label) {
    if (/%/.test(label) || /\//.test(label)) return '';
    if (/Yds/.test(label)) return 'yds';
    if (/Targets/.test(label)) return 'tgt';
    if (/Touches/.test(label)) return 'tch';
    if (/Carries/.test(label)) return 'car';
    if (/Receptions/.test(label)) return 'rec';
    if (/PPR/.test(label)) return 'pts';
    return '';
  }
  function _wt_row(label, series, color, suffix) {
    if (!series.some(function(v) { return v > 0; })) return '';
    var seasonAvg = series.reduce(function(s, v) { return s + v; }, 0) / series.length;
    var r3 = series.slice(-3);
    var recentAvg = r3.reduce(function(s, v) { return s + v; }, 0) / r3.length;
    var delta = recentAvg - seasonAvg;
    var deltaHtml = '';
    if (delta >= 0.5) deltaHtml = '<span class="pm-wt-delta" style="color:#10b981">&#9650; +' + delta.toFixed(1) + '</span>';
    else if (delta <= -0.5) deltaHtml = '<span class="pm-wt-delta" style="color:#ef4444">&#9660; ' + delta.toFixed(1) + '</span>';
    var lastWk = series[series.length - 1];
    // Per-week hover tooltip: "78 yds vs NYG" (opponent from the API), falling
    // back to "Wk 12 · 78 yds" when the opponent isn't known.
    var unit = _unitFor(label);
    var tips = series.map(function(val, i) {
      var w = weeks[i] || {};
      var num = Math.round(val * 10) / 10;
      var vTxt = (suffix === '%') ? (num + '%') : (num + (unit ? ' ' + unit : ''));
      return w.opponent ? (vTxt + ' vs ' + w.opponent)
                        : ('Wk ' + (w.week != null ? w.week : (i + 1)) + ' · ' + vTxt);
    });
    return '<div class="pm-wt-row">'
      + '<div class="pm-wt-label">' + label + '</div>'
      + pmSparkline(series, color, tips)
      + '<div class="pm-wt-stats">'
      + '<div class="pm-wt-stats-top">'
      + '<span class="pm-wt-last">' + seasonAvg.toFixed(1) + (suffix || '') + '</span>'
      + deltaHtml
      + '</div>'
      + '<span class="pm-wt-avg">LG: ' + lastWk.toFixed(1) + (suffix || '') + '</span>'
      + '</div></div>';
  }
  function rowFor(label, key, color, suffix) {
    return _wt_row(label, weeks.map(function(w) { return Number(w[key] || 0); }), color, suffix);
  }
  function rowForComputed(label, fn, color, suffix) {
    return _wt_row(label, weeks.map(fn), color, suffix);
  }
  var computedRows = '';
  var volumeRows = '';
  if (pos === 'RB') {
    volumeRows += rowFor('Carries', 'carries', '#f97316');
    volumeRows += rowFor('Rush Yds', 'rush_yards', '#fb923c');
    volumeRows += rowFor('Receptions', 'receptions', '#10b981');
    volumeRows += rowFor('Rec Yds', 'rec_yards', '#34d399');
    computedRows += rowForComputed('Yds/Carry', function(w) {
      var c = Number(w.carries || 0); return c > 0 ? Number(w.rush_yards || 0) / c : 0;
    }, '#f97316');
    computedRows += rowForComputed('Yds/Touch', function(w) {
      var t = Number(w.touches || 0); return t > 0 ? (Number(w.rush_yards || 0) + Number(w.rec_yards || 0)) / t : 0;
    }, '#ec4899');
    computedRows += rowForComputed('Catch %', function(w) {
      var tgt = Number(w.targets || 0); return tgt > 0 ? Number(w.receptions || 0) / tgt * 100 : 0;
    }, '#14b8a6', '%');
  } else if (pos === 'WR' || pos === 'TE') {
    volumeRows += rowFor('Receptions', 'receptions', '#10b981');
    volumeRows += rowFor('Rec Yds', 'rec_yards', '#34d399');
    volumeRows += rowFor('Rush Yds', 'rush_yards', '#fb923c');
    computedRows += rowForComputed('Yds/Target', function(w) {
      var t = Number(w.targets || 0); return t > 0 ? Number(w.rec_yards || 0) / t : 0;
    }, '#f97316');
    computedRows += rowForComputed('Yds/Rec', function(w) {
      var r = Number(w.receptions || 0); return r > 0 ? Number(w.rec_yards || 0) / r : 0;
    }, '#fb923c');
    computedRows += rowForComputed('Catch %', function(w) {
      var tgt = Number(w.targets || 0); return tgt > 0 ? Number(w.receptions || 0) / tgt * 100 : 0;
    }, '#14b8a6', '%');
  }
  return '<div class="pm-wt-grid">'
    + rowFor('Snap %', 'snap_pct', '#3b82f6', '%')
    + rowFor('Targets', 'targets', '#f59e0b')
    + rowFor('Touches', 'touches', '#22c55e')
    + volumeRows
    + computedRows
    + rowFor('PPR Pts', 'ppr_pts', '#8b5cf6')
    + '</div>'
    + '<div class="pm-wt-footer">Wks ' + weeks[0].week + '&ndash;' + weeks[weeks.length - 1].week
    + ' &middot; &#9650;&#9660; = 3-wk trend vs avg</div>';
}

// Collapse/expand a section in the player compare view.
// ── Advanced Metrics: tap-to-show definition tooltip (mobile-friendly) ──────
function _advGetTip() {
  let tip = document.getElementById('adv-metric-def-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'adv-metric-def-tip';
    tip.className = 'adv-def-tip';
    document.body.appendChild(tip);
    // Bubble phase so inline onclick handlers can call e.stopPropagation()
    // to prevent the metric-label click from immediately dismissing the tip.
    document.addEventListener('click', function(e) {
      if (!tip.contains(e.target)) tip.style.display = 'none';
    });
  }
  return tip;
}

function _advPositionTip(tip, anchorEl, mouseX, mouseY) {
  const tw = Math.min(240, window.innerWidth - 16);
  tip.style.maxWidth = tw + 'px';
  const tipH = tip.offsetHeight || 70;
  let left, top;
  if (mouseX != null && mouseY != null) {
    // Position near mouse cursor rather than element center (avoids wide-element offset).
    left = mouseX - tw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    top = mouseY + 16;
    if (top + tipH > window.innerHeight - 8) top = mouseY - tipH - 10;
    top = Math.max(8, top);
  } else {
    const rect = anchorEl.getBoundingClientRect();
    left = rect.left + rect.width / 2 - tw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    top = rect.bottom + 6;
    if (top + tipH > window.innerHeight - 8) top = rect.top - tipH - 6;
    top = Math.max(8, top);
  }
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

function advEnterMetricDef(e) {
  if (window.matchMedia('(pointer: coarse)').matches) return;
  const el = e.currentTarget;
  const def = el.dataset.def;
  if (!def) return;
  const tip = _advGetTip();
  clearTimeout(tip._hoverTid);
  tip.textContent = def;
  tip.dataset.src = def;
  tip.style.display = 'block';
  // Focus/keyboard events report clientX/Y as 0 — pin to the element instead.
  const fromPointer = e.type === 'mouseenter' || e.type === 'mousemove' || e.type === 'pointerover';
  _advPositionTip(tip, el, fromPointer ? e.clientX : null, fromPointer ? e.clientY : null);
}

function advLeaveMetricDef(e) {
  if (window.matchMedia('(pointer: coarse)').matches) return;
  const tip = _advGetTip();
  tip._hoverTid = setTimeout(function() { tip.style.display = 'none'; }, 120);
}

function advShowMetricDef(e) {
  e.stopPropagation();
  if (!window.matchMedia('(pointer: coarse)').matches) return; // desktop uses hover
  const el = e.currentTarget;
  const def = el.dataset.def;
  if (!def) return;
  const tip = _advGetTip();
  // Toggle off if same metric tapped again
  if (tip.style.display !== 'none' && tip.dataset.src === el.dataset.def) {
    tip.style.display = 'none';
    return;
  }
  tip.textContent = def;
  tip.dataset.src = def;
  tip.style.display = 'block';
  _advPositionTip(tip, el, e.clientX, e.clientY);
}

function advShowInfoTip(e) {
  e.stopPropagation();
  const tip = _advGetTip();
  if (tip.style.display !== 'none' && tip.dataset.src === '__info__') {
    tip.style.display = 'none';
    return;
  }
  tip.textContent = 'Tap any metric name to see its definition.';
  tip.dataset.src = '__info__';
  tip.style.display = 'block';
  _advPositionTip(tip, e.currentTarget);
}

function cmpToggleSection(wrapId, headerEl) {
  const wrap = document.getElementById(wrapId);
  if (!wrap) return;
  const collapsed = wrap.style.display === 'none';
  wrap.style.display = collapsed ? '' : 'none';
  const chev = headerEl ? headerEl.querySelector('.pm-collapse-chevron') : null;
  if (chev) chev.innerHTML = collapsed ? '&#9662;' : '&#9656;';
  const hint = headerEl ? headerEl.querySelector('.pm-collapse-hint') : null;
  if (hint) hint.textContent = collapsed ? 'click to collapse' : 'click to expand';
  // Keep the hint visible while a section is collapsed so it's discoverable
  if (hint) hint.style.opacity = collapsed ? '' : '0.8';
}

function pmWtRender(wrap, position) {
  var allWeeks = wrap._weeklyData || [];
  var wkStart = wrap.dataset.wkStart ? parseInt(wrap.dataset.wkStart) : null;
  var wkEnd   = wrap.dataset.wkEnd   ? parseInt(wrap.dataset.wkEnd)   : null;
  var rangeActive = (wkStart != null && wkEnd != null);

  var weeks;
  if (rangeActive) {
    weeks = allWeeks.filter(function(w) {
      var wk = Number(w.week); return wk >= wkStart && wk <= wkEnd;
    });
  } else {
    var activeTab = document.querySelector('.pm-wt-tab.pm-wt-tab-active');
    var nStr = activeTab ? activeTab.dataset.n : '8';
    var n = (nStr === '' || nStr === undefined) ? 0 : (parseInt(nStr) || 8);
    weeks = n > 0 ? allWeeks.slice(-n) : allWeeks;
  }

  // Hide tab bar when a specific range is selected; show it for full-season mode.
  var body = document.getElementById('pmWeeklyTrendsBody');
  if (body) {
    var filterBar = body.querySelector('.pm-wt-filter-bar');
    if (filterBar) filterBar.style.display = rangeActive ? 'none' : '';
  }

  var el = document.getElementById('pmWtContent');
  if (el) el.innerHTML = buildWeeklyTrendRows(weeks, position);
}

function pmToggleWeeklyTrends(playerId) {
  var wrap = document.getElementById('pmWeeklyTrendsWrap');
  var body = document.getElementById('pmWeeklyTrendsBody');
  var btn  = document.getElementById('pmWeeklyTrendsBtn');
  if (!wrap || !body || !btn) return;

  var open = body.style.display !== 'none';
  if (open) {
    body.style.display = 'none';
    btn.innerHTML = 'Trends &#9662;';
    return;
  }
  body.style.display = '';
  btn.innerHTML = 'Trends &#9652;';
  if (wrap.dataset.loaded) return;
  wrap.dataset.loaded = '1';

  body.innerHTML = '<div style="padding:10px 0;color:var(--text-muted);font-size:12px;">Loading trends…</div>';
  var seasonParam = wrap.dataset.season ? ('?season=' + wrap.dataset.season) : '';
  var wrapPosition = wrap.dataset.position || '';
  fetch('/api/player-weekly-metrics/' + encodeURIComponent(playerId) + seasonParam)
    .then(function(r) { return r.json(); })
    .then(function(d) {
      wrap._weeklyData = d.weeks || [];
      // Weekly ↔ Season mode toggle. Season is only offered when the player has
      // 2+ seasons of data (set by loadAdvancedMetrics on the wrap).
      var hasSeason = wrap.dataset.multiseason === '1';
      var modeBar = hasSeason
        ? '<div class="pm-trends-mode br-chip-pop">'
          + '<button type="button" class="pm-trends-mode-btn is-active" data-mode="weekly">Weekly</button>'
          + '<button type="button" class="pm-trends-mode-btn" data-mode="season">Season</button>'
          + '</div>'
        : '';
      body.innerHTML = modeBar
        + '<div id="pmWtWeekly">'
        + '<div class="pm-wt-filter-bar">'
        + '<div class="otc-day-filters pm-wt-tabs br-chip-pop">'
        + '<button class="otc-day-filter pm-wt-tab" data-n="">All</button>'
        + '<button class="otc-day-filter pm-wt-tab" data-n="4">L4</button>'
        + '<button class="otc-day-filter pm-wt-tab pm-wt-tab-active" data-n="8">L8</button>'
        + '<button class="otc-day-filter pm-wt-tab" data-n="12">L12</button>'
        + '</div>'
        + '</div>'
        + '<div id="pmWtContent"></div>'
        + '</div>'
        + '<div id="pmWtSeason" style="display:none;"></div>';
      var tabs = body.querySelectorAll('.pm-wt-tab');
      tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
          tabs.forEach(function(t) { t.classList.remove('pm-wt-tab-active'); });
          tab.classList.add('pm-wt-tab-active');
          pmWtRender(wrap, wrapPosition);
        });
      });
      body.querySelectorAll('.pm-trends-mode-btn').forEach(function(mb) {
        mb.addEventListener('click', function() {
          body.querySelectorAll('.pm-trends-mode-btn').forEach(function(b) { b.classList.remove('is-active'); });
          mb.classList.add('is-active');
          pmTrendsSetMode(playerId, mb.dataset.mode);
        });
      });
      pmWtRender(wrap, wrapPosition);
    })
    .catch(function() {
      window.brErrorState(body, 'Could not load trends.', null, { compact: true });
    });
}

const _ADV_METRIC_DESCS = {
  // keyed by metric key — used by renderCompareMetricRows
  vorp: "Value Over Replacement Points: season PPR points minus a replacement-level starter at the same position (league-size aware, FLEX included). This is a season total, so missed games (injury, bench) can make VORP negative even when per-game production was starter-level.",
  war: "Wins Above Replacement: season VORP divided by points-per-win (≈ the league's weekly scoring spread). Translates points above replacement into the wins they were worth; elite players are typically 4-6+.",
  role_score: "Overall opportunity score (0-100) blending snap share, touches, and red-zone usage relative to the player's position.",
  snap_share: "Percent of the team's offensive snaps the player was on the field for.",
  opportunity_share: "Share of the team's targets plus carries that went to this player.",
  red_zone_usage: "Targets and carries inside the opponent's 20-yard line per game; a proxy for scoring opportunity.",
  grades_offense: "PFF's overall offensive grade (0-100) from play-by-play charting.",
  yards_per_touch: "Yards gained per combined carry and reception.",
  yards_per_attempt: "Passing yards per attempt; core passing efficiency stat.",
  completion_pct: "Percent of pass attempts completed.",
  adjusted_completion_rate: "Completion percent adjusted for drops, throwaways, spikes, and batted passes.",
  td_rate: "Percent of pass attempts that result in a touchdown.",
  int_rate: "Percent of pass attempts intercepted. Lower is better.",
  big_time_throw_rate: "PFF rate of high-difficulty, high-value throws (deep and into tight windows).",
  pressure_to_sack_rate: "Percent of pressured dropbacks that turn into sacks. Lower is better.",
  nfl_passer_rating: "Standard NFL passer rating (0-158.3).",
  pff_passing_grade: "PFF's passing grade (0-100).",
  total_pass_tds: "Total passing touchdowns in the season.",
  pass_tds_per_game: "Passing touchdowns per game.",
  yards_per_carry: "Rushing yards gained per carry.",
  rush_td_rate: "Percent of carries that result in a touchdown.",
  breakaway_percentage: "Percent of rushing yards that came on runs of 15+ yards; explosiveness.",
  elusive_rating: "PFF metric for yards created after contact and missed tackles forced, independent of blocking.",
  pff_rushing_grade: "PFF's rushing grade (0-100).",
  explosive_runs_10_plus: "Count of runs gaining 10 or more yards in the season (PFF). Raw explosive-play volume.",
  avoided_tackles: "Tackles avoided (missed, broken, or forced) on rush attempts per PFF. Rewards runners who make defenders miss.",
  total_rush_tds: "Total rushing touchdowns in the season.",
  route_participation: "Percent of the team's pass-play snaps on which the WR/TE ran a route.",
  target_share: "Percent of the team's total targets directed at this player.",
  air_yards_per_game: "Receiving air yards per game; a measure of downfield target volume.",
  air_yards_share: "Share of the team's total passing air yards directed at this player; combines target share with depth of target.",
  yards_per_target: "Receiving yards earned per time targeted; measures efficiency on volume.",
  yards_per_reception: "Average yards gained per catch; higher means a more downfield/explosive role.",
  catch_rate: "Percent of targets caught.",
  target_quality_score: "Composite of how valuable a player's targets are (depth, location, situation).",
  avg_depth_of_target: "Average depth of target: how far downfield (in yards) the player is thrown to.",
  contested_catch_rate: "Percent of contested (tightly covered) targets the player came down with.",
  yards_after_catch_per_reception: "Average yards gained after the catch per reception.",
  yards_after_catch: "Total yards gained after the catch in the season.",
  drop_rate: "Percent of catchable targets dropped. Lower is better.",
  yprr: "Receiving yards earned per route run (from PFF). Elite WRs are typically 2.0+.",
  ngs_avg_separation: "Average yards of separation from the nearest defender at the moment of catch/incompletion (NFL Next Gen Stats).",
  ngs_avg_cushion: "Average yards of cushion the defender gives at the snap (NFL Next Gen Stats).",
  ngs_created_separation: "Separation minus pre-snap cushion (NFL Next Gen Stats). Positive means the receiver created space vs the look they were given.",
  ngs_avg_yac_above_expectation: "Yards after catch above what was expected given the catch situation (NFL Next Gen Stats).",
  ngs_avg_time_to_throw: "Average seconds from snap to throw (NFL Next Gen Stats).",
  ngs_aggressiveness: "Percent of attempts thrown into tight windows (NFL Next Gen Stats). A public analogue to big-time-throw rate.",
  ngs_avg_completed_air_yards: "Average air yards on completed passes (NFL Next Gen Stats).",
  ngs_avg_air_yards_differential: "Completed air yards minus intended air yards (NFL Next Gen Stats).",
  ngs_avg_air_yards_to_sticks: "Average air yards relative to the first-down marker (NFL Next Gen Stats).",
  ngs_cpoe: "Next Gen Stats completion percentage over expected.",
  ngs_max_completed_air_distance: "Longest completed air distance in yards (NFL Next Gen Stats).",
  ngs_avg_time_to_los: "Average seconds for the rusher to reach the line of scrimmage (NFL Next Gen Stats).",
  ngs_percent_attempts_gte_eight_defenders: "Percent of rush attempts against 8 or more defenders in the box (NFL Next Gen Stats).",
  qb_hit_rate: "Percent of dropbacks on which the passer was hit (nflverse). A public pressure-faced proxy.",
  explosive_pass_rate: "Percent of pass attempts that gained 16+ yards (nflverse).",
  play_action_rate: "Percent of dropbacks that are play-action (FTN charting).",
  play_action_epa: "Expected Points Added per play-action dropback (FTN + nflverse).",
  out_of_pocket_rate: "Percent of dropbacks from outside the pocket (FTN charting).",
  blitz_rate_faced: "Percent of dropbacks against a blitz (FTN charting).",
  epa_vs_blitz: "Expected Points Added per dropback against the blitz (FTN + nflverse).",
  epa_vs_stacked_box: "Expected Points Added per rush against 8+ defenders in the box (FTN + nflverse).",
  rushing_success_rate: "Percent of rushes with positive EPA (nflverse).",
  receiving_success_rate: "Percent of targets with positive EPA (nflverse).",
  rushing_epa_per_att: "Expected Points Added per rush attempt (nflverse).",
  receiving_epa_per_target: "Expected Points Added per target (nflverse).",
  pacr: "Passing Air Conversion Ratio: passing yards ÷ passing air yards.",
  racr: "Receiver Air Conversion Ratio: receiving yards ÷ air yards.",
  epa_per_play: "Expected Points Added per play: the average value of each play the player was involved in.",
  passing_epa: "Total Expected Points Added on the player's pass attempts over the season.",
  rushing_epa: "Total Expected Points Added on the player's rushing attempts over the season.",
  receiving_epa: "Total Expected Points Added on the player's targets over the season.",
  cpoe: "Completion Percentage Over Expected: accuracy adjusted for throw difficulty.",
  ngs_rush_yards_over_expected_per_att: "Rush Yards Over Expected per attempt: yards created beyond what blocking/situation expected (NFL Next Gen Stats).",
  sack_rate: "Percent of dropbacks that ended in a sack. Lower is better.",
  scramble_rate: "Percent of dropbacks where the QB scrambled.",
  success_rate: "Percent of plays with positive EPA (a 'successful' play).",
  slot_rate: "Percent of routes run from the slot.",
  wide_rate: "Percent of routes run from out wide.",
  inline_rate: "Percent of snaps a tight end lined up inline (attached to the formation).",
  pass_block_rate: "Percent of pass snaps spent blocking rather than running a route.",
  grades_pass_block: "PFF's pass blocking grade (0-100).",
  total_rec_tds: "Total receiving touchdowns in the season.",
  total_carries: "Total carries in the season.",
  carries_per_game: "Carries per game.",
  total_targets: "Total targets in the season.",
  targets_per_game: "Targets per game.",
  total_receptions: "Total receptions in the season.",
  receptions_per_game: "Receptions per game.",
  total_touches: "Total carries plus receptions in the season.",
  touches_per_game: "Carries plus receptions per game.",
  total_tds: "Total touchdowns (rush + receiving + passing) in the season.",
  // keyed by display label — used by buildAdvancedMetricsHTML _cells()
  'Role Score': "Overall opportunity score (0-100) blending snap share, touches, and red-zone usage relative to the player's position.",
  'Snap Share': "Percent of the team's offensive snaps the player was on the field for.",
  'Route Partic': "Percent of the team's pass-play snaps on which the WR/TE ran a route. High route participation means a consistent full-time route runner.",
  'Opp Share': "Share of the team's targets plus carries that went to this player.",
  'RZ Usage/G': "Targets and carries inside the opponent's 20-yard line per game; a proxy for scoring opportunity.",
  'PFF Off Grade': "PFF's overall offensive grade (0-100) from play-by-play charting.",
  'Yds/Touch': "Yards gained per combined carry and reception.",
  'PFF Pass Grade': "PFF's passing grade (0-100).",
  'BTT Rate': "PFF rate of high-difficulty, high-value throws (deep and into tight windows).",
  'Adj Comp %': "Completion percent adjusted for drops, throwaways, spikes, and batted passes.",
  'Passer Rating': "Standard NFL passer rating (0-158.3).",
  'Yds/Attempt': "Passing yards per attempt; core passing efficiency stat.",
  'Completion %': "Percent of pass attempts completed.",
  'TD/INT Ratio': "Touchdowns thrown per interception; combines pass TD rate and INT rate into a single efficiency ratio.",
  'Pressure→Sack%': "Percent of pressured dropbacks that turn into sacks. Lower is better.",
  'Yds/Carry': "Rushing yards gained per carry.",
  'Rush TD Rate': "Percent of carries that result in a touchdown.",
  'PFF Rush Grade': "PFF's rushing grade (0-100).",
  'Breakaway %': "Percent of rushing yards that came on runs of 15+ yards; explosiveness.",
  'Explosive Runs': "Count of runs gaining 10 or more yards in the season (PFF). Raw explosive-play volume.",
  'Elusive Rating': "PFF metric for yards created after contact and missed tackles forced, independent of blocking.",
  'Avoided Tackles': "Tackles avoided (missed, broken, or forced) on rush attempts per PFF. Rewards runners who make defenders miss.",
  'Catch Rate': "Percent of targets caught.",
  'Yds/Route Run': "Receiving yards earned per route run (from PFF). Elite WRs are typically 2.0+.",
  'Drop Rate': "Percent of catchable targets dropped. Lower is better.",
  'Yds/Target': "Receiving yards earned per time targeted; measures efficiency on volume.",
  'Yds/Reception': "Average yards gained per catch; higher means a more downfield/explosive role.",
  'YAC/Rec': "Average yards gained after the catch per reception.",
  'YAC (season)': "Total yards gained after the catch in the season.",
  'aDOT': "Average depth of target: how far downfield (in yards) the player is thrown to.",
  'Contested Catch %': "Percent of contested (tightly covered) targets the player came down with.",
  'Target Share': "Percent of the team's total targets directed at this player.",
  'Air Yds/Game': "Receiving air yards per game; a measure of downfield target volume.",
  'Air Yards Share': "Share of the team's total passing air yards directed at this player; combines target share with depth of target.",
  'WOPR': "Weighted Opportunity Rate: (1.5 × target share) + (0.7 × rush share). Combines air and ground touches into a single opportunity share signal; elite receivers typically exceed 0.50.",
  'VORP': "Value Over Replacement Points: season PPR points minus a replacement-level starter at the same position (league-size aware, FLEX included). This is a season total, so missed games (injury, bench) can make VORP negative even when per-game production was starter-level.",
  'WAR': "Wins Above Replacement: season VORP divided by points-per-win (≈ the league's weekly scoring spread). Translates points above replacement into the wins they were worth; elite players are typically 4-6+.",
  'Target Quality': "Composite of how valuable a player's targets are (depth, location, situation).",
  'Slot Rate': "Percent of routes run from the slot.",
  'Wide Rate': "Percent of routes run from out wide.",
  'Inline Rate': "Percent of snaps a tight end lined up inline (attached to the formation).",
  'Block Rate': "Percent of pass snaps spent blocking rather than running a route.",
  'PFF Block Grade': "PFF's pass blocking grade (0-100).",
  'Usage Trend': "Recent usage trend: positive means the player's usage has risen vs. their season average.",
  'Eff Trend': "Recent efficiency trend: positive means the player has been more efficient recently vs. their season average.",
  'Carries/G': "Carries per game.",
  'Carries': "Total carries in the season.",
  'Targets/G': "Targets per game.",
  'Targets': "Total targets in the season.",
  'Touches/G': "Carries plus receptions per game.",
  'Rush TDs': "Total rushing touchdowns in the season.",
  'Rec TDs': "Total receiving touchdowns in the season.",
  'Total TDs': "Total touchdowns (rush + receiving + passing) in the season.",
  'Receptions/G': "Receptions per game.",
  'Receptions': "Total receptions in the season.",
  'Pass TDs': "Total passing touchdowns in the season.",
  'Pass TDs/G': "Passing touchdowns per game.",
  // Labels that previously had no definition (efficiency / EPA / NGS / volume tiles)
  'Passing EPA': "Total Expected Points Added on the player's pass attempts over the season.",
  'EPA/Play': "Expected Points Added per play: the average value of each play the player was involved in.",
  'CPOE': "Completion Percentage Over Expected: accuracy adjusted for throw difficulty.",
  'Success Rate': "Percent of plays with positive EPA (a 'successful' play).",
  'Sack Rate': "Percent of dropbacks that ended in a sack. Lower is better.",
  'Rushing EPA': "Total Expected Points Added on the player's rushing attempts over the season.",
  'RYOE/Att': "Rush Yards Over Expected per attempt: yards created beyond what blocking/situation expected (NFL Next Gen Stats).",
  'Separation': "Average yards of separation from the nearest defender at the moment of catch/incompletion (NFL Next Gen Stats).",
  'Cushion': "Average yards of cushion the defender gives at the snap (NFL Next Gen Stats).",
  'YAC Over Expected': "Yards after catch above what was expected given the catch situation (NFL Next Gen Stats).",
  'Receiving EPA': "Total Expected Points Added on the player's targets over the season.",
  'Touches': "Total carries plus receptions in the season.",
  'PPR Points': "Total PPR fantasy points scored in the season.",
  'PPR Pts/G': "PPR fantasy points per game.",
  'Rec Yds/G': "Receiving yards per game.",
  'Rec Yards': "Total receiving yards in the season.",
  'Rush Yards': "Total rushing yards in the season.",
};

function buildAdvancedMetricsHTML(metricsData, ranks, cfg, weekActive, counts, bounds) {
  counts = counts || {};
  bounds = bounds || {};
  let metrics = metricsData.metrics || {};
  // Normalize PFF position codes to canonical fantasy positions
  const _posNorm = { HB: 'RB', FB: 'RB', SE: 'WR', FL: 'WR' };
  const position = _posNorm[(metricsData.position || '').toUpperCase()] || metricsData.position;

  // Week-range view: only metrics that can be sliced by week are meaningful, so
  // drop everything that isn't weekly-capable (per the leaderboard config).
  if (weekActive && cfg && Object.keys(cfg).length) {
    const _wkOk = new Set(Object.keys(cfg).filter(k => cfg[k] && cfg[k].weeklyCapable));
    ['games', 'position'].forEach(k => _wkOk.add(k));  // meta keys used for rates
    const _filtered = {};
    for (const [k, v] of Object.entries(metrics)) { if (_wkOk.has(k)) _filtered[k] = v; }
    metrics = _filtered;
  }

  const defs = [];
  const g = metrics.games || 0;
  function _rankSub(key) {
    if (!ranks || !ranks[key]) return null;
    return `(#${ranks[key]})`;
  }
  function _pg(total) { return (g > 0 && total != null) ? total / g : null; }
  // Bar fill from the metric's value relative to the position's min–max range
  // (from the ranks API `bounds`). Preserves real magnitude — a big lead at the
  // top shows a long bar with a visible gap; bunched values show small gaps —
  // while staying position-aware (bounds are built from that position's field).
  // Returns null when no usable bounds exist (→ caller uses a value-scale fallback).
  function _boundsFill(key, val) {
    if (val == null || !bounds || !bounds[key]) return null;
    const b = bounds[key];
    let lo = b[0], hi = b[1];
    if (!(hi > lo)) return null;  // degenerate range (all equal / single player)
    let t = (val - lo) / (hi - lo);
    if (cfg && cfg[key] && cfg[key].lower_better) t = 1 - t;
    t = Math.max(0, Math.min(1, t));
    return 8 + t * 92;  // 8% floor so the worst still shows a sliver
  }
  // Recover a metric's numeric value (stored, or derived per-game from a total)
  // so the bounds-fill post-process can scale any def's bar.
  const _PM_PER_GAME = {
    carries_per_game: 'total_carries', targets_per_game: 'total_targets',
    receptions_per_game: 'total_receptions', touches_per_game: 'total_touches',
    rush_tds_per_game: 'total_rush_tds', rec_tds_per_game: 'total_rec_tds',
    pass_tds_per_game: 'total_pass_tds', total_tds_per_game: 'total_tds',
    rec_yards_per_game: 'total_rec_yards', rush_yards_per_game: 'total_rush_yards',
    ppr_pts_per_game: 'ppr_pts',
  };
  function _metricVal(key) {
    if (metrics[key] != null) return metrics[key];
    if (_PM_PER_GAME[key]) return _pg(metrics[_PM_PER_GAME[key]]);
    return null;
  }
  // Realistic elite value for metrics with no positional bounds (e.g. career
  // view). Keeps bars from rendering at a misleading width via old heuristics.
  const _PM_FALLBACK_SCALE = {
    wopr: 0.65, air_yards_share: 38, air_yards_per_game: 110,
    targets_per_game: 11, receptions_per_game: 8,
    carries_per_game: 18, touches_per_game: 20,
    rz_carries_pg: 3, rz_targets_pg: 2.5, red_zone_usage: 3,
    rec_yards_per_game: 100, rush_yards_per_game: 110,
    yprr: 3, route_participation: 100, routes_per_game: 40,
    pass_tds_per_game: 2.5, rush_tds_per_game: 1, rec_tds_per_game: 1,
    total_tds_per_game: 1.5, fpts_per_carry: 1.5, fpts_per_reception: 3,
    explosive_runs_pg: 2, avoided_tackles_pg: 2.5,
    ngs_created_separation: 3, ngs_avg_time_to_throw: 3.5,
    ngs_aggressiveness: 25, ngs_avg_completed_air_yards: 12,
    ngs_avg_air_yards_differential: 4, ngs_avg_air_yards_to_sticks: 4,
    ngs_cpoe: 10, ngs_max_completed_air_distance: 60,
    ngs_avg_time_to_los: 3, ngs_percent_attempts_gte_eight_defenders: 50,
    qb_hit_rate: 25, explosive_pass_rate: 20,
    play_action_rate: 40, play_action_epa: 0.4,
    out_of_pocket_rate: 25, blitz_rate_faced: 40,
    epa_vs_blitz: 0.4, epa_vs_stacked_box: 0.3,
    rushing_success_rate: 55, receiving_success_rate: 55,
    rushing_epa_per_att: 0.3, receiving_epa_per_target: 0.6,
    pacr: 1.2, racr: 1.2,
  };

  // Value: VORP / WAR — season-only, injected by the API for season views.
  if (metrics.vorp != null) {
    const v = metrics.vorp;
    defs.push({ label: 'VORP', fill: Math.min(Math.max(v, 0) / 150 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1), key: 'vorp', sub: _rankSub('vorp'), cat: 'Value' });
  }
  if (metrics.war != null) {
    const v = metrics.war;
    defs.push({ label: 'WAR', fill: Math.min(Math.max(v, 0) / 6 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(2), key: 'war', sub: _rankSub('war'), cat: 'Value' });
  }

  // Role Score is an internal signal (feeds breakout detection) and is not
  // surfaced on the front end — the API no longer sends it.
  // Snap Share (0–1 → %).  85 % = starter ceiling → full bar.
  if (metrics.snap_share != null && position !== "QB") {
    const pct = metrics.snap_share * 100;
    defs.push({ label: 'Snap Share', fill: Math.min(pct / 85 * 100, 100), display: pct.toFixed(1) + '%', key: 'snap_share', sub: _rankSub('snap_share'), cat: 'General' });
  }

  if (position === 'QB') {
    if (metrics.pff_passing_grade != null) {
      const v = metrics.pff_passing_grade;
      defs.push({ label: 'PFF Pass Grade', fill: v, display: v.toFixed(1), key: 'pff_passing_grade', sub: _rankSub('pff_passing_grade'), cat: 'Passing' });
    }
    if (metrics.big_time_throw_rate != null) {
      const v = metrics.big_time_throw_rate;
      defs.push({ label: 'BTT Rate', fill: Math.min(v / 15 * 100, 100), display: v.toFixed(1) + '%', key: 'big_time_throw_rate', sub: _rankSub('big_time_throw_rate'), cat: 'Passing' });
    }
    if (metrics.adjusted_completion_rate != null) {
      const v = metrics.adjusted_completion_rate;
      defs.push({ label: 'Adj Comp %', fill: Math.min(Math.max(v - 55, 0) / 35 * 100, 100), display: v.toFixed(1) + '%', key: 'adjusted_completion_rate', sub: _rankSub('adjusted_completion_rate'), cat: 'Passing' });
    }
    if (metrics.nfl_passer_rating != null) {
      const v = metrics.nfl_passer_rating;
      defs.push({ label: 'Passer Rating', fill: Math.min(Math.max(v - 60, 0) / 70 * 100, 100), display: v.toFixed(1), key: 'nfl_passer_rating', sub: _rankSub('nfl_passer_rating'), cat: 'Passing' });
    }
    if (metrics.yards_per_attempt != null) {
      const v = metrics.yards_per_attempt;
      defs.push({ label: 'Yds/Attempt', fill: Math.min(Math.max(v - 4, 0) / 6 * 100, 100), display: v.toFixed(1), key: 'yards_per_attempt', sub: _rankSub('yards_per_attempt'), cat: 'Passing' });
    }
    if (metrics.completion_pct != null) {
      const pct = metrics.completion_pct;
      defs.push({ label: 'Completion %', fill: Math.min(Math.max(pct - 50, 0) / 35 * 100, 100), display: pct.toFixed(1) + '%', key: 'completion_pct', sub: _rankSub('completion_pct'), cat: 'Passing' });
    }
    if (metrics.td_rate != null && metrics.int_rate != null && metrics.int_rate > 0) {
      const ratio = metrics.td_rate / metrics.int_rate;
      defs.push({ label: 'TD/INT Ratio', fill: Math.min(ratio * 20, 100), display: ratio.toFixed(2), cat: 'Passing' });
    }
    if (metrics.passing_epa != null) {
      const v = metrics.passing_epa;
      defs.push({ label: 'Passing EPA', fill: Math.min(Math.max(v + 50, 0) / 200 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1), key: 'passing_epa', sub: _rankSub('passing_epa'), cat: 'Passing' });
    }
    if (metrics.epa_per_play != null) {
      const v = metrics.epa_per_play;
      defs.push({ label: 'EPA/Play', fill: Math.min(Math.max(v + 0.2, 0) / 0.5 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(2), key: 'epa_per_play', sub: _rankSub('epa_per_play'), cat: 'Passing' });
    }
    if (metrics.cpoe != null) {
      const v = metrics.cpoe;
      defs.push({ label: 'CPOE', fill: Math.min(Math.max(v + 5, 0) / 15 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1) + '%', key: 'cpoe', sub: _rankSub('cpoe'), cat: 'Passing' });
    }
    if (metrics.success_rate != null) {
      const v = metrics.success_rate;
      defs.push({ label: 'Success Rate', fill: Math.min(v / 55 * 100, 100), display: v.toFixed(1) + '%', key: 'success_rate', sub: _rankSub('success_rate'), cat: 'Passing' });
    }
    if (metrics.sack_rate != null) {
      const v = metrics.sack_rate;
      defs.push({ label: 'Sack Rate', fill: Math.max(0, 100 - v * 8), display: v.toFixed(1) + '%', key: 'sack_rate', sub: _rankSub('sack_rate'), cat: 'Passing' });
    }
    if (metrics.pressure_to_sack_rate != null) {
      const v = metrics.pressure_to_sack_rate;
      const fill = Math.max(0, 100 - v);
      defs.push({ label: 'Pressure→Sack%', fill, display: v.toFixed(1) + '%', key: 'pressure_to_sack_rate', sub: _rankSub('pressure_to_sack_rate'), cat: 'Passing' });
    }
    if (metrics.yards_per_carry != null) {
      const v = metrics.yards_per_carry;
      defs.push({ label: 'Yds/Carry', fill: Math.min(v / 7 * 100, 100), display: v.toFixed(1), key: 'yards_per_carry', sub: _rankSub('yards_per_carry'), cat: 'Rushing' });
    }
    if (metrics.rush_td_rate != null) {
      const v = metrics.rush_td_rate;
      defs.push({ label: 'Rush TD Rate', fill: Math.min(v * 800, 100), display: (v * 100).toFixed(1) + '%', key: 'rush_td_rate', sub: _rankSub('rush_td_rate'), cat: 'Rushing' });
    }
  } else if (position === 'RB') {
    if (metrics.pff_rushing_grade != null) {
      const v = metrics.pff_rushing_grade;
      defs.push({ label: 'PFF Rush Grade', fill: v, display: v.toFixed(1), key: 'pff_rushing_grade', sub: _rankSub('pff_rushing_grade'), cat: 'Rushing' });
    }
    if (metrics.breakaway_percentage != null) {
      const v = metrics.breakaway_percentage;
      defs.push({ label: 'Breakaway %', fill: Math.min(v * 2.5, 100), display: v.toFixed(1) + '%', key: 'breakaway_percentage', sub: _rankSub('breakaway_percentage'), cat: 'Rushing' });
    }
    if (metrics.explosive_runs_10_plus != null) {
      const v = metrics.explosive_runs_10_plus;
      defs.push({ label: 'Explosive Runs', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(0), key: 'explosive_runs_10_plus', sub: _rankSub('explosive_runs_10_plus'), cat: 'Rushing' });
    }
    if (metrics.rushing_epa != null) {
      const v = metrics.rushing_epa;
      defs.push({ label: 'Rushing EPA', fill: Math.min(Math.max(v + 20, 0) / 60 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1), key: 'rushing_epa', sub: _rankSub('rushing_epa'), cat: 'Rushing' });
    }
    if (metrics.ngs_rush_yards_over_expected_per_att != null) {
      const v = metrics.ngs_rush_yards_over_expected_per_att;
      defs.push({ label: 'RYOE/Att', fill: Math.min(Math.max(v + 1, 0) / 2.5 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(2), key: 'ngs_rush_yards_over_expected_per_att', sub: _rankSub('ngs_rush_yards_over_expected_per_att'), cat: 'Rushing' });
    }
    if (metrics.elusive_rating != null) {
      const v = metrics.elusive_rating;
      defs.push({ label: 'Elusive Rating', fill: Math.min(v / 200 * 100, 100), display: v.toFixed(1), key: 'elusive_rating', sub: _rankSub('elusive_rating'), cat: 'Rushing' });
    }
    if (metrics.avoided_tackles != null && metrics.avoided_tackles > 0) {
      const v = metrics.avoided_tackles;
      defs.push({ label: 'Avoided Tackles', fill: Math.min(v / 30 * 100, 100), display: v.toFixed(0), key: 'avoided_tackles', sub: _rankSub('avoided_tackles'), cat: 'Rushing' });
    }
    if (metrics.yards_per_carry != null) {
      const v = metrics.yards_per_carry;
      defs.push({ label: 'Yds/Carry', fill: Math.min(v / 7 * 100, 100), display: v.toFixed(1), key: 'yards_per_carry', sub: _rankSub('yards_per_carry'), cat: 'Rushing' });
    }
    if (metrics.yards_per_touch != null) {
      const v = metrics.yards_per_touch;
      defs.push({ label: 'Yds/Touch', fill: Math.min(v / 8 * 100, 100), display: v.toFixed(1), key: 'yards_per_touch', sub: _rankSub('yards_per_touch'), cat: 'General' });
    }
    if (metrics.rush_td_rate != null) {
      const v = metrics.rush_td_rate;
      defs.push({ label: 'Rush TD Rate', fill: Math.min(v * 1000, 100), display: (v * 100).toFixed(1) + '%', key: 'rush_td_rate', sub: _rankSub('rush_td_rate'), cat: 'Rushing' });
    }
    if (metrics.opportunity_share != null) {
      const oppShare = metrics.opportunity_share;
      const fillPercent = Math.min(oppShare * 4, 100);
      const color = oppShare >= 25 ? '#10b981' : oppShare >= 15 ? '#3b82f6' : oppShare >= 10 ? '#f59e0b' : '#6b7280';
      defs.push({ label: 'Opp Share', fill: fillPercent, display: oppShare.toFixed(1) + '%', key: 'opportunity_share', sub: _rankSub('opportunity_share'), cat: 'General' });
    }
    if (metrics.catch_rate != null) {
      const pct = metrics.catch_rate * 100;
      defs.push({ label: 'Catch Rate', fill: Math.min(pct / 95 * 100, 100), display: pct.toFixed(1) + '%', key: 'catch_rate', sub: _rankSub('catch_rate'), cat: 'Receiving' });
    }
    if (metrics.grades_offense != null) {
      const v = metrics.grades_offense;
      defs.push({ label: 'PFF Off Grade', fill: v, display: v.toFixed(1), key: 'grades_offense', sub: _rankSub('grades_offense'), cat: 'Rushing' });
    }
  } else if (position === 'WR' || position === 'TE') {
    if (metrics.grades_offense != null) {
      const v = metrics.grades_offense;
      defs.push({ label: 'PFF Off Grade', fill: v, display: v.toFixed(1), key: 'grades_offense', sub: _rankSub('grades_offense'), cat: 'Receiving' });
    }
    if (metrics.catch_rate != null) {
      const pct = metrics.catch_rate * 100;
      defs.push({ label: 'Catch Rate', fill: Math.min(pct / 85 * 100, 100), display: pct.toFixed(1) + '%', key: 'catch_rate', sub: _rankSub('catch_rate'), cat: 'Receiving' });
    }
    if (metrics.yprr != null) {
      const v = metrics.yprr;
      defs.push({ label: 'Yds/Route Run', fill: Math.min(v / 3.0 * 100, 100), display: v.toFixed(2), key: 'yprr', sub: _rankSub('yprr'), cat: 'Receiving' });
    }
    if (metrics.drop_rate != null) {
      const v = metrics.drop_rate;
      const fill = Math.max(0, 100 - v * 5);
      defs.push({ label: 'Drop Rate', fill, display: v.toFixed(1) + '%', key: 'drop_rate', sub: _rankSub('drop_rate'), cat: 'Receiving' });
    }
    if (metrics.yards_per_target != null) {
      const v = metrics.yards_per_target;
      defs.push({ label: 'Yds/Target', fill: Math.min(Math.max(v - 2, 0) / 10 * 100, 100), display: v.toFixed(1), key: 'yards_per_target', sub: _rankSub('yards_per_target'), cat: 'Receiving' });
    }
    if (metrics.yards_per_reception != null) {
      const v = metrics.yards_per_reception;
      defs.push({ label: 'Yds/Reception', fill: Math.min(Math.max(v - 4, 0) / 14 * 100, 100), display: v.toFixed(1), key: 'yards_per_reception', sub: _rankSub('yards_per_reception'), cat: 'Receiving' });
    }
    if (metrics.yards_after_catch_per_reception != null) {
      const v = metrics.yards_after_catch_per_reception;
      defs.push({ label: 'YAC/Rec', fill: Math.min(v / 10 * 100, 100), display: v.toFixed(1), key: 'yards_after_catch_per_reception', sub: _rankSub('yards_after_catch_per_reception'), cat: 'Receiving' });
    }
    if (metrics.yards_after_catch != null) {
      const v = metrics.yards_after_catch;
      defs.push({ label: 'YAC (season)', fill: Math.min(v / 600 * 100, 100), display: Math.round(v).toString(), cat: 'Volume' });
    }
    if (metrics.avg_depth_of_target != null) {
      const v = metrics.avg_depth_of_target;
      defs.push({ label: 'aDOT', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(1), key: 'avg_depth_of_target', sub: _rankSub('avg_depth_of_target'), cat: 'Receiving' });
    }
    if (metrics.contested_catch_rate != null) {
      const v = metrics.contested_catch_rate;
      defs.push({ label: 'Contested Catch %', fill: Math.min(v / 65 * 100, 100), display: v.toFixed(1) + '%', key: 'contested_catch_rate', sub: _rankSub('contested_catch_rate'), cat: 'Receiving' });
    }
    if (metrics.ngs_avg_separation != null) {
      const v = metrics.ngs_avg_separation;
      defs.push({ label: 'Separation', fill: Math.min(v / 5 * 100, 100), display: v.toFixed(1), key: 'ngs_avg_separation', sub: _rankSub('ngs_avg_separation'), cat: 'Receiving' });
    }
    if (metrics.ngs_avg_cushion != null) {
      const v = metrics.ngs_avg_cushion;
      defs.push({ label: 'Cushion', fill: Math.min(v / 9 * 100, 100), display: v.toFixed(1), key: 'ngs_avg_cushion', sub: _rankSub('ngs_avg_cushion'), cat: 'Receiving' });
    }
    if (metrics.ngs_avg_yac_above_expectation != null) {
      const v = metrics.ngs_avg_yac_above_expectation;
      defs.push({ label: 'YAC Over Expected', fill: Math.min(Math.max(v + 2, 0) / 4 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1), key: 'ngs_avg_yac_above_expectation', sub: _rankSub('ngs_avg_yac_above_expectation'), cat: 'Receiving' });
    }
    if (metrics.target_share != null) {
      const pct = metrics.target_share;
      defs.push({ label: 'Target Share', fill: Math.min(pct / 28 * 100, 100), display: pct.toFixed(1) + '%', key: 'target_share', sub: _rankSub('target_share'), cat: 'Receiving' });
    }
    if (metrics.air_yards_per_game != null) {
      const v = metrics.air_yards_per_game;
      defs.push({ label: 'Air Yds/Game', fill: Math.min(v / 110 * 100, 100), display: v.toFixed(1), key: 'air_yards_per_game', sub: _rankSub('air_yards_per_game'), cat: 'Receiving' });
    }
    if (metrics.air_yards_share != null) {
      const pct = metrics.air_yards_share;
      defs.push({ label: 'Air Yards Share', fill: Math.min(pct / 35 * 100, 100), display: pct.toFixed(1) + '%', key: 'air_yards_share', sub: _rankSub('air_yards_share'), cat: 'Receiving' });
    }
    if (metrics.wopr != null) {
      const v = metrics.wopr;
      // WOPR ≥ 0.50 is elite; scale bar so 0.65 = 100%
      defs.push({ label: 'WOPR', fill: Math.min(v / 0.65 * 100, 100), display: v.toFixed(3), key: 'wopr', sub: _rankSub('wopr'), cat: 'Receiving' });
    }
    if (metrics.target_quality_score != null) {
      const v = metrics.target_quality_score;
      defs.push({ label: 'Target Quality', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(1), key: 'target_quality_score', sub: _rankSub('target_quality_score'), cat: 'Receiving' });
    }
    if (metrics.receiving_epa != null) {
      const v = metrics.receiving_epa;
      defs.push({ label: 'Receiving EPA', fill: Math.min(Math.max(v + 20, 0) / 80 * 100, 100), display: (v >= 0 ? '+' : '') + v.toFixed(1), key: 'receiving_epa', sub: _rankSub('receiving_epa'), cat: 'Receiving' });
    }
    if (metrics.slot_rate != null) {
      const v = metrics.slot_rate;
      defs.push({ label: 'Slot Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%', cat: 'Receiving' });
    }
    if (metrics.wide_rate != null) {
      const v = metrics.wide_rate;
      defs.push({ label: 'Wide Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%', cat: 'Receiving' });
    }
    if (position === 'TE' && metrics.inline_rate != null) {
      const v = metrics.inline_rate;
      defs.push({ label: 'Inline Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%', cat: 'Receiving' });
    }
    if (metrics.pass_block_rate != null) {
      const v = metrics.pass_block_rate;
      defs.push({ label: 'Block Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%', cat: 'Receiving' });
    }
    if (metrics.grades_pass_block != null) {
      const v = metrics.grades_pass_block;
      defs.push({ label: 'PFF Block Grade', fill: v, display: v.toFixed(1), cat: 'Receiving' });
    }
    if (metrics.yards_per_touch != null) {
      const v = metrics.yards_per_touch;
      defs.push({ label: 'Yds/Touch', fill: Math.min(v / 8 * 100, 100), display: v.toFixed(1), key: 'yards_per_touch', sub: _rankSub('yards_per_touch'), cat: 'General' });
    }
    if (metrics.total_touches != null) {
      defs.push({ label: 'Touches', fill: Math.min(metrics.total_touches / 150 * 100, 100), display: Math.round(metrics.total_touches).toString(), key: 'total_touches', sub: _rankSub('total_touches'), cat: 'Volume' });
    }
  }

  if (metrics.target_quality_score != null && position === 'RB') {
    const v = metrics.target_quality_score;
    defs.push({ label: 'Target Quality', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(1), key: 'target_quality_score', sub: _rankSub('target_quality_score'), cat: 'Receiving' });
  }

  if (metrics.red_zone_usage != null && position !== 'QB') {
    const v = metrics.red_zone_usage;
    defs.push({ label: 'RZ Usage/G', fill: Math.min(v / 3 * 100, 100), display: v.toFixed(1), key: 'red_zone_usage', sub: _rankSub('red_zone_usage'), cat: 'General' });
  }

  if (metrics.usage_trend != null) {
    const trend = metrics.usage_trend;
    const icon = trend > 5 ? '<i class="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> ' : trend < -5 ? '<i class="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> ' : '';
    defs.push({
      label: 'Usage Trend',
      fill: Math.min(Math.max((trend + 50) / 100 * 100, 0), 100),
      display: icon + (trend > 0 ? '+' : '') + trend.toFixed(1) + '%',
      forceColor: trend > 5 ? '#10b981' : trend < -5 ? '#ef4444' : null,
      cat: 'General',
    });
  }

  if (metrics.efficiency_trend != null) {
    const trend = metrics.efficiency_trend;
    const icon = trend > 5 ? '<i class="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> ' : trend < -5 ? '<i class="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> ' : '';
    defs.push({
      label: 'Eff Trend',
      fill: Math.min(Math.max((trend + 50) / 100 * 100, 0), 100),
      display: icon + (trend > 0 ? '+' : '') + trend.toFixed(1) + '%',
      forceColor: trend > 5 ? '#10b981' : trend < -5 ? '#ef4444' : null,
      cat: 'General',
    });
  }

  // PPR fantasy points (all positions, when available — week range view)
  if (metrics.ppr_pts != null) {
    const _ppMax = position === 'QB' ? 500 : position === 'RB' ? 350 : 300;
    defs.push({ label: 'PPR Points', fill: Math.min(metrics.ppr_pts / _ppMax * 100, 100), display: Math.round(metrics.ppr_pts).toString(), key: 'ppr_pts', sub: _rankSub('ppr_pts'), cat: 'General' });
  }
  if (metrics.ppr_pts_per_game != null) {
    defs.push({ label: 'PPR Pts/G', fill: Math.min(metrics.ppr_pts_per_game / 30 * 100, 100), display: metrics.ppr_pts_per_game.toFixed(1), key: 'ppr_pts_per_game', sub: _rankSub('ppr_pts_per_game'), cat: 'General' });
  }

  // Volume metrics per position (replacing old volDefs section)
  if (position === 'WR' || position === 'TE') {
    const _tpg = metrics.targets_per_game != null ? metrics.targets_per_game : _pg(metrics.total_targets);
    if (_tpg != null) defs.push({ label: 'Targets/G', fill: Math.min(_tpg / (position === 'TE' ? 8 : 12) * 100, 100), display: _tpg.toFixed(1), key: 'targets_per_game', sub: _rankSub('targets_per_game'), cat: 'Receiving' });
    const _rpg = metrics.receptions_per_game != null ? metrics.receptions_per_game : _pg(metrics.total_receptions);
    if (_rpg != null) defs.push({ label: 'Receptions/G', fill: Math.min(_rpg / 9 * 100, 100), display: _rpg.toFixed(1), key: 'receptions_per_game', sub: _rankSub('receptions_per_game'), cat: 'Receiving' });
    const _recypg = metrics.rec_yards_per_game != null ? metrics.rec_yards_per_game : _pg(metrics.total_rec_yards);
    if (_recypg != null) defs.push({ label: 'Rec Yds/G', fill: Math.min(_recypg / 100 * 100, 100), display: _recypg.toFixed(1), key: 'rec_yards_per_game', sub: _rankSub('rec_yards_per_game'), cat: 'Receiving' });
    if (metrics.total_targets != null) defs.push({ label: 'Targets', fill: Math.min(metrics.total_targets / (position === 'TE' ? 120 : 180) * 100, 100), display: Math.round(metrics.total_targets).toString(), key: 'total_targets', sub: _rankSub('total_targets'), cat: 'Volume' });
    if (metrics.total_receptions != null) defs.push({ label: 'Receptions', fill: Math.min(metrics.total_receptions / 130 * 100, 100), display: Math.round(metrics.total_receptions).toString(), key: 'total_receptions', sub: _rankSub('total_receptions'), cat: 'Volume' });
    if (metrics.total_rec_yards != null) defs.push({ label: 'Rec Yards', fill: Math.min(metrics.total_rec_yards / 1500 * 100, 100), display: Math.round(metrics.total_rec_yards).toString(), key: 'total_rec_yards', sub: _rankSub('total_rec_yards'), cat: 'Volume' });
    const _recTdMax = position === 'TE' ? 12 : 14;
    if (metrics.total_rec_tds != null) defs.push({ label: 'Rec TDs', fill: Math.min(metrics.total_rec_tds / _recTdMax * 100, 100), display: Math.round(metrics.total_rec_tds).toString(), key: 'total_rec_tds', sub: _rankSub('total_rec_tds'), cat: 'Receiving' });
    if (metrics.total_tds != null) defs.push({ label: 'Total TDs', fill: Math.min(metrics.total_tds / 15 * 100, 100), display: Math.round(metrics.total_tds).toString(), key: 'total_tds', sub: _rankSub('total_tds'), cat: 'General' });
  } else if (position === 'RB') {
    const _cpg = metrics.carries_per_game != null ? metrics.carries_per_game : _pg(metrics.total_carries);
    if (_cpg != null) defs.push({ label: 'Carries/G', fill: Math.min(_cpg / 22 * 100, 100), display: _cpg.toFixed(1), key: 'carries_per_game', sub: _rankSub('carries_per_game'), cat: 'Rushing' });
    if (metrics.total_carries != null) defs.push({ label: 'Carries', fill: Math.min(metrics.total_carries / 300 * 100, 100), display: Math.round(metrics.total_carries).toString(), key: 'total_carries', sub: _rankSub('total_carries'), cat: 'Volume' });
    const _tpgRb = metrics.targets_per_game != null ? metrics.targets_per_game : _pg(metrics.total_targets);
    if (_tpgRb != null) defs.push({ label: 'Targets/G', fill: Math.min(_tpgRb / 7 * 100, 100), display: _tpgRb.toFixed(1), key: 'targets_per_game', sub: _rankSub('targets_per_game'), cat: 'Receiving' });
    if (metrics.total_targets != null) defs.push({ label: 'Targets', fill: Math.min(metrics.total_targets / 100 * 100, 100), display: Math.round(metrics.total_targets).toString(), key: 'total_targets', sub: _rankSub('total_targets'), cat: 'Volume' });
    const _thpgRb = metrics.touches_per_game != null ? metrics.touches_per_game : _pg(metrics.total_touches);
    if (_thpgRb != null) defs.push({ label: 'Touches/G', fill: Math.min(_thpgRb / 25 * 100, 100), display: _thpgRb.toFixed(1), key: 'touches_per_game', sub: _rankSub('touches_per_game'), cat: 'General' });
    if (metrics.total_touches != null) defs.push({ label: 'Touches', fill: Math.min(metrics.total_touches / 300 * 100, 100), display: Math.round(metrics.total_touches).toString(), key: 'total_touches', sub: _rankSub('total_touches'), cat: 'Volume' });
    if (metrics.total_rush_yards != null) defs.push({ label: 'Rush Yards', fill: Math.min(metrics.total_rush_yards / 1700 * 100, 100), display: Math.round(metrics.total_rush_yards).toString(), key: 'total_rush_yards', sub: _rankSub('total_rush_yards'), cat: 'Volume' });
    if (metrics.total_rush_tds != null) defs.push({ label: 'Rush TDs', fill: Math.min(metrics.total_rush_tds / 16 * 100, 100), display: Math.round(metrics.total_rush_tds).toString(), key: 'total_rush_tds', sub: _rankSub('total_rush_tds'), cat: 'Rushing' });
    if (metrics.total_rec_tds != null) defs.push({ label: 'Rec TDs', fill: Math.min(metrics.total_rec_tds / 8 * 100, 100), display: Math.round(metrics.total_rec_tds).toString(), key: 'total_rec_tds', sub: _rankSub('total_rec_tds'), cat: 'Receiving' });
    if (metrics.total_tds != null) defs.push({ label: 'Total TDs', fill: Math.min(metrics.total_tds / 20 * 100, 100), display: Math.round(metrics.total_tds).toString(), key: 'total_tds', sub: _rankSub('total_tds'), cat: 'General' });
  } else if (position === 'QB') {
    if (metrics.total_pass_tds != null) defs.push({ label: 'Pass TDs', fill: Math.min(metrics.total_pass_tds / 40 * 100, 100), display: Math.round(metrics.total_pass_tds).toString(), key: 'total_pass_tds', sub: _rankSub('total_pass_tds'), cat: 'Passing' });
    const _ptpg = metrics.pass_tds_per_game != null ? metrics.pass_tds_per_game : _pg(metrics.total_pass_tds);
    if (_ptpg != null) defs.push({ label: 'Pass TDs/G', fill: Math.min(_ptpg / 3 * 100, 100), display: _ptpg.toFixed(1), key: 'pass_tds_per_game', sub: _rankSub('pass_tds_per_game'), cat: 'Passing' });
    if (metrics.total_rush_tds != null) defs.push({ label: 'Rush TDs', fill: Math.min(metrics.total_rush_tds / 15 * 100, 100), display: Math.round(metrics.total_rush_tds).toString(), key: 'total_rush_tds', sub: _rankSub('total_rush_tds'), cat: 'Rushing' });
    if (metrics.total_tds != null) defs.push({ label: 'Total TDs', fill: Math.min(metrics.total_tds / 45 * 100, 100), display: Math.round(metrics.total_tds).toString(), key: 'total_tds', sub: _rankSub('total_tds'), cat: 'General' });
  }

  // Append any remaining cfg metrics not already covered above
  if (cfg && Object.keys(cfg).length) {
    const _shownLabels = new Set(defs.map(d => d.label));
    const _shownKeys = new Set([
      'vorp','war',
      'role_score','snap_share','route_participation','opportunity_share',
      'red_zone_usage','grades_offense','yards_per_touch',
      'pff_passing_grade','big_time_throw_rate','adjusted_completion_rate',
      'nfl_passer_rating','yards_per_attempt','completion_pct',
      'td_rate','int_rate','passing_epa','epa_per_play','cpoe',
      'success_rate','sack_rate','pressure_to_sack_rate','scramble_rate',
      'yards_per_carry','rush_td_rate',
      'pff_rushing_grade','breakaway_percentage','explosive_runs_10_plus',
      'rushing_epa','ngs_rush_yards_over_expected_per_att','elusive_rating',
      'avoided_tackles','catch_rate',
      'yprr','drop_rate','yards_per_target','yards_per_reception',
      'yards_after_catch_per_reception','yards_after_catch',
      'avg_depth_of_target','contested_catch_rate',
      'ngs_avg_separation','ngs_avg_cushion','ngs_avg_yac_above_expectation',
      'target_share','air_yards_per_game','air_yards_share',
      'target_quality_score','receiving_epa','slot_rate','wide_rate',
      'inline_rate','pass_block_rate','grades_pass_block',
      'total_carries','total_targets','total_touches','total_tds',
      'total_rush_tds','total_rec_tds','total_receptions','total_pass_tds',
      'usage_trend','efficiency_trend','games',
      'ppr_pts','ppr_pts_per_game',
      'total_rec_yards','total_rush_yards',
      'rec_yards_per_game','rush_yards_per_game',
      'carries_per_game','targets_per_game','receptions_per_game','touches_per_game',
      'total_routes','routes_per_game',
      'pass_tds_per_game','rush_tds_per_game','rec_tds_per_game',
    ]);
    for (const [key, spec] of Object.entries(cfg)) {
      if (_shownKeys.has(key)) continue;
      const val = metrics[key];
      if (val == null) continue;
      if (spec.positions && spec.positions.length && position) {
        if (!spec.positions.includes(position)) continue;
      }
      let fill, displayStr;
      // Display string (independent of bar scaling).
      if (spec.pct_frac) displayStr = (val * 100).toFixed(1) + '%';
      else if (spec.pct) displayStr = val.toFixed(1) + '%';
      else if (spec.integer) displayStr = Math.round(val).toString();
      else { const abs = Math.abs(val); displayStr = val.toFixed(abs >= 10 ? 1 : 2); }
      // Bar fill: prefer value-relative-to-position bounds (magnitude-preserving
      // + position-aware); fall back to a corrected per-metric value scale, then
      // a generic formula.
      const bf = _boundsFill(key, val);
      if (bf != null) {
        fill = bf;  // bounds already encode direction (good = high), no flip
      } else {
        const sc = _PM_FALLBACK_SCALE[key];
        if (spec.pct_frac) fill = Math.min(val * 100, 100);
        else if (sc) fill = Math.min(Math.abs(val) / sc * 100, 100);
        else if (spec.pct) fill = Math.min(Math.abs(val), 100);
        else if (spec.integer) fill = Math.min(Math.abs(val), 100);
        else { const abs = Math.abs(val); fill = abs < 1 ? abs * 100 : Math.min(abs * 5, 100); }
        if (spec.lower_better) fill = Math.max(0, 100 - fill);
      }
      defs.push({ label: spec.label || key, key, fill, display: displayStr, sub: _rankSub(key), desc: spec.desc || '' });
    }
  }

  // ── Bar fill model — four shapes, each matched to what the metric means ────
  //  • SCORE  — designed 0→ceiling scale; value ÷ ceiling (grades, ratings,
  //             VORP/WAR). Magnitude shows; below-replacement floors near empty.
  //  • MINMAX — wide-range value that can go negative (EPA totals); map the
  //             position's [min,max] onto the bar so a big lead AND negatives
  //             both render (lowest, often negative, sits at the floor).
  //  • RANK   — compressed efficiency rates that bunch near the top; percentile
  //             within position so a mid-pack player reads mid-pack.
  //  • LEADER — non-negative volume; value ÷ position leader (½ leader = ½ bar).
  const _SCORE_CEIL = { role_score: 100, grades_offense: 100, pff_passing_grade: 100,
    pff_rushing_grade: 100, nfl_passer_rating: 158.3, vorp: 150, war: 6 };
  const _MINMAX_KEYS = new Set(['passing_epa', 'rushing_epa', 'receiving_epa']);
  const _RATE_KEYS = new Set(['avoided_tackles_pg', 'explosive_runs_pg']);  // per-carry rates → rank
  function _rankFill(key) {
    const r = ranks && ranks[key];
    const n = counts && counts[key];
    if (!r || !n || n < 2) return null;
    return 8 + Math.max(0, Math.min(1, (n - r) / (n - 1))) * 92;  // #1 → 100, last → 8
  }
  function _leaderFill(key, val) {
    const b = bounds && bounds[key];
    if (!b || !(b[1] > 0)) return null;
    return Math.max(4, Math.min(100, (val / b[1]) * 100));  // relative to leader (max)
  }
  function _barFor(key, val) {
    const ceil = _SCORE_CEIL[key];
    if (ceil) return Math.max(4, Math.min(100, (val / ceil) * 100));
    const spec = cfg && cfg[key];
    if (_MINMAX_KEYS.has(key)) return _boundsFill(key, val) != null ? _boundsFill(key, val) : _rankFill(key);
    if ((spec && (spec.efficiency || spec.pct || spec.pct_frac)) || _RATE_KEYS.has(key)) {
      const r = _rankFill(key); return r != null ? r : _boundsFill(key, val);
    }
    const l = _leaderFill(key, val); return l != null ? l : _rankFill(key);  // volume
  }
  defs.forEach(function(d) {
    if (!d.key || d.forceColor) return;
    const val = _metricVal(d.key);
    if (val == null) return;
    let f = _barFor(d.key, val);
    if (f == null) f = _boundsFill(d.key, val);  // last resort
    if (f != null) d.fill = f;
  });

  if (defs.length === 0) return '';

  function _cells(m, isRank) {
    if (!m) return '<span></span><span></span><span></span>';
    const fill = Math.max(0, Math.min(100, m.fill));
    const color = m.forceColor || (fill >= 60 ? '#10b981' : fill >= 35 ? '#3b82f6' : '#f59e0b');
    const subCls = isRank ? ' rank-badge' : '';
    const subLine = m.sub ? `<div class="pm-comp-sub${subCls}">${m.sub}</div>` : '';
    const _desc = m.desc || _ADV_METRIC_DESCS[m.label] || '';
    // Custom hover tooltip only (data-def + advEnterMetricDef). No native title=
    // attribute — it would surface a second, plain browser tooltip on hover.
    const _defAttr = _desc ? ` data-def="${_desc.replace(/"/g, '&quot;')}" onclick="advShowMetricDef(event)" onmouseenter="advEnterMetricDef(event)" onmouseleave="advLeaveMetricDef(event)"` : '';
    return `<span class="pm-comp-label"${_defAttr}>${m.label}</span>` +
      `<div class="pm-comp-bar-wrap"><div class="pm-comp-bar" style="width:${fill.toFixed(1)}%;background:${color};"></div></div>` +
      `<div class="pm-comp-val" style="color:${color};">${m.display}${subLine}</div>`;
  }

  function _grid(arr, isRank) {
    const mid = Math.ceil(arr.length / 2);
    const left = arr.slice(0, mid);
    const right = arr.slice(mid);
    let rows = '';
    for (let i = 0; i < Math.max(left.length, right.length); i++) {
      rows += _cells(left[i], isRank);
      rows += '<div class="am-vert-sep"></div>';
      rows += _cells(right[i], isRank);
    }
    return `<div class="adv-metrics-grid">${rows}</div>`;
  }

  const rankNote = ranks && Object.keys(ranks).length
    ? '<div class="am-rank-note" title="Minimums: 4+ games · efficiency metrics also require 20+ carries (rush) · 15+ targets (receiving) · 50+ attempts (passing)">Ranked among qualified players</div>'
    : '';

  // Annotate defs with keys from cfg by matching labels
  if (cfg) {
    const _labelToKey = {};
    for (const [k, spec] of Object.entries(cfg)) {
      if (spec.label) _labelToKey[spec.label] = k;
    }
    for (const def of defs) {
      if (!def.key && def.label && _labelToKey[def.label]) {
        def.key = _labelToKey[def.label];
      }
    }
  }

  let html = rankNote;
  if (cfg && Object.keys(cfg).length) {
    // Group defs by category using cfg (best-effort — unlabeled defs go to 'Other')
    const _CAT_ORDER = ['Value', 'General', 'Passing', 'Rushing', 'Receiving', 'Volume'];
    const _catGroups = {};
    const _uncategorized = [];
    for (const def of defs) {
      const key = def.key;
      const spec = key && cfg[key];
      const cat = (spec && spec.category) || def.cat || null;
      if (cat) { (_catGroups[cat] = _catGroups[cat] || []).push(def); }
      else { _uncategorized.push(def); }
    }
    const orderedCats = _CAT_ORDER.filter(c => _catGroups[c]);
    for (const [c] of Object.entries(_catGroups)) { if (!_CAT_ORDER.includes(c)) orderedCats.push(c); }
    if (_uncategorized.length) orderedCats.push('__other__');
    if (_catGroups['__other__']) _catGroups['__other__'] = _uncategorized;
    else if (_uncategorized.length) _catGroups['__other__'] = _uncategorized;
    for (const cat of orderedCats) {
      // Alphabetical within each category for predictable scanning.
      const group = (_catGroups[cat] || []).slice()
        .sort((a, b) => (a.label || '').localeCompare(b.label || ''));
      if (!group.length) continue;
      const catLabel = cat === '__other__' ? '' : cat;
      if (catLabel) html += '<div class="am-metrics-cat-head">' + catLabel + '</div>';
      html += _grid(group, true);
    }
  } else {
    html += _grid(defs, true);
  }

  if (metricsData.as_of_date) {
    html += `<div style="font-size:11px;color:var(--text-muted);margin-top:10px;text-align:right;">As of ${metricsData.as_of_date}</div>`;
  }

  return html;
}

function toggleGameLogYear(arg) {
  // Resolve the header element. Accepts the clicked header (preferred - works
  // when duplicate year IDs exist, e.g. both players in the compare modal) or
  // a year string for backward compatibility.
  let header;
  if (typeof arg === 'string') {
    const toggle = document.getElementById(`toggle-${arg}`);
    header = toggle ? toggle.closest('.game-log-year-header') : null;
  } else {
    header = arg;
  }
  if (!header) return;

  const content = header.nextElementSibling;
  const toggle = header.querySelector('.game-log-year-toggle');
  if (!content || !toggle) return;

  // Toggle the collapsed class only; the CSS rotates the original arrow glyph.
  // (Swapping innerHTML to a Font Awesome icon here caused an inconsistent/red
  // arrow on iOS when the injected glyph fell back to a system font.)
  if (content.classList.contains('expanded')) {
    content.classList.remove('expanded');
    toggle.classList.add('collapsed');
  } else {
    content.classList.add('expanded');
    toggle.classList.remove('collapsed');
  }
}

function closePlayerModal() {
  const overlay = document.querySelector('.player-modal-overlay');
  if (overlay) {
    const _return = overlay._pmReturnFocus;
    document.body.style.overflow = '';
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 200);
    // Restore focus to whatever opened the modal (the clicked player row / chip),
    // so keyboard users are not dumped back at the top of the document.
    if (_return && typeof _return.focus === 'function') {
      try { _return.focus(); } catch (_) {}
    }
  }
}

window.openPlayerModal = openPlayerModal;
window.closePlayerModal = closePlayerModal;
if (window.openPlayerModal) {
  try { delete window.openPlayerModal.__stub; } catch (_) { window.openPlayerModal.__stub = false; }
}
