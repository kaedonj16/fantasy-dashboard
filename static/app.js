// ============================================================
// app.js (DROP-IN)
// - Idempotent init (safe across innerHTML swaps)
// - Standings sort supports DESC -> ASC -> none (and Shift+click multi-sort)
// - Refresh button supports platform/season and re-inits page root
// - Matchup carousel handlers delegated + resize-safe
// - Plotly: resize + relayout + redraw + viewport hooks (better with page zoom/layout shifts)
// ============================================================


// ------------------------------------------------------------
// Prevent scroll restoration on navigation (mobile fix)
// ------------------------------------------------------------
if ('scrollRestoration' in history) {
  history.scrollRestoration = 'manual';
}

// Immediately scroll to top on any page load
window.scrollTo(0, 0);
document.documentElement.scrollTop = 0;
document.body.scrollTop = 0;

// Prevent scroll during page load
// ── Page navigation progress bar ─────────────────────────────────────────────
(function () {
  var bar = null;
  function getBar() {
    if (!bar) bar = document.getElementById('page-load-bar');
    return bar;
  }
  function startBar() {
    var b = getBar(); if (!b) return;
    b.className = '';
    b.style.width = '0%';
    b.style.opacity = '1';
    requestAnimationFrame(function () { b.className = 'plb-active'; });
  }
  function finishBar() {
    var b = getBar(); if (!b) return;
    b.className = 'plb-done';
    setTimeout(function () { b.className = ''; b.style.width = '0%'; }, 500);
  }

  // Show bar when a same-origin link is clicked (not anchor, not external, not new tab)
  document.addEventListener('click', function (e) {
    var link = e.target.closest('a[href]');
    if (!link) return;
    var href = link.getAttribute('href') || '';
    if (link.target === '_blank' || href.startsWith('#') || href.startsWith('javascript:') ||
        href.startsWith('mailto:') || (href.startsWith('http') && !href.startsWith(window.location.origin))) return;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    startBar();
  }, true);

  // Finish bar once the new page is interactive
  window.addEventListener('pageshow', finishBar);
  window.addEventListener('DOMContentLoaded', finishBar);
})();

// ── Login / subscribe gate ────────────────────────────────────────────────────
/**
 * Render a sign-in or subscribe prompt inside a container element.
 * @param {string|Element} target  - element ID or DOM element
 * @param {object} opts            - { title, description, feature }
 */
function showLoginGate(target, opts) {
  const el = typeof target === 'string' ? document.getElementById(target) : target;
  if (!el) return;
  const ctx = window.__brctx || {};
  const hasLeague = ctx.leagueId && ctx.leagueId !== 'None' && ctx.leagueId !== '';
  opts = opts || {};

  if (!ctx.is_logged_in) {
    const signinBtn = hasLeague
      ? `<button class="login-gate-btn" onclick="document.getElementById('signinModal').style.display='flex'">Sign In</button>`
      : `<a class="login-gate-btn" href="/">Get Started</a>`;
    el.innerHTML = `
      <div class="login-gate">
        <div class="login-gate-icon">🔐</div>
        <div class="login-gate-title">${opts.title || 'Sign in to continue'}</div>
        <div class="login-gate-desc">${opts.description || 'Enter your Sleeper username to see personalized data for your team.'}</div>
        ${signinBtn}
      </div>`;
  } else if (!ctx.isPremium) {
    const pricingUrl = hasLeague
      ? `/${ctx.platform}/${ctx.season}/${ctx.leagueId}/pricing`
      : '/pricing';
    el.innerHTML = `
      <div class="login-gate">
        <div class="login-gate-icon">⭐</div>
        <div class="login-gate-title">${opts.title || 'Premium Feature'}</div>
        <div class="login-gate-desc">${opts.description || 'Subscribe to unlock this feature and get the full BR Fantasy experience.'}</div>
        <a class="login-gate-btn" href="${pricingUrl}">Subscribe</a>
        <a class="login-gate-btn-secondary" href="${pricingUrl}">View Plans</a>
      </div>`;
  }
}

// ── Data freshness chip ───────────────────────────────────────────────────────
(function () {
  var STALE_MS = 6 * 60 * 60 * 1000; // 6 hours - matches server CACHE_TTL
  function initFreshness() {
    var main = document.getElementById('page-root');
    if (!main) return;
    var ts = parseInt(main.dataset.cacheTs || '0', 10);
    if (!ts) return;
    var chip = document.getElementById('cache-freshness');
    if (!chip) {
      chip = document.createElement('div');
      chip.id = 'cache-freshness';
      document.body.appendChild(chip);
    }
    function update() {
      var diff = Date.now() - ts;
      var mins = Math.floor(diff / 60000);
      var label;
      if (mins < 1) label = 'Data just updated';
      else if (mins < 60) label = 'Data • ' + mins + 'm ago';
      else label = 'Data • ' + Math.floor(mins / 60) + 'h ago';
      chip.textContent = label;
      chip.classList.add('cf-visible');
      chip.classList.toggle('cf-stale', diff > STALE_MS);
    }
    update();
    setInterval(update, 60000);
  }
  document.addEventListener('DOMContentLoaded', initFreshness);
})();

window.addEventListener('beforeunload', function() {
  window.scrollTo(0, 0);
});

// Prevent any programmatic scrolls during initial page load
let scrollBlocked = true;
window.addEventListener('scroll', function() {
  if (scrollBlocked) {
    window.scrollTo(0, 0);
  }
}, { passive: true });

// Unblock scrolling after page is fully loaded and initialized
window.addEventListener('load', function() {
  setTimeout(function() {
    scrollBlocked = false;
  }, 300);
});

// Force scroll to top after a short delay to catch any delayed scrolls
setTimeout(function() {
  window.scrollTo(0, 0);
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
}, 0);

setTimeout(function() {
  window.scrollTo(0, 0);
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
}, 50);

setTimeout(function() {
  window.scrollTo(0, 0);
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
}, 100);

// ------------------------------------------------------------
// One-time guards (prevents listener stacking across swaps)
// ------------------------------------------------------------
window.__BR_INIT_FLAGS__ = window.__BR_INIT_FLAGS__ || {
  globalsBound: false,
  resizeBound: false,
  plotlyViewportBound: false,
};

// Small helper: bind an event only once per element+key
function bindOnce(el, key, type, handler, options) {
  if (!el) return;
  const k = `__bound_${key}`;
  if (el[k]) return;
  el.addEventListener(type, handler, options);
  el[k] = true;
}

// ------------------------------------------------------------
// Dark Mode Toggle
// ------------------------------------------------------------
(function initDarkMode() {
  // Apply saved theme immediately to prevent flash
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  function toggleDarkMode() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    if (newTheme === 'dark') {
      root.setAttribute('data-theme', 'dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.removeAttribute('data-theme');
      localStorage.setItem('theme', 'light');
    }

    // Update toggle button icons
    updateThemeIcons();
    
    // Update existing Plotly charts to match new theme
    updatePlotlyChartsTheme();
  }

  function updatePlotlyChartsTheme() {
    if (typeof Plotly === 'undefined') return;
    
    const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'plotly_dark' : 'plotly_white';
    
    // Update team modal charts if they exist
    const weeklyChart = document.getElementById('teamWeeklyChart');
    const radarChart = document.getElementById('teamRadarChart');
    
    if (weeklyChart) {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const textColor = isDark ? '#ffffff' : '#000000';
      const bgColor = isDark ? '#1f2937' : '#ffffff';
      const borderColor = isDark ? '#374151' : '#e5e7eb';
      
      Plotly.relayout(weeklyChart, {
        template: theme,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'hoverlabel.bgcolor': bgColor,
        'hoverlabel.bordercolor': borderColor,
        'hoverlabel.font.color': textColor
      });
    }
    
    if (radarChart) {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const textColor = isDark ? '#ffffff' : '#000000';
      const gridColor = isDark ? '#374151' : '#e5e7eb';
      const lineColor = isDark ? '#9ca3af' : '#6b7280';
      
      Plotly.relayout(radarChart, {
        template: theme,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'polar.radialaxis.tickcolor': textColor,
        'polar.radialaxis.gridcolor': gridColor,
        'polar.radialaxis.linecolor': lineColor,
        'polar.angularaxis.tickcolor': textColor,
        'polar.angularaxis.gridcolor': gridColor,
        'polar.angularaxis.linecolor': lineColor,
        'polar.bgcolor': 'rgba(0,0,0,0)',
        'font.color': textColor
      });
    }
  }

  function updateThemeIcons() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const lightIcons = document.querySelectorAll('.theme-icon.light-icon');
    const darkIcons = document.querySelectorAll('.theme-icon.dark-icon');
    const themeTexts = document.querySelectorAll('.theme-text');

    lightIcons.forEach(icon => {
      icon.style.display = isDark ? 'none' : 'inline';
    });
    darkIcons.forEach(icon => {
      icon.style.display = isDark ? 'inline' : 'none';
    });
    
    themeTexts.forEach(text => {
      text.textContent = isDark ? 'Light Mode' : 'Dark Mode';
    });
  }

  // Bind toggle button
  function bindDarkModeToggle() {
    const toggleBtn = document.getElementById('darkModeToggle');
    if (toggleBtn && !toggleBtn.__darkModeInitialized) {
      toggleBtn.addEventListener('click', toggleDarkMode);
      toggleBtn.__darkModeInitialized = true;
      updateThemeIcons();
    }
    
    // Also bind settings dropdown button
    const settingsBtn = document.getElementById('settingsDarkModeBtn');
    if (settingsBtn && !settingsBtn.__darkModeInitialized) {
      settingsBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleDarkMode();
      });
      settingsBtn.__darkModeInitialized = true;
      updateThemeIcons();
    }
  }

  // Initialize on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindDarkModeToggle);
  } else {
    bindDarkModeToggle();
  }

  // Re-bind after any page updates (for SPA-like behavior)
  const observer = new MutationObserver(function(mutations) {
    bindDarkModeToggle();
  });

  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
  }

  // Expose toggleDarkMode globally for settings dropdown
  window.toggleDarkMode = toggleDarkMode;
})();

// ------------------------------------------------------------
// Reusable Init Helpers (IDEMPOTENT)
// ------------------------------------------------------------

function initManagerPills(root = document) {
  const pills = Array.from(root.querySelectorAll(".manager-pill"));
  const panels = Array.from(root.querySelectorAll(".team-panel"));
  const leftArrow = root.querySelector(".pill-arrow-left");
  const rightArrow = root.querySelector(".pill-arrow-right");
  if (!pills.length) return;

  const stateHost = root.querySelector(".manager-pills") || root;
  let currentIndex = Number(stateHost.__currentIndex ?? -1);

  if (currentIndex < 0 || currentIndex >= pills.length) {
    currentIndex = pills.findIndex(p => p.classList.contains("active"));
    if (currentIndex === -1) currentIndex = 0;
  }

  function activateIndex(idx) {
    if (idx < 0) idx = pills.length - 1;
    if (idx >= pills.length) idx = 0;
    currentIndex = idx;
    stateHost.__currentIndex = currentIndex;

    const activePill = pills[currentIndex];
    const teamId = activePill.getAttribute("data-team-id");

    pills.forEach(p => p.classList.toggle("active", p === activePill));
    panels.forEach(panel => {
      const pid = panel.getAttribute("data-team-id");
      panel.classList.toggle("active", pid === teamId);
    });

    activePill.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }

  pills.forEach((pill, idx) => {
    bindOnce(pill, "managerPillClick", "click", () => activateIndex(idx));
  });

  bindOnce(leftArrow, "pillArrowLeft", "click", () => activateIndex(currentIndex - 1));
  bindOnce(rightArrow, "pillArrowRight", "click", () => activateIndex(currentIndex + 1));

  activateIndex(currentIndex);
}

function initCardTabs(root = document) {
  root.querySelectorAll(".card-tabs").forEach(card => {
    const tabs = Array.from(card.querySelectorAll(".tab-btn"));
    const panels = Array.from(card.querySelectorAll(".tab-panel"));
    if (!tabs.length) return;

    tabs.forEach(tab => {
      bindOnce(tab, "cardTabClick", "click", () => {
        const target = tab.dataset.tab;
        tabs.forEach(t => t.classList.toggle("active", t === tab));
        panels.forEach(p => p.classList.toggle("active", p.dataset.tab === target));
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Playoff Odds - lazy-loaded on first tab click
// ---------------------------------------------------------------------------

function initPlayoffOdds(root = document) {
  root.querySelectorAll('.tab-btn[data-tab="playoff-odds"]').forEach(btn => {
    bindOnce(btn, "playoffOddsLoad", "click", () => {
      const panel = root.getElementById
        ? root.getElementById("playoffOddsPanel")
        : document.getElementById("playoffOddsPanel");
      const _poTTL = 5 * 60 * 1000; // 5 minutes
      if (!panel) return;
      const _poLoadedAt = parseInt(panel.dataset.loadedAt || '0', 10);
      if (_poLoadedAt && Date.now() - _poLoadedAt < _poTTL) return;
      panel.dataset.loadedAt = String(Date.now());

      const leagueId = btn.dataset.leagueId;
      const platform = btn.dataset.platform;
      const season   = btn.dataset.season;

      fetch(
        `/api/playoff-odds?platform=${encodeURIComponent(platform)}` +
        `&league_id=${encodeURIComponent(leagueId)}` +
        `&season=${encodeURIComponent(season)}`
      )
        .then(r => r.json())
        .then(data => {
          if (data.error || !data.odds) {
            panel.innerHTML = '<p class="po-error">Unable to load playoff odds.</p>';
            return;
          }
          panel.innerHTML = _renderPlayoffOdds(data);
        })
        .catch(() => {
          panel.innerHTML = '<p class="po-error">Unable to load playoff odds.</p>';
          delete panel.dataset.loadedAt;
        });
    });
  });
}

function _calcPlayoffByes(N) {
  if (N < 2) return 0;
  if (N % 2 === 0) {
    const largest = 1 << (Math.floor(Math.log2(N)));
    return N - largest;
  }
  const next = 1 << Math.ceil(Math.log2(N));
  return next - N;
}

function _renderPlayoffOdds(data) {
  const { odds, is_complete, current_week, playoff_week_start, playoff_teams, playoff_byes } = data;
  if (!odds || !odds.length) return '<p class="po-error">No data available.</p>';

  const sorted = [...odds].sort((a, b) =>
    b.playoff_pct - a.playoff_pct || b.avg_final_wins - a.avg_final_wins
  );

  const isProjected = !is_complete && odds[0] && odds[0].is_projected;
  const weeksLeft = is_complete
    ? 0
    : Math.max(0, playoff_week_start - current_week - 1);
  const nByes = playoff_byes != null ? playoff_byes : _calcPlayoffByes(playoff_teams || 6);
  const subtitle = is_complete
    ? `Final standings · ${playoff_teams} playoff teams`
    : isProjected
      ? `Preseason projection · ${playoff_teams} playoff teams · ${(odds[0].n_sims || 10000).toLocaleString()} simulations`
      : `${weeksLeft} week${weeksLeft !== 1 ? 's' : ''} remaining · ${playoff_teams} playoff teams · ${(odds[0].n_sims || 10000).toLocaleString()} simulations`;

  const showBye = nByes > 0;

  const rows = sorted.map(t => {
    const pct    = t.playoff_pct;
    const barCls = pct >= 70 ? 'po-bar-green' : pct >= 35 ? 'po-bar-yellow' : 'po-bar-red';
    const rec    = `${t.wins}-${t.losses}${t.ties ? '-' + t.ties : ''}`;

    let oddsCell;
    if (is_complete) {
      oddsCell = pct === 100
        ? '<span class="po-made">Made Playoffs</span>'
        : '<span class="po-missed">Missed</span>';
    } else {
      oddsCell =
        `<div class="po-bar-wrap"><div class="po-bar ${barCls}" style="width:${Math.max(pct, 2)}%"></div></div>` +
        `<span class="po-label">${pct.toFixed(0)}<span class="po-pct-sym">%</span></span>`;
    }

    const byeCell = showBye
      ? `<td class="po-bye">${
          is_complete
            ? (t.bye_pct === 100 ? '✓' : '')
            : (t.bye_pct > 0 ? t.bye_pct.toFixed(0) + '%' : '-')
        }</td>`
      : '';

    const projCell = is_complete
      ? ''
      : `<td class="po-proj">${t.avg_final_wins.toFixed(1)}-${t.avg_final_losses.toFixed(1)}</td>`;

    const simAvgCell = '';

    const _ridAttr = t.roster_id != null ? ` data-roster-id="${t.roster_id}" data-team-name="${t.team_name}"` : '';
    return `<tr class="team-clickable"${_ridAttr}>
      <td class="po-team">${t.team_name}</td>
      <td class="po-rec">${rec}</td>
      <td class="po-odds">${oddsCell}</td>
      ${byeCell}${projCell}${simAvgCell}
    </tr>`;
  }).join('');

  const byeHdr    = showBye ? '<th class="po-bye">Bye</th>' : '';
  const projHdr   = is_complete ? '' : '<th class="po-proj">Proj W-L</th>';
  const simAvgHdr = '';

  return `<div class="po-wrap">
    <p class="po-subtitle">${subtitle}</p>
    <table class="po-table">
      <thead><tr>
        <th class="po-team">Team</th>
        <th class="po-rec">Record</th>
        <th class="po-odds">Playoff %</th>
        ${byeHdr}${projHdr}${simAvgHdr}
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function initTeamTabs(root = document) {
  const tabs = Array.from(root.querySelectorAll(".team-tab"));
  const panels = Array.from(root.querySelectorAll(".team-panel"));
  if (!tabs.length) return;

  tabs.forEach(tab => {
    bindOnce(tab, "teamTabClick", "click", () => {
      const id = tab.getAttribute("data-team-id");
      tabs.forEach(t => t.classList.toggle("active", t === tab));
      panels.forEach(p => p.classList.toggle("active", p.getAttribute("data-team-id") === id));
    });
  });
}

function initStandingsSort(root = document) {
  const marker = root.querySelector('[data-page="standings"]');
  if (!marker) return;

  const tbl = root.getElementById("stats") || root.querySelector("#stats");
  if (!tbl || !tbl.tHead || !tbl.tBodies?.length) return;

  if (tbl.__sortInited) return;
  tbl.__sortInited = true;

  const NUMERIC_COLS = new Set([2, 3, 4, 5, 6, 7, 8]);

  const getVal = (cell, idx) => {
    const t = (cell?.textContent || "").trim();
    if (!NUMERIC_COLS.has(idx)) return t.toLowerCase();
    const n = parseFloat(t.replace(/,/g, ""));
    return Number.isFinite(n) ? n : -Infinity;
  };

  let sortSpec = [{ col: 2, dir: -1 }, { col: 3, dir: -1 }];

  const applySort = () => {
    const tbody = tbl.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll("tr"));

    rows.sort((a, b) => {
      for (const { col, dir } of sortSpec) {
        const A = getVal(a.children[col], col);
        const B = getVal(b.children[col], col);
        if (A < B) return -1 * dir;
        if (A > B) return 1 * dir;
      }
      return 0;
    });

    tbody.replaceChildren(...rows);

    tbl.querySelectorAll("th").forEach(th =>
      th.classList.remove("sorted-asc", "sorted-desc", "sorted-secondary")
    );

    if (sortSpec.length) {
      const primary = sortSpec[0];
      const th = tbl.tHead.rows[0].children[primary.col];
      if (th) th.classList.add(primary.dir === 1 ? "sorted-asc" : "sorted-desc");

      for (let i = 1; i < sortSpec.length; i++) {
        const th2 = tbl.tHead.rows[0].children[sortSpec[i].col];
        if (th2) th2.classList.add("sorted-secondary");
      }
    }
  };

  const toggleSort = (col, additive = false) => {
    if (!additive) {
      const existing = sortSpec.find(s => s.col === col);
      sortSpec = existing ? [existing] : [];
    }

    const i = sortSpec.findIndex(s => s.col === col);
    if (i === -1) {
      sortSpec.unshift({ col, dir: NUMERIC_COLS.has(col) ? -1 : 1 });
    } else {
      const cur = sortSpec[i];
      if (cur.dir === -1) cur.dir = 1;
      else sortSpec.splice(i, 1);

      const j = sortSpec.findIndex(s => s.col === col);
      if (j > 0) sortSpec.unshift(sortSpec.splice(j, 1)[0]);
    }

    applySort();
  };

  bindOnce(tbl.tHead, "standingsHeadClick", "click", e => {
    const th = e.target.closest("th");
    if (!th) return;
    const colAttr = th.getAttribute("data-col");
    if (!colAttr) return;
    const col = parseInt(colAttr, 10);
    if (!Number.isNaN(col)) toggleSort(col, e.shiftKey);
  });

  applySort();
}

// ------------------------------------------------------------
// Matchup Carousel (delegated, debounced resize)
// ------------------------------------------------------------

function getCarouselState(card) {
  const track = card.querySelector(".m-track");
  if (!track) return { track: null, slides: [], width: 1, idx: 0 };

  const slides = track.querySelectorAll(".m-slide");
  const viewport = card.querySelector(".m-carousel");
  const width = viewport?.clientWidth || track.clientWidth || 1;

  const idx = width > 0 ? Math.round(track.scrollLeft / width) : 0;
  return { track, slides, width, idx };
}

function scrollToIndex(card, newIdx) {
  const { track, slides, width } = getCarouselState(card);
  if (!track || !slides.length) return;

  const maxIdx = slides.length - 1;
  const clamped = Math.max(0, Math.min(maxIdx, newIdx));

  track.scrollTo({ left: clamped * width, behavior: "smooth" });

  const prevBtn = card.querySelector(".m-btn-prev");
  const nextBtn = card.querySelector(".m-btn-next");
  if (prevBtn) prevBtn.disabled = clamped === 0;
  if (nextBtn) nextBtn.disabled = clamped === maxIdx;
}

function initAllCarousels(scope = document) {
  scope.querySelectorAll(".matchup-carousel").forEach(card => {
    const { idx } = getCarouselState(card);
    scrollToIndex(card, idx || 0);
  });
}

function bindGlobalCarouselHandlersOnce() {
  if (window.__BR_INIT_FLAGS__.globalsBound) return;
  window.__BR_INIT_FLAGS__.globalsBound = true;

  document.addEventListener("click", evt => {
    const prev = evt.target.closest(".m-btn-prev");
    const next = evt.target.closest(".m-btn-next");
    if (!prev && !next) return;

    const card = (prev || next).closest(".matchup-carousel");
    if (!card) return;

    const { idx } = getCarouselState(card);
    scrollToIndex(card, idx + (prev ? -1 : 1));
  });

  if (!window.__BR_INIT_FLAGS__.resizeBound) {
    window.__BR_INIT_FLAGS__.resizeBound = true;
    let t;
    window.addEventListener("resize", () => {
      clearTimeout(t);
      t = setTimeout(() => initAllCarousels(document), 150);
    });
  }
}

window.resetMatchupCarousels = function (root) {
  initAllCarousels(root || document);
};

// ------------------------------------------------------------
// Plotly Fixes (hover offset + axis title placement)
// ------------------------------------------------------------

function _fixOnePlotly(gd) {
  const P = window.Plotly;
  if (!P || !gd) return;

  try {
    P.Plots.resize(gd);
  } catch (e) {}

  try {
    const relayoutPatch = {
      "xaxis.automargin": true,
      "yaxis.automargin": true,
      "xaxis.title.standoff": 20,
      "yaxis.title.standoff": 20,
      "margin.l": 60,
      "margin.r": 20,
      "margin.t": 40,
      "margin.b": 60,
    };
    P.relayout(gd, relayoutPatch);
  } catch (e) {}

  try {
    P.redraw(gd);
  } catch (e) {}
}

window.resizeAllPlotly = function resizeAllPlotly(root = document) {
  const P = window.Plotly;
  if (!P) return;

  const run = () => {
    root.querySelectorAll(".js-plotly-plot").forEach(gd => _fixOnePlotly(gd));
  };

  requestAnimationFrame(() => {
    run();
    requestAnimationFrame(() => {
      run();
      setTimeout(run, 120);
      setTimeout(run, 300);
    });
  });
};

(function bindPlotlyViewportHooksOnce() {
  if (window.__BR_INIT_FLAGS__.plotlyViewportBound) return;
  window.__BR_INIT_FLAGS__.plotlyViewportBound = true;

  const handler = () => {
    const root = document.getElementById("page-root") || document;
    if (root.querySelector('[data-page="graphs"]')) {
      window.resizeAllPlotly(root);
    }
  };

  window.addEventListener("resize", handler);
  window.addEventListener("orientationchange", handler);

  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", handler);
    window.visualViewport.addEventListener("scroll", handler);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") handler();
  });
})();

// ------------------------------------------------------------
// Trade Page
// ------------------------------------------------------------

function tradePageExists(root = document) {
  return !!root.querySelector("#leagueIdInput");
}

window.initTradePage = function initTradePage(root = document) {
  const leagueInput = root.querySelector("#leagueIdInput");
  if (!leagueInput) return;

  const host = root.querySelector(".otc-layout") || root;
  if (host.__tradeInitDone) return;
  host.__tradeInitDone = true;

  let allPlayers = [];
  let activePosFilter = "ALL";
  let playerDeltas = {}; // Cache of player_id -> 7-day delta
  let playerIndicators = { rookies: [], breakouts: [] }; // Rookie and breakout flags

  const state = {
    sideAPlayers: [],
    sideBPlayers: [],
    sideAPicks: [],
    sideBPicks: [],
  };

  async function loadPlayerDeltas() {
    try {
      const leagueType = getLeagueType();
      const leagueSize = getLeagueSize();
      const res = await fetch(`/api/player-deltas?days=7&league_type=${leagueType}&league_size=${leagueSize}`, { cache: "no-store" });
      if (!res.ok) return;
      playerDeltas = await res.json();
    } catch (err) {
      console.error("[trade] Failed to load player deltas:", err);
    }
  }

  async function loadPlayerIndicators() {
    try {
      const leagueType = getLeagueType();
      const leagueSize = getLeagueSize();
      const res = await fetch(`/api/player-indicators?league_type=${leagueType}&league_size=${leagueSize}`, { cache: "no-store" });
      if (!res.ok) return;
      playerIndicators = await res.json();
    } catch (err) {
      console.error("[trade] Failed to load player indicators:", err);
    }
  }

  function isRookie(playerId) {
    return playerIndicators.rookies && playerIndicators.rookies.includes(String(playerId));
  }

  function isBreakout(playerId) {
    return playerIndicators.breakouts && playerIndicators.breakouts.includes(String(playerId));
  }

  function isElite(playerId) {
    return playerIndicators.elites && playerIndicators.elites.includes(String(playerId));
  }

  function isProspect(playerId) {
    return playerIndicators.prospects && playerIndicators.prospects.includes(String(playerId));
  }

  function getStorageKey() {
    const leagueId = leagueInput?.value || "";
    const season = root.querySelector("#seasonInput")?.value || "";
    return `tradeCalc_${leagueId}_${season}`;
  }

  function saveState() {
    try {
      const storageKey = getStorageKey();
      const stateToSave = {
        sideAPlayers: state.sideAPlayers.map(p => ({
          id: p.id,
          name: p.name,
          position: p.position,
          team: p.team,
          value: p.value,
          sf_value: p.sf_value,
          pos_rank_label: p.pos_rank_label,
          sf_pos_rank_label: p.sf_pos_rank_label
        })),
        sideBPlayers: state.sideBPlayers.map(p => ({
          id: p.id,
          name: p.name,
          position: p.position,
          team: p.team,
          value: p.value,
          sf_value: p.sf_value,
          pos_rank_label: p.pos_rank_label,
          sf_pos_rank_label: p.sf_pos_rank_label
        })),
        sideAPicks: state.sideAPicks.map(p => ({
          id: p.id,
          display: p.display
        })),
        sideBPicks: state.sideBPicks.map(p => ({
          id: p.id,
          display: p.display
        }))
      };
      localStorage.setItem(storageKey, JSON.stringify(stateToSave));
    } catch (err) {
      console.error("[Trade Calc] Failed to save state:", err);
    }
  }

  function loadState() {
    try {
      const storageKey = getStorageKey();
      const saved = localStorage.getItem(storageKey);
      if (!saved) return;

      const parsed = JSON.parse(saved);
      if (parsed.sideAPlayers) state.sideAPlayers = parsed.sideAPlayers;
      if (parsed.sideBPlayers) state.sideBPlayers = parsed.sideBPlayers;
      if (parsed.sideAPicks) state.sideAPicks = parsed.sideAPicks;
      if (parsed.sideBPicks) state.sideBPicks = parsed.sideBPicks;

      renderChips("A");
      renderChips("B");
    } catch (err) {
      console.error("[Trade Calc] Failed to load state:", err);
    }
  }

  function encodeTradeToURL() {
    const tradeData = {
      a: state.sideAPlayers.map(p => p.id).join(','),
      b: state.sideBPlayers.map(p => p.id).join(','),
      ap: state.sideAPicks.map(p => p.id).join(','),
      bp: state.sideBPicks.map(p => p.id).join(',')
    };

    const params = new URLSearchParams();
    if (tradeData.a) params.set('a', tradeData.a);
    if (tradeData.b) params.set('b', tradeData.b);
    if (tradeData.ap) params.set('ap', tradeData.ap);
    if (tradeData.bp) params.set('bp', tradeData.bp);

    const url = new URL(window.location.href);
    url.search = params.toString();
    return url.toString();
  }

  function loadTradeFromURL() {
    try {
      const params = new URLSearchParams(window.location.search);
      const aIds = params.get('a')?.split(',').filter(Boolean) || [];
      const bIds = params.get('b')?.split(',').filter(Boolean) || [];
      const apIds = params.get('ap')?.split(',').filter(Boolean) || [];
      const bpIds = params.get('bp')?.split(',').filter(Boolean) || [];

      if (aIds.length === 0 && bIds.length === 0) return false;

      // Load players from allPlayers
      state.sideAPlayers = aIds.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);
      state.sideBPlayers = bIds.map(id => allPlayers.find(p => p.id === id)).filter(Boolean);

      // Load picks
      state.sideAPicks = apIds.map(id => ({ id, display: formatPickId(id) }));
      state.sideBPicks = bpIds.map(id => ({ id, display: formatPickId(id) }));

      renderChips("A");
      renderChips("B");
      recomputeTrade();
      return true;
    } catch (err) {
      console.error("[Trade Calc] Failed to load trade from URL:", err);
      return false;
    }
  }

  function shareTradeToClipboard() {
    const url = encodeTradeToURL();
    navigator.clipboard.writeText(url).then(() => {
      // Show success feedback
      const btn = root.querySelector("#shareTradeBtn");
      if (btn) {
        const originalHTML = btn.innerHTML;
        btn.innerHTML = '<svg class="otc-share-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        btn.classList.add("otc-share-btn-success");
        setTimeout(() => {
          btn.innerHTML = originalHTML;
          btn.classList.remove("otc-share-btn-success");
        }, 2000);
      }
    }).catch(err => {
      console.error("Failed to copy to clipboard:", err);
      alert("Failed to copy link. Please try again.");
    });
  }

  function formatValue(v) {
    const num = Number(v) || 0;
    return num.toFixed(1);
  }

  function formatDelta(v) {
    const num = Number(v) || 0;
    return (num > 0 ? "+" : "") + num.toFixed(1);
  }

  function buildOverallRankMap(players) {
    const sorted = [...players].sort((a, b) => {
      return getPlayerValue(b) - getPlayerValue(a);
    });

    const rankMap = new Map();
    sorted.forEach((p, idx) => {
      if (p && p.id != null) rankMap.set(String(p.id), idx + 1);
    });
    return rankMap;
  }

  function buildMetaBits(p) {
    const metaBits = [];
    const pos = String(p.position || p.pos || "").toUpperCase();
    const leagueType = getLeagueType();

    if (pos === "PICK") {
      metaBits.push("PICK");
    } else if (pos) {
      // Use SF position rank when Superflex is selected
      if (leagueType === "sf" && p.sf_pos_rank_label) {
        metaBits.push(String(p.sf_pos_rank_label).toUpperCase());
      } else if (p.pos_rank_label) {
        metaBits.push(String(p.pos_rank_label).toUpperCase());
      } else {
        metaBits.push(pos);
      }
    }

    if (p.team) metaBits.push(p.team);

    // Display age (format to 1 decimal place)
    if (p.age != null && p.age !== "") {
      const ageNum = parseFloat(p.age);
      if (!isNaN(ageNum)) {
        metaBits.push(`${ageNum.toFixed(1)} yrs`);
      }
    }

    return metaBits;
  }

  function getSidePlayers(side) {
    return side === "A" ? state.sideAPlayers : state.sideBPlayers;
  }

  function getSidePicks(side) {
    return side === "A" ? state.sideAPicks : state.sideBPicks;
  }

  function getSideEmptyState(side) {
    return root.querySelector(side === "A" ? "#sideAEmptyState" : "#sideBEmptyState");
  }

  function syncEmptyState(side) {
    const emptyEl = getSideEmptyState(side);
    if (!emptyEl) return;

    const players = getSidePlayers(side);
    const picks = getSidePicks(side);
    emptyEl.style.display = players.length || picks.length ? "none" : "";
  }

  function buildPlayerValueRow(p, overallRank) {
    const row = document.createElement("div");
    row.className = "otc-value-row";

    // Make row clickable - use only onclick, not data-player-id, to avoid double-fire
    // from the global delegated handler in initGlobalPlayerModals
    if (p.id) {
      row.style.cursor = "pointer";
      row.onclick = (e) => {
        e.stopPropagation();
        const _drafted = p.is_rookie && p.is_rookie != 'False' && p.team && p.team !== 'FA';
        if (p.is_rookie && p.is_rookie != 'False' && !_drafted) {
          if (typeof rkOpenModal === 'function') {
            rkOpenModal(p);
          } else {
            openProspectModal(p.id, p.name || "Unknown");
          }
        } else {
          if (typeof openPlayerModal === 'function') {
            openPlayerModal(p.id, p.name || "Unknown");
          } else {
            console.error('openPlayerModal function not found');
          }
        }
      };
    }

    const rankWrap = document.createElement("div");
    rankWrap.className = "otc-value-rank";
    rankWrap.textContent = overallRank ? "#" + overallRank : "-";

    const mainWrap = document.createElement("div");
    mainWrap.className = "otc-value-main";

    const topLine = document.createElement("div");
    topLine.className = "otc-value-topline";

    const nameSpan = document.createElement("div");
    nameSpan.className = "otc-value-name player-clickable";
    nameSpan.textContent = p.name || "Unknown";

    const valueSpan = document.createElement("div");
    valueSpan.className = "otc-value-score";
    valueSpan.textContent = formatValue(getPlayerValue(p));

    topLine.appendChild(nameSpan);
    topLine.appendChild(valueSpan);

    const metaSpan = document.createElement("div");
    metaSpan.className = "otc-value-sub";

    const metaBits = buildMetaBits(p);
    if (isRookie(p.id)) {
      metaBits.push('<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> PROSPECT</span>');
    } else if (p.is_rookie) {
      metaBits.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>');
    }
    if (!p.is_rookie && isProspect(p.id)) {
      metaBits.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>');
    }
    if (isBreakout(p.id)) {
      metaBits.push('<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>');
    }

    metaSpan.innerHTML = metaBits.join(" • ");

    mainWrap.appendChild(topLine);
    mainWrap.appendChild(metaSpan);

    row.appendChild(rankWrap);
    row.appendChild(mainWrap);

    return row;
  }

  function buildDropdownItem(p, overallRank) {
    const item = document.createElement("div");
    item.className = "dropdown-item otc-dropdown-item";
    item.setAttribute("data-position", p.position || "");

    const left = document.createElement("div");
    left.className = "otc-dropdown-left";

    const top = document.createElement("div");
    top.className = "otc-dropdown-top";

    const rank = document.createElement("span");
    rank.className = "otc-dropdown-rank-inline";
    rank.textContent = overallRank ? "#" + overallRank : "";

    const name = document.createElement("span");
    name.className = "otc-dropdown-name";
    name.textContent = p.name || "Unknown";

    top.appendChild(rank);
    top.appendChild(name);

    const sub = document.createElement("div");
    sub.className = "otc-dropdown-sub";

    const metaBits = buildMetaBits(p);
    if (isRookie(p.id)) {
      metaBits.push('<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> PROSPECT</span>');
    } else if (p.is_rookie) {
      metaBits.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>');
    }
    if (!p.is_rookie && isProspect(p.id)) {
      metaBits.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>');
    }
    if (isBreakout(p.id)) {
      metaBits.push('<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>');
    }
    if (isElite(p.id)) {
        metaBits.push('<span class="player-badge player-badge-elite"><i class="fa-solid fa-star" aria-hidden="true"></i></span>');
    }


    sub.innerHTML = metaBits.join(" • ");

    left.appendChild(top);
    left.appendChild(sub);

    const value = document.createElement("div");
    value.className = "otc-dropdown-value";
    value.textContent = formatValue(getPlayerValue(p));

    item.appendChild(left);
    item.appendChild(value);

    return item;
  }

  function buildMoverRow(p, directionClass) {
    const row = document.createElement("div");
    row.className = "otc-mini-row " + directionClass;

    const name = document.createElement("div");
    name.className = "otc-mini-name player-clickable";
    name.textContent = p.name || "Unknown";

    // Make player clickable
    if (p.player_id) {
      name.dataset.playerId = p.player_id;
      name.dataset.playerName = p.name || "Unknown";
    }

    const delta = document.createElement("div");
    delta.className = "otc-mini-delta";
    delta.textContent = formatDelta(p.delta);

    row.appendChild(name);
    row.appendChild(delta);

    return row;
  }

  function renderMovers(data) {
    const risersEl = root.querySelector("#otcRisersList");
    const fallersEl = root.querySelector("#otcFallersList");
    if (!risersEl || !fallersEl) return;

    risersEl.innerHTML = "";
    fallersEl.innerHTML = "";

    const risers = Array.isArray(data?.risers) ? data.risers : [];
    const fallers = Array.isArray(data?.fallers) ? data.fallers : [];

    if (!risers.length) {
      risersEl.innerHTML = '<div class="otc-movers-empty">No risers yet.</div>';
    } else {
      risers.forEach(p => risersEl.appendChild(buildMoverRow(p, "up")));
    }

    if (!fallers.length) {
      fallersEl.innerHTML = '<div class="otc-movers-empty">No fallers yet.</div>';
    } else {
      fallers.forEach(p => fallersEl.appendChild(buildMoverRow(p, "down")));
    }
  }

  let loadMoversTimeout = null;
  let currentMoversDays = 7; // Default to 7 days

  // Function to change movers days period
  window.changeMoversDays = function(days) {
    currentMoversDays = days;

    // Update active button
    const dayFilters = document.querySelectorAll('.otc-day-filter');
    dayFilters.forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.getAttribute('data-days')) === days);
    });

    // Reload movers with new days
    loadTopMovers(true);
  };

  async function loadTopMovers(immediate = false) {
    // Debounce: wait 300ms before loading unless immediate
    if (!immediate) {
      clearTimeout(loadMoversTimeout);
      loadMoversTimeout = setTimeout(() => loadTopMovers(true), 300);
      return;
    }

    const risersEl = root.querySelector("#otcRisersList");
    const fallersEl = root.querySelector("#otcFallersList");
    const moversPanel = root.querySelector(".otc-movers-panel");

    // Show loading state
    if (moversPanel) {
      moversPanel.classList.add("otc-movers-loading");
    }
    if (risersEl) risersEl.innerHTML = '<div class="otc-movers-empty"><div class="loading-spinner"></div></div>';
    if (fallersEl) fallersEl.innerHTML = '<div class="otc-movers-empty"><div class="loading-spinner"></div></div>';

    try {
      const leagueType = getLeagueType();
      const leagueSize = getLeagueSize();
      const res = await fetch(`/api/value-movers?days=${currentMoversDays}&limit=5&league_type=${leagueType}&league_size=${leagueSize}`, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load movers.");

      const data = await res.json();
      const usedDays = data?.used_days;
      const latestDate = data?.latest_date;

      const sub = root.querySelector("#moversSub");
      if (sub && usedDays) {
        const leagueLabel = leagueType === "sf" ? "SF" : "1QB";
        const sizeLabel = leagueSize === 10 ? "" : ` ${leagueSize}-team`;

        // Add freshness indicator
        let freshnessText = "";
        if (latestDate) {
          const dataDate = new Date(latestDate);
          const now = new Date();
          const hoursDiff = Math.floor((now - dataDate) / (1000 * 60 * 60));
          if (hoursDiff < 1) {
            freshnessText = " • Updated recently";
          } else if (hoursDiff < 24) {
            freshnessText = ` • Updated ${hoursDiff}h ago`;
          } else {
            const daysDiff = Math.floor(hoursDiff / 24);
            freshnessText = ` • Updated ${daysDiff}d ago`;
          }
        }

        sub.textContent = `Biggest ${usedDays}-day changes in ${leagueLabel}${sizeLabel} BR value${freshnessText}`;
      }

      renderMovers(data);

      // Visual feedback - pulse animation
      if (moversPanel) {
        moversPanel.classList.remove("otc-movers-loading");
        moversPanel.classList.add("otc-movers-updated");
        setTimeout(() => moversPanel.classList.remove("otc-movers-updated"), 600);
      }
    } catch (err) {
      console.error("[trade] movers error:", err);
      if (risersEl) risersEl.innerHTML = '<div class="otc-movers-empty">Unable to load risers.</div>';
      if (fallersEl) fallersEl.innerHTML = '<div class="otc-movers-empty">Unable to load fallers.</div>';
      if (moversPanel) {
        moversPanel.classList.remove("otc-movers-loading");
      }
    }
  }

  // Load offseason breakout candidates
  async function loadBreakouts() {
    const breakoutsEl = root.querySelector("#otcBreakoutsList");
    const moversPanel = root.querySelector(".otc-movers-panel");
    if (!breakoutsEl) return;

    // Show loading state
    if (moversPanel) {
      moversPanel.classList.add("otc-movers-loading");
    }
    breakoutsEl.innerHTML = '<div class="otc-movers-empty"><div class="loading-spinner"></div></div>';

    try {
      const leagueType = getLeagueType();
      const nflState = await fetch('/api/nfl-state').then(r => r.json()).catch(() => ({}));
      const currentSeason = nflState.season || new Date().getFullYear();
      
      const res = await fetch(`/api/offseason-breakout-candidates?season=${currentSeason}&min_score=25`, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load breakouts.");

      let candidates = await res.json();

      // No additional filtering - show the same results as the main breakouts page
      if (!candidates || candidates.length === 0) {
        breakoutsEl.innerHTML = `
          <div class="otc-movers-empty" style="padding: 16px; text-align: center;">
            <div style="font-size: 13px; color: #64748b; line-height: 1.5;">
              No breakout candidates available yet.<br>
              <span style="font-size: 11px;">Top 5 breakout candidates will appear here once offseason roster changes are tracked.</span>
            </div>
          </div>
        `;
        if (moversPanel) moversPanel.classList.remove("otc-movers-loading");
        return;
      }

      // Show only top 5 candidates by breakout score
      candidates.sort((a, b) => (b.breakout_score || 0) - (a.breakout_score || 0));
      const topCandidates = candidates.slice(0, 5);

      breakoutsEl.innerHTML = "";
      topCandidates.forEach((c, idx) => {
        const row = document.createElement("div");
        row.className = "otc-mini-row otc-breakout-row";
        row.style.animationDelay = `${idx * 50}ms`;

        const name = document.createElement("div");
        name.className = "otc-mini-name";

        const playerName = document.createElement("span");
        playerName.className = "otc-player-name player-clickable";
        playerName.textContent = c.name || "Unknown";
        playerName.dataset.playerId = c.player_id;
        playerName.dataset.playerName = c.name || "Unknown";

        const meta = document.createElement("span");
        meta.className = "otc-player-meta";

        // Build meta string: position rank (RB13), team, age
        const metaParts = [];
        // Show position rank label (e.g., "RB13") instead of just position
        if (c.pos_rank_label) {
          metaParts.push(c.pos_rank_label);
        } else if (c.position && c.pos_rank) {
          metaParts.push(`${c.position}${c.pos_rank}`);
        } else if (c.position) {
          metaParts.push(c.position);
        }
        if (c.team) metaParts.push(c.team);
        const cAgeNum = parseFloat(c.age);
        if (!isNaN(cAgeNum)) metaParts.push(`${cAgeNum.toFixed(1)} yrs`);
        meta.textContent = metaParts.join(" · ");

        // Add opportunity badge
        const badge = document.createElement("div");
        badge.className = "otc-breakout-badge";
        badge.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>`;

        name.appendChild(playerName);
        name.appendChild(meta);

        row.appendChild(name);
        row.appendChild(badge);

        breakoutsEl.appendChild(row);
      });

      if (moversPanel) {
        moversPanel.classList.remove("otc-movers-loading");
        moversPanel.classList.add("otc-movers-updated");
        setTimeout(() => moversPanel.classList.remove("otc-movers-updated"), 600);
      }
    } catch (err) {
      console.error("[trade] breakouts error:", err);
      breakoutsEl.innerHTML = `
        <div class="otc-movers-empty" style="padding: 16px; text-align: center;">
          <div style="font-size: 13px; color: #ef4444; line-height: 1.5;">
            Unable to load breakout candidates.<br>
            <span style="font-size: 11px; color: #64748b;">Please try refreshing the page.</span>
          </div>
        </div>
      `;
      if (moversPanel) {
        moversPanel.classList.remove("otc-movers-loading");
      }
    }
  }

  // Handle tab switching
  function initMoversBreakoutsTabs() {
    const tabButtons = root.querySelectorAll(".otc-mini-tab");
    const moversContent = root.querySelector("#moversTabContent");
    const breakoutsContent = root.querySelector("#breakoutsTabContent");
    const targetsContent = root.querySelector("#targetsTabContent");
    const moversSub = root.querySelector("#moversSub");
    const dayFilters = root.querySelector(".otc-day-filters");
    const lockIcon = root.querySelector("#targetsLockIcon");
    const hasPremium = (root.querySelector("#otcHasPremium")?.value || "false") === "true";

    if (hasPremium && lockIcon) lockIcon.style.display = "none";

    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;

        if (tab === "targets" && !hasPremium) {
          if (typeof showPaywall === "function") showPaywall("trade-suggestions");
          return;
        }

        tabButtons.forEach(b => b.classList.remove("is-active"));
        btn.classList.add("is-active");

        moversContent?.classList.remove("is-active");
        breakoutsContent?.classList.remove("is-active");
        targetsContent?.classList.remove("is-active");

        if (tab === "movers") {
          moversContent?.classList.add("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            const leagueType = getLeagueType();
            const leagueSize = getLeagueSize();
            const leagueLabel = leagueType === "sf" ? "SF" : "1QB";
            const sizeLabel = leagueSize === 10 ? "" : ` ${leagueSize}-team`;
            moversSub.textContent = `Biggest 7-day changes in ${leagueLabel}${sizeLabel} BR value`;
          }
          if (dayFilters) dayFilters.style.display = "flex";

        } else if (tab === "breakouts") {
          breakoutsContent?.classList.add("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            moversSub.textContent = "Top 5 breakouts from Breakout Engine ";
          }
          if (dayFilters) dayFilters.style.display = "none";
          if (breakoutsContent && !breakoutsContent.dataset.loaded) {
            breakoutsContent.dataset.loaded = "true";
            loadBreakouts();
          }

        } else if (tab === "targets") {
          targetsContent?.classList.add("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            moversSub.textContent = "Players to pursue based on your roster gaps";
          }
          if (dayFilters) dayFilters.style.display = "none";
          if (targetsContent) {
            loadTradeTargets();
          }
        }
      });
    });
  }

  function normalizePlayerRow(p) {
    const pos = String(p.position || p.pos || "").toUpperCase();
    return {
      id: String(p.id),
      name: p.name || p.full_name || "Unknown",
      team: p.team || "",
      position: pos,
      age: p.age ?? null,
      value: Number(p.value || 0),
      sf_value: Number(p.sf_value || p.value || 0),
      value_8: Number(p.value_8 || p.value || 0),
      value_12: Number(p.value_12 || p.value || 0),
      value_14: Number(p.value_14 || p.value || 0),
      sf_value_8: Number(p.sf_value_8 || p.sf_value || p.value || 0),
      sf_value_12: Number(p.sf_value_12 || p.sf_value || p.value || 0),
      sf_value_14: Number(p.sf_value_14 || p.sf_value || p.value || 0),
      redraft_value_1qb: p.redraft_value_1qb != null ? Number(p.redraft_value_1qb) : null,
      redraft_value_sf: p.redraft_value_sf != null ? Number(p.redraft_value_sf) : null,
      pos_rank_label: p.pos_rank_label || "",
      sf_pos_rank_label: p.sf_pos_rank_label || "",
      is_rookie: p.is_rookie === true,
      search_name: p.search_name || "",
    };
  }

  let _tierThresholds = {};

  function _getOtcTier(player) {
    const lt = getLeagueType();
    const sz = String(getLeagueSize());
    const tbl = (_tierThresholds[lt] || {})[ sz] || (_tierThresholds["1qb"] || {})["10"] || [];
    if (!tbl.length) return null;
    const val = getPlayerValue(player);
    for (let i = 0; i < tbl.length; i++) { if (val >= tbl[i]) return i + 1; }
    return tbl.length + 1;
  }

  async function ensurePlayersLoaded() {
    if (allPlayers.length > 0) return;

    const errorBox = root.querySelector("#errorBox");

    let lastErr;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (attempt > 0) await new Promise(r => setTimeout(r, attempt * 1000));
      try {
        const res = await fetch("/api/league-players", { cache: "no-store" });
        if (!res.ok) throw new Error("Failed to load players (" + res.status + ").");
        const data = await res.json();
        const rawData = Array.isArray(data) ? data : (Array.isArray(data.players) ? data.players : []);
        if (!Array.isArray(data) && data.tier_thresholds) _tierThresholds = data.tier_thresholds;

        const players = rawData.filter(p => p.position !== "PICK");
        const picks = rawData.filter(p => p.position === "PICK");

        allPlayers = [
          ...players
            .filter(p => p && typeof p === "object" && p.id != null)
            .map(normalizePlayerRow)
            .filter(p => ["QB", "RB", "WR", "TE"].includes(p.position) || p.is_rookie),
          ...picks.map(p => ({ ...p, name: formatPickId(p.id) })),
        ].sort((a, b) => {
          const vb = Number(b.value || 0);
          const va = Number(a.value || 0);
          if (vb !== va) return vb - va;
          return String(a.name || "").localeCompare(String(b.name || ""));
        });

        lastErr = null;
        break;
      } catch (err) {
        lastErr = err;
        console.warn("[trade] ensurePlayersLoaded attempt", attempt + 1, "failed:", err.message);
      }
    }

    if (lastErr) {
      console.error("Error loading data:", lastErr);
      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = lastErr.message || "Failed to load data.";
      }
      return;
    }

    if (errorBox) {
      errorBox.style.display = "none";
      errorBox.textContent = "";
    }

    renderAllPlayersList();
    loadState();
  }

  function renderAllPlayersList() {
    const container = root.querySelector("#allPlayersList");
    if (!container) return;
    container.innerHTML = "";

    if (!allPlayers.length) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "Players will appear here once loaded.";
      container.appendChild(empty);
      return;
    }

    const overallRankMap = buildOverallRankMap(allPlayers);

    const items = allPlayers
      .filter(p => {
        const pos = String(p.position || "").toUpperCase();
        if (activePosFilter === "ALL") return true;
        if (activePosFilter === "ROOKIE") return !!p.is_rookie;
        if (activePosFilter === "PICK") return pos === "PICK";
        return pos === activePosFilter && !p.is_rookie;
      })
      .sort((a, b) => getPlayerValue(b) - getPlayerValue(a));

    function _tcColor(t) {
      const s=['#10b981','#22d3ee','#3b82f6','#818cf8','#a855f7','#d946ef','#f59e0b','#f97316','#ef4444','#94a3b8','#64748b','#475569','#334155','#1e293b','#0f172a'];
      return s[Math.min(t-1,s.length-1)];
    }
    function _tcLabel(t) {
      const l=['Elite','Star','High-End Starter','Starter','Flex','Bench Depth','Deep Bench','Handcuff','Fringe','Speculative'];
      return l[t-1]||('Tier '+t);
    }
    let prevTier = null;

    items.forEach(p => {
      const tier = activePosFilter === "ALL" ? Math.min(_getOtcTier(p) ?? Infinity, 9) || null : null;
      if (tier && tier !== prevTier) {
        const tc = _tcColor(tier);
        const div = document.createElement("div");
        div.className = "otc-tier-divider";
        div.innerHTML =
          `<div class="otc-tier-divider-line" style="background:${tc};"></div>` +
          `<span class="otc-tier-divider-label" style="color:${tc};" title="${_tcLabel(tier)}">T${tier}</span>` +
          `<div class="otc-tier-divider-line" style="background:${tc};"></div>`;
        container.appendChild(div);
      }
      if (tier) prevTier = tier;
      const overallRank = overallRankMap.get(String(p.id));
      container.appendChild(buildPlayerValueRow(p, overallRank));
    });
  }

  function setPosFilter(pos) {
    activePosFilter = pos;

    root.querySelectorAll(".pos-filter").forEach(btn => {
      const p = btn.getAttribute("data-pos") || "ALL";
      btn.classList.toggle("is-active", p === activePosFilter);
    });

    renderAllPlayersList();
  }

  function getLeagueType() {
    const toggle = root.querySelector("#leagueTypeToggle");
    return toggle?.checked ? "sf" : "1qb";
  }

  function getLeagueSize() {
    const sel = root.querySelector("#leagueSizeSelect");
    return parseInt(sel?.value || "10", 10);
  }

  function getScoringFormat() {
    const sel = root.querySelector("#scoringFormatSelect");
    return sel?.value || "ppr";
  }

  // Position multipliers matching the server-side _SCORING_MULTS table
  const SCORING_MULTS = {
    ppr:  { QB: 1.00, RB: 1.00, WR: 1.00, TE: 1.00 },
    half: { QB: 1.00, RB: 1.06, WR: 0.97, TE: 0.94 },
    std:  { QB: 1.00, RB: 1.13, WR: 0.93, TE: 0.87 },
  };

  function getPlayerValue(player) {
    const leagueType = getLeagueType();
    const size = getLeagueSize();
    const fmt = getScoringFormat();
    const mults = SCORING_MULTS[fmt] || SCORING_MULTS.ppr;
    const pos = (player.position || "").toUpperCase();
    const mult = mults[pos] ?? 1.0;

    let base;
    if (leagueType === "sf") {
      const key = size === 10 ? "sf_value" : `sf_value_${size}`;
      base = Number(player[key] ?? player.sf_value ?? player.value ?? 0);
    } else {
      const key = size === 10 ? "value" : `value_${size}`;
      base = Number(player[key] ?? player.value ?? 0);
    }
    return Math.round(base * mult * 10) / 10;
  }

  function isTargetsTabActive() {
    return !!root.querySelector("#targetsTabContent.is-active");
  }

  async function onLeagueTypeChange() {
    // Refresh all value displays
    await Promise.all([loadPlayerDeltas(), loadPlayerIndicators()]);
    renderChips("A");
    renderChips("B");
    recomputeTrade();
    renderAllPlayersList();
    loadTopMovers();
    if (isTargetsTabActive()) loadTradeTargets();
  }

  function syncViewerSideLabels() {
    const viewerSide =
      root.querySelector('input[name="viewerSide"]:checked')?.value || "a";

    const aTag = root.querySelector("#sideAOwnerTag");
    const bTag = root.querySelector("#sideBOwnerTag");
    const hidden = root.querySelector("#viewerSideInput");

    if (hidden) hidden.value = viewerSide;
    if (!aTag || !bTag) return;

    if (viewerSide === "a") {
      aTag.textContent = "Your side";
      aTag.classList.remove("otc-team-owner-tag-muted");
      bTag.textContent = "Other side";
      bTag.classList.add("otc-team-owner-tag-muted");
    } else {
      aTag.textContent = "Other side";
      aTag.classList.add("otc-team-owner-tag-muted");
      bTag.textContent = "Your side";
      bTag.classList.remove("otc-team-owner-tag-muted");
    }
  }

  function renderChips(side) {
    const container = root.querySelector(side === "A" ? "#sideAChips" : "#sideBChips");
    const players = getSidePlayers(side);
    const picks = getSidePicks(side);
    if (!container) return;

    container.innerHTML = "";
    container.className = "otc-selected-list";

    players.forEach((p, idx) => {
      const chip = document.createElement("div");
      chip.className = "otc-chip";
      if (p.id) chip.dataset.chipPid = String(p.id);

      const leftWrap = document.createElement("div");
      leftWrap.className = "otc-chip-main";

      const nameEl = document.createElement("div");
      nameEl.className = "otc-chip-name player-clickable";
      nameEl.textContent = p.name || "Unknown";

      // Make player name clickable - route prospects to prospect modal
      if (p.id) {
        nameEl.style.cursor = 'pointer';
        nameEl.onclick = (e) => {
          e.stopPropagation();
          if (isProspect(p.id)) {
            if (typeof rkOpenModal === 'function') {
              rkOpenModal(p);
            } else {
              openProspectModal(p.id, p.name || 'Unknown');
            }
          } else {
            if (typeof openPlayerModal === 'function') {
              openPlayerModal(p.id, p.name || 'Unknown');
            } else {
              console.error('openPlayerModal function not found');
            }
          }
        };
      }

      const metaEl = document.createElement("div");
      metaEl.className = "otc-chip-meta";

      const metaBits = buildMetaBits(p);

      // Add rookie/breakout/elite/prospect badges
      if (isRookie(p.id)) {
        metaBits.push('<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i></span>');
      } else if (p.is_rookie) {
        metaBits.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i></span>');
      }
      if (isElite(p.id)) {
        metaBits.push('<span class="player-badge player-badge-elite"><i class="fa-solid fa-star" aria-hidden="true"></i></span>');
      }
      if (!p.is_rookie && isProspect(p.id)) {
        metaBits.push('<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i></span>');
      }
      if (isBreakout(p.id)) {
        metaBits.push('<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i></span>');
      }

      metaEl.innerHTML = metaBits.map((bit, i) => {
        if (i === 0) return bit;
        if (p.team && bit === p.team) return `<span class="otc-chip-team"> • ${bit}</span>`;
        return ' • ' + bit;
      }).join('');

      leftWrap.appendChild(nameEl);
      leftWrap.appendChild(metaEl);

      const rightWrap = document.createElement("div");
      rightWrap.className = "otc-chip-value-wrap";

      const valueEl = document.createElement("span");
      valueEl.className = "otc-chip-value";
      valueEl.textContent = formatValue(getPlayerValue(p));

      // Add delta indicator if available
      const delta = p.delta || p.recent_delta || playerDeltas[p.id];
      if (delta && Math.abs(delta) >= 1) {
        const deltaEl = document.createElement("span");
        deltaEl.className = delta > 0 ? "otc-chip-delta otc-chip-delta-positive" : "otc-chip-delta otc-chip-delta-negative";
        deltaEl.textContent = delta > 0 ? `+${Math.round(delta)}` : Math.round(delta);
        rightWrap.appendChild(deltaEl);
      }

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "otc-chip-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        players.splice(idx, 1);
        saveState();
        renderChips(side);
      });

      rightWrap.appendChild(valueEl);
      rightWrap.appendChild(removeBtn);

      chip.appendChild(leftWrap);
      chip.appendChild(rightWrap);
      container.appendChild(chip);
    });

    picks.forEach((pk, idx) => {
      const chip = document.createElement("div");
      chip.className = "otc-chip otc-chip-pick";

      const leftWrap = document.createElement("div");
      leftWrap.className = "otc-chip-main";

      const nameEl = document.createElement("div");
      nameEl.className = "otc-chip-name";
      nameEl.textContent = pk.display || pk.id || "Pick";

      const metaEl = document.createElement("div");
      metaEl.className = "otc-chip-meta";
      metaEl.textContent = "Rookie pick";

      leftWrap.appendChild(nameEl);
      leftWrap.appendChild(metaEl);

      const rightWrap = document.createElement("div");
      rightWrap.className = "otc-chip-value-wrap";

      const valueEl = document.createElement("span");
      valueEl.className = "otc-chip-value";
      const pickData = allPlayers.find(p => p.id === pk.id && p.position === "PICK");
      valueEl.textContent = formatValue(pickData ? getPlayerValue(pickData) : 0);

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "otc-chip-remove";
      removeBtn.textContent = "×";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        picks.splice(idx, 1);
        saveState();
        renderChips(side);
      });

      rightWrap.appendChild(valueEl);
      rightWrap.appendChild(removeBtn);

      chip.appendChild(leftWrap);
      chip.appendChild(rightWrap);
      container.appendChild(chip);
    });

    syncEmptyState(side);
    recomputeTrade();
  }

  function formatPickId(id) {
    const parts = String(id).split("_");
    if (parts.length < 3) return String(id).replaceAll("_", " ");
    const year = parts[0];
    const round = parseInt(parts[1], 10);
    const third = parts.slice(2).join("_");
    const suffix = { 1: "st", 2: "nd", 3: "rd" }[round] || "th";
    const bucketLabel = { early: "Early", mid: "Mid", late: "Late" }[third];
    if (bucketLabel) return `${year} ${round}${suffix} (${bucketLabel})`;
    const slot = parseInt(third, 10);
    if (!isNaN(slot)) return `${year} ${round}.${String(slot).padStart(2, "0")}`;
    return String(id).replaceAll("_", " ");
  }

  function openPickPrompt(side) {
    const raw = window.prompt(
      "Enter a pick in this format:\n2026_1_04\nor\n2026_1_early"
    );
    if (!raw) return;

    const cleaned = String(raw).trim();
    if (!cleaned) return;

    const picks = getSidePicks(side);
    const isBucket = /_(early|mid|late)$/i.test(cleaned);
    if (!isBucket && picks.find(p => p.id === cleaned)) return;

    picks.push({
      id: cleaned,
      display: formatPickId(cleaned),
    });

    saveState();
    renderChips(side);
  }

  const _TIER_COLORS = ['', '#10b981', '#22d3ee', '#3b82f6', '#8b5cf6', '#a855f7', '#f59e0b', '#f97316', '#94a3b8', '#64748b'];
  const _TIER_LABELS = ['', 'Elite', 'Star', 'High-End Starter', 'Starter', 'Flex', 'Bench', 'Deep Bench', 'Handcuff', 'Fringe'];

  function _applyTierBadges(data) {
    ['a', 'b'].forEach(side => {
      const sideData = data['side_' + side];
      if (!sideData) return;
      const bd = sideData.breakdown || [];
      const byId = {};
      bd.forEach(item => { if (item.id) byId[String(item.id)] = item; });

      const container = root.querySelector(side === 'a' ? '#sideAChips' : '#sideBChips');
      if (!container) return;

      // Tag each player chip with its individual tier badge (tier is computed server-side
      // from live value-table gaps, so it reflects the current rankings)
      container.querySelectorAll('.otc-chip[data-chip-pid]').forEach(chip => {
        const pid = chip.dataset.chipPid;
        const item = byId[pid];
        if (!item) return;

        const playerObj = allPlayers.find(p => String(p.id) === pid);
        const tier = playerObj ? _getOtcTier(playerObj) : item.tier;
        const tc = _TIER_COLORS[tier] || '#6b7280';
        const label = _TIER_LABELS[tier] || ('Tier ' + tier);

        let badge = chip.querySelector('.otc-tier-badge');
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'otc-tier-badge';
          const metaEl = chip.querySelector('.otc-chip-meta');
          if (metaEl) metaEl.appendChild(badge);
        }
        badge.textContent = 'T' + tier;
        badge.title = label;
        badge.style.cssText = `display:inline-block;padding:1px 5px;border-radius:4px;font-size:10px;font-weight:700;margin-left:4px;background:${tc}22;color:${tc};border:1px solid ${tc}44;vertical-align:middle;cursor:default;`;
      });

      // Depth-adjustment note beneath the side total
      const noteId = 'sideDepthNote' + side.toUpperCase();
      let noteEl = root.querySelector('#' + noteId);
      const rawTotal = sideData.raw_total  || 0;
      const effTotal = sideData.effective_total || 0;
      const discount = rawTotal - effTotal;
      const totalEl  = root.querySelector(side === 'a' ? '#sideATotal' : '#sideBTotal');
      if (totalEl) {
        if (!noteEl) {
          noteEl = document.createElement('div');
          noteEl.id = noteId;
          noteEl.style.cssText = 'font-size:10px;color:var(--text-muted);margin-top:2px;text-align:center;';
          totalEl.parentNode.insertBefore(noteEl, totalEl.nextSibling);
        }
        noteEl.textContent = discount >= 5 ? `↓ ${Math.round(discount)} depth adj.` : '';
      }
    });

  }

  async function recomputeTrade() {
    const sideATotalEl = root.querySelector("#sideATotal");
    const sideBTotalEl = root.querySelector("#sideBTotal");
    const tradeDiffEl = root.querySelector("#tradeDiff");
    const verdictEl = root.querySelector("#tradeVerdict");
    const barIndicator = root.querySelector("#tradeBarIndicator");
    const errorBox = root.querySelector("#errorBox");

    const sideAIds = state.sideAPlayers.map(p => String(p.id));
    const sideBIds = state.sideBPlayers.map(p => String(p.id));
    const sideAPickIds = state.sideAPicks.map(p => String(p.id));
    const sideBPickIds = state.sideBPicks.map(p => String(p.id));

    if (
      sideAIds.length === 0 &&
      sideBIds.length === 0 &&
      sideAPickIds.length === 0 &&
      sideBPickIds.length === 0
    ) {
      if (sideATotalEl) sideATotalEl.textContent = "0.0";
      if (sideBTotalEl) sideBTotalEl.textContent = "0.0";
      if (tradeDiffEl) tradeDiffEl.textContent = "0.0";
      if (barIndicator) barIndicator.style.left = "50%";
      if (verdictEl) {
        verdictEl.textContent = "Add players to both sides to see the trade balance.";
        verdictEl.className = "otc-verdict";
      }
      if (errorBox) {
        errorBox.style.display = "none";
        errorBox.textContent = "";
      }
      const stlSec = root.querySelector("#similarTradesSection");
      if (stlSec) stlSec.style.display = "none";
      return;
    }

    const payload = {
      league_id: root.querySelector("#leagueIdInput")?.value || "",
      season: root.querySelector("#seasonInput")?.value || "",
      league_type: getLeagueType(),
      league_size: getLeagueSize(),
      scoring_format: getScoringFormat(),
      side_a_players: sideAIds,
      side_b_players: sideBIds,
      side_a_picks: sideAPickIds,
      side_b_picks: sideBPickIds,
    };

    try {
      const res = await fetch("/api/trade-eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Trade eval failed (" + res.status + ").");
      }

      const data = await res.json();
      const diff = Number(data.diff) || 0;
      const aEff = data.side_a ? Number(data.side_a.effective_total) || 0 : 0;
      const bEff = data.side_b ? Number(data.side_b.effective_total) || 0 : 0;

      if (sideATotalEl) sideATotalEl.textContent = formatValue(aEff);
      if (sideBTotalEl) sideBTotalEl.textContent = formatValue(bEff);
      if (tradeDiffEl) tradeDiffEl.textContent = formatValue(diff);

      const maxSideTotal = Math.max(Math.abs(aEff), Math.abs(bEff), 1);
      let normalizedDiff = diff / maxSideTotal;
      normalizedDiff = Math.max(-1, Math.min(1, normalizedDiff));
      // Invert the calculation: when diff > 0 (Team 1 favored), move bar left; when diff < 0 (Team 2 favored), move bar right
      const leftPct = ((1 - normalizedDiff) / 2) * 100;

      if (barIndicator) {
        barIndicator.style.left = leftPct + "%";
      }

      if (verdictEl) {
        verdictEl.textContent = data.verdict || "";
        verdictEl.className = "otc-verdict";
      }

      if (errorBox) {
        errorBox.style.display = "none";
        errorBox.textContent = "";
      }

      _applyTierBadges(data);

      Promise.all([fetchTradeIntel(), fetchSimilarTrades()]).catch(() => {});
    } catch (err) {
      console.error("[trade] error in recomputeTrade:", err);
      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = err.message || "Failed to evaluate trade.";
      }
    }
  }

  // ------------------------------------------------------------
  // fetchSimilarTrades - real trades from the DB involving these players
  // ------------------------------------------------------------
  async function fetchSimilarTrades() {
    const section = root.querySelector("#similarTradesSection");
    if (!section) return;

    const sideAIds = state.sideAPlayers.map(p => String(p.id))
      .filter(id => !id.startsWith("pick_") && !id.startsWith("PICK"));
    const sideBIds = state.sideBPlayers.map(p => String(p.id))
      .filter(id => !id.startsWith("pick_") && !id.startsWith("PICK"));

    if (sideAIds.length === 0 && sideBIds.length === 0) {
      section.style.display = "none";
      return;
    }

    const season = root.querySelector("#seasonInput")?.value || new Date().getFullYear();
    const listEl = root.querySelector("#similarTradesList");
    if (listEl) listEl.innerHTML = '<div class="stl-loading" style="display:flex;align-items:center;gap:8px;padding:12px;color:var(--text-muted);font-size:13px;"><div class="loading-spinner" style="width:14px;height:14px;margin:0;flex-shrink:0;"></div>Loading recent trades...</div>';
    section.style.display = "";

    try {
      const params = new URLSearchParams({ season, limit: 8 });
      if (sideAIds.length) params.set("side_a", sideAIds.join(","));
      if (sideBIds.length) params.set("side_b", sideBIds.join(","));

      const res = await fetch("/api/trade-intel/similar-trades?" + params);
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json();
      const trades = data.trades || [];

      if (!listEl) return;
      if (trades.length === 0) {
        listEl.innerHTML = '<div class="stl-empty">No matching trades found yet.</div>';
        return;
      }

      listEl.innerHTML = trades.map(t => {
        const sfBadge    = t.is_superflex === true  ? '<span class="stl-badge stl-badge-sf">SF</span>'
                         : t.is_superflex === false ? '<span class="stl-badge">1QB</span>' : '';
        const teamsBadge = t.num_teams    ? `<span class="stl-badge">${t.num_teams} Teams</span>` : '';
        const scoreBadge = t.scoring_type ? `<span class="stl-badge">${t.scoring_type.toUpperCase()}</span>` : '';

        function renderAsset(a) {
          const key  = a.is_key_player ? ' stl-key' : '';
          const pick = a.type === 'pick' ? ' stl-pick' : '';
          const pos  = a.type === 'player' ? `<span class="stl-pos">${a.position}</span>` : '';
          return `<div class="stl-asset${key}${pick}">${a.name}${pos}</div>`;
        }

        const sideA = (t.side_a || []).map(renderAsset).join('') || '<div class="stl-asset stl-muted">-</div>';
        const sideB = (t.side_b || []).map(renderAsset).join('') || '<div class="stl-asset stl-muted">-</div>';

        return `<div class="stl-card">
          <div class="stl-card-head">
            <span class="stl-date">${t.date || "-"}</span>
            <div class="stl-badges">${sfBadge}${teamsBadge}${scoreBadge}</div>
          </div>
          <div class="stl-card-body">
            <div class="stl-col">${sideA}</div>
            <div class="stl-col-divider"></div>
            <div class="stl-col">${sideB}</div>
          </div>
        </div>`;
      }).join('');

    } catch (e) {
      if (listEl) listEl.innerHTML = '<div class="stl-empty">Trade data unavailable.</div>';
    }
  }

  // ------------------------------------------------------------
  // fetchTradeIntel - loads real market data for players in the trade
  // ------------------------------------------------------------
  async function fetchTradeIntel() {
    const intelPanel = root.querySelector("#tradeIntelPanel");
    const intelBody = root.querySelector("#tradeIntelBody");
    if (!intelPanel || !intelBody) return;

    const allPlayers = [
      ...state.sideAPlayers.map(p => ({ id: String(p.id), name: p.name || p.id, side: "a" })),
      ...state.sideBPlayers.map(p => ({ id: String(p.id), name: p.name || p.id, side: "b" })),
    ];

    const playerIds = allPlayers.filter(p => !p.id.startsWith("pick_") && !p.id.startsWith("PICK"));
    if (playerIds.length === 0) {
      intelPanel.style.display = "none";
      return;
    }

    const season = root.querySelector("#seasonInput")?.value || new Date().getFullYear();
    const leagueType = getLeagueType();

    intelBody.innerHTML = '<div style="display:flex;align-items:center;gap:6px;color:#9ca3af;font-size:12px;padding:4px 0;"><div class="loading-spinner" style="width:12px;height:12px;margin:0;flex-shrink:0;"></div>Loading market data...</div>';
    intelPanel.style.display = "";

    const _timeout = ms => new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms));
    const results = await Promise.all(
      playerIds.map(p =>
        Promise.race([
          fetch(`/api/trade-intel/player/${p.id}?season=${season}&league_type=${leagueType}`)
            .then(r => r.ok ? r.json() : null)
            .then(d => d ? { ...d, name: p.name, side: p.side } : null),
          _timeout(8000),
        ]).catch(() => null)
      )
    );

    const valid = results.filter(r => r && r.trade_count_all > 0);
    if (valid.length === 0) {
      intelPanel.style.display = "none";
      return;
    }

    intelBody.innerHTML = valid.map(r => {
      const delta = r.value_delta;
      const deltaStr = delta != null
        ? `<span style="color:${delta >= 0 ? "#10b981" : "#ef4444"};font-weight:600;">${delta >= 0 ? "+" : ""}${delta}</span>`
        : "";
      const marketVal = r.market_value ? `<span style="color:#e2e8f0;">${Math.round(r.market_value)}</span>` : "-";
      const modelVal = r.model_value ? `<span style="color:#94a3b8;">${Math.round(r.model_value)}</span>` : "-";

      const bsr = r.buy_sell_ratio;
      const bsrLabel = bsr != null
        ? (bsr > 0.6 ? "<i class=\"fa-solid fa-circle\" style=\"color:#10b981;font-size:9px;vertical-align:middle;\" aria-hidden=\"true\"></i> Buy pressure" : bsr < 0.4 ? "<i class=\"fa-solid fa-circle\" style=\"color:#ef4444;font-size:9px;vertical-align:middle;\" aria-hidden=\"true\"></i> Sell pressure" : "<i class=\"fa-regular fa-circle\" style=\"color:var(--text-muted);font-size:9px;vertical-align:middle;\" aria-hidden=\"true\"></i> Neutral")
        : "";

      const packages = (r.common_packages || []).slice(0, 2).map(pkg => {
        const names = (pkg.companions || []).map(c => c.name).join(" + ");
        return `<div style="font-size:11px;color:#94a3b8;margin-top:2px;">w/ ${names} (${pkg.occurrence_count}x)</div>`;
      }).join("");

      return `
        <div style="border-bottom:1px solid #1e293b;padding-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:13px;font-weight:600;color:#f1f5f9;">${r.name}</span>
            <span style="font-size:11px;color:#64748b;">${r.trade_count_7d}x this week · ${r.trade_count_30d}x/mo</span>
          </div>
          <div style="display:flex;gap:12px;align-items:center;font-size:12px;margin-bottom:4px;">
            <span>Market ${marketVal}</span>
            <span>Model ${modelVal}</span>
            ${deltaStr ? `<span>${deltaStr} vs model</span>` : ""}
          </div>
          ${bsrLabel ? `<div style="font-size:11px;color:#94a3b8;">${bsrLabel}</div>` : ""}
          ${packages}
        </div>`;
    }).join("");
  }

  // ------------------------------------------------------------
  // loadTradeTargets - surfaces players to pursue based on positional rank gaps
  // ------------------------------------------------------------
  async function loadTradeTargets(containerEl) {
    const body = containerEl || root.querySelector("#tradeTargetsBody");
    if (!body) return;

    const hasPremium = (root.querySelector("#otcHasPremium")?.value || "false") === "true";
    if (!hasPremium) {
      body.innerHTML = '<div class="otc-movers-empty">Premium required.</div>';
      return;
    }

    const leagueId       = root.querySelector("#leagueIdInput")?.value || "";
    const season         = root.querySelector("#seasonInput")?.value   || new Date().getFullYear();
    const viewerRosterId = getCurrentRosterId();

    if (!leagueId || !viewerRosterId) {
      body.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">Sign in to see targets.</div>';
      return;
    }

    const pathParts = window.location.pathname.split("/").filter(Boolean);
    const platform = pathParts[0] || "sleeper";
    const leagueType = getLeagueType();

    body.innerHTML = '<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-muted);padding:4px 0;"><div class="loading-spinner" style="width:12px;height:12px;margin:0;flex-shrink:0;"></div>Loading targets…</div>';

    const leagueSize = getLeagueSize();

    try {
      const res = await fetch(
        `/api/trade-targets?platform=${encodeURIComponent(platform)}&league_id=${encodeURIComponent(leagueId)}` +
        `&season=${encodeURIComponent(season)}&viewer_roster_id=${encodeURIComponent(viewerRosterId)}` +
        `&league_type=${encodeURIComponent(leagueType)}&league_size=${encodeURIComponent(leagueSize)}`,
        { cache: "no-store" }
      );
      if (res.status === 403) {
        const errData = await res.json().catch(() => ({}));
        if (errData.paywall && typeof showPaywall === "function") showPaywall("trade-suggestions");
        else body.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">Premium required.</div>';
        return;
      }
      if (!res.ok) throw new Error("Failed");
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const grouped = data.by_position || {};
      const needPositions = Object.keys(grouped);
      const allGrouped = data.all_positions || {};
      const isBalanced = !needPositions.length;

      const posColor = { QB: "#3b82f6", RB: "#22c55e", WR: "#f59e0b", TE: "#8b5cf6" };

      // Each player row includes an inline hidden panel for package ideas
      function renderPlayerRow(t, pos) {
        const col      = posColor[pos] || "var(--text-muted)";
        const safeName = (t.name || "").replace(/&/g,"&amp;").replace(/"/g,"&quot;");
        const safePid  = (t.player_id || "").replace(/&/g,"&amp;").replace(/"/g,"&quot;");
        return `<div style="display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);">
          <div style="min-width:0;flex:1;">
            <div style="font-size:13px;font-weight:600;color:var(--text);display:flex;align-items:center;"><span class="player-clickable" data-player-id="${safePid}" data-player-name="${safeName}">${t.name}</span></div>
            <div style="font-size:11px;color:var(--text-muted);">${t.owner_team}</div>
          </div>
          <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
            <span style="font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px;background:${col}20;color:${col};">${t.pos_rank_label || t.position}</span>
            <span style="font-size:13px;font-weight:800;color:var(--text);">${parseFloat(t.value).toFixed(1)}</span>
            <button class="get-target-btn"
              data-pid="${safePid}" data-name="${safeName}"
              style="font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--border);background:transparent;color:var(--text-muted);cursor:pointer;white-space:nowrap;"
              title="Trade suggestions for this player">Get</button>
          </div>
        </div>`;
      }

      let html = "";

      if (isBalanced) {
        const allKeys = Object.keys(allGrouped);
        if (!allKeys.length) {
          body.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">No player data available.</div>';
          return;
        }
        html += `<div style="font-size:11px;color:var(--text-muted);padding:2px 0 8px;">Your roster is balanced - top available at each position:</div>`;
        allKeys.forEach(pos => {
          const players = allGrouped[pos] || [];
          if (!players.length) return;
          const col = posColor[pos] || "var(--text-muted)";
          html += `<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:${col};padding:4px 0 2px;">${pos}</div>`;
          players.forEach(t => { html += renderPlayerRow(t, pos); });
        });
      } else {
        needPositions.forEach(pos => {
          const players = grouped[pos] || [];
          if (!players.length) return;
          const col = posColor[pos] || "var(--text-muted)";
          html += `<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:${col};padding:4px 0 2px;">${pos} - need</div>`;
          players.forEach(t => { html += renderPlayerRow(t, pos); });
        });
      }

      body.innerHTML = html;

      // Delegate Get clicks — navigate to Suggestions tab with this player pre-loaded
      body.addEventListener("click", function(e) {
        const btn = e.target.closest(".get-target-btn");
        if (!btn) return;
        if (typeof window._openPlayerInSuggestions === "function") {
          window._openPlayerInSuggestions(btn.dataset.pid, btn.dataset.name);
        }
      });
    } catch (e) {
      body.innerHTML = `<div style="font-size:12px;color:var(--text-muted);">Could not load targets.</div>`;
    }
  }

  // Load a suggested package into the trade calculator
  function _loadPackageIntoCalc(targetPlayer, sendAssets) {
    // Clear both sides
    state.sideAPlayers.length = 0;
    state.sideBPlayers.length = 0;
    state.sideAPicks.length   = 0;
    state.sideBPicks.length   = 0;

    // Side A = viewer gets = target player
    state.sideAPlayers.push(targetPlayer);

    // Side B = viewer sends = package
    sendAssets.forEach(asset => {
      if (asset.is_pick) {
        // pick_id is the value-table key (e.g. "2026_1_early") that trade-eval
        // can parse; asset.name is the human-readable label ("2026 1.05")
        state.sideBPicks.push({ id: asset.pick_id || asset.name, display: asset.name });
      } else {
        state.sideBPlayers.push(asset);
      }
    });

    saveState();
    renderChips("A");
    renderChips("B");
    syncEmptyState("A");
    syncEmptyState("B");
    analyzeTrade();

    // Scroll calculator into view
    const calcEl = root.querySelector(".otc-main") || root.querySelector("#tradeCalcCard");
    if (calcEl) calcEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ── Suggestions tab ─────────────────────────────────────────────────────
  (function initSuggestionsTab() {
    const calcTab  = root.querySelector("#otcCalcTab");
    const suggTab  = root.querySelector("#otcSuggestionsTab");
    if (!calcTab || !suggTab) return;

    const tabs = root.querySelectorAll(".otc-main-tab");

    let suggTargetsLoaded = false;
    let suggCurrentPlayerId = null;
    let _fetchAbortCtrl = null;  // cancels in-flight fetchPackages requests
    let _untouchableIds = new Set(JSON.parse(localStorage.getItem('ti-untouchable') || '[]'));
    function _saveUntouchable() { localStorage.setItem('ti-untouchable', JSON.stringify([..._untouchableIds])); }

    function switchTab(name) {
      if (name === "suggestions") {
        const hasPremium = (root.querySelector("#otcHasPremium")?.value || "false") === "true";
        if (!hasPremium) {
          if (typeof showPaywall === "function") showPaywall("trade-suggestions");
          return;
        }
      }
      tabs.forEach(t => t.classList.toggle("is-active", t.dataset.tab === name));
      calcTab.style.display  = name === "calculator"   ? "" : "none";
      suggTab.style.display  = name === "suggestions"  ? "" : "none";
      if (name === "suggestions" && !suggTargetsLoaded) {
        loadSuggTargets();
      }
    }

    tabs.forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab)));

    // Retry targets automatically once a roster ID becomes available.
    // Covers both teamSelect changes and cases where viewerRosterIdInput is set late.
    const teamSelEl    = root.querySelector("#teamSelect");
    const rosterInputEl = root.querySelector("#viewerRosterIdInput");
    const leagueInputEl = root.querySelector("#leagueIdInput");
    const seasonInputEl = root.querySelector("#seasonInput");

    function _onRosterReady() {
      if (!suggTargetsLoaded && getCurrentRosterId()) loadSuggTargets();
    }
    function _onContextChange() {
      // League or season changed — stale targets must be re-fetched
      suggTargetsLoaded = false;
      if (suggTab.style.display !== "none") loadSuggTargets();
    }

    if (teamSelEl)    teamSelEl.addEventListener("change", _onRosterReady);
    if (leagueInputEl) leagueInputEl.addEventListener("change", _onContextChange);
    if (seasonInputEl) seasonInputEl.addEventListener("change", _onContextChange);
    if (rosterInputEl) {
      new MutationObserver(_onRosterReady).observe(rosterInputEl, { attributes: true, attributeFilter: ["value"] });
    }

    // expose so Load & Analyze can switch back
    function switchToCalc() { switchTab("calculator"); }

    // expose so Targets tab "Get" button can navigate here with a pre-selected player
    window._openPlayerInSuggestions = function(playerId, playerName) {
      switchTab("suggestions");
      if (playerInput) playerInput.value = playerName;
      fetchPackages(playerId, playerName);
      // On mobile the tab content is below the fold — scroll to it after paint.
      // Two frames: first lets the tab display change settle, second lets the
      // browser finish reflowing before measuring the scroll target's position.
      // Only scrolls on mobile-width viewports to avoid jarring jumps on desktop.
      if (window.innerWidth < 768) {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const el = root.querySelector("#otcSuggestionsTab") || root.querySelector(".otc-shell");
          if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
        }));
      }
    };

    // ── Player search ────────────────────────────────────────────
    const playerInput    = root.querySelector("#suggPlayerInput");
    const playerDropdown = root.querySelector("#suggPlayerDropdown");
    const resultsMeta    = root.querySelector("#suggResultsMeta");
    const resultsList    = root.querySelector("#suggResultsList");
    if (!playerInput) return;

    function posColor(pos) {
      return { QB: "#3b82f6", RB: "#22c55e", WR: "#f59e0b", TE: "#8b5cf6" }[pos] || "var(--accent)";
    }

    function renderDropdown(matches) {
      if (!matches.length) { playerDropdown.style.display = "none"; return; }
      playerDropdown.innerHTML = matches.slice(0, 12).map(p => {
        const col = posColor(p.position);
        return `<div class="otc-sugg-dropdown-item" data-id="${p.id}" data-name="${(p.name||"").replace(/"/g,"&quot;")}">
          <span class="otc-sugg-dropdown-pos" style="background:${col}20;color:${col};">${p.position}</span>
          <span class="otc-sugg-dropdown-name">${p.name || p.id}</span>
          <span class="otc-sugg-dropdown-val">${Math.round(getPlayerValue(p)) || ""}</span>
        </div>`;
      }).join("");
      playerDropdown.style.display = "block";
    }

    playerInput.addEventListener("input", () => {
      const q = playerInput.value.trim().toLowerCase();
      if (q.length < 2) { playerDropdown.style.display = "none"; return; }
      const matches = allPlayers.filter(p =>
        ["QB","RB","WR","TE"].includes(p.position) &&
        (p.name || "").toLowerCase().includes(q)
      ).sort((a,b) => getPlayerValue(b) - getPlayerValue(a));
      renderDropdown(matches);
    });

    playerDropdown.addEventListener("click", e => {
      const item = e.target.closest(".otc-sugg-dropdown-item");
      if (!item) return;
      playerInput.value = item.querySelector(".otc-sugg-dropdown-name").textContent;
      playerDropdown.style.display = "none";
      fetchPackages(item.dataset.id, item.dataset.name);
    });

    document.addEventListener("click", e => {
      if (!playerInput.contains(e.target) && !playerDropdown.contains(e.target))
        playerDropdown.style.display = "none";
    });

    // ── Fetch packages from API ──────────────────────────────────
    // Exposed so inline lock-icon handlers can trigger a re-fetch after toggling untouchable
    window._refetchTradeIntel = () => {
      if (suggCurrentPlayerId) fetchPackages(suggCurrentPlayerId, _pkgPlayerName);
    };
    window._toggleUntouchable = (pid) => {
      if (_untouchableIds.has(pid)) _untouchableIds.delete(pid);
      else _untouchableIds.add(pid);
      _saveUntouchable();
      window._refetchTradeIntel();
    };

    async function fetchPackages(playerId, playerName) {
      if (!playerId) return;

      // Cancel any previous in-flight request for a different player
      if (_fetchAbortCtrl) _fetchAbortCtrl.abort();
      _fetchAbortCtrl = new AbortController();
      const signal = _fetchAbortCtrl.signal;

      suggCurrentPlayerId = playerId;

      resultsMeta.style.display = "none";
      resultsList.innerHTML = `<div class="otc-sugg-loading">
        ${[1,2,3,4,5].map((_, i) => `<div class="otc-sugg-skeleton" style="display:flex;align-items:center;gap:10px;padding:10px 12px;">
          <div style="flex-shrink:0;width:64px;">
            <div class="otc-sugg-skeleton-line" style="height:18px;border-radius:5px;margin-bottom:5px;animation-delay:${i*0.1}s;"></div>
            <div class="otc-sugg-skeleton-line" style="height:10px;width:70%;animation-delay:${i*0.1+0.05}s;"></div>
          </div>
          <div style="flex:1;display:flex;align-items:center;gap:6px;">
            <div style="flex:1;">
              <div class="otc-sugg-skeleton-line" style="height:9px;width:45%;margin-bottom:5px;animation-delay:${i*0.1}s;"></div>
              <div class="otc-sugg-skeleton-line" style="height:13px;width:${70+i*5}%;animation-delay:${i*0.1+0.05}s;"></div>
            </div>
            <div class="otc-sugg-skeleton-line" style="width:14px;height:14px;border-radius:50%;flex-shrink:0;animation-delay:${i*0.1}s;"></div>
            <div style="flex:1;">
              <div class="otc-sugg-skeleton-line" style="height:9px;width:40%;margin-bottom:5px;animation-delay:${i*0.1}s;"></div>
              <div class="otc-sugg-skeleton-line" style="height:13px;width:${60+i*4}%;animation-delay:${i*0.1+0.05}s;"></div>
            </div>
          </div>
          <div class="otc-sugg-skeleton-line" style="flex-shrink:0;width:64px;height:28px;border-radius:8px;animation-delay:${i*0.1}s;"></div>
        </div>`).join("")}
      </div>`;

      const hasPremium = (root.querySelector("#otcHasPremium")?.value || "false") === "true";
      if (!hasPremium) {
        resultsList.innerHTML = `<div class="otc-sugg-empty">
          <div class="otc-sugg-empty-title">Pro Feature</div>
          <div class="otc-sugg-empty-sub">Upgrade to see real trade packages for any player.</div>
        </div>`;
        return;
      }

      const leagueId       = root.querySelector("#leagueIdInput")?.value  || "";
      const season         = root.querySelector("#seasonInput")?.value     || new Date().getFullYear();
      const viewerRosterId = getCurrentRosterId();
      const platform       = window.location.pathname.split("/").filter(Boolean)[0] || "sleeper";
      const leagueType     = getLeagueType();

      try {
        const res = await fetch(
          `/api/trade-intel/player-packages/${encodeURIComponent(playerId)}` +
          `?season=${season}&league_type=${leagueType}&league_id=${encodeURIComponent(leagueId)}` +
          `&platform=${encodeURIComponent(platform)}&viewer_roster_id=${encodeURIComponent(viewerRosterId)}` +
          `&untouchable_ids=${encodeURIComponent([..._untouchableIds].join(','))}`,
          { signal }
        );

        // A newer search was started — discard this response silently
        if (signal.aborted || suggCurrentPlayerId !== playerId) return;

        if (res.status === 403) {
          resultsList.innerHTML = `<div class="otc-sugg-empty">
            <div class="otc-sugg-empty-title">Pro Feature</div>
            <div class="otc-sugg-empty-sub">Upgrade to unlock trade package history for any player.</div>
          </div>`;
          return;
        }

        const data = await res.json();

        const hasRealPkgs = (data.real_packages && data.real_packages.length) || (data.archetype_patterns && data.archetype_patterns.length);
        if (!hasRealPkgs) {
          resultsList.innerHTML = `<div class="otc-sugg-empty">
            <div class="otc-sugg-empty-title">No trade data yet</div>
            <div class="otc-sugg-empty-sub">Not enough real trades for ${playerName} in similar leagues yet.</div>
          </div>`;
          return;
        }

        resultsMeta.style.display = "none";

        renderPackages(
          [],
          data.player_name, playerId, data.focus_value,
          data.real_packages, data.total_real_trades,
          data.archetype_patterns
        );

      } catch (err) {
        if (err.name === "AbortError") return;  // superseded by a newer search
        resultsList.innerHTML = `<div class="otc-sugg-empty">
          <div class="otc-sugg-empty-sub">Failed to load packages.</div></div>`;
      }
    }

    // ── Render package cards (paginated, 5 per page) ────────────
    const PAGE_SIZE = 5;
    let _pkgPage = 0;
    let _pkgAll  = [];
    let _pkgPlayerId       = null;
    let _pkgPlayerName     = null;
    let _pkgRealPkgs       = [];
    let _pkgRealTotal      = 0;
    let _pkgComboPkgs      = [];
    let _pkgArchetypes     = [];

    window.archToggle = function(uid, mode) {
      const allGrid  = document.getElementById('arch-grid-all-'  + uid);
      const teamGrid = document.getElementById('arch-grid-team-' + uid);
      const allBtn   = document.getElementById('arch-btn-all-'   + uid);
      const teamBtn  = document.getElementById('arch-btn-team-'  + uid);
      if (!allGrid || !teamGrid) return;
      const showAll = mode === 'all';
      allGrid.style.display  = showAll ? 'grid' : 'none';
      teamGrid.style.display = showAll ? 'none' : 'grid';
      allBtn.classList.toggle('is-active',  showAll);
      teamBtn.classList.toggle('is-active', !showAll);
    };

    function renderPackages(packages, playerName, playerId, focusValue, realPkgs, realTotal, archetypes) {
      _pkgAll        = packages;
      _pkgPage       = 0;
      _pkgPlayerId   = playerId;
      _pkgPlayerName = playerName;
      _pkgRealPkgs   = realPkgs   || [];
      _pkgRealTotal  = realTotal  || 0;
      _pkgComboPkgs  = [];
      _pkgArchetypes = archetypes || [];
      renderPackagePage();
    }

    function renderPackagePage() {
      const packages   = _pkgAll;
      const playerId   = _pkgPlayerId;
      const playerName = _pkgPlayerName;

      function valueClass(label) {
        if (label === "Great deal") return "great";
        if (label === "Fair value") return "fair";
        return "overpay";
      }

      const PROFILE_LABEL = {
        'young-rising':  { text: '↑ Young Rising',    color: '#10b981' },
        'young-stable':  { text: 'Young Stable',      color: '#3b82f6' },
        'young-falling': { text: '↓ Young Falling',   color: '#f59e0b' },
        'prime-rising':  { text: '↑ Prime Rising',    color: '#10b981' },
        'prime-stable':  { text: 'Prime',             color: '#6366f1' },
        'prime-falling': { text: '↓ Prime Falling',   color: '#f59e0b' },
        'vet-rising':    { text: '↑ Vet Resurgence',  color: '#f59e0b' },
        'vet-stable':    { text: 'Veteran',            color: '#9ca3af' },
        'vet-falling':   { text: '↓ Declining Vet',   color: '#ef4444' },
      };

      function assetHtml(a) {
        if (a.is_pick || a.type === "pick") {
          const label = a.name || "Pick";
          return `<div class="otc-sugg-pkg-asset"><span class="otc-sugg-pkg-asset-pos" style="background:rgba(99,102,241,.12);color:#6366f1;">PICK</span>${label}</div>`;
        }
        const col = posColor(a.position);
        const prof = a.profile ? PROFILE_LABEL[a.profile] : null;
        const profBadge = prof
          ? `<span style="font-size:10px;font-weight:600;color:${prof.color};margin-left:4px;white-space:nowrap;">${prof.text}</span>`
          : '';
        return `<div class="otc-sugg-pkg-asset" style="flex-wrap:wrap;gap:4px;">
          <span class="otc-sugg-pkg-asset-pos" style="background:${col}20;color:${col};">${a.position}</span>
          <span>${a.name}</span>${profBadge}
        </div>`;
      }

      const focusPlayer = allPlayers.find(p => String(p.id) === String(playerId));
      const focusPos    = focusPlayer?.position || "WR";
      const focusCol    = posColor(focusPos);

      const totalPages = Math.ceil(packages.length / PAGE_SIZE);
      const page       = _pkgPage;
      const slice      = packages.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

      const cardsHtml = slice.map((pkg) => {
        const vc = valueClass(pkg.value_label);
        const extra = pkg.extra_receive || null;
        const freqLabel = extra
          ? `<span class="otc-sugg-pkg-freq" style="color:#10b981;">+ bonus player</span>`
          : pkg.is_profile_match
            ? `<span class="otc-sugg-pkg-freq" style="color:var(--accent);">From your roster</span>`
            : `<span class="otc-sugg-pkg-freq">${pkg.frequency}× traded</span>`;

        // YOU GET side: always shows the target, plus the throw-in if present
        const extraCol  = extra ? posColor(extra.position) : null;
        const extraProf = extra?.profile ? PROFILE_LABEL[extra.profile] : null;
        const extraBadge = extraProf
          ? `<span style="font-size:10px;font-weight:600;color:${extraProf.color};margin-left:4px;">${extraProf.text}</span>`
          : '';
        const extraAssetHtml = extra
          ? `<div class="otc-sugg-pkg-asset" style="flex-wrap:wrap;gap:4px;">
               <span class="otc-sugg-pkg-asset-pos" style="background:${extraCol}20;color:${extraCol};">${extra.position}</span>
               <span>${extra.name}</span>${extraBadge}
             </div>`
          : '';

        const patternSigHtml = pkg.pattern_sig
          ? `<div style="font-size:10px;color:var(--text-muted);margin-top:4px;letter-spacing:.02em;">Market pattern: ${pkg.pattern_sig}</div>`
          : '';
        const throwInHtml = pkg.throw_in_sig
          ? `<div style="font-size:10px;color:var(--text-muted);margin-top:2px;letter-spacing:.02em;">Seller may include: ${pkg.throw_in_sig}</div>`
          : '';
        return `<div class="otc-sugg-package">
          <div class="otc-sugg-pkg-meta">
            <span class="otc-sugg-pkg-value ${vc}">${pkg.value_label}</span>
            ${freqLabel}
          </div>
          <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:start;gap:6px;margin-top:6px;">
            <div>
              <div class="otc-sugg-pkg-side-label">YOU GET</div>
              <div class="otc-sugg-pkg-assets">
                <div class="otc-sugg-pkg-asset" style="overflow:hidden;">
                  <span class="otc-sugg-pkg-asset-pos" style="background:${focusCol}20;color:${focusCol};flex-shrink:0;">${focusPos}</span>
                  <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${playerName}</span>
                </div>
                ${extraAssetHtml}
              </div>
            </div>
            <div class="otc-sugg-pkg-divider" style="padding-top:18px;">←</div>
            <div>
              <div class="otc-sugg-pkg-side-label">YOU GIVE</div>
              <div class="otc-sugg-pkg-assets">${pkg.assets.map(assetHtml).join("")}</div>
            </div>
          </div>
          ${patternSigHtml}
          ${throwInHtml}
          <button class="otc-sugg-pkg-load-btn"
            data-focus-id="${playerId}"
            data-assets="${encodeURIComponent(JSON.stringify(pkg.assets))}"
            ${extra ? `data-extra-receive="${encodeURIComponent(JSON.stringify(extra))}"` : ''}>
            Analyze
          </button>
        </div>`;
      }).join("");

      const paginationHtml = totalPages > 1 ? `
        <div class="otc-sugg-pagination">
          <button class="otc-sugg-page-btn" data-dir="-1" ${page === 0 ? "disabled" : ""}>← Prev</button>
          <span class="otc-sugg-page-label">${page + 1} / ${totalPages}</span>
          <button class="otc-sugg-page-btn" data-dir="1" ${page >= totalPages - 1 ? "disabled" : ""}>Next →</button>
        </div>` : "";

      // ── Profile labels shared across real-trade and combo sections ─────────────
      const PROF_LBL = {
        'young-rising':  { text: '↑ Young Rising',   color: '#10b981' },
        'young-stable':  { text: 'Young',             color: '#3b82f6' },
        'young-falling': { text: '↓ Young Falling',  color: '#f59e0b' },
        'prime-rising':  { text: '↑ Prime Rising',   color: '#10b981' },
        'prime-stable':  { text: 'Prime',             color: '#6366f1' },
        'prime-falling': { text: '↓ Prime Falling',  color: '#f59e0b' },
        'vet-rising':    { text: '↑ Vet',             color: '#f59e0b' },
        'vet-stable':    { text: 'Veteran',            color: '#9ca3af' },
        'vet-falling':   { text: '↓ Declining Vet',  color: '#ef4444' },
      };

      // ── Shared helper: render one archetype label ──────────────────────
      // Formats: "RB-T4-Prime"  "PICK:R1:Early"  "PICK:R1"  "PICK"
      function archetypeChip(lbl) {
        if (!lbl || lbl === '?') return '';
        if (lbl === 'PICK' || lbl.startsWith('PICK:')) {
          const segs   = lbl.split(':');           // ['PICK','R1','Early']
          const rnd    = segs[1] || '';
          const slot   = segs[2] || '';
          const slotColor = slot === 'Early' ? '#10b981' : slot === 'Late' ? '#ef4444' : '#a78bfa';
          return `<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 6px;border-radius:4px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.2);">
            <span style="font-size:10px;font-weight:700;color:#6366f1;">PICK</span>
            ${rnd ? `<span style="font-size:10px;font-weight:600;color:#a78bfa;">${rnd}</span>` : ''}
            ${slot ? `<span style="font-size:10px;color:${slotColor};">${slot}</span>` : ''}
          </span>`;
        }
        const parts   = lbl.split('-');
        const pos     = parts[0] || '';
        const tier    = parts[1] || '';
        const bracket = parts[2] || '';
        const col     = posColor(pos);
        const bracketColor = bracket === 'Young' ? '#10b981' : bracket === 'Vet' ? '#9ca3af' : '#6366f1';
        return `<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 6px;border-radius:4px;background:${col}12;border:1px solid ${col}30;">
          <span style="font-size:10px;font-weight:700;color:${col};">${pos}</span>
          <span style="font-size:10px;font-weight:600;color:var(--text);">${tier}</span>
          ${bracket ? `<span style="font-size:10px;color:${bracketColor};">· ${bracket}</span>` : ''}
        </span>`;
      }

      function archetypeSigHtml(pattern_sig, throw_in_sig) {
        if (!pattern_sig) return '';
        const chipCounts = new Map();
        pattern_sig.split(' + ').filter(Boolean).forEach(lbl => {
          chipCounts.set(lbl, (chipCounts.get(lbl) || 0) + 1);
        });
        const chips = Array.from(chipCounts.entries()).map(([lbl, n]) => {
          const chip = archetypeChip(lbl);
          return n > 1
            ? `<span style="display:inline-flex;align-items:center;gap:2px;"><span style="font-size:10px;font-weight:700;color:var(--text-muted);">${n}×</span>${chip}</span>`
            : chip;
        }).join(`<span style="color:var(--text-muted);font-size:11px;margin:0 1px;">+</span>`);
        const throwIn = throw_in_sig
          ? `<span style="font-size:10px;color:var(--text-muted);white-space:nowrap;">
               <span style="opacity:.5;margin:0 3px;">·</span>throw-in: ${throw_in_sig}
             </span>`
          : '';
        return chips + throwIn;
      }

      // ── "Based on real trades" section ─────────────────────────────────────────
      let realTradeHtml = "";
      if (_pkgRealPkgs.length || _pkgArchetypes.length) {
        realTradeHtml += `<div style="margin-top:14px;padding-top:12px;border-top:2px solid var(--border);">
          <div style="margin-bottom:10px;">
            <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:2px;">How people have acquired ${_pkgPlayerName}</div>
            <div style="font-size:11px;color:var(--text-muted);">Patterns from ${_pkgRealTotal} real trades in similar leagues</div>
          </div>`;

        // ── Common archetype patterns ─────────────────────────────
        if (_pkgArchetypes.length) {
          const archUid = String(playerId || Date.now());
          const teamArchetypes = _pkgArchetypes.filter(ap => ap.fits_your_team);

          function buildArchCells(list) {
            if (!list.length) return `<div style="grid-column:1/-1;padding:12px 10px;font-size:12px;color:var(--text-muted);">No top patterns match your current roster.</div>`;
            return list.map((ap, idx) => {
              const sigHtml     = archetypeSigHtml(ap.pattern_sig, ap.throw_in_sig);
              const pct         = ap.pct > 0 ? `<span style="font-size:11px;font-weight:700;color:#a78bfa;white-space:nowrap;">${ap.pct}%</span>` : '';
              const borderRight = idx % 2 === 0 ? 'border-right:1px solid var(--border);' : '';
              return `<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:6px;padding:7px 10px;border-bottom:1px solid var(--border);${borderRight}min-width:0;">
                <div style="display:flex;align-items:center;flex-wrap:wrap;gap:3px;min-width:0;">${sigHtml}</div>
                ${pct}
              </div>`;
            }).join('');
          }

          realTradeHtml += `<div style="margin-bottom:12px;border-radius:8px;border:1px solid var(--border);overflow:hidden;">
            <div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;background:var(--surface-2,rgba(0,0,0,.04));border-bottom:1px solid var(--border);">
              <span style="font-size:10px;font-weight:700;color:var(--text-muted);letter-spacing:.05em;text-transform:uppercase;">What teams typically send</span>
              <div class="otc-main-tabs" style="margin-bottom:0;">
                <button id="arch-btn-all-${archUid}" class="otc-main-tab is-active" onclick="archToggle('${archUid}','all')">Top patterns</button>
                <button id="arch-btn-team-${archUid}" class="otc-main-tab" onclick="archToggle('${archUid}','team')">Your team</button>
              </div>
            </div>
            <div id="arch-grid-all-${archUid}" style="display:grid;grid-template-columns:1fr 1fr;gap:0;">${buildArchCells(_pkgArchetypes)}</div>
            <div id="arch-grid-team-${archUid}" style="display:none;grid-template-columns:1fr 1fr;gap:0;">${buildArchCells(teamArchetypes)}</div>
          </div>`;
        }

        // ── Individual real-trade examples ──
        if (_pkgRealPkgs.length) {
          realTradeHtml += `<div style="font-size:10px;font-weight:700;color:var(--text-muted);letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;">Real trades — adapted to your roster</div>`;

          // Group packages by pattern_sig
          const _archGroups = new Map();
          _pkgRealPkgs.forEach(pkg => {
            const sig = pkg.pattern_sig || '';
            if (!_archGroups.has(sig)) _archGroups.set(sig, []);
            _archGroups.get(sig).push(pkg);
          });

          // Sort groups: archetypes with known patterns first (matching _pkgArchetypes order), then others
          const _archOrder = _pkgArchetypes.map(ap => ap.pattern_sig);
          const _sortedGroups = [..._archGroups.entries()].sort(([a], [b]) => {
            const ai = _archOrder.indexOf(a), bi = _archOrder.indexOf(b);
            if (ai === -1 && bi === -1) return 0;
            if (ai === -1) return 1;
            if (bi === -1) return -1;
            return ai - bi;
          });

          _sortedGroups.forEach(([sig, pkgs]) => {
            // Render a group header using the archetype chips
            const groupHeader = sig
              ? `<div style="font-size:10px;font-weight:700;color:var(--text-muted);letter-spacing:.05em;text-transform:uppercase;margin:12px 0 6px;display:flex;align-items:center;gap:6px;">
                   ${archetypeSigHtml(sig, '')}
                 </div>`
              : '';
            realTradeHtml += groupHeader;

            pkgs.forEach(pkg => {
            const count  = pkg.trades_like_this || 0;
            const isRef  = !!pkg.is_reference;
            const focusPos = focusPlayer?.position || 'WR';
            const focusCol = posColor(focusPos);

            // Deduplicate assets: group identical names → "2× 2026 1st"
            const assetCounts = new Map();
            (pkg.send || []).forEach(a => {
              const key = a.name || "Pick";
              if (!assetCounts.has(key)) assetCounts.set(key, { asset: a, n: 0 });
              assetCounts.get(key).n++;
            });
            const giveHtml = Array.from(assetCounts.values()).map(({ asset: a, n }) => {
              const prefix = n > 1 ? `<span style="font-size:10px;font-weight:700;color:var(--text-muted);margin-right:1px;">${n}×</span>` : '';
              if (a.is_pick || a.type === "pick") {
                return `<div class="otc-rt-asset">
                  ${prefix}<span class="otc-rt-pos" style="background:rgba(99,102,241,.1);color:#6366f1;">PICK</span>
                  <span class="otc-rt-name">${a.name || "Pick"}</span>
                </div>`;
              }
              const pid = a.player_id || a.id || '';
              const locked = pid && _untouchableIds.has(pid);
              const lockBtn = pid ? `<button onclick="window._toggleUntouchable('${pid}')" title="${locked?'Remove from excluded':'Exclude from suggestions'}" style="border:none;background:none;cursor:pointer;padding:0 0 0 4px;line-height:1;display:inline-flex;align-items:center;opacity:${locked?0.85:0.25};"><span class="fa-solid ${locked?'fa-lock':'fa-lock-open'}" style="width:10px;height:10px;"></span></button>` : '';
              const col = posColor(a.position);
              return `<div class="otc-rt-asset" style="display:flex;align-items:center;">
                ${prefix}<span class="otc-rt-pos" style="background:${col}18;color:${col};">${a.position}</span>
                <span class="otc-rt-name" style="${isRef ? 'color:var(--text-muted);' : ''}">${a.name}</span>
                ${lockBtn}
              </div>`;
            }).join('');

            const gradeMap = {
              steal:      { label: 'Steal 🟢',    color: '#10b981' },
              fair:       { label: 'Fair',         color: '#6366f1' },
              overpay:    { label: 'Slight overpay', color: '#f59e0b' },
              big_overpay:{ label: 'Overpay',      color: '#ef4444' },
            };
            const gradeInfo = gradeMap[pkg.value_grade];
            const gradeHtml = gradeInfo
              ? `<span style="font-size:10px;font-weight:700;color:${gradeInfo.color};white-space:nowrap;">${gradeInfo.label}</span>`
              : '';

            const takersHtml = (pkg.likely_takers && pkg.likely_takers.length)
              ? `<span style="font-size:10px;color:var(--text-muted);">· Likely: ${pkg.likely_takers.join(', ')}</span>`
              : '';

            realTradeHtml += `<div class="otc-real-trade-card">
              <div class="otc-rt-body">
                <div class="otc-rt-side">
                  <div class="otc-rt-label">YOU GET</div>
                  <div class="otc-rt-asset">
                    <span class="otc-rt-pos" style="background:${focusCol}18;color:${focusCol};">${focusPos}</span>
                    <span class="otc-rt-name" style="font-weight:700;">${_pkgPlayerName}</span>
                  </div>
                </div>
                <div class="otc-rt-divider"></div>
                <div class="otc-rt-side">
                  <div class="otc-rt-label">YOU GIVE</div>
                  ${giveHtml}
                </div>
              </div>
              <div class="otc-rt-footer">
                <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;min-width:0;">
                  <span class="otc-rt-count">${count} ${count === 1 ? 'trade' : 'trades'}</span>
                  ${gradeHtml}
                  ${takersHtml}
                </div>
                <button class="otc-sugg-pkg-load-btn"
                  data-focus-id="${_pkgPlayerId}"
                  data-assets="${encodeURIComponent(JSON.stringify(pkg.send || []))}">Analyze</button>
              </div>
            </div>`;
          }); // pkgs.forEach
          }); // _sortedGroups.forEach
        }
        realTradeHtml += `</div>`;
      }

      resultsList.innerHTML = cardsHtml + paginationHtml + realTradeHtml;

      resultsList.querySelectorAll(".otc-sugg-page-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const newPage = _pkgPage + parseInt(btn.dataset.dir);
          if (newPage < 0 || newPage >= totalPages) return;
          _pkgPage = newPage;
          renderPackagePage();
          resultsList.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
      });

      resultsList.querySelectorAll(".otc-sugg-pkg-load-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          const focusId    = btn.dataset.focusId;
          const assets     = JSON.parse(decodeURIComponent(btn.dataset.assets));
          const extraRaw   = btn.dataset.extraReceive
            ? JSON.parse(decodeURIComponent(btn.dataset.extraReceive))
            : null;
          const focusPObj = allPlayers.find(p => String(p.id) === String(focusId));
          if (!focusPObj) return;

          state.sideAPlayers.length = 0;
          state.sideBPlayers.length = 0;
          state.sideAPicks.length   = 0;
          state.sideBPicks.length   = 0;

          // Side A = what you receive (the target player, plus any combo throw-in)
          state.sideAPlayers.push(focusPObj);
          if (extraRaw) {
            const extraPObj = allPlayers.find(p => String(p.id) === String(extraRaw.player_id))
              || (extraRaw.name && allPlayers.find(p => p.name && p.name.toLowerCase() === extraRaw.name.toLowerCase()));
            if (extraPObj) state.sideAPlayers.push(extraPObj);
          }

          // Side B = what you give (the package)
          assets.forEach(a => {
            if (a.type === "pick" || a.is_pick) {
              const yr   = String(a.pick_season || "").replace(/\D/g, "");
              const rd   = String(a.pick_round  || "").replace(/\D/g, "");
              const slot = a.pick_slot ? String(a.pick_slot).replace(/\D/g, "").padStart(2, "0") : null;
              const order = (a.pick_order || "").toLowerCase().replace(/[^a-z]/g, "") || "mid";
              // Slot picks use numeric third segment (e.g. 2026_1_01), bucket picks use word (e.g. 2026_1_early)
              const pickId = yr && rd ? `${yr}_${rd}_${slot || order}` : null;
              const pickObj = pickId && (
                allPlayers.find(p => p.id === pickId) ||
                allPlayers.find(p => yr && rd && p.id && p.id.startsWith(`${yr}_${rd}_`))
              );
              if (pickObj) {
                state.sideBPicks.push({ id: pickObj.id, display: pickObj.name });
              } else if (pickId) {
                state.sideBPicks.push({ id: pickId, display: a.name || formatPickId(pickId) });
              }
            } else {
              const pObj = allPlayers.find(p => String(p.id) === String(a.player_id));
              if (pObj) state.sideBPlayers.push(pObj);
            }
          });

          saveState();
          renderChips("A");
          renderChips("B");
          syncEmptyState("A");
          syncEmptyState("B");
          analyzeTrade();
          switchToCalc();
          const shell = root.querySelector(".otc-shell");
          if (shell) shell.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });
    }

    // ── Suggestions-tab Trade Targets (different from sidebar) ───
    async function loadSuggTargets() {
      const container = root.querySelector("#otcSuggTargetsBody");
      if (!container) return;

      const hasPremium = (root.querySelector("#otcHasPremium")?.value || "false") === "true";
      if (!hasPremium) {
        suggTargetsLoaded = true;
        container.innerHTML = '<div class="otc-movers-empty">Premium required.</div>';
        return;
      }

      const leagueId       = root.querySelector("#leagueIdInput")?.value || "";
      const season         = root.querySelector("#seasonInput")?.value   || new Date().getFullYear();
      const viewerRosterId = getCurrentRosterId();

      if (!leagueId || !viewerRosterId) {
        // Don't mark loaded — retry next time the tab is opened
        container.innerHTML = '<div class="otc-movers-empty">Select your team to see targets.</div>';
        return;
      }

      const pathParts  = window.location.pathname.split("/").filter(Boolean);
      const platform   = pathParts[0] || "sleeper";
      const leagueType = getLeagueType();
      const leagueSize = getLeagueSize();

      container.innerHTML = '<div class="otc-movers-empty">Loading targets…</div>';

      try {
        const res = await fetch(
          `/api/trade-targets?platform=${encodeURIComponent(platform)}&league_id=${encodeURIComponent(leagueId)}` +
          `&season=${encodeURIComponent(season)}&viewer_roster_id=${encodeURIComponent(viewerRosterId)}` +
          `&league_type=${encodeURIComponent(leagueType)}&league_size=${encodeURIComponent(leagueSize)}`,
          { cache: "no-store" }
        );
        if (!res.ok) throw new Error("Failed");
        const data = await res.json();
        suggTargetsLoaded = true;  // only mark done after a successful response

        const grouped     = data.by_position || {};
        const allGrouped  = data.all_positions || {};
        const isBalanced  = !Object.keys(grouped).length;
        const posColor2   = { QB: "#3b82f6", RB: "#22c55e", WR: "#f59e0b", TE: "#8b5cf6" };

        function renderRow(t, pos) {
          const col      = posColor2[pos] || "var(--accent)";
          const safeName = (t.name || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;");
          const safePid  = (t.player_id || "");
          return `<div class="otc-sugg-target-row">
            <span class="otc-sugg-target-pos" style="background:${col}20;color:${col};">${pos}</span>
            <span class="otc-sugg-target-name">${t.name || ""}</span>
            <button class="sugg-target-get-btn otc-sugg-target-btn"
              data-pid="${safePid}" data-name="${safeName}">
              Find packages
            </button>
          </div>`;
        }

        let html = "";
        if (isBalanced) {
          html += `<div style="font-size:11px;color:var(--text-muted);padding:2px 0 8px;">Roster is balanced — top available at each position:</div>`;
          Object.keys(allGrouped).forEach(pos => {
            (allGrouped[pos] || []).forEach(t => { html += renderRow(t, pos); });
          });
        } else {
          Object.keys(grouped).forEach(pos => {
            (grouped[pos] || []).forEach(t => { html += renderRow(t, pos); });
          });
        }

        container.innerHTML = html || '<div class="otc-movers-empty">No targets found.</div>';

        container.addEventListener("click", e => {
          const btn = e.target.closest(".sugg-target-get-btn");
          if (!btn) return;
          const pid  = btn.dataset.pid;
          const name = btn.dataset.name;
          if (!pid || !name) return;
          // Populate the search input and fetch packages
          if (playerInput) {
            playerInput.value = name;
            playerDropdown.style.display = "none";
          }
          fetchPackages(pid, name);
          // Scroll search into view
          const buildHead = root.querySelector(".otc-sugg-build-head");
          if (buildHead) buildHead.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, { once: false });

      } catch (e) {
        container.innerHTML = '<div class="otc-movers-empty">Could not load targets.</div>';
      }
    }
  })();


  // ------------------------------------------------------------
  // analyzeTrade - owns ALL loading/result/empty state transitions
  // Never call tradeAiBody.innerHTML directly from outside this fn
  // ------------------------------------------------------------
  async function analyzeTrade() {
    // Don't run analysis in guest mode
    const isGuest = root.querySelector("#isGuestMode")?.value === "true";
    if (isGuest) {
      return;
    }

    const tradeAiBody = root.querySelector("#tradeAiBody");
    const errorBox = root.querySelector("#errorBox");
    const loadingState = root.querySelector("#aiLoadingState");
    const emptyState = root.querySelector("#aiEmptyState");
    const resultState = root.querySelector("#aiAnalysisResult");

    const sideAIds = state.sideAPlayers.map(p => String(p.id));
    const sideBIds = state.sideBPlayers.map(p => String(p.id));
    const sideAPickIds = state.sideAPicks.map(p => String(p.id));
    const sideBPickIds = state.sideBPicks.map(p => String(p.id));

    // Nothing on either side - show empty state and bail
    if (
      sideAIds.length === 0 &&
      sideBIds.length === 0 &&
      sideAPickIds.length === 0 &&
      sideBPickIds.length === 0
    ) {
      if (loadingState) loadingState.style.display = "none";
      if (resultState) resultState.style.display = "none";
      if (emptyState) emptyState.style.display = "block";
      return;
    }

    // Show loading, hide everything else
    if (loadingState) loadingState.style.display = "block";
    if (emptyState) emptyState.style.display = "none";
    if (resultState) resultState.style.display = "none";

    // Yield to browser so the loading state actually paints before fetch starts
    await new Promise(resolve => requestAnimationFrame(resolve));

    const viewerSide =
      root.querySelector('input[name="viewerSide"]:checked')?.value || "a";
    const teamSelector = root.querySelector("#teamSelect");
    const selectedTeamRosterId = teamSelector?.value || "";
    const selectedTeamName =
      teamSelector?.options[teamSelector?.selectedIndex]?.text || "";

    const payload = {
      league_id: root.querySelector("#leagueIdInput")?.value || "",
      season: root.querySelector("#seasonInput")?.value || "",
      league_type: getLeagueType(),
      scoring_format: getScoringFormat(),
      viewer_side: viewerSide,
      side_a_players: sideAIds,
      side_b_players: sideBIds,
      side_a_picks: sideAPickIds,
      side_b_picks: sideBPickIds,
      viewer_roster_id: selectedTeamRosterId,
      viewer_team_name: selectedTeamName,
    };

    try {
      const res = await fetch("/api/trade-eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Trade eval failed (" + res.status + ").");

      const data = await res.json();

      // Hide loading, show result panel
      if (loadingState) loadingState.style.display = "none";
      if (emptyState) emptyState.style.display = "none";
      if (resultState) {
        resultState.style.display = "block";
        // Write into resultState only - never clobber tradeAiBody directly
        // so that sibling state nodes (loading/empty) survive intact
        if (data.analysis_html) {
          resultState.innerHTML = data.analysis_html;
        } else if (data.error && data.error.includes("No user context")) {
          resultState.innerHTML = `
            <div class="otc-ai-empty">
              <div class="otc-ai-empty-title">Select Your Team</div>
              <div class="otc-ai-empty-sub">
                Please select your username/team from the league settings so the AI
                can analyze trades from your perspective.
              </div>
            </div>`;
        } else {
          resultState.innerHTML = `
            <div class="otc-ai-empty">
              <div class="otc-ai-empty-title">No AI take yet</div>
              <div class="otc-ai-empty-sub">Add assets on both sides to generate a front-office opinion.</div>
            </div>`;
        }
      }

      // Roster depth warnings for the viewer's post-trade roster
      const scarcityEl = document.getElementById('tradeScarcityNotes');
      if (scarcityEl) {
        const warnings = data.depth_warnings || {};
        const entries = Object.entries(warnings).filter(([, w]) => w.warning);
        if (entries.length > 0) {
          let html = '<div class="scarcity-notes-wrap"><div class="scarcity-notes-title">Roster Depth</div><div class="scarcity-notes-list">';
          entries.forEach(([pos, w]) => {
            const cls = w.severity === 'danger' ? 'depth-danger' : 'depth-caution';
            const icon = w.severity === 'danger' ? '<i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>' : '<i class="fa-solid fa-bolt" aria-hidden="true"></i>';
            html += `<div class="scarcity-note-row ${cls}"><span class="scarcity-pos pos-${pos.toLowerCase()}">${pos}</span><span class="scarcity-tier">${icon} ${w.warning}</span></div>`;
          });
          html += '</div></div>';
          scarcityEl.innerHTML = html;
          scarcityEl.style.display = 'block';
        } else {
          scarcityEl.style.display = 'none';
        }
      }

      if (errorBox) {
        errorBox.style.display = "none";
        errorBox.textContent = "";
      }
    } catch (err) {
      console.error("[trade] error in analyzeTrade:", err);

      if (loadingState) loadingState.style.display = "none";
      if (emptyState) emptyState.style.display = "none";
      if (resultState) {
        resultState.style.display = "block";
        resultState.innerHTML = `
          <div class="otc-ai-empty">
            <div class="otc-ai-empty-title">Analysis failed</div>
            <div class="otc-ai-empty-sub">There was an error generating the AI analysis.</div>
          </div>`;
      }

      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = err.message || "Failed to analyze trade.";
      }
    }
  }

  // Fuzzy player name matching - returns a score (higher = better match).
  // Handles: exact substring, word-start matches, and single-transposition typos.
  function fuzzyNameScore(name, query) {
    if (!name || !query) return 0;
    const n = name.toLowerCase();
    const q = query.toLowerCase();
    // Exact substring - highest priority
    if (n.includes(q)) return 100 + (100 - n.indexOf(q));
    // Check search_name field alias passed in via player object handled by caller
    // Word-start matching: "jax sm" matches "Jaxon Smith-Njigba"
    const nWords = n.split(/[\s\-]+/);
    const qWords = q.split(/\s+/).filter(Boolean);
    if (qWords.length > 1) {
      let wi = 0;
      for (const qw of qWords) {
        while (wi < nWords.length && !nWords[wi].startsWith(qw)) wi++;
        if (wi >= nWords.length) break;
        wi++;
      }
      if (wi <= nWords.length && qWords.every((qw, i) => {
        const start = nWords.slice(i).findIndex(w => w.startsWith(qw));
        return start !== -1;
      })) return 70;
    }
    // Any word starts with query
    if (nWords.some(w => w.startsWith(q))) return 60;
    // Typo tolerance: allow 1 character substitution/transposition for queries >= 4 chars
    if (q.length >= 4) {
      for (let i = 0; i < q.length; i++) {
        // deletion
        const del = q.slice(0, i) + q.slice(i + 1);
        if (n.includes(del)) return 40;
        // substitution with any char
        for (const c of "abcdefghijklmnopqrstuvwxyz") {
          const sub = q.slice(0, i) + c + q.slice(i + 1);
          if (n.includes(sub) && sub !== q) return 30;
        }
      }
    }
    return 0;
  }

  function setupSearch(side) {
    const input = root.querySelector(side === "A" ? "#sideASearch" : "#sideBSearch");
    const dropdown = root.querySelector(side === "A" ? "#sideADropdown" : "#sideBDropdown");
    const errorBox = root.querySelector("#errorBox");
    if (!input || !dropdown) return;
    if (input.__tradeSearchBound) return;
    input.__tradeSearchBound = true;

    input.addEventListener("input", async function () {
      const query = input.value.trim().toLowerCase();
      dropdown.innerHTML = "";
      dropdown.style.display = "none";
      dropdown.parentElement.classList.remove("dropdown-open");
      if (!query) return;

      try {
        await ensurePlayersLoaded();
      } catch (err) {
        console.error(err);
        if (errorBox) {
          errorBox.style.display = "block";
          errorBox.textContent = err.message || "Failed to load players.";
        }
        return;
      }

      const selected = getSidePlayers(side);
      const selectedPicks = getSidePicks(side);
      const leagueType = getLeagueType();

      const matches = allPlayers
        .filter(p => !selected.find(x => String(x.id) === String(p.id)))
        .filter(p => !selectedPicks.find(x => String(x.id) === String(p.id)))
        .map(p => {
          // Score against display name and search_name (normalized, no punctuation)
          const score = Math.max(
            fuzzyNameScore(p.name, query),
            fuzzyNameScore(p.search_name, query)
          );
          // Get player value for sorting
          const value = leagueType === "sf" ? (p.sf_value || p.value || 0) : (p.value || 0);
          return { p, score, value };
        })
        .filter(({ score }) => score > 0)
        .sort((a, b) => {
          // Primary sort by fuzzy score (better matches first)
          if (b.score !== a.score) return b.score - a.score;
          // Secondary sort by value (higher value first)
          return b.value - a.value;
        })
        .slice(0, 20)
        .map(({ p }) => p);

      const overallRankMap = buildOverallRankMap(allPlayers);
      if (!matches.length) return;

      matches.forEach(p => {
        const overallRank = overallRankMap.get(String(p.id));
        const item = buildDropdownItem(p, overallRank);

        item.addEventListener("click", () => {
          if (!selected.find(x => String(x.id) === String(p.id))) {
            if (p.position === "PICK") {
              const picks = getSidePicks(side);
              const isBucket = /_(early|mid|late)$/i.test(String(p.id));
              if (isBucket || !picks.find(x => String(x.id) === String(p.id))) {
                picks.push({ id: p.id, display: p.name });
              }
            } else {
              selected.push(p);
            }
            saveState();
            renderChips(side);
          }
          input.value = "";
          dropdown.style.display = "none";
          dropdown.parentElement.classList.remove("dropdown-open");
        });

        dropdown.appendChild(item);
      });

      dropdown.style.display = "block";
      dropdown.parentElement.classList.add("dropdown-open");
    });

    input.addEventListener("blur", function () {
      setTimeout(() => {
        dropdown.style.display = "none";
        dropdown.parentElement.classList.remove("dropdown-open");
      }, 150);
    });
  }

  function bindPickButtons() {
    root.querySelectorAll(".otc-add-pick-btn").forEach(btn => {
      bindOnce(btn, "tradeAddPick", "click", () => {
        const side = String(btn.getAttribute("data-side") || "").toUpperCase() === "B" ? "B" : "A";
        openPickPrompt(side);
      });
    });
  }

  function bindLeagueTypeControls() {
    const toggle = root.querySelector("#leagueTypeToggle");
    if (toggle) {
      bindOnce(toggle, "tradeLeagueTypeChange", "change", () => {
        onLeagueTypeChange();
      });
    }
  }

  function bindLeagueSizeControls() {
    const sel = root.querySelector("#leagueSizeSelect");
    if (sel) {
      bindOnce(sel, "tradeLeagueSizeChange", "change", async () => {
        await Promise.all([loadPlayerDeltas(), loadPlayerIndicators()]);
        renderChips("A");
        renderChips("B");
        recomputeTrade();
        renderAllPlayersList();
        loadTopMovers();
        if (isTargetsTabActive()) loadTradeTargets();
      });
    }
  }

  function bindScoringFormatControls() {
    const sel = root.querySelector("#scoringFormatSelect");
    if (sel) {
      bindOnce(sel, "tradeScoringFormatChange", "change", () => {
        renderChips("A");
        renderChips("B");
        recomputeTrade();
        renderAllPlayersList();
        if (isTargetsTabActive()) loadTradeTargets();
      });
    }
  }

  function bindViewerSideControls() {
    root.querySelectorAll('input[name="viewerSide"]').forEach(el => {
      bindOnce(el, "tradeViewerSideChange", "change", () => {
        syncViewerSideLabels();
        recomputeTrade();
      });
    });

    syncViewerSideLabels();
  }

  function bindShareButton() {
    const btn = root.querySelector("#shareTradeBtn");
    if (btn) {
      bindOnce(btn, "shareTradeClick", "click", () => {
        shareTradeToClipboard();
      });
    }
  }

  // ------------------------------------------------------------
  // Button handlers - lean; analyzeTrade() owns all state logic
  // ------------------------------------------------------------

  function bindClearTradeButton() {
    const btn = root.querySelector("#clearTradeBtn");
    if (!btn) return;
    // Only bind analyzeTrade if NOT in guest mode
    // (guest mode has its own handler to show login modal)
    const isGuest = root.querySelector("#isGuestMode")?.value === "true";
    if (isGuest) return;

    bindOnce(btn, "clearTradeBtn", "click", () => {
      const teamSelector = root.querySelector("#teamSelect");
      const teamSelectWrap = root.querySelector(".otc-summary-team-select");

      // Check if team is selected
      if (!teamSelector?.value) {
        // Shake the dropdown to indicate it needs to be filled
        if (teamSelectWrap) {
          teamSelectWrap.classList.add("shake");
          // Remove shake class after animation completes
          setTimeout(() => {
            teamSelectWrap.classList.remove("shake");
          }, 500);
        }
        // Focus the dropdown
        if (teamSelector) {
          teamSelector.focus();
        }
        return;
      }

      analyzeTrade();

      // Scroll to BR Trade Analyst section after triggering analysis
      const aiPanel = root.querySelector("#tradeAiPanel");
      if (aiPanel) {
        setTimeout(() => {
          aiPanel.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
            inline: 'nearest'
          });
        }, 100);
      }
    });
  }

  function bindAnalyzeTrade() {
    const btn = root.querySelector("#analyzeTradeBtn");
    const teamSelector = root.querySelector("#teamSelector");
    if (!btn) return;

    bindOnce(btn, "tradeAnalyzeBtn", "click", () => {
      const hasLeague = root.querySelector("#leagueIdInput")?.value;
      if (hasLeague && !teamSelector?.value) {
        showTeamSelectionPopup();
        return;
      }
      analyzeTrade();
    });

    btn.addEventListener("mouseenter", () => {
      const hasLeague = root.querySelector("#leagueIdInput")?.value;
      if (hasLeague && !teamSelector?.value) showTeamSelectionPopup();
    });
  }

  function bindSetupButton() {
    const setupBtn = root.querySelector("#setupTradeBtn");
    const usernameInput = root.querySelector("#username");
    const leagueSelect = root.querySelector("#league");
    if (!setupBtn) return;

    if (usernameInput) {
      usernameInput.addEventListener("input", debounce(async () => {
        const username = usernameInput.value.trim();
        if (username.length >= 3) {
          try {
            const response = await fetch(`/api/sleeper-user-leagues?username=${encodeURIComponent(username)}`);
            const data = await response.json();

            leagueSelect.innerHTML = '<option value="">Choose your league...</option>';

            if (data.ok && data.leagues) {
              data.leagues.forEach(league => {
                const option = document.createElement("option");
                option.value = league.league_id;
                option.textContent = league.name;
                leagueSelect.appendChild(option);
              });
            }
          } catch (err) {
            console.error("Failed to fetch leagues:", err);
          }
        }
      }, 500));
    }

    if (leagueSelect) {
      leagueSelect.addEventListener("change", updateSetupButtonState);
    }

    function updateSetupButtonState() {
      const username = usernameInput?.value?.trim() || "";
      const leagueId = leagueSelect?.value || "";
      const btn = root.querySelector("#setupTradeBtn");
      if (btn) {
        btn.disabled = !username || !leagueId;
        btn.style.cursor = username && leagueId ? "pointer" : "not-allowed";
      }
    }

    bindOnce(setupBtn, "setupTradeBtn", "click", async () => {
      const username = usernameInput?.value;
      const leagueId = leagueSelect?.value;

      if (!username || !leagueId) {
        alert("Please enter both username and select a league");
        return;
      }

      try {
        setupBtn.disabled = true;
        setupBtn.textContent = "Setting up...";
        setupBtn.style.cursor = "not-allowed";

        const response = await fetch("/api/setup-trade-context", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, league_id: leagueId }),
        });

        if (response.ok) {
          const platform = "sleeper";
          const season = new Date().getFullYear();
          window.location.href = `/${platform}/${season}/${leagueId}/trade`;
        } else {
          alert("Failed to setup trade context. Please try again.");
        }
      } catch (err) {
        console.error("Setup failed:", err);
        alert("Failed to setup trade context. Please try again.");
      } finally {
        setupBtn.disabled = false;
        setupBtn.textContent = "Setup Trade Calculator";
        setupBtn.style.cursor = "pointer";
      }
    });
  }

  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  function bindTeamSelector() {
    const selector = root.querySelector("#teamSelect");
    if (!selector) return;

    const leagueId = root.querySelector("#leagueIdInput")?.value || "";
    if (leagueId) {
      fetch(`/api/teams?league_id=${encodeURIComponent(leagueId)}&platform=sleeper`)
        .then(res => res.json())
        .then(teams => {
          selector.innerHTML = '<option value="">Select your team...</option>';

          teams.forEach(team => {
            const option = document.createElement("option");
            option.value = team.roster_id;
            option.textContent = team.team_name;
            option.setAttribute("data-username", team.username || "");
            selector.appendChild(option);
          });

          // Priority 1: viewer_roster_id injected server-side (most reliable)
          const injectedRosterId = root.querySelector("#viewerRosterIdInput")?.value || "";
          if (injectedRosterId) {
            selector.value = injectedRosterId;
          }

          // Priority 2: username match
          if (!selector.value) {
            const currentUsername = getCurrentUsername();
            if (currentUsername) {
              const userTeam = teams.find(
                team =>
                  team.username === currentUsername ||
                  team.team_name.toLowerCase().includes(currentUsername.toLowerCase())
              );
              if (userTeam) selector.value = userTeam.roster_id;
            }
          }

          // Priority 3: URL param / existing selection
          if (!selector.value) {
            const currentRosterId = getCurrentRosterId();
            if (currentRosterId) selector.value = currentRosterId;
          }

          if (selector.value && root.querySelector("#targetsTabContent.is-active")) loadTradeTargets();
          updateAnalyzeButtonState();
        })
        .catch(err => {
          console.error("Failed to load teams:", err);
        });
    }

    bindOnce(selector, "teamSelectorChange", "change", () => {
      const selectedRosterId = selector.value;
      if (selectedRosterId) {
        if (root.querySelector("#targetsTabContent.is-active")) loadTradeTargets();
        fetch("/api/set-viewer-roster", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ roster_id: selectedRosterId }),
        }).then(res => {
          if (res.ok) {
            // Team changed - reset to empty state so user re-triggers analysis
            const loadingState = root.querySelector("#aiLoadingState");
            const emptyState = root.querySelector("#aiEmptyState");
            const resultState = root.querySelector("#aiAnalysisResult");
            if (loadingState) loadingState.style.display = "none";
            if (resultState) resultState.style.display = "none";
            if (emptyState) {
              emptyState.style.display = "block";
              emptyState.innerHTML = `
                <div class="otc-ai-empty-title">Team Selected</div>
                <div class="otc-ai-empty-sub">
                  Now click Analyze Trade to generate analysis for
                  ${selector.options[selector.selectedIndex].text}.
                </div>`;
            }
          }
        });
      }

      updateAnalyzeButtonState();
    });
  }

  function updateAnalyzeButtonState() {
    const selector = root.querySelector("#teamSelect");
    const analyzeBtn = root.querySelector("#clearTradeBtn");
    const hasLeague = root.querySelector("#leagueIdInput")?.value;

    if (!analyzeBtn) return;

    if (!hasLeague) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Trade";
      analyzeBtn.classList.remove("otc-btn-disabled");
      analyzeBtn.removeAttribute("data-tooltip");
      return;
    }

    if (!selector) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyze Trade";
      analyzeBtn.classList.add("otc-btn-disabled");
      analyzeBtn.setAttribute("data-tooltip", "Please select your team first to analyze trades");
      return;
    }

    const hasSelection = selector.value && selector.value !== "";
    if (hasSelection) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Trade";
      analyzeBtn.classList.remove("otc-btn-disabled");
      analyzeBtn.removeAttribute("data-tooltip");
    } else {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyze Trade";
      analyzeBtn.classList.add("otc-btn-disabled");
      analyzeBtn.setAttribute("data-tooltip", "Please select your team first to analyze trades");
    }
  }

  function showTeamSelectionPopup() {
    hideTeamSelectionPopup();

    const popup = document.createElement("div");
    popup.id = "team-selection-popup";
    popup.className = "otc-team-popup";
    popup.innerHTML = `
      <div class="otc-popup-content">
        <div class="otc-popup-header">
          <h3>Select Your Team</h3>
          <button class="otc-popup-close" onclick="hideTeamSelectionPopup()">×</button>
        </div>
        <div class="otc-popup-body">
          <p>Please select your team from the dropdown above to enable personalized AI analysis.</p>
          <div class="otc-popup-arrow"></div>
        </div>
      </div>`;

    document.body.appendChild(popup);

    const analyzeBtn = root.querySelector("#analyzeTradeBtn");
    if (analyzeBtn) {
      const btnRect = analyzeBtn.getBoundingClientRect();
      const popupRect = popup.getBoundingClientRect();
      popup.style.position = "fixed";
      popup.style.top = btnRect.top - popupRect.height - 10 + "px";
      popup.style.left = btnRect.left + btnRect.width / 2 - popupRect.width / 2 + "px";
      popup.style.zIndex = "1000";
    }

    setTimeout(hideTeamSelectionPopup, 3000);
  }

  function hideTeamSelectionPopup() {
    const popup = document.getElementById("team-selection-popup");
    if (popup) popup.remove();
  }

  window.hideTeamSelectionPopup = hideTeamSelectionPopup;

  document.addEventListener("click", function (event) {
    const popup = document.getElementById("team-selection-popup");
    if (popup && !popup.contains(event.target) && !event.target.closest("#clearTradeBtn")) {
      hideTeamSelectionPopup();
    }
  });

  // Info tooltip handler
  const infoBtn = root.querySelector("#otcInfoBtn");
  const infoTooltip = root.querySelector("#otcInfoTooltip");
  const infoWrapper = root.querySelector(".otc-info-tooltip-wrapper");

  if (infoBtn && infoTooltip && infoWrapper) {
    // Show tooltip on hover
    bindOnce(infoBtn, "otcInfoMouseenter", "mouseenter", (e) => {
      infoTooltip.style.display = "block";
    });

    // Hide tooltip when mouse leaves
    bindOnce(infoBtn, "otcInfoMouseleave", "mouseleave", (e) => {
      infoTooltip.style.display = "none";
    });

    // Also hide when mouse leaves the tooltip wrapper
    bindOnce(infoWrapper, "otcWrapperMouseleave", "mouseleave", (e) => {
      if (!infoWrapper.contains(e.relatedTarget)) {
        infoTooltip.style.display = "none";
      }
    });

    // Fetch trade count from database
    fetch('/api/trade-count')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        const tradeCountElement = document.getElementById('tradeCount');
        if (tradeCountElement && data.count !== undefined) {
          tradeCountElement.textContent = data.count.toLocaleString();
        }
      })
      .catch(error => {
        const tradeCountElement = document.getElementById('tradeCount');
        if (tradeCountElement) {
          tradeCountElement.textContent = '150,000+';
        }
      });
  }

  root.querySelectorAll(".pos-filter").forEach(btn => {
    bindOnce(btn, "tradePosFilterClick", "click", () => {
      const pos = btn.getAttribute("data-pos") || "ALL";
      setPosFilter(pos);
    });
  });

  Promise.allSettled([
    ensurePlayersLoaded(),
    loadTopMovers(),
    loadPlayerDeltas(),
    loadPlayerIndicators(),
  ]).then(() => {
    setupSearch("A");
    setupSearch("B");
    bindLeagueTypeControls();
    bindLeagueSizeControls();
    bindScoringFormatControls();
    bindViewerSideControls();
    bindAnalyzeTrade();
    bindTeamSelector();
    bindSetupButton();
    bindClearTradeButton();
    bindShareButton();
    initMoversBreakoutsTabs();

    // Re-verify premium status client-side so stale page renders don't lock out subscribers
    (async function checkPremiumState() {
      const leagueId = root.querySelector("#leagueIdInput")?.value || "";
      if (!leagueId) return;
      try {
        const res = await fetch(`/api/subscription-status?league_id=${encodeURIComponent(leagueId)}`, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        if (data.has_premium) {
          const premiumInput = root.querySelector("#otcHasPremium");
          if (premiumInput) premiumInput.value = "true";
          const lockIcon = root.querySelector("#targetsLockIcon");
          if (lockIcon) lockIcon.style.display = "none";
        }
      } catch (_) {}
    })();

    // Try to load trade from URL first, otherwise load from localStorage
    const loadedFromURL = loadTradeFromURL();
    if (!loadedFromURL) {
      loadState();
    }

    updateAnalyzeButtonState();
    syncEmptyState("A");
    syncEmptyState("B");
    recomputeTrade();

    // Targets tab loads lazily when opened - no eager fetch needed
  });
};

// Global utility functions
function getCurrentUsername() {
  const urlParams = new URLSearchParams(window.location.search);
  return (
    urlParams.get("username") ||
    sessionStorage.getItem("trade_username") ||
    localStorage.getItem("trade_username") ||
    ""
  );
}

function getCurrentRosterId() {
  const urlParams = new URLSearchParams(window.location.search);
  return (
    urlParams.get("viewer_roster_id") ||
    document.querySelector("#viewerRosterIdInput")?.value ||
    document.querySelector("#teamSelect")?.value ||
    ""
  );
}

// ------------------------------------------------------------
// History Page Season Recap
// ------------------------------------------------------------
async function generateSeasonRecap() {
  const root = document;
  const errorBox = root.querySelector("#errorBox");
  const loadingState = root.querySelector("#aiLoadingState");
  const emptyState = root.querySelector("#aiEmptyState");
  const resultState = root.querySelector("#aiAnalysisResult");
  const teamDropdown = root.querySelector("#recapTeamDropdown");
  const generateBtn = root.querySelector("#generateRecapBtn");
  
  const selectedTeam = teamDropdown?.value;
  if (!selectedTeam) {
    alert("Please select a team first");
    return;
  }
  
  // Show loading, hide everything else
  if (loadingState) loadingState.style.display = "block";
  if (emptyState) emptyState.style.display = "none";
  if (resultState) resultState.style.display = "none";

  // Yield to browser so the loading state actually paints before fetch starts
  await new Promise(resolve => requestAnimationFrame(resolve));
  
  try {
    const leagueId = root.querySelector("#leagueIdInput")?.value || "";
    const season = root.querySelector("#seasonInput")?.value || "";
    const resolvedLeagueId = root.querySelector("#resolvedLeagueIdInput")?.value || leagueId;
    const historySeasonSelect = root.querySelector("#history-season-select");
    let historySeason = season;
    
    // Extract season number from the selected option value (which might be a URL)
    if (historySeasonSelect && historySeasonSelect.value) {
      const seasonMatch = historySeasonSelect.value.match(/history_season=(\d+)/);
      if (seasonMatch) {
        historySeason = seasonMatch[1];
      }
    }
    
    const url = `/api/history/ai-recap?league_id=${encodeURIComponent(resolvedLeagueId)}&season=${encodeURIComponent(historySeason)}&roster_id=${encodeURIComponent(selectedTeam)}&base_season=${encodeURIComponent(season)}`;

    const res = await fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    
    if (!res.ok) {
      throw new Error(`Failed to generate recap (${res.status})`);
    }
    
    const data = await res.json();
    
    // Hide loading, show result panel
    if (loadingState) loadingState.style.display = "none";
    if (emptyState) emptyState.style.display = "none";
    if (resultState) {
      resultState.style.display = "block";
      if (data.html) {
        resultState.innerHTML = data.html;
      } else {
        resultState.innerHTML = `
          <div class="otc-ai-empty">
            <div class="otc-ai-empty-title">No AI take yet</div>
            <div class="otc-ai-empty-sub">Unable to generate season analysis.</div>
          </div>`;
      }
    }

    if (errorBox) {
      errorBox.style.display = "none";
      errorBox.textContent = "";
    }
    
  } catch (err) {
    console.error("[recap] Error:", err);
    
    // Show error state
    if (loadingState) loadingState.style.display = "none";
    if (emptyState) emptyState.style.display = "none";
    if (resultState) {
      resultState.style.display = "block";
      resultState.innerHTML = `
        <div class="otc-ai-empty">
          <div class="otc-ai-empty-title">Analysis failed</div>
          <div class="otc-ai-empty-sub">There was an error generating the AI analysis.</div>
        </div>`;
    }

    if (errorBox) {
      errorBox.style.display = "block";
      errorBox.textContent = err.message || "Failed to analyze season.";
    }
  } finally {
    // Reset button
    if (generateBtn) generateBtn.disabled = false;
    if (generateBtn) generateBtn.textContent = "Generate Recap";
  }
}

function bindRecapTeamSelector() {
  const root = document;
  const teamDropdown = root.querySelector("#recapTeamDropdown");
  const generateBtn = root.querySelector("#generateRecapBtn");
  
  if (!teamDropdown || !generateBtn) return;
  
  const leagueId = root.querySelector("#leagueIdInput")?.value || "";
  const season = root.querySelector("#seasonInput")?.value || "";
  const resolvedLeagueId = root.querySelector("#resolvedLeagueIdInput")?.value || leagueId;
  const historySeasonSelect = root.querySelector("#history-season-select");
  let historySeason = season;
  
  // Extract season number from the selected option value (which might be a URL)
  if (historySeasonSelect && historySeasonSelect.value) {
    const seasonMatch = historySeasonSelect.value.match(/history_season=(\d+)/);
    if (seasonMatch) {
      historySeason = seasonMatch[1];
    }
  }
  
  if (resolvedLeagueId) {
    const teamsUrl = `/api/teams?league_id=${encodeURIComponent(resolvedLeagueId)}&platform=sleeper&season=${encodeURIComponent(historySeason)}`;

    fetch(teamsUrl)
      .then(res => res.json())
      .then(teams => {
        teamDropdown.innerHTML = '<option value="">Choose a team for recap...</option>';
        
        teams.forEach(team => {
          const option = document.createElement("option");
          option.value = team.roster_id;
          option.textContent = team.team_name;
          teamDropdown.appendChild(option);
        });
        
        // Auto-select current user's team if available
        const currentUsername = getCurrentUsername();
        if (currentUsername) {
          const userTeam = teams.find(
            team =>
              team.username === currentUsername ||
              team.team_name.toLowerCase().includes(currentUsername.toLowerCase())
          );
          if (userTeam) {
            teamDropdown.value = userTeam.roster_id;
            generateBtn.disabled = false;
          }
        }
      })
      .catch(err => {
        console.error("Failed to load recap teams:", err);
        teamDropdown.innerHTML = '<option value="">Error loading teams</option>';
      });
  }
  
  // Enable/disable button based on selection
  teamDropdown.addEventListener("change", () => {
    generateBtn.disabled = !teamDropdown.value;
  });
  
  // Bind generate button
  generateBtn.addEventListener("click", generateSeasonRecap);
}

function recapPageExists(root = document) {
  return !!root.querySelector("#recapTeamDropdown");
}

function initRecapPage(root = document) {
  if (!recapPageExists(root)) return;
  
  bindRecapTeamSelector();
}

// ------------------------------------------------------------
// Master Initializer
// ------------------------------------------------------------

window.initPageRoot = function initPageRoot(root = document) {
  initManagerPills(root);
  initCardTabs(root);
  initPlayoffOdds(root);
  initTeamTabs(root);
  initStandingsSort(root);

  bindGlobalCarouselHandlersOnce();
  window.resetMatchupCarousels?.(root);

  if (root.querySelector('[data-page="graphs"]')) {
    window.resizeAllPlotly?.(root);
  }
  if (tradePageExists(root)) {
    window.initTradePage?.(root);
  }
  if (recapPageExists(root)) {
    initRecapPage(root);
  }
  
  // Setup fun awards grid if present
  setupFunAwardsGrid();
};

bindOnce(document, "domContentLoadedInit", "DOMContentLoaded", () => {
  // Force scroll to top
  window.scrollTo(0, 0);

  // Initialize page
  window.initPageRoot(document);

  // Force scroll to top again after initialization
  requestAnimationFrame(() => {
    window.scrollTo(0, 0);
  });
});

// ------------------------------------------------------------
// Refresh Button Handler (platform/season aware)
// ------------------------------------------------------------
(function () {
  const refreshBtn = document.getElementById("refreshBtn");
  if (!refreshBtn) return;

  function readCtxFromRefreshBtn(btn) {
    const league = (btn.dataset.league || "").trim();
    const page = (btn.dataset.page || "").trim();
    const platform = (btn.dataset.platform || "").trim().toLowerCase();
    const season = parseInt(btn.dataset.season || "0", 10);
    return { league, page, platform, season };
  }

  bindOnce(refreshBtn, "refreshBtnClick", "click", async () => {
    const { league, page, platform, season } = readCtxFromRefreshBtn(refreshBtn);

    if (!league || !page || !platform || !season) {
      console.error("Missing league/page/platform/season for refresh.", { league, page, platform, season });
      window.location.reload();
      return;
    }

    refreshBtn.disabled = true;
    refreshBtn.classList.add("refresh-spinner");

    try {
      const res = await fetch("/api/refresh-page", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ league_id: league, page, platform, season }),
      });

      const data = await res.json().catch(() => null);

      if (!res.ok || !data || !data.ok) {
        console.error("Refresh failed:", data?.error || res.statusText);
        window.location.reload();
        return;
      }

      const root = document.getElementById("page-root");

      if (root && data.body_html) {
        root.innerHTML = data.body_html;
        requestAnimationFrame(() => window.initPageRoot?.(root));
      } else {
        window.location.reload();
      }
    } catch (err) {
      console.error("Error during refresh:", err);
      window.location.reload();
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.classList.remove("refresh-spinner");
    }
  });
})();

document.addEventListener("DOMContentLoaded", () => {
  const platformBtns = document.querySelectorAll(".platform-btn");
  const sleeperFlow = document.getElementById("sleeperFlow");
  const espnFlow = document.getElementById("espnFlow");
  const sleeperHint = document.getElementById("sleeperHint");
  const lookupBtn = document.getElementById("lookupBtn");
  const usernameInput = document.getElementById("username");
  const leagueSelect = document.getElementById("league");
  const leagueSelectWrap = document.getElementById("leagueSelectWrap");
  const generateWrap = document.getElementById("generateWrap");
  const errorBox = document.getElementById("lookupError");
  const formPlatform = document.getElementById("formPlatform");

  const espnLeagueIdInput = document.getElementById("espnLeagueIdInput");
  const espnTeamName = document.getElementById("espnTeamName");
  const espnSubmitBtn = document.getElementById("espnSubmitBtn");
  const espnErrorBox = document.getElementById("espnError");

  if (!platformBtns.length) return;

  let currentPlatform = "sleeper";

  function switchPlatform(platform) {
    currentPlatform = platform;
    platformBtns.forEach(b => b.classList.remove("active"));
    const activeBtn = [...platformBtns].find(b => b.dataset.platform === platform);
    if (activeBtn) activeBtn.classList.add("active");

    if (formPlatform) formPlatform.value = platform;

    // Reset shared state
    if (leagueSelectWrap) leagueSelectWrap.style.display = "none";
    if (generateWrap) generateWrap.style.display = "none";
    if (errorBox) errorBox.style.display = "none";

    if (platform === "espn") {
      if (sleeperFlow) sleeperFlow.style.display = "none";
      if (espnFlow) espnFlow.style.display = "block";
      if (sleeperHint) sleeperHint.style.display = "none";
    } else {
      if (sleeperFlow) sleeperFlow.style.display = "block";
      if (espnFlow) espnFlow.style.display = "none";
      if (sleeperHint) sleeperHint.style.display = "";
    }
  }

  // Platform switching
  platformBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      switchPlatform(btn.dataset.platform);
    });
  });

  // Sleeper username input
  if (usernameInput) {
    usernameInput.addEventListener("input", () => {
      const formUsername = document.getElementById("formUsername");
      if (formUsername) formUsername.value = usernameInput.value.trim();
    });
  }

  // Save viewer to localStorage on form submit so returning users can skip re-entry
  const leagueSelectFormEl = document.getElementById("leagueSelectForm");
  if (leagueSelectFormEl) {
    leagueSelectFormEl.addEventListener("submit", (e) => {
      const lId = leagueSelect?.value;
      const uname = usernameInput?.value.trim();
      const platform = formPlatform?.value || "sleeper";
      const seasonVal = leagueSelectFormEl.querySelector('input[name="season"]')?.value;
      if (lId && uname) {
        localStorage.setItem("saved_viewer", JSON.stringify({
          username: uname,
          league_id: lId,
          platform: platform,
          season: seasonVal || new Date().getFullYear(),
          ts: Date.now()
        }));
      }

      // Show full-page loading overlay while the server builds the dashboard
      const overlay = document.getElementById("dashboardLoadingOverlay");
      const submitBtn = leagueSelectFormEl.querySelector('button[type="submit"]');
      if (overlay) overlay.style.display = "flex";
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Building Dashboard…";
      }
    });
  }

  // Show "Continue as X" CTA for returning users
  const saved = JSON.parse(localStorage.getItem("saved_viewer") || "null");
  if (saved?.league_id && saved?.username) {
    const platform = saved.platform || "sleeper";
    const season = saved.season || new Date().getFullYear();
    const dashboardUrl = `/${platform}/${season}/${saved.league_id}/dashboard`;

    const cta = document.createElement("div");
    cta.className = "saved-viewer-cta";
    cta.innerHTML = `
      <div class="saved-viewer-info">
        <span class="saved-viewer-label">Welcome back!</span>
        <form method="POST" action="/set-viewer" style="display:inline">
          <input type="hidden" name="league_id" value="${saved.league_id}">
          <input type="hidden" name="username" value="${saved.username}">
          <input type="hidden" name="platform" value="${platform}">
          <input type="hidden" name="season" value="${season}">
          <button type="submit" class="saved-viewer-btn">Continue as <strong>${saved.username}</strong></button>
        </form>
      </div>
      <button type="button" class="saved-viewer-dismiss" aria-label="Dismiss">×</button>
    `;

    const homeCard = document.querySelector(".home-card");
    if (homeCard) homeCard.insertAdjacentElement("afterbegin", cta);

    cta.querySelector(".saved-viewer-dismiss")?.addEventListener("click", () => {
      localStorage.removeItem("saved_viewer");
      cta.remove();
    });
  }

  // Sleeper lookup
  if (lookupBtn) {
    lookupBtn.addEventListener("click", async () => {
      const username = usernameInput?.value.trim();
      if (!username) {
        errorBox.textContent = "Enter a Sleeper username.";
        errorBox.style.display = "block";
        leagueSelectWrap.style.display = "none";
        generateWrap.style.display = "none";
        return;
      }

      errorBox.style.display = "none";
      lookupBtn.disabled = true;
      lookupBtn.textContent = "Loading...";

      try {
        const res = await fetch(`/api/sleeper-user-leagues?username=${encodeURIComponent(username)}`);
        const data = await res.json();

        if (!res.ok || !data.ok) {
          throw new Error(data.error || "Unable to load leagues.");
        }

        leagueSelect.innerHTML = `<option value="">Select a league</option>`;

        for (const league of data.leagues || []) {
          const option = document.createElement("option");
          option.value = league.league_id;
          option.textContent = league.label;
          leagueSelect.appendChild(option);
        }

        if (!data.leagues || !data.leagues.length) {
          throw new Error("No leagues found for that user this season.");
        }

        leagueSelectWrap.style.display = "block";
        generateWrap.style.display = "block";
      } catch (err) {
        errorBox.textContent = err.message || "Unable to load leagues.";
        errorBox.style.display = "block";
        leagueSelectWrap.style.display = "none";
        generateWrap.style.display = "none";
      } finally {
        lookupBtn.disabled = false;
        lookupBtn.textContent = "Find My Leagues";
      }
    });
  }

  if (espnSubmitBtn) {
    espnSubmitBtn.addEventListener("click", async () => {
      const leagueId = espnLeagueIdInput?.value.trim();
      if (!leagueId || !/^\d+$/.test(leagueId)) {
        if (espnErrorBox) {
          espnErrorBox.textContent = "Enter a valid ESPN League ID (numbers only).";
          espnErrorBox.style.display = "block";
        }
        return;
      }

      if (espnErrorBox) espnErrorBox.style.display = "none";
      espnSubmitBtn.disabled = true;
      espnSubmitBtn.textContent = "Validating...";

      try {
        const res = await fetch(`/api/espn-validate-league?league_id=${encodeURIComponent(leagueId)}`);
        const data = await res.json();

        if (!res.ok || !data.ok) {
          throw new Error(data.error || "Unable to load ESPN league.");
        }

        // Populate the shared league select and submit the form
        if (leagueSelect) {
          leagueSelect.innerHTML = `<option value="${leagueId}" selected>${data.league?.name || "ESPN League"}</option>`;
        }
        if (formPlatform) formPlatform.value = "espn";

        const teamName = espnTeamName?.value.trim() || "";
        const formUsername = document.getElementById("formUsername");
        if (formUsername) formUsername.value = teamName;

        // Save for the "Continue as" returning-user CTA
        const seasonVal = document.querySelector('input[name="season"]')?.value || new Date().getFullYear();
        if (teamName) {
          localStorage.setItem("saved_viewer", JSON.stringify({
            username: teamName,
            league_id: leagueId,
            platform: "espn",
            season: seasonVal,
            ts: Date.now(),
          }));
        }

        document.getElementById("leagueSelectForm")?.submit();
      } catch (err) {
        if (espnErrorBox) {
          espnErrorBox.textContent = err.message || "Unable to load ESPN league.";
          espnErrorBox.style.display = "block";
        }
        espnSubmitBtn.disabled = false;
        espnSubmitBtn.textContent = "Find My League";
      }
    });
  }

});

// Trade Calculator Login Modal
document.addEventListener("DOMContentLoaded", () => {
  const isGuestMode = document.getElementById("isGuestMode")?.value === "true";
  if (!isGuestMode) return;

  const analyzeBtn = document.getElementById("clearTradeBtn");
  const modal = document.getElementById("tradeLoginModal");
  const closeBtn = document.getElementById("closeLoginModal");
  const overlay = modal?.querySelector(".trade-login-overlay");
  const usernameInput = document.getElementById("tradeUsername");
  const lookupBtn = document.getElementById("tradeLookupBtn");
  const leagueSelect = document.getElementById("tradeLeagueSelect");
  const leagueWrap = document.getElementById("tradeLeagueSelectWrap");
  const goWrap = document.getElementById("tradeGoWrap");
  const goBtn = document.getElementById("tradeGoBtn");
  const errorBox = document.getElementById("tradeLookupError");
  const seasonInput = document.getElementById("seasonInput");

  if (!analyzeBtn || !modal) return;

  function openModal() {
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    modal.style.display = "none";
    document.body.style.overflow = "";
  }

  analyzeBtn.addEventListener("click", (e) => {
    openModal();
  });
  closeBtn?.addEventListener("click", closeModal);
  overlay?.addEventListener("click", closeModal);

  // Add click handler for guest "Log in to see more" links
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("otc-guest-link")) {
      e.preventDefault();
      openModal();
    }
  });

  lookupBtn?.addEventListener("click", async () => {
    const username = usernameInput?.value.trim();
    if (!username) {
      errorBox.textContent = "Enter a Sleeper username.";
      errorBox.style.display = "block";
      leagueWrap.style.display = "none";
      goWrap.style.display = "none";
      return;
    }

    errorBox.style.display = "none";
    lookupBtn.disabled = true;
    lookupBtn.textContent = "Loading...";

    try {
      const res = await fetch(`/api/sleeper-user-leagues?username=${encodeURIComponent(username)}`);
      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "Unable to load leagues.");
      }

      leagueSelect.innerHTML = `<option value="">Select a league</option>`;

      for (const league of data.leagues || []) {
        const option = document.createElement("option");
        option.value = league.league_id;
        option.textContent = league.label;
        leagueSelect.appendChild(option);
      }

      if (!data.leagues || !data.leagues.length) {
        throw new Error("No leagues found for that user this season.");
      }

      leagueWrap.style.display = "block";
      goWrap.style.display = "block";
    } catch (err) {
      errorBox.textContent = err.message || "Unable to load leagues.";
      errorBox.style.display = "block";
      leagueWrap.style.display = "none";
      goWrap.style.display = "none";
    } finally {
      lookupBtn.disabled = false;
      lookupBtn.textContent = "Find My Leagues";
    }
  });

  goBtn?.addEventListener("click", () => {
    const leagueId = leagueSelect?.value;
    const season = seasonInput?.value || new Date().getFullYear();

    if (!leagueId) {
      errorBox.textContent = "Please select a league.";
      errorBox.style.display = "block";
      return;
    }

    window.location.href = `/sleeper/${season}/${leagueId}/trade`;
  });

  leagueSelect?.addEventListener("change", () => {
    if (leagueSelect.value) {
      errorBox.style.display = "none";
    }
  });
});

// ------------------------------------------------------------
// Changelog Bell
// ------------------------------------------------------------

// Global variable to track notification state
let hasNewNotifications = false;

// Global function for notification dots (accessible from settings dropdown)
function setChangelogDot(hasNew, showSettingsDot = false) {
  // Update the global notification state
  hasNewNotifications = hasNew;

  // Update gear dot - hide it when settings dropdown is opened and we want to show the dot there instead
  const gearDot = document.getElementById("gearDot");
  if (gearDot) {
    gearDot.style.display = (hasNew && !showSettingsDot) ? "block" : "none";
  }
  
  // Only show settings notification dot when explicitly requested (when settings is opened)
  const settingsDot = document.getElementById("settingsNotifDot");
  if (settingsDot) {
    settingsDot.style.display = (hasNew && showSettingsDot) ? "block" : "none";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const bellWrapper = document.querySelector(".changelog-bell-wrapper");
  const bellBtn = document.getElementById("changelogBell");
  const dropdown = document.getElementById("changelogDropdown");
  const dot = document.querySelector(".changelog-dot");

  if (!bellBtn || !dropdown || !dot) return;

  let changelogData = [];
  let isDropdownOpen = false;

  // Fetch changelog and initialize
  async function initChangelog() {
    try {
      const res = await fetch("/api/changelog");
      if (!res.ok) throw new Error("Failed to load changelog");

      changelogData = await res.json();

      if (!changelogData || changelogData.length === 0) return;

      // Detect if user is logged in (check for league context in URL)
      const pathParts = window.location.pathname.split('/').filter(p => p);
      const isLoggedIn = pathParts.length >= 3 && pathParts[0] && !isNaN(pathParts[1]);

      // Filter out history page entries if not logged in
      if (!isLoggedIn) {
        changelogData = changelogData.filter(entry => !entry.link?.includes('/history'));
      }

      if (changelogData.length === 0) return;

      // Check if we should show the red dot
      const latestDate = changelogData[0].date;
      const lastSeen = localStorage.getItem("changelog_last_seen");


      if (!lastSeen || latestDate > lastSeen) {
        setChangelogDot(true);
      }

      // Build dropdown HTML
      buildDropdown();
    } catch (err) {
      console.error("[changelog] Failed to load:", err);
    }
  }

  // Build the dropdown HTML
  function buildDropdown() {
    // Detect league context from URL
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const isLoggedIn = pathParts.length >= 3 && pathParts[0] && !isNaN(pathParts[1]);
    const leaguePrefix = isLoggedIn ? `/${pathParts[0]}/${pathParts[1]}/${pathParts[2]}` : '';

    const entries = changelogData.slice(0, 5).map(entry => {
      const tagClass = `changelog-tag changelog-tag-${entry.tag}`;
      const formattedDate = formatDate(entry.date);
      let link = entry.link || "#";

      // If logged in and link starts with /, prepend league context
      if (isLoggedIn && link.startsWith('/')) {
        link = leaguePrefix + link;
      }

      return `
        <a href="${link}" class="changelog-entry" data-link="${link}">
          <div class="changelog-entry-top">
            <span class="${tagClass}">${entry.tag}</span>
            <span class="changelog-entry-date">${formattedDate}</span>
          </div>
          <div class="changelog-entry-text">${entry.text}</div>
        </a>
      `;
    }).join("");

    dropdown.innerHTML = `
      <div class="changelog-dropdown-header">Recent Updates</div>
      ${entries}
    `;
  }

  // Format date (e.g., "2026-03-26" -> "Mar 26")
  function formatDate(dateStr) {
    const date = new Date(dateStr);
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return `${months[date.getMonth()]} ${date.getDate()}`;
  }

  // Toggle dropdown
  function toggleDropdown() {
    isDropdownOpen = !isDropdownOpen;

    if (isDropdownOpen) {
      dropdown.style.display = "block";

      // Mark as seen - hide dots on both bell and gear
      if (changelogData && changelogData.length > 0) {
        const latestDate = changelogData[0].date;
        localStorage.setItem("changelog_last_seen", latestDate);
setChangelogDot(false);
      }
    } else {
      dropdown.style.display = "none";
    }
  }

  // Close dropdown
  function closeDropdown() {
    if (isDropdownOpen) {
      isDropdownOpen = false;
      dropdown.style.display = "none";
    }
  }

  // Bind bell click
  bindOnce(bellBtn, "changelogBellClick", "click", (e) => {
    e.stopPropagation();
    toggleDropdown();
  });

  // Close on click outside
  document.addEventListener("click", (e) => {
    if (bellWrapper && !bellWrapper.contains(e.target)) {
      closeDropdown();
    }
  });

  // Initialize
  initChangelog();
});

// Settings Gear Dropdown
document.addEventListener("DOMContentLoaded", () => {
  const gearWrapper = document.querySelector(".settings-gear-wrapper");
  const gearBtn = document.getElementById("settingsGearBtn");
  const dropdown = document.getElementById("settingsDropdown");

  if (!gearBtn || !dropdown) return;

  let isDropdownOpen = false;

  function toggleDropdown() {
    isDropdownOpen = !isDropdownOpen;
    dropdown.style.display = isDropdownOpen ? "block" : "none";
    

    // When opening settings, show bell dot if there are new notifications
    if (isDropdownOpen) {
      // Move notification dot from gear to notification button if there are new notifications
      if (hasNewNotifications) {
        setChangelogDot(true, true); // Keep gear dot hidden, show settings notification dot
      }
    }
  }

  function closeDropdown() {
    if (isDropdownOpen) {
      isDropdownOpen = false;
      dropdown.style.display = "none";
      // Hide settings notification dot when dropdown closes
      setChangelogDot(true, false);
    }
  }

  gearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleDropdown();
  });

  document.addEventListener("click", (e) => {
    if (gearWrapper && !gearWrapper.contains(e.target)) {
      closeDropdown();
    }
  });

  // Refresh button - close dropdown after click
  const refreshBtn = document.getElementById("refreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      closeDropdown();
    });
  }

  // Changelog button triggers changelog dropdown
  const settingsChangelogBtn = document.getElementById("settingsChangelogBtn");
  const changelogBellBtn = document.getElementById("changelogBell");
  if (settingsChangelogBtn && changelogBellBtn) {
    settingsChangelogBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeDropdown();
      changelogBellBtn.click();
    });
  }

  
  // Prevent dropdown from closing when clicking inside (except on links)
  if (dropdown) {
    dropdown.addEventListener("click", (e) => {
      // Let links work normally, but stop propagation for other elements
      if (e.target.tagName !== 'A' && !e.target.closest('a')) {
        e.stopPropagation();
      }
    });
  }

  // Clear saved viewer on logout so the "Continue as X" CTA doesn't reappear
  const logoutLink = dropdown?.querySelector('a[href="/logout"]');
  if (logoutLink) {
    logoutLink.addEventListener("click", () => {
      localStorage.removeItem("saved_viewer");
    });
  }
});

// League switcher functionality
document.addEventListener('DOMContentLoaded', function() {
  // Helper function to show full-screen loading overlay
  function showFullscreenLoading(message = 'Loading...') {
    // Remove any existing overlay
    const existingOverlay = document.getElementById('fullscreenLoadingOverlay');
    if (existingOverlay) {
      existingOverlay.remove();
    }

    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'fullscreenLoadingOverlay';
    overlay.className = 'fullscreen-loading-overlay';
    overlay.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="fullscreen-loading-text">${message}</div>
    `;
    document.body.appendChild(overlay);
  }

  // Handle league switcher
  const leagueSwitcher = document.getElementById('leagueSwitcher');

  if (leagueSwitcher) {
    const currentLeagueId = leagueSwitcher.getAttribute('data-current-league');
    const currentPlatform = leagueSwitcher.getAttribute('data-current-platform');
    const currentSeason = leagueSwitcher.getAttribute('data-current-season');
    const username = leagueSwitcher.getAttribute('data-current-username');

    // Fetch user leagues
    fetch('/api/sleeper-user-leagues?username=' + username)
      .then(res => res.json())
      .then(data => {
        // Handle error response
        if (!data.ok) {
          console.warn('League switcher API error:', data.error || 'Unknown error');
          leagueSwitcher.innerHTML = '<option value="">No leagues available</option>';
          return;
        }

        // Handle success
        if (data.leagues && data.leagues.length > 0) {
          leagueSwitcher.innerHTML = '';
          data.leagues.forEach(league => {
            const option = document.createElement('option');
            option.value = league.league_id;
            option.textContent = league.label;
            if (league.league_id === currentLeagueId) {
              option.selected = true;
            }
            leagueSwitcher.appendChild(option);
          });
        } else {
          leagueSwitcher.innerHTML = '<option value="">No leagues found</option>';
        }
      })
      .catch(err => {
        console.error('Failed to load leagues:', err);
        leagueSwitcher.innerHTML = '<option value="">Error loading leagues</option>';
      });

    // Handle league change
    leagueSwitcher.addEventListener('change', function() {
      const selectedLeagueId = this.value;
      if (selectedLeagueId && selectedLeagueId !== currentLeagueId) {
        // Show full-screen loading overlay
        showFullscreenLoading('Switching leagues...');

        // Get current page from URL
        const pathParts = window.location.pathname.split('/');
        const currentPage = pathParts[pathParts.length - 1] || 'dashboard';

        // Redirect to new league with same page
        window.location.href = `/${currentPlatform}/${currentSeason}/${selectedLeagueId}/${currentPage}`;
      }
    });
  }
});

// Mobile nav toggle functionality
document.addEventListener('DOMContentLoaded', function() {
  const navToggle = document.getElementById('navToggle');
  const navPillsContainer = document.querySelector('.nav-pills-container');

  if (navToggle && navPillsContainer) {
    navToggle.addEventListener('click', function(e) {
      e.stopPropagation();
      navPillsContainer.classList.toggle('nav-open');

      // Update hamburger icon
      navToggle.innerHTML = navPillsContainer.classList.contains('nav-open') ? '<i class="fa-solid fa-xmark" aria-hidden="true"></i>' : '<i class="fa-solid fa-bars" aria-hidden="true"></i>';
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
      if (!navToggle.contains(e.target) && !navPillsContainer.contains(e.target)) {
        navPillsContainer.classList.remove('nav-open');
        navToggle.innerHTML = '<i class="fa-solid fa-bars" aria-hidden="true"></i>';
      }
    });

    // Close menu when clicking a nav link
    navPillsContainer.querySelectorAll('.nav-pill').forEach(pill => {
      pill.addEventListener('click', function() {
        if (pill.closest('.nav-pill-dropdown-wrapper')) return;  // dropdown trigger - keep hamburger open
        navPillsContainer.classList.remove('nav-open');
        navToggle.innerHTML = '<i class="fa-solid fa-bars" aria-hidden="true"></i>';
      });
    });

    // Close hamburger when clicking a Players sub-menu item
    navPillsContainer.querySelectorAll('.nav-pill-dropdown-item').forEach(item => {
      item.addEventListener('click', function() {
        navPillsContainer.classList.remove('nav-open');
        navToggle.innerHTML = '<i class="fa-solid fa-bars" aria-hidden="true"></i>';
        const wrapper = document.getElementById('playersNavDropdown');
        if (wrapper) wrapper.classList.remove('open');
      });
    });
  }
});

// Generic nav dropdown toggle (works for Players, Stats, or any future dropdown)
function toggleNavDropdown(e, wrapperId) {
  e.stopPropagation();
  const wrapper = document.getElementById(wrapperId);
  if (!wrapper) return;
  // Close all other open dropdowns first
  document.querySelectorAll('.nav-pill-dropdown-wrapper.open').forEach(function(el) {
    if (el.id !== wrapperId) el.classList.remove('open');
  });
  wrapper.classList.toggle('open');
}

// Legacy alias kept in case any rendered HTML still calls it
function togglePlayersNav(e) { toggleNavDropdown(e, 'playersNavDropdown'); }

document.addEventListener('DOMContentLoaded', function() {
  // Close all nav dropdowns when clicking outside
  document.addEventListener('click', function(e) {
    if (!e.target.closest('.nav-pill-dropdown-wrapper')) {
      document.querySelectorAll('.nav-pill-dropdown-wrapper.open').forEach(function(el) {
        el.classList.remove('open');
      });
    }
  });

  // Close dropdowns when a sub-menu item is clicked (mobile nav)
  const navPillsContainer = document.getElementById('navPillsContainer') ||
                            document.querySelector('.nav-pills-container');
  if (navPillsContainer) {
    navPillsContainer.querySelectorAll('.nav-pill-dropdown-item').forEach(function(item) {
      item.addEventListener('click', function() {
        document.querySelectorAll('.nav-pill-dropdown-wrapper.open').forEach(function(el) {
          el.classList.remove('open');
        });
      });
    });
  }
});

// Card collapse toggle functionality
document.addEventListener('DOMContentLoaded', function() {
  const collapseToggles = document.querySelectorAll('.card-collapse-toggle');

  collapseToggles.forEach(toggle => {
    toggle.addEventListener('click', function() {
      const targetId = this.getAttribute('data-target');
      const targetBody = document.getElementById(targetId);

      if (targetBody) {
        const isCollapsed = targetBody.classList.contains('collapsed');

        if (isCollapsed) {
          // Expand
          targetBody.classList.remove('collapsed');
          this.classList.remove('collapsed');
          this.innerHTML = '<i class="fa-solid fa-chevron-down" aria-hidden="true"></i>';
        } else {
          // Collapse
          targetBody.classList.add('collapsed');
          this.classList.add('collapsed');
          this.innerHTML = '<i class="fa-solid fa-chevron-right" aria-hidden="true"></i>';
        }
      }
    });
  });

  // GM Memo generation functionality
  const generateGmMemoBtn = document.getElementById('generateGmMemoBtn');
  if (generateGmMemoBtn) {
    generateGmMemoBtn.addEventListener('click', async function() {
      const leagueId = this.dataset.leagueId;
      const season = this.dataset.season;
      const platform = this.dataset.platform;
      const viewerRosterId = this.dataset.viewerRosterId;

      const emptyState = document.getElementById('gm-memo-empty');
      const loadingState = document.getElementById('gm-memo-loading');
      const resultState = document.getElementById('gm-memo-result');

      // Show loading, hide empty state
      if (emptyState) emptyState.style.display = 'none';
      if (loadingState) loadingState.style.display = 'block';
      if (resultState) resultState.style.display = 'none';

      try {
        const response = await fetch('/api/gm-memo', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            league_id: leagueId,
            season: parseInt(season),
            platform: platform,
            viewer_roster_id: viewerRosterId
          })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (data.success) {
          // Hide loading, show result
          if (loadingState) loadingState.style.display = 'none';
          if (resultState) {
            resultState.style.display = 'block';
            resultState.innerHTML = data.gm_memo_html;
          }
        } else {
          // Show error
          if (loadingState) loadingState.style.display = 'none';
          if (emptyState) {
            emptyState.style.display = 'block';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'gm-memo-error';
            errorDiv.textContent = data.error || 'Failed to generate GM memo. Please try again.';
            emptyState.appendChild(errorDiv);
          }
        }
      } catch (error) {
        console.error('Error generating GM memo:', error);
        if (loadingState) loadingState.style.display = 'none';
        if (emptyState) {
          emptyState.style.display = 'block';
          const errorDiv = document.createElement('div');
          errorDiv.className = 'gm-memo-error';
          errorDiv.textContent = 'Network error. Please try again.';
          emptyState.appendChild(errorDiv);
        }
      }
    });
  }

  // History page dynamic loading
  const historyPage = document.querySelector('.history-page');
  if (historyPage) {
    const platform = document.getElementById('platformInput')?.value;
    const season = document.getElementById('seasonInput')?.value;
    const leagueId = document.getElementById('leagueIdInput')?.value;
    const historySeason = document.getElementById('historySeasonInput')?.value;
    const tourMode = !!document.getElementById('historyTourMode');

    if (!tourMode && platform && season && leagueId && historySeason) {
      // Load awards section
      const awardsContent = document.getElementById('historyAwardsContent');
      if (awardsContent) {
        fetch(`/api/history/${platform}/${season}/${leagueId}/summary?history_season=${historySeason}`, { cache: 'no-store' })
          .then(res => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
          .then(data => {
            if (data.html) {
              awardsContent.innerHTML = data.html;
              // Setup dynamic grid columns for fun awards
              setupFunAwardsGrid();
            } else {
              awardsContent.innerHTML = '<div class="history-empty">Failed to load season awards.</div>';
            }
          })
          .catch(err => {
            console.error('Error loading history awards:', err);
            awardsContent.innerHTML = '<div class="history-empty">Error loading season awards.</div>';
          });
      }

      // Load standings section
      const standingsContent = document.getElementById('historyStandingsContent');
      if (standingsContent) {
        fetch(`/api/history/${platform}/${season}/${leagueId}/standings?history_season=${historySeason}`, { cache: 'no-store' })
          .then(res => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
          .then(data => {
            if (data.html) {
              standingsContent.innerHTML = data.html;
            } else {
              standingsContent.innerHTML = '<div class="history-empty">Failed to load standings.</div>';
            }
          })
          .catch(err => {
            console.error('Error loading history standings:', err);
            standingsContent.innerHTML = '<div class="history-empty">Error loading standings.</div>';
          });
      }

      // Load chart section
      const chartContent = document.getElementById('historyChartContent');
      if (chartContent) {
        fetch(`/api/history/${platform}/${season}/${leagueId}/chart?history_season=${historySeason}`, { cache: 'no-store' })
          .then(res => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
          .then(data => {
            if (data.html) {
              // Empty state or error message
              chartContent.innerHTML = data.html;
            } else if (data.data && data.data.length > 0) {
              // Create div for Plotly chart
              chartContent.innerHTML = '<div id="historyChartPlotly" style="width: 100%; height: 430px;"></div>';

              // Build Plotly traces
              const traces = data.data.map(team => ({
                x: team.x,
                y: team.y,
                mode: 'lines+markers',
                name: team.name,
                hovertemplate: '%{fullData.name}<br>Week %{x}<br>%{y:.1f} pts<extra></extra>'
              }));

              // Plotly layout
              const layout = {
                template: 'plotly_white',
                height: 430,
                margin: { l: 40, r: 20, t: 20, b: 40 },
                legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0 },
                xaxis: { 
                  title: 'Week',
                  tickmode: 'auto',
                  nticks: Math.min(10, Math.max(...data.data.flatMap(team => team.x)) || 18)
                },
                yaxis: { title: 'Points' }
              };

              // Render chart
              Plotly.newPlot('historyChartPlotly', traces, layout, { displayModeBar: false });
            } else {
              chartContent.innerHTML = '<div class="history-empty">No chart data available.</div>';
            }
          })
          .catch(err => {
            console.error('Error loading history chart:', err);
            chartContent.innerHTML = '<div class="history-empty">Error loading season chart.</div>';
          });
      }
    }
  }
});
// Player Modal
function openPlayerModal(playerId, playerName, opts) {

  // Extract league context from URL path: /<platform>/<season>/<league_id>/<page>
  const pathParts = window.location.pathname.split('/').filter(p => p);
  const platform = pathParts[0] || 'sleeper';
  const season = pathParts[1] || new Date().getFullYear();
  const leagueId = pathParts[2] || null;

  // Use page-level league settings when available (set for logged-in users)
  const modalLt = (typeof _leagueType !== 'undefined') ? _leagueType : '1qb';
  const modalLs = (typeof _leagueSize !== 'undefined') ? _leagueSize : 10;
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

  modal.innerHTML = `
    <div class="player-modal-header">
        <div class="player-modal-headshot-container">
          <img class="player-modal-headshot" id="playerModalHeadshot" src="" alt="${playerName || 'Player'}" />
        </div>
      <div class="player-modal-title-section">
        <div class="player-modal-title-text">
          <h2 class="player-modal-name">${playerName || 'Loading...'}</h2>
          <div class="player-modal-meta" id="playerModalMeta">
            <div class="loading-spinner" style="width: 16px; height: 16px;"></div>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
        <button class="player-modal-watchlist-btn" id="playerModalWatchlistBtn" title="Add to watchlist" style="display: none;"><i class="fa-regular fa-star" aria-hidden="true"></i></button>
        <button class="player-modal-compare-btn" id="playerModalCompareBtn" title="Compare players">Compare Player</button>
        <button class="player-modal-close" onclick="closePlayerModal()">×</button>
      </div>
    </div>
    <div class="pm-tab-bar" id="pmTabBar" style="display:none">
      <button class="pm-tab active" data-tab="overview" onclick="pmSwitchTab('overview')">Overview</button>
      <button class="pm-tab" data-tab="stats" onclick="pmSwitchTab('stats')">Stats</button>
      <button class="pm-tab" id="pmTabMetrics" data-tab="metrics" onclick="pmSwitchTab('metrics')" style="display:none">Adv Metrics</button>
      <button class="pm-tab" id="pmTabProspect" data-tab="prospect" onclick="pmSwitchTab('prospect')" style="display:none">Prospect</button>
      <button class="pm-tab" id="pmTabBreakout" data-tab="breakout" onclick="pmSwitchTab('breakout')" style="display:none">Breakout</button>
      <button class="pm-tab" data-tab="trades" onclick="pmSwitchTab('trades')">Trades</button>
    </div>
    <div class="player-modal-body" id="playerModalBody">
      <div class="player-modal-loading">
        <div class="loading-spinner"></div>
        <div>Loading player data...</div>
      </div>
    </div>
  `;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);
  document.body.style.overflow = 'hidden';

  // Fetch player data (with 5-min localStorage cache to speed up re-opens)
  const _cacheKey = 'pm_cache_' + apiUrl;
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
        modalBody.innerHTML = `
          <div class="player-modal-loading">
            <div style="color: #ef4444; font-weight: 500;">Error loading player data</div>
            <div style="font-size: 13px;">${data.error}</div>
          </div>
        `;
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

      // Check if player has no game logs (indicating a rookie without NFL stats)
      const hasGameLogs = data.game_logs_by_year && Object.keys(data.game_logs_by_year).length > 0;
      const isRookieWithoutGameLogs = !hasGameLogs && data.prospect_data && data.prospect_data.prospect_score != null;

      if (isElite(pid)) {
        badges += '<span class="player-badge player-badge-elite"><i class="fa-solid fa-star" aria-hidden="true"></i> ELITE</span>';
      }
      // Show rookie badge for both years_exp === 0 AND players with no game logs (rookies without NFL stats)
      if ((yearsExp != null && yearsExp === 0) || isRookieWithoutGameLogs) {
        badges += '<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>';
      }
      if (isProspect(pid)) {
        badges += '<span class="player-badge player-badge-prospect"><i class="fa-solid fa-seedling" aria-hidden="true"></i> PROSPECT</span>';
      }
      if (isBreakout(pid)) {
        badges += '<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>';
      }

      // Name with inline badges
      const nameEl = document.querySelector('.player-modal-name');
      if (!nameEl) return;
      nameEl.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;';
      nameEl.innerHTML = `<span>${playerName || 'Unknown Player'}</span>${badges}`;

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
        const _ownerStr = data.fantasy_team_owner ? ` · <span style="opacity:.65;">@${data.fantasy_team_owner}</span>` : '';
        metaHTML += `<div style="font-size:11px;font-weight:600;color:var(--accent);margin-top:3px;opacity:.9;">${data.fantasy_team}${_ownerStr}</div>`;
      }
      metaEl.innerHTML = metaHTML;

      // Update headshot
      const headshotEl = document.getElementById('playerModalHeadshot');
      if (headshotEl && data.espnHeadshot) {
        headshotEl.src = data.espnHeadshot;
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
      const ppgSeason    = data.stats?.ppg_season;
      const totalPts     = data.stats?.total_pts;
      const totalPtsRank = data.stats?.total_pts_rank;
      const seasonLabel  = ppgSeason ? ` · ${ppgSeason}` : '';
      const ppgCard = ppgVal != null
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">PPG${seasonLabel}</div>
            <div class="pm-hero-val">${ppgVal}</div>
            <div class="pm-hero-sub">${ppgRank ? `${pos}${ppgRank}` : '-'}</div>
          </div>`
        : '';
      const totalCard = totalPts != null
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">Total${seasonLabel}</div>
            <div class="pm-hero-val">${totalPts}</div>
            <div class="pm-hero-sub">${totalPtsRank ? `${pos}${totalPtsRank}` : '-'}</div>
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
      const _heroCardCount = 3 + (ppgCard ? 1 : 0) + (totalCard ? 1 : 0);
      const heroGridStyle = `style="grid-template-columns:repeat(${_heroCardCount},1fr);"`;
      let overviewHTML = `
        <div class="pm-hero-row" ${heroGridStyle}>
          <div class="pm-hero-stat pm-hero-primary">
            <div class="pm-hero-label">1QB Value</div>
            <div class="pm-hero-val" style="color:#3b82f6;">${val1qb > 0 ? val1qb : '-'}</div>
          </div>
          <div class="pm-hero-stat">
            <div class="pm-hero-label">SF Value</div>
            <div class="pm-hero-val">${valsf > 0 ? valsf : '-'}</div>
          </div>
          ${thirdValueCard}
          ${ppgCard}
          ${totalCard}
        </div>
      `;

      if (hasChart) {
        overviewHTML += `
          <hr class="pm-section-divider">
          <div class="pm-section-header"><span class="pm-section-label">Value History</span></div>
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
      const metricsHTML = hasMetrics ? `
        <div id="advancedMetricsSection">
          <div class="pm-section-header">
            <span class="pm-section-label">Advanced Metrics <span id="advMetricsSeasonLabel" style="font-size:12px;opacity:.6;"></span></span>
          </div>
          <div id="advMetricsPills"></div>
          <div id="advancedMetricsContent">
            <div style="padding:12px 0;display:flex;align-items:center;gap:10px;">
              <div class="loading-spinner" style="width:16px;height:16px;"></div>
              <span style="font-size:13px;color:var(--text-muted);">Loading...</span>
            </div>
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
        <div class="pm-panel" id="pm-panel-metrics">${metricsHTML}</div>
        <div class="pm-panel" id="pm-panel-prospect">${prospectPanelHTML}</div>
        <div class="pm-panel" id="pm-panel-breakout">${breakoutHTML}</div>
        <div class="pm-panel" id="pm-panel-trades">${tradesHTML}</div>
      `;

      // ── Show tab bar and configure it ─────────────────────────────────────
      const pmTabBar = document.getElementById('pmTabBar');
      pmTabBar.style.display = '';
      pmTabBar.dataset.pmPlayerId = playerId;
      pmTabBar.dataset.pmSeason = season;
      pmTabBar.dataset.pmPlayerName = data.name || playerName || '';

      // Show/hide conditional tabs
      const tabMetrics = document.getElementById('pmTabMetrics');
      if (tabMetrics) tabMetrics.style.display = hasMetrics ? '' : 'none';
      const tabProspect = document.getElementById('pmTabProspect');
      // Prospect tab: only for players with no NFL game logs drafted in the current season
      const _currentNFLYear = new Date().getFullYear();
      const _isCurrentYearProspect = hasProspectData && !hasGameLogs
        && String(pd.draft_class_year) === String(_currentNFLYear);
      if (tabProspect) tabProspect.style.display = _isCurrentYearProspect ? '' : 'none';
      const tabBreakout = document.getElementById('pmTabBreakout');
      if (tabBreakout) tabBreakout.style.display = isBreakout(pid) ? '' : 'none';

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
            const tierColors = ['','#10b981','#3b82f6','#8b5cf6','#f59e0b','#6b7280','#9ca3af'];
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

      // ── Wire up compare button ────────────────────────────────────────────
      const cmpBtn = document.getElementById('playerModalCompareBtn');
      if (cmpBtn) {
        cmpBtn.addEventListener('click', () => openCompareSearch(data));
      }

      // ── Render value history chart in Overview panel ───────────────────────
      if (data.value_history && data.value_history.length > 0) {
        const chartDiv = document.getElementById('playerValueChart');
        if (chartDiv && typeof Plotly !== 'undefined') {
          // Robust date formatters (handle YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS)
          const formatDateLabel = (dateStr) => {
            if (!dateStr) return '';
            const m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})/);
            if (!m) return '';
            const [, year, month, day] = m;
            // Use hardcoded month names to avoid locale/timezone issues
            const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            return `${monthNames[parseInt(month, 10) - 1]} ${parseInt(day, 10)}`;
          };

          const xData = data.value_history.map(d => formatDateLabel(d.as_of_date));
          const n = xData.length;

          // Tight Y-axis range based on actual data
          const yValues = data.value_history.map(d => d.value);
          const yMin = Math.min(...yValues);
          const yMax = Math.max(...yValues);
          const yRange = yMax - yMin;
          const yPad = Math.max(yRange * 0.15, 30);

          // Ticks at their actual positions (first / middle / last)
          const midIdx = Math.floor((n - 1) / 2);
          const firstDateStr  = formatDateLabel(data.value_history[0].as_of_date);
          const midDateStr    = formatDateLabel(data.value_history[midIdx].as_of_date);
          const latestDateStr = formatDateLabel(data.value_history[n - 1].as_of_date);
          const tickvals = n <= 1 ? [xData[0]]
                         : n <= 2 ? [xData[0], xData[n - 1]]
                         : [xData[0], xData[midIdx], xData[n - 1]];
          const ticktext  = n <= 1 ? [latestDateStr]
                          : n <= 2 ? [firstDateStr, latestDateStr]
                          : [firstDateStr, midDateStr, latestDateStr];

          // Read theme text color so ticks are visible in both light and dark mode
          const mutedColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--text-muted').trim() || '#6b7280';

          // Add empty space after the data to center the last point
          const extendedX = [...xData, '', '', '', ''];
          const extendedY = [...yValues, null, null, null, null];

          // Create hover text for actual data points only
          const hoverText = [...xData.map(date => `<b>${date}</b><br>Value: ${yValues[xData.indexOf(date)]?.toFixed(1) || ''}`), '', '', '', ''];

          const trace = {
            x: extendedX,
            y: extendedY,
            type: 'scatter',
            mode: 'lines',
            name: 'Value',
            line: { color: '#3b82f6', width: 2, shape: 'spline', smoothing: 1.2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: '%{text}<extra></extra>',
            text: hoverText
          };

          // Adjust chart height based on screen size
          const isMobile = window.innerWidth <= 768;
          const chartHeight = isMobile ? 200 : 250;

          const layout = {
            margin: { l: 30, r: 20, t: 10, b: 36 },
            height: chartHeight,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
              showgrid: false,
              type: 'category',
              tickmode: 'array',
              tickvals: [...tickvals, tickvals.length, tickvals.length + 1, tickvals.length + 2, tickvals.length + 3],
              ticktext: [...ticktext, '', '', ''],
              tickangle: 0,
              tickfont: { size: 11, color: mutedColor },
              fixedrange: true,
              range: [-(xData.length * 0.3), xData.length + 2],
            },
            yaxis: {
              showgrid: true,
              showticklabels: true,
              range: [yMin - yPad, yMax + yPad],
              tickfont: { size: 11, color: mutedColor },
            },
            hovermode: 'closest',
          };

          Plotly.newPlot('playerValueChart', [trace], layout, {
            displayModeBar: false,
            responsive: true
          });
        }
      }

      // Store whether this player has metrics so pmSwitchTab can lazy-load them
      if (pmTabBar) pmTabBar.dataset.pmHasMetrics = hasMetrics ? '1' : '';
    })
    .catch(err => {
      console.error('Error loading player data:', err);
      const b = document.getElementById('playerModalBody');
      if (b) b.innerHTML = `
        <div class="player-modal-loading">
          <div style="color: #ef4444; font-weight: 500;">Error loading player data</div>
          <div style="font-size: 13px;">Please try again</div>
        </div>
      `;
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

// ── Player Modal Tab Switching (global) ──────────────────────────────────────
function pmSwitchTab(tab) {
  document.querySelectorAll('.pm-panel').forEach(p => p.classList.remove('pm-panel-active'));
  document.querySelectorAll('.pm-tab').forEach(t => t.classList.remove('active'));
  const panel = document.getElementById('pm-panel-' + tab);
  const btn = document.querySelector('.pm-tab[data-tab="' + tab + '"]');
  if (panel) panel.classList.add('pm-panel-active');
  if (btn) btn.classList.add('active');

  const pmTabBar = document.getElementById('pmTabBar');
  if (!pmTabBar) return;
  const playerId = pmTabBar.dataset.pmPlayerId;
  const season = pmTabBar.dataset.pmSeason;

  // ── Lazy-load Adv Metrics tab ────────────────────────────────────────────
  if (tab === 'metrics' && panel && !panel.dataset.loaded && pmTabBar && pmTabBar.dataset.pmHasMetrics) {
    panel.dataset.loaded = '1';
    const path = window.location.pathname;
    const match = path.match(/\/(sleeper|espn)\/(\d+)\/([^\/]+)/);
    const leagueIdForMetrics = match ? match[3] : null;
    loadAdvancedMetrics(playerId, leagueIdForMetrics, null);
  }

  // ── Lazy-load Breakout tab ───────────────────────────────────────────────
  if (tab === 'breakout' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    fetch(`/api/breakout/player/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (!panel.isConnected) return;
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

  // ── Lazy-load Stats tab ──────────────────────────────────────────────────
  if (tab === 'stats' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    const pathParts2 = window.location.pathname.split('/').filter(p => p);
    const _platform = pathParts2[0] || 'sleeper';
    const _season   = pathParts2[1] || new Date().getFullYear();
    const _leagueId = pathParts2[2] || null;
    const _lt = (typeof _leagueType !== 'undefined') ? _leagueType : '1qb';
    const _ls = (typeof _leagueSize !== 'undefined') ? _leagueSize : 10;
    let logsUrl = `/api/player-game-logs/${encodeURIComponent(playerId)}?season=${_season}&league_type=${_lt}&league_size=${_ls}`;
    if (_leagueId) logsUrl += `&league_id=${_leagueId}&platform=${_platform}`;
    fetch(logsUrl)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (!panel.isConnected) return;
        const logsByYear = data.game_logs_by_year || {};
        if (!Object.keys(logsByYear).length) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:40px 0;"><div style="color:var(--text-muted);font-size:13px;">No game log data available.</div></div>';
          return;
        }
        panel.innerHTML = _buildStatsHTML(logsByYear);
      })
      .catch(() => {
        if (panel.isConnected) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:40px 0;"><div style="color:var(--text-muted);font-size:13px;">Could not load stats.</div></div>';
        }
      });
  }

  // ── Lazy-load Trades tab ─────────────────────────────────────────────────
  if (tab === 'trades' && panel && !panel.dataset.loaded) {
    panel.dataset.loaded = '1';
    const playerName = pmTabBar.dataset.pmPlayerName || '';
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const tdbPlatform = pathParts[0];
    const tdbSeason   = pathParts[1];
    const tdbLeague   = pathParts[2];
    const tdbBase = (tdbPlatform && tdbSeason && tdbLeague && !['players','breakouts','prospects','trade-database','trade-intel'].includes(tdbPlatform))
      ? `/${tdbPlatform}/${tdbSeason}/${tdbLeague}/trade-database`
      : '/trade-database';
    const tdbLink = playerName
      ? `${tdbBase}?q=${encodeURIComponent(playerName)}`
      : tdbBase;

    fetch(`/api/trade-intel/player-trades/${encodeURIComponent(playerId)}?season=${encodeURIComponent(season)}&limit=20`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(data => {
        if (!panel.isConnected) return;
        const trades = data.trades || [];
        const linkHTML = `<div style="text-align:center;padding:12px 0 2px;"><a href="${tdbLink}" style="font-size:12px;color:var(--accent,#3b82f6);font-weight:600;text-decoration:none;">Search all trades in Trade Database →</a></div>`;
        if (!trades.length) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">No recent trades found for this player.</div></div>' + linkHTML;
          return;
        }
        panel.innerHTML = '<div style="padding:4px 0;">' + trades.map(t => {
          const dateStr = t.date ? new Date(t.date).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'}) : '-';
          const sfBadge = t.league_type === 'sf' || t.league_type === 'superflex'
            ? '<span style="padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;background:rgba(139,92,246,.15);color:#8b5cf6;border:1px solid rgba(139,92,246,.3);">SF</span>'
            : '<span style="padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;background:rgba(59,130,246,.15);color:#3b82f6;border:1px solid rgba(59,130,246,.3);">1QB</span>';
          const scoreBadge = t.fairness_score != null
            ? `<span style="padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;background:rgba(0,0,0,.05);color:var(--text-muted);">${parseFloat(t.fairness_score).toFixed(0)}</span>`
            : '';
          const renderAssets = (assets) => {
            if (!assets || !assets.length) return '<span style="font-size:12px;color:var(--text-muted);">-</span>';
            return assets.map(a => {
              const isPick = a.is_pick || (a.name || '').toLowerCase().includes('pick') || (a.name || '').toLowerCase().includes('round');
              const isFocus = String(a.player_id || '') === String(playerId);
              const cls = isPick ? 'pm-trade-asset pm-pick' : (isFocus ? 'pm-trade-asset pm-focus' : 'pm-trade-asset');
              return `<div class="${cls}">${a.name || a.player_name || '?'}</div>`;
            }).join('');
          };
          const sideA = renderAssets(t.side_a);
          const sideB = renderAssets(t.side_b);
          return `<div class="pm-trade-card">
            <div class="pm-trade-head">
              <span class="pm-trade-date">${dateStr}</span>
              <div style="display:flex;gap:5px;">${sfBadge}${scoreBadge}</div>
            </div>
            <div class="pm-trade-body">
              <div class="pm-trade-col">${sideA}</div>
              <div style="color:var(--text-muted);font-size:18px;align-self:center;">⇄</div>
              <div class="pm-trade-col">${sideB}</div>
            </div>
          </div>`;
        }).join('') + '</div>' + linkHTML;
      })
      .catch(() => {
        if (panel.isConnected) {
          panel.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">Could not load trade history.</div></div>';
        }
      });
  }
}

// ── Breakout tab HTML builder (returns HTML string, no DOM side effects) ─────
function _buildBkTabHTML(data, scoreColor) {
  const score = parseFloat(data.breakout_opportunity_score || 0);
  const scoreStr = score.toFixed(1);
  if (!scoreColor) {
    scoreColor = '#10b981';
    if (score < 50) scoreColor = '#3b82f6';
    if (score < 40) scoreColor = '#f59e0b';
    if (score < 30) scoreColor = '#6b7280';
  }

  const breakoutType = data.breakout_type || {};
  const formattedPhase = data.phase
    ? data.phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : '-';

  const reasons = (data.key_reasons || '').split('\n')
    .filter(r => r.trim() && r.startsWith('•'))
    .map(r => r.substring(1).trim());

  const txnSummary    = data.vacated_usage_summary || '';
  const addedCompSumm = data.added_competition_summary || '';

  // ── Hero row ───────────────────────────────────────────────────────────────
  let html = `
    <div class="pm-hero-row">
      <div class="pm-hero-stat" style="background:${scoreColor}1a;border-color:${scoreColor}33;">
        <div class="pm-hero-label" style="color:${scoreColor};">Breakout Score</div>
        <div class="pm-hero-val" style="color:${scoreColor};">${scoreStr}</div>
      </div>
      <div class="pm-hero-stat" style="text-align:center;padding-left:16px;">
        <div class="pm-hero-label">Profile</div>
        <div style="font-size:13px;font-weight:700;line-height:1.3;color:var(--text);margin:4px 0;">${breakoutType.profile_label || '-'}</div>
      </div>
      <div class="pm-hero-stat">
        <div class="pm-hero-label">Phase</div>
        <div style="font-size:13px;font-weight:700;color:var(--text);line-height:1.3;margin:4px 0;">${formattedPhase}</div>
      </div>
    </div>
  `;

  // ── Component breakdown ────────────────────────────────────────────────────
  const components = [
    { label: 'Opportunity',     val: data.opportunity_opened_score,  color: '#10b981' },
    { label: 'Competition',     val: data.competition_removed_score, color: '#3b82f6' },
    { label: 'Team Env.',       val: data.team_environment_score,    color: null      },
    { label: 'Readiness',       val: data.player_readiness_score,    color: '#8b5cf6' },
    { label: 'Role Trajectory', val: data.role_trajectory_score,     color: null      },
    { label: 'Confidence',      val: data.confidence_score,          color: '#6b7280', suffix: '%' },
  ];

  html += `<div class='pm-two-column'><div class='pm-left-column'>`;
  html += `<hr class="pm-section-divider">`;
  html += `<div class="pm-section-header"><span class="pm-section-label">Component Breakdown</span></div>`;
  html += '<div class="pm-comp-list-bo">';
  components.forEach(c => {
    const v    = parseFloat(c.val || 0);
    const fill = Math.min(100, Math.max(0, v));
    const color = c.color || (v >= 60 ? '#10b981' : v >= 35 ? '#3b82f6' : '#f59e0b');
    const disp = c.suffix ? v.toFixed(0) + c.suffix : v.toFixed(1);
    html += `
      <div class="pm-comp-row">
        <span class="pm-comp-label">${c.label}</span>
        <div class="pm-comp-bar-wrap"><div class="pm-comp-bar" style="width:${fill.toFixed(1)}%;background:${color};"></div></div>
        <span class="pm-comp-val" style="color:${color};">${disp}</span>
      </div>`;
  });
  html += '</div></div>';

  // ── Key factors ────────────────────────────────────────────────────────────
  if (reasons.length) {
    html += `
      <div class='pm-right-column'>
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Key Factors</span></div>
      <div style="display:flex;flex-direction:column;gap:6px;">
    `;
    reasons.forEach(r => {
      html += `<div style="font-size:13px;color:var(--text-muted);display:flex;gap:15px;align-items:flex-start;">
        <span style="color:${scoreColor};font-weight:700;flex-shrink:0;">•</span><span>${r}</span>
      </div>`;
    });
    html += '</div></div>';
  }
  html += '</div>';

  // ── Context boxes ──────────────────────────────────────────────────────────
  if (txnSummary && txnSummary !== 'No departures') {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Vacated Opportunity</span></div>
      <div class="pm-context-box">${txnSummary}</div>
    `;
  }
  if (addedCompSumm && addedCompSumm !== 'No new competition added') {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Added Competition</span></div>
      <div class="pm-context-box competition">${addedCompSumm}</div>
    `;
  }

  return html;
}

// ── Stats tab HTML builder (returns HTML string, no DOM side effects) ─────────
function _buildStatsHTML(game_logs_by_year) {
  let statsHTML = '';
  if (game_logs_by_year && Object.keys(game_logs_by_year).length > 0) {
    statsHTML += `
      <div class="player-modal-section">
        <div class="pm-section-header"><span class="pm-section-label">Game Logs</span></div>
    `;

    // Sort years in descending order (most recent first)
    const years = Object.keys(game_logs_by_year).sort((a, b) => b - a);

    years.forEach((year, index) => {
      const gameLogs = game_logs_by_year[year];
      const isFirstYear = index === 0;

      // Calculate season totals
      let totalFantasyPts = 0;
      let totalPassYd = 0, totalPassTd = 0, totalPassInt = 0;
      let totalRushAtt = 0, totalRushYd = 0, totalRushTd = 0;
      let totalRecTgt = 0, totalRec = 0, totalRecYd = 0, totalRecTd = 0;
      let totalFumLost = 0;

      gameLogs.forEach(game => {
        totalFantasyPts += game.fantasy_pts || 0;
        const s = game.stats;
        totalPassYd += s.pass_yd || 0;
        totalPassTd += s.pass_td || 0;
        totalPassInt += s.pass_int || 0;
        totalRushAtt += s.rush_att || 0;
        totalRushYd += s.rush_yd || 0;
        totalRushTd += s.rush_td || 0;
        totalRecTgt += s.rec_tgt || 0;
        totalRec += s.rec || 0;
        totalRecYd += s.rec_yd || 0;
        totalRecTd += s.rec_td || 0;
        totalFumLost += s.fum_lost || 0;
      });

      // Build season summary for header
      const seasonSummaryParts = [];
      seasonSummaryParts.push(`${totalFantasyPts.toFixed(1)} pts`);
      if (totalPassYd > 0) seasonSummaryParts.push(`${Math.round(totalPassYd)} pass yds`);
      if (totalRushYd > 0) seasonSummaryParts.push(`${Math.round(totalRushYd)} rush yds`);
      if (totalRec > 0) seasonSummaryParts.push(`${totalRec} rec`);
      const seasonSummary = seasonSummaryParts.join(' • ');

      statsHTML += `
        <div class="game-log-year-section">
          <div class="game-log-year-header" onclick="toggleGameLogYear('${year}')">
            <div class="game-log-year-header-main">
              <span class="game-log-year-toggle" id="toggle-${year}">▼</span>
              <span class="game-log-year-title">${year} Season</span>
            </div>
          </div>
          <div class="game-log-year-content ${isFirstYear ? 'expanded' : ''}" id="year-${year}">
            <table class="game-log-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th>Pts</th>
                  <th>Pass Yd</th>
                  <th>Pass TD</th>
                  <th>INT</th>
                  <th>Rush Att</th>
                  <th>Rush Yd</th>
                  <th>Rush TD</th>
                  <th>Tgt</th>
                  <th>Rec</th>
                  <th>Rec Yd</th>
                  <th>Rec TD</th>
                </tr>
              </thead>
              <tbody>
      `;

      gameLogs.forEach(game => {
        const stats = game.stats;

        // Format date: 20240908 -> 9/8
        let dateStr = game.date || '';
        if (dateStr.length === 8) {
          const month = parseInt(dateStr.substring(4, 6));
          const day = parseInt(dateStr.substring(6, 8));
          dateStr = `${month}/${day}`;
        }

        // Check if player has any stats at all
        const hasAnyStats = stats.pass_yd != null || stats.rush_att != null ||
                           stats.rec != null || stats.rec_tgt != null;

        const val = (v) => v != null && v > 0 ? v : '-';
        const rowClass = hasAnyStats ? 'game-log-table-row' : 'game-log-table-row game-log-no-stats';

        statsHTML += `
          <tr class="${rowClass}">
            <td>${dateStr}</td>
            <td class="game-log-table-opp">${game.opponent || '-'}</td>
            <td class="game-log-table-pts">${hasAnyStats ? (game.fantasy_pts != null ? game.fantasy_pts.toFixed(1) : '-') : '<span style="color:#9ca3af;">DNP</span>'}</td>
            <td>${val(stats.pass_yd) !== '-' ? Math.round(stats.pass_yd) : '-'}</td>
            <td>${val(stats.pass_td)}</td>
            <td>${val(stats.pass_int)}</td>
            <td>${val(stats.rush_att)}</td>
            <td>${val(stats.rush_yd) !== '-' ? Math.round(stats.rush_yd) : '-'}</td>
            <td>${val(stats.rush_td)}</td>
            <td>${val(stats.rec_tgt)}</td>
            <td>${val(stats.rec)}</td>
            <td>${val(stats.rec_yd) !== '-' ? Math.round(stats.rec_yd) : '-'}</td>
            <td>${val(stats.rec_td)}</td>
          </tr>
        `;
      });

      const valTotal = (v) => v != null && v > 0 ? v : '-';

      // Add season totals row in table format (inside the table)
      statsHTML += `
              </tbody>
              <tfoot>
                <tr class="game-log-table-total">
                  <td><strong>Total</strong></td>
                  <td><strong>${gameLogs.length}G</strong></td>
                  <td class="game-log-table-pts"><strong>${totalFantasyPts.toFixed(1)}</strong></td>
                  <td><strong>${valTotal(totalPassYd) !== '-' ? Math.round(totalPassYd) : '-'}</strong></td>
                  <td><strong>${valTotal(totalPassTd)}</strong></td>
                  <td><strong>${valTotal(totalPassInt)}</strong></td>
                  <td><strong>${valTotal(totalRushAtt)}</strong></td>
                  <td><strong>${valTotal(totalRushYd) !== '-' ? Math.round(totalRushYd) : '-'}</strong></td>
                  <td><strong>${valTotal(totalRushTd)}</strong></td>
                  <td><strong>${valTotal(totalRecTgt)}</strong></td>
                  <td><strong>${valTotal(totalRec)}</strong></td>
                  <td><strong>${valTotal(totalRecYd) !== '-' ? Math.round(totalRecYd) : '-'}</strong></td>
                  <td><strong>${valTotal(totalRecTd)}</strong></td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      `;
    });

    statsHTML += `</div>`;
  }
  return statsHTML || '<div class="player-modal-loading" style="padding:40px 0;"><div style="color:var(--text-muted);font-size:13px;">No game log data available.</div></div>';
}

function getRoleGrade(roleScore) {
  if (roleScore >= 80) return 'Elite';
  if (roleScore >= 70) return 'Great';
  if (roleScore >= 60) return 'Good';
  if (roleScore >= 50) return 'Average';
  if (roleScore >= 40) return 'Below Avg';
  return 'Limited';
}

function loadAdvancedMetrics(playerId, leagueId, season) {
  const contentEl = document.getElementById('advancedMetricsContent');
  if (!contentEl) return;

  // Show spinner in the bars column only (values on the left stay intact)
  contentEl.innerHTML = `
    <div style="padding:12px 0;display:flex;align-items:center;gap:10px;">
      <div class="loading-spinner" style="width:16px;height:16px;"></div>
      <span style="font-size:13px;color:var(--text-muted);">Loading...</span>
    </div>
  `;

  const leagueParam = leagueId ? `&league_id=${leagueId}` : '';
  const seasonParam = season != null ? `&season=${season}` : '';
  const url = `/api/player-advanced-metrics/${playerId}?_=1${leagueParam}${seasonParam}`;

  fetch(url)
    .then(res => { if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
    .then(metricsData => {
      if (metricsData.error || metricsData.premium_required) {
        const section = document.getElementById('advancedMetricsSection');
        if (section) section.style.display = 'none';
        return;
      }

      const availableSeasons = metricsData.available_seasons || [];
      const activeSeason = metricsData.season;

      // Update year label in section header
      const seasonLabelEl = document.getElementById('advMetricsSeasonLabel');
      if (seasonLabelEl && activeSeason) seasonLabelEl.textContent = activeSeason;

      // Season pills above the layout
      const pillsEl = document.getElementById('advMetricsPills');
      if (pillsEl && availableSeasons.length > 1) {
        let pillsHTML = '<div class="adv-metrics-season-pills">';
        availableSeasons.forEach(yr => {
          const activeClass = yr === activeSeason ? ' active' : '';
          pillsHTML += `<button class="adv-season-pill${activeClass}" onclick="loadAdvancedMetrics('${playerId}', ${leagueId ? `'${leagueId}'` : 'null'}, ${yr})">${yr}</button>`;
        });
        pillsHTML += '</div>';
        pillsEl.innerHTML = pillsHTML;
      }

      // Populate just the bars column
      contentEl.innerHTML = buildAdvancedMetricsHTML(metricsData);
    })
    .catch(err => {
      console.error('Error loading advanced metrics:', err);
      const section = document.getElementById('advancedMetricsSection');
      if (section) section.style.display = 'none';
    });
}

function buildAdvancedMetricsHTML(metricsData) {
  const metrics = metricsData.metrics || {};
  const position = metricsData.position;

  const defs = [];

  // Role Score (0–100)
  if (metrics.role_score != null) {
    defs.push({ label: 'Role Score', fill: metrics.role_score, display: metrics.role_score.toFixed(1), sub: getRoleGrade(metrics.role_score) });
  }
  // Snap Share (0–1 → %)
  if (metrics.snap_share != null && position !== "QB") {
    const pct = metrics.snap_share * 100;
    defs.push({ label: 'Snap Share', fill: pct, display: pct.toFixed(1) + '%' });
  }

  if (position === 'QB') {
    if (metrics.pff_passing_grade != null) {
      const v = metrics.pff_passing_grade;
      defs.push({ label: 'PFF Pass Grade', fill: v, display: v.toFixed(1) });
    }
    if (metrics.big_time_throw_rate != null) {
      const v = metrics.big_time_throw_rate;
      defs.push({ label: 'BTT Rate', fill: Math.min(v * 5, 100), display: v.toFixed(1) + '%' });
    }
    if (metrics.adjusted_completion_rate != null) {
      const v = metrics.adjusted_completion_rate;
      defs.push({ label: 'Adj Comp %', fill: v, display: v.toFixed(1) + '%' });
    }
    if (metrics.nfl_passer_rating != null) {
      const v = metrics.nfl_passer_rating;
      defs.push({ label: 'Passer Rating', fill: Math.min(v / 130 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.yards_per_attempt != null) {
      const v = metrics.yards_per_attempt;
      defs.push({ label: 'Yds/Attempt', fill: Math.min(v / 10 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.completion_pct != null) {
      const pct = metrics.completion_pct;
      defs.push({ label: 'Completion %', fill: pct, display: pct.toFixed(1) + '%' });
    }
    if (metrics.td_rate != null && metrics.int_rate != null && metrics.int_rate > 0) {
      const ratio = metrics.td_rate / metrics.int_rate;
      defs.push({ label: 'TD/INT Ratio', fill: Math.min(ratio * 20, 100), display: ratio.toFixed(2) });
    }
    if (metrics.pressure_to_sack_rate != null) {
      const v = metrics.pressure_to_sack_rate;
      // Lower is better: elite QBs sack <20% of pressures
      const fill = Math.max(0, 100 - v);
      defs.push({ label: 'Pressure→Sack%', fill, display: v.toFixed(1) + '%', forceColor: v <= 20 ? '#10b981' : v <= 35 ? '#3b82f6' : '#ef4444' });
    }
    if (metrics.yards_per_carry != null) {
      const v = metrics.yards_per_carry;
      defs.push({ label: 'Yds/Carry', fill: Math.min(v / 7 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.rush_td_rate != null) {
      const v = metrics.rush_td_rate;
      defs.push({ label: 'Rush TD Rate', fill: Math.min(v * 800, 100), display: (v * 100).toFixed(1) + '%' });
    }
  } else if (position === 'RB') {
    if (metrics.pff_rushing_grade != null) {
      const v = metrics.pff_rushing_grade;
      defs.push({ label: 'PFF Rush Grade', fill: v, display: v.toFixed(1) });
    }
    if (metrics.breakaway_percentage != null) {
      const v = metrics.breakaway_percentage;
      defs.push({ label: 'Breakaway %', fill: Math.min(v * 2.5, 100), display: v.toFixed(1) + '%' });
    }
    if (metrics.explosive_runs_10_plus != null) {
      const v = metrics.explosive_runs_10_plus;
      defs.push({ label: 'Explosive Runs', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(0) });
    }
    if (metrics.elusive_rating != null) {
      const v = metrics.elusive_rating;
      defs.push({ label: 'Elusive Rating', fill: Math.min(v / 180 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.avoided_tackles != null && metrics.avoided_tackles > 0) {
      const v = metrics.avoided_tackles;
      defs.push({ label: 'Avoided Tackles', fill: Math.min(v / 30 * 100, 100), display: v.toFixed(0) });
    }
    if (metrics.yards_per_carry != null) {
      const v = metrics.yards_per_carry;
      defs.push({ label: 'Yds/Carry', fill: Math.min(v / 7 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.yards_per_touch != null) {
      const v = metrics.yards_per_touch;
      defs.push({ label: 'Yds/Touch', fill: Math.min(v / 8 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.rush_td_rate != null) {
      const v = metrics.rush_td_rate;
      defs.push({ label: 'Rush TD Rate', fill: Math.min(v * 1000, 100), display: (v * 100).toFixed(1) + '%' });
    }
    if (metrics.opportunity_share != null) {
      const oppShare = metrics.opportunity_share;
      const fillPercent = Math.min(oppShare * 4, 100);
      const color = oppShare >= 25 ? '#10b981' : oppShare >= 15 ? '#3b82f6' : oppShare >= 10 ? '#f59e0b' : '#6b7280';
      defs.push({ label: 'Opp Share', fill: fillPercent, display: oppShare.toFixed(1) + '%', forceColor: color });
    }
    if (metrics.catch_rate != null) {
      const pct = metrics.catch_rate * 100;
      defs.push({ label: 'Catch Rate', fill: pct, display: pct.toFixed(1) + '%' });
    }
  } else if (position === 'WR' || position === 'TE') {
    if (metrics.grades_offense != null) {
      const v = metrics.grades_offense;
      defs.push({ label: 'PFF Off Grade', fill: v, display: v.toFixed(1) });
    }
    if (metrics.catch_rate != null) {
      const pct = metrics.catch_rate * 100;
      defs.push({ label: 'Catch Rate', fill: pct, display: pct.toFixed(1) + '%' });
    }
    if (metrics.drop_rate != null) {
      const v = metrics.drop_rate;
      // Lower is better; flip color: green = low drop rate
      const fill = Math.max(0, 100 - v * 5);
      defs.push({ label: 'Drop Rate', fill, display: v.toFixed(1) + '%', forceColor: v <= 5 ? '#10b981' : v <= 10 ? '#f59e0b' : '#ef4444' });
    }
    if (metrics.yards_per_target != null) {
      const v = metrics.yards_per_target;
      defs.push({ label: 'Yds/Target', fill: Math.min(v / 14 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.yards_per_reception != null) {
      const v = metrics.yards_per_reception;
      defs.push({ label: 'Yds/Reception', fill: Math.min(v / 18 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.yards_after_catch_per_reception != null) {
      const v = metrics.yards_after_catch_per_reception;
      defs.push({ label: 'YAC/Rec', fill: Math.min(v / 12 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.yards_after_catch != null) {
      const v = metrics.yards_after_catch;
      defs.push({ label: 'YAC (season)', fill: Math.min(v / 800 * 100, 100), display: Math.round(v).toString() });
    }
    if (metrics.avg_depth_of_target != null) {
      const v = metrics.avg_depth_of_target;
      defs.push({ label: 'aDOT', fill: Math.min(v / 20 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.contested_catch_rate != null) {
      const v = metrics.contested_catch_rate;
      defs.push({ label: 'Contested Catch %', fill: v, display: v.toFixed(1) + '%' });
    }
    if (metrics.target_share != null) {
      const pct = metrics.target_share * 100;
      defs.push({ label: 'Target Share', fill: pct, display: pct.toFixed(1) + '%' });
    }
    if (metrics.target_quality_score != null) {
      const v = metrics.target_quality_score;
      defs.push({ label: 'Target Quality', fill: Math.min(v, 100), display: v.toFixed(1) });
    }
    // Alignment rates (slot / wide / inline)
    if (metrics.slot_rate != null) {
      const v = metrics.slot_rate;
      defs.push({ label: 'Slot Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%' });
    }
    if (metrics.wide_rate != null) {
      const v = metrics.wide_rate;
      defs.push({ label: 'Wide Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%' });
    }
    if (position === 'TE' && metrics.inline_rate != null) {
      const v = metrics.inline_rate;
      defs.push({ label: 'Inline Rate', fill: Math.min(v, 100), display: v.toFixed(1) + '%' });
    }
  }

  if (metrics.target_quality_score != null && position === 'RB') {
    const v = metrics.target_quality_score;
    defs.push({ label: 'Target Quality', fill: Math.min(v, 100), display: v.toFixed(1) });
  }

  if (metrics.red_zone_usage != null && position !== 'QB') {
    const v = metrics.red_zone_usage;
    defs.push({ label: 'RZ Usage/G', fill: Math.min(v / 3 * 100, 100), display: v.toFixed(1) });
  }

  if (metrics.usage_trend != null) {
    const trend = metrics.usage_trend;
    const icon = trend > 5 ? '<i class="fa-solid fa-arrow-trend-up" aria-hidden="true"></i> ' : trend < -5 ? '<i class="fa-solid fa-arrow-trend-down" aria-hidden="true"></i> ' : '';
    defs.push({
      label: 'Usage Trend',
      fill: Math.min(Math.max((trend + 50) / 100 * 100, 0), 100),
      display: icon + (trend > 0 ? '+' : '') + trend.toFixed(1) + '%',
      forceColor: trend > 5 ? '#10b981' : trend < -5 ? '#ef4444' : null,
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
    });
  }

  if (defs.length === 0) return '';

  let html = '<div class="pm-comp-list">';
  defs.forEach(m => {
    const fill = Math.max(0, Math.min(100, m.fill));
    const color = m.forceColor || (fill >= 60 ? '#10b981' : fill >= 35 ? '#3b82f6' : '#f59e0b');
    const subLine = m.sub ? `<div style="font-size:10px;font-weight:500;opacity:.65;line-height:1;">${m.sub}</div>` : '';
    html += `
      <div class="pm-comp-row">
        <span class="pm-comp-label">${m.label}</span>
        <div class="pm-comp-bar-wrap"><div class="pm-comp-bar" style="width:${fill.toFixed(1)}%;background:${color};"></div></div>
        <div class="pm-comp-val" style="color:${color};">${m.display}${subLine}</div>
      </div>`;
  });
  html += '</div>';

  if (metricsData.as_of_date) {
    html += `<div style="font-size:11px;color:var(--text-muted);margin-top:10px;text-align:right;">As of ${metricsData.as_of_date}</div>`;
  }

  return html;
}

function toggleGameLogYear(year) {
  const content = document.getElementById(`year-${year}`);
  const toggle = document.getElementById(`toggle-${year}`);

  if (content.classList.contains('expanded')) {
    content.classList.remove('expanded');
    toggle.innerHTML = '<i class="fa-solid fa-chevron-right" aria-hidden="true"></i>';
  } else {
    content.classList.add('expanded');
    toggle.innerHTML = '<i class="fa-solid fa-chevron-down" aria-hidden="true"></i>';
  }
}

// ── Watchlist ─────────────────────────────────────────────────────────────────
const _WL_KEY = 'brfantasy_watchlist';

function _getWatchlist() {
  try { return JSON.parse(localStorage.getItem(_WL_KEY) || '[]'); }
  catch { return []; }
}

function _saveWatchlist(list) {
  localStorage.setItem(_WL_KEY, JSON.stringify(list));
  window.dispatchEvent(new Event('watchlist-updated'));
}

function _isWatched(player_id) {
  return _getWatchlist().some(p => p.player_id === player_id);
}

function _toggleWatchlist(player) {
  const list = _getWatchlist();
  const idx = list.findIndex(p => p.player_id === player.player_id);
  if (idx >= 0) list.splice(idx, 1);
  else list.unshift(player);
  _saveWatchlist(list);
}

function _updateWatchlistBtn(btn, player_id) {
  const watched = _isWatched(player_id);
  btn.innerHTML = watched ? '<i class="fa-solid fa-star" aria-hidden="true"></i>' : '<i class="fa-regular fa-star" aria-hidden="true"></i>';
  btn.title = watched ? 'Remove from watchlist' : 'Add to watchlist';
  btn.classList.toggle('player-modal-watchlist-btn--active', watched);
}

function _refreshWatchlistNav() {
  const countEl = document.getElementById('watchlistNavCount');
  const listEl  = document.getElementById('watchlistNavList');
  const list = _getWatchlist();
  if (countEl) {
    countEl.style.display = list.length > 0 ? '' : 'none';
    countEl.textContent   = list.length > 9 ? '9+' : String(list.length);
  }
  if (!listEl) return;
  if (!list.length) {
    listEl.innerHTML = '<div class="watchlist-nav-empty">No players watched yet.<br>Click the <i class="fa-regular fa-star" aria-hidden="true"></i> icon in a player card to add.</div>';
    return;
  }
  listEl.innerHTML = list.map(p =>
    '<div class="watchlist-nav-item" onclick="openPlayerModal(' + JSON.stringify(p.player_id) + ',' + JSON.stringify(p.name || '') + ')">' +
      '<span>' + (p.name || p.player_id) + (p.position ? ' <span style="color:var(--text-muted);font-size:11px">' + p.position + '</span>' : '') + '</span>' +
      '<button class="watchlist-nav-item-remove" onclick="event.stopPropagation();_removeWatchlistNav(' + JSON.stringify(p.player_id) + ')" title="Remove">&times;</button>' +
    '</div>'
  ).join('');
}

function _removeWatchlistNav(player_id) {
  const list = _getWatchlist().filter(p => p.player_id !== player_id);
  _saveWatchlist(list);
  _refreshWatchlistNav();
}

function toggleWatchlistPanel() {
  const panel = document.getElementById('watchlistNavPanel');
  if (!panel) return;
  const isOpen = panel.style.display !== 'none';
  // Close any other open dropdowns
  document.querySelectorAll('.settings-dropdown, .watchlist-nav-panel').forEach(d => { d.style.display = 'none'; });
  if (!isOpen) {
    _refreshWatchlistNav();
    panel.style.display = '';
  }
}

window.addEventListener('watchlist-updated', _refreshWatchlistNav);
document.addEventListener('click', function(e) {
  const wrapper = document.querySelector('.watchlist-nav-wrapper');
  if (wrapper && !wrapper.contains(e.target)) {
    const panel = document.getElementById('watchlistNavPanel');
    if (panel) panel.style.display = 'none';
  }
});
document.addEventListener('DOMContentLoaded', _refreshWatchlistNav);

// Create rkModal structure and CSS if they don't exist (for pages other than rookies page)
function createRkModalIfMissing() {
  // Check if modal already exists
  if (document.getElementById('rkModal')) return;
  
  // Add CSS styles if not already present
  if (!document.getElementById('rkModalStyles')) {
    var css = `
      /* Modal */
      .rk-modal-header {
        padding: 24px 24px 0;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
      }
      .rk-modal-close {
        background: var(--accent-soft);
        border: none;
        width: 32px; height: 32px;
        border-radius: 8px;
        cursor: pointer;
        color: var(--accent);
        font-size: 18px;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
      }
      .rk-modal-body { padding: 16px 24px 24px; }

      /* Hero row */
      .rk-hero-row {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1fr;
        gap: 8px;
        margin-bottom: 10px;
      }
      .rk-hero-stat {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
      }
      .rk-hero-primary {
        background: var(--accent-soft);
        border-color: transparent;
      }
      .rk-hero-label {
        font-size: 10px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
      }
      .rk-hero-val {
        font-size: 26px;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
      }
      .rk-hero-sub {
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      /* Draft + measurables */
      .rk-info-row {
        display: flex;
        align-items: center;
        gap: 12px;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 9px 14px;
        margin-bottom: 10px;
      }
      .rk-meas-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 4px;
      }
      .rk-meas-cell {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
      }
      .rk-meas-label {
        font-size: 10px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
      }
      .rk-meas-val {
        font-size: 14px;
        font-weight: 700;
        color: var(--text);
      }

      /* Section divider */
      .rk-section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 14px 0;
      }

      /* Component breakdown with bars */
      .rk-comp-list {
        display: flex;
        flex-direction: column;
        gap: 9px;
      }
      .rk-comp-row {
        display: grid;
        grid-template-columns: 90px 1fr 32px;
        align-items: center;
        gap: 10px;
      }
      .rk-comp-label {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
      }
      .rk-comp-bar-wrap {
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
      }
      .rk-comp-bar {
        height: 100%;
        border-radius: 3px;
      }
      .rk-comp-val {
        font-size: 12px;
        font-weight: 700;
        text-align: right;
      }

      /* Modal mobile */
      @media (max-width: 480px) {
        .rk-hero-row { grid-template-columns: 1fr 1fr; }
        .rk-hero-primary { grid-column: 1 / -1; }
        .rk-meas-grid { grid-template-columns: repeat(2, 1fr); }
        .rk-comp-row { grid-template-columns: 76px 1fr 28px; gap: 8px; }
      }
    `;
    
    var style = document.createElement('style');
    style.id = 'rkModalStyles';
    style.textContent = css;
    document.head.appendChild(style);
  }
  
  // Create modal HTML structure
  var modalHtml = `
    <div id="rkModal" style="display:none;position:fixed;inset:0;z-index:10500;
         align-items:center;justify-content:center;padding:20px;
         background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);">
      <div id="rkModalContent"
        style="background:var(--card);border-radius:16px;max-width:680px;width:100%;
               max-height:90vh;overflow-y:auto;
               box-shadow:0 24px 48px rgba(15,23,42,0.25);">
      </div>
    </div>
  `;
  
  var tempDiv = document.createElement('div');
  tempDiv.innerHTML = modalHtml;
  document.body.appendChild(tempDiv.firstElementChild);
  
  // Add close on backdrop click
  document.getElementById('rkModal').addEventListener('click', function(e) {
    if (e.target === this) rkCloseModal();
  });
}

// Global rkCloseModal function if it doesn't exist
if (typeof rkCloseModal === 'undefined') {
  function rkCloseModal() {
    var modal = document.getElementById('rkModal');
    if (modal) {
      modal.style.display = 'none';
      document.body.style.overflow = '';
    }
  }
}

// Global prospect modal - used when rkOpenModal (rookies page only) is not defined
function openProspectModal(playerId, playerName) {
  // Use the existing rkModal from the rankings page, or create it if it doesn't exist
  var modal   = document.getElementById('rkModal');
  var content = document.getElementById('rkModalContent');
  
  if (!modal || !content) {
    createRkModalIfMissing();
    modal   = document.getElementById('rkModal');
    content = document.getElementById('rkModalContent');
  }

  // Show loading state
  content.innerHTML = '<div style="text-align:center;padding:40px;"><div class="loading-spinner" style="margin:0 auto 12px;"></div>Loading prospect data…</div>';
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  // Fetch prospect data using the correct API endpoint
  fetch(`/api/prospects/player/${encodeURIComponent(playerId)}`)
    .then(r => {
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      }
      return r.json();
    })
    .then(r => {
      if (r.error) {
        content.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-muted);">${r.error}</div>`;
        return;
      }

      // Use the same data processing as the rankings page
      var val1qb = parseFloat(r.rookie_value||0);
      var valsf  = parseFloat(r.rookie_sf_value||0);
      var score  = parseFloat(r.prospect_score||0);
      var conf   = parseFloat(r.confidence_score||0);
      var age    = r.age != null ? parseFloat(r.age).toFixed(1) : '-';
      var tier   = r.tier || '?';
      var tierColors = ['','#10b981','#3b82f6','#8b5cf6','#f59e0b','#6b7280','#9ca3af'];
      var tierColor  = tierColors[tier] || '#9ca3af';

      var reasons = (r.key_reasons||'').split('\\n').filter(function(l){ return l.trim(); });

      // Measurables
      var ht = r.height_inches;
      var heightStr = ht ? (Math.floor(ht/12) + "'" + (ht%12) + '"') : '-';
      var weightStr = r.weight_lbs ? r.weight_lbs + ' lbs' : '-';
      var fortyStr  = r.forty_yard ? r.forty_yard + 's' : '-';
      var rasStr    = r.ras_score  ? parseFloat(r.ras_score).toFixed(1) + '/10' : '-';

      // Draft info - single consolidated line
      var draftCapLabel = r.draft_capital_label || (r.projected_pick ? 'Pick #' + r.projected_pick : null);
      var draftStr = draftCapLabel
        ? draftCapLabel + (r.num_mocks_used ? '  ·  ' + r.num_mocks_used + ' mocks' : '')
        : 'Undrafted / Unknown';

      // Component scores (Confidence lives in the section header, not here)
      var components = [
        {label:'Production',  val: r.production_score,              color:'#10b981'},
        {label:'Efficiency',  val: r.efficiency_score,              color:'#3b82f6'},
        {label:'Age',         val: r.age_score,                     color:'#8b5cf6'},
        {label:'Breakout',    val: r.breakout_profile_score,        color:'#f59e0b'},
        {label:'Athleticism', val: r.athleticism_score,             color:'#ef4444'},
        {label:'Competition', val: r.competition_score,             color:'#06b6d4'},
        {label:'Draft Cap.',  val: r.projected_draft_capital_score, color:'#f97316'},
      ];

      var compsHtml = components.map(function(c) {
        var v = parseFloat(c.val||0);
        return '<div class="rk-comp-row">' +
          '<div class="rk-comp-label">' + c.label + '</div>' +
          '<div class="rk-comp-bar-wrap"><div class="rk-comp-bar" style="width:' + Math.round(v) + '%;background:' + c.color + ';"></div></div>' +
          '<div class="rk-comp-val" style="color:' + c.color + ';">' + v.toFixed(0) + '</div>' +
        '</div>';
      }).join('');

      var reasonsHtml = reasons.length > 0
        ? '<div class="rk-section-divider"></div>' +
          '<div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px;">Scouting Notes</div>' +
          '<div style="font-size:13px;color:var(--text-muted);line-height:1.7;">' +
            reasons.map(function(l){ return '<div style="padding:2px 0;">' + l + '</div>'; }).join('') +
          '</div>'
        : '';

      // Build the modal content using the same structure as rankings page
      content.innerHTML =
        // ── Header: name + tier badge + close ───────────────────────────────
        '<div class="rk-modal-header">' +
          '<div>' +
            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
              '<span style="font-size:22px;font-weight:700;color:var(--text);">' + (r.name||playerName||'') + '</span>' +
              '<span style="padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;' +
                    'background:' + tierColor + '22;color:' + tierColor + ';border:1px solid ' + tierColor + '44;">' +
                'Tier ' + tier +
              '</span>' +
            '</div>' +
            '<div style="font-size:13px;color:var(--text-muted);margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">' +
              '<span style="font-weight:600;color:var(--text);">' + (r.position||'') + (r.position_rank ? ' #'+r.position_rank : '') + '</span>' +
              (r.school ? '<span style="opacity:.4;">·</span><span>' + r.school + '</span>' : '') +
              '<span style="opacity:.4;">·</span><span>' + age + ' yrs</span>' +
              (r.draft_class_year ? '<span style="opacity:.4;">·</span><span>' + r.draft_class_year + ' Draft</span>' : '') +
            '</div>' +
          '</div>' +
          '<button class="rk-modal-close" onclick="rkCloseModal()">✕</button>' +
        '</div>' +

        '<div class="rk-modal-body">' +

          // ── Hero: Prospect Score + 1QB Value + SF Value ──────────────────
          '<div class="rk-hero-row">' +
            '<div class="rk-hero-stat rk-hero-primary">' +
              '<div class="rk-hero-label">Prospect Score</div>' +
              '<div class="rk-hero-val" style="color:var(--accent);">' + score.toFixed(1) + '</div>' +
              '<div class="rk-hero-sub">' + (r.tier_label||'') + '</div>' +
            '</div>' +
            '<div class="rk-hero-stat">' +
              '<div class="rk-hero-label">1QB Value</div>' +
              '<div class="rk-hero-val">' + (val1qb > 0 ? val1qb.toFixed(1) : '-') + '</div>' +
              '<div class="rk-hero-sub">10-team</div>' +
            '</div>' +
            '<div class="rk-hero-stat">' +
              '<div class="rk-hero-label">SF Value</div>' +
              '<div class="rk-hero-val">' + (valsf > 0 ? valsf.toFixed(1) : '-') + '</div>' +
              '<div class="rk-hero-sub">10-team</div>' +
            '</div>' +
          '</div>' +

          // ── Draft (consolidated) ─────────────────────────────────────────
          '<div class="rk-info-row">' +
            '<span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap;">Draft</span>' +
            '<span style="font-size:13px;font-weight:600;color:var(--text);">' + draftStr + '</span>' +
          '</div>' +

          // ── Measurables ──────────────────────────────────────────────────
          '<div class="rk-meas-grid">' +
            '<div class="rk-meas-cell"><div class="rk-meas-label">Height</div><div class="rk-meas-val">' + heightStr + '</div></div>' +
            '<div class="rk-meas-cell"><div class="rk-meas-label">Weight</div><div class="rk-meas-val">' + weightStr + '</div></div>' +
            '<div class="rk-meas-cell"><div class="rk-meas-label">40 Dash</div><div class="rk-meas-val">' + fortyStr + '</div></div>' +
            '<div class="rk-meas-cell"><div class="rk-meas-label">RAS</div><div class="rk-meas-val">' + rasStr + '</div></div>' +
          '</div>' +

          // ── Component scores with bars ───────────────────────────────────
          '<div class="rk-section-divider"></div>' +
          '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">' +
            '<span style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.04em;">Component Scores</span>' +
            '<span style="font-size:11px;color:var(--text-muted);">Data confidence: <strong style="color:var(--text);">' + conf.toFixed(0) + '</strong></span>' +
          '</div>' +
          '<div class="rk-comp-list">' + compsHtml + '</div>' +

          reasonsHtml +

          // ── Historical Comparables ────────────────────────────────────────
          '<div class="rk-section-divider"></div>' +
          '<div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px;">Historical Comparables</div>' +
          '<div id="rkComparablesBody" style="font-size:13px;color:var(--text-muted);">' +
            '<div style="display:flex;align-items:center;gap:8px;"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading…</div>' +
          '</div>' +

        '</div>';

      // Auto-link to Sleeper ID silently in background
      if (r.player_id) {
        fetch('/api/prospects/auto-link/' + encodeURIComponent(r.player_id)).catch(function(){});
      }

      // Fetch comparables
      fetch('/api/prospects/comparables/' + encodeURIComponent(r.player_id))
        .then(function(res){ if (!res.ok) throw new Error('HTTP ' + res.status); return res.json(); })
        .then(function(cd) {
          var cb = document.getElementById('rkComparablesBody');
          if (!cb) return;
          var comps = cd.comparables || [];
          if (!comps.length) {
            cb.innerHTML = '<span>No close historical comps found.</span>';
            return;
          }
          var tc_ = ['','#10b981','#3b82f6','#8b5cf6','#f59e0b','#6b7280','#9ca3af'];
          cb.innerHTML = comps.map(function(c) {
            var tc = tc_[c.tier] || '#9ca3af';
            var pickStr = c.actual_pick ? ' · Pick ' + c.actual_pick : '';
            return '<div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);">' +
              '<div>' +
                '<span style="font-weight:600;color:var(--text);font-size:13px;">' + c.name + '</span>' +
                '<span style="color:var(--text-muted);font-size:12px;margin-left:6px;">' + c.draft_class_year + pickStr + '</span>' +
                (c.school ? '<span style="color:var(--text-muted);font-size:12px;"> · ' + c.school + '</span>' : '') +
              '</div>' +
              '<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">' +
                '<span style="font-size:12px;color:var(--text-muted);">' + parseFloat(c.prospect_score).toFixed(1) + '</span>' +
                '<span style="padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;background:' + tc + '22;color:' + tc + ';border:1px solid ' + tc + '44;">T' + c.tier + '</span>' +
              '</div>' +
            '</div>';
          }).join('');
        })
        .catch(function() {
          var cb = document.getElementById('rkComparablesBody');
          if (cb) cb.innerHTML = '<span>Could not load comparables.</span>';
        });
    })
    .catch(function(err) {
      console.error('Error loading prospect data:', err);
      content.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-muted);">Could not load prospect data.</div>';
    });
}

function closePlayerModal() {
  const overlay = document.querySelector('.player-modal-overlay');
  if (overlay) {
    document.body.style.overflow = '';
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 200);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Player Comparison
// ─────────────────────────────────────────────────────────────────────────────

let _playerListCache = null;

function _cmpFuzzyScore(name, query) {
  if (!name || !query) return 0;
  const n = name.toLowerCase();
  const q = query.toLowerCase();
  if (n.includes(q)) return 100 + (100 - n.indexOf(q));
  const nWords = n.split(/[\s\-]+/);
  if (nWords.some(w => w.startsWith(q))) return 60;
  if (q.length >= 4) {
    for (let i = 0; i < q.length; i++) {
      const del = q.slice(0, i) + q.slice(i + 1);
      if (n.includes(del)) return 40;
    }
  }
  return 0;
}

function openCompareSearch(player1Data) {
  const modal = document.getElementById('playerModal');
  const body = document.getElementById('playerModalBody');
  if (!modal || !body) return;

  modal.classList.add('compare-mode');
  const tabBar = document.getElementById('pmTabBar');
  if (tabBar) tabBar.style.display = 'none';

  body.innerHTML = `
    <div class="compare-search-panel">
      <div class="compare-search-header">
        <div class="compare-search-p1">
          <img src="${player1Data.espnHeadshot || ''}" class="compare-search-headshot" alt="${player1Data.name}" />
          <div>
            <div class="compare-search-p1-name">${player1Data.name}</div>
            <div class="compare-search-p1-meta">${player1Data.position || ''} · ${player1Data.team || ''}</div>
          </div>
        </div>
        <div class="compare-vs-badge">VS</div>
        <div class="compare-search-picker">
          <input
            type="text"
            id="comparePlayerInput"
            class="compare-search-input"
            placeholder="Search for a player to compare..."
            autocomplete="off"
          />
          <div id="compareSearchResults" class="compare-search-results"></div>
        </div>
      </div>
    </div>
  `;

  const input = document.getElementById('comparePlayerInput');
  const resultsBox = document.getElementById('compareSearchResults');
  input.focus();

  function renderResults(players) {
    if (!players.length) {
      resultsBox.innerHTML = '<div class="compare-search-empty">No players found</div>';
      return;
    }
    resultsBox.innerHTML = players.slice(0, 10).map(p => `
      <div class="compare-search-result-item" data-pid="${p.player_id}" data-pname="${p.name}">
        <img src="${p.espnHeadshot || ''}" class="compare-result-headshot" alt="${p.name}" />
        <div class="compare-result-info">
          <div class="compare-result-name">${p.name}</div>
          <div class="compare-result-meta">${p.position || ''} · ${p.team || ''}</div>
        </div>
        <div class="compare-result-value">${p.value || '-'}</div>
      </div>
    `).join('');

    resultsBox.querySelectorAll('.compare-search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        const pid = item.dataset.pid;
        const pname = item.dataset.pname;
        // Show loading state
        body.innerHTML = '<div class="player-modal-loading"><div class="loading-spinner"></div><div>Loading comparison...</div></div>';
        // Fetch second player details
        const pathParts = window.location.pathname.split('/').filter(p => p);
        const platform = pathParts[0] || 'sleeper';
        const season = pathParts[1] || new Date().getFullYear();
        const leagueId = pathParts[2] || null;
        const apiUrl = leagueId
          ? `/api/player-details/${pid}?league_id=${leagueId}&platform=${platform}&season=${season}`
          : `/api/player-details/${pid}`;
        fetch(apiUrl)
          .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
          .then(p2Data => openComparisonView(player1Data, p2Data))
          .catch(() => {
            body.innerHTML = '<div class="player-modal-loading"><div style="color:#ef4444;">Failed to load player data</div></div>';
          });
      });
    });
  }

  let _debounce;
  let _currentQuery = '';

  function doSearch(q) {
    if (!q || q.length < 2) { resultsBox.innerHTML = ''; return; }
    _currentQuery = q;
    resultsBox.innerHTML = '<div class="compare-search-empty" style="opacity:.5;">Searching...</div>';
    fetch(`/api/players?q=${encodeURIComponent(q)}&limit=20`)
      .then(r => r.json())
      .then(list => {
        if (_currentQuery !== q) return; // stale response
        // Support both paginated {players:[...]} and flat [...] responses
        const players = Array.isArray(list) ? list : (list.players || []);
        renderResults(players.filter(p => p.player_id !== player1Data.player_id));
      })
      .catch(() => {
        resultsBox.innerHTML = '<div class="compare-search-empty" style="color:#ef4444;">Search failed</div>';
      });
  }

  input.addEventListener('input', () => {
    clearTimeout(_debounce);
    const q = input.value.trim();
    if (!q || q.length < 2) { resultsBox.innerHTML = ''; return; }
    _debounce = setTimeout(() => doSearch(q), 250);
  });
}

function _computeSeasonStats(p) {
  const yearMap = p.game_logs_by_year || {};
  const latestYear = Object.keys(yearMap).sort((a, b) => b - a)[0];
  if (!latestYear) return { ppg: null, total: null, season: null, games: 0 };
  const logs = yearMap[latestYear] || [];
  const played = logs.filter(g => parseFloat(g.fantasy_pts || 0) > 0);
  const total = played.reduce((s, g) => s + parseFloat(g.fantasy_pts || 0), 0);
  return {
    ppg: played.length > 0 ? (total / played.length).toFixed(1) : null,
    total: played.length > 0 ? total.toFixed(1) : null,
    season: latestYear,
    games: played.length,
  };
}

function _buildComparePPGRow(p1, p2) {
  // game_logs_by_year is lazy-loaded and empty at compare time; use stats directly
  const ppg1 = p1.stats?.ppg;
  const ppg2 = p2.stats?.ppg;
  if (ppg1 == null && ppg2 == null) return '';

  const season = p1.stats?.ppg_season || p2.stats?.ppg_season || '';

  function scoringBlock(p) {
    const pos = p.position || '';
    const ppg = p.stats?.ppg;
    const total = p.stats?.total_pts;
    const games = p.stats?.ppg_games;
    const ppgRank = p.stats?.ppg_rank;
    const totalRank = p.stats?.total_pts_rank;
    const valLine = ppg != null
      ? `${ppg}${total != null ? ` | ${total}` : ''}`
      : '-';
    const rankLine = `PPG · ${ppgRank ? `${pos}${ppgRank}` : '-'} | TOTAL · ${totalRank ? `${pos}${totalRank}` : '-'}`;
    return {
      cell: `<div class="compare-pts-cell">
        <div class="compare-pts-val">${valLine}</div>
        <div class="compare-pts-label">${rankLine}</div>
      </div>`,
      games,
    };
  }

  const b1 = scoringBlock(p1);
  const b2 = scoringBlock(p2);

  return `
    <div class="compare-pts-row">
      <div class="compare-pts-player">
        <div class="compare-pts-stats">${b1.cell}</div>
        ${b1.games ? `<div class="compare-pts-meta">${season} · ${b1.games}g</div>` : ''}
      </div>
      <div class="compare-pts-divider"></div>
      <div class="compare-pts-player compare-pts-player-right">
        <div class="compare-pts-stats">${b2.cell}</div>
        ${b2.games ? `<div class="compare-pts-meta">${season} · ${b2.games}g</div>` : ''}
      </div>
    </div>
  `;
}

function _buildCompareHeroHTML(p) {
  const val1qb = p.stats?.value || 0;
  const valsf  = p.stats?.sf_value || 0;
  const posRankLabel = p.stats?.pos_rank_label || (p.stats?.pos_rank ? `${p.position}${p.stats.pos_rank}` : '-');
  const pos    = p.position || '';
  const ppg    = p.stats?.ppg;
  const ppgRank = p.stats?.ppg_rank;
  const total  = p.stats?.total_pts;
  const totalRank = p.stats?.total_pts_rank;
  const season = p.stats?.ppg_season ? ` · ${p.stats.ppg_season}` : '';
  const ppgCard = ppg != null ? `
      <div class="pm-hero-stat" style="padding:10px 10px;">
        <div class="pm-hero-label">PPG${season}</div>
        <div class="pm-hero-val" style="font-size:20px;">${ppg}</div>
        <div class="pm-hero-sub">${ppgRank ? `${pos}${ppgRank}` : '-'}</div>
      </div>` : '';
  const totalCard = total != null ? `
      <div class="pm-hero-stat" style="padding:10px 10px;">
        <div class="pm-hero-label">Total${season}</div>
        <div class="pm-hero-val" style="font-size:20px;">${total}</div>
        <div class="pm-hero-sub">${totalRank ? `${pos}${totalRank}` : '-'}</div>
      </div>` : '';
  const cardCount = 3 + (ppgCard ? 1 : 0) + (totalCard ? 1 : 0);
  return `
    <div class="compare-hero-row" style="grid-template-columns:repeat(${cardCount},1fr);">
      <div class="pm-hero-stat pm-hero-primary" style="padding:10px 10px;">
        <div class="pm-hero-label">1QB Value</div>
        <div class="pm-hero-val" style="font-size:20px;color:#3b82f6;">${val1qb > 0 ? val1qb : '-'}</div>
      </div>
      <div class="pm-hero-stat" style="padding:10px 10px;">
        <div class="pm-hero-label">SF Value</div>
        <div class="pm-hero-val" style="font-size:20px;">${valsf > 0 ? valsf : '-'}</div>
      </div>
      <div class="pm-hero-stat" style="padding:10px 10px;">
        <div class="pm-hero-label">Pos Rank</div>
        <div class="pm-hero-val" style="font-size:20px;">${posRankLabel}</div>
      </div>
      ${ppgCard}
      ${totalCard}
    </div>
  `;
}

function _buildComparePlayerHeader(p) {
  const sep = '<span style="opacity:.3;margin:0 4px;">·</span>';
  const metaParts = [];
  if (p.position) metaParts.push(`<span style="font-weight:600;">${p.position}</span>`);
  if (p.team) metaParts.push(`<span>${p.team}</span>`);
  const _pAge = parseFloat(p.age);
  if (!isNaN(_pAge)) metaParts.push(`<span>${_pAge.toFixed(1)} yrs</span>`);
  const _ownerLine = ''; // omit fantasy team from compare header — too cluttered
  return `
    <div class="compare-player-header">
      <img src="${p.espnHeadshot || ''}" class="compare-player-headshot" alt="${p.name}" />
      <div class="compare-player-header-info">
        <div class="compare-player-name">${p.name}</div>
        <div class="compare-player-meta">${metaParts.join(sep)}</div>
        ${_ownerLine}
      </div>
    </div>
  `;
}

function renderCompareMetricRows(m1, m2, p1, p2) {
  // Build a map of metrics present in both players
  const allKeys = new Set([...Object.keys(m1 || {}), ...Object.keys(m2 || {})]);
  
  // Position-aware metric groups
  const qbMetrics = [
    'completion_pct', 'yards_per_attempt', 'td_rate', 'int_rate', 'nfl_passer_rating',
    'pff_passing_grade', 'big_time_throw_rate', 'adjusted_completion_rate', 
    'pressure_to_sack_rate', 'snap_share', 'role_score'
  ];
  
  const rbMetrics = [
    'yards_per_carry', 'yards_per_touch', 'rush_td_rate', 'snap_share', 
    'opportunity_share', 'red_zone_usage', 'role_score', 'explosive_runs_10_plus',
    'breakaway_percentage', 'elusive_rating', 'pff_rushing_grade', 'grades_offense'
  ];
  
  const wrTeMetrics = [
    'yards_per_target', 'catch_rate', 'yards_per_reception', 'target_quality_score',
    'snap_share', 'opportunity_share', 'red_zone_usage', 'role_score',
    'yards_after_catch', 'yards_after_catch_per_reception', 'avg_depth_of_target',
    'contested_catch_rate', 'avoided_tackles', 'drop_rate', 'slot_rate',
    'wide_rate', 'inline_rate', 'grades_offense'
  ];
  
  const olMetrics = [
    'snap_share', 'pass_block_rate', 'grades_offense', 'grades_pass_block'
  ];
  
  const allLabelMap = {
    // Receiving metrics
    yards_per_target: 'Yards/Target',
    catch_rate: 'Catch Rate',
    yards_per_reception: 'Yards/Rec',
    target_quality_score: 'Target Quality',
    yards_after_catch: 'YAC',
    yards_after_catch_per_reception: 'YAC/Rec',
    avg_depth_of_target: 'aDOT',
    contested_catch_rate: 'Contested Catch %',
    drop_rate: 'Drop Rate',
    slot_rate: 'Slot Rate',
    wide_rate: 'Wide Rate',
    inline_rate: 'Inline Rate',
    
    // Rushing metrics
    yards_per_carry: 'Yards/Carry',
    yards_per_touch: 'Yards/Touch',
    rush_td_rate: 'Rush TD Rate',
    explosive_runs_10_plus: '10+ Yd Runs',
    breakaway_percentage: 'Breakaway %',
    elusive_rating: 'Elusive Rating',
    pff_rushing_grade: 'PFF Rush Grade',
    
    // Passing metrics
    completion_pct: 'Completion %',
    yards_per_attempt: 'Yards/Attempt',
    td_rate: 'TD Rate',
    int_rate: 'INT Rate',
    nfl_passer_rating: 'Passer Rating',
    pff_passing_grade: 'PFF Pass Grade',
    big_time_throw_rate: 'BTT Rate',
    adjusted_completion_rate: 'Adj Comp %',
    pressure_to_sack_rate: 'P2S Rate',
    
    // Usage/Role metrics
    snap_share: 'Snap Share',
    opportunity_share: 'Opportunity Share',
    red_zone_usage: 'Red Zone Usage',
    role_score: 'Role Score',
  };
  
  // Determine relevant metrics based on positions
  let relevantMetrics = [];
  const pos1 = (p1?.position || '').toUpperCase();
  const pos2 = (p2?.position || '').toUpperCase();
  
  if (pos1 === 'QB' || pos2 === 'QB') {
    relevantMetrics.push(...qbMetrics);
  }
  if (pos1 === 'RB' || pos2 === 'RB') {
    relevantMetrics.push(...rbMetrics);
  }
  if ((pos1 === 'WR' || pos1 === 'TE') || (pos2 === 'WR' || pos2 === 'TE')) {
    relevantMetrics.push(...wrTeMetrics);
  }
  if ((pos1 === 'OT' || pos1 === 'OG' || pos1 === 'C') || (pos2 === 'OT' || pos2 === 'OG' || pos2 === 'C')) {
    relevantMetrics.push(...olMetrics);
  }
  
  // Add universal metrics if no position-specific ones found
  if (relevantMetrics.length === 0) {
    relevantMetrics = ['snap_share', 'role_score', 'grades_offense'];
  }
  
  // Filter to only include relevant metrics that exist in the data
  const displayKeys = relevantMetrics.filter(k => 
    allLabelMap[k] && allKeys.has(k) && (m1?.[k] != null || m2?.[k] != null)
  );
  if (!displayKeys.length) return '<div style="color:var(--text-muted);font-size:13px;padding:8px 0;">No shared metrics available</div>';

  // Determine max per metric for bar scaling using metric-specific ranges
  return displayKeys.map(key => {
    const v1 = m1?.[key] ?? null;
    const v2 = m2?.[key] ?? null;
    
    // Metric-specific scaling ranges for meaningful comparison
    const metricRanges = {
      // Percentage metrics (0-100)
      'catch_rate': 1, 'completion_pct': 100, 'contested_catch_rate': 100,
      'drop_rate': 100, 'slot_rate': 100, 'wide_rate': 100, 'inline_rate': 100,
      'pass_block_rate': 100, 'snap_share': 1, 'opportunity_share': 25,
      'red_zone_usage': 4,
      
      // Rate metrics (0-10 or 0-20)
      'rush_td_rate': .05, 'td_rate': 8, 'int_rate': 3, 'avoided_tackles': 1,
      'explosive_runs_10_plus': 20,
      
      // Yards per attempt metrics (0-20)
      'yards_per_target': 20, 'yards_per_reception': 20, 'yards_per_carry': 10,
      'yards_per_touch': 15, 'yards_per_attempt': 15, 'avg_depth_of_target': 20,
      'yards_after_catch': 20, 'yards_after_catch_per_reception': 20,
      
      // PFF grades (0-100)
      'pff_passing_grade': 100, 'pff_rushing_grade': 100, 'grades_offense': 100,
      'grades_pass_block': 100,
      
      // Role score (0-100)
      'role_score': 100,
      
      'big_time_throw_rate': 1, 'adjusted_completion_rate': 1,
      
      // Rating metrics (0-160)
      'nfl_passer_rating': 160,
      
      // Other metrics
      'target_quality_score': 50, 'elusive_rating': 100,
    };
    
    const range = metricRanges[key] || 100; // Default to 100 if not specified
    const pct1 = v1 != null ? Math.min(100, Math.round((v1 / range) * 100)) : 0;
    const pct2 = v2 != null ? Math.min(100, Math.round((v2 / range) * 100)) : 0;

    function barColor(pct, raw) {
      if (raw == null) return '#374151';
      
      // Inverse metrics where lower is better (INT rate, drop rate)
      const inverseMetrics = ['int_rate', 'drop_rate', 'fumble_rate'];
      const isInverse = inverseMetrics.includes(key);
      
      if (isInverse) {
        // For inverse metrics: lower values are better (green), higher are worse (red)
        if (pct <= 20) return '#10b981';  // Excellent
        if (pct <= 40) return '#3b82f6';  // Good
        return '#f59e0b';  // Poor
      } else {
        // For normal metrics: higher values are better
        if (pct >= 60) return '#10b981';
        if (pct >= 35) return '#3b82f6';
        return '#f59e0b';
      }
    }

    const fmt = v => {
      if (v == null) return 'â';
      
      // Metrics that should be displayed as percentages
      const percentageMetrics = [
        'catch_rate', 'snap_share', 'rush_td_rate',
        'big_time_throw_rate', 'adjusted_completion_rate', 'pressure_to_sack_rate',
        'breakaway_percentage'
      ];
      
      // Check if current metric is a percentage metric
      const isPercentageMetric = percentageMetrics.includes(key);
      
      if (isPercentageMetric) {
        // Display as percentage (e.g., .6 becomes 60.0%)
        if (v < 0.1 && v > 0) return (v * 100).toFixed(1) + '%';
        return (v * 100).toFixed(0) + '%';
      } else {
        // Regular formatting for other metrics
        if (v < 0.1 && v > 0) return v.toFixed(3);
        return Number.isInteger(v) ? v : v.toFixed(1);
      }
    };

    return `
      <div class="compare-metric-row">
        <div class="compare-metric-p1-val">${fmt(v1)}</div>
        <div class="compare-bar-left">
          <div class="compare-bar-fill" style="width:${pct1}%;background:${barColor(pct1, v1)};"></div>
        </div>
        <div class="compare-metric-label">${allLabelMap[key]}</div>
        <div class="compare-bar-right">
          <div class="compare-bar-fill" style="width:${pct2}%;background:${barColor(pct2, v2)};"></div>
        </div>
        <div class="compare-metric-p2-val">${fmt(v2)}</div>
      </div>
    `;
  }).join("");
}function loadCompareMetrics(playerId1, playerId2, season) {
  const metricsUrl = (pid, season) => {
    if (season === null) {
      return `/api/player-advanced-metrics/${pid}?season=career`;
    }
    return `/api/player-advanced-metrics/${pid}?season=${season}`;
  };

  Promise.all([
    fetch(metricsUrl(playerId1, season)).then(r => r.json()).catch(() => ({})),
    fetch(metricsUrl(playerId2, season)).then(r => r.json()).catch(() => ({})),
  ]).then(([data1, data2]) => {
    const metricsDiv = document.getElementById('compareMetricsContent');
    if (metricsDiv) {
      // Extract metrics from the nested structure returned by API
      const m1 = data1.metrics || {};
      const m2 = data2.metrics || {};
      
      // Get available seasons from both players
      const seasons1 = data1.available_seasons || [];
      const seasons2 = data2.available_seasons || [];
      const allSeasons = [...new Set([...seasons1, ...seasons2])].sort((a, b) => b - a);
      
      // Rebuild season selector with updated active state
      let seasonSelectorHTML = '';
      if (allSeasons.length > 1) {
        seasonSelectorHTML = `
          <div class="compare-season-selector">
            <div class="compare-season-label">Season:</div>
            <div class="compare-season-pills">
              <button class="adv-season-pill ${season === null ? 'active' : ''}" 
                      onclick="loadCompareMetrics('${playerId1}', '${playerId2}', null)">
                Career
              </button>
              ${allSeasons.map(s => `
                <button class="adv-season-pill ${season === s ? 'active' : ''}" 
                        onclick="loadCompareMetrics('${playerId1}', '${playerId2}', ${s})">
                  ${s}
                </button>
              `).join('')}
            </div>
          </div>
        `;
      }
      
      const rows = renderCompareMetricRows(m1, m2, data1, data2);
      metricsDiv.innerHTML = `
        ${seasonSelectorHTML}
        ${rows}
      `;
    }
  });
}

function openComparisonView(p1, p2) {
  const modal = document.getElementById('playerModal');
  const body = document.getElementById('playerModalBody');
  if (!modal || !body) return;

  modal.classList.add('compare-mode');

  // Update the header to show both players instead of single player
  const headerTitleSection = modal.querySelector('.player-modal-title-section');
  const headshotContainer = modal.querySelector('.player-modal-headshot-container');
  if (headshotContainer) headshotContainer.style.display = 'none';
  if (headerTitleSection) {
    headerTitleSection.innerHTML = `
      <div class="compare-dual-header">
        ${_buildComparePlayerHeader(p1)}
        <div class="compare-vs-badge">VS</div>
        ${_buildComparePlayerHeader(p2)}
      </div>
    `;
  }

  // Build the comparison body
  body.innerHTML = `
    <div class="compare-body">
      <div class="compare-hero-section">
        <div class="compare-hero-player" id="compareHero1" data-name="${p1.full_name || ''}">${_buildCompareHeroHTML(p1)}</div>
        <div class="compare-hero-player" id="compareHero2" data-name="${p2.full_name || ''}">${_buildCompareHeroHTML(p2)}</div>
      </div>

      <hr class="pm-section-divider">

      <div class="pm-section-header"><span class="pm-section-label">Advanced Metrics Comparison</span></div>
      <div id="compareMetricsContent" class="compare-metrics-section">
        <div style="display:flex;align-items:center;gap:10px;padding:12px 0;">
          <div class="loading-spinner" style="width:16px;height:16px;"></div>
          <span style="font-size:13px;color:var(--text-muted);">Loading metrics...</span>
        </div>
      </div>

      <hr class="pm-section-divider">

      <div class="pm-section-header"><span class="pm-section-label">Value History</span></div>
      <div id="compareValueChart" class="player-modal-chart-container" style="min-height:200px;"></div>

      <div class="compare-nav-btns">
        <button class="compare-back-btn" id="compareBackBtn">← Back to ${p1.name}</button>
        <button class="compare-profile-btn" id="compareP2ProfileBtn">${p2.name}'s Profile →</button>
      </div>
    </div>
  `;

  document.getElementById('compareBackBtn')?.addEventListener('click', () => {
    closePlayerModal();
    openPlayerModal(p1.player_id, p1.name);
  });

  document.getElementById('compareP2ProfileBtn')?.addEventListener('click', () => {
    closePlayerModal();
    openPlayerModal(p2.player_id, p2.name);
  });

  // Fetch NFL state to determine if it's offseason
  fetch('/api/nfl-state').then(r => r.json()).catch(() => ({}))
    .then(nflState => {
      const isOffseason = (nflState.season_type || '').toLowerCase() === 'off';
      const currentSeason = nflState.season || new Date().getFullYear();
      
      // During offseason, default to career metrics (no season parameter)
      const defaultSeason = isOffseason ? null : currentSeason;
      
      // Fetch advanced metrics for both players in parallel
      const metricsUrl = (pid, season) => {
        if (season === null) {
          return `/api/player-advanced-metrics/${pid}?season=career`;
        }
        return `/api/player-advanced-metrics/${pid}?season=${season}`;
      };

      Promise.all([
        fetch(metricsUrl(p1.player_id, defaultSeason)).then(r => r.json()).catch(() => ({})),
        fetch(metricsUrl(p2.player_id, defaultSeason)).then(r => r.json()).catch(() => ({})),
      ]).then(([data1, data2]) => {
        const metricsDiv = document.getElementById('compareMetricsContent');
        if (metricsDiv) {
          // Extract metrics from the nested structure returned by API
          const m1 = data1.metrics || {};
          const m2 = data2.metrics || {};
          
          // Get available seasons from both players
          const seasons1 = data1.available_seasons || [];
          const seasons2 = data2.available_seasons || [];
          const allSeasons = [...new Set([...seasons1, ...seasons2])].sort((a, b) => b - a);
          
          // Build season selector if multiple seasons available
          let seasonSelectorHTML = '';
          if (allSeasons.length > 1) {
            const activeSeason = defaultSeason || allSeasons[0];
            seasonSelectorHTML = `
              <div class="compare-season-selector">
                <div class="compare-season-label">Season:</div>
                <div class="compare-season-pills">
                  <button class="adv-season-pill ${activeSeason === null ? 'active' : ''}" 
                          onclick="loadCompareMetrics('${p1.player_id}', '${p2.player_id}', null)">
                    Career
                  </button>
                  ${allSeasons.map(season => `
                    <button class="adv-season-pill ${activeSeason === season ? 'active' : ''}" 
                            onclick="loadCompareMetrics('${p1.player_id}', '${p2.player_id}', ${season})">
                      ${season}
                    </button>
                  `).join('')}
                </div>
              </div>
            `;
          }
          
          const rows = renderCompareMetricRows(m1, m2, p1, p2);
          metricsDiv.innerHTML = `
            ${seasonSelectorHTML}
            ${rows}
          `;
        }
      });
    });

  // Render value history chart with two lines
  if (typeof Plotly !== 'undefined') {
    const chartDiv = document.getElementById('compareValueChart');
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)';
    const textColor = isDark ? '#9ca3af' : '#6b7280';
    const bgColor = isDark ? '#1e293b' : '#f8fafc';

    const makeTrace = (history, name, color) => {
      const dates = (history || []).map(h => h.as_of_date || h.date);
      const vals  = (history || []).map(h => h.value);
      return {
        x: dates, y: vals, mode: 'lines', name,
        line: { color, width: 2.5 },
        hovertemplate: `<b>${name}</b><br>%{x}<br>Value: %{y}<extra></extra>`,
      };
    };

    const traces = [
      makeTrace(p1.value_history, p1.name, '#3b82f6'),
      makeTrace(p2.value_history, p2.name, '#f59e0b'),
    ];

    Plotly.newPlot(chartDiv, traces, {
      paper_bgcolor: 'transparent',
      plot_bgcolor: bgColor,
      margin: { t: 10, r: 16, b: 40, l: 46 },
      xaxis: { showgrid: false, tickfont: { size: 11, color: textColor }, zeroline: false },
      yaxis: { showgrid: true, gridcolor: gridColor, tickfont: { size: 11, color: textColor }, zeroline: false },
      legend: { font: { size: 12, color: textColor }, bgcolor: 'transparent', orientation: 'h', x: 0, y: 1.1 },
      hovermode: 'x unified',
      showlegend: true,
    }, { responsive: true, displayModeBar: false });
  }
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closePlayerModal();
  }
});
// Make player modal work site-wide
document.addEventListener('DOMContentLoaded', function() {
  initGlobalPlayerModals();
  loadGlobalPlayerIndicators();
  addBreakoutBadgesToTeamsPage();
});

// Add breakout badges to teams page
function addBreakoutBadgesToTeamsPage() {
  // Wait a bit for indicators to load, then process players
  setTimeout(() => {
    const players = document.querySelectorAll('[data-breakout-check="true"]');
    players.forEach(playerEl => {
      const playerId = playerEl.dataset.playerId;
      const badges = [];
      
      // Get player data from data attributes
      const value = parseFloat(playerEl.dataset.value) || 0;
      const position = playerEl.dataset.position || '';
      const yearsExp = playerEl.dataset.yearsExp;
      
      // Position-specific elite thresholds
      const eliteThresholds = {
        'RB': 650,   // Elite young backs
        'WR': 650,   // Elite WRs
        'TE': 500,   // Premium TE scarcity
        'QB': 360,   // Solid QB1s
        'K': 9999,   // No elite kickers
        'DEF': 9999  // No elite defenses
      };
      const threshold = eliteThresholds[position] || 750;
      const isElite = value >= threshold;
      const isRookie = yearsExp !== null && yearsExp !== '' && parseInt(yearsExp) === 0;
      
      if (isElite) {
        badges.push('<span class="player-badge player-badge-elite"><i class="fa-solid fa-star" aria-hidden="true"></i> ELITE</span>');
      }
      if (isRookie) {
        badges.push('<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>');
      }
      if (!isElite && isBreakout(playerId)) {
        badges.push('<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>');
      }
      
      // Add badges after player name
      if (badges.length > 0) {
        const badgeContainer = document.createElement('span');
        badgeContainer.innerHTML = badges.join(' ');
        badgeContainer.style.marginLeft = '5px';
        playerEl.parentNode.insertBefore(badgeContainer, playerEl.nextSibling);
      }
      
      // Remove the check attribute
      playerEl.removeAttribute('data-breakout-check');
    });
  }, 1000);
}

let _globalPlayerModalsReady = false;
function initGlobalPlayerModals() {
  // Guard: only attach the delegated listener once
  if (_globalPlayerModalsReady) return;
  _globalPlayerModalsReady = true;

  document.addEventListener('click', function(e) {
    const target = e.target.closest('[data-player-id]');
    if (target && target.dataset.playerId) {
      const playerId = target.dataset.playerId;
      const playerName = target.dataset.playerName || target.textContent || 'Player';

      // Don't interfere with chip remove buttons
      if (e.target.classList.contains('chip-remove')) {
        return;
      }

      e.preventDefault();
      e.stopPropagation();
      openPlayerModal(playerId, playerName);
    }
  });
}

// Helper function to make any element open player modal
function makePlayerClickable(element, playerId, playerName) {
  element.dataset.playerId = playerId;
  element.dataset.playerName = playerName;
  element.style.cursor = 'pointer';
  element.classList.add('player-clickable');
}

// ============================================
// Team Modal Functions
// ============================================

function openTeamModal(rosterId, teamName) {
  // Create modal overlay
  const overlay = document.createElement('div');
  overlay.className = 'team-modal-overlay';
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      closeTeamModal();
    }
  });

  // Create modal
  const modal = document.createElement('div');
  modal.className = 'team-modal';
  modal.id = 'teamModal';

  window._tmRosterId = rosterId;

  modal.innerHTML = `
    <div class="team-modal-header">
      <div class="team-modal-avatar" id="teamModalAvatar">
        <div class="loading-spinner" style="width: 32px; height: 32px;"></div>
      </div>
      <div class="team-modal-title-section">
        <h2 class="team-modal-name">${teamName || 'Loading...'}</h2>
        <div class="team-modal-meta" id="teamModalMeta">
          <div class="loading-spinner" style="width: 16px; height: 16px;"></div>
        </div>
      </div>
      <button class="team-modal-close" onclick="closeTeamModal()">×</button>
    </div>
    <div class="tm-tab-bar">
      <button class="tm-tab active" data-tab="roster" onclick="tmSwitchTab('roster')">Roster</button>
      <button class="tm-tab" data-tab="charts" onclick="tmSwitchTab('charts')">Charts</button>
      <button class="tm-tab" data-tab="trades" onclick="tmSwitchTab('trades')">Trades</button>
    </div>
    <div class="team-modal-body">
      <div class="tm-panel active" id="tm-panel-roster">
        <div class="team-modal-loading">
          <div class="loading-spinner"></div>
          <div>Loading team details...</div>
        </div>
      </div>
      <div class="tm-panel" id="tm-panel-charts"></div>
      <div class="tm-panel" id="tm-panel-trades"></div>
    </div>
  `;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);
  document.body.style.overflow = 'hidden';

  // Fetch team details
  fetchTeamDetails(rosterId);
}

function closeTeamModal() {
  const overlay = document.querySelector('.team-modal-overlay');
  const modal = document.getElementById('teamModal');

  if (overlay) overlay.remove();
  if (modal) modal.remove();
  document.body.style.overflow = '';
  window._tmRosterId = null;
  window._tmTradesLoaded = false;
}

function tmSwitchTab(tab) {
  document.querySelectorAll('.tm-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tm-tab').forEach(t => t.classList.remove('active'));
  const panel = document.getElementById('tm-panel-' + tab);
  const btn = document.querySelector('.tm-tab[data-tab="' + tab + '"]');
  if (panel) panel.classList.add('active');
  if (btn) btn.classList.add('active');

  if (tab === 'trades' && !window._tmTradesLoaded) {
    window._tmTradesLoaded = true;
    tmLoadTrades(window._tmRosterId);
  }
}

async function tmLoadTrades(rosterId) {
  const panel = document.getElementById('tm-panel-trades');
  if (!panel) return;
  panel.innerHTML = '<div class="team-modal-loading"><div class="loading-spinner"></div><div>Loading trade history…</div></div>';

  try {
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const platform = pathParts[0] || 'sleeper';
    const season = pathParts[1] || new Date().getFullYear();
    const leagueId = pathParts[2];

    if (!leagueId) throw new Error('League ID not found');

    const res = await fetch(`/api/team-trades/${rosterId}?league_id=${leagueId}&platform=${platform}&season=${season}`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const trades = data.trades || [];
    if (!trades.length) {
      panel.innerHTML = '<div class="player-modal-loading" style="padding:32px 0;"><div style="color:var(--text-muted);font-size:13px;">No trades found for this team this season.</div></div>';
      return;
    }

    const renderAssets = (players, picks) => {
      const parts = [
        ...players.map(p => `<div class="pm-trade-asset player-clickable" data-player-id="${p.player_id}" data-player-name="${p.name}" style="cursor:pointer">${p.name}${p.position ? `<span style="font-size:11px;color:var(--text-muted);margin-left:4px;">${p.position}</span>` : ''}</div>`),
        ...picks.map(p => `<div class="pm-trade-asset pm-pick">${p.season} Rd ${p.round}</div>`),
      ];
      return parts.length ? parts.join('') : '<div class="pm-trade-asset" style="color:var(--text-muted);">—</div>';
    };

    const cards = trades.map(tr => {
      const weekLabel = tr.week ? `Week ${tr.week}` : '';
      const dateLabel = tr.date || '';
      const headRight = weekLabel && dateLabel ? `${weekLabel} · ${dateLabel}` : weekLabel || dateLabel;

      return `<div class="pm-trade-card">
        <div class="pm-trade-head">
          <span class="pm-trade-date">${headRight}</span>
        </div>
        <div class="pm-trade-body">
          <div class="pm-trade-col">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#10b981;margin-bottom:4px;">Received</div>
            ${renderAssets(tr.my_gets, tr.my_pick_gets)}
          </div>
          <div style="color:var(--text-muted);font-size:18px;align-self:center;">⇄</div>
          <div class="pm-trade-col">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#ef4444;margin-bottom:4px;">Sent</div>
            ${renderAssets(tr.my_sends, tr.my_pick_sends)}
          </div>
        </div>
      </div>`;
    }).join('');

    panel.innerHTML = `<div style="padding:4px 0;">${cards}</div>`;
  } catch (err) {
    panel.innerHTML = `<div class="team-modal-error"><div>Failed to load trades</div><div style="font-size:13px;color:#9ca3af">${err.message}</div></div>`;
  }
}

async function checkTradeOutcome(btn) {
  const card = btn.closest('.trade-card');
  if (!card) return;
  const resultEl = card.querySelector('.trade-outcome-result');
  if (!resultEl) return;

  if (resultEl.style.display === 'block') {
    resultEl.style.display = 'none';
    btn.textContent = 'Check Outcome';
    return;
  }

  btn.textContent = 'Loading…';
  btn.disabled = true;

  try {
    const teamsData = JSON.parse(btn.dataset.tradeTeams || '[]');
    const tradeDate = btn.dataset.tradeDate || '';

    // Use first team's perspective: what they received = assets_received, what they sent = assets_sent
    const firstTeam = teamsData[0] || {};
    const assets_received = firstTeam.gets || [];
    const assets_sent = firstTeam.sends || [];
    
    // Validate that we have assets to analyze
    if (!assets_received.length && !assets_sent.length) {
      throw new Error('No trade assets found to analyze');
    }
    
    const payload = {
      assets_received,
      assets_sent,
      trade_date: tradeDate,
    };

    console.log('[trade-outcome] Sending payload:', payload);

    const res = await fetch('/api/trade-outcome', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      resultEl.innerHTML = `<p class="outcome-error">Could not load outcome data.</p>`;
      return;
    }
    const data = await res.json();

    if (!data.success) {
      resultEl.innerHTML = `<p class="outcome-error">Could not load outcome data.</p>`;
    } else {
      const verdictCls = data.verdict === 'WIN' ? 'outcome-win' : data.verdict === 'LOSS' ? 'outcome-loss' : 'outcome-even';
      const sign = data.net_delta_now >= 0 ? '+' : '';

      function buildThenRow(r, tag, tagCls) {
        if (r.is_pick || r.value_then == null) return '';
        return `<div class="outcome-row"><span class="outcome-name">${r.name}</span><span class="outcome-tag ${tagCls}">${tag}</span><span class="outcome-val">${r.value_then.toFixed(0)}</span></div>`;
      }

      function buildNowRow(r, tag, tagCls) {
        const deltaStr = r.delta != null ? (r.delta >= 0 ? ` (+${r.delta.toFixed(0)})` : ` (${r.delta.toFixed(0)})`) : '';
        const cls = r.delta == null ? 'outcome-neutral' : (r.delta >= 0 ? 'outcome-plus' : 'outcome-minus');
        return `<div class="outcome-row"><span class="outcome-name">${r.name}</span><span class="outcome-tag ${tagCls}">${tag}</span><span class="outcome-val ${cls}">${(r.value_now ?? 0).toFixed(0)}${deltaStr}</span></div>`;
      }

      let thenRows = '';
      (data.received || []).forEach(r => { thenRows += buildThenRow(r, 'GOT', 'outcome-got'); });
      (data.sent    || []).forEach(r => { thenRows += buildThenRow(r, 'GAVE', 'outcome-gave'); });

      let nowRows = '';
      (data.received || []).forEach(r => { nowRows += buildNowRow(r, 'GOT', 'outcome-got'); });
      (data.sent    || []).forEach(r => { nowRows += buildNowRow(r, 'GAVE', 'outcome-gave'); });

      const thenSection = thenRows ? `<div class="outcome-section-label">At Trade Date</div><div class="outcome-rows">${thenRows}</div>` : '';
      resultEl.innerHTML = `
        <div class="trade-outcome-wrap">
          <div class="outcome-header">
            <span class="outcome-verdict ${verdictCls}">${data.verdict}</span>
            <span class="outcome-delta">${firstTeam.team_name || 'Team 1'}: ${sign}${data.net_delta_now.toFixed(0)} value since trade</span>
          </div>
          ${thenSection}
          <div class="outcome-section-label outcome-section-label--current">Current Value</div>
          <div class="outcome-rows">${nowRows}</div>
        </div>`;
    }

    resultEl.style.display = 'block';
    btn.textContent = 'Hide Outcome';
  } catch (e) {
    resultEl.innerHTML = `<p class="outcome-error">Error loading outcome.</p>`;
    resultEl.style.display = 'block';
    btn.textContent = 'Check Outcome';
  } finally {
    btn.disabled = false;
  }
}

async function fetchTeamDetails(rosterId) {
  try {
    // Extract league context from URL path: /<platform>/<season>/<league_id>/<page>
    const pathParts = window.location.pathname.split('/').filter(p => p);
    const platform = pathParts[0] || 'sleeper';
    const season = pathParts[1] || new Date().getFullYear();
    const leagueId = pathParts[2];

    if (!leagueId) {
      throw new Error('League ID not found in URL');
    }

    const _tmLt = (typeof _leagueType !== 'undefined') ? _leagueType : '1qb';
    const response = await fetch(`/api/team-details/${rosterId}?league_id=${leagueId}&platform=${platform}&season=${season}&league_type=${encodeURIComponent(_tmLt)}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderTeamDetails(data);

  } catch (error) {
    console.error('[team-modal] Error fetching team details:', error);
    const _errPanel = document.getElementById('tm-panel-roster');
    if (_errPanel) _errPanel.innerHTML = `
      <div class="team-modal-error">
        <div>Failed to load team details</div>
        <div style="color: #9ca3af; font-size: 14px;">${error.message}</div>
      </div>
    `;
  }
}

// Global player indicators for team modal
let globalPlayerIndicators = { rookies: [], breakouts: [], elites: [], prospects: [] };

// Load player indicators globally
async function loadGlobalPlayerIndicators() {
  try {
    const res = await fetch('/api/player-indicators?league_type=all&league_size=12', { cache: "no-store" });
    if (!res.ok) return;
    globalPlayerIndicators = await res.json();
  } catch (err) {
    console.error("[global] Failed to load player indicators:", err);
  }
}

// Global indicator helpers (used by openPlayerModal and any non-trade-calc context)
function isBreakout(playerId) {
  return globalPlayerIndicators.breakouts && globalPlayerIndicators.breakouts.includes(String(playerId));
}
function isElite(playerId) {
  return globalPlayerIndicators.elites && globalPlayerIndicators.elites.includes(String(playerId));
}
function isProspect(playerId) {
  return globalPlayerIndicators.prospects && globalPlayerIndicators.prospects.includes(String(playerId));
}
function isRookie(playerId) {
  return globalPlayerIndicators.rookies && globalPlayerIndicators.rookies.includes(String(playerId));
}

// =============================================================================
// BREAKOUT MODAL
// =============================================================================

function openBreakoutModal(playerId, playerName) {
  // Remove any existing breakout modal
  const existing = document.getElementById('bkModalOverlay');
  if (existing) existing.remove();

  const displayName = playerName || 'Breakout Profile';

  const overlay = document.createElement('div');
  overlay.id = 'bkModalOverlay';
  overlay.className = 'player-modal-overlay';
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeBkModal();
  });

  const modal = document.createElement('div');
  modal.className = 'player-modal';
  modal.id = 'bkModal';
  modal.innerHTML = `
    <div class="player-modal-header">
      <div class="player-modal-title-section">
        <div class="player-modal-headshot-container">
          <img class="player-modal-headshot" id="bkModalHeadshot" src="" alt="${displayName}" />
        </div>
        <div class="player-modal-title-text">
          <h2 class="player-modal-name" id="bkModalName">${displayName}</h2>
          <div class="player-modal-meta" id="bkModalMeta">
            <div class="loading-spinner" style="width:16px;height:16px;"></div>
          </div>
        </div>
      </div>
      <button class="player-modal-close" onclick="closeBkModal()">×</button>
    </div>
    <div class="player-modal-body" id="bkModalBody">
      <div class="player-modal-loading">
        <div class="loading-spinner"></div>
        <div>Loading breakout data...</div>
      </div>
    </div>
  `;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);
  document.body.style.overflow = 'hidden';

  // Extract season from URL
  const pathParts = window.location.pathname.split('/').filter(p => p);
  const season = pathParts[1] || new Date().getFullYear();

  fetch(`/api/breakout/player/${playerId}?season=${season}`)
    .then(res => {
      if (!res.ok) {
        console.error('Breakout API request failed:', res.status, res.statusText);
        document.getElementById('bkModalBody').innerHTML = `
          <div class="player-modal-loading">
            <div style="color:#ef4444;">Failed to load breakout data (${res.status})</div>
            <div style="font-size:13px;color:var(--text-muted);">Please try again later.</div>
          </div>
        `;
        return;
      }
      return res.json();
    })
    .then(data => {
      if (!data) return;
      if (data.error) {
        document.getElementById('bkModalBody').innerHTML = `
          <div class="player-modal-loading">
            <div style="color:#ef4444;font-weight:500;">No breakout data available</div>
            <div style="font-size:13px;color:var(--text-muted);">This player doesn't have a breakout analysis for the current season.</div>
          </div>
        `;
        return;
      }
      _renderBkModalContent(data, playerId);
    })
    .catch(() => {
      document.getElementById('bkModalBody').innerHTML = `
        <div class="player-modal-loading">
          <div style="color:#ef4444;">Failed to load breakout data</div>
        </div>
      `;
    });
}

function closeBkModal() {
  const overlay = document.getElementById('bkModalOverlay');
  if (overlay) overlay.remove();
  document.body.style.overflow = '';
}

function _renderBkModalContent(data, playerId) {
  const name  = data.player_name || 'Unknown';
  const team  = data.team || '';
  const pos   = data.position || '';
  const score = parseFloat(data.breakout_opportunity_score || 0);
  const scoreStr = score.toFixed(1);

  const breakoutType = data.breakout_type || {};
  const iconHtml  = breakoutType.icon_html || '<i class="fa-solid fa-chart-simple" aria-hidden="true"></i>';
  const label  = breakoutType.profile_label || 'Breakout Candidate';
  const driver = breakoutType.primary_driver || 'balanced';
  const formattedPhase = data.phase
    ? data.phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : '-';

  // Header: name + meta with dots
  const nameEl = document.getElementById('bkModalName');
  if (nameEl) nameEl.textContent = name;
  const metaParts = [];
  if (pos)  metaParts.push(`<span style="font-weight:600;color:var(--text);">${pos}</span>`);
  if (team) metaParts.push(`<span>${team}</span>`);
  if (formattedPhase !== '-') metaParts.push(`<span>${formattedPhase}</span>`);
  document.getElementById('bkModalMeta').innerHTML = metaParts.join('<span style="opacity:.35;margin:0 4px;">·</span>');

  // Update headshot
  const headshotEl = document.getElementById("bkModalHeadshot");
  if (headshotEl && data.espnHeadshot) {
    headshotEl.src = data.espnHeadshot;
  }
  let scoreColor = '#10b981';
  if (score < 50) scoreColor = '#3b82f6';
  if (score < 40) scoreColor = '#f59e0b';
  if (score < 30) scoreColor = '#6b7280';

  // Key reasons
  const reasons = (data.key_reasons || '').split('\n')
    .filter(r => r.trim() && r.startsWith('•'))
    .map(r => r.substring(1).trim());

  const txnSummary    = data.vacated_usage_summary || '';
  const addedCompSumm = data.added_competition_summary || '';

  // ── Hero row ──────────────────────────────────────────────────────────────
  let html = `
    <div class="pm-hero-row">
      <div class="pm-hero-stat" style="background:${scoreColor}1a;border-color:${scoreColor}33;">
        <div class="pm-hero-label" style="color:${scoreColor};">Breakout Score</div>
        <div class="pm-hero-val" style="color:${scoreColor};">${scoreStr}</div>
      </div>
      <div class="pm-hero-stat" style="text-align:center;padding-left:16px;">
        <div class="pm-hero-label">Profile</div>
        <div style="font-size:13px;font-weight:700;line-height:1,3;color:var(--text);margin:4px 0;">${breakoutType.profile_label}</div>
      </div>
      <div class="pm-hero-stat">
        <div class="pm-hero-label">Phase</div>
        <div style="font-size:13px;font-weight:700;color:var(--text);line-height:1.3;margin:4px 0;">${formattedPhase}</div>
      </div>
    </div>
  `;

  // ── Component breakdown with bars ─────────────────────────────────────────
  html += `<div class='pm-two-column'>`;
  html += `<div class='pm-left-column'>`;
  html += `<hr class="pm-section-divider">`;
  html += `<div class="pm-section-header"><span class="pm-section-label">Component Breakdown</span></div>`;

  const components = [
    { label: 'Opportunity',     val: data.opportunity_opened_score,  color: '#10b981' },
    { label: 'Competition',     val: data.competition_removed_score, color: '#3b82f6' },
    { label: 'Team Env.',       val: data.team_environment_score,    color: null      },
    { label: 'Readiness',       val: data.player_readiness_score,    color: '#8b5cf6' },
    { label: 'Role Trajectory', val: data.role_trajectory_score,     color: null      },
    { label: 'Confidence',      val: data.confidence_score,          color: '#6b7280', suffix: '%' },
  ];

  html += '<div class="pm-comp-list-bo">';
  components.forEach(c => {
    const v    = parseFloat(c.val || 0);
    const fill = Math.min(100, Math.max(0, v));
    const color = c.color || (v >= 60 ? '#10b981' : v >= 35 ? '#3b82f6' : '#f59e0b');
    const disp = c.suffix ? v.toFixed(0) + c.suffix : v.toFixed(1);
    html += `
      <div class="pm-comp-row">
        <span class="pm-comp-label">${c.label}</span>
        <div class="pm-comp-bar-wrap"><div class="pm-comp-bar" style="width:${fill.toFixed(1)}%;background:${color};"></div></div>
        <span class="pm-comp-val" style="color:${color};">${disp}</span>
      </div>`;
  });
  html += '</div>';
  html += '</div>';


  // ── Key factors ───────────────────────────────────────────────────────────
  if (reasons.length) {
    html += `
      <div class='pm-right-column'>
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Key Factors</span></div>
      <div style="display:flex;flex-direction:column;gap:6px;">
    `;
    reasons.forEach(r => {
      html += `<div style="font-size:13px;color:var(--text-muted);display:flex;gap:15px;align-items:flex-start;">
        <span style="color:${scoreColor};font-weight:700;flex-shrink:0;">•</span><span>${r}</span>
      </div>`;
    });
    html += '</div>';
    html += '</div>';
    html += '</div>';
  }

  // ── Context boxes ─────────────────────────────────────────────────────────
  if (txnSummary && txnSummary !== "No departures") {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Vacated Opportunity</span></div>
      <div class="pm-context-box">${txnSummary}</div>
    `;
  }
  if (addedCompSumm && addedCompSumm !== "No new competition added") {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Added Competition</span></div>
      <div class="pm-context-box competition">${addedCompSumm}</div>
    `;
  }
  // ── Footer CTA ────────────────────────────────────────────────────────────
  html += `
    <div class="pm-footer">
      <button id="bkViewProfileBtn" class="pm-profile-btn">View Full Player Profile</button>
    </div>
  `;

  document.getElementById('bkModalBody').innerHTML = html;

  const profileBtn = document.getElementById('bkViewProfileBtn');
  if (profileBtn) {
    profileBtn.addEventListener('click', () => {
      closeBkModal();
      openPlayerModal(playerId, name);
    });
  }
}

function renderTeamDetails(data) {
  // Update avatar
  const avatarHTML = data.avatar 
    ? `<img src="${data.avatar}" alt="${data.team_name || data.username}" class="team-modal-avatar-img" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
       <div class="team-modal-avatar-placeholder" style="display: none;">
         <span>${(data.team_name || data.username || 'Team').charAt(0).toUpperCase()}</span>
       </div>`
    : `<div class="team-modal-avatar-placeholder">
         <span>${(data.team_name || data.username || 'Team').charAt(0).toUpperCase()}</span>
       </div>`;
  document.getElementById('teamModalAvatar').innerHTML = avatarHTML;

  // Update header
  const metaHTML = `
    <div class="team-modal-stat-row">
      <span class="team-modal-stat-label">Record:</span>
      <span class="team-modal-stat-value">${data.record}</span>
    </div>
    <div class="team-modal-stat-row">
      <span class="team-modal-stat-label">Manager:</span>
      <span class="team-modal-stat-value">@${data.username || 'Unknown'}</span>
    </div>
    <div class="team-modal-stat-row">
      <span class="team-modal-stat-label">Total Value:</span>
      <span class="team-modal-stat-value">${data.total_value}</span>
    </div>
  `;
  document.getElementById('teamModalMeta').innerHTML = metaHTML;

  // Build roster table
  let rosterHTML = '<div class="team-modal-section"><h3>Roster</h3>';

  if (data.roster && data.roster.length > 0) {
    rosterHTML += '<table class="team-roster-table">';
    rosterHTML += `
      <thead>
        <tr>
          <th>Player</th>
          <th>Pos</th>
          <th>Team</th>
          <th>Age</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
    `;

    data.roster.forEach(player => {
      const isUnknown = !player.name || player.name === 'Unknown' || /^\d+$/.test(player.name);
      // Determine badges with position-aware thresholds
      let badges = '';
      const value = player.value || 0;
      const pos = player.position;

      // Position-specific elite thresholds (players who would make any team better)
      const eliteThresholds = {
        'RB': 650,   // Elite young backs
        'WR': 650,   // Elite WRs
        'TE': 500,   // Premium TE scarcity
        'QB': 360,   // Solid QB1s
        'K': 9999,   // No elite kickers
        'DEF': 9999  // No elite defenses
      };

      const threshold = eliteThresholds[pos] || 750;
      const isElite = value >= threshold;
      const isRookie = player.years_exp != null && player.years_exp === 0;
      const isBreakoutPlayer = !isElite && isBreakout(player.player_id);

      if (isElite) {
        badges += '<span class="player-badge player-badge-elite"><i class="fa-solid fa-star" aria-hidden="true"></i> ELITE</span>';
      }
      if (isRookie) {
        badges += '<span class="player-badge player-badge-rookie"><i class="fa-solid fa-registered-solid" aria-hidden="true"></i> ROOKIE</span>';
      }
      if (isBreakoutPlayer) {
        badges += '<span class="player-badge player-badge-breakout"><i class="fa-solid fa-fire" aria-hidden="true"></i> BREAKOUT</span>';
      }

      rosterHTML += `
        <tr ${isUnknown ? '' : `style="cursor:pointer;" data-player-id="${player.player_id}" data-player-name="${player.name}"`}>
          <td>
            ${isUnknown
              ? `<span style="color:var(--text-muted);">${/^\d+$/.test(player.name) ? `Unknown ${player.position || ''}`.trim() : (player.name || 'Unknown')}</span>`
              : `<strong class="player-clickable">${player.name}</strong>${badges}`}
          </td>
          <td><span class="pos-badge ${player.position}">${player.position}</span></td>
          <td>${player.team || '-'}</td>
          <td>${player.age != null && !isNaN(parseFloat(player.age)) ? parseFloat(player.age).toFixed(1) : '-'}</td>
          <td>${player.value != null && !isNaN(parseFloat(player.value)) ? parseFloat(player.value).toFixed(1) : '-'}</td>
        </tr>
      `;
    });

    rosterHTML += '</tbody></table>';
  } else {
    rosterHTML += '<div class="team-modal-empty">No players on roster</div>';
  }

  rosterHTML += '</div>';

  // Build picks section
  let picksHTML = '<div class="team-modal-section"><h3>Draft Picks</h3>';

  if (data.picks && data.picks.length > 0) {
    picksHTML += '<div class="team-picks-list">';

    data.picks.forEach(pick => {
      const viaText = pick.via ? ` <span class="pick-via">via ${pick.via}</span>` : '';
      picksHTML += `
        <div class="team-pick-item">
          <span class="pick-label">${pick.year} Round ${pick.round}</span>
          ${viaText}
        </div>
      `;
    });

    picksHTML += '</div>';
  } else {
    picksHTML += '<div class="team-modal-empty">No future picks</div>';
  }

  picksHTML += '</div>';

  // Build graphs section — each chart in its own section for side-by-side layout
  let graphsHTML = '';

  if (data.graphs && (data.graphs.weekly_scores || data.graphs.radar)) {
    if (data.graphs.weekly_scores && data.graphs.weekly_scores.length > 0) {
      graphsHTML += '<div class="team-modal-section tm-chart-weekly"><h3>Weekly Scoring</h3><div class="team-chart-container" id="teamWeeklyChart"></div></div>';
    }
    if (data.graphs.radar && data.graphs.radar.z_scores) {
      graphsHTML += '<div class="team-modal-section tm-chart-radar"><h3>Team Breakdown</h3><div class="team-chart-container" id="teamRadarChart"></div></div>';
    }
  }

  // Populate tab panels
  const rosterPanel = document.getElementById('tm-panel-roster');
  if (rosterPanel) {
    rosterPanel.innerHTML = `<div class="team-modal-body-left">${rosterHTML}</div><div class="team-modal-body-right">${picksHTML}</div>`;
  }
  const chartsPanel = document.getElementById('tm-panel-charts');
  if (chartsPanel) {
    chartsPanel.innerHTML = graphsHTML || '<div class="team-modal-empty">No chart data available</div>';
  }

  // Helper function to get theme-appropriate Plotly styling
  function getPlotlyTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
      template: isDark ? 'plotly_dark' : 'plotly_white',
      textColor: isDark ? '#ffffff' : '#000000',
      gridColor: isDark ? '#374151' : '#e5e7eb',
      lineColor: isDark ? '#9ca3af' : '#6b7280'
    };
  }

  // Render charts using Plotly (if data exists)
  if (data.graphs && typeof Plotly !== 'undefined') {
    // Render weekly scores chart
    if (data.graphs.weekly_scores && data.graphs.weekly_scores.length > 0) {
      const weeks = data.graphs.weekly_scores.map(d => d.week);
      const points = data.graphs.weekly_scores.map(d => d.points);

      const theme = getPlotlyTheme();
      const traces = [{
        x: weeks,
        y: points,
        type: 'scatter',
        mode: 'lines+markers',
        name: data.team_name,
        line: { color: '#667eea', width: 3 },
        marker: { size: 8 },
        hovertemplate: `<b>${data.team_name}</b><br>Week: %{x}<br>Points: %{y:.1f}<extra></extra>`
      }];

      // Add league average if available
      if (data.graphs.league_avg_scores && data.graphs.league_avg_scores.length > 0) {
        const avgWeeks = data.graphs.league_avg_scores.map(d => d.week);
        const avgPoints = data.graphs.league_avg_scores.map(d => d.points);
        traces.push({
          x: avgWeeks,
          y: avgPoints,
          type: 'scatter',
          mode: 'lines',
          name: 'League Avg',
          line: { dash: 'dash', color: '#9ca3af', width: 2 },
          opacity: 0.7,
          hovertemplate: `<b>League Average</b><br>Week: %{x}<br>Points: %{y:.1f}<extra></extra>`
        });
      }

      const weeklyLayout = {
        template: theme.template,
        xaxis: { 
          title: 'Week', 
          standoff: 12,
          color: theme.textColor,
          gridcolor: theme.gridColor
        },
        yaxis: { 
          title: 'Points',
          color: theme.textColor,
          gridcolor: theme.gridColor
        },
        hovermode: 'x unified',
        hoverlabel: {
          bgcolor: theme.template === 'plotly_dark' ? '#1f2937' : '#ffffff',
          bordercolor: theme.template === 'plotly_dark' ? '#374151' : '#e5e7eb',
          font: { color: theme.textColor }
        },
        margin: { l: 50, r: 20, t: 20, b: 50 },
        showlegend: true,
        legend: { 
          x: 0, 
          y: 1.1, 
          orientation: 'h',
          font: { color: theme.textColor }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
      };

      Plotly.newPlot('teamWeeklyChart', traces, weeklyLayout, { responsive: true });
    }

    // Render radar chart
    if (data.graphs.radar && data.graphs.radar.z_scores) {
      const metrics = data.graphs.radar.metrics;
      const zScores = data.graphs.radar.z_scores;
      const rawStats = data.graphs.radar.raw_stats;

      // Close the ring
      const closedMetrics = [...metrics, metrics[0]];
      const closedZScores = [...zScores, zScores[0]];

      // Create hover text with raw stats
      const hoverText = metrics.map((metric, i) =>
        `${metric}: ${rawStats[metric]} (z: ${zScores[i] != null && !isNaN(zScores[i]) ? zScores[i].toFixed(2) : 'N/A'})`
      );
      hoverText.push(hoverText[0]); // Close the ring for hover text too

      const radarTrace = {
        type: 'scatterpolar',
        r: closedZScores,
        theta: closedMetrics,
        fill: 'toself',
        fillcolor: 'rgba(102, 126, 234, 0.3)',
        line: { color: '#667eea', width: 2 },
        marker: { size: 6, color: '#667eea' },
        name: data.team_name,
        text: hoverText,
        hoverinfo: 'text'
      };

      const theme = getPlotlyTheme();
      const radarLayout = {
        template: theme.template,
        polar: {
          radialaxis: {
            visible: true,
            range: [-3, 3],
            tickvals: [-3, -2, -1, 0, 1, 2, 3],
            tickcolor: theme.textColor,
            gridcolor: theme.gridColor,
            linecolor: theme.lineColor
          },
          angularaxis: {
            tickcolor: theme.textColor,
            gridcolor: theme.gridColor,
            linecolor: theme.lineColor
          },
          bgcolor: 'rgba(0,0,0,0)'
        },
        margin: { l: 60, r: 60, t: 40, b: 40 },
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
          color: theme.textColor
        }
      };

      Plotly.newPlot('teamRadarChart', [radarTrace], radarLayout, { responsive: true });
    }
  }
}

// Team click handler (event delegation)
document.addEventListener('click', (e) => {
  const teamCard = e.target.closest('.team-clickable');
  if (teamCard) {
    const rosterId = teamCard.dataset.rosterId;
    const teamName = teamCard.dataset.teamName;
    if (rosterId) {
      openTeamModal(rosterId, teamName);
    }
  }
});

// ── New Subscriber Welcome Tour ───────────────────────────────────────────────
(function () {
  'use strict';
  if (new URLSearchParams(window.location.search).get('new_subscriber') !== '1') return;
  if (localStorage.getItem('br_sub_tour_done')) return;
  localStorage.setItem('br_sub_tour_done', '1');

  // Clean the URL so a refresh doesn't re-trigger it
  const cleanUrl = window.location.pathname + window.location.search.replace(/[?&]new_subscriber=1/, '').replace(/^\?$/, '');
  history.replaceState(null, '', cleanUrl);

  // Build league-aware links from path
  const parts = window.location.pathname.split('/').filter(Boolean);
  const hasLeague = parts.length >= 3;
  const base = hasLeague ? '/' + parts.slice(0, 3).join('/') : '';

  const features = [
    { icon: '📊', label: 'Trade Intelligence', desc: 'Market values, momentum, and real trade frequency', href: base + '/trade-intel' },
    { icon: '🔥', label: 'Breakout Engine', desc: 'Opportunity projections and breakout candidate rankings', href: base + '/breakouts' },
    { icon: '🤖', label: 'AI Insights', desc: 'Front office briefings and trade analysis personalized to your roster', href: base + '/dashboard' },
    { icon: '📈', label: 'Roster Grades', desc: 'Letter grades, archetypes, and portfolio value trends', href: base + '/teams' },
  ];

  const overlay = document.createElement('div');
  overlay.id = 'subWelcomeTour';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.65);z-index:10002;display:flex;align-items:center;justify-content:center;padding:16px;';
  overlay.innerHTML = `
    <div style="background:var(--card);border:1px solid var(--border);border-radius:20px;padding:36px 32px 28px;max-width:480px;width:100%;box-shadow:0 24px 64px rgba(0,0,0,0.35);position:relative;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:40px;margin-bottom:12px;">🎉</div>
        <h2 style="margin:0 0 8px;font-size:22px;font-weight:800;color:var(--text);">Welcome to Premium!</h2>
        <p style="margin:0;font-size:14px;color:var(--text-muted);line-height:1.5;">Here's what you've unlocked. Explore at your own pace or jump in now.</p>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:24px;">
        ${features.map(f => `
          <a href="${f.href}" style="display:flex;align-items:center;gap:14px;padding:12px 14px;border-radius:12px;border:1px solid var(--border);background:var(--bg-alt);text-decoration:none;transition:background 0.12s;" onmouseover="this.style.background='var(--accent-soft)'" onmouseout="this.style.background='var(--bg-alt)'">
            <span style="font-size:22px;width:32px;text-align:center;flex-shrink:0;">${f.icon}</span>
            <div>
              <div style="font-size:13px;font-weight:700;color:var(--text);">${f.label}</div>
              <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">${f.desc}</div>
            </div>
            <span style="margin-left:auto;color:var(--text-muted);font-size:14px;">→</span>
          </a>`).join('')}
      </div>
      <button onclick="document.getElementById('subWelcomeTour').remove()" style="width:100%;padding:12px;border-radius:10px;border:none;background:var(--accent);color:#fff;font-size:14px;font-weight:700;cursor:pointer;transition:opacity .12s;" onmouseover="this.style.opacity='.88'" onmouseout="this.style.opacity='1'">
        Let's go →
      </button>
    </div>`;
  document.body.appendChild(overlay);
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
})();

// ── New User Tour ──────────────────────────────────────────────────────────
(function () {
  'use strict';

  // Determine league context from URL
  const pathParts = window.location.pathname.split('/').filter(Boolean);
  const hasLeague = pathParts.length >= 3 && !isNaN(pathParts[1]);
  if (!hasLeague) return;

  const platform  = pathParts[0];
  const season    = pathParts[1];
  const leagueId  = pathParts[2];
  const tourKey   = 'tour_done_' + leagueId;

  // Current page from the page-shell data attribute
  const pageShell = document.querySelector('.page-shell');
  const currentPage = pageShell ? (pageShell.dataset.page || '') : '';

  function buildLeagueUrl(page) {
    // Extract league context from current URL
    const pathParts = window.location.pathname.split('/').filter(Boolean);

    if (pathParts.length >= 3 && !isNaN(pathParts[1])) {
      const platform = pathParts[0];
      const season = pathParts[1];
      const leagueId = pathParts[2];
      const url = '/' + platform + '/' + season + '/' + leagueId + '/' + page;
      return url;
    }
    // Fallback to just page name if no league context
    const fallbackUrl = '/' + page;
    return fallbackUrl;
  }

  // ── Tour steps ──────────────────────────────────────────────────────────
  const TOUR_STEPS = [
    {
      page: 'dashboard', selector: null,
      title: 'Welcome to your dynasty hub!',
      body: "Let's take a quick tour of the key features. Hit Next to get started.",
    },
    {
      page: 'dashboard', selector: '#settingsGearBtn',
      title: 'Settings & Leagues',
      body: 'Manage leagues, refresh your data, toggle dark mode, and view the changelog here.',
    },
    {
      page: 'dashboard', selector: '#settingsDropdown',
      title: 'Settings Menu',
      body: 'Switch between leagues, force a data refresh, or flip to dark mode - all in one place.',
      beforeShow: function () {
        var dd = document.getElementById('settingsDropdown');
        if (dd) dd.style.display = 'block';
      },
      afterLeave: function () {
        var dd = document.getElementById('settingsDropdown');
        if (dd) dd.style.display = '';
      },
    },
    {
      page: 'dashboard', selector: '#settingsChangelogBtn',
      title: "What's New",
      body: 'The bell lights up when new features or data drop. Never miss an update.',
      beforeShow: function () {
        var dd = document.getElementById('settingsDropdown');
        if (dd) dd.style.display = 'block';
      },
      afterLeave: function () {
        var dd = document.getElementById('settingsDropdown');
        if (dd) dd.style.display = '';
      },
    },
    {
      page: 'dashboard', selector: '.nav-pills-container',
      title: 'Navigation',
      body: 'Jump to Trade Calc, Teams, Activity, Players, Breakouts, Prospects, and more from here.',
    },
    {
      page: 'dashboard', selector: '#navSearchWrapper',
      title: 'Player Search',
      body: 'Search any player from the nav bar — click the magnifying glass or press Ctrl+K, then click a result to open their full profile.',
    },
    {
      page: 'dashboard', selector: '.player-row, .otc-player-row',
      title: 'Player Cards',
      body: 'Click any player to see their dynasty value history, trend, news, and breakout score.',
      action: 'openPlayerModal',
    },
    {
      page: 'dashboard', selector: '.team-clickable, .team-row',
      title: 'Team Cards',
      body: 'Click any team to pull up their full roster, positional grades, and asset values.',
      action: 'openTeamModal',
    },
    {
      page: 'dashboard', selector: 'a[href*="/weekly"]',
      title: 'Weekly Hub',
      body: 'Live scores, matchup previews, and weekly projections during the season.',
      onlyIfPresent: false,
    },
    {
      page: 'history', selector: '.card',
      title: 'Season History',
      body: 'Week-by-week recaps, season standings, power rankings, and scoring trends for every past year.',
      navigate: 'history',
    },
    {
      page: 'awards', selector: '.card',
      title: 'All-Time Awards',
      body: 'Career records, championship history, and fun awards like "most bench points left on the table."',
      navigate: 'awards',
    },
    {
      page: 'graphs', selector: '.card',
      title: 'League Analytics',
      body: 'PF vs PA scatter plots, weekly scoring lines, and head-to-head radar charts for every team.',
      navigate: 'graphs',
    },
    {
      page: 'dashboard', selector: null,
      title: "You're all set!",
      body: "Explore your dynasty empire. Check out Trade Calc, Breakouts, and Prospects when you're ready.",
    },
  ];

  // ── State ────────────────────────────────────────────────────────────────
  let currentStep  = 0;
  let tourActive   = false;
  const overlays   = [];
  let tooltipEl    = null;

  // ── Init ─────────────────────────────────────────────────────────────────
  function initTour() {
    const params     = new URLSearchParams(window.location.search);
    const resumeAt   = params.get('tour');

    if (resumeAt !== null) {
      // Came here via a tour navigation - clean the URL then resume
      history.replaceState(null, '', window.location.pathname);
      const idx = parseInt(resumeAt, 10);
      if (!isNaN(idx)) {
        startTour(idx);
        return;
      }
    }

    // Only auto-start on the dashboard for new users with league context
    if (currentPage !== 'dashboard') return;
    if (localStorage.getItem(tourKey)) return;
    
    // Check if we have league context (not on demo/home page)
    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const hasLeague = pathParts.length >= 3 && !isNaN(pathParts[1]);
    if (!hasLeague) return;

    setTimeout(function () { startTour(0); }, 800);
  }

  // ── DOM creation ─────────────────────────────────────────────────────────
  function startTour(fromStep) {
    tourActive   = true;
    currentStep  = fromStep;
    createOverlayDOM();
    showStep(currentStep);
  }

  function createOverlayDOM() {
    ['top', 'bottom', 'left', 'right'].forEach(function (side) {
      const el = document.createElement('div');
      el.className = 'tour-overlay-piece';
      el.dataset.side = side;
      document.body.appendChild(el);
      overlays.push(el);
    });

    tooltipEl = document.createElement('div');
    tooltipEl.className = 'tour-tooltip';
    document.body.appendChild(tooltipEl);
  }

  // ── Step rendering ───────────────────────────────────────────────────────
  function showStep(idx) {
    if (idx < 0 || idx >= TOUR_STEPS.length) { endTour(); return; }
    const step = TOUR_STEPS[idx];

    // Wrong page? Navigate there
    if (step.page && step.page !== currentPage) {
      window.location.href = buildLeagueUrl(step.page) + '?tour=' + idx;
      return;
    }

    // Skip if element required but missing
    if (step.onlyIfPresent) {
      const el = step.selector ? document.querySelector(step.selector) : null;
      if (!el) { advanceTour(); return; }
    }

    // Run beforeShow hook (e.g. open settings dropdown)
    if (typeof step.beforeShow === 'function') step.beforeShow();

    const target = step.selector ? document.querySelector(step.selector) : null;
    positionOverlays(target);
    renderTooltip(step, idx, target);

    // Auto-open modals as a demo, then reposition the spotlight onto the modal
    if (step.action === 'openPlayerModal') {
      var playerEl = document.querySelector('[data-player-id]');
      if (playerEl) {
        setTimeout(function () {
          var pid   = playerEl.dataset.playerId;
          var pname = playerEl.dataset.playerName || playerEl.textContent.trim();
          if (typeof openPlayerModal === 'function') openPlayerModal(pid, pname);

          var attempts = 0;
          var maxAttempts = 80;
          function checkAndPosition() {
            attempts++;
            var modal = document.getElementById('playerModal');
            if (modal) {
              var body = document.getElementById('playerModalBody');
              var isVisible = modal.style.display !== 'none' && modal.offsetParent !== null;
              var hasAdvancedMetrics = body && body.querySelector('#advancedMetricsSection');

              if (isVisible && hasAdvancedMetrics) {
                positionOverlays(modal);
                placeTooltip(modal);
                return;
              }

              if (attempts < maxAttempts) {
                setTimeout(checkAndPosition, 100);
              } else {
                positionOverlays(modal);
                placeTooltip(modal);
              }
            }
          }
          setTimeout(checkAndPosition, 400);
        }, 200);
      }
    } else if (step.action === 'openTeamModal') {
      var teamEl = document.querySelector('.team-clickable[data-roster-id]');
      if (teamEl) {
        setTimeout(function () {
          var rid   = teamEl.dataset.rosterId;
          var tname = teamEl.dataset.teamName || '';
          if (typeof openTeamModal === 'function') openTeamModal(rid, tname);

          var attempts = 0;
          var maxAttempts = 80;
          function checkAndPosition() {
            attempts++;
            var modal = document.getElementById('teamModal');
            if (modal) {
              var body = document.getElementById('teamModalBody');
              var isVisible = modal.style.display !== 'none' && modal.offsetParent !== null;
              var hasWeeklyChart = body && body.querySelector('#teamWeeklyChart');

              if (isVisible && hasWeeklyChart) {
                positionOverlays(modal);
                placeTooltip(modal);
                return;
              }

              if (attempts < maxAttempts) {
                setTimeout(checkAndPosition, 100);
              } else {
                // Fallback: position on modal even if weekly chart didn't load
                positionOverlays(modal); 
                placeTooltip(modal);
              }
            }
          }
          setTimeout(checkAndPosition, 400);
        }, 200);
      }
    }
  }

  function positionOverlays(target) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const PAD = 8;

    if (!target) {
      // Full-screen dim - all coverage via top piece, rest zeroed
      overlays.forEach(function (p) {
        if (p.dataset.side === 'top') {
          Object.assign(p.style, { top: '0', left: '0', width: vw + 'px', height: vh + 'px' });
        } else {
          Object.assign(p.style, { top: '0', left: '0', width: '0', height: '0' });
        }
      });
      return;
    }

    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const r  = target.getBoundingClientRect();
    const x1 = Math.max(0, r.left   - PAD);
    const y1 = Math.max(0, r.top    - PAD);
    const x2 = Math.min(vw, r.right  + PAD);
    const y2 = Math.min(vh, r.bottom + PAD);

    overlays.forEach(function (p) {
      switch (p.dataset.side) {
        case 'top':    Object.assign(p.style, { top: '0',        left: '0',       width: vw + 'px',       height: y1 + 'px'       }); break;
        case 'bottom': Object.assign(p.style, { top: y2 + 'px',  left: '0',       width: vw + 'px',       height: (vh - y2) + 'px' }); break;
        case 'left':   Object.assign(p.style, { top: y1 + 'px',  left: '0',       width: x1 + 'px',       height: (y2 - y1) + 'px' }); break;
        case 'right':  Object.assign(p.style, { top: y1 + 'px',  left: x2 + 'px', width: (vw - x2) + 'px', height: (y2 - y1) + 'px' }); break;
      }
    });
  }

  function renderTooltip(step, idx, target) {
    const isLast  = idx === TOUR_STEPS.length - 1;
    const isFirst = idx === 0;
    const count   = (idx + 1) + ' / ' + TOUR_STEPS.length;

    tooltipEl.innerHTML =
      '<div class="tour-tooltip-header">' +
        '<span class="tour-step-count">' + count + '</span>' +
        '<button class="tour-skip-btn" type="button">Skip tour</button>' +
      '</div>' +
      '<div class="tour-tooltip-title">' + step.title + '</div>' +
      '<div class="tour-tooltip-body">'  + step.body  + '</div>' +
      '<div class="tour-tooltip-footer">' +
        (isFirst ? '<span></span>' : '<button class="tour-btn tour-btn-secondary" data-action="prev">Back</button>') +
        '<button class="tour-btn tour-btn-primary" data-action="next">' + (isLast ? 'Finish' : 'Next →') + '</button>' +
      '</div>';

    // Position
    if (!target) {
      Object.assign(tooltipEl.style, { top: '50%', left: '50%', transform: 'translate(-50%,-50%)', display: 'block' });
    } else {
      placeTooltip(target);
      tooltipEl.style.display = 'block';
    }

    tooltipEl.querySelector('[data-action="next"]').addEventListener('click', advanceTour);
    const prevBtn = tooltipEl.querySelector('[data-action="prev"]');
    if (prevBtn) prevBtn.addEventListener('click', function () { currentStep--; showStep(currentStep); });
    tooltipEl.querySelector('.tour-skip-btn').addEventListener('click', endTour);
  }

  function placeTooltip(target) {
    const TH   = 200;
    const PAD  = 12;
    const vw   = window.innerWidth;
    const vh   = window.innerHeight;
    const r    = target.getBoundingClientRect();

    let top = r.bottom + PAD;
    if (top + TH > vh) top = Math.max(PAD, r.top - PAD - TH);
    if (top < PAD) top = PAD;

    // On mobile stretch full-width; on desktop use fixed 300px centered on target
    if (vw <= 540) {
      Object.assign(tooltipEl.style, {
        top: top + 'px',
        left: PAD + 'px',
        right: PAD + 'px',
        width: 'auto',
        transform: '',
      });
    } else {
      const TW = 300;
      let left = r.left + r.width / 2 - TW / 2;
      left = Math.max(PAD, Math.min(left, vw - TW - PAD));
      Object.assign(tooltipEl.style, {
        top: top + 'px',
        left: left + 'px',
        right: '',
        width: '',
        transform: '',
      });
    }
  }

  // ── Navigation ───────────────────────────────────────────────────────────
  function leaveCurrentStep() {
    // Call afterLeave hook (e.g. close settings dropdown)
    const step = TOUR_STEPS[currentStep];
    if (step && typeof step.afterLeave === 'function') step.afterLeave();

    // Close any open player/team modals
    if (typeof closePlayerModal === 'function' && document.getElementById('playerModal')) {
      closePlayerModal();
    }
    if (typeof closeTeamModal === 'function' && document.getElementById('teamModal')) {
      closeTeamModal();
    }
  }

  function advanceTour() {
    leaveCurrentStep();
    const nextIdx  = currentStep + 1;
    if (nextIdx >= TOUR_STEPS.length) { endTour(); return; }
    const nextStep = TOUR_STEPS[nextIdx];

    if (nextStep.navigate && nextStep.page !== currentPage) {
      window.location.href = buildLeagueUrl(nextStep.page) + '?tour=' + nextIdx;
      return;
    }
    currentStep = nextIdx;
    showStep(currentStep);
  }

  function endTour() {
    leaveCurrentStep();
    localStorage.setItem(tourKey, '1');
    removeTourDOM();
    tourActive = false;
  }

  function removeTourDOM() {
    overlays.forEach(function (p) { p.remove(); });
    overlays.length = 0;
    if (tooltipEl) { tooltipEl.remove(); tooltipEl = null; }
  }

  // Reposition on resize
  window.addEventListener('resize', function () {
    if (!tourActive) return;
    const step   = TOUR_STEPS[currentStep];
    const target = step && step.selector ? document.querySelector(step.selector) : null;
    positionOverlays(target);
    if (tooltipEl && target) placeTooltip(target);
  });

  document.addEventListener('DOMContentLoaded', initTour);
}());

// ── Pick value modal ──────────────────────────────────────────────────────────
function showPickModal(el) {
  var raw = el.getAttribute('data-pick');
  if (!raw) return;
  var pick;
  try { pick = JSON.parse(raw.replace(/&quot;/g, '"')); } catch(e) { return; }

  var label  = pick.label  || 'Draft Pick';
  var value  = pick.value  || 0;
  var tiers  = pick.tiers  || {};
  var rnd    = pick.round  || 0;
  var season = pick.season || '';

  var suffix = rnd === 1 ? '1st' : rnd === 2 ? '2nd' : rnd === 3 ? '3rd' : rnd + 'th';
  var roundName = season + ' ' + suffix + ' Round';

  var maxTier = Math.max(tiers.early || 0, tiers.mid || 0, tiers.late || 0, 1);
  function bar(tierVal, tierName) {
    if (!tierVal) return '';
    var pct = Math.round(tierVal / maxTier * 100);
    var cls = tierName === 'early' ? 'analytics-bar-neg' : tierName === 'late' ? 'analytics-bar-pos' : 'analytics-bar-mid';
    return '<div class="analytics-bar-row" style="margin:4px 0">' +
      '<span class="analytics-bar-name" style="width:48px;text-transform:capitalize">' + tierName + '</span>' +
      '<div class="analytics-bar-track"><div class="analytics-bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
      '<span class="analytics-bar-val">' + tierVal.toFixed(1) + '</span>' +
    '</div>';
  }

  var tiersHtml = bar(tiers.early, 'early') + bar(tiers.mid, 'mid') + bar(tiers.late, 'late');
  var currentVal = value > 0 ? '<div style="margin:8px 0;font-size:14px;color:var(--text-muted)">Current value: <strong>' + value.toFixed(1) + '</strong></div>' : '';

  var html = '<div id="pickModalOverlay" style="position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:9000;display:flex;align-items:center;justify-content:center" onclick="if(event.target===this)closePickModal()">' +
    '<div style="background:var(--card-bg,#1e293b);border-radius:12px;padding:24px;max-width:380px;width:92%;box-shadow:0 8px 32px rgba(0,0,0,.4)">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">' +
        '<h3 style="margin:0;font-size:16px">' + label + '</h3>' +
        '<button onclick="closePickModal()" style="background:none;border:none;cursor:pointer;font-size:20px;color:var(--text-muted)">&times;</button>' +
      '</div>' +
      currentVal +
      '<div style="margin-top:12px">' +
        '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">' + roundName + ' - Value by tier</div>' +
        tiersHtml +
      '</div>' +
      (tiersHtml ? '' : '<p style="color:var(--text-muted);font-size:13px">No value data available for this pick.</p>') +
    '</div>' +
  '</div>';

  document.body.insertAdjacentHTML('beforeend', html);
  document.body.style.overflow = 'hidden';
}

function closePickModal() {
  var overlay = document.getElementById('pickModalOverlay');
  if (overlay) overlay.remove();
  document.body.style.overflow = '';
}

function toggleAnalyticsDraftTeam(header) {
  const teamElement = header.parentElement;
  teamElement.classList.toggle('collapsed');
}

function setupFunAwardsGrid() {
  const funAwardsGrid = document.querySelector('.fun-awards-grid');
  if (!funAwardsGrid) return;
  
  const items = funAwardsGrid.querySelectorAll('.fun-award-item');
  const itemCount = items.length;

  if (itemCount === 0) return;
  
  // Check if mobile (screen width <= 768px)
  const isMobile = window.innerWidth <= 768;
  
  if (isMobile) {
    // Single column on mobile
    funAwardsGrid.style.gridTemplateColumns = '1fr';
  } else {
    // Calculate optimal columns: ceil(items / 2) for exactly 2 rows on desktop
    const columns = Math.ceil(itemCount / 2);
    funAwardsGrid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
  }
}

// Mobile sidebar: wrap sidebar content in a collapsible toggle drawer.
// Handles both .page-sidebar (league pages, ≤1180px) and .otc-side (trade calc, ≤1200px).
// Both default to CLOSED on mobile.
(function initMobileSidebar() {
  var configs = [
    { selector: '.page-sidebar', breakpoint: 1180, toggleClass: 'page-sidebar-toggle', bodyClass: 'page-sidebar-body', label: 'League Analytics' },
    { selector: '.otc-side',     breakpoint: 1200, toggleClass: 'otc-side-toggle',     bodyClass: 'otc-side-body',     label: 'Player Insights' },
  ];

  function setupSidebar(sidebar, cfg) {
    if (sidebar.dataset.mobileToggleInit) return;
    sidebar.dataset.mobileToggleInit = '1';

    var label = sidebar.dataset.sidebarLabel || cfg.label;
    var toggle = document.createElement('button');
    toggle.className = cfg.toggleClass;
    toggle.innerHTML = '<span>' + label + '</span><span class="sidebar-toggle-icon">▼</span>';

    var body = document.createElement('div');
    body.className = cfg.bodyClass;
    while (sidebar.firstChild) {
      body.appendChild(sidebar.firstChild);
    }

    sidebar.appendChild(toggle);
    sidebar.appendChild(body);
    // Default: closed

    toggle.addEventListener('click', function() {
      var open = body.classList.toggle('open');
      toggle.classList.toggle('open', open);
    });
  }

  function setup() {
    configs.forEach(function(cfg) {
      if (window.innerWidth > cfg.breakpoint) return;
      document.querySelectorAll(cfg.selector).forEach(function(el) {
        setupSidebar(el, cfg);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
  window.addEventListener('resize', setup);
})();

// ── Nav-wide player search ────────────────────────────────────────────────────
(function initNavSearch() {
  function setup() {
  const wrapper  = document.getElementById('navSearchWrapper');
  const input    = document.getElementById('navPlayerSearch');
  const dropdown = document.getElementById('navSearchDropdown');
  const clearBtn = document.getElementById('navSearchClear');
  if (!wrapper || !input || !dropdown) return;

  const POS_COLORS = { QB: 'qb', RB: 'rb', WR: 'wr', TE: 'te', K: 'k', DEF: 'def' };
  let _players = null;
  let _loading = false;
  let _debounce = null;
  let _focusIdx = -1;

  async function loadPlayers() {
    if (_players !== null || _loading) return;
    _loading = true;
    try {
      const res = await fetch('/api/league-players', { cache: 'default' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const raw = Array.isArray(data) ? data : (Array.isArray(data.players) ? data.players : []);
      _players = raw
        .filter(p => p && p.id && p.name && p.position !== 'PICK' && !String(p.id).startsWith('pick_'))
        .map(p => ({
          id:   String(p.id),
          name: String(p.name || ''),
          pos:  String(p.position || '').toUpperCase(),
          team: String(p.team || p.nfl_team || ''),
        }));
    } catch (e) {
      console.warn('[nav-search] Failed to load players:', e);
      _players = [];
    }
    _loading = false;
  }

  function headshot(id) {
    return `https://sleepercdn.com/content/nfl/players/thumb/${id}.jpg`;
  }

  function posColor(pos) {
    return POS_COLORS[pos] || 'def';
  }

  function renderResults(query) {
    const q = query.trim().toLowerCase();
    if (!q) { closeDropdown(); return; }
    if (!_players) { dropdown.innerHTML = '<div class="nav-search-empty">Loading…</div>'; openDropdown(); return; }

    const words = q.split(/\s+/);
    const matches = _players
      .filter(p => {
        const n = p.name.toLowerCase();
        return words.every(w => n.includes(w));
      })
      .slice(0, 8);

    if (!matches.length) {
      dropdown.innerHTML = '<div class="nav-search-empty">No players found</div>';
      openDropdown();
      return;
    }

    dropdown.innerHTML = matches.map((p, i) => `
      <div class="nav-search-result" data-idx="${i}" data-ns-pid="${p.id}" data-ns-name="${p.name.replace(/"/g, '&quot;')}">
        <img class="nav-search-avatar" src="${headshot(p.id)}" alt="" loading="lazy"
             onerror="this.style.visibility='hidden'" />
        <div class="nav-search-info">
          <div class="nav-search-name">${p.name}</div>
          <div class="nav-search-meta">${p.team || '—'}</div>
        </div>
        <span class="nav-search-pos nav-search-pos-${posColor(p.pos)}">${p.pos}</span>
      </div>
    `).join('');

    _focusIdx = -1;
    openDropdown();
  }

  function openDropdown() { dropdown.classList.add('open'); }
  function closeDropdown() { dropdown.classList.remove('open'); _focusIdx = -1; }

  function setFocus(idx) {
    const items = dropdown.querySelectorAll('.nav-search-result');
    items.forEach(el => el.classList.remove('focused'));
    if (idx >= 0 && idx < items.length) {
      items[idx].classList.add('focused');
      items[idx].scrollIntoView({ block: 'nearest' });
    }
    _focusIdx = idx;
  }

  function selectCurrent() {
    const items = dropdown.querySelectorAll('.nav-search-result');
    const el = _focusIdx >= 0 ? items[_focusIdx] : items[0];
    if (!el) return;
    openPlayerModal(el.dataset.nsPid, el.dataset.nsName);
    input.value = '';
    clearBtn.style.display = 'none';
    closeDropdown();
  }

  // Click anywhere on collapsed wrapper → expand and focus input
  wrapper.querySelector('.nav-search-inner').addEventListener('click', () => {
    if (document.activeElement !== input) input.focus();
  });

  input.addEventListener('focus', () => { loadPlayers(); if (input.value.trim()) openDropdown(); });

  input.addEventListener('input', () => {
    const val = input.value;
    clearBtn.style.display = val ? 'block' : 'none';
    clearTimeout(_debounce);
    _debounce = setTimeout(() => renderResults(val), 120);
  });

  input.addEventListener('keydown', e => {
    const items = dropdown.querySelectorAll('.nav-search-result');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setFocus(Math.min(_focusIdx + 1, items.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocus(Math.max(_focusIdx - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      selectCurrent();
    } else if (e.key === 'Escape') {
      closeDropdown();
      input.blur();
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    clearBtn.style.display = 'none';
    closeDropdown();
    input.focus();
  });

  dropdown.addEventListener('click', e => {
    const row = e.target.closest('.nav-search-result');
    if (!row) return;
    e.stopPropagation();
    openPlayerModal(row.dataset.nsPid, row.dataset.nsName);
    input.value = '';
    clearBtn.style.display = 'none';
    closeDropdown();
  });

  document.addEventListener('click', e => {
    if (!wrapper.contains(e.target)) closeDropdown();
  });

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();

// ── Rookie Draft Assistant ────────────────────────────────────────────────────
(function () {
  let daProspects  = [];
  let daDrafted    = new Set(); // insertion order = overall draft pick order
  let myPicks      = new Set(); // subset of daDrafted that are the user's own picks
  let myPickOrder  = [];        // ordered subset of myPicks (for grading)
  let daLocalNeeds = {};        // position -> need level delta from my picks
  let daFilter    = 'ALL';
  let daSubView   = 'available'; // 'available' | 'drafted'
  let daNeeds      = {};
  let daLeagueType = '1qb';
  let daLeagueSize = 10;
  let daYear       = new Date().getFullYear();
  let daInitialized = false;

  const POS_COLORS = { QB: '#a78bfa', RB: '#34d399', WR: '#60a5fa', TE: '#fb923c' };
  const NEED_LABEL = { 2: 'Major Need', 1: 'Need', 0: 'Neutral', '-1': 'Depth', '-2': 'Stacked' };
  const NEED_COLOR = { 2: '#ef4444', 1: '#f59e0b', 0: '#9ca3af', '-1': '#10b981', '-2': '#059669' };
  const NEED_BONUS = { 2: 1.5, 1: 1.2, 0: 1.0, '-1': 0.85, '-2': 0.7 };

  function effectiveNeed(pos) {
    const raw   = daNeeds[pos] ?? 0;
    const delta = daLocalNeeds[pos] || 0;
    let need    = Math.max(-2, Math.min(2, raw + delta));
    // In 1QB leagues cap QB need at Neutral if roster already has 2+ QBs (including my picks)
    if (pos === 'QB' && daLeagueType !== 'sf') {
      const myQBs = myPickOrder.filter(id => {
        const p = daProspects.find(x => String(x.player_id) === id);
        return p && p.position === 'QB';
      }).length;
      if ((daNeeds.QB_count || 0) + myQBs >= 2) need = Math.min(-1, need);
    }
    return need;
  }

  function needBonus(pos) {
    return NEED_BONUS[String(effectiveNeed(pos))] ?? 1.0;
  }

  function adjustNeedsForDraft(playerId, delta) {
    const p = daProspects.find(x => String(x.player_id) === String(playerId));
    if (!p || !p.position) return;
    const pos = p.position.toUpperCase();
    daLocalNeeds[pos] = (daLocalNeeds[pos] || 0) + delta;
    renderNeeds();
  }

  function daScore(p) {
    const val = parseFloat(p.display_value || p.rookie_value || 0);
    return val * 0.6 + val * needBonus(p.position) * 0.4;
  }

  // 1 rec normally; 2 only if there's a major need AND the top pick doesn't address a need
  function recCount(scored) {
    const hasMajorNeed = Object.values(daNeeds).some(v => typeof v === 'number' && v === 2);
    if (!hasMajorNeed) return 1;
    const topNeed = scored[0] ? (daNeeds[scored[0].position] ?? 0) : 0;
    return topNeed >= 1 ? 1 : 2;
  }

  function daToggleNeeds() {
    const panel = document.getElementById('daNeedsPanel');
    if (!panel) return;
    const collapsed = panel.classList.toggle('da-needs-collapsed');
    const chevron = panel.querySelector('.da-needs-chevron');
    if (chevron) chevron.style.transform = collapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
  }

  function renderNeeds() {
    const panel = document.getElementById('daNeedsPanel');
    if (!panel) return;
    const collapsed = panel.classList.contains('da-needs-collapsed');
    const chevron = `<span class="da-needs-chevron" style="margin-left:auto;font-size:12px;transition:transform 0.2s;${collapsed?'transform:rotate(-90deg)':''}">&#8964;</span>`;
    const titleHtml = `<div class="da-needs-title" onclick="window._da.toggleNeeds()">My Roster Needs${chevron}</div>`;
    if (!Object.keys(daNeeds).length) {
      panel.innerHTML = titleHtml + '<div class="da-needs-body"><div style="font-size:12px;color:var(--text-muted);padding-top:8px;">Log in with your league to see personalized needs.</div></div>';
      return;
    }
    const rows = ['QB','RB','WR','TE'].map(pos => {
      const need  = effectiveNeed(pos);
      const col   = POS_COLORS[pos] || '#9ca3af';
      const count = daNeeds[`${pos}_count`] ?? 0;
      const val   = Math.round(daNeeds[`${pos}_value`] || 0);
      const avg   = Math.round(daNeeds[`${pos}_avg`]   || 0);
      return `<div class="da-need-row">
        <span class="pos-badge ${pos}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 7px;">${pos}</span>
        <div class="da-need-info">
          <span class="da-need-label" style="color:${NEED_COLOR[String(need)] || '#9ca3af'}">${NEED_LABEL[String(need)] ?? 'Neutral'}</span>
          <span class="da-need-meta">${count} players · ${val} (avg ${avg})</span>
        </div>
      </div>`;
    }).join('');
    panel.innerHTML = `${titleHtml}<div class="da-needs-body">${rows}</div>`;
  }

  function updateDraftedBadge() {
    const el = document.getElementById('daDraftedCount');
    if (!el) return;
    if (daDrafted.size === 0) { el.style.display = 'none'; }
    else { el.style.display = ''; el.textContent = daDrafted.size; }
  }

  function render() {
    const listEl = document.getElementById('daBoardList');
    if (!listEl) return;
    updateDraftedBadge();

    // Tag .da-board with current view so CSS can use different grid per view
    const boardEl = listEl.closest('.da-board');
    if (boardEl) boardEl.dataset.view = daSubView;

    if (daSubView === 'drafted') {
      // Sort by insertion order in daDrafted (first pick = index 0 = top)
      const draftedArr = [...daDrafted];
      const drafted = draftedArr
        .map(sid => daProspects.find(p => String(p.player_id) === sid))
        .filter(Boolean);
      if (!drafted.length) {
        listEl.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">No players drafted yet.</div>';
        return;
      }
      const endBtn = myPicks.size > 0
        ? `<div style="padding:12px 10px 4px;"><button class="da-end-draft-btn" onclick="window._da.endDraft()">End Draft &amp; Grade My Picks</button></div>`
        : '';
      listEl.innerHTML = endBtn + drafted.map((p, i) => {
        const sid   = String(p.player_id);
        const isMine = myPicks.has(sid);
        const col   = POS_COLORS[p.position] || '#9ca3af';
        const dAdp  = daLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
        const dTeam = p.actual_nfl_team || p.school || '';
        const dMeta = [dTeam, dAdp != null ? `ADP ${parseFloat(dAdp).toFixed(1)}` : ''].filter(Boolean).join(' · ');
        const overallPick = draftedArr.indexOf(sid) + 1;
        return `<div class="da-row${isMine ? ' da-my-pick' : ''}">
          <div class="da-rank"><span style="color:${isMine ? 'var(--accent)' : 'var(--text-muted)'};font-weight:${isMine ? '800' : '400'};">${overallPick}</span></div>
          <div class="da-info"><span class="da-name">${p.name || '—'}</span><span class="da-meta">${dMeta}</span></div>
          <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 6px;">${p.position}</span>
          <label class="da-mine-label" title="My pick">
            <input type="checkbox" class="da-mine-cb" ${isMine ? 'checked' : ''} onchange="window._da.toggleMine('${p.player_id}')">
            <span>Mine</span>
          </label>
          <div class="da-col-right da-val">${Math.round(parseFloat(p.display_value||0))||'—'}</div>
          <button class="otc-chip-remove" onclick="window._da.undraft('${p.player_id}')" title="Remove">×</button>
        </div>`;
      }).join('');
      return;
    }

    // Available view
    let visible = daProspects.filter(p => !daDrafted.has(String(p.player_id)));
    if (daFilter !== 'ALL') visible = visible.filter(p => p.position === daFilter);
    const scored = visible.map(p => ({ ...p, _s: daScore(p) })).sort((a, b) => b._s - a._s);
    const nRec   = recCount(scored);
    const recIds = new Set(scored.slice(0, nRec).map(p => String(p.player_id)));

    if (!scored.length) {
      listEl.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">No prospects available.</div>';
      return;
    }

    listEl.innerHTML = scored.map((p, i) => {
      const isRec    = recIds.has(String(p.player_id));
      const col      = POS_COLORS[p.position] || '#9ca3af';
      const val      = Math.round(parseFloat(p.display_value || 0));
      const needLvl  = daNeeds[p.position] ?? 0;
      const isNeed   = needLvl >= 1;
      const needCol  = NEED_COLOR[String(needLvl)] || '#9ca3af';
      const needTxt  = NEED_LABEL[String(needLvl)] || '';

      // Recommendation row: add grade + ADP in meta
      const adpRaw   = daLeagueType === 'sf' ? p.sf_avg_pick : p.avg_pick;
      const adpTxt   = adpRaw != null ? `ADP ${parseFloat(adpRaw).toFixed(1)}` : '';
      const gradeTxt = p.tier_label || '';
      const teamTxt  = p.actual_nfl_team || p.school || '';
      const baseMeta = [teamTxt, adpTxt].filter(Boolean).join(' · ');
      const recMeta  = isRec
        ? [teamTxt, gradeTxt, adpTxt].filter(Boolean).join(' · ')
        : baseMeta;

      // Need badge goes in the badge column (col 4) — same slot as PICK for rec rows
      const needBadge = isNeed && !isRec
        ? `<span style="font-size:10px;font-weight:700;color:${needCol};background:${needCol}18;border:1px solid ${needCol}33;border-radius:4px;padding:2px 6px;">${needTxt}</span>`
        : '';

      return `<div class="da-row${isRec ? ' da-recommended' : ''}">
        <div class="da-rank">${i + 1}</div>
        <div class="da-info">
          <span class="da-name">${p.name || '—'}${isRec && isNeed ? `<span style="font-size:10px;font-weight:700;color:${needCol};margin-left:6px;">▲ ${needTxt}</span>` : ''}</span>
          <span class="da-meta">${recMeta}</span>
        </div>
        <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 6px;">${p.position}</span>
        ${isRec ? '<div class="da-rec-badge">PICK</div>' : (needBadge || '<div></div>')}
        <div class="da-col-right da-val">${val || '—'}</div>
        <button class="da-draft-btn" onclick="window._da.draft('${p.player_id}')">Draft</button>
      </div>`;
    }).join('');
  }

  function saveSession() {
    const key = 'da_' + location.pathname;
    try {
      sessionStorage.setItem(key, JSON.stringify([...daDrafted]));
      sessionStorage.setItem(key + '_mine', JSON.stringify(myPickOrder));
    } catch (_) {}
  }

  function showDraftHelp() {
    const steps = [
      { icon: '1', title: 'Draft players in order', body: 'As each pick happens — yours or anyone else\'s — tap <strong>Draft</strong> to remove them from the board. Do this in real draft order so pick numbers are accurate.' },
      { icon: '2', title: 'Mark your picks', body: 'Switch to the <strong>Drafted</strong> tab and tap <strong>Mine</strong> on each player you actually selected. The pick number is set automatically based on when you drafted them.' },
      { icon: '3', title: 'Watch your needs update', body: 'The <strong>Roster Needs</strong> panel reflects your current roster vs. the league. Marking a pick as Mine adjusts the needs panel live.' },
      { icon: '4', title: 'End Draft &amp; grade', body: 'Once you\'ve marked your picks, tap <strong>End Draft &amp; Grade My Picks</strong>. Each pick is graded A+–F using ADP value, positional need, and QB context — the same formula as the Teams page Draft Grades.' },
    ];
    const html = `
      <div style="padding:20px 20px 0;display:flex;align-items:center;justify-content:space-between;">
        <div style="font-size:16px;font-weight:700;color:var(--text);">How to use the Draft Assistant</div>
        <button onclick="document.getElementById('daHelpModal').style.display='none'" style="background:none;border:none;font-size:20px;color:var(--text-muted);cursor:pointer;">✕</button>
      </div>
      <div style="padding:16px 20px 20px;display:flex;flex-direction:column;gap:16px;">
        ${steps.map(s => `
          <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;background:var(--accent);color:#fff;font-size:13px;font-weight:800;display:flex;align-items:center;justify-content:center;">${s.icon}</div>
            <div>
              <div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:3px;">${s.title}</div>
              <div style="font-size:12px;color:var(--text-muted);line-height:1.5;">${s.body}</div>
            </div>
          </div>`).join('')}
      </div>`;

    let modal = document.getElementById('daHelpModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'daHelpModal';
      modal.style.cssText = 'display:none;position:fixed;inset:0;z-index:10600;align-items:center;justify-content:center;padding:20px;background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);';
      modal.innerHTML = '<div id="daHelpModalContent" style="background:var(--card);border-radius:16px;max-width:420px;width:100%;box-shadow:0 24px 48px rgba(15,23,42,0.25);"></div>';
      modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });
      document.body.appendChild(modal);
    }
    document.getElementById('daHelpModalContent').innerHTML = html;
    modal.style.display = 'flex';
  }

  // Exact port of pick_grade() and team_grade() from app.py (including BPA logic)
  function _pickGrade(adpDiff, need, pos, isSF, qbCount, numTeams, isBpa, bpaGap) {
    if (adpDiff === null) return 'N/A';
    const bigReach = -(numTeams * 1.1);
    let score;
    if      (adpDiff >= 4)          score = 4;
    else if (adpDiff >= 2)          score = 3;
    else if (adpDiff >= -3)         score = 2;
    else if (adpDiff >= bigReach)   score = 1;
    else                            score = 0;

    // BPA bonus / penalty (mirrors Python logic)
    if (isBpa) {
      score += adpDiff < -3 ? 1 : 2;
    } else if (bpaGap != null && bpaGap >= 5) {
      score = Math.max(score - 1, 0);
    }

    if (need) {
      score += 1;
    } else {
      if (pos === 'QB' && !isSF && qbCount >= 2) score = Math.max(score - 2, 0);
      else if (pos === 'QB' && !isSF && qbCount >= 1) score = Math.max(score - 1, 0);
    }
    if (adpDiff >= -3)            score = Math.max(score, 1);
    if (need && adpDiff >= -4)    score = Math.max(score, 2);
    return ({5:'A+',4:'A',3:'B',2:'C',1:'D',0:'F'})[Math.min(score, 5)] || 'F';
  }

  function _teamGrade(grades) {
    if (!grades.length) return 'N/A';
    const v = {'A+':5,'A':4,'B':3,'C':2,'D':1,'F':0,'N/A':2};
    const avg = grades.reduce((s, g) => s + (v[g] ?? 2), 0) / grades.length;
    if (avg >= 4.5) return 'A+';
    if (avg >= 3.5) return 'A';
    if (avg >= 2.5) return 'B';
    if (avg >= 1.5) return 'C';
    if (avg >= 0.5) return 'D';
    return 'F';
  }

  function showDraftGrade() {
    const GRADE_COLOR = { 'A+': '#10b981', 'A': '#10b981', 'B': '#3b82f6', 'C': '#f59e0b', 'D': '#ef4444', 'F': '#ef4444', 'N/A': '#9ca3af' };
    const GRADE_BG    = { 'A+': 'rgba(16,185,129,.08)', 'A': 'rgba(16,185,129,.08)', 'B': 'rgba(59,130,246,.08)', 'C': 'rgba(245,158,11,.08)', 'D': 'rgba(239,68,68,.08)', 'F': 'rgba(239,68,68,.08)', 'N/A': 'transparent' };
    const isSF = daLeagueType === 'sf';

    const draftedArr = [...daDrafted]; // preserves insertion order = actual pick sequence

    // Build a lookup of adp for BPA computation
    const adpKey = p => parseFloat(isSF ? p.sf_avg_pick : p.avg_pick) || 9999;

    const picks = myPickOrder.map((sid, idx) => {
      const p = daProspects.find(x => String(x.player_id) === sid);
      if (!p) return null;
      const actualPick = draftedArr.indexOf(sid) + 1; // overall pick # in draft order
      const adp = parseFloat(isSF ? p.sf_avg_pick : p.avg_pick) || null;
      const adpDiff = adp !== null ? actualPick - adp : null;
      const need = (daNeeds[p.position] ?? 0) >= 1;
      const qbsBefore = myPickOrder.slice(0, idx).filter(id => {
        const q = daProspects.find(x => String(x.player_id) === id);
        return q && q.position === 'QB';
      }).length;
      const qbCount = (daNeeds.QB_count || 0) + qbsBefore;

      // BPA: who was available at this pick with a better ADP?
      const takenBefore = new Set(draftedArr.slice(0, actualPick - 1));
      const available = daProspects.filter(x => !takenBefore.has(String(x.player_id)));
      const bpa = available.reduce((best, x) => adpKey(x) < adpKey(best) ? x : best, available[0]);
      const bpaAdp = bpa ? adpKey(bpa) : null;
      const isBpa = bpa ? String(bpa.player_id) === sid : false;
      const bpaGap = (adp !== null && bpaAdp !== null && !isBpa) ? adp - bpaAdp : 0;

      const grade = _pickGrade(adpDiff, need, p.position, isSF, qbCount, daLeagueSize, isBpa, bpaGap);
      const needLabel = NEED_LABEL[String(daNeeds[p.position] ?? 0)] || 'Neutral';
      const tier = p.tier_label || '';
      return { p, actualPick, adp, adpDiff, grade, need, needLabel, tier, isBpa, bpaGap };
    }).filter(Boolean);

    if (!picks.length) return;

    const overall = _teamGrade(picks.map(x => x.grade));

    const rows = picks.map(({ p, actualPick, adp, adpDiff, grade, needLabel, tier, isBpa }) => {
      const col    = POS_COLORS[p.position] || '#9ca3af';
      const gc     = GRADE_COLOR[grade] || '#9ca3af';
      const gbg    = GRADE_BG[grade] || 'transparent';
      const adpTxt = adp ? `ADP ${adp.toFixed(1)}` : '';
      const pickTxt = `Pick ${actualPick}`;
      const diffTxt = adpDiff !== null
        ? (adpDiff >= 0 ? `+${adpDiff.toFixed(1)} value` : `${adpDiff.toFixed(1)} reach`)
        : '';
      const diffCol = adpDiff !== null ? (adpDiff >= 0 ? '#10b981' : '#ef4444') : 'var(--text-muted)';
      const tierTxt = tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : '';
      const bpaTxt  = isBpa ? '<span style="font-size:10px;font-weight:700;color:#10b981;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.25);border-radius:4px;padding:1px 5px;margin-left:4px;">BPA</span>' : '';
      const meta = [pickTxt, adpTxt, tierTxt].filter(Boolean).join(' · ');
      return `<div style="display:grid;grid-template-columns:1fr 38px 32px;align-items:center;gap:8px;padding:10px 14px 10px 12px;border-top:1px solid var(--border);border-left:3px solid ${gc};">
        <div style="display:flex;flex-direction:column;gap:2px;min-width:0;">
          <div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
            <span style="font-size:13px;font-weight:700;color:var(--text);">${p.name}</span>${bpaTxt}
          </div>
          <span style="font-size:11px;color:var(--text-muted);">${meta}</span>
          ${diffTxt ? `<span style="font-size:11px;font-weight:600;color:${diffCol};">${diffTxt}</span>` : ''}
        </div>
        <span class="pos-badge ${p.position}" style="background:${col}22;color:${col};border:1px solid ${col}44;font-size:10px;padding:2px 5px;text-align:center;">${p.position}</span>
        <div style="font-size:18px;font-weight:800;color:${gc};text-align:right;">${grade}</div>
      </div>`;
    }).join('');

    const gc  = GRADE_COLOR[overall] || '#9ca3af';
    const gbg = GRADE_BG[overall] || 'transparent';
    const html = `
      <div style="padding:16px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);">
        <div>
          <div style="font-size:15px;font-weight:700;color:var(--text);">My Draft Grade</div>
          <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">${picks.length} pick${picks.length !== 1 ? 's' : ''} graded</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:48px;height:48px;border-radius:50%;background:${gbg};border:2px solid ${gc};display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <span style="font-size:20px;font-weight:900;color:${gc};line-height:1;">${overall}</span>
          </div>
          <button onclick="document.getElementById('daGradeModal').style.display='none'" style="background:none;border:none;font-size:18px;color:var(--text-muted);cursor:pointer;padding:6px;line-height:1;flex-shrink:0;">✕</button>
        </div>
      </div>
      <div>${rows}</div>
      <div style="padding:14px 16px;display:flex;gap:8px;">
        <button onclick="document.getElementById('daGradeModal').style.display='none';daReset();" style="flex:1;padding:9px;background:transparent;color:var(--text-muted);border:1px solid var(--border);border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;">Reset Board</button>
        <button onclick="document.getElementById('daGradeModal').style.display='none'" class="da-end-draft-btn" style="flex:2;">Done</button>
      </div>`;

    let modal = document.getElementById('daGradeModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'daGradeModal';
      modal.style.cssText = 'display:none;position:fixed;inset:0;z-index:10600;align-items:center;justify-content:center;padding:20px;background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);';
      modal.innerHTML = '<div id="daGradeModalContent" style="background:var(--card);border-radius:16px;max-width:480px;width:100%;max-height:85vh;overflow-y:auto;box-shadow:0 24px 48px rgba(15,23,42,0.25);"></div>';
      modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });
      document.body.appendChild(modal);
    }
    document.getElementById('daGradeModalContent').innerHTML = html;
    modal.style.display = 'flex';
  }

  window._da = {
    draft(id)      { daDrafted.add(String(id));    saveSession(); render(); },
    undraft(id) {
      const sid = String(id);
      daDrafted.delete(sid);
      if (myPicks.has(sid)) {
        myPicks.delete(sid);
        myPickOrder = myPickOrder.filter(x => x !== sid);
        adjustNeedsForDraft(sid, +1);
      }
      saveSession(); render();
    },
    toggleMine(id) {
      const sid = String(id);
      if (myPicks.has(sid)) {
        myPicks.delete(sid);
        myPickOrder = myPickOrder.filter(x => x !== sid);
        adjustNeedsForDraft(sid, +1);
      } else {
        myPicks.add(sid);
        myPickOrder.push(sid);
        adjustNeedsForDraft(sid, -1);
      }
      saveSession(); render();
    },
    toggleNeeds()  { daToggleNeeds(); },
    endDraft()     { showDraftGrade(); },
    showHelp()     { showDraftHelp(); },
  };

  window.daFilterPos = function (pos) {
    daFilter = pos;
    document.querySelectorAll('.da-filter').forEach(b => b.classList.toggle('active', b.dataset.pos === pos));
    render();
  };

  window.daSubTab = function (sub) {
    daSubView = sub;
    document.querySelectorAll('.da-sub-tab').forEach(b => b.classList.toggle('active', b.dataset.sub === sub));
    render();
  };

  window.daReset = function () {
    daDrafted.clear();
    myPicks.clear();
    myPickOrder = [];
    daLocalNeeds = {};
    daFilter  = 'ALL';
    daSubView = 'available';
    document.querySelectorAll('.da-filter').forEach(b => b.classList.toggle('active', b.dataset.pos === 'ALL'));
    document.querySelectorAll('.da-sub-tab').forEach(b => b.classList.toggle('active', b.dataset.sub === 'available'));
    saveSession();
    render();
  };

  // Page-level tab switcher (Rankings / Draft Board)
  window.rkPageTab = function (tab) {
    document.querySelectorAll('.rk-page-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('rk-panel-rankings').style.display = tab === 'rankings' ? '' : 'none';
    document.getElementById('rk-panel-draft').style.display    = tab === 'draft'    ? '' : 'none';
    if (tab === 'draft' && !daInitialized) {
      daInitialized = true;
      initDA();
    }
  };

  async function initDA() {
    const _sessKey = 'da_' + location.pathname;
    try { daDrafted = new Set(JSON.parse(sessionStorage.getItem(_sessKey) || '[]')); } catch (_) {}
    try { myPickOrder = JSON.parse(sessionStorage.getItem(_sessKey + '_mine') || '[]'); myPicks = new Set(myPickOrder); } catch (_) {}

    // Derive league context from URL: /<platform>/<season>/<league_id>/...
    const parts    = location.pathname.split('/').filter(Boolean);
    const platform = parts[0] || 'sleeper';
    const season   = parts[1] || new Date().getFullYear();
    const leagueId = parts[2];
    daYear         = parseInt(season);

    // Fetch the active draft class year (may differ from NFL season in URL)
    try {
      const acr = await fetch('/api/prospects/active-class');
      if (acr.ok) { const acd = await acr.json(); if (acd.year) daYear = acd.year; }
    } catch (_) {}

    // Fetch league-calibrated prospect rankings settings if in a league
    if (leagueId && !['players','breakouts','prospects','trade-database','trade-intel'].includes(platform)) {
      try {
        // Detect league type / size from rankings context (use rkLeagueType/rkLeagueSize if set by the Rankings tab)
        daLeagueType = (typeof rkLeagueType !== 'undefined' ? rkLeagueType : null)
          || localStorage.getItem('rk_league_type') || '1qb';
        daLeagueSize = parseInt((typeof rkLeagueSize !== 'undefined' ? rkLeagueSize : null)
          || localStorage.getItem('rk_league_size') || '10');

        // Get viewer roster_id if available on this page; backend falls back to session
        const viewerRid = (typeof getCurrentRosterId === 'function' ? getCurrentRosterId() : null)
          || document.querySelector('#viewerRosterIdInput')?.value || '';
        const needsUrl = `/api/draft-needs?league_id=${leagueId}&platform=${platform}&season=${season}`
          + (viewerRid ? `&roster_id=${encodeURIComponent(viewerRid)}` : '');
        const nr = await fetch(needsUrl);
        if (nr.ok) {
          const nd = await nr.json();
          if (nd.error) {
            const np = document.getElementById('daNeedsPanel');
            if (np) np.innerHTML = '<div class="da-needs-title">My Roster Needs<span class="da-needs-chevron" style="margin-left:auto;font-size:12px;">&#8964;</span></div><div class="da-needs-body"><div style="padding:12px 0;font-size:12px;color:var(--text-muted);">Log in with your league to see personalized needs.</div></div>';
          } else {
            daNeeds      = nd.needs || {};
            daLeagueType = nd.league_type || daLeagueType;
            daLeagueSize = nd.league_size || daLeagueSize;
          }
        }
      } catch (_) {}
    }

    renderNeeds();

    const listEl = document.getElementById('daBoardList');
    try {
      const r = await fetch(`/api/prospects/rankings?year=${daYear}&league_type=${encodeURIComponent(daLeagueType)}&league_size=${daLeagueSize}&limit=200`);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      daProspects = data.rankings || [];
      render();
    } catch (e) {
      if (listEl) listEl.innerHTML = `<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:13px;">Could not load prospects: ${e.message}</div>`;
    }
  }
})();
