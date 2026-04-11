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
window.addEventListener('beforeunload', function() {
  window.scrollTo(0, 0);
});

// Prevent any programmatic scrolls during initial page load
let scrollBlocked = true;
window.addEventListener('scroll', function(e) {
  if (scrollBlocked) {
    window.scrollTo(0, 0);
  }
}, { passive: false });

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
  }

  function updateThemeIcons() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const lightIcons = document.querySelectorAll('.theme-icon.light-icon');
    const darkIcons = document.querySelectorAll('.theme-icon.dark-icon');

    lightIcons.forEach(icon => {
      icon.style.display = isDark ? 'none' : 'inline';
    });
    darkIcons.forEach(icon => {
      icon.style.display = isDark ? 'inline' : 'none';
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
      state.sideAPicks = apIds.map(id => ({ id, display: id }));
      state.sideBPicks = bpIds.map(id => ({ id, display: id }));

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

    // Make row clickable
    if (p.id) {
      row.style.cursor = "pointer";
      row.dataset.playerId = p.id;
      row.dataset.playerName = p.name || "Unknown";
    }

    const rankWrap = document.createElement("div");
    rankWrap.className = "otc-value-rank";
    rankWrap.textContent = overallRank ? "#" + overallRank : "—";

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
      metaBits.push('<span class="player-badge player-badge-rookie">ROOKIE</span>');
    }
    if (p.is_rookie) {
      metaBits.push('<span class="player-badge player-badge-rookie">PROSPECT</span>');
    }
    if (isBreakout(p.id)) {
      metaBits.push('<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>');
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
      metaBits.push('<span class="player-badge player-badge-rookie">ROOKIE</span>');
    }
    if (isBreakout(p.id)) {
      metaBits.push('<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>');
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
        if (c.age != null) metaParts.push(`${c.age.toFixed(1)} yrs`);
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
    const moversSub = root.querySelector("#moversSub");
    const dayFilters = root.querySelector(".otc-day-filters");

    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;

        // Update active tab button
        tabButtons.forEach(b => b.classList.remove("is-active"));
        btn.classList.add("is-active");

        // Update content visibility
        if (tab === "movers") {
          moversContent?.classList.add("is-active");
          breakoutsContent?.classList.remove("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            // Restore movers subtitle (will be updated by loadTopMovers)
            const leagueType = getLeagueType();
            const leagueSize = getLeagueSize();
            const leagueLabel = leagueType === "sf" ? "SF" : "1QB";
            const sizeLabel = leagueSize === 10 ? "" : ` ${leagueSize}-team`;
            moversSub.textContent = `Biggest 7-day changes in ${leagueLabel}${sizeLabel} BR value`;
          }
          // Show day filters for movers
          if (dayFilters) {
            dayFilters.style.display = "flex";
          }
        } else if (tab === "breakouts") {
          moversContent?.classList.remove("is-active");
          breakoutsContent?.classList.add("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            moversSub.textContent = "Top 5 breakouts from Breakout Engine ";
          }
          // Hide day filters for breakouts
          if (dayFilters) {
            dayFilters.style.display = "none";
          }

          // Load breakouts when tab is opened for the first time
          if (!breakoutsContent.dataset.loaded) {
            breakoutsContent.dataset.loaded = "true";
            loadBreakouts();
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
      pos_rank_label: p.pos_rank_label || "",
      sf_pos_rank_label: p.sf_pos_rank_label || "",
      is_rookie: p.is_rookie === true,
      search_name: p.search_name || "",
    };
  }

  async function ensurePlayersLoaded() {
    if (allPlayers.length > 0) return;

    const errorBox = root.querySelector("#errorBox");

    try {
      const res = await fetch("/api/league-players", { cache: "no-store" });
      if (!res.ok) {
        throw new Error("Failed to load players (" + res.status + ").");
      }
      const data = await res.json();
      const rawData = Array.isArray(data) ? data : [];

      const players = rawData.filter(p => p.position !== "PICK");
      const picks = rawData.filter(p => p.position === "PICK");

      allPlayers = [
        ...players
          .filter(p => p && typeof p === "object" && p.id != null)
          .map(normalizePlayerRow)
          .filter(p => ["QB", "RB", "WR", "TE"].includes(p.position) || p.is_rookie),
        ...picks,
      ].sort((a, b) => {
        const vb = Number(b.value || 0);
        const va = Number(a.value || 0);
        if (vb !== va) return vb - va;
        return String(a.name || "").localeCompare(String(b.name || ""));
      });
    } catch (err) {
      console.error("Error loading data:", err);
      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = err.message || "Failed to load data.";
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

    items.forEach(p => {
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

  async function onLeagueTypeChange() {
    // Refresh all value displays
    await Promise.all([loadPlayerDeltas(), loadPlayerIndicators()]);
    renderChips("A");
    renderChips("B");
    recomputeTrade();
    renderAllPlayersList();
    loadTopMovers();
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

      const leftWrap = document.createElement("div");
      leftWrap.className = "otc-chip-main";

      const nameEl = document.createElement("div");
      nameEl.className = "otc-chip-name player-clickable";
      nameEl.textContent = p.name || "Unknown";

      // Make player name clickable
      if (p.id) {
        nameEl.dataset.playerId = p.id;
        nameEl.dataset.playerName = p.name || "Unknown";
      }

      const metaEl = document.createElement("div");
      metaEl.className = "otc-chip-meta";

      const metaBits = buildMetaBits(p);

      // Add rookie/breakout badges
      if (isRookie(p.id)) {
        metaBits.push('<span class="player-badge player-badge-rookie">ROOKIE</span>');
      }
      if (isBreakout(p.id)) {
        metaBits.push('<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>');
      }

      metaEl.innerHTML = metaBits.join(" • ");

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
      removeBtn.addEventListener("click", () => {
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
      removeBtn.addEventListener("click", () => {
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

  function openPickPrompt(side) {
    const raw = window.prompt(
      "Enter a pick in this format:\n2026_1_04\nor\n2026_1_early"
    );
    if (!raw) return;

    const cleaned = String(raw).trim();
    if (!cleaned) return;

    const picks = getSidePicks(side);
    if (picks.find(p => p.id === cleaned)) return;

    picks.push({
      id: cleaned,
      display: cleaned.replaceAll("_", " "),
    });

    saveState();
    renderChips(side);
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
      return;
    }

    const payload = {
      league_id: root.querySelector("#leagueIdInput")?.value || "",
      season: root.querySelector("#seasonInput")?.value || "",
      league_type: getLeagueType(),
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
      const leftPct = ((normalizedDiff + 1) / 2) * 100;

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
    } catch (err) {
      console.error("[trade] error in recomputeTrade:", err);
      if (errorBox) {
        errorBox.style.display = "block";
        errorBox.textContent = err.message || "Failed to evaluate trade.";
      }
    }
  }

  // ------------------------------------------------------------
  // analyzeTrade — owns ALL loading/result/empty state transitions
  // Never call tradeAiBody.innerHTML directly from outside this fn
  // ------------------------------------------------------------
  async function analyzeTrade() {
    // Don't run analysis in guest mode
    const isGuest = root.querySelector("#isGuestMode")?.value === "true";
    if (isGuest) {
      console.log("[trade] Skipping analyzeTrade in guest mode");
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

    // Nothing on either side — show empty state and bail
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
        // Write into resultState only — never clobber tradeAiBody directly
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

  // Fuzzy player name matching — returns a score (higher = better match).
  // Handles: exact substring, word-start matches, and single-transposition typos.
  function fuzzyNameScore(name, query) {
    if (!name || !query) return 0;
    const n = name.toLowerCase();
    const q = query.toLowerCase();
    // Exact substring — highest priority
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
              if (!picks.find(x => String(x.id) === String(p.id))) {
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
        // Same refresh as league type change — all values need recalculating
        await Promise.all([loadPlayerDeltas(), loadPlayerIndicators()]);
        renderChips("A");
        renderChips("B");
        recomputeTrade();
        renderAllPlayersList();
        loadTopMovers();
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
  // Button handlers — lean; analyzeTrade() owns all state logic
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

          const currentUsername = getCurrentUsername();
          if (currentUsername) {
            const userTeam = teams.find(
              team =>
                team.username === currentUsername ||
                team.team_name.toLowerCase().includes(currentUsername.toLowerCase())
            );
            if (userTeam) selector.value = userTeam.roster_id;
          }

          const currentRosterId = getCurrentRosterId();
          if (!selector.value && currentRosterId) {
            selector.value = currentRosterId;
          }

          updateAnalyzeButtonState();
        })
        .catch(err => {
          console.error("Failed to load teams:", err);
        });
    }

    bindOnce(selector, "teamSelectorChange", "change", () => {
      const selectedRosterId = selector.value;
      if (selectedRosterId) {
        fetch("/api/set-viewer-roster", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ roster_id: selectedRosterId }),
        }).then(res => {
          if (res.ok) {
            // Team changed — reset to empty state so user re-triggers analysis
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
    let isInfoOpen = false;

    bindOnce(infoBtn, "otcInfoClick", "click", (e) => {
      e.stopPropagation();
      isInfoOpen = !isInfoOpen;
      infoTooltip.style.display = isInfoOpen ? "block" : "none";
    });

    // Close on click outside
    document.addEventListener("click", (e) => {
      if (infoWrapper && !infoWrapper.contains(e.target) && isInfoOpen) {
        isInfoOpen = false;
        infoTooltip.style.display = "none";
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

    // Try to load trade from URL first, otherwise load from localStorage
    const loadedFromURL = loadTradeFromURL();
    if (!loadedFromURL) {
      loadState();
    }

    updateAnalyzeButtonState();
    syncEmptyState("A");
    syncEmptyState("B");
    recomputeTrade();
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
  return urlParams.get("viewer_roster_id") || document.querySelector("#teamSelect")?.value || "";
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

  // ESPN elements - DISABLED
  // const espnLeagueIdInput = document.getElementById("espnLeagueIdInput");
  // const espnTeamName = document.getElementById("espnTeamName");
  // const espnSubmitBtn = document.getElementById("espnSubmitBtn");

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

    // ESPN platform disabled
    // if (platform === "espn") {
    //   if (sleeperFlow) sleeperFlow.style.display = "none";
    //   if (espnFlow) espnFlow.style.display = "block";
    //   if (sleeperHint) sleeperHint.style.display = "none";
    // } else {
      if (sleeperFlow) sleeperFlow.style.display = "block";
      if (espnFlow) espnFlow.style.display = "none";
      if (sleeperHint) sleeperHint.style.display = "";
    // }
  }

  // Platform switching
  platformBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      // Ignore clicks on disabled ESPN button
      if (btn.dataset.platform === "espn" && btn.disabled) {
        return;
      }
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

  // ESPN submit - DISABLED
  /*
  if (espnSubmitBtn) {
    espnSubmitBtn.addEventListener("click", async () => {
      const leagueId = espnLeagueIdInput?.value.trim();
      if (!leagueId || !/^\d+$/.test(leagueId)) {
        if (errorBox) {
          errorBox.textContent = "Enter a valid ESPN League ID (numbers only).";
          errorBox.style.display = "block";
        }
        return;
      }

      if (errorBox) errorBox.style.display = "none";
      espnSubmitBtn.disabled = true;
      espnSubmitBtn.textContent = "Validating...";

      try {
        const res = await fetch(`/api/espn-validate-league?league_id=${encodeURIComponent(leagueId)}`);
        const data = await res.json();

        if (!res.ok || !data.ok) {
          throw new Error(data.error || "Unable to load ESPN league.");
        }

        // Inject league id into the form's league select and submit
        leagueSelect.innerHTML = `<option value="${leagueId}" selected>${data.league?.name || "ESPN League"}</option>`;
        if (formPlatform) formPlatform.value = "espn";

        // Set optional team name as username for viewer matching
        const teamName = espnTeamName?.value.trim() || "";
        const formUsername = document.getElementById("formUsername");
        if (formUsername) formUsername.value = teamName;

        document.getElementById("leagueSelectForm")?.submit();
      } catch (err) {
        if (errorBox) {
          errorBox.textContent = err.message || "Unable to load ESPN league.";
          errorBox.style.display = "block";
        }
        espnSubmitBtn.disabled = false;
        espnSubmitBtn.textContent = "Go to Dashboard";
      }
    });
  }
  */

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
        dot.classList.remove("changelog-dot-hidden");
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

      // Mark as seen
      if (changelogData && changelogData.length > 0) {
        const latestDate = changelogData[0].date;
        localStorage.setItem("changelog_last_seen", latestDate);
        dot.classList.add("changelog-dot-hidden");
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
  }

  function closeDropdown() {
    if (isDropdownOpen) {
      isDropdownOpen = false;
      dropdown.style.display = "none";
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

  // Dark mode toggle
  const settingsDarkModeBtn = document.getElementById("settingsDarkModeBtn");
  if (settingsDarkModeBtn) {
    settingsDarkModeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (window.toggleDarkMode) {
        window.toggleDarkMode();
      }
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
      navToggle.textContent = navPillsContainer.classList.contains('nav-open') ? '✕' : '☰';
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
      if (!navToggle.contains(e.target) && !navPillsContainer.contains(e.target)) {
        navPillsContainer.classList.remove('nav-open');
        navToggle.textContent = '☰';
      }
    });

    // Close menu when clicking a nav link
    navPillsContainer.querySelectorAll('.nav-pill').forEach(pill => {
      pill.addEventListener('click', function() {
        if (pill.closest('.nav-pill-dropdown-wrapper')) return;  // dropdown trigger — keep hamburger open
        navPillsContainer.classList.remove('nav-open');
        navToggle.textContent = '☰';
      });
    });

    // Close hamburger when clicking a Players sub-menu item
    navPillsContainer.querySelectorAll('.nav-pill-dropdown-item').forEach(item => {
      item.addEventListener('click', function() {
        navPillsContainer.classList.remove('nav-open');
        navToggle.textContent = '☰';
        const wrapper = document.getElementById('playersNavDropdown');
        if (wrapper) wrapper.classList.remove('open');
      });
    });
  }
});

// Players nav dropdown toggle
function togglePlayersNav(e) {
  e.stopPropagation();
  const wrapper = document.getElementById('playersNavDropdown');
  if (!wrapper) return;
  wrapper.classList.toggle('open');
}

document.addEventListener('DOMContentLoaded', function() {
  // Close players nav dropdown when clicking outside
  document.addEventListener('click', function(e) {
    const wrapper = document.getElementById('playersNavDropdown');
    if (wrapper && !wrapper.contains(e.target)) {
      wrapper.classList.remove('open');
    }
  });
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
          this.textContent = '▼';
        } else {
          // Collapse
          targetBody.classList.add('collapsed');
          this.classList.add('collapsed');
          this.textContent = '▶';
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

    if (platform && season && leagueId && historySeason) {
      // Load awards section
      const awardsContent = document.getElementById('historyAwardsContent');
      if (awardsContent) {
        fetch(`/api/history/${platform}/${season}/${leagueId}/summary?history_season=${historySeason}`, { cache: 'no-store' })
          .then(res => res.json())
          .then(data => {
            if (data.html) {
              awardsContent.innerHTML = data.html;
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
          .then(res => res.json())
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
          .then(res => res.json())
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
                xaxis: { title: 'Week', dtick: 1 },
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
function openPlayerModal(playerId, playerName) {
  console.log('Opening player modal for ID:', playerId, 'Name:', playerName); // Debug: Log player info
  
  // Extract league context from URL path: /<platform>/<season>/<league_id>/<page>
  const pathParts = window.location.pathname.split('/').filter(p => p);
  const platform = pathParts[0] || 'sleeper';
  const season = pathParts[1] || new Date().getFullYear();
  const leagueId = pathParts[2] || null;
  
  // Build API URL with league context if available
  const apiUrl = leagueId 
    ? `/api/player-details/${playerId}?league_id=${leagueId}&platform=${platform}&season=${season}`
    : `/api/player-details/${playerId}`;
  
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
      <div class="player-modal-title-section">
        <h2 class="player-modal-name">${playerName || 'Loading...'}</h2>
        <div class="player-modal-meta" id="playerModalMeta">
          <div class="loading-spinner" style="width: 16px; height: 16px;"></div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
        <span id="playerModalBreakoutSlot"></span>
        <button class="player-modal-close" onclick="closePlayerModal()">×</button>
      </div>
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

  // Fetch player data
  fetch(apiUrl)
    .then(res => res.json())
    .then(data => {
      console.log('Player modal data received:', data); // Debug: Log received data
      
      if (data.error) {
        document.getElementById('playerModalBody').innerHTML = `
          <div class="player-modal-loading">
            <div style="color: #ef4444; font-weight: 500;">Error loading player data</div>
            <div style="font-size: 13px;">${data.error}</div>
          </div>
        `;
        return;
      }
      
      // Check if data has expected structure
      if (!data.name) {
        document.getElementById('playerModalBody').innerHTML = `
          <div class="player-modal-loading">
            <div style="color: #f59e0b; font-weight: 500;">Player data incomplete</div>
            <div style="font-size: 13px;">Player ID: ${playerId}</div>
            <div style="font-size: 11px; color: #6b7280;">Raw data: ${JSON.stringify(data, null, 2)}</div>
          </div>
        `;
        return;
      }

      // Determine badges with position-aware thresholds
      let badges = '';
      const value = data.stats?.value || 0;
      const pos = data.position;
      const yearsExp = data.stats?.years_exp;

      // Position-specific elite thresholds (players who would make any team better)
      const eliteThresholds = {
        'RB': 650,   // Elite young backs
        'WR': 650,   // Elite WRs
        'TE': 550,   // Premium TE scarcity
        'QB': 400,   // Solid QB1s
        'K': 9999,   // No elite kickers
        'DEF': 9999  // No elite defenses
      };

      const threshold = eliteThresholds[pos] || 750;
      const isElite = value >= threshold;
      const isRookie = yearsExp != null && yearsExp === 0;
      const isBreakoutPlayer = !isElite && isBreakout(data.player_id);

      if (isElite) {
        badges += '<span class="elite-badge">ELITE</span>';
      }
      if (isRookie) {
        badges += '<span class="rookie-badge">ROOKIE</span>';
      }
      if (isBreakoutPlayer) {
        badges += '<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>';
      }

      // Name with inline badges
      const nameEl = document.querySelector('.player-modal-name');
      nameEl.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;';
      nameEl.innerHTML = `<span>${playerName || 'Unknown Player'}</span>${badges}`;

      // Meta with dots separator
      const metaParts = [];
      if (data.position && data.pos_rank) metaParts.push(`<span style="font-weight:600;color:var(--text);">${data.position}${data.pos_rank}</span>`);
      if (data.team) metaParts.push(`<span>${data.team}</span>`);
      if (data.age) metaParts.push(`<span>${data.age.toFixed(1)} yrs</span>`);
      document.getElementById('playerModalMeta').innerHTML = metaParts.join('<span style="opacity:.35;margin:0 3px;">·</span>');

      // Build modal body
      let bodyHTML = '';

      // Breakout button in header slot (shown when player has a breakout score)
      if (isBreakoutPlayer) {
        const slot = document.getElementById('playerModalBreakoutSlot');
        if (slot) {
          const bkHeaderBtn = document.createElement('button');
          bkHeaderBtn.id = 'playerModalBreakoutBtn';
          bkHeaderBtn.textContent = '🔥 Breakout Analysis';
          bkHeaderBtn.style.cssText = `
            background: rgba(16,185,129,0.1);
            border: 1px solid rgba(16,185,129,0.3);
            color: #10b981;
            border-radius: 7px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
          `;
          slot.appendChild(bkHeaderBtn);
        }
      }

      // ── Hero row ─────────────────────────────────────────────────────────
      const val1qb = data.stats?.value || 0;
      const valsf  = data.stats?.sf_value || 0;
      const posRankLabel = data.stats?.pos_rank_label || (data.stats?.pos_rank ? `${pos}${data.stats.pos_rank}` : '');
      const expLabel = data.stats?.years_exp === 0 ? 'Rookie'
        : data.stats?.years_exp != null ? `${data.stats.years_exp} yr${data.stats.years_exp !== 1 ? 's' : ''}`
        : '—';

      const thirdValueCard = data.stats?.pos_rank
        ? `<div class="pm-hero-stat">
            <div class="pm-hero-label">Pos Rank</div>
            <div class="pm-hero-val">${posRankLabel || data.stats.pos_rank}</div>
            <div class="pm-hero-sub">${expLabel}</div>
          </div>`
        : `<div class="pm-hero-stat">
            <div class="pm-hero-label">Experience</div>
            <div class="pm-hero-val">${expLabel}</div>
            <div class="pm-hero-sub">${pos || ''}</div>
          </div>`;

      bodyHTML += `
        <div class="pm-hero-row">
          <div class="pm-hero-stat pm-hero-primary">
            <div class="pm-hero-label">1QB Value</div>
            <div class="pm-hero-val" style="color:#3b82f6;">${val1qb > 0 ? val1qb : '—'}</div>
            <div class="pm-hero-sub">${posRankLabel || pos || ''}</div>
          </div>
          <div class="pm-hero-stat">
            <div class="pm-hero-label">SF Value</div>
            <div class="pm-hero-val">${valsf > 0 ? valsf : '—'}</div>
            <div class="pm-hero-sub">Superflex</div>
          </div>
          ${thirdValueCard}
        </div>
      `;

      // ── Advanced Metrics + Value History (side by side) ───────────────────
      const hasMetrics = pos && pos !== 'K' && pos !== 'DEF';
      const hasChart   = data.value_history && data.value_history.length > 0;

      if (hasMetrics || hasChart) {
        bodyHTML += `<hr class="pm-section-divider"><div class="pm-metrics-chart-grid">`;

        if (hasMetrics) {
          bodyHTML += `
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
          `;
        } else {
          bodyHTML += `<div></div>`; // empty left cell so chart stays right
        }

        if (hasChart) {
          bodyHTML += `
            <div>
              <div class="pm-section-header"><span class="pm-section-label">Value History</span></div>
              <div class="player-modal-chart-container" id="playerValueChart" style="min-height:200px;"></div>
            </div>
          `;
        } else {
          bodyHTML += `<div></div>`;
        }

        bodyHTML += `</div>`;
      }

      // ── Game Logs ─────────────────────────────────────────────────────────
      if (data.game_logs_by_year && Object.keys(data.game_logs_by_year).length > 0) {
        bodyHTML += `
          <hr class="pm-section-divider">
          <div class="player-modal-section">
            <div class="pm-section-header"><span class="pm-section-label">Game Logs</span></div>
        `;

        // Sort years in descending order (most recent first)
        const years = Object.keys(data.game_logs_by_year).sort((a, b) => b - a);

        years.forEach((year, index) => {
          const gameLogs = data.game_logs_by_year[year];
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
          if (totalRec >
           0) seasonSummaryParts.push(`${totalRec} rec`);
          const seasonSummary = seasonSummaryParts.join(' • ');

          bodyHTML += `
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

            const val = (v) => v != null && v > 0 ? v : '—';
            const rowClass = hasAnyStats ? 'game-log-table-row' : 'game-log-table-row game-log-no-stats';

            bodyHTML += `
              <tr class="${rowClass}">
                <td>${dateStr}</td>
                <td class="game-log-table-opp">${game.opponent || '—'}</td>
                <td class="game-log-table-pts">${hasAnyStats ? (game.fantasy_pts != null ? game.fantasy_pts.toFixed(1) : '—') : '<span style="color:#9ca3af;">DNP</span>'}</td>
                <td>${val(stats.pass_yd) !== '—' ? Math.round(stats.pass_yd) : '—'}</td>
                <td>${val(stats.pass_td)}</td>
                <td>${val(stats.pass_int)}</td>
                <td>${val(stats.rush_att)}</td>
                <td>${val(stats.rush_yd) !== '—' ? Math.round(stats.rush_yd) : '—'}</td>
                <td>${val(stats.rush_td)}</td>
                <td>${val(stats.rec_tgt)}</td>
                <td>${val(stats.rec)}</td>
                <td>${val(stats.rec_yd) !== '—' ? Math.round(stats.rec_yd) : '—'}</td>
                <td>${val(stats.rec_td)}</td>
              </tr>
            `;
          });

          const valTotal = (v) => v != null && v > 0 ? v : '—';

          // Add season totals row in table format (inside the table)
          bodyHTML += `
                  </tbody>
                  <tfoot>
                    <tr class="game-log-table-total">
                      <td><strong>Total</strong></td>
                      <td><strong>${gameLogs.length}G</strong></td>
                      <td class="game-log-table-pts"><strong>${totalFantasyPts.toFixed(1)}</strong></td>
                      <td><strong>${valTotal(totalPassYd) !== '—' ? Math.round(totalPassYd) : '—'}</strong></td>
                      <td><strong>${valTotal(totalPassTd)}</strong></td>
                      <td><strong>${valTotal(totalPassInt)}</strong></td>
                      <td><strong>${valTotal(totalRushAtt)}</strong></td>
                      <td><strong>${valTotal(totalRushYd) !== '—' ? Math.round(totalRushYd) : '—'}</strong></td>
                      <td><strong>${valTotal(totalRushTd)}</strong></td>
                      <td><strong>${valTotal(totalRecTgt)}</strong></td>
                      <td><strong>${valTotal(totalRec)}</strong></td>
                      <td><strong>${valTotal(totalRecYd) !== '—' ? Math.round(totalRecYd) : '—'}</strong></td>
                      <td><strong>${valTotal(totalRecTd)}</strong></td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>
          `;
        });

        bodyHTML += `
          </div>
        `;
      }

      document.getElementById('playerModalBody').innerHTML = bodyHTML || '<div class="player-modal-loading"><div>No data available</div></div>';

      // Wire up breakout header button
      const bkBtn = document.getElementById('playerModalBreakoutBtn');
      if (bkBtn) {
        bkBtn.addEventListener('click', () => {
          closePlayerModal();
          openBreakoutModal(playerId, playerName);
        });
      }

      // Render value history chart if data exists
      if (data.value_history && data.value_history.length > 0) {
        const chartDiv = document.getElementById('playerValueChart');
        if (chartDiv && typeof Plotly !== 'undefined') {
          // Robust date formatters (handle YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS)
          const formatDateLabel = (dateStr) => {
            if (!dateStr) return '';
            const m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})/);
            if (!m) return '';
            const d = new Date(+m[1], +m[2] - 1, +m[3]);
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          };
          const formatMonthOnly = (dateStr) => {
            if (!dateStr) return '';
            const m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})/);
            if (!m) return '';
            return new Date(+m[1], +m[2] - 1, +m[3]).toLocaleDateString('en-US', { month: 'short' });
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

          const trace = {
            x: xData,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Value',
            line: { color: '#3b82f6', width: 2, shape: 'spline', smoothing: 1.2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: '%{y:.1f}<extra></extra>'
          };

          // Adjust chart height based on screen size
          const isMobile = window.innerWidth <= 768;
          const chartHeight = isMobile ? 200 : 250;

          const layout = {
            margin: { l: 40, r: 20, t: 10, b: 36 },
            height: chartHeight,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
              showgrid: false,
              type: 'category',
              tickmode: 'array',
              tickvals: tickvals,
              ticktext: ticktext,
              tickangle: 0,
              tickfont: { size: 11, color: mutedColor },
            },
            yaxis: {
              showgrid: true,
              showticklabels: false,
              range: [yMin - yPad, yMax + yPad],
            },
            hovermode: 'x unified',
          };

          Plotly.newPlot('playerValueChart', [trace], layout, {
            displayModeBar: false,
            responsive: true
          });
        }
      }

      // Fetch and render advanced metrics (bars go into #advancedMetricsContent)
      const advancedSection = document.getElementById('advancedMetricsSection');
      if (advancedSection) {
        const path = window.location.pathname;
        const match = path.match(/\/(sleeper|espn)\/(\d+)\/([^\/]+)/);
        const leagueId = match ? match[3] : null;
        loadAdvancedMetrics(playerId, leagueId, null);
      }
    })
    .catch(err => {
      console.error('Error loading player data:', err);
      document.getElementById('playerModalBody').innerHTML = `
        <div class="player-modal-loading">
          <div style="color: #ef4444; font-weight: 500;">Error loading player data</div>
          <div style="font-size: 13px;">Please try again</div>
        </div>
      `;
    });
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
    .then(res => res.json())
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
  if (metrics.snap_share != null) {
    const pct = metrics.snap_share * 100;
    defs.push({ label: 'Snap Share', fill: pct, display: pct.toFixed(1) + '%' });
  }

  if (position === 'QB') {
    if (metrics.yards_per_attempt != null) {
      const v = metrics.yards_per_attempt;
      defs.push({ label: 'Yds/Attempt', fill: Math.min(v / 10 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.completion_rate != null) {
      const pct = metrics.completion_rate * 100;
      defs.push({ label: 'Completion %', fill: pct, display: pct.toFixed(1) + '%' });
    }
  } else if (position === 'RB') {
    if (metrics.yards_per_carry != null) {
      const v = metrics.yards_per_carry;
      defs.push({ label: 'Yds/Carry', fill: Math.min(v / 7 * 100, 100), display: v.toFixed(1) });
    }
    if (metrics.opportunity_share != null) {
      defs.push({ label: 'Opp Share', fill: Math.min(metrics.opportunity_share, 100), display: metrics.opportunity_share.toFixed(1) + '%' });
    }
  } else if (position === 'WR' || position === 'TE') {
    if (metrics.target_share != null) {
      const pct = metrics.target_share * 100;
      defs.push({ label: 'Target Share', fill: pct, display: pct.toFixed(1) + '%' });
    }
    if (metrics.catch_rate != null) {
      const pct = metrics.catch_rate * 100;
      defs.push({ label: 'Catch Rate', fill: pct, display: pct.toFixed(1) + '%' });
    }
    if (metrics.yards_per_target != null) {
      const v = metrics.yards_per_target;
      defs.push({ label: 'Yds/Target', fill: Math.min(v / 14 * 100, 100), display: v.toFixed(1) });
    }
  }

  if (metrics.red_zone_usage != null && position !== 'QB') {
    const v = metrics.red_zone_usage;
    defs.push({ label: 'RZ Usage/G', fill: Math.min(v / 3 * 100, 100), display: v.toFixed(1) });
  }

  if (metrics.efficiency_trend != null) {
    const trend = metrics.efficiency_trend;
    const icon = trend > 5 ? '↗ ' : trend < -5 ? '↘ ' : '';
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
    toggle.textContent = '▶';
  } else {
    content.classList.add('expanded');
    toggle.textContent = '▼';
  }
}

function closePlayerModal() {
  const overlay = document.querySelector('.player-modal-overlay');
  if (overlay) {
    document.body.style.overflow = '';
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 200);
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
        'RB': 650, 'WR': 650, 'TE': 550, 'QB': 400, 'K': 9999, 'DEF': 9999
      };
      const threshold = eliteThresholds[position] || 750;
      const isElite = value >= threshold;
      const isRookie = yearsExp !== null && yearsExp !== '' && parseInt(yearsExp) === 0;
      
      if (isElite) {
        badges.push('<span class="elite-badge">ELITE</span>');
      }
      if (isRookie) {
        badges.push('<span class="rookie-badge">ROOKIE</span>');
      }
      
      // Only show breakout badge if player is not elite
      if (!isElite && isBreakout(playerId)) {
        badges.push('<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>');
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

  modal.innerHTML = `
    <div class="team-modal-header">
      <div class="team-modal-title-section">
        <h2 class="team-modal-name">${teamName || 'Loading...'}</h2>
        <div class="team-modal-meta" id="teamModalMeta">
          <div class="loading-spinner" style="width: 16px; height: 16px;"></div>
        </div>
      </div>
      <button class="team-modal-close" onclick="closeTeamModal()">×</button>
    </div>
    <div class="team-modal-body" id="teamModalBody">
      <div class="team-modal-loading">
        <div class="loading-spinner"></div>
        <div>Loading team details...</div>
      </div>
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

    const response = await fetch(`/api/team-details/${rosterId}?league_id=${leagueId}&platform=${platform}&season=${season}`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderTeamDetails(data);

  } catch (error) {
    console.error('[team-modal] Error fetching team details:', error);
    document.getElementById('teamModalBody').innerHTML = `
      <div class="team-modal-error">
        <div>Failed to load team details</div>
        <div style="color: #9ca3af; font-size: 14px;">${error.message}</div>
      </div>
    `;
  }
}

// Global player indicators for team modal
let globalPlayerIndicators = { rookies: [], breakouts: [] };

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

// Global isBreakout function
function isBreakout(playerId) {
  return globalPlayerIndicators.breakouts && globalPlayerIndicators.breakouts.includes(String(playerId));
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
        <h2 class="player-modal-name" id="bkModalName">${displayName}</h2>
        <div class="player-modal-meta" id="bkModalMeta">
          <div class="loading-spinner" style="width:16px;height:16px;"></div>
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
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        document.getElementById('bkModalBody').innerHTML = `
          <div class="player-modal-loading">
            <div style="color:#ef4444;font-weight:500;">No breakout data found</div>
            <div style="font-size:13px;color:var(--text-muted);">This player has no breakout score for the current season.</div>
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
  const emoji  = breakoutType.emoji || '📊';
  const label  = breakoutType.profile_label || 'Breakout Candidate';
  const driver = breakoutType.primary_driver || 'balanced';
  const formattedPhase = data.phase
    ? data.phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : '—';

  // Header: name + meta with dots
  const nameEl = document.getElementById('bkModalName');
  if (nameEl) nameEl.textContent = name;
  const metaParts = [];
  if (pos)  metaParts.push(`<span style="font-weight:600;color:var(--text);">${pos}</span>`);
  if (team) metaParts.push(`<span>${team}</span>`);
  if (formattedPhase !== '—') metaParts.push(`<span>${formattedPhase}</span>`);
  document.getElementById('bkModalMeta').innerHTML = metaParts.join('<span style="opacity:.35;margin:0 4px;">·</span>');

  // Score color
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
        <div class="pm-hero-sub">${label}</div>
      </div>
      <div class="pm-hero-stat" style="text-align:left;padding-left:16px;">
        <div class="pm-hero-label">Profile</div>
        <div style="font-size:22px;line-height:1;margin:4px 0;">${emoji}</div>
        <div class="pm-hero-sub" style="font-weight:600;color:var(--text);">${driver} driven</div>
      </div>
      <div class="pm-hero-stat">
        <div class="pm-hero-label">Phase</div>
        <div style="font-size:13px;font-weight:700;color:var(--text);line-height:1.3;margin:4px 0;">${formattedPhase}</div>
        <div class="pm-hero-sub">${pos}${pos && team ? ' · ' : ''}${team}</div>
      </div>
    </div>
  `;

  // ── Component breakdown with bars ─────────────────────────────────────────
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

  html += '<div class="pm-comp-list">';
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

  // ── Context boxes ─────────────────────────────────────────────────────────
  if (txnSummary) {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Vacated Opportunity</span></div>
      <div class="pm-context-box">${txnSummary}</div>
    `;
  }
  if (addedCompSumm) {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Added Competition</span></div>
      <div class="pm-context-box competition">${addedCompSumm}</div>
    `;
  }

  // ── Key factors ───────────────────────────────────────────────────────────
  if (reasons.length) {
    html += `
      <hr class="pm-section-divider">
      <div class="pm-section-header"><span class="pm-section-label">Key Factors</span></div>
      <div style="display:flex;flex-direction:column;gap:6px;">
    `;
    reasons.forEach(r => {
      html += `<div style="font-size:13px;color:var(--text-muted);display:flex;gap:8px;align-items:flex-start;">
        <span style="color:${scoreColor};font-weight:700;flex-shrink:0;">•</span><span>${r}</span>
      </div>`;
    });
    html += '</div>';
  }

  // ── Footer CTA ────────────────────────────────────────────────────────────
  html += `
    <div class="pm-footer">
      <button id="bkViewProfileBtn" class="pm-profile-btn">View Full Player Profile →</button>
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
      // Determine badges with position-aware thresholds
      let badges = '';
      const value = player.value || 0;
      const pos = player.position;

      // Position-specific elite thresholds (players who would make any team better)
      const eliteThresholds = {
        'RB': 650,   // Elite young backs
        'WR': 650,   // Elite WRs
        'TE': 550,   // Premium TE scarcity
        'QB': 400,   // Solid QB1s
        'K': 9999,   // No elite kickers
        'DEF': 9999  // No elite defenses
      };

      const threshold = eliteThresholds[pos] || 750;
      const isElite = value >= threshold;
      const isRookie = player.years_exp != null && player.years_exp === 0;
      const isBreakoutPlayer = !isElite && isBreakout(player.player_id);

      if (isElite) {
        badges += '<span class="elite-badge">ELITE</span>';
      }
      if (isRookie) {
        badges += '<span class="rookie-badge">ROOKIE</span>';
      }
      if (isBreakoutPlayer) {
        badges += '<span class="player-badge player-badge-breakout">🔥 BREAKOUT</span>';
      }

      rosterHTML += `
        <tr style="cursor:pointer;" data-player-id="${player.player_id}" data-player-name="${player.name}">
          <td>
            <strong class="player-clickable">${player.name}</strong>
            ${badges}
          </td>
          <td><span class="pos-badge ${player.position}">${player.position}</span></td>
          <td>${player.team || '—'}</td>
          <td>${player.age != null ? player.age.toFixed(1) : '—'}</td>
          <td>${player.value != null ? player.value.toFixed(1) : '—'}</td>
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

  // Build graphs section
  let graphsHTML = '';
  console.log('[team-modal] Graphs data:', data.graphs);

  if (data.graphs && (data.graphs.weekly_scores || data.graphs.radar)) {
    graphsHTML += '<div class="team-modal-section"><h3>Performance Charts</h3>';

    // Weekly scores line chart
    if (data.graphs.weekly_scores && data.graphs.weekly_scores.length > 0) {
      graphsHTML += '<div class="team-chart-container" id="teamWeeklyChart"></div>';
    }

    // Radar chart
    if (data.graphs.radar && data.graphs.radar.z_scores) {
      graphsHTML += '<div class="team-chart-container" id="teamRadarChart"></div>';
    }

    graphsHTML += '</div>';
  }

  // Set body content with two-column layout
  const leftColumn = `<div class="team-modal-body-left">${rosterHTML}</div>`;
  const rightColumn = `<div class="team-modal-body-right">${graphsHTML}${picksHTML}</div>`;
  document.getElementById('teamModalBody').innerHTML = leftColumn + rightColumn;

  // Render charts using Plotly (if data exists)
  if (data.graphs && typeof Plotly !== 'undefined') {
    // Render weekly scores chart
    if (data.graphs.weekly_scores && data.graphs.weekly_scores.length > 0) {
      const weeks = data.graphs.weekly_scores.map(d => d.week);
      const points = data.graphs.weekly_scores.map(d => d.points);

      const traces = [{
        x: weeks,
        y: points,
        type: 'scatter',
        mode: 'lines+markers',
        name: data.team_name,
        line: { color: '#667eea', width: 3 },
        marker: { size: 8 }
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
          opacity: 0.7
        });
      }

      const weeklyLayout = {
        xaxis: { title: 'Week', standoff: 12 },
        yaxis: { title: 'Points' },
        hovermode: 'x unified',
        margin: { l: 50, r: 20, t: 20, b: 50 },
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h' }
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
        `${metric}: ${rawStats[metric]} (z: ${zScores[i].toFixed(2)})`
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

      const radarLayout = {
        polar: {
          radialaxis: {
            visible: true,
            range: [-3, 3],
            tickvals: [-3, -2, -1, 0, 1, 2, 3]
          }
        },
        margin: { l: 60, r: 60, t: 40, b: 40 },
        showlegend: false
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
