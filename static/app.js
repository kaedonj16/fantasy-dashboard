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

    const rankWrap = document.createElement("div");
    rankWrap.className = "otc-value-rank";
    rankWrap.textContent = overallRank ? "#" + overallRank : "—";

    const mainWrap = document.createElement("div");
    mainWrap.className = "otc-value-main";

    const topLine = document.createElement("div");
    topLine.className = "otc-value-topline";

    const nameSpan = document.createElement("div");
    nameSpan.className = "otc-value-name";
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
    name.className = "otc-mini-name";
    name.textContent = p.name || "Unknown";

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
      const res = await fetch(`/api/value-movers?days=7&limit=5&league_type=${leagueType}&league_size=${leagueSize}`, { cache: "no-store" });
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
      const res = await fetch(`/api/offseason-breakout-candidates?min_score=30&limit=10`, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load breakouts.");

      let candidates = await res.json();

      // Filter out high-value players (already established stars)
      // Only show players with dynasty value < 2000 (roughly outside top 50)
      // Also filter out players older than 25 (breakouts are typically young players)
      const valueThreshold = 2000;
      const maxAge = 25;
      candidates = candidates.filter(c => {
        const value = leagueType === "sf" ? (c.sf_value || c.value || 0) : (c.value || 0);
        const age = c.age;
        return value < valueThreshold && (age == null || age <= maxAge);
      });

      if (!candidates || candidates.length === 0) {
        breakoutsEl.innerHTML = '<div class="otc-movers-empty">No breakout candidates found.</div>';
        if (moversPanel) moversPanel.classList.remove("otc-movers-loading");
        return;
      }

      // Limit to top 8 for display
      candidates = candidates.slice(0, 8);

      breakoutsEl.innerHTML = "";
      candidates.forEach(c => {
        const row = document.createElement("div");
        row.className = "otc-mini-row";

        const name = document.createElement("div");
        name.className = "otc-mini-name";

        const playerName = document.createElement("span");
        playerName.className = "otc-player-name";
        playerName.textContent = c.name || "Unknown";

        const meta = document.createElement("span");
        meta.className = "otc-player-meta";

        // Build meta string: position, team, age
        const metaParts = [];
        if (c.position) metaParts.push(c.position);
        if (c.team) metaParts.push(c.team);
        if (c.age != null) metaParts.push(`${c.age.toFixed(1)} yrs`);
        meta.textContent = metaParts.join(" · ");

        name.appendChild(playerName);
        name.appendChild(meta);

        row.appendChild(name);

        breakoutsEl.appendChild(row);
      });

      if (moversPanel) {
        moversPanel.classList.remove("otc-movers-loading");
        moversPanel.classList.add("otc-movers-updated");
        setTimeout(() => moversPanel.classList.remove("otc-movers-updated"), 600);
      }
    } catch (err) {
      console.error("[trade] breakouts error:", err);
      breakoutsEl.innerHTML = '<div class="otc-movers-empty">Unable to load breakouts.</div>';
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
        } else if (tab === "breakouts") {
          moversContent?.classList.remove("is-active");
          breakoutsContent?.classList.add("is-active");
          if (moversSub) {
            moversSub.style.display = "block";
            moversSub.textContent = "Young players poised for expanded roles";
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
          .filter(p => ["QB", "RB", "WR", "TE"].includes(p.position)),
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
        if (activePosFilter === "PICK") return pos === "PICK";
        return pos === activePosFilter;
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
      nameEl.className = "otc-chip-name";
      nameEl.textContent = p.name || "Unknown";

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

      metaEl.innerHTML = metaBits.join(" · ");

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

  // ESPN elements
  const espnLeagueIdInput = document.getElementById("espnLeagueIdInput");
  const espnTeamName = document.getElementById("espnTeamName");
  const espnSubmitBtn = document.getElementById("espnSubmitBtn");

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

  // ESPN submit
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

// League switcher functionality
document.addEventListener('DOMContentLoaded', function() {
  const leagueSwitcher = document.getElementById('leagueSwitcher');

  if (leagueSwitcher) {
    const currentLeagueId = leagueSwitcher.getAttribute('data-current-league');
    const currentPlatform = leagueSwitcher.getAttribute('data-current-platform');
    const currentSeason = leagueSwitcher.getAttribute('data-current-season');

    // Fetch user leagues
    fetch('/api/sleeper-user-leagues?season=' + currentSeason)
      .then(res => res.json())
      .then(data => {
        if (data.ok && data.leagues) {
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
  const navLinksWrapper = document.querySelector('.nav-links-wrapper');

  if (navToggle && navLinksWrapper) {
    navToggle.addEventListener('click', function() {
      navLinksWrapper.classList.toggle('nav-open');
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
      if (!navToggle.contains(e.target) && !navLinksWrapper.contains(e.target)) {
        navLinksWrapper.classList.remove('nav-open');
      }
    });

    // Close menu when clicking a link
    navLinksWrapper.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', function() {
        navLinksWrapper.classList.remove('nav-open');
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
});