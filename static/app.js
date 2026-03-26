// ============================================================
// app.js (DROP-IN)
// - Idempotent init (safe across innerHTML swaps)
// - Standings sort supports DESC -> ASC -> none (and Shift+click multi-sort)
// - Refresh button supports platform/season and re-inits page root
// - Matchup carousel handlers delegated + resize-safe
// - Plotly: resize + relayout + redraw + viewport hooks (better with page zoom/layout shifts)
// ============================================================

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

  const state = {
    sideAPlayers: [],
    sideBPlayers: [],
    sideAPicks: [],
    sideBPicks: [],
  };

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
          value: p.value
        })),
        sideBPlayers: state.sideBPlayers.map(p => ({
          id: p.id,
          name: p.name,
          position: p.position,
          team: p.team,
          value: p.value
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
      const va = Number(a?.value || 0);
      const vb = Number(b?.value || 0);
      return vb - va;
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

    if (pos === "PICK") {
      metaBits.push("PICK");
    } else if (pos) {
      if (p.pos_rank_label) metaBits.push(String(p.pos_rank_label).toUpperCase());
      else metaBits.push(pos);
    }

    if (p.team) metaBits.push(p.team);
    if (p.age != null && p.age !== "") metaBits.push(`${p.age} yrs`);

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
    valueSpan.textContent = formatValue(p.value);

    topLine.appendChild(nameSpan);
    topLine.appendChild(valueSpan);

    const metaSpan = document.createElement("div");
    metaSpan.className = "otc-value-sub";
    metaSpan.textContent = buildMetaBits(p).join(" • ");

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
    sub.textContent = buildMetaBits(p).join(" • ");

    left.appendChild(top);
    left.appendChild(sub);

    const value = document.createElement("div");
    value.className = "otc-dropdown-value";
    value.textContent = formatValue(p.value);

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

  async function loadTopMovers() {
    const risersEl = root.querySelector("#otcRisersList");
    const fallersEl = root.querySelector("#otcFallersList");

    try {
      const res = await fetch("/api/value-movers?days=7&limit=5", { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load movers.");

      const data = await res.json();
      const usedDays = data?.used_days;

      const sub = root.querySelector("#moversSub");
      if (sub && usedDays) {
        sub.textContent = `Biggest ${usedDays}-day changes in BR value`;
      }

      renderMovers(data);
    } catch (err) {
      console.error("[trade] movers error:", err);
      if (risersEl) risersEl.innerHTML = '<div class="otc-movers-empty">Unable to load risers.</div>';
      if (fallersEl) fallersEl.innerHTML = '<div class="otc-movers-empty">Unable to load fallers.</div>';
    }
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
      pos_rank_label: p.pos_rank_label || "",
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
      .sort((a, b) => Number(b.value || 0) - Number(a.value || 0));

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
      metaEl.textContent = buildMetaBits(p).join(" · ");

      leftWrap.appendChild(nameEl);
      leftWrap.appendChild(metaEl);

      const rightWrap = document.createElement("div");
      rightWrap.className = "otc-chip-value-wrap";

      const valueEl = document.createElement("span");
      valueEl.className = "otc-chip-value";
      valueEl.textContent = formatValue(p.value);

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
      valueEl.textContent = formatValue(pickData ? pickData.value : 0);

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

    const viewerSide =
      root.querySelector('input[name="viewerSide"]:checked')?.value || "a";

    const payload = {
      league_id: root.querySelector("#leagueIdInput")?.value || "",
      season: root.querySelector("#seasonInput")?.value || "",
      viewer_side: viewerSide,
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
      const matches = allPlayers
        .filter(p => p.name && p.name.toLowerCase().includes(query))
        .filter(p => !selected.find(x => String(x.id) === String(p.id)))
        .filter(p => !selectedPicks.find(x => String(x.id) === String(p.id)))
        .slice(0, 20);

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

  function bindViewerSideControls() {
    root.querySelectorAll('input[name="viewerSide"]').forEach(el => {
      bindOnce(el, "tradeViewerSideChange", "change", () => {
        syncViewerSideLabels();
        recomputeTrade();
      });
    });

    syncViewerSideLabels();
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
  ]).then(() => {
    setupSearch("A");
    setupSearch("B");
    bindViewerSideControls();
    bindAnalyzeTrade();
    bindTeamSelector();
    bindSetupButton();
    bindClearTradeButton();
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
  window.initPageRoot(document);
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
  const lookupBtn = document.getElementById("lookupBtn");
  const usernameInput = document.getElementById("username");
  const leagueSelect = document.getElementById("league");
  const leagueSelectWrap = document.getElementById("leagueSelectWrap");
  const generateWrap = document.getElementById("generateWrap");
  const errorBox = document.getElementById("lookupError");
  const formPlatform = document.getElementById("formPlatform");

  if (!platformBtns.length) return;

  let currentPlatform = "sleeper";

  // Platform switching (ESPN coming soon, only Sleeper active)
  platformBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const platform = btn.dataset.platform;

      // Only handle Sleeper for now
      if (platform !== "sleeper") return;

      currentPlatform = platform;

      // Update active state
      platformBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Update form platform
      if (formPlatform) formPlatform.value = platform;

      // Reset state
      if (leagueSelectWrap) leagueSelectWrap.style.display = "none";
      if (generateWrap) generateWrap.style.display = "none";
      if (errorBox) errorBox.style.display = "none";
    });
  });

  // Sleeper username input
  if (usernameInput) {
    usernameInput.addEventListener("input", () => {
      const formUsername = document.getElementById("formUsername");
      if (formUsername) {
        formUsername.value = usernameInput.value.trim();
      }
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