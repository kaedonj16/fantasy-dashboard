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

// Stronger than just Plotly.Plots.resize:
// 1) resize
// 2) relayout automargins + title standoff + bottom margin
// 3) redraw
function _fixOnePlotly(gd) {
  const P = window.Plotly;
  if (!P || !gd) return;

  try {
    P.Plots.resize(gd);
  } catch (e) {}

  // If margins are too small, Plotly can shove axis titles into plot area.
  // Also helps after zoom/layout shifts.
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

  // Run multiple passes because CSS zoom / font rendering / grid sizing can settle late.
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

// viewport/zoom-ish hooks
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

  // When page becomes visible again, Plotly often needs a resize to fix hover mapping
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") handler();
  });
})();

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
  const lookupBtn = document.getElementById("lookupBtn");
  const usernameInput = document.getElementById("username");
  const leagueSelect = document.getElementById("league");
  const leagueSelectWrap = document.getElementById("leagueSelectWrap");
  const generateWrap = document.getElementById("generateWrap");
  const errorBox = document.getElementById("lookupError");

  if (!lookupBtn || !usernameInput || !leagueSelect) return;

  lookupBtn.addEventListener("click", async () => {
    const username = usernameInput.value.trim();
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
});