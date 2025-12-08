// ------------------------------------------------------------
// Reusable Init Helpers
// ------------------------------------------------------------

function initManagerPills(root = document) {
  const pills = Array.from(root.querySelectorAll(".manager-pill"));
  const panels = Array.from(root.querySelectorAll(".team-panel"));
  const leftArrow = root.querySelector(".pill-arrow-left");
  const rightArrow = root.querySelector(".pill-arrow-right");

  if (!pills.length) return;

  let currentIndex = pills.findIndex(p => p.classList.contains("active"));
  if (currentIndex === -1) currentIndex = 0;

  function activateIndex(idx) {
    if (idx < 0) idx = pills.length - 1;
    if (idx >= pills.length) idx = 0;
    currentIndex = idx;

    const activePill = pills[currentIndex];
    const teamId = activePill.getAttribute("data-team-id");

    pills.forEach(p => p.classList.remove("active"));
    activePill.classList.add("active");

    panels.forEach(panel => {
      const pid = panel.getAttribute("data-team-id");
      panel.classList.toggle("active", pid === teamId);
    });

    activePill.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest",
    });
  }

  pills.forEach((pill, idx) => {
    pill.addEventListener("click", () => activateIndex(idx));
  });

  if (leftArrow) leftArrow.addEventListener("click", () => activateIndex(currentIndex - 1));
  if (rightArrow) rightArrow.addEventListener("click", () => activateIndex(currentIndex + 1));

  activateIndex(currentIndex);
}


function initCardTabs(root = document) {
  root.querySelectorAll(".card-tabs").forEach(card => {
    const tabs = card.querySelectorAll(".tab-btn");
    const panels = card.querySelectorAll(".tab-panel");

    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;

        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        panels.forEach(p => {
          p.classList.toggle("active", p.dataset.tab === target);
        });
      });
    });
  });
}


function initTeamTabs(root = document) {
  const tabs = root.querySelectorAll(".team-tab");
  const panels = root.querySelectorAll(".team-panel");

  if (!tabs.length) return;

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const id = tab.getAttribute("data-team-id");

      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      panels.forEach(p => {
        p.classList.toggle("active", p.getAttribute("data-team-id") === id);
      });
    });
  });
}


function initStandingsSort(root = document) {
  const marker = root.querySelector('[data-page="standings"]');
  if (!marker) return;

  const tbl = root.getElementById("stats") || root.querySelector("#stats");
  if (!tbl) return;

  if (tbl.dataset.sortInited === "1") return;
  tbl.dataset.sortInited = "1";

  const NUMERIC_COLS = new Set([2,3,4,5,6,7,8]);

  const getVal = (cell, idx) => {
    const t = cell.textContent.trim();
    if (!NUMERIC_COLS.has(idx)) return t.toLowerCase();
    const n = parseFloat(t.replace(/,/g, ""));
    return isNaN(n) ? -Infinity : n;
  };

  let sortSpec = [];

  const applySort = () => {
    const tbody = tbl.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll("tr"));

    rows.sort((a, b) => {
      for (const { col, dir } of sortSpec) {
        const A = getVal(a.children[col], col);
        const B = getVal(b.children[col], col);
        if (A < B) return -1 * dir;
        if (A > B) return  1 * dir;
      }
      return 0;
    });

    tbody.innerHTML = "";
    rows.forEach(r => tbody.appendChild(r));

    tbl.querySelectorAll("th").forEach(th =>
      th.classList.remove("sorted-asc","sorted-desc","sorted-secondary")
    );

    if (sortSpec.length) {
      const primary = sortSpec[0];
      const th = tbl.tHead.rows[0].children[primary.col];
      th.classList.add(primary.dir === 1 ? "sorted-asc" : "sorted-desc");

      for (let i = 1; i < sortSpec.length; i++) {
        tbl.tHead.rows[0]
          .children[sortSpec[i].col]
          .classList.add("sorted-secondary");
      }
    }
  };

  const toggleSort = (col, additive = false) => {
    if (!additive) sortSpec = [];

    const i = sortSpec.findIndex(s => s.col === col);
    if (i === -1) {
      const dir = NUMERIC_COLS.has(col) ? -1 : 1;
      sortSpec.push({ col, dir });
    } else {
      const cur = sortSpec[i];
      cur.dir = cur.dir === -1 ? 1 : null;
      if (cur.dir === null) sortSpec.splice(i, 1);
    }

    applySort();
  };

  tbl.tHead.addEventListener("click", e => {
    if (e.target.tagName !== "TH") return;
    const colAttr = e.target.getAttribute("data-col");
    if (!colAttr) return;
    const col = parseInt(colAttr, 10);
    if (!Number.isNaN(col)) toggleSort(col, e.shiftKey);
  });

  sortSpec = [{ col: 2, dir: -1 }, { col: 3, dir: -1 }];
  applySort();
}


// ------------------------------------------------------------
// Master Initializer
// ------------------------------------------------------------

window.initPageRoot = function initPageRoot(root = document) {
  initManagerPills(root);
  initCardTabs(root);
  initTeamTabs(root);
  initStandingsSort(root);

  if (window.resetMatchupCarousels) {
    window.resetMatchupCarousels(root);
  }
};

document.addEventListener("DOMContentLoaded", () => {
  window.initPageRoot(document);
});


// ------------------------------------------------------------
// Refresh Button Handler
// ------------------------------------------------------------

(function () {
  const refreshBtn = document.getElementById("refreshBtn");
  if (!refreshBtn) return;

  refreshBtn.addEventListener("click", async () => {
    const page   = refreshBtn.dataset.page || "";
    const league = (refreshBtn.dataset.league || "").trim();

    if (!league || !page) {
      console.error("Missing league or page for refresh.");
      return;
    }

    refreshBtn.disabled = true;
    refreshBtn.classList.add("refresh-spinner");

    try {
      const res = await fetch("/api/refresh-page", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ league_id: league, page }),
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
        if (window.initPageRoot) window.initPageRoot(root);
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


// ------------------------------------------------------------
// Matchup Carousel
// ------------------------------------------------------------

function getCarouselState(card) {
  const track = card.querySelector(".m-track");
  if (!track) return { track: null, slides: [], width: 1, idx: 0 };

  const slides = track.querySelectorAll(".m-slide");
  const viewport = card.querySelector(".m-carousel");
  const width = (viewport?.clientWidth) || track.clientWidth || 1;

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
    scrollToIndex(card, 0);
  });
}

document.addEventListener("DOMContentLoaded", () => initAllCarousels(document));
window.addEventListener("resize", () => initAllCarousels(document));

document.addEventListener("click", evt => {
  const prev = evt.target.closest(".m-btn-prev");
  const next = evt.target.closest(".m-btn-next");
  if (!prev && !next) return;

  const card = (prev || next).closest(".matchup-carousel");
  if (!card) return;

  const { idx } = getCarouselState(card);
  scrollToIndex(card, idx + (prev ? -1 : 1));
});

window.resetMatchupCarousels = function (root) {
  initAllCarousels(root || document);
};
