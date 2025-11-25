document.addEventListener("DOMContentLoaded", function () {
  const pills = Array.from(document.querySelectorAll(".manager-pill"));
  const panels = Array.from(document.querySelectorAll(".team-panel"));
  const leftArrow = document.querySelector(".pill-arrow-left");
  const rightArrow = document.querySelector(".pill-arrow-right");

  if (!pills.length) return;

  let currentIndex = pills.findIndex(p => p.classList.contains("active"));
  if (currentIndex === -1) currentIndex = 0;

  function activateIndex(idx) {
    // wrap around
    if (idx < 0) idx = pills.length - 1;
    if (idx >= pills.length) idx = 0;
    currentIndex = idx;

    const activePill = pills[currentIndex];
    const teamId = activePill.getAttribute("data-team-id");

    // update pills
    pills.forEach(p => p.classList.remove("active"));
    activePill.classList.add("active");

    // update panels
    panels.forEach(panel => {
      const pid = panel.getAttribute("data-team-id");
      panel.classList.toggle("active", pid === teamId);
    });

    // scroll the pill into view inside the pills row
    activePill.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest"
    });
  }

  // existing pill click behavior
  pills.forEach((pill, idx) => {
    pill.addEventListener("click", () => {
      activateIndex(idx);
    });
  });

  // arrows
  if (leftArrow) {
    leftArrow.addEventListener("click", () => {
      activateIndex(currentIndex - 1);
    });
  }

  if (rightArrow) {
    rightArrow.addEventListener("click", () => {
      activateIndex(currentIndex + 1);
    });
  }

  // ensure initial state is synced
  activateIndex(currentIndex);
});

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.card-tabs').forEach(card => {
    const tabs = card.querySelectorAll('.tab-btn');
    const panels = card.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;

        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        panels.forEach(p => {
          p.classList.toggle('active', p.dataset.tab === target);
        });
      });
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const tabs = document.querySelectorAll(".team-tab");
  const panels = document.querySelectorAll(".team-panel");

  if (!tabs.length) return;

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const id = tab.getAttribute("data-team-id");

      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      panels.forEach(p => {
        if (p.getAttribute("data-team-id") === id) {
          p.classList.add("active");
        } else {
          p.classList.remove("active");
        }
      });
    });
  });
});

    (function () {
      const tbl = document.getElementById('stats');
      const NUMERIC_COLS = new Set([2,3,4,5,6,7,8]);
      const getVal = (cell, idx) => {
        const t = cell.textContent.trim();
        if (!NUMERIC_COLS.has(idx)) return t.toLowerCase();
        const n = parseFloat(t.replace(/,/g,'')); return isNaN(n) ? -Infinity : n;
      };
      let sortSpec = [];
      const applySort = () => {
        const tbody = tbl.tBodies[0];
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {
          for (const {col, dir} of sortSpec) {
            const A = getVal(a.children[col], col), B = getVal(b.children[col], col);
            if (A < B) return -1 * dir; if (A > B) return  1 * dir;
          } return 0;
        });
        tbody.innerHTML = ''; rows.forEach(r => tbody.appendChild(r));
        tbl.querySelectorAll('th').forEach(th => th.classList.remove('sorted-asc','sorted-desc','sorted-secondary'));
        if (sortSpec.length) {
          const primary = sortSpec[0];
          const th = tbl.tHead.rows[0].children[primary.col];
          th.classList.add(primary.dir === 1 ? 'sorted-asc' : 'sorted-desc');
          for (let i=1;i<sortSpec.length;i++){
            tbl.tHead.rows[0].children[sortSpec[i].col].classList.add('sorted-secondary');
          }
        }
      };
      const toggleSort = (col, additive=false) => {
        if (!additive) sortSpec = [];
        const i = sortSpec.findIndex(s => s.col === col);
        if (i === -1) {
          const dir = NUMERIC_COLS.has(col) ? -1 : 1;
          sortSpec.push({col, dir});
        } else {
          const cur = sortSpec[i];
          cur.dir = cur.dir === -1 ? 1 : null;
          if (cur.dir === null) sortSpec.splice(i, 1);
        }
        applySort();
      };
      tbl.tHead.addEventListener('click', (e) => {
        if (e.target.tagName !== 'TH') return;
        const col = parseInt(e.target.getAttribute('data-col'));
        toggleSort(col, e.shiftKey);
      });
      sortSpec = [{col: 2, dir: -1}, {col: 3, dir: -1}];
      applySort();
    })();
