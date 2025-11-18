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