// Isolated-world content script on ESPN's football draft room. Receives the
// MAIN-world pick snapshot (CustomEvent) and forwards it to the service worker,
// which relays to open BR Fantasy Draft Room tabs. Also shows a small sync
// status chip so the user knows BR Fantasy is watching.

(function () {
  "use strict";

  const EVENT = "brfantasy:espn-draft-raw";
  let lastSent = "";
  let lastPickCount = 0;
  let chip = null;

  function leagueFromUrl() {
    try {
      const u = new URL(location.href);
      return {
        leagueId: (u.searchParams.get("leagueId") || "").trim(),
        season: (u.searchParams.get("seasonId") || u.searchParams.get("season") || "").trim(),
      };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function ensureChip() {
    if (chip && document.documentElement.contains(chip)) return chip;
    chip = document.createElement("div");
    chip.id = "br-fantasy-espn-sync-chip";
    chip.setAttribute(
      "style",
      [
        "position:fixed",
        "z-index:2147483646",
        "right:12px",
        "bottom:12px",
        "max-width:min(280px,calc(100vw - 24px))",
        "padding:10px 12px",
        "border-radius:10px",
        "background:#0f172a",
        "color:#f8fafc",
        "font:600 12px/1.35 system-ui,-apple-system,sans-serif",
        "box-shadow:0 10px 28px rgba(0,0,0,.35)",
        "pointer-events:none",
      ].join(";")
    );
    chip.textContent = "BR Fantasy · watching draft…";
    document.documentElement.appendChild(chip);
    return chip;
  }

  function setChip(text, ok) {
    const el = ensureChip();
    el.textContent = text;
    el.style.background = ok ? "#065f46" : "#0f172a";
  }

  function forward(detail) {
    if (!detail || !Array.isArray(detail.picks)) return;
    const ids = leagueFromUrl();
    const payload = {
      type: "espnDraftRelay",
      leagueId: detail.leagueId || ids.leagueId,
      season: detail.season || ids.season,
      inProgress: !!detail.inProgress,
      drafted: !!detail.drafted,
      picks: detail.picks,
      source: detail.source || "espn-draft-room",
      at: detail.at || Date.now(),
    };
    if (!payload.leagueId) return;
    const fp = [
      payload.leagueId,
      payload.season,
      payload.picks.length,
      payload.picks.length ? payload.picks[payload.picks.length - 1].overallPickNumber : 0,
      payload.picks.length ? payload.picks[payload.picks.length - 1].playerId : "",
    ].join("|");
    if (fp === lastSent) return;
    lastSent = fp;
    const grew = payload.picks.length > lastPickCount;
    lastPickCount = payload.picks.length;
    try {
      chrome.runtime.sendMessage(payload, (resp) => {
        void chrome.runtime.lastError;
        if (resp && resp.sent > 0) {
          setChip(
            grew
              ? "BR Fantasy · synced " + payload.picks.length + " picks"
              : "BR Fantasy · connected · " + payload.picks.length + " picks",
            true
          );
        } else {
          setChip("BR Fantasy · open Draft Room to receive picks", false);
        }
      });
    } catch (_e) {
      setChip("BR Fantasy · reload extension", false);
    }
  }

  window.addEventListener(EVENT, (ev) => {
    forward(ev && ev.detail);
  });

  ensureChip();
  try {
    chrome.runtime.sendMessage(
      {
        type: "espnDraftTabReady",
        leagueId: leagueFromUrl().leagueId,
        season: leagueFromUrl().season,
        href: location.href,
      },
      () => {
        void chrome.runtime.lastError;
      }
    );
  } catch (_e) {
    /* ignore */
  }
})();
