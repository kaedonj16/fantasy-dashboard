// Isolated-world content script on ESPN's football draft room. Receives the
// MAIN-world pick snapshot (CustomEvent) and forwards it to the service worker,
// which relays to open BR Fantasy Draft Room tabs.

(function () {
  "use strict";

  const EVENT = "brfantasy:espn-draft-raw";
  let lastSent = "";

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
    try {
      chrome.runtime.sendMessage(payload, () => {
        void chrome.runtime.lastError;
      });
    } catch (_e) {
      /* extension context invalidated */
    }
  }

  window.addEventListener(EVENT, (ev) => {
    forward(ev && ev.detail);
  });

  // Announce presence so the popup / BR page can hint that the draft tab is live.
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
