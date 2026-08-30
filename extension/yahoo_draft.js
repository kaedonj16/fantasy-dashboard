// Isolated-world content script on Yahoo's football draft room. Receives the
// MAIN-world pick snapshot (CustomEvent) and forwards it to the service worker,
// which relays to open BR Fantasy Draft Room tabs. Also shows a small sync
// status chip so the user knows BR Fantasy is watching.

(function () {
  "use strict";

  const EVENT = "brfantasy:yahoo-draft-raw";
  const RETRY_MS = 3000;
  let lastDelivered = "";
  let lastPickCount = 0;
  let pendingPayload = null;
  let retryTimer = null;
  let chip = null;

  function leagueFromUrl() {
    try {
      const u = new URL(location.href);
      let leagueId = (u.searchParams.get("leagueId") || u.searchParams.get("league") || "").trim();
      let season = (u.searchParams.get("seasonId") || u.searchParams.get("season") || "").trim();
      if (!leagueId) {
        const m = u.pathname.match(/\/f1\/(\d+)(?:\/|$)/i);
        if (m) leagueId = m[1];
      }
      if (!leagueId) {
        const key = (u.searchParams.get("leagueKey") || u.searchParams.get("key") || "").trim();
        if (key && key.indexOf(".l.") >= 0) leagueId = key.split(".l.").pop() || "";
      }
      return { leagueId, season };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function ensureChip() {
    if (chip && document.documentElement.contains(chip)) return chip;
    chip = document.createElement("div");
    chip.id = "br-fantasy-yahoo-sync-chip";
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
    chip.textContent = "BR Fantasy · watching Yahoo draft…";
    document.documentElement.appendChild(chip);
    return chip;
  }

  function setChip(text, ok) {
    const el = ensureChip();
    el.textContent = text;
    el.style.background = ok ? "#065f46" : "#0f172a";
  }

  function payloadFingerprint(payload) {
    const picks = payload.picks || [];
    const last = picks.length ? picks[picks.length - 1] : null;
    return [
      payload.leagueId,
      payload.season,
      picks.length,
      last ? last.overallPickNumber : 0,
      last ? last.playerId : "",
    ].join("|");
  }

  function clearRetry() {
    if (!retryTimer) return;
    clearTimeout(retryTimer);
    retryTimer = null;
  }

  function scheduleRetry() {
    if (retryTimer) return;
    retryTimer = setTimeout(function () {
      retryTimer = null;
      deliverPending();
    }, RETRY_MS);
  }

  function deliverPending() {
    if (!pendingPayload) return;
    const payload = pendingPayload;
    const fp = payloadFingerprint(payload);
    if (fp === lastDelivered) {
      pendingPayload = null;
      clearRetry();
      return;
    }
    const grew = payload.picks.length > lastPickCount;
    try {
      chrome.runtime.sendMessage(payload, function (resp) {
        void chrome.runtime.lastError;
        if (resp && resp.sent > 0) {
          lastDelivered = fp;
          pendingPayload = null;
          clearRetry();
          lastPickCount = payload.picks.length;
          setChip(
            grew
              ? "BR Fantasy · synced " + payload.picks.length + " picks"
              : "BR Fantasy · connected · " + payload.picks.length + " picks",
            true
          );
        } else {
          setChip("BR Fantasy · open Draft Room to receive picks", false);
          scheduleRetry();
        }
      });
    } catch (_e) {
      setChip("BR Fantasy · reload extension", false);
      scheduleRetry();
    }
  }

  function forward(detail) {
    if (!detail || !Array.isArray(detail.picks)) return;
    const ids = leagueFromUrl();
    const payload = {
      type: "yahooDraftRelay",
      leagueId: detail.leagueId || ids.leagueId,
      season: detail.season || ids.season,
      inProgress: !!detail.inProgress,
      drafted: !!detail.drafted,
      picks: detail.picks,
      source: detail.source || "yahoo-draft-room",
      at: detail.at || Date.now(),
    };
    if (!payload.leagueId) return;
    const fp = payloadFingerprint(payload);
    if (fp === lastDelivered) return;
    pendingPayload = payload;
    deliverPending();
  }

  window.addEventListener(EVENT, (ev) => {
    forward(ev && ev.detail);
  });

  ensureChip();
  try {
    chrome.runtime.sendMessage(
      {
        type: "yahooDraftTabReady",
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
