// Isolated-world content script on ESPN's football draft room. Receives the
// MAIN-world pick snapshot (CustomEvent) and forwards it to the service worker,
// which relays to open BR Fantasy Draft Room tabs. Also shows a small sync
// status chip so the user knows BR Fantasy is watching.

(function () {
  "use strict";

  const EVENT = "brfantasy:espn-draft-raw";
  const RETRY_MS = 3000;
  const RECONNECT_COOLDOWN_MS = 5000;
  let lastDelivered = "";
  let lastPickCount = 0;
  let pendingPayload = null;
  let retryTimer = null;
  let lastManualReconnectAt = 0;
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
      ].join(";")
    );
    chip.innerHTML =
      '<span class="br-chip-text">BR Fantasy · watching draft…</span>'
      + '<button type="button" class="br-chip-reconnect" title="Reestablish sync" style="margin-top:6px;display:block;width:100%;padding:4px 8px;border-radius:6px;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.08);color:#f8fafc;font:600 11px/1.2 system-ui,sans-serif;cursor:pointer;">↻ Reconnect</button>';
    const btn = chip.querySelector(".br-chip-reconnect");
    if (btn) {
      btn.addEventListener("click", function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        manualReconnect();
      });
    }
    document.documentElement.appendChild(chip);
    return chip;
  }

  function setChip(text, ok) {
    const el = ensureChip();
    const textEl = el.querySelector(".br-chip-text");
    if (textEl) textEl.textContent = text;
    else el.textContent = text;
    el.style.background = ok ? "#065f46" : "#0f172a";
  }

  function forceResend() {
    lastDelivered = "";
    clearRetry();
    try {
      window.dispatchEvent(new CustomEvent("brfantasy:draft-rescan"));
    } catch (_e) {
      /* ignore */
    }
    if (pendingPayload) deliverPending(true);
  }

  function manualReconnect() {
    const now = Date.now();
    if (now - lastManualReconnectAt < RECONNECT_COOLDOWN_MS) {
      setChip("BR Fantasy · wait a few seconds…", false);
      return;
    }
    lastManualReconnectAt = now;
    setChip("BR Fantasy · reconnecting…", false);
    forceResend();
    try {
      chrome.runtime.sendMessage({ type: "reconnectDraftRelay", source: "espn-chip" }, function (resp) {
        void chrome.runtime.lastError;
        if (resp && resp.throttled) {
          setChip("BR Fantasy · wait a few seconds…", false);
          return;
        }
        if (resp && ((resp.br && resp.br.pinged > 0) || (resp.draft && resp.draft.pinged > 0))) {
          setChip("BR Fantasy · reconnect sent", false);
        }
      });
    } catch (_e) {
      /* ignore */
    }
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

  function relayFailureText(resp) {
    if (resp && resp.reason === "tabs_query_failed") return "BR Fantasy · reload extension";
    if (resp && resp.tabs > 0) return "BR Fantasy · reload Draft Room tab";
    return "BR Fantasy · open Draft Room on brfantasyfootball.com";
  }

  function deliverPending(force) {
    if (!pendingPayload) return;
    const payload = pendingPayload;
    const fp = payloadFingerprint(payload);
    if (!force && fp === lastDelivered) {
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
          setChip(relayFailureText(resp), false);
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
    const fp = payloadFingerprint(payload);
    if (fp === lastDelivered) return;
    pendingPayload = payload;
    deliverPending();
  }

  window.addEventListener(EVENT, (ev) => {
    forward(ev && ev.detail);
  });

  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.type === "forceDraftRelay") forceResend();
    });
  } catch (_e) {
    /* ignore */
  }

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
