// Isolated-world content script on Yahoo's football draft room. Receives the
// MAIN-world pick snapshot (postMessage) and forwards it to the service worker,
// which relays to open BR Fantasy Draft Room tabs. Also shows a small sync
// status chip so the user knows BR Fantasy is watching.

(function () {
  "use strict";

  const EVENT = "brfantasy:yahoo-draft-raw";
  const RELAY_STATUS = "brfantasy:yahoo-relay-status";
  const RESCAN = "brfantasy:draft-rescan";
  const BRIDGE = "brfantasy-bridge-v1";
  const RECONNECT_SETTLE_MS = 1200;
  const RETRY_MS = 3000;
  const RECONNECT_COOLDOWN_MS = 5000;
  let lastDelivered = "";
  let lastPickCount = 0;
  let pendingPayload = null;
  let retryTimer = null;
  let lastManualReconnectAt = 0;
  let chip = null;
  let lastMySlot = 0;
  let lastUserTeamId = null;
  let lastPicks = [];

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
      const client = u.pathname.match(/\/draftclient\/(?:nfl\/|f1\/)?(\d+)\/(\d+)/i);
      if (client) {
        if (!leagueId) leagueId = client[1];
        if (!lastUserTeamId) lastUserTeamId = client[2];
      }
      return { leagueId, season };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function overlayTeamMeta(picks) {
    const ids = {};
    (picks || []).forEach(function (p) {
      if (p && p.teamId != null && p.teamId !== "") ids[String(p.teamId)] = true;
    });
    const teams = Object.keys(ids).length;
    const out = {};
    if (teams >= 4) out.teams = teams;
    return out;
  }

  function resolveMySlot(picks, detail) {
    const meta = overlayTeamMeta(picks);
    const teams = meta.teams || 12;
    const hinted = Number(detail && detail.mySlot) || 0;
    if (hinted >= 1) {
      lastMySlot = hinted;
      return hinted;
    }
    const teamId =
      (detail && (detail.userTeamId || detail.myTeamId)) || lastUserTeamId;
    if (teamId != null && window.BRDraftSlot) {
      const fromTeam = window.BRDraftSlot.slotFromTeamId(picks, teamId, teams);
      if (fromTeam) {
        lastMySlot = fromTeam;
        return fromTeam;
      }
      lastUserTeamId = teamId;
    }
    if (window.BRDraftSlot) {
      const yahoo = window.BRDraftSlot.detectYahooSlot
        ? window.BRDraftSlot.detectYahooSlot(teams)
        : window.BRDraftSlot.detectDomSlot();
      if (yahoo) {
        lastMySlot = yahoo;
        return yahoo;
      }
    }
    const urlTeam = window.BRDraftSlot && BRDraftSlot.yahooClientTeamId
      ? BRDraftSlot.yahooClientTeamId()
      : "";
    if (urlTeam && !lastUserTeamId) lastUserTeamId = urlTeam;
    return lastMySlot || 0;
  }

  function overlaySyncText(picks, ok, mySlot) {
    if (window.BRDraftSlot) {
      return window.BRDraftSlot.compactSync("yahoo", (picks || []).length, mySlot, ok);
    }
    return (picks || []).length ? "YAHOO · " + picks.length : "YAHOO · LIVE";
  }

  function feedAssistant(picks, syncText, ok, extra) {
    lastPicks = picks || lastPicks || [];
    const mySlot = resolveMySlot(lastPicks, extra || {});
    const text = overlaySyncText(lastPicks, ok, mySlot);
    if (typeof window.__brDaPushPicks === "function" && lastPicks) {
      window.__brDaPushPicks(
        Object.assign(
          {
            platform: "yahoo",
            picks: lastPicks,
            syncText: text,
            mySlot: mySlot || undefined,
          },
          overlayTeamMeta(lastPicks),
          extra || {}
        )
      );
    }
    if (typeof window.__brDaSetSync === "function") {
      window.__brDaSetSync(!!ok, text);
    }
  }

  function listenFromMain(type, fn) {
    window.addEventListener("message", (ev) => {
      if (!ev.data || ev.data.__br !== BRIDGE || ev.data.type !== type) return;
      fn(ev.data.detail);
    });
  }

  function requestRescan() {
    try {
      window.postMessage({ __br: BRIDGE, type: RESCAN }, "*");
    } catch (_e) {
      /* ignore */
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
      ].join(";")
    );
    chip.innerHTML =
      '<span class="br-chip-text">BR Fantasy · watching Yahoo draft…</span>'
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
    if (typeof window.__brDaSetSync === "function") {
      window.__brDaSetSync(!!ok, overlaySyncText(lastPicks, !!ok, lastMySlot));
    }
  }

  function applyRelayStatus(detail, force) {
    if (!detail) return;
    const count = Number(detail.pickCount || 0);
    if (detail.sent > 0) {
      lastPickCount = count;
      setChip(
        count > 0
          ? "BR Fantasy · connected · " + count + " picks"
          : "BR Fantasy · connected · watching for picks",
        true
      );
      return;
    }
    if (count > 0 || force) {
      setChip(relayFailureText(detail), false);
      if (pendingPayload) scheduleRetry();
    }
  }

  function finishReconnect(resp) {
    if (resp && resp.throttled) {
      setChip("BR Fantasy · wait a few seconds…", false);
      return;
    }
    setTimeout(function () {
      if (pendingPayload) {
        relayPending(true);
        return;
      }
      if (lastPickCount > 0) {
        setChip("BR Fantasy · connected · " + lastPickCount + " picks", true);
        return;
      }
      const hasDraftRoom = resp && resp.br && resp.br.pinged > 0;
      const hasDraftTab = resp && resp.draft && resp.draft.pinged > 0;
      if (!hasDraftRoom) {
        setChip("BR Fantasy · open Draft Room on brfantasyfootball.com", false);
      } else if (!hasDraftTab) {
        setChip("BR Fantasy · reload this Yahoo draft tab", false);
      } else {
        setChip("BR Fantasy · watching draft · no picks detected yet", false);
        requestRescan();
      }
    }, RECONNECT_SETTLE_MS);
  }

  function forceResend() {
    lastDelivered = "";
    clearRetry();
    requestRescan();
    if (pendingPayload) relayPending(true);
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
      chrome.runtime.sendMessage(
        {
          type: "reconnectDraftRelay",
          source: "yahoo-chip",
          leagueId: leagueFromUrl().leagueId,
          season: leagueFromUrl().season,
          platform: "yahoo",
        },
        function (resp) {
          void chrome.runtime.lastError;
          finishReconnect(resp);
        }
      );
    } catch (_e) {
      finishReconnect(null);
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
      last ? last.playerName || last.name || "" : "",
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
      relayPending();
    }, RETRY_MS);
  }

  function relayFailureText(resp) {
    if (resp && resp.reason === "tabs_query_failed") return "BR Fantasy · reload extension";
    if (resp && resp.tabs > 0) return "BR Fantasy · reload Draft Room tab";
    return "BR Fantasy · open Draft Room on brfantasyfootball.com";
  }

  function relayPending(force) {
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
      type: "yahooDraftRelay",
      leagueId: detail.leagueId || ids.leagueId,
      season: detail.season || ids.season,
      inProgress: !!detail.inProgress,
      drafted: !!detail.drafted,
      picks: detail.picks,
      source: detail.source || "yahoo-draft-room",
      at: detail.at || Date.now(),
    };
    if (window.BRDraftSlot && BRDraftSlot.mergeYahooPicks) {
      payload.picks = BRDraftSlot.mergeYahooPicks(lastPicks, payload.picks);
    }
    feedAssistant(payload.picks, "", true, {
      mySlot: detail.mySlot,
      userTeamId: detail.userTeamId,
      rounds: detail.rounds,
      teams: detail.teams,
      inProgress: detail.inProgress,
      drafted: detail.drafted,
      roster: detail.roster,
      sf: detail.sf,
      ppr: detail.ppr,
      tep: detail.tep,
      passTd: detail.passTd,
    });
    if (!payload.leagueId) {
      setChip("BR Fantasy · leagueId missing in URL", false);
      return;
    }
    pendingPayload = payload;
    lastPickCount = Math.max(lastPickCount, payload.picks.length);
    relayPending();
  }

  listenFromMain(EVENT, forward);
  listenFromMain(RELAY_STATUS, function (detail) {
    applyRelayStatus(detail, false);
  });

  document.addEventListener("brfantasy:assistant-reconnect", function () {
    manualReconnect();
  });

  try {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.type === "forceDraftRelay") forceResend();
      if (msg && msg.type === "draftRelayResult") applyRelayStatus(msg, true);
    });
  } catch (_e) {
    /* ignore */
  }

  ensureChip();

  function lastPickLabel(picks) {
    const last = picks && picks.length ? picks[picks.length - 1] : null;
    if (!last) return "";
    return [last.overallPickNumber, last.playerId || "", last.playerName || last.name || ""].join("|");
  }

  function pushMergedPicks(incoming, extra) {
    const helper = window.BRDraftSlot;
    const merged = helper && helper.mergeYahooPicks
      ? helper.mergeYahooPicks(lastPicks, incoming)
      : (incoming && incoming.length >= (lastPicks || []).length ? incoming : lastPicks);
    if (!merged || !merged.length) return false;
    const grew = merged.length > (lastPicks || []).length || lastPickLabel(merged) !== lastPickLabel(lastPicks);
    if (!grew) return false;
    lastPicks = merged;
    feedAssistant(merged, "", true, extra || {});
    const ids = leagueFromUrl();
    if (ids.leagueId) {
      pendingPayload = {
        type: "yahooDraftRelay",
        leagueId: ids.leagueId,
        season: ids.season,
        inProgress: true,
        drafted: false,
        picks: merged,
        source: "yahoo-draft-room",
        at: Date.now(),
      };
      relayPending();
    }
    return true;
  }

  function pollYahooLive() {
    const prev = lastMySlot;
    const slot = resolveMySlot(lastPicks || [], {});
    const extra = { mySlot: slot || lastMySlot, teams: overlayTeamMeta(lastPicks).teams };
    if (window.BRDraftSlot && BRDraftSlot.scrapeYahooBoard) {
      const scraped = BRDraftSlot.scrapeYahooBoard(document, extra.teams);
      if (scraped && scraped.length) pushMergedPicks(scraped, extra);
    }
    if (slot && slot !== prev) feedAssistant(lastPicks || [], "", true, extra);
    const expected = window.BRDraftSlot && BRDraftSlot.completedFromYahooClock
      ? BRDraftSlot.completedFromYahooClock(extra.teams || 12)
      : 0;
    if (expected > (lastPicks || []).length) requestRescan();
  }
  pollYahooLive();
  setTimeout(pollYahooLive, 500);
  setInterval(pollYahooLive, 1000);
  let livePollTimer = null;
  try {
    const mo = new MutationObserver(function () {
      if (document.hidden || livePollTimer) return;
      livePollTimer = setTimeout(function () {
        livePollTimer = null;
        pollYahooLive();
      }, 350);
    });
    mo.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  } catch (_e) {
    /* ignore */
  }
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
