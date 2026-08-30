// Service worker: ESPN cookies for league connect, plus live-draft pick relay
// from an open ESPN or Yahoo draft room tab to BR Fantasy Draft Room tabs.
// Nothing here submits picks to ESPN or Yahoo. Cookies are only returned to the
// BR tab that asked for them (ESPN connect flow).

const ESPN_URLS = [
  "https://www.espn.com",
  "https://fantasy.espn.com",
  "https://espn.com",
];

const BR_TAB_URLS = [
  "https://www.brfantasyfootball.com/*",
  "https://brfantasyfootball.com/*",
  "http://localhost/*",
  "http://127.0.0.1/*",
];

const ESPN_DRAFT_TAB_URLS = [
  "https://fantasy.espn.com/football/draft*",
  "https://fantasy.espn.com/*/football/draft*",
  "https://fantasy.espn.com/*draft*",
];

const YAHOO_DRAFT_TAB_URLS = [
  "https://football.fantasysports.yahoo.com/f1/*/draft*",
  "https://football.fantasysports.yahoo.com/f1/*/livedraft*",
  "https://football.fantasysports.yahoo.com/draftclient*",
  "https://*.fantasysports.yahoo.com/*/draft*",
  "https://sports.yahoo.com/fantasy/*/draft*",
];

const RECONNECT_COOLDOWN_MS = 5000;
const SESSION_TABS_KEY = "brDraftRoomTabs";
let lastReconnectAt = 0;
/** @type {Map<number, {href:string, platform:string, season:string, leagueId:string, at:number}>} */
const brDraftRoomTabs = new Map();
let tabsLoaded = false;

async function loadBrDraftRoomTabs() {
  if (tabsLoaded) return;
  tabsLoaded = true;
  try {
    const data = await chrome.storage.session.get(SESSION_TABS_KEY);
    const entries = data[SESSION_TABS_KEY];
    if (!Array.isArray(entries)) return;
    for (const row of entries) {
      if (!Array.isArray(row) || row.length < 2) continue;
      const tabId = Number(row[0]);
      const meta = row[1];
      if (tabId && meta && typeof meta === "object") brDraftRoomTabs.set(tabId, meta);
    }
  } catch (_e) {
    /* ignore */
  }
}

async function persistBrDraftRoomTabs() {
  try {
    await chrome.storage.session.set({
      [SESSION_TABS_KEY]: Array.from(brDraftRoomTabs.entries()),
    });
  } catch (_e) {
    /* ignore */
  }
}

async function pruneStaleDraftRoomTabs() {
  let changed = false;
  for (const tabId of brDraftRoomTabs.keys()) {
    try {
      await chrome.tabs.get(tabId);
    } catch (_e) {
      brDraftRoomTabs.delete(tabId);
      changed = true;
    }
  }
  if (changed) await persistBrDraftRoomTabs();
}


function parseDraftRoomHref(href) {
  try {
    const u = new URL(String(href || ""));
    const m = u.pathname.match(/^\/(espn|yahoo|sleeper)\/(\d{4})\/([^/]+)\/draft\b/i);
    if (m) {
      return { platform: m[1].toLowerCase(), season: m[2], leagueId: m[3] };
    }
  } catch (_e) {
    /* ignore */
  }
  return { platform: "", season: "", leagueId: "" };
}

function registerBrDraftRoomTab(tabId, meta) {
  if (!tabId) return;
  brDraftRoomTabs.set(Number(tabId), {
    href: String((meta && meta.href) || ""),
    platform: String((meta && meta.platform) || ""),
    season: String((meta && meta.season) || ""),
    leagueId: String((meta && meta.leagueId) || ""),
    at: Date.now(),
  });
  void persistBrDraftRoomTabs();
}

function leagueIdsMatch(a, b) {
  const x = String(a || "").trim();
  const y = String(b || "").trim();
  if (!x || !y) return true;
  if (x === y) return true;
  const xn = x.replace(/\D/g, "");
  const yn = y.replace(/\D/g, "");
  return !!(xn && yn && xn === yn);
}

chrome.tabs.onRemoved.addListener((tabId) => {
  if (!brDraftRoomTabs.delete(Number(tabId))) return;
  void persistBrDraftRoomTabs();
});

async function queryBrDraftRoomTabs() {
  await loadBrDraftRoomTabs();
  await pruneStaleDraftRoomTabs();
  const seen = new Set();
  const out = [];

  for (const [tabId, meta] of brDraftRoomTabs.entries()) {
    if (seen.has(tabId)) continue;
    seen.add(tabId);
    out.push({ id: tabId, url: meta.href, ...meta });
  }

  try {
    const queried = await chrome.tabs.query({ url: BR_TAB_URLS });
    for (const tab of queried) {
      if (!tab || !tab.id || seen.has(tab.id)) continue;
      seen.add(tab.id);
      const parsed = parseDraftRoomHref(tab.url || "");
      out.push({ ...tab, ...parsed });
      registerBrDraftRoomTab(tab.id, {
        href: tab.url || "",
        platform: parsed.platform,
        season: parsed.season,
        leagueId: parsed.leagueId,
      });
    }
  } catch (_e) {
    if (!out.length) return [];
  }

  return out;
}

function filterTabsForRelay(tabs, payload) {
  const lid = String((payload && payload.leagueId) || "");
  const draftTabs = tabs.filter((tab) => {
    const href = tab.url || tab.href || "";
    return /\/draft\b/i.test(href) || (tab.leagueId && tab.platform);
  });
  const pool = draftTabs.length ? draftTabs : tabs;
  if (!lid) return pool;
  const matched = pool.filter((tab) => leagueIdsMatch(tab.leagueId, lid));
  return matched.length ? matched : pool;
}

function relayEventName(messageType) {
  return messageType === "yahooDraftRelay"
    ? "brfantasy:yahoo-draft-relay"
    : "brfantasy:espn-draft-relay";
}

async function readCookie(name) {
  for (const url of ESPN_URLS) {
    try {
      const cookie = await chrome.cookies.get({ url, name });
      if (cookie && cookie.value) return cookie.value;
    } catch (_e) {
      // keep trying the next host
    }
  }
  return "";
}

async function getEspnCreds() {
  const [swid, espn_s2] = await Promise.all([
    readCookie("SWID"),
    readCookie("espn_s2"),
  ]);
  return { swid, espn_s2 };
}

// Content scripts run in an isolated JS world; Draft Room listens in the page
// (MAIN) world. Always inject a MAIN-world document event so the board hears it.
async function injectPageEvent(tabId, eventName, detail) {
  if (!tabId || !eventName) return false;
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      world: "MAIN",
      func: (evt, data) => {
        document.dispatchEvent(
          new CustomEvent(evt, { detail: data, bubbles: true, composed: true })
        );
        if (!data || !Array.isArray(data.picks) || !data.picks.length) return;
        var path =
          evt.indexOf("yahoo") >= 0 ? "/api/draft/yahoo-relay" : "/api/draft/espn-relay";
        fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "same-origin",
          cache: "no-store",
          body: JSON.stringify({
            leagueId: data.leagueId,
            season: data.season,
            inProgress: data.inProgress !== false,
            drafted: !!data.drafted,
            picks: data.picks,
            source: data.source || "extension-inject",
            forceReplay: true,
          }),
        }).catch(function () {});
      },
      args: [eventName, detail || {}],
    });
    return true;
  } catch (_e) {
    return false;
  }
}

async function deliverRelayToTab(tab, messageType, payload) {
  if (!tab || !tab.id) return false;
  const eventName = relayEventName(messageType);
  if (await injectPageEvent(tab.id, eventName, payload)) return true;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: messageType, payload });
    return true;
  } catch (_e) {
    return false;
  }
}

async function deliverReconnectToBrTab(tab, detail) {
  if (!tab || !tab.id) return false;
  if (await injectPageEvent(tab.id, "brfantasy:extension-reconnect", detail || {})) {
    return true;
  }
  try {
    await chrome.tabs.sendMessage(tab.id, {
      type: "brDraftRoomReconnect",
      detail: detail || {},
    });
    return true;
  } catch (_e) {
    return false;
  }
}

async function relayDraftToBrTabs(messageType, payload) {
  let tabs = [];
  try {
    tabs = filterTabsForRelay(await queryBrDraftRoomTabs(), payload);
  } catch (_e) {
    return { ok: false, reason: "tabs_query_failed", sent: 0, tabs: 0 };
  }
  let sent = 0;
  await Promise.all(
    tabs.map(async (tab) => {
      if (await deliverRelayToTab(tab, messageType, payload)) sent += 1;
    })
  );
  return { ok: true, sent, tabs: tabs.length, registered: brDraftRoomTabs.size };
}

async function ensureEspnDraftObserver(tabId) {
  if (!tabId) return false;
  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      world: "MAIN",
      files: ["espn_draft_main.js"],
    });
    return true;
  } catch (_e) {
    return false;
  }
}

async function nudgeDraftTabScan(tab) {
  if (!tab || !tab.id) return false;
  let ok = false;
  try {
    await chrome.tabs.sendMessage(tab.id, { type: "forceDraftRelay" });
    ok = true;
  } catch (_e) {
    /* isolated bridge may not be ready yet */
  }
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      world: "MAIN",
      func: () => {
        window.postMessage({ __br: "brfantasy-bridge-v1", type: "brfantasy:draft-rescan" }, "*");
        if (typeof window.__brFantasyEspnForceScan === "function") window.__brFantasyEspnForceScan();
        if (typeof window.__brFantasyYahooForceScan === "function") window.__brFantasyYahooForceScan();
      },
    });
    ok = true;
  } catch (_e) {
    /* ignore */
  }
  return ok;
}

async function notifyDraftTabRelayResult(tabId, pickCount, result) {
  if (!tabId) return;
  try {
    await chrome.tabs.sendMessage(tabId, {
      type: "draftRelayResult",
      sent: result && result.sent,
      tabs: result && result.tabs,
      pickCount: pickCount || 0,
      reason: result && result.reason,
    });
  } catch (_e) {
    /* chip script not ready */
  }
}

async function pingDraftTabs() {
  let espnTabs = [];
  let yahooTabs = [];
  try {
    espnTabs = await chrome.tabs.query({ url: ESPN_DRAFT_TAB_URLS });
    yahooTabs = await chrome.tabs.query({ url: YAHOO_DRAFT_TAB_URLS });
  } catch (_e) {
    return { ok: false, espn: 0, yahoo: 0, pinged: 0 };
  }
  let pinged = 0;
  const tabs = [...espnTabs, ...yahooTabs];
  await Promise.all(
    tabs.map(async (tab) => {
      if (await nudgeDraftTabScan(tab)) pinged += 1;
    })
  );
  return { ok: true, espn: espnTabs.length, yahoo: yahooTabs.length, pinged };
}

async function pingBrDraftRooms(detail) {
  let tabs = [];
  try {
    tabs = filterTabsForRelay(await queryBrDraftRoomTabs(), detail);
  } catch (_e) {
    return { ok: false, tabs: 0, pinged: 0 };
  }
  let pinged = 0;
  await Promise.all(
    tabs.map(async (tab) => {
      if (await deliverReconnectToBrTab(tab, detail)) pinged += 1;
    })
  );
  return { ok: true, tabs: tabs.length, pinged, registered: brDraftRoomTabs.size };
}

async function reconnectDraftRelay(detail) {
  const now = Date.now();
  if (now - lastReconnectAt < RECONNECT_COOLDOWN_MS) {
    return {
      ok: true,
      throttled: true,
      draft: { pinged: 0 },
      br: { pinged: 0 },
      message: "Reconnect already sent — wait a few seconds",
    };
  }
  lastReconnectAt = now;
  const payload = detail && typeof detail === "object" ? detail : {};
  // Draft tab first (resend full pick snapshot), then Draft Room (apply/cache).
  const draft = await pingDraftTabs();
  await new Promise((resolve) => setTimeout(resolve, 350));
  const br = await pingBrDraftRooms(payload);
  return {
    ok: true,
    draft,
    br,
    message:
      br.pinged > 0 || draft.pinged > 0
        ? "Reconnect sent to open tabs"
        : "No open Draft Room or draft tabs found",
  };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || typeof msg !== "object") return false;

  if (msg.type === "getEspnCookies") {
    getEspnCreds()
      .then(sendResponse)
      .catch(() => sendResponse({ swid: "", espn_s2: "" }));
    return true;
  }

  if (msg.type === "reconnectDraftRelay") {
    reconnectDraftRelay({
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      platform: String(msg.platform || ""),
      source: String(msg.source || "manual"),
    })
      .then(sendResponse)
      .catch(() =>
        sendResponse({
          ok: false,
          draft: { pinged: 0 },
          br: { pinged: 0 },
          message: "Reconnect failed",
        })
      );
    return true;
  }

  if (msg.type === "espnDraftRelay") {
    const pickCount = Array.isArray(msg.picks) ? msg.picks.length : 0;
    relayDraftToBrTabs("espnDraftRelay", {
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      inProgress: !!msg.inProgress,
      drafted: !!msg.drafted,
      picks: Array.isArray(msg.picks) ? msg.picks : [],
      source: msg.source || "espn-draft-room",
      at: msg.at || Date.now(),
      forceReplay: true,
    })
      .then((result) => {
        notifyDraftTabRelayResult(sender && sender.tab && sender.tab.id, pickCount, result);
        sendResponse(result);
      })
      .catch(() => sendResponse({ ok: false, sent: 0, tabs: 0 }));
    return true;
  }

  if (msg.type === "yahooDraftRelay") {
    const pickCount = Array.isArray(msg.picks) ? msg.picks.length : 0;
    relayDraftToBrTabs("yahooDraftRelay", {
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      inProgress: !!msg.inProgress,
      drafted: !!msg.drafted,
      picks: Array.isArray(msg.picks) ? msg.picks : [],
      source: msg.source || "yahoo-draft-room",
      at: msg.at || Date.now(),
      forceReplay: true,
    })
      .then((result) => {
        notifyDraftTabRelayResult(sender && sender.tab && sender.tab.id, pickCount, result);
        sendResponse(result);
      })
      .catch(() => sendResponse({ ok: false, sent: 0, tabs: 0 }));
    return true;
  }

  if (msg.type === "ensureEspnDraftObserver") {
    const tabId = sender && sender.tab && sender.tab.id;
    ensureEspnDraftObserver(tabId)
      .then((ok) => sendResponse({ ok }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }

  if (msg.type === "espnDraftTabReady") {
    const tabId = sender && sender.tab && sender.tab.id;
    ensureEspnDraftObserver(tabId).finally(() => sendResponse({ ok: true }));
    return true;
  }

  if (msg.type === "yahooDraftTabReady") {
    sendResponse({ ok: true });
    return false;
  }

  if (msg.type === "brDraftRoomReady") {
    const tabId = sender && sender.tab && sender.tab.id;
    const href = String(msg.href || (sender.tab && sender.tab.url) || "");
    const parsed = parseDraftRoomHref(href);
    registerBrDraftRoomTab(tabId, {
      href,
      platform: String(msg.platform || parsed.platform || ""),
      season: String(msg.season || parsed.season || ""),
      leagueId: String(msg.leagueId || parsed.leagueId || ""),
    });
    sendResponse({ ok: true, registered: brDraftRoomTabs.size });
    return false;
  }

  return false;
});
