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
];

const YAHOO_DRAFT_TAB_URLS = [
  "https://football.fantasysports.yahoo.com/f1/*/draft*",
  "https://football.fantasysports.yahoo.com/f1/*/livedraft*",
  "https://football.fantasysports.yahoo.com/draftclient*",
  "https://*.fantasysports.yahoo.com/*/draft*",
  "https://sports.yahoo.com/fantasy/*/draft*",
];

const RECONNECT_COOLDOWN_MS = 5000;
let lastReconnectAt = 0;

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
    tabs = await chrome.tabs.query({ url: BR_TAB_URLS });
  } catch (_e) {
    return { ok: false, reason: "tabs_query_failed", sent: 0, tabs: 0 };
  }
  let sent = 0;
  await Promise.all(
    tabs.map(async (tab) => {
      if (await deliverRelayToTab(tab, messageType, payload)) sent += 1;
    })
  );
  return { ok: true, sent, tabs: tabs.length };
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
      if (!tab || !tab.id) return;
      try {
        await chrome.tabs.sendMessage(tab.id, { type: "forceDraftRelay" });
        pinged += 1;
      } catch (_e) {
        /* draft tab has no bridge yet */
      }
    })
  );
  return { ok: true, espn: espnTabs.length, yahoo: yahooTabs.length, pinged };
}

async function pingBrDraftRooms(detail) {
  let tabs = [];
  try {
    tabs = await chrome.tabs.query({ url: BR_TAB_URLS });
  } catch (_e) {
    return { ok: false, tabs: 0, pinged: 0 };
  }
  let pinged = 0;
  await Promise.all(
    tabs.map(async (tab) => {
      if (await deliverReconnectToBrTab(tab, detail)) pinged += 1;
    })
  );
  return { ok: true, tabs: tabs.length, pinged };
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

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
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
    relayDraftToBrTabs("espnDraftRelay", {
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      inProgress: !!msg.inProgress,
      drafted: !!msg.drafted,
      picks: Array.isArray(msg.picks) ? msg.picks : [],
      source: msg.source || "espn-draft-room",
      at: msg.at || Date.now(),
    })
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, sent: 0, tabs: 0 }));
    return true;
  }

  if (msg.type === "yahooDraftRelay") {
    relayDraftToBrTabs("yahooDraftRelay", {
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      inProgress: !!msg.inProgress,
      drafted: !!msg.drafted,
      picks: Array.isArray(msg.picks) ? msg.picks : [],
      source: msg.source || "yahoo-draft-room",
      at: msg.at || Date.now(),
    })
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, sent: 0, tabs: 0 }));
    return true;
  }

  if (
    msg.type === "espnDraftTabReady" ||
    msg.type === "yahooDraftTabReady" ||
    msg.type === "brDraftRoomReady"
  ) {
    sendResponse({ ok: true });
    return false;
  }

  return false;
});
