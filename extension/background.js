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

async function deliverRelayToTab(tab, messageType, payload) {
  if (!tab || !tab.id) return false;
  const msg = { type: messageType, payload };
  try {
    await chrome.tabs.sendMessage(tab.id, msg);
    return true;
  } catch (_e) {
    // Content script missing — common when Draft Room was opened before the
    // extension loaded. Inject a one-shot MAIN-world dispatch instead.
  }
  const eventName = relayEventName(messageType);
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      world: "MAIN",
      func: (evt, detail) => {
        window.dispatchEvent(new CustomEvent(evt, { detail: detail }));
      },
      args: [eventName, payload],
    });
    return true;
  } catch (_e2) {
    return false;
  }
}

async function relayDraftToBrTabs(messageType, payload) {
  let tabs = [];
  try {
    // host_permissions already cover these URLs — no broad `tabs` permission.
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

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || typeof msg !== "object") return false;

  if (msg.type === "getEspnCookies") {
    getEspnCreds()
      .then(sendResponse)
      .catch(() => sendResponse({ swid: "", espn_s2: "" }));
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
