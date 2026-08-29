// Service worker: ESPN cookies for league connect, plus live-draft pick relay
// from an open ESPN draft room tab to BR Fantasy Draft Room tabs.
// Nothing here submits picks to ESPN. Cookies are only returned to the BR tab
// that asked for them (connect flow).

const ESPN_URLS = [
  "https://www.espn.com",
  "https://fantasy.espn.com",
  "https://espn.com",
];

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

async function relayEspnDraft(payload) {
  let tabs = [];
  try {
    // host_permissions already cover these URLs — no broad `tabs` permission.
    tabs = await chrome.tabs.query({
      url: [
        "https://www.brfantasyfootball.com/*",
        "https://brfantasyfootball.com/*",
        "http://localhost/*",
        "http://127.0.0.1/*",
      ],
    });
  } catch (_e) {
    return { ok: false, reason: "tabs_query_failed" };
  }
  let sent = 0;
  await Promise.all(
    tabs.map(async (tab) => {
      if (!tab || !tab.id) return;
      try {
        await chrome.tabs.sendMessage(tab.id, {
          type: "espnDraftRelay",
          payload,
        });
        sent += 1;
      } catch (_e) {
        // Tab has no content script (or is mid-load) — skip.
      }
    })
  );
  return { ok: true, sent };
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
    relayEspnDraft({
      leagueId: String(msg.leagueId || ""),
      season: String(msg.season || ""),
      inProgress: !!msg.inProgress,
      drafted: !!msg.drafted,
      picks: Array.isArray(msg.picks) ? msg.picks : [],
      source: msg.source || "espn-draft-room",
      at: msg.at || Date.now(),
    })
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, sent: 0 }));
    return true;
  }

  if (msg.type === "espnDraftTabReady") {
    sendResponse({ ok: true });
    return false;
  }

  return false;
});
