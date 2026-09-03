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

const SLEEPER_HOST_RE = /(^|\.)sleeper\.(com|app)$/i;

function isSleeperDraftTabUrl(url) {
  try {
    const u = new URL(String(url || ""));
    if (!SLEEPER_HOST_RE.test(u.hostname)) return false;
    const blob = (u.pathname + "\n" + u.hash + "\n" + u.search).toLowerCase();
    if (/[?&]draft_id=/.test(blob)) return true;
    if (/\/draft\/[a-z0-9]+/.test(blob)) return true;
    if (/\/leagues\/\d{6,20}\/draft(?:\/|$|\?|#)/.test(blob.replace(/\n/g, ""))) return true;
    return false;
  } catch (_e) {
    return false;
  }
}

async function ensureSleeperDraftAssistant(tabId) {
  if (!tabId) return false;
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      world: "MAIN",
      files: ["sleeper_draft_main.js"],
    });
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["draft_slot.js", "assistant_inject.js", "sleeper_draft.js"],
    });
    return true;
  } catch (_e) {
    return false;
  }
}

async function openDraftAssistantOnTab(tabId) {
  if (!tabId) return { ok: false, message: "No active tab." };
  let tab = null;
  try {
    tab = await chrome.tabs.get(tabId);
  } catch (_e) {
    return { ok: false, message: "No active tab." };
  }
  const url = String((tab && tab.url) || "");
  if (isSleeperDraftTabUrl(url)) {
    const injected = await ensureSleeperDraftAssistant(tabId);
    if (!injected) {
      return { ok: false, message: "Could not attach to this Sleeper tab. Reload the draft page." };
    }
  }
  try {
    await chrome.tabs.sendMessage(tabId, { type: "openDraftAssistant" });
    return { ok: true };
  } catch (_e) {
    return { ok: false, message: "Open a Sleeper, Yahoo, or ESPN draft tab first." };
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const url = changeInfo.url || (tab && tab.url) || "";
  if (!url || !isSleeperDraftTabUrl(url)) return;
  if (changeInfo.url || changeInfo.status === "complete") {
    void ensureSleeperDraftAssistant(tabId);
  }
});

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
      message: "Reconnect already sent - wait a few seconds",
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

const BR_API_HOSTS = [
  "https://www.brfantasyfootball.com",
  "https://brfantasyfootball.com",
];
const POOL_TTL_MS = 5 * 60 * 1000;
/** @type {{key:string, at:number, players:object[]}|null} */
let draftPoolCache = null;

function poolNum(v) {
  const n = Number(v);
  return isFinite(n) ? n : null;
}

function normDraftPos(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "PK") return "K";
  if (p === "DST" || p === "D/ST" || p === "D-ST" || p === "D ST") return "DEF";
  return p;
}

function compactDraftPlayer(raw, scoringType, isSf, teams, adpSource) {
  if (!raw || raw.id == null) return null;
  const pos = normDraftPos(raw.position || raw.pos);
  const isKd = pos === "K" || pos === "DEF";
  if (pos !== "QB" && pos !== "RB" && pos !== "WR" && pos !== "TE" && !isKd) return null;
  const by = raw.adp_by_source && typeof raw.adp_by_source === "object" ? raw.adp_by_source : {};
  const src = (adpSource && by[adpSource]) || {};
  const cons = by.consensus || {};
  const sleeper = by.sleeper || {};
  const size = Number(teams) >= 8 ? Number(teams) : 12;
  const redraft = scoringType !== "dynasty" && scoringType !== "rookie";
  let adp;
  let val;
  if (redraft) {
    adp =
      poolNum(isSf ? src.sf_redraft_avg_pick : src.redraft_avg_pick) ||
      poolNum(isSf ? raw.sf_redraft_avg_pick : raw.redraft_avg_pick) ||
      poolNum(isSf ? cons.sf_redraft_avg_pick : cons.redraft_avg_pick) ||
      poolNum(isSf ? sleeper.sf_redraft_avg_pick : sleeper.redraft_avg_pick);
    if (isSf) {
      val =
        (size !== 10 ? poolNum(raw["redraft_sf_value_" + size]) : null) ||
        poolNum(raw.redraft_value_sf) ||
        poolNum(raw.redraft_value_1qb);
    } else {
      val =
        (size !== 10 ? poolNum(raw["redraft_value_" + size]) : null) ||
        poolNum(raw.redraft_value_1qb);
    }
  } else {
    adp =
      poolNum(isSf ? src.sf_avg_pick : src.avg_pick) ||
      poolNum(isSf ? raw.sf_avg_pick : raw.avg_pick) ||
      poolNum(isSf ? cons.sf_avg_pick : cons.avg_pick);
    val = poolNum(isSf ? raw.sf_value : raw.value) || poolNum(raw.value);
  }
  if (!(val > 0) && !(adp > 0) && !isKd) return null;
  const adpN = adp && adp > 0 ? adp : 999;
  let tier = 6;
  if (adpN <= 12) tier = 1;
  else if (adpN <= 24) tier = 2;
  else if (adpN <= 48) tier = 3;
  else if (adpN <= 84) tier = 4;
  else if (adpN <= 120) tier = 5;
  const id = String(raw.id);
  let headshot = String(raw.espnHeadshot || "").trim();
  if (!headshot && /^\d+$/.test(id)) {
    headshot = "https://sleepercdn.com/content/nfl/players/" + id + ".jpg";
  }
  const vorp = isSf
    ? (poolNum(raw["sf_vorp_" + size]) || poolNum(raw.sf_vorp) || poolNum(raw.vorp))
    : (poolNum(raw["vorp_" + size]) || poolNum(raw.vorp));
  const market = isSf
    ? (poolNum(raw.sf_market_vs_adp) || poolNum(raw.market_vs_adp))
    : (poolNum(raw.market_vs_adp_1qb) || poolNum(raw.market_vs_adp));
  const inj = String(raw.injury || raw.injury_status || "").trim();
  const yearsExp = poolNum(raw.years_exp);
  return {
    id: id,
    name: String(raw.name || ""),
    pos: pos,
    team: String(raw.team || "").toUpperCase(),
    age: poolNum(raw.age) || 0,
    bye: poolNum(raw.bye_week) || poolNum(raw.bye) || 0,
    adp: adpN,
    val: Math.round(val > 0 ? val : 0),
    ppg: poolNum(raw.proj_ppg) || 0,
    proj_ppg: poolNum(raw.proj_ppg),
    proj_pts: poolNum(raw.proj_pts),
    last_ppg: poolNum(raw.ppg),
    ppg_season: raw.ppg_season != null ? String(raw.ppg_season) : "",
    vorp: vorp,
    market: market,
    years_exp: yearsExp,
    is_rookie: raw.is_rookie === true || yearsExp === 0,
    injury: inj && !/^(active|act)$/i.test(inj) ? inj : "",
    headshot: headshot,
    tier: tier,
    rank_change_7d: poolNum(raw.rank_change_7d),
    breakout_score: poolNum(raw.breakout_score),
    projected_role: raw.projected_role ? String(raw.projected_role) : "",
    bye_week: poolNum(raw.bye_week) || poolNum(raw.bye) || 0,
  };
}

function adpOptionsFromBody(body, scoringType) {
  if (!body || Array.isArray(body)) return [];
  const opts = body.adp_source_options || {};
  const key = scoringType === "dynasty" ? "startup" : scoringType;
  const list = opts[key] || opts.redraft || [];
  if (!Array.isArray(list)) return [];
  return list.filter(function (o) { return o && o.value; }).map(function (o) {
    return { value: String(o.value), label: String(o.label || o.value) };
  });
}

async function fetchDraftPool(opts) {
  const scoringType = String((opts && opts.scoringType) || "redraft").toLowerCase();
  const sf = !!(opts && opts.sf);
  const adpSource = String((opts && opts.adpSource) || "consensus").toLowerCase();
  const teams = Number((opts && opts.teams) || 12) || 12;
  const ppr = (opts && opts.ppr != null) ? Number(opts.ppr) : 1;
  const tep = (opts && opts.tep != null) ? Number(opts.tep) : 0;
  const passTd = (opts && opts.passTd != null) ? Number(opts.passTd) : 4;
  const kdef = opts.kdef !== false && scoringType !== "rookie";
  const key = [scoringType, sf ? "sf" : "1qb", adpSource, teams || "", ppr, tep, passTd, kdef ? "kdef" : "nokd"].join("|");
  const now = Date.now();
  if (!opts.force && draftPoolCache && draftPoolCache.key === key && now - draftPoolCache.at < POOL_TTL_MS) {
    return {
      ok: true,
      players: draftPoolCache.players,
      scoringType: scoringType,
      sf: sf,
      adpSource: adpSource,
      adpOptions: draftPoolCache.adpOptions || [],
      cached: true,
    };
  }
  const params = [
    "adp_source=" + encodeURIComponent(adpSource),
    "scoring_type=" + encodeURIComponent(scoringType === "startup" ? "dynasty" : scoringType),
    "league_type=" + (sf ? "sf" : "1qb"),
    "proj_rec=" + encodeURIComponent(String((opts && opts.ppr != null) ? opts.ppr : 1)),
    "proj_te_bonus=" + encodeURIComponent(String((opts && opts.tep != null) ? opts.tep : 0)),
    "proj_pass_td=" + encodeURIComponent(String((opts && opts.passTd != null) ? opts.passTd : 4)),
  ];
  if (teams >= 8) params.push("league_size=" + encodeURIComponent(String(teams)));
  if (kdef) params.push("kdef=1");
  const path = "/api/league-players?" + params.join("&");
  let lastErr = "fetch failed";
  for (const host of BR_API_HOSTS) {
    try {
      const res = await fetch(host + path);
      if (!res.ok) {
        lastErr = "HTTP " + res.status;
        continue;
      }
      const body = await res.json();
      const raw = Array.isArray(body) ? body : body.players || [];
      const players = raw.map(function (p) {
        return compactDraftPlayer(p, scoringType, sf, teams, adpSource);
      }).filter(Boolean);
      if (!players.length) {
        lastErr = "empty pool";
        continue;
      }
      const adpOptions = adpOptionsFromBody(body, scoringType);
      draftPoolCache = { key: key, at: now, players: players, adpOptions: adpOptions };
      return {
        ok: true,
        players: players,
        scoringType: scoringType,
        sf: sf,
        adpSource: adpSource,
        adpOptions: adpOptions,
        cached: false,
      };
    } catch (err) {
      lastErr = String(err && err.message ? err.message : err);
    }
  }
  if (draftPoolCache && draftPoolCache.players && draftPoolCache.players.length) {
    return {
      ok: true,
      players: draftPoolCache.players,
      scoringType: scoringType,
      sf: sf,
      adpSource: adpSource,
      adpOptions: draftPoolCache.adpOptions || [],
      cached: true,
      stale: true,
    };
  }
  return { ok: false, players: [], error: lastErr, scoringType: scoringType, sf: sf, adpSource: adpSource, adpOptions: [] };
}

async function fetchDraftPlayoffOdds(opts) {
  const body = {
    season: Number((opts && opts.season) || 0) || 0,
    ppr: (opts && opts.ppr != null) ? Number(opts.ppr) : 1,
    tep: (opts && opts.tep != null) ? Number(opts.tep) : 0,
    pass_td: (opts && opts.passTd != null) ? Number(opts.passTd) : 4,
    roster: (opts && opts.roster) || {},
    playoff_teams: Number((opts && opts.playoffTeams) || 6) || 6,
    platform: String((opts && opts.platform) || "sleeper"),
    league_id: String((opts && opts.leagueId) || ""),
    use_league: opts && opts.useLeague !== false,
    viewer_slot: (opts && opts.viewerSlot) || null,
    teams: (opts && opts.teams) || [],
  };
  let lastErr = "fetch failed";
  for (const host of BR_API_HOSTS) {
    try {
      const res = await fetch(host + "/api/draft-playoff-odds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        lastErr = "HTTP " + res.status;
        continue;
      }
      const json = await res.json();
      if (json && Array.isArray(json.odds) && json.odds.length) {
        return { ok: true, odds: json.odds, source: json.source || "", playoffTeams: json.playoff_teams };
      }
      lastErr = (json && json.error) || "empty odds";
    } catch (err) {
      lastErr = String(err && err.message ? err.message : err);
    }
  }
  return { ok: false, odds: [], error: lastErr };
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

  if (msg.type === "openDraftAssistantOnTab") {
    const tabId = Number(msg.tabId || (sender && sender.tab && sender.tab.id) || 0);
    openDraftAssistantOnTab(tabId)
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, message: "Open a Sleeper, Yahoo, or ESPN draft tab first." }));
    return true;
  }

  if (msg.type === "fetchDraftPool") {
    fetchDraftPool({
      scoringType: String(msg.scoringType || "redraft"),
      sf: !!msg.sf,
      adpSource: String(msg.adpSource || "consensus"),
      teams: Number(msg.teams || 0),
      force: !!msg.force,
      kdef: msg.kdef !== false,
      ppr: msg.ppr,
      tep: msg.tep,
      passTd: msg.passTd,
    })
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, players: [] }));
    return true;
  }

  if (msg.type === "fetchDraftPlayoffOdds") {
    fetchDraftPlayoffOdds(msg)
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false, odds: [] }));
    return true;
  }

  return false;
});
