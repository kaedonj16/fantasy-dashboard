// Isolated-world content script on Sleeper draft rooms. Polls the public
// Sleeper draft API (no pick submission) and feeds the docked overlay.

(function () {
  "use strict";

  if (window.__brFantasySleeperDraft) return;
  window.__brFantasySleeperDraft = true;

  const POLL_PICKS_MS = 350;
  const POLL_META_MS = 2500;
  const POLL_IDLE_MS = 2000;
  let lastFp = "";
  let lastCount = 0;
  let lastMySlot = 0;
  let lastStatus = "";
  let pickTimer = null;
  let metaTimer = null;
  let pickInFlight = false;
  let pickQueued = false;
  let cachedLeagueId = "";
  let cachedScoring = { ppr: 1, tep: 0, passTd: 4 };
  let cachedLeague = null;
  let cachedUser = { username: "", userId: "" };
  let mainIdentity = { userIds: [], username: "", displayName: "", teamName: "" };
  let cachedOwnerMap = { leagueId: "", map: {} };
  let cachedUsers = { leagueId: "", rows: [] };
  let cachedTrades = { draftId: "", rows: [], at: 0 };
  let cachedDraft = null;
  let cachedPicks = [];
  let lastClockSent = "";

  async function leagueScoring(leagueId) {
    if (!leagueId) return cachedScoring;
    if (cachedLeagueId === String(leagueId) && cachedLeague) return cachedScoring;
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/league/" + encodeURIComponent(leagueId),
        { cache: "no-store" }
      );
      const league = res.ok ? await res.json() : null;
      if (league && typeof league === "object") cachedLeague = league;
      const src = (league && league.scoring_settings) || {};
      cachedScoring = window.BRDraftSlot && BRDraftSlot.scoringFromSleeperSettings
        ? BRDraftSlot.scoringFromSleeperSettings(src)
        : {
            ppr: Number(src.rec != null ? src.rec : 1),
            tep: Number(src.bonus_rec_te || 0),
            passTd: Number(src.pass_td != null ? src.pass_td : 4),
          };
      cachedLeagueId = String(leagueId);
    } catch (_e) {
      /* keep last scoring */
    }
    return cachedScoring;
  }

  let cachedLeagueDraftId = "";

  function draftIdFromUrl() {
    if (window.BRDraftSlot && typeof window.BRDraftSlot.sleeperDraftIdFromUrl === "function") {
      const id = window.BRDraftSlot.sleeperDraftIdFromUrl();
      if (id) return id;
    }
    try {
      const blob = String(location.pathname || "") + String(location.hash || "") + String(location.search || "");
      const m = blob.match(/\/draft\/(?:nfl\/|nba\/|ncaaf\/|cbb\/|epl\/)?([a-zA-Z0-9]+)/i);
      if (m) return m[1];
      const q = new URLSearchParams(location.search || "").get("draft_id")
        || new URLSearchParams(String(location.hash || "").replace(/^#/, "").split("?")[1] || "").get("draft_id");
      return q && /^[a-zA-Z0-9]+$/.test(q) ? q : "";
    } catch (_e) {
      return "";
    }
  }

  function leagueIdFromUrl() {
    if (window.BRDraftSlot && typeof window.BRDraftSlot.sleeperLeagueIdFromUrl === "function") {
      return window.BRDraftSlot.sleeperLeagueIdFromUrl() || "";
    }
    try {
      const m = (String(location.pathname || "") + String(location.hash || "")).match(/\/leagues\/(\d{6,20})/i);
      return m ? m[1] : "";
    } catch (_e) {
      return "";
    }
  }

  function isSleeperDraftRoom() {
    if (window.BRDraftSlot && typeof window.BRDraftSlot.isSleeperDraftRoom === "function") {
      return window.BRDraftSlot.isSleeperDraftRoom();
    }
    return !!(draftIdFromUrl() || /\/leagues\/\d{6,20}\/draft/i.test(location.pathname + location.hash));
  }

  async function resolveDraftId() {
    const fromUrl = draftIdFromUrl();
    if (fromUrl) return fromUrl;
    if (cachedLeagueDraftId) return cachedLeagueDraftId;
    const leagueId = leagueIdFromUrl();
    if (!leagueId) return "";
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/league/" + encodeURIComponent(leagueId) + "/drafts",
        { cache: "no-store" }
      );
      const rows = res.ok ? await res.json() : [];
      const list = Array.isArray(rows) ? rows : [];
      const live = list.filter(function (d) {
        return d && /^(drafting|paused)$/i.test(String(d.status || ""));
      })[0] || list[0];
      if (live && live.draft_id) {
        cachedLeagueDraftId = String(live.draft_id);
        return cachedLeagueDraftId;
      }
    } catch (_e) {
      /* keep empty */
    }
    return "";
  }

  function mergeIdentity(base, extra) {
    const out = {
      userIds: ((base && base.userIds) || []).slice(),
      username: (base && base.username) || "",
      displayName: (base && base.displayName) || "",
      teamName: (base && base.teamName) || "",
    };
    ((extra && extra.userIds) || []).forEach(function (id) {
      const s = String(id || "");
      if (s && out.userIds.indexOf(s) < 0) out.userIds.unshift(s);
    });
    if (extra && extra.username && !out.username) out.username = extra.username;
    if (extra && extra.displayName && !out.displayName) out.displayName = extra.displayName;
    if (extra && extra.teamName && !out.teamName) out.teamName = extra.teamName;
    return out;
  }

  function applyMainIdentity(detail) {
    if (!detail) return;
    const uid = detail.userId ? String(detail.userId) : "";
    if (uid && /^\d{6,20}$/.test(uid)) {
      cachedUser.userId = uid;
      mainIdentity.userIds = [uid].concat(mainIdentity.userIds.filter(function (x) { return x !== uid; }));
    }
    if (detail.username) {
      mainIdentity.username = String(detail.username);
      cachedUser.username = mainIdentity.username;
    }
    if (detail.displayName) mainIdentity.displayName = String(detail.displayName);
    if (detail.teamName) mainIdentity.teamName = String(detail.teamName);
  }

  function sleeperIdentity() {
    let ident = { userIds: [], username: "", displayName: "", teamName: "" };
    if (window.BRDraftSlot && BRDraftSlot.collectSleeperIdentity) {
      ident = window.BRDraftSlot.collectSleeperIdentity() || ident;
    } else {
      try {
        for (let i = 0; i < localStorage.length; i++) {
          const v = localStorage.getItem(localStorage.key(i)) || "";
          const m = v.match(/"user_id"\s*:\s*"?(\d{6,20})/);
          if (m) {
            ident = { userIds: [m[1]], username: "", displayName: "", teamName: "" };
            break;
          }
        }
      } catch (_e) {
        /* ignore */
      }
    }
    ident = mergeIdentity(ident, mainIdentity);
    if (cachedUser.userId && ident.userIds.indexOf(cachedUser.userId) < 0) {
      ident.userIds.unshift(cachedUser.userId);
    }
    if (cachedUser.username && !ident.username) ident.username = cachedUser.username;
    return ident;
  }

  async function resolveSleeperUserId(ident) {
    const ids = (ident && ident.userIds) || [];
    if (ids.length) {
      cachedUser.userId = ids[0];
      return ids[0];
    }
    if (cachedUser.userId) return cachedUser.userId;
    const un =
      (ident && ident.username) ||
      (window.BRDraftSlot && BRDraftSlot.sleeperUsernameFromDom
        ? BRDraftSlot.sleeperUsernameFromDom()
        : "");
    if (!un) return "";
    if (cachedUser.username === un && cachedUser.userId) return cachedUser.userId;
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/user/" + encodeURIComponent(un),
        { cache: "no-store" }
      );
      const user = res.ok ? await res.json() : null;
      if (user && user.user_id) {
        cachedUser = { username: un, userId: String(user.user_id) };
        return cachedUser.userId;
      }
    } catch (_e) {
      /* ignore */
    }
    return "";
  }

  async function leagueUsers(leagueId) {
    if (!leagueId) return [];
    if (cachedUsers.leagueId === String(leagueId)) return cachedUsers.rows;
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/league/" + encodeURIComponent(leagueId) + "/users",
        { cache: "no-store" }
      );
      const rows = res.ok ? await res.json() : [];
      cachedUsers = { leagueId: String(leagueId), rows: Array.isArray(rows) ? rows : [] };
      return cachedUsers.rows;
    } catch (_e) {
      return cachedUsers.rows || [];
    }
  }

  async function tradedPicks(draftId) {
    if (!draftId) return [];
    if (cachedTrades.draftId === String(draftId) && Date.now() - cachedTrades.at < 8000) {
      return cachedTrades.rows;
    }
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/draft/" + encodeURIComponent(draftId) + "/traded_picks",
        { cache: "no-store" }
      );
      const rows = res.ok ? await res.json() : [];
      cachedTrades = { draftId: String(draftId), rows: Array.isArray(rows) ? rows : [], at: Date.now() };
      return cachedTrades.rows;
    } catch (_e) {
      return cachedTrades.rows || [];
    }
  }

  async function ownerToRosterMap(leagueId) {
    if (!leagueId) return {};
    if (cachedOwnerMap.leagueId === String(leagueId)) return cachedOwnerMap.map;
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/league/" + encodeURIComponent(leagueId) + "/rosters",
        { cache: "no-store" }
      );
      const rows = res.ok ? await res.json() : [];
      const map = {};
      (Array.isArray(rows) ? rows : []).forEach(function (r) {
        if (!r) return;
        if (r.owner_id) map[String(r.owner_id)] = r.roster_id;
        (r.co_owners || []).forEach(function (c) {
          if (c) map[String(c)] = r.roster_id;
        });
      });
      cachedOwnerMap = { leagueId: String(leagueId), map: map };
      return map;
    } catch (_e) {
      return cachedOwnerMap.map || {};
    }
  }

  function resolveMySlot(draft, picks, teams, ident, userId, ownerToRoster) {
    const ids = ((ident && ident.userIds) || []).slice();
    if (userId && ids.indexOf(String(userId)) < 0) ids.unshift(String(userId));
    if (window.BRDraftSlot && BRDraftSlot.sleeperUserIdFromUsers) {
      const fromName = BRDraftSlot.sleeperUserIdFromUsers(cachedUsers.rows || [], ident || {});
      if (fromName && ids.indexOf(fromName) < 0) ids.unshift(fromName);
      if (fromName) cachedUser.userId = fromName;
    }
    if (window.BRDraftSlot && BRDraftSlot.userIdsInDraftOrder) {
      const filtered = BRDraftSlot.userIdsInDraftOrder((draft && draft.draft_order) || {}, ids);
      if (filtered.length) {
        ids.length = 0;
        filtered.forEach(function (id) { ids.push(id); });
      }
    }
    let slot = 0;
    if (window.BRDraftSlot && BRDraftSlot.detectSleeperSlot) {
      slot = BRDraftSlot.detectSleeperSlot({
        draft: draft || {},
        picks: picks || [],
        teams: teams,
        identity: {
          userIds: ids,
          username: ident && ident.username,
          displayName: ident && ident.displayName,
          teamName: ident && ident.teamName,
        },
        ownerToRoster: ownerToRoster || {},
        currentPick: (picks && picks.length ? picks[picks.length - 1].overallPickNumber : 0) + 1,
        auction: String((draft && draft.type) || "").toLowerCase() === "auction",
      });
    }
    if (slot) lastMySlot = slot;
    return lastMySlot || 0;
  }

  function normalizePick(row) {
    if (!row) return null;
    const md = row.metadata || {};
    const first = md.first_name || md.firstName || "";
    const last = md.last_name || md.lastName || "";
    const name = (first + " " + last).trim() || md.player_name || "";
    const pn = Number(row.pick_no || row.pickNo || 0);
    if (!pn) return null;
    return {
      overallPickNumber: pn,
      playerId: row.player_id != null ? String(row.player_id) : "",
      playerName: name,
      pos: window.BRDraftSlot && BRDraftSlot.normDraftPos
        ? BRDraftSlot.normDraftPos(md.position || row.position)
        : String(md.position || row.position || "").toUpperCase(),
      nflTeam: String(md.team || "").toUpperCase(),
      slot: Number(row.draft_slot || row.slot || 0),
      pickedBy: row.picked_by != null ? String(row.picked_by) : "",
      teamId: row.roster_id != null ? String(row.roster_id) : "",
    };
  }

  function fingerprint(picks) {
    const last = picks.length ? picks[picks.length - 1] : null;
    return [
      picks.length,
      last ? last.overallPickNumber : 0,
      last ? last.playerId : "",
      last ? last.playerName || "" : "",
    ].join("|");
  }

  function push(detail) {
    if (typeof window.__brDaPushPicks === "function") window.__brDaPushPicks(detail);
    lastCount = (detail.picks || []).length;
    if (typeof window.__brDaSetSync === "function") {
      window.__brDaSetSync(
        true,
        window.BRDraftSlot
          ? window.BRDraftSlot.compactSync("sleeper", lastCount, detail.mySlot, true)
          : (lastCount ? "SLEEPER · " + lastCount : "SLEEPER · LIVE")
      );
    }
  }

  function emitPayload(draft, picks) {
    const settings = (draft && draft.settings) || {};
    const teams = Number(settings.teams || 12);
    const ident = sleeperIdentity();
    const uid = cachedUser.userId || ((ident && ident.userIds) || [])[0] || "";
    const ownerToRoster = cachedOwnerMap.map || {};
    const users = cachedUsers.rows || [];
    const trades = cachedTrades.rows || [];
    const mySlot = resolveMySlot(draft, picks, teams, ident, uid, ownerToRoster);
    const teamNames = window.BRDraftSlot && BRDraftSlot.teamNamesFromSleeperDraft
      ? BRDraftSlot.teamNamesFromSleeperDraft(draft, users)
      : {};
    const status = String((draft && draft.status) || "").toLowerCase();
    lastStatus = status;
    const inProgress = status === "drafting" || (status !== "complete" && picks.length > 0);
    const drafted = status === "complete";
    const pickOwners = window.BRDraftSlot && BRDraftSlot.sleeperPickOwners
      ? BRDraftSlot.sleeperPickOwners({
          teams: teams,
          rounds: Number(settings.rounds || 15),
          draft: draft || {},
          ownerToRoster: ownerToRoster,
          tradedPicks: trades,
          picks: picks,
        })
      : {};
    const timer = Number((settings && settings.pick_timer) || 0) || undefined;
    const clockSeconds = window.BRDraftSlot && BRDraftSlot.sleeperClockRemaining
      ? BRDraftSlot.sleeperClockRemaining(draft)
      : null;
    const league = cachedLeague;
    const roster = window.BRDraftSlot && BRDraftSlot.rosterFromSleeperLeague
      ? BRDraftSlot.rosterFromSleeperLeague(league ? Object.assign({}, league, { settings: settings }) : { settings: settings })
      : (window.BRDraftSlot && BRDraftSlot.rosterFromSleeperSettings
        ? BRDraftSlot.rosterFromSleeperSettings(settings)
        : null);
    const scoring = cachedScoring || { ppr: 1, tep: 0, passTd: 4 };
    const sf = window.BRDraftSlot && BRDraftSlot.isSleeperSuperflex
      ? BRDraftSlot.isSleeperSuperflex(league, settings)
      : (Number(settings.slots_super_flex || settings.slots_sf || 0) > 0
        || !!(roster && roster.SF));
    const draftName = String((draft && draft.metadata && draft.metadata.name) || "").trim();
    const leagueName = String((league && league.name) || "").trim()
      || (draftName && !/^draft$/i.test(draftName) ? draftName : "");
    const payload = {
      platform: "sleeper",
      teams: Number(settings.teams || (league && league.total_rosters) || 12),
      rounds: Number(settings.rounds || 15),
      mySlot: mySlot || undefined,
      sf: sf,
      leagueName: leagueName || undefined,
      roster: roster || undefined,
      ppr: scoring.ppr,
      tep: scoring.tep,
      passTd: scoring.passTd,
      picks: picks,
      teamNames: teamNames,
      pickOwners: pickOwners,
      inProgress: inProgress,
      drafted: drafted,
      clockSeconds: clockSeconds,
      pickTimer: timer,
      syncText: window.BRDraftSlot
        ? window.BRDraftSlot.compactSync("sleeper", picks.length, mySlot || undefined, true)
        : (picks.length ? "SLEEPER · " + picks.length : "SLEEPER · LIVE"),
    };
    const settingsFp = [
      fingerprint(picks),
      mySlot || "",
      status,
      Object.keys(teamNames).length,
      Object.keys(pickOwners).length,
      window.BRDraftSlot && BRDraftSlot.rosterKey ? BRDraftSlot.rosterKey(roster) : "",
      sf ? 1 : 0,
      leagueName,
      scoring.ppr,
      scoring.tep,
      scoring.passTd,
    ].join("|");
    if (settingsFp !== lastFp) {
      lastFp = settingsFp;
      push(payload);
    }
    const clockKey = String(clockSeconds) + "|" + String(timer || "");
    if (clockKey !== lastClockSent && typeof window.__brDaPushClock === "function") {
      lastClockSent = clockKey;
      window.__brDaPushClock({ clockSeconds: clockSeconds, pickTimer: timer });
    }
  }

  async function fetchDraftAndPicks(id) {
    const [draftRes, pickRes] = await Promise.all([
      fetch("https://api.sleeper.app/v1/draft/" + encodeURIComponent(id), { cache: "no-store" }),
      fetch("https://api.sleeper.app/v1/draft/" + encodeURIComponent(id) + "/picks", { cache: "no-store" }),
    ]);
    const draft = draftRes.ok ? await draftRes.json() : cachedDraft;
    if (draft && draft.league_id && cachedLeagueId !== String(draft.league_id)) {
      await leagueScoring(draft.league_id);
    }
    let picks = cachedPicks;
    if (pickRes.ok) {
      const rows = await pickRes.json();
      const next = (Array.isArray(rows) ? rows : []).map(normalizePick).filter(Boolean);
      next.sort(function (a, b) { return a.overallPickNumber - b.overallPickNumber; });
      const status = String((draft && draft.status) || lastStatus || "").toLowerCase();
      if (next.length >= cachedPicks.length || status === "complete") {
        picks = next;
        cachedPicks = next;
      }
    }
    if (draft) cachedDraft = draft;
    return { draft: draft || cachedDraft, picks: picks };
  }

  async function pollPicks() {
    const id = await resolveDraftId();
    if (!id) {
      if (typeof window.__brDaSetSync === "function") {
        window.__brDaSetSync(false, "SLEEPER · …");
      }
      return;
    }
    try {
      const snap = await fetchDraftAndPicks(id);
      emitPayload(snap.draft, snap.picks);
    } catch (_e) {
      if (typeof window.__brDaSetSync === "function") {
        window.__brDaSetSync(false, "SLEEPER · …");
      }
    }
  }

  async function pollMeta() {
    const id = await resolveDraftId();
    const draft = cachedDraft;
    if (!id || !draft) return;
    const ident = sleeperIdentity();
    try {
      await resolveSleeperUserId(ident);
      const leagueId = draft.league_id;
      await Promise.all([
        ownerToRosterMap(leagueId),
        leagueUsers(leagueId),
        tradedPicks(id),
        leagueScoring(leagueId),
      ]);
      emitPayload(draft, cachedPicks);
    } catch (_e) {
      /* keep last live payload */
    }
  }

  function requestPicks() {
    if (pickInFlight) {
      pickQueued = true;
      return Promise.resolve();
    }
    pickInFlight = true;
    return pollPicks().then(function () {
      pickInFlight = false;
      if (pickQueued) {
        pickQueued = false;
        return requestPicks();
      }
    }, function () {
      pickInFlight = false;
    });
  }

  function pickInterval() {
    if (lastStatus === "complete") return POLL_IDLE_MS;
    if (lastStatus === "pre_draft" || lastStatus === "paused") return POLL_IDLE_MS;
    return POLL_PICKS_MS;
  }

  function schedulePicks() {
    if (pickTimer) clearTimeout(pickTimer);
    pickTimer = setTimeout(function () {
      requestPicks();
      schedulePicks();
    }, pickInterval());
  }

  function scheduleMeta() {
    if (metaTimer) clearTimeout(metaTimer);
    metaTimer = setTimeout(function () {
      pollMeta().finally(scheduleMeta);
    }, POLL_META_MS);
  }

  document.addEventListener("brfantasy:assistant-reconnect", function () {
    lastFp = "";
    requestPicks().then(pollMeta);
  });

  window.addEventListener("message", function (ev) {
    const msg = ev && ev.data;
    if (!msg || msg.__br !== "brfantasy-sleeper-v1" || msg.type !== "identity") return;
    applyMainIdentity(msg.detail);
    lastFp = "";
    requestPicks().then(pollMeta);
  });

  let started = false;
  function startPolling() {
    if (started) return;
    started = true;
    requestPicks().then(function () {
      pollMeta();
      schedulePicks();
      scheduleMeta();
    });
    setTimeout(requestPicks, 200);
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) requestPicks();
    });
    let boardTick = null;
    try {
      const mo = new MutationObserver(function () {
        if (document.hidden || boardTick) return;
        boardTick = setTimeout(function () {
          boardTick = null;
          requestPicks();
        }, 120);
      });
      mo.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
    } catch (_e) {
      /* ignore */
    }
  }

  function waitForDraftRoom() {
    if (isSleeperDraftRoom()) {
      startPolling();
      return;
    }
    const wait = setInterval(function () {
      if (!isSleeperDraftRoom()) return;
      clearInterval(wait);
      startPolling();
    }, 800);
  }

  waitForDraftRoom();
})();
