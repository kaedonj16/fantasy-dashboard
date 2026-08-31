// Isolated-world content script on Sleeper draft rooms. Polls the public
// Sleeper draft API (no pick submission) and feeds the docked overlay.

(function () {
  "use strict";

  const POLL_DRAFTING_MS = 900;
  const POLL_IDLE_MS = 2000;
  let lastFp = "";
  let lastCount = 0;
  let lastMySlot = 0;
  let lastStatus = "";
  let pollTimer = null;
  let cachedLeagueId = "";
  let cachedScoring = { ppr: 1, tep: 0, passTd: 4 };
  let cachedUser = { username: "", userId: "" };
  let cachedOwnerMap = { leagueId: "", map: {} };
  let cachedUsers = { leagueId: "", rows: [] };

  async function leagueScoring(leagueId) {
    if (!leagueId) return cachedScoring;
    if (cachedLeagueId === String(leagueId) && cachedScoring) return cachedScoring;
    try {
      const res = await fetch(
        "https://api.sleeper.app/v1/league/" + encodeURIComponent(leagueId),
        { cache: "no-store" }
      );
      const league = res.ok ? await res.json() : null;
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

  function draftIdFromUrl() {
    try {
      const m = location.pathname.match(/\/draft\/(?:nfl\/)?([a-zA-Z0-9]+)/i);
      return m ? m[1] : "";
    } catch (_e) {
      return "";
    }
  }

  function sleeperIdentity() {
    if (window.BRDraftSlot && BRDraftSlot.collectSleeperIdentity) {
      return BRDraftSlot.collectSleeperIdentity();
    }
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const v = localStorage.getItem(localStorage.key(i)) || "";
        const m = v.match(/"user_id"\s*:\s*"?(\d{6,20})/);
        if (m) return { userIds: [m[1]], username: "", displayName: "", teamName: "" };
      }
    } catch (_e) {
      /* ignore */
    }
    return { userIds: [], username: "", displayName: "", teamName: "" };
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
      pos: String(md.position || row.position || "").toUpperCase(),
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

  async function poll() {
    const id = draftIdFromUrl();
    if (!id) {
      if (typeof window.__brDaSetSync === "function") {
        window.__brDaSetSync(false, "SLEEPER · …");
      }
      return;
    }
    try {
      const [draftRes, pickRes] = await Promise.all([
        fetch("https://api.sleeper.app/v1/draft/" + encodeURIComponent(id), { cache: "no-store" }),
        fetch("https://api.sleeper.app/v1/draft/" + encodeURIComponent(id) + "/picks", { cache: "no-store" }),
      ]);
      const draft = draftRes.ok ? await draftRes.json() : null;
      const rows = pickRes.ok ? await pickRes.json() : [];
      const picks = (Array.isArray(rows) ? rows : []).map(normalizePick).filter(Boolean);
      picks.sort(function (a, b) { return a.overallPickNumber - b.overallPickNumber; });
      const fp = fingerprint(picks);
      const settings = (draft && draft.settings) || {};
      const teams = Number(settings.teams || 12);
      const ident = sleeperIdentity();
      const uid = await resolveSleeperUserId(ident);
      const leagueId = draft && draft.league_id;
      const ownerToRoster = await ownerToRosterMap(leagueId);
      const users = await leagueUsers(leagueId);
      const mySlot = resolveMySlot(draft, picks, teams, ident, uid, ownerToRoster);
      const teamNames = window.BRDraftSlot && BRDraftSlot.teamNamesFromSleeperDraft
        ? BRDraftSlot.teamNamesFromSleeperDraft(draft, users)
        : {};
      const status = String((draft && draft.status) || "").toLowerCase();
      lastStatus = status;
      const inProgress = status === "drafting" || (status !== "complete" && picks.length > 0);
      const drafted = status === "complete";
      const roster = window.BRDraftSlot && BRDraftSlot.rosterFromSleeperSettings
        ? BRDraftSlot.rosterFromSleeperSettings(settings)
        : null;
      const scoring = await leagueScoring(draft && draft.league_id);
      const payload = {
        platform: "sleeper",
        teams: Number(settings.teams || 12),
        rounds: Number(settings.rounds || 15),
        mySlot: mySlot || undefined,
        sf: Number(settings.slots_super_flex || settings.slots_sf || 0) > 0,
        roster: roster || undefined,
        ppr: scoring.ppr,
        tep: scoring.tep,
        passTd: scoring.passTd,
        picks: picks,
        teamNames: teamNames,
        inProgress: inProgress,
        drafted: drafted,
        syncText: window.BRDraftSlot
          ? window.BRDraftSlot.compactSync("sleeper", picks.length, mySlot || undefined, true)
          : (picks.length ? "SLEEPER · " + picks.length : "SLEEPER · LIVE"),
      };
      const settingsFp = [
        fp,
        mySlot || "",
        status,
        Object.keys(teamNames).length,
        window.BRDraftSlot && BRDraftSlot.rosterKey ? BRDraftSlot.rosterKey(roster) : "",
        scoring.ppr,
        scoring.tep,
        scoring.passTd,
      ].join("|");
      if (settingsFp === lastFp) return;
      lastFp = settingsFp;
      push(payload);
    } catch (_e) {
      if (typeof window.__brDaSetSync === "function") {
        window.__brDaSetSync(false, "SLEEPER · …");
      }
    }
  }

  document.addEventListener("brfantasy:assistant-reconnect", function () {
    lastFp = "";
    poll();
  });

  function schedulePoll() {
    if (pollTimer) clearTimeout(pollTimer);
    const ms = lastStatus === "drafting" ? POLL_DRAFTING_MS : POLL_IDLE_MS;
    pollTimer = setTimeout(function () {
      poll().then(schedulePoll);
    }, ms);
  }
  poll().then(schedulePoll);
  setTimeout(poll, 500);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) poll();
  });
  let slotTick = null;
  try {
    const mo = new MutationObserver(function () {
      if (document.hidden || lastMySlot || slotTick) return;
      slotTick = setTimeout(function () {
        slotTick = null;
        poll();
      }, 400);
    });
    mo.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  } catch (_e) {
    /* ignore */
  }
})();
