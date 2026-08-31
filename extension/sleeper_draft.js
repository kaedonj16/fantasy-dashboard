// Isolated-world content script on Sleeper draft rooms. Polls the public
// Sleeper draft API (no pick submission) and feeds the docked overlay.

(function () {
  "use strict";

  const POLL_MS = 1500;
  let lastFp = "";
  let lastCount = 0;
  let cachedLeagueId = "";
  let cachedScoring = { ppr: 1, tep: 0, passTd: 4 };

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

  function sleeperUserId() {
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        const v = localStorage.getItem(k) || "";
        const m = v.match(/"user_id"\s*:\s*"?(\d{3,})/);
        if (m) return m[1];
      }
    } catch (_e) {
      /* ignore */
    }
    return "";
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
      teamId: row.roster_id != null ? String(row.roster_id) : "",
    };
  }

  function fingerprint(picks) {
    const last = picks.length ? picks[picks.length - 1] : null;
    return [picks.length, last ? last.overallPickNumber : 0, last ? last.playerId : ""].join("|");
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
      const uid = sleeperUserId();
      let mySlot = 0;
      if (uid && draft && draft.draft_order && draft.draft_order[uid] != null) {
        mySlot = Number(draft.draft_order[uid]);
      }
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
        syncText: window.BRDraftSlot
          ? window.BRDraftSlot.compactSync("sleeper", picks.length, mySlot || undefined, true)
          : (picks.length ? "SLEEPER · " + picks.length : "SLEEPER · LIVE"),
      };
      const settingsFp = [
        fp,
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

  poll();
  setInterval(poll, POLL_MS);
})();
