// Runs in the MAIN world on Yahoo's draft room so it can see page JS state
// and fetch/XHR traffic. Yahoo's draftresults API usually updates mid-draft;
// the open draft room is still the fastest, most reliable source. We observe
// in-page state and forward a compact pick snapshot to the isolated content
// script via postMessage (no cookies leave this page).

(function () {
  "use strict";

  const EVENT = "brfantasy:yahoo-draft-raw";
  const RESCAN = "brfantasy:draft-rescan";
  const RELAY_STATUS = "brfantasy:yahoo-relay-status";
  const BRIDGE = "brfantasy-bridge-v1";
  const MAX_WALK = 14000;
  let lastFingerprint = "";
  let lastEmitAt = 0;
  const pickAccumulator = new Map();
  let bestOverallSeen = 0;
  let detectedUserTeamId = null;
  let detectedSlot = 0;
  let detectedTeams = 0;
  let detectedRounds = 0;
  let detectedRoster = null;
  let detectedSf = false;
  let detectedPpr = 1;
  let detectedTep = 0;
  let detectedPassTd = 4;
  const detectedTeamNamesById = {};
  const detectedTeamSlotsById = {};
  let lastClockSeconds = null;
  /** @type {Map<string, {playerName: string, pos: string, nflTeam: string}>} */
  const playerMetaById = new Map();

  function leagueFromUrl() {
    try {
      const u = new URL(location.href);
      let leagueId = (u.searchParams.get("leagueId") || u.searchParams.get("league") || "").trim();
      let season = (u.searchParams.get("seasonId") || u.searchParams.get("season") || "").trim();
      // Path: /f1/{leagueId}/draft or /f1/{leagueId}/livedraft
      if (!leagueId) {
        const m = u.pathname.match(/\/f1\/(\d+)(?:\/|$)/i);
        if (m) leagueId = m[1];
      }
      // Full Yahoo league key in query: 461.l.12345
      if (!leagueId) {
        const key = (u.searchParams.get("leagueKey") || u.searchParams.get("key") || "").trim();
        if (key && key.indexOf(".l.") >= 0) leagueId = key.split(".l.").pop() || "";
      }
      const client = u.pathname.match(/\/draftclient\/(?:nfl\/|f1\/)?(\d+)\/(\d+)/i);
      if (client) {
        if (!leagueId) leagueId = client[1];
        if (!detectedUserTeamId) detectedUserTeamId = client[2];
      }
      return { leagueId, season };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function bridgeToExtension(type, detail) {
    const msg = { __br: BRIDGE, type: type, detail: detail || {} };
    try {
      window.postMessage(msg, "*");
    } catch (_e) { /* ignore */ }
    try {
      if (window.top && window.top !== window) window.top.postMessage(msg, "*");
    } catch (_e) { /* ignore */ }
  }

  function yahooIdFromKey(key) {
    if (key == null) return null;
    const s = String(key);
    if (s.indexOf(".p.") >= 0) return s.split(".p.").pop();
    return s;
  }

  function teamIdFromKey(key) {
    if (key == null) return null;
    const s = String(key);
    if (s.indexOf(".t.") >= 0) return s.split(".t.").pop();
    const parts = s.split(".");
    return parts.length ? parts[parts.length - 1] : s;
  }

  function rawPlayerId(obj) {
    if (!obj || typeof obj !== "object") return null;
    const direct =
      obj.playerId ??
      obj.player_id ??
      yahooIdFromKey(obj.player_key || obj.playerKey);
    if (direct != null && String(direct) !== "" && String(direct) !== "0") return String(direct);
    const player = yahooPlayerBlob(obj);
    if (player && player !== obj) return rawPlayerId(player);
    return null;
  }

  function rawOverall(obj) {
    if (!obj || typeof obj !== "object") return 0;
    const direct =
      obj.overallPickNumber ??
      obj.overallPick ??
      obj.overall_pick ??
      obj.pick_no ??
      obj.pickNumber ??
      obj.pick_number ??
      obj.overall ??
      obj.pickOverall ??
      obj.overallPickNo ??
      obj.draftPickNumber;
    if (direct != null && Number(direct) >= 1) return Number(direct);
    const pick = obj.pick ?? obj.selection ?? obj.slot ?? obj.pickSlot;
    if (typeof pick === "string" && /^\d{1,2}\.\d{1,2}$/.test(pick.trim())) {
      const parts = pick.trim().split(".");
      const rd = Number(parts[0]);
      const pk = Number(parts[1]);
      const n = detectedTeams >= 4 ? detectedTeams : 12;
      if (rd >= 1 && pk >= 1) return (rd - 1) * n + pk;
    }
    const nPick = Number(pick);
    if (!(nPick >= 1)) return 0;
    const nRound = Number(obj.round ?? obj.roundId ?? obj.round_id);
    const teams = detectedTeams || 0;
    if (nRound >= 2 && nPick <= (teams || 16)) {
      const n = teams >= 4 ? teams : 12;
      return (nRound - 1) * n + (nRound % 2 === 1 ? nPick : n - nPick + 1);
    }
    return nPick;
  }

  function fantasyTeamId(obj) {
    if (!obj || typeof obj !== "object") return null;
    const key = obj.team_key || obj.teamKey;
    if (key && String(key).indexOf(".t.") >= 0) return teamIdFromKey(key);
    const tid = obj.teamId ?? obj.team_id ?? obj.fantasyTeamId;
    if (tid != null && String(tid) !== "" && String(tid) !== "0") return String(tid);
    return null;
  }

  function looksDraftedYahoo(obj) {
    if (!obj || typeof obj !== "object") return false;
    if (obj.drafted === false || obj.isDrafted === false || obj.available === true) return false;
    const name = yahooPlayerName(obj);
    const pid = rawPlayerId(obj);
    if (!name && !pid) return false;
    if (obj.drafted === true || obj.isDrafted === true || obj.selected === true || obj.isPicked === true) return true;
    if (obj.status && /drafted|taken|selected|picked/i.test(String(obj.status))) return true;
    const key = obj.team_key || obj.teamKey;
    if (key && String(key).indexOf(".l.") >= 0 && String(key).indexOf(".t.") >= 0) return true;
    return !!rawOverall(obj);
  }

  function isPickRow(obj) {
    if (!obj || typeof obj !== "object") return false;
    const overall = rawOverall(obj);
    if (overall) {
      if (rawPlayerId(obj)) return true;
      const name = yahooPlayerName(obj);
      return !!(name && name.length >= 3);
    }
    return looksDraftedYahoo(obj);
  }

  function pickLooksMade(norm) {
    if (!norm || !norm.overallPickNumber) return false;
    const pid = norm.playerId != null ? String(norm.playerId) : "";
    if (pid && pid !== "0" && pid !== "-1") return true;
    return !!(norm.playerName && String(norm.playerName).trim().length >= 3);
  }

  function yahooPlayerBlob(raw) {
    if (!raw || typeof raw !== "object") return null;
    let player = raw.player;
    if (Array.isArray(player)) player = player[0];
    return player && typeof player === "object" ? player : null;
  }

  function yahooPlayerName(raw) {
    if (!raw || typeof raw !== "object") return "";
    const direct = raw.playerName || raw.player_name || raw.fullName || raw.display_name || raw.name_full;
    if (direct && typeof direct === "string") return direct.trim();
    if (typeof raw.name === "string" && raw.name.trim().length >= 3) {
      const n = raw.name.trim();
      if (!/^(yahoo|draft|team|round|pick|available|queue|board)$/i.test(n)) return n;
    }
    const nameObj = raw.name;
    if (nameObj && typeof nameObj === "object") {
      const full = nameObj.full || nameObj.fullName;
      if (full) return String(full).trim();
      const joined = ((nameObj.first || nameObj.firstName || "") + " " + (nameObj.last || nameObj.lastName || "")).trim();
      if (joined) return joined;
    }
    const first = raw.firstName || raw.first_name || "";
    const last = raw.lastName || raw.last_name || "";
    const joined = (String(first) + " " + String(last)).trim();
    if (joined) return joined;
    const player = yahooPlayerBlob(raw);
    if (player && player !== raw) return yahooPlayerName(player);
    return "";
  }

  function yahooPlayerPos(raw) {
    if (!raw || typeof raw !== "object") return "";
    const pos = raw.pos || raw.position || raw.display_position || raw.displayPosition || raw.eligible_positions;
    if (typeof pos === "string" && pos) return pos.toUpperCase().split(",")[0].trim();
    if (Array.isArray(pos) && pos[0]) return String(pos[0]).toUpperCase();
    const player = yahooPlayerBlob(raw);
    if (player && player !== raw) return yahooPlayerPos(player);
    return "";
  }

  function yahooPlayerTeam(raw) {
    if (!raw || typeof raw !== "object") return "";
    const team =
      raw.nflTeam ||
      raw.editorial_team_abbr ||
      raw.editorialTeamAbbr ||
      raw.team_abbr ||
      raw.display_team;
    if (typeof team === "string" && /[A-Za-z]/.test(team)) return team.toUpperCase().slice(0, 3);
    const player = yahooPlayerBlob(raw);
    if (player && player !== raw) return yahooPlayerTeam(player);
    return "";
  }

  function rememberYahooPlayer(raw) {
    if (!raw || typeof raw !== "object") return;
    const pid = rawPlayerId(raw);
    if (pid == null || String(pid) === "" || String(pid) === "0") return;
    const name = yahooPlayerName(raw);
    const pos = yahooPlayerPos(raw);
    const nfl = yahooPlayerTeam(raw);
    if (!name && !pos && !nfl) return;
    const prev = playerMetaById.get(String(pid)) || {};
    playerMetaById.set(String(pid), {
      playerName: name || prev.playerName || "",
      pos: pos || prev.pos || "",
      nflTeam: nfl || prev.nflTeam || "",
    });
  }

  function normalizePick(raw) {
    if (!isPickRow(raw)) {
      rememberYahooPlayer(raw);
      return null;
    }
    rememberYahooPlayer(raw);
    const playerId = rawPlayerId(raw);
    const overall = rawOverall(raw);
    const teamId =
      raw.teamId ??
      raw.team_id ??
      fantasyTeamId(raw) ??
      teamIdFromKey(raw.team_key || raw.teamKey);
    const roundId = raw.roundId ?? raw.round ?? raw.round_id;
    const roundPick = raw.roundPickNumber ?? raw.roundPick ?? raw.round_pick;
    const cost = raw.bidAmount ?? raw.cost ?? raw.auction_cost;
    const cached = playerId != null ? playerMetaById.get(String(playerId)) : null;
    const playerName = yahooPlayerName(raw) || (cached && cached.playerName) || "";
    const pos = yahooPlayerPos(raw) || (cached && cached.pos) || "";
    const nflTeam = yahooPlayerTeam(raw) || (cached && cached.nflTeam) || "";
    if (!Number(overall) && !playerName && !playerId) return null;
    return {
      overallPickNumber: Number(overall) || 0,
      playerId: playerId == null ? null : String(playerId),
      playerName: playerName || undefined,
      pos: pos || undefined,
      nflTeam: nflTeam || undefined,
      teamId: teamId == null ? null : String(teamId),
      roundId: roundId == null ? null : Number(roundId),
      roundPickNumber: roundPick == null ? null : Number(roundPick),
      keeper: !!(raw.keeper || raw.is_keeper),
      bidAmount: cost != null ? cost : null,
    };
  }

  function fingerprint(picks, meta) {
    const last = picks.length ? picks[picks.length - 1] : null;
    return [
      meta.inProgress ? 1 : 0,
      meta.drafted ? 1 : 0,
      picks.length,
      last ? last.overallPickNumber : 0,
      last ? last.playerId : "",
      last ? last.playerName || "" : "",
      last ? last.teamId : "",
      detectedUserTeamId || "",
      detectedSlot || "",
      detectedTeams || "",
      detectedRounds || "",
      detectedSf ? 1 : 0,
      detectedPpr,
      detectedTep,
      detectedPassTd,
      rosterFingerprint(detectedRoster),
    ].join("|");
  }

  function emptyRoster() {
    return { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, RB_WR: 0, WR_TE: 0, RB_TE: 0, K: 0, DEF: 0, BN: 0 };
  }

  function rosterFromYahooPositions(positions) {
    const out = emptyRoster();
    (positions || []).forEach(function (p) {
      const raw = String((p && (p.position || p.display_position || p)) || "").toUpperCase();
      const n = Number((p && (p.count || p.num || p.slots)) || 1) || 1;
      if (!raw || raw === "IR" || raw === "IR+" || raw === "TAXI") return;
      if (raw === "QB") out.QB += n;
      else if (raw === "RB") out.RB += n;
      else if (raw === "WR") out.WR += n;
      else if (raw === "TE") out.TE += n;
      else if (raw === "K") out.K += n;
      else if (raw === "DEF" || raw === "DST") out.DEF += n;
      else if (raw === "BN" || raw === "BENCH") out.BN += n;
      else if (raw === "Q/W/R/T" || raw === "QP" || raw === "SUPER_FLEX" || raw === "SF") out.SF += n;
      else if (raw === "W/R" || raw === "RB/WR" || raw === "WRRB_FLEX" || raw === "RB_WR") out.RB_WR += n;
      else if (raw === "W/T" || raw === "WR/TE" || raw === "REC_FLEX" || raw === "WR_TE") out.WR_TE += n;
      else if (raw === "R/T" || raw === "RB/TE" || raw === "RB_TE") out.RB_TE += n;
      else if (raw === "W/R/T" || raw === "FLEX") out.FLEX += n;
    });
    return out;
  }

  function rosterFingerprint(rs) {
    if (!rs) return "";
    return ["QB", "SF", "RB", "WR", "TE", "FLEX", "RB_WR", "WR_TE", "RB_TE", "K", "DEF", "BN"].map(function (k) {
      return k + (Number(rs[k]) || 0);
    }).join("");
  }

  function applyYahooScoring(obj) {
    if (!obj || typeof obj !== "object") return;
    const rec = obj.rec ?? obj.reception_points ?? obj.ppr;
    if (rec != null && Number.isFinite(Number(rec))) detectedPpr = Number(rec);
    const tep = obj.bonus_rec_te ?? obj.tep ?? obj.te_premium;
    if (tep != null && Number.isFinite(Number(tep))) detectedTep = Number(tep);
    const passTd = obj.pass_td ?? obj.passing_td ?? obj.passTd;
    if (passTd != null && Number.isFinite(Number(passTd))) detectedPassTd = Number(passTd);
    function walkStats(stats) {
      const list = Array.isArray(stats)
        ? stats
        : (stats && (stats.stat || stats.stats || stats.modifiers));
      const arr = Array.isArray(list)
        ? list
        : (list && typeof list === "object" ? Object.keys(list).map(function (k) { return list[k]; }) : []);
      arr.forEach(function (st) {
        if (!st || typeof st !== "object") return;
        const id = Number(st.stat_id != null ? st.stat_id : (st.statId != null ? st.statId : st.id));
        const val = Number(st.value != null ? st.value : (st.points != null ? st.points : st.modifier));
        if (!Number.isFinite(val)) return;
        if (id === 11) detectedPpr = val;
        if (id === 4) detectedPassTd = val;
      });
    }
    walkStats(obj.stat_modifiers);
    if (obj.stat_modifiers && obj.stat_modifiers.stats) walkStats(obj.stat_modifiers.stats);
    if (obj.stat_categories && obj.stat_categories.stats) walkStats(obj.stat_categories.stats);
  }

  function rememberYahooSettings(obj) {
    if (!obj || typeof obj !== "object") return;
    const nTeams = obj.num_teams ?? obj.numTeams ?? obj.size;
    if (nTeams >= 4 && nTeams <= 32) detectedTeams = Number(nTeams);
    const nRounds = obj.num_rounds ?? obj.numRounds ?? obj.draft_rounds;
    if (nRounds >= 6 && nRounds <= 40) detectedRounds = Number(nRounds);
    const positions = obj.roster_positions || obj.rosterPositions;
    if (Array.isArray(positions)) {
      let n = 0;
      for (let i = 0; i < positions.length; i++) {
        const p = positions[i];
        const pos = String((p && (p.position || p.display_position || p)) || "").toUpperCase();
        const c = Number((p && (p.count || p.num || p.slots)) || 1) || 1;
        if (pos && pos !== "IR" && pos !== "IR+" && pos !== "TAXI") n += c;
      }
      if (n >= 6 && n <= 40) detectedRounds = n;
      const mapped = rosterFromYahooPositions(positions);
      if ((mapped.QB || 0) + (mapped.RB || 0) + (mapped.WR || 0) + (mapped.TE || 0) + (mapped.FLEX || 0) + (mapped.SF || 0) >= 4) {
        detectedRoster = mapped;
        detectedSf = (mapped.SF || 0) > 0;
      }
    }
    applyYahooScoring(obj);
    if (obj.settings && obj.settings !== obj) applyYahooScoring(obj.settings);
  }

  function rememberYahooUser(obj) {
    if (!obj || typeof obj !== "object") return;
    rememberYahooSettings(obj);
    const uid = obj.userTeamId ?? obj.myTeamId ?? obj.currentUserTeamId ?? obj.viewerTeamId;
    if (uid != null && uid !== "" && String(uid) !== "0") detectedUserTeamId = uid;
    const owned =
      obj.is_owned_by_current_login === 1 ||
      obj.is_owned_by_current_login === true ||
      obj.isOwnedByCurrentLogin === true ||
      obj.isCurrentUser === true ||
      obj.isUser === true;
    const managers = obj.managers || obj.manager;
    let managerOwned = false;
    const list = Array.isArray(managers) ? managers : managers ? [managers] : [];
    for (let i = 0; i < list.length; i++) {
      const m = list[i];
      if (!m || typeof m !== "object") continue;
      if (m.is_current_login === 1 || m.is_current_login === true || m.isCurrentLogin === true) {
        managerOwned = true;
      }
    }
    const teamId = obj.teamId ?? obj.team_id ?? teamIdFromKey(obj.team_key || obj.teamKey);
    let teamName = obj.name || obj.team_name || obj.teamName || obj.nickname;
    if (!teamName && list[0]) {
      teamName = list[0].nickname || list[0].manager_nickname || list[0].nickname;
    }
    if (teamId != null && teamId !== "" && teamName) {
      detectedTeamNamesById[String(teamId)] = String(teamName).trim();
    }
    const draftPos = obj.draftPosition ?? obj.draft_position ?? obj.draftSlot ?? obj.draft_slot;
    if (teamId != null && Number(draftPos) >= 1) detectedTeamSlotsById[String(teamId)] = Number(draftPos);
    if (owned || managerOwned) {
      if (teamId != null && teamId !== "") detectedUserTeamId = teamId;
      if (draftPos != null && Number(draftPos) >= 1) detectedSlot = Number(draftPos);
    }
  }

  function snakeSlotFromPick(overall, teams) {
    const n = Number(teams) || 0;
    const pn = Number(overall) || 0;
    if (n < 2 || pn < 1) return 0;
    const r = Math.ceil(pn / n);
    const i = (pn - 1) % n;
    return r % 2 === 1 ? i + 1 : n - i;
  }

  function computeMySlot(picks) {
    if (detectedSlot >= 1) return detectedSlot;
    if (detectedUserTeamId == null || detectedUserTeamId === "") return 0;
    const want = String(detectedUserTeamId);
    const ids = {};
    let first = 0;
    (picks || []).forEach(function (p) {
      if (p && p.teamId != null && p.teamId !== "") ids[String(p.teamId)] = true;
      if (!p || String(p.teamId) !== want) return;
      const n = Number(p.overallPickNumber || 0);
      if (n && (!first || n < first)) first = n;
    });
    const teams = Object.keys(ids).length >= 4 ? Object.keys(ids).length : 12;
    return first ? snakeSlotFromPick(first, teams) : 0;
  }

  function relayToBackground(detail) {
    if (!detail || !detail.leagueId) return;
    try {
      chrome.runtime.sendMessage(
        {
          type: "yahooDraftRelay",
          leagueId: detail.leagueId,
          season: detail.season || "",
          inProgress: !!detail.inProgress,
          drafted: !!detail.drafted,
          picks: Array.isArray(detail.picks) ? detail.picks : [],
          source: detail.source || "yahoo-draft-room",
          at: detail.at || Date.now(),
        },
        function (resp) {
          void chrome.runtime.lastError;
          bridgeToExtension(RELAY_STATUS, {
            sent: resp && resp.sent,
            tabs: resp && resp.tabs,
            pickCount: (detail.picks || []).length,
            reason: resp && resp.reason,
          });
        }
      );
    } catch (_e) {
      bridgeToExtension(RELAY_STATUS, {
        sent: 0,
        pickCount: (detail.picks || []).length,
        reason: "runtime_error",
      });
    }
  }

  function mergeIntoAccumulator(rawPicks, source) {
    let grew = false;
    const list = rawPicks || [];
    const hasOverall = list.some(function (raw) {
      if (!raw || typeof raw !== "object") return false;
      return rawOverall(raw) >= 1 || Number(raw.overallPickNumber) >= 1;
    });
    const draftedCount = list.filter(looksDraftedYahoo).length;
    const allowSeq = !hasOverall && list.length <= 320 && draftedCount >= Math.max(1, list.length * 0.8);
    let seq = 0;
    for (const raw of list) {
      let norm = normalizePick(raw) || (raw && Number(raw.overallPickNumber) >= 1 ? raw : null);
      if (!norm && looksDraftedYahoo(raw)) {
        rememberYahooPlayer(raw);
        norm = {
          overallPickNumber: 0,
          playerId: rawPlayerId(raw),
          playerName: yahooPlayerName(raw) || undefined,
          pos: yahooPlayerPos(raw) || undefined,
          nflTeam: yahooPlayerTeam(raw) || undefined,
          teamId: fantasyTeamId(raw),
        };
      }
      if (norm && !(Number(norm.overallPickNumber) >= 1)) {
        if (!allowSeq) continue;
        seq += 1;
        norm.overallPickNumber = seq;
      }
      if (!norm || !pickLooksMade(norm)) continue;
      const n = Number(norm.overallPickNumber);
      if (!n || n <= 0) continue;
      if (!pickAccumulator.has(n)) grew = true;
      const prev = pickAccumulator.get(n);
      if (prev) {
        if (prev.playerName && !norm.playerName) norm.playerName = prev.playerName;
        if (prev.pos && !norm.pos) norm.pos = prev.pos;
        if (prev.nflTeam && !norm.nflTeam) norm.nflTeam = prev.nflTeam;
        if (prev.playerId && !norm.playerId) norm.playerId = prev.playerId;
        if (prev.teamId && !norm.teamId) norm.teamId = prev.teamId;
      } else {
        grew = true;
      }
      pickAccumulator.set(n, norm);
      if (n > bestOverallSeen) bestOverallSeen = n;
    }
    void source;
    return grew;
  }

  function emitAccumulated(meta, source) {
    const ids = leagueFromUrl();
    let clean = Array.from(pickAccumulator.values())
      .filter(pickLooksMade)
      .sort((a, b) => a.overallPickNumber - b.overallPickNumber);
    if (window.BRDraftSlot && BRDraftSlot.filterYahooPicksToClock) {
      const capped = BRDraftSlot.filterYahooPicksToClock(clean, detectedTeams || 12);
      if (capped.length !== clean.length) {
        pickAccumulator.clear();
        bestOverallSeen = 0;
        capped.forEach(function (p) {
          pickAccumulator.set(Number(p.overallPickNumber), p);
          if (p.overallPickNumber > bestOverallSeen) bestOverallSeen = p.overallPickNumber;
        });
        clean = capped;
      }
    }
    if (!clean.length && !lastFingerprint) return;
    const fp = fingerprint(clean, meta || {});
    const now = Date.now();
    if (fp === lastFingerprint && now - lastEmitAt < 1200) return;
    lastFingerprint = fp;
    lastEmitAt = now;
    const mySlot = computeMySlot(clean);
    const teamNames = window.BRDraftSlot && BRDraftSlot.teamNamesFromTeamIds
      ? BRDraftSlot.teamNamesFromTeamIds(detectedTeamNamesById, detectedTeamSlotsById, clean, detectedTeams)
      : {};
    if (window.BRDraftSlot && BRDraftSlot.scrapeHostClockSeconds) {
      const scraped = BRDraftSlot.scrapeHostClockSeconds(document);
      if (scraped != null) lastClockSeconds = scraped;
    }
    const detail = {
      source: source || "accumulated",
      leagueId: ids.leagueId,
      season: ids.season,
      inProgress: meta && meta.inProgress == null ? true : !!(meta && meta.inProgress),
      drafted: !!(meta && meta.drafted),
      picks: clean,
      mySlot: mySlot || undefined,
      userTeamId: detectedUserTeamId || undefined,
      teams: detectedTeams || undefined,
      rounds: detectedRounds || undefined,
      roster: detectedRoster || undefined,
      sf: detectedSf,
      ppr: detectedPpr,
      tep: detectedTep,
      passTd: detectedPassTd,
      teamNames: teamNames,
      clockSeconds: lastClockSeconds,
      at: now,
    };
    const clockDone = window.BRDraftSlot && BRDraftSlot.completedFromYahooClock
      ? BRDraftSlot.completedFromYahooClock(detectedTeams || 12)
      : -1;
    if (clockDone >= 0) detail.current = clockDone + 1;
    bridgeToExtension(EVENT, detail);
    relayToBackground(detail);
  }

  function emit(picks, meta, source) {
    const incoming = picks || [];
    const grew = mergeIntoAccumulator(incoming, source);
    if (!grew && pickAccumulator.size) {
      const mapped = incoming.map(normalizePick).filter(Boolean);
      const maxIncoming = mapped.length ? mapped[mapped.length - 1].overallPickNumber : 0;
      if (maxIncoming <= bestOverallSeen && pickAccumulator.size >= mapped.length) {
        emitAccumulated(meta, source);
        return;
      }
    }
    emitAccumulated(meta, source);
  }

  function picksFromDraftResults(block) {
    if (!block || typeof block !== "object") return null;
    const out = [];
    if (Array.isArray(block)) {
      for (const item of block) {
        const dr = (item && item.draft_result) || item;
        if (isPickRow(dr)) out.push(dr);
      }
      return out.length ? out : null;
    }
    const count = Number(block.count || 0);
    if (count > 0) {
      for (let i = 0; i < count; i++) {
        const entry = block[String(i)] || block[i];
        const dr = (entry && entry.draft_result) || entry;
        if (isPickRow(dr)) out.push(dr);
      }
      return out.length ? out : null;
    }
    for (const k of Object.keys(block)) {
      if (k === "count") continue;
      const entry = block[k];
      const dr = (entry && entry.draft_result) || entry;
      if (isPickRow(dr)) out.push(dr);
    }
    return out.length ? out : null;
  }

  function maybeFromDraftResults(block, source, meta) {
    const picks = picksFromDraftResults(block);
    if (!picks) return false;
    emit(picks, meta || { inProgress: true, drafted: false }, source);
    return true;
  }

  function collectPickArrays(root) {
    const seen = new Set();
    const arrays = [];
    const q = [root];
    let n = 0;
    function take(arr) {
      if (!Array.isArray(arr) || !arr.length || arr.length > 500) return;
      if (arr.some(isPickRow)) arrays.push(arr);
    }
    while (q.length && n < MAX_WALK) {
      const cur = q.shift();
      n++;
      if (!cur || typeof cur !== "object") continue;
      if (seen.has(cur)) continue;
      try {
        seen.add(cur);
      } catch (_e) {
        continue;
      }
      rememberYahooUser(cur);
      if (cur.draft_results) {
        const fromBlock = picksFromDraftResults(cur.draft_results);
        if (fromBlock) arrays.push(fromBlock);
      }
      if (cur.draftResults) {
        const fromBlock = picksFromDraftResults(cur.draftResults);
        if (fromBlock) arrays.push(fromBlock);
      }
      take(cur.picks);
      take(cur.draftPicks);
      take(cur.draft_picks);
      take(cur.selections);
      take(cur.results);
      take(cur.pickHistory);
      take(cur.pickedPlayers);
      take(cur.draftBoard);
      take(cur.draftedPlayers);
      take(cur.drafted);
      take(cur.takenPlayers);
      take(cur.selectedPlayers);
      take(cur.history);
      if (Array.isArray(cur) && cur.some(isPickRow)) arrays.push(cur);
      const next = [];
      if (cur.memoizedProps) next.push(cur.memoizedProps);
      if (cur.pendingProps) next.push(cur.pendingProps);
      if (cur.stateNode) next.push(cur.stateNode);
      if (cur.state) next.push(cur.state);
      if (cur.return) next.push(cur.return);
      if (cur.child) next.push(cur.child);
      if (cur.sibling) next.push(cur.sibling);
      if (cur.props) next.push(cur.props);
      try {
        for (const k of Object.keys(cur)) {
          if (k === "draft_results" || k === "draftResults") {
            const fromKey = picksFromDraftResults(cur[k]);
            if (fromKey) arrays.push(fromKey);
          }
          const v = cur[k];
          if (v && typeof v === "object" && !seen.has(v)) next.push(v);
          if (next.length > 80) break;
        }
      } catch (_e) {
        /* ignore */
      }
      for (let i = 0; i < Math.min(next.length, 40); i++) q.push(next[i]);
    }
    return arrays;
  }

  function walkForPicks(root) {
    const arrays = collectPickArrays(root);
    if (!arrays.length) return false;
    arrays.forEach(function (arr) {
      mergeIntoAccumulator(arr, "react");
    });
    emitAccumulated({ inProgress: true, drafted: false }, "react");
    return pickAccumulator.size > 0;
  }

  function scanReact() {
    const roots = [];
    const docs = window.BRDraftSlot && BRDraftSlot.sameOriginDocuments
      ? BRDraftSlot.sameOriginDocuments(document)
      : [document];
    const candidates = [];
    docs.forEach(function (doc) {
      if (!doc) return;
      [
        doc.getElementById && doc.getElementById("draft"),
        doc.getElementById && doc.getElementById("draftapp"),
        doc.getElementById && doc.getElementById("root"),
        doc.querySelector && doc.querySelector("[data-reactroot]"),
        doc.body,
        doc.getElementById && doc.getElementById("app"),
        doc.getElementById && doc.getElementById("__next"),
        doc.querySelector && doc.querySelector("main"),
      ].forEach(function (el) { if (el) candidates.push(el); });
      try {
        const extra = doc.querySelectorAll("div[id],section");
        for (let i = 0; i < Math.min(extra.length, 40); i++) candidates.push(extra[i]);
      } catch (_e) { /* ignore */ }
    });
    for (const el of candidates) {
      for (const key of Object.keys(el)) {
        if (
          key.startsWith("__reactFiber$") ||
          key.startsWith("__reactInternalInstance$") ||
          key.startsWith("__reactContainer$") ||
          key.startsWith("_reactRootContainer")
        ) {
          try {
            roots.push(el[key]);
          } catch (_e) {
            /* ignore */
          }
        }
      }
    }
    let hit = false;
    for (const r of roots) {
      if (walkForPicks(r)) hit = true;
    }
    return hit;
  }

  function inspectJson(data, source) {
    if (!data) return;
    if (Array.isArray(data)) {
      if (data.some(isPickRow)) emit(data, { inProgress: true, drafted: false }, source);
      for (const item of data) inspectJson(item, source);
      return;
    }
    if (typeof data !== "object") return;
    rememberYahooPlayer(data);
    rememberYahooUser(data);
    if (data.draft_results || data.draftResults) {
      maybeFromDraftResults(data.draft_results || data.draftResults, source, null);
    }
    if (Array.isArray(data.picks) && data.picks.some(isPickRow)) {
      emit(data.picks, { inProgress: data.inProgress !== false, drafted: data.drafted === true }, source);
    }
    if (Array.isArray(data.draftPicks) && data.draftPicks.some(isPickRow)) {
      emit(data.draftPicks, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.selections) && data.selections.some(isPickRow)) {
      emit(data.selections, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.pickHistory) && data.pickHistory.some(isPickRow)) {
      emit(data.pickHistory, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.draftBoard) && data.draftBoard.some(isPickRow)) {
      emit(data.draftBoard, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.pickedPlayers) && data.pickedPlayers.some(isPickRow)) {
      emit(data.pickedPlayers, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.draftedPlayers) && data.draftedPlayers.some(isPickRow)) {
      emit(data.draftedPlayers, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.drafted) && data.drafted.some(isPickRow)) {
      emit(data.drafted, { inProgress: true, drafted: false }, source);
    }
    if (Array.isArray(data.takenPlayers) && data.takenPlayers.some(isPickRow)) {
      emit(data.takenPlayers, { inProgress: true, drafted: false }, source);
    }
    if (isPickRow(data)) {
      emit([data], { inProgress: true, drafted: false }, source);
    }
    if (data.fantasy_content) inspectJson(data.fantasy_content, source);
    if (data.league && data.league !== data) inspectJson(data.league, source);
    for (const key of ["draft", "data", "payload", "result", "message", "pick"]) {
      if (data[key] && typeof data[key] === "object" && data[key] !== data) inspectJson(data[key], source);
    }
  }

  function looksLikeDraftPayload(text) {
    return (
      text.indexOf("draft_result") >= 0 ||
      text.indexOf("draftResults") >= 0 ||
      text.indexOf("player_key") >= 0 ||
      text.indexOf("playerKey") >= 0 ||
      text.indexOf("overallPick") >= 0 ||
      text.indexOf("player_id") >= 0 ||
      text.indexOf("playerId") >= 0 ||
      text.indexOf("firstName") >= 0 ||
      text.indexOf("nflPlayer") >= 0 ||
      text.indexOf("pickedPlayer") >= 0 ||
      text.indexOf("draftedPlayer") >= 0 ||
      text.indexOf("takenPlayer") >= 0 ||
      text.indexOf("\"drafted\"") >= 0 ||
      text.indexOf("\"pick\":") >= 0
    );
  }

  function inspectText(text, source) {
    if (!text || text.length < 8 || text.length > 2500000) return;
    if (!looksLikeDraftPayload(text)) return;
    try {
      let payload = text;
      if (payload.charAt(0) === "4" && payload.charAt(1) === "2") {
        const idx = payload.indexOf("[");
        if (idx >= 0) payload = payload.slice(idx);
      }
      inspectJson(JSON.parse(payload), source);
    } catch (_e) {
      /* ignore */
    }
  }

  function looksLikeYahooDraftUrl(url) {
    return /fantasysports\.yahoo|yahooapis\.com|sports\.yahoo\.com|draftclient|fantasy\.yahoo|yimg\.com/i.test(
      String(url || "")
    );
  }

  let pollInFlight = false;
  let lastHtmlPollAt = 0;

  function harvestPageJson() {
    const docs = window.BRDraftSlot && BRDraftSlot.sameOriginDocuments
      ? BRDraftSlot.sameOriginDocuments(document)
      : [document];
    docs.forEach(function (doc) {
      if (!doc || !doc.querySelectorAll) return;
      const scripts = doc.querySelectorAll("script");
      const limit = Math.min(scripts.length, 48);
      for (let i = 0; i < limit; i++) {
        const t = scripts[i].textContent || "";
        if (t.length < 40 || t.length > 2500000) continue;
        if (!looksLikeDraftPayload(t)) continue;
        const startO = t.indexOf("{");
        const startA = t.indexOf("[");
        const idx = startO >= 0 && (startA < 0 || startO < startA) ? startO : startA;
        if (idx >= 0) inspectText(t.slice(idx), "script");
      }
    });
    ["__PRELOADED_STATE__", "__NEXT_DATA__", "__INITIAL_STATE__", "__APP_STATE__"].forEach(function (k) {
      try {
        if (window[k]) inspectJson(window[k], "boot-" + k);
      } catch (_e) { /* ignore */ }
    });
  }

  function refetchSeenDraftUrls() {
    let entries = [];
    try {
      entries = performance.getEntriesByType("resource") || [];
    } catch (_e) {
      return;
    }
    const seen = {};
    for (let i = 0; i < entries.length; i++) {
      const url = String(entries[i].name || "");
      if (seen[url] || !looksLikeYahooDraftUrl(url)) continue;
      if (!/draft|pick|result|player/i.test(url)) continue;
      seen[url] = true;
      fetch(url, { credentials: "include", cache: "no-store" })
        .then(function (r) { return r.text(); })
        .then(function (t) { inspectText(t, "perf"); })
        .catch(function () {});
    }
  }

  function pollDraftResultPages() {
    const ids = leagueFromUrl();
    if (!ids.leagueId || pollInFlight) return;
    const now = Date.now();
    if (now - lastHtmlPollAt < 8000 && pickAccumulator.size > 0) return;
    lastHtmlPollAt = now;
    pollInFlight = true;
    const origin = location.origin;
    const lid = ids.leagueId;
    let leagueKey = "";
    try {
      const m = String(location.href || "").match(/(\d{3})\.l\.(\d+)/);
      if (m) leagueKey = m[1] + ".l." + m[2];
    } catch (_e) { /* ignore */ }
    if (!leagueKey) {
      try {
        const blob = String((document.body && document.body.innerText) || "").slice(0, 8000);
        const m = blob.match(/(\d{3})\.l\.(\d+)/);
        if (m) leagueKey = m[1] + ".l." + m[2];
      } catch (_e) { /* ignore */ }
    }
    const urls = [
      origin + "/f1/" + lid + "/draftresults",
      origin + "/f1/" + lid + "/draftresults?format=json",
      origin + "/f1/" + lid + "/draftresults?xhr=1",
    ];
    if (leagueKey) {
      urls.push(origin + "/sitedirectory/fantasy/resource/league/draftresults?league_key=" + encodeURIComponent(leagueKey));
    }
    let i = 0;
    function next() {
      if (i >= urls.length) {
        pollInFlight = false;
        return;
      }
      const url = urls[i++];
      fetch(url, { credentials: "include", cache: "no-store" })
        .then(function (r) { return r.text(); })
        .then(function (t) {
          inspectText(t, "draftresults");
          if (window.BRDraftSlot && BRDraftSlot.parseYahooDraftResultsHtml) {
            const rows = BRDraftSlot.parseYahooDraftResultsHtml(t);
            if (rows && rows.length) emit(rows, { inProgress: true, drafted: false }, "draftresults-html");
          }
          if (pickAccumulator.size < 8) next();
          else pollInFlight = false;
        })
        .catch(function () { next(); });
    }
    next();
  }

  function scrapeYahooDomPicks() {
    if (!document.body) return false;
    const helper = window.BRDraftSlot;
    const scraped = helper && helper.scrapeYahooBoard
      ? helper.scrapeYahooBoard(document, detectedTeams)
      : [];
    if (!scraped || !scraped.length) return false;
    emit(scraped, { inProgress: true, drafted: false }, "dom-scrape");
    return true;
  }

  function scanAll() {
    harvestPageJson();
    refetchSeenDraftUrls();
    scrapeYahooDomPicks();
    scanReact();
    const clockDone = window.BRDraftSlot && BRDraftSlot.completedFromYahooClock
      ? BRDraftSlot.completedFromYahooClock(detectedTeams || 12)
      : -1;
    if (clockDone >= 0 && clockDone <= 40 && pickAccumulator.size > clockDone + 6) {
      pickAccumulator.clear();
      bestOverallSeen = 0;
      lastFingerprint = "";
    }
    if (pickAccumulator.size < 8 && !(clockDone >= 0 && clockDone < 8)) pollDraftResultPages();
    if (pickAccumulator.size) emitAccumulated({ inProgress: true, drafted: false }, "scan");
  }

  let domScrapeTimer = null;
  function scheduleDomScrape() {
    if (domScrapeTimer) return;
    domScrapeTimer = setTimeout(function () {
      domScrapeTimer = null;
      scrapeYahooDomPicks();
    }, 300);
  }

  function watchDom() {
    if (window.__brFantasyYahooDomWatch || !document.documentElement) return;
    window.__brFantasyYahooDomWatch = true;
    try {
      const mo = new MutationObserver(function () {
        if (document.hidden) return;
        scheduleDomScrape();
      });
      mo.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
    } catch (_e) {
      /* ignore */
    }
  }

  function hookNetwork() {
    if (window.__brFantasyYahooFetchHooked) return;
    window.__brFantasyYahooFetchHooked = true;

    const origFetch = window.fetch;
    if (typeof origFetch === "function") {
      window.fetch = function () {
        const args = arguments;
        return origFetch.apply(this, args).then((res) => {
          try {
            const url = String((args[0] && args[0].url) || args[0] || "");
            if (looksLikeYahooDraftUrl(url)) {
              res
                .clone()
                .text()
                .then((text) => inspectText(text, "fetch"))
                .catch(() => {});
            }
          } catch (_e) {
            /* ignore */
          }
          return res;
        });
      };
    }

    const XO = XMLHttpRequest.prototype.open;
    const XS = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function (method, url) {
      this.__brUrl = url;
      return XO.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function () {
      this.addEventListener("load", function () {
        try {
          const url = String(this.__brUrl || "");
          if (!looksLikeYahooDraftUrl(url)) return;
          const text = typeof this.responseText === "string" ? this.responseText : "";
          inspectText(text, "xhr");
        } catch (_e) {
          /* ignore */
        }
      });
      return XS.apply(this, arguments);
    };

    const OrigWS = window.WebSocket;
    if (typeof OrigWS === "function") {
      window.WebSocket = function (url, protocols) {
        const ws = protocols !== undefined ? new OrigWS(url, protocols) : new OrigWS(url);
        ws.addEventListener("message", function (ev) {
          try {
            const text = typeof ev.data === "string" ? ev.data : "";
            inspectText(text, "ws");
          } catch (_e) {
            /* ignore */
          }
        });
        return ws;
      };
      window.WebSocket.prototype = OrigWS.prototype;
    }
  }

  leagueFromUrl();
  hookNetwork();
  watchDom();
  function onRescan() {
    lastFingerprint = "";
    scanAll();
  }
  window.addEventListener("message", (ev) => {
    if (!ev.data || ev.data.__br !== BRIDGE || ev.data.type !== RESCAN) return;
    onRescan();
  });
  document.addEventListener(RESCAN, onRescan);
  setInterval(() => {
    if (document.hidden) return;
    scanAll();
  }, 1200);
  setTimeout(scanAll, 400);
  setTimeout(scanAll, 1500);
  setTimeout(scanAll, 4000);
  window.__brFantasyYahooForceScan = onRescan;
})();
