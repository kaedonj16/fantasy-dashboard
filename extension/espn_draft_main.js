// Runs in the MAIN world on ESPN's draft room so it can see React fiber state,
// page fetch/XHR/WebSocket traffic, and poll mDraftDetail with the user's session.
// Forwards pick snapshots to the extension via chrome.runtime + postMessage.

(function () {
  "use strict";

  if (window.__brFantasyEspnDraftObserver) return;
  // Disney/registerdisney iframes also match the content-script URL pattern; skip
  // them so React/JSON walks never touch cross-origin Window objects.
  try {
    if (window.top !== window) return;
    if (!/fantasy\.espn\.com$/i.test(String(location.hostname || ""))) return;
  } catch (_e) {
    return;
  }

  function isEspnDraftRoom() {
    try {
      const path = String(location.pathname || "").toLowerCase();
      if (/mockdraftlobby|draftlobby/.test(path)) return false;
      if (/(?:^|\/)(?:live)?draft(?:\/|$)/.test(path)) return true;
      if (/(?:^|\/)mockdraft(?:\/|$)/.test(path)) return true;
      return false;
    } catch (_e) {
      return false;
    }
  }

  const EVENT = "brfantasy:espn-draft-raw";
  const RESCAN = "brfantasy:draft-rescan";
  const RELAY_STATUS = "brfantasy:espn-relay-status";
  const OBSERVER_READY = "brfantasy:espn-observer-ready";
  const BRIDGE = "brfantasy-bridge-v1";
  const MAX_WALK = 12000;
  const MAX_ELEM_SCAN = 1200;
  let lastFingerprint = "";
  let lastEmitAt = 0;
  let apiPollInFlight = false;
  /** @type {Map<number, object>} */
  const pickAccumulator = new Map();
  /** @type {Map<number, Set<string>>} */
  const pickSources = new Map();
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
  /** @type {Map<string, {playerName: string, pos: string, nflTeam: string}>} */
  const playerMetaById = new Map();
  const ESPN_POS = { 1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST" };

  function isTraversableObject(val) {
    try {
      if (!val || typeof val !== "object") return false;
      if (Array.isArray(val)) return true;
      if (typeof Window !== "undefined" && val instanceof Window) return false;
      if (typeof Location !== "undefined" && val instanceof Location) return false;
      const tag = Object.prototype.toString.call(val);
      if (
        tag === "[object Window]" ||
        tag === "[object HTMLDocument]" ||
        tag === "[object Document]" ||
        tag === "[object Location]"
      ) {
        return false;
      }
      if (typeof HTMLIFrameElement !== "undefined" && val instanceof HTMLIFrameElement) {
        return false;
      }
      if (typeof Element !== "undefined" && val instanceof Element) return false;
      if (typeof Node !== "undefined" && val instanceof Node) return false;
      return true;
    } catch (_e) {
      return false;
    }
  }

  function safeProp(obj, key) {
    try {
      if (obj == null || typeof obj !== "object") return undefined;
      if (typeof Window !== "undefined" && obj instanceof Window) return undefined;
      if (typeof Location !== "undefined" && obj instanceof Location) return undefined;
      return obj[key];
    } catch (_e) {
      return undefined;
    }
  }

  function safeKeys(obj) {
    try {
      if (!isTraversableObject(obj)) return [];
      return Object.keys(obj);
    } catch (_e) {
      return [];
    }
  }

  function bridgeToExtension(type, detail) {
    try {
      window.postMessage({ __br: BRIDGE, type: type, detail: detail || {} }, "*");
    } catch (_e) {
      /* ignore */
    }
  }

  function leagueFromUrl() {
    try {
      const u = new URL(location.href);
      let leagueId = (u.searchParams.get("leagueId") || u.searchParams.get("league") || "").trim();
      let season =
        (u.searchParams.get("seasonId") || u.searchParams.get("season") || "").trim();
      if (!leagueId) {
        const hm = u.hash.match(/[?&]leagueId=(\d+)/i);
        if (hm) leagueId = hm[1];
      }
      if (!leagueId) {
        const pm = u.pathname.match(/\/(?:football\/)?draft\/(?:league\/)?(\d+)/i);
        if (pm) leagueId = pm[1];
      }
      if (!leagueId) {
        const gm = u.pathname.match(/\/(?:games\/)?draft[^/]*\/(\d{6,})/i);
        if (gm) leagueId = gm[1];
      }
      if (!leagueId) {
        const el = document.querySelector("[data-league-id],[data-leagueid]");
        if (el) {
          leagueId = (
            el.getAttribute("data-league-id") ||
            el.getAttribute("data-leagueid") ||
            ""
          ).trim();
        }
      }
      if (!leagueId) {
        const html = document.documentElement && document.documentElement.innerHTML;
        if (html) {
          const m = html.match(/["']leagueId["']\s*:\s*(\d{5,})/i);
          if (m) leagueId = m[1];
        }
      }
      if (!season) {
        const y = new Date().getFullYear();
        season = String(y);
      }
      return { leagueId, season };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function playerIdSelected(pid) {
    if (pid == null) return false;
    const text = String(pid).trim();
    if (!text || text === "0" || text === "-1" || text === "None" || text === "null") return false;
    return true;
  }

  function pickPlayerId(obj) {
    if (!isTraversableObject(obj)) return null;
    const pool = safeProp(obj, "playerPoolEntry") || safeProp(obj, "player_pool_entry");
    const poolPlayer = pool && safeProp(pool, "player");
    const fromPool =
      pool &&
      (safeProp(pool, "playerId") ??
        safeProp(pool, "player_id") ??
        safeProp(pool, "id") ??
        (poolPlayer &&
          (safeProp(poolPlayer, "id") ?? safeProp(poolPlayer, "playerId"))));
    const player = safeProp(obj, "player");
    const fromPlayer = player && (safeProp(player, "id") ?? safeProp(player, "playerId"));
    const explicit =
      safeProp(obj, "playerId") ??
      safeProp(obj, "player_id") ??
      safeProp(obj, "athleteId") ??
      safeProp(obj, "athlete_id") ??
      fromPool ??
      fromPlayer;
    if (explicit != null) return explicit;
    // Empty ESPN seats have overallPickNumber + a row `id` that is not a player.
    if (
      safeProp(obj, "overallPickNumber") != null ||
      safeProp(obj, "overallPickNo") != null ||
      safeProp(obj, "overallPick") != null ||
      safeProp(obj, "pickNumber") != null
    ) {
      return null;
    }
    return safeProp(obj, "id");
  }

  function pickOverall(obj) {
    if (!isTraversableObject(obj)) return null;
    return (
      safeProp(obj, "overallPickNumber") ??
      safeProp(obj, "overallPickNo") ??
      safeProp(obj, "overallPick") ??
      safeProp(obj, "overall_pick_number") ??
      safeProp(obj, "pickNumber") ??
      safeProp(obj, "pick_no") ??
      safeProp(obj, "pick")
    );
  }

  function isPlaceholderPlayerName(name) {
    return /^pick\s*#?\s*\d+$/i.test(String(name || "").trim());
  }

  function pickLooksMade(p) {
    if (!p) return false;
    const name = String(p.playerName || "").trim();
    if (isPlaceholderPlayerName(name)) return false;
    if (playerIdSelected(p.playerId)) return true;
    return !!name;
  }

  function isPickRow(obj) {
    if (!isTraversableObject(obj)) return false;
    const pid = pickPlayerId(obj);
    const overall = pickOverall(obj);
    if (overall == null || !playerIdSelected(pid)) return false;
    const name = playerNameFrom(obj);
    if (isPlaceholderPlayerName(name)) return false;
    return true;
  }

  function playerNameFrom(obj, depth) {
    if (!isTraversableObject(obj) || (depth || 0) > 4) return "";
    const direct =
      safeProp(obj, "playerName") ||
      safeProp(obj, "player_name") ||
      safeProp(obj, "fullName") ||
      safeProp(obj, "full_name") ||
      safeProp(obj, "displayName") ||
      safeProp(obj, "display_name");
    if (direct) return String(direct).trim();
    const first = safeProp(obj, "firstName") || safeProp(obj, "first_name") || "";
    const last = safeProp(obj, "lastName") || safeProp(obj, "last_name") || "";
    const joined = (String(first || "") + " " + String(last || "")).trim();
    if (joined) return joined;
    const nameObj = safeProp(obj, "name");
    if (isTraversableObject(nameObj)) {
      const full = safeProp(nameObj, "full") || safeProp(nameObj, "fullName");
      if (full) return String(full).trim();
      const n = (
        String(safeProp(nameObj, "first") || safeProp(nameObj, "firstName") || "") +
        " " +
        String(safeProp(nameObj, "last") || safeProp(nameObj, "lastName") || "")
      ).trim();
      if (n) return n;
    }
    const player = safeProp(obj, "player");
    if (player && player !== obj) {
      const nested = playerNameFrom(player, (depth || 0) + 1);
      if (nested) return nested;
    }
    const pool = safeProp(obj, "playerPoolEntry") || safeProp(obj, "player_pool_entry");
    if (pool && pool !== obj) {
      const nested = playerNameFrom(pool, (depth || 0) + 1);
      if (nested) return nested;
    }
    return "";
  }

  function playerPosFrom(obj, depth) {
    if (!isTraversableObject(obj) || (depth || 0) > 4) return "";
    const raw =
      safeProp(obj, "pos") ||
      safeProp(obj, "position") ||
      safeProp(obj, "defaultPosition") ||
      safeProp(obj, "eligiblePosition");
    if (raw && typeof raw === "string") return String(raw).toUpperCase().replace("D/ST", "DST");
    const id = safeProp(obj, "defaultPositionId") || safeProp(obj, "default_position_id");
    if (id != null && ESPN_POS[Number(id)]) return ESPN_POS[Number(id)];
    const player = safeProp(obj, "player");
    if (player && player !== obj) {
      const nested = playerPosFrom(player, (depth || 0) + 1);
      if (nested) return nested;
    }
    const pool = safeProp(obj, "playerPoolEntry") || safeProp(obj, "player_pool_entry");
    if (pool && pool !== obj) return playerPosFrom(pool, (depth || 0) + 1);
    return "";
  }

  function playerNflTeamFrom(obj, depth) {
    if (!isTraversableObject(obj) || (depth || 0) > 4) return "";
    const raw =
      safeProp(obj, "nflTeam") ||
      safeProp(obj, "proTeam") ||
      safeProp(obj, "proTeamAbbreviation") ||
      safeProp(obj, "teamAbbr") ||
      safeProp(obj, "proTeamAbbrev");
    if (raw && typeof raw === "string" && /[A-Za-z]/.test(raw)) {
      return String(raw).toUpperCase().slice(0, 3);
    }
    const player = safeProp(obj, "player");
    if (player && player !== obj) {
      const nested = playerNflTeamFrom(player, (depth || 0) + 1);
      if (nested) return nested;
    }
    const pool = safeProp(obj, "playerPoolEntry") || safeProp(obj, "player_pool_entry");
    if (pool && pool !== obj) return playerNflTeamFrom(pool, (depth || 0) + 1);
    return "";
  }

  function rememberPlayerMeta(obj) {
    if (!isTraversableObject(obj)) return;
    const pid = pickPlayerId(obj);
    if (!playerIdSelected(pid)) return;
    const name = playerNameFrom(obj);
    const pos = playerPosFrom(obj);
    const nfl = playerNflTeamFrom(obj);
    if (!name && !pos && !nfl) return;
    const prev = playerMetaById.get(String(pid)) || {};
    playerMetaById.set(String(pid), {
      playerName: name || prev.playerName || "",
      pos: pos || prev.pos || "",
      nflTeam: nfl || prev.nflTeam || "",
    });
  }

  function normalizePick(raw) {
    if (!isPickRow(raw)) return null;
    rememberPlayerMeta(raw);
    const playerId = pickPlayerId(raw);
    const overall = pickOverall(raw);
    const team = safeProp(raw, "team");
    const teamId = safeProp(raw, "teamId") ?? safeProp(raw, "team_id") ?? (team && safeProp(team, "id"));
    const roundId = safeProp(raw, "roundId") ?? safeProp(raw, "round") ?? safeProp(raw, "round_id");
    const roundPick =
      safeProp(raw, "roundPickNumber") ??
      safeProp(raw, "roundPick") ??
      safeProp(raw, "round_pick") ??
      safeProp(raw, "slot");
    const cached = playerId != null ? playerMetaById.get(String(playerId)) : null;
    const playerName = playerNameFrom(raw) || (cached && cached.playerName) || "";
    const pos = playerPosFrom(raw) || (cached && cached.pos) || "";
    const nflTeam = playerNflTeamFrom(raw) || (cached && cached.nflTeam) || "";
    return {
      overallPickNumber: Number(overall),
      playerId: playerId == null ? null : playerId,
      playerName: playerName || undefined,
      pos: pos || undefined,
      nflTeam: nflTeam || undefined,
      teamId: teamId == null ? null : teamId,
      roundId: roundId == null ? null : Number(roundId),
      roundPickNumber: roundPick == null ? null : Number(roundPick),
      keeper: !!(safeProp(raw, "keeper") || safeProp(raw, "reservedForKeeper") || safeProp(raw, "isKeeper")),
      bidAmount: safeProp(raw, "bidAmount") != null ? safeProp(raw, "bidAmount") : null,
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

  function espnSwid() {
    try {
      const m = document.cookie.match(/(?:^|; )SWID=([^;]*)/i);
      return m ? decodeURIComponent(m[1]).replace(/[{}]/g, "").toLowerCase() : "";
    } catch (_e) {
      return "";
    }
  }

  function rosterRoundsFromLineupSlots(counts) {
    if (!counts || typeof counts !== "object") return 0;
    let n = 0;
    const keys = safeKeys(counts);
    for (let i = 0; i < keys.length; i++) {
      const id = Number(keys[i]);
      if (id === 21 || id === 22) continue;
      const c = Number(safeProp(counts, keys[i]) || counts[keys[i]]) || 0;
      if (c > 0) n += c;
    }
    return n >= 6 && n <= 40 ? n : 0;
  }

  function rosterFromEspnSlots(counts) {
    const out = { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, K: 0, DEF: 0, BN: 0 };
    if (!counts || typeof counts !== "object") return out;
    const map = { 0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "SF", 16: "DEF", 17: "K", 20: "BN", 23: "FLEX", 3: "FLEX", 5: "FLEX" };
    const keys = safeKeys(counts);
    for (let i = 0; i < keys.length; i++) {
      const id = Number(keys[i]);
      if (id === 21 || id === 22) continue;
      const dest = map[id];
      const n = Number(safeProp(counts, keys[i]) || counts[keys[i]]) || 0;
      if (dest && n > 0) out[dest] += n;
    }
    return out;
  }

  function rosterFingerprint(rs) {
    if (!rs) return "";
    return ["QB", "SF", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN"].map(function (k) {
      return k + (Number(rs[k]) || 0);
    }).join("");
  }

  function scoringFromEspnSettings(settings) {
    const out = { ppr: 1, passTd: 4, tep: 0 };
    if (!isTraversableObject(settings)) return out;
    const scoring = safeProp(settings, "scoringSettings") || safeProp(settings, "scoring_settings") || settings;
    const items = scoring && (safeProp(scoring, "scoringItems") || safeProp(scoring, "scoring_items"));
    if (Array.isArray(items)) {
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (!isTraversableObject(it)) continue;
        const id = Number(safeProp(it, "statId") != null ? safeProp(it, "statId") : safeProp(it, "stat_id"));
        const pts = Number(safeProp(it, "points") != null ? safeProp(it, "points") : safeProp(it, "pts"));
        if (!Number.isFinite(pts)) continue;
        if (id === 53) out.ppr = pts;
        if (id === 4) out.passTd = pts;
      }
    }
    const bonus = scoring && (safeProp(scoring, "bonusScoringItems") || safeProp(scoring, "bonus_scoring_items"));
    if (Array.isArray(bonus)) {
      for (let i = 0; i < bonus.length; i++) {
        const it = bonus[i];
        if (!isTraversableObject(it)) continue;
        const id = Number(safeProp(it, "statId") != null ? safeProp(it, "statId") : safeProp(it, "stat_id"));
        const pts = Number(safeProp(it, "points") != null ? safeProp(it, "points") : safeProp(it, "pts"));
        const elig = safeProp(it, "eligiblePositionIds") || safeProp(it, "eligible_position_ids") || [];
        const te = Array.isArray(elig) && elig.some(function (p) { return Number(p) === 6; });
        if (id === 53 && te && Number.isFinite(pts) && pts > 0) out.tep = pts;
      }
    }
    return out;
  }

  function rememberEspnSettings(obj) {
    if (!isTraversableObject(obj)) return;
    const size = safeProp(obj, "size");
    if (size >= 4 && size <= 32) detectedTeams = Number(size);
    const settings = safeProp(obj, "settings") || obj;
    if (isTraversableObject(settings)) {
      const sz = safeProp(settings, "size");
      if (sz >= 4 && sz <= 32) detectedTeams = Number(sz);
      const roster = safeProp(settings, "rosterSettings") || safeProp(obj, "rosterSettings");
      const counts = roster && (safeProp(roster, "lineupSlotCounts") || safeProp(roster, "lineup_slot_counts"));
      const r = rosterRoundsFromLineupSlots(counts);
      if (r) detectedRounds = r;
      if (counts) {
        const mapped = rosterFromEspnSlots(counts);
        if ((mapped.QB || 0) + (mapped.RB || 0) + (mapped.WR || 0) + (mapped.TE || 0) + (mapped.FLEX || 0) + (mapped.SF || 0) >= 4) {
          detectedRoster = mapped;
          detectedSf = (mapped.SF || 0) > 0;
        }
      }
      const scoring = scoringFromEspnSettings(settings);
      if (scoring) {
        detectedPpr = scoring.ppr;
        detectedTep = scoring.tep;
        detectedPassTd = scoring.passTd;
      }
    }
  }

  function rememberEspnUser(obj) {
    if (!isTraversableObject(obj)) return;
    rememberEspnSettings(obj);
    const uid =
      safeProp(obj, "userTeamId") ??
      safeProp(obj, "myTeamId") ??
      safeProp(obj, "currentUserTeamId") ??
      safeProp(obj, "viewerTeamId");
    if (uid != null && uid !== "" && Number(uid) !== 0) detectedUserTeamId = uid;
    const teams = safeProp(obj, "teams");
    if (Array.isArray(teams) && teams.length >= 4 && teams.length <= 32) {
      detectedTeams = teams.length;
    }
    if (!Array.isArray(teams)) return;
    const swid = espnSwid();
    for (let i = 0; i < teams.length; i++) {
      const t = teams[i];
      if (!isTraversableObject(t)) continue;
      const id = safeProp(t, "id") ?? safeProp(t, "teamId");
      const owner = String(safeProp(t, "primaryOwner") || "").replace(/[{}]/g, "").toLowerCase();
      const isUser =
        safeProp(t, "isCurrentUser") === true ||
        safeProp(t, "isUser") === true ||
        safeProp(t, "isUserTeam") === true ||
        (swid && owner && owner === swid);
      if (!isUser || id == null || id === "") continue;
      detectedUserTeamId = id;
      const pos =
        safeProp(t, "draftPosition") ??
        safeProp(t, "draftSlot") ??
        safeProp(t, "pickNumber") ??
        safeProp(t, "draftPickNumber");
      if (pos != null && Number(pos) >= 1) detectedSlot = Number(pos);
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
    let roundPick = 0;
    (picks || []).forEach(function (p) {
      if (p && p.teamId != null && p.teamId !== "") ids[String(p.teamId)] = true;
      if (!p || String(p.teamId) !== want) return;
      const n = Number(p.overallPickNumber || 0);
      if (n && (!first || n < first)) first = n;
      if (Number(p.roundId || p.round) === 1 && Number(p.roundPickNumber) >= 1) {
        roundPick = Number(p.roundPickNumber);
      }
    });
    const teams = Object.keys(ids).length >= 4 ? Object.keys(ids).length : 12;
    if (roundPick >= 1 && roundPick <= teams) return roundPick;
    return first ? snakeSlotFromPick(first, teams) : 0;
  }

  function relayToBackground(detail) {
    if (!detail || !detail.leagueId) return;
    try {
      chrome.runtime.sendMessage(
        {
          type: "espnDraftRelay",
          leagueId: detail.leagueId,
          season: detail.season || "",
          inProgress: !!detail.inProgress,
          drafted: !!detail.drafted,
          picks: Array.isArray(detail.picks) ? detail.picks : [],
          source: detail.source || "espn-draft-room",
          at: detail.at || Date.now(),
          forceReplay: true,
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

  function isDomSource(source) {
    return String(source || "").indexOf("dom") >= 0;
  }

  function trustedMaxOverall() {
    let max = 0;
    for (const [n, sources] of pickSources.entries()) {
      let trusted = false;
      for (const src of sources) {
        if (!isDomSource(src)) {
          trusted = true;
          break;
        }
      }
      if (trusted && n > max) max = n;
    }
    return max;
  }

  function mergeIntoAccumulator(rawPicks, source) {
    let grew = false;
    for (const raw of rawPicks || []) {
      const norm = normalizePick(raw);
      if (!norm || !norm.overallPickNumber) continue;
      const n = Number(norm.overallPickNumber);
      if (!n || n <= 0 || !pickLooksMade(norm)) continue;
      if (!pickAccumulator.has(n)) grew = true;
      const prev = pickAccumulator.get(n);
      if (prev) {
        if (prev.playerName && !norm.playerName) norm.playerName = prev.playerName;
        if (prev.pos && !norm.pos) norm.pos = prev.pos;
        if (prev.nflTeam && !norm.nflTeam) norm.nflTeam = prev.nflTeam;
      }
      pickAccumulator.set(n, norm);
      if (!pickSources.has(n)) pickSources.set(n, new Set());
      pickSources.get(n).add(String(source || "unknown"));
      if (n > bestOverallSeen) bestOverallSeen = n;
    }
    return grew;
  }

  function emitAccumulated(meta, source) {
    const ids = leagueFromUrl();
    const trustedMax = trustedMaxOverall();
    let clean = Array.from(pickAccumulator.values()).sort(
      (a, b) => a.overallPickNumber - b.overallPickNumber
    );
    clean = clean.filter(pickLooksMade);
    if (trustedMax > 0) {
      clean = clean.filter(function (p) {
        return p.overallPickNumber <= trustedMax;
      });
    }
    for (const n of Array.from(pickAccumulator.keys())) {
      if (!pickLooksMade(pickAccumulator.get(n))) {
        pickAccumulator.delete(n);
        pickSources.delete(n);
      }
    }
    if (!clean.length && !lastFingerprint) return;
    const fp = fingerprint(clean, meta || {});
    const now = Date.now();
    if (fp === lastFingerprint && now - lastEmitAt < 1500) return;
    lastFingerprint = fp;
    lastEmitAt = now;
    const mySlot = computeMySlot(clean);
    const detail = {
      source: source || "accumulated",
      leagueId: ids.leagueId,
      season: ids.season,
      inProgress: !!(meta && meta.inProgress),
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
      at: now,
    };
    bridgeToExtension(EVENT, detail);
    relayToBackground(detail);
  }

  function emit(picks, meta, source) {
    if (!mergeIntoAccumulator(picks, source)) {
      const incoming = (picks || []).map(normalizePick).filter(Boolean);
      if (!incoming.length) return;
      const maxIncoming = incoming[incoming.length - 1].overallPickNumber;
      if (maxIncoming <= bestOverallSeen && pickAccumulator.size >= incoming.length) return;
    }
    emitAccumulated(meta, source);
  }

  function maybeFromDraftDetail(detail, source) {
    if (!isTraversableObject(detail)) return false;
    const picks = safeProp(detail, "picks");
    if (!Array.isArray(picks) || !picks.length) return false;
    const selected = picks.filter(isPickRow);
    if (!selected.length) return false;
    emit(
      selected,
      {
        inProgress:
          safeProp(detail, "inProgress") === true || safeProp(detail, "in_progress") === true,
        drafted: safeProp(detail, "drafted") === true,
      },
      source
    );
    return true;
  }

  function findBestDraftDetail(data, depth, best) {
    if (!isTraversableObject(data)) return best;
    if (depth == null) depth = 0;
    if (!best) best = { detail: null, count: 0 };
    if (depth > 16) return best;
    if (Array.isArray(data)) {
      for (let i = 0; i < Math.min(data.length, 32); i++) {
        best = findBestDraftDetail(data[i], depth + 1, best);
      }
      return best;
    }
    const draftDetail = safeProp(data, "draftDetail");
    if (draftDetail && typeof draftDetail === "object") {
      const picks = safeProp(draftDetail, "picks");
      if (Array.isArray(picks)) {
        const sel = picks.filter(isPickRow);
        if (sel.length > best.count) {
          best = { detail: draftDetail, count: sel.length };
        }
      }
    }
    const topPicks = safeProp(data, "picks");
    if (Array.isArray(topPicks) && topPicks.some(isPickRow)) {
      const sel = topPicks.filter(isPickRow);
      if (sel.length > best.count) {
        best = { detail: data, count: sel.length };
      }
    }
    for (const k of safeKeys(data)) {
      if (k === "draftDetail") continue;
      const v = safeProp(data, k);
      if (isTraversableObject(v)) {
        best = findBestDraftDetail(v, depth + 1, best);
      }
    }
    return best;
  }

  function deepFindDraftDetail(data, depth) {
    const best = findBestDraftDetail(data, depth, null);
    return best && best.detail ? best.detail : null;
  }

  function inspectJson(data, source) {
    if (!isTraversableObject(data)) return;
    rememberEspnUser(data);
    const players = safeProp(data, "players");
    if (Array.isArray(players)) {
      for (let i = 0; i < Math.min(players.length, 1200); i++) rememberPlayerMeta(players[i]);
    }
    const detail = deepFindDraftDetail(data) || safeProp(data, "draftDetail") || null;
    if (detail && maybeFromDraftDetail(detail, source)) return;
    if (Array.isArray(data)) {
      for (const item of data) inspectJson(item, source);
      return;
    }
    const picks = safeProp(data, "picks");
    if (Array.isArray(picks) && picks.some(isPickRow)) {
      emit(
        picks.filter(isPickRow),
        {
          inProgress: safeProp(data, "inProgress") === true,
          drafted: safeProp(data, "drafted") === true,
        },
        source + "-picks"
      );
    }
  }

  function walkForDraftDetail(root) {
    const seen = new Set();
    const q = [root];
    let n = 0;
    let found = false;
    while (q.length && n < MAX_WALK) {
      const cur = q.shift();
      n++;
      if (!isTraversableObject(cur)) continue;
      if (seen.has(cur)) continue;
      seen.add(cur);
      rememberEspnUser(cur);
      const draftDetail = safeProp(cur, "draftDetail");
      if (draftDetail && maybeFromDraftDetail(draftDetail, "react")) found = true;
      const picks = safeProp(cur, "picks");
      if (Array.isArray(picks) && picks.some(isPickRow)) {
        emit(
          picks.filter(isPickRow),
          {
            inProgress: safeProp(cur, "inProgress") === true,
            drafted: safeProp(cur, "drafted") === true,
          },
          "react-picks"
        );
        found = true;
      }
      const next = [];
      const memoizedProps = safeProp(cur, "memoizedProps");
      const pendingProps = safeProp(cur, "pendingProps");
      const stateNode = safeProp(cur, "stateNode");
      const state = safeProp(cur, "state");
      const ret = safeProp(cur, "return");
      const child = safeProp(cur, "child");
      const sibling = safeProp(cur, "sibling");
      const props = safeProp(cur, "props");
      if (memoizedProps) next.push(memoizedProps);
      if (pendingProps) next.push(pendingProps);
      if (stateNode && isTraversableObject(stateNode)) next.push(stateNode);
      if (state) next.push(state);
      if (ret) next.push(ret);
      if (child) next.push(child);
      if (sibling) next.push(sibling);
      if (props) next.push(props);
      for (const k of safeKeys(cur)) {
        if (k === "draftDetail") {
          const dd = safeProp(cur, k);
          if (dd && maybeFromDraftDetail(dd, "react-key")) found = true;
        }
        const v = safeProp(cur, k);
        if (isTraversableObject(v) && !seen.has(v)) next.push(v);
        if (next.length > 48) break;
      }
      for (let i = 0; i < Math.min(next.length, 28); i++) q.push(next[i]);
    }
    return found;
  }

  function collectReactRoots() {
    const roots = [];
    const seenRoot = new Set();
    if (!document || !document.documentElement) return roots;
    const candidates = [
      document.getElementById("espn-aria-root"),
      document.getElementById("root"),
      document.querySelector("[data-reactroot]"),
      document.body,
    ].filter(Boolean);
    function pushRoot(node) {
      if (!node || seenRoot.has(node)) return;
      seenRoot.add(node);
      roots.push(node);
    }
    for (const el of candidates) {
      for (const key of Object.keys(el)) {
        if (
          key.startsWith("__reactFiber$") ||
          key.startsWith("__reactInternalInstance$") ||
          key.startsWith("_reactRootContainer")
        ) {
          try {
            pushRoot(el[key]);
          } catch (_e) {
            /* ignore */
          }
        }
      }
    }
    let scanned = 0;
    const nodes = document.querySelectorAll("*");
    for (const el of nodes) {
      if (scanned++ > MAX_ELEM_SCAN) break;
      for (const key of Object.keys(el)) {
        if (!key.startsWith("__reactFiber$") && !key.startsWith("__reactInternalInstance$")) continue;
        try {
          pushRoot(el[key]);
        } catch (_e) {
          /* ignore */
        }
      }
    }
    return roots;
  }

  function scanReact() {
    const roots = collectReactRoots();
    for (const r of roots) {
      if (walkForDraftDetail(r)) return true;
    }
    return false;
  }

  function playerIdFromImg(img) {
    if (!img) return null;
    const src = img.getAttribute("src") || img.getAttribute("data-src") || "";
    const m =
      src.match(/\/(?:full|scale)\/(-?\d+)\.(?:png|jpg|webp)/i) ||
      src.match(/playerId[=:/](-?\d+)/i);
    return m ? m[1] : null;
  }

  function parsePickLabel(text) {
    const s = String(text || "").replace(/\s+/g, " ");
    const rd = s.match(/\b(\d+)\.\s*(\d{1,2})\b/);
    if (rd) {
      const teams = guessTeamCount() || 12;
      return (parseInt(rd[1], 10) - 1) * teams + parseInt(rd[2], 10);
    }
    const ov = s.match(/(?:overall|pick)\s*#?\s*(\d{1,3})\b/i);
    if (ov) return parseInt(ov[1], 10);
    return null;
  }

  function guessTeamCount() {
    if (!document || !document.querySelectorAll) return 0;
    const headers = document.querySelectorAll(
      '[class*="team-column"], [class*="teamColumn"], [class*="draft-team"], [class*="team-header"]'
    );
    if (headers.length >= 4 && headers.length <= 32) return headers.length;
    return 0;
  }

  function reactPropsNear(el) {
    if (!el || el.nodeType !== 1) return null;
    for (const key of Object.keys(el)) {
      if (!key.startsWith("__reactFiber$") && !key.startsWith("__reactInternalInstance$")) continue;
      let node = el[key];
      for (let depth = 0; depth < 10 && node; depth++) {
        const props = node.memoizedProps || node.pendingProps || node.props;
        if (props) {
          if (isPickRow(props)) return props;
          if (props.pick && isPickRow(props.pick)) return props.pick;
          if (props.draftPick && isPickRow(props.draftPick)) return props.draftPick;
          if (props.player && props.overallPickNumber != null) return props;
        }
        node = node.return;
      }
    }
    return null;
  }

  function addDomPick(map, pickNo, playerId, teamId, sourceTag, extra) {
    if (!playerIdSelected(playerId) || pickNo == null || pickNo <= 0) return;
    const n = Number(pickNo);
    if (!map.has(n)) {
      map.set(n, Object.assign({
        overallPickNumber: n,
        playerId: playerId,
        teamId: teamId == null ? null : teamId,
        roundId: null,
        roundPickNumber: null,
        keeper: false,
        bidAmount: null,
        __source: sourceTag,
      }, extra || {}));
    } else if (extra) {
      const cur = map.get(n);
      if (extra.playerName && !cur.playerName) cur.playerName = extra.playerName;
      if (extra.pos && !cur.pos) cur.pos = extra.pos;
      if (extra.nflTeam && !cur.nflTeam) cur.nflTeam = extra.nflTeam;
    }
  }

  function scrapeDomPicks() {
    if (!document.body) return false;
    const byOverall = new Map();
    const scope =
      document.querySelector('[class*="draftContainer"], [class*="draft-container"], main, #root') ||
      document.body;
    if (!scope || typeof scope.querySelectorAll !== "function") return false;

    const cellSelector = [
      '[class*="player-column"]',
      '[class*="playerColumn"]',
      '[class*="pick-cell"]',
      '[class*="pickCell"]',
      '[class*="draft-pick"]',
      '[class*="draftPick"]',
      '[class*="pick-history"] [class*="row"]',
      '[class*="pickHistory"] [class*="row"]',
      '[class*="recent-pick"]',
      '[class*="recentPick"]',
      ".fixedDataTableRowLayout_rowWrapper",
    ].join(",");

    scope.querySelectorAll(cellSelector).forEach(function (cell) {
      const img = cell.querySelector('img[src*="headshot"], img[src*="players/full"]');
      const pid =
        (img && playerIdFromImg(img)) ||
        cell.getAttribute("data-player-id") ||
        cell.getAttribute("data-playerid");
      const pickNo =
        parsePickLabel(cell.textContent) ||
        parsePickLabel(cell.getAttribute("aria-label") || "") ||
        parseInt(cell.getAttribute("data-pick") || cell.getAttribute("data-overall-pick") || "", 10) ||
        null;
      const altName = (img && (img.getAttribute("alt") || img.getAttribute("title"))) || "";
      addDomPick(byOverall, pickNo, pid, null, "dom-cell", altName ? { playerName: String(altName).trim() } : null);
      const props = reactPropsNear(cell);
      if (props) {
        const norm = normalizePick(props);
        if (norm) {
          addDomPick(byOverall, norm.overallPickNumber, norm.playerId, norm.teamId, "dom-react", {
            playerName: norm.playerName,
            pos: norm.pos,
            nflTeam: norm.nflTeam,
          });
        }
      }
    });

    scope.querySelectorAll('[data-player-id], [data-playerid]').forEach(function (el) {
      const pid = el.getAttribute("data-player-id") || el.getAttribute("data-playerid");
      const pickNo =
        parsePickLabel(el.textContent) ||
        parseInt(el.getAttribute("data-pick") || el.getAttribute("data-overall-pick") || "", 10) ||
        null;
      addDomPick(byOverall, pickNo, pid, null, "dom-data");
    });

    const imgs = scope.querySelectorAll('img[src*="headshots/nfl/players"], img[src*="players/full"]');
    imgs.forEach(function (img) {
      const pid = playerIdFromImg(img);
      if (!playerIdSelected(pid)) return;
      let pickNo = null;
      let node = img.parentElement;
      for (let depth = 0; depth < 6 && node; depth++) {
        pickNo =
          parsePickLabel(node.textContent) ||
          parsePickLabel(node.getAttribute("aria-label") || "") ||
          parseInt(node.getAttribute("data-pick") || node.getAttribute("data-overall-pick") || "", 10) ||
          null;
        if (pickNo) break;
        node = node.parentElement;
      }
      addDomPick(byOverall, pickNo, pid, null, "dom-img");
    });

    if (!byOverall.size) return false;

    let picks = Array.from(byOverall.values());
    picks.sort(function (a, b) {
      return a.overallPickNumber - b.overallPickNumber;
    });

    const missingNo = picks.some(function (p) {
      return !p.overallPickNumber;
    });
    // Never invent pick numbers 1..N for headshots without labels (recent-pick strip).
    if (missingNo) {
      picks = picks.filter(function (p) {
        return p.overallPickNumber > 0;
      });
      if (!picks.length) return false;
    }

    picks = picks.map(function (p) {
      const out = Object.assign({}, p);
      delete out.__source;
      return out;
    });

    mergeIntoAccumulator(picks, "dom-scrape");
    emitAccumulated({ inProgress: true, drafted: false }, "dom-scrape");
    return true;
  }

  function scanAll() {
    if (!isEspnDraftRoom() || !document.body) return;
    scrapeDomPicks();
    scanReact();
  }

  let domScrapeTimer = null;
  function scheduleDomScrape() {
    if (domScrapeTimer) return;
    domScrapeTimer = setTimeout(function () {
      domScrapeTimer = null;
      scrapeDomPicks();
    }, 350);
  }

  function watchDom() {
    if (window.__brFantasyEspnDomWatch) return;
    window.__brFantasyEspnDomWatch = true;
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

  function espnApiUrls(leagueId, season) {
    const s = String(season || new Date().getFullYear());
    const lid = String(leagueId);
    const q = "?view=mDraftDetail&view=mSettings&view=mTeam";
    return [
      "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/" +
        encodeURIComponent(s) +
        "/segments/0/leagues/" +
        encodeURIComponent(lid) +
        q,
      "https://fantasy.espn.com/apis/v3/games/ffl/seasons/" +
        encodeURIComponent(s) +
        "/segments/0/leagues/" +
        encodeURIComponent(lid) +
        q,
      "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/leagueHistory/" +
        encodeURIComponent(lid) +
        "?view=mDraftDetail&view=mTeam&seasonId=" +
        encodeURIComponent(s),
    ];
  }

  function pollEspnApi() {
    if (apiPollInFlight || !isEspnDraftRoom()) return;
    const ids = leagueFromUrl();
    if (!ids.leagueId) return;
    apiPollInFlight = true;
    const urls = espnApiUrls(ids.leagueId, ids.season);
    let i = 0;
    function next() {
      if (i >= urls.length) {
        apiPollInFlight = false;
        return;
      }
      const url = urls[i++];
      fetch(url, { credentials: "include", cache: "no-store" })
        .then(function (res) {
          return res.ok ? res.json() : null;
        })
        .then(function (data) {
          if (data) inspectJson(data, "api-poll");
          next();
        })
        .catch(function () {
          next();
        });
    }
    next();
  }

  function looksLikeEspnFantasyUrl(url) {
    return /fantasy\.espn\.com|lm-api-reads\.fantasy\.espn\.com|lm-api\.fantasy\.espn\.com|\/apis\/v3\/games\/ffl/i.test(
      String(url || "")
    );
  }

  function hookNetwork() {
    if (window.__brFantasyEspnFetchHooked) return;
    window.__brFantasyEspnFetchHooked = true;

    const origFetch = window.fetch;
    if (typeof origFetch === "function") {
      window.fetch = function () {
        const args = arguments;
        return origFetch.apply(this, args).then(function (res) {
          try {
            const url = String((args[0] && args[0].url) || args[0] || "");
            if (looksLikeEspnFantasyUrl(url)) {
              res
                .clone()
                .json()
                .then(function (data) {
                  inspectJson(data, "fetch");
                })
                .catch(function () {});
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
          if (!looksLikeEspnFantasyUrl(url)) return;
          const text = typeof this.responseText === "string" ? this.responseText : "";
          if (
            text.indexOf("draftDetail") < 0 &&
            text.indexOf("overallPickNumber") < 0 &&
            text.indexOf("playerId") < 0
          ) {
            return;
          }
          const data = JSON.parse(text);
          inspectJson(data, "xhr");
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
            if (
              text.indexOf("draftDetail") < 0 &&
              text.indexOf("overallPickNumber") < 0 &&
              text.indexOf("playerId") < 0
            ) {
              return;
            }
            const data = JSON.parse(text);
            inspectJson(data, "ws");
          } catch (_e) {
            /* ignore */
          }
        });
        return ws;
      };
      window.WebSocket.prototype = OrigWS.prototype;
    }
  }

  function startEspnDraftObserver() {
    if (window.__brFantasyEspnDraftObserver) return;
    window.__brFantasyEspnDraftObserver = true;
    hookNetwork();
    watchDom();
    bridgeToExtension(OBSERVER_READY, { href: location.href, leagueId: leagueFromUrl().leagueId });

    function onRescan() {
      lastFingerprint = "";
      pickAccumulator.clear();
      pickSources.clear();
      bestOverallSeen = 0;
      scanAll();
      pollEspnApi();
    }
    window.addEventListener("message", function (ev) {
      if (!ev.data || ev.data.__br !== BRIDGE || ev.data.type !== RESCAN) return;
      onRescan();
    });
    document.addEventListener(RESCAN, onRescan);

    setInterval(function () {
      if (document.hidden || !isEspnDraftRoom()) return;
      scanAll();
    }, 2000);
    setInterval(function () {
      if (document.hidden || !isEspnDraftRoom()) return;
      pollEspnApi();
    }, 3000);

    setTimeout(onRescan, 800);
    setTimeout(onRescan, 2500);
    setTimeout(onRescan, 6000);

    window.__brFantasyEspnForceScan = onRescan;
  }

  if (isEspnDraftRoom()) {
    startEspnDraftObserver();
  } else if (!window.__brFantasyEspnLobbyWait) {
    window.__brFantasyEspnLobbyWait = true;
    setInterval(function () {
      if (isEspnDraftRoom()) startEspnDraftObserver();
    }, 1000);
  }
})();
