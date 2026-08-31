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
  const MAX_WALK = 4000;
  let lastFingerprint = "";
  let lastEmitAt = 0;
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
      return { leagueId, season };
    } catch (_e) {
      return { leagueId: "", season: "" };
    }
  }

  function bridgeToExtension(type, detail) {
    try {
      window.postMessage({ __br: BRIDGE, type: type, detail: detail || {} }, "*");
    } catch (_e) {
      /* ignore */
    }
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

  function isPickRow(obj) {
    if (!obj || typeof obj !== "object") return false;
    const pid =
      obj.playerId ??
      obj.player_id ??
      yahooIdFromKey(obj.player_key || obj.playerKey);
    const overall =
      obj.overallPickNumber ?? obj.overallPick ?? obj.pick_no ?? obj.pick;
    return overall != null && pid != null && String(pid) !== "" && String(pid) !== "0";
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
    const pid =
      raw.playerId ??
      raw.player_id ??
      yahooIdFromKey(raw.player_key || raw.playerKey);
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
    const playerId =
      raw.playerId ??
      raw.player_id ??
      yahooIdFromKey(raw.player_key || raw.playerKey);
    const overall =
      raw.overallPickNumber ?? raw.overallPick ?? raw.pick_no ?? raw.pick;
    const teamId =
      raw.teamId ??
      raw.team_id ??
      teamIdFromKey(raw.team_key || raw.teamKey);
    const roundId = raw.roundId ?? raw.round ?? raw.round_id;
    const roundPick = raw.roundPickNumber ?? raw.roundPick ?? raw.round_pick;
    const cost = raw.bidAmount ?? raw.cost ?? raw.auction_cost;
    const cached = playerId != null ? playerMetaById.get(String(playerId)) : null;
    const playerName = yahooPlayerName(raw) || (cached && cached.playerName) || "";
    const pos = yahooPlayerPos(raw) || (cached && cached.pos) || "";
    const nflTeam = yahooPlayerTeam(raw) || (cached && cached.nflTeam) || "";
    return {
      overallPickNumber: Number(overall),
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
      last ? last.teamId : "",
    ].join("|");
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

  function emit(picks, meta, source) {
    const ids = leagueFromUrl();
    const clean = (picks || []).map(normalizePick).filter(Boolean);
    clean.sort((a, b) => a.overallPickNumber - b.overallPickNumber);
    const fp = fingerprint(clean, meta || {});
    const now = Date.now();
    if (fp === lastFingerprint && now - lastEmitAt < 1500) return;
    lastFingerprint = fp;
    lastEmitAt = now;
    const detail = {
      source: source || "unknown",
      leagueId: ids.leagueId,
      season: ids.season,
      inProgress: !!(meta && meta.inProgress),
      drafted: !!(meta && meta.drafted),
      picks: clean,
      at: now,
    };
    bridgeToExtension(EVENT, detail);
    relayToBackground(detail);
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

  function walkForPicks(root) {
    const seen = new Set();
    const q = [root];
    let n = 0;
    while (q.length && n < MAX_WALK) {
      const cur = q.shift();
      n++;
      if (!cur || typeof cur !== "object") continue;
      if (seen.has(cur)) continue;
      seen.add(cur);
      if (cur.draft_results && maybeFromDraftResults(cur.draft_results, "react", null)) return true;
      if (cur.draftResults && maybeFromDraftResults(cur.draftResults, "react", null)) return true;
      if (Array.isArray(cur.picks) && cur.picks.some(isPickRow)) {
        emit(
          cur.picks,
          { inProgress: cur.inProgress !== false, drafted: cur.drafted === true },
          "react-picks"
        );
        return true;
      }
      if (Array.isArray(cur.draft_results) && cur.draft_results.length) {
        if (maybeFromDraftResults(cur.draft_results, "react-arr", null)) return true;
      }
      const next = [];
      if (cur.memoizedProps) next.push(cur.memoizedProps);
      if (cur.pendingProps) next.push(cur.pendingProps);
      if (cur.stateNode) next.push(cur.stateNode);
      if (cur.state) next.push(cur.state);
      if (cur.return) next.push(cur.return);
      if (cur.child) next.push(cur.child);
      if (cur.sibling) next.push(cur.sibling);
      if (cur.props) next.push(cur.props);
      for (const k of Object.keys(cur)) {
        if (k === "draft_results" || k === "draftResults") {
          if (maybeFromDraftResults(cur[k], "react-key", null)) return true;
        }
        const v = cur[k];
        if (v && typeof v === "object" && !seen.has(v)) next.push(v);
        if (next.length > 40) break;
      }
      for (let i = 0; i < Math.min(next.length, 24); i++) q.push(next[i]);
    }
    return false;
  }

  function scanReact() {
    const roots = [];
    const candidates = [
      document.getElementById("draft"),
      document.getElementById("draftapp"),
      document.getElementById("root"),
      document.querySelector("[data-reactroot]"),
      document.body,
    ].filter(Boolean);
    for (const el of candidates) {
      for (const key of Object.keys(el)) {
        if (
          key.startsWith("__reactFiber$") ||
          key.startsWith("__reactInternalInstance$") ||
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
    for (const r of roots) {
      if (walkForPicks(r)) return true;
    }
    return false;
  }

  function inspectJson(data, source) {
    if (!data) return;
    if (Array.isArray(data)) {
      for (const item of data) inspectJson(item, source);
      return;
    }
    if (typeof data !== "object") return;
    rememberYahooPlayer(data);
    if (data.draft_results || data.draftResults) {
      maybeFromDraftResults(data.draft_results || data.draftResults, source, null);
      return;
    }
    if (data.fantasy_content) {
      inspectJson(data.fantasy_content, source);
      return;
    }
    if (data.league) {
      inspectJson(data.league, source);
      return;
    }
    for (const key of ["draft", "data", "payload", "result"]) {
      if (data[key] && typeof data[key] === "object") inspectJson(data[key], source);
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
            if (/fantasysports\.yahoo|yahooapis\.com|sports\.yahoo\.com/i.test(url)) {
              res
                .clone()
                .json()
                .then((data) => inspectJson(data, "fetch"))
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
          if (!/fantasysports\.yahoo|yahooapis\.com|sports\.yahoo\.com/i.test(url)) return;
          const text = typeof this.responseText === "string" ? this.responseText : "";
          if (
            text.indexOf("draft_result") < 0 &&
            text.indexOf("player_key") < 0 &&
            text.indexOf("overallPick") < 0
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
  }

  hookNetwork();
  function onRescan() {
    lastFingerprint = "";
    scanReact();
  }
  window.addEventListener("message", (ev) => {
    if (!ev.data || ev.data.__br !== BRIDGE || ev.data.type !== RESCAN) return;
    onRescan();
  });
  document.addEventListener(RESCAN, onRescan);
  setInterval(() => {
    if (document.hidden) return;
    scanReact();
  }, 2000);
  setTimeout(scanReact, 1500);
  setTimeout(scanReact, 4000);
  window.__brFantasyYahooForceScan = onRescan;
})();
