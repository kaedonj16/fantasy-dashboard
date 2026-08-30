// Runs in the MAIN world on ESPN's draft room so it can see React fiber state,
// page fetch/XHR/WebSocket traffic, and poll mDraftDetail with the user's session.
// Forwards pick snapshots to the extension via chrome.runtime + postMessage.

(function () {
  "use strict";

  if (window.__brFantasyEspnDraftObserver) return;
  window.__brFantasyEspnDraftObserver = true;

  const EVENT = "brfantasy:espn-draft-raw";
  const RESCAN = "brfantasy:draft-rescan";
  const RELAY_STATUS = "brfantasy:espn-relay-status";
  const OBSERVER_READY = "brfantasy:espn-observer-ready";
  const BRIDGE = "brfantasy-bridge-v1";
  const MAX_WALK = 8000;
  const MAX_ELEM_SCAN = 800;
  let lastFingerprint = "";
  let lastEmitAt = 0;
  let apiPollInFlight = false;

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
    if (!obj || typeof obj !== "object") return null;
    const pool = obj.playerPoolEntry || obj.player_pool_entry;
    const fromPool =
      pool &&
      (pool.playerId ??
        pool.player_id ??
        pool.id ??
        (pool.player && (pool.player.id ?? pool.player.playerId)));
    return (
      obj.playerId ??
      obj.player_id ??
      obj.athleteId ??
      obj.athlete_id ??
      fromPool ??
      (obj.player && (obj.player.id ?? obj.player.playerId)) ??
      obj.id
    );
  }

  function pickOverall(obj) {
    if (!obj || typeof obj !== "object") return null;
    return (
      obj.overallPickNumber ??
      obj.overallPickNo ??
      obj.overallPick ??
      obj.overall_pick_number ??
      obj.pickNumber ??
      obj.pick_no ??
      obj.pick
    );
  }

  function isPickRow(obj) {
    if (!obj || typeof obj !== "object") return false;
    const pid = pickPlayerId(obj);
    const overall = pickOverall(obj);
    return overall != null && playerIdSelected(pid);
  }

  function normalizePick(raw) {
    if (!isPickRow(raw)) return null;
    const playerId = pickPlayerId(raw);
    const overall = pickOverall(raw);
    const teamId = raw.teamId ?? raw.team_id ?? raw.team?.id;
    const roundId = raw.roundId ?? raw.round ?? raw.round_id;
    const roundPick = raw.roundPickNumber ?? raw.roundPick ?? raw.round_pick ?? raw.slot;
    return {
      overallPickNumber: Number(overall),
      playerId: playerId == null ? null : playerId,
      teamId: teamId == null ? null : teamId,
      roundId: roundId == null ? null : Number(roundId),
      roundPickNumber: roundPick == null ? null : Number(roundPick),
      keeper: !!(raw.keeper || raw.reservedForKeeper || raw.isKeeper),
      bidAmount: raw.bidAmount != null ? raw.bidAmount : null,
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
          type: "espnDraftRelay",
          leagueId: detail.leagueId,
          season: detail.season || "",
          inProgress: !!detail.inProgress,
          drafted: !!detail.drafted,
          picks: Array.isArray(detail.picks) ? detail.picks : [],
          source: detail.source || "espn-draft-room",
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
    if (!clean.length) return;
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

  function maybeFromDraftDetail(detail, source) {
    if (!detail || typeof detail !== "object") return false;
    const picks = Array.isArray(detail.picks) ? detail.picks : null;
    if (!picks || !picks.length) return false;
    const selected = picks.filter(isPickRow);
    if (!selected.length) return false;
    emit(
      selected,
      {
        inProgress: detail.inProgress === true || detail.in_progress === true,
        drafted: detail.drafted === true,
      },
      source
    );
    return true;
  }

  function deepFindDraftDetail(data, depth) {
    if (!data || typeof data !== "object") return null;
    if (depth == null) depth = 0;
    if (depth > 14) return null;
    if (Array.isArray(data)) {
      if (data.length && data[0] && typeof data[0] === "object" && data[0].draftDetail) {
        const inner = deepFindDraftDetail(data[0], depth + 1);
        if (inner) return inner;
      }
      for (let i = 0; i < Math.min(data.length, 8); i++) {
        const inner = deepFindDraftDetail(data[i], depth + 1);
        if (inner) return inner;
      }
      return null;
    }
    if (data.draftDetail && typeof data.draftDetail === "object") {
      const dd = data.draftDetail;
      if (Array.isArray(dd.picks)) return dd;
    }
    for (const k of Object.keys(data)) {
      if (k === "draftDetail") continue;
      const v = data[k];
      if (v && typeof v === "object") {
        const inner = deepFindDraftDetail(v, depth + 1);
        if (inner) return inner;
      }
    }
    return null;
  }

  function inspectJson(data, source) {
    if (!data) return;
    const detail = deepFindDraftDetail(data) || (data.draftDetail ? data.draftDetail : null);
    if (detail && maybeFromDraftDetail(detail, source)) return;
    if (Array.isArray(data)) {
      for (const item of data) inspectJson(item, source);
      return;
    }
    if (typeof data !== "object") return;
    if (Array.isArray(data.picks) && data.picks.some(isPickRow)) {
      emit(
        data.picks.filter(isPickRow),
        { inProgress: data.inProgress === true, drafted: data.drafted === true },
        source + "-picks"
      );
    }
  }

  function walkForDraftDetail(root) {
    const seen = new Set();
    const q = [root];
    let n = 0;
    while (q.length && n < MAX_WALK) {
      const cur = q.shift();
      n++;
      if (!cur || typeof cur !== "object") continue;
      if (seen.has(cur)) continue;
      seen.add(cur);
      if (cur.draftDetail && maybeFromDraftDetail(cur.draftDetail, "react")) return true;
      if (Array.isArray(cur.picks) && cur.picks.some(isPickRow)) {
        emit(
          cur.picks.filter(isPickRow),
          { inProgress: cur.inProgress === true, drafted: cur.drafted === true },
          "react-picks"
        );
        return true;
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
        if (k === "draftDetail") {
          if (maybeFromDraftDetail(cur[k], "react-key")) return true;
        }
        const v = cur[k];
        if (v && typeof v === "object" && !seen.has(v)) next.push(v);
        if (next.length > 48) break;
      }
      for (let i = 0; i < Math.min(next.length, 28); i++) q.push(next[i]);
    }
    return false;
  }

  function collectReactRoots() {
    const roots = [];
    const seenRoot = new Set();
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

  function espnApiUrls(leagueId, season) {
    const s = String(season || new Date().getFullYear());
    const lid = String(leagueId);
    const q = "?view=mDraftDetail&view=mSettings";
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
        "?view=mDraftDetail&seasonId=" +
        encodeURIComponent(s),
    ];
  }

  function pollEspnApi() {
    if (apiPollInFlight) return;
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

  hookNetwork();
  bridgeToExtension(OBSERVER_READY, { href: location.href, leagueId: leagueFromUrl().leagueId });

  function onRescan() {
    lastFingerprint = "";
    scanReact();
    pollEspnApi();
  }
  window.addEventListener("message", function (ev) {
    if (!ev.data || ev.data.__br !== BRIDGE || ev.data.type !== RESCAN) return;
    onRescan();
  });
  document.addEventListener(RESCAN, onRescan);

  setInterval(function () {
    if (document.hidden) return;
    scanReact();
  }, 2000);
  setInterval(function () {
    if (document.hidden) return;
    pollEspnApi();
  }, 5000);

  setTimeout(onRescan, 800);
  setTimeout(onRescan, 2500);
  setTimeout(onRescan, 6000);

  window.__brFantasyEspnForceScan = onRescan;
})();
