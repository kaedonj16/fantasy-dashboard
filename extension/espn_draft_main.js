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
  const MAX_WALK = 12000;
  const MAX_ELEM_SCAN = 1200;
  let lastFingerprint = "";
  let lastEmitAt = 0;
  let apiPollInFlight = false;
  /** @type {Map<number, object>} */
  const pickAccumulator = new Map();
  let bestOverallSeen = 0;

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

  function mergeIntoAccumulator(rawPicks) {
    let grew = false;
    for (const raw of rawPicks || []) {
      const norm = normalizePick(raw);
      if (!norm || !norm.overallPickNumber) continue;
      const n = Number(norm.overallPickNumber);
      if (!n || n <= 0) continue;
      if (!pickAccumulator.has(n)) grew = true;
      pickAccumulator.set(n, norm);
      if (n > bestOverallSeen) bestOverallSeen = n;
    }
    return grew;
  }

  function emitAccumulated(meta, source) {
    const ids = leagueFromUrl();
    const clean = Array.from(pickAccumulator.values()).sort(
      (a, b) => a.overallPickNumber - b.overallPickNumber
    );
    if (!clean.length) return;
    const fp = fingerprint(clean, meta || {});
    const now = Date.now();
    if (fp === lastFingerprint && now - lastEmitAt < 1500) return;
    lastFingerprint = fp;
    lastEmitAt = now;
    const detail = {
      source: source || "accumulated",
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

  function emit(picks, meta, source) {
    if (!mergeIntoAccumulator(picks)) {
      const incoming = (picks || []).map(normalizePick).filter(Boolean);
      if (!incoming.length) return;
      const maxIncoming = incoming[incoming.length - 1].overallPickNumber;
      if (maxIncoming <= bestOverallSeen && pickAccumulator.size >= incoming.length) return;
    }
    emitAccumulated(meta, source);
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

  function findBestDraftDetail(data, depth, best) {
    if (!data || typeof data !== "object") return best;
    if (depth == null) depth = 0;
    if (!best) best = { detail: null, count: 0 };
    if (depth > 16) return best;
    if (Array.isArray(data)) {
      for (let i = 0; i < Math.min(data.length, 32); i++) {
        best = findBestDraftDetail(data[i], depth + 1, best);
      }
      return best;
    }
    if (data.draftDetail && typeof data.draftDetail === "object") {
      const dd = data.draftDetail;
      if (Array.isArray(dd.picks)) {
        const sel = dd.picks.filter(isPickRow);
        if (sel.length > best.count) {
          best = { detail: dd, count: sel.length };
        }
      }
    }
    if (Array.isArray(data.picks) && data.picks.some(isPickRow)) {
      const sel = data.picks.filter(isPickRow);
      if (sel.length > best.count) {
        best = { detail: data, count: sel.length };
      }
    }
    for (const k of Object.keys(data)) {
      if (k === "draftDetail") continue;
      const v = data[k];
      if (v && typeof v === "object") {
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
    let found = false;
    while (q.length && n < MAX_WALK) {
      const cur = q.shift();
      n++;
      if (!cur || typeof cur !== "object") continue;
      if (seen.has(cur)) continue;
      seen.add(cur);
      if (cur.draftDetail) {
        if (maybeFromDraftDetail(cur.draftDetail, "react")) found = true;
      }
      if (Array.isArray(cur.picks) && cur.picks.some(isPickRow)) {
        emit(
          cur.picks.filter(isPickRow),
          { inProgress: cur.inProgress === true, drafted: cur.drafted === true },
          "react-picks"
        );
        found = true;
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
          if (maybeFromDraftDetail(cur[k], "react-key")) found = true;
        }
        const v = cur[k];
        if (v && typeof v === "object" && !seen.has(v)) next.push(v);
        if (next.length > 48) break;
      }
      for (let i = 0; i < Math.min(next.length, 28); i++) q.push(next[i]);
    }
    return found;
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

  function addDomPick(map, pickNo, playerId, teamId, sourceTag) {
    if (!playerIdSelected(playerId) || pickNo == null || pickNo <= 0) return;
    const n = Number(pickNo);
    if (!map.has(n)) {
      map.set(n, {
        overallPickNumber: n,
        playerId: playerId,
        teamId: teamId == null ? null : teamId,
        roundId: null,
        roundPickNumber: null,
        keeper: false,
        bidAmount: null,
        __source: sourceTag,
      });
    }
  }

  function scrapeDomPicks() {
    const byOverall = new Map();
    const scope =
      document.querySelector('[class*="draftContainer"], [class*="draft-container"], main, #root') ||
      document.body;

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
      addDomPick(byOverall, pickNo, pid, null, "dom-cell");
      const props = reactPropsNear(cell);
      if (props) {
        const norm = normalizePick(props);
        if (norm) addDomPick(byOverall, norm.overallPickNumber, norm.playerId, norm.teamId, "dom-react");
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

    mergeIntoAccumulator(picks);
    emitAccumulated({ inProgress: true, drafted: false }, "dom-scrape");
    return true;
  }

  function scanAll() {
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
  watchDom();
  bridgeToExtension(OBSERVER_READY, { href: location.href, leagueId: leagueFromUrl().leagueId });

  function onRescan() {
    lastFingerprint = "";
    pickAccumulator.clear();
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
    if (document.hidden) return;
    scanAll();
  }, 2000);
  setInterval(function () {
    if (document.hidden) return;
    pollEspnApi();
  }, 3000);

  setTimeout(onRescan, 800);
  setTimeout(onRescan, 2500);
  setTimeout(onRescan, 6000);

  window.__brFantasyEspnForceScan = onRescan;
})();
