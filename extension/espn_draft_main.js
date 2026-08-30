// Runs in the MAIN world on ESPN's draft room so it can see React fiber state
// and page fetch/XHR traffic. ESPN's documented mDraftDetail REST view often
// does not update mid-draft; the live draft room itself does. We observe that
// in-page state and forward a compact pick snapshot to the isolated content
// script via postMessage (no cookies leave this page).

(function () {
  "use strict";

  const EVENT = "brfantasy:espn-draft-raw";
  const RESCAN = "brfantasy:draft-rescan";
  const RELAY_STATUS = "brfantasy:espn-relay-status";
  const BRIDGE = "brfantasy-bridge-v1";
  const MAX_WALK = 4000;
  let lastFingerprint = "";
  let lastEmitAt = 0;

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

  function playerIdSelected(pid) {
    if (pid == null) return false;
    const text = String(pid).trim();
    if (!text || text === "0" || text === "-1" || text === "None" || text === "null") return false;
    return true;
  }

  function isPickRow(obj) {
    if (!obj || typeof obj !== "object") return false;
    const pid =
      obj.playerId ??
      obj.player_id ??
      obj.athleteId ??
      obj.athlete_id ??
      (obj.player && (obj.player.id ?? obj.player.playerId)) ??
      obj.id;
    const overall = obj.overallPickNumber ?? obj.overallPick ?? obj.pick_no;
    return overall != null && playerIdSelected(pid);
  }

  function normalizePick(raw) {
    if (!isPickRow(raw)) return null;
    const playerId =
      raw.playerId ??
      raw.player_id ??
      raw.athleteId ??
      raw.athlete_id ??
      (raw.player && (raw.player.id ?? raw.player.playerId)) ??
      raw.id;
    const overall = raw.overallPickNumber ?? raw.overallPick ?? raw.pick_no;
    const teamId = raw.teamId ?? raw.team_id;
    const roundId = raw.roundId ?? raw.round ?? raw.round_id;
    const roundPick = raw.roundPickNumber ?? raw.roundPick ?? raw.round_pick;
    return {
      overallPickNumber: Number(overall),
      playerId: playerId == null ? null : playerId,
      teamId: teamId == null ? null : teamId,
      roundId: roundId == null ? null : Number(roundId),
      roundPickNumber: roundPick == null ? null : Number(roundPick),
      keeper: !!(raw.keeper || raw.reservedForKeeper),
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
    if (!picks) return false;
    emit(
      picks,
      {
        inProgress: detail.inProgress === true,
        drafted: detail.drafted === true,
      },
      source
    );
    return true;
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
          cur.picks,
          { inProgress: cur.inProgress === true, drafted: cur.drafted === true },
          "react-picks"
        );
        return true;
      }
      // Prefer props / memoizedProps / stateNode paths used by React fiber.
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
        if (next.length > 40) break;
      }
      for (let i = 0; i < Math.min(next.length, 24); i++) q.push(next[i]);
    }
    return false;
  }

  function scanReact() {
    const roots = [];
    const candidates = [
      document.getElementById("espn-aria-root"),
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
      if (walkForDraftDetail(r)) return true;
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
    if (data.draftDetail) {
      maybeFromDraftDetail(data.draftDetail, source);
      return;
    }
    // Some live payloads nest under league or draft keys.
    for (const key of ["league", "draft", "data", "payload"]) {
      if (data[key] && typeof data[key] === "object") inspectJson(data[key], source);
    }
  }

  function hookNetwork() {
    if (window.__brFantasyEspnFetchHooked) return;
    window.__brFantasyEspnFetchHooked = true;

    const origFetch = window.fetch;
    if (typeof origFetch === "function") {
      window.fetch = function () {
        const args = arguments;
        return origFetch.apply(this, args).then((res) => {
          try {
            const url = String((args[0] && args[0].url) || args[0] || "");
            if (/fantasy\.espn\.com|lm-api-reads\.fantasy\.espn\.com/i.test(url)) {
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
          if (!/fantasy\.espn\.com|lm-api-reads\.fantasy\.espn\.com/i.test(url)) return;
          const ct = (this.getResponseHeader("content-type") || "").toLowerCase();
          if (ct.indexOf("json") < 0 && typeof this.responseText === "string") {
            if (this.responseText.indexOf("draftDetail") < 0 && this.responseText.indexOf("overallPickNumber") < 0)
              return;
          }
          const data = JSON.parse(this.responseText);
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
  // Poll React state a few times a second while the tab is visible. Cheap relative
  // to a missed pick during a live draft.
  setInterval(() => {
    if (document.hidden) return;
    scanReact();
  }, 2000);
  // First pass after the draft UI settles.
  setTimeout(scanReact, 1500);
  setTimeout(scanReact, 4000);
  window.__brFantasyEspnForceScan = onRescan;
})();
