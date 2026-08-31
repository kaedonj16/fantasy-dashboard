// Shared draft-slot helpers for isolated-world host scripts.
// Detects the logged-in manager's snake slot (1-indexed first-round seat)
// from host JSON, cookies, or the draft-board "You" column. Read-only.

(function (root) {
  "use strict";

  function snakeSlot(overall, teams) {
    const n = Number(teams) || 0;
    const pn = Number(overall) || 0;
    if (n < 2 || pn < 1) return 0;
    const r = Math.ceil(pn / n);
    const i = (pn - 1) % n;
    return r % 2 === 1 ? i + 1 : n - i;
  }

  function teamCountFromPicks(picks) {
    const ids = {};
    let max = 0;
    (picks || []).forEach(function (p) {
      if (p && p.teamId != null && p.teamId !== "") ids[String(p.teamId)] = true;
      const n = Number(p && (p.overallPickNumber || p.pick_no));
      if (n > max) max = n;
    });
    const teams = Object.keys(ids).length;
    if (teams >= 4 && teams <= 32) return teams;
    if (max >= 8) {
      const guess = [8, 10, 12, 14, 16, 18, 20].filter(function (n) {
        return max % n === 0 || max % n <= 2;
      })[0];
      if (guess) return guess;
    }
    return 0;
  }

  function slotFromTeamId(picks, teamId, teams) {
    if (teamId == null || teamId === "") return 0;
    const want = String(teamId);
    let first = 0;
    let roundPick = 0;
    (picks || []).forEach(function (p) {
      if (!p || String(p.teamId) !== want) return;
      const n = Number(p.overallPickNumber || p.pick_no || 0);
      if (n && (!first || n < first)) first = n;
      const rd = Number(p.roundId || p.round || 0);
      const rp = Number(p.roundPickNumber || p.roundPick || 0);
      if (rd === 1 && rp >= 1) roundPick = rp;
    });
    const nTeams = Number(teams) || teamCountFromPicks(picks) || 12;
    if (roundPick >= 1 && roundPick <= nTeams) return roundPick;
    if (!first) return 0;
    return snakeSlot(first, nTeams);
  }

  function readCookie(name) {
    try {
      const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
      return m ? decodeURIComponent(m[1]) : "";
    } catch (_e) {
      return "";
    }
  }

  function espnSwid() {
    const raw = readCookie("SWID") || readCookie("swid");
    return String(raw || "").replace(/[{}]/g, "").toLowerCase();
  }

  function compactSync(platform, pickCount, mySlot, ok) {
    const plat = String(platform || "LIVE").replace(/[^a-z]/gi, "").slice(0, 7).toUpperCase() || "LIVE";
    const n = Number(pickCount) || 0;
    if (ok === false) return plat + " · …";
    const parts = [plat];
    if (n) parts.push(String(n));
    if (mySlot) parts.push("YOU " + Number(mySlot));
    else if (!n) parts.push("LIVE");
    return parts.join(" · ");
  }

  function looksLikeYouLabel(text) {
    const s = String(text || "").replace(/\s+/g, " ").trim();
    if (!s || s.length > 64) return false;
    if (/on the clock|your turn|waiting for you|draft assistant/i.test(s)) return false;
    if (/^you(?:'re| are)\b/i.test(s)) return false;
    return /\(\s*you\s*\)/i.test(s) || /\byour\s+team\b/i.test(s) || /^you$/i.test(s) || /^you\s+\d/i.test(s);
  }

  function slotFromColumnNode(node) {
    let col = node;
    for (let i = 0; i < 10 && col && col !== document.body; i++) {
      const parent = col.parentElement;
      if (!parent) break;
      const kids = Array.prototype.filter.call(parent.children, function (c) {
        return c && c.nodeType === 1 && c.offsetWidth > 36 && c.offsetHeight > 16;
      });
      if (kids.length >= 4 && kids.length <= 20) {
        const idx = kids.indexOf(col);
        if (idx >= 0) return idx + 1;
      }
      col = parent;
    }
    return 0;
  }

  function detectDomSlot() {
    const highlighted = document.querySelector(
      [
        '[class*="userTeam"]',
        '[class*="user-team"]',
        '[class*="yourTeam"]',
        '[class*="your-team"]',
        '[class*="is-you"]',
        '[class*="isYou"]',
        '[class*="current-user"]',
        '[class*="currentUser"]',
        '[data-is-user="true"]',
        '[data-user-team="true"]',
        '[aria-label*="your team" i]',
        '[aria-label*="(you)" i]',
        '[aria-label="You"]',
        '[class*="isSelf"]',
        '[class*="self-team"]',
        '[class*="selfTeam"]',
      ].join(",")
    );
    if (highlighted) {
      const slot = slotFromColumnNode(highlighted);
      if (slot) return slot;
    }

    const attr = document.querySelector("[data-draft-slot], [data-draftslot], [data-pick-slot]");
    if (attr) {
      const raw =
        attr.getAttribute("data-draft-slot") ||
        attr.getAttribute("data-draftslot") ||
        attr.getAttribute("data-pick-slot");
      const n = Number(raw);
      if (n >= 1 && n <= 32) {
        const youish =
          looksLikeYouLabel(attr.getAttribute("aria-label") || attr.textContent) ||
          attr.closest('[class*="user"], [class*="you"], [class*="your"]');
        if (youish) return n;
      }
    }

    const scope =
      document.querySelector(
        '[class*="draft-board"], [class*="draftBoard"], [class*="draft-container"], [class*="draftContainer"], [id*="draft"]'
      ) || document.body;
    if (!scope || typeof document.createTreeWalker !== "function") return 0;
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_ELEMENT);
    let el;
    let scanned = 0;
    while ((el = walker.nextNode()) && scanned < 3500) {
      scanned++;
      if (el.children && el.children.length > 4) continue;
      const label = (el.getAttribute("aria-label") || "") + " " + String(el.textContent || "").slice(0, 80);
      if (!looksLikeYouLabel(label)) continue;
      const slot = slotFromColumnNode(el);
      if (slot) return slot;
    }
    return 0;
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

  function yahooClientTeamId() {
    try {
      const m = String(location.pathname || "").match(/\/draftclient\/(?:nfl\/|f1\/)?(\d+)\/(\d+)/i);
      return m ? m[2] : "";
    } catch (_e) {
      return "";
    }
  }

  function parseYahooClock(text) {
    const s = String(text || "").replace(/\s+/g, " ");
    const up = s.match(/you(?:'re| are) up in\s+(\d+)\s+picks?/i);
    const onClock = /your pick|you(?:'re| are) on the clock|it(?:'s| is) your (?:pick|turn)/i.test(s);
    const rd = s.match(/round\s+(\d+)\s*[,•·]\s*pick\s+(\d+)/i);
    return {
      upIn: up ? Number(up[1]) : null,
      onClock: onClock,
      round: rd ? Number(rd[1]) : 0,
      roundPick: rd ? Number(rd[2]) : 0,
    };
  }

  function slotFromYahooClock(text, teams) {
    const c = parseYahooClock(text);
    const nTeams = Number(teams) || 0;
    if (c.onClock && c.round === 1 && c.roundPick >= 1) return c.roundPick;
    if (c.onClock && nTeams >= 2 && c.round >= 1 && c.roundPick >= 1) {
      return snakeSlot((c.round - 1) * nTeams + c.roundPick, nTeams);
    }
    if (c.upIn != null && c.upIn >= 0) {
      if (c.round === 1 && c.roundPick >= 1) return c.roundPick + c.upIn;
      if (c.round <= 1) return 1 + c.upIn;
      if (nTeams >= 2 && c.round >= 1 && c.roundPick >= 1) {
        return snakeSlot((c.round - 1) * nTeams + c.roundPick + c.upIn, nTeams);
      }
    }
    return 0;
  }

  function yahooClockText() {
    if (!document || !document.querySelectorAll) return "";
    const bits = [];
    const nodes = document.querySelectorAll(
      "h1,h2,h3,header,[class*='Clock'],[class*='clock'],[class*='Status'],[class*='status'],[class*='Banner'],[class*='banner'],[class*='Pick']"
    );
    const n = Math.min(nodes.length, 60);
    for (let i = 0; i < n; i++) bits.push(String(nodes[i].textContent || "").slice(0, 220));
    let blob = bits.join(" ");
    if (!/you(?:'re| are) up in|your pick/i.test(blob) && document.body) {
      blob = String(document.body.innerText || "").slice(0, 4000);
    }
    return blob;
  }

  function detectYahooSlot(teams) {
    const fromClock = slotFromYahooClock(yahooClockText(), teams);
    if (fromClock) return clampSlot(fromClock, teams || 32);
    const dom = detectDomSlot();
    if (dom) return dom;
    const tid = Number(yahooClientTeamId());
    if (tid >= 1 && tid <= 32 && (!teams || tid <= Number(teams))) return tid;
    return 0;
  }

  function clampSlot(slot, teams) {
    const n = Number(slot) || 0;
    const max = Number(teams) || 32;
    if (n < 1) return 0;
    return Math.max(1, Math.min(max, n));
  }

  function emptyRoster() {
    return { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, K: 0, DEF: 0, BN: 0 };
  }

  function rosterFromEspnSlots(counts) {
    const out = emptyRoster();
    if (!counts || typeof counts !== "object") return out;
    const map = { 0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "SF", 16: "DEF", 17: "K", 20: "BN", 23: "FLEX", 3: "FLEX", 5: "FLEX" };
    Object.keys(counts).forEach(function (k) {
      const id = Number(k);
      if (id === 21 || id === 22) return;
      const dest = map[id];
      const n = Number(counts[k]) || 0;
      if (dest && n > 0) out[dest] += n;
    });
    return out;
  }

  function rosterFromSleeperSettings(s) {
    const out = emptyRoster();
    if (!s || typeof s !== "object") return out;
    out.QB = Number(s.slots_qb || 0);
    out.RB = Number(s.slots_rb || 0);
    out.WR = Number(s.slots_wr || 0);
    out.TE = Number(s.slots_te || 0);
    out.FLEX = Number(s.slots_flex || s.slots_wr_rb_te || 0);
    out.SF = Number(s.slots_super_flex || s.slots_sf || 0);
    out.K = Number(s.slots_k || 0);
    out.DEF = Number(s.slots_def || 0);
    out.BN = Number(s.slots_bn || 0);
    return out;
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
      else if (raw === "W/R/T" || raw === "W/R" || raw === "W/T" || raw === "R/T" || raw === "FLEX") out.FLEX += n;
    });
    return out;
  }

  function rosterHasStarters(rs) {
    if (!rs) return false;
    return (rs.QB || 0) + (rs.RB || 0) + (rs.WR || 0) + (rs.TE || 0) + (rs.FLEX || 0) + (rs.SF || 0) >= 4;
  }

  function slotListFromRoster(rs) {
    const order = ["QB", "SF", "RB", "WR", "TE", "FLEX", "K", "DEF"];
    const out = [];
    order.forEach(function (k) {
      const n = Number(rs && rs[k]) || 0;
      for (let i = 0; i < n; i++) out.push(k === "SF" ? "SF" : k);
    });
    return out;
  }

  function settingsLabel(rs, scoring) {
    if (!rosterHasStarters(rs)) return "";
    const bits = [];
    bits.push((rs.SF ? "SF" : "1QB"));
    const ppr = scoring && scoring.ppr;
    if (ppr === 0.5) bits.push("HALF");
    else if (ppr === 0) bits.push("STD");
    else bits.push("PPR");
    if (scoring && Number(scoring.tep) > 0) bits.push("TEP");
    bits.push((rs.QB || 0) + "/" + (rs.RB || 0) + "/" + (rs.WR || 0) + "/" + (rs.TE || 0));
    if (rs.FLEX) bits.push("FLEX" + (rs.FLEX > 1 ? rs.FLEX : ""));
    return bits.join(" · ");
  }

  function scoringFromSleeperSettings(s) {
    const src = s && typeof s === "object" ? s : {};
    return {
      ppr: Number(src.rec != null ? src.rec : 1),
      tep: Number(src.bonus_rec_te || 0),
      passTd: Number(src.pass_td != null ? src.pass_td : 4),
    };
  }

  function rosterKey(rs) {
    if (!rs || typeof rs !== "object") return "";
    return ["QB", "SF", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN"].map(function (k) {
      return k + (Number(rs[k]) || 0);
    }).join("");
  }

  root.BRDraftSlot = {
    snakeSlot: snakeSlot,
    teamCountFromPicks: teamCountFromPicks,
    slotFromTeamId: slotFromTeamId,
    readCookie: readCookie,
    espnSwid: espnSwid,
    compactSync: compactSync,
    detectDomSlot: detectDomSlot,
    detectYahooSlot: detectYahooSlot,
    yahooClientTeamId: yahooClientTeamId,
    slotFromYahooClock: slotFromYahooClock,
    parseYahooClock: parseYahooClock,
    isEspnDraftRoom: isEspnDraftRoom,
    clampSlot: clampSlot,
    rosterFromEspnSlots: rosterFromEspnSlots,
    rosterFromSleeperSettings: rosterFromSleeperSettings,
    rosterFromYahooPositions: rosterFromYahooPositions,
    rosterHasStarters: rosterHasStarters,
    slotListFromRoster: slotListFromRoster,
    settingsLabel: settingsLabel,
    scoringFromSleeperSettings: scoringFromSleeperSettings,
    rosterKey: rosterKey,
  };
})(window);
