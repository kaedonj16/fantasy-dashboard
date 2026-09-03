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
        '[class*="is-me"]',
        '[class*="isMe"]',
        '[class*="my-team"]',
        '[class*="myTeam"]',
        '[class*="my-column"]',
        '[class*="myColumn"]',
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

  function sleeperHrefString(href) {
    const raw = String(href || (typeof location !== "undefined" ? location.href : "") || "");
    if (raw) return raw;
    if (typeof location === "undefined") return "";
    return String(location.pathname || "") + String(location.search || "") + String(location.hash || "");
  }

  function sleeperNavText(href) {
    return sleeperHrefString(href).toLowerCase();
  }

  function sleeperDraftIdFromUrl(href) {
    try {
      const text = sleeperNavText(href);
      const q = text.match(/[?&#]draft_id=([a-z0-9]+)/i) || text.match(/[?&#]draftid=([a-z0-9]+)/i);
      if (q && q[1]) return q[1];
      const m = text.match(/\/draft\/(?:nfl\/|nba\/|ncaaf\/|cbb\/|epl\/)?([a-z0-9]+)/i);
      return m ? m[1] : "";
    } catch (_e) {
      return "";
    }
  }

  function sleeperLeagueIdFromUrl(href) {
    try {
      const m = sleeperNavText(href).match(/\/leagues\/(\d{6,20})/);
      return m ? m[1] : "";
    } catch (_e) {
      return "";
    }
  }

  function isSleeperDraftRoom(href) {
    const text = sleeperNavText(href);
    if (!text) return false;
    if (sleeperDraftIdFromUrl(href)) return true;
    if (/\/leagues\/\d{6,20}\/draft(?:\/|$|\?|#)/.test(text)) return true;
    if (/\/draft\/[a-z0-9]+/.test(text)) return true;
    return false;
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
      const rd = s.match(/round\s+(\d+)\s*[,•·]?\s*pick\s+(\d+)/i);
    let round = 0;
    let roundPick = 0;
    let overall = 0;
    if (rd) {
      round = Number(rd[1]);
      const n = Number(rd[2]);
      // "Round 14, Pick 185" is overall 185. "Round 1, Pick 7" is seat 7.
      if (n > 16) overall = n;
      else roundPick = n;
    }
    if (!overall) {
      const ov = s.match(/\bpick\s+(\d{2,3})\b/i);
      if (ov && Number(ov[1]) > 16) overall = Number(ov[1]);
    }
    return {
      upIn: up ? Number(up[1]) : null,
      onClock: onClock,
      round: round,
      roundPick: roundPick,
      overall: overall,
    };
  }

  function slotFromYahooClock(text, teams) {
    const c = parseYahooClock(text);
    const nTeams = Number(teams) || 0;
    if (c.onClock && c.overall >= 1 && nTeams >= 2) return snakeSlot(c.overall, nTeams);
    if (c.onClock && c.round === 1 && c.roundPick >= 1) return c.roundPick;
    if (c.onClock && nTeams >= 2 && c.round >= 1 && c.roundPick >= 1) {
      return snakeSlot((c.round - 1) * nTeams + c.roundPick, nTeams);
    }
    if (c.upIn != null && c.upIn >= 0) {
      if (c.overall >= 1 && nTeams >= 2) return snakeSlot(c.overall + c.upIn, nTeams);
      if (c.round === 1 && c.roundPick >= 1) return c.roundPick + c.upIn;
      if (c.round <= 1) return 1 + c.upIn;
      if (nTeams >= 2 && c.round >= 1 && c.roundPick >= 1) {
        return snakeSlot((c.round - 1) * nTeams + c.roundPick + c.upIn, nTeams);
      }
    }
    return 0;
  }

  function yahooClockText() {
    const docs = sameOriginDocuments();
    const bits = [];
    docs.forEach(function (doc) {
      if (!doc || !doc.querySelectorAll) return;
      const nodes = doc.querySelectorAll(
        "h1,h2,h3,header,[class*='Clock'],[class*='clock'],[class*='Status'],[class*='status'],[class*='Banner'],[class*='banner'],[class*='Pick'],[class*='round'],[class*='Round']"
      );
      const n = Math.min(nodes.length, 80);
      for (let i = 0; i < n; i++) bits.push(String(nodes[i].textContent || "").slice(0, 240));
      if (doc.body) bits.push(String(doc.body.innerText || "").slice(0, 6000));
    });
    return bits.join(" ");
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
    return { QB: 0, SF: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, RB_WR: 0, WR_TE: 0, RB_TE: 0, K: 0, DEF: 0, BN: 0 };
  }

  function normDraftPos(pos) {
    const p = String(pos || "").toUpperCase();
    if (p === "PK") return "K";
    if (p === "DST" || p === "D/ST" || p === "D-ST" || p === "D ST") return "DEF";
    return p;
  }

  function isKDefPos(pos) {
    const p = normDraftPos(pos);
    return p === "K" || p === "DEF";
  }

  function rosterFromEspnSlots(counts) {
    const out = emptyRoster();
    if (!counts || typeof counts !== "object") return out;
    const map = { 0: "QB", 2: "RB", 4: "WR", 6: "TE", 7: "SF", 16: "DEF", 17: "K", 20: "BN", 23: "FLEX", 3: "RB_WR", 5: "WR_TE" };
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
    out.RB_WR = Number(s.slots_wr_rb || s.slots_rb_wr || 0);
    out.WR_TE = Number(s.slots_rec_flex || s.slots_wr_te || 0);
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
      else if (raw === "W/R" || raw === "RB/WR" || raw === "WRRB_FLEX" || raw === "RB_WR") out.RB_WR += n;
      else if (raw === "W/T" || raw === "WR/TE" || raw === "REC_FLEX" || raw === "WR_TE") out.WR_TE += n;
      else if (raw === "R/T" || raw === "RB/TE" || raw === "RB_TE") out.RB_TE += n;
      else if (raw === "W/R/T" || raw === "FLEX") out.FLEX += n;
    });
    return out;
  }

  function rosterHasStarters(rs) {
    if (!rs) return false;
    return (rs.QB || 0) + (rs.RB || 0) + (rs.WR || 0) + (rs.TE || 0) + (rs.FLEX || 0) + (rs.SF || 0) + (rs.RB_WR || 0) + (rs.WR_TE || 0) + (rs.RB_TE || 0) >= 4;
  }

  function slotListFromRoster(rs) {
    const order = ["QB", "SF", "RB", "WR", "TE", "RB_WR", "WR_TE", "RB_TE", "FLEX", "K", "DEF"];
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
    if (rs.RB_WR) bits.push("RB/WR" + (rs.RB_WR > 1 ? rs.RB_WR : ""));
    if (rs.WR_TE) bits.push("WR/TE" + (rs.WR_TE > 1 ? rs.WR_TE : ""));
    if (rs.RB_TE) bits.push("RB/TE" + (rs.RB_TE > 1 ? rs.RB_TE : ""));
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
    return ["QB", "SF", "RB", "WR", "TE", "FLEX", "RB_WR", "WR_TE", "RB_TE", "K", "DEF", "BN"].map(function (k) {
      return k + (Number(rs[k]) || 0);
    }).join("");
  }

  function completedFromYahooClock(teams) {
    const c = parseYahooClock(yahooClockText());
    if (c.overall >= 1) return Math.max(0, c.overall - 1);
    if (c.round >= 1 && c.roundPick >= 1) {
      const nTeams = Number(teams) >= 2 ? Number(teams) : 12;
      return Math.max(0, (c.round - 1) * nTeams + c.roundPick - 1);
    }
    return 0;
  }

  function yahooInAvailableList(el, doc) {
    let n = el;
    for (let i = 0; i < 10 && n && n !== (doc && doc.body); i++) {
      const cls = String((n.className && n.className.baseVal != null ? n.className.baseVal : n.className) || "");
      const id = String(n.id || "");
      const aria = String((n.getAttribute && n.getAttribute("aria-label")) || "");
      if (/available|playerlist|player-list|player_list|rankings|search-result|searchResult|draft.?kit|ultra.?draft|adp.?rank/i.test(cls + " " + id + " " + aria)) {
        return true;
      }
      n = n.parentElement;
    }
    return false;
  }

  function parseYahooNamePos(text) {
    const s = String(text || "").replace(/\s+/g, " ").trim().replace(/\s*[·•()]\s*/g, " ");
    const m = s.match(
      /([A-Z](?:[a-zA-Z.'\-]+|\.)(?:\s+(?:[A-Z][a-zA-Z.'\-]+|Jr\.?|Sr\.?|II|III|IV|V)){0,4})\s+(QB|RB|WR|TE|K|DEF|DST|D\/ST)\b(?:\s*-?\s*([A-Z]{2,3}))?/
    );
    if (!m) return null;
    const name = m[1].replace(/\s+/g, " ").trim();
    if (name.length < 3 || name.length > 48) return null;
    if (/^(round|pick|you|team|bench|available|search|draft|flex)$/i.test(name)) return null;
    if (/you(?:'re| are) up|on the clock|time remaining/i.test(name)) return null;
    return { name: name, pos: m[2].replace("D/ST", "DEF"), team: m[3] || "" };
  }

  function parseYahooLooseName(text) {
    const typed = parseYahooNamePos(text);
    if (typed) return typed;
    const s = String(text || "").replace(/\s+/g, " ").trim().replace(/\s*[·•]\s*/g, " ");
    const m = s.match(
      /^([A-Z](?:[a-zA-Z.'\-]+|\.)(?:\s+(?:[A-Z][a-zA-Z.'\-]+|Jr\.?|Sr\.?|II|III|IV|V)){0,4})(?:\s+([A-Z]{2,3}))?$/
    );
    if (!m) return null;
    const name = m[1].replace(/\s+/g, " ").trim();
    if (name.length < 5 || name.length > 48) return null;
    if (!/\s/.test(name)) return null;
    if (/^(round|pick|you|team|bench|available|search|draft|flex|yahoo)$/i.test(name)) return null;
    if (/you(?:'re| are) up|on the clock|time remaining/i.test(name)) return null;
    return { name: name, pos: "", team: m[2] || "" };
  }

  function sameOriginDocuments(doc) {
    const start = doc || (typeof document !== "undefined" ? document : null);
    const out = [];
    if (!start) return out;
    out.push(start);
    try {
      const frames = start.querySelectorAll("iframe");
      for (let i = 0; i < Math.min(frames.length, 12); i++) {
        try {
          const d = frames[i].contentDocument;
          if (d && d.documentElement && out.indexOf(d) < 0) out.push(d);
        } catch (_e) { /* cross-origin */ }
      }
    } catch (_e) { /* ignore */ }
    return out;
  }

  function yahooDottedToOverall(rd, pk, teams) {
    const r = Number(rd) || 0;
    const p = Number(pk) || 0;
    const n = Number(teams) || 0;
    if (!(r >= 1 && p >= 1 && p <= 32)) return 0;
    if (n >= 4) return (r - 1) * n + p;
    if (r === 1) return p;
    return 0;
  }

  function parseYahooCompactPick(text, teams) {
    const s = String(text || "").replace(/\s+/g, " ").replace(/[•·()]/g, " ").trim();
    const m = s.match(/^(\d{1,2})\.(\d{1,2})\s+(.+)$/);
    if (!m) return null;
    const rd = Number(m[1]);
    const pk = Number(m[2]);
    if (!(rd >= 1 && rd <= 30 && pk >= 1 && pk <= 20)) return null;
    const rest = m[3];
    const info = parseYahooNamePos(rest) || parseYahooLooseName(rest);
    const abbr = !info
      ? rest.match(/^([A-Z]\.?\s+[A-Z][A-Za-z.'\-]+)\s+(QB|RB|WR|TE|K|DEF|DST|D\/ST)\b(?:\s+([A-Z]{2,3}))?/)
      : null;
    const parsed = info || (abbr
      ? { name: abbr[1].replace(/\s+/g, " ").trim(), pos: String(abbr[2]).replace("D/ST", "DEF").replace("DST", "DEF"), team: abbr[3] || "" }
      : null);
    if (!parsed || !parsed.name || parsed.name.length < 3) return null;
    const pn = yahooDottedToOverall(rd, pk, teams);
    if (!pn) return null;
    return {
      overallPickNumber: pn,
      playerName: parsed.name,
      pos: parsed.pos || "",
      nflTeam: parsed.team || "",
      roundId: rd,
      roundPickNumber: pk,
    };
  }

  function filterYahooPicksToClock(picks, teams) {
    const list = picks || [];
    const expected = completedFromYahooClock(teams);
    if (!(expected >= 0) || !list.length) return list;
    const maxPn = list.reduce(function (m, p) {
      const n = Number(p && (p.overallPickNumber || p.pick_no)) || 0;
      return n > m ? n : m;
    }, 0);
    if (expected <= 48 && (maxPn > expected + 6 || list.length > expected + 6)) {
      return [];
    }
    return list;
  }

  function mergeYahooPicks(a, b) {
    const map = {};
    function add(p) {
      if (!p) return;
      const pn = Number(p.overallPickNumber || p.pick_no || 0);
      if (pn < 1 || pn > 400) return;
      const name = String(p.playerName || p.name || "").replace(/\s+/g, " ").trim();
      const pid = p.playerId != null && String(p.playerId) !== "0" ? String(p.playerId) : "";
      if (!name && !pid) return;
      const prev = map[pn];
      if (
        !prev ||
        (name && name.length > String(prev.playerName || "").length) ||
        (pid && !prev.playerId)
      ) {
        map[pn] = {
          overallPickNumber: pn,
          playerId: pid || (prev && prev.playerId) || "",
          playerName: name || (prev && prev.playerName) || "",
          pos: p.pos || (prev && prev.pos) || "",
          nflTeam: p.nflTeam || (prev && prev.nflTeam) || "",
          teamId: p.teamId != null && p.teamId !== "" ? String(p.teamId) : (prev && prev.teamId) || null,
          roundId: p.roundId || p.round || (prev && prev.roundId) || null,
          roundPickNumber: p.roundPickNumber || (prev && prev.roundPickNumber) || null,
        };
      }
    }
    (a || []).forEach(add);
    (b || []).forEach(add);
    return Object.keys(map)
      .map(Number)
      .sort(function (x, y) {
        return x - y;
      })
      .map(function (k) {
        return map[k];
      });
  }

  function addYahooDomPick(byPn, pn, info, pid) {
    if (!info || !info.name) return;
    const n = Number(pn);
    if (!(n >= 1 && n <= 400)) return;
    const prev = byPn.get(n);
    if (prev && prev.playerName && prev.playerName.length >= info.name.length && prev.playerId) return;
    byPn.set(n, {
      overallPickNumber: n,
      playerId: pid ? String(pid) : (prev && prev.playerId) || "",
      playerName: info.name,
      pos: info.pos || (prev && prev.pos) || "",
      nflTeam: info.team || (prev && prev.nflTeam) || "",
    });
  }

  function scrapeYahooLabeledPicks(doc, byPn) {
    const nodes = doc.querySelectorAll(
      "[data-pick],[data-overall-pick],[data-pick-number],[class*='Pick'],[class*='pick'],[class*='draft-cell'],[class*='draftCell'],[class*='PickCard'],[class*='player-name']"
    );
    const limit = Math.min(nodes.length, 900);
    for (let i = 0; i < limit; i++) {
      const el = nodes[i];
      if (yahooInAvailableList(el, doc)) continue;
      const text = String(el.textContent || "").replace(/\s+/g, " ").trim();
      if (text.length < 5 || text.length > 180) continue;
      const info = parseYahooNamePos(text) || parseYahooLooseName(text);
      if (!info) continue;
      let pn = Number(
        el.getAttribute("data-pick") ||
          el.getAttribute("data-overall-pick") ||
          el.getAttribute("data-pick-number") ||
          0
      );
      if (!pn) {
        const labeled = text.match(/pick\s*#?\s*(\d{1,3})\b/i);
        if (labeled) pn = Number(labeled[1]);
      }
      if (!pn) {
        const compact = parseYahooCompactPick(text, Number(doc.documentElement && doc.documentElement.getAttribute("data-br-da-teams")) || 0);
        if (compact) pn = compact.overallPickNumber;
        if (!pn) {
          const dotted = text.match(/\b(\d{1,2})\.(\d{1,2})\b/);
          if (dotted) {
            const teams = Number(doc.documentElement && doc.documentElement.getAttribute("data-br-da-teams")) || 0;
            pn = yahooDottedToOverall(dotted[1], dotted[2], teams);
          }
        }
      }
      if (!pn) continue;
      addYahooDomPick(byPn, pn, info, el.getAttribute("data-player-id") || el.getAttribute("data-playerid"));
    }
  }

  function scrapeYahooColumns(doc, byPn, teamsHint) {
    const scope =
      doc.querySelector(
        "#draft, #draftapp, [class*='draft-board'], [class*='draftBoard'], [class*='DraftBoard'], [class*='pick-grid'], [class*='PickGrid']"
      ) || doc.body;
    if (!scope || !doc.createTreeWalker) return;
    const walker = doc.createTreeWalker(scope, NodeFilter.SHOW_ELEMENT);
    let el;
    let scanned = 0;
    let cols = [];
    while ((el = walker.nextNode()) && scanned < 2500) {
      scanned++;
      const kids = el.children;
      if (!kids || kids.length < 8 || kids.length > 18) continue;
      const widths = [];
      let ok = true;
      for (let i = 0; i < kids.length; i++) {
        const w = kids[i].offsetWidth || 0;
        const h = kids[i].offsetHeight || 0;
        if (w < 28 || h < 28) {
          ok = false;
          break;
        }
        widths.push(w);
      }
      if (!ok || widths.length < 8) continue;
      const avg = widths.reduce(function (a, b) {
        return a + b;
      }, 0) / widths.length;
      if (!widths.every(function (w) { return Math.abs(w - avg) < avg * 0.5; })) continue;
      if (kids.length > cols.length) cols = Array.prototype.slice.call(kids);
    }
    if (cols.length < 8) return;
    const teams = Number(teamsHint) >= 4 ? Number(teamsHint) : cols.length;
    cols.forEach(function (col, idx) {
      if (yahooInAvailableList(col, doc)) return;
      const slot = idx + 1;
      const text = String(col.innerText || col.textContent || "");
      const re =
        /([A-Z][a-zA-Z.'\-]+(?:\s+(?:[A-Z][a-zA-Z.'\-]+|Jr\.?|Sr\.?|II|III|IV|V)){0,4})\s+(QB|RB|WR|TE|K|DEF|DST|D\/ST)\b(?:\s+([A-Z]{2,3}))?/g;
      const names = [];
      let m;
      while ((m = re.exec(text))) {
        const parsed = parseYahooNamePos(m[0]);
        if (parsed) names.push(parsed);
      }
      names.forEach(function (info, ridx) {
        const rd = ridx + 1;
        const pn = (rd - 1) * teams + (rd % 2 === 1 ? slot : teams - slot + 1);
        addYahooDomPick(byPn, pn, info, "");
      });
    });
  }

  function scrapeYahooDottedPicks(doc, byPn, teamsHint) {
    const teams = Number(teamsHint) >= 4 ? Number(teamsHint) : 0;
    const nodes = doc.querySelectorAll("div,span,li,td,article,section");
    const limit = Math.min(nodes.length, 1500);
    for (let i = 0; i < limit; i++) {
      const el = nodes[i];
      if (yahooInAvailableList(el, doc)) continue;
      const text = String(el.textContent || "").replace(/\s+/g, " ").trim();
      if (text.length < 6 || text.length > 80) continue;
      if (!/^\d{1,2}\.\d{1,2}\b/.test(text)) continue;
      const compact = parseYahooCompactPick(text, teams);
      if (!compact) continue;
      addYahooDomPick(byPn, compact.overallPickNumber, {
        name: compact.playerName,
        pos: compact.pos,
        team: compact.nflTeam,
      }, "");
    }
    const lastBits = String((doc.body && doc.body.innerText) || "").slice(0, 2500);
    const last = lastBits.match(/last[:\s]+([A-Z]\.?\s+[A-Z][A-Za-z.'\-]+|[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){0,3})\s*[\(]?\s*(QB|RB|WR|TE|K|DEF|DST)/i);
    if (last) {
      const expected = completedFromYahooClock(teams);
      const info = parseYahooNamePos(last[1] + " " + last[2]) || {
        name: last[1].replace(/\s+/g, " ").trim(),
        pos: String(last[2]).replace("DST", "DEF"),
        team: "",
      };
      if (expected >= 1 && info.name) addYahooDomPick(byPn, expected, info, "");
    }
  }

  function parseYahooDraftResultsHtml(html) {
    const byPn = new Map();
    const text = String(html || "").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/\s+/g, " ");
    const re =
      /(?:^|\s)(\d{1,3})[.)]?\s+([A-Z](?:[a-zA-Z.'\-]+|\.)(?:\s+(?:[A-Z][a-zA-Z.'\-]+|Jr\.?|Sr\.?|II|III|IV|V)){0,4})\s+(QB|RB|WR|TE|K|DEF|DST|D\/ST)\b/g;
    let m;
    let n = 0;
    while ((m = re.exec(text)) && n < 400) {
      n++;
      const info = parseYahooNamePos(m[2] + " " + m[3]);
      if (info) addYahooDomPick(byPn, Number(m[1]), info, "");
    }
    return Array.from(byPn.values()).sort(function (a, b) {
      return a.overallPickNumber - b.overallPickNumber;
    });
  }

  function scrapeYahooBoard(doc, teamsHint) {
    const docs = sameOriginDocuments(doc || (typeof document !== "undefined" ? document : null));
    const byPn = new Map();
    docs.forEach(function (d) {
      if (!d || !d.querySelectorAll) return;
      scrapeYahooLabeledPicks(d, byPn);
      scrapeYahooColumns(d, byPn, teamsHint);
      scrapeYahooDottedPicks(d, byPn, teamsHint);
    });
    const rows = Array.from(byPn.values()).sort(function (a, b) {
      return a.overallPickNumber - b.overallPickNumber;
    });
    return filterYahooPicksToClock(rows, teamsHint);
  }

  function addSleeperUserId(out, id, front) {
    const s = String(id || "").trim();
    if (!/^\d{6,20}$/.test(s)) return;
    if (out.seen[s]) {
      if (front) {
        out.userIds = [s].concat(out.userIds.filter(function (x) { return x !== s; }));
      }
      return;
    }
    out.seen[s] = true;
    if (front) out.userIds.unshift(s);
    else out.userIds.push(s);
  }

  function harvestSleeperUserObject(obj, key, out, depth) {
    if (!obj || typeof obj !== "object" || depth > 5) return;
    if (Array.isArray(obj)) {
      if (obj.length > 2 && obj[0] && typeof obj[0] === "object" && (obj[0].user_id || obj[0].display_name)) {
        return;
      }
      obj.slice(0, 6).forEach(function (item) {
        harvestSleeperUserObject(item, key, out, depth + 1);
      });
      return;
    }
    const uid = obj.user_id || obj.userId;
    const un = obj.username || obj.user_name;
    const dn = obj.display_name || obj.displayName;
    const team = obj.team_name || obj.teamName;
    const loggedInHint =
      obj.token ||
      obj.email ||
      obj.avatar ||
      obj.phone ||
      obj.verification ||
      obj.real_name != null ||
      obj.is_bot === false ||
      /user|auth|session|token|login|\bme\b/i.test(String(key || ""));
    if (uid && loggedInHint && (un || dn || obj.token || obj.email || obj.access_token || obj.accessToken)) {
      addSleeperUserId(out, uid, true);
      if (un) out.username = String(un);
      if (dn) out.displayName = String(dn);
      if (team) out.teamName = String(team);
      return;
    }
    ["user", "data", "session", "profile", "me", "account", "viewer"].forEach(function (k) {
      if (obj[k] && typeof obj[k] === "object") harvestSleeperUserObject(obj[k], k, out, depth + 1);
    });
  }

  function collectSleeperIdentity() {
    const out = { userIds: [], username: "", displayName: "", teamName: "", seen: {} };
    function inspect(text, key) {
      if (!text || text.length > 800000) return;
      const trimmed = String(text).trim();
      if (trimmed.charAt(0) === "{" || trimmed.charAt(0) === "[") {
        try {
          harvestSleeperUserObject(JSON.parse(trimmed), key, out, 0);
        } catch (_e) {
          /* fall through */
        }
      }
      const bare = trimmed.match(/^\d{6,20}$/);
      if (bare && /user_id|userid|sleeper_user/i.test(String(key || ""))) {
        addSleeperUserId(out, bare[0], true);
      }
      if (!out.userIds.length && /^(?:user|currentuser|current_user|me|session|auth|token|login|sleeper.?user)(?:[_-].*)?$/i.test(String(key || ""))) {
        const idRe = /"(?:user_id|userId)"\s*:\s*"?(\d{6,20})"?/g;
        const found = [];
        let m;
        while ((m = idRe.exec(trimmed))) {
          if (found.indexOf(m[1]) < 0) found.push(m[1]);
        }
        if (found.length && found.length <= 2) {
          found.forEach(function (id) { addSleeperUserId(out, id); });
        }
        if (!out.username) {
          const un = trimmed.match(/"username"\s*:\s*"([A-Za-z0-9_]{2,32})"/);
          if (un) out.username = un[1];
        }
        if (!out.displayName) {
          const dn = trimmed.match(/"display_name"\s*:\s*"([^"]{2,40})"/);
          if (dn) out.displayName = dn[1];
        }
      }
    }
    try {
      [localStorage, sessionStorage].forEach(function (store) {
        if (!store) return;
        for (let i = 0; i < store.length; i++) {
          const k = store.key(i);
          inspect(store.getItem(k) || "", k);
        }
      });
    } catch (_e) {
      /* ignore */
    }
    try {
      inspect(document.cookie || "", "cookie");
    } catch (_e) {
      /* ignore */
    }
    if (!out.username) {
      const fromDom = sleeperUsernameFromDom();
      if (fromDom) out.username = fromDom;
    }
    delete out.seen;
    return out;
  }

  function sleeperUsernameFromHref(href) {
    const m = String(href || "").match(/\/u\/([A-Za-z0-9_]+)/);
    if (!m) return "";
    const name = m[1];
    if (/^(help|support|blog|about|settings|login|signup)$/i.test(name)) return "";
    return name;
  }

  function sleeperUsernameFromDom(doc) {
    doc = doc || (typeof document !== "undefined" ? document : null);
    if (!doc || !doc.querySelectorAll) return "";
    const header = doc.querySelector("header, nav, [class*='Header'], [class*='header'], [class*='NavBar']");
    if (header && header.querySelectorAll) {
      const pinned = header.querySelectorAll('a[href*="/u/"]');
      for (let i = 0; i < Math.min(pinned.length, 8); i++) {
        const name = sleeperUsernameFromHref(pinned[i].getAttribute("href"));
        if (name) return name;
      }
    }
    const links = doc.querySelectorAll('a[href*="/u/"]');
    const counts = {};
    const n = Math.min(links.length, 40);
    for (let i = 0; i < n; i++) {
      const name = sleeperUsernameFromHref(links[i].getAttribute("href"));
      if (!name) continue;
      counts[name] = (counts[name] || 0) + 1;
    }
    const uniq = Object.keys(counts);
    if (uniq.length === 1) return uniq[0];
    return "";
  }

  function sleeperUserIdFromUsers(users, ident) {
    const names = [];
    const seen = {};
    function add(s) {
      const n = String(s || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (n.length < 2 || n.length > 40 || seen[n]) return;
      seen[n] = true;
      names.push(n);
    }
    add(ident && ident.username);
    add(ident && ident.displayName);
    add(ident && ident.teamName);
    if (!names.length) return "";
    for (let i = 0; i < (users || []).length; i++) {
      const u = users[i];
      if (!u || u.user_id == null) continue;
      const meta = u.metadata || {};
      const cands = [u.username, u.display_name, u.displayName, meta.team_name, u.team_name];
      const hit = cands.some(function (c) {
        const n = String(c || "").replace(/\s+/g, " ").trim().toLowerCase();
        return !!n && names.indexOf(n) >= 0;
      });
      if (hit) return String(u.user_id);
    }
    return "";
  }

  function userIdsInDraftOrder(order, userIds) {
    const ids = userIds || [];
    if (!order || typeof order !== "object") return ids;
    const hit = ids.filter(function (id) {
      return id && order[String(id)] != null;
    });
    return hit.length ? hit : ids;
  }

  function slotFromSleeperDraftOrder(order, userIds) {
    if (!order || typeof order !== "object") return 0;
    const ids = userIds || [];
    for (let i = 0; i < ids.length; i++) {
      const uid = String(ids[i] || "");
      if (!uid) continue;
      if (order[uid] != null && Number(order[uid]) >= 1) return Number(order[uid]);
    }
    return 0;
  }

  function slotFromSleeperPickedBy(picks, userIds) {
    const want = {};
    (userIds || []).forEach(function (id) {
      if (id) want[String(id)] = true;
    });
    if (!Object.keys(want).length) return 0;
    let first = 0;
    let slot = 0;
    (picks || []).forEach(function (p) {
      const who = p && (p.pickedBy != null ? p.pickedBy : p.picked_by);
      if (who == null || !want[String(who)]) return;
      const pn = Number(p.overallPickNumber || p.pick_no || 0);
      const ds = Number(p.slot || p.draft_slot || p.draftSlot || 0);
      if (ds >= 1 && (!first || (pn && pn < first) || !pn)) {
        first = pn || first;
        slot = ds;
      }
    });
    return slot;
  }

  function slotFromSleeperRosterMap(slotToRoster, ownerToRoster, userIds) {
    if (!slotToRoster || !ownerToRoster) return 0;
    for (let i = 0; i < (userIds || []).length; i++) {
      const uid = String(userIds[i] || "");
      if (!uid) continue;
      const rid = ownerToRoster[uid];
      if (rid == null) continue;
      const keys = Object.keys(slotToRoster);
      for (let k = 0; k < keys.length; k++) {
        if (String(slotToRoster[keys[k]]) === String(rid)) {
          const slot = Number(keys[k]);
          if (slot >= 1) return slot;
        }
      }
    }
    return 0;
  }

  function parseSleeperClock(text) {
    const s = String(text || "").replace(/\s+/g, " ");
    return {
      onClock: /you(?:'re| are) on the clock|waiting for you to pick|you(?:'re| are) up(?:\s+now)?(?:\s*[!.])?(?:\s|$)|your turn to pick|it(?:'s| is) your (?:pick|turn)/i.test(s),
      upIn: (function () {
        const m = s.match(/you(?:'re| are) up in\s+(\d+)\s+picks?/i);
        return m ? Number(m[1]) : null;
      })(),
    };
  }

  function sleeperClockText(doc) {
    doc = doc || (typeof document !== "undefined" ? document : null);
    if (!doc || !doc.querySelectorAll) return "";
    const bits = [];
    const nodes = doc.querySelectorAll(
      "h1,h2,h3,header,[class*='Clock'],[class*='clock'],[class*='Status'],[class*='status'],[class*='Banner'],[class*='banner'],[class*='Pick']"
    );
    const n = Math.min(nodes.length, 60);
    for (let i = 0; i < n; i++) bits.push(String(nodes[i].textContent || "").slice(0, 220));
    let blob = bits.join(" ");
    if (!/you(?:'re| are) on the clock|your (?:pick|turn)|you(?:'re| are) up in/i.test(blob) && doc.body) {
      blob = String(doc.body.innerText || "").slice(0, 4000);
    }
    return blob;
  }

  function slotFromSleeperClock(text, currentPick, teams) {
    const c = parseSleeperClock(text);
    const pn = Number(currentPick) || 0;
    const nTeams = Number(teams) || 0;
    if (nTeams < 2 || pn < 1) return 0;
    if (c.onClock) return snakeSlot(pn, nTeams);
    if (c.upIn != null && c.upIn >= 0) return snakeSlot(pn + c.upIn, nTeams);
    return 0;
  }

  function detectSleeperDomSlot(identity, teams) {
    const names = [];
    const seen = {};
    function addName(s) {
      const n = String(s || "").replace(/\s+/g, " ").trim().toLowerCase();
      if (n.length < 2 || n.length > 40) return;
      if (seen[n]) return;
      seen[n] = true;
      names.push(n);
    }
    addName(identity && identity.displayName);
    addName(identity && identity.username);
    addName(identity && identity.teamName);
    if (names.length && typeof document !== "undefined" && document.createTreeWalker) {
      const scope =
        document.querySelector(
          '[class*="draft-board"], [class*="draftBoard"], [class*="draft-container"], [class*="draftContainer"], [id*="draft"]'
        ) || document.body;
      if (scope) {
        const walker = document.createTreeWalker(scope, NodeFilter.SHOW_ELEMENT);
        let el;
        let scanned = 0;
        while ((el = walker.nextNode()) && scanned < 3500) {
          scanned++;
          if (el.children && el.children.length > 3) continue;
          const label = String(el.getAttribute("aria-label") || el.textContent || "")
            .replace(/\s+/g, " ")
            .trim()
            .toLowerCase();
          if (!label || label.length > 48) continue;
          const hit = names.some(function (n) {
            return label === n || label.indexOf(n) === 0 || label.indexOf(n + " ") === 0;
          });
          if (!hit) continue;
          const slot = slotFromColumnNode(el);
          if (slot) return clampSlot(slot, teams || 32);
        }
      }
    }
    return detectDomSlot();
  }

  function detectSleeperSlot(opts) {
    opts = opts || {};
    const teams = Number(opts.teams) || 0;
    const identity = opts.identity || (opts.skipCollect ? { userIds: [] } : collectSleeperIdentity());
    const userIds = userIdsInDraftOrder(opts.draft && opts.draft.draft_order, (identity && identity.userIds) || []);
    const draft = opts.draft || {};
    const picks = opts.picks || [];
    const max = teams || 32;
    const current = Number(opts.currentPick) || (picks.length ? picks[picks.length - 1].overallPickNumber + 1 : 1);
    const auction = !!opts.auction || String((draft && draft.type) || "").toLowerCase() === "auction";
    // Host clock wins. draft_order / roster maps can bind a league-mate id
    // and park you on the wrong seat while Sleeper has you picking 1.07.
    if (!auction) {
      const clockText = opts.clockText != null ? opts.clockText : opts.skipDom ? "" : sleeperClockText();
      const clockSlot = slotFromSleeperClock(clockText, current, teams || 12);
      if (clockSlot) return clampSlot(clockSlot, max);
    }
    let slot = slotFromSleeperDraftOrder(draft.draft_order, userIds);
    if (slot) return clampSlot(slot, max);
    slot = slotFromSleeperPickedBy(picks, userIds);
    if (slot) return clampSlot(slot, max);
    const rosterMap = draft.slot_to_roster_id || (draft.metadata && draft.metadata.slot_to_roster_id);
    slot = slotFromSleeperRosterMap(rosterMap, opts.ownerToRoster, userIds);
    if (slot) return clampSlot(slot, max);
    if (!opts.skipDom) {
      slot = detectSleeperDomSlot(identity, max);
      if (slot) return clampSlot(slot, max);
    }
    return 0;
  }

  function teamNamesFromSleeperDraft(draft, users) {
    const names = {};
    const order = (draft && draft.draft_order) || {};
    const byId = {};
    (users || []).forEach(function (u) {
      if (!u || u.user_id == null) return;
      byId[String(u.user_id)] = u;
    });
    Object.keys(order).forEach(function (uid) {
      const slot = Number(order[uid]);
      if (!(slot >= 1 && slot <= 32)) return;
      const u = byId[uid];
      if (!u) return;
      const meta = u.metadata || {};
      const name = String(meta.team_name || u.display_name || u.username || "").trim();
      if (name) names[slot] = name;
    });
    return names;
  }

  function teamNamesFromTeamIds(namesById, slotsById, picks, teams) {
    const out = {};
    const ids = namesById || {};
    const slots = slotsById || {};
    Object.keys(ids).forEach(function (tid) {
      const name = String(ids[tid] || "").trim();
      if (!name) return;
      let slot = Number(slots[tid]) || 0;
      if (!slot) slot = slotFromTeamId(picks, tid, teams);
      if (slot >= 1 && slot <= 32) out[slot] = name;
    });
    return out;
  }

  function sleeperPickOwners(opts) {
    opts = opts || {};
    const teams = Number(opts.teams) || 0;
    const rounds = Number(opts.rounds) || 0;
    const out = {};
    if (teams < 2 || rounds < 1) return out;
    const draft = opts.draft || {};
    const slotToRoster = {};
    const rosterToSlot = {};
    const rawMap = draft.slot_to_roster_id || (draft.metadata && draft.metadata.slot_to_roster_id) || {};
    Object.keys(rawMap).forEach(function (sl) {
      const slot = Number(sl);
      const rid = rawMap[sl];
      if (slot >= 1 && rid != null) {
        slotToRoster[slot] = rid;
        rosterToSlot[String(rid)] = slot;
      }
    });
    const order = draft.draft_order || {};
    const ownerToRoster = opts.ownerToRoster || {};
    Object.keys(order).forEach(function (uid) {
      const slot = Number(order[uid]);
      const rid = ownerToRoster[uid];
      if (slot >= 1 && rid != null) {
        slotToRoster[slot] = rid;
        rosterToSlot[String(rid)] = slot;
      }
    });
    const traded = {};
    (opts.tradedPicks || []).forEach(function (tp) {
      if (!tp || tp.roster_id == null || tp.round == null || tp.owner_id == null) return;
      traded[String(tp.roster_id) + ":" + String(tp.round)] = tp.owner_id;
    });
    (opts.picks || []).forEach(function (p) {
      const pn = Number(p && (p.overallPickNumber || p.pick_no));
      if (!pn) return;
      const rid = p.teamId != null ? p.teamId : p.roster_id;
      if (rid != null && rosterToSlot[String(rid)] != null) {
        out[pn] = rosterToSlot[String(rid)];
      } else if (Number(p.slot || p.draft_slot) >= 1) {
        out[pn] = Number(p.slot || p.draft_slot);
      }
    });
    const tot = teams * rounds;
    for (let pn = 1; pn <= tot; pn++) {
      if (out[pn]) continue;
      const home = snakeSlot(pn, teams);
      const origRid = slotToRoster[home];
      const rnd = Math.ceil(pn / teams);
      if (origRid != null) {
        const key = String(origRid) + ":" + String(rnd);
        const ownRid = Object.prototype.hasOwnProperty.call(traded, key) ? traded[key] : origRid;
        out[pn] = rosterToSlot[String(ownRid)] != null ? rosterToSlot[String(ownRid)] : home;
      } else {
        out[pn] = home;
      }
    }
    return out;
  }

  function parseClockSeconds(text) {
    const s = String(text || "");
    const mmss = s.match(/\b(\d{1,2}):([0-5]\d)\b/);
    if (mmss) return Number(mmss[1]) * 60 + Number(mmss[2]);
    const sec = s.match(/\b(\d{1,3})\s*(?:s|sec|secs|seconds)\b/i);
    if (sec) return Number(sec[1]);
    return null;
  }

  function sleeperClockRemaining(draft, now) {
    draft = draft || {};
    const settings = draft.settings || {};
    const timer = Number(settings.pick_timer || settings.pickTimer || 0);
    if (!(timer > 0)) return null;
    const last = Number(draft.last_picked || draft.lastPicked || draft.start_time || 0);
    if (!(last > 0)) return timer;
    const elapsed = Math.floor(((now || Date.now()) - last) / 1000);
    return Math.max(0, Math.round(timer - elapsed));
  }

  function scrapeHostClockSeconds(doc) {
    doc = doc || (typeof document !== "undefined" ? document : null);
    if (!doc || !doc.querySelectorAll) return null;
    const nodes = doc.querySelectorAll(
      "[class*='clock'],[class*='Clock'],[class*='timer'],[class*='Timer'],[class*='countdown'],[class*='Countdown']"
    );
    const n = Math.min(nodes.length, 40);
    for (let i = 0; i < n; i++) {
      const sec = parseClockSeconds(nodes[i].textContent);
      if (sec != null && sec <= 600) return sec;
    }
    if (doc.body) {
      const blob = String(doc.body.innerText || "").slice(0, 2500);
      const near = blob.match(/(?:clock|timer|remaining)[^\d]{0,24}(\d{1,2}:[0-5]\d)/i);
      if (near) return parseClockSeconds(near[1]);
    }
    return null;
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
    isSleeperDraftRoom: isSleeperDraftRoom,
    sleeperDraftIdFromUrl: sleeperDraftIdFromUrl,
    sleeperLeagueIdFromUrl: sleeperLeagueIdFromUrl,
    clampSlot: clampSlot,
    normDraftPos: normDraftPos,
    isKDefPos: isKDefPos,
    rosterFromEspnSlots: rosterFromEspnSlots,
    rosterFromSleeperSettings: rosterFromSleeperSettings,
    rosterFromYahooPositions: rosterFromYahooPositions,
    rosterHasStarters: rosterHasStarters,
    slotListFromRoster: slotListFromRoster,
    settingsLabel: settingsLabel,
    scoringFromSleeperSettings: scoringFromSleeperSettings,
    rosterKey: rosterKey,
    scrapeYahooBoard: scrapeYahooBoard,
    parseYahooDraftResultsHtml: parseYahooDraftResultsHtml,
    parseYahooCompactPick: parseYahooCompactPick,
    filterYahooPicksToClock: filterYahooPicksToClock,
    yahooDottedToOverall: yahooDottedToOverall,
    mergeYahooPicks: mergeYahooPicks,
    parseYahooNamePos: parseYahooNamePos,
    parseYahooLooseName: parseYahooLooseName,
    sameOriginDocuments: sameOriginDocuments,
    completedFromYahooClock: completedFromYahooClock,
    collectSleeperIdentity: collectSleeperIdentity,
    sleeperUsernameFromDom: sleeperUsernameFromDom,
    sleeperUserIdFromUsers: sleeperUserIdFromUsers,
    userIdsInDraftOrder: userIdsInDraftOrder,
    slotFromSleeperDraftOrder: slotFromSleeperDraftOrder,
    slotFromSleeperPickedBy: slotFromSleeperPickedBy,
    slotFromSleeperRosterMap: slotFromSleeperRosterMap,
    parseSleeperClock: parseSleeperClock,
    slotFromSleeperClock: slotFromSleeperClock,
    detectSleeperSlot: detectSleeperSlot,
    detectSleeperDomSlot: detectSleeperDomSlot,
    teamNamesFromSleeperDraft: teamNamesFromSleeperDraft,
    teamNamesFromTeamIds: teamNamesFromTeamIds,
    sleeperPickOwners: sleeperPickOwners,
    parseClockSeconds: parseClockSeconds,
    sleeperClockRemaining: sleeperClockRemaining,
    scrapeHostClockSeconds: scrapeHostClockSeconds,
  };
})(window);
