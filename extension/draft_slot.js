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
    return /\(\s*you\s*\)/i.test(s) || /\byour\s+team\b/i.test(s) || /^you$/i.test(s);
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

  function clampSlot(slot, teams) {
    const n = Number(slot) || 0;
    const max = Number(teams) || 32;
    if (n < 1) return 0;
    return Math.max(1, Math.min(max, n));
  }

  root.BRDraftSlot = {
    snakeSlot: snakeSlot,
    teamCountFromPicks: teamCountFromPicks,
    slotFromTeamId: slotFromTeamId,
    readCookie: readCookie,
    espnSwid: espnSwid,
    compactSync: compactSync,
    detectDomSlot: detectDomSlot,
    isEspnDraftRoom: isEspnDraftRoom,
    clampSlot: clampSlot,
  };
})(window);
