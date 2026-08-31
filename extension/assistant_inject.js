// Docks the BR Draft Assistant overlay as a sidebar iframe on Sleeper,
// Yahoo, and ESPN draft rooms. Reads host picks (never submits).

(function () {
  "use strict";

  if (window.__brFantasyAssistantInject) return;
  try {
    if (window.top !== window) return;
  } catch (_e) {
    return;
  }
  window.__brFantasyAssistantInject = true;

  const ROOT_ID = "br-fantasy-assistant-root";
  const WIDTH = 400;
  const COLLAPSED = 48;
  const SLIDE_MS = 240;
  const SLIDE_EASE = "cubic-bezier(0.22, 1, 0.36, 1)";
  let iframe = null;
  let ready = false;
  let queuedPicks = null;
  let queuedPool = null;
  let collapsed = false;
  let poolRetryTimer = null;
  let lastPoolKey = "";
  let adpSource = "consensus";
  let lastScoring = { ppr: 1, tep: 0, passTd: 4 };
  let slideTimer = null;

  function platformFromHost() {
    const h = String(location.hostname || "").toLowerCase();
    if (h.indexOf("espn") >= 0) return "espn";
    if (h.indexOf("yahoo") >= 0) return "yahoo";
    return "sleeper";
  }

  function isHostDraftRoom() {
    if (platformFromHost() !== "espn") return true;
    if (window.BRDraftSlot && typeof window.BRDraftSlot.isEspnDraftRoom === "function") {
      return window.BRDraftSlot.isEspnDraftRoom();
    }
    const path = String(location.pathname || "").toLowerCase();
    if (/mockdraftlobby|draftlobby/.test(path)) return false;
    return /(?:^|\/)(?:live)?draft(?:\/|$)/.test(path) || /(?:^|\/)mockdraft(?:\/|$)/.test(path);
  }

  function dockShiftPx() {
    return collapsed ? COLLAPSED : WIDTH;
  }

  function hostRoots() {
    const out = [];
    ["root", "app", "__next", "draft", "draftapp", "main-content"].forEach(function (id) {
      const el = document.getElementById(id);
      if (el) out.push(el);
    });
    ["main", "[data-reactroot]", "[data-app]"].forEach(function (sel) {
      try {
        const el = document.querySelector(sel);
        if (el) out.push(el);
      } catch (_e) { /* ignore */ }
    });
    return out;
  }

  function looksFullBleed(el) {
    if (!el || !el.getBoundingClientRect) return false;
    const r = el.getBoundingClientRect();
    if (r.width >= window.innerWidth - 24 && r.left <= 12) return true;
    const inline = String((el.style && el.style.width) || "") + String((el.style && el.style.maxWidth) || "");
    if (/100vw/.test(inline)) return true;
    try {
      const st = window.getComputedStyle(el);
      if ((st.position === "fixed" || st.position === "absolute") && r.width >= window.innerWidth - 24 && r.left <= 12) {
        return true;
      }
    } catch (_e) { /* ignore */ }
    return false;
  }

  function shiftShell(el, px, cap, root) {
    if (!el || el === root || (root && root.contains(el))) return;
    if (/^(SCRIPT|STYLE|LINK|META|NOSCRIPT)$/.test(el.tagName)) return;
    if (el === document.body) return;
    el.setAttribute("data-br-da-shifted", "1");
    el.style.setProperty("box-sizing", "border-box", "important");
    el.style.setProperty("min-width", "0", "important");
    try {
      const st = window.getComputedStyle(el);
      if (st.position === "fixed" || st.position === "absolute") {
        const r = el.getBoundingClientRect();
        if (r.left <= 12 && r.width >= window.innerWidth - 80) {
          el.style.setProperty("left", "0px", "important");
          el.style.setProperty("right", px + "px", "important");
          el.style.setProperty("width", "auto", "important");
          el.style.setProperty("max-width", "none", "important");
          return;
        }
      }
    } catch (_e) { /* ignore */ }
    el.style.setProperty("width", "100%", "important");
    el.style.setProperty("max-width", "100%", "important");
    void cap;
  }

  function applyDockShift() {
    const px = dockShiftPx();
    const root = document.getElementById(ROOT_ID);
    const cap = Math.max(320, window.innerWidth - px);
    const html = document.documentElement;
    html.classList.add("br-da-docked");
    html.classList.add("br-da-" + platformFromHost());
    html.style.setProperty("--br-da-shift", px + "px");
    const shells = [];
    if (document.body) {
      Array.prototype.forEach.call(document.body.children, function (el) { shells.push(el); });
      const nested = document.querySelectorAll("body > *:not(#" + ROOT_ID + ") > *");
      for (let i = 0; i < Math.min(nested.length, 48); i++) shells.push(nested[i]);
    }
    hostRoots().forEach(function (el) { shells.push(el); });
    try {
      const vw = document.querySelectorAll('[style*="100vw"]');
      for (let i = 0; i < Math.min(vw.length, 40); i++) shells.push(vw[i]);
    } catch (_e) { /* ignore */ }
    shells.forEach(function (el) {
      if (!el || el === root || (root && root.contains(el))) return;
      if (/^(SCRIPT|STYLE|LINK|META|NOSCRIPT)$/.test(el.tagName)) return;
      if (el.getAttribute("data-br-da-shifted") === "1" || looksFullBleed(el)) {
        shiftShell(el, px, cap, root);
      }
    });
  }

  function ensureDockCss() {
    if (document.getElementById("br-da-dock-css")) return;
    const style = document.createElement("style");
    style.id = "br-da-dock-css";
    style.textContent =
      "html.br-da-docked{--br-da-shift:" + WIDTH + "px;width:100%!important;max-width:100%!important;height:100%!important;box-sizing:border-box!important;overflow-x:hidden!important;}" +
      "html.br-da-docked.br-da-collapsed{--br-da-shift:" + COLLAPSED + "px;}" +
      "html.br-da-docked body{display:flex!important;flex-direction:row!important;align-items:stretch!important;width:100%!important;max-width:100%!important;min-height:100%!important;height:100%!important;margin:0!important;overflow-x:hidden!important;box-sizing:border-box!important;}" +
      "html.br-da-docked body>*:not(#" + ROOT_ID + "){flex:1 1 auto!important;min-width:0!important;max-width:100%!important;box-sizing:border-box!important;}" +
      "html.br-da-docked.br-da-yahoo #root,html.br-da-docked.br-da-yahoo #app,html.br-da-docked.br-da-yahoo #__next,html.br-da-docked.br-da-yahoo #draft,html.br-da-docked.br-da-yahoo #draftapp,html.br-da-docked.br-da-sleeper #root,html.br-da-docked.br-da-sleeper #app,html.br-da-docked.br-da-sleeper [data-reactroot]{width:auto!important;max-width:100%!important;min-width:0!important;flex:1 1 auto!important;box-sizing:border-box!important;}" +
      "html.br-da-docked.br-da-ready body{transition:none;}" +
      "#" + ROOT_ID + "{position:relative;flex:0 0 var(--br-da-shift);width:var(--br-da-shift);align-self:stretch;min-height:100vh;z-index:2147483645;box-shadow:none;border-left:1px solid rgba(18,45,75,.35);background:#122d4b;" +
      "contain:layout paint;overflow:hidden;}" +
      "html.br-da-ready #" + ROOT_ID + "{transition:flex-basis " + SLIDE_MS + "ms " + SLIDE_EASE + ",width " + SLIDE_MS + "ms " + SLIDE_EASE + ";}" +
      "#" + ROOT_ID + " iframe{width:100%;height:100%;border:0;background:transparent;display:block;min-height:100%;}" +
      "html.br-da-collapsed #" + ROOT_ID + " iframe{pointer-events:none;}" +
      "#br-fantasy-assistant-expand{display:flex;position:absolute;left:0;top:0;bottom:0;width:" +
      COLLAPSED + "px;z-index:2;margin:0;padding:10px 0;border:0;border-left:1px solid rgba(255,255,255,.14);" +
      "background:#122d4b;color:#fff;cursor:pointer;flex-direction:column;align-items:center;" +
      "justify-content:flex-start;gap:12px;opacity:0;pointer-events:none;}" +
      "html.br-da-ready #br-fantasy-assistant-expand{transition:opacity 120ms ease;}" +
      "html.br-da-collapsed #br-fantasy-assistant-expand{opacity:1;pointer-events:auto;}" +
      "#br-fantasy-assistant-expand:hover{background:#1a3d63;}" +
      "#br-fantasy-assistant-expand .br-da-expand-logo{width:34px;height:auto;margin-top:4px;display:block;}" +
      "#br-fantasy-assistant-expand svg{width:16px;height:16px;flex-shrink:0;}" +
      "@media (prefers-reduced-motion:reduce){html.br-da-ready #" +
      ROOT_ID + ",html.br-da-ready #br-fantasy-assistant-expand{transition:none !important;}}" +
      "#br-fantasy-espn-sync-chip,#br-fantasy-yahoo-sync-chip{display:none!important;}";
    (document.head || document.documentElement).appendChild(style);
  }

  function mount() {
    if (document.getElementById(ROOT_ID)) {
      iframe = document.querySelector("#" + ROOT_ID + " iframe");
      return;
    }
    ensureDockCss();
    document.documentElement.classList.add("br-da-docked");
    document.documentElement.classList.add("br-da-" + platformFromHost());
    const wrap = document.createElement("div");
    wrap.id = ROOT_ID;
    iframe = document.createElement("iframe");
    iframe.title = "BR Fantasy Draft Assistant";
    iframe.src = chrome.runtime.getURL("overlay.html") + "?embed=1";
    wrap.appendChild(iframe);
    const expand = document.createElement("button");
    expand.id = "br-fantasy-assistant-expand";
    expand.type = "button";
    expand.title = "Open Draft Assistant";
    expand.setAttribute("aria-label", "Open Draft Assistant");
    const logoUrl = chrome.runtime.getURL("icons/br-logo-dark.png");
    expand.innerHTML =
      '<img class="br-da-expand-logo" alt="BR Fantasy" src="' + logoUrl + '">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';
    expand.addEventListener("click", function () { setCollapsed(false); });
    wrap.appendChild(expand);
    (document.body || document.documentElement).appendChild(wrap);
    applyDockShift();
    requestAnimationFrame(function () {
      if (wrap.parentNode !== document.body && document.body) {
        document.body.appendChild(wrap);
      }
      applyDockShift();
      requestAnimationFrame(function () {
        document.documentElement.classList.add("br-da-ready");
        applyDockShift();
      });
    });
    window.addEventListener("resize", applyDockShift);
    setInterval(applyDockShift, 2500);
    try {
      if (!window.__brDaDockMo && document.documentElement) {
        window.__brDaDockMo = new MutationObserver(function () {
          applyDockShift();
        });
        window.__brDaDockMo.observe(document.documentElement, { childList: true, subtree: false });
        if (document.body) {
          window.__brDaDockMo.observe(document.body, { childList: true, subtree: false, attributes: true, attributeFilter: ["style", "class"] });
        }
      }
    } catch (_e) { /* ignore */ }
  }

  function postToOverlay(msg) {
    if (!msg) return;
    if (!iframe || !iframe.contentWindow || !ready) {
      if (msg.type === "pool") queuedPool = msg;
      else if (msg.type === "picks" || msg.type === "sync") queuedPicks = msg;
      return;
    }
    try {
      iframe.contentWindow.postMessage(Object.assign({ __br: "br-da" }, msg), "*");
    } catch (_e) {
      /* ignore */
    }
  }

  function requestPool(extra) {
    const opts = Object.assign(
      {
        type: "fetchDraftPool",
        scoringType: "redraft",
        adpSource: adpSource || "consensus",
        sf: false,
        teams: 12,
        ppr: lastScoring.ppr,
        tep: lastScoring.tep,
        passTd: lastScoring.passTd,
      },
      extra || {}
    );
    lastPoolKey = [
      opts.scoringType,
      opts.sf ? "sf" : "1qb",
      opts.adpSource,
      opts.teams || 12,
      opts.ppr,
      opts.tep,
      opts.passTd,
    ].join("|");
    adpSource = String(opts.adpSource || "consensus");
    try {
      chrome.runtime.sendMessage(opts, function (resp) {
        void chrome.runtime.lastError;
        if (!resp || !Array.isArray(resp.players) || !resp.players.length) {
          if (typeof window.__brDaSetSync === "function") {
            window.__brDaSetSync(false, "BR ranks · retrying…");
          }
          if (!poolRetryTimer) {
            poolRetryTimer = setTimeout(function () {
              poolRetryTimer = null;
              requestPool(extra);
            }, 6000);
          }
          return;
        }
        if (poolRetryTimer) {
          clearTimeout(poolRetryTimer);
          poolRetryTimer = null;
        }
        postToOverlay({
          type: "pool",
          players: resp.players,
          scoringType: resp.scoringType || "redraft",
          sf: !!resp.sf,
          adpSource: resp.adpSource || adpSource,
          adpOptions: Array.isArray(resp.adpOptions) ? resp.adpOptions : [],
        });
      });
    } catch (_e) {
      /* ignore */
    }
  }

  function setCollapsed(on) {
    collapsed = !!on;
    if (iframe) iframe.style.visibility = "";
    if (slideTimer) {
      clearTimeout(slideTimer);
      slideTimer = null;
    }
    const root = document.getElementById(ROOT_ID);
    if (root) root.style.willChange = "width,flex-basis";
    document.documentElement.classList.toggle("br-da-collapsed", collapsed);
    applyDockShift();
    if (collapsed) {
      slideTimer = setTimeout(function () {
        slideTimer = null;
        if (root) root.style.willChange = "";
        postToOverlay({ type: "collapsed", on: true });
      }, SLIDE_MS);
    } else {
      postToOverlay({ type: "collapsed", on: false });
      slideTimer = setTimeout(function () {
        slideTimer = null;
        if (root) root.style.willChange = "";
      }, SLIDE_MS);
    }
  }

  let lastPickFp = "";

  window.__brDaPushPicks = function (detail) {
    const payload = Object.assign(
      {
        type: "picks",
        platform: platformFromHost(),
        syncText: platformFromHost().toUpperCase() + " · SYNCED",
      },
      detail || {}
    );
    const picks = payload.picks || [];
    const last = picks.length ? picks[picks.length - 1] : null;
    const fp = [
      picks.length,
      last ? (last.overallPickNumber || last.pick_no || 0) : 0,
      last ? (last.playerId || "") : "",
      last ? (last.playerName || last.name || "") : "",
      payload.teams || "",
      payload.mySlot || "",
      payload.rounds || "",
      payload.inProgress ? 1 : 0,
      payload.drafted ? 1 : 0,
      payload.sf ? 1 : 0,
      payload.ppr != null ? payload.ppr : "",
      payload.tep != null ? payload.tep : "",
      payload.passTd != null ? payload.passTd : "",
      payload.teamNames ? Object.keys(payload.teamNames).length : "",
      payload.pickOwners ? Object.keys(payload.pickOwners).length : "",
      window.BRDraftSlot && BRDraftSlot.rosterKey ? BRDraftSlot.rosterKey(payload.roster) : ""
    ].join("|");
    const teams = Number(payload.teams || 0);
    const sf = !!payload.sf;
    if (payload.ppr != null && isFinite(Number(payload.ppr))) lastScoring.ppr = Number(payload.ppr);
    if (payload.tep != null && isFinite(Number(payload.tep))) lastScoring.tep = Number(payload.tep);
    if (payload.passTd != null && isFinite(Number(payload.passTd))) lastScoring.passTd = Number(payload.passTd);
    const key = [
      "redraft",
      sf ? "sf" : "1qb",
      adpSource,
      teams >= 8 ? teams : 12,
      lastScoring.ppr,
      lastScoring.tep,
      lastScoring.passTd,
    ].join("|");
    if (key !== lastPoolKey) {
      requestPool({
        teams: teams >= 8 ? teams : 12,
        sf: sf,
        adpSource: adpSource,
        ppr: lastScoring.ppr,
        tep: lastScoring.tep,
        passTd: lastScoring.passTd,
      });
    }
    if (fp === lastPickFp) return;
    lastPickFp = fp;
    postToOverlay(payload);
  };

  window.__brDaSetSync = function (ok, text) {
    postToOverlay({ type: "sync", ok: !!ok, text: text || "" });
  };

  window.__brDaPushClock = function (detail) {
    postToOverlay(Object.assign({ type: "clock", clockAt: Date.now() }, detail || {}));
  };

  window.addEventListener("message", function (ev) {
    const msg = ev.data;
    if (!msg || msg.__br !== "br-da") return;
    if (msg.type === "ready") {
      ready = true;
      if (msg.adpSource) adpSource = String(msg.adpSource).toLowerCase() || adpSource;
      const hadPool = !!queuedPool;
      if (queuedPool) {
        postToOverlay(queuedPool);
        queuedPool = null;
      }
      if (queuedPicks) {
        postToOverlay(queuedPicks);
        queuedPicks = null;
      } else {
        postToOverlay({
          type: "picks",
          platform: platformFromHost(),
          picks: [],
          syncText: platformFromHost().toUpperCase() + " · watching",
        });
      }
      if (!hadPool) requestPool({ adpSource: adpSource });
      return;
    }
    if (msg.type === "adp") {
      adpSource = String(msg.adpSource || "consensus").toLowerCase() || "consensus";
      requestPool({ adpSource: adpSource, force: true });
      return;
    }
    if (msg.type === "collapse") {
      setCollapsed(!collapsed);
      return;
    }
    if (msg.type === "reconnect") {
      requestPool({ force: true });
      try {
        document.dispatchEvent(new CustomEvent("brfantasy:assistant-reconnect", { bubbles: true }));
      } catch (_e) {
        /* ignore */
      }
      return;
    }
    if (msg.type === "open") {
      const dest = msg.dest === "sheet" ? "/draft/cheat-sheet" : "/draft";
      try {
        window.open("https://www.brfantasyfootball.com" + dest, "_blank", "noopener");
      } catch (_e) {
        /* ignore */
      }
    }
  });

  function tryMount() {
    if (!isHostDraftRoom()) return false;
    requestPool();
    mount();
    return true;
  }

  if (!tryMount()) {
    const wait = setInterval(function () {
      if (tryMount()) clearInterval(wait);
    }, 1000);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryMount);
  }
})();
