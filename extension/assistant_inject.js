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

  function ensureDockCss() {
    if (document.getElementById("br-da-dock-css")) return;
    const style = document.createElement("style");
    style.id = "br-da-dock-css";
    style.textContent =
      "html.br-da-docked{margin-right:" + WIDTH + "px !important;}" +
      "html.br-da-docked.br-da-collapsed{margin-right:" + COLLAPSED + "px !important;}" +
      "html.br-da-docked.br-da-ready{transition:margin-right " + SLIDE_MS + "ms " + SLIDE_EASE + " !important;}" +
      "#" + ROOT_ID + "{position:fixed;top:0;right:0;bottom:0;width:" + WIDTH +
      "px;z-index:2147483645;box-shadow:-8px 0 28px rgba(0,0,0,.22);background:#122d4b;" +
      "contain:layout paint;transform:translateX(0);}" +
      "html.br-da-ready #" + ROOT_ID + "{transition:transform " + SLIDE_MS + "ms " + SLIDE_EASE + ";}" +
      "html.br-da-collapsed #" + ROOT_ID + "{transform:translateX(calc(100% - " + COLLAPSED + "px));}" +
      "#" + ROOT_ID + " iframe{width:100%;height:100%;border:0;background:transparent;display:block;}" +
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
      "@media (prefers-reduced-motion:reduce){html.br-da-docked.br-da-ready,html.br-da-ready #" +
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
    document.documentElement.appendChild(wrap);
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        document.documentElement.classList.add("br-da-ready");
      });
    });
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
      },
      extra || {}
    );
    lastPoolKey = [opts.scoringType, opts.sf ? "sf" : "1qb", opts.adpSource, opts.teams || 12].join("|");
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
    if (root) root.style.willChange = "transform";
    document.documentElement.classList.toggle("br-da-collapsed", collapsed);
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
      last ? (last.playerId || last.playerName || "") : "",
      payload.teams || "",
      payload.mySlot || "",
      payload.rounds || "",
      payload.inProgress ? 1 : 0,
      payload.drafted ? 1 : 0,
      payload.sf ? 1 : 0,
      payload.ppr != null ? payload.ppr : "",
      payload.tep != null ? payload.tep : "",
      payload.passTd != null ? payload.passTd : "",
      window.BRDraftSlot && BRDraftSlot.rosterKey ? BRDraftSlot.rosterKey(payload.roster) : ""
    ].join("|");
    const teams = Number(payload.teams || 0);
    const sf = !!payload.sf;
    const key = ["redraft", sf ? "sf" : "1qb", adpSource, teams >= 8 ? teams : 12].join("|");
    if (key !== lastPoolKey) requestPool({ teams: teams >= 8 ? teams : 12, sf: sf, adpSource: adpSource });
    if (fp === lastPickFp) return;
    lastPickFp = fp;
    postToOverlay(payload);
  };

  window.__brDaSetSync = function (ok, text) {
    postToOverlay({ type: "sync", ok: !!ok, text: text || "" });
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
