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
  let iframe = null;
  let ready = false;
  let queuedPicks = null;
  let queuedPool = null;
  let collapsed = false;
  let poolRetryTimer = null;
  let lastPoolKey = "";

  function platformFromHost() {
    const h = String(location.hostname || "").toLowerCase();
    if (h.indexOf("espn") >= 0) return "espn";
    if (h.indexOf("yahoo") >= 0) return "yahoo";
    return "sleeper";
  }

  function ensureDockCss() {
    if (document.getElementById("br-da-dock-css")) return;
    const style = document.createElement("style");
    style.id = "br-da-dock-css";
    style.textContent =
      "html.br-da-docked{margin-right:" + WIDTH + "px !important;}" +
      "html.br-da-docked.br-da-collapsed{margin-right:" + COLLAPSED + "px !important;}" +
      "#" + ROOT_ID + "{position:fixed;top:0;right:0;bottom:0;width:" + WIDTH +
      "px;z-index:2147483645;box-shadow:-8px 0 28px rgba(0,0,0,.22);background:transparent;}" +
      "html.br-da-collapsed #" + ROOT_ID + "{width:" + COLLAPSED + "px;}" +
      "#" + ROOT_ID + " iframe{width:100%;height:100%;border:0;background:transparent;display:block;}" +
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
    document.documentElement.appendChild(wrap);
  }

  function postToOverlay(msg) {
    if (!msg) return;
    if (!iframe || !iframe.contentWindow || !ready) {
      if (msg.type === "pool") queuedPool = msg;
      else queuedPicks = msg;
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
        adpSource: "consensus",
        sf: false,
        teams: 12,
      },
      extra || {}
    );
    lastPoolKey = [opts.scoringType, opts.sf ? "sf" : "1qb", opts.adpSource, opts.teams || 12].join("|");
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
        });
      });
    } catch (_e) {
      /* ignore */
    }
  }

  function setCollapsed(on) {
    collapsed = !!on;
    document.documentElement.classList.toggle("br-da-collapsed", collapsed);
  }

  window.__brDaPushPicks = function (detail) {
    const payload = Object.assign(
      {
        type: "picks",
        platform: platformFromHost(),
        syncText: platformFromHost().toUpperCase() + " · SYNCED",
      },
      detail || {}
    );
    const teams = Number(payload.teams || 0);
    const sf = !!payload.sf;
    const key = ["redraft", sf ? "sf" : "1qb", "consensus", teams >= 8 ? teams : 12].join("|");
    if (key !== lastPoolKey) requestPool({ teams: teams >= 8 ? teams : 12, sf: sf });
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

  requestPool();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
