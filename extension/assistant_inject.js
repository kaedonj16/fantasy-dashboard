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
  const INVITE_ID = "br-fantasy-assistant-invite";
  const LAUNCH_KEY = "br-da-launch";
  const PRODUCT_VERSION = "1.0.0";
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

  function hostLabel() {
    const p = platformFromHost();
    if (p === "espn") return "ESPN";
    if (p === "yahoo") return "Yahoo";
    return "Sleeper";
  }

  function extensionAsset(path) {
    try {
      return chrome.runtime.getURL(path);
    } catch (_e) {
      return "";
    }
  }

  function isHostDraftRoom() {
    const platform = platformFromHost();
    if (platform === "sleeper") {
      if (window.BRDraftSlot && typeof window.BRDraftSlot.isSleeperDraftRoom === "function") {
        return window.BRDraftSlot.isSleeperDraftRoom();
      }
      const path = String(location.pathname || "") + String(location.hash || "");
      return /\/draft\//i.test(path) || /\/leagues\/\d+\/draft/i.test(path);
    }
    if (platform !== "espn") return true;
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
    const logoUrl = extensionAsset("icons/br-logo-dark.png");
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

  function readLaunchChoice() {
    try {
      return String(sessionStorage.getItem(LAUNCH_KEY) || "");
    } catch (_e) {
      return "";
    }
  }

  function writeLaunchChoice(value) {
    try {
      sessionStorage.setItem(LAUNCH_KEY, value);
    } catch (_e) { /* ignore */ }
  }

  function removeInvite() {
    const el = document.getElementById(INVITE_ID);
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  function ensureInviteCss() {
    if (document.getElementById("br-da-invite-css")) return;
    const style = document.createElement("style");
    style.id = "br-da-invite-css";
    style.textContent =
      "@keyframes br-da-invite-in{from{opacity:0;transform:translateY(12px) scale(.98);}to{opacity:1;transform:none;}}" +
      "#" + INVITE_ID + "{position:fixed;inset:0;z-index:2147483646;display:flex;align-items:center;justify-content:center;" +
      "padding:20px;background:rgba(6,12,22,.62);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);}" +
      "#" + INVITE_ID + " .br-da-invite-card{position:relative;width:min(400px,100%);padding:24px 24px 18px;border-radius:20px;" +
      "background:linear-gradient(165deg,#183a5e 0%,#102842 55%,#0d2038 100%);color:#fff;overflow:hidden;" +
      "font:600 13px/1.45 system-ui,-apple-system,sans-serif;" +
      "box-shadow:0 24px 60px -12px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.06);" +
      "border:1px solid rgba(255,255,255,.1);animation:br-da-invite-in .28s cubic-bezier(.22,1,.36,1) both;}" +
      "#" + INVITE_ID + " .br-da-invite-card:before{content:'';position:absolute;inset:0;border-radius:inherit;pointer-events:none;" +
      "background:radial-gradient(120% 80% at 85% -10%,rgba(125,211,252,.18),transparent 60%);}" +
      "#" + INVITE_ID + " .br-da-invite-card>*{position:relative;}" +
      "#" + INVITE_ID + " .br-da-invite-brand{display:flex;align-items:center;gap:11px;margin:0 0 16px;}" +
      "#" + INVITE_ID + " .br-da-invite-logo{width:44px;height:44px;object-fit:contain;flex:0 0 auto;}" +
      "#" + INVITE_ID + " .br-da-invite-brand-txt{min-width:0;flex:1;}" +
      "#" + INVITE_ID + " .br-da-invite-kicker{margin:0;font:800 11px/1.2 inherit;letter-spacing:.09em;text-transform:uppercase;color:rgba(255,255,255,.66);}" +
      "#" + INVITE_ID + " .br-da-invite-ver{flex:0 0 auto;margin:0;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.18);" +
      "background:rgba(255,255,255,.05);font:800 11px/1 inherit;letter-spacing:.02em;color:rgba(255,255,255,.9);}" +
      "#" + INVITE_ID + " .br-da-invite-title{font:800 19px/1.3 inherit;margin:0 0 8px;letter-spacing:-.01em;}" +
      "#" + INVITE_ID + " .br-da-invite-copy{margin:0 0 16px;color:rgba(255,255,255,.82);font-weight:500;}" +
      "#" + INVITE_ID + " .br-da-invite-perks{margin:0 0 20px;padding:0;list-style:none;}" +
      "#" + INVITE_ID + " .br-da-invite-perks li{position:relative;margin:0 0 9px;padding:0 0 0 27px;color:rgba(255,255,255,.9);font-weight:600;}" +
      "#" + INVITE_ID + " .br-da-invite-perks li:last-child{margin-bottom:0;}" +
      "#" + INVITE_ID + " .br-da-invite-perks li:before{content:'';position:absolute;left:0;top:1px;width:17px;height:17px;border-radius:50%;" +
      "background:rgba(125,211,252,.16);border:1px solid rgba(125,211,252,.4);}" +
      "#" + INVITE_ID + " .br-da-invite-perks li:after{content:'';position:absolute;left:6px;top:5px;width:4px;height:7px;" +
      "border:solid #7dd3fc;border-width:0 2px 2px 0;transform:rotate(45deg);}" +
      "#" + INVITE_ID + " .br-da-invite-row{display:flex;gap:10px;}" +
      "#" + INVITE_ID + " button{flex:1;margin:0;padding:12px 12px;border-radius:12px;font:700 13px/1.2 inherit;cursor:pointer;" +
      "transition:transform .12s ease,background .15s ease,box-shadow .15s ease,border-color .15s ease;}" +
      "#" + INVITE_ID + " button:focus-visible{outline:2px solid #7dd3fc;outline-offset:2px;}" +
      "#" + INVITE_ID + " button:active{transform:translateY(1px);}" +
      "#" + INVITE_ID + " .br-da-invite-skip{border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.02);color:rgba(255,255,255,.9);}" +
      "#" + INVITE_ID + " .br-da-invite-skip:hover{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.32);}" +
      "#" + INVITE_ID + " .br-da-invite-open{flex:1.4;border:0;background:#fff;color:#0d2038;box-shadow:0 6px 16px -4px rgba(0,0,0,.45);}" +
      "#" + INVITE_ID + " .br-da-invite-open:hover{background:#eaf4fd;box-shadow:0 10px 24px -6px rgba(125,211,252,.5);transform:translateY(-1px);}" +
      "@media (prefers-reduced-motion:reduce){#" + INVITE_ID + " .br-da-invite-card{animation:none;}" +
      "#" + INVITE_ID + " button:hover,#" + INVITE_ID + " button:active{transform:none;}}";
    (document.head || document.documentElement).appendChild(style);
  }

  function openAssistant() {
    writeLaunchChoice("open");
    removeInvite();
    requestPool();
    mount();
  }

  function skipAssistant() {
    writeLaunchChoice("skip");
    removeInvite();
  }

  function showInvite() {
    if (document.getElementById(ROOT_ID) || document.getElementById(INVITE_ID)) return;
    ensureInviteCss();
    const wrap = document.createElement("div");
    wrap.id = INVITE_ID;
    wrap.setAttribute("role", "dialog");
    wrap.setAttribute("aria-modal", "true");
    wrap.setAttribute("aria-label", "Use the BR Fantasy Draft Assistant");
    const host = hostLabel();
    const logoUrl = extensionAsset("icons/br-logo-dark.png");
    wrap.innerHTML =
      '<div class="br-da-invite-card">' +
      '<div class="br-da-invite-brand">' +
      (logoUrl ? '<img class="br-da-invite-logo" alt="BR Fantasy" src="' + logoUrl + '">' : "") +
      '<div class="br-da-invite-brand-txt">' +
      '<p class="br-da-invite-kicker">BR Fantasy extension</p>' +
      "</div>" +
      '<p class="br-da-invite-ver">' + PRODUCT_VERSION + "</p>" +
      "</div>" +
      '<p class="br-da-invite-title">Use Draft Assistant on this ' + host + " draft</p>" +
      '<p class="br-da-invite-copy">The extension docks beside the board, follows live ' +
      host + " picks, and ranks who is left. It never submits a pick.</p>" +
      '<ul class="br-da-invite-perks">' +
      "<li>Live picks from this room</li>" +
      "<li>Ranked recommendations for the pick on the clock</li>" +
      "<li>Read-only. You still pick in " + host + "</li>" +
      "</ul>" +
      '<div class="br-da-invite-row">' +
      '<button type="button" class="br-da-invite-skip">Not now</button>' +
      '<button type="button" class="br-da-invite-open">Open Draft Assistant</button>' +
      "</div></div>";
    wrap.addEventListener("click", function (ev) {
      if (ev.target === wrap) skipAssistant();
    });
    wrap.querySelector(".br-da-invite-open").addEventListener("click", function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      openAssistant();
    });
    wrap.querySelector(".br-da-invite-skip").addEventListener("click", function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      skipAssistant();
    });
    (document.body || document.documentElement).appendChild(wrap);
  }

  function tryMount() {
    if (!isHostDraftRoom()) {
      removeInvite();
      return false;
    }
    if (document.getElementById(ROOT_ID)) return true;
    if (readLaunchChoice() === "skip") return true;
    if (readLaunchChoice() === "open") {
      ready = false;
      iframe = null;
      mount();
      requestPool();
      return true;
    }
    showInvite();
    const invite = document.getElementById(INVITE_ID);
    if (invite && document.body && invite.parentNode !== document.body) {
      document.body.appendChild(invite);
    }
    return true;
  }

  function hookHistory(cb) {
    try {
      const wrap = function (fn) {
        return function () {
          const ret = fn.apply(this, arguments);
          try { cb(); } catch (_err) { /* ignore */ }
          return ret;
        };
      };
      history.pushState = wrap(history.pushState.bind(history));
      history.replaceState = wrap(history.replaceState.bind(history));
    } catch (_e) { /* ignore */ }
    window.addEventListener("popstate", cb);
  }

  function watchHost() {
    let lastHref = location.href;
    const check = function () {
      if (location.href !== lastHref) lastHref = location.href;
      tryMount();
    };
    hookHistory(check);
    window.addEventListener("hashchange", check);
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", check);
    }
    setInterval(check, 800);
    try {
      if (!window.__brDaInviteMo && document.documentElement) {
        window.__brDaInviteMo = new MutationObserver(function () { tryMount(); });
        window.__brDaInviteMo.observe(document.documentElement, { childList: true, subtree: false });
        if (document.body) {
          window.__brDaInviteMo.observe(document.body, { childList: true, subtree: false });
        }
      }
    } catch (_e) { /* ignore */ }
  }

  try {
    chrome.runtime.onMessage.addListener(function (msg) {
      if (!msg || msg.type !== "openDraftAssistant") return;
      openAssistant();
    });
  } catch (_e) { /* ignore */ }

  tryMount();
  watchHost();
})();
