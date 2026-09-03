// MAIN-world helper on Sleeper. Reads the logged-in user from page JS /
// storage the isolated world cannot see, and posts it across the bridge.
// Never submits a pick.

(function () {
  "use strict";

  if (window.__brFantasySleeperMain) return;
  try {
    if (window.top !== window) return;
    if (!/(^|\.)sleeper\.(com|app)$/i.test(String(location.hostname || ""))) return;
  } catch (_e) {
    return;
  }
  window.__brFantasySleeperMain = true;

  const BRIDGE = "brfantasy-sleeper-v1";
  const MAX_WALK = 4000;
  let lastFp = "";
  let walked = 0;

  function postIdentity(ident) {
    if (!ident || !ident.userId) return;
    const fp = [ident.userId, ident.username || "", ident.displayName || ""].join("|");
    if (fp === lastFp) return;
    lastFp = fp;
    try {
      window.postMessage({ __br: BRIDGE, type: "identity", detail: ident }, "*");
    } catch (_e) { /* ignore */ }
  }

  function asId(v) {
    const s = String(v == null ? "" : v).trim();
    return /^\d{6,20}$/.test(s) ? s : "";
  }

  function harvest(obj, key, depth) {
    if (!obj || typeof obj !== "object" || depth > 5) return;
    if (walked++ > MAX_WALK) return;
    try {
      if (obj === window || obj === document || obj instanceof Window) return;
    } catch (_e) {
      return;
    }
    if (Array.isArray(obj)) {
      if (obj.length > 2) return;
      obj.slice(0, 4).forEach(function (item) { harvest(item, key, depth + 1); });
      return;
    }
    const uid = asId(obj.user_id || obj.userId || obj.sleeper_user_id);
    const un = obj.username || obj.user_name || obj.userName || "";
    const dn = obj.display_name || obj.displayName || "";
    const team = (obj.metadata && obj.metadata.team_name) || obj.team_name || obj.teamName || "";
    const loggedIn =
      obj.token ||
      obj.access_token ||
      obj.accessToken ||
      obj.email ||
      obj.phone ||
      obj.verification ||
      obj.real_name != null ||
      /user|auth|session|token|login|\bme\b|current/i.test(String(key || ""));
    if (uid && loggedIn) {
      postIdentity({
        userId: uid,
        username: un ? String(un) : "",
        displayName: dn ? String(dn) : "",
        teamName: team ? String(team) : "",
      });
      return;
    }
    ["user", "data", "session", "profile", "me", "account", "viewer", "auth", "currentUser"].forEach(function (k) {
      if (obj[k] && typeof obj[k] === "object") harvest(obj[k], k, depth + 1);
    });
  }

  function scanStorage() {
    try {
      [localStorage, sessionStorage].forEach(function (store) {
        if (!store) return;
        for (let i = 0; i < store.length; i++) {
          const k = store.key(i);
          const v = store.getItem(k) || "";
          if (!v || v.length > 400000) continue;
          const trimmed = v.trim();
          if (trimmed.charAt(0) === "{" || trimmed.charAt(0) === "[") {
            try { harvest(JSON.parse(trimmed), k, 0); } catch (_e) { /* ignore */ }
          }
          const bare = asId(trimmed);
          if (bare && /user_id|userId|sleeper_user/i.test(String(k || ""))) {
            postIdentity({ userId: bare, username: "", displayName: "", teamName: "" });
          }
        }
      });
    } catch (_e) { /* ignore */ }
  }

  function scanGlobals() {
    walked = 0;
    const keys = [
      "__PRELOADED_STATE__",
      "__NEXT_DATA__",
      "store",
      "sleeper",
      "__SLEEPER_USER__",
      "__INITIAL_STATE__",
    ];
    keys.forEach(function (k) {
      try { if (window[k]) harvest(window[k], k, 0); } catch (_e) { /* ignore */ }
    });
    try {
      const names = Object.getOwnPropertyNames(window);
      const n = Math.min(names.length, 80);
      for (let i = 0; i < n; i++) {
        const k = names[i];
        if (!/user|auth|session|sleeper|store|state/i.test(k)) continue;
        try { harvest(window[k], k, 0); } catch (_e) { /* ignore */ }
      }
    } catch (_e) { /* ignore */ }
  }

  function scan() {
    scanStorage();
    scanGlobals();
  }

  scan();
  setInterval(scan, 4000);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) scan();
  });
})();
