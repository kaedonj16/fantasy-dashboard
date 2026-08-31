// Runs on brfantasyfootball.com.
// 1) Autofill ESPN cookies into the private-league connect box.
// 2) Receive live-draft pick relays from the service worker (sourced from an
//    open ESPN or Yahoo draft room tab) and hand them to Draft Room via CustomEvent.
// 3) Bridge page-initiated reconnect requests to the service worker.

(function () {
  "use strict";

  const BLOB_IDS = ["espnCookieBlob", "linkEspnBlob"];
  const ESPN_RELAY_EVENT = "brfantasy:espn-draft-relay";
  const YAHOO_RELAY_EVENT = "brfantasy:yahoo-draft-relay";
  const RECONNECT_REQ = "brfantasy:request-extension-reconnect";
  const RECONNECT_EVT = "brfantasy:extension-reconnect";
  const RECONNECT_RESULT = "brfantasy:extension-reconnect-result";

  function icon() {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "15");
    svg.setAttribute("height", "15");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", "M13 2 3 14h7l-1 8 10-12h-7z");
    svg.appendChild(p);
    return svg;
  }

  function statusEl(blob) {
    let el = blob.parentNode.querySelector(".br-ext-status");
    if (!el) {
      el = document.createElement("p");
      el.className = "br-ext-status";
      blob.insertAdjacentElement("afterend", el);
    }
    return el;
  }

  function setStatus(blob, text, kind) {
    const el = statusEl(blob);
    el.textContent = text;
    el.dataset.kind = kind || "";
  }

  function fill(blob, btn) {
    btn.disabled = true;
    const label = btn.querySelector(".br-ext-label");
    const prev = label.textContent;
    label.textContent = "Reading ESPN session…";
    chrome.runtime.sendMessage({ type: "getEspnCookies" }, (creds) => {
      btn.disabled = false;
      label.textContent = prev;
      if (chrome.runtime.lastError || !creds || (!creds.swid && !creds.espn_s2)) {
        setStatus(blob, "No ESPN session found. Sign in at espn.com in this browser, then try again.", "err");
        return;
      }
      const parts = [];
      if (creds.swid) parts.push("SWID=" + creds.swid);
      if (creds.espn_s2) parts.push("espn_s2=" + creds.espn_s2);
      blob.value = parts.join("; ");
      blob.dispatchEvent(new Event("input", { bubbles: true }));
      if (creds.swid && creds.espn_s2) {
        setStatus(blob, "✓ Filled from your ESPN session - click Connect below.", "ok");
      } else {
        setStatus(
          blob,
          "Only found " +
            (creds.swid ? "SWID" : "espn_s2") +
            ". Open a league on espn.com so both cookies are set, then try again.",
          "err"
        );
      }
    });
  }

  function inject(blob) {
    if (blob.dataset.brExtWired) return;
    blob.dataset.brExtWired = "1";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "br-ext-fill-btn";
    btn.appendChild(icon());
    const label = document.createElement("span");
    label.className = "br-ext-label";
    label.textContent = "Autofill from ESPN (extension)";
    btn.appendChild(label);
    btn.addEventListener("click", () => fill(blob, btn));
    blob.insertAdjacentElement("beforebegin", btn);
  }

  function scan() {
    BLOB_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (el) inject(el);
    });
  }

  function dispatchToPage(eventName, detail) {
    try {
      document.dispatchEvent(
        new CustomEvent(eventName, { detail: detail || {}, bubbles: true, composed: true })
      );
    } catch (_e) {
      /* ignore */
    }
  }

  function dispatchRelay(eventName, payload) {
    if (!payload || !Array.isArray(payload.picks)) return;
    dispatchToPage(eventName, payload);
  }

  function dispatchReconnect(detail) {
    dispatchToPage(RECONNECT_EVT, detail || {});
  }

  function requestReconnect(detail) {
    try {
      chrome.runtime.sendMessage(
        {
          type: "reconnectDraftRelay",
          leagueId: detail && detail.leagueId,
          season: detail && detail.season,
          platform: detail && detail.platform,
          source: (detail && detail.source) || "draft-room",
        },
        (resp) => {
          void chrome.runtime.lastError;
          try {
            dispatchToPage(RECONNECT_RESULT, resp || {});
          } catch (_e) {
            /* ignore */
          }
        }
      );
    } catch (_e) {
      /* ignore */
    }
  }

  window.addEventListener(RECONNECT_REQ, (ev) => {
    requestReconnect((ev && ev.detail) || {});
  });
  document.addEventListener(RECONNECT_REQ, (ev) => {
    requestReconnect((ev && ev.detail) || {});
  });

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg && msg.type === "espnDraftRelay" && msg.payload) {
      dispatchRelay(ESPN_RELAY_EVENT, msg.payload);
      sendResponse({ ok: true });
      return false;
    }
    if (msg && msg.type === "yahooDraftRelay" && msg.payload) {
      dispatchRelay(YAHOO_RELAY_EVENT, msg.payload);
      sendResponse({ ok: true });
      return false;
    }
    if (msg && msg.type === "brDraftRoomReconnect") {
      dispatchReconnect(msg.detail || {});
      sendResponse({ ok: true });
      return false;
    }
    return false;
  });

  function draftRoomMeta() {
    const m = location.pathname.match(/^\/(espn|yahoo|sleeper)\/(\d{4})\/([^/]+)\/draft\b/i);
    return {
      href: location.href,
      platform: m ? m[1].toLowerCase() : "",
      season: m ? m[2] : "",
      leagueId: m ? m[3] : "",
    };
  }

  function announceDraftRoom() {
    try {
      chrome.runtime.sendMessage(
        { type: "brDraftRoomReady", ...draftRoomMeta() },
        () => {
          void chrome.runtime.lastError;
        }
      );
    } catch (_e) {
      /* ignore */
    }
  }

  scan();
  const mo = new MutationObserver(scan);
  mo.observe(document.documentElement, { childList: true, subtree: true });

  announceDraftRoom();
  setInterval(announceDraftRoom, 12000);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) announceDraftRoom();
  });
  window.addEventListener("pageshow", announceDraftRoom);
})();
