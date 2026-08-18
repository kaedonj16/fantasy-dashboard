// Runs on brfantasyfootball.com. When the private-ESPN connect UI is present,
// it adds an "Autofill from ESPN" button just above the paste box. Clicking it
// asks the service worker for the ESPN cookies and drops them into the box,
// then fires the same `input` event a human paste would — so the site's own
// parser fills SWID + espn_s2 and validates. The extension writes nothing else
// and sends the cookies nowhere; they ride the normal Connect request.

(function () {
  "use strict";

  // The two paste boxes shipped by the site (main connect modal + link modal).
  const BLOB_IDS = ["espnCookieBlob", "linkEspnBlob"];

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
      // Let the site's own listener parse + validate exactly as if pasted.
      blob.dispatchEvent(new Event("input", { bubbles: true }));
      if (creds.swid && creds.espn_s2) {
        setStatus(blob, "✓ Filled from your ESPN session — click Connect below.", "ok");
      } else {
        setStatus(blob, "Only found " + (creds.swid ? "SWID" : "espn_s2") +
          ". Open a league on espn.com so both cookies are set, then try again.", "err");
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

  scan();
  // The connect modals mount/unmount dynamically, so keep watching.
  const mo = new MutationObserver(scan);
  mo.observe(document.documentElement, { childList: true, subtree: true });
})();
