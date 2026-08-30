// The popup runs in the extension context, so it can read the ESPN cookies
// directly. It reports whether a session was found and lets the user copy the
// two values for pasting — a fallback for when they aren't on a BR Fantasy tab
// (where the injected "Autofill from ESPN" button is the smoother path).

const ESPN_URLS = [
  "https://www.espn.com",
  "https://fantasy.espn.com",
  "https://espn.com",
];

async function readCookie(name) {
  for (const url of ESPN_URLS) {
    try {
      const cookie = await chrome.cookies.get({ url, name });
      if (cookie && cookie.value) return cookie.value;
    } catch (_e) { /* try next host */ }
  }
  return "";
}

const card = document.getElementById("card");
const titleEl = document.getElementById("title");
const detailEl = document.getElementById("detail");
const copyBtn = document.getElementById("copyBtn");
const reconnectBtn = document.getElementById("reconnectBtn");
const reconnectStatus = document.getElementById("reconnectStatus");

let blob = "";

function render(swid, espn_s2) {
  card.classList.remove("state-loading", "state-ok", "state-warn", "state-err");
  if (swid && espn_s2) {
    card.classList.add("state-ok");
    titleEl.textContent = "ESPN session detected";
    detailEl.textContent =
      "Both SWID and espn_s2 are ready. Live drafts: open ESPN or Yahoo draft + BR Draft Room — picks relay automatically.";
    blob = "SWID=" + swid + "; espn_s2=" + espn_s2;
    copyBtn.disabled = false;
  } else if (swid || espn_s2) {
    card.classList.add("state-warn");
    titleEl.textContent = "Almost there";
    detailEl.textContent = "Found " + (swid ? "SWID" : "espn_s2") +
      " only. Open a league on espn.com so both cookies are set.";
    copyBtn.disabled = true;
  } else {
    card.classList.add("state-err");
    titleEl.textContent = "Not signed into ESPN";
    detailEl.textContent = "Sign in at espn.com in this browser, then reopen this.";
    copyBtn.disabled = true;
  }
}

function reconnectMessage(resp) {
  if (!resp) return "Reconnect failed — reload the extension.";
  if ((resp.br && resp.br.pinged > 0) || (resp.draft && resp.draft.pinged > 0)) {
    const parts = [];
    if (resp.br && resp.br.pinged > 0) parts.push(resp.br.pinged + " Draft Room tab(s)");
    if (resp.draft && resp.draft.pinged > 0) parts.push(resp.draft.pinged + " draft tab(s)");
    return "Reconnect sent to " + parts.join(" and ") + ".";
  }
  return resp.message || "No open Draft Room or draft tabs found.";
}

copyBtn.addEventListener("click", async () => {
  if (!blob) return;
  try {
    await navigator.clipboard.writeText(blob);
    copyBtn.textContent = "Copied ✓";
    setTimeout(() => { copyBtn.textContent = "Copy SWID & espn_s2"; }, 1600);
  } catch (_e) {
    copyBtn.textContent = "Copy failed — select manually";
  }
});

reconnectBtn.addEventListener("click", () => {
  reconnectBtn.disabled = true;
  reconnectStatus.textContent = "Reconnecting…";
  chrome.runtime.sendMessage({ type: "reconnectDraftRelay", source: "popup" }, (resp) => {
    reconnectBtn.disabled = false;
    void chrome.runtime.lastError;
    reconnectStatus.textContent = reconnectMessage(resp);
  });
});

(async function init() {
  const [swid, espn_s2] = await Promise.all([readCookie("SWID"), readCookie("espn_s2")]);
  render(swid, espn_s2);
})();
