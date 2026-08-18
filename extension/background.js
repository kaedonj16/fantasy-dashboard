// Service worker: the only place with the `cookies` permission. It reads the
// two ESPN session cookies (SWID + espn_s2, both HttpOnly, so a page script
// can't) and hands them to the content script on request. Nothing here talks
// to any network — the values only ever go to the BR Fantasy tab the user is on.

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
    } catch (_e) {
      // keep trying the next host
    }
  }
  return "";
}

async function getEspnCreds() {
  const [swid, espn_s2] = await Promise.all([
    readCookie("SWID"),
    readCookie("espn_s2"),
  ]);
  return { swid, espn_s2 };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "getEspnCookies") {
    getEspnCreds().then(sendResponse).catch(() => sendResponse({ swid: "", espn_s2: "" }));
    return true; // response is async
  }
  return false;
});
