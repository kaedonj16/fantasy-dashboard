# BR Fantasy — ESPN League Connector (browser extension)

One-click connect for **private ESPN** fantasy leagues. It removes the last
manual step of the streamlined paste flow: instead of copying `SWID` and
`espn_s2` out of dev tools, the extension reads them for you and drops them
into the BR Fantasy connect box.

## Why an extension (and not a bookmarklet)

ESPN's `espn_s2` cookie is `HttpOnly`, so page JavaScript — and therefore any
bookmarklet — **cannot** read it. A browser extension with the `cookies`
permission is the only client-side way to read it. That's the whole reason this
exists.

## How it works

1. **`background.js`** (service worker) is the only place with the `cookies`
   permission. On request it reads `SWID` + `espn_s2` from `*.espn.com`.
2. **`content.js`** runs on `brfantasyfootball.com`. When the private-league
   connect box (`#espnCookieBlob` / `#linkEspnBlob`) is on the page, it adds an
   **Autofill from ESPN** button above it. Clicking asks the service worker for
   the cookies, writes `SWID=…; espn_s2=…` into the box, and fires the same
   `input` event a paste would — so the site's own parser fills and validates
   the two values. The user clicks **Connect**.
3. **`popup.js`** is a fallback for when you're not on a BR Fantasy tab: it
   reports whether an ESPN session is detected and copies the two values to your
   clipboard to paste in.

The extension **sends the cookies nowhere itself**. They only leave the browser
in the normal Connect request the user triggers on BR Fantasy, over HTTPS, where
they're validated and stored encrypted — exactly like a manual paste.

> Dependency: the content script targets the paste-box element IDs shipped by
> the app (`espnCookieBlob`, `linkEspnBlob`). Keep those IDs stable, or update
> `BLOB_IDS` in `content.js`.

## Permissions

| Permission | Why |
|---|---|
| `cookies` | Read `SWID` + `espn_s2` from ESPN. |
| `host_permissions: *.espn.com` | Scope the cookie reads to ESPN only. |
| `host_permissions: brfantasyfootball.com` | Run the autofill content script on the connect page. |
| `localhost` / `127.0.0.1` | Local development only — drop before publishing if undesired. |

No `tabs`, no broad host access, no remote code.

## Install for development (Chrome / Edge)

1. Regenerate icons if needed: `python3 extension/icons/make_icons.py`
2. Visit `chrome://extensions`, enable **Developer mode**.
3. **Load unpacked** → select this `extension/` folder.
4. Sign into `espn.com`, open a private league on BR Fantasy → **Private League**,
   and click **Autofill from ESPN**.

## Firefox

The manifest is MV3 with `browser_specific_settings.gecko` and works on Firefox
121+ (`about:debugging` → **This Firefox** → **Load Temporary Add-on** → pick
`manifest.json`). If you target older Firefox, swap the `background.service_worker`
key for `background.scripts: ["background.js"]`.

## Publishing (later)

- **Chrome Web Store**: zip the folder contents, submit via the Developer
  Dashboard (one-time \$5 fee). Provide the store icon (`icons/icon128.png`),
  screenshots, and a privacy justification for the `cookies` permission — the
  "sends nowhere itself" note above is the honest summary.
- **Firefox AMO**: submit the same zip at addons.mozilla.org.
- Bump `version` in `manifest.json` for each release.

## Files

```
extension/
  manifest.json       MV3 manifest
  background.js       reads ESPN cookies (cookies permission)
  content.js          injects "Autofill from ESPN" on BR Fantasy
  content.css         button + status styling
  popup.html/.js/.css toolbar popup (detect + copy fallback)
  icons/              football icons + pure-Python generator
```
