# BR Fantasy — ESPN League Connector (browser extension)

Connect private ESPN fantasy leagues to BR Fantasy, and **auto-relay live draft
picks** into Draft Room while you draft on ESPN.

## Why an extension

1. **Cookies:** ESPN's `espn_s2` cookie is `HttpOnly` — only an extension can
   read it for one-click private-league connect.
2. **Live drafts:** ESPN's `mDraftDetail` REST view often does **not** update
   mid-draft. The live draft room UI does. This extension reads that in-page
   state and relays picks to BR Fantasy. Picks are never submitted to ESPN.

## Desktop live draft (automatic)

1. Install the extension (Load unpacked for now, or the Chrome Web Store build).
2. Open **Draft Room** for your ESPN league → **Connect Live Draft**.
3. Open the ESPN draft in another tab (`fantasy.espn.com/football/draft?...`).
4. A small **BR Fantasy** chip appears on the ESPN page. Picks flow into Draft
   Room within ~1–2 seconds — keep both tabs open.

## Mobile live draft (bookmarklet / Shortcut)

Extensions don't run on iOS/Android browsers. ESPN's default mobile page pushes
the Fantasy app; use **Request Desktop Website** / **Desktop site** so the live
draft board loads in Safari/Chrome, then:

### Android
1. Draft Room → **Mobile Sync** → copy bookmarklet.
2. Bookmark the desktop-mode ESPN draft → edit URL to the bookmarklet.
3. Run the bookmark after picks.

### iPhone (Shortcuts — Safari only)
Chrome cannot run this action (it shares a URL, not a Safari page).

1. Draft Room → **Mobile Sync** → **Copy iOS Shortcut JS**.
2. Shortcuts → new shortcut → **Show in Share Sheet**.
3. Receive **only** Safari web pages (not “Apps and 18 more”).
4. **Run JavaScript on Web Page** → **Shortcut Input** → paste Shortcut JS
   (not the bookmarklet; must call `completion(…)`).
5. Open ESPN draft in **Safari** → Request Desktop Website → Share → run the
   shortcut. Do not press Play inside the Shortcuts app.

The ESPN Fantasy **app** cannot run bookmarks — use manual tracking there, or
draft with a laptop and the Chrome extension.

## Private league connect

1. **`background.js`** reads `SWID` + `espn_s2`.
2. **`content.js`** on BR Fantasy adds **Autofill from ESPN**.
3. Values only leave the browser on the site's normal Connect request.

## Permissions

| Permission | Why |
|---|---|
| `cookies` | Read `SWID` + `espn_s2` (connect flow only). |
| `*.espn.com` | Cookie reads + draft-room observers. |
| `brfantasyfootball.com` | Autofill + receive live pick relay. |

No broad `tabs` permission. No pick submission to ESPN.

## Install for development (Chrome / Edge)

1. `chrome://extensions` → Developer mode → **Load unpacked** → `extension/`
2. Sign into espn.com; open Draft Room + ESPN draft for a live test.

## Production zip (Chrome Web Store / AMO)

```bash
python3 extension/pack_extension.py
# → artifacts/br-fantasy-espn-connector-vX.Y.Z.zip
```

That build strips `localhost` permissions. Upload the zip in the Chrome Web
Store developer dashboard (one-time $5) with screenshots + a privacy policy
URL. Justify `cookies` as local-only until the user clicks Connect.

## Firefox

MV3 + `browser_specific_settings.gecko` — load via `about:debugging` or submit
the same production zip to addons.mozilla.org.

## Files

```
extension/
  manifest.json
  background.js          cookies + pick relay
  content.js             BR Fantasy autofill + receive relay
  content.css
  espn_draft_main.js     ESPN draft room MAIN-world observer
  espn_draft.js          ESPN isolated bridge + status chip
  popup.html/.js/.css
  pack_extension.py      production zip builder
  icons/
```
