# BR Fantasy — League Connector (browser extension)

Connect private ESPN fantasy leagues to BR Fantasy, and **auto-relay live draft
picks** from ESPN or Yahoo into Draft Room.

## Why an extension

1. **Cookies:** ESPN's `espn_s2` cookie is `HttpOnly` — only an extension can
   read it for one-click private-league connect.
2. **Live drafts:** ESPN's `mDraftDetail` REST view often does **not** update
   mid-draft. Yahoo's `draftresults` usually does, but the open draft room UI is
   still faster. This extension reads in-page state and relays picks to BR
   Fantasy. Picks are never submitted to ESPN or Yahoo.

## Desktop live draft (automatic)

1. Install the extension (Load unpacked for now, or the Chrome Web Store build).
2. Open **Draft Room** for your ESPN or Yahoo league → **Connect Live Draft**.
3. Open the host draft in another tab (ESPN `fantasy.espn.com/.../draft` or
   Yahoo `football.fantasysports.yahoo.com/f1/{id}/draft`).
4. A small **BR Fantasy** chip appears on the draft page. Picks flow into Draft
   Room within ~1–2 seconds — keep both tabs open.

Phone drafts: track picks manually in Draft Room, or use a laptop with this
extension for auto-sync.

## Private league connect (ESPN)

1. **`background.js`** reads `SWID` + `espn_s2`.
2. **`content.js`** on BR Fantasy adds **Autofill from ESPN**.
3. Values only leave the browser on the site's normal Connect request.

## Permissions

| Permission | Why |
|---|---|
| `cookies` | Read `SWID` + `espn_s2` (ESPN connect flow only). |
| `*.espn.com` | Cookie reads + ESPN draft-room observers. |
| `*.fantasysports.yahoo.com` / `sports.yahoo.com` | Yahoo draft-room observers. |
| `brfantasyfootball.com` | Autofill + receive live pick relay. |

No broad `tabs` permission. No pick submission to ESPN or Yahoo.

## Install for development (Chrome / Edge)

1. `chrome://extensions` → Developer mode → **Load unpacked** → `extension/`
2. Sign into espn.com / Yahoo; open Draft Room + the host draft for a live test.

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
  yahoo_draft_main.js    Yahoo draft room MAIN-world observer
  yahoo_draft.js         Yahoo isolated bridge + status chip
  popup.html/.js/.css
  pack_extension.py      production zip builder
  icons/
```
