# BR Fantasy — League Connector (browser extension)

Connect private ESPN fantasy leagues to BR Fantasy, **auto-relay live draft
picks** from ESPN or Yahoo into Draft Room, and **dock a read-only Draft
Assistant overlay** on Sleeper, Yahoo, and ESPN draft rooms.

## Extension parity checklist (R15)

| Item | Status |
|------|--------|
| ESPN private-league connect (cookies → autofill) | Shipped |
| ESPN live draft relay | Shipped |
| Yahoo live draft relay | Shipped |
| Sleeper / Yahoo / ESPN in-page Draft Assistant overlay | Shipped |
| Phone drafts | Manual pick entry in Draft Room (no mobile auto-sync) |
| Production zip | `python3 extension/pack_extension.py` → `artifacts/br-fantasy-espn-connector-vX.Y.Z.zip` |
| Chrome Web Store / AMO | Upload the production zip (see below) |

## Why an extension

1. **Cookies:** ESPN's `espn_s2` cookie is `HttpOnly` — only an extension can
   read it for one-click private-league connect.
2. **Live drafts:** ESPN's `mDraftDetail` REST view often does **not** update
   mid-draft. Yahoo's `draftresults` usually does, but the open draft room UI is
   still faster. This extension reads in-page state and relays picks to BR
   Fantasy. Picks are never submitted to ESPN or Yahoo.
3. **Draft Assistant overlay:** the same read-only pick stream drives a docked
   sidebar (Best Available, roster, grades) inside the host draft tab. Rankings
   use the site `/api/league-players` pool (consensus ADP, BR values,
   projections) and real ESPN / Sleeper headshots — not the standalone mock board.

## Desktop live draft

1. Install the extension (Load unpacked for now, or the Chrome Web Store build).
2. Open the host draft (Sleeper, ESPN, or Yahoo). A **BR Draft Assistant**
   sidebar docks on the right and follows picks. It never submits a pick —
   draft in the host room.
3. For ESPN or Yahoo, also open **Draft Room** → **Connect Live Draft** if you
   want picks mirrored into the web app. Use **Reconnect** in the overlay (or
   the popup) if sync stalls.

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
| `*.espn.com` | Cookie reads + ESPN draft-room observers + overlay. |
| `*.fantasysports.yahoo.com` / `sports.yahoo.com` | Yahoo draft-room observers + overlay. |
| `sleeper.app` / `sleeper.com` / `api.sleeper.app` | Sleeper overlay + public draft API (read-only). |
| `brfantasyfootball.com` | Autofill + receive live pick relay. |

No pick submission to Sleeper, ESPN, or Yahoo.

## Install for development (Chrome / Edge)

1. `chrome://extensions` → Developer mode → **Load unpacked** → `extension/`
2. Sign into espn.com / Yahoo / Sleeper; open a draft tab to see the overlay.

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
  assistant_inject.js    docks overlay iframe on host draft pages
  overlay.html/.css/.js  Draft Assistant UI (extension page, MV3-safe)
  sleeper_draft.js       Sleeper public draft API → overlay
  espn_draft_main.js     ESPN draft room MAIN-world observer
  espn_draft.js          ESPN isolated bridge + overlay feed
  yahoo_draft_main.js    Yahoo draft room MAIN-world observer
  yahoo_draft.js         Yahoo isolated bridge + overlay feed
  popup.html/.js/.css
  pack_extension.py      production zip builder
  icons/
```
