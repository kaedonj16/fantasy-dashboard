# BR Fantasy — ESPN League Connector (browser extension)

One-click connect for **private ESPN** fantasy leagues, plus **live draft pick
relay** into BR Fantasy Draft Room.

## Why an extension

1. **Cookies:** ESPN's `espn_s2` cookie is `HttpOnly`, so page JavaScript — and
   any bookmarklet — **cannot** read it. A browser extension with the `cookies`
   permission is the only client-side way to read it for private-league connect.
2. **Live drafts:** ESPN's documented `mDraftDetail` REST view often does **not**
   update while a draft is in progress (picks appear only after the draft). The
   live draft room UI itself does update. This extension reads that in-page
   state (React + draft API traffic) and relays picks to an open BR Fantasy
   Draft Room tab. Picks are never submitted to ESPN.

## How it works

### Private league connect

1. **`background.js`** reads `SWID` + `espn_s2` from `*.espn.com`.
2. **`content.js`** on BR Fantasy adds **Autofill from ESPN** above the cookie
   paste box. Values only leave the browser on the site's normal Connect request.

### Live draft relay

1. Open your draft on ESPN (`fantasy.espn.com/football/draft?...`) **and** open
   Draft Room for that league on BR Fantasy → **Connect Live Draft**.
2. **`espn_draft_main.js`** (MAIN world) observes ESPN's draft-room React state
   and fantasy API responses.
3. **`espn_draft.js`** forwards a compact pick list to the service worker.
4. **`background.js`** relays to BR Fantasy tabs; **`content.js`** dispatches
   `brfantasy:espn-draft-relay`.
5. Draft Room posts the picks to `/api/draft/espn-relay` for ESPN→canonical
   player mapping and applies them on the board.

Keep both tabs open during the draft. If REST sync stalls, the extension path
keeps the board live without manual pick entry.

## Permissions

| Permission | Why |
|---|---|
| `cookies` | Read `SWID` + `espn_s2` from ESPN (connect flow only). |
| `host_permissions: *.espn.com` | Cookie reads + draft-room content scripts. |
| `host_permissions: brfantasyfootball.com` | Autofill + receive live pick relay. |
| `localhost` / `127.0.0.1` | Local development only — drop before publishing if undesired. |

No broad `tabs` permission. The extension does not submit picks to ESPN and does
not send cookies to any third party.

## Install for development (Chrome / Edge)

1. Regenerate icons if needed: `python3 extension/icons/make_icons.py`
2. Visit `chrome://extensions`, enable **Developer mode**.
3. **Load unpacked** → select this `extension/` folder.
4. Sign into `espn.com`. For live drafts, open the ESPN draft room in one tab
   and BR Fantasy Draft Room in another.

## Firefox

The manifest is MV3 with `browser_specific_settings.gecko` and works on Firefox
121+ (`about:debugging` → **This Firefox** → **Load Temporary Add-on** → pick
`manifest.json`).

## Publishing (later)

- **Chrome Web Store** / **Firefox AMO**: bump `version` in `manifest.json`.
- Privacy justification: cookies stay local until the user clicks Connect on BR
  Fantasy; live relay sends only pick numbers / ESPN player ids / team ids for
  the league the user already has open.

## Files

```
extension/
  manifest.json         MV3 manifest
  background.js         cookies + pick relay
  content.js            BR Fantasy: autofill + receive relay
  content.css           button + status styling
  espn_draft_main.js    ESPN draft room (MAIN world observer)
  espn_draft.js         ESPN draft room (isolated → background)
  popup.html/.js/.css   toolbar popup (detect + copy fallback)
  icons/                BR Fantasy logo icons
```
