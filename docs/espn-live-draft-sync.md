# ESPN live-draft companion (Draft Room)

Observe an ESPN snake/linear draft from this site's Draft Room. ESPN still
makes the picks. This app maps players onto the existing board and recalculates
Pick Scores / recommendations. **Picks are never submitted to ESPN.**

## Sync paths

| Path | How picks arrive | Devices |
|------|------------------|---------|
| **REST poll** | Server polls ESPN `mDraftDetail` | All (often stale mid-draft) |
| **Extension relay** | Extension reads open ESPN draft room → Draft Room + server store | Desktop Chrome/Edge/Firefox |
| **Mobile bookmarklet / Shortcut** | Token-authenticated POST from ESPN page → server store → Draft Room poll | Phone browsers with **Request Desktop Website** |

Use them together. Connect Live Draft, then:

- **Desktop:** install the Chrome extension (Draft Room → **Get Chrome
  extension**, or load unpacked `extension/`), keep the ESPN draft tab open.
- **Mobile:** ESPN's default mobile page shows “download the ESPN Fantasy App.”
  Use **Request Desktop Website** (Safari) / **Desktop site** (Chrome) so the
  live draft board loads, then **Mobile Sync** (bookmarklet / Shortcut). The
  native ESPN app cannot run bookmarks — use manual tracking there instead.

Backend: `dashboard_services/draft_sync.py`, `espn_draft.py`,
`espn_draft_relay.py` (tokens + snapshot store). APIs:
`GET|POST /api/draft/espn-relay`, `POST /api/draft/espn-relay/token`,
`GET /api/draft/live` (merges stored relay when fresher than REST).

## Verify

### Desktop extension

1. Load unpacked `extension/` → Connect Live Draft → open ESPN draft.
2. Make a pick on ESPN → BR board updates; ESPN page shows a BR Fantasy chip.

### Mobile bookmarklet

1. Connect Live Draft → **Mobile Sync** → copy bookmarklet / Shortcut JS.
2. Open ESPN draft → **Request Desktop Website** (confirm the draft board loads).
3. **Android:** bookmark URL = bookmarklet. **iPhone:** Shortcuts Share Sheet
   receives Safari web pages → Run JavaScript on Web Page → **Shortcut Input**
   → paste Shortcut JS → Share from the ESPN tab to run it.
4. Return to Draft Room; picks appear within one poll cycle (~5–10s).

## Diagnostics

`ESPN_DRAFT_SYNC_DEBUG=1` logs REST snapshots. Relay tokens use
`ESPN_RELAY_SECRET` or `FLASK_SECRET_KEY`. Snapshots live in process memory and
optional Redis (`REDIS_URL`), TTL ~12h.

## Known ESPN API limits

`mDraftDetail` often does not grow mid-draft. Treat frozen REST as unavailable
unless extension / bookmarklet relay is feeding picks.

ESPN mobile web defaults to an app-download interstitial; Request Desktop
Website restores the draft UI. There is no supported way to inject JS into the
ESPN Fantasy native app.
