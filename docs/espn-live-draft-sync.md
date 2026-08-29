# ESPN live-draft companion (Draft Room)

Observe an ESPN snake/linear draft from this site's Draft Room. ESPN still
makes the picks. This app maps players onto the existing board and recalculates
Pick Scores / recommendations. **Picks are never submitted to ESPN.**

## Sync paths

| Path | How picks arrive | Devices |
|------|------------------|---------|
| **REST poll** | Server polls ESPN `mDraftDetail` | All (often stale mid-draft) |
| **Extension relay** | Extension reads open ESPN draft room → Draft Room + server store | Desktop Chrome/Edge/Firefox |
| **Mobile bookmarklet / Shortcut** | Token-authenticated POST from ESPN page → server store → Draft Room poll | iOS / Android |

Use them together. Connect Live Draft, then:

- **Desktop:** install `extension/`, keep the ESPN draft tab open.
- **Mobile:** tap **Mobile Sync** in Draft Room, install the bookmarklet /
  Shortcut, run it on the ESPN draft page when picks move.

Backend: `dashboard_services/draft_sync.py`, `espn_draft.py`,
`espn_draft_relay.py` (tokens + snapshot store). APIs:
`GET|POST /api/draft/espn-relay`, `POST /api/draft/espn-relay/token`,
`GET /api/draft/live` (merges stored relay when fresher than REST).

## Verify

### Desktop extension

1. Load unpacked `extension/` → Connect Live Draft → open ESPN draft.
2. Make a pick on ESPN → BR board updates; ESPN page shows a BR Fantasy chip.

### Mobile bookmarklet

1. Connect Live Draft (phone or desktop) → **Mobile Sync** → copy bookmarklet.
2. On phone, open ESPN draft → run bookmark → return to Draft Room; picks appear
   within one poll cycle (~5–10s).

## Diagnostics

`ESPN_DRAFT_SYNC_DEBUG=1` logs REST snapshots. Relay tokens use
`ESPN_RELAY_SECRET` or `FLASK_SECRET_KEY`. Snapshots live in process memory and
optional Redis (`REDIS_URL`), TTL ~12h.

## Known ESPN API limits

`mDraftDetail` often does not grow mid-draft. Treat frozen REST as unavailable
unless extension / bookmarklet relay is feeding picks.
