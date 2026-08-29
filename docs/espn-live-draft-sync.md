# ESPN live-draft companion (Draft Room)

Observe an ESPN snake/linear draft from this site's Draft Room. ESPN still
makes the picks. This app maps players onto the existing board and recalculates
Pick Scores / recommendations. **Picks are never submitted to ESPN.**

## Sync paths

| Path | How picks arrive | Devices |
|------|------------------|---------|
| **REST poll** | Server polls ESPN `mDraftDetail` | All (often stale mid-draft) |
| **Extension relay** | Extension reads open ESPN draft room → Draft Room + server store | Desktop Chrome/Edge |

Connect Live Draft, then install the Chrome extension (Draft Room → **Get Chrome
extension**, or load unpacked `extension/`) and keep the ESPN draft tab open.

On a phone, track picks manually in Draft Room (or use a laptop for auto-sync).

Backend: `dashboard_services/draft_sync.py`, `espn_draft.py`,
`espn_draft_relay.py` (snapshot store). APIs:
`GET|POST /api/draft/espn-relay`, `GET /api/draft/live` (merges stored relay when
fresher than REST).

## Verify

1. Load unpacked `extension/` → Connect Live Draft → open ESPN draft.
2. Make a pick on ESPN → BR board updates; ESPN page shows a BR Fantasy chip.

## Diagnostics

`ESPN_DRAFT_SYNC_DEBUG=1` logs REST snapshots. Snapshots live in process memory
and optional Redis (`REDIS_URL`), TTL ~12h.

## Known ESPN API limits

`mDraftDetail` often does not grow mid-draft. Treat frozen REST as unavailable
unless the desktop extension relay is feeding picks.
