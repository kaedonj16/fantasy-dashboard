# ESPN & Yahoo live-draft companion (Draft Room)

Observe an ESPN or Yahoo snake/linear draft from this site's Draft Room. The
host site still makes the picks. This app maps players onto the existing board
and recalculates Pick Scores / recommendations. **Picks are never submitted to
ESPN or Yahoo.**

## Sync paths

| Path | How picks arrive | Devices |
|------|------------------|---------|
| **REST poll** | Server polls ESPN `mDraftDetail` or Yahoo `draftresults` | All (ESPN often stale mid-draft; Yahoo usually updates) |
| **Extension relay** | Extension reads open draft room → Draft Room + server store | Desktop Chrome/Edge |

Connect Live Draft, then install the Chrome extension (Draft Room → **Get Chrome
extension**, or load unpacked `extension/`) and keep the ESPN or Yahoo draft tab
open.

On a phone, track picks manually in Draft Room (or use a laptop for auto-sync).

Backend: `dashboard_services/draft_sync.py`, `espn_draft.py`, `yahoo_draft.py`,
`espn_draft_relay.py` / `yahoo_draft_relay.py` (snapshot store). APIs:
`GET|POST /api/draft/espn-relay`, `GET|POST /api/draft/yahoo-relay`,
`GET /api/draft/live` (merges stored relay when fresher than REST).

## Verify

1. Load unpacked `extension/` → Connect Live Draft → open ESPN or Yahoo draft.
2. Make a pick on the host site → BR board updates; draft page shows a BR Fantasy chip.

## Diagnostics

`ESPN_DRAFT_SYNC_DEBUG=1` logs ESPN REST snapshots. Snapshots live in process
memory and optional Redis (`REDIS_URL`), TTL ~12h.

## Known host API limits

ESPN `mDraftDetail` often does not grow mid-draft. Yahoo `draftresults` usually
does, but the desktop extension still gives faster updates and covers API gaps.
Treat frozen REST as unavailable unless the desktop extension relay is feeding
picks (especially for ESPN).
