# ESPN live-draft companion (Draft Room)

Observe an ESPN snake/linear draft from this site's Draft Room. ESPN still
makes the picks. This app maps players onto the existing board and recalculates
Pick Scores / recommendations. **Picks are never submitted to ESPN.**

## Two sync paths

| Path | How picks arrive | When it works |
|------|------------------|---------------|
| **REST poll** | Server polls ESPN `mDraftDetail` every ~5–10s | Sometimes mid-draft; reliably after completion |
| **Extension relay** | BR Fantasy extension reads the open ESPN draft room and relays picks | During live drafts when REST is stale |

Use both: Connect Live Draft as usual, keep the ESPN draft tab open, and install
the extension (`extension/`). If REST never grows picks, the extension keeps the
board live so you do not have to click each pick manually.

Live ESPN picks use the same `state.picks` / `drafted` / `render()` path as
manual, mock, and Sleeper live sync. There is not a second Draft Room.

Backend: `dashboard_services/draft_sync.py` (normalized picks + reconcile) and
`dashboard_services/providers/espn_draft.py` (REST `mDraftDetail`). Extension
relay posts to `POST /api/draft/espn-relay`. Frontend: `static/draft_room.js`
(`applyOneLivePick` / `applyMissingLivePicks` / `applyEspnExtensionRelay`).

## Verify against an ESPN mock / test draft

1. Link a public or private ESPN league (private: `SWID` + `espn_s2` stay
   server-side). Open **Draft Room** from that league.
2. Load the unpacked extension from `extension/` (see `extension/README.md`).
3. Start an ESPN mock draft (or join a real one) on ESPN in another tab.
4. In Draft Room click **Connect Live Draft**. You should see
   `ESPN Draft · Connecting` then `ESPN Draft · LIVE · Pick X.YY` once
   `inProgress` is true.
5. Make a pick **on ESPN**. Within a couple of seconds the board should:
   - show the pick in the correct seat
   - remove the player from available
   - update your roster if it was your team
   - refresh Pick Score / recommendation order
6. Refresh this site. The first poll after reload reconciles ESPN's full pick
   list when REST has data; otherwise reopen the ESPN draft tab so the
   extension can re-relay.
7. Duplicate ESPN responses must not create a second copy of a pick.

If ESPN's REST feed never grows **and** the extension is not installed / draft
tab is closed, Draft Room falls back to **ESPN live sync unavailable** and
**Switch to Manual Tracking**.

## Diagnostics

Set `ESPN_DRAFT_SYNC_DEBUG=1` (same env-flag style as
`MARKET_DEBUG_PROVIDER_RESPONSES`). Logs look like:

```
[espn-draft-sync] league_id=… season=… inProgress=True drafted=False
status=drafting picks=20 latest_overall=20 latest_player=4039057
latest_team=3 picks_observed=True detail_present=True unresolved=0
changed=True fingerprint=drafting|1|0|20|20|4039057|3
```

**REST live sync is working only when `picks` and `latest_overall` increase after
ESPN selections.** HTTP 200 with `picks_observed=False` / `detail_present=False`
or a frozen `picks=0` while `inProgress=True` means ESPN is not exposing live
picks over REST — use the extension relay. Credentials (`espn_s2`, `SWID`,
cookies) are never logged.

Optional:

- `ESPN_DRAFT_SYNC_POLL_SECONDS` — 5–10 (default 8)
- `ESPN_DRAFT_SYNC_STALL_POLLS` — consecutive in-progress polls with no pick
  growth before fallback (default 8)

## Known ESPN API limits

`mDraftDetail` is undocumented. ESPN often snapshots `picks` at draft start or
only after completion; many live drafts never update the REST view. The
maintainer of the popular `espn-api` Python package confirms live drafts use a
different path. Treat frozen REST as **sync unavailable** unless the extension
relay is feeding picks.

Predraft `picks` is often a full grid of empty slots (`playerId` 0 / -1 / null).
Those are not selections: Draft Room leaves the board empty (no Unknown names,
no grade) until a real player id appears. Keepers with a real `playerId` still
show.
