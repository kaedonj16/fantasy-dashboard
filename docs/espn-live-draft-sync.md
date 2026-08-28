# ESPN live-draft companion (Draft Room)

Observe an ESPN snake/linear draft from this site's Draft Room. ESPN still
makes the picks. This app polls `mDraftDetail`, maps players onto the existing
board, and recalculates Pick Scores / recommendations. **Picks are never
submitted to ESPN.**

## How it fits the existing Draft Room

Live ESPN picks use the same `state.picks` / `drafted` / `render()` path as
manual, mock, and Sleeper live sync. There is not a second Draft Room.

Backend: `dashboard_services/draft_sync.py` (normalized picks + reconcile) and
`dashboard_services/providers/espn_draft.py` (one `mDraftDetail` fetch using
the existing ESPN auth). Frontend: `static/draft_room.js` (`applyOneLivePick` /
`applyMissingLivePicks`).

## Verify against an ESPN mock / test draft

1. Link a public or private ESPN league (private: `SWID` + `espn_s2` stay
   server-side). Open **Draft Room** from that league.
2. Start an ESPN mock draft (or join a real one) on ESPN in another tab.
3. In Draft Room click **Connect Live Draft**. You should see
   `ESPN Draft · Connecting` then `ESPN Draft · LIVE · Pick X.YY` once
   `inProgress` is true.
4. Make a pick **on ESPN**. Within ~5–10 seconds the board should:
   - show the pick in the correct seat
   - remove the player from available
   - update your roster if it was your team
   - refresh Pick Score / recommendation order
5. Refresh this site. The first poll after reload reconciles ESPN's full pick
   list (not only picks seen in the previous browser session).
6. Duplicate ESPN responses must not create a second copy of a pick.

If ESPN reports a live draft but `picks` never grow, Draft Room falls back to
**ESPN live sync unavailable** and **Switch to Manual Tracking** (same engine,
you click Draft yourself).

## Diagnostics

Set `ESPN_DRAFT_SYNC_DEBUG=1` (same env-flag style as
`MARKET_DEBUG_PROVIDER_RESPONSES`). Logs look like:

```
[espn-draft-sync] league_id=… season=… inProgress=True drafted=False
status=drafting picks=20 latest_overall=20 latest_player=4039057
latest_team=3 picks_observed=True detail_present=True unresolved=0
changed=True fingerprint=drafting|1|0|20|20|4039057|3
```

**Live sync is working only when `picks` and `latest_overall` increase after
ESPN selections.** HTTP 200 with `picks_observed=False` / `detail_present=False`
or a frozen `picks=0` while `inProgress=True` means ESPN is not exposing live
picks. Credentials (`espn_s2`, `SWID`, cookies) are never logged.

Optional:

- `ESPN_DRAFT_SYNC_POLL_SECONDS` — 5–10 (default 8)
- `ESPN_DRAFT_SYNC_STALL_POLLS` — consecutive in-progress polls with no pick
  growth before fallback (default 8)

## Known ESPN API limits

`mDraftDetail` is undocumented. ESPN often snapshots `picks` at draft start or
only after completion; some live mocks never update the REST view. This feature
treats that as **sync unavailable**, not as a successful empty draft.

Predraft `picks` is often a full grid of empty slots (`playerId` 0 / -1 / null).
Those are not selections: Draft Room leaves the board empty (no Unknown names,
no grade) until a real player id appears. Keepers with a real `playerId` still
show.
