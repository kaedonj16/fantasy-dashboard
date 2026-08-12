# Custom Draft Board (personal ranking overrides)

Status: prototype (client-side, cheat sheet only). Pro feature.

## Goal

Let a manager bend the model's board to their own view without throwing away
the work the model already did. The board still loads fully ranked by value
over replacement (VOR); the user layers light overrides on top: bump a player up
or down a tier, pin a player to the top, or mute a player to the bottom. Their
edits persist per league and, in the full version, follow them into the live
Draft Room.

## Principles

1. Start from the model, never a blank slate. Zero-setup value on first open;
   overrides are optional tweaks.
2. Overrides are intent, not absolute positions. We store "up 2 tiers" or
   "muted", not "rank 14", so a weekly value refresh that re-ranks the pool keeps
   the user's intent instead of pinning stale numbers.
3. The board stays tier-organized and monotonic. Overrides move a player between
   tiers rather than inserting them mid-list, so tier dividers never go
   non-monotonic.
4. Always reversible. A per-player reset and a whole-board "Reset to model".

## Override model

Per player, at most one bucket plus a fine delta:

| Action    | Stored              | Effect                                             |
|-----------|---------------------|----------------------------------------------------|
| Bump up   | `d += 1`            | effective tier = clamp(modelTier - d, 1..maxTier)  |
| Bump down | `d -= 1`            | effective tier = clamp(modelTier - d, 1..maxTier)  |
| Pin       | `p = true`          | floats to a "Pinned" bucket above Tier 1           |
| Mute      | `m = true`          | sinks to a "Muted" bucket below every tier         |

Pin and mute are terminal buckets and are mutually exclusive; taking one clears
the other. Bumping up/down clears pin/mute and edits the fine delta.

Storage shape (per player id): `{ d?: int, p?: true, m?: true }`. Empty entries
are pruned so an untouched board stores nothing.

## Ordering

For each player compute a sort bucket:

- pinned  -> `-1`
- muted   -> `+Infinity`
- else    -> `effectiveTier = clamp(modelTier - d, 1, maxTier)`

Sort by `(bucket asc, modelRank asc)`. A bumped-up player joins the target
tier at the bottom (their VOR is genuinely lower than the natives, so they sit
last in that tier). Renumber the RK column to the resulting order. Tier
dividers are driven by the bucket, so the sequence reads:
`Pinned -> Tier 1 -> Tier 2 -> ... -> Muted`.

## Persistence

- Prototype: `localStorage`, keyed by `csboard:<leagueId>:<mode>:<sf>` so a
  dynasty-SF board and a redraft-1QB board are independent.
- Full version: server-side per (account, league, format) so it is cross-device
  and can be read by the Draft Room. Migrate the localStorage blob on first
  authenticated load.

## Reset semantics

- Per player: an active override chip on the row clears that player.
- Whole board: a "Reset board" control wipes the current view's overrides.

## Draft Room integration (full version, not in the prototype)

`DraftBoardCore` already produces the best-available order the Draft Room shows.
The override map (loaded from the server) applies as the same bucket/sort step
after the core ranks the pool, so the live best-available list and the cheat
sheet agree. Live cross-off is unchanged and stays pro.

## Pro gating

The override controls, the custom ordering, and reset are premium
(`cfg.hasPremium`). A free viewer always sees the pure model board. This matches
the live-assistant layer: the static model board is free, personalization is
pro.

## Rollout phases

1. (this prototype) Cheat-sheet overrides with localStorage, pin/mute/bump,
   reset, pro-gated.
2. Server persistence per (account, league, format); migrate localStorage.
3. Draft Room best-available reads the same overrides.
4. Optional: full drag-to-reorder, once the intent-based persistence across
   model refreshes is proven.

## Open questions

- Drag reorder later: map a dropped position back to an intent (nearest tier +
  delta) rather than an absolute rank, to stay refresh-safe.
- Sharing/exporting a personal board (CSV already exports the current order).
