# Custom Draft Board (personal ranking overrides)

Status: cheat sheet, with server persistence and drag-reorder. Pro feature.
Draft Room integration still pending.

## Goal

Let a manager bend the model's board to their own view without throwing away
the work the model already did. The board still loads fully ranked by value
over replacement (VOR); the user layers light overrides on top: drag a player
to a new spot (or nudge one row at a time with the arrows), pin a player to the
top, or mute a player to the bottom. Their edits persist per league and, in the
full version, follow them into the live Draft Room.

## Principles

1. Start from the model, never a blank slate. Zero-setup value on first open;
   overrides are optional tweaks.
2. Overrides ride the model scale, not stale absolute numbers. We store a
   fractional rank between two neighbours ("halfway between the players the model
   ranks 13th and 14th"), not "rank 14". A weekly value refresh that re-ranks the
   pool re-anchors those neighbours, so the moved player stays put relative to
   them instead of pinning a stale integer.
3. The board stays tier-organized and monotonic. A moved player adopts the tier
   it settles into, so the cliff dividers never repeat or go non-monotonic.
4. Always reversible. A per-player revert and a whole-board "Reset board".

## Override model

Per player, exactly one of: a fractional rank, pinned, or muted.

| Action     | Stored          | Effect                                              |
|------------|-----------------|-----------------------------------------------------|
| Move/drag  | `r = <float>`   | sorts at `r`, set midway between the chosen neighbours |
| Pin        | `p = true`      | floats to a "Pinned" bucket above Tier 1            |
| Mute       | `m = true`      | sinks to a "Muted" bucket below every tier          |

Pin and mute are terminal buckets and are mutually exclusive; taking one clears
the other. Moving (drag or arrow) clears pin/mute and writes a fresh `r`.

Storage shape (per player id): `{ r?: number, p?: true, m?: true }`. Empty
entries are pruned so an untouched board stores nothing.

## Ordering

For each player compute a sort bucket and an effective rank:

- pinned  -> bucket `-1`
- muted   -> bucket `+1`
- else    -> bucket `0`, effective rank = `r` if moved else the model index

Sort by `(bucket asc, effectiveRank asc, modelRank asc)`. Because a move writes
`r` midway between the two neighbours it was dropped between, a drop (or a
one-row arrow nudge) lands exactly there and never ties into place. Renumber the
RK column to the resulting order, and give each moved row the tier of the
model-ranked player above it so the sequence reads
`Pinned -> Tier 1 -> Tier 2 -> ... -> Muted`. The name chip shows the net move
(`▲N` / `▼N`) versus where the model had the player.

## Persistence (implemented)

Two layers, local for speed and server for durability:

- `localStorage`, keyed by `csboard:<leagueId>:<mode>:<sf>`, is the instant cache
  and offline fallback. It backs the synchronous board compute.
- Server table `draft_board_overrides`, keyed by
  `(owner_key, platform, league_id, board_key)` where `owner_key` is
  `acct:<accountId>` for a logged-in account or `sleeper:<viewerId>` otherwise,
  and `board_key` is `<mode>:<1qb|sf>`. This makes the board cross-device.

Flow: on opening a board, read localStorage immediately, then `GET
/api/draft-board/overrides` and adopt the server copy if it differs (server is
the source of truth across devices). On every edit, write localStorage
synchronously and `POST` to the server debounced (~600ms). All endpoints are
premium-gated; a free viewer never loads or writes overrides.

## Reset semantics

- Per player: a revert (↶) control appears on any overridden row and clears that
  player back to its model spot.
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

1. [done] Cheat-sheet overrides with localStorage, pin/mute/bump, reset,
   pro-gated.
2. [done] Server persistence per (owner, league, format); localStorage is now a
   cache in front of it.
3. [done] Full drag-to-reorder (pointer-based, mouse + touch) plus one-row arrow
   nudges, on the fractional-rank model so drops stay refresh-safe.
4. [next] Draft Room best-available reads the same overrides (the API and
   `board_key` scheme are already in place; `DraftBoardCore` applies the same
   bucket/sort step after it ranks the pool).

## Open questions

- Sharing/exporting a personal board (CSV already exports the current order).
