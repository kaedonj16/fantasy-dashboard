# Redzone Plays implementation plan (revised)

## Status

Revised against `main` after **#1643** (`a2519bae` / `44ae90bb`): real Tank01
PBP, soft rank, saved prefs, Big Plays toggle, and richer play cards are
**already shipped**. This plan does **not** rebuild those surfaces. It extends
them to fix confirmed correctness and layout gaps.

Prior chat roadmap (six Plays upgrades) is treated as **done**. Work below is
the next implementation pass.

## Scope and guardrails

**In scope**

- Play / contribution identity and feed dedupe
- Event card delta + post-play total under league scoring context
- Per-game PBP coverage vs snapshot (box-score) updates
- Complete fantasy scoring for PBP / diff events (K, DEF, bonuses, corrections)
- Historical play metadata honesty + Latest / For You sort modes
- My Teams compact rows, full expandable bench, NFL matchup filter + game strip
- FINAL labeling and partial DOM refresh

**Out of scope / do not rebuild**

- Tank01 `playByPlay=true` → `utils/redzone_pbp.extract_pbp_plays` →
  `pbp_by_game` payload attachment in `app.py`
- Booth-style `_describe` / `demo_play_text` copy
- Down / distance / yard line / Q-clock rendering on event cards (`_downDist`,
  `_eventHtml` meta bits)
- Prefs via `_prefsKey('hero-mid'|'my-team'|'big-only')` →
  `rz-{kind}:{platform}:{league_id}:{scope}` in `localStorage`
- Big Plays toggle + `_isBigPlay` (TD or ≥4 pts) + existing FX
- Soft-rank as the default **For You** ordering (mine → opp → rest, then TD /
  pts / recency) — keep it; add an alternate **Latest** mode beside it
- Demo discovery via empty-state CTA + Weekly nav `?demo=1` (no header Demo pill)
- Scope-switch feed reset / `pbp_by_game` merge already covered by
  `tests/test_redzone_scope_switch.py`

**Principles**

- Extend `_eventsFromPbp`, `_detectChanges`, `_lineToPts` / shared scoring,
  `_eventHtml`, `_rosterCard`, `_nflOptions`, `_partialUpdate` — do not replace
  the feed architecture.
- Prefer honesty: missing history → unavailable/pending; never invent totals or
  play clocks from live scoreboard.
- Cross-league scoring stays per-league (`scoring_by_league` / `_scFor`); store
  totals with an explicit league scoring context on the event.
- Mobile: keep existing density; compact My Teams and matchup filter must remain
  usable on narrow viewports.

---

## Current-state data-flow (post-#1643)

### Server

1. `_redzone_boxscore(game_id, play_by_play=…)` caches Tank01 boxscores; live
   games (`game_code == "1"`) request PBP; finals use plain boxscore.
2. `_rz_extract_pbp_plays` flattens `allPlayByPlay` into **one row per fantasy
   contributor**, sharing Tank01 `playId` / `play_id` and attaching `stat_line`
   via `rz_stat_line_from_ps` / `rz_def_stat_line`.
3. Payload includes `pbp_by_game[game_id] → [plays…]`, `player_info` (current
   clock/scoreboard), `matchups`, `scoring` / `scoring_by_league`.
4. Demo scripts emit the same `pbp_by_game` shape (demo `play_id`s are already
   pid-scoped and therefore hide the live dedupe bug).

### Client

1. `_detectChanges` prefers `_eventsFromPbp`, then box-score `_playsFromDiff`,
   then `players_points` delta fallback.
2. `_eventsFromPbp` dedupes with `_seenPlayIds` keyed by raw `play_id` when
   present; stamps `_gameLine(pid)` (live scoreboard) and may fall back to live
   `game_clock` / `game_quarter`.
3. New batches are `_softRank`ed and unshifted into `_feed` (cap 200). Cold-boot
   PBP is sliced to the last **40** events before insert.
4. `_eventHtml` shows a large signed delta only; `_impactLine` repeats points and
   can label any owned player “your starter”.
5. Poll path always full `_render()`; `_partialUpdate` exists but is unused.

---

## Confirmed issues → work items

### 1. Play identity and dropped player contributions

**Evidence**

- `utils/redzone_pbp.py` emits multiple contributions with one shared
  `play_id` (QB + receiver on a completion).
- `_eventsFromPbp` uses
  `playKey = String(play.play_id || (gid + ':' + seq + ':' + pid))` then
  `_seenPlayIds.add(playKey)` **before** roster / describe gates. Same Tank01
  `playId` across games collides; second contributor on a play is dropped.
- Once seen, corrections never update the existing feed row.

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ1.1 | Separate **NFL play identity** = `game_id` + provider `play_id` (fallback `game_id:seq`) from **contribution identity** = play identity + `pid` | `_eventsFromPbp`; optionally stamp both fields in `extract_pbp_plays` |
| RZ1.2 | Dedupe / `_eid` on contribution identity, not raw `play_id` | `_eventsFromPbp`, `_eid` |
| RZ1.3 | Do **not** mark a contribution processed until it is accepted into the feed (rostered + displayable). Unmapped pids may retry on later polls | `_eventsFromPbp` |
| RZ1.4 | Support **updates** to existing contributions (stat_line / play_text / pts / totals) instead of permanently ignoring revisions | `_eventsFromPbp` + `_syncFeed` update path |
| RZ1.5 | Keep league scoring attachment on the contribution (`scoring_league_id` or equivalent), separate from NFL play identity | event object |

**Regression**

- Completion awarding both passing and receiving points → **two** feed events
  with distinct contribution ids and shared play identity.
- Identical provider `play_id` in two different `game_id`s → no collision.

---

### 2. Keep the delta; add total underneath

**Evidence**

- `_eventHtml` renders only `.rz-event-delta` with signed `ev.pts`.
- No `total` / post-play fantasy points stored on the event.
- Cold-boot visible PBP is capped at 40; recomputing totals from the visible
  feed or from the latest `players_points` at render time is wrong for older
  plays.
- `_impactLine` repeats the delta (`+6.6 pts`) and uses “your starter” for any
  `mine` non-TD (bench included — `mine` is roster ownership, not starter set).

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ2.1 | Preserve large signed delta appearance; add smaller muted “`12.8 total`” under it in the same right-aligned column | `_eventHtml` + CSS |
| RZ2.2 | Store `pts_total` (or `afterPts`) on the event under an **explicit league scoring context** at event-build time | `_eventsFromPbp`, `_playsFromDiff`, points-delta path |
| RZ2.3 | Reconstruct totals from complete ordered contribution history for that `(league, pid)` **or** an authoritative baseline (`players_points` at hydrate + subsequent accepted deltas). If history is incomplete → show unavailable / pending, never a guessed number | new helper; cold-boot must baseline before or instead of trusting the 40-cap slice alone |
| RZ2.4 | `_impactLine`: remove delta repetition; reserve for ownership / matchup context only. Starter vs bench must use `matchup.starters` — never call a bench player “your starter” | `_impactLine` |

---

### 3. Switching between PBP and snapshot updates

**Evidence**

```js
if (gid && _pbpGames[gid] && pbpEvents.length) {
  handled[pid] = true;
  return;
}
```

- `_pbpGames[gid] = true` for any key in `pbp_by_game`, even if every play is
  skipped.
- `pbpEvents.length` is newly accepted events **across all games** this poll,
  not coverage for `gid`.
- A quiet PBP poll (`length === 0`) re-enables box-score fiction → duplicates.
- No explicit “PBP unavailable → labeled snapshot → reconcile when PBP returns”
  transition.

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ3.1 | Replace boolean `_pbpGames` with per-game coverage: e.g. `{ covered, lastSeq, lastPollHadPbp, mode: 'pbp'|'snapshot'|'reconciling' }` | module state + `_eventsFromPbp` / `_detectChanges` |
| RZ3.2 | Suppress snapshot diffs for a game only when that game’s PBP coverage is fresh for this poll (including “PBP present, zero new plays”) | `_detectChanges` |
| RZ3.3 | If PBP drops for a previously covered game, switch to **labeled** snapshot updates (`fromPbp: false`, source tag) rather than silent fiction | event fields + optional UI chip |
| RZ3.4 | When PBP returns, reconcile: prefer PBP contributions; collapse / supersede overlapping snapshot rows for the same contribution window where safe | `_detectChanges` |

---

### 4. Incomplete scoring

**Evidence**

- `_lineToPts` only covers pass/rush/rec yards + TDs + INT. K (`fgm`/`xpm`) and
  DEF (`sacks`/`def_int`/`fum_rec`/`def_td`) narrate via `_describe` but score
  **0**.
- Shared engine `utils.fantasy_scoring.score_stats` is unused by Redzone live
  math. Stat keys also differ (`pass_yds` vs `pass_yd`, `sacks` vs Sleeper
  `sack`, etc.).
- `rz_stat_line_from_ps` only reads nested `Passing`/`Rushing`/`Receiving`/
  `Kicking` — flat top-level Tank01 deltas (if present) become empty lines and
  are dropped by `_stat_line_nonzero`.
- Points-delta fallback rejects `delta <= 0.05`, so negative corrections never
  appear.

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ4.1 | Map redzone `stat_line` → scoring-engine stat keys; call shared scoring (JS port of the supported subset **or** precompute pts server-side with `score_stats`). Cover K, DEF, fum_lost, and league bonuses/deductions present in settings | replace/extend `_lineToPts`; optionally thin Python helper used by tests |
| RZ4.2 | Audit `rz_stat_line_from_ps` / `rz_def_stat_line` for flat + nested Tank01 shapes and for fields the engine needs; preserve genuine zeroes vs missing | `utils/redzone_stats.py` + tests |
| RZ4.3 | Points-delta path: accept meaningful negative adjustments (corrections / point losses); keep a small noise floor for float jitter only | `_detectChanges` fallback |
| RZ4.4 | Document provider fields still unavailable after the audit (see § Provider gaps) | this doc + PR notes |

---

### 5. Historical context and ordering

**Evidence**

- Missing play clock falls back to live `player_info.game_clock` /
  `game_quarter`.
- `_gameLine` always uses current home/away points — stamped as if it were play
  context.
- Soft-rank prioritizes ownership, TDs, and point size ahead of recency; small
  recent plays stay buried. No Latest / For You control.
- `ts: Date.now() + seq * 0.001` is browser arrival with a tiny seq nudge — not
  cross-game chronology.

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ5.1 | Prefer play metadata (`quarter`, `clock`, down/distance/yardline) only when present; do **not** substitute live clock. Omit or mark “live board” separately from play context | `_eventsFromPbp` |
| RZ5.2 | Do not present `_gameLine` live scoreboard as verified historical play score; keep current board as a clearly current status affordance if shown | `_eventHtml` / event fields |
| RZ5.3 | Preserve provider `seq` / play order **within** a game; do not invent precise cross-game chronology from arrival time | event `seq` + sort keys |
| RZ5.4 | Add sort modes **Latest** (recency / within-game seq first) and **For You** (current `_softRank`). Persist with existing prefs | prefs + `_syncFeed` / filter chrome |
| RZ5.5 | Default remains For You so soft-rank behavior is preserved for returning users | prefs defaults |

---

### 6. Layout changes

**Evidence**

- `_renderMyTeams` renders a section label + full `_rosterCard` per league.
- `_rosterCard` uses `bench.slice(0, 6)` with no expand / count.
- `_nflOptions` lists individual team abbrevs; no selected-game scoreboard strip
  (logos, possession, down/distance, field position, quarter, clock).

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ6.1 | My Teams: compact expandable rows — fantasy matchup score, lead/deficit, playing-now count, yet-to-play count; expand reveals existing roster detail | `_renderMyTeams` (+ CSS) |
| RZ6.2 | Replace silent bench truncation with expandable full bench + player count | `_rosterCard` |
| RZ6.3 | NFL filter: list **matchups** (`AWAY @ HOME` / game_id), not lone teams | `_nflOptions`, `_eventMatches`, `_topMatches` |
| RZ6.4 | When an NFL matchup is selected, show a scoreboard: opposing logos, score, possession, down/distance, field position, quarter, clock — using verified fields only; hide or mark pending when provider fields are missing | new strip near Plays filters; server may need to attach game-level snapshot fields |

Preserve mobile usability: compact rows and matchup chips must fit existing
chip-bar / panel patterns.

---

### 7. Status and refresh behavior

**Evidence**

- Hero cards and `_renderScoreboard` use
  `anyLive ? LIVE : anyFinal ? FINAL : PRE` where `anyFinal` is “any starter
  final”. Mid-slate matchups (some finals, rest pre) label **FINAL** once no
  one is live.
- `_renderLeagueOthers` is worse: `anyLive ? LIVE : FINAL` with no PRE.
- `_partialUpdate` updates hero, chips, feed in place and preserves hero scroll,
  but `_tick` / `_refresh` always call full `_render()`.

**Plan**

| ID | Change | Where |
|----|--------|--------|
| RZ7.1 | FINAL only when authoritative completion **or** all applicable starters are finished (ignore empty slots / bye as needed). Otherwise LIVE / PRE / in-progress hybrid label | hero, scoreboard, Around the League |
| RZ7.2 | Wire quiet polls to `_partialUpdate` (or equivalent targeted updates) while preserving filters, focus, expanded teams, and scroll | `_refresh` / `_tick` |
| RZ7.3 | Ensure every visible tab (Plays, My Team(s), Opp, Top) still refreshes correctly under partial updates | `_partialUpdate` extensions |
| RZ7.4 | Keep full `_render` for structural changes (tab switch, scope switch, filter panel open, expand toggles) | call sites |

---

## Cross-cutting requirements (retained)

- **Cross-league scoring clarity** — events keep league tag / scoring context;
  shared players resolve via viewer roster + `_scFor` (already started; do not
  regress).
- **Unavailable / stale** — totals, clocks, possession, and snapshot mode must
  surface pending/stale rather than fake precision.
- **Corrections** — contribution updates + negative point paths (RZ1.4, RZ4.3).
- **Mobile** — compact My Teams, matchup filter, delta+total column, and sort
  control must remain usable ≤430px (existing header overflow lessons).

---

## Provider gaps (report / track)

Attached today on `player_info` / scores: `game_id`, `game_status`,
`game_code`, `game_clock`, `game_quarter`, home/away + pts, kickoff epoch.

**Not reliably attached for the selected-game scoreboard (RZ6.4)**

| Field | Notes |
|-------|--------|
| Possession | Not on current Redzone `player_info`; may exist on Tank01 score/box payloads — audit `get_nfl_scores_for_date` / boxscore before inventing UI |
| Game-level down / distance / ball-on | Play-level only via PBP today; game snapshot may need a new extract |
| Team logos | Not in Redzone payload; site may already have NFL logo URLs elsewhere — reuse, don’t hardcode one-offs without a shared helper |
| Red-zone / goal-to-go flag | Not attached |

**Scoring / normalizer gaps to confirm in RZ4.2**

| Field | Notes |
|-------|--------|
| Flat per-play Tank01 playerStats | Nested-only mapper may miss deltas |
| FG miss / XP miss / blocks | Often absent from `stat_line` |
| Individual IDP | Out of scope unless already in league scoring + feed |
| Pts allowed / DST yards buckets | Needed for full DEF engine parity; may only update on snapshot, not per play |

After implementation, PR description must list any of the above that remain
unavailable.

---

## Suggested implementation order

1. **RZ1** identity / dedupe / updates — unblocks correct feed content  
2. **RZ4** scoring + normalizer — unblocks truthful pts / Big Plays for K/DEF  
3. **RZ3** per-game PBP coverage — stops snapshot duplicates  
4. **RZ2** totals + impact line — needs stable contributions + scoring  
5. **RZ5** metadata honesty + Latest / For You  
6. **RZ7** FINAL + `_partialUpdate`  
7. **RZ6** layout (My Teams, bench, NFL matchup strip) — may depend on provider
   audit for possession / logos  

Each step ships with focused regressions (below). Prefer small commits per RZ
group.

---

## Tests

Extend existing harnesses; avoid brittle full-DOM snapshots.

| Area | Tests |
|------|--------|
| RZ1 | Python: extract still emits 2 rows; **new JS or pure-helper tests** for contribution keys across games + both scorers kept. Prefer extracting pure key helpers if JS harness is thin |
| RZ2 | Total present when baseline complete; unavailable when history truncated; impact line has no raw delta repeat; bench ≠ “your starter” |
| RZ3 | Quiet PBP poll does not emit snapshot dupes for covered game; PBP loss labels snapshot; PBP return reconciles |
| RZ4 | K FG / DEF sack score non-zero under standard settings; negative `players_points` correction accepted; zero FG miss stays zero; missing line ≠ scored as zero bonus |
| RZ5 | Missing play clock does not copy live clock onto `fromPbp` events; Latest vs For You order differs on fixture feed |
| RZ6 | Source-level or DOM-contract tests for compact My Teams fields, full bench expand, matchup filter values |
| RZ7 | FINAL helper: mixed final+pre starters → not FINAL; `_partialUpdate` invoked on quiet refresh path (call-order / source contract like `test_redzone_scope_switch.py`) |

Keep `tests/test_redzone_pbp.py` and scope-switch contracts green.

---

## Files likely to change

| File | Role |
|------|------|
| `static/redzone.js` | Identity, coverage, scoring client, UI, sort, partial update |
| `static/dashboard.css` | Delta+total, compact My Teams, matchup scoreboard, sort control |
| `utils/redzone_pbp.py` | Optional play vs contribution fields; scoring helpers if server-side |
| `utils/redzone_stats.py` | Normalizer audit (flat + engine fields) |
| `utils/fantasy_scoring.py` | Reuse; avoid duplicating bonus math |
| `app.py` | Game-level scoreboard fields if provider supports them; demo fixtures |
| `tests/test_redzone_pbp.py` | Normalizer / extract regressions |
| `tests/test_redzone_scope_switch.py` | Source contracts for refresh / identity / FINAL |
| New focused test module(s) as needed for pure helpers |

---

## Acceptance checklist

- [ ] QB + receiver both appear for one completion; cross-game play ids isolated  
- [ ] Event card: large delta unchanged visually; muted total underneath when known  
- [ ] Incomplete history → pending/unavailable total, never a wrong number  
- [ ] Impact line has no duplicate delta; bench never “your starter”  
- [ ] Quiet PBP polls don’t spawn box-score duplicates; snapshot mode labeled when used  
- [ ] K/DEF (and configured bonuses) score via shared engine path  
- [ ] Negative corrections appear on the fallback path  
- [ ] No live clock/score presented as verified play history  
- [ ] Latest and For You sorts both work; For You remains default soft-rank  
- [ ] My Teams compact rows + expandable full bench  
- [ ] NFL filter is matchups; selected game strip uses only verified fields  
- [ ] FINAL only when matchup actually complete / all applicable starters done  
- [ ] Partial refresh preserves filters, focus, expanded state, scroll; all tabs update  
- [ ] Prefs, Big Plays, down/distance, real PBP pipeline unchanged in spirit  
- [ ] PR lists remaining unavailable provider fields  

---

## Non-goals this pass

- Rewriting Redzone as a new SPA or replacing Tank01  
- Changing the 15s boxscore cache vs longer PBP cache policy (explicitly deferred
  earlier)  
- Header Demo button (stays removed)  
- Auto-activating My Team filter on load (stays off)  
