# Draft evaluation and projection data-flow audit

This audit was completed before the canonical resolver migration. It records
the architectural causes rather than player-specific symptoms.

| Feature | Before source/path | Projection type | Conversion/fallback | Root cause / after path |
|---|---|---|---|---|
| Draft Room, board, Pick Score, recommendation, grade | `/api/league-players` → `fetch_sleeper_season_projections` plus `proj_ppg_by` → `DraftBoardCore.scoringProjPpg` | season average | Browser selected/scaled a variant | Authority and scoring selection were split across server and browser. The API now emits `projection` from `utils.projection_resolver`; the board consumes that exact canonical value. |
| Cheat Sheet | shared league-player payload | season average | same browser board helper | Now consumes the same canonical API field and provenance. |
| Player modal/page | model table and league-player payload; some weekly page bundles | mixed | labels did not always expose context | Season values use the canonical payload. Weekly values remain valid but must retain `projection_type=weekly`. |
| Start/Sit, matchups, scout | `build_projections_by_week` / cached Sleeper weeks | weekly | `projection_points` | Weekly use is intentional. The resolver reuses the same centralized scoring function and makes week/type explicit. |
| Waivers | direct season fetch plus recent-PPG fallback | season/forward role | feature-local fallback | Direct provider selection caused fallback drift. New work should consume resolver results; historical PPG remains a distinct signal, not projected PPG. |
| VOR and Draft Grade | `proj_ppg` already attached to player rows | season average | sometimes `proj_pts`, otherwise PPG × games | They now receive the displayed canonical PPG in the shared row; VOR total derivation is a model operation, not a second PPG authority. |
| Playoff simulations | cached Sleeper weekly maps and season aggregate | weekly/season | feature-owned map selection | Weekly simulation is intentionally different from season average; diagnostics compare both explicit contexts. |

## Duplicate logic and observed divergence

* Draft Room's higher value came from the client-side `proj_ppg_by` ratio path
  in `DraftBoardCore.scoringProjPpg`: a PPR baseline was scaled again for the
  selected variant. The displayed value and model input shared that transformed
  number, but it was not the single authoritative Sleeper context.
* Modal/player data could use a model-table or current-week bundle. A weekly or
  historical/fallback value could therefore appear under a generic projected
  PPG label, explaining lower values such as the reported 11.8.
* Fetching Sleeper was centralized only by convention. There was no result
  contract with source, type, season, scoring fingerprint, week, or fallback
  metadata, and existing cache names did not encode scoring context.
* K/DEF exposed a unit-safety hole: a numeric weekly-cache entry is accepted by
  `projection_points` as already-scored week points. If an upstream season total
  such as ~95 was stored in that shape, the previous resolver treated it as one
  weekly PPG observation. The resolver now rejects position-aware implausible
  PPG, logs the corrupt origin, and resolves explicit Sleeper `pts_* / gp`
  season fields through the shared active-game denominator instead. It never
  clamps a season total into a plausible-looking PPG.

## Draft-evaluation audit

`DraftBoardCore` already separates the absolute Pick Score kernel from live
`decisionScore`, calibrated survival, opportunity-cost verdicts, ADP uncertainty,
reach classification, conservative late-round utility, and impact-based bye
severity. `draft_grade_team.js` and `utils/draft_grade.py` already mirror a
role-weighted mean with starter/primary/fringe weights, smooth round decay,
format-aware QB/TE depth, functional depth, and zero K/DEF influence. The
late-draft Pick Score depth normalization is retained: it remains an
absolute-at-slot calibration shared by JS/Python, while Board PS is calculated
against the historical remaining pool. Removing it without a demonstrated
backtest gain would blur rather than improve those semantics.

## ESPN scoring normalization follow-up

ESPN league loading previously inferred reception scoring from
`espn_api.Settings.scoring_type`, defaulted a missing/unknown label to
`standard`, and never read the authoritative
`settings.scoringSettings.scoringItems` returned by the `mSettings` view.
League-context construction then discarded the normalized non-Sleeper scoring
object from `raw_scoring_settings`, causing projection and weekly consumers to
fall back independently. ESPN stat ID **53** is the actual "Each reception"
item; its `points` (or explicit slot-16 override) now populates canonical `rec`
without truthiness coercion. The same normalized object is retained for modal
stat math, Draft Room configuration, projections, Start/Sit and simulations.

## Runtime contract cleanup follow-up

The first resolver implementation loaded weeks 1-18 and selected their median
before consulting Sleeper's season projection. That made `season_average` mean
"typical weekly product" even when the distinct Sleeper season product existed.
The resolver and legacy season helper now use: Sleeper season stat line first,
weekly-derived median only when absent, then explicit secondary/conservative
fallbacks. Results expose `source_projection_type` so `sleeper_season`,
`sleeper_weekly_derived`, and `sleeper_week` cannot be confused.

Normal `/api/league-players` rows always carry a canonical `projection` object,
including an explicit unavailable object if resolution fails. The projection
schema/cache version is v2, Sleeper season cache is v2 (and preserves raw stats
for custom scoring), and Player Modal localStorage is `pm_cache_v3_`, preventing
pre-migration scoring/projection payloads from surviving deployment.

### Projection consumer matrix

| Feature | Producer / runtime path | Type | Frontend transformation |
|---|---|---|---|
| Draft cards, Recommendation, Pick Score, Grade, Deep Dive | `/api/league-players` canonical row | season_average | `DraftBoardCore.scoringProjPpg` reads `projection.ppg` |
| Cheat Sheet | same league-player payload | season_average | formatting only |
| Player Modal / Player Page season data | canonical league-player/player payload | season_average | formatting only; game-log rows are actual/weekly |
| VOR/VORP, team strength, trade roster inputs | canonical `proj_ppg` copied from the row | season_average | model aggregation only |
| Matchups, Start/Sit, Scout | `build_projections_by_week` / `projection_points` | weekly | explicit requested week |
| Waivers | canonical resolver plus distinct recent/weekly opportunity inputs | season_average + explicit forward signals | no provider reselection |
| Simulations / playoff odds | canonical resolver preseason; Sleeper requested weeks in-season | explicit season/weekly | simulation aggregation |
| Breakout / market intelligence | canonical row when PPG is consumed; raw projected stats for feature modeling | explicit by feature | no display authority |
