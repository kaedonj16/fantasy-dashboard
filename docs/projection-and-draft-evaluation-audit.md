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
