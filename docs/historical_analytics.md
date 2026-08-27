# Historical Analytics

Phase 1–2 of the Draft Cheat Sheet historical layer: a reusable
player-season warehouse, positional finishes, leakage-safe prior-career
features, and **descriptive** age / career-stage / prior-elite /
draft-capital rates. Later phases (usage/efficiency, comps, ADP backfill,
board columns, Pick Score) build on this. They are **not** in this PR.

This document is the source of truth for datasets reused, season coverage,
scoring/tier definitions, age methodology, leakage rules, and how the two
age statistics and the two “breakout” labels differ.

## Phase 1 audit — what already exists (reused, not replaced)

Creating a parallel warehouse or a second backtester is a failure condition.
The following are the datasets and functions this layer extends.

### Cheat sheet / draft surface (read-only in Phase 1)

| Path | Role |
|---|---|
| `dashboard_services/pages/cheat_sheet_page.py` | Server-built `window.__cheatCfg` blob |
| `static/cheat_sheet.js` | Board UI; strings asserted by `test_product_honesty.py` / `test_cheat_sheet_page.py` |
| `static/draft_board_core.js`, `static/pick_score.js` | Shared board/Pick Score kernels — **untouched** |
| `dashboard_services/pages/draft_room_page.py`, `static/draft_room.js` | Draft Room — **untouched** |

Board data is batch (`/api/league-players`), not per-player REST. Historical
summaries will ride that payload in Phase 8. A per-player REST endpoint is for
the deep modal only (Phase 8). Core analytical logic must not live in JS.

### Historical warehouse (extended)

| Dataset / function | Season range | Notes |
|---|---|---|
| `data_building/external_data/player_history.py` | usage_rows **2018–2025** committed; parquet 2023–2025 (pre-Phase-1 schema) | One row per player per season. Overwrite guard refuses target-share-starved writes. |
| `build_usage_rows_for_season` | driven by `cache/sleeper_stats/` **2016–2026** | Sleeper weekly stats → per-game averages (2023+ schema) |
| `build_player_history_for_season` | same | Legacy `avg_*` columns for live valuation; **kept** |
| `build_player_history_features` | latest-season collapse | Live `player_value.py` / `value_model_training.py`. **Unchanged** (zero-fill, includes current season). New leakage-safe API is `build_prior_career_features`. |
| `cache/player_history/usage_rows_{2018-2022}.json` | 2018–2022 | **Legacy totals** schema from the breakout engine (nested `usage.gsis_id`, `ppr_total`, season totals) |
| `cache/player_history/usage_rows_{2023-2025}.json` | 2023–2025 | **Sleeper averages** schema (`avg_targets`, `ppr_ppg`, Footballguys `target_share`) |

`build_canonical_player_history_for_season` / `dashboard_services.historical.seasons.canonicalize_usage_row`
normalize **both** schemas onto one column set. Missing stats are `None`, never a
meaningful zero.

### nflverse / identity / usage (reused)

| Dataset / function | Season range | Notes |
|---|---|---|
| `nflverse_metrics._gsis_to_sleeper` (`import_ids`) | all nflverse ids | Authoritative GSIS↔Sleeper. Optional; Phase 1 also backfills GSIS from 2018–2022 usage_rows so we do not require `nfl_data_py` at rebuild time. |
| `nflverse_metrics.season_team_by_sleeper` | roster seasons | Historical team stamp (already used by `build_usage_rows_for_season`) |
| `nflverse_metrics` NGS / FTN / EPA | NGS **2016+**, FTN **2022+** | Phase 3 efficiency. Not joined in Phase 1 (`air_yards` / `adot` stay null). |
| `pfr_snap_counts.fetch_season_snap_counts` | nflverse snap counts; breakout engine treats snaps as **~2022+** | Used by `sleeper_usage.py`, **not** yet by `player_history.py`. `snap_pct == 0` with real volume is treated as missing. |
| `sleeper_bulk_stats.fetch_season_stats` | weekly cache **2016+** | Fantasy points / volume backbone |
| `sleeper_bulk_stats.fetch_season_redzone_stats` | cached **2016+** | `rec_rz_tgt_pg` / `rush_rz_att_pg` |
| `nfl_target_share.fetch_league_target_share` | Footballguys scrape (current-ish) | Comment in `player_history` says “nfl_data_py PBP”; the implementation is Footballguys. 0-share with `targets > 0` is treated as missing. |
| `cache/players_index.json` | current players | `bDay`, `exp`, `draft_year` (no round/pick). Age is **recomputed as of Sept 1 of the season**, not “age today”. |
| `player_investment.py` + `cache/player_investment/draft_history.parquet` | draft **2014+** (nflverse players.parquet / draft_picks.parquet) | `sleeper_id → draft_year / draft_round / draft_pick`. Reused for warehouse identity. |
| `dashboard_services/market_intelligence/identity.py` | n/a | Name+pos+team resolver. **Not** the player GSIS crosswalk. |
| `dashboard_services/historical_identity.py` | n/a | **Fantasy-owner** identity (`owner_id`), not player IDs. Do not reuse for ADP/player matching. |

### Breakout engine / backtester (reuse later; do not duplicate)

| Dataset / function | Season range | Notes |
|---|---|---|
| `breakout_engine/backtest_breakout_model.py` | walk-forward; outcomes from `usage_rows_{N+1}` | Breakout = not top-12 prior → top-12 next. `BREAKOUT_RANK_THRESHOLD = 12`. |
| `breakout_engine/build_historical_scores.py` | scores labelled season=N predict N+1 | Anti-leakage season labeling. `SNAP_COUNT_SEASONS = {2022,2023,2024}`. `lookback_start = max(..., 2016)`. |
| `breakout_engine/train_hit_probability.py` + `hit_probability_model.json` | trained 2022–2024 | Calibrated hit model. Phase 9 reuses this harness. |
| `_load_historical_fantasy_rankings` | prefers `fantasy_rankings_{season}.json` (absent); falls back to usage proxy | Phase 1 **replaces the proxy** with real total-points positional finishes on the warehouse. Later phases should read those finishes instead of re-deriving. |

Phase 1 does **not** add a second backtester. Repeat/breakout *rates* (Phase 2)
must use this labeling.

### ADP (Phase 5+; documented so we do not invent a parallel store)

| Dataset / function | Season range | Notes |
|---|---|---|
| `dashboard_services/adp_formats.py` | n/a | Pure capability model (`exact` / `compatible` / `generic` / `excluded`). Slim-CI tested. |
| `dashboard_services/adp_service.py` | I/O | Snapshots + `resolve_market_adp` |
| `providers/global_adp.py` | year-parameterized `fetch_mfl_adp(season)`, ESPN, Yahoo | MFL `is_mock=0`, selected-only `MIN_MFL_DRAFT_PCT` floor |
| `migrations/015_draft_adp.sql` | BR observed drafts | Native TEP |
| `migrations/029_adp_snapshots.sql` | normalized mirror | Persist via `write_adp_snapshot` |
| `docs/adp_sources.md` | source matrix | Superflex / TEP historical ADP essentially does not exist in free sources |

Phase 1 has **zero ADP dependency**. All analytics work with ADP absent.

### Projections (Phase 7; keep out of historical comps)

| Dataset / function | Notes |
|---|---|
| `data_building/fetch_projections.py` | Sleeper-only; current/upcoming season |
| `utils/projection_resolver.py` + tests | Current-year signal |
| `breakout_engine/projections.py` | Live engine |

Canonical season rows have **no** `projected_*` columns. Tests lock that.

## Architecture (slim-CI split)

```
dashboard_services/historical/     # pure logic — pytest -m "not integration"
    definitions.py                 # tiers, age-as-of, buckets, bust, smoothing
    seasons.py                     # canonicalize both usage_rows schemas
    finishes.py                    # ranks + prior-career features
    finish_rates.py                # cohort rates, sample size, Bayes, confidence
    age_curves.py                  # distribution vs conditional; data-derived prime
    career_profiles.py             # stage / repeat / engine-breakout / capital
data_building/historical/          # pandas / parquet I/O
    build_player_seasons.py        # rebuild warehouse from cache
    build_outcomes.py              # finishes applied in the same rebuild
    build_profiles.py              # parquet → historical_profile_aggregates.json
data_building/external_data/player_history.py   # existing paths + new wrappers
```

`definitions.py` / `seasons.py` / `finishes.py` / `finish_rates.py` /
`age_curves.py` / `career_profiles.py` must not import pandas or Flask.
`test_product_honesty.py` and `test_adp_formats.py` stay green: this package is
not on their import graph.

Persistence: large per-season rows stay in committed parquet under
`cache/player_history/`. Precomputed **profile aggregates** are a small JSON
file (`historical_profile_aggregates.json`) rebuilt in cron after the
warehouse. Request paths must not scan parquet row-by-row. No Postgres table
yet (next migration remains `031_*` when a request path needs it).

Rebuild:

```bash
python -m data_building.historical.build_player_seasons
python -m data_building.historical.build_profiles
```

Cron steps: `rebuild_historical_warehouse` then `rebuild_historical_profiles`
(cache only, no live NFL APIs).

## Scoring, tiers, age

- **Finish metric:** total points for the scoring format (`ppr_points`,
  `half_ppr_points`, `standard_points`). PPG ranks exist as optional
  `ppr_ppg_*_finish` columns and are **not** the default.
- **Ties:** competition rank (1, 2, 2, 4). Unranked (missing points) stay
  `None`, not last place.
- **Positional labels:** `RB1` = ranks 1–12, `RB2` = 13–24, same width for
  WR/TE/QB. Cutoffs live only in `TIER_CUTOFFS` / `POSITION_TIER_WIDTH`.
  `top_5` is in that map so “previous RB1 → top-5” does not invent a second
  cutoff.
- **Age:** `(season_start − birth_date) / 365.25`, truncated to 1 decimal,
  `season_start = Sept 1 of the season`. Buckets are UI convenience only
  (`AGE_BUCKETS`); exact age is stored.
- **Years experience:** `season − draft_year` (0 in the draft year). Missing
  draft year → `None`, never a fake rookie 0 for a veteran.
- **Draft capital:** `round_1` / `day_2` / `day_3` / `undrafted`. Missing round
  is `None`, not inferred UDFA, unless explicitly flagged.

## Prior-career features (no hindsight)

For player-season **S**, only seasons **&lt; S** enter:

`previous_season_finish/ppg/games`, `career_best_finish_before_season`,
`career_best_ppg_before_season`, `prior_top{3,5,6,12,24}_count`,
`previously_top{3,5,6,12,24}`, `first_time_top12_candidate`,
`career_seasons_before_current`.

PPR uses those names; half-PPR / standard are prefixed (`half_ppr_previous_season_finish`).

Leakage test: mutating 2022 actuals must not change any 2022 *feature* column.
Outcome columns on the 2022 row may change. 2023 features *should* move — 2022
is prior to 2023.

`build_player_history_features` still collapses to the latest season and still
zero-fills for live valuation. Do not point the historical layer at it.

## Coverage by feature (honest floors)

| Feature | Reliable from | Notes |
|---|---|---|
| Sleeper weekly volume / PPR points | **2016+** (stats cache); warehouse rows **2018+** until 2016–17 usage_rows are built | 2016–17 weekly JSON exists; Phase 1 does not rebuild those usage_rows (Footballguys target share / network). |
| Half-PPR / standard points | 2023+ recorded; 2018–22 **derived** as PPR − 0.5×rec / PPR − rec | Documented scoring identity, not an independent source. |
| Target share | where Footballguys/legacy share is present and consistent with targets | Else `None`. |
| Red-zone targets/carries | Sleeper RZ cache, 2016+ where joined | Per-game × games. |
| Snap % / snaps | sparse before ~2022; 0% + real volume → `None` | PFR/nflverse snaps not joined in Phase 1. |
| Age as-of Sept 1 | players with `bDay` (~5.6k in current index) | Retired players missing from index → `None` unless birth date is on the usage row. |
| Draft year | players_index + draft_history | |
| Draft round / pick | `draft_history.parquet` (nflverse) | |
| GSIS | 2018–22 usage_rows; 2023+ only if that sleeper id appeared in a legacy file | Full `import_ids` join is Phase 1-compatible but optional. |
| Air yards / aDOT / starts | **not in Phase 1** | `None` until Phase 3 nflverse join. |
| Historical ADP | **not in Phase 1** | |
| SF / TEP historical ADP | **does not exist** in free sources; deferred | |
| Current-season projections | excluded from this warehouse | |

Do not claim “2012+” uniformly.

## Phase 2 — age, career-stage, prior-elite, draft-capital rates

Descriptive only. Qualifying set is Phase 1 **appeared** rows (games > 0).
Default era is **2016+** (warehouse rows are 2018+). Age known preseason +
same-season finish is not leakage for `P(hit | age)`. Repeat rates use Phase 1
prior-career fields (pre-season). Nothing here enters ranking or Pick Score.

Every rate record has `sample_size`, `season_range`, `raw_rate`,
`smoothed_rate` (`empirical_bayes`, prior = position baseline,
`DEFAULT_BAYES_PRIOR_N = 10`), `confidence` (&lt;15 low / 15–39 moderate /
40–99 good / 100+ strong), and `display_pct` (whole percent). Empty cohorts
keep `raw_rate=None` (not a fake 0%). Small samples still emit raw + smoothed
+ n; they do not fake precision.

### Two age statistics (never collapse these)

1. **Distribution** — among seasons that *did* hit (e.g. RB1), what share
   sat in an age window. “X% of RB1 seasons came from ages 23–27.”
2. **Conditional hit rate** — among qualifying seasons *in* that window, what
   share hit. “X% of age-23–27 RB seasons finished RB1.”

Exact age is canonical; `AGE_BUCKETS` are UI-only. Rows with **missing age
are omitted from age curves only** (not treated as 0).

**Prime window** is data-derived, not hard-coded 23–27: Bayes-smoothed
`P(top-12 | integer age, position)` vs the known-age position baseline;
prime ages are at/above baseline with n ≥ 15 (not “low”). The window is the
longest consecutive run (ties broken by successes). The JSON stores both
stats on that derived window as `prime_window_pair`.

### Engine breakout vs first-time elite (both kept)

Copied from `breakout_engine/backtest_breakout_model.py` — do not “fix”:

- `BREAKOUT_RANK_THRESHOLD = 12`
- `PRIOR_NON_STARTER_RANK = 13`
- `was_non_starter = prior is None or prior > 13` (rank 13 is **not** a
  non-starter)
- Engine breakout = non-starter AND this season ≤ 12

**First-time elite** = `not previously_top12` AND this season ≤ 12.

These disagree: prior rank 13 → top-12 is first-time elite but not an engine
breakout; a former RB1 who was RB20 last year → RB8 is an engine breakout
but not first-time. Prior rank is Phase 1 `previous_season_finish` (last
*observed* prior season, not a calendar year-1 join). Slim tests
string-assert the engine source still has those constant lines.

Also emitted: previous top-12 → top-12, previous top-12 → top-5, 2+ prior
top-12 → top-12 (WR/TE/QB equivalents).

### Career stage

From `years_experience` (0 = rookie year): `rookie` / `year_2` / `year_3` /
`year_4` / `year_5` / `year_6_plus` (≥5). **Missing exp is omitted, not
labeled rookie.** Stage × draft-capital × UI age-bucket cells emit only when
n > 0.

### Draft capital (descriptive)

Buckets: `round_1` / `day_2` / `day_3` / `undrafted`. **Missing capital is
omitted, not labeled UDFA.** Season-level P(top-12 / top-5 / top-24) and
absolute-bust rates, plus player-level cumulative windows once the calendar
window has closed (`draft_year + max_exp <= max warehouse season`):

- `top12_as_rookie` (exp 0)
- `top12_by_year_2` (exp 0–1)
- `top12_in_years_2_4` (exp 1–3)

The `undrafted` bucket is only players **explicitly** flagged UDFA (draft
round 0). Missing round stays excluded, so that bucket can be empty even
though many UDFAs appear in the warehouse. Do not backfill them.

### Phase 2 snapshot (2018–2025 warehouse, PPR)

Rebuilt aggregates: 4,647 appeared skill-position seasons. Prime windows are
data-derived (not 23–27):

| Pos | Known-age n | P(top-12) | Prime ages | % of top-12s in prime | P(top-12 \| prime) | Prev top-12 → top-12 | Engine BO among non-starters | R1 top-12 by year 2 |
|---|---|---|---|---|---|---|---|---|
| QB | 585 | 16% | 22–23 | 22% | 23% | 52% (n=83) | 9% | 32% (n=40, good) |
| RB | 957 | 10% | 25–27 | 40% | 11% | 40% (n=84) | 5% | 29% (n=15, moderate) |
| WR | 1,640 | 6% | 28–30 | 20% | 8% | 35% (n=84) | 4% | 13% (n=50, good) |
| TE | 866 | 10% | 28–30 | 24% | 14% | 42% (n=82) | 6% | 35% (n=11, low) |

Distribution vs conditional stay apart: 40% of RB1 seasons are ages 25–27,
but only 11% of age-25–27 RB seasons finish RB1. Age is missing for a
non-trivial slice (especially 2018); those rows are omitted from the age
columns above, not zero-filled. R1 TE-by-year-2 is a low-confidence cell —
raw + smoothed + n are still stored.

### I/O

`assemble_profile_aggregates` is pure (slim CI). `build_profiles.py` loads
`player_history_all.parquet`, writes
`cache/player_history/historical_profile_aggregates.json`. PPR-primary. No
ADP columns, no `projected_*`.

## Limitations (Phase 1–2)

- No comparable-player engine, no board UI columns, no ADP, no Pick Score
  change. Phase 2 rates are informational.
- 2016–2017 not in the parquet warehouse yet (Sleeper week files are on disk).
- 2023 usage_rows contain many null-name rows; identity join from
  `players_index` recovers current players, not all retirees.
- Legacy 2018–2022 names are nflverse short names (`T.Brady`); current index
  name wins when present.
- `historical_identity.py` is owner identity — player matching uses sleeper id
  plus `nflverse_metrics._gsis_to_sleeper` / usage_rows GSIS.
- Age coverage is thin in 2018 (retired players missing from the current
  `players_index`). Missing age/exp/capital must not be treated as 0 / rookie
  / UDFA.
- Snap % is not uniform before ~2022. Air yards / aDOT stay null until Phase 3.
- Ranking integration is gated on backtesting (Phase 9).

## Phase 1 warehouse snapshot

Rebuilt from committed usage_rows (skill-position players who appeared).
~4,650 player-seasons, 2018–2025. Finishes and prior-career features live on
the same parquet (no duplicate outcomes file in Phase 1).

Example (PPR positional finish, leakage-safe priors):

| Player | Season | Age | Exp | Draft | PPR pts | Pos finish | Prior finish | Prior top-12s | First-time top-12? |
|---|---|---|---|---|---|---|---|---|---|
| Bijan Robinson | 2023 | 21.5 | 0 | R1 | 246.3 | RB9 | — | 0 | yes |
| Bijan Robinson | 2024 | 22.5 | 1 | R1 | 341.7 | RB3 | RB9 | 1 | no |
| Jaxon Smith-Njigba | 2023 | 21.5 | 0 | R1 | 149.8 | WR48 | — | 0 | yes |
| Jaxon Smith-Njigba | 2024 | 22.5 | 1 | R1 | 253.0 | WR9 | WR48 | 0 | yes |
| CeeDee Lamb | 2022 | 23.4 | 2 | R1 | 301.6 | WR5 | WR19 | 0 | yes |
| CeeDee Lamb | 2023 | 24.3 | 3 | R1 | 403.2 | WR1 | WR5 | 1 | no |
| Sam LaPorta | 2023 | 22.6 | 0 | R2 | 239.3 | TE1 | — | 0 | yes |

`air_yards` / `adot` / `starts` are null until Phase 3.

## Rebuild

```bash
python -m data_building.historical.build_player_seasons
python -m data_building.historical.build_profiles
```

Writes `cache/player_history/player_history_{season}.parquet`,
`player_history_all.parquet`, `historical_coverage.json`, and
`historical_profile_aggregates.json`. Empty games=0 padding rows from
2023–2025 Sleeper dumps are dropped. Missing fields stay null; coverage JSON
reports present/missing per field. Profile JSON is the request-path artifact
for Phase 2 rates.

## Follow-up (gated)

3. Previous-season usage/efficiency (nflverse NGS/FTN) on this warehouse.
4. Comps + smoothed board probabilities (Postgres `031_*` if a request path
   needs it).
5–6. ADP snapshot preservation + multi-source backfill + hit rates.
7. Current projections as a **separate** signal; History vs Projection vs Market.
8. Compact board columns + lazy deep panel.
9+. League-winner proxy, walk-forward comparison via existing backtester, bounded
    Pick Score only if validated.

