# Historical Analytics

Phase 1–9 of the Draft Cheat Sheet historical layer: a reusable
player-season warehouse, positional finishes, leakage-safe prior-career
features, descriptive age / career-stage / draft-capital rates,
previous-season usage/efficiency (NGS + snaps), comparable-player cells
with smoothed board probabilities, frozen preseason ADP with hit rates,
History vs Projection vs Market as three separate live signals, a
**compact Hist column plus a lazy similar-player panel** on the cheat
sheet, league-winner labels, and a **walk-forward honesty check** of
History P vs Market P. Pick Score is unchanged: the gate did not pass.

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
| `static/draft_board_core.js`, `static/pick_score.js` | Shared board/Pick Score kernels — **untouched** (Hist is display-only) |
| `dashboard_services/pages/draft_room_page.py`, `static/draft_room.js` | Draft Room UI — **untouched**; `/api/league-players` may carry `historical` |

Board data is batch (`/api/league-players`). Compact Hist fields ride that
payload. A per-player REST endpoint (`/api/historical-player/<id>`) is for
the deep modal only. Core analytical logic must not live in JS.

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
| `nflverse_metrics` NGS / FTN / EPA | NGS **2016+**, FTN **2022+** | Phase 3 joins NGS receiving/passing/rushing onto the warehouse via committed `nflverse_metrics_{season}.json` (same column names as `build_ngs_*`). FTN/PBP EPA is wired but **not cached** here (`nfl_data_py` unavailable in this environment; PBP dumps are heavy). |
| `pfr_snap_counts.fetch_season_snap_counts` | nflverse snap counts | Name-keyed, used by `sleeper_usage.py`. Phase 3 does **not** call it (and does not use `estimate_snap_share_from_usage`). Snaps join `pfr_player_id` → sleeper via the same DynastyProcess ids file `nfl_data_py.import_ids` uses, cached as `snap_counts_{season}.json`. |
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
| `breakout_engine/train_hit_probability.py` + `hit_probability_model.json` | trained 2022–2024 | Calibrated hit model. Phase 9 does **not** call this; it reuses the walk-forward *pattern* (train < S, test S) with warehouse finishes. |
| `_load_historical_fantasy_rankings` | prefers `fantasy_rankings_{season}.json` (absent); falls back to usage proxy | Phase 1 **replaces the proxy** with real total-points positional finishes on the warehouse. Phase 9 scores those finishes — it does not call this helper. |

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

Phase 5–6 **reuse** this store: completed-season snapshots are frozen so
cron cannot overwrite a preseason board, and usable historical maps are
committed under `cache/player_history/adp/`. Superflex / TEP historical
ADP still does not exist in free sources.

### Projections (Phase 7; live signal, not a warehouse column)

| Dataset / function | Notes |
|---|---|
| `data_building/fetch_projections.py` | Sleeper-only; current/upcoming season |
| `utils/projection_resolver.py` + tests | Current-year signal. Callers of `compare_board_signals` must pass this PPG. |
| `breakout_engine/projections.py` | Live engine |
| `dashboard_services/historical/signals.py` | History vs Projection vs Market. Does not fetch. |

Canonical season rows have **no** `projected_*` columns. Tests lock that.
Sleeper has no honest historical preseason projection archive, so Phase 7
does **not** backfill `P(hit | projected WR1)`. PPG is never converted into
a fake probability.

## Architecture (slim-CI split)

```
dashboard_services/historical/     # pure logic — pytest -m "not integration"
    definitions.py                 # tiers, age-as-of, buckets, bust, smoothing
    seasons.py                     # canonicalize both usage_rows schemas
    finishes.py                    # ranks + prior-career features
    finish_rates.py                # cohort rates, sample size, Bayes, confidence
    age_curves.py                  # distribution vs conditional; data-derived prime
    career_profiles.py             # stage / repeat / engine-breakout / capital
    usage.py                       # overlay, prior-usage features, usage rates
    comps.py                       # pre-season matching, cells, board lookup
    adp.py                         # normalize 999→None, ADP hit rates, ADP-bust
    signals.py                     # History vs Projection vs Market (no blend)
    filters.py                     # shared Scout / cohort predicates (OR/AND)
    cohorts.py                     # compact observation index + combined rates
    board.py                       # compact board payload + deep-panel lookup
    aggregates_store.py            # JSON load + mtime cache (no pandas/Flask)
data_building/historical/          # pandas / parquet I/O
    build_player_seasons.py        # rebuild warehouse from cache
    build_outcomes.py              # finishes applied in the same rebuild
    build_profiles.py              # parquet → historical_profile_aggregates.json
    build_usage_efficiency.py      # NGS/snap cache refresh + overlay loader
    build_adp.py                   # frozen ADP backfill + join onto profiles
data_building/external_data/player_history.py   # existing paths + new wrappers
```

`definitions.py` / `seasons.py` / `finishes.py` / `finish_rates.py` /
`age_curves.py` / `career_profiles.py` / `usage.py` / `comps.py` / `adp.py`
/ `signals.py` / `filters.py` / `cohorts.py` / `board.py` must not import pandas, Flask, or `nfl_data_py`.
`aggregates_store.py` reads the JSON artifact (pathlib + json only).
`test_product_honesty.py` and `test_adp_formats.py` stay green: this package is
not on their import graph.

## Combined cohorts, shrinkage, and Wilson intervals

Selected Trends buckets are evaluated against a compact in-memory observation
index (`cohort_index` in `historical_profile_aggregates.json`, or the sibling
`cohort_index.json` overlay). Hits are counted from actual matching
player-seasons. Intersections are never estimated by multiplying bucket
probabilities.

Same-group filters OR; different groups AND. Scout (current-board matching)
uses the same predicates in `filters.py`.

Display rates on the selected-profile card are **raw** observed percents.
Ranking / selected-profile **adjusted_edge** uses empirical-Bayes shrinkage
with prior n=30 toward the positional baseline:

`adjusted_rate = (successes + baseline * 30) / (n + 30)`

Table cells still use prior n=10 (`DEFAULT_BAYES_PRIOR_N`). Top 10 edges and
Historical red flags rank by `adjusted_edge`, not raw lift.

Wilson 95% intervals (`ci_low` / `ci_high`) appear on the selected multi-factor
cohort, expanded Hist, and the selected trend row only.

Market-adjusted cohort edge is Top-12 only: for each matched observation with
a historical preseason ADP bucket, look up the existing ADP-bucket P(top-12),
average those probabilities, and compare to the observed raw rate. If coverage
is thin (`n_with_adp < 15` or share `< 0.40`), the field is omitted rather than
faked.

Year-over-year trajectory buckets (target share +5 pts, snap +15 pts, workload
cliffs) use two consecutive **prior** seasons only. Outcome year S never
enters those deltas. Missing change is omitted, never 0.

`POST /api/historical-cohort` reads the compact index. It does not scan parquet
and does not send the warehouse to the browser. Historical outputs remain
excluded from Pick Score, Recommendation, VOR, Draft Grade, and Cheat Sheet
ranking.

Persistence: large per-season rows stay in committed parquet under
`cache/player_history/`. Precomputed **profile aggregates** are a small JSON
file (`historical_profile_aggregates.json`) rebuilt in cron after the
warehouse. Request paths must not scan parquet row-by-row. No Postgres table
yet (next migration remains `031_*` when a request path needs it). Phase 4
board probabilities, Phase 6 ADP rates, and Phase 7 live-signal comparison
are JSON lookups plus caller-supplied current PPG/ADP, not a new table.

Rebuild:

```bash
python -m data_building.historical.build_usage_efficiency  # optional if JSON caches exist
python -m data_building.historical.build_adp               # optional if ADP caches exist
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
| Air yards / aDOT | NGS **2016+** where a receiving summary row exists (~20–23% of appeared seasons) | `adot` from NGS intended air yards; `air_yards` = adot × targets when total air yards are absent. |
| Snap % / snaps | stored **2018+** from nflverse PFR snaps + legacy `snap_share`; **rates use 2022+** priors (`SNAP_RELIABLE_FLOOR`) | 0% + real volume → `None`. Not estimated from touches. Live `avg_off_snap_pct` is **not** overwritten. |
| NGS separation / YAC / CPOE / RYOE | **2016+** on NGS-covered players | Same names as `nflverse_metrics`. |
| FTN drop / contested | **2022+** when cached | **Not in this cache** (needs FTN+PBP join). Overlay will fill if JSON later includes `drop_rate`. |
| Starts | **not filled** | No reliable starts column in the NGS/snap files used here. |
| Age as-of Sept 1 | players with `bDay` (~5.6k in current index) | Retired players missing from index → `None` unless birth date is on the usage row. |
| Draft year | players_index + draft_history | |
| Draft round / pick | `draft_history.parquet` (nflverse) | |
| GSIS | 2018–22 usage_rows; 2023+ only if that sleeper id appeared in a legacy file | Full `import_ids` join is Phase 1-compatible but optional. Overlay joins NGS/snaps by sleeper id from the ids file, so 2023+ NGS does not require GSIS on the warehouse row. |
| Historical ADP | **redraft PPR 1QB**, frozen snapshots **2018–2025** | Sleeper `adp_ppr` 2020+ (empty 2018–19); MFL PPR 12-team real drafts 2018+ (selected-only ≥25%); ESPN only when the board passes a preseason quality gate (2021/22/25 fail the “170 wall”). Yahoo has **no season axis**. Sleeper 999 → missing. |
| SF / TEP historical ADP | **does not exist** in free sources | Not claimed. Sleeper `adp_2qb` is stored on the Sleeper snapshot but is not blended into 1QB hit rates. |
| Current-season projections | live Sleeper only (Phase 7) | Not stored on warehouse rows. No historical preseason backfill. |
| Season-long projected team offense | **2016–2026** nflverse `spread_line` + `total_line` on regular-season games | Rank 1 = highest average implied total among games with a line. Overlay `team_offense_overlay.json`. Request path does not import `nfl_data_py`. Live 2026 uses every posted line, not week 1 only. |
| Extra 2016–17 cohort seasons | skill-position PPR finishes from nflverse seasonal + rosters | Stamped onto the cohort index at load for Trends/Hist only. Not a full warehouse rebuild. |

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

## Phase 3 — previous-season usage / efficiency

Same-season NGS and snap values are **outcomes** of that season (like points).
Hit rates that claim a usage → finish relationship use only
**previous-season** fields (`previous_season_target_share`,
`previous_season_snap_pct`, `previous_season_adot`, …). Mutating 2024
actuals must not change 2024 prior-usage features.

Two statistics, same rule as age:

1. **Distribution** — among this-season WR1s, what share had 80%+ snaps last year.
2. **Conditional** — among WRs who had 80%+ snaps last year, what share finished WR1.

Missing previous usage is omitted from those cohorts, not bucketed as 0.
Snap **rates** require `previous_season_year >= 2022`. NGS aDOT/RYOE use
2016+. Target-share rates are WR/RB/TE only (QBs are not a target-share
cohort). Estimated snaps from touches are never used. Overlay does not
rewrite live-valuation `avg_*` columns.

### Phase 3 snapshot (2018–2025 warehouse, PPR)

NGS receiving summaries cover ~20–23% of appeared seasons (aDOT / air yards /
separation). Snap % is present on nearly all appeared skill rows after the
PFR join. FTN drop/contested is **empty** in this cache.

| Signal | Example | Conditional P(top-12) | Share of top-12s | n / confidence |
|---|---|---|---|---|
| WR previous target share 25%+ | vs &lt;10% | 35% vs 1% | 36% vs 5% of WR1s | 61 good / 575 strong |
| WR previous snap 80%+ | vs &lt;40% | 19% vs 0% | **61%** of WR1s vs 0% | 97 good / 224 strong |
| WR previous aDOT 10–13 | vs 13+ | 16% vs 8% | 52% vs 14% of WR1s | 240 / 144 strong |
| RB previous RYOE ≥ 0 | vs below expected | 26% vs 14% | 71% vs 29% of RB1s | 194 / 145 strong |

61% of WR1 seasons had 80%+ snaps the year before, but only 19% of those
high-snap WR seasons finished WR1 — the two stats stay apart.

### I/O

`python -m data_building.historical.build_usage_efficiency` refreshes
`nflverse_metrics_{season}.json` and `snap_counts_{season}.json` from nflverse
GitHub parquets. Warehouse rebuild is still cache-only. Cron does not fetch
NGS live.

## Phase 4 — comps + smoothed board probabilities

Comparable-player matching is a **pre-season profile lookup**, not a
learned similarity model and not a ranking input. The board will (Phase 8)
show smoothed P(top-5 / top-12 / top-24) from historical player-seasons
that looked like this *before* the season started.

Matching dimensions (missing omitted, never faked as 0 / UDFA / last-place):

| Dimension | Source | Notes |
|---|---|---|
| position | warehouse | required |
| career_stage | `years_experience` | missing exp omitted, not rookie |
| draft_capital | `draft_capital_bucket` | missing omitted, not UDFA |
| prior_finish | `previous_season_finish` | rookies (exp 0) use explicit `none`; veterans with a missing prior skip this dim. Rank 13 is `top_24` |
| age_bucket | age as of Sept 1 | missing age omitted |
| target_share | **previous-season** share | WR/RB/TE only; QBs skip |
| snap_pct | **previous-season** snap % | only when `previous_season_year >= 2022` |

Same-season points, finishes, snaps, NGS, ADP, and projections are **not**
matching features. Mutating 2024 actuals does not change a 2024 comp key.

Two products stay distinct:

1. **Conditional board probabilities** — P(this-season hit | profile).
   Stored as finest-grain *leaves* (one per unique present-dimension
   signature) plus position baselines. `lookup_board_probabilities` pools
matching leaves and walks `COMP_RELAXATION_ORDER` (`target_share` →
`snap_pct` → `age_bucket` → `draft_capital` → `career_stage` →
`prior_finish`; position is never dropped) until `n >= 15` (walk-forward /
default). The live Hist column and modal use `HIST_PANEL_MIN_N = 1` so an
exact R1 rookie cell is not mixed with every rookie. Those tiny exact
cells still empirical-Bayes shrink (`DEFAULT_BAYES_PRIOR_N = 10`), but
the prior is the nearest parent cell with n ≥ 15 (e.g. last-year top-5
RBs), not the all-appeared position baseline. Empty cells keep
`raw_rate=None`.
2. **Named comps** — a few example player-seasons from the matched cell
   (hits first, then PPR points). The query player is excluded. A
   historical query with `as_of_season` cannot use later seasons.
   `match_comps` is the in-memory primitive; the request path reads
   examples off the JSON leaves and must not scan parquet.

Rates are **pooled historical** (all warehouse seasons) on the live board.
Walk-forward of the same lookups is Phase 9 (`walkforward` on the JSON).
Descriptive only — comps do not enter ranking or Pick Score.

No `031_*` migration. The request-path artifact remains
`historical_profile_aggregates.json` (`phase: 9`, `comps` + `adp` + `signals`
+ `board` + `preseason_profiles` + `walkforward`). Live PPG is supplied by
the caller, not stored on warehouse rows.

### Phase 4 snapshot (2018–2025 warehouse, PPR)

1,676 leaves, ~4,650 seasons, JSON ≈ 2 MB. Position-wide P(top-12) is low
because the warehouse includes every appeared skill player (WR ≈ 5%,
RB ≈ 8%). Profile cells move that number.

| Profile (pre-season) | Fallback | n / conf | raw P(top-12) | smoothed | Named examples |
|---|---|---|---|---|---|
| Rookie WR, R1, age ≤22 | none | 30 moderate | 17% | **14%** | J.Chase 2021 WR5; Brian Thomas Jr. 2024 WR4; J.Jefferson 2020 WR6 |
| Year-2 RB who was RB1 last year | dropped usage/age/capital/stage | 49 good | 43% | **37%** | CMC 2023 RB1; A.Kamara 2020 RB1; J.Taylor 2021 RB1 |
| Year-6+ WR who was top-5 last year | dropped usage/age/capital | 17 moderate | 59% | **39%** | Tyreek Hill 2023 WR2; D.Adams 2021 WR2 |
| Year-5 day-3 WR, prior outside 36 | dropped usage | 38 moderate | 0% | **1%** | Jauan Jennings 2024 WR24 |

Tiny exact cells on the live Hist surface keep the profile and shrink
toward a parent cell (last-year top-5 RBs), not every appeared RB.
Walk-forward still relaxes to n ≥ 15, then shrinks toward the position
baseline. 0% raw with n>0 is a real zero and still smooths up slightly
(the day-3 cell). Request-path lookup passes `sleeper_id` so a player is
not listed as their own comp.

## Phase 5–6 — frozen ADP + hit rates

ADP is a **preseason market** signal, kept separate from comps (profile) and
from current projections (Phase 7). It does not enter ranking or Pick Score.

**Preservation (Phase 5).** `write_adp_snapshot(..., frozen=True)` refuses later
overwrites. Cron freezes Yahoo/ESPN/MFL for `current_season - 1` before
refreshing the live season. Empty fetches still cannot clobber last-good
data. Historical copies live under `cache/player_history/adp/` (committed);
`data/adp_snapshots/` stays the gitignored live resolver disk.

**Backfill (Phase 6).** Reuses `fetch_sleeper_adp`, `fetch_mfl_adp`
(`is_mock=0`, `MIN_MFL_DRAFT_PCT=25`), and `fetch_espn_global_adp`. Yahoo is
not backfilled. ESPN boards that fail `source_map_is_usable` (no first-round
picks, or the 12th player later than pick 36) are stored as `usable: false`
and skipped at join time.

One overall ADP per player-season, source order **sleeper → mfl → espn →
yahoo**. Positional ADP is a competition rank among that season’s position
mates who have ADP (missing stays `None`, not last place).

Two statistics, same split as age/usage:

1. **Distribution** — among this-season WR1s, what share were ADP round 1.
2. **Conditional** — among ADP round-1 WRs, what share finished WR1.

ADP-relative bust: drafted inside positional ADP top-12 and finished
outside. Missing ADP is not a bust. Absolute bust (NFL draft capital)
stays a separate definition.

### Phase 6 snapshot (2018–2025 warehouse, redraft PPR 1QB)

Coverage is uneven: 2019 is MFL-only (168 of 547 appeared skill rows); 2024–25
are nearly complete via Sleeper. 3,088 of 4,647 seasons have an ADP.

| Signal | Conditional P(top-12) | Share of top-12s | n / confidence |
|---|---|---|---|
| WR ADP round 1 vs rounds 11+ | **58%** vs 1% | 27% of WR1s vs 8% | 36 moderate / 795 strong |
| WR ADP positional top-12 vs outside 36 | **36%** vs 1% | 24% of WR1s | 56 good / 913 strong |
| RB ADP round 1 vs rounds 11+ | **57%** vs 1% | 38% of RB1s | 55 good / 492 strong |
| WR ADP-relative bust (ADP WR1 → outside WR1) | **46%** | — | 96 good |

Round-1 TE/QB cells are tiny (those positions rarely go in the first 12
picks); use positional ADP top-12 there. Comps keys still have no ADP
dimension.

## Phase 7 — History vs Projection vs Market

Three live signals, compared in **native units**. Descriptive only — nothing
here enters ranking or Pick Score. Warehouse parquet still has no
`projected_*` columns. Comps matching still ignores projections and ADP.

| Signal | Source | Unit |
|---|---|---|
| **History** | `lookup_board_probabilities` on JSON comps leaves | `P(top-12)` (smoothed) |
| **Market** | current overall ADP → Phase 6 `by_overall_bucket` conditional rate | `P(top-12)` (smoothed) |
| **Projection** | caller-supplied current Sleeper PPG (`utils.projection_resolver`) | PPG + implied positional rank among the live projected field |

`compare_board_signals` is the request-path primitive: one batch over the
board so implied ranks are well-defined. It does not scan parquet, fetch
Sleeper, or invent a blended score (`blended_score` is always `None`).

Rules:

- Missing projection / ADP / empty rate cell → `unknown`, not 0% or last place.
- Board `ppg` / warehouse `ppr_ppg` / season `projected_points` are **not**
  read as a projection. Only `projected_ppg` / `proj_ppg`.
- Sleeper ADP `999` is missing market, same as Phase 6.
- History vs market: `|ΔP| < 0.10` is aligned; otherwise `history_higher` /
  `market_higher`.
- Projection vs market: `|Δ rank| ≤ 6` is aligned (competition rank, 1/2/2/4).
- Projection vs history is **qualitative** (`history_skeptical` /
  `history_bullish` / `agree_hit` / `agree_miss`). PPG is not turned into a
  fake `P(top-12)`. Sleeper cannot backfill historical preseason projections,
  so there is no honest `P(hit | projected WR1)` table.

Phase 8 attaches a compact slice of this payload onto `/api/league-players`.
This phase does not change Pick Score.

## Phase 8 — compact Hist column + lazy deep panel

Display only. VOR order and Pick Score are unchanged. JS does not compute
rates; it renders `player.historical` from the batch payload.

- **Batch:** `_league_players_response` stamps a compact `historical` dict
  on skill-position players after projections and ADP overlays, then
  jsonifies. Fields: `p_hit` / `p_hit_pct`, `mkt_p` / `mkt_pct`, `h_vs_m`,
  `proj_rk`, `adp_rk`, `p_vs_m`, `p_vs_h`. Missing stays `null` / `unknown`.
- **Preseason profiles** live on the JSON (`preseason_profiles.by_player`).
  Each player's last warehouse season supplies previous finish/usage; years
  experience and Sept-1 age step forward to `upcoming_season`. Request paths
  do not scan parquet.
- **Cheat sheet:** a Hist column (percent + disagreement tint) with an `i`
  button. Row tap still crosses a player off. The button opens a lazy panel.
- **Deep panel:** `GET /api/historical-player/<id>` returns named comps,
  dropped dimensions, and market bucket rate from JSON leaves.
- **ADP axis** remains redraft 1QB. Superflex / TEP historical ADP is not
  claimed. Dynasty `avg_pick` is not used as market ADP.
- Draft Room UI is unchanged; it may ignore the extra payload field.

## Phase 9 — league-winner proxy + walk-forward gate

Honesty check, not a second backtester. Live Hist stays pooled and
display-only. `pick_score.js`, `utils/pick_score.py`, and
`draft_board_core.js` are untouched.

**Labels** (warehouse positional finish, PPR totals):

| Label | Definition |
|---|---|
| League winner | this-season finish ≤ 5 (`TIER_CUTOFFS['top_5']` — no new cutoff) |
| League-winner smash | that finish **and** prior finish is None or **> 12**. Rank 13 last year finishing top-5 *is* a smash. That is not the engine non-starter cut (prior > 13). |
| Engine breakout / first-time elite | unchanged from Phase 2 |

**Walk-forward.** `dashboard_services/historical/walkforward.py` rebuilds
comps (no named examples) and ADP hit rates on seasons **< S**, then scores
season **S** with the existing `lookup_board_probabilities` /
`lookup_market_probability`. Outcomes are warehouse finishes, not the
breakout engine's usage-points proxy, and not `BreakoutEngine`. Missing P
is skipped, never 0. Test seasons **2021–2025** (2018–19 ADP is too thin).
History P for league-winner is comps `P(top-5)`; for top-12 / breakout it
is comps `P(top-12)`. Market P is the matching ADP-bucket rate rebuilt on
train. Position baseline is the train P(hit | position). Metrics are
Mann–Whitney AUC and Brier (no sklearn).

**Pick Score gate** (all must hold on `league_winner`): ≥3 test seasons,
≥50 scored player-seasons with both P per season, ≥200 History-P rows
total, hist AUC ≥ 0.55, and hist AUC ≥ market AUC + 0.02. Even a passing
gate does not mutate live ranking in this phase
(`pick_score_in_live_ranking: false`).

### Phase 9 snapshot (2018–2025 warehouse, PPR)

5 folds, 2,975 test player-seasons, 2,152 with both History and Market P.
JSON ≈ 2.48 MB. Gate **failed** (0/3 seasons beat market by 0.02).
`pick_score.validated = false`. Mean league-winner AUC: History **0.825**,
Market **0.878**, position baseline ~0.60. History is real information vs
a position prior; ADP is still the better ranking of league winners.

| Test S | n train / test / both | Hist AUC (LW) | Mkt AUC (LW) | Margin | Hist AUC (top-12) | Mkt AUC (top-12) |
|---|---|---|---|---|---|---|
| 2021 | 1672 / 595 / 332 | 0.701 | 0.896 | −0.195 | 0.687 | 0.861 |
| 2022 | 2267 / 574 / 269 | 0.906 | 0.904 | +0.001 | 0.904 | 0.809 |
| 2023 | 2841 / 598 / 389 | 0.797 | 0.822 | −0.025 | 0.767 | 0.817 |
| 2024 | 3439 / 599 / 558 | 0.883 | 0.909 | −0.026 | 0.849 | 0.888 |
| 2025 | 4038 / 609 / 604 | 0.839 | 0.860 | −0.021 | 0.865 | 0.888 |

2022 is a near-tie on league-winner and the one season where History P
beats market on top-12. That is not three seasons with a 0.02 margin.
Smash / engine-breakout / first-time-elite AUCs are lower for both
signals (rarer positives); market still leads on mean AUC.

No `031_*` table. Request path is still the JSON artifact.

## Limitations (Phase 1–9)

- No Pick Score change. Phase 2–9 rates, the Hist column, and the
  walk-forward verdict are informational. Comps, ADP, and live projections
  do not enter ranking. The gate requires History P to beat Market P; it
  did not.
- Live comp cell rates and ADP hit rates stay pooled historical. Walk-forward
  is the validation section on the JSON, not a replacement of board leaves.
- Current projections are a live signal only. There is no historical
  preseason projection archive, so projection vs history cannot be a
  calibrated probability.
- Superflex / TEP historical ADP does not exist in free sources.
- Yahoo public ADP has no season parameter; it is current-only.
- ESPN 2021/2022/2025 boards fail the preseason quality gate and are unused.
- Sleeper ADP is empty for 2018–2019; MFL 2018 mapping is thin (~149 ids).
- 2016–2017 not in the parquet warehouse yet (Sleeper week files are on disk).
- 2023 usage_rows contain many null-name rows; identity join from
  `players_index` recovers current players, not all retirees.
- Legacy 2018–2022 names are nflverse short names (`T.Brady`); current index
  name wins when present.
- `historical_identity.py` is owner identity — player matching uses sleeper id
  plus `nflverse_metrics._gsis_to_sleeper` / usage_rows GSIS.
- Age coverage is thin in 2018 (retired players missing from the current
  `players_index`). Missing age/exp/capital/usage must not be treated as 0 /
  rookie / UDFA.
- Snap **rates** are 2022+ priors even though snap *values* are stored 2018+.
- FTN drop/contested and PBP EPA are not in the committed cache (`nfl_data_py`
  could not be installed here; PBP dumps are heavy). `starts` stay null.
- Ranking integration is gated on walk-forward beating the market (Phase 9
  ran; gate failed).

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

`air_yards` / `adot` are filled from NGS when a receiving summary exists.
`starts` stay null. JSN 2024: snap 86%, aDOT 8.8 (previous-season snap 65%).

## Rebuild

```bash
python -m data_building.historical.build_usage_efficiency
python -m data_building.historical.build_adp
python -m data_building.historical.build_player_seasons
python -m data_building.historical.build_profiles
```

Writes `cache/player_history/player_history_{season}.parquet`,
`player_history_all.parquet`, `historical_coverage.json`,
`nflverse_metrics_{season}.json`, `snap_counts_{season}.json`,
`adp/{source}_redraft_{season}.json`, `historical_adp_coverage.json`, and
`historical_profile_aggregates.json`. Empty games=0 padding rows from
2023–2025 Sleeper dumps are dropped. Missing fields stay null; coverage JSON
reports present/missing per field. Profile JSON is the request-path artifact
for Phase 2–9 rates (comp leaves + ADP hit rates + signal contract +
preseason profiles for the board + walk-forward verdict). The usage-efficiency and
ADP backfills are optional when those JSON caches are already committed;
warehouse rebuild does not call nflverse or ADP APIs live. Live projections
are not written into this JSON.

## Follow-up (gated)

- Bounded Pick Score term **only if** a later walk-forward passes the gate
  (History P beating Market P with the documented margins). Do not add Hist
  to VOR / Recommendation / `decisionScore` on the same evidence.
- Postgres `031_*` only if a request path outgrows the JSON artifact.

