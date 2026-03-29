# Changelog

All notable changes to the BR Fantasy Dashboard are recorded here.

---

## [Unreleased]

### Value Model — Red Zone & Age Curves
- **Red zone stats for past seasons** — Historical `rec_rz_tgt_pg` and `rush_rz_att_pg` are now propagated through `build_player_history_features()` as 3-year weighted averages (`three_year_weighted_rec_rz`, `three_year_weighted_rush_rz`, `rz_trend_1yr`). The value engine now falls back to these historical averages in the offseason when current-season data is absent, the same pattern used for rush yards.
- **TEs included in red zone fetch** — `ALLOWED_POS` in `sleeper_bulk_stats.py` was `["RB", "WR"]`; TEs are now included since they're among the most redzone-dependent positions.
- **Age curves updated** — WR peak moved from 25 → 26, QB from 28.5 → 29.5, TE from 26.5 → 27.5, reflecting recent NFL career arc data. RB unchanged at 23.5. Changes reduce systematic undervaluation of 26–28 year old WRs and QBs in their prime.
- **Historical parquet files rebuilt** — All three cached history files (2023, 2024, 2025) regenerated with the new redzone columns.

### Trade Calculator
- **Draft picks extended to rounds 4–5** — Pick value table previously capped at round 3; rounds 4 and 5 are now included from both FantasyCalc and DynastyProcess sources. Dynasty rebuilds that accumulate late-round capital now get meaningful values instead of 0.
- **Fuzzy player search** — The trade calculator search now uses a scored fuzzy matcher instead of a plain `.includes()` filter. Handles: exact substring (highest priority), multi-word initials ("jsn" → Jaxon Smith-Njigba), any-word-starts match, and single-character typo tolerance for queries ≥ 4 characters. Results are ranked by match quality.

### Projections
- **Silent failure fixed** — `load_week_projection()` no longer throws or returns `None`; it always returns a dict (empty if unavailable) and logs the failure. Projected scores showing as 0 without explanation is no longer possible.
- **"Projections unavailable" banner** — When no projection data can be loaded for a season/week, the weekly hub now shows a yellow warning banner instead of silently displaying 0-point projections.

### Value Model
- **SF value floor for non-QB players** — Non-QB players no longer drop in Superflex value relative to their 1QB value. The DP 2QB vendor blend was pulling pass-catchers down; their SF value is now floored at their 1QB value. QBs still receive the full Superflex premium. Fixed 295 players that were incorrectly discounted.
- **Per-league-size values (8, 10, 12, 14 teams)** — Player values now vary by league size. Larger leagues push more mid-tier players above the replacement line (higher value); smaller leagues concentrate value at the top. Fields `value_8`, `value_12`, `value_14` and their SF equivalents are populated using a position-rank scarcity model.
- **Elite cutoffs scale with league size** — The elite tier thresholds in `player_value.py` (QB4, RB12, WR18, TE5) now multiply by `num_teams / 10` so 8-team and 14-team leagues get proportionally correct elite cutoffs instead of hardcoded 10-team numbers.

### Trade Calculator
- **Scoring format selector** — PPR / Half-PPR / Standard toggle added to the trade calculator. Guest mode shows a radio group; logged-in users get their league's format auto-detected from scoring settings. Values adjust using position-based multipliers (Standard: RB +13%, WR −7%, TE −13%; Half-PPR: RB +6%, WR −3%, TE −6%).
- **Fix: Draft pick value was always 0 for named buckets** — `bucket_for_slot(slot, ...)` had a typo (`slot` instead of `slot_str`). Future picks with early/mid/late buckets now return correct values instead of silently returning 0.
- **Pick value table cached outside loop** — `load_pick_value_table()` was called once per pick inside `build_side()`. It is now loaded once before evaluation, eliminating redundant disk reads on trades with multiple picks.
- **Smarter fair-trade threshold** — The fairness band now scales with trade size: 10% for small trades (<300 value), 7% for mid-size, 5% for large trades (>600 value), with a 25-point absolute floor so tiny trades aren't hair-trigger unfair.
- **Multi-for-one adjustment caps raised** — The consolidation bonus cap increased from 60%/35% to 75%/50% (stud cap / side cap), allowing complex multi-piece trades to receive a more realistic bonus.
- **League size selector in guest mode** — Trade calculator now shows an 8/10/12/14-team selector for guest users; logged-in users inherit league size automatically.

### ESPN Support
- **ESPN league integration** — Added `espn_api.py` provider with `get_league_globals()`, `get_transactions()`, and `get_drafts()`. Platform abstraction layer routes Sleeper vs ESPN calls transparently.
- **ESPN globals sync** — Scoring settings, roster positions, and total rosters are now synced from ESPN leagues via `sync_league_globals()` before any page render.
- **ESPN empty-state messaging** — Activity and history pages show a clear message for ESPN leagues explaining which features require Sleeper data.

### Platform / API Fixes
- **Hardcoded `platform="sleeper"` removed** — `/api/history/ai-recap`, `/set-viewer`, and `/api/trade-eval` all read platform from the request instead of defaulting to Sleeper.
- **`/set-viewer` missing season variable fixed** — The endpoint previously referenced an undefined `season` variable; it now reads from the form.
- **Manual league refresh endpoint** — `POST /api/refresh-league` immediately expires the cached league context so the next page load rebuilds from source, without waiting up to 6 hours for TTL expiry.
- **`total_rosters` fallback** — When `get_total_rosters()` fails, the context now falls back to `len(rosters)` instead of leaving `total_rosters` absent. All previously-silent `except: pass` blocks now log errors.

### Performance
- **Parallel transaction fetching** — `get_transactions_by_week()` now uses `ThreadPoolExecutor` (up to 8 workers) instead of fetching weeks serially, significantly reducing activity page load time mid-season.
- **Model value table cache** — Teams, trade, and offseason dashboard page refreshes now check `ctx.get("model_value_table")` before re-reading from disk via `load_model_value_table()`.

### Code Quality
- **`_build_roster_map()` helper extracted** — Identical 10-line roster map construction blocks in `build_league_context` and `refresh_league_ctx_section` replaced with a single shared helper.
- **Scoring format flows end-to-end** — Scoring format is now read from league context in `page_trade`, passed into `build_trade_calculator_body`, emitted as a hidden input for logged-in users, and sent in both trade-eval API payloads.
