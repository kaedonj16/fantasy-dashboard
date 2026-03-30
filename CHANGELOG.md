# Changelog

All notable changes to the BR Fantasy Dashboard are recorded here.

---

## [Unreleased]

### Offseason Breakout Detection System
- **Roster change tracking** — New `roster_changes` table tracks player departures (free agent, trade, retirement, cut) with previous season usage stats (targets, carries, snap share, opportunity share). Enables detection of vacated opportunity before season starts.
- **Vacated opportunity calculation** — `vacated_opportunity` table aggregates targets/carries/snaps left behind per team/position. Example: "TB WR has 140 vacated targets from Mike Evans departure". Function: `calculate_vacated_opportunity()`.
- **Opportunity redistribution projection** — `projected_opportunity` table projects how vacated opportunity distributes to remaining players. Uses proportional allocation based on previous usage. Projects target increase, snap share increase, and assigns offseason breakout scores.
- **Offseason breakout scoring** — 5-factor system (0-100 points): absolute opportunity increase (0-30), relative increase % (0-25), team vacancy size (0-20), youth/experience bonus (0-15), established role bonus (0-10). Threshold: 30+ points. Example: Egbuka scores 84.9 (WR2→WR1 after Evans departure).
- **Auto-detection via season comparison** — `detect_roster_changes_between_seasons()` compares player teams in current vs previous season usage tables to identify departures. Enriches with usage stats automatically.
- **Manual roster change entry** — `manual_add_roster_change()` allows adding high-profile moves (FA signings, trades) with player name lookup and usage stat retrieval.
- **API endpoint** — `/api/offseason-breakout-candidates` returns ranked list with projection details, departed players context, and score breakdown. Filters: season, min_score, position.
- **Season-aware player indicators** — `/api/player-indicators` switches breakout detection mode based on NFL season type. Offseason: uses roster change projections. In-season: uses performance-based metrics. Seamless transition.
- **Population script** — `populate_roster_changes.py` runs full pipeline: detect changes → calculate vacated opp → project redistribution. Command: `python populate_roster_changes.py 2025`.

### Home Page UX
- **Changelog sidebar** — Recent updates now appear in a compact sidebar on the home page instead of a large centered section. Sidebar is sticky on desktop, stacks below features on mobile. Displays 5 most recent user-facing updates from `dashboard_services/changelog.py` with color-coded tags (feature/improvement/new).
- **User-facing changelog separation** — Technical CHANGELOG.md tracks implementation details; `dashboard_services/changelog.py` contains user-friendly update descriptions without formulas or internal architecture mentions.

### Trade Calculator UX Improvements
- **Real-time value change indicators** — Player chips now display 7-day value deltas (+15, -8) with color coding (green for risers, red for fallers). Only shown for changes ≥1 point.
- **Shareable trade links** — New share button (🔗) generates URL-encoded trade links. Copy to clipboard with one click; trades auto-load from shared URLs with fallback to localStorage for personal saves.
- **Rookie & breakout badges** — Players identified as rookies show a blue "ROOKIE" badge; players with +50 value in 7 days get an orange "🔥 BREAKOUT" badge. Appears in trade chips, dropdowns, and value list.
- **1QB ↔ Superflex toggle** — League type control redesigned as a pill toggle switch between 1QB and SF modes for quicker switching.
- **Dropdown controls** — Team size (8/10/12/14) and scoring format (PPR/Half/STD) now always visible as compact dropdowns instead of hidden for logged-in users.

### Top Movers Enhancements
- **Loading states & animations** — Movers panel shows spinner during load with reduced opacity. Pulse highlight animation when data refreshes.
- **Debounced refresh** — 300ms debounce on control changes prevents excessive API calls when rapidly toggling settings.
- **Data freshness indicators** — Movers subtitle shows when values were last updated ("Updated 2h ago", "Updated 3d ago") based on database snapshot timestamps.
- **League size-aware movers** — Top risers/fallers now respect selected league size (8/10/12/14-team) and display in subtitle (e.g., "Biggest 7-day changes in SF 12-team BR value").

### Database & Performance
- **Value history for all league sizes** — `player_value_history` table now stores `value_8`, `value_12`, `value_14`, `sf_value_8`, `sf_value_12`, `sf_value_14` fields. Historical tracking works across all league configurations.
- **Database fallback for ephemeral filesystems** — `load_model_value_table()` now falls back to loading from database when JSON file doesn't exist, solving Render's ephemeral filesystem issue where cron_daily writes aren't accessible to the main app.
- **Performance indexes** — Added 4 new indexes on `player_value_history`: `(as_of_date, value DESC)`, `(as_of_date, sf_value DESC)`, `(player_id, position)`, `(source, as_of_date DESC)` for faster movers queries.

### Infrastructure
- **ads.txt support** — Created `/ads.txt` endpoint for ad network authorization (Google AdSense, Media.net). Template file included with instructions for adding publisher credentials.

### Advanced Metrics & Breakout Detection
- **Advanced efficiency metrics table** — New `player_advanced_metrics` database table stores position-specific efficiency calculations: yards per target, catch rate, yards per reception (WR/TE/RB), yards per carry, yards per touch (RB), yards per attempt, completion %, TD/INT rates (QB), plus snap share, opportunity share, red zone usage, and composite role scores (0-100).
- **Daily metrics calculation** — `cron_daily.py` now runs `build_daily_advanced_metrics()` after usage table generation to calculate and store efficiency metrics for all players daily.
- **Offseason metrics handling** — Value model automatically uses most recent available metrics (from previous season) when in offseason. Logs indicate when using historical data (e.g., "Using metrics from 2025-12-30 (90 days old - likely previous season)"). Efficiency metrics remain relevant across seasons since player skill persists.
- **Multi-factor breakout algorithm** — Breakout detection upgraded from simple +50 value threshold to composite scoring system analyzing: snap share increase (0-25 pts), opportunity share increase (0-30 pts), role score improvement (0-25 pts), efficiency gains (0-20 pts), red zone usage increase (0-15 pts), and age bonus for players under 26 (0-15 pts). Requires 30+ total score to qualify.
- **Year-over-year breakout factors** — Added YoY snap increase (0-20 pts), YoY opportunity increase (0-25 pts), and second-year player bonus (10 pts) to capture depth chart promotions (WR2→WR1 when starter leaves), sophomore leaps (Jefferson, Chase, Lamb pattern), and situation changes. Uses `get_year_over_year_metrics()` to query metrics from ~365 days ago (±30 day window) for comparison. Dual-timeframe approach: 14-day trends identify hot hands, YoY comparisons identify structural advantages.
- **Improved breakout thresholds** — Simple value-based fallback now uses ≥75 for QB/RB/WR, ≥100 for TE (reduced from uniform ≥50). Eliminates false positives: previous threshold flagged 858 players as breakouts, new threshold identifies 6 meaningful candidates.
- **Role score calculation** — Position-specific composite metric combining usage volume (snaps, touches), efficiency (yards per touch, catch rate), and opportunity quality (red zone usage). Weights tailored per position (e.g., pass-catching RBs valued higher, WR target volume weighted more than snap count).
- **Trend tracking** — `usage_trend` and `efficiency_trend` fields calculate 14-day % changes in opportunity share and role score, identifying emerging usage patterns before value changes.
- **Advanced metrics API endpoints** — Three new endpoints: `/api/player-advanced-metrics/<id>` (individual player efficiency stats), `/api/advanced-metrics/top-role-players` (highest usage+efficiency composite scores by position), `/api/advanced-metrics/breakout-candidates` (multi-factor breakout analysis with score breakdowns).
- **Breakout candidate scoring transparency** — Breakout API returns score components breakdown showing exactly why a player qualified (snap increase 15.2 pts, opportunity increase 22.5 pts, youth bonus 9.0 pts, etc.) along with current/previous role scores and value delta for context.
- **Value model integration** — Advanced efficiency metrics now feed directly into the ML value model as training features. The gradient boosting model learns patterns like "high YPT + increasing snap share = rising value" and "declining efficiency trend = value risk". Metrics joined into both training and inference dataframes via `load_advanced_metrics_df()`. 18 new features: receiving efficiency (YPT, catch rate, YPR, target quality), rushing efficiency (YPC, yards per touch, TD rate), passing efficiency (YPA, completion %, TD/INT rates), usage (snap share, opportunity share, red zone usage), and trends (usage trend, efficiency trend, role score).

### Trade Calculator UX
- **Age display formatting** — Player ages now display consistently as "27.0 yrs" in trade chips with proper float-to-string conversion. Previously ages could appear missing due to type coercion issues.
- **Position rank labels** — Position ranks (WR7, RB5, etc.) now display correctly in trade chips. Uses SF rank when Superflex mode selected, 1QB rank otherwise.

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
