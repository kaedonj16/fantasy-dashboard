# Dynasty Player Valuation Model - Critical Analysis

**Analysis Date:** 2026-03-26
**Analyst:** Claude Code
**Scope:** Complete end-to-end review of dynasty fantasy football player valuation system

---

## Step 2: Complete Data Pipeline Map

### External Data Sources

1. **Sleeper API**
   - Player metadata (name, team, position, age, DOB)
   - Weekly stats (weeks 1-18): snaps, targets, receptions, carries, fantasy points
   - Redzone stats: RZ targets/attempts per game
   - **Refresh:** Real-time via API calls

2. **FantasyCalc API**
   - Dynasty trade values (10-team, 1QB, PPR)
   - Pick values by exact slot
   - **Refresh:** Daily scrape
   - **Location:** `data/fantasycalc_api_values_{date}.csv`

3. **DynastyProcess GitHub**
   - Dynasty trade values (consensus community values)
   - Pick values
   - **Refresh:** Daily download
   - **Location:** `data/dynastyprocess_values_{date}.csv`

4. **NFLVerse GitHub**
   - Draft history (2014-present)
   - Player draft round, pick, team
   - **Refresh:** On-demand fetch
   - **Location:** `cache/player_investment/draft_history.parquet`

5. **OverTheCap.com (scraped)**
   - QB/RB/WR/TE contracts
   - APY, guaranteed money, years, free agency year
   - **Refresh:** On-demand scrape
   - **Location:** `cache/player_investment/contracts_latest.parquet`

6. **Tank01 RapidAPI**
   - Team offensive stats (pass/rush yards, TDs per game)
   - **Refresh:** On-demand API call
   - **Location:** `cache/teams_index.json`

7. **TeamRankings.com (scraped)**
   - Team rush attempts, rush/pass yards per game
   - Opponent defensive stats
   - **Refresh:** On-demand scrape
   - **Location:** `cache/teams_index.json`

8. **Footballguys.com**
   - Season target share by team
   - **Refresh:** Embedded in usage table build

---

### Processing Pipeline (Execution Order)

#### **Phase 1: Raw Data Collection**

**File:** `data_building/external_data/external_values_scraper.py`

```
scrape_all_vendor_values()
  ├─> Fetch FantasyCalc API (numTeams=10, numQbs=1, ppr=1)
  │   └─> Output: data/fantasycalc_api_values_{date}.csv
  └─> Download DynastyProcess values.csv
      └─> Output: data/dynastyprocess_values_{date}.csv
```

#### **Phase 2: Usage & History Aggregation**

**File:** `data_building/external_data/sleeper_usage.py` + `player_history.py`

```
write_usage_table_snapshot(season, weeks)
  ├─> fetch_season_stats(season, weeks 1-18)  [Sleeper API]
  ├─> fetch_season_redzone_stats(season)      [Sleeper API]
  ├─> fetch_league_target_share(season)       [Footballguys]
  ├─> Aggregate per-player season stats:
  │   • games, avg_off_snap_pct, avg_targets, avg_carries
  │   • ppr_ppg, rec_rz_tgt_pg, rush_rz_att_pg
  │   • target_share (season-level from Footballguys)
  └─> Output: data/usage_table_{date}.json

build_multi_season_player_history(current_season, num_past_seasons=2)
  ├─> Build usage for seasons N-2, N-1, N
  ├─> Save per-season parquet files
  └─> Output: cache/player_history/player_history_all.parquet

build_player_history_features(history_df)
  ├─> Calculate 3-year weighted features (60% last year, 30% prev, 10% 2 yrs ago):
  │   • three_year_weighted_ppg
  │   • three_year_weighted_snap_pct
  │   • three_year_weighted_target_share
  ├─> Calculate trend features:
  │   • ppg_trend_1yr, ppg_trend_2yr
  │   • target_share_trend_1yr
  ├─> Calculate career features:
  │   • career_best_ppg, career_avg_ppg
  │   • seasons_played, games_last_3yr
  └─> Output: DataFrame with 12+ historical features per player
```

**Key Columns in Usage Table:**
- `id` (Sleeper ID), `name`, `team`, `position`, `age`
- `usage.games`, `usage.avg_off_snap_pct`, `usage.avg_targets`, `usage.avg_carries`
- `usage.ppr_ppg`, `usage.rec_rz_tgt_pg`, `usage.rush_rz_att_pg`
- `usage.target_share` (0-1 scale, season-level from Footballguys)

#### **Phase 3: Investment Context**

**File:** `data_building/external_data/player_investment.py`

```
build_player_investment_context(start_draft_season=2014)
  ├─> load_nflverse_players_draft_history()
  │   ├─> Fetch https://github.com/nflverse/nflverse-data/.../players.parquet
  │   ├─> Extract: draft_year, draft_round, draft_pick, draft_team
  │   └─> Calculate draft_capital_score (0-1 scale):
  │       • score = 1.0 / (1.0 + (pick - 1) / 32.0)
  │       • Round multipliers: R1=1.0, R2=0.92, R3=0.84, R4-5=0.68, R6-7=0.52
  │
  ├─> scrape_otc_contracts()
  │   ├─> Scrape OTC for QB/RB/WR/TE
  │   ├─> Extract: contract_total_value, contract_apy, contract_years,
  │   │            guaranteed_money, free_agency_year
  │   └─> Calculate:
  │       • years_to_fa = free_agency_year - current_year
  │       • guaranteed_pct = guaranteed_money / contract_total_value
  │
  ├─> Calculate positional percentiles:
  │   • contract_apy_pos_pct (rank within position)
  │   • guaranteed_money_pos_pct
  │   • guaranteed_pct_pos_pct
  │   • draft_capital_pos_pct
  │
  ├─> Calculate contract_score:
  │   • 0.45 * contract_apy_pos_pct +
  │     0.35 * guaranteed_money_pos_pct +
  │     0.20 * guaranteed_pct_pos_pct
  │
  └─> Calculate team_investment_score (blended draft + contract):
      • 0.35 * contract_score +
        0.25 * draft_capital_score (raw) +
        0.40 * draft_capital_pos_pct
      └─> Output: cache/player_investment/player_investment_latest.parquet
```

**Key Investment Signals:**
- `draft_capital_score`: 0-1, heavily weighted to early picks
- `contract_score`: 0-1, blends APY + guarantees (positional percentiles)
- `team_investment_score`: 0-1, final blended metric
- `years_to_fa`: Numeric, contract security signal

#### **Phase 4: Team Context Enrichment**

**File:** `data_building/external_data/team_enrichment.py`

```
enrich_all_team_info(season)
  ├─> fetch_team_offense_per_game(season)  [Tank01 API]
  │   ├─> Extract per-team: pass_yds_pg, pass_att_pg, pass_td_pg
  │   │                      rush_yds_pg, rush_att_pg, rush_td_pg
  │   └─> Store in teams_index
  │
  └─> enrich_teams_index_with_rushing()    [TeamRankings scrape]
      ├─> Scrape 5 URLs in parallel:
      │   • rush_att_pg, rush_yds_pg, pass_yds_pg
      │   • opp_pass_yds_pg, opp_rush_yds_pg
      └─> Update cache/teams_index.json
```

**Team Context Used in Valuation:**
- Offensive volume signals (pass attempts, rush attempts per game)
- Touchdown distribution (pass/rush TDs per game)
- Defensive context (opponent yards allowed)

#### **Phase 5: Engine Value Calculation**

**File:** `data_building/player_value.py`

**Entry Point:** `build_value_table_for_usage()`

```
build_value_table_for_usage()
  ├─> load_usage_table()  [from Phase 2]
  ├─> load_teams_index()  [from Phase 4]
  │
  ├─> For each player:
  │   ├─> _age_factor(age, position)
  │   │   • 3-year dynasty horizon age curves
  │   │   • Peak ages: QB=27-31, RB=24-27, WR=26-29, TE=26-29
  │   │   • Polynomial curves with rapid dropoff for RB after 27
  │   │
  │   ├─> _production_component_fixed(usage, position)
  │   │   • Position-specific production scoring:
  │   │     - QB: 0.40*ppr_ppg + 0.35*pass_att + 0.25*snaps
  │   │     - RB: 0.45*ppr_ppg + 0.30*carries + 0.25*targets
  │   │     - WR: 0.50*ppr_ppg + 0.35*targets + 0.15*snaps
  │   │     - TE: 0.50*ppr_ppg + 0.35*targets + 0.15*snaps
  │   │   • All inputs normalized to 0-1 scale via percentiles
  │   │
  │   ├─> _usage_role_security(usage, position)
  │   │   • Opportunity consistency scoring:
  │   │     - Snap % (high weight for non-QB)
  │   │     - Target/carry volume
  │   │     - Games played (availability)
  │   │     - Redzone usage
  │   │
  │   ├─> _investment_score(player_id)
  │   │   • Loads team_investment_score from Phase 3
  │   │   • 0-1 scale (draft capital + contract blend)
  │   │
  │   ├─> _risk_penalty(age, games, usage, position)
  │   │   • Age risk: steep penalties for RB 28+, QB 34+
  │   │   • Sample size risk: < 4 games played
  │   │   • Role risk: low snap % with production (TD variance)
  │   │   • Injury risk: games missed
  │   │
  │   ├─> Blend components (position-specific weights):
  │   │   • production_weight: QB=0.45, RB=0.50, WR/TE=0.48
  │   │   • age_weight: QB=0.18, RB=0.15, WR/TE=0.17
  │   │   • role_weight: QB=0.20, RB=0.18, WR/TE=0.20
  │   │   • investment_weight: all=0.15
  │   │   • risk_penalty_weight: all=0.02
  │   │   → raw_value = sum of weighted components (0-1)
  │   │
  │   ├─> Apply market compression:
  │   │   • _apply_qb_market_compression(raw, position)
  │   │     - 1QB leagues: reduce QB values by 35-60%
  │   │     - Compress based on starter scarcity (10 starters for 10 teams)
  │   │   • _apply_te_market_compression(raw, position)
  │   │     - Boost TE scarcity premium by 8-15%
  │   │
  │   └─> Scale to 0-999.9:
  │       • value = (raw_value ^ 0.72) * 999.9
  │       • GAMMA=0.72 creates exponential spread at top
  │
  └─> Output: {player_id: engine_value, ...}
```

**Engine Value Characteristics:**
- **Scale:** 0-999.9 (elite RBs/WRs typically 800-999, QBs compressed to 300-600 in 1QB)
- **Horizon:** 3-year dynasty outlook
- **League Context:** Hardcoded 10-team, 1QB
- **Key Assumptions:**
  - `STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}`
  - `NUM_TEAMS = 10`
  - `GAMMA = 0.72` (value curve exponent)

#### **Phase 6: Export Engine Values**

**File:** `data_building/value_exports.py`

```
export_engine_values()
  ├─> build_value_table_for_usage()  [from Phase 5]
  ├─> load_relevant_index()  [player metadata]
  ├─> Merge engine values with player names/positions/teams
  ├─> Filter: must have name, position, team, non-null value
  └─> Output: data/engine_values_{date}.csv
```

#### **Phase 7: ML Consensus Model Training**

**File:** `data_building/value_model_training.py`

**Entry Point:** `rewrite_value_table_with_model()`

```
train_trade_value_model()
  ├─> build_training_dataframe()
  │   ├─> load_fantasycalc_df()
  │   │   • Read CSV from Phase 1
  │   │   • Normalize: fc_value (keep raw scale ~0-10000)
  │   │
  │   ├─> load_dynastyprocess_df()
  │   │   • Read CSV from Phase 1
  │   │   • Normalize: dp_value
  │   │
  │   ├─> load_engine_df()
  │   │   • Read CSV from Phase 6
  │   │   • Normalize: engine_value (0-999.9 scale)
  │   │
  │   ├─> Merge all 3 sources on player name normalization
  │   │   • Drop players missing any vendor value
  │   │
  │   ├─> Calculate consensus_value (target):
  │   │   • consensus = 0.50 * fc_value +
  │   │                 0.35 * dp_value +
  │   │                 0.15 * engine_value
  │   │   • FC gets 50% weight (most stable)
  │   │   • DP gets 35% (community wisdom)
  │   │   • Engine gets 15% (proprietary signals)
  │   │
  │   ├─> Merge player_history features (from Phase 2):
  │   │   • three_year_weighted_ppg
  │   │   • three_year_weighted_snap_pct
  │   │   • ppg_trend_1yr, ppg_trend_2yr
  │   │   • career_best_ppg, seasons_played
  │   │   • (12+ historical features)
  │   │
  │   ├─> Merge player_investment features (from Phase 3):
  │   │   • draft_capital_score
  │   │   • contract_apy, guaranteed_money
  │   │   • team_investment_score
  │   │   • years_to_fa
  │   │
  │   ├─> Merge team context (from Phase 4):
  │   │   • team_pass_att_pg, team_rush_att_pg
  │   │   • team_pass_yds_pg, team_rush_yds_pg
  │   │
  │   └─> Feature set (30+ features):
  │       • Vendor values: fc_value, dp_value, engine_value
  │       • Age: current age
  │       • Production: ppg, snap_pct, targets, carries (current season)
  │       • History: 3yr weighted, trends, career bests
  │       • Investment: draft capital, contracts, team investment
  │       • Team context: offensive volume, TDs per game
  │       • Position: one-hot encoded (QB, RB, WR, TE)
  │
  ├─> Train GradientBoostingRegressor:
  │   • n_estimators=250
  │   • learning_rate=0.03
  │   • max_depth=2 (shallow trees prevent overfitting)
  │   • Target: consensus_value
  │   • Features: all 30+ features
  │   • Result: model predicts trade value for any player
  │
  └─> Return trained model + feature columns

rewrite_value_table_with_model()
  ├─> model, feature_cols = train_trade_value_model()
  │
  ├─> Load full player universe (including players without vendor values):
  │   • usage_table (all active players)
  │   • history features
  │   • investment features
  │   • team context
  │
  ├─> Build feature matrix for ALL players:
  │   • Use vendor values if available
  │   • Fill missing vendor values with 0 or engine value
  │   • Fill missing history with 0
  │   • Fill missing investment with 0
  │
  ├─> Predict values:
  │   • model.predict(X) → model_value for each player
  │
  ├─> Load pick values:
  │   • load_pick_value_table() [from picks.py]
  │   • Merge FC picks (55%) + DP picks (45%)
  │   • Remap 12-team picks to 10-team equivalents
  │   • Bucket future picks:
  │     - Upcoming draft: exact slots (2026_1_01, 2026_1_02, ...)
  │     - Future drafts: buckets (2027_1_early, 2027_2_mid, 2028_3_late)
  │
  ├─> Combine players + picks into single value table:
  │   • [{id, name, team, position, age, value, ...}, ...]
  │   • 'value' = model_value (or pick value for picks)
  │
  └─> Output: data/model_value_table_{date}.json
```

**ML Model Characteristics:**
- **Target:** Consensus of FC (50%) + DP (35%) + Engine (15%)
- **Model:** GradientBoostingRegressor (250 trees, lr=0.03, depth=2)
- **Features:** 30+ including vendor values, age, production, history, investment, team context
- **Training Set:** Only players with all 3 vendor values present
- **Prediction Set:** All active players (fills missing features with 0/defaults)
- **Output Scale:** Same as FantasyCalc (~0-10000 range)

#### **Phase 8: Pick Valuation**

**File:** `dashboard_services/picks.py`

```
load_pick_value_table()
  ├─> Load FC pick values (from Phase 1 CSV)
  ├─> Load DP pick values (from Phase 1 CSV)
  │
  ├─> For each pick:
  │   ├─> _remap_pick_to_league_size(round, pick, from_teams=12, to_teams=10)
  │   │   • Example: 12-team 1.12 → 10-team 1.10
  │   │   • Adjusts pick numbers to match league size
  │   │
  │   ├─> _build_pick_key(year, round, pick):
  │   │   • Upcoming draft (offseason): exact slots
  │   │     - 2026_1_01, 2026_1_02, ..., 2026_3_10
  │   │   • Future drafts: bucketed by thirds
  │   │     - 2027_1_early (picks 1-3)
  │   │     - 2027_1_mid (picks 4-7)
  │   │     - 2027_1_late (picks 8-10)
  │   │     - 2027_2_early, 2027_2_mid, 2027_2_late
  │   │     - 2028_3_early, 2028_3_mid, 2028_3_late
  │   │
  │   └─> Blend FC + DP values:
  │       • pick_value = 0.55 * fc_pick_value + 0.45 * dp_pick_value
  │
  └─> Return {pick_key: pick_value, ...}
```

**Pick Value Characteristics:**
- **Blend:** 55% FC + 45% DP (FC slightly favored for stability)
- **League Size:** Remapped from 12-team to 10-team
- **Bucketing Strategy:**
  - Current draft year (offseason): Exact slots (30 picks: 1.01-3.10)
  - Future drafts: Buckets by third (early/mid/late per round)
  - Time discount: Future picks inherently less valuable (baked into vendor values)
- **Scale:** Same as model_value (~0-10000 range)

#### **Phase 9: Historical Snapshot**

**File:** `data_building/player_value_history.py`

```
record_model_value_snapshot(model_value_table)
  ├─> Insert into database: value_snapshots table
  │   • Columns: date, player_id, name, position, value
  │   • Used for tracking value changes over time
  │   • Enables trend analysis and value history charts
  └─> Return: number of rows inserted
```

#### **Phase 10: API Endpoint**

**File:** `app.py` (lines 5716-5724)

```python
@app.route("/api/league-players")
def api_league_players():
    model_value_table = load_model_value_table()
    # Loads: data/model_value_table_{date}.json

    cleaned_players = _sanitize_for_json(model_value_table)
    return jsonify(cleaned_players)
```

**API Response Schema:**
```json
[
  {
    "id": "sleeper_id",
    "name": "Player Name",
    "team": "KC",
    "position": "RB",
    "age": 25.3,
    "value": 8543.2,
    "engine_value": 875.4,
    "fc_value": 8800,
    "dp_value": 8200,
    ...
  }
]
```

---

### Daily Rebuild Orchestrator

**File:** `data_building/build_daily_value_table.py`

**Entry Point:** `build_daily_data(season, week)`

```
build_daily_data(season, week)
  ├─> Check NFL season state:
  │   • offseason_mode = season_type == "off"
  │   • If in-season and week >= 1:
  │       - get_live_game_ids_for_today()
  │       - build_and_save_week_stats_for_league()
  │
  ├─> Step 1: Scrape vendor values (if missing)
  │   • if load_fantasycalc_api_values() is None
  │       OR load_dynastyprocess_values() is None:
  │       → scrape_all_vendor_values()  [Phase 1]
  │
  ├─> Step 2: Build usage + investment (if missing)
  │   • if load_usage_table() is None
  │       OR load_engine_table() is None:
  │       → write_usage_table_snapshot(season, weeks=1-18)  [Phase 2]
  │       → enrich_all_team_info(season)  [Phase 4]
  │       → enrich_teams_index_with_rushing()  [Phase 4]
  │       → export_engine_values()  [Phase 6]
  │
  ├─> Step 3: Train model (if missing)
  │   • if load_model_value_table() is None:
  │       → rewrite_value_table_with_model()  [Phase 7]
  │       → record_model_value_snapshot()  [Phase 9]
  │
  └─> Result: data/model_value_table_{date}.json ready for API
```

**Rebuild Triggers:**
- Manual: `python data_building/build_daily_value_table.py`
- Scheduled: Daily cron job (implied, not in code)
- Conditional: Only rebuilds if files missing

---

## Step 3: Critical Analysis

### 3.1 Value Accuracy

#### **Positional Hierarchy**

**ISSUE: QB Compression May Be Too Aggressive**

The engine applies 35-60% QB value compression in 1QB leagues:

```python
# player_value.py:789-801
def _apply_qb_market_compression(raw_value, position, num_teams=10, starters_map=None):
    if position != "QB":
        return raw_value

    num_starters = 1 * num_teams  # 10 starters in 10-team 1QB
    # ... compression logic ...
    if rank <= num_starters:
        return raw_value * 0.65  # Top 10 QBs lose 35% value
    elif rank <= num_starters * 1.5:
        return raw_value * 0.50  # QB11-15 lose 50%
    else:
        return raw_value * 0.40  # QB16+ lose 60%
```

**Problem:** This creates a massive gap between QB and RB/WR in 1QB. In real 1QB dynasty trades:
- Patrick Mahomes trades for ~2-3 early 2nds (value ~3000-4500)
- Elite QBs (Allen, Burrow) trade for mid-1st to early-2nd (value ~4000-6000)
- But compression reduces elite QB raw value from ~0.92 to 0.60 → ~600 engine value

After ML model blending, QBs get boosted back up by FC/DP vendor values, but the engine signal is artificially weak.

**Impact:** Engine values systematically undervalue elite QBs relative to market. ML model corrects this by leaning on FC/DP, but loses proprietary signal strength.

**ISSUE: TE Scarcity Boost May Be Insufficient**

```python
# player_value.py:804-816
def _apply_te_market_compression(raw_value, position):
    if position != "TE":
        return raw_value

    # Boost TE value due to scarcity
    if raw_value >= 0.75:
        return raw_value * 1.15  # Elite TEs get 15% boost
    elif raw_value >= 0.50:
        return raw_value * 1.10  # Mid TEs get 10% boost
    else:
        return raw_value * 1.08  # Replacement TEs get 8% boost
```

**Problem:** TE scarcity in dynasty is extreme. The top 3 TEs (Kelce/Andrews/LaPorta era) trade for early 1sts. But 8-15% boost may not capture true scarcity premium.

Real market behavior:
- TE1-3: Early-mid 1st round picks (~7000-9000 value)
- TE4-6: Late 1st to early 2nd (~4000-6000 value)
- TE7-12: Mid 2nd to early 3rd (~2000-4000 value)
- TE13+: Late 3rd or waiver (~500-1500 value)

The cliff after TE3 is much steeper than 8-15% suggests.

**Impact:** Engine likely undervalues elite TEs (though ML model corrects via vendor values).

#### **Value Ranges & Calibration**

**ISSUE: Engine Scale (0-999.9) Doesn't Match ML Output Scale (~0-10000)**

- Engine outputs values on 0-999.9 scale
- FantasyCalc values range ~0-10000+
- DynastyProcess values range ~0-15000+
- ML model trains on consensus target mixing all three scales

**Problem:** When building training features, engine values are ~10x smaller than vendor values. This creates feature scale imbalance.

```python
# value_model_training.py - feature columns include:
# - fc_value (scale ~0-10000)
# - dp_value (scale ~0-15000)
# - engine_value (scale ~0-999.9)
```

The GradientBoostingRegressor will learn to weight engine_value ~10x higher to compensate, but this is fragile.

**Impact:** Model may be overly sensitive to engine value fluctuations, or may ignore engine signal if vendor values dominate.

**Recommendation:** Normalize all value inputs to 0-1 scale before model training.

#### **Aging Curves**

**GOOD: Position-Specific Age Curves Are Well-Designed**

```python
# player_value.py:260-310
def _age_factor(age, position):
    if position == "QB":
        # Peak 27-31, gradual decline
        peak_start, peak_end = 27, 31
        young_curve = min(1.0, (age - 20) / 7)
        old_curve = max(0.4, 1.0 - ((age - 31) / 8))

    elif position == "RB":
        # Peak 24-27, steep decline
        peak_start, peak_end = 24, 27
        young_curve = min(1.0, (age - 19) / 5)
        old_curve = max(0.25, 1.0 - ((age - 27) / 4) ** 1.8)  # Exponential

    elif position in ("WR", "TE"):
        # Peak 26-29, moderate decline
        peak_start, peak_end = 26, 29
        young_curve = min(1.0, (age - 20) / 6)
        old_curve = max(0.35, 1.0 - ((age - 29) / 7) ** 1.3)
```

**Strengths:**
- RB age cliff is appropriately aggressive (exponential decay after 27)
- QB longevity is properly modeled (gradual decline)
- Rookie discounts are applied (young_curve ramps up)

**POTENTIAL ISSUE: No Elite Player Age Extension**

Elite players maintain value longer than average. Examples:
- CMC produced at age 28-29 (should be steep decline per curve)
- Tyreek Hill elite at 30+ (should be declining)
- Aaron Rodgers elite at 38-39 (curve predicts 0.4 factor)

**Current Approach:** Age factor is uniform per position, no adjustment for talent/production.

**Impact:** May undervalue elite old players, overvalue mediocre young players.

**Recommendation:** Add interaction term: `age_factor * production_percentile` to extend primes for elites.

#### **Pick Valuation**

**ISSUE: Pick Bucketing Loses Granularity**

```python
# picks.py:180-195
def _bucket_for_pick_in_round(pick_in_round, num_teams):
    third = num_teams / 3
    if pick_in_round <= third:
        return "early"
    elif pick_in_round <= third * 2:
        return "mid"
    else:
        return "late"

# For 10-team league:
# early = picks 1-3
# mid = picks 4-7
# late = picks 8-10
```

**Problem:** In dynasty, 1.01 vs 1.03 is a significant difference (Bijan vs Wilson, Marvin Harrison vs Nabers). Bucketing loses this signal.

Real market behavior for future 1sts:
- 1.01-1.02: Elite RB/WR prospect tier (~8000-9000 value)
- 1.03-1.05: High-end RB/WR tier (~6500-7500 value)
- 1.06-1.08: Solid starter tier (~5000-6000 value)
- 1.09-1.10: Dart throw tier (~4000-5000 value)

Bucketing "early" (1-3) and "mid" (4-7) flattens this curve.

**Impact:** Future picks may be overvalued (mid bucket) or undervalued (early bucket) depending on actual slot.

**ISSUE: No Explicit Time Discount**

Picks lose value the further out they are (time value of assets). Current approach relies on vendor values baking in time discount.

```python
# No time discount formula in code
# Just trusts FC/DP to price 2027 picks < 2026 picks
```

**Problem:** If vendor values don't properly discount, model inherits the error.

**Recommendation:** Add explicit time decay: `pick_value * (0.90 ** years_away)` (10% annual discount).

---

### 3.2 Model Inputs

#### **Current Signals - Strengths**

**EXCELLENT: Comprehensive Production Metrics**

The engine uses a rich set of production signals:
- PPR points per game (outcome-based)
- Snap % (opportunity-based)
- Targets, carries, receptions (volume-based)
- Redzone targets/attempts (efficiency/TD equity)
- Target share (team-relative opportunity)

This multi-dimensional approach avoids over-indexing on any single metric.

**EXCELLENT: 3-Year Weighted History**

```python
# player_history.py:242-244
weighted_ppg_3yr = (0.6 * ppg_3[-1]) + (0.3 * ppg_3[-2]) + (0.1 * ppg_3[-3])
```

Recency weighting (60/30/10) properly balances:
- Recent performance (most predictive)
- Historical consistency (regression to mean)
- Career trajectory (trend signals)

**GOOD: Investment Context**

Blending draft capital + contract data captures team commitment:
- High draft picks get more opportunity
- High APY/guarantees signal team belief
- Years to FA indicates job security

#### **Current Signals - Weaknesses**

**MISSING: Efficiency Metrics**

Current signals focus on volume (targets, carries, snaps) but lack efficiency:
- Yards per carry (RB talent signal)
- Yards per target (WR/TE separation)
- TD rate vs expected (talent vs luck)
- Catch rate (WR skill)
- Air yards (WR role)

**Impact:** Model can't distinguish volume-driven production from talent-driven efficiency.

**Example:** Two WRs with 120 targets and 1200 yards:
- WR A: 120 targets, 90 catches (75% catch rate), 1200 yards, 8 TDs
- WR B: 120 targets, 65 catches (54% catch rate), 1200 yards, 8 TDs

Current model sees identical production. But WR A is more talented (higher catch rate = better separation, hands, route running). WR A should value higher.

**MISSING: Supporting Cast Context**

Player value depends on teammates:
- WR paired with elite QB vs backup QB
- RB in committee vs bell cow
- TE1 vs TE2 on same team

Current model has team offensive volume (pass_att_pg) but not teammate context.

**Example:**
- Garrett Wilson with Aaron Rodgers (elite QB) vs Garrett Wilson with Zach Wilson (backup QB)
- Model doesn't adjust for QB quality change

**MISSING: Coaching/Scheme Signals**

Offensive scheme heavily impacts fantasy production:
- Pass-heavy vs run-heavy offense
- Zone-blocking vs power-blocking (RB fit)
- Play-action rate (benefits certain WRs)
- Red-zone tendencies (goal-line RB vs passing)

Current model has team pass/rush yards but not scheme stability signals.

**MISSING: Injury History**

Durability is a major dynasty factor. Current model has:
- `games` played (current season)
- `games_last_3yr` (historical)

But lacks:
- Injury type/severity (ACL, Achilles, concussion)
- Games missed due to injury vs healthy scratch
- Injury recurrence risk

**Impact:** Model treats missed games equally, but ACL recovery year is different from healthy scratch.

**WEAK: Rookie/Breakout Identification**

Model struggles with rookies and breakout candidates:
- Rookies have no `three_year_weighted_ppg` (filled with 0)
- Breakouts have low historical production (drags down value)

Example: Puka Nacua 2023 breakout
- Before season: ~2nd year WR, minimal rookie production
- Model would undervalue due to weak history features
- Post-breakout: Strong season stats, but history still weak

**Recommendation:** Add "breakout indicators":
- Target share increase > 5% year-over-year
- Snap % increase > 15% year-over-year
- Age < 24 and production spike (youth + opportunity = breakout)

#### **Recency Weighting**

**CURRENT: 3-Year Window with 60/30/10 Weighting**

```python
# player_history.py:242-244
weighted_ppg_3yr = (0.6 * ppg_3[-1]) + (0.3 * ppg_3[-2]) + (0.1 * ppg_3[-3])
```

**ISSUE: May Be Too Focused on Recent Year**

Dynasty values balance recent performance + long-term upside/risk. 60% weight on last year may:
- Overreact to fluky seasons (injury-shortened, TD luck)
- Underweight career consistency for veterans
- Miss multi-year trends

**Alternative Approach:**
- Equal weighting (33/33/33) for established vets
- Recency weighting (60/30/10) for young players (< 26)
- Exponential smoothing with alpha=0.5 for all history

**MISSING: In-Season Recency Weighting**

During season, model uses full-season averages. But recent weeks are more predictive than early-season stats.

**Example:** RB Week 1-8 vs Week 9-17
- Weeks 1-8: 12 carries/game, 3.8 YPC (backup role)
- Weeks 9-17: 18 carries/game, 4.6 YPC (starter role after injury)

Current model: `avg_carries = 15/game` (blend of both)
Better approach: Weight recent weeks higher (70% last 4 weeks, 30% earlier)

**Recommendation:** Add rolling window features:
- `last_4_weeks_ppg`
- `last_4_weeks_snap_pct`
- `ppg_acceleration` (is player trending up or down?)

#### **Position-Specific Needs**

**QB:**

Missing:
- Rushing upside (Josh Allen, Lamar Jackson)
- Supporting cast quality (OL, weapons)
- Passing volume (team pass rate)

Current model has `pass_att` but not adjusted for pass-heavy vs run-heavy offenses.

**Recommendation:** Add:
- `rush_att_pg` for QB (dual-threat signal)
- `team_pass_rate` (pass_att / (pass_att + rush_att))
- `oline_rank` (from PFF or similar)

**RB:**

Missing:
- Receiving role (pass-catching RBs more valuable)
- Goal-line role (TD equity)
- Offensive line quality

Current model has `targets` but not RB-specific receiving metrics.

**Recommendation:** Add:
- `target_share_among_rbs` (RB1 vs committee)
- `goal_line_carry_share` (TD equity)
- `team_oline_rank` (run-blocking quality)

**WR:**

Missing:
- Target quality (air yards, deep targets)
- Catch rate (talent signal)
- Slot vs outside (role diversity)

**Recommendation:** Add:
- `avg_air_yards_per_target`
- `catch_rate`
- `slot_rate` (versatility signal)

**TE:**

Missing:
- Blocking role (snap % without targets = blocker)
- Target competition (TE1 vs TE2)

**Recommendation:** Add:
- `routes_run_per_snap` (receiving vs blocking)
- `target_share_among_tes`

---

### 3.3 Pick Valuation Deep Dive

#### **Bucketing Strategy**

**CURRENT APPROACH:**
- Upcoming draft (offseason): Exact slots (2026_1_01, 2026_1_02, ...)
- Future drafts: Buckets (2027_1_early, 2027_1_mid, 2027_1_late)

**ISSUES:**

1. **Lost Precision for Future Drafts**

Example: Owner promises "my 2027 1st" in June 2026. Team looks strong (projected 1.09-1.10). Model values as "2027_1_late". But if team injuries tank, pick becomes 1.03. Value jump is huge (~4000 → 7000), but model didn't see it coming.

**Recommendation:** Add probabilistic pick projection:
- Track team roster strength, age, injury risk
- Estimate pick landing spot distribution (e.g., 30% early, 50% mid, 20% late)
- Value = weighted average of bucket values

2. **No Rookie Class Adjustment**

Draft pick values fluctuate based on rookie class strength:
- 2024 class (Bijan, Gibbs, Stroud) was elite → 1sts very valuable
- 2025 class (Bowers, Nabers, MHJ) was elite → 1sts very valuable
- 2026 class (projected weaker) → 1sts less valuable

Current model treats all 2027_1_early picks equally, regardless of class strength.

**Recommendation:** Add rookie class strength multiplier:
- Elite class: 1.10x pick value
- Average class: 1.00x pick value
- Weak class: 0.90x pick value

3. **No Owner Context**

Pick value depends on league context:
- Contender values future 1sts less (win-now mode)
- Rebuilder values future 1sts more (tanking)

Current model outputs universal pick values, not contextual.

**Recommendation:** Add user context parameter:
- `team_mode`: "contending", "competing", "rebuilding"
- Adjust pick values:
  - Contending: future picks -10%, current picks +5%
  - Rebuilding: future picks +10%, current picks -5%

#### **Time Discounting**

**CURRENT: Relies on Vendor Values**

FantasyCalc and DynastyProcess bake time discount into pick values:
- 2026 1.01 > 2027 1.01 > 2028 1.01

But model doesn't explicitly model time decay.

**ISSUE: Vendor Values May Misprices Time Discount**

If FC/DP undervalue future picks, model inherits error. No independent signal.

**Recommendation:** Add explicit time decay formula:
```python
time_discount = 0.90 ** years_away  # 10% annual discount
pick_value = base_value * time_discount
```

This ensures future picks are systematically discounted even if vendors misprice.

#### **Pick-to-Player Conversions**

**MISSING: Draft Hit Rate Modeling**

Picks are probabilistic assets. Model should account for:
- % of 1st round picks that become stars (RB ~40%, WR ~30%, QB ~25%, TE ~20%)
- % of 2nd round picks that become starters (~20-30%)
- % of 3rd round picks that bust (~60-70%)

Current model values picks based on vendor values, but doesn't model expected value explicitly.

**Recommendation:** Add "pick expected value" formula:
```python
# 1st round pick expected value
EV_1st = (
    0.30 * elite_player_value +  # 30% hit elite (8000+)
    0.25 * solid_starter_value + # 25% hit starter (4000-6000)
    0.25 * bench_player_value +  # 25% bench (1000-2000)
    0.20 * bust_value             # 20% bust (0-500)
)
```

This makes pick valuation transparent and adjustable.

---

### 3.4 Edge Cases & Failure Modes

#### **1. Rookie Valuation**

**PROBLEM:** Rookies have no historical production data.

Current model fills history features with 0:
- `three_year_weighted_ppg = 0`
- `ppg_trend_1yr = 0`
- `career_best_ppg = 0`

**Impact:** Model undervalues hyped rookies (Bijan, MHJ) because history features are weak.

**Mitigation:** ML model learns to lean on:
- Draft capital (1st round picks = high value)
- Vendor values (FC/DP price rookies correctly)

But engine loses signal strength.

**Recommendation:**
- Add "pre-NFL production" features: college stats, draft position, combine metrics
- Use transfer learning: train separate model for rookies
- Add "rookie adjustment factor" based on draft capital and position

#### **2. Injured Players**

**PROBLEM:** Injured players have 0 games, 0 production.

Current model:
- `games = 0` → production = 0
- Model tanks their value

But dynasty values injured stars highly (buy-low opportunity).

**Example:**
- Jonathan Taylor Week 1-17 injury in 2023
- Model sees: 0 games, 0 production → low value
- Reality: Elite RB talent, temporary injury → high dynasty value

**Current Mitigation:**
- `three_year_weighted_ppg` uses historical data (60/30/10)
- If player was elite in 2022, history features prop up value

**Remaining Issue:**
- Multi-year injuries (ACL in Year N, recovery in Year N+1)
- Model sees 2 years of low/no production → value crashes
- Reality: Talent remains, just injured

**Recommendation:**
- Add "injury context" flag (out vs healthy scratch)
- Use "healthy games only" for production averages
- Add "pre-injury production" features

#### **3. Breakout Candidates**

**PROBLEM:** Players with sudden opportunity spikes are undervalued.

**Example:**
- Backup RB with 3 carries/game
- Starter gets injured Week 8
- Backup now gets 18 carries/game

Current model:
- Season average: 10 carries/game (blend of backup + starter role)
- History features: Low (was backup previous years)
- Model undervalues post-injury breakout

**Recommendation:**
- Add "recent weeks" features (last 4 weeks PPG, snap %)
- Add "role change" detection (snap % jump > 20%)
- Weight recent weeks higher in production signals

#### **4. Committee Backfields**

**PROBLEM:** RBs in committee are overvalued by raw stats.

**Example:**
- RB A: 12 carries, 4 targets, 15 PPR points per game
- RB B (committee): 12 carries, 4 targets, 15 PPR points per game

Raw stats identical. But:
- RB A: Clear starter, 75% snap share, secure role
- RB B: Committee member, 40% snap share, TD-dependent

Current model:
- Both get same production score
- Snap % differentiates slightly
- But committee risk not fully captured

**Recommendation:**
- Add "carries/targets per snap" (efficiency)
- Add "goal-line carry share" (TD equity signal)
- Penalize low snap % players with high TD rates (regression risk)

#### **5. Aging Stars**

**PROBLEM:** Model may overreact to age for elite players.

**Example:**
- Derrick Henry age 30: Still elite production
- Age curve predicts steep decline (RB 27+ drops fast)
- Model tanks value despite current elite performance

**Current Mitigation:**
- Recent production (60% weight) props up value
- ML model learns elite players age better

**Remaining Issue:**
- Engine age factor is uniform (no talent adjustment)
- Elite players maintain primes 2-3 years longer than average

**Recommendation:**
- Add interaction: `age_factor * career_best_ppg`
- Elite players (career_best_ppg > 20) get extended primes
- Average players (career_best_ppg < 15) decline on schedule

#### **6. Quarterback Changes**

**PROBLEM:** WR/TE values spike/crash with QB changes.

**Example:**
- Garrett Wilson with Zach Wilson (bad QB): WR2 value
- Garrett Wilson with Aaron Rodgers (elite QB): WR1 value

Current model:
- Has team offensive volume (pass_att_pg)
- But no QB quality signal

**Impact:** Model can't predict value changes from offseason QB acquisitions.

**Recommendation:**
- Add "QB quality" feature:
  - Elite QB (Mahomes, Allen): 1.2x WR/TE value
  - Average QB: 1.0x
  - Backup QB: 0.8x
- Track QB changes in offseason, adjust projections

#### **7. Prospect vs Proven Trade-offs**

**PROBLEM:** Model struggles with prospect vs proven player trades.

**Example:**
- Trade: Proven RB2 (age 28, 12 PPG, declining)
- For: 2027 1st (could be Bijan-level prospect)

Current model:
- RB2 has solid production + history → high value
- 2027 1st is bucketed "early/mid/late" → moderate value

**Reality:** Dynasty managers often prefer prospect upside over declining veterans.

**Recommendation:**
- Add "upside score" for young players (< 24)
- Discount "declining vets" (age > 28 with negative trends)
- Weight age curves more heavily for dynasty (3-year horizon)

#### **8. Superflex / 2QB Leagues**

**PROBLEM:** Model is hardcoded for 1QB leagues.

```python
# player_value.py:37-38
NUM_TEAMS = 10
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
```

In Superflex/2QB leagues:
- QB values 2-3x higher than 1QB
- QB scarcity is extreme (20+ starting QBs needed)

Current model applies 35-60% QB compression, destroying QB value in Superflex context.

**Impact:** Model completely breaks for Superflex leagues.

**Recommendation:**
- Add `league_type` parameter: "1QB", "Superflex", "2QB"
- Adjust QB compression based on league type:
  - 1QB: 35-60% compression (current)
  - Superflex: 0-10% compression
  - 2QB: -20% compression (actually boost QBs)

---

### 3.5 Data Quality Issues

#### **1. Sleeper API Consistency**

**RISK:** Sleeper API sometimes returns incomplete/inconsistent data.

```python
# sleeper_usage.py:66-69
for week, players in season_stats.items():
    if not isinstance(players, dict):
        # Sleeper sometimes returns {"message": "..."} if no data
        continue
```

**Impact:** If Sleeper API fails during data build, usage table is incomplete. Model trains on partial data.

**Mitigation:** Code has defensive checks (`if not isinstance(players, dict)`).

**Remaining Risk:** Partial failures (some weeks missing) may go undetected.

**Recommendation:**
- Add validation: check expected week count (1-18)
- Alert if any weeks missing
- Add fallback data source (NFL.com, ESPN, etc.)

#### **2. Name Normalization Fragility**

**RISK:** Player name matching across sources is error-prone.

```python
# utils/utils.py (inferred)
def normalize_name(name):
    # Strips punctuation, lowers, removes suffixes (Jr., III)
    # But: "D.J. Moore" vs "DJ Moore" vs "DJMoore"
```

**Impact:** If names don't match, players lose vendor values or investment data.

**Example:**
- FantasyCalc: "Kenneth Walker III"
- DynastyProcess: "Kenneth Walker"
- Sleeper: "Ken Walker"

Model may fail to merge these as same player.

**Recommendation:**
- Add fuzzy matching (Levenshtein distance)
- Use Sleeper ID as primary key everywhere
- Fallback to name matching only when ID missing

#### **3. Draft Capital Data Freshness**

**RISK:** NFLVerse draft data may lag.

```python
# player_investment.py:28-34
NFLVERSE_PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
)
```

**Impact:** If 2025 draft picks aren't in NFLVerse yet (post-draft delay), rookies have no draft capital.

**Mitigation:** Code has fallback to secondary draft picks dataset.

**Recommendation:**
- Add manual override: maintain local CSV with current-year picks
- Merge manual + NFLVerse data

#### **4. Contract Data Staleness**

**RISK:** OverTheCap scraping may be blocked/throttled.

```python
# player_investment.py:315-324
def scrape_otc_contracts():
    for pos, url in OTC_POSITION_URLS.items():
        try:
            tables = pd.read_html(url)
        except Exception as e:
            print(f"failed OTC scrape for {pos}: {e}")
            continue
```

**Impact:** If OTC blocks scraping, contract data becomes stale. Investment scores based on old contracts.

**Recommendation:**
- Add "last updated" timestamp to contracts
- Alert if contracts > 30 days old
- Add fallback: Spotrac.com or manual CSV

#### **5. Team Context Lag**

**RISK:** TeamRankings stats update mid-week.

```python
# team_enrichment.py:225-236
def _fetch_teamrankings_table(url, session):
    # Scrapes "Current Season" column
```

**Impact:** Monday-Wednesday, stats may be from previous week. Thursday-Sunday, stats are current week.

**Recommendation:**
- Add "as of week" tracking
- Only rebuild values on Wednesdays (after stats finalize)

---

## Step 4: Ranked Improvement Suggestions

### Tier 1: Critical (High Impact, Low Effort)

#### **1. Normalize Value Scales Before ML Training**

**Problem:** Engine values (0-999.9) mixed with FC (0-10000+) and DP (0-15000+) creates feature scale imbalance.

**Solution:**
```python
def normalize_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Before training:
df['fc_value_norm'] = normalize_value(df['fc_value'], 0, 10000)
df['dp_value_norm'] = normalize_value(df['dp_value'], 0, 15000)
df['engine_value_norm'] = normalize_value(df['engine_value'], 0, 999.9)
```

**Impact:** Model learns proper feature weights, reduces overfitting to any single source.

**Effort:** 30 lines of code, 1 hour

**Files:** `value_model_training.py:build_training_dataframe()`

---

#### **2. Add Explicit Time Discount to Picks**

**Problem:** Future picks rely on vendor time discounting, no independent signal.

**Solution:**
```python
def time_discounted_pick_value(base_value, draft_year, current_year):
    years_away = draft_year - current_year
    discount = 0.90 ** years_away  # 10% annual discount
    return base_value * discount
```

**Impact:** Future picks systematically valued correctly even if vendors misprice.

**Effort:** 15 lines of code, 30 minutes

**Files:** `picks.py:load_pick_value_table()`

---

#### **3. Add Recent Weeks Features (Rolling Windows)**

**Problem:** Season averages hide recent role changes.

**Solution:**
```python
# In player_history.py
def calculate_recent_windows(usage_by_week):
    last_4_weeks_ppg = usage_by_week[-4:].mean()
    last_4_weeks_snap_pct = usage_by_week[-4:].mean()
    ppg_acceleration = last_4_weeks_ppg - usage_by_week[-8:-4].mean()
    return last_4_weeks_ppg, last_4_weeks_snap_pct, ppg_acceleration
```

**Impact:** Model captures breakouts, role changes, hot streaks.

**Effort:** 50 lines of code, 2 hours

**Files:** `player_history.py:build_player_history_features()`

---

#### **4. Fix QB Compression for Superflex**

**Problem:** QB compression hardcoded for 1QB, breaks Superflex leagues.

**Solution:**
```python
def _apply_qb_market_compression(raw_value, position, league_type="1QB"):
    if position != "QB":
        return raw_value

    if league_type == "1QB":
        compression = 0.65  # 35% reduction
    elif league_type == "Superflex":
        compression = 0.95  # 5% reduction
    elif league_type == "2QB":
        compression = 1.20  # 20% boost

    return raw_value * compression
```

**Impact:** Model works for Superflex users (huge market).

**Effort:** 20 lines of code, 1 hour

**Files:** `player_value.py:_apply_qb_market_compression()`

---

#### **5. Add Validation Alerts for Data Completeness**

**Problem:** Silent failures when Sleeper/OTC data incomplete.

**Solution:**
```python
def validate_usage_table(usage_table, expected_weeks=18):
    player_week_counts = defaultdict(int)
    for player in usage_table:
        player_week_counts[player['id']] = player['usage']['games']

    incomplete = [pid for pid, weeks in player_week_counts.items() if weeks < 4]
    if incomplete:
        print(f"WARNING: {len(incomplete)} players with < 4 games")

    if len(usage_table) < 500:
        raise ValueError("Usage table too small - data fetch failed")
```

**Impact:** Catch data failures before model trains on garbage.

**Effort:** 40 lines of code, 1 hour

**Files:** `sleeper_usage.py:write_usage_table_snapshot()`

---

### Tier 2: High Impact, Medium Effort

#### **6. Add Efficiency Metrics (YPC, YPT, Catch Rate)**

**Problem:** Model lacks efficiency signals, can't distinguish talent from volume.

**Solution:**
```python
# In sleeper_usage.py
usage[pid]['yards_per_carry'] = rush_yards / carries if carries > 0 else 0
usage[pid]['yards_per_target'] = rec_yards / targets if targets > 0 else 0
usage[pid]['catch_rate'] = receptions / targets if targets > 0 else 0
usage[pid]['td_rate'] = (rec_tds + rush_tds) / (targets + carries) if (targets + carries) > 0 else 0
```

**Impact:** Model values efficient players higher (better talent signal).

**Effort:** 60 lines of code, 3 hours (+ testing)

**Files:** `sleeper_usage.py:build_usage_map_for_season()`, `player_value.py:_production_component_fixed()`

---

#### **7. Add Injury Context Tracking**

**Problem:** Model treats all missed games equally (injury vs healthy scratch).

**Solution:**
- Scrape injury reports from Sleeper/ESPN
- Flag games missed due to injury
- Calculate "healthy games PPG" separate from "all games PPG"
- Add "injury risk score" (# injuries, severity)

**Impact:** Injured stars (JT, CMC) retain value during recovery.

**Effort:** 150 lines of code, 8 hours (scraping + parsing)

**Files:** New file `injury_tracking.py`, integrate into `player_history.py`

---

#### **8. Add Breakout Detection Signals**

**Problem:** Model undervalues players with sudden opportunity spikes.

**Solution:**
```python
# In player_history.py
def detect_breakout(player_history):
    snap_pct_increase = current_snap_pct - prev_snap_pct
    target_share_increase = current_target_share - prev_target_share

    breakout_score = 0
    if player_age < 24:
        breakout_score += 0.3  # Youth bonus
    if snap_pct_increase > 0.15:
        breakout_score += 0.4  # Big role increase
    if target_share_increase > 0.05:
        breakout_score += 0.3  # Market share gain

    return breakout_score
```

**Impact:** Puka Nacua, Jahmyr Gibbs type breakouts captured earlier.

**Effort:** 80 lines of code, 4 hours

**Files:** `player_history.py:build_player_history_features()`

---

#### **9. Add Probabilistic Pick Projection**

**Problem:** Future 1sts bucketed without accounting for team strength.

**Solution:**
- Track roster age, injury history, wins/losses
- Project pick landing spot distribution (e.g., 30% early, 50% mid, 20% late)
- Value = weighted average of outcomes

```python
def project_pick_distribution(team_roster, season):
    # Analyze roster age, talent, schedule
    # Return {early: 0.3, mid: 0.5, late: 0.2}
    pass

def expected_pick_value(distribution, pick_values):
    return (
        distribution['early'] * pick_values['early'] +
        distribution['mid'] * pick_values['mid'] +
        distribution['late'] * pick_values['late']
    )
```

**Impact:** More accurate future pick valuation.

**Effort:** 200 lines of code, 10 hours

**Files:** New file `pick_projection.py`, integrate into `picks.py`

---

#### **10. Add QB Quality Signal for WR/TE**

**Problem:** WR/TE values don't adjust for QB quality.

**Solution:**
- Maintain QB quality ratings (PFF grade, EPA, etc.)
- Adjust WR/TE values based on QB:
  - Elite QB: 1.2x value
  - Average QB: 1.0x value
  - Backup QB: 0.8x value

```python
QB_QUALITY = {
    "Mahomes": 1.25,
    "Allen": 1.20,
    "Burrow": 1.15,
    # ...
    "Backup QBs": 0.75
}

def adjust_for_qb_quality(wr_value, team, qb_quality_map):
    qb_quality = qb_quality_map.get(team, 1.0)
    return wr_value * qb_quality
```

**Impact:** Garrett Wilson, Davante Adams type QB-dependent values captured.

**Effort:** 100 lines of code, 5 hours (+ manual QB ratings)

**Files:** New file `qb_quality.py`, integrate into `player_value.py`

---

### Tier 3: Medium Impact, High Effort

#### **11. Train Separate Rookie Model**

**Problem:** Rookies have no production history, model undervalues them.

**Solution:**
- Collect college stats (yards, TDs, draft position, combine metrics)
- Train separate model on "pre-NFL features → NFL success"
- Blend rookie model (60%) + draft capital (40%) for rookie values

**Impact:** Bijan, MHJ, Nabers valued correctly pre-NFL.

**Effort:** 500+ lines, 20 hours (data collection + modeling)

**Files:** New file `rookie_model.py`, integrate into `value_model_training.py`

---

#### **12. Add Coaching/Scheme Stability Tracking**

**Problem:** Offensive scheme changes impact player values.

**Solution:**
- Track coaching changes (OC, HC)
- Classify offensive schemes (pass-heavy, run-heavy, zone-blocking, etc.)
- Penalize players in unstable situations (new coach, scheme change)

**Impact:** Captures scheme fit risks (e.g., new OC implements run-heavy offense, WRs lose value).

**Effort:** 300+ lines, 15 hours (manual scheme classification + integration)

**Files:** New file `coaching_tracker.py`, integrate into `player_value.py`

---

#### **13. Build Interactive Value Explorer UI**

**Problem:** Users can't inspect why a player is valued X.

**Solution:**
- Build web UI showing value breakdown:
  - Production score: 0.82
  - Age factor: 0.91
  - Role security: 0.78
  - Investment: 0.65
  - → Raw value: 0.84
  - → Scaled value: 887

**Impact:** Transparency builds trust, users can validate model logic.

**Effort:** Full-stack dev, 40+ hours

**Files:** New Flask routes, React components

---

### Tier 4: Low Impact, Low Effort (Quick Wins)

#### **14. Add "Last Updated" Timestamps**

**Problem:** Can't tell if data is stale.

**Solution:**
```python
# Add to all data files
{
  "last_updated": "2026-03-26T10:30:00Z",
  "source": "sleeper_api",
  "players": [...]
}
```

**Impact:** Users know if values are fresh.

**Effort:** 20 lines, 30 minutes

**Files:** All data writers (`sleeper_usage.py`, `external_values_scraper.py`, etc.)

---

#### **15. Add Player "Tier" Classification**

**Problem:** Users want quick heuristics (elite, starter, bench, waiver).

**Solution:**
```python
def assign_tier(value, position):
    if value > 7000:
        return "Elite"
    elif value > 4000:
        return "Starter"
    elif value > 1500:
        return "Bench"
    else:
        return "Waiver"
```

**Impact:** User-friendly labels.

**Effort:** 30 lines, 1 hour

**Files:** `value_model_training.py:rewrite_value_table_with_model()`

---

#### **16. Add Value Change Tracking**

**Problem:** Can't see if player value is rising or falling.

**Solution:**
```python
# Compare today's value to last week's
value_change = current_value - previous_value
value_change_pct = (current_value - previous_value) / previous_value
```

**Impact:** "Buy low" and "sell high" signals.

**Effort:** 40 lines, 2 hours

**Files:** `player_value_history.py:record_model_value_snapshot()`

---

## Step 5: Quick Wins & Bug Fixes

### **Bug 1: Engine Scale Mismatch (CRITICAL)**

**Location:** `value_model_training.py:build_training_dataframe()`

**Issue:** Engine values (0-999.9) mixed with FC (0-10000+) and DP (0-15000+).

**Fix:**
```python
# Add normalization before training
df['fc_value_norm'] = df['fc_value'] / 10000.0
df['dp_value_norm'] = df['dp_value'] / 15000.0
df['engine_value_norm'] = df['engine_value'] / 999.9

# Update consensus target
df['consensus_value'] = (
    0.50 * df['fc_value'] +
    0.35 * df['dp_value'] +
    0.15 * (df['engine_value_norm'] * 10000)  # Scale engine to match
)
```

---

### **Quick Win 1: Add Time Discount to Picks**

**Location:** `picks.py:load_pick_value_table()`

**Current:** No explicit time decay.

**Fix:**
```python
def apply_time_discount(pick_value, draft_year, current_year):
    years_away = draft_year - current_year
    if years_away <= 0:
        return pick_value
    discount_rate = 0.90  # 10% per year
    return pick_value * (discount_rate ** years_away)

# Apply to all picks
for pick_key, pick_value in pick_table.items():
    draft_year = int(pick_key.split('_')[0])
    pick_table[pick_key] = apply_time_discount(pick_value, draft_year, current_year)
```

---

### **Quick Win 2: Add Validation Checks**

**Location:** `sleeper_usage.py:write_usage_table_snapshot()`

**Current:** Silent failures if Sleeper API returns partial data.

**Fix:**
```python
def validate_usage_table(usage_table):
    if len(usage_table) < 400:
        raise ValueError(f"Usage table too small: {len(usage_table)} players (expected 500+)")

    zero_games = sum(1 for p in usage_table if p['usage']['games'] == 0)
    if zero_games > len(usage_table) * 0.5:
        raise ValueError(f"Too many players with 0 games: {zero_games}")

    print(f"[VALIDATION] Usage table OK: {len(usage_table)} players, {zero_games} with 0 games")

# Call after building usage table
usage_table = build_usage_map_for_season(season, weeks)
validate_usage_table(usage_table)
```

---

### **Quick Win 3: Add Rolling Window Features**

**Location:** `player_history.py:build_player_history_features()`

**Current:** Only season-level and 3-year weighted features.

**Fix:**
```python
# Add recent performance features
def calculate_recent_windows(grp):
    # Last 4 games
    recent_games = grp.tail(4)
    last_4_ppg = recent_games['ppr_ppg'].mean()
    last_4_snap_pct = recent_games['avg_off_snap_pct'].mean()

    # Acceleration (recent vs previous)
    previous_games = grp.iloc[-8:-4] if len(grp) >= 8 else grp.iloc[:-4]
    ppg_acceleration = last_4_ppg - previous_games['ppr_ppg'].mean()

    return last_4_ppg, last_4_snap_pct, ppg_acceleration

# Add to features dataframe
row['last_4_weeks_ppg'] = last_4_ppg
row['last_4_weeks_snap_pct'] = last_4_snap_pct
row['ppg_acceleration'] = ppg_acceleration
```

---

### **Quick Win 4: Fix Superflex QB Compression**

**Location:** `player_value.py:_apply_qb_market_compression()`

**Current:** Hardcoded 1QB compression (35-60%).

**Fix:**
```python
def _apply_qb_market_compression(raw_value, position, league_type="1QB"):
    if position != "QB":
        return raw_value

    # Adjust compression based on league type
    if league_type == "1QB":
        compression = 0.65  # Reduce by 35%
    elif league_type == "Superflex":
        compression = 0.95  # Reduce by 5%
    elif league_type == "2QB":
        compression = 1.20  # Boost by 20%
    else:
        compression = 1.0  # No adjustment

    return raw_value * compression

# Add league_type parameter to build_value_table_for_usage()
def build_value_table_for_usage(league_type="1QB"):
    # ... existing code ...
    compressed = _apply_qb_market_compression(raw_value, position, league_type)
```

---

### **Quick Win 5: Add Efficiency Metrics**

**Location:** `sleeper_usage.py:build_usage_map_for_season()`

**Current:** Only volume metrics (targets, carries, snaps).

**Fix:**
```python
# Add efficiency calculations
usage[pid] = {
    # ... existing volume metrics ...

    # Efficiency metrics
    'yards_per_carry': acc['rush_yards'] / acc['carries'] if acc['carries'] > 0 else 0,
    'yards_per_target': acc['rec_yards'] / acc['targets'] if acc['targets'] > 0 else 0,
    'catch_rate': acc['receptions'] / acc['targets'] if acc['targets'] > 0 else 0,
    'td_rate_per_touch': (acc['rec_tds'] + acc['rush_tds']) / (acc['targets'] + acc['carries'])
                         if (acc['targets'] + acc['carries']) > 0 else 0,
}
```

Then update `player_value.py:_production_component_fixed()` to incorporate efficiency:
```python
# Add efficiency weights
if position == "RB":
    efficiency_boost = (ypc - 4.0) / 2.0  # YPC above 4.0 boosts value
    production = base_production * (1 + efficiency_boost * 0.1)
elif position in ("WR", "TE"):
    efficiency_boost = (catch_rate - 0.65) / 0.20  # Catch rate above 65% boosts value
    production = base_production * (1 + efficiency_boost * 0.08)
```

---

## Summary

### Current System Strengths
1. ✅ Comprehensive production signals (snap %, targets, carries, redzone usage)
2. ✅ Well-designed position-specific age curves
3. ✅ 3-year weighted history (60/30/10 recency)
4. ✅ Investment context (draft capital + contracts)
5. ✅ ML consensus blending 3 sources (FC 50%, DP 35%, Engine 15%)

### Critical Issues
1. ⚠️ **Value scale mismatch:** Engine (0-999.9) vs FC (0-10000+) creates ML training instability
2. ⚠️ **QB compression too aggressive:** 35-60% reduction breaks QB value in 1QB, and completely breaks Superflex
3. ⚠️ **Pick bucketing loses precision:** Future 1sts lumped as "early/mid/late" misses 1.01 vs 1.09 gap
4. ⚠️ **No time discount:** Future picks rely entirely on vendor values, no independent decay formula
5. ⚠️ **Rookie undervaluation:** No historical features, model leans only on draft capital

### Top 5 Improvements (by impact/effort)
1. **Normalize value scales** (1 hour) → Fixes ML training stability
2. **Add time discount to picks** (30 min) → Future picks properly valued
3. **Add rolling window features** (2 hours) → Captures breakouts, role changes
4. **Fix Superflex QB compression** (1 hour) → Model works for Superflex leagues
5. **Add validation alerts** (1 hour) → Catches data failures before training

### Total Estimated Effort: Tier 1 (Quick Wins)
- **5 critical fixes:** ~6 hours total
- **Impact:** Fixes core issues, makes model production-ready

---

**END OF ANALYSIS**
