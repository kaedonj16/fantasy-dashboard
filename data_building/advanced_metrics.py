"""
Advanced player efficiency metrics for dynasty valuation.

Calculates position-specific efficiency metrics from usage data:
- WR/TE: Yards per target, catch rate, yards per reception, YPRR proxy
- RB: Yards per carry, yards per touch, broken tackle proxy
- QB: Yards per attempt, completion %, TD rate, INT rate

These metrics inform the breakout detection algorithm and can be displayed in the UI.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

from dashboard_services.api import get_nfl_state
from dashboard_services.db import get_conn

# Map PFF/non-standard position codes to the canonical fantasy set
# (QB / RB / WR / TE).  Anything not in this map is returned as-is.
_POS_NORM: Dict[str, str] = {
    "HB": "RB", "FB": "RB",
    "SE": "WR", "FL": "WR",
}


def _normalize_position(pos: Optional[str]) -> Optional[str]:
    """Return the canonical fantasy position for a raw position string."""
    if not pos:
        return pos
    return _POS_NORM.get(pos.upper(), pos.upper())


def init_advanced_metrics_db():
    """
    Create player_advanced_metrics table if it doesn't exist.

    Stores calculated efficiency metrics per player per date.
    """
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_advanced_metrics (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                as_of_date DATE NOT NULL,
                position VARCHAR(5),

                -- Receiving efficiency (WR/TE/RB)
                yards_per_target NUMERIC,
                catch_rate NUMERIC,
                yards_per_reception NUMERIC,
                target_quality_score NUMERIC,

                -- Rushing efficiency (RB)
                yards_per_carry NUMERIC,
                yards_per_touch NUMERIC,
                rush_td_rate NUMERIC,

                -- Passing efficiency (QB)
                yards_per_attempt NUMERIC,
                completion_pct NUMERIC,
                td_rate NUMERIC,
                int_rate NUMERIC,

                -- Usage metrics (all positions)
                snap_share NUMERIC,
                opportunity_share NUMERIC,
                red_zone_usage NUMERIC,

                -- Role indicators
                role_score NUMERIC,
                usage_trend NUMERIC,
                efficiency_trend NUMERIC,

                UNIQUE(player_id, as_of_date)
            );

            CREATE INDEX IF NOT EXISTS idx_adv_metrics_player_date
                ON player_advanced_metrics (player_id, as_of_date DESC);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_date_pos
                ON player_advanced_metrics (as_of_date, position);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_role_score
                ON player_advanced_metrics (as_of_date, role_score DESC);
        """)

        # Add season column if it doesn't exist yet (migration)
        conn.execute("""
            ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS season INTEGER;
        """)

        # Backfill season from as_of_date for any rows that are missing it.
        # Regular season runs Sep–Dec of year Y and Jan of year Y+1, so:
        #   Jan–Feb  → season = year - 1  (e.g. 2026-01 → 2025 season)
        #   Sep–Dec  → season = year      (e.g. 2025-10 → 2025 season)
        # Mar–Aug rows won't exist (cron skips offseason), but we handle them
        # conservatively as the prior season.
        conn.execute("""
            UPDATE player_advanced_metrics
            SET season = CASE
                WHEN EXTRACT(MONTH FROM as_of_date) <= 2
                    THEN EXTRACT(YEAR FROM as_of_date)::INTEGER - 1
                ELSE EXTRACT(YEAR FROM as_of_date)::INTEGER
            END
            WHERE season IS NULL;
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_player_season
                ON player_advanced_metrics (player_id, season, as_of_date DESC);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_season
                ON player_advanced_metrics (season, as_of_date DESC);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_player_seasons_notnull
                ON player_advanced_metrics (player_id, season DESC)
                WHERE season IS NOT NULL;
        """)

        # Add rookie evaluation columns (safe migration - all nullable)
        _add_rookie_eval_columns(conn)


def _add_rookie_eval_columns(conn) -> None:
    """
    Add rookie_eval_* columns to player_advanced_metrics if they don't exist.

    All columns are nullable so existing NFL-player rows are unaffected.
    Safe to call repeatedly (ADD COLUMN IF NOT EXISTS).
    """
    conn.execute("""
        ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS rookie_eval_routes_run         NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_yprr               NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_tprr               NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_yac_per_att        NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_mtf_per_att        NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_explosive_run_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_adjusted_comp_pct  NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_twp_rate           NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_player_level_sos   NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_perf_vs_top_def    NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_true_early_declare BOOLEAN,
            ADD COLUMN IF NOT EXISTS rookie_eval_draft_class_year   INTEGER,
            ADD COLUMN IF NOT EXISTS rookie_eval_completeness       NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_prospect_score     NUMERIC,
            ADD COLUMN IF NOT EXISTS rookie_eval_is_rookie          BOOLEAN;
    """)

    # PFF feed columns (NFL) for player evaluation and modal display
    conn.execute("""
        ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS yards_after_catch NUMERIC,
            ADD COLUMN IF NOT EXISTS yards_after_catch_per_reception NUMERIC,
            ADD COLUMN IF NOT EXISTS avg_depth_of_target NUMERIC,
            ADD COLUMN IF NOT EXISTS contested_catch_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS avoided_tackles NUMERIC,
            ADD COLUMN IF NOT EXISTS drop_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS slot_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS wide_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS inline_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS pass_block_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS grades_offense NUMERIC,
            ADD COLUMN IF NOT EXISTS grades_pass_block NUMERIC,
            ADD COLUMN IF NOT EXISTS explosive_runs_10_plus NUMERIC,
            ADD COLUMN IF NOT EXISTS breakaway_percentage NUMERIC,
            ADD COLUMN IF NOT EXISTS elusive_rating NUMERIC,
            ADD COLUMN IF NOT EXISTS pff_rushing_grade NUMERIC,
            ADD COLUMN IF NOT EXISTS pff_passing_grade NUMERIC,
            ADD COLUMN IF NOT EXISTS big_time_throw_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS adjusted_completion_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS pressure_to_sack_rate NUMERIC,
            ADD COLUMN IF NOT EXISTS nfl_passer_rating NUMERIC;
    """)


def _extract_metric_value(metrics: Dict, metric_name: str):
    """Safely pull the scalar value from a metric payload dict."""
    entry = metrics.get(metric_name)
    if not isinstance(entry, dict):
        return None
    return entry.get("value")


def merge_rookie_profiles_to_advanced_metrics(
    profiles: List[Dict],
    as_of_date: str,
    conn=None,
) -> Dict[str, int]:
    """
    Upsert rookie evaluation metrics from profile snapshots into
    player_advanced_metrics so they are available as model-training features.

    For each profile:
    - Extracts the latest-season values of each rookie_eval_* metric.
    - If a row already exists (player_id, as_of_date), updates only the
      rookie_eval_* columns (leaves NFL-side columns untouched).
    - If no row exists yet (prospect not yet drafted / in usage table),
      inserts a minimal row with position + rookie_eval columns; all
      NFL-side metrics remain NULL.

    Processes in batches of 25 to keep memory low (Render constraint).

    Args:
        profiles:   List of rookie profile dicts from run_rookie_evaluation_pipeline.
        as_of_date: ISO date string (YYYY-MM-DD).
        conn:       Optional existing psycopg connection; acquires one if None.

    Returns:
        {"updated": n, "inserted": n, "skipped": n}
    """
    if not profiles:
        return {"updated": 0, "inserted": 0, "skipped": 0}

    def _run(db_conn):
        updated = inserted = skipped = 0
        batch_size = 25

        for i in range(0, len(profiles), batch_size):
            batch = profiles[i : i + batch_size]
            for profile in batch:
                player_id = (
                    profile.get("sleeper_id")
                    or profile.get("player_id")
                )
                if not player_id:
                    skipped += 1
                    continue

                rp = profile.get("rookie_profile") or {}
                metrics = rp.get("metrics") or {}

                # Pull latest values for each rookie_eval column
                rv = {
                    "routes_run":         _extract_metric_value(metrics, "routes_run"),
                    "yprr":               _extract_metric_value(metrics, "yprr"),
                    "tprr":               _extract_metric_value(metrics, "tprr"),
                    "yac_per_att":        _extract_metric_value(metrics, "yac_per_att"),
                    "mtf_per_att":        _extract_metric_value(metrics, "mtf_per_att"),
                    "explosive_run_rate": _extract_metric_value(metrics, "explosive_run_rate"),
                    "adjusted_comp_pct":  _extract_metric_value(metrics, "adjusted_comp_pct"),
                    "twp_rate":           _extract_metric_value(metrics, "twp_rate"),
                    "player_level_sos":   _extract_metric_value(metrics, "player_level_sos"),
                    "perf_vs_top_def":    _extract_metric_value(metrics, "performance_vs_top_defenses"),
                    "true_early_declare": _extract_metric_value(metrics, "true_early_declare"),
                }
                draft_class_year = profile.get("draft_class_year") or rp.get("draft_class_year")
                completeness = rp.get("completeness")
                prospect_score = profile.get("prospect_score")
                position = (profile.get("position") or "").upper() or None

                # Check if row already exists
                row = db_conn.execute(
                    "SELECT 1 FROM player_advanced_metrics WHERE player_id = %s AND as_of_date = %s",
                    (str(player_id), as_of_date),
                ).fetchone()

                if row:
                    db_conn.execute(
                        """
                        UPDATE player_advanced_metrics SET
                            rookie_eval_routes_run         = %s,
                            rookie_eval_yprr               = %s,
                            rookie_eval_tprr               = %s,
                            rookie_eval_yac_per_att        = %s,
                            rookie_eval_mtf_per_att        = %s,
                            rookie_eval_explosive_run_rate = %s,
                            rookie_eval_adjusted_comp_pct  = %s,
                            rookie_eval_twp_rate           = %s,
                            rookie_eval_player_level_sos   = %s,
                            rookie_eval_perf_vs_top_def    = %s,
                            rookie_eval_true_early_declare = %s,
                            rookie_eval_draft_class_year   = %s,
                            rookie_eval_completeness       = %s,
                            rookie_eval_prospect_score     = %s,
                            rookie_eval_is_rookie          = TRUE
                        WHERE player_id = %s AND as_of_date = %s
                        """,
                        (
                            rv["routes_run"], rv["yprr"], rv["tprr"],
                            rv["yac_per_att"], rv["mtf_per_att"], rv["explosive_run_rate"],
                            rv["adjusted_comp_pct"], rv["twp_rate"],
                            rv["player_level_sos"], rv["perf_vs_top_def"],
                            rv["true_early_declare"],
                            draft_class_year, completeness, prospect_score,
                            str(player_id), as_of_date,
                        ),
                    )
                    updated += 1
                else:
                    # INSERT minimal row: prospects not yet in usage table
                    db_conn.execute(
                        """
                        INSERT INTO player_advanced_metrics (
                            player_id, as_of_date, position,
                            rookie_eval_routes_run, rookie_eval_yprr, rookie_eval_tprr,
                            rookie_eval_yac_per_att, rookie_eval_mtf_per_att,
                            rookie_eval_explosive_run_rate, rookie_eval_adjusted_comp_pct,
                            rookie_eval_twp_rate, rookie_eval_player_level_sos,
                            rookie_eval_perf_vs_top_def, rookie_eval_true_early_declare,
                            rookie_eval_draft_class_year, rookie_eval_completeness,
                            rookie_eval_prospect_score, rookie_eval_is_rookie
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE
                        )
                        ON CONFLICT (player_id, as_of_date) DO UPDATE SET
                            rookie_eval_routes_run         = EXCLUDED.rookie_eval_routes_run,
                            rookie_eval_yprr               = EXCLUDED.rookie_eval_yprr,
                            rookie_eval_tprr               = EXCLUDED.rookie_eval_tprr,
                            rookie_eval_yac_per_att        = EXCLUDED.rookie_eval_yac_per_att,
                            rookie_eval_mtf_per_att        = EXCLUDED.rookie_eval_mtf_per_att,
                            rookie_eval_explosive_run_rate = EXCLUDED.rookie_eval_explosive_run_rate,
                            rookie_eval_adjusted_comp_pct  = EXCLUDED.rookie_eval_adjusted_comp_pct,
                            rookie_eval_twp_rate           = EXCLUDED.rookie_eval_twp_rate,
                            rookie_eval_player_level_sos   = EXCLUDED.rookie_eval_player_level_sos,
                            rookie_eval_perf_vs_top_def    = EXCLUDED.rookie_eval_perf_vs_top_def,
                            rookie_eval_true_early_declare = EXCLUDED.rookie_eval_true_early_declare,
                            rookie_eval_draft_class_year   = EXCLUDED.rookie_eval_draft_class_year,
                            rookie_eval_completeness       = EXCLUDED.rookie_eval_completeness,
                            rookie_eval_prospect_score     = EXCLUDED.rookie_eval_prospect_score,
                            rookie_eval_is_rookie          = TRUE
                        """,
                        (
                            str(player_id), as_of_date, position,
                            rv["routes_run"], rv["yprr"], rv["tprr"],
                            rv["yac_per_att"], rv["mtf_per_att"], rv["explosive_run_rate"],
                            rv["adjusted_comp_pct"], rv["twp_rate"],
                            rv["player_level_sos"], rv["perf_vs_top_def"],
                            rv["true_early_declare"],
                            draft_class_year, completeness, prospect_score,
                        ),
                    )
                    inserted += 1

        return {"updated": updated, "inserted": inserted, "skipped": skipped}

    if conn is not None:
        return _run(conn)

    with get_conn() as db_conn:
        return _run(db_conn)


def calculate_receiving_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate receiving efficiency metrics."""
    targets = usage.get("avg_targets", 0) or 0
    receptions = usage.get("avg_receptions", 0) or 0
    rec_yards = usage.get("avg_rec_yards", 0) or 0
    rec_tds = usage.get("avg_rec_tds", 0) or 0

    yards_per_target = rec_yards / targets if targets > 0 else None
    catch_rate = receptions / targets if targets > 0 else None
    yards_per_reception = rec_yards / receptions if receptions > 0 else None

    # Target quality score: combines volume + efficiency
    # High target volume with good efficiency = elite
    target_quality = None
    if targets > 0 and yards_per_target is not None:
        target_quality = (targets * 2) + (yards_per_target * 1.5)
        if rec_tds > 0:
            target_quality += (rec_tds * 15)  # TD boost

    return {
        "yards_per_target": yards_per_target,
        "catch_rate": catch_rate,
        "yards_per_reception": yards_per_reception,
        "target_quality_score": target_quality,
    }


def calculate_rushing_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate rushing efficiency metrics."""
    carries = usage.get("avg_carries", 0) or 0
    rush_yards = usage.get("avg_rush_yards", 0) or 0
    rush_tds = usage.get("avg_rush_tds", 0) or 0
    targets = usage.get("avg_targets", 0) or 0
    receptions = usage.get("avg_receptions", 0) or 0

    yards_per_carry = rush_yards / carries if carries > 0 else None

    # Yards per touch: total scrimmage yards / total touches
    total_touches = carries + receptions
    total_yards = rush_yards + usage.get("avg_rec_yards", 0)
    yards_per_touch = total_yards / total_touches if total_touches > 0 else None

    # Rush TD rate
    rush_td_rate = rush_tds / carries if carries > 0 else None

    return {
        "yards_per_carry": yards_per_carry,
        "yards_per_touch": yards_per_touch,
        "rush_td_rate": rush_td_rate,
    }


def calculate_passing_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate passing efficiency metrics."""
    pass_att = usage.get("avg_pass_att", 0) or 0
    pass_cmp = usage.get("avg_pass_cmp", 0) or 0
    pass_yds = usage.get("avg_pass_yds", 0) or 0
    pass_tds = usage.get("avg_pass_tds", 0) or 0
    pass_int = usage.get("avg_pass_int", 0) or 0

    yards_per_attempt = pass_yds / pass_att if pass_att > 0 else None
    completion_pct = (pass_cmp / pass_att * 100) if pass_att > 0 else None
    td_rate = (pass_tds / pass_att * 100) if pass_att > 0 else None
    int_rate = (pass_int / pass_att * 100) if pass_att > 0 else None

    return {
        "yards_per_attempt": yards_per_attempt,
        "completion_pct": completion_pct,
        "td_rate": td_rate,
        "int_rate": int_rate,
    }


def calculate_usage_metrics(usage: Dict[str, float], position: str) -> Dict[str, Optional[float]]:
    """Calculate usage and opportunity metrics."""
    snap_share = usage.get("avg_off_snap_pct", 0) or 0

    # Opportunity share: targets + carries normalized by games
    targets = usage.get("avg_targets", 0) or 0
    carries = usage.get("avg_carries", 0) or 0
    opportunity_share = targets + carries  # per-game touches

    # Red zone usage
    rz_targets = usage.get("rec_rz_tgt_pg", 0) or 0
    rz_carries = usage.get("rush_rz_att_pg", 0) or 0
    red_zone_usage = rz_targets + rz_carries

    return {
        "snap_share": snap_share if snap_share > 0 else None,
        "opportunity_share": opportunity_share if opportunity_share > 0 else None,
        "red_zone_usage": red_zone_usage if red_zone_usage > 0 else None,
    }


from typing import Dict, Optional


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe(v: Optional[float], default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm(
    value: float,
    low: float,
    high: float,
    *,
    cap_low: bool = True,
    cap_high: bool = True,
) -> float:
    if high <= low:
        return 0.0

    x = (value - low) / (high - low)

    if cap_low:
        x = max(0.0, x)
    if cap_high:
        x = min(1.0, x)

    return x


def _sample_confidence(games: float) -> float:
    """
    Soft confidence multiplier.
    Keeps small samples from dominating, but does not crush ceilings.
    """
    if games <= 0:
        return 0.0
    return _clip(games / 10.0, 0.35, 1.0)


def calculate_role_score(
    usage: Dict[str, float],
    position: str,
    receiving_metrics: Dict[str, Optional[float]],
    rushing_metrics: Dict[str, Optional[float]],
    passing_metrics: Dict[str, Optional[float]],
) -> Optional[float]:
    """
    Calculate a role/opportunity score from 0-100.

    What it measures:
    - How much the player is actually used in their offense
    - Volume first, efficiency second
    - High-value usage (red zone) matters
    - Position-specific scoring
    - Nonlinear dropoff so middling / low-usage players fall faster
    """
    snap_pct = _safe(usage.get("avg_off_snap_pct"))
    games = _safe(usage.get("games"))

    if games <= 0 or snap_pct <= 0:
        return None

    avg_targets = _safe(usage.get("avg_targets"))
    avg_carries = _safe(usage.get("avg_carries"))
    rec_rz_tgt_pg = _safe(usage.get("rec_rz_tgt_pg"))
    rush_rz_att_pg = _safe(usage.get("rush_rz_att_pg"))

    ypt = _safe(receiving_metrics.get("yards_per_target"))
    catch_rate = _safe(receiving_metrics.get("catch_rate"))
    rec_td_rate = _safe(receiving_metrics.get("td_rate"))

    ypc = _safe(rushing_metrics.get("yards_per_carry"))
    rush_td_rate = _safe(rushing_metrics.get("td_rate"))

    pass_att = _safe(usage.get("avg_pass_att"))
    qb_rush_att = _safe(usage.get("avg_carries"))
    ypa = _safe(passing_metrics.get("yards_per_attempt"))
    pass_td_rate = _safe(passing_metrics.get("td_rate"))
    int_rate = _safe(passing_metrics.get("int_rate"))

    sample_mult = _sample_confidence(games)

    if position == "QB":
        snap_score = _norm(snap_pct, 55, 100) ** 1.05
        att_score = _norm(pass_att, 18, 40) ** 1.20
        rush_score = _norm(qb_rush_att, 0, 8) ** 1.15
        ypa_score = _norm(ypa, 5.5, 8.8)
        td_score = _norm(pass_td_rate, 0.02, 0.08)
        int_penalty = _norm(int_rate, 0.01, 0.05) if int_rate > 0 else 0.0

        base = (
            snap_score * 0.20 +
            att_score * 0.42 +
            rush_score * 0.12 +
            ypa_score * 0.14 +
            td_score * 0.17
        )
        base -= int_penalty * 0.05

    elif position == "RB":
        weighted_opps = avg_carries + (avg_targets * 1.6)
        high_value_usage = rush_rz_att_pg + (rec_rz_tgt_pg * 1.25)

        snap_score = _norm(snap_pct, 20, 85) ** 1.10
        opp_score = _norm(weighted_opps, 5, 24) ** 1.35
        carry_score = _norm(avg_carries, 3, 20) ** 1.20
        target_score = _norm(avg_targets, 1, 7) ** 1.20
        rz_score = _norm(high_value_usage, 0.1, 4.0) ** 1.15

        ypc_score = _norm(ypc, 3.5, 5.5)
        ypt_score = _norm(ypt, 4.5, 9.0)
        rush_td_score = _norm(rush_td_rate, 0.01, 0.06) if rush_td_rate > 0 else 0.0
        rec_td_score = _norm(rec_td_rate, 0.02, 0.12) if rec_td_rate > 0 else 0.0

        base = (
            snap_score * 0.18 +
            opp_score * 0.30 +
            carry_score * 0.16 +
            target_score * 0.14 +
            rz_score * 0.14 +
            ypc_score * 0.04 +
            ypt_score * 0.02 +
            rush_td_score * 0.015 +
            rec_td_score * 0.005
        )

    elif position == "WR":
        snap_score = _norm(snap_pct, 35, 95) ** 1.10
        target_score = _norm(avg_targets, 2.5, 11.5) ** 1.35
        rz_score = _norm(rec_rz_tgt_pg, 0.1, 2.5) ** 1.20
        catch_score = _norm(catch_rate, 0.50, 0.78)
        ypt_score = _norm(ypt, 6.0, 11.0)

        base = (
            snap_score * 0.22 +
            target_score * 0.42 +
            rz_score * 0.16 +
            catch_score * 0.08 +
            ypt_score * 0.12
        )

    elif position == "TE":
        snap_score = _norm(snap_pct, 35, 90) ** 1.10
        target_score = _norm(avg_targets, 2.0, 8.5) ** 1.30
        rz_score = _norm(rec_rz_tgt_pg, 0.05, 1.8) ** 1.20
        catch_score = _norm(catch_rate, 0.55, 0.82)
        ypt_score = _norm(ypt, 5.5, 9.5)

        base = (
            snap_score * 0.24 +
            target_score * 0.38 +
            rz_score * 0.18 +
            catch_score * 0.10 +
            ypt_score * 0.10
        )

    else:
        return None

    base = _clip(base, 0.0, 1.0)

    # keep the dropoff
    base = base ** 1.25

    # rescale more generously so elite/great scores are reachable
    base = base / 0.68

    base = _clip(base, 0.0, 1.0)

    score = base * 100.0

    # very light sample effect
    score *= (0.94 + 0.06 * sample_mult)

    return round(_clip(score, 0.0, 100.0), 1)


# ===========================================================================
# Role score v2 — team-relative shares, opportunity-only, position percentile
# ===========================================================================
#
# Why a rewrite (see calculate_role_score above for v1):
#   * v1 scored raw per-game *volume* against absolute thresholds, so a back on
#     a pass-heavy team looked identical to one on a run-heavy team — it never
#     measured share of the team's opportunities.
#   * v1 baked efficiency (ypt/ypc/catch rate/TD) into "role", conflating "is he
#     featured" with "is he good". Those live in the dedicated efficiency
#     metrics; role should be a near-orthogonal opportunity axis.
#   * v1's snap term was effectively dead: avg_off_snap_pct is a 0-1 fraction but
#     v1 normalised it against 35-95, so _norm(0.85, 35, 95) clipped to 0.
#
# v2 is a two-pass batch computation (needs the whole cohort), exposed via
# finalize_role_scores_v2():
#   Pass 1 — a raw 0-1 opportunity index from team-relative shares.
#   Pass 2 — map each index to its percentile within the position cohort.
# Toggle with the ROLE_SCORE_V2 env var (default on); v1 stays reachable for A/B.

# Min qualified players in a position cohort before percentile ranking is
# trustworthy; below this we fall back to an absolute scaling of the index.
_ROLE_MIN_COHORT = 8
# Index value treated as "elite" when the cohort is too small to percentile.
_ROLE_ABS_REFERENCE = 0.62
# A player needs this many games to earn full sample confidence.
_ROLE_FULL_SAMPLE_GAMES = 4.0
# Qualification floor for the *reference* distribution (keeps small-sample flukes
# from defining the top of the cohort). Players below it are still scored.
_ROLE_QUAL_GAMES = 4.0
_ROLE_QUAL_SNAP = 0.30


def use_role_score_v2() -> bool:
    """v2 is the default; set ROLE_SCORE_V2=0 to fall back to the v1 formula."""
    return os.getenv("ROLE_SCORE_V2", "1").strip().lower() not in ("0", "false", "no", "")


def build_team_opportunity_context(usage_table: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Sum per-game opportunities by team so each player's share can be derived.
    usage_table entries are {id, team, position, usage:{...}}; only players with
    games > 0 contribute. Returns {team: {targets, carries, rz_tgt, rz_rush}}.
    """
    ctx: Dict[str, Dict[str, float]] = {}
    for p in usage_table:
        team = p.get("team")
        usage = p.get("usage") or {}
        if not team or (usage.get("games") or 0) <= 0:
            continue
        agg = ctx.setdefault(team, {"targets": 0.0, "carries": 0.0, "rz_tgt": 0.0, "rz_rush": 0.0})
        agg["targets"] += _safe(usage.get("avg_targets"))
        agg["carries"] += _safe(usage.get("avg_carries"))
        agg["rz_tgt"]  += _safe(usage.get("rec_rz_tgt_pg"))
        agg["rz_rush"] += _safe(usage.get("rush_rz_att_pg"))
    return ctx


def _share(part: float, whole: float) -> float:
    return _clip(part / whole, 0.0, 1.0) if whole > 0 else 0.0


def role_opportunity_index(
    usage: Dict[str, float],
    position: str,
    team_ctx: Dict[str, float],
) -> Optional[float]:
    """
    Pass 1: a 0-1 opportunity index from team-relative shares (no efficiency).
    team_ctx is the entry from build_team_opportunity_context for this player's
    team. Returns None for non-skill positions or players who never played.
    """
    games = _safe(usage.get("games"))
    if games <= 0:
        return None

    snap = _clip(_safe(usage.get("avg_off_snap_pct")), 0.0, 1.0)

    # Receiving share: prefer Footballguys target_share (true team share);
    # fall back to deriving it from the team aggregate.
    tshare = _safe(usage.get("target_share"))
    if tshare <= 0:
        tshare = _share(_safe(usage.get("avg_targets")), _safe(team_ctx.get("targets")))
    tshare = _clip(tshare, 0.0, 1.0)

    rz_tgt_share  = _share(_safe(usage.get("rec_rz_tgt_pg")),  _safe(team_ctx.get("rz_tgt")))
    rz_rush_share = _share(_safe(usage.get("rush_rz_att_pg")), _safe(team_ctx.get("rz_rush")))

    if position == "WR":
        # Alpha / slot / deep / RZ specialist all reachable; snap (0.28) keeps
        # lower-target-share field-stretchers from cratering until air yards land.
        idx = 0.50 * tshare + 0.28 * snap + 0.22 * rz_tgt_share

    elif position == "TE":
        # Snap deliberately low — TE snaps include blocking, which is not a
        # fantasy role. Receiving + RZ involvement carry the score.
        idx = 0.55 * tshare + 0.27 * rz_tgt_share + 0.18 * snap

    elif position == "RB":
        # PPR-weighted dual role: the 1.7x target premium lets pass-catching
        # backs register, while rush + goal-line share reward early-down bellcows.
        rshare = _share(_safe(usage.get("avg_carries")), _safe(team_ctx.get("carries")))
        core   = _clip(rshare + 1.7 * tshare, 0.0, 1.0)
        idx = 0.46 * core + 0.20 * rz_rush_share + 0.18 * snap + 0.16 * rz_tgt_share

    elif position == "QB":
        # No "share" at QB — workload + dual-threat, ranked. Rushing is additive
        # upside (0.18) so pocket passers are not penalised: pass + snap = 0.72.
        pass_vol = _norm(_safe(usage.get("avg_pass_att")), 18, 42)
        rush_vol = _norm(_safe(usage.get("avg_carries")), 0, 9)
        rz_vol   = _norm(_safe(usage.get("rush_rz_att_pg")), 0, 2.0)
        idx = 0.42 * pass_vol + 0.30 * snap + 0.18 * rush_vol + 0.10 * rz_vol

    else:
        return None

    return _clip(idx, 0.0, 1.0)


def _percentile_of(sorted_ref: List[float], x: float) -> float:
    """Midrank percentile (0-100) of x within an ascending reference list."""
    n = len(sorted_ref)
    if n == 0:
        return 0.0
    import bisect
    lo = bisect.bisect_left(sorted_ref, x)
    hi = bisect.bisect_right(sorted_ref, x)
    # midpoint of the equal-value band gives ties a fair shared rank
    return ((lo + hi) / 2.0) / n * 100.0


def finalize_role_scores_v2(
    metrics_list: List[Dict[str, Any]],
    usage_table: List[Dict[str, Any]],
) -> None:
    """
    Overwrite each metrics dict's "role_score" with the v2 percentile score.

    No-op (leaves the v1 values from calculate_player_metrics in place) when
    ROLE_SCORE_V2 is disabled. Mutates metrics_list in place.
    """
    if not use_role_score_v2():
        return

    team_ctx_map = build_team_opportunity_context(usage_table)
    usage_by_id = {str(p.get("id")): p for p in usage_table}

    # Pass 1: raw opportunity index per player.
    raw_by_id: Dict[str, float] = {}
    games_by_id: Dict[str, float] = {}
    ref_by_pos: Dict[str, List[float]] = {}
    for m in metrics_list:
        pid = str(m.get("player_id"))
        position = m.get("position")
        entry = usage_by_id.get(pid)
        if entry is None:
            continue
        usage = entry.get("usage") or {}
        team_ctx = team_ctx_map.get(entry.get("team"), {})
        idx = role_opportunity_index(usage, position, team_ctx)
        if idx is None:
            continue
        raw_by_id[pid] = idx
        games = _safe(usage.get("games"))
        games_by_id[pid] = games
        snap = _safe(usage.get("avg_off_snap_pct"))
        # Build the reference distribution from qualified players only.
        if games >= _ROLE_QUAL_GAMES and snap >= _ROLE_QUAL_SNAP:
            ref_by_pos.setdefault(position, []).append(idx)

    for refs in ref_by_pos.values():
        refs.sort()

    # Pass 2: percentile within position (absolute fallback for thin cohorts),
    # lightly shrunk toward 0 for small samples so week-1 flukes can't top a cohort.
    for m in metrics_list:
        pid = str(m.get("player_id"))
        if pid not in raw_by_id:
            continue
        position = m.get("position")
        idx = raw_by_id[pid]
        refs = ref_by_pos.get(position, [])
        if len(refs) >= _ROLE_MIN_COHORT:
            score = _percentile_of(refs, idx)
        else:
            score = _clip(idx / _ROLE_ABS_REFERENCE, 0.0, 1.0) * 100.0
        conf = _clip(games_by_id.get(pid, 0.0) / _ROLE_FULL_SAMPLE_GAMES, 0.0, 1.0)
        m["role_score"] = round(_clip(score * conf, 0.0, 100.0), 1)


def calculate_player_metrics(
        player_id: str,
        usage: Dict[str, float],
        position: str,
) -> Dict[str, Any]:
    """
    Calculate all advanced metrics for a single player.

    Returns dict with all metrics ready for database insertion.
    """
    receiving = calculate_receiving_metrics(usage)
    rushing = calculate_rushing_metrics(usage)
    passing = calculate_passing_metrics(usage)
    usage_metrics = calculate_usage_metrics(usage, position)
    role_score = calculate_role_score(usage, position, receiving, rushing, passing)

    return {
        "player_id": player_id,
        "position": position,
        **receiving,
        **rushing,
        **passing,
        **usage_metrics,
        "role_score": role_score,
        "usage_trend": None,  # Will be calculated from historical data
        "efficiency_trend": None,  # Will be calculated from historical data
    }


def save_metrics_snapshot(metrics_list: List[Dict[str, Any]], as_of_date: str, season: Optional[int] = None):
    """
    Save calculated metrics to database for a specific date.

    Args:
        metrics_list: List of metric dicts from calculate_player_metrics()
        as_of_date: Date string (YYYY-MM-DD)
        season: NFL season year (e.g. 2025). If None, inferred from as_of_date.
    """
    init_advanced_metrics_db()

    if season is None:
        from datetime import datetime as _dt
        _d = _dt.strptime(as_of_date, "%Y-%m-%d")
        season = _d.year - 1 if _d.month <= 2 else _d.year

    with get_conn() as conn:
        for metrics in metrics_list:
            # Upsert: update if exists, insert if not
            conn.execute("""
                INSERT INTO player_advanced_metrics (
                    player_id, as_of_date, season, position,
                    yards_per_target, catch_rate, yards_per_reception, target_quality_score,
                    yards_per_carry, yards_per_touch, rush_td_rate,
                    yards_per_attempt, completion_pct, td_rate, int_rate,
                    snap_share, opportunity_share, red_zone_usage,
                    role_score, usage_trend, efficiency_trend
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (player_id, as_of_date)
                DO UPDATE SET
                    season = EXCLUDED.season,
                    position = EXCLUDED.position,
                    yards_per_target = EXCLUDED.yards_per_target,
                    catch_rate = EXCLUDED.catch_rate,
                    yards_per_reception = EXCLUDED.yards_per_reception,
                    target_quality_score = EXCLUDED.target_quality_score,
                    yards_per_carry = EXCLUDED.yards_per_carry,
                    yards_per_touch = EXCLUDED.yards_per_touch,
                    rush_td_rate = EXCLUDED.rush_td_rate,
                    yards_per_attempt = EXCLUDED.yards_per_attempt,
                    completion_pct = EXCLUDED.completion_pct,
                    td_rate = EXCLUDED.td_rate,
                    int_rate = EXCLUDED.int_rate,
                    snap_share = EXCLUDED.snap_share,
                    opportunity_share = EXCLUDED.opportunity_share,
                    red_zone_usage = EXCLUDED.red_zone_usage,
                    role_score = EXCLUDED.role_score,
                    usage_trend = EXCLUDED.usage_trend,
                    efficiency_trend = EXCLUDED.efficiency_trend
            """, (
                metrics["player_id"], as_of_date, season, metrics["position"],
                metrics.get("yards_per_target"), metrics.get("catch_rate"),
                metrics.get("yards_per_reception"), metrics.get("target_quality_score"),
                metrics.get("yards_per_carry"), metrics.get("yards_per_touch"),
                metrics.get("rush_td_rate"),
                metrics.get("yards_per_attempt"), metrics.get("completion_pct"),
                metrics.get("td_rate"), metrics.get("int_rate"),
                metrics.get("snap_share"), metrics.get("opportunity_share"),
                metrics.get("red_zone_usage"),
                metrics.get("role_score"), metrics.get("usage_trend"),
                metrics.get("efficiency_trend")
            ))

    print(f"[advanced_metrics] Saved {len(metrics_list)} player metrics for {as_of_date} (season {season})")


def calculate_trends(player_id: str, current_date: str, lookback_days: int = 14) -> Dict[str, Optional[float]]:
    """
    Calculate usage and efficiency trends by comparing current metrics to previous period.

    Returns:
        {
            "usage_trend": % change in opportunity_share over lookback period
            "efficiency_trend": % change in role_score over lookback period
        }
    """
    from datetime import datetime, timedelta

    current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    lookback_dt = current_dt - timedelta(days=lookback_days)
    lookback_str = lookback_dt.strftime("%Y-%m-%d")

    with get_conn() as conn:
        # Get current metrics
        current = conn.execute("""
            SELECT opportunity_share, role_score
            FROM player_advanced_metrics
            WHERE player_id = %s AND as_of_date = %s
        """, (player_id, current_date)).fetchone()

        # Get previous metrics
        previous = conn.execute("""
            SELECT opportunity_share, role_score
            FROM player_advanced_metrics
            WHERE player_id = %s AND as_of_date >= %s AND as_of_date < %s
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id, lookback_str, current_date)).fetchone()

        if not current or not previous:
            return {"usage_trend": None, "efficiency_trend": None}

        current_opp = current["opportunity_share"] or 0
        prev_opp = previous["opportunity_share"] or 0
        current_role = current["role_score"] or 0
        prev_role = previous["role_score"] or 0

        usage_trend = ((current_opp - prev_opp) / prev_opp * 100) if prev_opp > 0 else None
        efficiency_trend = ((current_role - prev_role) / prev_role * 100) if prev_role > 0 else None

        return {
            "usage_trend": usage_trend,
            "efficiency_trend": efficiency_trend,
        }


def get_player_metrics(player_id: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve advanced metrics for a player.

    Args:
        player_id: Sleeper player ID
        as_of_date: Specific date (YYYY-MM-DD) or None for latest

    Returns:
        Dict with all metrics or None if not found
    """
    with get_conn() as conn:
        if as_of_date:
            row = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE player_id = %s AND as_of_date = %s
            """, (player_id, as_of_date)).fetchone()
        else:
            row = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE player_id = %s
                ORDER BY as_of_date DESC
                LIMIT 1
            """, (player_id,)).fetchone()

        return dict(row) if row else None


def get_player_metrics_by_season(player_id: str, season: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve advanced metrics for a player in a given season.

    A single season can have multiple snapshot rows with complementary columns
    — e.g. a PFF NFL import (completion %, passer rating, snap share) on one
    date and the computed efficiency snapshot (role_score, yards_per_target,
    target quality) on another. Taking only the most recent row would silently
    drop whichever set landed on the older date, which is why the season view
    showed fewer metrics than Career. Instead, coalesce all rows for the season
    newest-first: the latest non-null value wins for every column.

    Args:
        player_id: Sleeper player ID
        season: NFL season year (e.g. 2025)

    Returns:
        Dict with all metrics or None if not found
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM player_advanced_metrics
            WHERE player_id = %s AND season = %s
            ORDER BY as_of_date DESC
        """, (player_id, season)).fetchall()

        if not rows:
            return None

        rows = [dict(r) for r in rows]
        # Base = most recent row (carries id, position, as_of_date, season).
        merged = dict(rows[0])
        # Fill any remaining nulls from older rows in the same season.
        for older in rows[1:]:
            for key, value in older.items():
                if merged.get(key) is None and value is not None:
                    merged[key] = value
        return merged


def get_player_career_metrics(player_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve career-advanced metrics aggregated across all seasons for a player.

    Args:
        player_id: Sleeper player ID

    Returns:
        Dict with aggregated career metrics or None if not found
    """
    from collections import defaultdict

    _NUMERIC_METRICS = [
        'yards_per_target', 'catch_rate', 'yards_per_reception', 'target_quality_score',
        'yards_per_carry', 'yards_per_touch', 'rush_td_rate',
        'yards_per_attempt', 'completion_pct', 'td_rate', 'int_rate',
        'snap_share', 'opportunity_share', 'red_zone_usage', 'role_score',
        'yards_after_catch', 'yards_after_catch_per_reception', 'avg_depth_of_target',
        'contested_catch_rate', 'avoided_tackles', 'drop_rate', 'slot_rate',
        'wide_rate', 'inline_rate', 'pass_block_rate', 'grades_offense',
        'grades_pass_block', 'explosive_runs_10_plus', 'breakaway_percentage',
        'elusive_rating', 'pff_rushing_grade', 'pff_passing_grade',
        'big_time_throw_rate', 'adjusted_completion_rate', 'pressure_to_sack_rate',
        'nfl_passer_rating',
    ]

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM player_advanced_metrics
            WHERE player_id = %s
            ORDER BY as_of_date DESC
        """, (player_id,)).fetchall()

        if not rows:
            return None

        metrics_list = [dict(row) for row in rows]
        latest = metrics_list[0]

        # Prefer the canonical fantasy position (QB/RB/WR/TE); some import
        # paths write PFF codes like "HB".  Search all rows newest-first.
        _canonical = {"QB", "RB", "WR", "TE"}
        position = None
        for r in metrics_list:
            p = _normalize_position(r.get("position"))
            if p in _canonical:
                position = p
                break
        if position is None:
            position = _normalize_position(latest.get("position"))

        # Step 1 — coalesce rows within each season so that a season with
        # two DB rows (one PFF import + one computed snapshot) contributes
        # the full column set, not just the most recent row.
        season_buckets: Dict[Optional[int], List[Dict]] = defaultdict(list)
        for r in metrics_list:
            season_buckets[r.get("season")].append(r)

        seasons_ordered = sorted(
            (s for s in season_buckets if s is not None), reverse=True
        )

        if seasons_ordered:
            season_snapshots = []
            for s in seasons_ordered:
                s_rows = season_buckets[s]  # already newest-first from ORDER BY
                merged = dict(s_rows[0])
                for older in s_rows[1:]:
                    for key, value in older.items():
                        if merged.get(key) is None and value is not None:
                            merged[key] = value
                season_snapshots.append(merged)
        else:
            season_snapshots = [latest]

        # Step 2 — weighted average across seasons (most recent = weight 1,
        # previous = 0.5, etc.).  Track weight per column separately so a
        # metric that only exists in one season isn't diluted by the total
        # season count (e.g. PFF grades imported for just one year).
        col_sums: Dict[str, float] = {}
        col_weights: Dict[str, float] = {}

        for i, snap in enumerate(season_snapshots):
            w = 1.0 / (i + 1)
            for metric in _NUMERIC_METRICS:
                v = snap.get(metric)
                if v is not None:
                    col_sums[metric] = col_sums.get(metric, 0.0) + float(v) * w
                    col_weights[metric] = col_weights.get(metric, 0.0) + w

        aggregated: Dict[str, Any] = {
            "player_id": player_id,
            "position": position,
            "season": None,
            "as_of_date": latest.get("as_of_date"),
        }
        for metric in _NUMERIC_METRICS:
            if col_weights.get(metric, 0.0) > 0:
                aggregated[metric] = col_sums[metric] / col_weights[metric]

        return aggregated


def get_available_seasons_for_player(player_id: str) -> List[int]:
    """
    Return a list of seasons (descending) for which metrics exist for this player.

    Args:
        player_id: Sleeper player ID

    Returns:
        List of season years, e.g. [2025, 2024]
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT season
            FROM player_advanced_metrics
            WHERE player_id = %s AND season IS NOT NULL
            ORDER BY season DESC
        """, (player_id,)).fetchall()
        return [row["season"] for row in rows]


def get_top_role_players(position: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get players with highest role scores (usage + efficiency).

    Args:
        position: Filter by position (QB/RB/WR/TE) or None for all
        limit: Max number of players to return

    Returns:
        List of player metrics sorted by role_score descending
    """
    with get_conn() as conn:
        # Get latest date with metrics
        latest = conn.execute("""
            SELECT MAX(as_of_date) as max_date
            FROM player_advanced_metrics
        """).fetchone()

        if not latest or not latest["max_date"]:
            return []

        latest_date = latest["max_date"]

        if position:
            rows = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE as_of_date = %s AND position = %s
                ORDER BY role_score DESC NULLS LAST
                LIMIT %s
            """, (latest_date, position, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE as_of_date = %s
                ORDER BY role_score DESC NULLS LAST
                LIMIT %s
            """, (latest_date, limit)).fetchall()

        return [dict(row) for row in rows]


def get_year_over_year_metrics(conn, player_id: str, current_dt: "datetime") -> Optional[Dict[str, float]]:
    """
    Get metrics from the same point in the previous season for year-over-year comparison.
    Captures second-year leaps and depth chart changes (WR2 → WR1).

    Args:
        conn: Database connection
        player_id: Sleeper player ID
        current_dt: Current date as datetime object

    Returns:
        Dict with snap_share and opportunity_share from last year, or None
    """
    from datetime import timedelta

    # Look back approximately 1 year (365 days)
    # Use a window of +/- 30 days to handle season timing variations
    last_year_target = current_dt - timedelta(days=365)
    window_start = last_year_target - timedelta(days=30)
    window_end = last_year_target + timedelta(days=30)

    try:
        # Find closest metrics from previous season within window
        row = conn.execute("""
            SELECT
                snap_share,
                opportunity_share,
                as_of_date,
                ABS(EXTRACT(EPOCH FROM (as_of_date::date - %s::date))) as days_diff
            FROM player_advanced_metrics
            WHERE player_id = %s
              AND as_of_date BETWEEN %s AND %s
            ORDER BY days_diff ASC
            LIMIT 1
        """, (
            last_year_target.strftime("%Y-%m-%d"),
            player_id,
            window_start.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d")
        )).fetchone()

        if row:
            return {
                "snap_share": row["snap_share"],
                "opportunity_share": row["opportunity_share"],
                "as_of_date": row["as_of_date"]
            }

    except Exception as e:
        # Silently fail - YoY comparison is optional enhancement
        pass

    return None


def detect_breakout_candidates_legacy(
        lookback_days: int = 14,
        min_games: int = 2,
        age_threshold: float = 26.0,
) -> List[Dict[str, Any]]:
    """
    LEGACY: Detect breakout candidates using multi-factor analysis.

    This is the original implementation. New code should use the unified breakout engine
    via detect_breakout_candidates() which wraps the BreakoutEngine.

    Factors considered:
    1. Short-term usage increase (snap %, opportunity share vs 14 days ago)
    2. Efficiency improvements (role score, yards per touch)
    3. Value momentum (from player_value_history)
    4. Age (younger players weighted higher)
    5. Year-over-year opportunity increase (captures depth chart promotions)
    6. Second-year player bonus (sophomore breakouts)

    Scoring breakdown:
    - Snap share increase (14d): 0-25 pts
    - Opportunity share increase (14d): 0-30 pts
    - Role score improvement: 0-25 pts
    - Efficiency gains: 0-20 pts
    - Red zone usage increase: 0-15 pts
    - Youth bonus (<26 yrs): 0-15 pts
    - YoY snap increase: 0-20 pts (NEW - captures WR2→WR1 scenarios)
    - YoY opportunity increase: 0-25 pts (NEW - captures expanded roles)
    - Second-year bonus: 10 pts (NEW - sophomore leap)

    Threshold: 30+ points required to qualify

    Args:
        lookback_days: Days to look back for trend comparison
        min_games: Minimum games played to qualify
        age_threshold: Max age for "young breakout" bonus

    Returns:
        List of breakout candidates with composite scores
    """
    from datetime import datetime, timedelta
    from utils.utils import load_players_index

    players_index = load_players_index() or {}

    with get_conn() as conn:
        # Get latest date
        latest = conn.execute("""
            SELECT MAX(as_of_date) as max_date
            FROM player_advanced_metrics
        """).fetchone()

        if not latest or not latest["max_date"]:
            return []

        latest_date = latest["max_date"]
        latest_dt = datetime.strptime(str(latest_date), "%Y-%m-%d")
        lookback_dt = latest_dt - timedelta(days=lookback_days)
        lookback_date = lookback_dt.strftime("%Y-%m-%d")

        # Get current and historical metrics for all players
        current_metrics = conn.execute("""
            SELECT
                player_id,
                position,
                snap_share,
                opportunity_share,
                role_score,
                yards_per_target,
                yards_per_carry,
                yards_per_touch,
                red_zone_usage
            FROM player_advanced_metrics
            WHERE as_of_date = %s
        """, (latest_date,)).fetchall()

        breakouts = []

        for current in current_metrics:
            player_id = current["player_id"]
            position = current["position"]

            # Get previous metrics for comparison
            previous = conn.execute("""
                SELECT
                    snap_share,
                    opportunity_share,
                    role_score,
                    yards_per_target,
                    yards_per_carry,
                    yards_per_touch,
                    red_zone_usage
                FROM player_advanced_metrics
                WHERE player_id = %s
                  AND as_of_date >= %s
                  AND as_of_date < %s
                ORDER BY as_of_date DESC
                LIMIT 1
            """, (player_id, lookback_date, latest_date)).fetchone()

            if not previous:
                continue

            # Get player metadata
            player_meta = players_index.get(str(player_id), {})
            age = player_meta.get("age")
            name = player_meta.get("name", "Unknown")

            # Calculate breakout score components
            score_components = []

            # 1. Snap share increase (0-25 points)
            curr_snaps = current["snap_share"] or 0
            prev_snaps = previous["snap_share"] or 0
            if prev_snaps > 0:
                snap_increase = ((curr_snaps - prev_snaps) / prev_snaps) * 100
                if snap_increase > 20:  # 20%+ increase
                    score_components.append(("snap_increase", min(snap_increase, 100) * 0.25))

            # 2. Opportunity share increase (0-30 points)
            curr_opp = current["opportunity_share"] or 0
            prev_opp = previous["opportunity_share"] or 0
            if prev_opp > 0 and curr_opp > prev_opp:
                opp_increase = ((curr_opp - prev_opp) / prev_opp) * 100
                if opp_increase > 15:  # 15%+ increase
                    score_components.append(("opportunity_increase", min(opp_increase, 150) * 0.2))

            # 3. Role score improvement (0-25 points)
            curr_role = current["role_score"] or 0
            prev_role = previous["role_score"] or 0
            if prev_role > 0 and curr_role > prev_role:
                role_improvement = ((curr_role - prev_role) / prev_role) * 100
                if role_improvement > 10:  # 10%+ improvement
                    score_components.append(("role_improvement", min(role_improvement, 100) * 0.25))

            # 4. Efficiency gains (0-20 points)
            efficiency_score = 0
            if position in ("WR", "TE", "RB"):
                curr_ypt = current["yards_per_target"] or 0
                prev_ypt = previous["yards_per_target"] or 0
                if prev_ypt > 0 and curr_ypt > prev_ypt:
                    ypt_gain = ((curr_ypt - prev_ypt) / prev_ypt) * 100
                    if ypt_gain > 15:
                        efficiency_score += min(ypt_gain, 50) * 0.2

            if position == "RB":
                curr_ypc = current["yards_per_carry"] or 0
                prev_ypc = previous["yards_per_carry"] or 0
                if prev_ypc > 0 and curr_ypc > prev_ypc:
                    ypc_gain = ((curr_ypc - prev_ypc) / prev_ypc) * 100
                    if ypc_gain > 15:
                        efficiency_score += min(ypc_gain, 50) * 0.2

            if efficiency_score > 0:
                score_components.append(("efficiency_gains", efficiency_score))

            # 5. Red zone usage increase (0-15 points)
            curr_rz = current["red_zone_usage"] or 0
            prev_rz = previous["red_zone_usage"] or 0
            if curr_rz > prev_rz and prev_rz > 0:
                rz_increase = ((curr_rz - prev_rz) / prev_rz) * 100
                if rz_increase > 20:
                    score_components.append(("red_zone_increase", min(rz_increase, 150) * 0.1))

            # 6. Age bonus (0-15 points for players under age threshold)
            age_bonus = 0
            if age and age < age_threshold:
                # Younger = higher bonus (max 15 points for 21 year olds)
                age_bonus = (age_threshold - age) * 3
                score_components.append(("youth_bonus", min(age_bonus, 15)))

            # 7. Year-over-year opportunity increase (0-25 points)
            # Captures second-year leaps and depth chart promotions
            yoy_metrics = get_year_over_year_metrics(conn, player_id, latest_dt)
            if yoy_metrics:
                prev_year_snaps = yoy_metrics.get("snap_share") or 0
                prev_year_opp = yoy_metrics.get("opportunity_share") or 0

                # YoY snap share increase
                if prev_year_snaps > 0 and curr_snaps > prev_year_snaps:
                    yoy_snap_increase = ((curr_snaps - prev_year_snaps) / prev_year_snaps) * 100
                    if yoy_snap_increase > 30:  # 30%+ YoY increase
                        score_components.append(("yoy_snap_increase", min(yoy_snap_increase * 0.15, 20)))

                # YoY opportunity share increase (higher weight)
                if prev_year_opp > 0 and curr_opp > prev_year_opp:
                    yoy_opp_increase = ((curr_opp - prev_year_opp) / prev_year_opp) * 100
                    if yoy_opp_increase > 25:  # 25%+ YoY increase
                        score_components.append(("yoy_opportunity_increase", min(yoy_opp_increase * 0.2, 25)))

            # 8. Second-year player bonus (0-10 points)
            # Players in year 2 are prime breakout candidates
            years_exp = player_meta.get("years_exp")
            if years_exp == 1:
                score_components.append(("second_year_bonus", 10))

            # Calculate total breakout score
            total_score = sum(score for _, score in score_components)

            # Only include players with score > 30 (significant breakout signals)
            if total_score >= 30:
                # Get value change for context
                value_delta = get_value_delta_for_player(player_id, lookback_days)

                breakouts.append({
                    "player_id": player_id,
                    "name": name,
                    "position": position,
                    "age": age,
                    "breakout_score": round(total_score, 1),
                    "score_components": {k: round(v, 1) for k, v in score_components},
                    "current_role_score": current["role_score"],
                    "previous_role_score": previous["role_score"],
                    "snap_share": curr_snaps,
                    "opportunity_share": curr_opp,
                    "value_delta": value_delta,
                })

        # Sort by breakout score descending
        breakouts.sort(key=lambda x: x["breakout_score"], reverse=True)

        return breakouts


def detect_breakout_candidates(
        lookback_days: int = 14,
        min_games: int = 2,
        age_threshold: float = 26.0,
        use_unified_engine: bool = True
) -> List[Dict[str, Any]]:
    """
    Detect in-season breakout candidates.

    This function now uses the unified breakout engine by default, which provides
    year-round scoring with phase-based weighting and explainability.

    Args:
        lookback_days: Days to look back for trend comparison
        min_games: Minimum games played
        age_threshold: Age threshold for youth bonus
        use_unified_engine: Use new unified engine (default True) vs legacy

    Returns:
        List of breakout candidates sorted by score
    """
    if not use_unified_engine:
        # Use legacy implementation
        return detect_breakout_candidates_legacy(lookback_days, min_games, age_threshold)

    # Use unified breakout engine
    try:
        from data_building.breakout_engine import BreakoutEngine
        from utils.utils import load_players_index, load_model_value_table
        from datetime import datetime, date

        # Get current season
        nfl_state = get_nfl_state() or {}
        current_season = int(nfl_state.get("season") or datetime.now().year)

        players_index = load_players_index() or {}
        value_table = load_model_value_table() or []

        # Build values lookup
        values_by_id = {str(p.get("id")): p for p in value_table}

        # Get top 600 players by value
        sorted_values = sorted(value_table, key=lambda x: x.get("value", 0), reverse=True)
        top_player_ids = set(str(p.get("id")) for p in sorted_values[:600])

        # Build player list for engine
        player_list = []
        for player_id in top_player_ids:
            player_meta = players_index.get(player_id, {})
            player_value = values_by_id.get(player_id, {})

            position = player_meta.get("pos") or player_value.get("position")
            if not position:
                continue

            player_list.append({
                'player_id': player_id,
                'player_name': player_meta.get("name") or player_meta.get("full_name"),
                'team': player_meta.get("team"),
                'position': position,
                'age': player_value.get("age"),
                'years_exp': player_meta.get("years_exp", 0)
            })

        # Initialize engine for in-season
        engine = BreakoutEngine(season=current_season, as_of_date=date.today())

        # Calculate scores
        candidates = engine.calculate_breakout_scores(player_list, min_score=30)

        # Convert to legacy format for compatibility
        results = []
        for cand in candidates:
            player_value = values_by_id.get(cand.player_id, {})

            results.append({
                "player_id": cand.player_id,
                "name": cand.player_name,
                "team": cand.team,
                "position": cand.position,
                "age": player_value.get("age"),
                "years_exp": player_value.get("years_exp"),
                "value": player_value.get("value", 0),
                "breakout_score": cand.breakout_opportunity_score,
                "score_components": {
                    "opportunity_opened": cand.opportunity_opened_score,
                    "competition_removed": cand.competition_removed_score,
                    "competition_added": cand.competition_added_penalty,
                    "team_environment": cand.team_environment_score,
                    "player_readiness": cand.player_readiness_score,
                    "role_trajectory": cand.role_trajectory_score,
                    "confidence": cand.confidence_score
                },
                "key_reasons": cand.key_reasons,
                "projected_role": cand.projected_role_tag,
                "directional_trend": cand.directional_trend,
                "phase": cand.phase
            })

        return results

    except Exception as e:
        print(f"[detect_breakout_candidates] Error using unified engine: {e}")
        print("[detect_breakout_candidates] Falling back to legacy implementation")
        import traceback
        traceback.print_exc()
        # Fallback to legacy
        return detect_breakout_candidates_legacy(lookback_days, min_games, age_threshold)


def get_value_delta_for_player(player_id: str, days: int) -> Optional[float]:
    """
    Get value change for a player over the last N days.

    Args:
        player_id: Sleeper player ID
        days: Number of days to look back

    Returns:
        Value change or None if not available
    """
    from datetime import datetime, timedelta

    with get_conn() as conn:
        # Get latest value
        latest = conn.execute("""
            SELECT value, as_of_date
            FROM player_value_history
            WHERE player_id = %s AND source = 'model'
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id,)).fetchone()

        if not latest:
            return None

        latest_date = latest["as_of_date"]
        latest_value = latest["value"] or 0

        # Get value from N days ago
        lookback_dt = datetime.strptime(str(latest_date), "%Y-%m-%d") - timedelta(days=days)
        lookback_date = lookback_dt.strftime("%Y-%m-%d")

        previous = conn.execute("""
            SELECT value
            FROM player_value_history
            WHERE player_id = %s
              AND source = 'model'
              AND as_of_date >= %s
              AND as_of_date < %s
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id, lookback_date, latest_date)).fetchone()

        if not previous:
            return None

        previous_value = previous["value"] or 0

        return latest_value - previous_value
