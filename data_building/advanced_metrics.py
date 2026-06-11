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

                -- Sample size (games played in the snapshot window)
                games NUMERIC,

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

        # Add games column if it doesn't exist yet (migration). Populated on the
        # next snapshot/backfill; older rows stay NULL and pass the min-games gate.
        conn.execute("""
            ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS games NUMERIC;
        """)

        # Volume totals used for per-stat minimum filters on the leaderboard.
        conn.execute("""
            ALTER TABLE player_advanced_metrics
                ADD COLUMN IF NOT EXISTS total_targets  NUMERIC,
                ADD COLUMN IF NOT EXISTS total_receptions NUMERIC,
                ADD COLUMN IF NOT EXISTS total_carries  NUMERIC,
                ADD COLUMN IF NOT EXISTS total_touches  NUMERIC,
                ADD COLUMN IF NOT EXISTS total_pass_att NUMERIC,
                ADD COLUMN IF NOT EXISTS total_routes   NUMERIC;
        """)

        # Target share and air yards (usage/stats CSV sourced).
        conn.execute("""
            ALTER TABLE player_advanced_metrics
                ADD COLUMN IF NOT EXISTS target_share    NUMERIC,
                ADD COLUMN IF NOT EXISTS air_yards_share NUMERIC,
                ADD COLUMN IF NOT EXISTS air_yards_per_game NUMERIC;
        """)

        # Red zone breakdown (split from red_zone_usage).
        conn.execute("""
            ALTER TABLE player_advanced_metrics
                ADD COLUMN IF NOT EXISTS rz_targets_pg NUMERIC,
                ADD COLUMN IF NOT EXISTS rz_carries_pg NUMERIC;
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

        # Derive total_routes from (yards_per_reception × total_receptions) / yprr.
        # yprr and the receiving yards live on different as_of_date rows within a
        # season, so we aggregate the best non-null values per player/season and
        # write the result to the most-recent row (which is what DISTINCT ON picks
        # in get_metric_leaderboard).
        conn.execute("""
            UPDATE player_advanced_metrics m
            SET total_routes = ROUND((b.ypr * b.recs) / b.yprr)
            FROM (
                SELECT player_id, season,
                    MAX(yprr) FILTER (WHERE yprr IS NOT NULL AND yprr > 0) AS yprr,
                    MAX(yards_per_reception) FILTER (WHERE yards_per_reception IS NOT NULL) AS ypr,
                    MAX(total_receptions) FILTER (WHERE total_receptions IS NOT NULL) AS recs
                FROM player_advanced_metrics
                WHERE season IS NOT NULL
                GROUP BY player_id, season
            ) b
            WHERE m.player_id = b.player_id
              AND m.season = b.season
              AND b.yprr IS NOT NULL
              AND b.ypr  IS NOT NULL
              AND b.recs IS NOT NULL
              AND m.total_routes IS NULL
              AND m.as_of_date = (
                  SELECT MAX(as_of_date) FROM player_advanced_metrics
                  WHERE player_id = m.player_id AND season = m.season
              )
        """)


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

    # Route participation proxy (WR/TE offensive snap % = fraction of dropbacks
    # they were on the field to run a route, sourced from nfl_data_py snap counts).
    conn.execute("""
        ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS route_participation NUMERIC;
    """)
    # Backfill from snap_share for existing WR/TE rows that are missing it.
    conn.execute("""
        UPDATE player_advanced_metrics
        SET route_participation = snap_share
        WHERE position IN ('WR', 'TE')
          AND snap_share IS NOT NULL
          AND route_participation IS NULL;
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
            ADD COLUMN IF NOT EXISTS nfl_passer_rating NUMERIC,
            ADD COLUMN IF NOT EXISTS yprr NUMERIC;
    """)

    # Touchdown volume totals.
    conn.execute("""
        ALTER TABLE player_advanced_metrics
            ADD COLUMN IF NOT EXISTS total_rush_tds  NUMERIC,
            ADD COLUMN IF NOT EXISTS total_rec_tds   NUMERIC,
            ADD COLUMN IF NOT EXISTS total_pass_tds  NUMERIC,
            ADD COLUMN IF NOT EXISTS total_tds       NUMERIC;
    """)


def _extract_metric_value(metrics: Dict, metric_name: str):
    """Safely pull the scalar value from a metric payload dict."""
    entry = metrics.get(metric_name)
    if not isinstance(entry, dict):
        return None
    return entry.get("value")


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
        "rz_targets_pg": rz_targets if rz_targets > 0 else None,
        "rz_carries_pg": rz_carries if rz_carries > 0 else None,
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
    # avg_off_snap_pct is a 0-1 fraction (from PFR); the _norm bounds below are on a
    # 0-100 percentage scale, so convert once here.
    snap_pct = _safe(usage.get("avg_off_snap_pct")) * 100.0
    games = _safe(usage.get("games"))

    if games <= 0 or snap_pct <= 0:
        return None

    avg_targets = _safe(usage.get("avg_targets"))
    avg_carries = _safe(usage.get("avg_carries"))
    rec_rz_tgt_pg = _safe(usage.get("rec_rz_tgt_pg"))
    rush_rz_att_pg = _safe(usage.get("rush_rz_att_pg"))

    ypt = _safe(receiving_metrics.get("yards_per_target"))
    catch_rate = _safe(receiving_metrics.get("catch_rate"))
    # Receiving TD rate isn't part of the receiving-metrics dict; derive per-target
    # rate directly from usage (matches the 0.02-0.12 _norm bounds used below).
    _avg_tgts = _safe(usage.get("avg_targets"))
    rec_td_rate = (_safe(usage.get("avg_rec_tds")) / _avg_tgts) if _avg_tgts > 0 else 0.0

    ypc = _safe(rushing_metrics.get("yards_per_carry"))
    rush_td_rate = _safe(rushing_metrics.get("rush_td_rate"))

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
        # passing td_rate / int_rate come back as percentages (pass_tds/att * 100),
        # so the bounds are on a 0-100 scale (2-8% TD, 1-5% INT), not fractions.
        td_score = _norm(pass_td_rate, 2.0, 8.0)
        int_penalty = _norm(int_rate, 1.0, 5.0) if int_rate > 0 else 0.0

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
# v2 is a two-pass batch computation (needs team aggregates for shares), exposed
# via finalize_role_scores_v2():
#   Pass 1 — a raw 0-1 opportunity index from team-relative shares.
#   Pass 2 — scale that index against a fixed per-position "elite role" anchor so
#            100 means an alpha/bellcow workload. Absolute (not percentile) so a
#            middling role reads as middling regardless of how strong the rest of
#            the cohort is that season, and so scores are comparable across years.
# Toggle with the ROLE_SCORE_V2 env var (default on); v1 stays reachable for A/B.

# Index value that maps to 100 per position. Calibrated empirically against the
# 2025 cohort's opportunity-index distribution so that only the genuine top 2-4
# roles per position reach 100, with a smooth gradient below (elite tier ~88-100,
# strong starters ~75-88, role players lower).
#
# Calibration history (top WR / players at 100 across all positions):
#   v2a  WR=0.48 TE=0.40 RB=0.70 QB=0.78  -> top WR ~75, only ~1 at 100 (too high)
#   v2b  WR=0.40 TE=0.33 RB=0.58 QB=0.68  -> ~29 players at 100 (too low)
#   v2c  WR=0.46 TE=0.35 RB=0.74 QB=0.77  -> ~8 players at 100 (this) ✓
#
# Anchors sit near the 2025 p98-p99 index per position (RB/WR/QB show a clear
# elite cluster then a gap; the anchor lands just below the cluster's top).
_ROLE_ELITE_ANCHOR: Dict[str, float] = {"WR": 0.46, "TE": 0.35, "RB": 0.74, "QB": 0.77}
# A player needs this many games to earn full sample confidence (small samples
# shrink toward 0 so a 1-2 game fluke can't post an elite role score).
_ROLE_FULL_SAMPLE_GAMES = 4.0


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
    rz_available: bool = True,
) -> Optional[float]:
    """
    Pass 1: a 0-1 opportunity index from team-relative shares (no efficiency).
    team_ctx is the entry from build_team_opportunity_context for this player's
    team. Returns None for non-skill positions or players who never played.

    rz_available=False (no red-zone data in the slate, e.g. offseason / early
    season) drops the RZ components and renormalises the remaining weights, so
    the index is not artificially capped below the elite anchor — otherwise the
    missing 0.22-0.27 RZ weight would undersell TEs/RBs.
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

    # Each component is (value, weight, is_red_zone).
    if position == "WR":
        # Alpha / slot / deep / RZ specialist all reachable; snap keeps
        # lower-target-share field-stretchers from cratering until air yards land.
        comps = [(tshare, 0.50, False), (snap, 0.28, False), (rz_tgt_share, 0.22, True)]

    elif position == "TE":
        # Snap deliberately low — TE snaps include blocking, not a fantasy role.
        # RZ involvement is a big slice of TE value (they are red-zone weapons).
        comps = [(tshare, 0.55, False), (rz_tgt_share, 0.27, True), (snap, 0.18, False)]

    elif position == "RB":
        # PPR-weighted dual role: the 1.7x target premium lets pass-catching
        # backs register, while rush + goal-line share reward early-down bellcows.
        rshare = _share(_safe(usage.get("avg_carries")), _safe(team_ctx.get("carries")))
        core   = _clip(rshare + 1.7 * tshare, 0.0, 1.0)
        comps = [(core, 0.46, False), (rz_rush_share, 0.20, True),
                 (snap, 0.18, False), (rz_tgt_share, 0.16, True)]

    elif position == "QB":
        # No "share" at QB — workload + dual-threat, ranked. No red-zone term:
        # "RZ role" isn't a meaningful axis for a QB (unlike a goal-line RB or a
        # jump-ball WR/TE), and goal-line rushing is already captured by the
        # designed-rush component. Rushing stays additive upside so pocket
        # passers are not penalised: pass + snap = 0.80.
        pass_vol = _norm(_safe(usage.get("avg_pass_att")), 18, 42)
        rush_vol = _norm(_safe(usage.get("avg_carries")), 0, 9)
        comps = [(pass_vol, 0.47, False), (snap, 0.33, False), (rush_vol, 0.20, False)]

    else:
        return None

    if rz_available:
        idx = sum(value * weight for value, weight, _ in comps)
    else:
        kept = [(value, weight) for value, weight, is_rz in comps if not is_rz]
        wsum = sum(weight for _, weight in kept) or 1.0
        idx = sum(value * (weight / wsum) for value, weight in kept)

    return _clip(idx, 0.0, 1.0)


def finalize_role_scores_v2(
    metrics_list: List[Dict[str, Any]],
    usage_table: List[Dict[str, Any]],
) -> None:
    """
    Overwrite each metrics dict's "role_score" with the v2 score: the
    team-relative opportunity index scaled against a fixed per-position elite
    anchor (absolute, not percentile), lightly shrunk for small samples.

    No-op (leaves the v1 values from calculate_player_metrics in place) when
    ROLE_SCORE_V2 is disabled. Mutates metrics_list in place.
    """
    if not use_role_score_v2():
        return

    team_ctx_map = build_team_opportunity_context(usage_table)
    usage_by_id = {str(p.get("id")): p for p in usage_table}

    # Does the slate actually carry red-zone data? If not (offseason / early
    # season / RZ source down), the index renormalises so TEs/RBs aren't
    # undersold by a silently-zero RZ term.
    rz_available = any(
        (_safe(c.get("rz_tgt")) > 0 or _safe(c.get("rz_rush")) > 0)
        for c in team_ctx_map.values()
    )

    for m in metrics_list:
        pid = str(m.get("player_id"))
        position = m.get("position")
        entry = usage_by_id.get(pid)
        if entry is None:
            continue
        usage = entry.get("usage") or {}
        idx = role_opportunity_index(usage, position, team_ctx_map.get(entry.get("team"), {}), rz_available)
        anchor = _ROLE_ELITE_ANCHOR.get(position)
        if idx is None or not anchor:
            continue
        # Absolute scaling: elite role -> ~100; middling role stays middling
        # regardless of cohort strength. Small samples shrink toward 0.
        conf = _clip(_safe(usage.get("games")) / _ROLE_FULL_SAMPLE_GAMES, 0.0, 1.0)
        m["role_score"] = round(_clip(idx / anchor, 0.0, 1.0) * 100.0 * conf, 1)


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

    games = _safe(usage.get("games")) or None

    def _vol(avg_key: str) -> Optional[float]:
        avg = _safe(usage.get(avg_key))
        if avg is None or games is None:
            return None
        return round(avg * games, 1)

    raw_tshare = _safe(usage.get("target_share"))
    target_share = round(raw_tshare * 100.0, 1) if raw_tshare > 0 else None

    return {
        "player_id": player_id,
        "position": position,
        **receiving,
        **rushing,
        **passing,
        **usage_metrics,
        "role_score": role_score,
        "usage_trend": None,
        "efficiency_trend": None,
        "games": games,
        "total_targets":    _vol("avg_targets"),
        "total_receptions": _vol("avg_receptions"),
        "total_carries":    _vol("avg_carries"),
        "total_touches":    (
            None if games is None else
            round(
                ((_safe(usage.get("avg_carries")) or 0) + (_safe(usage.get("avg_receptions")) or 0)) * games,
                1
            )
        ),
        "total_pass_att":   _vol("avg_pass_att"),
        "target_share":     target_share,
        "total_rush_tds":   _vol("avg_rush_tds"),
        "total_rec_tds":    _vol("avg_rec_tds"),
        "total_pass_tds":   _vol("avg_pass_tds"),
        "total_tds": (
            None if all(
                _vol(k) is None for k in ("avg_rush_tds", "avg_rec_tds", "avg_pass_tds")
            ) else round(sum(
                _vol(k) or 0 for k in ("avg_rush_tds", "avg_rec_tds", "avg_pass_tds")
            ), 1)
        ),
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
            pos = (metrics.get("position") or "").upper()
            route_partic = metrics.get("snap_share") if pos in ("WR", "TE") else None

            # Upsert: update if exists, insert if not
            conn.execute("""
                INSERT INTO player_advanced_metrics (
                    player_id, as_of_date, season, position,
                    yards_per_target, catch_rate, yards_per_reception, target_quality_score,
                    yards_per_carry, yards_per_touch, rush_td_rate,
                    yards_per_attempt, completion_pct, td_rate, int_rate,
                    snap_share, opportunity_share, red_zone_usage,
                    rz_targets_pg, rz_carries_pg,
                    role_score, usage_trend, efficiency_trend, games,
                    total_targets, total_receptions, total_carries, total_touches, total_pass_att,
                    target_share, route_participation,
                    total_rush_tds, total_rec_tds, total_pass_tds, total_tds
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s
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
                    rz_targets_pg = EXCLUDED.rz_targets_pg,
                    rz_carries_pg = EXCLUDED.rz_carries_pg,
                    role_score = EXCLUDED.role_score,
                    usage_trend = EXCLUDED.usage_trend,
                    efficiency_trend = EXCLUDED.efficiency_trend,
                    games = EXCLUDED.games,
                    total_targets = EXCLUDED.total_targets,
                    total_receptions = EXCLUDED.total_receptions,
                    total_carries = EXCLUDED.total_carries,
                    total_touches = EXCLUDED.total_touches,
                    total_pass_att = EXCLUDED.total_pass_att,
                    target_share = EXCLUDED.target_share,
                    route_participation = COALESCE(EXCLUDED.route_participation, player_advanced_metrics.route_participation),
                    total_rush_tds = EXCLUDED.total_rush_tds,
                    total_rec_tds = EXCLUDED.total_rec_tds,
                    total_pass_tds = EXCLUDED.total_pass_tds,
                    total_tds = EXCLUDED.total_tds
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
                metrics.get("rz_targets_pg"), metrics.get("rz_carries_pg"),
                metrics.get("role_score"), metrics.get("usage_trend"),
                metrics.get("efficiency_trend"), metrics.get("games"),
                metrics.get("total_targets"), metrics.get("total_receptions"),
                metrics.get("total_carries"), metrics.get("total_touches"),
                metrics.get("total_pass_att"),
                metrics.get("target_share"),
                route_partic,
                metrics.get("total_rush_tds"), metrics.get("total_rec_tds"),
                metrics.get("total_pass_tds"), metrics.get("total_tds"),
            ))

    print(f"[advanced_metrics] Saved {len(metrics_list)} player metrics for {as_of_date} (season {season})")


def import_air_yards_from_stats_csv(season: int) -> int:
    """
    Read cache/stats_player_reg_{season}.csv (produced by nfl_data_py) and upsert
    air_yards_per_game and air_yards_share into player_advanced_metrics.

    Matches rows by player_id (sleeper id from players_index) and updates the
    most-recent snapshot row for each player/season.  Safe to call repeatedly.
    Returns the number of rows updated.
    """
    import os
    import csv
    from pathlib import Path
    from utils.paths import CACHE_DIR

    csv_path = Path(CACHE_DIR) / f"stats_player_reg_{season}.csv"
    if not csv_path.exists():
        return 0

    # Load players index to map player display name → sleeper player_id
    players_idx_path = Path(CACHE_DIR) / "players_index_relevant.json"
    if not players_idx_path.exists():
        return 0

    import json
    with open(players_idx_path) as f:
        players_idx = json.load(f)

    # Build name (lowercase) → sleeper_id map for skill-position players.
    # players_index_relevant.json has no NFL/stats ID fields; name matching
    # achieves ~70% coverage for QB/RB/WR/TE.
    _skill = {"QB", "RB", "WR", "TE"}
    name_to_sleeper: dict = {}
    for sleeper_id, info in players_idx.items():
        pos = (info.get("pos") or "").upper()
        if pos not in _skill:
            continue
        name = (info.get("name") or "").strip().lower()
        if name:
            name_to_sleeper[name] = str(sleeper_id)

    rows_by_player: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_name = (row.get("player_display_name") or "").strip().lower()
            sleeper_id = name_to_sleeper.get(display_name)
            if not sleeper_id:
                continue
            try:
                games = float(row.get("games") or 0) or None
                rec_air = float(row.get("receiving_air_yards") or 0) or None
                ayr_share = float(row.get("air_yards_share") or 0) or None
            except (TypeError, ValueError):
                continue
            air_pg = round(rec_air / games, 1) if rec_air and games else None
            ayr_pct = round(ayr_share * 100.0, 1) if ayr_share else None
            rows_by_player[sleeper_id] = {"air_yards_per_game": air_pg, "air_yards_share": ayr_pct}

    if not rows_by_player:
        return 0

    updated = 0
    with get_conn() as conn:
        for sleeper_id, vals in rows_by_player.items():
            result = conn.execute("""
                UPDATE player_advanced_metrics
                SET air_yards_per_game = %s,
                    air_yards_share    = %s
                WHERE player_id = %s
                  AND season = %s
                  AND as_of_date = (
                      SELECT MAX(as_of_date) FROM player_advanced_metrics
                      WHERE player_id = %s AND season = %s
                  )
            """, (
                vals["air_yards_per_game"], vals["air_yards_share"],
                sleeper_id, season, sleeper_id, season,
            ))
            if result.rowcount:
                updated += result.rowcount

    return updated


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
        'snap_share', 'route_participation', 'opportunity_share', 'red_zone_usage', 'role_score',
        'yards_after_catch', 'yards_after_catch_per_reception', 'avg_depth_of_target',
        'contested_catch_rate', 'avoided_tackles', 'drop_rate', 'slot_rate',
        'wide_rate', 'inline_rate', 'pass_block_rate', 'grades_offense',
        'grades_pass_block', 'explosive_runs_10_plus', 'breakaway_percentage',
        'elusive_rating', 'pff_rushing_grade', 'pff_passing_grade',
        'big_time_throw_rate', 'adjusted_completion_rate', 'pressure_to_sack_rate',
        'nfl_passer_rating', 'yprr',
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


# Minimum snap share required for a player to appear on an *efficiency* metric
# leaderboard. Rate/grade metrics are noisy for low-snap players (e.g. a backup QB
# who threw twice with 0 INTs shows a perfect INT rate), so they're excluded.
# Players whose snap_share is unknown (NULL) are NOT filtered, to avoid dropping
# PFF-sourced rows that may lack a snap share.
_MIN_SNAP_FOR_EFFICIENCY = 0.20

# Metrics exposed on the standalone Advanced Metrics leaderboard page, each mapped
# to the positions where it is meaningful (drives the page's auto position filter).
# Internal trend deltas (usage_trend, efficiency_trend) are intentionally excluded.
#   lower_better  → a smaller value ranks better (e.g. INT rate, drop rate).
#   efficiency    → rate/grade metric; gated by _MIN_SNAP_FOR_EFFICIENCY so low-snap
#                   players with degenerate values don't crowd the leaderboard.
# Usage/role metrics (role_score, snap_share, opportunity_share, red_zone_usage)
# are NOT gated — a low value simply sorts low.
_V_TARGETS   = {"col": "total_targets",    "label": "Min Targets",    "opts": [20, 40, 60, 80]}
_V_RECS      = {"col": "total_receptions", "label": "Min Receptions", "opts": [15, 25, 40, 60]}
_V_CARRIES   = {"col": "total_carries",    "label": "Min Carries",    "opts": [30, 50, 100, 150]}
_V_TOUCHES   = {"col": "total_touches",    "label": "Min Touches",    "opts": [40, 60, 100, 150]}
_V_PASS_ATT  = {"col": "total_pass_att",   "label": "Min Attempts",   "opts": [100, 200, 300, 400]}
_V_GAMES     = {"col": "games",            "label": "Min Games",      "opts": [4, 8, 12, 16]}

LEADERBOARD_METRICS: Dict[str, Dict[str, Any]] = {
    # ── General (applies across positions) ───────────────────────────────────
    "role_score":           {"label": "Role Score",          "category": "General", "positions": ["QB", "RB", "WR", "TE"], "min_vol": _V_GAMES, "desc": "Overall opportunity score (0-100) blending snap share, touches, and red-zone usage relative to the player's position."},
    "snap_share":           {"label": "Snap Share",          "category": "General", "positions": ["QB", "RB", "WR", "TE"], "pct": True, "pct_frac": True, "min_vol": _V_GAMES, "desc": "Percent of the team's offensive snaps the player was on the field for."},
    "opportunity_share":    {"label": "Opportunity Share",   "category": "General", "positions": ["RB", "WR", "TE"], "min_vol": _V_GAMES, "desc": "Share of the team's targets plus carries that went to this player."},
    "red_zone_usage":       {"label": "Red Zone Usage",      "category": "General", "positions": ["QB", "RB", "WR", "TE"], "min_vol": _V_GAMES, "desc": "Targets and carries inside the opponent's 20-yard line per game; a proxy for scoring opportunity."},
    "rz_targets_pg":        {"label": "RZ Targets/G",        "category": "General", "positions": ["QB", "WR", "TE", "RB"], "min_vol": _V_GAMES, "desc": "Red zone targets per game (inside opponent's 20-yard line)."},
    "rz_carries_pg":        {"label": "RZ Carries/G",        "category": "General", "positions": ["QB", "RB"], "min_vol": _V_GAMES, "desc": "Red zone rushing attempts per game (inside opponent's 20-yard line)."},
    "grades_offense":       {"label": "PFF Off Grade",       "category": "General", "positions": ["QB", "RB", "WR", "TE"], "efficiency": True, "min_vol": _V_GAMES, "desc": "PFF's overall offensive grade (0-100) from play-by-play charting."},
    "yards_per_touch":      {"label": "Yards / Touch",       "category": "General", "positions": ["RB", "WR", "TE"], "efficiency": True, "min_vol": _V_TOUCHES, "desc": "Yards gained per combined carry and reception."},
    # ── Passing ──────────────────────────────────────────────────────────────
    "yards_per_attempt":    {"label": "Yards / Attempt",    "category": "Passing", "positions": ["QB"], "efficiency": True, "min_vol": _V_PASS_ATT, "desc": "Passing yards per attempt; core passing efficiency stat."},
    "completion_pct":       {"label": "Completion %",       "category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "min_vol": _V_PASS_ATT, "desc": "Percent of pass attempts completed."},
    "adjusted_completion_rate": {"label": "Adj Completion %", "category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "min_vol": _V_PASS_ATT, "desc": "Completion percent adjusted for drops, throwaways, spikes, and batted passes."},
    "td_rate":              {"label": "Pass TD Rate",        "category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "min_vol": _V_PASS_ATT, "desc": "Percent of pass attempts that result in a touchdown."},
    "int_rate":             {"label": "INT Rate",            "category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "lower_better": True, "min_vol": _V_PASS_ATT, "desc": "Percent of pass attempts intercepted. Lower is better."},
    "big_time_throw_rate":  {"label": "Big-Time Throw %",   "category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "min_vol": _V_PASS_ATT, "desc": "PFF rate of high-difficulty, high-value throws (deep and into tight windows)."},
    "pressure_to_sack_rate": {"label": "Pressure to Sack %","category": "Passing", "positions": ["QB"], "efficiency": True, "pct": True, "lower_better": True, "min_vol": _V_PASS_ATT, "desc": "Percent of pressured dropbacks that turn into sacks. Lower is better."},
    "nfl_passer_rating":    {"label": "Passer Rating",       "category": "Passing", "positions": ["QB"], "efficiency": True, "min_vol": _V_PASS_ATT, "desc": "Standard NFL passer rating (0-158.3)."},
    "pff_passing_grade":    {"label": "PFF Pass Grade",      "category": "Passing", "positions": ["QB"], "efficiency": True, "min_vol": _V_PASS_ATT, "desc": "PFF's passing grade (0-100)."},
    # ── Rushing ──────────────────────────────────────────────────────────────
    "yards_per_carry":      {"label": "Yards / Carry",       "category": "Rushing", "positions": ["RB", "QB"], "efficiency": True, "min_vol": _V_CARRIES, "desc": "Rushing yards gained per carry."},
    "rush_td_rate":         {"label": "Rush TD Rate",        "category": "Rushing", "positions": ["RB", "QB"], "efficiency": True, "pct": True, "pct_frac": True, "min_vol": _V_CARRIES, "desc": "Percent of carries that result in a touchdown."},
    "breakaway_percentage": {"label": "Breakaway %",         "category": "Rushing", "positions": ["RB"], "efficiency": True, "pct": True, "min_vol": _V_CARRIES, "desc": "Percent of rushing yards that came on runs of 15+ yards; explosiveness."},
    "elusive_rating":       {"label": "Elusive Rating",      "category": "Rushing", "positions": ["RB"], "efficiency": True, "min_vol": _V_CARRIES, "desc": "PFF metric for yards created after contact and missed tackles forced, independent of blocking."},
    "pff_rushing_grade":    {"label": "PFF Rush Grade",      "category": "Rushing", "positions": ["RB", "QB"], "efficiency": True, "min_vol": _V_CARRIES, "desc": "PFF's rushing grade (0-100)."},
    "explosive_runs_10_plus": {"label": "Explosive Runs",   "category": "Rushing", "positions": ["RB"], "min_vol": _V_CARRIES, "integer": True, "desc": "Count of runs gaining 10 or more yards in the season (PFF). Raw explosive-play volume."},
    "avoided_tackles":      {"label": "Avoided Tackles",    "category": "Rushing", "positions": ["RB"], "min_vol": _V_CARRIES, "desc": "Tackles avoided (missed, broken, or forced) on rush attempts per PFF. Rewards runners who make defenders miss."},
    # ── Receiving ────────────────────────────────────────────────────────────
    "route_participation":  {"label": "Route Partic %",      "category": "Receiving", "positions": ["WR", "TE"], "pct": True, "pct_frac": True, "min_vol": _V_GAMES, "desc": "Percent of the team's pass-play snaps on which the WR/TE ran a route. High route participation means the player is a consistent full-time route runner."},
    "target_share":         {"label": "Target Share",        "category": "Receiving", "positions": ["WR", "TE", "RB"], "pct": True, "min_vol": _V_GAMES, "desc": "Percent of the team's total targets directed at this player."},
    "air_yards_per_game":   {"label": "Air Yards / Game",    "category": "Receiving", "positions": ["WR", "TE"], "min_vol": _V_GAMES, "desc": "Receiving air yards (distance thrown in the air to the player) per game; a measure of downfield target volume."},
    "air_yards_share":      {"label": "Air Yards Share",     "category": "Receiving", "positions": ["WR", "TE"], "pct": True, "min_vol": _V_GAMES, "desc": "Share of the team's total passing air yards directed at this player; combines target share with depth of target."},
    "yards_per_target":     {"label": "Yards / Target",      "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "min_vol": _V_TARGETS, "desc": "Receiving yards earned per time targeted; measures efficiency on volume."},
    "yards_per_reception":  {"label": "Yards / Reception",   "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "min_vol": _V_RECS, "desc": "Average yards gained per catch; higher means a more downfield/explosive role."},
    "catch_rate":           {"label": "Catch Rate",          "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "pct": True, "pct_frac": True, "min_vol": _V_TARGETS, "desc": "Percent of targets caught."},
    "target_quality_score": {"label": "Target Quality",      "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "min_vol": _V_TARGETS, "desc": "Composite of how valuable a player's targets are (depth, location, situation)."},
    "avg_depth_of_target":  {"label": "aDOT",                "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "min_vol": _V_TARGETS, "desc": "Average depth of target: how far downfield (in yards) the player is thrown to."},
    "contested_catch_rate": {"label": "Contested Catch %",   "category": "Receiving", "positions": ["WR", "TE"], "efficiency": True, "pct": True, "min_vol": _V_TARGETS, "desc": "Percent of contested (tightly covered) targets the player came down with."},
    "yards_after_catch_per_reception": {"label": "YAC / Reception", "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "min_vol": _V_RECS, "desc": "Average yards gained after the catch per reception."},
    "drop_rate":            {"label": "Drop Rate",           "category": "Receiving", "positions": ["WR", "RB", "TE"], "efficiency": True, "pct": True, "lower_better": True, "min_vol": _V_TARGETS, "desc": "Percent of catchable targets dropped. Lower is better."},
    "yprr":                 {"label": "Yards / Route Run",   "category": "Receiving", "positions": ["WR", "TE", "RB"], "efficiency": True, "min_vol": _V_GAMES, "desc": "Receiving yards earned per route run (from PFF). Elite WRs are typically 2.0+; accounts for targets indirectly by rewarding yards on every snap."},
    "slot_rate":            {"label": "Slot Rate",           "category": "Receiving", "positions": ["WR", "TE"], "efficiency": True, "pct": True, "min_vol": _V_GAMES, "desc": "Percent of routes run from the slot."},
    "wide_rate":            {"label": "Wide Rate",           "category": "Receiving", "positions": ["WR", "TE"], "efficiency": True, "pct": True, "min_vol": _V_GAMES, "desc": "Percent of routes run from out wide."},
    "inline_rate":          {"label": "Inline Rate",         "category": "Receiving", "positions": ["TE"], "efficiency": True, "pct": True, "min_vol": _V_GAMES, "desc": "Percent of snaps a tight end lined up inline (attached to the formation)."},
    "pass_block_rate":      {"label": "Block Rate",          "category": "Receiving", "positions": ["TE", "RB"], "efficiency": True, "pct": True, "min_vol": _V_GAMES, "desc": "Percent of pass snaps spent blocking rather than running a route."},
    "explosive_runs_pg":       {"label": "Explosive Runs/G",  "category": "Rushing", "positions": ["RB"], "min_vol": _V_GAMES, "desc": "Explosive runs (10+ yards) per game.", "computed_sql": "m.explosive_runs_10_plus::float / NULLIF(m.games, 0)", "computed_null": "m.explosive_runs_10_plus IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "avoided_tackles_pg":      {"label": "Avoided Tackles/G", "category": "Rushing", "positions": ["RB"], "min_vol": _V_GAMES, "desc": "Tackles avoided per game (PFF).", "computed_sql": "m.avoided_tackles::float / NULLIF(m.games, 0)", "computed_null": "m.avoided_tackles IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    # ── Volume counts with paired per-game rates ─────────────────────────────
    "total_carries":      {"label": "Carries",      "category": "Volume", "positions": ["RB", "QB"], "integer": True, "desc": "Total carries in the season."},
    "carries_per_game":   {"label": "Carries/G",    "category": "Volume", "positions": ["RB", "QB"], "min_vol": _V_GAMES, "desc": "Carries per game.", "computed_sql": "m.total_carries::float / NULLIF(m.games, 0)", "computed_null": "m.total_carries IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_targets":      {"label": "Targets",      "category": "Volume", "positions": ["WR", "RB", "TE"], "integer": True, "desc": "Total targets in the season."},
    "targets_per_game":   {"label": "Targets/G",    "category": "Volume", "positions": ["WR", "RB", "TE"], "min_vol": _V_GAMES, "desc": "Targets per game.", "computed_sql": "m.total_targets::float / NULLIF(m.games, 0)", "computed_null": "m.total_targets IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_receptions":   {"label": "Receptions",   "category": "Volume", "positions": ["WR", "RB", "TE"], "integer": True, "desc": "Total receptions in the season."},
    "receptions_per_game": {"label": "Receptions/G", "category": "Volume", "positions": ["WR", "RB", "TE"], "min_vol": _V_GAMES, "desc": "Receptions per game.", "computed_sql": "m.total_receptions::float / NULLIF(m.games, 0)", "computed_null": "m.total_receptions IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_routes":       {"label": "Routes",       "category": "Volume", "positions": ["WR", "TE", "RB"], "integer": True, "desc": "Estimated total routes run (= season receiving yards ÷ yprr). Requires both yprr and receptions data."},
    "routes_per_game":    {"label": "Routes/G",     "category": "Volume", "positions": ["WR", "TE", "RB"], "min_vol": _V_GAMES, "desc": "Routes run per game.", "computed_sql": "m.total_routes::float / NULLIF(m.games, 0)", "computed_null": "m.total_routes IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_touches":      {"label": "Touches",      "category": "Volume", "positions": ["RB", "WR", "TE"], "integer": True, "desc": "Total carries plus receptions in the season."},
    "touches_per_game":   {"label": "Touches/G",    "category": "Volume", "positions": ["RB", "WR", "TE"], "min_vol": _V_GAMES, "desc": "Carries plus receptions per game.", "computed_sql": "m.total_touches::float / NULLIF(m.games, 0)", "computed_null": "m.total_touches IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_rush_tds":     {"label": "Rush TDs",     "category": "Volume", "positions": ["RB", "QB"], "integer": True, "desc": "Total rushing touchdowns in the season."},
    "rush_tds_per_game":  {"label": "Rush TDs/G",   "category": "Volume", "positions": ["RB", "QB"], "min_vol": _V_GAMES, "desc": "Rushing touchdowns per game.", "computed_sql": "m.total_rush_tds::float / NULLIF(m.games, 0)", "computed_null": "m.total_rush_tds IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_rec_tds":      {"label": "Rec TDs",      "category": "Volume", "positions": ["WR", "TE", "RB"], "integer": True, "desc": "Total receiving touchdowns in the season."},
    "rec_tds_per_game":   {"label": "Rec TDs/G",    "category": "Volume", "positions": ["WR", "TE", "RB"], "min_vol": _V_GAMES, "desc": "Receiving touchdowns per game.", "computed_sql": "m.total_rec_tds::float / NULLIF(m.games, 0)", "computed_null": "m.total_rec_tds IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_pass_tds":     {"label": "Pass TDs",     "category": "Volume", "positions": ["QB"], "integer": True, "desc": "Total passing touchdowns in the season."},
    "pass_tds_per_game":  {"label": "Pass TDs/G",   "category": "Volume", "positions": ["QB"], "min_vol": _V_GAMES, "desc": "Passing touchdowns per game.", "computed_sql": "m.total_pass_tds::float / NULLIF(m.games, 0)", "computed_null": "m.total_pass_tds IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
    "total_tds":          {"label": "Total TDs",    "category": "Volume", "positions": ["QB", "RB", "WR", "TE"], "integer": True, "desc": "Total touchdowns (rush + receiving + passing) in the season."},
    "total_tds_per_game": {"label": "Total TDs/G",  "category": "Volume", "positions": ["QB", "RB", "WR", "TE"], "min_vol": _V_GAMES, "desc": "Total touchdowns per game.", "computed_sql": "m.total_tds::float / NULLIF(m.games, 0)", "computed_null": "m.total_tds IS NOT NULL AND m.games IS NOT NULL AND m.games > 0"},
}


def get_available_seasons() -> List[int]:
    """Return distinct seasons that have real player data, newest first.

    Filters to seasons with at least one non-null metric so empty 2026-tagged
    rows (snapshot taken in the offseason but no games played yet) don't appear.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT season FROM player_advanced_metrics "
                "WHERE season IS NOT NULL "
                "AND (role_score IS NOT NULL OR snap_share IS NOT NULL "
                "     OR yards_per_target IS NOT NULL OR yards_per_carry IS NOT NULL) "
                "ORDER BY season DESC"
            ).fetchall()
            return [int(r["season"]) for r in rows]
    except Exception:
        return []


def get_metric_leaderboard(
    metric: str,
    position: Optional[str] = None,
    limit: int = 500,
    season: Optional[int] = None,
    min_vol: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Players ranked by a single advanced metric.

    `metric` must be a key of LEADERBOARD_METRICS. When `season` is given the
    snapshot for that season is used; otherwise the latest snapshot with data for
    this metric is used. `min_vol` filters by the metric's natural volume stat
    (e.g. targets for catch rate, carries for YPC, attempts for completion %) so
    small-sample players with degenerate rates don't crowd the leaderboard. Rows
    with a NULL volume count are kept since older snapshots predate the columns.
    Returns [{player_id, name, team, position, value, games}].
    """
    if metric not in LEADERBOARD_METRICS:
        return []
    pos = (position or "").upper().strip() or None
    _spec = LEADERBOARD_METRICS[metric]
    _computed_sql = _spec.get("computed_sql")
    _computed_null = _spec.get("computed_null")
    vol_spec = _spec.get("min_vol") or {}
    vol_col = vol_spec.get("col") or "games"

    with get_conn() as conn:
        # Pre-check which columns exist to avoid aborting the transaction on
        # a missing column reference. All volume columns were added together, so
        # checking one proxy column tells us if they're all present.
        existing_cols = {
            r["column_name"]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='player_advanced_metrics' "
                "AND column_name = ANY(%s)",
                (["games", "total_targets", "total_receptions",
                  "total_carries", "total_touches", "total_pass_att",
                  "total_rush_tds", "total_rec_tds", "total_pass_tds", "total_tds"],),
            ).fetchall()
        }
        has_games = "games" in existing_cols
        has_vol_col = vol_col in existing_cols

        # Resolve season if not provided: use the latest season with metric data.
        # Computed metrics don't have a direct column, so fall back to games.
        if season is None:
            if _computed_sql:
                srow = conn.execute(
                    "SELECT season FROM player_advanced_metrics "
                    "WHERE games IS NOT NULL AND season IS NOT NULL "
                    "ORDER BY season DESC LIMIT 1"
                ).fetchone()
            else:
                srow = conn.execute(
                    f"SELECT season FROM player_advanced_metrics "
                    f"WHERE {metric} IS NOT NULL AND season IS NOT NULL "
                    f"ORDER BY season DESC LIMIT 1"
                ).fetchone()
            if not srow:
                return []
            season = int(srow["season"])

        # Volume join: coalesce the season-max vol count across all snapshot rows
        # so PFF-imported metrics (which land on a different as_of_date from the
        # computed volume totals) still see the correct carry/target/games count.
        season_for_vol = season
        use_vol_join = bool(has_vol_col and season_for_vol is not None)

        # Check whether this season has non-zero vol data for the vol column so
        # we know whether a NULL/0 count means "no data" vs "predates the column".
        season_has_vol = False
        if use_vol_join:
            chk = conn.execute(
                f"SELECT 1 FROM player_advanced_metrics "
                f"WHERE season = %s AND {vol_col} IS NOT NULL AND {vol_col} > 0 LIMIT 1",
                (season_for_vol,),
            ).fetchone()
            season_has_vol = chk is not None

        apply_vol = bool(season_has_vol and min_vol and min_vol > 0)

        gate = ""
        vol_join = ""
        params: list = []
        if use_vol_join:
            if vol_col == "games":
                # When gating by games, fall back to volume totals (carries + targets)
                # to avoid dropping players whose games column is NULL because the
                # snapshot was written before their season data was available.
                vol_join = (
                    " LEFT JOIN (SELECT player_id,"
                    " COALESCE(MAX(games),"
                    "   CASE WHEN COALESCE(MAX(total_carries),0)+COALESCE(MAX(total_targets),0)>0"
                    "   THEN 1 ELSE NULL END) AS vol"
                    " FROM player_advanced_metrics WHERE season = %s GROUP BY player_id) v"
                    " ON v.player_id = m.player_id"
                )
            else:
                vol_join = (
                    f" LEFT JOIN (SELECT player_id, MAX({vol_col}) AS vol "
                    "FROM player_advanced_metrics WHERE season = %s GROUP BY player_id) v "
                    "ON v.player_id = m.player_id"
                )
            params.append(season_for_vol)
        # Season filter always applied (required for correct DISTINCT ON results).
        gate += " AND m.season = %s"
        params.append(season)
        if pos:
            gate += " AND m.position = %s"
            params.append(pos)
        if LEADERBOARD_METRICS[metric].get("efficiency"):
            gate += " AND (m.snap_share IS NULL OR m.snap_share >= %s)"
            params.append(_MIN_SNAP_FOR_EFFICIENCY)
        if season_has_vol:
            # Always require vol > 0 when the season has vol data: hides players
            # with no recorded volume (null carries, null targets, etc.) even
            # when "Any" minimum is selected. Keeps "Any" clean of zero-vol rows.
            if apply_vol:
                gate += " AND v.vol IS NOT NULL AND v.vol >= %s"
                params.append(min_vol)
            else:
                gate += " AND v.vol IS NOT NULL AND v.vol > 0"
        params.append(limit)

        if has_games:
            games_col = (
                "COALESCE(m.games, v.vol) AS games,"
                if (use_vol_join and vol_col == "games") else "m.games AS games,"
            )
        else:
            games_col = ""
        has_specific_vol = vol_col != "games" and vol_col in existing_cols
        if has_specific_vol:
            specific_vol_col = (
                f"COALESCE(m.{vol_col}, v.vol) AS vol,"
                if use_vol_join else f"m.{vol_col} AS vol,"
            )
        else:
            specific_vol_col = ""

        # Computed metrics (per-game rates) use SQL expressions instead of columns.
        if _computed_sql:
            metric_value_expr = f"{_computed_sql} AS value"
            metric_where = _computed_null
        else:
            metric_value_expr = f"m.{metric} AS value"
            metric_where = f"m.{metric} IS NOT NULL"

        # DISTINCT ON picks each player's most recent non-null snapshot for this
        # metric within the season. This prevents the old single-max-date approach
        # from dropping players whose computed metric (e.g. yards_per_carry) was
        # written on a different date than the latest PFF sync.
        rows = conn.execute(
            f"""SELECT t.*
                FROM (
                    SELECT DISTINCT ON (m.player_id)
                        m.player_id, m.position, {games_col} {specific_vol_col}
                        {metric_value_expr}
                    FROM player_advanced_metrics m{vol_join}
                    WHERE {metric_where}{gate}
                    ORDER BY m.player_id, m.as_of_date DESC
                ) t
                ORDER BY t.value DESC LIMIT %s""",
            tuple(params),
        ).fetchall()

    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
    except Exception:
        idx = {}

    def _player_age(meta: dict) -> Optional[int]:
        bday = meta.get("bDay") or meta.get("bday") or ""
        if not bday:
            return None
        try:
            from datetime import date as _date
            parts = str(bday).split("/")
            if len(parts) == 3:
                m_b, d_b, y_b = int(parts[0]), int(parts[1]), int(parts[2])
                born = _date(y_b, m_b, d_b)
                today = _date.today()
                return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        except Exception:
            return None
        return None

    out: List[Dict[str, Any]] = []
    for r in rows:
        pid = str(r["player_id"])
        meta = idx.get(pid) or {}
        games_val = (int(r["games"]) if r["games"] is not None else None) if has_games else None
        # Use the metric-specific volume column when available; fall back to games.
        if has_specific_vol:
            vol_val = int(r["vol"]) if r["vol"] is not None else None
        else:
            vol_val = games_val
        out.append({
            "player_id": pid,
            "name": meta.get("name") or "Unknown",
            "team": meta.get("team") or "",
            "position": r["position"],
            "value": float(r["value"]) if r["value"] is not None else None,
            "games": games_val,
            "vol": vol_val,
            "age": _player_age(meta),
        })
    return out


def get_player_metric_ranks(player_id: str, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Return position-relative volume ranks for one player in a given season.
    Ranks each volume metric (total carries, targets, TDs, per-game rates, etc.)
    using SQL window functions across all players at the same position.
    Returns {position, season, ranks: {metric_key: rank}}.
    Empty dict when the player has no data or the season has no volume records.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT position, season FROM player_advanced_metrics "
            "WHERE player_id = %s AND season IS NOT NULL "
            "ORDER BY as_of_date DESC LIMIT 1",
            (player_id,)
        ).fetchone()
        if not row:
            return {}
        position = row["position"]
        if season is None:
            season = int(row["season"])

        try:
            result = conn.execute("""
                WITH snapshot AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        total_carries, total_targets, total_receptions,
                        total_touches, total_rush_tds, total_rec_tds,
                        total_pass_tds, total_tds, games,
                        CASE WHEN games > 0 THEN total_carries::float / games    END AS carries_pg,
                        CASE WHEN games > 0 THEN total_targets::float / games    END AS targets_pg,
                        CASE WHEN games > 0 THEN total_receptions::float / games END AS recs_pg,
                        CASE WHEN games > 0 THEN total_touches::float / games    END AS touches_pg,
                        CASE WHEN games > 0 THEN total_rush_tds::float / games   END AS rush_tds_pg,
                        CASE WHEN games > 0 THEN total_rec_tds::float / games    END AS rec_tds_pg,
                        CASE WHEN games > 0 THEN total_pass_tds::float / games   END AS pass_tds_pg,
                        CASE WHEN games > 0 THEN total_tds::float / games        END AS total_tds_pg
                    FROM player_advanced_metrics
                    WHERE season = %s AND position = %s
                      AND games IS NOT NULL AND games > 0
                    ORDER BY player_id, as_of_date DESC
                ),
                r AS (
                    SELECT player_id,
                        RANK() OVER (ORDER BY total_carries DESC NULLS LAST)    AS total_carries,
                        RANK() OVER (ORDER BY total_targets DESC NULLS LAST)    AS total_targets,
                        RANK() OVER (ORDER BY total_receptions DESC NULLS LAST) AS total_receptions,
                        RANK() OVER (ORDER BY total_touches DESC NULLS LAST)    AS total_touches,
                        RANK() OVER (ORDER BY total_rush_tds DESC NULLS LAST)   AS total_rush_tds,
                        RANK() OVER (ORDER BY total_rec_tds DESC NULLS LAST)    AS total_rec_tds,
                        RANK() OVER (ORDER BY total_pass_tds DESC NULLS LAST)   AS total_pass_tds,
                        RANK() OVER (ORDER BY total_tds DESC NULLS LAST)        AS total_tds,
                        RANK() OVER (ORDER BY carries_pg DESC NULLS LAST)       AS carries_per_game,
                        RANK() OVER (ORDER BY targets_pg DESC NULLS LAST)       AS targets_per_game,
                        RANK() OVER (ORDER BY recs_pg DESC NULLS LAST)          AS receptions_per_game,
                        RANK() OVER (ORDER BY touches_pg DESC NULLS LAST)       AS touches_per_game,
                        RANK() OVER (ORDER BY rush_tds_pg DESC NULLS LAST)      AS rush_tds_per_game,
                        RANK() OVER (ORDER BY rec_tds_pg DESC NULLS LAST)       AS rec_tds_per_game,
                        RANK() OVER (ORDER BY pass_tds_pg DESC NULLS LAST)      AS pass_tds_per_game,
                        RANK() OVER (ORDER BY total_tds_pg DESC NULLS LAST)     AS total_tds_per_game
                    FROM snapshot
                )
                SELECT * FROM r WHERE player_id = %s
            """, (season, position, player_id)).fetchone()
        except Exception:
            return {"position": position, "season": season, "ranks": {}}

        if not result:
            return {"position": position, "season": season, "ranks": {}}

        rank_dict = dict(result)
        rank_dict.pop("player_id", None)
        return {
            "position": position,
            "season": season,
            "ranks": {k: int(v) for k, v in rank_dict.items() if v is not None},
        }


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
