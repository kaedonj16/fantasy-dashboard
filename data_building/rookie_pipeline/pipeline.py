"""
Rookie pipeline orchestration.

Run the full pipeline:
    1. Determine active draft class year
    2. Load / refresh prospect data (ingestion)
    3. Build mock draft consensus
    4. Score all prospects (prospect_model)
    5. Translate scores to dynasty values (value_translation)
    6. Upsert everything to the database
    7. Snapshot value history

Entry point:
    run_rookie_pipeline(draft_year=None)   # None = auto-detect active class

Individual steps are also exported for targeted refreshes.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Active class detection
# ─────────────────────────────────────────────────────────────────────────────

def get_active_rookie_class(today: Optional[date] = None) -> int:
    """
    Return the draft class year that should currently be displayed.

    Rules (mirrors the DB seed in migration 009):
    - Rookie season ends ≈ second week of January each year (wild-card weekend).
    - If today is on or after season_end for class Y, show class Y+1.
    - Otherwise show class Y (either in pre-draft evaluation or rookie season).

    Falls back to hardcoded logic when DB is unavailable.
    """
    if today is None:
        today = date.today()

    # Try DB first
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT draft_class_year, season_end
                    FROM   rookie_active_class
                    ORDER  BY draft_class_year
                    """
                )
                rows = cur.fetchall()
        if rows:
            for row in rows:
                year = int(row['draft_class_year'])  # Ensure integer type
                season_end = row['season_end']
                # Convert season_end to date if needed
                if season_end is not None:
                    if isinstance(season_end, str):
                        from datetime import datetime
                        try:
                            season_end = datetime.strptime(season_end, '%Y-%m-%d').date()
                        except ValueError:
                            # Try other common formats
                            try:
                                season_end = datetime.strptime(season_end, '%Y-%m-%d %H:%M:%S').date()
                            except ValueError:
                                log.warning(f"Unable to parse season_end date: {season_end}")
                                season_end = None
                    elif not isinstance(season_end, date):
                        log.warning(f"Unexpected season_end type: {type(season_end)}")
                        season_end = None
                
                if season_end is None or today <= season_end:
                    return year
            # All classes have ended → return latest + 1
            return int(rows[-1]['draft_class_year']) + 1
    except Exception as exc:
        log.warning("[pipeline] DB unavailable for active class lookup: %s", exc)

    # Fallback: heuristic
    # NFL Draft is late April. Rookie season ends ≈ wild-card weekend (Jan 12-ish).
    # Jan 1–11: still watching prior year's class in the playoffs → show year-1
    # Jan 12 onward: prior class is done; next class is upcoming → show year
    if today.month == 1 and today.day <= 11:
        return today.year - 1
    return today.year


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_available() -> bool:
    db_url = os.getenv("DATABASE_URL", "").strip()
    return bool(db_url) and not any(t in db_url for t in ("USER", "PASSWORD", "HOST"))


def upsert_prospects(prospects: List[Dict], conn) -> int:
    saved = 0
    with conn.cursor() as cur:
        for p in prospects:
            cur.execute(
                """
                INSERT INTO rookie_prospects
                    (player_id, sleeper_id, name, position, school, age,
                     height_inches, weight_lbs, hometown, state,
                     draft_class_year, early_declare, transfer_history,
                     headshot_url, updated_at)
                VALUES
                    (%(player_id)s, %(sleeper_id)s, %(name)s, %(position)s,
                     %(school)s, %(age)s, %(height_inches)s, %(weight_lbs)s,
                     %(hometown)s, %(state)s, %(draft_class_year)s,
                     %(early_declare)s, %(transfer_history)s,
                     %(headshot_url)s, NOW())
                ON CONFLICT (player_id) DO UPDATE SET
                    sleeper_id       = EXCLUDED.sleeper_id,
                    name             = EXCLUDED.name,
                    position         = EXCLUDED.position,
                    school           = EXCLUDED.school,
                    age              = EXCLUDED.age,
                    height_inches    = EXCLUDED.height_inches,
                    weight_lbs       = EXCLUDED.weight_lbs,
                    early_declare    = EXCLUDED.early_declare,
                    transfer_history = EXCLUDED.transfer_history,
                    updated_at       = NOW()
                """,
                {
                    "player_id":       p["player_id"],
                    "sleeper_id":      p.get("sleeper_id"),
                    "name":            p["name"],
                    "position":        p["position"],
                    "school":          p.get("school"),
                    "age":             p.get("age"),
                    "height_inches":   p.get("height_inches"),
                    "weight_lbs":      p.get("weight_lbs"),
                    "hometown":        p.get("hometown"),
                    "state":           p.get("state"),
                    "draft_class_year":p["draft_class_year"],
                    "early_declare":   p.get("early_declare", False),
                    "transfer_history":p.get("transfer_history"),
                    "headshot_url":    p.get("headshot_url"),
                },
            )
            saved += 1

            # Source data (seasons)
            for s in p.get("seasons") or []:
                cur.execute(
                    """
                    INSERT INTO rookie_prospect_source_data
                        (player_id, season, games_played,
                         pass_yards, pass_tds, pass_attempts, completions, interceptions,
                         rush_attempts, rush_yards, rush_tds,
                         receptions, targets, receiving_yards, receiving_tds,
                         dominator_rating, market_share_yards, market_share_tds,
                         yds_per_carry, yds_per_reception, yds_per_attempt,
                         completion_pct, td_int_ratio, team, conference,
                         team_pass_rate, team_total_yards, team_total_tds, source)
                    VALUES
                        (%(player_id)s, %(season)s, %(games_played)s,
                         %(pass_yards)s, %(pass_tds)s, %(pass_attempts)s,
                         %(completions)s, %(interceptions)s,
                         %(rush_attempts)s, %(rush_yards)s, %(rush_tds)s,
                         %(receptions)s, %(targets)s, %(receiving_yards)s, %(receiving_tds)s,
                         %(dominator_rating)s, %(market_share_yards)s, %(market_share_tds)s,
                         %(yds_per_carry)s, %(yds_per_reception)s, %(yds_per_attempt)s,
                         %(completion_pct)s, %(td_int_ratio)s, %(team)s, %(conference)s,
                         %(team_pass_rate)s, %(team_total_yards)s, %(team_total_tds)s,
                         %(source)s)
                    ON CONFLICT (player_id, season, source) DO UPDATE SET
                        games_played       = EXCLUDED.games_played,
                        receptions         = EXCLUDED.receptions,
                        receiving_yards    = EXCLUDED.receiving_yards,
                        receiving_tds      = EXCLUDED.receiving_tds,
                        rush_yards         = EXCLUDED.rush_yards,
                        rush_tds           = EXCLUDED.rush_tds,
                        pass_yards         = EXCLUDED.pass_yards,
                        pass_tds           = EXCLUDED.pass_tds,
                        dominator_rating   = EXCLUDED.dominator_rating,
                        market_share_yards = EXCLUDED.market_share_yards,
                        market_share_tds   = EXCLUDED.market_share_tds,
                        team_pass_rate     = EXCLUDED.team_pass_rate
                    """,
                    {
                        "player_id":         p["player_id"],
                        "season":            s.get("season"),
                        "games_played":      s.get("games_played"),
                        "pass_yards":        s.get("pass_yards"),
                        "pass_tds":          s.get("pass_tds"),
                        "pass_attempts":     s.get("pass_attempts"),
                        "completions":       s.get("completions"),
                        "interceptions":     s.get("interceptions"),
                        "rush_attempts":     s.get("rush_attempts"),
                        "rush_yards":        s.get("rush_yards"),
                        "rush_tds":          s.get("rush_tds"),
                        "receptions":        s.get("receptions"),
                        "targets":           s.get("targets"),
                        "receiving_yards":   s.get("receiving_yards"),
                        "receiving_tds":     s.get("receiving_tds"),
                        "dominator_rating":  s.get("dominator_rating"),
                        "market_share_yards":s.get("market_share_yards"),
                        "market_share_tds":  s.get("market_share_tds"),
                        "yds_per_carry":     s.get("yds_per_carry"),
                        "yds_per_reception": s.get("yds_per_reception"),
                        "yds_per_attempt":   s.get("yds_per_attempt"),
                        "completion_pct":    s.get("completion_pct"),
                        "td_int_ratio":      s.get("td_int_ratio"),
                        "team":              s.get("team"),
                        "conference":        s.get("conference"),
                        "team_pass_rate":    s.get("team_pass_rate"),
                        "team_total_yards":  s.get("team_total_yards"),
                        "team_total_tds":    s.get("team_total_tds"),
                        "source":            s.get("source", "cfbd"),
                    },
                )

            # Athleticism
            ath = p.get("athleticism") or {}
            if ath:
                cur.execute(
                    """
                    INSERT INTO rookie_prospect_athleticism
                        (player_id, forty_yard, vertical_inches, broad_jump_in,
                         three_cone, short_shuttle, bench_reps,
                         speed_score, ras_score, source, updated_at)
                    VALUES
                        (%(player_id)s, %(forty_yard)s, %(vertical_inches)s,
                         %(broad_jump_in)s, %(three_cone)s, %(short_shuttle)s,
                         %(bench_reps)s, %(speed_score)s, %(ras_score)s,
                         %(source)s, NOW())
                    ON CONFLICT (player_id) DO UPDATE SET
                        forty_yard       = EXCLUDED.forty_yard,
                        vertical_inches  = EXCLUDED.vertical_inches,
                        broad_jump_in    = EXCLUDED.broad_jump_in,
                        three_cone       = EXCLUDED.three_cone,
                        short_shuttle    = EXCLUDED.short_shuttle,
                        bench_reps       = EXCLUDED.bench_reps,
                        speed_score      = EXCLUDED.speed_score,
                        ras_score        = EXCLUDED.ras_score,
                        updated_at       = NOW()
                    """,
                    {
                        "player_id":       p["player_id"],
                        "forty_yard":      ath.get("forty_yard"),
                        "vertical_inches": ath.get("vertical_inches"),
                        "broad_jump_in":   ath.get("broad_jump_in"),
                        "three_cone":      ath.get("three_cone"),
                        "short_shuttle":   ath.get("short_shuttle"),
                        "bench_reps":      ath.get("bench_reps"),
                        "speed_score":     ath.get("speed_score"),
                        "ras_score":       ath.get("ras_score"),
                        "source":          ath.get("source", "seed"),
                    },
                )
    return saved


def upsert_mock_consensus(consensus_map: Dict[str, Dict], draft_year: int, conn) -> int:
    saved = 0
    with conn.cursor() as cur:
        for pid, c in consensus_map.items():
            cur.execute(
                """
                INSERT INTO rookie_mock_draft_consensus
                    (player_id, draft_class_year, projected_round, projected_pick,
                     projected_pick_low, projected_pick_high,
                     projected_draft_capital_score, num_mocks_used,
                     consensus_confidence, mock_sources, calculated_at)
                VALUES
                    (%(player_id)s, %(draft_class_year)s, %(projected_round)s,
                     %(projected_pick)s, %(projected_pick_low)s, %(projected_pick_high)s,
                     %(projected_draft_capital_score)s, %(num_mocks_used)s,
                     %(consensus_confidence)s, %(mock_sources)s::jsonb, NOW())
                ON CONFLICT (player_id) DO UPDATE SET
                    projected_round               = EXCLUDED.projected_round,
                    projected_pick                = EXCLUDED.projected_pick,
                    projected_pick_low            = EXCLUDED.projected_pick_low,
                    projected_pick_high           = EXCLUDED.projected_pick_high,
                    projected_draft_capital_score = EXCLUDED.projected_draft_capital_score,
                    num_mocks_used                = EXCLUDED.num_mocks_used,
                    consensus_confidence          = EXCLUDED.consensus_confidence,
                    mock_sources                  = EXCLUDED.mock_sources::jsonb,
                    calculated_at                 = NOW()
                """,
                {
                    "player_id":                    pid,
                    "draft_class_year":             draft_year,
                    "projected_round":              c.get("projected_round"),
                    "projected_pick":               c.get("projected_pick"),
                    "projected_pick_low":           c.get("projected_pick_low"),
                    "projected_pick_high":          c.get("projected_pick_high"),
                    "projected_draft_capital_score":c.get("projected_draft_capital_score"),
                    "num_mocks_used":               c.get("num_mocks_used"),
                    "consensus_confidence":         c.get("consensus_confidence"),
                    "mock_sources":                 str(c.get("mock_sources", [])).replace("'", '"'),
                },
            )
            saved += 1
    return saved


def upsert_rankings(scores: List[Dict], values: List[Dict], conn) -> int:
    """Merge score + value dicts and upsert into rookie_rankings."""
    value_by_pid = {v["player_id"]: v for v in values}
    saved = 0
    with conn.cursor() as cur:
        for s in scores:
            pid = s["player_id"]
            v   = value_by_pid.get(pid, {})
            cur.execute(
                """
                INSERT INTO rookie_rankings
                    (player_id, draft_class_year, overall_rank, position_rank,
                     production_score, efficiency_score, age_score,
                     breakout_profile_score, athleticism_score,
                     competition_score, environment_adjustment,
                     durability_score, projected_draft_capital_score,
                     fantasy_translation_score, confidence_score,
                     prospect_score, rookie_value, rookie_sf_value,
                     rookie_value_8, rookie_value_12, rookie_value_14,
                     rookie_sf_value_8, rookie_sf_value_12, rookie_sf_value_14,
                     tier, tier_label, key_reasons, calculated_at)
                VALUES
                    (%(player_id)s, %(draft_class_year)s, %(overall_rank)s, %(position_rank)s,
                     %(production_score)s, %(efficiency_score)s, %(age_score)s,
                     %(breakout_profile_score)s, %(athleticism_score)s,
                     %(competition_score)s, %(environment_adjustment)s,
                     %(durability_score)s, %(projected_draft_capital_score)s,
                     %(fantasy_translation_score)s, %(confidence_score)s,
                     %(prospect_score)s, %(rookie_value)s, %(rookie_sf_value)s,
                     %(rookie_value_8)s, %(rookie_value_12)s, %(rookie_value_14)s,
                     %(rookie_sf_value_8)s, %(rookie_sf_value_12)s, %(rookie_sf_value_14)s,
                     %(tier)s, %(tier_label)s, %(key_reasons)s, NOW())
                ON CONFLICT (player_id, draft_class_year) DO UPDATE SET
                    overall_rank                  = EXCLUDED.overall_rank,
                    position_rank                 = EXCLUDED.position_rank,
                    production_score              = EXCLUDED.production_score,
                    efficiency_score              = EXCLUDED.efficiency_score,
                    age_score                     = EXCLUDED.age_score,
                    breakout_profile_score        = EXCLUDED.breakout_profile_score,
                    athleticism_score             = EXCLUDED.athleticism_score,
                    competition_score             = EXCLUDED.competition_score,
                    environment_adjustment        = EXCLUDED.environment_adjustment,
                    durability_score              = EXCLUDED.durability_score,
                    projected_draft_capital_score = EXCLUDED.projected_draft_capital_score,
                    fantasy_translation_score     = EXCLUDED.fantasy_translation_score,
                    confidence_score              = EXCLUDED.confidence_score,
                    prospect_score                = EXCLUDED.prospect_score,
                    rookie_value                  = EXCLUDED.rookie_value,
                    rookie_sf_value               = EXCLUDED.rookie_sf_value,
                    rookie_value_8                = EXCLUDED.rookie_value_8,
                    rookie_value_12               = EXCLUDED.rookie_value_12,
                    rookie_value_14               = EXCLUDED.rookie_value_14,
                    rookie_sf_value_8             = EXCLUDED.rookie_sf_value_8,
                    rookie_sf_value_12            = EXCLUDED.rookie_sf_value_12,
                    rookie_sf_value_14            = EXCLUDED.rookie_sf_value_14,
                    tier                          = EXCLUDED.tier,
                    tier_label                    = EXCLUDED.tier_label,
                    key_reasons                   = EXCLUDED.key_reasons,
                    calculated_at                 = NOW()
                """,
                {
                    "player_id":                    pid,
                    "draft_class_year":             s["draft_class_year"],
                    "overall_rank":                 s.get("overall_rank"),
                    "position_rank":                s.get("position_rank"),
                    "production_score":             s.get("production_score"),
                    "efficiency_score":             s.get("efficiency_score"),
                    "age_score":                    s.get("age_score"),
                    "breakout_profile_score":       s.get("breakout_profile_score"),
                    "athleticism_score":            s.get("athleticism_score"),
                    "competition_score":            s.get("competition_score"),
                    "environment_adjustment":       s.get("environment_adjustment"),
                    "durability_score":             s.get("durability_score"),
                    "projected_draft_capital_score":s.get("projected_draft_capital_score"),
                    "fantasy_translation_score":    s.get("fantasy_translation_score"),
                    "confidence_score":             s.get("confidence_score"),
                    "prospect_score":               s.get("prospect_score"),
                    "rookie_value":                 v.get("rookie_value"),
                    "rookie_sf_value":              v.get("rookie_sf_value"),
                    "rookie_value_8":               v.get("rookie_value_8"),
                    "rookie_value_12":              v.get("rookie_value_12"),
                    "rookie_value_14":              v.get("rookie_value_14"),
                    "rookie_sf_value_8":            v.get("rookie_sf_value_8"),
                    "rookie_sf_value_12":           v.get("rookie_sf_value_12"),
                    "rookie_sf_value_14":           v.get("rookie_sf_value_14"),
                    "tier":                         v.get("tier"),
                    "tier_label":                   v.get("tier_label"),
                    "key_reasons":                  s.get("key_reasons"),
                },
            )
            saved += 1

            # Snapshot value history
            today = date.today()
            cur.execute(
                """
                INSERT INTO rookie_value_history
                    (player_id, draft_class_year, snapshot_date,
                     overall_rank, position_rank,
                     rookie_value, rookie_sf_value, prospect_score)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, snapshot_date) DO UPDATE SET
                    overall_rank    = EXCLUDED.overall_rank,
                    position_rank   = EXCLUDED.position_rank,
                    rookie_value    = EXCLUDED.rookie_value,
                    rookie_sf_value = EXCLUDED.rookie_sf_value,
                    prospect_score  = EXCLUDED.prospect_score
                """,
                (
                    pid, s["draft_class_year"], today,
                    s.get("overall_rank"), s.get("position_rank"),
                    v.get("rookie_value"), v.get("rookie_sf_value"),
                    s.get("prospect_score"),
                ),
            )
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Public API — in-memory path (no DB required)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_active_nfl_players(prospects: List[Dict]) -> List[Dict]:
    """
    Remove any prospect who is already in the NFL player database.

    Uses Sleeper's /players/nfl endpoint (no auth, public API) to get the
    full player index.  A prospect is filtered out when:
      - their normalized name matches a Sleeper player
      - AND that player has years_exp is not None (they have NFL experience)
           OR has a non-null NFL team (just drafted this cycle)

    Failures (network error, no key needed) are non-fatal — the full list is
    returned unchanged so the page still works.
    """
    try:
        from dashboard_services.api import get_nfl_players
        nfl_players = get_nfl_players()
    except Exception as exc:
        log.warning("[pipeline] Could not fetch NFL player index for dedup: %s", exc)
        return prospects

    if not nfl_players:
        return prospects

    # Build a set of lowercase names for active/drafted NFL players
    active_names: set = set()
    for pid, p in nfl_players.items():
        name = (
            p.get("full_name") or
            " ".join(filter(None, [p.get("first_name"), p.get("last_name")]))
        ).strip().lower()
        if not name:
            continue
        years_exp = p.get("years_exp")
        team      = p.get("team")
        # Include if they're in the NFL system (drafted or active)
        if years_exp is not None or team:
            active_names.add(name)

    before = len(prospects)
    filtered = [p for p in prospects if p["name"].lower() not in active_names]
    removed = before - len(filtered)
    if removed:
        log.info("[pipeline] Filtered %d already-drafted players from prospect list", removed)
    return filtered


def run_rookie_pipeline_inmemory(draft_year: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the full pipeline without writing to the database.
    Returns a dict with prospects, scores, consensus, values — ready for the
    page to consume directly or for the DB path to persist.
    """
    from .ingestion          import load_prospects_for_year
    from .mock_draft_consensus import build_mock_draft_consensus
    from .prospect_model     import score_all_prospects
    from .value_translation  import translate_all

    if draft_year is None:
        draft_year = get_active_rookie_class()

    log.info("[pipeline] Running in-memory pipeline for %d draft class", draft_year)

    prospects    = load_prospects_for_year(draft_year)
    prospects    = _filter_active_nfl_players(prospects)
    consensus    = build_mock_draft_consensus(draft_year)
    scores       = score_all_prospects(prospects, consensus)
    values       = translate_all(scores, prospects, consensus)

    return {
        "draft_year":  draft_year,
        "prospects":   prospects,
        "consensus":   consensus,
        "scores":      scores,
        "values":      values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API — full DB path
# ─────────────────────────────────────────────────────────────────────────────

def run_rookie_pipeline(draft_year: Optional[int] = None) -> Dict[str, Any]:
    """
    Full pipeline: ingest → score → translate → persist to DB.
    Falls back to in-memory mode when DB is unavailable.
    """
    result = run_rookie_pipeline_inmemory(draft_year)

    if not _db_available():
        log.info("[pipeline] DATABASE_URL not configured — returning in-memory results only")
        return result

    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            n_prospects = upsert_prospects(result["prospects"], conn)
            n_mocks     = upsert_mock_consensus(result["consensus"], result["draft_year"], conn)
            n_rankings  = upsert_rankings(result["scores"], result["values"], conn)
            conn.commit()

        log.info(
            "[pipeline] Saved %d prospects, %d mock entries, %d rankings for %d class",
            n_prospects, n_mocks, n_rankings, result["draft_year"],
        )
    except Exception as exc:
        log.error("[pipeline] DB save failed: %s", exc)

    return result


def get_rookie_rankings_from_db(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch persisted rankings from the database.  Falls back to in-memory pipeline
    if DB is unavailable or empty.
    """
    # Ensure draft_year is an integer
    draft_year = int(draft_year)
    if _db_available():
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            rr.player_id, rr.draft_class_year,
                            rp.name, rp.position, rp.school, rp.age,
                            rp.height_inches, rp.weight_lbs,
                            rp.early_declare, rp.transfer_history,
                            rr.overall_rank, rr.position_rank,
                            rr.prospect_score, rr.rookie_value, rr.rookie_sf_value,
                            rr.rookie_value_8, rr.rookie_value_12, rr.rookie_value_14,
                            rr.rookie_sf_value_8, rr.rookie_sf_value_12, rr.rookie_sf_value_14,
                            rr.tier, rr.tier_label, rr.key_reasons,
                            rr.production_score, rr.efficiency_score, rr.age_score,
                            rr.breakout_profile_score, rr.athleticism_score,
                            rr.competition_score, rr.projected_draft_capital_score,
                            rr.confidence_score, rr.calculated_at,
                            rmc.projected_round, rmc.projected_pick,
                            rmc.projected_pick_low, rmc.projected_pick_high,
                            rmc.num_mocks_used, rmc.consensus_confidence,
                            rpa.forty_yard, rpa.ras_score
                        FROM   rookie_rankings rr
                        JOIN   rookie_prospects rp  ON rp.player_id = rr.player_id
                        LEFT   JOIN rookie_mock_draft_consensus rmc ON rmc.player_id = rr.player_id
                        LEFT   JOIN rookie_prospect_athleticism rpa ON rpa.player_id = rr.player_id
                        WHERE  rr.draft_class_year = %s
                        ORDER  BY rr.overall_rank
                        """,
                        (draft_year,),
                    )
                    rows = cur.fetchall()

            if rows:
                return rows

            # DB available but tables empty — run the full pipeline to seed them
            log.info("[pipeline] DB empty for %d — running full pipeline to populate tables", draft_year)
            run_rookie_pipeline(draft_year)

            # Re-query after population
            with get_conn() as conn2:
                with conn2.cursor() as cur2:
                    cur2.execute(
                        """
                        SELECT
                            rr.player_id, rr.draft_class_year,
                            rp.name, rp.position, rp.school, rp.age,
                            rp.height_inches, rp.weight_lbs,
                            rp.early_declare, rp.transfer_history,
                            rr.overall_rank, rr.position_rank,
                            rr.prospect_score, rr.rookie_value, rr.rookie_sf_value,
                            rr.rookie_value_8, rr.rookie_value_12, rr.rookie_value_14,
                            rr.rookie_sf_value_8, rr.rookie_sf_value_12, rr.rookie_sf_value_14,
                            rr.tier, rr.tier_label, rr.key_reasons,
                            rr.production_score, rr.efficiency_score, rr.age_score,
                            rr.breakout_profile_score, rr.athleticism_score,
                            rr.competition_score, rr.projected_draft_capital_score,
                            rr.confidence_score, rr.calculated_at,
                            rmc.projected_round, rmc.projected_pick,
                            rmc.projected_pick_low, rmc.projected_pick_high,
                            rmc.num_mocks_used, rmc.consensus_confidence,
                            rpa.forty_yard, rpa.ras_score
                        FROM   rookie_rankings rr
                        JOIN   rookie_prospects rp  ON rp.player_id = rr.player_id
                        LEFT   JOIN rookie_mock_draft_consensus rmc ON rmc.player_id = rr.player_id
                        LEFT   JOIN rookie_prospect_athleticism rpa ON rpa.player_id = rr.player_id
                        WHERE  rr.draft_class_year = %s
                        ORDER  BY rr.overall_rank
                        """,
                        (draft_year,),
                    )
                    rows = cur2.fetchall()
            if rows:
                return rows

        except Exception as exc:
            log.warning("[pipeline] DB read failed: %s", exc)

    # Final fallback to in-memory (DB unavailable or pipeline population also failed)
    log.info("[pipeline] Falling back to in-memory pipeline for %d", draft_year)
    result = run_rookie_pipeline_inmemory(draft_year)
    return _merge_inmemory_result(result)


def _merge_inmemory_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge in-memory pipeline output into a flat list of row dicts."""
    from .value_translation import format_draft_capital

    prospects_by_id = {p["player_id"]: p for p in result["prospects"]}
    values_by_id    = {v["player_id"]: v for v in result["values"]}
    consensus       = result["consensus"]

    rows = []
    for s in result["scores"]:
        pid = s["player_id"]
        p   = prospects_by_id.get(pid, {})
        v   = values_by_id.get(pid, {})
        dc  = consensus.get(pid, {})

        rows.append({
            "player_id":                    pid,
            "draft_class_year":             s["draft_class_year"],
            "name":                         p.get("name", pid),
            "position":                     p.get("position"),
            "school":                       p.get("school"),
            "age":                          p.get("age"),
            "height_inches":               p.get("height_inches"),
            "weight_lbs":                  p.get("weight_lbs"),
            "early_declare":               p.get("early_declare"),
            "transfer_history":            p.get("transfer_history"),
            "overall_rank":                s.get("overall_rank"),
            "position_rank":               s.get("position_rank"),
            "prospect_score":              s.get("prospect_score"),
            "rookie_value":                v.get("rookie_value"),
            "rookie_sf_value":             v.get("rookie_sf_value"),
            "tier":                        v.get("tier"),
            "tier_label":                  v.get("tier_label"),
            "key_reasons":                 s.get("key_reasons"),
            "production_score":            s.get("production_score"),
            "efficiency_score":            s.get("efficiency_score"),
            "age_score":                   s.get("age_score"),
            "breakout_profile_score":      s.get("breakout_profile_score"),
            "athleticism_score":           s.get("athleticism_score"),
            "competition_score":           s.get("competition_score"),
            "projected_draft_capital_score": s.get("projected_draft_capital_score"),
            "confidence_score":            s.get("confidence_score"),
            "calculated_at":               None,
            "projected_round":             dc.get("projected_round"),
            "projected_pick":              dc.get("projected_pick"),
            "projected_pick_low":          dc.get("projected_pick_low"),
            "projected_pick_high":         dc.get("projected_pick_high"),
            "num_mocks_used":              dc.get("num_mocks_used"),
            "consensus_confidence":        dc.get("consensus_confidence"),
            "forty_yard":                  (p.get("athleticism") or {}).get("forty_yard"),
            "ras_score":                   (p.get("athleticism") or {}).get("ras_score"),
        })
    return rows
