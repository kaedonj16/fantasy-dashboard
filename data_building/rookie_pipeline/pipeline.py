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

import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional


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
                                print(f"Unable to parse season_end date: {season_end}")
                                season_end = None
                    elif not isinstance(season_end, date):
                        print(f"Unexpected season_end type: {type(season_end)}")
                        season_end = None
                
                if season_end is None or today <= season_end:
                    return year
            # All classes have ended → return latest + 1
            return int(rows[-1]['draft_class_year']) + 1
    except Exception as exc:
        print(f"[pipeline] DB unavailable for active class lookup: {exc}")

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
    if not db_url:
        print("[_db_available] DATABASE_URL environment variable is not set")
        return False
    if any(t in db_url for t in ("USER", "PASSWORD", "HOST")):
        print("[_db_available] DATABASE_URL contains placeholder values (USER, PASSWORD, or HOST)")
        return False
    return True


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
                    age              = COALESCE(EXCLUDED.age, rookie_prospects.age),
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
                         team_pass_rate, team_total_yards, team_total_tds,
                         team_pass_yards, sagarin_team_rating, source)
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
                         %(team_pass_yards)s, %(sagarin_team_rating)s, %(source)s)
                    ON CONFLICT (player_id, season, source) DO UPDATE SET
                        games_played         = EXCLUDED.games_played,
                        pass_yards           = EXCLUDED.pass_yards,
                        pass_tds             = EXCLUDED.pass_tds,
                        pass_attempts        = EXCLUDED.pass_attempts,
                        completions          = EXCLUDED.completions,
                        interceptions        = EXCLUDED.interceptions,
                        rush_attempts        = EXCLUDED.rush_attempts,
                        rush_yards           = EXCLUDED.rush_yards,
                        rush_tds             = EXCLUDED.rush_tds,
                        receptions           = EXCLUDED.receptions,
                        targets              = EXCLUDED.targets,
                        receiving_yards      = EXCLUDED.receiving_yards,
                        receiving_tds        = EXCLUDED.receiving_tds,
                        dominator_rating     = EXCLUDED.dominator_rating,
                        market_share_yards   = EXCLUDED.market_share_yards,
                        market_share_tds     = EXCLUDED.market_share_tds,
                        yds_per_carry        = EXCLUDED.yds_per_carry,
                        yds_per_reception    = EXCLUDED.yds_per_reception,
                        yds_per_attempt      = EXCLUDED.yds_per_attempt,
                        completion_pct       = EXCLUDED.completion_pct,
                        td_int_ratio         = EXCLUDED.td_int_ratio,
                        team_pass_rate       = EXCLUDED.team_pass_rate,
                        team                 = EXCLUDED.team,
                        conference           = EXCLUDED.conference,
                        team_total_yards     = EXCLUDED.team_total_yards,
                        team_total_tds       = EXCLUDED.team_total_tds,
                        team_pass_yards      = EXCLUDED.team_pass_yards,
                        sagarin_team_rating  = EXCLUDED.sagarin_team_rating
                    """,
                    {
                        "player_id":           p["player_id"],
                        "season":              s.get("season"),
                        "games_played":        s.get("games_played"),
                        "pass_yards":          s.get("pass_yards"),
                        "pass_tds":            s.get("pass_tds"),
                        "pass_attempts":       s.get("pass_attempts"),
                        "completions":         s.get("completions"),
                        "interceptions":       s.get("interceptions"),
                        "rush_attempts":       s.get("rush_attempts"),
                        "rush_yards":          s.get("rush_yards"),
                        "rush_tds":            s.get("rush_tds"),
                        "receptions":          s.get("receptions"),
                        "targets":             s.get("targets"),
                        "receiving_yards":     s.get("receiving_yards"),
                        "receiving_tds":       s.get("receiving_tds"),
                        "dominator_rating":    s.get("dominator_rating"),
                        "market_share_yards":  s.get("market_share_yards"),
                        "market_share_tds":    s.get("market_share_tds"),
                        "yds_per_carry":       s.get("yds_per_carry"),
                        "yds_per_reception":   s.get("yds_per_reception"),
                        "yds_per_attempt":     s.get("yds_per_attempt"),
                        "completion_pct":      s.get("completion_pct"),
                        "td_int_ratio":        s.get("td_int_ratio"),
                        "team":                s.get("team"),
                        "conference":          s.get("conference"),
                        "team_pass_rate":      s.get("team_pass_rate"),
                        "team_total_yards":    s.get("team_total_yards"),
                        "team_total_tds":      s.get("team_total_tds"),
                        "team_pass_yards":     s.get("team_pass_yards"),
                        "sagarin_team_rating": s.get("sagarin_team_rating"),
                        "source":              s.get("source", "cfbd"),
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


def _slug(name: str) -> str:
    """Convert 'Travis Hunter' → 'TRAVIS_HUNTER'. Strips periods so 'K.C.' → 'KC'."""
    import re
    return re.sub(r"[^A-Z0-9]+", "_", name.upper().replace(".", "")).strip("_")


def upsert_prospect_source_data(prospects: List[Dict], cfbd_stats: Dict, draft_year: int, conn) -> int:
    """
    Save college stats to rookie_prospect_source_data.

    Args:
        prospects: List of prospect dicts with player_id
        cfbd_stats: Dict of {name_lower: [season_dicts]}
        draft_year: Draft year
        conn: Database connection
    """
    saved = 0
    with conn.cursor() as cur:
        for prospect in prospects:
            name_key = prospect["name"].lower()
            seasons = cfbd_stats.get(name_key, [])

            for season_data in seasons:
                try:
                    # Use savepoint to allow rollback on individual errors
                    cur.execute("SAVEPOINT save_stats")
                    cur.execute(
                        """
                        INSERT INTO rookie_prospect_source_data
                            (player_id, season, games_played, pass_yards, pass_tds, pass_attempts,
                             completions, interceptions, rush_attempts, rush_yards, rush_tds,
                             receptions, targets, receiving_yards, receiving_tds,
                             dominator_rating, market_share_yards, market_share_tds,
                             yds_per_carry, yds_per_reception, yds_per_attempt,
                             completion_pct, td_int_ratio, team, conference, team_pass_rate,
                             team_total_yards, team_total_tds,
                             team_pass_yards, sagarin_team_rating, source)
                        VALUES
                            (%(player_id)s, %(season)s, %(games_played)s, %(pass_yards)s, %(pass_tds)s,
                             %(pass_attempts)s, %(completions)s, %(interceptions)s, %(rush_attempts)s,
                             %(rush_yards)s, %(rush_tds)s, %(receptions)s, %(targets)s,
                             %(receiving_yards)s, %(receiving_tds)s, %(dominator_rating)s,
                             %(market_share_yards)s, %(market_share_tds)s, %(yds_per_carry)s,
                             %(yds_per_reception)s, %(yds_per_attempt)s, %(completion_pct)s,
                             %(td_int_ratio)s, %(team)s, %(conference)s, %(team_pass_rate)s,
                             %(team_total_yards)s, %(team_total_tds)s,
                             %(team_pass_yards)s, %(sagarin_team_rating)s, %(source)s)
                        ON CONFLICT (player_id, season, source) DO UPDATE SET
                            games_played        = EXCLUDED.games_played,
                            pass_yards          = EXCLUDED.pass_yards,
                            pass_tds            = EXCLUDED.pass_tds,
                            pass_attempts       = EXCLUDED.pass_attempts,
                            completions         = EXCLUDED.completions,
                            interceptions       = EXCLUDED.interceptions,
                            rush_attempts       = EXCLUDED.rush_attempts,
                            rush_yards          = EXCLUDED.rush_yards,
                            rush_tds            = EXCLUDED.rush_tds,
                            receptions          = EXCLUDED.receptions,
                            targets             = EXCLUDED.targets,
                            receiving_yards     = EXCLUDED.receiving_yards,
                            receiving_tds       = EXCLUDED.receiving_tds,
                            dominator_rating    = EXCLUDED.dominator_rating,
                            market_share_yards  = EXCLUDED.market_share_yards,
                            market_share_tds    = EXCLUDED.market_share_tds,
                            yds_per_carry       = EXCLUDED.yds_per_carry,
                            yds_per_reception   = EXCLUDED.yds_per_reception,
                            yds_per_attempt     = EXCLUDED.yds_per_attempt,
                            completion_pct      = EXCLUDED.completion_pct,
                            td_int_ratio        = EXCLUDED.td_int_ratio,
                            team                = EXCLUDED.team,
                            conference          = EXCLUDED.conference,
                            team_pass_rate      = EXCLUDED.team_pass_rate,
                            team_total_yards    = EXCLUDED.team_total_yards,
                            team_total_tds      = EXCLUDED.team_total_tds,
                            team_pass_yards     = EXCLUDED.team_pass_yards,
                            sagarin_team_rating = EXCLUDED.sagarin_team_rating
                        """,
                        {
                            "player_id":           prospect["player_id"],
                            "season":              season_data.get("season"),
                            "games_played":        season_data.get("games_played"),
                            "pass_yards":          season_data.get("pass_yards"),
                            "pass_tds":            season_data.get("pass_tds"),
                            "pass_attempts":       season_data.get("pass_attempts"),
                            "completions":         season_data.get("completions"),
                            "interceptions":       season_data.get("interceptions"),
                            "rush_attempts":       season_data.get("rush_attempts"),
                            "rush_yards":          season_data.get("rush_yards"),
                            "rush_tds":            season_data.get("rush_tds"),
                            "receptions":          season_data.get("receptions"),
                            "targets":             season_data.get("targets"),
                            "receiving_yards":     season_data.get("receiving_yards"),
                            "receiving_tds":       season_data.get("receiving_tds"),
                            "dominator_rating":    season_data.get("dominator_rating"),
                            "market_share_yards":  season_data.get("market_share_yards"),
                            "market_share_tds":    season_data.get("market_share_tds"),
                            "yds_per_carry":       season_data.get("yds_per_carry"),
                            "yds_per_reception":   season_data.get("yds_per_reception"),
                            "yds_per_attempt":     season_data.get("yds_per_attempt"),
                            "completion_pct":      season_data.get("completion_pct"),
                            "td_int_ratio":        season_data.get("td_int_ratio"),
                            "team":                season_data.get("team"),
                            "conference":          season_data.get("conference"),
                            "team_pass_rate":      season_data.get("team_pass_rate"),
                            "team_total_yards":    season_data.get("team_total_yards"),
                            "team_total_tds":      season_data.get("team_total_tds"),
                            "team_pass_yards":     season_data.get("team_pass_yards"),
                            "sagarin_team_rating": season_data.get("sagarin_team_rating"),
                            "source":              "cfbd",
                        }
                    )
                    cur.execute("RELEASE SAVEPOINT save_stats")
                    saved += 1
                except Exception as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT save_stats")
                    print(f"[pipeline] Failed to save stats for {prospect['name']} season {season_data.get('season')}: {exc}")
    return saved


def upsert_prospect_athleticism(prospects: List[Dict], combine_data: Dict, conn) -> int:
    """
    Save combine data to rookie_prospect_athleticism.

    Args:
        prospects: List of prospect dicts with player_id
        combine_data: Dict of {name_lower: {athleticism: {...}, height_inches, weight_lbs}}
        conn: Database connection
    """
    saved = 0
    with conn.cursor() as cur:
        for prospect in prospects:
            name_key = prospect["name"].lower()
            data = combine_data.get(name_key, {})
            ath = data.get("athleticism", {})

            if not ath:
                continue  # Skip if no athleticism data

            try:
                # Use savepoint to allow rollback on individual errors
                cur.execute("SAVEPOINT save_combine")
                cur.execute(
                    """
                    INSERT INTO rookie_prospect_athleticism
                        (player_id, forty_yard, vertical_inches, broad_jump_in,
                         three_cone, short_shuttle, bench_reps, ras_score, source)
                    VALUES
                        (%(player_id)s, %(forty_yard)s, %(vertical_inches)s, %(broad_jump_in)s,
                         %(three_cone)s, %(short_shuttle)s, %(bench_reps)s, %(ras_score)s, 'nflverse')
                    ON CONFLICT (player_id) DO UPDATE SET
                        forty_yard = EXCLUDED.forty_yard,
                        vertical_inches = EXCLUDED.vertical_inches,
                        broad_jump_in = EXCLUDED.broad_jump_in,
                        three_cone = EXCLUDED.three_cone,
                        short_shuttle = EXCLUDED.short_shuttle,
                        bench_reps = EXCLUDED.bench_reps,
                        ras_score = EXCLUDED.ras_score,
                        updated_at = now()
                    """,
                    {
                        "player_id": prospect["player_id"],
                        "forty_yard": ath.get("forty_yard"),
                        "vertical_inches": ath.get("vertical_inches"),
                        "broad_jump_in": ath.get("broad_jump_in"),
                        "three_cone": ath.get("three_cone"),
                        "short_shuttle": ath.get("short_shuttle"),
                        "bench_reps": ath.get("bench_reps"),
                        "ras_score": ath.get("ras_score"),
                    }
                )
                cur.execute("RELEASE SAVEPOINT save_combine")
                saved += 1
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT save_combine")
                print(f"[pipeline] Failed to save combine data for {prospect['name']}: {exc}")
    return saved


def upsert_mock_entries_from_scraped(scraped_picks: List[Dict], draft_year: int, conn) -> int:
    """
    Save scraped mock draft entries to rookie_mock_draft_entries.

    Args:
        scraped_picks: List of dicts from scraper with player_name, position, etc.
        draft_year: Draft year
        conn: Database connection
    """
    saved = 0
    skipped = 0

    with conn.cursor() as cur:
        for pick in scraped_picks:
            player_name = pick.get("player_name", "").strip()
            if not player_name:
                continue

            # Generate player_id
            player_id = f"ROOKIE_{draft_year}_{_slug(player_name)}"

            try:
                # Use savepoint to allow rollback on individual errors
                cur.execute("SAVEPOINT save_mock")
                cur.execute(
                    """
                    INSERT INTO rookie_mock_draft_entries
                        (player_id, draft_class_year, source_name, source_url,
                         projected_round, projected_pick, mock_date, analyst_name)
                    SELECT %(player_id)s, %(draft_class_year)s, %(source_name)s, %(source_url)s,
                           %(projected_round)s, %(projected_pick)s, %(mock_date)s, %(analyst_name)s
                    WHERE EXISTS (
                        SELECT 1 FROM rookie_prospects WHERE player_id = %(player_id)s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    {
                        "player_id": player_id,
                        "draft_class_year": draft_year,
                        "source_name": pick.get("source", "Unknown"),
                        "source_url": pick.get("source_url"),
                        "projected_round": pick.get("projected_round"),
                        "projected_pick": pick.get("projected_pick"),
                        "mock_date": pick.get("mock_date"),
                        "analyst_name": pick.get("analyst_name"),
                    },
                )
                if cur.rowcount > 0:
                    saved += 1
                else:
                    skipped += 1
                cur.execute("RELEASE SAVEPOINT save_mock")
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT save_mock")
                print(f"[pipeline] Failed to save mock entry for {player_name}: {exc}")
                skipped += 1

    if skipped > 0:
        print(f"[pipeline] Skipped {skipped} mock entries (player not in prospects table)")

    return saved


def upsert_mock_entries(draft_year: int, conn) -> int:
    """Write mock draft entries to rookie_mock_draft_entries.

    Scrapes individual mocks from CBS Sports and other sources, then upserts to DB.
    Only inserts entries whose player_id already exists in rookie_prospects
    (FK constraint) — players not yet persisted are silently skipped.
    """
    from .mock_draft_consensus import get_seed_mocks
    from .mock_draft_scraper import scrape_individual_mocks
    from .pfn_scraper import scrape_pfn_mock_consensus

    # Get seed mocks (if any)
    seed_entries = get_seed_mocks(draft_year)

    # Scrape individual analyst mocks
    scraped_picks = scrape_individual_mocks(draft_year)
    
    # Scrape PFN position consensus data
    pfn_picks = scrape_pfn_mock_consensus(draft_year)
    scraped_picks.extend(pfn_picks)

    # Convert scraped picks to entry format with player_ids
    scraped_entries = []
    for pick in scraped_picks:
        player_name = pick.get("player_name", "").strip()
        if not player_name:
            continue

        # Generate player_id using same format as prospects
        player_id = f"ROOKIE_{draft_year}_{_slug(player_name)}"

        scraped_entries.append({
            "player_id": player_id,
            "source_name": pick.get("source", "Unknown"),
            "source_url": pick.get("source_url"),
            "projected_round": pick.get("projected_round"),
            "projected_pick": pick.get("projected_pick"),
            "mock_date": pick.get("mock_date"),
            "analyst_name": pick.get("analyst_name"),
        })

    # Combine seed and scraped entries
    all_entries = seed_entries + scraped_entries
    print(f"[pipeline] Upserting {len(all_entries)} mock entries ({len(seed_entries)} seed + {len(scraped_entries)} scraped)")

    saved = 0
    skipped = 0

    with conn.cursor() as cur:
        for e in all_entries:
            try:
                cur.execute(
                    """
                    INSERT INTO rookie_mock_draft_entries
                        (player_id, draft_class_year, source_name, source_url,
                         projected_round, projected_pick, mock_date, analyst_name)
                    SELECT %(player_id)s, %(draft_class_year)s, %(source_name)s, %(source_url)s,
                           %(projected_round)s, %(projected_pick)s, %(mock_date)s, %(analyst_name)s
                    WHERE EXISTS (
                        SELECT 1 FROM rookie_prospects WHERE player_id = %(player_id)s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    {
                        "player_id":        e["player_id"],
                        "draft_class_year": draft_year,
                        "source_name":      e.get("source_name", "Unknown"),
                        "source_url":       e.get("source_url"),
                        "projected_round":  e.get("projected_round"),
                        "projected_pick":   e.get("projected_pick"),
                        "mock_date":        e.get("mock_date"),
                        "analyst_name":     e.get("analyst_name"),
                    },
                )
                if cur.rowcount > 0:
                    saved += 1
                else:
                    skipped += 1
            except Exception as exc:
                print(f"[pipeline] Failed to insert mock entry for {e.get('player_id')}: {exc}")
                skipped += 1
                continue

    if skipped > 0:
        print(f"[pipeline] Skipped {skipped} mock entries (player not in prospects table)")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Post-draft: actual pick data
# ─────────────────────────────────────────────────────────────────────────────

def is_draft_complete(draft_year: int, conn=None) -> bool:
    """
    Return True if the NFL Draft for draft_year has already occurred.

    Priority:
    1. rookie_active_class.draft_date (explicit DB record)
    2. rookie_prospects table has any draft_confirmed=TRUE rows
    3. Fallback: today > April 28 of draft_year (typical draft end)
    """
    today = date.today()

    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT draft_date FROM rookie_active_class WHERE draft_class_year = %s",
                    (draft_year,),
                )
                row = cur.fetchone()
            if row and row["draft_date"]:
                draft_date = row["draft_date"]
                if isinstance(draft_date, str):
                    from datetime import datetime as _dt
                    draft_date = _dt.strptime(draft_date[:10], "%Y-%m-%d").date()
                return today >= draft_date
        except Exception:
            pass

        # Check if any actual picks are already stored
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM rookie_prospects WHERE draft_class_year = %s AND draft_confirmed = TRUE LIMIT 1",
                    (draft_year,),
                )
                if cur.fetchone():
                    return True
        except Exception:
            pass

    # Fallback: NFL Draft is held in late April (typically April 24–27)
    typical_draft_end = date(draft_year, 4, 28)
    return today >= typical_draft_end


def fetch_nflverse_draft_picks(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch actual NFL draft picks from nflverse open-data CSV.

    Returns list of dicts with keys:
        player_name, position, pick, round, nfl_team, season
    Returns [] on failure so the pipeline degrades gracefully.
    """
    import csv
    import io
    try:
        import urllib.request
        url = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[pipeline] Could not fetch nflverse draft CSV: {exc}")
        return []

    picks = []
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        try:
            season = int(row.get("season") or 0)
        except (ValueError, TypeError):
            continue
        if season != draft_year:
            continue

        try:
            pick_num = int(row.get("pick") or 0)
            rnd      = int(row.get("round") or 0)
        except (ValueError, TypeError):
            continue

        if pick_num <= 0:
            continue

        picks.append({
            "player_name": (row.get("full_name") or row.get("pfr_player_name") or "").strip(),
            "position":    (row.get("position") or "").upper().strip(),
            "pick":        pick_num,
            "round":       rnd,
            "nfl_team":    (row.get("team") or "").strip(),
            "season":      season,
        })

    print(f"[pipeline] Fetched {len(picks)} actual {draft_year} draft picks from nflverse")
    return picks


def upsert_actual_draft_picks(picks: List[Dict[str, Any]], draft_year: int, conn) -> int:
    """
    Write actual draft results into rookie_prospects
    (actual_pick, actual_round, actual_nfl_team, draft_confirmed).

    Matches on name fuzzy-equality (lowercased, suffix-stripped).
    Only updates prospects already in rookie_prospects for this draft class.
    """
    import re

    def _norm(name: str) -> str:
        n = name.lower().strip()
        n = re.sub(r"[\s,]+(jr\.?|sr\.?|ii|iii|iv|v\.?)$", "", n).strip()
        return re.sub(r"[^a-z\s]", "", n).strip()

    # Load existing prospects so we can match by name
    with conn.cursor() as cur:
        cur.execute(
            "SELECT player_id, name FROM rookie_prospects WHERE draft_class_year = %s",
            (draft_year,),
        )
        existing = {_norm(row["name"]): row["player_id"] for row in cur.fetchall()}

    updated = 0
    with conn.cursor() as cur:
        for pick in picks:
            norm = _norm(pick["player_name"])
            pid  = existing.get(norm)
            if not pid:
                continue   # not a tracked prospect — skip (undrafted / other class)

            cur.execute(
                """
                UPDATE rookie_prospects
                SET actual_pick      = %(pick)s,
                    actual_round     = %(round)s,
                    actual_nfl_team  = %(nfl_team)s,
                    draft_confirmed  = TRUE,
                    updated_at       = NOW()
                WHERE player_id = %(player_id)s
                """,
                {
                    "pick":      pick["pick"],
                    "round":     pick["round"],
                    "nfl_team":  pick["nfl_team"],
                    "player_id": pid,
                },
            )
            updated += cur.rowcount

    print(f"[pipeline] Updated actual draft picks for {updated} prospects")
    return updated


def load_actual_picks_from_db(draft_year: int, conn) -> Dict[str, Dict[str, Any]]:
    """
    Return a map of player_id → actual pick info for all prospects with confirmed picks.
    Only returns rows where draft_confirmed = TRUE.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT player_id, position, name,
                   actual_pick, actual_round, actual_nfl_team
            FROM   rookie_prospects
            WHERE  draft_class_year = %s
              AND  draft_confirmed  = TRUE
              AND  actual_pick IS NOT NULL
            """,
            (draft_year,),
        )
        rows = cur.fetchall()

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        result[row["player_id"]] = {
            "player_id":   row["player_id"],
            "position":    row["position"] or "WR",
            "name":        row["name"],
            "actual_pick": int(row["actual_pick"]),
            "actual_round":int(row["actual_round"] or ((int(row["actual_pick"]) - 1) // 32) + 1),
            "nfl_team":    row["actual_nfl_team"],
        }

    return result


def build_consensus_from_db_entries(draft_year: int, conn) -> Dict[str, Dict]:
    """
    Aggregate per-analyst rows from rookie_mock_draft_entries into a consensus
    map.  If the draft is complete, actual picks overlay the mock consensus
    with full confidence (100) and is_actual_pick=True.

    Pre-draft: uses mock medians + stdev-based confidence.
    Post-draft: actual picks replace projections for drafted players; undrafted
    prospects remain at their final mock consensus with is_actual_pick=False.

    Returns {} if no data exists at all.
    """
    from .mock_draft_consensus import pick_to_draft_capital_score
    import json, math

    # ── Step 1: build mock consensus from stored entries ──────────────────────
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.player_id,
                e.draft_class_year,
                p.name,
                p.position,
                p.school,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.projected_pick) AS median_pick,
                MIN(e.projected_pick)  AS pick_low,
                MAX(e.projected_pick)  AS pick_high,
                COUNT(*)               AS num_mocks,
                STDDEV(e.projected_pick) AS pick_stdev,
                ARRAY_AGG(DISTINCT COALESCE(e.analyst_name, e.source_name)
                          ORDER BY COALESCE(e.analyst_name, e.source_name)) AS sources
            FROM rookie_mock_draft_entries e
            JOIN rookie_prospects p ON p.player_id = e.player_id
            WHERE e.draft_class_year = %(year)s
            GROUP BY e.player_id, e.draft_class_year, p.name, p.position, p.school
            """,
            {"year": draft_year},
        )
        rows = cur.fetchall()

    def _int(v, default=0):
        try:
            return int(round(float(v))) if v is not None else default
        except (TypeError, ValueError):
            return default

    def _float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    consensus_map: Dict[str, Dict] = {}
    for row in rows:
        pid         = row["player_id"]
        median_pick = _int(row["median_pick"], 300)
        pick_low    = _int(row["pick_low"],    median_pick)
        pick_high   = _int(row["pick_high"],   median_pick)
        num_mocks   = _int(row["num_mocks"],   1)
        stdev       = _float(row["pick_stdev"], 0.0)
        sources     = [s for s in (row["sources"] or []) if s]

        if num_mocks >= 2 and stdev > 0:
            confidence = round(max(40.0, 100.0 - stdev * 5.0), 1)
        elif num_mocks == 1:
            confidence = 55.0
        else:
            confidence = 100.0

        position = row["position"] or "WR"
        consensus_map[pid] = {
            "player_name":                  row["name"] or pid,
            "position":                     position,
            "school":                       row["school"] or "",
            "projected_round":              ((median_pick - 1) // 32) + 1,
            "projected_pick":               median_pick,
            "projected_pick_low":           pick_low,
            "projected_pick_high":          pick_high,
            "projected_draft_capital_score": pick_to_draft_capital_score(median_pick, position),
            "num_mocks_used":               num_mocks,
            "consensus_confidence":         confidence,
            "mock_sources":                 sources,
            "is_actual_pick":               False,
        }

    # ── Step 2: overlay actual picks post-draft ────────────────────────────
    if is_draft_complete(draft_year, conn):
        actual_picks = load_actual_picks_from_db(draft_year, conn)
        n_actual = 0
        for pid, ap in actual_picks.items():
            position = ap["position"]
            pick     = ap["actual_pick"]
            rnd      = ap["actual_round"]
            dc_score = pick_to_draft_capital_score(pick, position)

            # Overlay on top of mock entry (or create new entry if player
            # wasn't in any mock — e.g. undrafted players won't appear here)
            existing = consensus_map.get(pid, {})
            consensus_map[pid] = {
                "player_name":                  ap["name"],
                "position":                     position,
                "school":                       existing.get("school", ""),
                "projected_round":              rnd,
                "projected_pick":               pick,
                "projected_pick_low":           pick,   # exact — no range
                "projected_pick_high":          pick,
                "projected_draft_capital_score": dc_score,
                "num_mocks_used":               existing.get("num_mocks_used", 0),
                "consensus_confidence":         100.0,  # actual pick = certainty
                "mock_sources":                 existing.get("mock_sources", []),
                "is_actual_pick":               True,
                "actual_nfl_team":              ap.get("nfl_team"),
            }
            n_actual += 1

        if n_actual:
            print(f"[pipeline] Post-draft: overlaid {n_actual} actual picks (draft complete)")

    return consensus_map


def upsert_mock_consensus(consensus_map: Dict[str, Dict], draft_year: int, conn) -> int:
    import json as _json
    from decimal import Decimal as _Decimal

    def _safe_dumps(obj):
        if isinstance(obj, _Decimal):
            return int(obj) if obj == obj.to_integral_value() else float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    saved = 0
    skipped = 0
    with conn.cursor() as cur:
        for pid, c in consensus_map.items():
            try:
                cur.execute("SAVEPOINT save_consensus")
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
                        "mock_sources":                 _json.dumps(c.get("mock_sources") or [], default=_safe_dumps),
                    },
                )
                row_count = cur.rowcount
                cur.execute("RELEASE SAVEPOINT save_consensus")
                saved += max(row_count, 1)  # rowcount=1 insert or update; treat 0 as 1 for upserts
            except Exception as exc:
                cur.execute("ROLLBACK TO SAVEPOINT save_consensus")
                print(f"[pipeline] Failed to upsert consensus for {pid}: {exc}")
                skipped += 1

    if skipped:
        print(f"[pipeline] Skipped {skipped} consensus entries (see errors above)")
    return saved


def upsert_rankings(scores: List[Dict], values: List[Dict], conn) -> int:
    """Merge score + value dicts and upsert into rookie_rankings.
    
    Priority: model_values > calculated rookie values > fallback values
    """
    
    value_by_pid = {v["player_id"]: v for v in values}
    saved = 0
    
    # Load model values to check for existing player values
    model_values = {}
    name_to_model_values = {}  # For rookie ID name matching
    try:
        from utils.utils import load_model_value_table
        model_table = load_model_value_table() or []
        for player in model_table:
            if isinstance(player, dict) and player.get("id"):
                pid = str(player["id"])
                model_values[pid] = {
                    "rookie_value": player.get("value", 0),
                    "rookie_sf_value": player.get("sf_value", 0),
                    "rookie_value_8": player.get("value_8", 0),
                    "rookie_value_12": player.get("value_12", 0),
                    "rookie_value_14": player.get("value_14", 0),
                    "rookie_sf_value_8": player.get("sf_value_8", 0),
                    "rookie_sf_value_12": player.get("sf_value_12", 0),
                    "rookie_sf_value_14": player.get("sf_value_14", 0),
                }
                
                # Also create name lookup for rookie ID matching
                name = player.get("name", "").lower().strip()
                if name:
                    name_to_model_values[name] = model_values[pid]
        
    except Exception as e:
        print(f"[upsert_rankings] Could not load model values: {e}")
    
    with conn.cursor() as cur:
        for s in scores:
            pid = s["player_id"]
            # Debug: Check if scores have tier information
            # Priority 1: Use model values if available (by player_id)
            if pid in model_values:
                v = model_values[pid]
                # Add tier information from scores
                if s.get("tier") is not None:
                    v["tier"] = s.get("tier")
                if s.get("tier_label") is not None:
                    v["tier_label"] = s.get("tier_label")
            # Priority 1.5: Use model values by name matching (for rookie IDs like "ROOKIE_2026_JEREMIYAH_LOVE")
            elif pid.startswith("ROOKIE_"):
                # Extract name from rookie ID
                name_parts = pid.split("_")
                if len(name_parts) >= 4:
                    name = "_".join(name_parts[2:]).replace("_", " ").title()
                    name_lower = name.lower().strip()
                    if name_lower in name_to_model_values:
                        v = name_to_model_values[name_lower]
                        # Add tier information from scores
                        if s.get("tier") is not None:
                            v["tier"] = s.get("tier")
                        if s.get("tier_label") is not None:
                            v["tier_label"] = s.get("tier_label")
                    else:
                        # Name matching failed, try calculated values
                        if pid in value_by_pid:
                            v = value_by_pid.get(pid, {})
                        else:
                            v = {}
                else:
                    # Name extraction failed, try calculated values
                    if pid in value_by_pid:
                        v = value_by_pid.get(pid, {})
                    else:
                        v = {}

            # Priority 2: Use calculated rookie values
            elif pid in value_by_pid:
                v = value_by_pid.get(pid, {})
                # Ensure calculated values have tier information from scores if not present
                if "tier" not in v and s.get("tier") is not None:
                    v["tier"] = s.get("tier")
                if "tier_label" not in v and s.get("tier_label") is not None:
                    v["tier_label"] = s.get("tier_label")
            # Priority 3: Use empty values (already initialized as {})
            else:
                v = {}
                # Ensure fallback values have tier information from scores
                if s.get("tier") is not None:
                    v["tier"] = s.get("tier")
                if s.get("tier_label") is not None:
                    v["tier_label"] = s.get("tier_label")
                print(f"[upsert_rankings] No values found for player {pid}, using defaults")
            
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

def _norm_name(name: str) -> str:
    """
    Normalize a player name for dedup comparison.
    Lowercases, strips punctuation, removes common generational suffixes
    so 'Harold Fannin Jr.' == 'Harold Fannin Jr' == 'Harold Fannin'.
    Also handles period normalization: 'K.C. Concepcion' == 'KC Concepcion' == 'K C Concepcion'.
    """
    import re
    n = name.lower().strip()
    # Remove periods from initials (K.C. -> KC, K. C. -> K C)
    n = re.sub(r'\.', '', n)
    # Remove trailing generational suffixes (jr, sr, ii, iii, iv, v)
    n = re.sub(r'[\s,]+(jr\.?|sr\.?|ii|iii|iv|v\.?)$', '', n).strip()
    # Strip any remaining punctuation except spaces
    n = re.sub(r'[^a-z\s]', '', n).strip()
    # Normalize multiple spaces to single space
    n = re.sub(r'\s+', ' ', n)
    # Remove spaces between single-letter initials (K C -> KC)
    n = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', n)
    return n


def _deduplicate_prospects(prospects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate prospects that have name variations.
    
    Keeps the most complete record (with more data fields) when duplicates are found.
    """
        
    # Group prospects by normalized name
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for prospect in prospects:
        norm_name = _norm_name(prospect.get("name", ""))
        if not norm_name:
            continue
        groups.setdefault(norm_name, []).append(prospect)
    
    # Find duplicates and merge them
    deduped = []
    duplicates_found = 0
    
    for norm_name, group in groups.items():
        if len(group) == 1:
            # No duplicate, keep as-is
            deduped.append(group[0])
        else:
            # Found duplicates, merge them
            duplicates_found += len(group) - 1
            merged = _merge_duplicate_prospects(group)
            deduped.append(merged)
    
    return deduped


def _merge_duplicate_prospects(duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple prospect records for the same player.
    
    Strategy: Keep the record with the most complete data, then fill in missing fields
    from other records.
    """
    if not duplicates:
        return {}
    
    # Sort by completeness (more non-None fields = more complete)
    def completeness_score(p: Dict[str, Any]) -> int:
        score = 0
        for key, value in p.items():
            if value is not None and value != "" and value != []:
                score += 1
        return score
    
    sorted_duplicates = sorted(duplicates, key=completeness_score, reverse=True)
    merged = dict(sorted_duplicates[0])  # Start with most complete record
    
    # Fill in missing fields from other records
    for other in sorted_duplicates[1:]:
        for key, value in other.items():
            if key == "name":
                # Keep the original name from the most complete record
                continue
            elif merged.get(key) in [None, "", []] and value not in [None, "", []]:
                merged[key] = value
            elif key == "seasons" and isinstance(value, list):
                # Merge seasons data
                existing_seasons = merged.get("seasons", [])
                merged_seasons = existing_seasons.copy()
                for season in value:
                    season_year = season.get("season")
                    if season_year and not any(s.get("season") == season_year for s in existing_seasons):
                        merged_seasons.append(season)
                merged["seasons"] = merged_seasons
            elif key == "athleticism" and isinstance(value, dict):
                # Merge athleticism data
                existing_ath = merged.get("athleticism", {})
                merged_ath = dict(existing_ath)
                for ath_key, ath_value in value.items():
                    if ath_value is not None and existing_ath.get(ath_key) is None:
                        merged_ath[ath_key] = ath_value
                merged["athleticism"] = merged_ath
    
    return merged


def _filter_active_nfl_players(prospects: List[Dict], draft_year: int) -> List[Dict]:
    """
    Remove any prospect who was already drafted in a PREVIOUS draft class.

    For the target draft year, we DON'T filter prospects from that class.
    We only filter veterans who were drafted 2+ years before the target year.

    The goal is to allow seed data for the current/upcoming draft class to work
    while removing players from much older classes.

    Logic:
    - Don't filter at all for now - rely on seed data to be accurate
    - Sleeper may have speculative data (years_exp=0 or 1) for upcoming draft
    - Better to have duplicates than miss prospects

    Name comparison strips generational suffixes (Jr./Sr./II/III) and
    punctuation so 'Harold Fannin Jr.' matches 'Harold Fannin Jr'.

    Failures are non-fatal — the full list is returned unchanged.
    """
    # For now, disable filtering completely to allow seed data through
    # The seed data should be curated to only include relevant prospects
    print(f"[pipeline] Keeping all {len(prospects)} prospects from seed data (filter disabled)")
    return prospects


def load_prospects_from_db(draft_year: int, conn) -> List[Dict[str, Any]]:
    """
    Load complete prospect data from database including seasons and athleticism.

    Returns prospects in the format expected by the scoring model:
    {
        "player_id": str,
        "name": str,
        "position": str,
        "age": float,
        "draft_class_year": int,
        "seasons": [...],
        "athleticism": {...}
    }
    """
    with conn.cursor() as cur:
        # Load base prospect data
        cur.execute("""
            SELECT player_id, name, position, school, age, height_inches, weight_lbs,
                   draft_class_year, early_declare
            FROM rookie_prospects
            WHERE draft_class_year = %s
        """, (draft_year,))
        prospects_rows = cur.fetchall()

        if not prospects_rows:
            print(f"[pipeline] No prospects found in database for {draft_year}")
            return []

        prospects = []
        for row in prospects_rows:
            prospect = dict(row)
            prospect.setdefault("seasons", [])
            prospect.setdefault("athleticism", {})
            prospects.append(prospect)

        print(f"[pipeline] Loaded {len(prospects)} base prospects from database")

        # Load season stats for all prospects
        cur.execute("""
            SELECT player_id, season, games_played,
                   pass_yards, pass_tds, pass_attempts, completions, interceptions,
                   rush_attempts, rush_yards, rush_tds,
                   receptions, targets, receiving_yards, receiving_tds,
                   dominator_rating, market_share_yards, market_share_tds,
                   yds_per_carry, yds_per_reception, yds_per_attempt,
                   completion_pct, td_int_ratio, team, conference, team_pass_rate,
                   team_pass_yards, sagarin_team_rating,
                   eval_snap_counts,
                   yards_after_catch, yards_after_catch_per_reception,
                   avg_depth_of_target, contested_catch_rate, avoided_tackles,
                   drop_rate, slot_rate, wide_rate, inline_rate,
                   pass_block_rate, grades_offense, grades_pass_block,
                   explosive_runs_10_plus, breakaway_percentage, elusive_rating,
                   pff_rushing_grade, pff_passing_grade, big_time_throw_rate,
                   adjusted_completion_rate, pressure_to_sack_rate, nfl_passer_rating
            FROM rookie_prospect_source_data
            WHERE player_id IN (
                SELECT player_id FROM rookie_prospects WHERE draft_class_year = %s
            )
            AND source != 'rookie_eval'
            ORDER BY player_id, season
        """, (draft_year,))
        stats_rows = cur.fetchall()

        # Group stats by player_id
        stats_by_player: Dict[str, List[Dict]] = {}
        for stat_row in stats_rows:
            stat_dict = dict(stat_row)
            player_id = stat_dict.pop("player_id")
            stats_by_player.setdefault(player_id, []).append(stat_dict)

        print(f"[pipeline] Loaded {len(stats_rows)} season stat records")

        # Load athleticism for all prospects
        cur.execute("""
            SELECT player_id, forty_yard, vertical_inches, broad_jump_in,
                   three_cone, short_shuttle, bench_reps, speed_score, ras_score
            FROM rookie_prospect_athleticism
            WHERE player_id IN (
                SELECT player_id FROM rookie_prospects WHERE draft_class_year = %s
            )
        """, (draft_year,))
        ath_rows = cur.fetchall()

        # Map athleticism by player_id
        ath_by_player: Dict[str, Dict] = {}
        for ath_row in ath_rows:
            ath_dict = dict(ath_row)
            player_id = ath_dict.pop("player_id")
            ath_by_player[player_id] = ath_dict

        print(f"[pipeline] Loaded {len(ath_rows)} athleticism records")

        # ── Load eval metrics (primary: snapshots; fallback: per-season rows) ──
        player_ids_for_class = [p["player_id"] for p in prospects]
        cur.execute("""
            SELECT DISTINCT ON (player_id)
                player_id,
                profile_json -> 'rookie_profile' -> 'metrics' AS eval_metrics
            FROM rookie_profiles_snapshots
            WHERE draft_class_year = %s
              AND player_id = ANY(%s)
            ORDER BY player_id, snapshot_date DESC
        """, (draft_year, player_ids_for_class))
        snapshot_rows = cur.fetchall()

        eval_by_player: Dict[str, Dict] = {}
        players_with_snapshot = set()
        for snap_row in snapshot_rows:
            pid = snap_row["player_id"]
            raw = snap_row["eval_metrics"]
            if isinstance(raw, dict) and raw:
                eval_by_player[pid] = raw
                players_with_snapshot.add(pid)

        # Fallback: for players without a snapshot, merge per-season JSONB rows
        players_needing_fallback = [
            p["player_id"] for p in prospects
            if p["player_id"] not in players_with_snapshot
        ]
        if players_needing_fallback:
            cur.execute("""
                SELECT player_id, season, rookie_eval_metrics
                FROM rookie_prospect_source_data
                WHERE player_id = ANY(%s)
                  AND rookie_eval_metrics IS NOT NULL
                ORDER BY player_id, season ASC
            """, (players_needing_fallback,))
            fallback_rows = cur.fetchall()

            fallback_by_player: Dict[str, Dict] = {}
            for fb_row in fallback_rows:
                pid = fb_row["player_id"]
                season_metrics = fb_row["rookie_eval_metrics"]
                if isinstance(season_metrics, dict):
                    fallback_by_player.setdefault(pid, {}).update(season_metrics)
            eval_by_player.update(fallback_by_player)

        n_eval = sum(1 for v in eval_by_player.values() if v)
        print(f"[pipeline] Loaded eval metrics for {n_eval}/{len(prospects)} prospects "
              f"({len(players_with_snapshot)} from snapshots, "
              f"{n_eval - len(players_with_snapshot)} from fallback rows)")

        # Attach stats, athleticism, and eval metrics to prospects
        for prospect in prospects:
            player_id = prospect["player_id"]
            prospect["seasons"] = stats_by_player.get(player_id, [])
            prospect["athleticism"] = ath_by_player.get(player_id, {})
            eval_metrics = eval_by_player.get(player_id)
            if eval_metrics:
                prospect["_eval_metrics"] = eval_metrics

        return prospects


def run_rookie_pipeline_inmemory(
    draft_year: Optional[int] = None,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
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

    print(f"[pipeline] Running in-memory pipeline for {draft_year} draft class")

    prospects    = load_prospects_for_year(draft_year)
    prospects    = _filter_active_nfl_players(prospects, draft_year)
    consensus    = build_mock_draft_consensus(draft_year)

    # If no prospects but we have mock draft data, create prospects from mocks
    if not prospects and consensus:
        print("[pipeline] No prospects found - creating from mock draft data")
        from .ingestion import prospects_from_mock_draft
        from .mock_draft_scraper import scrape_consensus_mock_draft

        mock_picks = scrape_consensus_mock_draft(draft_year)
        prospects = prospects_from_mock_draft(mock_picks, draft_year)

    scores       = score_all_prospects(
        prospects,
        consensus,
        position_weights_override=position_weights_override,
    )
    values       = translate_all(scores, prospects, consensus)

    return {
        "draft_year":  draft_year,
        "prospects":   prospects,
        "consensus":   consensus,
        "scores":      scores,
        "values":      values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DB-only scoring path (used when ingestion APIs are unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def _score_from_db(
    draft_year: int,
    get_conn_fn,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Score prospects using data already persisted in the DB.

    Called when Sportradar is unreachable (no key, 403, etc.) but the DB
    already has prospects, stats, athleticism, and eval metrics from a prior
    evaluation pipeline run.  Runs Stages 5-6 only:
      5. Build mock-draft consensus from existing rookie_mock_draft_entries
      6. Load prospects + eval metrics → score → translate → upsert rankings
    """
    from .prospect_model import score_all_prospects
    from .value_translation import translate_all

    print("[pipeline] ====== DB-ONLY SCORING: checking existing prospects ======")

    # Stage 4: Scrape mock drafts (including PFN) even in DB-only mode
    print("[pipeline] ====== STAGE 4: Scrape Mock Drafts ======")
    with get_conn_fn() as conn:
        n_mock_entries = upsert_mock_entries(draft_year, conn)
    print(f"[pipeline] STAGE 4 COMPLETE: Saved {n_mock_entries} mock entries to rookie_mock_draft_entries")

    with get_conn_fn() as conn:
        existing_prospects = load_prospects_from_db(draft_year, conn)

    if not existing_prospects:
        print(f"[pipeline] No prospects in DB for {draft_year} — cannot score. "
              "Run evaluation pipeline and ensure prospects are seeded first.")
        return {}

    print(f"[pipeline] Found {len(existing_prospects)} prospects in DB for {draft_year}")
    n_with_eval = sum(1 for p in existing_prospects if p.get("_eval_metrics"))
    print(f"[pipeline] Eval metrics loaded from DB: {n_with_eval}/{len(existing_prospects)} prospects")
    if n_with_eval == 0:
        print("[pipeline] WARNING: No eval metrics in DB. Run evaluation pipeline first to populate them.")

    # Stage 5: build consensus from stored mock entries
    print("[pipeline] ====== STAGE 5: Build Mock Draft Consensus ======")
    with get_conn_fn() as conn:
        consensus_map = build_consensus_from_db_entries(draft_year, conn)

    print(f"[pipeline] Built consensus from {len(consensus_map)} players")
    if consensus_map:
        sample = list(consensus_map.items())[:3]
        for pid, c in sample:
            print(f"[pipeline]   sample: {pid} pick={c.get('projected_pick')} mocks={c.get('num_mocks_used')}")

    with get_conn_fn() as conn:
        n_consensus = upsert_mock_consensus(consensus_map, draft_year, conn)
    print(f"[pipeline] STAGE 5 COMPLETE: Saved {n_consensus} consensus records")

    # Stage 6: score + translate + save rankings
    print("[pipeline] ====== STAGE 6: Calculate Rookie Values ======")
    scores = score_all_prospects(
        existing_prospects,
        consensus_map,
        position_weights_override=position_weights_override,
    )
    print(f"[pipeline] Scored {len(scores)} prospects")

    values = translate_all(scores, existing_prospects, consensus_map)
    print(f"[pipeline] Calculated values for {len(values)} prospects")

    with get_conn_fn() as conn:
        n_rankings = upsert_rankings(scores, values, conn)
    print(f"[pipeline] STAGE 6 COMPLETE: Saved {n_rankings} rankings to rookie_rankings")

    print("[pipeline] ====== DB-ONLY SCORING COMPLETE ======")

    return {
        "draft_year": draft_year,
        "prospects": existing_prospects,
        "consensus": consensus_map,
        "scores": scores,
        "values": values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API — full DB path
# ─────────────────────────────────────────────────────────────────────────────

def run_rookie_pipeline_staged(
    draft_year: Optional[int] = None,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Staged pipeline with DB saves after each step:
    1. Fetch prospects + bio data → save to rookie_prospects
    2. Fetch stats → save to rookie_prospect_source_data
    3. Fetch combine data → save to rookie_prospect_athleticism
    4. Scrape mock drafts → save to rookie_mock_draft_entries
    5. Calculate consensus → save to rookie_mock_draft_consensus
    6. Calculate values → save to rookie_rankings
    """
    from dashboard_services.db import get_conn
    from .ingestion import (
        fetch_sportradar_prospects,
        fetch_cfbd_college_stats,
        fetch_local_combine_csv,
    )
    from .sagarin import get_team_rating as _sagarin_get_team_rating
    from .mock_draft_scraper import scrape_consensus_mock_draft, scrape_individual_mocks
    from .prospect_model import score_all_prospects
    from .value_translation import translate_all

    if draft_year is None:
        draft_year = get_active_rookie_class()

    if not _db_available():
        print("[pipeline] DATABASE_URL not configured — cannot run staged pipeline")
        return {}

    print("[pipeline] ====== STAGE 1: Fetch Prospects + Bio Data ======")

    # Check for required API keys
    import os
    _has_sr_key = bool(os.getenv("SPORTRADAR_API_KEY"))
    if not _has_sr_key:
        print("[pipeline] SPORTRADAR_API_KEY not set - attempting to use cached data")

    # Fetch prospects from Sportradar (use cache if no key)
    if _has_sr_key:
        sr_prospects = fetch_sportradar_prospects(draft_year)
    else:
        # Try to load from cache when API key is not available
        try:
            from .ingestion import get_seed_prospects
            sr_prospects = get_seed_prospects(draft_year)
            print(f"[pipeline] Loaded {len(sr_prospects)} prospects from cached seed data")
        except Exception as exc:
            print(f"[pipeline] Failed to load cached prospects: {exc}")
            sr_prospects = []
    if not sr_prospects:
        print("[pipeline] No prospects from Sportradar, attempting to create from mock data")
        try:
            from .ingestion import prospects_from_mock_draft
            from .mock_draft_scraper import scrape_consensus_mock_draft

            mock_picks = scrape_consensus_mock_draft(draft_year)
            sr_prospects = prospects_from_mock_draft(mock_picks, draft_year)
            print(f"[pipeline] Created {len(sr_prospects)} prospects from mock draft data")
        except Exception as exc:
            print(f"[pipeline] Failed to create prospects from mock data: {exc}")
            return {}

        if not sr_prospects:
            print("[pipeline] No prospects from Sportradar or mock data, cannot continue")
            return {}
        if _has_sr_key:
            print("[pipeline] No prospects from Sportradar — falling back to DB-only scoring")

        # Persist mock-created prospects so Stage 4 can link their picks (FK constraint).
        # Without this upsert, all 194 scraped mock entries are skipped because the
        # player_ids don't exist in rookie_prospects yet.
        try:
            from .ingestion import normalize_prospect
            normalized = [normalize_prospect(p) for p in sr_prospects]
            with get_conn() as conn:
                n_saved = upsert_prospects(normalized, conn)
                conn.commit()
            print(f"[pipeline] Saved {n_saved} mock-created prospects to DB")
        except Exception as exc:
            print(f"[pipeline] Failed to save mock prospects to DB (non-fatal): {exc}")

        # Fall back: score from the DB — now includes the newly inserted prospects
        return _score_from_db(
            draft_year,
            get_conn,
            position_weights_override=position_weights_override,
        )

    print(f"[pipeline] Loaded {len(sr_prospects)} prospects")

    # Deduplicate prospects with name variations (e.g., "K.C. Concepcion" == "KC Concepcion")
    sr_prospects = _deduplicate_prospects(sr_prospects)

    # Estimate ages from experience (SR/JR/SO/FR) — rough fallback when ESPN fails
    from .ingestion import _estimate_age
    for p in sr_prospects:
        experience = p.get("_experience")
        if not p.get("age") and experience:
            p["age"] = _estimate_age(experience, draft_year)

    ages_from_exp = sum(1 for p in sr_prospects if p.get("age"))
    print(f"[pipeline] Estimated ages for {ages_from_exp}/{len(sr_prospects)} prospects from experience")

    # Age lookup — ESPN first, PlayerProfiler fallback.
    # Only fetches for prospects that don't already have an age in the DB.
    print("[pipeline] ====== STAGE 1b: Age Lookup (ESPN → PlayerProfiler) ======")

    # Pre-populate ages from DB so we skip prospects we already know
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, age FROM rookie_prospects "
                "WHERE draft_class_year = %s AND age IS NOT NULL",
                (draft_year,),
            )
            _db_ages = {row["name"].lower().strip(): row["age"] for row in cur.fetchall()}

    _preloaded = 0
    for p in sr_prospects:
        if not p.get("age"):
            db_age = _db_ages.get(p["name"].lower().strip())
            if db_age is not None:
                p["age"] = db_age
                _preloaded += 1
    print(f"[pipeline] Pre-loaded {_preloaded} ages from DB ({len(_db_ages)} total on record)")

    _missing = [p["name"] for p in sr_prospects if not p.get("age")]
    print(f"[pipeline] {len(_missing)} prospects still need age lookup")

    if len(_missing) > 50:
        try:
            print("[pipeline] Retrieving PlayerProfiler ages")
            from .playerprofiler_scraper import fetch_playerprofiler_ages
            age_map = fetch_playerprofiler_ages(_missing)

            _resolved = 0
            for p in sr_prospects:
                key = p["name"].lower().strip()
                if key in age_map:
                    p["age"] = age_map[key]
                    _resolved += 1
            print(f"[pipeline] Resolved {_resolved}/{len(_missing)} missing ages")
        except Exception as exc:
            print(f"[pipeline] Age lookup failed — {type(exc).__name__}: {exc} (continuing without ages)")

    ages_total = sum(1 for p in sr_prospects if p.get("age"))
    print(f"[pipeline] Total prospects with age set: {ages_total}/{len(sr_prospects)}")

    # Save prospects to DB (age reflects ESPN DOB where available, else experience estimate)
    with get_conn() as conn:
        n_prospects = upsert_prospects(sr_prospects, conn)
        # Let context manager handle commit

    # Check after transaction completes
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM rookie_prospects WHERE draft_class_year = %s", (draft_year,))
            count_in_db = cur.fetchone()["count"]
            print(f"[pipeline] DEBUG: Database shows {count_in_db} prospects after transaction")

    print(f"[pipeline] STAGE 1 COMPLETE: Saved {n_prospects} prospects to rookie_prospects")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 2: Fetch College Stats ======")

    # Skip CFBD fetch if we already have non-zero stats in the DB for this class.
    # CFBD has a strict rate limit (~600 req/hr); college stats don't change once
    # the season ends, so there's no value in re-fetching on every pipeline run.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) as count FROM rookie_prospect_source_data rsd
                JOIN rookie_prospects rp ON rsd.player_id = rp.player_id
                WHERE rp.draft_class_year = %s
                  AND (rsd.receiving_yards > 0 OR rsd.rush_yards > 0 OR rsd.pass_yards > 0)
                """,
                (draft_year,),
            )
            existing_stats = cur.fetchone()["count"]

    if existing_stats > 0:
        print(f"[pipeline] STAGE 2 SKIPPED: {existing_stats} non-zero stat records already in DB for {draft_year}")
        n_stats = 0
    else:
        cfbd_stats = fetch_cfbd_college_stats(draft_year)
        print(f"[pipeline] Fetched stats for {len(cfbd_stats)} players")

        with get_conn() as conn:
            n_stats = upsert_prospect_source_data(sr_prospects, cfbd_stats, draft_year, conn)

        print(f"[pipeline] STAGE 2 COMPLETE: Saved {n_stats} stat records to rookie_prospect_source_data")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 3: Fetch Combine Data ======")

    combine_data = fetch_local_combine_csv(draft_year)
    print(f"[pipeline] Fetched combine data for {len(combine_data)} players")

    # Back-fill ages for prospects ESPN missed using NFLVerse combine birthdate
    # Also copy height and weight from combine data if not already present
    from datetime import date as _date
    from .espn_scraper import parse_dob_and_calculate_age as _parse_dob
    _ref = _date(draft_year, 4, 25)
    _combine_ages = 0
    _combine_height = 0
    _combine_weight = 0
    for p in sr_prospects:
        key = p["name"].lower().strip()
        combine = combine_data.get(key) or {}
        
        # Back-fill age if missing
        if not p.get("age"):
            bd = combine.get("birthdate")
            if bd:
                _, age = _parse_dob(bd, _ref)
                if age:
                    p["age"] = age
                    _combine_ages += 1
        
        # Copy height from combine if missing
        if not p.get("height_inches"):
            height = combine.get("height_inches")
            if height:
                p["height_inches"] = height
                _combine_height += 1
        
        # Copy weight from combine if missing
        if not p.get("weight_lbs"):
            weight = combine.get("weight_lbs")
            if weight:
                p["weight_lbs"] = weight
                _combine_weight += 1
    
    if _combine_ages or _combine_height or _combine_weight:
        print(f"[pipeline] Resolved {_combine_ages} ages, {_combine_height} heights, {_combine_weight} weights from combine data")
        with get_conn() as conn:
            upsert_prospects(sr_prospects, conn)
    else:
        print("[pipeline] No additional data from combine")

    # Save combine data to DB
    with get_conn() as conn:
        n_combine = upsert_prospect_athleticism(sr_prospects, combine_data, conn)

    print(f"[pipeline] STAGE 3 COMPLETE: Saved {n_combine} records to rookie_prospect_athleticism")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 4: Scrape Mock Drafts ======")

    all_mock_picks: List[Dict] = []

    # 4a — CBS Sports: individual analyst mocks
    individual_mocks = scrape_individual_mocks(draft_year)
    print(f"[pipeline] Scraped {len(individual_mocks)} CBS individual mock entries")
    all_mock_picks.extend(individual_mocks)

    # 4b — FantasyPros: consensus mock (counts as one analyst "source")
    fantasypros_picks = scrape_consensus_mock_draft(draft_year)
    print(f"[pipeline] Scraped {len(fantasypros_picks)} FantasyPros consensus picks")
    for pick in fantasypros_picks:
        pick.setdefault("source", "FantasyPros")
        pick.setdefault("analyst_name", "FantasyPros Consensus")
    all_mock_picks.extend(fantasypros_picks)

    # Save all entries to DB (CBS + FantasyPros in one pass)
    with get_conn() as conn:
        n_mock_entries = upsert_mock_entries_from_scraped(all_mock_picks, draft_year, conn)

    print(f"[pipeline] STAGE 4 COMPLETE: Saved {n_mock_entries} mock entries to rookie_mock_draft_entries")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 4b: Actual Draft Results (post-draft only) ======")

    with get_conn() as conn:
        draft_done = is_draft_complete(draft_year, conn)

    if draft_done:
        print(f"[pipeline] Draft complete — fetching actual {draft_year} picks from nflverse")
        actual_picks = fetch_nflverse_draft_picks(draft_year)
        if actual_picks:
            with get_conn() as conn:
                n_actual = upsert_actual_draft_picks(actual_picks, draft_year, conn)
            print(f"[pipeline] STAGE 4b COMPLETE: Stored actual picks for {n_actual} prospects")
        else:
            print("[pipeline] STAGE 4b: No actual picks fetched (nflverse data not yet available)")
    else:
        print(f"[pipeline] Draft not yet complete for {draft_year} — skipping actual pick fetch")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 5: Build Mock Draft Consensus ======")

    # Aggregate all stored entries (CBS analysts + FantasyPros) into one consensus.
    # Median pick, spread, and confidence are computed across all sources.
    with get_conn() as conn:
        consensus_map = build_consensus_from_db_entries(draft_year, conn)

    if consensus_map:
        print(f"[pipeline] Built consensus from {len(consensus_map)} players across all sources")

    # Save consensus to DB
    with get_conn() as conn:
        n_consensus = upsert_mock_consensus(consensus_map, draft_year, conn)

    print(f"[pipeline] STAGE 5 COMPLETE: Saved {n_consensus} consensus records to rookie_mock_draft_consensus")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== STAGE 6: Calculate Rookie Values ======")

    # Load complete prospect data from database (with seasons and athleticism)
    with get_conn() as conn:
        complete_prospects = load_prospects_from_db(draft_year, conn)

    print(f"[pipeline] Loaded {len(complete_prospects)} complete prospects from database")

    # ── Eval metrics were loaded from DB by load_prospects_from_db ─────────
    # run_rookie_evaluation_pipeline must have been called before this pipeline
    # (see scripts/populate_rookie_data.py) so metrics are already in the DB.
    n_with_eval = sum(1 for p in complete_prospects if p.get("_eval_metrics"))
    print(f"[pipeline] Eval metrics loaded from DB: {n_with_eval}/{len(complete_prospects)} prospects")
    if n_with_eval == 0:
        print("[pipeline] WARNING: No eval metrics in DB. Run evaluation pipeline first to populate them.")

    # Score prospects
    scores = score_all_prospects(
        complete_prospects,
        consensus_map,
        position_weights_override=position_weights_override,
    )
    print(f"[pipeline] Scored {len(scores)} prospects")

    # Translate to values
    values = translate_all(scores, complete_prospects, consensus_map)
    print(f"[pipeline] Calculated values for {len(values)} prospects")

    # Save rankings to DB
    with get_conn() as conn:
        n_rankings = upsert_rankings(scores, values, conn)

    print(f"[pipeline] STAGE 6 COMPLETE: Saved {n_rankings} rankings to rookie_rankings")

    # ──────────────────────────────────────────────────────────────────────────
    print("[pipeline] ====== PIPELINE COMPLETE ======")
    print(f"[pipeline] Summary: {n_prospects} prospects, {n_stats} stats, {n_combine} combine, {n_mock_entries} mock entries, {n_consensus} consensus, {n_rankings} rankings")

    return {
        "draft_year": draft_year,
        "prospects": complete_prospects,
        "consensus": consensus_map,
        "scores": scores,
        "values": values,
    }


def run_rookie_pipeline(
    draft_year: Optional[int] = None,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: ingest → score → translate → persist to DB.

    Caller-supplied position_weights_override is used as-is.
    To apply calibrated weights, pass them explicitly via
    historical_calibration.get_calibrated_weights().
    """
    return run_rookie_pipeline_staged(
        draft_year,
        position_weights_override=position_weights_override,
    )


def get_rookie_rankings_from_db(draft_year: int) -> List[Dict[str, Any]]:
    """
    Fetch persisted rankings from the database.  Returns an empty list if no
    data exists for the requested year — does not auto-run the pipeline.
    
    Priority: Model values table > rookie_rankings table > calculated values
    """
    # Ensure draft_year is an integer
    draft_year = int(draft_year)
    
    # First, check if player exists in model values table
    model_values = {}
    try:
        from utils.utils import load_model_value_table
        model_table = load_model_value_table() or []
        for player in model_table:
            player_id = str(player.get('id'))
            if player_id:
                model_values[player_id] = {
                    "rookie_value": float(player.get('value', 0)),
                    "rookie_sf_value": float(player.get('sf_value', player.get('value', 0))),
                    "rookie_value_8": float(player.get('value_8', player.get('value', 0))),
                    "rookie_value_12": float(player.get('value_12', player.get('value', 0))),
                    "rookie_value_14": float(player.get('value_14', player.get('value', 0))),
                    "rookie_sf_value_8": float(player.get('sf_value_8', player.get('sf_value', player.get('value', 0)))),
                    "rookie_sf_value_12": float(player.get('sf_value_12', player.get('sf_value', player.get('value', 0)))),
                    "rookie_sf_value_14": float(player.get('sf_value_14', player.get('sf_value', player.get('value', 0)))),
                }
        print(f"[get_rookie_rankings] Found {len(model_values)} players in model values table")
    except Exception as e:
        print(f"[get_rookie_rankings] Could not load model values: {e}")
    
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
                            rp.headshot_url,
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

            print(f"[pipeline] No rankings in DB for draft_year={draft_year}")
            return []
        except Exception as exc:
            print(f"[pipeline] Error loading rankings: {exc}")
            return []

    return []


def _merge_inmemory_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge in-memory pipeline output into a flat list of row dicts."""

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
