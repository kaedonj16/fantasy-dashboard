from __future__ import annotations

import logging
import time
from pathlib import Path

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.api import get_nfl_state
from data_building.external_data.sleeper_usage import write_usage_table_snapshot
from data_building.external_data.team_enrichment import (
    enrich_all_team_info,
    enrich_teams_index_with_rushing,
)
from data_building.player_value_history import record_model_value_snapshot
from data_building.value_model_training import rewrite_value_table_with_model
from utils.utils import (
    path_teams_index,
    load_teams_index,
    load_model_value_table,
    get_live_game_ids_for_today,
    load_week_schedule,
    build_and_save_week_stats_for_league, )
from utils.coerce import safe_float as _safe_float


def build_daily_data(season: int, week: int, force: bool = False):
    """
    Build usage table and vendor values.

    NOTE: This does NOT build model values - that should be done AFTER
    advanced metrics are calculated (see build_daily_model_values).

    force=True bypasses the same-day vendor-CSV freshness guard. The daily cron
    runs on an ephemeral container that checks out the repo fresh, so the
    committed vendor CSVs get a "today" mtime even though their data is stale —
    the guard would otherwise skip the scrape and keep serving old vendor values.
    """
    from data_building.external_data.external_values_scraper import (
        scrape_all_vendor_values,
        load_fantasycalc_api_values,
        load_dynastyprocess_values,
    )
    nfl_state = get_nfl_state() or {}
    season_type = (nfl_state.get("season_type") or "").lower()
    offseason_mode = season_type == "off"

    if not offseason_mode and week >= 1:
        live_game_ids = get_live_game_ids_for_today(load_week_schedule(season, week))
        build_and_save_week_stats_for_league(load_teams_index(), season, week, live_game_ids)

    # Refresh vendor CSVs only when they are stale (not written today).
    # Originally ungated to fix a bug where CSVs were never re-downloaded;
    # the freshness guard below preserves daily freshness while avoiding
    # redundant vendor API hits on every app restart within the same day.
    from datetime import date as _date
    _data_dir = Path(__file__).resolve().parents[1] / "data"
    def _csvs_fresh_today():
        for _name in ("fantasycalc_api_values.csv", "fantasycalc_sf_api_values.csv", "dynastyprocess_values.csv"):
            _p = _data_dir / _name
            if not _p.exists() or _date.fromtimestamp(_p.stat().st_mtime) != _date.today():
                return False
        return True
    if not force and _csvs_fresh_today():
        print("[build_daily_data] Vendor CSVs already fresh today, skipping scrape")
    else:
        if force:
            print("[build_daily_data] force=True — scraping vendor values regardless of CSV mtime")
        scrape_all_vendor_values()

    # Only fetch weeks 1 through current week (or max 18)
    # In offseason, fetch last season's full data
    weeks_to_fetch = range(1, min(week + 1, 19)) if not offseason_mode and week >= 1 else range(1, 19)

    print(f"[build_daily_data] Refreshing usage table for weeks {list(weeks_to_fetch)}")

    # Always refresh usage table daily (for up-to-date stats)
    write_usage_table_snapshot(season, weeks=weeks_to_fetch)
    enrich_all_team_info(season)
    enrich_teams_index_with_rushing(Path(path_teams_index()))

    # Rookie evaluation snapshots (prospect-focused, draft-class aware)
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.rookie_evaluation_pipeline import run_rookie_evaluation_pipeline

        draft_class_year = get_active_rookie_class()
        rookie_result = run_rookie_evaluation_pipeline(draft_class_year)
        db_res = rookie_result.get("db_result") or {}
        bridge = db_res.get("db_bridge_rows") or {}
        print(
            "[build_daily_data] Rookie evaluation updated "
            f"class={rookie_result.get('draft_class_year')} "
            f"profiles={rookie_result.get('profile_count')} "
            f"bridge_updated={bridge.get('updated', 0)} "
            f"bridge_inserted={bridge.get('inserted', 0)}"
        )
    except Exception as exc:
        print(f"[build_daily_data] Rookie evaluation pipeline failed: {exc}")


def build_daily_model_values():
    """
    Build model values using ML model.

    MUST be called AFTER build_daily_advanced_metrics() so that advanced
    metrics are available for the model to use.
    """
    import sys
    from pathlib import Path
    from datetime import date

    print(f"[build_daily_data] Building model values with advanced metrics")

    json_path = Path(__file__).resolve().parents[1] / "data" / "model_values.json"
    mtime_before = json_path.stat().st_mtime if json_path.exists() else None

    try:
        out_path = rewrite_value_table_with_model()
        print(f"[build_daily_data] rewrite_value_table_with_model wrote {out_path}")
    except Exception as e:
        print(f"[build_daily_data] FAILED at rewrite_value_table_with_model: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    if json_path.exists():
        mtime_after = json_path.stat().st_mtime
        if mtime_before == mtime_after:
            print(f"[build_daily_data] WARNING: {json_path.name} mtime unchanged — write may have silently failed", file=sys.stderr)
        elif date.fromtimestamp(mtime_after) != date.today():
            print(f"[build_daily_data] WARNING: {json_path.name} mtime is {date.fromtimestamp(mtime_after)}, not today", file=sys.stderr)
        else:
            print(f"[build_daily_data] {json_path.name} refreshed (mtime now {date.fromtimestamp(mtime_after)})")
    else:
        print(f"[build_daily_data] FAILED: {json_path} does not exist after rewrite", file=sys.stderr)
        raise RuntimeError(f"{json_path.name} missing after rewrite")

    try:
        model_value_table = load_model_value_table(apply_calibration=False) or []
    except Exception as e:
        print(f"[build_daily_data] FAILED at load_model_value_table: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    if not model_value_table:
        print(f"[build_daily_data] WARNING: load_model_value_table returned empty list", file=sys.stderr)

    try:
        n = record_model_value_snapshot(model_value_table, ema_alpha=1.0)
        print(f"[build_daily_data] record_model_value_snapshot wrote {n} rows")
    except Exception as e:
        print(f"[build_daily_data] FAILED at record_model_value_snapshot: {type(e).__name__}: {e}", file=sys.stderr)
        raise


def record_calibrated_history_snapshot() -> int:
    """
    Write today's player_value_history snapshot using COALESCE(calibrated, raw)
    values from player_values - the same values shown on the rankings page.

    Call this AFTER run_trade_value_model() has written calibrated_value_1qb.
    Uses ON CONFLICT UPDATE so it overwrites any raw-model entry already written
    today by build_daily_model_values().
    """
    from datetime import date
    from dashboard_services.db import get_conn
    from utils.utils import load_players_index, load_model_value_table

    today = date.today().isoformat()

    # Build id -> name map: model table first (has picks + all players), then players_index fallback
    name_map: dict = {}
    try:
        mv = load_model_value_table(apply_calibration=False) or []
        for p in mv:
            pid = str(p.get("id") or "")
            nm = p.get("name") or ""
            if pid and nm and nm != "Unknown":
                name_map[pid] = nm
    except Exception:
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    # Fill gaps with players_index
    try:
        players_index = load_players_index() or {}
        for pid, info in players_index.items():
            if pid not in name_map:
                nm = (info or {}).get("name") or ""
                if nm:
                    name_map[str(pid)] = nm
    except Exception:
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    with get_conn() as conn:
        try:
            rows = conn.execute(
                """
                SELECT
                    player_id,
                    position,
                    team,
                    COALESCE(calibrated_value_1qb, value_1qb)                    AS value,
                    COALESCE(calibrated_value_sf,  value_sf, value_1qb)          AS value_sf,
                    COALESCE(calibrated_value_8,   GREATEST(value_8,  value_1qb)) AS value_8,
                    COALESCE(calibrated_value_12,  GREATEST(value_12, value_1qb)) AS value_12,
                    COALESCE(calibrated_value_14,  GREATEST(value_14, value_1qb)) AS value_14,
                    COALESCE(calibrated_sf_value_8,  sf_value_8,  value_sf) AS sf_value_8,
                    COALESCE(calibrated_sf_value_12, sf_value_12, value_sf) AS sf_value_12,
                    COALESCE(calibrated_sf_value_14, sf_value_14, value_sf) AS sf_value_14
                FROM player_values
                WHERE COALESCE(calibrated_value_1qb, value_1qb) > 0
                """
            ).fetchall()
        except Exception:
            # Fallback if calibrated_value_8 columns haven't been migrated yet
            rows = conn.execute(
                """
                SELECT
                    player_id,
                    position,
                    team,
                    COALESCE(calibrated_value_1qb, value_1qb)                    AS value,
                    COALESCE(calibrated_value_sf, value_sf, value_1qb)           AS value_sf,
                    GREATEST(value_8,  value_1qb)               AS value_8,
                    GREATEST(value_12, value_1qb)               AS value_12,
                    GREATEST(value_14, value_1qb)               AS value_14,
                    COALESCE(GREATEST(sf_value_8,  value_sf), value_1qb)  AS sf_value_8,
                    COALESCE(GREATEST(sf_value_12, value_sf), value_1qb)  AS sf_value_12,
                    COALESCE(GREATEST(sf_value_14, value_sf), value_1qb)  AS sf_value_14
                FROM player_values
                WHERE COALESCE(calibrated_value_1qb, value_1qb) > 0
                """
            ).fetchall()

    if not rows:
        print("[record_calibrated_history_snapshot] No rows in player_values - skipping")
        return 0

    written = 0
    
    # Write in batches with connection recovery to prevent timeouts
    BATCH = 500
    for batch_start in range(0, len(rows), BATCH):
        batch = rows[batch_start : batch_start + BATCH]
        batch_written = 0
        
        # Retry each batch up to 3 times with fresh connections
        for attempt in range(3):
            try:
                with get_conn(autocommit=True) as conn:
                    for r in batch:
                        player_name = name_map.get(str(r["player_id"])) or f"Player {r['player_id']}"
                        conn.execute(
                            """
                            INSERT INTO player_value_history
                                (as_of_date, player_id, name, position, team,
                                 value, value_sf, value_8, value_12, value_14,
                                 sf_value_8, sf_value_12, sf_value_14, source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'model')
                            ON CONFLICT (as_of_date, player_id, source)
                            DO UPDATE SET
                                name        = EXCLUDED.name,
                                value       = EXCLUDED.value,
                                value_sf    = EXCLUDED.value_sf,
                                value_8     = EXCLUDED.value_8,
                                value_12    = EXCLUDED.value_12,
                                value_14    = EXCLUDED.value_14,
                                sf_value_8  = EXCLUDED.sf_value_8,
                                sf_value_12 = EXCLUDED.sf_value_12,
                                sf_value_14 = EXCLUDED.sf_value_14,
                                position    = EXCLUDED.position,
                                team        = EXCLUDED.team
                            """,
                            (
                                today, r["player_id"], player_name, r["position"], r["team"],
                                float(r["value"] or 0),
                                float(r["value_sf"] or 0),
                                float(r["value_8"] or 0),
                                float(r["value_12"] or 0),
                                float(r["value_14"] or 0),
                                float(r["sf_value_8"] or 0),
                                float(r["sf_value_12"] or 0),
                                float(r["sf_value_14"] or 0),
                            ),
                        )
                        batch_written += 1
                    
                    written += batch_written
                    print(f"[record_calibrated_history_snapshot] Written batch {batch_start}-{batch_start + len(batch) - 1} ({batch_written} rows) - Total: {written} / {len(rows)}")
                    break  # Success, exit retry loop
                    
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    print(f"[record_calibrated_history_snapshot] Failed to write batch {batch_start}-{batch_start + len(batch) - 1} after 3 attempts, skipping. Error: {e}")
                    # Continue with next batch instead of failing completely
                    break
                else:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) + 1
                    print(f"[record_calibrated_history_snapshot] Batch {batch_start}-{batch_start + len(batch) - 1} failed (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

    print(f"[record_calibrated_history_snapshot] Wrote {written} calibrated values to player_value_history")
    return written


def sync_history_to_player_values() -> int:
    """
    Copy the most recent player_value_history snapshot (source='model') into
    player_values.value_1qb / value_sf, and clear calibrated columns so the
    EMA-smoothed values are immediately visible on the site.

    Picks are left untouched — they don't appear in player_value_history.

    Call this AFTER record_model_value_snapshot() and
    update_player_values_with_rankings() so step 4 metadata (pos_rank, age,
    team, etc.) is preserved but value_1qb is replaced with the EMA value.

    This ensures the site serves smoothed values rather than raw model output.
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT MAX(as_of_date) AS latest_date
            FROM player_value_history
            WHERE source = 'model' AND value > 0
            """
        ).fetchone()

    if not row or not row["latest_date"]:
        print("[sync_history_to_player_values] No history rows found — skipping")
        return 0

    latest_date = str(row["latest_date"])
    print(f"[sync_history_to_player_values] Syncing EMA values from as_of_date={latest_date}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE player_values pv
                SET value_1qb            = ph.value,
                    value_sf             = COALESCE(ph.sf_value, ph.value_sf, ph.value),
                    calibrated_value_1qb = NULL,
                    calibrated_value_sf  = NULL,
                    last_updated         = NOW()
                FROM player_value_history ph
                WHERE ph.player_id  = pv.player_id
                  AND ph.source     = 'model'
                  AND ph.as_of_date = %s
                  AND ph.value      > 0
                """,
                (latest_date,),
            )
            updated = cur.rowcount

    print(f"[sync_history_to_player_values] Updated {updated} player rows in player_values")
    return updated


def build_daily_market_pulse_for_league_type(league_type: str = "1qb"):
    """
    Build market pulse for a specific league type.

    Args:
        league_type: "1qb" or "sf" (superflex)

    Returns:
        HTML string with market pulse for the specified league type
    """
    value_table = load_model_value_table() or []

    if league_type == "sf":
        # Build top assets for Superflex
        top_assets = sorted(
            [
                {
                    "name": p.get("name"),
                    "position": p.get("position"),
                    "team": p.get("team"),
                    "value": _safe_float(p.get("sf_value")),
                }
                for p in value_table
                if isinstance(p, dict) and str(p.get("position") or "").upper() in {"QB", "RB", "WR", "TE"}
            ],
            key=lambda x: x["value"],
            reverse=True,
        )[:15]

        html = "<div class='ai-copy'><p><strong>Daily market pulse (Superflex):</strong> Elite QBs dominate the top of the market. Build around a QB1 or acquire multiple QB2s to remain competitive.</p></div>"
        payload = {"top_assets": top_assets, "league_type": "superflex"}
        cache_key = build_ai_cache_key("daily_market_pulse_sf", payload, "v1")
    else:
        # Build top assets for 1QB
        top_assets = sorted(
            [
                {
                    "name": p.get("name"),
                    "position": p.get("position"),
                    "team": p.get("team"),
                    "value": _safe_float(p.get("value")),
                }
                for p in value_table
                if isinstance(p, dict) and str(p.get("position") or "").upper() in {"QB", "RB", "WR", "TE"}
            ],
            key=lambda x: x["value"],
            reverse=True,
        )[:15]

        html = "<div class='ai-copy'><p><strong>Daily market pulse (1QB):</strong> Elite value remains concentrated at the top of the board. Monitor shifting tiers around your weakest position group before forcing trades.</p></div>"
        payload = {"top_assets": top_assets, "league_type": "1qb"}
        cache_key = build_ai_cache_key("daily_market_pulse_1qb", payload, "v1")

    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    save_cached_ai_text(cache_key, html)
    return html


if __name__ == "__main__":
    current = get_nfl_state() or {}
    current_season = int(current.get("season"))
    current_week = int(current.get("week"))
    build_daily_data(current_season, current_week)
