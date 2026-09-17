"""Repair RZ columns on the latest stored snapshot for each player/season.

Run with ``python -m scripts.backfill_redzone_metrics``. Other metrics and older
point-in-time snapshots are untouched. Re-running is safe, including after a
partial failure. Weekly rows missing the new columns repair during daily builds.
"""
from data_building.advanced_metrics import init_advanced_metrics_db
from data_building.external_data.sleeper_usage import build_usage_map_for_season
from dashboard_services.db import get_conn
from data_building.weekly_metrics import build_weekly_metrics


def backfill_redzone_metrics():
    init_advanced_metrics_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT ON (season, player_id) id, season, player_id
            FROM player_advanced_metrics
            WHERE season IS NOT NULL AND position IN ('QB', 'RB', 'WR', 'TE')
            ORDER BY season, player_id, as_of_date DESC
        """).fetchall()
    by_season = {}
    for row in rows:
        by_season.setdefault(int(row['season']), []).append(row)
    updated = unavailable = 0
    for season, snapshots in sorted(by_season.items()):
        usage = build_usage_map_for_season(season, range(1, 19))
        if not usage:
            unavailable += len(snapshots)
            continue
        build_weekly_metrics(season)
        with get_conn() as conn:
            for row in snapshots:
                player = usage.get(str(row['player_id'])) or {}
                if not player.get('games'):
                    continue
                if not player.get('red_zone_available'):
                    unavailable += 1
                    continue
                targets = player.get('rec_rz_tgt_pg')
                carries = player.get('rush_rz_att_pg')
                if targets is None or carries is None:
                    unavailable += 1
                    continue
                conn.execute("""
                    UPDATE player_advanced_metrics
                    SET rz_targets_pg = %s, rz_carries_pg = %s, red_zone_usage = %s
                    WHERE id = %s
                """, (targets, carries, targets + carries, row['id']))
                updated += 1
        print(f'[redzone backfill] season={season} updated_total={updated} unavailable_total={unavailable}')
    return {'updated': updated, 'unavailable': unavailable}


if __name__ == '__main__':
    print(backfill_redzone_metrics())
