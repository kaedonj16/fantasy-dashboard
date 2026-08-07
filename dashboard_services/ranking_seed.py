"""
Daily seeder for team-ranking movement arrows.

Runs once a day (from the notification cron) over every subscribed league and:

  * records TODAY's real snapshot for the value, dashboard-value, power and
    playoff-odds rankings, so day-over-day movement accrues going forward;
  * on the first run for a league, reconstructs YESTERDAY's value rankings from
    the transaction log + the daily value history and backfills them as the
    baseline — so ▲/▼ arrows appear immediately instead of a day later, using
    only real data (no fabrication).

Power and playoff-odds rankings can't be truthfully reconstructed (they depend on
projections / a Monte Carlo sim whose past inputs aren't stored), so those are
record-forward only. All of this is best-effort: any failure is logged and
skipped, never raised.
"""
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _team_totals(rosters, values_map, pick_v_by_rid):
    """{roster_id: sum(player values) + pick value} for one value map."""
    out = {}
    for r in rosters or []:
        rid = r.get("roster_id")
        if rid is None:
            continue
        rid = str(rid)
        pv = sum(float(values_map.get(str(p), 0.0)) for p in (r.get("players") or []))
        out[rid] = pv + float(pick_v_by_rid.get(rid, 0.0))
    return out


def _order(totals):
    """roster ids ranked by total value, highest first."""
    return [rid for rid, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]


def seed_league_rankings(platform: str, league_id: str, season: int) -> None:
    """Snapshot (and, first time, reconstruct) every ranking for one league."""
    from app import (
        get_league_ctx_from_cache, _team_pick_value, _safe_int, apply_te_premium,
        te_premium_from_settings, load_pick_value_table, get_model_value_table_cached,
        _owner_to_rid_map,
    )
    from dashboard_services.ranking_movement import (
        record_ranks_for_date, has_snapshot,
    )

    ctx = get_league_ctx_from_cache(platform, league_id, season)
    if not ctx:
        return
    rosters = ctx.get("rosters") or []
    if not rosters:
        return
    roster_map = ctx.get("roster_map") or {}
    picks_by_roster = ctx.get("picks_by_roster") or {}
    league_id_str = str(ctx.get("resolved_league_id") or ctx.get("league_id") or league_id)
    _season = _safe_int(season, 0)

    today = datetime.now(timezone.utc).date()
    today_s = today.isoformat()
    yday_s = (today - timedelta(days=1)).isoformat()

    # ── value maps (raw = standings table; te = dashboard snapshot) ─────────────
    model_rows = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])
    values_raw, pos_by_id = {}, {}
    for row in model_rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        pid = str(row["id"])
        try:
            values_raw[pid] = float(row.get("value") or 0.0)
        except (TypeError, ValueError):
            values_raw[pid] = 0.0
        pos_by_id[pid] = str(row.get("position") or row.get("pos") or "").upper()

    tep = te_premium_from_settings(ctx.get("scoring_settings"))

    def _te_map(base_values):
        return {pid: apply_te_premium(v, pos_by_id.get(pid, ""), tep)
                for pid, v in base_values.items()}

    pick_by_key = load_pick_value_table() or {}
    pick_v_by_rid = {}
    for r in rosters:
        rid = r.get("roster_id")
        if rid is None:
            continue
        pick_v_by_rid[str(rid)] = _team_pick_value(
            picks_by_roster.get(str(rid), []) if isinstance(picks_by_roster, dict) else [],
            pick_by_key, platform=platform, league_id=league_id_str, season=_season)

    values_te = _te_map(values_raw)

    # ── today's snapshots (record-forward) ──────────────────────────────────────
    record_ranks_for_date(league_id_str, _season, "value",
                          _order(_team_totals(rosters, values_raw, pick_v_by_rid)),
                          today_s, overwrite=True)
    record_ranks_for_date(league_id_str, _season, "dash_value",
                          _order(_team_totals(rosters, values_te, pick_v_by_rid)),
                          today_s, overwrite=True)

    _seed_power(ctx, roster_map, league_id_str, _season, today_s, _owner_to_rid_map)
    _seed_odds(ctx, platform, league_id_str, _season)

    # ── first-run backfill: reconstruct yesterday's value rankings ──────────────
    if not has_snapshot(league_id_str, _season, "value", yday_s):
        _reconstruct_value_baseline(
            platform, league_id_str, _season, rosters, pos_by_id, pick_v_by_rid,
            tep, yday_s, today)


def _seed_power(ctx, roster_map, league_id_str, season, today_s, owner_to_rid_fn):
    try:
        from dashboard_services.ai.context_builders import build_power_rankings_context
        from dashboard_services.ranking_movement import record_ranks_for_date
        teams = (build_power_rankings_context(ctx) or {}).get("teams") or []
        if not teams:
            return
        o2r = owner_to_rid_fn(roster_map=roster_map)
        ordered = [o2r.get(str(t.get("team_name")))
                   for t in sorted(teams, key=lambda t: float(t.get("power_score") or 0.0),
                                   reverse=True)]
        record_ranks_for_date(league_id_str, season, "power",
                              [r for r in ordered if r is not None], today_s, overwrite=True)
    except Exception:
        logger.debug("[ranking-seed] power failed for %s", league_id_str, exc_info=True)


def _seed_odds(ctx, platform, league_id_str, season):
    try:
        from data_building.simulate_playoff_odds import simulate_playoff_odds
        from dashboard_services.playoff_odds_history import record_daily_and_movement
        odds = simulate_playoff_odds(ctx, platform=platform) or []
        if odds:
            record_daily_and_movement(league_id_str, season, odds, write=True)
    except Exception:
        logger.debug("[ranking-seed] odds failed for %s", league_id_str, exc_info=True)


def _reconstruct_value_baseline(platform, league_id_str, season, rosters, pos_by_id,
                                pick_v_by_rid, tep, yday_s, today):
    """Backfill yesterday's value + dash_value rankings from real history."""
    try:
        from app import apply_te_premium
        from dashboard_services.service import get_transactions_by_week
        from dashboard_services.roster_history import reconstruct_rosters_as_of
        from dashboard_services.player_value_history import get_values_as_of
        from dashboard_services.ranking_movement import record_ranks_for_date

        # All transactions since the start of today get reversed to recover the
        # rosters as they stood at end of yesterday.
        start_today_ms = int(datetime(today.year, today.month, today.day,
                                      tzinfo=timezone.utc).timestamp() * 1000)
        tx_by_week = get_transactions_by_week(
            league_id_str, list(range(0, 19)), platform=platform, season=int(season)) or {}
        all_tx = [t for wk in tx_by_week.values() for t in (wk or [])]
        yroster = reconstruct_rosters_as_of(rosters, all_tx, start_today_ms)

        yvals_raw = get_values_as_of(yday_s)
        if not yvals_raw:
            return  # no value history for yesterday — can't reconstruct honestly
        yvals_te = {pid: apply_te_premium(v, pos_by_id.get(pid, ""), tep)
                    for pid, v in yvals_raw.items()}

        def _tot(vals):
            return {rid: sum(float(vals.get(p, 0.0)) for p in players)
                         + float(pick_v_by_rid.get(rid, 0.0))
                    for rid, players in yroster.items()}

        def _ord(tot):
            return [rid for rid, _ in sorted(tot.items(), key=lambda kv: kv[1], reverse=True)]

        record_ranks_for_date(league_id_str, season, "value", _ord(_tot(yvals_raw)),
                              yday_s, overwrite=False)
        record_ranks_for_date(league_id_str, season, "dash_value", _ord(_tot(yvals_te)),
                              yday_s, overwrite=False)
        logger.info("[ranking-seed] reconstructed yesterday value baseline for %s",
                    league_id_str)
    except Exception:
        logger.debug("[ranking-seed] value reconstruction failed for %s",
                     league_id_str, exc_info=True)


def snapshot_all_rankings() -> None:
    """Daily entry point: seed rankings for every subscribed league (current
    season only — past seasons don't change)."""
    try:
        from utils.push_notifications import _get_subscribed_leagues
        from dashboard_services.api import get_nfl_state
        leagues = _get_subscribed_leagues()
        if not leagues:
            return
        season = int((get_nfl_state() or {}).get("season") or datetime.now(timezone.utc).year)
        for league_id, platform in leagues:
            try:
                seed_league_rankings(platform, league_id, season)
            except Exception:
                logger.debug("[ranking-seed] league %s failed", league_id, exc_info=True)
    except Exception:
        logger.warning("[ranking-seed] snapshot_all_rankings failed", exc_info=True)
