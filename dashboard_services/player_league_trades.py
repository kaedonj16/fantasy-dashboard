"""
Player-scoped league trade history with draft-pick resolution.

Powers the player-modal Trades tab "This League" view: every season in the
league chain, real counterparties, and (once a draft is complete) who each
traded pick eventually became.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _player_info(pid: str, players_index: dict, *, is_focus: bool = False) -> dict:
    meta = players_index.get(str(pid)) or {}
    return {
        "type": "player",
        "player_id": str(pid),
        "name": meta.get("name") or str(pid),
        "position": meta.get("pos") or meta.get("position") or "",
        "is_focus": bool(is_focus),
    }


def _pick_label(season: Any, round_: Any, slot: Optional[int], order: Optional[str] = None) -> str:
    s = str(season) if season not in (None, "") else "?"
    r = str(round_) if round_ not in (None, "") else "?"
    if slot:
        return f"{s} Pick {r}.{str(int(slot)).zfill(2)}"
    if order:
        return f"{s} Round {r} ({order})"
    return f"{s} Round {r}"


def build_draft_resolution_map(
    platform: str,
    league_id: str,
    seasons: list[int] | set[int] | None = None,
) -> dict[tuple[int, int, int], dict]:
    """
    Map (pick_season, pick_round, pick_slot) -> drafted player info.

    Sleeper-only today (completed drafts). Empty for other platforms.
    """
    plat = (platform or "sleeper").strip().lower()
    if plat != "sleeper" or not league_id:
        return {}

    try:
        from dashboard_services.api import get_draft_picks, get_drafts
        from utils.utils import load_players_index
    except Exception:
        logger.debug("[player-league-trades] draft imports failed", exc_info=True)
        return {}

    players_index = load_players_index() or {}
    season_filter = {int(s) for s in seasons} if seasons else None
    out: dict[tuple[int, int, int], dict] = {}

    try:
        drafts = get_drafts(str(league_id)) or []
    except Exception:
        logger.debug("[player-league-trades] drafts list failed for %s", league_id, exc_info=True)
        return {}

    for d in drafts:
        if str(d.get("status") or "") != "complete":
            continue
        try:
            d_season = int(d.get("season"))
        except (TypeError, ValueError):
            continue
        if season_filter is not None and d_season not in season_filter:
            continue
        draft_id = d.get("draft_id")
        if not draft_id:
            continue
        try:
            picks = get_draft_picks(str(draft_id)) or []
        except Exception:
            logger.debug("[player-league-trades] picks failed for %s", draft_id, exc_info=True)
            continue
        for p in picks:
            pid = str(p.get("player_id") or "").strip()
            if not pid:
                continue
            try:
                rnd = int(p.get("round"))
                slot = int(p.get("draft_slot") or 0)
            except (TypeError, ValueError):
                continue
            if rnd <= 0 or slot <= 0:
                continue
            key = (d_season, rnd, slot)
            if key in out:
                continue
            out[key] = _player_info(pid, players_index)

    return out


def resolve_pick_asset(
    *,
    pick_season: Any,
    pick_round: Any,
    pick_slot: Any,
    pick_order: Any = None,
    resolution_map: dict[tuple[int, int, int], dict] | None = None,
) -> dict:
    """Build a pick asset dict, attaching drafted_player when the draft is done."""
    slot_i: Optional[int] = None
    try:
        if pick_slot not in (None, ""):
            slot_i = int(pick_slot)
    except (TypeError, ValueError):
        slot_i = None

    name = _pick_label(pick_season, pick_round, slot_i, pick_order if isinstance(pick_order, str) else None)
    asset: dict[str, Any] = {
        "type": "pick",
        "name": name,
        "is_focus": False,
        "pick_season": int(pick_season) if str(pick_season).isdigit() else pick_season,
        "pick_round": int(pick_round) if str(pick_round).isdigit() else pick_round,
        "pick_slot": slot_i,
    }

    drafted = None
    if resolution_map and slot_i:
        try:
            key = (int(pick_season), int(pick_round), int(slot_i))
            drafted = resolution_map.get(key)
        except (TypeError, ValueError):
            drafted = None
    if drafted:
        asset["drafted_player"] = {
            "player_id": drafted.get("player_id"),
            "name": drafted.get("name"),
            "position": drafted.get("position") or "",
        }
        # Surface the outcome in the primary label once known.
        asset["name"] = f"{name} → {drafted.get('name')}"

    return asset


def _slot_map_for_league(platform: str, league_id: str) -> dict[tuple[str, str], int]:
    """{(season_str, roster_id_str): draft_slot} for labeling traded picks."""
    plat = (platform or "sleeper").strip().lower()
    if plat != "sleeper":
        return {}
    try:
        from data_building.trade_intel.trade_crawler import _fetch_draft_slot_map
        return _fetch_draft_slot_map(str(league_id)) or {}
    except Exception:
        logger.debug("[player-league-trades] slot map failed for %s", league_id, exc_info=True)
        return {}


def _roster_names(platform: str, league_id: str, season: int) -> dict[str, str]:
    try:
        from dashboard_services.platform_api import get_rosters, get_users
        from dashboard_services.players import build_roster_map
        users = get_users(platform, league_id, season) or []
        rosters = get_rosters(platform, league_id, season) or []
        return {str(k): str(v) for k, v in (build_roster_map(users, rosters) or {}).items()}
    except Exception:
        logger.debug("[player-league-trades] roster map failed", exc_info=True)
        return {}


def _trade_timestamp(txn: dict) -> Optional[datetime]:
    ts_raw = txn.get("status_updated") or txn.get("created")
    if not ts_raw:
        return None
    try:
        return datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _format_trade_sides(
    txn: dict,
    *,
    focus_pid: str,
    roster_names: dict[str, str],
    players_index: dict,
    slot_map: dict[tuple[str, str], int],
    resolution_map: dict[tuple[int, int, int], dict],
) -> tuple[dict, dict] | None:
    """
    Split a Sleeper-style trade into two sides centered on the focus player.

    side_a = team that received the focus player
    side_b = the counterparty that sent the focus player (their received package)
    """
    adds = txn.get("adds") or {}
    drops = txn.get("drops") or {}
    draft_picks = txn.get("draft_picks") or []

    focus = str(focus_pid)
    if focus not in {str(k) for k in adds.keys()} and focus not in {str(k) for k in drops.keys()}:
        # Player must appear in the trade (add or drop). Adds cover both sides
        # on Sleeper; require an add for the focus player.
        if focus not in {str(k) for k in adds.keys()}:
            return None

    # Receiver of the focus player
    recv_rid = str(adds.get(focus) or "")
    if not recv_rid:
        return None

    # Sender: drop map values are the roster losing the player
    send_rid = str(drops.get(focus) or "")
    if not send_rid:
        # Infer other roster from roster_ids / adds
        rids = {str(r) for r in (txn.get("roster_ids") or [])}
        rids |= {str(v) for v in adds.values()}
        rids |= {str(v) for v in drops.values()}
        rids.discard(recv_rid)
        send_rid = next(iter(sorted(rids)), "")
    if not send_rid or send_rid == recv_rid:
        return None

    def assets_for(rid: str) -> list[dict]:
        out: list[dict] = []
        for pid, to_rid in adds.items():
            if str(to_rid) != str(rid):
                continue
            out.append(_player_info(str(pid), players_index, is_focus=(str(pid) == focus)))
        for pick in draft_picks:
            if str(pick.get("owner_id") or "") != str(rid):
                continue
            p_season = pick.get("season")
            p_round = pick.get("round")
            roster_id = pick.get("roster_id")
            slot = None
            if slot_map and roster_id is not None and p_season is not None:
                slot = slot_map.get((str(p_season), str(roster_id)))
            out.append(
                resolve_pick_asset(
                    pick_season=p_season,
                    pick_round=p_round,
                    pick_slot=slot,
                    resolution_map=resolution_map,
                )
            )
        return out

    side_a_assets = assets_for(recv_rid)
    side_b_assets = assets_for(send_rid)
    if not side_a_assets and not side_b_assets:
        return None

    return (
        {
            "team_name": roster_names.get(recv_rid) or f"Team {recv_rid}",
            "roster_id": recv_rid,
            "assets": side_a_assets,
        },
        {
            "team_name": roster_names.get(send_rid) or f"Team {send_rid}",
            "roster_id": send_rid,
            "assets": side_b_assets,
        },
    )


def get_player_league_trades(
    *,
    player_id: str,
    platform: str,
    league_id: str,
    season: int,
    limit: int = 20,
) -> dict[str, Any]:
    """
    All trades involving ``player_id`` across every season in this league's
    history (via previous_league_id / ESPN year probe), newest first.
    """
    from dashboard_services.api import build_league_history_map
    from dashboard_services.service import get_transactions_by_week
    from utils.utils import load_players_index

    pid = str(player_id or "").strip()
    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "").strip()
    if not pid or not lid:
        return {"trades": [], "total": 0, "source": "league"}

    limit = max(1, min(int(limit or 20), 50))
    season_map = build_league_history_map(plat, lid, int(season)) or {int(season): lid}
    players_index = load_players_index() or {}

    # Prefetch draft resolution for every league_id in the chain (picks resolve
    # against the season the pick belongs to, which may be a later league year).
    resolution_by_league: dict[str, dict[tuple[int, int, int], dict]] = {}
    slot_by_league: dict[str, dict[tuple[str, str], int]] = {}
    for hist_lid in {str(v) for v in season_map.values()}:
        resolution_by_league[hist_lid] = build_draft_resolution_map(plat, hist_lid)
        slot_by_league[hist_lid] = _slot_map_for_league(plat, hist_lid)

    # Also resolve against the current league id — dynasty drafts for future
    # pick seasons often live on the newest league record.
    if lid not in resolution_by_league:
        resolution_by_league[lid] = build_draft_resolution_map(plat, lid)
    # Merge all resolution maps (later seasons / newer leagues win on conflict)
    merged_resolution: dict[tuple[int, int, int], dict] = {}
    for m in resolution_by_league.values():
        merged_resolution.update(m)

    collected: list[dict] = []
    for hist_season in sorted(season_map.keys(), reverse=True):
        hist_lid = str(season_map[hist_season])
        roster_names = _roster_names(plat, hist_lid, int(hist_season))
        slot_map = slot_by_league.get(hist_lid) or {}
        try:
            tx_by_week = get_transactions_by_week(
                hist_lid, range(1, 19), platform=plat, season=int(hist_season)
            ) or {}
        except Exception:
            logger.debug(
                "[player-league-trades] tx fetch failed %s %s",
                hist_lid, hist_season, exc_info=True,
            )
            continue

        for week in sorted(tx_by_week.keys(), reverse=True):
            for txn in (tx_by_week[week] or []):
                if (txn.get("type") or "") != "trade":
                    continue
                status = (txn.get("status") or "complete").lower()
                if status in ("failed", "cancelled", "canceled", "rejected"):
                    continue
                adds = txn.get("adds") or {}
                if str(pid) not in {str(k) for k in adds.keys()}:
                    continue
                sides = _format_trade_sides(
                    txn,
                    focus_pid=pid,
                    roster_names=roster_names,
                    players_index=players_index,
                    slot_map=slot_map,
                    resolution_map=merged_resolution,
                )
                if not sides:
                    continue
                side_a, side_b = sides
                ts = _trade_timestamp(txn)
                date_str = ""
                if ts:
                    try:
                        date_str = ts.strftime("%-m/%-d/%y")
                    except Exception:
                        date_str = ts.strftime("%m/%d/%y")
                collected.append({
                    "date": date_str,
                    "season": int(hist_season),
                    "week": int(week) if week is not None else None,
                    "ts": ts.timestamp() if ts else 0,
                    "side_a": side_a,
                    "side_b": side_b,
                    "is_superflex": None,
                    "source": "league",
                })

    collected.sort(key=lambda t: (t.get("ts") or 0), reverse=True)
    total = len(collected)
    trimmed = collected[:limit]
    for t in trimmed:
        t.pop("ts", None)
    return {"trades": trimmed, "total": total, "source": "league"}


def attach_drafted_players_to_trade_db_assets(
    trades: list[dict],
    *,
    platform: str = "sleeper",
) -> list[dict]:
    """
    For Trade DB cards: when a pick has pick_slot + the trade's league_id is
    known, resolve who was drafted with that pick after the draft completed.
    """
    if not trades:
        return trades

    # Group league_ids that need resolution
    league_ids = sorted({str(t.get("league_id") or "") for t in trades if t.get("league_id")})
    league_ids = [x for x in league_ids if x]
    if not league_ids:
        return trades

    # Collect pick seasons we care about to avoid scanning unrelated drafts
    pick_seasons: set[int] = set()
    for t in trades:
        for side in (t.get("side_a") or []) + (t.get("side_b") or []):
            if side.get("type") == "pick" and side.get("pick_season") is not None:
                try:
                    pick_seasons.add(int(side["pick_season"]))
                except (TypeError, ValueError):
                    pass

    res_cache: dict[str, dict[tuple[int, int, int], dict]] = {}
    for lid in league_ids[:40]:  # hard cap: one page of trades
        res_cache[lid] = build_draft_resolution_map(platform, lid, pick_seasons or None)

    for t in trades:
        lid = str(t.get("league_id") or "")
        res_map = res_cache.get(lid) or {}
        if not res_map:
            continue
        for key in ("side_a", "side_b"):
            assets = t.get(key) or []
            new_assets = []
            for a in assets:
                if a.get("type") != "pick":
                    new_assets.append(a)
                    continue
                new_assets.append(
                    resolve_pick_asset(
                        pick_season=a.get("pick_season"),
                        pick_round=a.get("pick_round"),
                        pick_slot=a.get("pick_slot"),
                        pick_order=a.get("pick_order"),
                        resolution_map=res_map,
                    )
                )
            t[key] = new_assets
    return trades
