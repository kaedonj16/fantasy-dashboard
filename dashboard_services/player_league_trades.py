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

# Sleeper (and ESPN scoringPeriodId) store the entire offseason as week 0.
# Dynasty trades cluster there, so "This League" has to include it or it
# looks like current-season in-season activity only.
_TX_WEEKS = range(0, 19)


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
        return f"{s} {r}.{str(int(slot)).zfill(2)}"
    if order:
        return f"{s} {r}st ({order})" if r == "1" else f"{s} Round {r} ({order})"
    return f"{s} {r}st" if r == "1" else f"{s} Round {r}"


def build_draft_resolution_map(
    platform: str,
    league_id: str,
    seasons: list[int] | set[int] | None = None,
) -> dict[tuple[int, int, str], dict]:
    """
    Map ``(season, round, original_roster_id)`` to an authoritative selection.

    The original owner is the stable identity Sleeper puts on every transaction
    pick as ``roster_id``.  It survives repeated trades.  ``pick_no`` supplies
    the displayed within-round position, so snake/3RR/custom orders are not
    inferred from a roster's starting slot.  Multiple matching drafts are left
    unresolved rather than guessed.
    """
    plat = (platform or "sleeper").strip().lower()
    if plat != "sleeper" or not league_id:
        return {}

    try:
        from dashboard_services.api import get_draft, get_draft_picks, get_drafts
        from utils.utils import load_players_index
    except Exception:
        logger.debug("[player-league-trades] draft imports failed", exc_info=True)
        return {}

    players_index = load_players_index() or {}
    season_filter = {int(s) for s in seasons} if seasons else None
    candidates: dict[tuple[int, int, str], list[dict]] = {}

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
            detail = get_draft(str(draft_id)) or {}
            picks = get_draft_picks(str(draft_id)) or []
        except Exception:
            logger.debug("[player-league-trades] picks failed for %s", draft_id, exc_info=True)
            continue
        slot_to_roster = detail.get("slot_to_roster_id") or d.get("slot_to_roster_id") or {}
        team_count = int(detail.get("settings", {}).get("teams") or d.get("settings", {}).get("teams") or 0)
        if not slot_to_roster or team_count <= 0:
            continue
        for p in picks:
            pid = str(p.get("player_id") or "").strip()
            if not pid:
                continue
            try:
                rnd = int(p.get("round"))
                draft_slot = int(p.get("draft_slot") or 0)
                pick_no = int(p.get("pick_no") or 0)
            except (TypeError, ValueError):
                continue
            if rnd <= 0 or draft_slot <= 0 or pick_no <= 0:
                continue
            original_roster = slot_to_roster.get(str(draft_slot), slot_to_roster.get(draft_slot))
            if original_roster in (None, ""):
                continue
            within_round = ((pick_no - 1) % team_count) + 1
            key = (d_season, rnd, str(original_roster))
            resolved = _player_info(pid, players_index)
            resolved.update({"draft_id": str(draft_id), "pick_no": pick_no,
                             "within_round": within_round, "original_roster_id": str(original_roster)})
            candidates.setdefault(key, []).append(resolved)

    return {key: rows[0] for key, rows in candidates.items() if len(rows) == 1}


def resolve_pick_asset(
    *,
    pick_season: Any,
    pick_round: Any,
    pick_slot: Any,
    pick_roster_id: Any = None,
    pick_order: Any = None,
    resolution_map: dict[tuple[int, int, str], dict] | None = None,
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
    if resolution_map and pick_roster_id not in (None, ""):
        try:
            key = (int(pick_season), int(pick_round), str(pick_roster_id))
            drafted = resolution_map.get(key)
        except (TypeError, ValueError):
            drafted = None
    if drafted:
        actual_slot = int(drafted.get("within_round") or slot_i or 0) or None
        name = _pick_label(pick_season, pick_round, actual_slot, None)
        asset["pick_slot"] = actual_slot
        asset["pick_roster_id"] = str(pick_roster_id)
        asset["resolved_draft_id"] = drafted.get("draft_id")
        asset["drafted_player"] = {
            "player_id": drafted.get("player_id"),
            "name": drafted.get("name"),
            "position": drafted.get("position") or "",
        }
        # Surface the outcome in the primary label once known.
        asset["name"] = f"{name} ({drafted.get('name')})"

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
        from dashboard_services.players import build_roster_map
        names = build_roster_map(str(league_id), platform, season) or {}
        return {str(k): str(v) for k, v in names.items() if v}
    except Exception:
        logger.debug("[player-league-trades] roster map failed", exc_info=True)
        return {}


def _cached_resolution_map(league_ids: set[str], players_index: dict) -> dict[tuple[int, int, str], dict]:
    """Read verified resolutions without contacting a provider.

    Deploys remain compatible while migration 037 is rolling out: an absent
    table column simply means there is no cache yet.
    """
    if not league_ids:
        return {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT a.pick_season, a.pick_round, a.pick_roster_id, "
                "a.resolved_draft_id, a.resolved_player_id, a.resolved_pick_no, "
                "a.resolved_round_slot FROM trade_intel_assets a "
                "JOIN trade_intel_trades t ON t.id=a.trade_id "
                "WHERE t.league_id = ANY(%s) AND a.resolved_player_id IS NOT NULL",
                (list(league_ids),),
            ).fetchall()
    except Exception:
        logger.debug("[player-league-trades] verified resolution cache unavailable", exc_info=True)
        return {}
    out = {}
    conflicts = set()
    for row in rows:
        key = (int(row["pick_season"]), int(row["pick_round"]), str(row["pick_roster_id"]))
        value = _player_info(str(row["resolved_player_id"]), players_index)
        value.update({"draft_id": row["resolved_draft_id"], "pick_no": row["resolved_pick_no"],
                      "within_round": row["resolved_round_slot"],
                      "original_roster_id": str(row["pick_roster_id"])})
        if key in out and (out[key]["draft_id"], out[key]["player_id"]) != (value["draft_id"], value["player_id"]):
            conflicts.add(key)
        else:
            out[key] = value
    return {key: value for key, value in out.items() if key not in conflicts}


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
    resolution_map: dict[tuple[int, int, str], dict],
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
    matching_pick = None
    for pick in draft_picks:
        try:
            key = (int(pick.get("season")), int(pick.get("round")), str(pick.get("roster_id")))
        except (TypeError, ValueError):
            continue
        if str((resolution_map.get(key) or {}).get("player_id") or "") == focus:
            matching_pick = pick
            break

    # Direction always comes from transaction participants.  A resolved player
    # did not exist as a player asset at trade time; its pick's owner/previous
    # owner determine received/sent instead.
    recv_rid = str((matching_pick or {}).get("owner_id") or adds.get(focus) or "")
    if not recv_rid:
        return None

    # Sender: drop map values are the roster losing the player
    send_rid = str((matching_pick or {}).get("previous_owner_id") or drops.get(focus) or "")
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
            asset = resolve_pick_asset(
                    pick_season=p_season,
                    pick_round=p_round,
                    pick_slot=slot,
                    pick_roster_id=roster_id,
                    resolution_map=resolution_map,
                )
            if matching_pick is pick:
                asset["is_focus"] = True
                asset["via_draft_pick"] = True
            out.append(asset)
        return out

    side_a_assets = assets_for(recv_rid)
    side_b_assets = assets_for(send_rid)
    if not side_a_assets and not side_b_assets:
        return None

    return (
        {
            "team_name": roster_names.get(str(recv_rid)) or roster_names.get(recv_rid) or f"Team {recv_rid}",
            "roster_id": recv_rid,
            "assets": side_a_assets,
            "direction": "Received",
        },
        {
            "team_name": roster_names.get(str(send_rid)) or roster_names.get(send_rid) or f"Team {send_rid}",
            "roster_id": send_rid,
            "assets": side_b_assets,
            "direction": "Sent",
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

    limit = max(1, min(int(limit or 50), 50))
    season_map = build_league_history_map(plat, lid, int(season)) or {int(season): lid}
    players_index = load_players_index() or {}

    # Prefetch draft resolution for every league_id in the chain (picks resolve
    # against the season the pick belongs to, which may be a later league year).
    resolution_by_league: dict[str, dict[tuple[int, int, str], dict]] = {}
    slot_by_league: dict[str, dict[tuple[str, str], int]] = {}
    for hist_lid in {str(v) for v in season_map.values()}:
        resolution_by_league[hist_lid] = build_draft_resolution_map(plat, hist_lid)
        slot_by_league[hist_lid] = _slot_map_for_league(plat, hist_lid)

    # Also resolve against the current league id -- dynasty drafts for future
    # pick seasons often live on the newest league record.
    if lid not in resolution_by_league:
        resolution_by_league[lid] = build_draft_resolution_map(plat, lid)
    # Merge all resolution maps (later seasons / newer leagues win on conflict)
    lineage_ids = {str(v) for v in season_map.values()} | {lid}
    merged_resolution: dict[tuple[int, int, str], dict] = _cached_resolution_map(lineage_ids, players_index)
    ambiguous: set[tuple[int, int, str]] = set()
    for m in resolution_by_league.values():
        for key, value in m.items():
            prior = merged_resolution.get(key)
            if prior and (prior.get("draft_id"), prior.get("player_id")) != (value.get("draft_id"), value.get("player_id")):
                ambiguous.add(key)
            else:
                merged_resolution[key] = value
    for key in ambiguous:
        merged_resolution.pop(key, None)

    collected: list[dict] = []
    seen_txn_ids: set[str] = set()
    for hist_season in sorted(season_map.keys(), reverse=True):
        hist_lid = str(season_map[hist_season])
        roster_names = _roster_names(plat, hist_lid, int(hist_season))
        slot_map = slot_by_league.get(hist_lid) or {}
        try:
            tx_by_week = get_transactions_by_week(
                hist_lid, _TX_WEEKS, platform=plat, season=int(hist_season)
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
                txn_id = str(txn.get("transaction_id") or "").strip()
                if txn_id:
                    if txn_id in seen_txn_ids:
                        continue
                    seen_txn_ids.add(txn_id)
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
                        date_str = f"{ts.month}/{ts.day}/{ts.strftime('%y')}"
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
                    "via_draft_pick": any(
                        asset.get("via_draft_pick")
                        for side in (side_a, side_b)
                        for asset in side.get("assets", [])
                    ),
                })

    collected.sort(key=lambda t: (t.get("ts") or 0), reverse=True)
    total = len(collected)
    trimmed = collected[:limit]
    for t in trimmed:
        t.pop("ts", None)
    return {"trades": trimmed, "total": total, "source": "league"}


def get_player_acquisition_events(
    player_id: str,
    *,
    platform: str = "sleeper",
    league_id: str = "",
    season: int = 0,
    limit: int = 20,
) -> dict:
    """Non-trade acquisition events for a player across the league history chain:
    the draft pick that selected them and any waiver / free-agent adds (with FAAB
    when present). Best-effort and defensive: returns whatever it can derive.

    Shape: {"events": [{"kind": "draft"|"add", "season", ...}]}, newest last so
    the caller can render a chronological timeline.
    """
    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "")
    pid = str(player_id or "")
    if not lid or not pid:
        return {"events": []}

    from dashboard_services.api import build_league_history_map
    from dashboard_services.service import get_transactions_by_week

    try:
        season_map = build_league_history_map(plat, lid, int(season)) or {int(season): lid}
    except Exception:
        season_map = {int(season): lid}

    events: list[dict] = []

    # ── Draft pick that selected this player (completed drafts only) ──────────
    try:
        from dashboard_services.api import get_drafts, get_draft_picks
        for hist_lid in {str(v) for v in season_map.values()}:
            drafts = get_drafts(hist_lid) or []
            for d in drafts:
                if str(d.get("status") or "") != "complete":
                    continue
                draft_id = d.get("draft_id")
                if not draft_id:
                    continue
                try:
                    d_season = int(d.get("season"))
                except (TypeError, ValueError):
                    d_season = None
                picks = get_draft_picks(str(draft_id)) or []
                names = _roster_names(plat, hist_lid, int(d_season)) if d_season else {}
                for p in picks:
                    if str(p.get("player_id") or "") != pid:
                        continue
                    rid = str(p.get("roster_id") or p.get("picked_by") or "")
                    try:
                        rnd = int(p.get("round") or 0)
                    except (TypeError, ValueError):
                        rnd = 0
                    try:
                        slot = int(p.get("draft_slot") or 0)
                    except (TypeError, ValueError):
                        slot = 0
                    events.append({
                        "kind": "draft",
                        "season": d_season,
                        "round": rnd or None,
                        "slot": slot or None,
                        "pick_no": p.get("pick_no"),
                        "team": names.get(rid) or (f"Team {rid}" if rid else None),
                    })
    except Exception:
        logger.debug("[player-acquisition] draft scan failed", exc_info=True)

    # ── Waiver / free-agent adds (with FAAB when the league uses it) ──────────
    try:
        for hist_season in sorted(season_map.keys(), reverse=True):
            hist_lid = str(season_map[hist_season])
            names = _roster_names(plat, hist_lid, int(hist_season))
            try:
                tx_by_week = get_transactions_by_week(
                    hist_lid, _TX_WEEKS, platform=plat, season=int(hist_season)
                ) or {}
            except Exception:
                continue
            for week in sorted(tx_by_week.keys()):
                for txn in (tx_by_week[week] or []):
                    if (txn.get("type") or "") not in ("waiver", "free_agent", "waiver_add"):
                        continue
                    if (txn.get("status") or "complete").lower() in ("failed", "cancelled", "canceled", "rejected"):
                        continue
                    adds = txn.get("adds") or {}
                    if pid not in {str(k) for k in adds.keys()}:
                        continue
                    rid = str(adds.get(pid) or adds.get(int(pid) if pid.isdigit() else pid) or "")
                    faab = None
                    try:
                        faab = (txn.get("settings") or {}).get("waiver_bid")
                    except Exception:
                        faab = None
                    events.append({
                        "kind": "add",
                        "season": int(hist_season),
                        "week": int(week) if week is not None else None,
                        "faab": faab,
                        "team": names.get(rid) or (f"Team {rid}" if rid else None),
                    })
    except Exception:
        logger.debug("[player-acquisition] add scan failed", exc_info=True)

    # Chronological: draft first, then adds by season/week.
    def _sort_key(e):
        return (int(e.get("season") or 0), 0 if e.get("kind") == "draft" else int(e.get("week") or 0))
    events.sort(key=_sort_key)
    return {"events": events[:limit]}


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

    res_cache: dict[str, dict[tuple[int, int, str], dict]] = {}
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
                        pick_roster_id=a.get("pick_roster_id"),
                        pick_order=a.get("pick_order"),
                        resolution_map=res_map,
                    )
                )
            t[key] = new_assets
    return trades
