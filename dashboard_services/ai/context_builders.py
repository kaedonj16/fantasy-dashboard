from __future__ import annotations

import math
from typing import Any, Dict, List


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def build_model_value_lookup(model_value_table: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in model_value_table or []:
        pid = str(row.get("id") or row.get("player_id") or "")
        if pid:
            out[pid] = row
    return out


def summarize_roster_players(
        roster: dict,
        players_index: dict,
        players_map: dict,
        model_value_lookup: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    for pid in roster.get("players") or []:
        spid = str(pid)
        meta = players_index.get(spid) or players_map.get(spid) or {}
        mv = model_value_lookup.get(spid) or {}

        name = meta.get("full_name") or meta.get("name") or mv.get("name") or "Unknown"
        pos = meta.get("position") or meta.get("pos") or mv.get("position") or "?"
        team = meta.get("team") or mv.get("team") or "FA"
        age = meta.get("age") or mv.get("age")
        value = _safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"), 0.0)

        out.append({
            "id": spid,
            "name": name,
            "position": pos,
            "team": team,
            "age": age,
            "value": round(value, 1),
        })

    out.sort(key=lambda x: x["value"], reverse=True)
    return out


def group_position_strength(players: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {}
    for p in players:
        pos = p.get("position") or "?"
        buckets.setdefault(pos, []).append(_safe_float(p.get("value")))

    out: dict[str, dict] = {}
    for pos, vals in buckets.items():
        vals = sorted(vals, reverse=True)
        out[pos] = {
            "count": len(vals),
            "top_3_sum": round(sum(vals[:3]), 1),
            "top_5_sum": round(sum(vals[:5]), 1),
            "best": round(vals[0], 1) if vals else 0.0,
        }
    return out


def detect_team_direction(players: list[dict], future_picks: list[dict]) -> str:
    ages = [
        _safe_float(p.get("age"))
        for p in players
        if p.get("age") not in (None, "")
    ]
    avg_age = sum(ages) / len(ages) if ages else 0.0

    elite_assets = sum(1 for p in players if _safe_float(p.get("value")) >= 6500)
    strong_assets = sum(1 for p in players if _safe_float(p.get("value")) >= 4500)
    firsts = sum(1 for p in future_picks if "1." in str(p.get("display") or ""))

    if elite_assets >= 3 and avg_age and avg_age <= 27.5:
        return "contender"
    if firsts >= 2 and avg_age and avg_age >= 26.8 and elite_assets < 2:
        return "retool"
    if firsts >= 3 and strong_assets <= 3:
        return "rebuild"
    return "balanced"


def build_team_gm_context(ctx: dict, viewer_roster_id: str) -> Union[dict, None]:
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return None

    roster_map = ctx.get("roster_map") or {}
    team_name = roster_map.get(str(viewer_roster_id)) or f"Roster {viewer_roster_id}"

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])
    roster_players = summarize_roster_players(
        roster=roster,
        players_index=ctx.get("players_index") or {},
        players_map=ctx.get("players_map") or {},
        model_value_lookup=model_value_lookup,
    )

    position_strength = group_position_strength(roster_players)
    future_picks = ctx.get("picks_by_roster", {}).get(str(viewer_roster_id), [])

    top_assets = roster_players[:8]
    aging_assets = [
        p for p in roster_players
        if p.get("age") not in (None, "") and _safe_float(p.get("age")) >= 28 and _safe_float(p.get("value")) >= 2500
    ][:5]

    direction = detect_team_direction(roster_players, future_picks)

    standing = (ctx.get("standings_map") or {}).get(str(viewer_roster_id), {})
    record = standing.get("record") or standing.get("display_record") or ""
    pf = _safe_float(standing.get("PF"))
    pa = _safe_float(standing.get("PA"))

    return {
        "league_id": ctx.get("league_id"),
        "season": ctx.get("current_season"),
        "week": ctx.get("current_week"),
        "viewer_roster_id": str(viewer_roster_id),
        "team_name": team_name,
        "record": record,
        "points_for": round(pf, 1),
        "points_against": round(pa, 1),
        "direction": direction,
        "top_assets": top_assets,
        "aging_assets": aging_assets,
        "future_picks": future_picks,
        "position_strength": position_strength,
        "roster_size": len(roster_players),
    }


def _safe_str(v, default: str = "") -> str:
    if v is None:
        return default
    return str(v).strip() or default


def _format_pick_display(pk: str) -> str:
    """
    Converts:
      2026_1_04 -> 2026 1.04
      2026_1_early -> 2026 Early 1st
      fallback -> original string
    """
    raw = _safe_str(pk)
    if not raw:
        return ""

    parts = raw.split("_")
    if len(parts) == 3:
        year, rnd, slot = parts
        if slot.isdigit():
            return f"{year} {rnd}.{slot.zfill(2)}"
        bucket = slot.lower()
        bucket_label = bucket.capitalize()
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {bucket_label} {suffix}"

    if len(parts) == 2:
        year, rnd = parts
        suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(_safe_int(rnd), f"{rnd}th")
        return f"{year} {suffix}"

    return raw


def _find_roster(ctx: dict, roster_id: str) -> Union[dict, None]:
    for r in ctx.get("rosters") or []:
        if str(r.get("roster_id")) == str(roster_id):
            return r
    return None


def _top_position_edges(position_strength: dict[str, dict]) -> tuple[str, str]:
    if not position_strength:
        return "Unknown", "Unknown"

    scored = []
    for pos, meta in position_strength.items():
        scored.append((pos, _safe_float(meta.get("top_3_sum")), _safe_float(meta.get("best"))))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0][0]
    worst = scored[-1][0]
    return best, worst


def build_front_office_brief_context(ctx: dict, viewer_roster_id: str) -> Union[dict, None]:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return None

    pos_strength = team_ctx.get("position_strength") or {}
    best_pos, worst_pos = _top_position_edges(pos_strength)

    top_assets = team_ctx.get("top_assets") or []
    future_picks = team_ctx.get("future_picks") or []

    return {
        "league_id": team_ctx.get("league_id"),
        "season": team_ctx.get("season"),
        "week": team_ctx.get("week"),
        "team_name": team_ctx.get("team_name"),
        "record": team_ctx.get("record"),
        "direction": team_ctx.get("direction"),
        "points_for": team_ctx.get("points_for"),
        "points_against": team_ctx.get("points_against"),
        "best_position": best_pos,
        "weakest_position": worst_pos,
        "top_assets": top_assets[:6],
        "aging_assets": (team_ctx.get("aging_assets") or [])[:4],
        "future_picks": future_picks[:8],
        "position_strength": pos_strength,
    }


def build_trade_ai_context(
        ctx: dict,
        viewer_roster_id: str,
        viewer_side: str,
        side_a: dict,
        side_b: dict,
) -> Union[dict, None]:
    """
    side_a / side_b are the objects your /api/trade-eval route already builds.
    """
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return None

    viewer_side = (viewer_side or "a").lower().strip()
    viewer_gets = side_a if viewer_side == "a" else side_b
    viewer_gives = side_b if viewer_side == "a" else side_a

    def clean_asset(asset: dict) -> dict:
        return {
            "id": _safe_str(asset.get("id")),
            "name": _safe_str(asset.get("name")),
            "position": _safe_str(asset.get("position")),
            "team": _safe_str(asset.get("team")),
            "age": asset.get("age"),
            "value": round(_safe_float(asset.get("value")), 1),
        }

    def clean_pick(pk: Any) -> dict:
        raw = _safe_str(pk)
        return {
            "id": raw,
            "display": _format_pick_display(raw),
        }

    return {
        "viewer_team": {
            "roster_id": str(viewer_roster_id),
            "team_name": team_ctx.get("team_name"),
            "direction": team_ctx.get("direction"),
            "record": team_ctx.get("record"),
            "points_for": team_ctx.get("points_for"),
            "points_against": team_ctx.get("points_against"),
            "top_assets": (team_ctx.get("top_assets") or [])[:6],
            "aging_assets": (team_ctx.get("aging_assets") or [])[:4],
            "future_picks": (team_ctx.get("future_picks") or [])[:8],
            "position_strength": team_ctx.get("position_strength") or {},
        },
        "viewer_side": viewer_side,
        "viewer_gets": {
            "players": [clean_asset(a) for a in (viewer_gets.get("assets") or []) if str(a.get("position")) != "PICK"],
            "picks": [clean_pick(pk) for pk in (viewer_gets.get("pick_ids") or [])],
            "raw_total": round(_safe_float(viewer_gets.get("raw_total")), 1),
            "effective_total": round(_safe_float(viewer_gets.get("effective_total")), 1),
            "adjustment": round(_safe_float(viewer_gets.get("adjustment")), 1),
        },
        "viewer_gives": {
            "players": [clean_asset(a) for a in (viewer_gives.get("assets") or []) if str(a.get("position")) != "PICK"],
            "picks": [clean_pick(pk) for pk in (viewer_gives.get("pick_ids") or [])],
            "raw_total": round(_safe_float(viewer_gives.get("raw_total")), 1),
            "effective_total": round(_safe_float(viewer_gives.get("effective_total")), 1),
            "adjustment": round(_safe_float(viewer_gives.get("adjustment")), 1),
        },
        "net_effective_delta": round(
            _safe_float(viewer_gets.get("effective_total")) - _safe_float(viewer_gives.get("effective_total")),
            1,
        ),
    }
