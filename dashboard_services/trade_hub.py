"""Unified Trade Hub: thin glue over the existing trade engines.

The hub converges the three "who should I trade for" surfaces (archetype
trade suggestions, positional trade targets, trade-intel market data) into one
tabbed UI. This module holds the shared, server-side pieces:

- one consistent ranking-explanation line per row ("why this"), reusing the
  Front Office "why they'd say yes" pattern (deterministic, server-computed,
  never AI-only),
- the "shop this to all teams" repricer, which prices a send-side package
  against every roster's positional needs using the same starter-slot-weighted
  strength model as Trade Targets.

It never reimplements valuation: player values come from the league context's
model value table, needs from utils.trade_targets / utils.roster_strength.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")

# Pick value estimates mirror api_trade_targets' _pick_val_est so the shop
# repricer prices picks the same way the rest of the trade suite does.
PICK_VALUE_BY_ROUND = {1: 650.0, 2: 220.0, 3: 80.0}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any) -> str:
    return str(v or "").strip()


# ---------------------------------------------------------------------------
# Shared "why this" ranking-explanation lines
# ---------------------------------------------------------------------------
# One consistent component across every hub row: a short deterministic line
# explaining why the row is ranked where it is. Computed server-side from the
# same fields the rankers already produced (mirrors the Front Office
# why_they_say_yes pattern). No em dashes in copy.


def why_line_for_suggestion(s: Dict[str, Any]) -> str:
    """Explanation line for an archetype-engine suggestion row."""
    bits: List[str] = []
    fit = _safe_str(s.get("fit_note")) or _safe_str(s.get("why"))
    if fit:
        bits.append(fit.rstrip(".") + ".")
    acpt = s.get("acceptance_pct")
    if acpt is not None:
        try:
            bits.append(f"About {int(round(float(acpt)))}% likely to be accepted.")
        except (TypeError, ValueError):
            pass
    pod = s.get("net_playoff_odds_delta", s.get("playoff_odds_delta"))
    if pod is not None:
        try:
            pct = float(pod) * 100
            if abs(pct) >= 0.05:
                direction = "Adds" if pct > 0 else "Costs"
                bits.append(f"{direction} {abs(pct):.1f}% playoff odds.")
        except (TypeError, ValueError):
            pass
    partner = _safe_str(s.get("partner_team"))
    parch = _safe_str(s.get("partner_arch"))
    if partner and parch:
        bits.append(f"Partner fits a {parch} build.")
    return " ".join(bits)


def why_line_for_target(t: Dict[str, Any]) -> str:
    """Explanation line for a trade-targets row."""
    bits: List[str] = []
    why = _safe_str(t.get("why"))
    if why:
        bits.append(why.rstrip(".") + ".")
    owner = _safe_str(t.get("owner_team"))
    if owner:
        bits.append(f"On {owner}.")
    owner_needs = [str(n).upper() for n in (t.get("owner_needs") or [])]
    if owner_needs:
        bits.append(f"They need {'/'.join(owner_needs)}.")
    return " ".join(bits)


def why_line_for_market(p: Dict[str, Any]) -> str:
    """Explanation line for a trade-intel market row."""
    bits: List[str] = []
    cnt7 = p.get("trade_count_7d") or 0
    try:
        cnt7 = int(cnt7)
    except (TypeError, ValueError):
        cnt7 = 0
    if cnt7 > 0:
        bits.append(f"In {cnt7} real trades this week.")
    trend = p.get("market_trend")
    if trend is not None:
        try:
            tr = float(trend)
            if tr >= 5:
                bits.append("Market price rising.")
            elif tr <= -5:
                bits.append("Market price falling.")
        except (TypeError, ValueError):
            pass
    delta = p.get("value_delta")
    model = p.get("model_value")
    if delta is not None and model:
        try:
            d = float(delta)
            if d <= -5:
                bits.append("Priced below the BR model. Buy-low window.")
            elif d >= 5:
                bits.append("Priced above the BR model. Sell-high window.")
        except (TypeError, ValueError):
            pass
    bsr = p.get("buy_sell_ratio")
    if bsr is not None:
        try:
            r = float(bsr)
            if r >= 1.2:
                bits.append("More buyers than sellers in real deals.")
            elif r <= 0.8:
                bits.append("More sellers than buyers in real deals.")
        except (TypeError, ValueError):
            pass
    return " ".join(bits)


def why_line_for_shop(row: Dict[str, Any]) -> str:
    """Explanation line for a shop-to-all-teams row."""
    bits: List[str] = []
    needs = [str(n).upper() for n in (row.get("team_needs") or [])]
    if needs:
        bits.append(f"They need {'/'.join(needs)}.")
    get_names = [g.get("name") for g in (row.get("suggested_get") or []) if g.get("name")]
    if get_names:
        bits.append(f"Fair return: {', '.join(get_names[:3])}.")
    fairness = row.get("fairness")
    if fairness is not None:
        try:
            f = float(fairness)
            if f >= 1.08:
                bits.append("You win the value math.")
            elif f <= 0.92:
                bits.append("You pay a little extra for the fit.")
        except (TypeError, ValueError):
            pass
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Value table (mirrors api_trade_targets' key selection)
# ---------------------------------------------------------------------------

def values_by_id(
    ctx: Dict[str, Any],
    league_type: str = "1qb",
    league_size: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Build {player_id: {value, position, name, team, age}} from the ctx."""
    league_type = (league_type or "1qb").strip().lower()
    league_size = int(league_size or 10)
    if league_type == "sf":
        val_key = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
        val_fallback = "sf_value"
    else:
        val_key = "value" if league_size == 10 else f"value_{league_size}"
        val_fallback = "value"
    model_value_table = ctx.get("model_value_table") or []
    players_index = ctx.get("players_index") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        if not pid:
            continue
        out[pid] = {
            "value": _safe_float(row.get(val_key) or row.get(val_fallback) or row.get("value")),
            "position": str(row.get("position") or "").upper(),
            "name": row.get("name") or players_index.get(pid, {}).get("name", f"Player {pid}"),
            "team": row.get("team") or "",
            "age": row.get("age"),
        }
    return out


def pick_value_for(pick_id: str) -> float:
    """Value estimate for a draft-pick asset id."""
    try:
        from utils.pick_slots import parse_pick_asset
    except Exception:
        return 0.0
    parsed = parse_pick_asset(pick_id)
    if not parsed:
        return 0.0
    return PICK_VALUE_BY_ROUND.get(int(parsed.get("round") or 0), 40.0)


# ---------------------------------------------------------------------------
# Shop this to all teams
# ---------------------------------------------------------------------------

def _roster_needs(
    roster_vals: Dict[str, Dict[str, List[float]]],
    roster_positions: Sequence[str],
    num_teams: int,
    slot_counts: Dict[str, int],
    is_sf: bool,
) -> Dict[str, List[str]]:
    """Positional needs per roster via starter-slot-weighted strength."""
    from utils.roster_strength import derive_league_thresholds, weighted_pos_strength
    from utils.trade_targets import detect_needed_positions

    pos_strength = {
        rid: {pos: weighted_pos_strength(vals.get(pos, []), pos, slot_counts)
              for pos in POSITIONS}
        for rid, vals in roster_vals.items()
    }
    pos_ranks: Dict[str, Dict[str, int]] = {}
    for pos in POSITIONS:
        ordered = sorted(pos_strength.keys(),
                         key=lambda rid: pos_strength[rid].get(pos, 0.0),
                         reverse=True)
        pos_ranks[pos] = {rid: i + 1 for i, rid in enumerate(ordered)}
    thr, floors = derive_league_thresholds(
        list(roster_positions or []), num_teams, is_sf=is_sf,
    )
    return {
        rid: detect_needed_positions(
            {pos: pos_ranks[pos].get(rid, num_teams) for pos in POSITIONS},
            vals, num_teams, thr, floors,
        )
        for rid, vals in roster_vals.items()
    }


def _asset_value(asset_id: str, values: Dict[str, Dict[str, Any]]) -> float:
    info = values.get(str(asset_id))
    if info:
        return _safe_float(info.get("value"))
    return pick_value_for(str(asset_id))


def _cheapest_fair_return(
    candidates: List[Dict[str, Any]],
    target_value: float,
    max_pieces: int = 3,
) -> Tuple[List[Dict[str, Any]], float]:
    """Best asset set reaching >= 85% of target_value (cap max_pieces).

    Tries combinations of 1..max_pieces and picks the set closest to 100%
    of the target (preferring fewer pieces and viewer-needed positions on
    ties). A set over the 85% floor always beats nothing: the why-line calls
    out when you win or lose the value math.
    """
    from itertools import combinations

    floor = target_value * 0.85
    if target_value <= 0 or not candidates:
        return [], 0.0
    # Cap the pool so combination search stays cheap; prefer need-fit and
    # cheap assets since those make the fairest, most useful returns.
    pool = sorted(candidates, key=lambda c: (not c.get("need_fit"), c["value"]))[:14]
    best: Optional[Tuple[Tuple[float, int, int], Tuple[Dict[str, Any], ...], float]] = None
    for n in range(1, max_pieces + 1):
        for combo in combinations(pool, n):
            total = sum(_safe_float(c.get("value")) for c in combo)
            if total < floor:
                continue
            overpay = total / target_value
            fits = sum(1 for c in combo if c.get("need_fit"))
            key = (abs(overpay - 1.0), -fits, n)
            if best is None or key < best[0]:
                best = (key, combo, total)
    if best is None:
        return [], 0.0
    return list(best[1]), best[2]


def shop_package(
    ctx: Dict[str, Any],
    viewer_roster_id: str,
    send_ids: Sequence[str],
    league_type: str = "1qb",
    league_size: int = 10,
) -> Dict[str, Any]:
    """Re-price a send-side package against every other roster's needs.

    Returns per-team rows: team name, their needs, a value-matched return
    built from their roster (viewer-needed positions first, then picks), the
    fairness ratio, and a shared why-line.
    """
    league_type = (league_type or "1qb").strip().lower()
    league_size = int(league_size or 10)
    is_sf = league_type == "sf"
    viewer_roster_id = str(viewer_roster_id or "")
    send_ids = [str(s) for s in (send_ids or []) if str(s).strip()]

    rosters = ctx.get("rosters") or []
    roster_map = ctx.get("roster_map") or {}
    values = values_by_id(ctx, league_type, league_size)

    # Redraft leagues have no pick assets to price.
    from dashboard_services.ai.context_builders import ctx_scoring_type
    is_redraft = ctx_scoring_type(ctx or {}) == "redraft"
    picks_by_roster: Dict[str, list] = {}
    if not is_redraft:
        picks_by_roster = ctx.get("picks_by_roster") or {}

    try:
        from utils.utils import count_roster_positions
    except Exception:
        count_roster_positions = None  # type: ignore

    rp_list = ctx.get("roster_positions") or []
    slot_counts = count_roster_positions(rp_list) if count_roster_positions else {}
    if not any(slot_counts.get(p) for p in ("QB", "RB", "WR", "TE", "FLEX")):
        slot_counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}

    def _pos_vals(player_ids: list) -> Dict[str, List[float]]:
        vals: Dict[str, List[float]] = {p: [] for p in POSITIONS}
        for pid in player_ids:
            info = values.get(str(pid))
            if info and info["position"] in POSITIONS and info["value"] > 0:
                vals[info["position"]].append(info["value"])
        return vals

    roster_vals = {
        str(r.get("roster_id")): _pos_vals(r.get("players") or [])
        for r in rosters
    }
    num_teams = max(len(rosters), 1)
    needs_by_roster = _roster_needs(roster_vals, rp_list, num_teams, slot_counts, is_sf)
    viewer_needs = set(needs_by_roster.get(viewer_roster_id, []))

    send_assets: List[Dict[str, Any]] = []
    send_value = 0.0
    for aid in send_ids:
        v = _asset_value(aid, values)
        info = values.get(aid, {})
        send_assets.append({
            "id": aid,
            "name": info.get("name") or aid,
            "position": info.get("position") or ("PICK" if pick_value_for(aid) else "?"),
            "value": round(v, 1),
        })
        send_value += v

    rows: List[Dict[str, Any]] = []
    for roster in rosters:
        rid = str(roster.get("roster_id"))
        if rid == viewer_roster_id:
            continue
        team_needs = needs_by_roster.get(rid, [])
        team_players = [str(p) for p in (roster.get("players") or [])]

        candidates: List[Dict[str, Any]] = []
        # Viewer-needed positions first, then the rest of their startable
        # assets. Cheap, startable pieces make the fairest return.
        for pid in team_players:
            info = values.get(pid)
            if not info or info["position"] not in POSITIONS:
                continue
            if info["value"] < 150:
                continue
            candidates.append({
                "id": pid,
                "name": info["name"],
                "position": info["position"],
                "value": round(info["value"], 1),
                "need_fit": info["position"] in viewer_needs,
            })
        for pk in (picks_by_roster.get(rid) or []):
            try:
                rnd = int((pk.get("round") or 0))
            except (TypeError, ValueError):
                continue
            if rnd not in (1, 2):
                continue
            season = pk.get("season")
            slot = pk.get("slot")
            pid = f"{season}_{rnd}_{int(slot):02d}" if slot else f"{season}_{rnd}"
            pv = PICK_VALUE_BY_ROUND.get(rnd, 0.0)
            candidates.append({
                "id": pid,
                "name": f"{season} {rnd}{'st' if rnd == 1 else 'nd'} pick",
                "position": "PICK",
                "value": round(pv, 1),
                "need_fit": False,
            })

        # Prefer assets at positions the viewer needs; keep picks as filler.
        candidates.sort(key=lambda c: (not c["need_fit"], c["value"]))
        chosen, get_value = _cheapest_fair_return(candidates, send_value)
        if not chosen:
            continue
        fairness = round(get_value / send_value, 3) if send_value > 0 else 0.0
        row = {
            "team": roster_map.get(rid, f"Roster {rid}"),
            "roster_id": rid,
            "team_needs": team_needs,
            "send_value": round(send_value, 1),
            "suggested_get": [
                {"id": c["id"], "name": c["name"], "position": c["position"],
                 "value": c["value"]}
                for c in chosen
            ],
            "get_value": round(get_value, 1),
            "fairness": fairness,
        }
        row["why_line"] = why_line_for_shop(row)
        rows.append(row)

    rows.sort(key=lambda r: (-float(r["fairness"] or 0), r["team"]))
    result = {
        "send": send_assets,
        "send_value": round(send_value, 1),
        "viewer_needs": sorted(viewer_needs),
        "teams": rows,
    }
    return result
