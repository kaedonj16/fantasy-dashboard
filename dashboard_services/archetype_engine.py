"""
Archetype-based trade suggestion engine.

Computes Contending / Rebuilding / Consolidate / Distribute suggestions
based on roster composition, value trends, and win probability modeling.

No imports from app.py — safe for use from any blueprint or endpoint.
"""
from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PEAK_AGE: Dict[str, int] = {"QB": 29, "RB": 26, "WR": 27, "TE": 27}
SKILL_POS = {"QB", "RB", "WR", "TE"}
FLEX_POS  = {"RB", "WR", "TE"}

# Required starters per format (FLEX filled after dedicated slots)
SLOTS_1QB: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
SLOTS_SF:  Dict[str, int] = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}

# Position weight for target scoring (higher = better consolidation target)
SCARCITY_1QB = {"QB": 0.80, "RB": 1.00, "WR": 1.00, "TE": 0.85}
SCARCITY_SF  = {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 0.85}

COMPLEMENT = {
    "contending":  "rebuilding",
    "rebuilding":  "contending",
    "consolidate": "distribute_candidate",
    "distribute":  "distribute_candidate",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _seed(standings_map: Dict, roster_id: Any, fallback: int = 99) -> int:
    """Look up standings seed tolerating int/str key mismatch."""
    rid_int = int(roster_id) if str(roster_id).isdigit() else None
    rid_str = str(roster_id)
    return (
        standings_map.get(rid_int)
        or standings_map.get(rid_str)
        or fallback
    )


def _roster_name(roster_map: Dict, roster_id: Any) -> str:
    rid_int = int(roster_id) if str(roster_id).isdigit() else None
    rid_str = str(roster_id)
    return (
        roster_map.get(rid_int)
        or roster_map.get(rid_str)
        or f"Team {roster_id}"
    )


# ── Lineup optimizer ─────────────────────────────────────────────────────────

def _optimal_lineup_value(
    player_ids: List[str],
    values_by_id: Dict[str, Any],
    league_type: str = "1qb",
) -> float:
    """
    Greedy lineup optimizer. Fill dedicated slots first (QB/RB/WR/TE),
    then fill FLEX with best remaining eligible players.
    """
    slots = SLOTS_SF if league_type == "sf" else SLOTS_1QB
    flex_pos = SKILL_POS if league_type == "sf" else FLEX_POS

    by_pos: Dict[str, List[float]] = {}
    for pid in player_ids:
        info = values_by_id.get(pid)
        if not info:
            continue
        pos = str(info.get("position") or "").upper()
        val = _f(info.get("value"))
        if pos in SKILL_POS and val > 0:
            by_pos.setdefault(pos, []).append(val)

    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    total = 0.0
    used: Dict[str, int] = {}

    for pos, count in slots.items():
        if pos == "FLEX":
            continue
        vals = by_pos.get(pos, [])
        n = min(count, len(vals))
        total += sum(vals[:n])
        used[pos] = n

    # Collect remaining values eligible for FLEX
    flex_pool: List[float] = []
    for pos in (SKILL_POS if league_type == "sf" else FLEX_POS):
        already = used.get(pos, 0)
        flex_pool.extend(by_pos.get(pos, [])[already:])

    flex_pool.sort(reverse=True)
    flex_count = slots.get("FLEX", 0)
    total += sum(flex_pool[:flex_count])
    return total


# ── Win-probability model ─────────────────────────────────────────────────────

def _win_prob(team_val: float, league_avg: float) -> float:
    """Logistic win probability from relative lineup value."""
    ratio = team_val / max(1.0, league_avg)
    return 1.0 / (1.0 + math.exp(-4.0 * (ratio - 1.0)))


def _wp_delta(
    viewer_val: float,
    target_val: float,
    replace_val: float,
    league_avg: float,
) -> float:
    new_val = viewer_val - replace_val + target_val
    return _win_prob(new_val, league_avg) - _win_prob(viewer_val, league_avg)


# ── Team archetype inference ──────────────────────────────────────────────────

def _infer_archetype(
    player_ids: List[str],
    values_by_id: Dict[str, Any],
    seed: int,
    num_teams: int,
    playoff_spots: int,
) -> str:
    skill = [p for p in player_ids if values_by_id.get(p, {}).get("position") in SKILL_POS]
    if not skill:
        return "balanced"

    dyn_vals  = [_f(values_by_id[p].get("value")) for p in skill]
    rdft_vals = [_f(values_by_id[p].get("redraft_value")) for p in skill]
    ages      = [_f(values_by_id[p].get("age"), PEAK_AGE.get(values_by_id[p].get("position", "WR"), 27))
                 for p in skill]

    total_dyn  = sum(dyn_vals)
    total_rdft = sum(rdft_vals)
    avg_age    = sum(ages) / len(ages) if ages else 26.0
    rdft_ratio = total_rdft / total_dyn if total_dyn > 0 else 0.5
    is_above   = seed <= playoff_spots

    # Distribute candidate: top-2 players hold > 55 % of total dynasty value
    sorted_dyn = sorted(dyn_vals, reverse=True)
    top2_share = sum(sorted_dyn[:2]) / max(1, total_dyn) if len(sorted_dyn) >= 4 else 0
    if top2_share > 0.55:
        return "distribute_candidate"

    if is_above and rdft_ratio > 0.80:
        return "contending"
    if not is_above and avg_age < 26.5:
        return "rebuilding"
    return "balanced"


# ── 30-day value trend ────────────────────────────────────────────────────────

def _get_30d_values(player_ids: List[str]) -> Dict[str, float]:
    """
    Batch-fetch the earliest value snapshot within the past 30 days per player.
    Returns {player_id: old_value}. Falls back to {} if DB unavailable.
    """
    if not player_ids:
        return {}
    try:
        from datetime import date, timedelta
        from dashboard_services.db import get_conn
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (player_id) player_id, value
                    FROM player_value_history
                    WHERE player_id = ANY(%s)
                      AND as_of_date <= %s
                    ORDER BY player_id, as_of_date DESC
                    """,
                    (player_ids, cutoff),
                )
                rows = cur.fetchall()
        return {r["player_id"]: _f(r["value"]) for r in (rows or [])}
    except Exception as exc:
        log.debug("[archetype] 30d trend fetch skipped: %s", exc)
        return {}


def _trend_pct(pid: str, current: float, old_vals: Dict[str, float],
               values_by_id: Dict[str, Any]) -> float:
    old = old_vals.get(pid, 0)
    if old > 0:
        return (current - old) / old
    # Proxy: rank_change_7d (positive = rank improved = value rising)
    rc = _f(values_by_id.get(pid, {}).get("rank_change_7d"), 0)
    return rc * -0.003  # rank improves by 1 → ~0.3 % value rise


# ── "Why" explanation generator ───────────────────────────────────────────────

def _build_why(
    t: Dict[str, Any],
    archetype: str,
    tp: float,
    wp_d: float,
) -> str:
    name     = t.get("name", "")
    pos      = t.get("position", "")
    age      = _f(t.get("age"))
    val      = _f(t.get("value"))
    rdft     = _f(t.get("redraft_value"))
    partner  = t.get("partner_name", "")
    peak     = PEAK_AGE.get(pos, 27)

    age_tag = ""
    if age:
        diff = age - peak
        if diff < -2:
            age_tag = f"age {age:.0f}, well below peak"
        elif diff <= 0:
            age_tag = f"age {age:.0f}, prime window"
        elif diff <= 2:
            age_tag = f"age {age:.0f}, near peak"
        else:
            age_tag = f"age {age:.0f}, post-peak"

    trend_tag = ""
    if tp > 0.04:
        trend_tag = "rising value"
    elif tp < -0.05:
        trend_tag = "declining trend"

    if archetype == "contending":
        parts = []
        if rdft > val * 1.08:
            parts.append("redraft value tops dynasty, productive now")
        if age_tag:
            parts.append(age_tag)
        if trend_tag:
            parts.append(trend_tag)
        parts.append(t.get("partner_phrase") or f"{partner} may be a seller")
        body = ", ".join(parts[:3])
        return f"{name} offers immediate impact. {body[:1].upper()}{body[1:]}."

    if archetype == "rebuilding":
        parts = []
        if val > rdft * 1.08:
            parts.append("dynasty value exceeds current production")
        if age_tag:
            parts.append(age_tag)
        if trend_tag:
            parts.append(trend_tag)
        parts.append(t.get("partner_phrase") or f"{partner} may move youth for win-now pieces")
        body = ", ".join(parts[:3])
        return f"{name} is a long-term asset. {body[:1].upper()}{body[1:]}."

    if archetype == "consolidate":
        parts = []
        if rdft > val * 1.05:
            parts.append("proven current production")
        if age_tag:
            parts.append(age_tag)
        tail = t.get("partner_phrase") or f"{partner} has concentrated value and can absorb depth pieces"
        return f"Consolidating around {name} ({', '.join(parts[:2])}). {tail}."

    if archetype == "distribute":
        return (
            f"Distribute {name}'s value into multiple starters. "
            f"{partner} needs depth and can offer a multi-player return."
        )

    return f"{name} improves your win probability by {wp_d:+.1%}."


# ── Send candidate scoring ────────────────────────────────────────────────────

def _score_sends(
    player_ids: List[str],
    values_by_id: Dict[str, Any],
    archetype: str,
    untouchable_ids=None,
) -> List[Dict[str, Any]]:
    """Score viewer's players as send candidates; returns sorted list."""
    out = []
    for pid in player_ids:
        if untouchable_ids and pid in untouchable_ids:
            continue
        info = values_by_id.get(pid)
        if not info:
            continue
        pos  = str(info.get("position") or "").upper()
        val  = _f(info.get("value"))
        rdft = _f(info.get("redraft_value"))
        age  = _f(info.get("age"), 0)
        if pos not in SKILL_POS or val <= 0:
            continue
        peak = PEAK_AGE.get(pos, 27)

        sc = 0.0
        if archetype == "contending":
            if age > 0 and age < 24 and rdft < val * 0.85:
                sc = 0.6 + min(val / 1000, 0.4)       # young upside
            elif age >= peak and rdft < val * 0.92:
                sc = 0.65 + min(val / 1000, 0.35)     # post-peak, sell
            else:
                sc = 0.15
        elif archetype == "consolidate":
            if 300 <= val <= 650:
                sc = 0.85 - abs(val - 475) / 475      # sweet spot mid-tier
            else:
                sc = 0.05
        elif archetype == "distribute":
            if val >= 800:
                sc = 1.0
            elif val >= 600:
                sc = 0.7
            else:
                sc = 0.05

        out.append({
            "player_id": pid,
            "name":      info.get("name", ""),
            "position":  pos,
            "value":     round(val, 1),
            "_sc":       sc,
        })

    out.sort(key=lambda x: x["_sc"], reverse=True)
    for c in out:
        c.pop("_sc", None)
    return out


def _select_package(
    sends: List[Dict], target_val: float, archetype: str
) -> List[Dict]:
    """Choose 1–3 send players that approximately match target value (±30 %)."""
    if not sends:
        return []
    lo, hi = target_val * 0.70, target_val * 1.30
    pool = sends[:10]

    if archetype == "consolidate":
        # Prefer 2–3 mid-tier combo
        best, best_diff = [], float("inf")
        for n in (2, 3):
            for combo in combinations(pool, n):
                if sum(1 for c in combo if c["position"] == "QB") > 1:
                    continue
                s = sum(c["value"] for c in combo)
                if lo <= s <= hi and abs(s - target_val) < best_diff:
                    best_diff = abs(s - target_val)
                    best = list(combo)
        return best if best else pool[:2]

    if archetype == "distribute":
        return pool[:1]

    # Contending: 1-player first, then 2-combo
    for c in pool:
        if lo <= c["value"] <= hi:
            return [c]
    for a, b in combinations(pool, 2):
        if a["position"] == "QB" and b["position"] == "QB":
            continue
        s = a["value"] + b["value"]
        if lo <= s <= hi:
            return [a, b]
    return pool[:1]


# ── Pick send candidates ──────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _pick_send_candidates(picks: List[Dict], num_teams: int) -> List[Dict[str, Any]]:
    """Convert a roster's future picks into send-candidate dicts with est. values."""
    if not picks:
        return []
    pick_tbl: Dict[str, float] = {}
    try:
        from dashboard_services.picks import load_pick_value_table
        pick_tbl = load_pick_value_table(league_teams=num_teams) or {}
    except Exception:
        pick_tbl = {}

    out = []
    for pk in picks:
        if not isinstance(pk, dict):
            continue
        season = str(pk.get("season") or pk.get("year") or "")
        rnd    = int(pk.get("round") or 0)
        if not season or rnd <= 0:
            continue
        val = 0.0
        for key in (f"{season}_{rnd}_mid", f"{season}_{rnd}", f"{season}_{rnd}_early"):
            if key in pick_tbl and float(pick_tbl[key]) > 0:
                val = float(pick_tbl[key])
                break
        if val <= 0:
            val = {1: 650.0, 2: 220.0}.get(rnd, 80.0)
        out.append({
            "player_id": f"pick_{season}_{rnd}",
            "name":      f"{season} {_ordinal(rnd)}",
            "position":  "PICK",
            "value":     round(val, 1),
            "is_pick":   True,
        })
    return out


# ── Partner phrasing (signal-driven, varied) ──────────────────────────────────

def _partner_phrase(
    arch: str, name: str, seed: int, playoff_spots: int,
    avg_age: Optional[float], rdft_ratio: Optional[float],
    partner_weak: Optional[List[str]] = None,
) -> str:
    need_sfx = ""
    if partner_weak:
        need_sfx = f", needs {'/'.join(partner_weak[:2])} depth"
    if arch == "rebuilding":
        if avg_age and avg_age < 25:
            return f"{name} skews young and may sell win-now vets{need_sfx}"
        if seed > playoff_spots + 1:
            return f"{name} sits outside playoff position, likely selling{need_sfx}"
        return f"{name} looks like a rebuild partner{need_sfx}"
    if arch == "contending":
        if rdft_ratio and rdft_ratio > 0.95:
            return f"{name} is built to win now and may pay up{need_sfx}"
        if seed <= max(2, playoff_spots // 2):
            return f"{name} is a title contender chasing pieces{need_sfx}"
        return f"{name} is in win-now mode{need_sfx}"
    if arch == "distribute_candidate":
        return f"{name} is top-heavy and needs depth{need_sfx}"
    return f"{name} could be a match{need_sfx}"


# ── Distribute suggestion builder (viewer sends one stud for depth) ───────────

def _build_distribute(
    viewer_players: List[str],
    values_by_id: Dict[str, Any],
    targets_by_owner: Dict[str, List[Dict]],
    owner_meta: Dict[str, Dict],
    roster_map: Dict,
    league_type: str,
    viewer_lineup_val: float,
    league_avg: float,
    untouchable_ids=None,
) -> List[Dict[str, Any]]:
    """
    Viewer sends one concentrated stud and receives a 2–3 player depth package.
    Each card = one stud → one partner's multi-player return.
    Only keeps trades where the optimal lineup ceiling rises.
    """
    studs = sorted(
        [p for p in viewer_players
         if values_by_id.get(p, {}).get("position") in SKILL_POS
         and _f(values_by_id[p].get("value")) >= 600
         and (not untouchable_ids or p not in untouchable_ids)],
        key=lambda p: _f(values_by_id[p].get("value")),
        reverse=True,
    )[:3]

    results: List[Dict[str, Any]] = []
    used_owners: set = set()

    for stud in studs:
        sval  = _f(values_by_id[stud].get("value"))
        sname = values_by_id[stud].get("name", "")
        spos  = values_by_id[stud].get("position", "")
        lo, hi = sval * 0.75, sval * 1.25

        best: Optional[Tuple[str, List[Dict], float]] = None
        for owner, pool in targets_by_owner.items():
            if owner in used_owners:
                continue
            cand = sorted(pool, key=lambda x: x["value"], reverse=True)[:8]
            for n in (2, 3):
                for combo in combinations(cand, n):
                    s = sum(c["value"] for c in combo)
                    if lo <= s <= hi:
                        diff = abs(s - sval)
                        if best is None or diff < best[2]:
                            best = (owner, list(combo), diff)

        if not best:
            continue
        owner, combo, _ = best
        used_owners.add(owner)

        recv_ids   = [c["player_id"] for c in combo]
        new_players = [p for p in viewer_players if p != stud] + recv_ids
        new_lineup  = _optimal_lineup_value(new_players, values_by_id, league_type)
        wpd         = _win_prob(new_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)

        pname  = _roster_name(roster_map, owner)
        p_arch = owner_meta.get(owner, {}).get("arch", "")
        ceiling_note = "lineup ceiling rises" if wpd >= 0 else "adds depth but trims your ceiling"

        results.append({
            "player_id":      stud,
            "name":           sname,
            "position":       spos,
            "nfl_team":       values_by_id[stud].get("team", ""),
            "age":            _f(values_by_id[stud].get("age")),
            "value":          round(sval, 1),
            "redraft_value":  round(_f(values_by_id[stud].get("redraft_value")), 1),
            "pos_rank_label": values_by_id[stud].get("pos_rank_label", ""),
            "why":            (f"Spread {sname}'s value into {len(combo)} starters from {pname}. "
                               f"{ceiling_note.capitalize()}, filling multiple holes at once."),
            "partner_team":   pname,
            "partner_arch":   p_arch,
            "win_prob_delta": round(wpd, 4),
            "direction":      "distribute",
            "suggested_send": [{
                "player_id": stud, "name": sname,
                "position": spos, "value": round(sval, 1),
            }],
            "suggested_receive": [{
                "player_id": c["player_id"], "name": c["name"],
                "position": c["position"], "value": round(c["value"], 1),
            } for c in combo],
        })
        if len(results) >= 5:
            break

    return results


# ── Rebuilding suggestion builder (sell a vet, acquire youth) ─────────────────

def _build_rebuilding(
    viewer_players: List[str],
    values_by_id: Dict[str, Any],
    all_targets: List[Dict[str, Any]],
    league_type: str,
    viewer_lineup_val: float,
    league_avg: float,
    untouchable_ids=None,
) -> List[Dict[str, Any]]:
    """
    Rebuild = sell win-now vets for younger, ascending players of similar value.

    Anchored on each sellable vet (the SEND): pair it with the rival young
    target (the GET) whose value most closely matches the vet, so every card
    has a realistic, fundable send and targets stay within reach (no dangling
    elite players the viewer could never actually trade for).
    """
    # Sellable vets: at/past positional peak age with real remaining value.
    vets = sorted(
        [p for p in viewer_players
         if values_by_id.get(p, {}).get("position") in SKILL_POS
         and _f(values_by_id[p].get("age")) >= PEAK_AGE.get(
             values_by_id[p].get("position", "WR"), 27) - 1
         and _f(values_by_id[p].get("value")) >= 250
         and (not untouchable_ids or p not in untouchable_ids)],
        key=lambda p: (
            max(0.0, _f(values_by_id[p].get("age"))
                - PEAK_AGE.get(values_by_id[p].get("position", "WR"), 27))
            * _f(values_by_id[p].get("value"))
        ),
        reverse=True,
    )
    if not vets:
        return []

    # Young, ascending rival targets (below positional peak, real value).
    young = sorted(
        [t for t in all_targets
         if t["age"] and t["age"] < PEAK_AGE.get(t["position"], 27)
         and t["value"] >= 150],
        key=lambda t: -t["value"],
    )
    if not young:
        return []

    old_vals = _get_30d_values([t["player_id"] for t in young])

    results: List[Dict[str, Any]] = []
    used_targets: set = set()

    for vet in vets:
        vval  = _f(values_by_id[vet].get("value"))
        vname = values_by_id[vet].get("name", "")
        vpos  = values_by_id[vet].get("position", "")
        lo, hi = vval * 0.75, vval * 1.30

        candidates = [
            t for t in young
            if t["player_id"] not in used_targets and lo <= t["value"] <= hi
        ]
        if not candidates:
            continue
        pick = min(candidates, key=lambda t: abs(t["value"] - vval))
        used_targets.add(pick["player_id"])

        # Honest win-prob: drop the vet from the lineup, add the young player.
        new_players = [p for p in viewer_players if p != vet] + [pick["player_id"]]
        new_lineup  = _optimal_lineup_value(new_players, values_by_id, league_type)
        wpd = _win_prob(new_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)

        tp  = _trend_pct(pick["player_id"], pick["value"], old_vals, values_by_id)
        pick["win_prob_delta"] = wpd
        why = _build_why(pick, "rebuilding", tp, wpd)

        results.append({
            "player_id":      pick["player_id"],
            "name":           pick["name"],
            "position":       pick["position"],
            "nfl_team":       pick["team"],
            "age":            pick["age"],
            "value":          round(pick["value"], 1),
            "redraft_value":  round(pick.get("redraft_value", 0), 1),
            "pos_rank_label": pick["pos_rank_label"],
            "why":            why,
            "partner_team":   pick["partner_name"],
            "partner_arch":   pick["partner_arch"],
            "win_prob_delta": round(wpd, 4),
            "direction":      "acquire",
            "suggested_send": [{
                "player_id": vet, "name": vname,
                "position": vpos, "value": round(vval, 1),
            }],
        })
        if len(results) >= 5:
            break

    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def get_archetype_suggestions(
    archetype: str,
    platform: str,
    league_id: str,
    season: int,
    viewer_roster_id: str,
    league_type: str = "1qb",
    league_size: int = 10,
    ctx: Optional[Dict[str, Any]] = None,
    untouchable_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Returns up to 5 archetype-targeted trade suggestions.
    Gracefully degrades when DB or league context is unavailable.

    The caller (API endpoint) should pass `ctx` from get_league_ctx_from_cache.
    A lazy fallback import is used only if ctx is not supplied.
    """
    archetype = archetype.lower().strip()

    # ── League context ────────────────────────────────────────────────────────
    if ctx is None:
        try:
            # Lazy, runtime-only import: this engine is itself imported lazily
            # from the request handler, so app.py is fully initialized by now.
            from app import get_league_ctx_from_cache
            ctx = get_league_ctx_from_cache(platform=platform, league_id=league_id, season=season) or {}
        except Exception as exc:
            log.warning("[archetype] ctx load failed: %s", exc)
            ctx = {}
    ctx = ctx or {}

    rosters         = ctx.get("rosters") or []
    roster_map      = ctx.get("roster_map") or {}
    standings_map   = ctx.get("standings_map") or {}
    model_tbl       = ctx.get("model_value_table") or []
    picks_by_roster = ctx.get("picks_by_roster") or {}

    num_teams     = max(len(rosters), league_size, 8)
    playoff_spots = max(4, round(num_teams * 0.4))

    # ── Build values_by_id ────────────────────────────────────────────────────
    val_key  = "sf_value" if league_type == "sf" else "value"
    rdft_key = "redraft_value_sf" if league_type == "sf" else "redraft_value_1qb"

    values_by_id: Dict[str, Any] = {}
    for p in model_tbl:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        pos = str(p.get("position") or "").upper()
        values_by_id[pid] = {
            "name":           p.get("name", ""),
            "position":       pos,
            "team":           p.get("team", ""),
            "age":            p.get("age"),
            "value":          _f(p.get(val_key) or p.get("value")),
            "redraft_value":  0.0,   # filled from DB below
            "pos_rank_label": p.get("pos_rank_label") or "",
            "rank_change_7d": p.get("rank_change_7d"),
        }

    # Augment with calibrated values + redraft from DB
    try:
        from dashboard_services.player_value_history import load_current_values_from_db
        for p in (load_current_values_from_db() or []):
            pid = str(p.get("id") or "")
            if not pid:
                continue
            if pid in values_by_id:
                values_by_id[pid]["value"]          = _f(p.get(val_key) or p.get("value"))
                values_by_id[pid]["redraft_value"]  = _f(p.get(rdft_key))
                values_by_id[pid]["rank_change_7d"] = p.get("rank_change_7d")
            else:
                pos = str(p.get("position") or "").upper()
                values_by_id[pid] = {
                    "name":           p.get("name", ""),
                    "position":       pos,
                    "team":           p.get("team", ""),
                    "age":            p.get("age"),
                    "value":          _f(p.get(val_key) or p.get("value")),
                    "redraft_value":  _f(p.get(rdft_key)),
                    "pos_rank_label": p.get("pos_rank_label") or "",
                    "rank_change_7d": p.get("rank_change_7d"),
                }
    except Exception as exc:
        log.debug("[archetype] DB value augment skipped: %s", exc)

    # ── Viewer roster ──────────────────────────────────────────────────────────
    viewer_players: List[str] = []
    for r in rosters:
        if str(r.get("roster_id")) == str(viewer_roster_id):
            viewer_players = [str(p) for p in (r.get("players") or [])]
            break

    viewer_lineup_val = _optimal_lineup_value(viewer_players, values_by_id, league_type)

    # League-average lineup value
    lineup_vals = [
        _optimal_lineup_value(
            [str(p) for p in (r.get("players") or [])], values_by_id, league_type
        )
        for r in rosters
    ]
    league_avg = sum(lineup_vals) / max(1, len(lineup_vals)) if lineup_vals else viewer_lineup_val

    viewer_seed      = _seed(standings_map, viewer_roster_id, num_teams)
    viewer_above     = viewer_seed <= playoff_spots
    preferred_arch   = COMPLEMENT.get(archetype, "balanced")
    scarcity         = SCARCITY_SF if league_type == "sf" else SCARCITY_1QB

    # ── Infer each rival's archetype & collect tradeable players ──────────────
    all_targets: List[Dict[str, Any]] = []
    targets_by_owner: Dict[str, List[Dict]] = {}
    owner_meta: Dict[str, Dict] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        if rid == str(viewer_roster_id):
            continue
        pids    = [str(p) for p in (r.get("players") or [])]
        seed    = _seed(standings_map, rid, num_teams)
        p_arch  = _infer_archetype(pids, values_by_id, seed, num_teams, playoff_spots)
        p_name  = _roster_name(roster_map, rid)
        is_pref = p_arch == preferred_arch

        # Partner stats for phrasing
        p_skill = [p for p in pids if values_by_id.get(p, {}).get("position") in SKILL_POS]
        p_ages  = [_f(values_by_id[p].get("age"),
                      PEAK_AGE.get(values_by_id[p].get("position", "WR"), 27)) for p in p_skill]
        p_dyn   = sum(_f(values_by_id[p].get("value")) for p in p_skill)
        p_rdft  = sum(_f(values_by_id[p].get("redraft_value")) for p in p_skill)
        p_avg_age   = (sum(p_ages) / len(p_ages)) if p_ages else None
        p_rdft_ratio = (p_rdft / p_dyn) if p_dyn > 0 else None
        p_pos_counts: Dict[str, int] = {}
        for p in p_skill:
            pos = values_by_id.get(p, {}).get("position", "")
            if pos:
                p_pos_counts[pos] = p_pos_counts.get(pos, 0) + 1
        p_weak = [pos for pos in ("RB", "WR", "TE", "QB")
                  if p_pos_counts.get(pos, 0) < 3][:2]
        p_phrase = _partner_phrase(p_arch, p_name, seed, playoff_spots, p_avg_age, p_rdft_ratio, p_weak)
        owner_meta[rid] = {"arch": p_arch, "name": p_name, "phrase": p_phrase}

        for pid in pids:
            info = values_by_id.get(pid)
            if not info:
                continue
            pos  = info.get("position", "")
            val  = _f(info.get("value"))
            rdft = _f(info.get("redraft_value"))
            age  = _f(info.get("age"), 0)
            if pos not in SKILL_POS or val <= 0:
                continue

            tgt = {
                "player_id":      pid,
                "name":           info.get("name", ""),
                "position":       pos,
                "team":           info.get("team", ""),
                "age":            age,
                "value":          val,
                "redraft_value":  rdft,
                "pos_rank_label": info.get("pos_rank_label", ""),
                "rank_change_7d": info.get("rank_change_7d"),
                "owner_roster_id": rid,
                "partner_name":   p_name,
                "partner_arch":   p_arch,
                "partner_phrase": p_phrase,
                "is_pref":        is_pref,
            }
            all_targets.append(tgt)
            targets_by_owner.setdefault(rid, []).append({
                "player_id": pid, "name": info.get("name", ""),
                "position": pos, "value": val,
            })

    # ── Distribute: viewer sends a stud for a depth package ───────────────────
    if archetype == "distribute":
        return _build_distribute(
            viewer_players, values_by_id, targets_by_owner, owner_meta,
            roster_map, league_type, viewer_lineup_val, league_avg,
            untouchable_ids=untouchable_ids,
        )

    # ── Rebuilding: viewer sells a win-now vet for a younger player ───────────
    if archetype == "rebuilding":
        return _build_rebuilding(
            viewer_players, values_by_id, all_targets,
            league_type, viewer_lineup_val, league_avg,
            untouchable_ids=untouchable_ids,
        )

    # ── 30-day trend ──────────────────────────────────────────────────────────
    all_pids = list({t["player_id"] for t in all_targets} | set(viewer_players))
    old_vals = _get_30d_values(all_pids)

    # ── Score & filter targets per archetype ──────────────────────────────────
    scored: List[Tuple[float, Dict]] = []
    peak_pos = PEAK_AGE

    for t in all_targets:
        pid  = t["player_id"]
        val  = t["value"]
        rdft = t["redraft_value"]
        age  = t["age"]
        pos  = t["position"]
        peak = peak_pos.get(pos, 27)
        tp   = _trend_pct(pid, val, old_vals, values_by_id)
        sc   = scarcity.get(pos, 0.9)
        pref_bonus = 1.15 if t["is_pref"] else 1.0

        score: Optional[float] = None

        if archetype == "contending":
            if val < 200:
                continue
            rdft_ratio = rdft / max(1, val) if rdft > 0 else 0.5
            age_sc     = max(0.0, 1.0 - abs(age - peak) / 6.0) if age else 0.5
            score = (
                0.45 * rdft_ratio +
                0.30 * age_sc +
                0.15 * min(1.0, val / 1000) +
                0.10 * (1.0 if tp >= 0 else 0.5)
            ) * sc * pref_bonus

        elif archetype == "consolidate":
            if val < 600:
                continue
            rdft_ratio = rdft / max(1, val) if rdft > 0 else 0.6
            score = (
                0.60 * min(1.0, val / 1200) +
                0.25 * rdft_ratio +
                0.15 * (1.0 if tp >= 0 else 0.7)
            ) * sc * pref_bonus

        if score is None:
            continue

        # Win-prob delta: assume replacing weakest player of same position
        same_pos_vals = [
            _f(values_by_id.get(p2, {}).get("value"))
            for p2 in viewer_players
            if values_by_id.get(p2, {}).get("position") == pos
        ]
        replace_val = min(same_pos_vals) if same_pos_vals else 0
        wpd = _wp_delta(viewer_lineup_val, val, replace_val, league_avg)

        t["win_prob_delta"] = wpd
        scored.append((score + wpd * 0.25, t))

    # ── Rank: one player per (team, position), best overall ───────────────────
    # Allows e.g. a team's stud RB *and* stud WR to both surface, but not two
    # RBs from the same team. Falls back to player-only dedup if results are thin.
    scored.sort(key=lambda x: x[0], reverse=True)
    seen_owner_pos: set = set()
    seen_players:   set = set()
    top: List[Dict] = []
    for _, t in scored:
        if t["player_id"] in seen_players:
            continue
        key = (t["owner_roster_id"], t["position"])
        if key in seen_owner_pos:
            continue
        seen_owner_pos.add(key)
        seen_players.add(t["player_id"])
        top.append(t)
        if len(top) >= 5:
            break

    # Relax (team, position) cap if not enough results
    if len(top) < 3:
        for _, t in scored:
            if t["player_id"] not in seen_players and len(top) < 5:
                seen_players.add(t["player_id"])
                top.append(t)

    # ── Build send packages & assemble response ───────────────────────────────
    send_candidates = _score_sends(viewer_players, values_by_id, archetype, untouchable_ids=untouchable_ids)
    # Add viewer's draft picks to the send pool so packages can use them as
    # value-fillers (e.g. two players + a pick for consolidate).
    viewer_picks = picks_by_roster.get(str(viewer_roster_id)) or \
                   picks_by_roster.get(viewer_roster_id) or []
    send_candidates += _pick_send_candidates(viewer_picks, num_teams)

    results = []
    for t in top:
        tp  = _trend_pct(t["player_id"], t["value"], old_vals, values_by_id)
        wpd = t.get("win_prob_delta", 0.0)
        why = _build_why(t, archetype, tp, wpd)
        pkg = _select_package(send_candidates, t["value"], archetype)

        results.append({
            "player_id":      t["player_id"],
            "name":           t["name"],
            "position":       t["position"],
            "nfl_team":       t["team"],
            "age":            t["age"],
            "value":          round(t["value"], 1),
            "redraft_value":  round(t.get("redraft_value", 0), 1),
            "pos_rank_label": t["pos_rank_label"],
            "why":            why,
            "partner_team":   t["partner_name"],
            "partner_arch":   t["partner_arch"],
            "win_prob_delta": round(wpd, 4),
            "direction":      "acquire",
            "suggested_send": pkg,
        })

    return results
