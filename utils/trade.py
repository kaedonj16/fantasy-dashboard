"""Consolidated utils module: trade.

trade math, values, tiers, roster strength, VORP

Merged from: utils/trade_targets.py, utils/trade_value.py, utils/trade_window.py, utils/tier_stack.py, utils/tier_thresholds.py, utils/player_tiers.py, utils/roster_strength.py, utils/vorp.py, utils/value_helpers.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/trade_targets.py
# ======================================================================

"""Roster-fit ranking for Trade Targets.

Need detection still uses starter-slot-weighted positional strength (same as
the Teams page). Candidate *selection* used to sort the other teams' players
by raw value, so every QB-needy roster saw the same elites. This module ranks
candidates by how well they fill THIS roster's gap at a price the viewer can
actually pay, prefers owners who need the viewer's surplus, and returns a
mixed list (not four elites per weak position).
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


POSITIONS = ("QB", "RB", "WR", "TE")
PEAK_AGE = {"QB": 29, "RB": 26, "WR": 27, "TE": 27}

# Bottom-of-league cutoff used by the API before this ranker existed.
NEED_RANK_FRACTION = 0.35

# A "quality" starter — bottom-ranked but already this good is not a shop list.
QUALITY_STARTER_MULT = 1.4

# Clearing this fraction of the starter bar counts as filling a hole, so a
# 350 QB can fill a 1QB hole (threshold 500) and a 400 QB can be a SF QB2.
_HOLE_FILL_FRAC = 0.50

# Don't let a stack of future 1sts make every elite look reachable.
_MAX_PICK_IN_CEILING = 870.0  # one 1st + one 2nd

# Mixed-list caps: never dump four elites at the same spot.
MAX_TARGETS = 8
MAX_PER_POS_HARD = 3
MAX_PER_POS_SOFT = 2

# Value floor for a "real" trade chip. Mirrors the API's collect filter.
_MIN_ASSET = 150.0


def one_for_one_chip(asset_values: Sequence[float]) -> float:
    """Typical 1-for-1 send: the 2nd-best real asset, so the cornerstone stays.

    A one-stud roster falls back to that stud. Empty / dart-throw rosters get
    a conservative placeholder so elites still look like a stretch.
    """
    vals = sorted((float(v or 0.0) for v in (asset_values or []) if float(v or 0.0) >= _MIN_ASSET),
                  reverse=True)
    if len(vals) >= 2:
        return vals[1]
    if vals:
        return vals[0]
    return 250.0


def package_ceiling(asset_values: Sequence[float], pick_value: float = 0.0) -> float:
    """Hard acquire cap: best 1-for-1 overpay, or top-2 + picks with a premium."""
    vals = sorted((float(v or 0.0) for v in (asset_values or []) if float(v or 0.0) >= _MIN_ASSET),
                  reverse=True)
    offer_1for1 = (vals[0] * 1.25) if vals else 300.0
    offer_package = ((sum(vals[:2]) + float(pick_value or 0.0)) * 1.2) if vals else 500.0
    return max(offer_1for1, offer_package)


def affordability_multiplier(value: float, one_for_one: float, package_max: float) -> float:
    """Peak when the target is a realistic 1-for-1; fade package-stretch elites.

    A 900-value QB against a 400-value chip scores ~0.12. The same QB against
    an 800-value chip (loaded roster) stays in the sweet spot.
    """
    value = float(value or 0.0)
    chip = max(80.0, float(one_for_one or 0.0))
    ceiling = max(chip, float(package_max or 0.0))
    if value <= 0:
        return 0.0
    if value > ceiling:
        return 0.10
    ratio = value / chip
    if 0.55 <= ratio <= 1.15:
        return 1.20
    if 0.35 <= ratio < 0.55:
        return 1.00
    if ratio < 0.35:
        return 0.70
    # Above a 1-for-1: fade from the chip toward the package ceiling.
    span = max(ceiling - chip * 1.15, 1.0)
    stretch = (value - chip * 1.15) / span
    return max(0.12, 0.55 * (1.0 - stretch))


def infer_roster_window(
    valued_ages: Sequence[Tuple[float, float]],
    is_redraft: bool = False,
) -> str:
    """rebuild | contend | balanced from the ages of the viewer's top assets."""
    if is_redraft:
        return "balanced"
    top = sorted(((float(v or 0.0), float(a)) for v, a in (valued_ages or [])
                  if a is not None and float(v or 0.0) > 0),
                 key=lambda x: x[0], reverse=True)[:8]
    ages = [a for _, a in top]
    if len(ages) < 3:
        return "balanced"
    avg = sum(ages) / len(ages)
    if avg <= 25.2:
        return "rebuild"
    if avg >= 27.8:
        return "contend"
    return "balanced"


def age_fit_multiplier(
    age: Optional[float],
    pos: str,
    window: str,
    is_redraft: bool = False,
) -> float:
    if is_redraft or not age or window == "balanced":
        return 1.0
    try:
        years = float(age)
    except (TypeError, ValueError):
        return 1.0
    peak = PEAK_AGE.get(str(pos or "").upper(), 27)
    if window == "rebuild":
        if years <= peak - 3:
            return 1.25
        if years <= peak:
            return 1.05
        if years <= peak + 2:
            return 0.70
        return 0.40
    # contend: prime-age (including young stars) over aging vets
    if years <= peak + 1:
        return 1.15
    if years <= peak + 3:
        return 1.00
    return 0.75


def availability_multiplier(depth_rank: int, pos_count: int) -> float:
    """How movable a rival's player is. Their #1 at a spot stays a keeper."""
    try:
        rank = int(depth_rank or 1)
    except (TypeError, ValueError):
        rank = 1
    try:
        count = int(pos_count or 1)
    except (TypeError, ValueError):
        count = 1
    if rank <= 1:
        return 0.75
    if count >= 4:
        return 1.25
    if count >= 3:
        return 1.10
    if count <= 1:
        return 0.85
    return 1.0


# Long enough to occupy every weight ``weighted_pos_strength`` will apply
# (RB/WR with 2+ flex uses 5). Padding missing slots as 0 means adding a
# QB2/TE2 is credited as filling an empty starter/depth slot instead of
# diluting a lone elite when the short list only used the first weight.
_STRENGTH_PAD = 6


def _padded_vals(vals: Sequence[float], extra: Sequence[float] = ()) -> List[float]:
    out = [float(v or 0.0) for v in (vals or [])]
    out.extend(float(v or 0.0) for v in extra)
    if len(out) < _STRENGTH_PAD:
        out.extend([0.0] * (_STRENGTH_PAD - len(out)))
    return out


def strength_gain(
    viewer_vals: Sequence[float],
    candidate_val: float,
    pos: str,
    slot_counts: Dict[str, int],
) -> float:
    """How much starter-slot-weighted strength this player adds.

    Missing starter/depth slots are scored as 0 on both sides so a QB2 added
    next to an elite QB1 is a real gain, not a dilution of the QB1-only average.
    """
    slots = slot_counts or {}
    before = weighted_pos_strength(_padded_vals(viewer_vals), pos, slots)
    after = weighted_pos_strength(_padded_vals(viewer_vals, (candidate_val,)), pos, slots)
    return max(0.0, after - before)


def classify_position_needs(
    pos_ranks: Dict[str, int],
    viewer_vals: Dict[str, Sequence[float]],
    num_teams: int,
    starter_thresholds: Dict[str, float],
    starter_floors: Dict[str, int],
) -> List[Tuple[str, str]]:
    """``(pos, 'hard'|'soft')`` the viewer should shop, worst gap first.

    Hard = missing a starter-caliber body. Soft = bottom of the league *and*
    the current best is still below a quality-starter bar. A 7th-place QB
    room that already has a 700 QB is not a shop-for-Allen list.
    """
    n = max(int(num_teams or 1), 1)
    cutoff = max(1, round(n * NEED_RANK_FRACTION))
    scored: List[Tuple[int, int, str, str]] = []
    for pos in POSITIONS:
        vals = [float(v or 0.0) for v in (viewer_vals or {}).get(pos, [])]
        threshold = float((starter_thresholds or {}).get(pos) or 0.0)
        floor = int((starter_floors or {}).get(pos) or 1)
        starters = sum(1 for v in vals if v >= threshold) if threshold else 0
        deficit = max(0, floor - starters)
        rank = int((pos_ranks or {}).get(pos) or n)
        is_bottom = rank > n - cutoff
        best = max(vals, default=0.0)
        quality_bar = threshold * QUALITY_STARTER_MULT if threshold else 0.0
        if deficit:
            scored.append((deficit, rank, pos, "hard"))
        elif is_bottom and (not quality_bar or best < quality_bar):
            scored.append((0, rank, pos, "soft"))
    scored.sort(key=lambda t: (-t[0], -t[1], POSITIONS.index(t[2])))
    return [(pos, kind) for _, _, pos, kind in scored]


def detect_needed_positions(
    pos_ranks: Dict[str, int],
    viewer_vals: Dict[str, Sequence[float]],
    num_teams: int,
    starter_thresholds: Dict[str, float],
    starter_floors: Dict[str, int],
) -> List[str]:
    """Positions the viewer should shop: starter hole, or thin + bottom 35%."""
    return [pos for pos, _ in classify_position_needs(
        pos_ranks, viewer_vals, num_teams, starter_thresholds, starter_floors,
    )]


def detect_surplus_positions(
    pos_ranks: Dict[str, int],
    viewer_vals: Dict[str, Sequence[float]],
    num_teams: int,
    starter_thresholds: Dict[str, float],
    starter_floors: Dict[str, int],
) -> List[str]:
    """Positions the viewer can actually deal from: extra starter-caliber bodies.

    Rank alone is not surplus — a 4th-place 1QB room still has only one QB.
    """
    surplus: List[str] = []
    for pos in POSITIONS:
        vals = [float(v or 0.0) for v in (viewer_vals or {}).get(pos, [])]
        threshold = float((starter_thresholds or {}).get(pos) or 0.0)
        floor = int((starter_floors or {}).get(pos) or 1)
        starters = sum(1 for v in vals if v >= threshold) if threshold else 0
        if starters - floor >= 1:
            surplus.append(pos)
    return surplus


def complementary_multiplier(
    owner_needs: Sequence[str],
    viewer_surplus: Sequence[str],
) -> Tuple[float, Optional[str]]:
    """Boost when the owner needs a position the viewer can send."""
    overlap = [p for p in POSITIONS if p in (owner_needs or []) and p in (viewer_surplus or [])]
    if overlap:
        return 1.18, overlap[0]
    return 1.0, None


def need_summary(needed: Sequence[Tuple[str, str]], window: str = "balanced") -> str:
    """One line for the UI: which hole this list is answering."""
    holes = [p for p, k in (needed or []) if k == "hard"]
    thin = [p for p, k in (needed or []) if k == "soft"]
    if not holes and not thin:
        return "No glaring gaps — upgrades that fit your roster"
    bits: List[str] = []
    if holes:
        if len(holes) == 1:
            bits.append(f"your {holes[0]} hole")
        else:
            bits.append("your " + " & ".join(holes) + " holes")
    if thin:
        bits.append("thin " + "/".join(thin))
    line = " and ".join(bits)
    if window == "rebuild":
        return f"Based on {line} · rebuild window"
    if window == "contend":
        return f"Based on {line} · win-now window"
    return f"Based on {line}"


def annotate_owner_depth(candidates: Iterable[Dict[str, Any]]) -> None:
    """Set depth_rank (1 = owner's best at the pos) and owner_pos_count in place."""
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in candidates or []:
        rid = str(row.get("owner_roster_id") or "")
        pos = str(row.get("position") or "").upper()
        by_key.setdefault((rid, pos), []).append(row)
    for rows in by_key.values():
        rows.sort(key=lambda r: float(r.get("value") or 0.0), reverse=True)
        n = len(rows)
        for i, row in enumerate(rows):
            row["depth_rank"] = i + 1
            row["owner_pos_count"] = n


def _fill_bar(threshold: float) -> float:
    return max(0.0, float(threshold or 0.0) * _HOLE_FILL_FRAC)


def fit_reason(
    *,
    pos: str,
    value: float,
    age: Optional[float],
    viewer_best: float,
    starter_count: int,
    floor: int,
    threshold: float,
    window: str,
    one_for_one: float,
    depth_rank: int,
    owner_pos_count: int,
    complementary_pos: Optional[str] = None,
) -> str:
    """One short line for the UI: why this player is on THIS roster's list."""
    pos = str(pos or "").upper()
    fill_bar = _fill_bar(threshold)
    if starter_count < floor and value >= fill_bar:
        return f"Fills your {pos} hole"
    if viewer_best > 0 and value >= viewer_best * 1.15:
        return f"Upgrades your {pos}1"
    if floor >= 2 and starter_count >= 1 and value >= fill_bar:
        return f"Adds {pos}{starter_count + 1} depth"
    if complementary_pos:
        return f"They need your {complementary_pos}s"
    if owner_pos_count >= 3 and depth_rank >= 2:
        return f"Their {pos} surplus"
    if window == "rebuild" and age and float(age) <= PEAK_AGE.get(pos, 27) - 2:
        return "Fits rebuild"
    if window == "contend" and age and float(age) >= PEAK_AGE.get(pos, 27) - 1:
        return "Win-now piece"
    if one_for_one and 0.55 <= (value / max(one_for_one, 1.0)) <= 1.15:
        return "Reachable upgrade"
    return f"Helps your {pos}s"


def _candidate_score(
    row: Dict[str, Any],
    *,
    viewer_vals: Sequence[float],
    pos: str,
    slot_counts: Dict[str, int],
    one_for_one: float,
    package_max: float,
    window: str,
    is_redraft: bool,
    starter_count: int,
    floor: int,
    threshold: float,
    viewer_surplus: Sequence[str] = (),
) -> Tuple[float, str]:
    value = float(row.get("value") or 0.0)
    age = row.get("age")
    viewer_best = max((float(v or 0.0) for v in (viewer_vals or [])), default=0.0)
    fill_bar = _fill_bar(threshold)
    gain = strength_gain(viewer_vals, value, pos, slot_counts)
    # Absolute gain fills a hole; efficiency stops 900-value elites from
    # always beating the mid-tier player who actually fits the budget.
    efficiency = gain / max(value, 1.0)
    raw = (0.55 * gain) + (0.45 * efficiency * 400.0)
    if starter_count < floor and value >= fill_bar:
        raw *= 1.25
    elif starter_count >= floor and value < fill_bar:
        raw *= 0.55
    elif starter_count >= floor and viewer_best > 0:
        # Already have a starter: prefer a real upgrade, fade 1.5x trophy hunts
        # so a 7th-place QB room with a 550 QB doesn't list Josh Allen.
        ratio = value / viewer_best
        if 1.10 <= ratio <= 1.50:
            raw *= 1.15
        elif ratio > 1.50:
            raw *= 0.62
        elif ratio < 1.0:
            raw *= 0.70
    afford = affordability_multiplier(value, one_for_one, package_max)
    avail = availability_multiplier(row.get("depth_rank") or 1, row.get("owner_pos_count") or 1)
    age_m = age_fit_multiplier(age, pos, window, is_redraft=is_redraft)
    comp, matched = complementary_multiplier(row.get("owner_needs") or [], viewer_surplus)
    score = raw * afford * avail * age_m * comp
    why = fit_reason(
        pos=pos, value=value, age=age,
        viewer_best=viewer_best,
        starter_count=starter_count, floor=floor, threshold=threshold,
        window=window, one_for_one=one_for_one,
        depth_rank=int(row.get("depth_rank") or 1),
        owner_pos_count=int(row.get("owner_pos_count") or 1),
        complementary_pos=matched,
    )
    return score, why


def rank_position_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    viewer_vals: Sequence[float],
    pos: str,
    slot_counts: Dict[str, int],
    one_for_one: float,
    package_max: float,
    window: str,
    is_redraft: bool,
    starter_threshold: float,
    starter_floor: int,
    limit: int,
    viewer_surplus: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Highest-fit players at one position, already annotated with why/score."""
    vals = [float(v or 0.0) for v in (viewer_vals or [])]
    threshold = float(starter_threshold or 0.0)
    floor = int(starter_floor or 1)
    starter_count = sum(1 for v in vals if v >= threshold) if threshold else 0
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for row in candidates or []:
        if float(row.get("value") or 0.0) > package_max:
            continue
        score, why = _candidate_score(
            row, viewer_vals=vals, pos=pos, slot_counts=slot_counts,
            one_for_one=one_for_one, package_max=package_max,
            window=window, is_redraft=is_redraft,
            starter_count=starter_count, floor=floor, threshold=threshold,
            viewer_surplus=viewer_surplus,
        )
        out = dict(row)
        out["why"] = why
        out["fit_score"] = round(score, 3)
        # Trade Hub: shared server-computed "Why this" line (owner_needs still
        # present here; _public() strips it later).
        try:
            from dashboard_services.trade_hub import why_line_for_target
            out["why_line"] = why_line_for_target(out)
        except Exception:
            pass
        ranked.append((score, out))
    ranked.sort(key=lambda t: (-t[0], -float(t[1].get("value") or 0.0)))
    return [row for _, row in ranked[: max(0, int(limit))]]


def select_trade_targets(
    *,
    viewer_vals: Dict[str, Sequence[float]],
    pos_ranks: Dict[str, int],
    num_teams: int,
    slot_counts: Dict[str, int],
    candidates_by_pos: Dict[str, Sequence[Dict[str, Any]]],
    viewer_asset_values: Sequence[float],
    pick_value: float = 0.0,
    valued_ages: Sequence[Tuple[float, float]] = (),
    starter_thresholds: Optional[Dict[str, float]] = None,
    starter_floors: Optional[Dict[str, int]] = None,
    is_redraft: bool = False,
    per_pos_limit: int = 4,
    balanced_per_pos: int = 2,
    owner_needs_by_roster: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Pick the targets the UI lists, grouped the same way the API always has.

    ``targets`` is the mixed, fit-ranked list (capped per position) so the
    page is not "top four at QB, top four at TE". ``by_position`` stays
    populated when the roster has a real need. Balanced rosters get
    ``all_positions`` — still fit-ranked, not top-by-value.
    """
    thresholds = dict(starter_thresholds or {})
    floors = dict(starter_floors or {})
    for pos in POSITIONS:
        thresholds.setdefault(pos, {"QB": 500, "RB": 350, "WR": 350, "TE": 200}[pos])
        floors.setdefault(pos, 1 if pos in ("QB", "TE") else 2)

    chip = one_for_one_chip(viewer_asset_values)
    ceiling = package_ceiling(
        viewer_asset_values, min(float(pick_value or 0.0), _MAX_PICK_IN_CEILING),
    )
    window = infer_roster_window(valued_ages, is_redraft=is_redraft)

    classified = classify_position_needs(
        pos_ranks, viewer_vals, num_teams, thresholds, floors,
    )
    needed = [pos for pos, _ in classified]
    severity = {pos: kind for pos, kind in classified}
    surplus = detect_surplus_positions(
        pos_ranks, viewer_vals, num_teams, thresholds, floors,
    )
    owner_needs = owner_needs_by_roster or {}

    # Depth ranks are a property of the owner's room, so annotate the full
    # candidate pool before we slice it.
    flat: List[Dict[str, Any]] = []
    for pos in POSITIONS:
        for row in candidates_by_pos.get(pos) or []:
            item = dict(row)
            item["position"] = str(item.get("position") or pos).upper()
            rid = str(item.get("owner_roster_id") or "")
            item["owner_needs"] = list(owner_needs.get(rid) or [])
            flat.append(item)
    annotate_owner_depth(flat)
    pooled: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POSITIONS}
    for row in flat:
        pos = row.get("position")
        if pos in pooled:
            pooled[pos].append(row)

    def _public(row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in row.items() if k != "owner_needs"}

    def _rank(pos: str, limit: int) -> List[Dict[str, Any]]:
        return rank_position_candidates(
            pooled.get(pos) or [],
            viewer_vals=viewer_vals.get(pos) or [],
            pos=pos,
            slot_counts=slot_counts,
            one_for_one=chip,
            package_max=ceiling,
            window=window,
            is_redraft=is_redraft,
            starter_threshold=thresholds[pos],
            starter_floor=floors[pos],
            limit=limit,
            viewer_surplus=surplus,
        )

    def _allocate(rows: Sequence[Dict[str, Any]], pos_caps: Dict[str, int]) -> List[Dict[str, Any]]:
        picked: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        # Once a position has a reachable hole-fill, skip the 1.5x trophy hunts
        # so Maye doesn't drag Burrow/Lamar along as "also QBs."
        filled_reachable: set[str] = set()
        stretch_cut = chip * 1.35
        reachable_cut = chip * 1.15
        ordered = sorted(
            rows,
            key=lambda r: (-float(r.get("fit_score") or 0.0), -float(r.get("value") or 0.0)),
        )
        for row in ordered:
            pos = str(row.get("position") or "")
            val = float(row.get("value") or 0.0)
            cap = pos_caps.get(pos, MAX_PER_POS_SOFT)
            if counts.get(pos, 0) >= cap:
                continue
            if pos in filled_reachable and val > stretch_cut:
                continue
            picked.append(_public(row))
            counts[pos] = counts.get(pos, 0) + 1
            why = str(row.get("why") or "")
            if why.startswith("Fills your") and val <= reachable_cut:
                filled_reachable.add(pos)
            if len(picked) >= MAX_TARGETS:
                break
        return picked

    if needed:
        pool_limit = max(int(per_pos_limit or 4) * 2, 8)
        pool: List[Dict[str, Any]] = []
        for pos in needed:
            pool.extend(_rank(pos, pool_limit))
        caps = {
            pos: MAX_PER_POS_HARD if severity.get(pos) == "hard" else MAX_PER_POS_SOFT
            for pos in needed
        }
        targets = _allocate(pool, caps)
        by_position: Dict[str, List[Dict[str, Any]]] = {p: [] for p in needed}
        for row in targets:
            pos = str(row.get("position") or "")
            if pos in by_position:
                by_position[pos].append(row)
        by_position = {pos: rows for pos, rows in by_position.items() if rows}
        return {
            "by_position": by_position,
            "all_positions": {},
            "targets": targets,
            "needed_positions": needed,
            "window": window,
            "summary": need_summary(classified, window),
            "surplus_positions": surplus,
        }

    all_positions = {pos: [_public(r) for r in rows]
                     for pos in POSITIONS if (rows := _rank(pos, balanced_per_pos))}
    balanced_pool = [r for rows in all_positions.values() for r in rows]
    targets = _allocate(balanced_pool, {p: balanced_per_pos for p in POSITIONS})
    return {
        "by_position": {},
        "all_positions": all_positions,
        "targets": targets,
        "needed_positions": [],
        "window": window,
        "summary": need_summary([], window),
        "surplus_positions": surplus,
    }


# ======================================================================
# From utils/trade_value.py
# ======================================================================

"""Trade-calculator player value math shared by the server eval and the JS preview.

``SCORING_MULTS`` and ``player_trade_value`` must stay in lockstep with
``SCORING_MULTS`` / ``getPlayerValue`` in static/app.js. tests/test_scoring_mult_parity.py
and tests/test_trade_value_parity.py fail if they drift.
"""

import math
from typing import Mapping

SUPPORTED_LEAGUE_SIZES = (8, 10, 12, 14)


def snap_league_size(n) -> int:
    """Nearest supported value-table size (8 / 10 / 12 / 14)."""
    try:
        size = int(n or 10)
    except (TypeError, ValueError):
        return 10
    if size in SUPPORTED_LEAGUE_SIZES:
        return size
    return min(SUPPORTED_LEAGUE_SIZES, key=lambda s: abs(s - size))


SCORING_MULTS = {
    "ppr": {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 1.00},
    "half": {"QB": 1.00, "RB": 1.06, "WR": 0.97, "TE": 0.94},
    "std": {"QB": 1.00, "RB": 1.13, "WR": 0.93, "TE": 0.87},
}


def player_trade_value(
    player: Mapping,
    *,
    league_type: str = "1qb",
    league_size: int = 10,
    scoring_format: str = "ppr",
    scoring_type: str = "dynasty",
    te_premium: float = 0.0,
) -> float:
    """Per-player value used by ``/api/trade-eval`` and the live trade preview."""
    fmt = (scoring_format or "ppr").strip().lower()
    scoring_mults = SCORING_MULTS.get(fmt, SCORING_MULTS["ppr"])
    lt = (league_type or "1qb").strip().lower()
    st = (scoring_type or "dynasty").strip().lower()
    size = snap_league_size(league_size)
    try:
        tep = float(te_premium or 0)
    except (TypeError, ValueError):
        tep = 0.0

    def _n(v) -> float:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0

    if st == "redraft":
        # Redraft is size-invariant. The 10-team base columns are the
        # FantasyCalc-ratio board the player modal shows. Size-bucketed
        # redraft_*_{8,12,14} columns are a WLS overlay and can invert
        # Superflex (elite QBs priced like 1QB). League-size controls are
        # disabled in redraft for the same reason. Ranked surfaces
        # (My Leagues / Teams) pin league_size=10 so they match the modal.
        if lt == "sf":
            val = _n(player.get("redraft_value_sf") or player.get("redraft_value_1qb"))
        else:
            val = _n(player.get("redraft_value_1qb"))
    elif lt == "sf":
        size_key = "sf_value" if size == 10 else f"sf_value_{size}"
        val = _n(player.get(size_key) or player.get("sf_value") or player.get("value"))
    else:
        size_key = "value" if size == 10 else f"value_{size}"
        val = _n(player.get(size_key) or player.get("value"))

    pos = str(player.get("position") or "").upper()
    mult = scoring_mults.get(pos, 1.0)
    if tep and pos == "TE":
        mult *= (1 + tep * 0.20)
    return math.floor(val * mult * 10 + 0.5) / 10


def fair_value_band(baseline: float, floor: float = 25.0) -> float:
    """Shared "fair trade" band: the max value delta still called fair.

    Continuous in the baseline (the larger side's total value): a flat 7%
    with a ``floor`` minimum. This replaced the old tiered 5%/7%/10% bands,
    which had a discontinuity at the 600 threshold (band 41.9 at baseline
    599, dropping to 30.0 at 600) so near-identical trades flipped verdicts.

    Used by the trade calculator (api_trade_eval) and the trade outcome
    analyzer (api_trade_outcome) so both surfaces apply the same definition
    of fair and can't give contradictory verdicts.
    """
    try:
        b = max(float(baseline or 0.0), 1.0)
    except (TypeError, ValueError):
        b = 1.0
    try:
        f = max(float(floor or 0.0), 0.0)
    except (TypeError, ValueError):
        f = 25.0
    return max(b * 0.07, f)


def fairness_label(net_delta: float, baseline: Optional[float] = None) -> str:
    """Classify a trade's net value delta using the shared fair band.

    ``baseline`` is the larger side's total value; when omitted the band
    falls back to the 25.0 floor so the label still works for delta-only
    callers.
    """
    try:
        delta = float(net_delta or 0)
    except (TypeError, ValueError):
        delta = 0.0
    band = fair_value_band(baseline if baseline is not None else 0.0)
    if delta > band:
        return "strong_win"
    if delta < -band:
        return "strong_loss"
    return "fair"


# ======================================================================
# From utils/trade_window.py
# ======================================================================

"""Buy/sell trade-window classification.

Pure logic for the Season Hub advisor: given a team's playoff odds, roster-age
standing, and the weeks remaining before the league trade deadline, decide
whether the team should be buying (contender consolidating), selling
(rebuilder cashing vets), or holding, and pick the trade partners on the
opposite side of the market.
"""

# Playoff-odds thresholds for the verdict. Between them is a genuine coin
# flip where pushing someone to buy or sell would be false confidence.
BUY_THRESHOLD = 65.0
SELL_THRESHOLD = 35.0

# A deadline this close makes the verdict urgent.
URGENT_WEEKS = 3

# Redraft Season Hub cards are labeled "Trade deadline: …". Only show them
# once the deadline is known and within this many weeks — otherwise Week 1
# leagues (especially ESPN, which historically lacked trade_deadline) get a
# misleading mid-season notif.
REDRAFT_DEADLINE_WINDOW = 4


def redraft_deadline_card_visible(weeks_to_deadline: Optional[int]) -> bool:
    """True when a redraft league should paint the trade-deadline action card."""
    if weeks_to_deadline is None:
        return False
    try:
        weeks = int(weeks_to_deadline)
    except (TypeError, ValueError):
        return False
    return 0 <= weeks <= REDRAFT_DEADLINE_WINDOW


def deadline_line_visible(weeks_to_deadline: Optional[int]) -> bool:
    """True when the "deadline in N weeks" context line is worth showing.

    Dynasty paints the trade-window advisor year-round, so far-off deadlines
    would make a standing buy/sell card read like a countdown alert. Gate the
    deadline line to the same near-deadline window redraft uses for its card.
    """
    return redraft_deadline_card_visible(weeks_to_deadline)


def trade_window_verdict(
    playoff_pct: float,
    weeks_to_deadline: Optional[int] = None,
    age_rank: Optional[int] = None,
    n_teams: Optional[int] = None,
) -> dict:
    """Classify a team's trade posture.

    Args:
        playoff_pct: 0-100 playoff probability.
        weeks_to_deadline: whole weeks until the trade deadline; None when the
            league has no usable deadline.
        age_rank: 1 = oldest core in the league (optional flavor signal).
        n_teams: league size, required for age_rank to mean anything.

    Returns {"verdict": "buy"|"sell"|"hold", "urgent": bool, "modifier": str}.
    modifier is "" or a refinement: "all_in" (buying with an old core, the
    window is now), "youth" (selling with a young core, rebuild is on
    schedule), "aging_bubble" (holding with an old core on the playoff bubble,
    the riskiest place to sit).
    """
    pct = float(playoff_pct or 0.0)
    if pct >= BUY_THRESHOLD:
        verdict = "buy"
    elif pct <= SELL_THRESHOLD:
        verdict = "sell"
    else:
        verdict = "hold"

    urgent = weeks_to_deadline is not None and 0 <= int(weeks_to_deadline) <= URGENT_WEEKS

    modifier = ""
    if age_rank and n_teams and n_teams >= 4:
        old_third = age_rank <= max(1, round(n_teams / 3))
        young_third = age_rank > n_teams - max(1, round(n_teams / 3))
        if verdict == "buy" and old_third:
            modifier = "all_in"
        elif verdict == "sell" and young_third:
            modifier = "youth"
        elif verdict == "hold" and old_third:
            modifier = "aging_bubble"

    return {"verdict": verdict, "urgent": urgent, "modifier": modifier}


def trade_partners(teams: List[dict], verdict: str, limit: int = 3) -> List[str]:
    """Names of the best trade partners on the opposite side of the market.

    teams: [{"name", "playoff_pct", "is_viewer"}]. Buyers should call the
    clearest sellers (lowest playoff odds) and vice versa; holders get no
    partner list. The viewer is always excluded (by flag and by name, so a
    mis-tagged roster_id cannot list your own team as a seller to call).
    """
    if verdict not in ("buy", "sell"):
        return []
    viewer_names = {
        str(t.get("name") or "").strip().lower()
        for t in (teams or [])
        if t.get("is_viewer") and t.get("name")
    }
    pool = []
    for t in teams or []:
        if t.get("is_viewer"):
            continue
        name = t.get("name")
        if not name:
            continue
        if str(name).strip().lower() in viewer_names:
            continue
        pool.append(t)
    if verdict == "buy":
        pool = [t for t in pool if float(t.get("playoff_pct") or 0) <= SELL_THRESHOLD]
        pool.sort(key=lambda t: float(t.get("playoff_pct") or 0))
    else:
        pool = [t for t in pool if float(t.get("playoff_pct") or 0) >= BUY_THRESHOLD]
        pool.sort(key=lambda t: -float(t.get("playoff_pct") or 0))
    return [str(t["name"]) for t in pool[: max(0, int(limit))]]


# ======================================================================
# From utils/tier_stack.py
# ======================================================================

"""Pure asset-tier classification and multi-for-one trade adjustment.

Extracted from app.py so this logic can be unit-tested without the pandas/DB
stack.

``asset_tier`` maps a fantasy value to a tier number (T1 elite ... TN) given a
set of thresholds. ``apply_tier_stack_adjustment`` applies the depth penalty for
the side of an unequal trade that gives up more players: the extra players are
discounted toward true waiver-wire value, mutating the passed-in side dicts in
place with tier/effective-value annotations.
"""


# Number of tiers the classifier caps out at (T1 elite ... T9 catch-all).
NUM_TIERS = 9


def build_tier_caps(num_tiers: int) -> dict:
    """Per-tier value-retention caps, linearly interpolated from 1.0 (T1)
    down to 0.38 (bottom tier). Returns {tier: cap}."""
    high, low = 1.0, 0.38
    if num_tiers <= 1:
        return {1: 1.0}
    return {t: round(high - (high - low) * (t - 1) / (num_tiers - 1), 3)
            for t in range(1, num_tiers + 1)}


def asset_tier(value: float, thresholds: list = None) -> int:
    t = thresholds if thresholds is not None else FALLBACK_THRESHOLDS
    for i, threshold in enumerate(t):
        if value >= threshold:
            return min(i + 1, NUM_TIERS)
    return NUM_TIERS  # catch-all T9


def apply_tier_stack_adjustment(side_a: dict, side_b: dict,
                                 tier_thresholds: list = None,
                                 is_sf: bool = False,
                                 value_table: list = None,
                                 league_size: int = 10) -> None:
    """
    Depth penalty for the bigger side in unequal trades.

    The side giving up more players is discounted by a fraction of the value of
    true waiver-wire players (ranked well below the roster cutoff).  The first
    reference player sits at roughly rank (league_size × 38) and each successive
    extra player steps deeper; 50% of that value is applied as the penalty.

    Bigger side: effective_total = raw_total - sum(bench_values).
    Smaller side: effective_total = raw_total (no adjustment).
    """
    a_count = len(side_a.get("breakdown") or []) or len(side_a.get("player_values") or [])
    b_count = len(side_b.get("breakdown") or []) or len(side_b.get("player_values") or [])

    thresholds = tier_thresholds if tier_thresholds is not None else FALLBACK_THRESHOLDS

    # Annotate tiers first (informational, no multiplier applied)
    for side in (side_a, side_b):
        for item in (side.get("breakdown") or []):
            val = item.get("value", 0.0)
            item["tier"]            = asset_tier(val, thresholds)
            item["stack_mult"]      = 1.0
            item["effective_value"] = round(val, 1)

    if a_count == b_count:
        return

    delta   = abs(a_count - b_count)
    smaller = side_a if a_count < b_count else side_b
    bigger  = side_b if a_count < b_count else side_a

    # Build sorted value list for bench-rank lookup
    sorted_vals: list = []
    if value_table:
        sorted_vals = sorted(
            [float(p.get("value") or 0) for p in value_table
             if isinstance(p, dict) and float(p.get("value") or 0) > 10],
            reverse=True,
        )

    # Rank well below the roster cutoff so the penalty reflects true waiver-wire value
    base_rank      = league_size * 38   # ~380 for 10-team (below 27-spot roster cutoff)
    _BENCH_BASE    = 80.0               # fallback when value_table unavailable
    _BENCH_STEP    = -5.0               # each extra player is worth slightly less
    _PENALTY_FRAC  = 0.5                # apply 50% of waiver-wire value as the penalty

    def _bench_value(i: int) -> float:
        rank = min(len(sorted_vals) if sorted_vals else 9999, base_rank + i * 10)
        if sorted_vals:
            idx = min(rank - 1, len(sorted_vals) - 1)
            return sorted_vals[idx] * _PENALTY_FRAC
        return max(20.0, (_BENCH_BASE + i * _BENCH_STEP) * _PENALTY_FRAC)

    bench_total = sum(_bench_value(i) for i in range(delta))

    bigger["effective_total"]  = float(bigger.get("raw_total") or 0.0) - bench_total
    bigger["adjustment"]       = -bench_total
    smaller["effective_total"] = float(smaller.get("raw_total") or 0.0)
    smaller["adjustment"]      = 0.0


# ======================================================================
# From utils/tier_thresholds.py
# ======================================================================

"""Pure tier-threshold computation.

Extracted from app.py so the drop-based tiering logic can be unit-tested without
importing the full application (pandas / DB) stack.

Given a table of player value dicts, ``compute_tier_thresholds`` returns the
value boundaries between tiers (T1 elite ... Tn). Boundaries are placed at
natural value drops scored by *local* significance, subject to two hard rules
(max span per tier, min players per tier). See the function docstring for the
full description of the algorithm.
"""

# Fallback thresholds used when the value distribution is too small or too
# degenerate to derive meaningful tiers from.
FALLBACK_THRESHOLDS = [850.0, 700.0, 550.0, 420.0, 300.0, 200.0, 120.0, 60.0]

# Per-position value-rank cutoffs that define an "elite" player, matching the
# ELITE chip (api_player_indicators). A player ranked at/above the cutoff by
# positional value is elite. Single source of truth so the chip and the
# consolidate/distribute engine can't drift apart.
ELITE_RANK_CUTOFFS = {"QB": 5, "RB": 6, "WR": 6, "TE": 5}

# Tier display caps out here (T1 elite ... T9).
MAX_DISPLAY_TIERS = 9


def compute_tier_thresholds(value_table, league_type: str = "1qb", league_size: int = 10,
                            num_tiers: int = MAX_DISPLAY_TIERS, t1_size: int = None) -> list:
    """
    Drop-based tier boundaries with two hard constraints, relative to fantasy value.

    Boundaries are placed at natural value drops, scored by *local* significance
    (a gap measured against the median of nearby gaps) so a real cliff registers
    whether it sits among the sparse elites or the dense mid/low range. Two hard
    rules are enforced:

      1. MAX_SPAN: no tier may span more than ~220 value. Span splits are
         mandatory and take priority over discretionary drop boundaries, so an
         otherwise-flat region (e.g. a wall of similarly-valued QBs in SF) still
         gets broken up.
      2. MIN_SIZE: no tier smaller than 5 players, except the elite T1 which may
         be as small as 3 - so there are never tiny tiers outside the top.

    At most num_tiers (default 12) tiers are produced. Tiers naturally widen
    toward the bottom because low values are densely packed.
    """
    if league_type == "sf":
        primary = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
    else:
        primary = "value" if league_size == 10 else f"value_{league_size}"

    vals = []
    for p in (value_table or []):
        if not isinstance(p, dict):
            continue
        pos = (p.get("position") or "").upper()
        if pos in ("K", "DEF", "PICK"):
            continue
        v = float(p.get(primary) or p.get("value") or 0)
        if v >= 5:
            vals.append(v)

    vals.sort(reverse=True)
    n = len(vals)
    if n < num_tiers * 3:
        return FALLBACK_THRESHOLDS

    MIN_SIZE   = 5      # minimum players per tier (non-elite)
    ELITE_MIN  = 3      # T1 may be smaller (elite cluster)
    MAX_SPAN   = 220.0  # no tier spans more than this in value
    WINDOW     = 10     # neighborhood for local-significance scoring
    SIG_MIN    = 2.0    # a gap must be >= 2x the local median to count as a drop

    # Local significance of each gap: gap size vs the median of nearby gaps.
    score = [0.0] * (n - 1)
    for i in range(n - 1):
        gap = vals[i] - vals[i + 1]
        lo = max(0, i - WINDOW)
        hi = min(n - 1, i + WINDOW)
        nbrs = sorted(vals[j] - vals[j + 1] for j in range(lo, hi) if j != i)
        med = nbrs[len(nbrs) // 2] if nbrs else 1.0
        score[i] = gap / max(med, 0.5)

    bounds: list = []   # boundary index i = split between player i and i+1

    def _segment(i):
        lower = max([b for b in bounds if b < i], default=-1)
        upper = min([b for b in bounds if b > i], default=n - 1)
        return lower, upper

    def _valid(i):
        lower, upper = _segment(i)
        top = i - lower
        bot = upper - i
        tmin = ELITE_MIN if lower == -1 else MIN_SIZE
        return top >= tmin and bot >= MIN_SIZE

    while len(bounds) < num_tiers - 1:
        # 1) Mandatory: split the worst over-span segment at its biggest gap.
        prev = -1
        worst = None
        worst_span = MAX_SPAN
        for b in sorted(bounds) + [n - 1]:
            lo, hi = prev + 1, b
            prev = b
            sp = vals[lo] - vals[hi]
            if sp > worst_span:
                worst_span = sp
                worst = (lo, hi)
        if worst is not None:
            lo, hi = worst
            best_i, best_g = None, -1.0
            for j in range(lo + MIN_SIZE - 1, hi - MIN_SIZE + 1):
                g = vals[j] - vals[j + 1]
                if g > best_g:
                    best_g = g
                    best_i = j
            if best_i is not None and _valid(best_i):
                bounds.append(best_i)
                continue

        # 2) Discretionary: the most locally-significant remaining drop.
        cand = [(score[i], i) for i in range(n - 1)
                if i not in bounds and score[i] >= SIG_MIN and _valid(i)]
        if not cand:
            break
        cand.sort(reverse=True)
        bounds.append(cand[0][1])

    thresholds = sorted(
        [round((vals[b] + vals[b + 1]) / 2.0, 1) for b in sorted(bounds)],
        reverse=True,
    )
    return thresholds if len(thresholds) >= 2 else FALLBACK_THRESHOLDS


# ======================================================================
# From utils/player_tiers.py
# ======================================================================

"""Shared player-tier classification for trade suggestions.

ONE model, used by every surface that reasons about "elite / pure starter /
flex" so they can never disagree:

  - ELITE   : top-N at a position by *value rank*, matching the ELITE chip
              (api_player_indicators / ELITE_RANK_CUTOFFS). Rank-based on purpose:
              "elite" means the handful of best players at the position, not an
              absolute value bar.
  - STARTER : startable-caliber by *value*, matching the depth-warning thresholds
              (utils.roster_strength), scaled by league size. Value-based on
              purpose: "startable" is an absolute production floor, not a
              headcount.
  - FLEX    : rostered but below the starter value bar.
  - DEPTH   : unknown / no value.

The two metrics are intentional and internally consistent: an elite player is
always above the starter value bar, so checking rank (elite) before value
(starter) never contradicts itself.

Keeping this in one place is what lets the proactive Suggestions tab
(build_trade_suggestions_context) and the archetype engine
(get_archetype_suggestions) apply the *same* roster-aware ceiling.
"""


from utils.core import safe_float as _f

SKILL_POS = {"QB", "RB", "WR", "TE"}


def positional_ranks(values_by_id: Dict[str, Any]) -> Dict[str, int]:
    """pid -> value rank within its own position (1 = highest-value at that
    position) across the whole value table. Feeds pos_category."""
    pool: Dict[str, list] = {}
    for pid, v in values_by_id.items():
        p = str((v or {}).get("position") or "").upper()
        if p in SKILL_POS:
            pool.setdefault(p, []).append((str(pid), _f((v or {}).get("value"))))
    ranks: Dict[str, int] = {}
    for lst in pool.values():
        for rk, (pid, _val) in enumerate(sorted(lst, key=lambda x: -x[1]), start=1):
            ranks[pid] = rk
    return ranks


def pos_category(
    pos: str,
    rank: Optional[int],
    value: Optional[float],
    starter_threshold: Dict[str, float],
) -> str:
    """'elite' | 'starter' | 'flex' | 'depth' for a player at a position. Elite is
    the ELITE chip's per-position rank cutoff; starter is the league's
    starter-caliber value bar; below that is flex; unknown/valueless is depth."""
    if not rank:
        return "depth"
    if rank <= ELITE_RANK_CUTOFFS.get(pos, 3):
        return "elite"
    if _f(value) >= starter_threshold.get(pos, STARTER_THRESHOLD.get(pos, 350)):
        return "starter"
    return "flex"


def roster_position_counts(
    player_values,
    starter_threshold: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Given an iterable of (position, value) for a roster, count per position how
    many players are on the roster (`total`) and how many clear the starter-caliber
    value bar (`starters`). The building block of the starter-gap need model."""
    counts: Dict[str, Dict[str, int]] = {p: {"total": 0, "starters": 0} for p in SKILL_POS}
    for pos, value in player_values:
        p = str(pos or "").upper()
        if p in counts:
            counts[p]["total"] += 1
            if _f(value) >= starter_threshold.get(p, STARTER_THRESHOLD.get(p, 350)):
                counts[p]["starters"] += 1
    return counts


def starter_gap_needs(counts: Dict[str, Dict[str, int]], depth_floor: Dict[str, int]) -> list:
    """Positions where you can't field your starters: fewer startable players than
    the league's required starting slots. This is a real lineup hole, unlike a
    low positional value *total* (which flex depth can inflate). Ordered by the
    size of the gap so the most urgent hole comes first (a marginal-need proxy)."""
    gaps = []
    for p in SKILL_POS:
        gap = depth_floor.get(p, 1) - counts.get(p, {}).get("starters", 0)
        if gap > 0:
            gaps.append((p, gap))
    gaps.sort(key=lambda x: -x[1])
    return [p for p, _ in gaps]


def startable_surplus(counts: Dict[str, Dict[str, int]], depth_floor: Dict[str, int]) -> list:
    """Positions where you roster more startable players than starting slots, so
    you can trade a starter away and still field your lineup (real strength)."""
    return [p for p in SKILL_POS
            if counts.get(p, {}).get("starters", 0) > depth_floor.get(p, 1)]


def ceiling_needs(
    counts: Dict[str, Dict[str, int]],
    best_cat_by_pos: Dict[str, str],
    depth_floor: Dict[str, int],
) -> list:
    """Positions where you field your starters (no hole) but your best player is
    not elite - a 'ceiling gap' rather than a roster hole. Contenders chasing a
    difference-maker want these; rebuilders don't. `best_cat_by_pos` maps a
    position to the category of the viewer's best player there (from pos_category)."""
    out = []
    for p in SKILL_POS:
        if counts.get(p, {}).get("starters", 0) >= depth_floor.get(p, 1):
            if best_cat_by_pos.get(p) not in ("elite",):
                out.append(p)
    return out


def consolidate_target_allowed(target_cat: str, viewer_best_cat: str) -> bool:
    """Whether a consolidation should aim at a target of `target_cat` given the
    viewer's best existing player at that position.

    Deliberate, position-internal product rule: you only reach for an ELITE
    (top-of-position) when you already roster a pure starter (or elite) there. A
    team with only flex-worthy depth at the position is steered to a pure starter
    instead of an unrealistic wall-of-depth-for-a-superstar reach. Applied
    identically on every suggestion surface.
    """
    if target_cat == "elite" and viewer_best_cat not in ("starter", "elite"):
        return False
    return True


# ======================================================================
# From utils/roster_strength.py
# ======================================================================

"""Pure positional roster-strength scoring.

Extracted from app.py so the weighting logic can be unit-tested without the
pandas/DB stack.

``weighted_pos_strength`` collapses a list of player values at one position into
a single strength number that emphasizes top-end talent over pure depth, so a
handful of mid-tier players never outscores two elite starters. The weights
adapt to how many FLEX slots the league runs (more flex -> more depth credit).
"""


from utils.lineups import FLEX_SLOT_NAMES as _FLEX_SLOT_NAMES, RB_TE_SLOT_NAMES as _RB_TE_SLOT_NAMES, RB_WR_SLOT_NAMES as _RB_WR_SLOT_NAMES, SKILL_POSITIONS, SUPERFLEX_SLOT_NAMES as _SUPERFLEX_SLOT_NAMES, WR_TE_SLOT_NAMES as _WR_TE_SLOT_NAMES, canonicalize_slot, slot_total as _slot_total
from utils.projections import confidence_from_inputs


def weighted_pos_strength(vals: List[float], pos: str, slot_counts: Dict[str, int]) -> float:
    """
    Emphasize top-end talent over pure depth.

    Examples:
      - QB: mostly QB1, tiny credit for QB2
      - RB/WR: strong weight on top 2, smaller weight on next few
      - TE: mostly TE1, tiny credit for TE2

    This prevents 5 mid players from outscoring 2 elite starters.
    """
    if not vals:
        return 0.0

    vals = sorted((float(v or 0.0) for v in vals), reverse=True)

    # Sleeper, ESPN, and Yahoo do not use one canonical name for FLEX/SF. A
    # strength formula must read all of them or the same roster is graded
    # differently depending on which platform supplied its settings.
    flex_slots = _slot_total(slot_counts, _FLEX_SLOT_NAMES)
    superflex_slots = _slot_total(slot_counts, _SUPERFLEX_SLOT_NAMES)
    rb_wr_slots = _slot_total(slot_counts, _RB_WR_SLOT_NAMES | {"RB_WR"})
    wr_te_slots = _slot_total(slot_counts, _WR_TE_SLOT_NAMES | {"WR_TE"})
    rb_te_slots = _slot_total(slot_counts, _RB_TE_SLOT_NAMES | {"RB_TE"})

    if pos == "QB":
        qb_starters = max(1, int(slot_counts.get("QB") or 0) + superflex_slots)
        # In 1QB, QB2 is bench insurance. In superflex/2QB, QB2 is a weekly
        # starter and must carry nearly the same weight as QB1; the old fixed
        # 0.20 weight materially overrated teams with one elite QB and no QB2.
        weights = [1.0] + [0.90] * (qb_starters - 1) + [0.20]

    elif pos == "RB":
        # RB1/RB2 matter most, then some flex/depth credit. Restricted
        # RB/WR and RB/TE spots count; WR/TE-only does not.
        rb_flex = flex_slots + rb_wr_slots + rb_te_slots
        if rb_flex >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif rb_flex == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "WR":
        wr_flex = flex_slots + rb_wr_slots + wr_te_slots
        if wr_flex >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif wr_flex == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "TE":
        # TE premium on starter, little on TE2 unless you want more.
        # WR/RB-only flex does not make a TE2 startable.
        te_flex = flex_slots + wr_te_slots + rb_te_slots
        if te_flex >= 1:
            weights = [1.0, 0.20, 0.08]
        else:
            weights = [1.0, 0.15]

    else:
        weights = [1.0]

    used = vals[:len(weights)]
    denom = sum(weights[:len(used)]) or 1.0
    return sum(v * w for v, w in zip(used, weights)) / denom


CORE_POSITIONS = ("QB", "RB", "WR", "TE")


def _player_position(
    pid: str,
    values_by_id: Mapping[str, Mapping],
    players_index: Optional[Mapping] = None,
) -> str:
    row = values_by_id.get(pid) or {}
    pos = str(row.get("position") or row.get("pos") or "").upper()
    if pos:
        return pos
    meta = (players_index or {}).get(pid) or {}
    return str(meta.get("pos") or meta.get("position") or "").upper()


def roster_pos_value_lists(
    rosters: Sequence[Mapping],
    values_by_id: Mapping[str, Mapping],
    *,
    positions: Sequence[str] = CORE_POSITIONS,
    players_index: Optional[Mapping] = None,
    rid_cast=None,
) -> Dict[Any, Dict[str, List[float]]]:
    """Per-roster lists of positive player values, bucketed by skill position.

    Players missing from the value table, or with value <= 0, are omitted so
    they do not occupy a starter-weight slot. ``rid_cast`` defaults to identity
    (the Teams page keeps the provider roster_id type); pass ``str`` for
    My Leagues so lookups match ``str(viewer_roster_id)``.
    """
    wanted = {str(p).upper() for p in positions}
    out: Dict[Any, Dict[str, List[float]]] = {}
    for roster in rosters or []:
        rid = roster.get("roster_id")
        if rid is None:
            continue
        key = rid_cast(rid) if rid_cast else rid
        buckets: Dict[str, List[float]] = {pos: [] for pos in wanted}
        for pid in roster.get("players") or []:
            spid = str(pid)
            row = values_by_id.get(spid)
            if not row:
                continue
            pos = _player_position(spid, values_by_id, players_index)
            if pos not in wanted:
                continue
            try:
                val = float(row.get("value") or 0.0)
            except (TypeError, ValueError):
                val = 0.0
            if val <= 0:
                continue
            buckets[pos].append(val)
        out[key] = buckets
    return out


def rank_rosters_by_position(
    team_pos_values: Mapping[Any, Mapping[str, Sequence[float]]],
    slot_counts: Mapping[str, int],
    *,
    positions: Sequence[str] = CORE_POSITIONS,
) -> tuple[Dict[Any, Dict[str, float]], Dict[str, Dict[Any, int]]]:
    """Rank every roster at each skill position with ``weighted_pos_strength``.

    This is the single ranking used by My Leagues and the Teams page. The
    Teams detail strip may still show a starter/depth/fragility profile, but
    the visible ``#N`` place is this order — 1 = strongest.

    Returns ``(strengths, ranks)``:
      strengths[rid][pos] = float
      ranks[pos][rid] = 1-based rank

    Ties break by ``str(rid)`` so two pages cannot assign adjacent places to
    the same pair in opposite orders.
    """
    slots = dict(slot_counts or {})
    strengths: Dict[Any, Dict[str, float]] = {}
    for rid, pos_map in (team_pos_values or {}).items():
        strengths[rid] = {
            pos: weighted_pos_strength(list((pos_map or {}).get(pos) or []), pos, slots)
            for pos in positions
        }
    ranks: Dict[str, Dict[Any, int]] = {}
    for pos in positions:
        ordered = sorted(
            strengths.keys(),
            key=lambda rid: (-float(strengths[rid].get(pos) or 0.0), str(rid)),
        )
        ranks[pos] = {rid: i + 1 for i, rid in enumerate(ordered)}
    return strengths, ranks


def strength_percentile(user_strength: float, all_strengths: Sequence[float]) -> float:
    """0-100 percentile of ``user_strength`` among ``all_strengths``.

    100 means strictly best (better than every other roster), 0 means strictly
    worst. Ties split the difference with the other tied rosters, so two teams
    tied for first in a 12-team league land around the 95th percentile rather
    than both claiming 100. A one-team (or empty) league returns 50 — there is
    no field to outrank.

    Averaging this across leagues is what the My Leagues positional-strength
    card shows: a #1 finish in one league and a #12 in another read as ~50th,
    instead of a signed percent-vs-median that the weak league drags negative.
    """
    strengths = [float(s or 0.0) for s in (all_strengths or [])]
    n = len(strengths)
    if n <= 1:
        return 50.0
    user = float(user_strength or 0.0)
    n_worse = sum(1 for s in strengths if s < user)
    n_equal = sum(1 for s in strengths if s == user)
    others = n - 1
    if n_equal <= 0:
        # Caller compared against a field that does not include the user.
        return 100.0 * n_worse / n
    tied_others = n_equal - 1
    return 100.0 * (n_worse + 0.5 * tied_others) / others


def average_league_percentiles(
    league_pctiles: Iterable[Mapping[str, float]],
    positions: Sequence[str] = CORE_POSITIONS,
) -> Dict[str, float]:
    """Mean in-league percentile per position, rounded to one decimal.

    Missing positions are skipped, not treated as zero — a league that could
    not score TE must not pull the cross-league TE number to 0th. An empty
    input yields 50 for every requested position.
    """
    buckets: Dict[str, List[float]] = {pos: [] for pos in positions}
    for row in league_pctiles or []:
        for pos in positions:
            if pos not in row:
                continue
            raw = row.get(pos)
            if raw is None:
                continue
            buckets[pos].append(float(raw))
    return {
        pos: round((sum(vals) / len(vals)) if vals else 50.0, 1)
        for pos, vals in buckets.items()
    }


# Holdout-oriented composite: starter utility is the primary outcome, while
# depth and resilience remain independently visible instead of being hidden in
# one hand-tuned average. Keep these centralized for future backtest promotion.
ROSTER_COMPONENT_WEIGHTS = {"starter": 0.62, "depth": 0.23, "resilience": 0.15}


def fit_roster_component_weights(samples: list, step: float = 0.05) -> dict:
    """Fit non-negative component weights against realized roster outcomes.

    ``samples`` contain starter/depth/resilience (all normalized to comparable
    scales) plus ``outcome``. A deterministic simplex search is intentionally
    dependency-free so seasonal calibration can run in the existing jobs.
    """
    if not samples:
        return dict(ROSTER_COMPONENT_WEIGHTS)
    units = max(1, round(1.0 / max(0.01, float(step))))
    best = None
    for starter_units in range(units + 1):
        for depth_units in range(units - starter_units + 1):
            resilience_units = units - starter_units - depth_units
            weights = (starter_units / units, depth_units / units, resilience_units / units)
            error = 0.0
            for row in samples:
                pred = (weights[0] * float(row.get("starter") or 0)
                        + weights[1] * float(row.get("depth") or 0)
                        + weights[2] * float(row.get("resilience") or 0))
                error += (pred - float(row.get("outcome") or 0)) ** 2
            candidate = (error / len(samples), weights)
            if best is None or candidate < best:
                best = candidate
    w = best[1]
    return {"starter": w[0], "depth": w[1], "resilience": w[2],
            "mse": round(best[0], 6), "samples": len(samples)}


def positional_strength_profile(vals: List[float], pos: str,
                                slot_counts: Dict[str, int]) -> dict:
    """Starter, bench and top-player-loss profile for one position group."""
    clean = sorted((float(v or 0) for v in (vals or [])), reverse=True)
    if not clean:
        return {"starter": 0.0, "depth": 0.0, "resilience": 0.0,
                "fragility": 1.0, "composite": 0.0,
                "confidence": confidence_from_inputs(0, 1)}
    flex = _slot_total(slot_counts or {}, _FLEX_SLOT_NAMES)
    sf = _slot_total(slot_counts or {}, _SUPERFLEX_SLOT_NAMES)
    rb_wr = _slot_total(slot_counts or {}, _RB_WR_SLOT_NAMES | {"RB_WR"})
    wr_te = _slot_total(slot_counts or {}, _WR_TE_SLOT_NAMES | {"WR_TE"})
    rb_te = _slot_total(slot_counts or {}, _RB_TE_SLOT_NAMES | {"RB_TE"})
    dedicated = max(1, int((slot_counts or {}).get(pos) or (2 if pos in {"RB", "WR"} else 1)))
    if pos == "QB":
        starters = dedicated + sf
    elif pos == "RB":
        starters = dedicated + flex // 2 + rb_wr // 2 + rb_te // 2
    elif pos == "WR":
        starters = dedicated + flex // 2 + rb_wr // 2 + wr_te // 2
    elif pos == "TE":
        starters = dedicated + wr_te // 2 + rb_te // 2
    else:
        starters = dedicated
    starter_vals = clean[:starters]
    depth_vals = clean[starters:starters + max(1, starters)]
    starter = sum(starter_vals) / starters  # missing required starters count as zero
    depth = sum(depth_vals) / len(depth_vals) if depth_vals else 0.0
    without_top = clean[1:starters + 1]
    replacement = sum(without_top) / starters if without_top else 0.0
    resilience = min(1.0, replacement / starter) if starter > 0 else 0.0
    fragility = 1.0 - resilience
    # Resilience is converted to the same value scale before blending.
    composite = (ROSTER_COMPONENT_WEIGHTS["starter"] * starter
                 + ROSTER_COMPONENT_WEIGHTS["depth"] * depth
                 + ROSTER_COMPONENT_WEIGHTS["resilience"] * starter * resilience)
    return {"starter": starter, "depth": depth, "resilience": resilience,
            "fragility": fragility, "composite": composite,
            "confidence": confidence_from_inputs(len(clean), starters * 2)}


# ── Starter-caliber value thresholds ──────────────────────────────────────────
# The value at/above which a player is a "pure starter" at each position, plus
# the number of that position a team is expected to start. Single source of
# truth for the depth-warning system AND the consolidate/distribute engine, so
# they never disagree on who is startable.
#
# TE sits well below RB/WR because tight-end values are compressed: only ~2-3 TEs
# clear 400, yet a 1-TE league starts one per team (~10-14 startable TEs, down to
# ~215 in value). A 200 bar captures that real starter pool (and, after the
# small-league x1.2 scale, still keeps clear starters like a TE5 above the line)
# instead of flagging every non-elite TE as un-startable.
STARTER_THRESHOLD = {"QB": 500, "RB": 350, "WR": 350, "TE": 200}
DEPTH_FLOOR = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

# Superflex / "OP" (offensive player) slots are QB-eligible, so they raise the
# expected number of startable QBs by one. Kept separate from _FLEX_SLOT_NAMES
# because those never take a QB.

# In superflex, QB values sit on the same 0-999.9 scale but are lifted well above
# their 1QB level (the scale is anchored to non-QB skill players), so the 1QB
# starter threshold would wave through almost every rostered QB. Lift the QB bar
# by roughly the SF premium so only genuinely startable SF QBs clear it.
_SF_QB_THRESHOLD_MULT = 1.6


def derive_league_thresholds(
    roster_positions: List[str],
    num_teams: int,
    is_sf: bool = False,
) -> "tuple[Dict[str, int], Dict[str, int]]":
    """Derive starter-caliber value thresholds and depth floors from actual
    league settings.

    Depth floor  = number of that position in the starting lineup (including
                   FLEX split evenly across RB/WR, and superflex as +1 QB).
    Value threshold scales down with league size: larger leagues spread talent
    thinner, so a lower absolute value still constitutes a starter. In superflex
    the QB threshold is lifted (see _SF_QB_THRESHOLD_MULT).
    """
    pos_counts: Dict[str, int] = {}
    flex_count = 0
    superflex_count = 0
    rb_wr = wr_te = rb_te = 0
    for slot in roster_positions:
        s = canonicalize_slot(slot)
        if s in SKILL_POSITIONS:
            pos_counts[s] = pos_counts.get(s, 0) + 1
        elif s == "SUPER_FLEX":
            superflex_count += 1
        elif s == "FLEX":
            flex_count += 1
        elif s == "RB_WR":
            rb_wr += 1
        elif s == "WR_TE":
            wr_te += 1
        elif s == "RB_TE":
            rb_te += 1

    # A league with a superflex slot is a superflex league even if the caller
    # didn't flag it (and vice-versa) — treat either signal as SF.
    sf = bool(is_sf or superflex_count)

    rb_flex = flex_count // 2 + rb_wr // 2 + rb_te // 2
    wr_flex = (flex_count - flex_count // 2) + (rb_wr - rb_wr // 2) + (wr_te - wr_te // 2)
    te_flex = (wr_te // 2) + (rb_te - rb_te // 2)
    floor: Dict[str, int] = {
        "QB": max(1, pos_counts.get("QB", 1) + superflex_count),
        "RB": max(1, pos_counts.get("RB", 1) + rb_flex),
        "WR": max(1, pos_counts.get("WR", 1) + wr_flex),
        "TE": max(1, pos_counts.get("TE", 1) + te_flex),
    }

    scale = 12 / max(num_teams, 6)
    qb_mult = _SF_QB_THRESHOLD_MULT if sf else 1.0
    threshold: Dict[str, int] = {
        "QB": round(STARTER_THRESHOLD["QB"] * scale * qb_mult),
        "RB": round(STARTER_THRESHOLD["RB"] * scale),
        "WR": round(STARTER_THRESHOLD["WR"] * scale),
        "TE": round(STARTER_THRESHOLD["TE"] * scale),
    }
    return threshold, floor


def dedicated_starter_counts(roster_positions: List[str]) -> "Dict[str, int]":
    """How many players a manager must field at each position to fill their
    position-LOCKED starting slots.

    Standard FLEX (RB/WR/TE) is deliberately excluded: it's fungible, so trading
    a surplus WR for an RB you'll start doesn't reduce your ability to fill your
    dedicated WR slots. Superflex IS counted toward QB, since SF managers field a
    second quarterback there.

    Used by the depth warning so a lateral position swap (WR-rich -> RB) stays
    quiet as long as you can still fill your locked slots. Falls back to a
    standard 1QB/2RB/2WR/1TE lineup when league settings are unknown.
    """
    if not roster_positions:
        return {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    counts: Dict[str, int] = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    superflex = 0
    for slot in roster_positions:
        s = canonicalize_slot(slot)
        if s in counts:
            counts[s] += 1
        elif s == "SUPER_FLEX":
            superflex += 1
        # standard flex slots intentionally ignored (fungible)
    counts["QB"] += superflex
    return counts


# ======================================================================
# From utils/vorp.py
# ======================================================================

"""Pure VORP / replacement-level helpers.

Fantasy VORP is season points minus a replacement-level starter at the same
position. Replacement rank is (starters × teams + FLEX share), matching the
league-aware math in ``data_building.advanced_metrics``.

These helpers are importable without the DB/Flask stack so unit tests can pin
the Kraft-style case: a top-10 projected TE must not inherit last season's
injury-shortened totals as VORP next to upcoming-season proj PPG.
"""

import os

# Standard starters per team used to locate the replacement-level player.
# FLEX is modeled as one extra RB/WR/TE slot per team, split by typical usage.
VALUE_STARTERS = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 1.0}
VALUE_FLEX_ALLOC = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
# Marginal PPR points worth one head-to-head win (≈ weekly team-score stdev).
POINTS_PER_WIN_DEFAULT = 28.0
PROJ_SEASON_GAMES = 17.0

_POS_NORM = {"HB": "RB", "FB": "RB", "SE": "WR", "FL": "WR"}


def normalize_position(pos: Optional[str]) -> Optional[str]:
    """Return the canonical fantasy position for a raw position string."""
    if not pos:
        return pos
    upper = str(pos).upper()
    return _POS_NORM.get(upper, upper)


def points_per_win() -> float:
    try:
        v = float(os.getenv("POINTS_PER_WIN", "").strip())
        return v if v > 0 else POINTS_PER_WIN_DEFAULT
    except (TypeError, ValueError):
        return POINTS_PER_WIN_DEFAULT


def stamp_value_metrics(
    recs: List[Dict[str, Any]],
    num_teams: int = 12,
    starters: Optional[Dict[str, float]] = None,
    points_per_win_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Stamp ``vorp``, ``war``, and position ranks onto recs with ``position`` + ``pts``.

    Mutates ``recs`` in place and returns them. Replacement is the player ranked
    at ``round(starters×teams + flex_share×teams)`` within the position.
    """
    teams = int(num_teams) if num_teams and num_teams > 0 else 12
    start_slots = {**VALUE_STARTERS, **(starters or {})}
    ppw = (
        points_per_win_value
        if (points_per_win_value and points_per_win_value > 0)
        else points_per_win()
    )

    pool_by_pos: Dict[str, List[float]] = {}
    for rec in recs:
        pos = rec.get("position")
        if pos not in start_slots:
            continue
        pool_by_pos.setdefault(pos, []).append(float(rec.get("pts") or 0.0))

    repl_pts: Dict[str, float] = {}
    for pos, base in start_slots.items():
        rank = base * teams + VALUE_FLEX_ALLOC.get(pos, 0.0) * teams
        pool = sorted(pool_by_pos.get(pos, []), reverse=True)
        if not pool:
            repl_pts[pos] = 0.0
            continue
        idx = int(round(rank)) - 1
        idx = max(0, min(len(pool) - 1, idx))
        repl_pts[pos] = pool[idx]

    for rec in recs:
        pos = rec.get("position")
        if pos not in start_slots:
            continue
        vorp = float(rec.get("pts") or 0.0) - repl_pts.get(pos, 0.0)
        rec["vorp"] = round(vorp, 3)
        rec["war"] = round(vorp / ppw, 3)

    for pos in start_slots:
        pos_recs = [r for r in recs if r.get("position") == pos]
        for key in ("vorp", "war"):
            for i, rec in enumerate(
                sorted(pos_recs, key=lambda x: x[key], reverse=True), 1
            ):
                rec[f"{key}_rank"] = i
    return recs


def projected_season_pts(
    player: Mapping[str, Any],
    season_games: float = PROJ_SEASON_GAMES,
) -> Optional[float]:
    """Upcoming-season points: ``proj_pts``, else ``proj_ppg × season_games``.

    Last-season actuals (``ppg``, ``total_pts``) are ignored on purpose — those
    are the injury-shortened numbers that made a TE10 project as −40 VORP.
    """
    pts = player.get("proj_pts")
    try:
        if pts is not None and float(pts) > 0:
            return float(pts)
    except (TypeError, ValueError):
        pass
    ppg = player.get("proj_ppg")
    try:
        if ppg is not None and float(ppg) > 0:
            return float(ppg) * float(season_games)
    except (TypeError, ValueError):
        pass
    return None


def projected_vorp_map(
    players: Iterable[Mapping[str, Any]],
    num_teams: int = 12,
    starters: Optional[Dict[str, float]] = None,
    season_games: float = PROJ_SEASON_GAMES,
) -> Dict[str, float]:
    """Player-id → VORP from projected season points (draft/rankings overlay).

    Uses the same replacement rank as historical VORP, but the inputs are
    upcoming-season projections so a top-10 projected TE cannot inherit last
    year's missed-game totals.
    """
    recs: List[Dict[str, Any]] = []
    for p in players or []:
        pos = normalize_position(str(p.get("position") or ""))
        if pos not in VALUE_STARTERS:
            continue
        pts = projected_season_pts(p, season_games)
        if pts is None:
            continue
        pid = str(p.get("id") or p.get("player_id") or "")
        if not pid:
            continue
        recs.append({"player_id": pid, "position": pos, "pts": pts})
    if not recs:
        return {}
    stamp_value_metrics(recs, num_teams=num_teams, starters=starters)
    return {r["player_id"]: float(r["vorp"]) for r in recs}


# ======================================================================
# From utils/value_helpers.py
# ======================================================================

"""Pure TE-premium and format-aware value helpers.

Extracted from app.py so this logic can be unit-tested without importing the
full application (pandas / DB) stack. A league that awards bonus points per TE
reception ("TE premium") makes tight ends more valuable; these helpers snap the
league's Sleeper ``bonus_rec_te`` to the supported tiers (0 / 0.5 / 1.0) and
scale TE values by +20% per full point — matching the trade calculator, activity
feed and player modal so values stay consistent on every page that shows them.

Also owns the redraft/dynasty column picker and the unpriced-redraft depth fill
so waiver, start/sit, and the draft board all read the same numbers.
"""

from collections import defaultdict

_SKILL_POS = {"QB", "RB", "WR", "TE"}


def scoring_format_from_settings(scoring_settings) -> str:
    """PPR / half / std the player modal uses for this league.

    Missing reception scoring defaults to PPR, matching ``/api/player-details``.
    ESPN leagues may only set ``pointsPerReception``.
    """
    settings = scoring_settings if isinstance(scoring_settings, dict) else {}
    rec = settings.get("rec")
    if rec is None:
        rec = settings.get("pointsPerReception")
    try:
        rec_f = float(rec) if rec is not None else 1.0
    except (TypeError, ValueError):
        return "ppr"
    if rec_f >= 1.0:
        return "ppr"
    if rec_f >= 0.5:
        return "half"
    return "std"


def te_premium_from_settings(scoring_settings) -> float:
    """Snap a league's Sleeper ``bonus_rec_te`` to a supported premium tier.

    Returns 1.0 (full), 0.5 (half), or 0.0 (none). Non-dict / non-numeric input
    yields 0.0 rather than raising.
    """
    try:
        b = float((scoring_settings or {}).get("bonus_rec_te") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return 1.0 if b >= 0.75 else 0.5 if b >= 0.25 else 0.0


def apply_te_premium(value, position, te_premium) -> float:
    """Scale a TE's value up for TE-premium leagues; pass-through otherwise.

    +20% per full premium point. ``value`` is coerced to float (DB values arrive
    as Decimal, which would otherwise raise on ``Decimal * float``), returning
    0.0 on non-numeric input rather than raising.
    """
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if te_premium and str(position or "").upper() == "TE":
        return v * (1.0 + te_premium * 0.20)
    return v


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_value_keys(*, is_redraft: bool, is_sf: bool) -> tuple[str, str]:
    """(primary, fallback) value columns for a league's scoring type + QB format.

    Redraft/keeper uses ``redraft_value_*``. Dynasty Superflex uses ``sf_value``.
    Fallback is the dynasty column so a missing redraft price still ranks the
    player instead of dropping them below the value floor.
    """
    if is_redraft:
        return (
            ("redraft_value_sf" if is_sf else "redraft_value_1qb"),
            ("sf_value" if is_sf else "value"),
        )
    return (("sf_value" if is_sf else "value"), "value")


def format_rank_label_key(*, is_redraft: bool, is_sf: bool) -> str:
    """Position-rank label field matching ``format_value_keys``."""
    if is_redraft:
        return "redraft_sf_pos_rank_label" if is_sf else "redraft_pos_rank_label"
    return "sf_pos_rank_label" if is_sf else "pos_rank_label"


def format_rank_key(*, is_redraft: bool, is_sf: bool) -> str:
    """Numeric position-rank field matching ``format_rank_label_key``."""
    if is_redraft:
        return "redraft_sf_pos_rank" if is_sf else "redraft_pos_rank"
    return "sf_pos_rank" if is_sf else "pos_rank"


def row_format_value(row: dict, primary: str, fallback: str) -> float:
    """Numeric value from ``primary``, then ``fallback``, then 0."""
    if not isinstance(row, dict):
        return 0.0
    return _num(row.get(primary) or row.get(fallback) or 0.0)


def row_format_rank_label(row: dict, label_key: str) -> str:
    """Rank label for the active format, falling back to the dynasty 1QB label."""
    if not isinstance(row, dict):
        return ""
    return str(row.get(label_key) or row.get("pos_rank_label") or "")


def rerank_pos_labels(
    table: list,
    val_key: str,
    rank_key: str,
    label_key: str,
    positions: Iterable[str] | None = None,
) -> list:
    """Stamp ``rank_key`` / ``label_key`` from descending ``val_key`` within each position."""
    if not table:
        return table
    allowed = {str(p).upper() for p in positions} if positions is not None else None
    grp: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if not pos or pos == "PICK":
            continue
        if allowed is not None and pos not in allowed:
            continue
        grp[pos].append(i)
    for pos, idxs in grp.items():
        idxs.sort(key=lambda i: _num(table[i].get(val_key)), reverse=True)
        for rank, i in enumerate(idxs, 1):
            table[i][rank_key] = rank
            table[i][label_key] = f"{pos}{rank}"
    return table


def fill_unpriced_redraft_values(table: list) -> list:
    """Give skill players with no redraft price a value scaled below the priced floor.

    FantasyCalc only prices roughly the top ~64 RB / ~150 WR. Without this fill,
    waiver/start-sit ``or``-fallback to dynasty ``value`` and a redraft league
    shows dynasty numbers for most free agents. Derived values stay strictly
    below every priced player so real redraft prices always rank first.
    Mutates rows in place and returns the same list. Idempotent when a
    positive redraft value is already present.
    """
    if not table:
        return table
    for rd_field, dyn_field in (
        ("redraft_value_1qb", "value"),
        ("redraft_value_sf", "sf_value"),
    ):
        priced = [
            _num(row.get(rd_field))
            for row in table
            if isinstance(row, dict)
            and str(row.get("position") or "").upper() in _SKILL_POS
            and _num(row.get(rd_field)) > 0
        ]
        floor = min(priced) if priced else 1.0
        unpriced_dyn = [
            _num(row.get(dyn_field))
            for row in table
            if isinstance(row, dict)
            and str(row.get("position") or "").upper() in _SKILL_POS
            and _num(row.get(rd_field)) <= 0
        ]
        dyn_max = max(unpriced_dyn) if unpriced_dyn else 0.0
        if dyn_max <= 0:
            continue
        cap = floor * 0.9
        for row in table:
            if not isinstance(row, dict):
                continue
            if str(row.get("position") or "").upper() not in _SKILL_POS:
                continue
            if _num(row.get(rd_field)) > 0:
                continue
            dyn = _num(row.get(dyn_field))
            if dyn > 0:
                row[rd_field] = round(cap * (dyn / dyn_max), 2)
    return table


def apply_redraft_display_fields(table: list) -> list:
    """Fill missing redraft values and stamp redraft position-rank labels."""
    fill_unpriced_redraft_values(table)
    if table:
        rerank_pos_labels(table, "redraft_value_1qb", "redraft_pos_rank", "redraft_pos_rank_label")
        rerank_pos_labels(table, "redraft_value_sf", "redraft_sf_pos_rank", "redraft_sf_pos_rank_label")
    return table
