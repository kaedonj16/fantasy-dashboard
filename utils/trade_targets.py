"""Roster-fit ranking for Trade Targets.

Need detection still uses starter-slot-weighted positional strength (same as
the Teams page). Candidate *selection* used to sort the other teams' players
by raw value, so every QB-needy roster saw the same elites. This module ranks
candidates by how well they fill THIS roster's gap at a price the viewer can
actually pay, with owner-surplus and age-window as tie-breakers.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.roster_strength import weighted_pos_strength

POSITIONS = ("QB", "RB", "WR", "TE")
PEAK_AGE = {"QB": 29, "RB": 26, "WR": 27, "TE": 27}

# Bottom-of-league cutoff used by the API before this ranker existed.
NEED_RANK_FRACTION = 0.35

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


def detect_needed_positions(
    pos_ranks: Dict[str, int],
    viewer_vals: Dict[str, Sequence[float]],
    num_teams: int,
    starter_thresholds: Dict[str, float],
    starter_floors: Dict[str, int],
) -> List[str]:
    """Positions the viewer should shop: bottom 35% and/or a starter hole.

    Ordered worst-gap first (starter deficit, then league rank).
    """
    n = max(int(num_teams or 1), 1)
    cutoff = max(1, round(n * NEED_RANK_FRACTION))
    scored: List[Tuple[int, int, str]] = []
    for pos in POSITIONS:
        vals = [float(v or 0.0) for v in (viewer_vals or {}).get(pos, [])]
        threshold = float((starter_thresholds or {}).get(pos) or 0.0)
        floor = int((starter_floors or {}).get(pos) or 1)
        starters = sum(1 for v in vals if v >= threshold) if threshold else 0
        deficit = max(0, floor - starters)
        rank = int((pos_ranks or {}).get(pos) or n)
        is_bottom = rank > n - cutoff
        if deficit or is_bottom:
            scored.append((deficit, rank, pos))
    scored.sort(key=lambda t: (-t[0], -t[1], POSITIONS.index(t[2])))
    return [pos for _, _, pos in scored]


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
) -> str:
    """One short line for the UI: why this player is on THIS roster's list."""
    pos = str(pos or "").upper()
    if starter_count < floor and value >= threshold:
        return f"Fills your {pos} hole"
    if viewer_best > 0 and value >= viewer_best * 1.15:
        return f"Upgrades your {pos}1"
    if floor >= 2 and starter_count >= 1 and value >= threshold:
        return f"Adds {pos}{starter_count + 1} depth"
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
) -> Tuple[float, str]:
    value = float(row.get("value") or 0.0)
    age = row.get("age")
    gain = strength_gain(viewer_vals, value, pos, slot_counts)
    # Absolute gain fills a hole; efficiency stops 900-value elites from
    # always beating the mid-tier player who actually fits the budget.
    efficiency = gain / max(value, 1.0)
    raw = (0.55 * gain) + (0.45 * efficiency * 400.0)
    if starter_count < floor and value >= threshold:
        raw *= 1.25
    elif starter_count >= floor and value < threshold:
        raw *= 0.55
    afford = affordability_multiplier(value, one_for_one, package_max)
    avail = availability_multiplier(row.get("depth_rank") or 1, row.get("owner_pos_count") or 1)
    age_m = age_fit_multiplier(age, pos, window, is_redraft=is_redraft)
    score = raw * afford * avail * age_m
    why = fit_reason(
        pos=pos, value=value, age=age,
        viewer_best=max((float(v or 0.0) for v in (viewer_vals or [])), default=0.0),
        starter_count=starter_count, floor=floor, threshold=threshold,
        window=window, one_for_one=one_for_one,
        depth_rank=int(row.get("depth_rank") or 1),
        owner_pos_count=int(row.get("owner_pos_count") or 1),
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
        )
        out = dict(row)
        out["why"] = why
        out["fit_score"] = round(score, 3)
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
) -> Dict[str, Any]:
    """Pick the targets the UI lists, grouped the same way the API always has.

    ``by_position`` is populated when the roster has a real need. Balanced
    rosters get ``all_positions`` — still fit-ranked, not top-by-value.
    """
    thresholds = dict(starter_thresholds or {})
    floors = dict(starter_floors or {})
    for pos in POSITIONS:
        thresholds.setdefault(pos, {"QB": 500, "RB": 350, "WR": 350, "TE": 200}[pos])
        floors.setdefault(pos, 1 if pos in ("QB", "TE") else 2)

    chip = one_for_one_chip(viewer_asset_values)
    ceiling = package_ceiling(viewer_asset_values, pick_value)
    window = infer_roster_window(valued_ages, is_redraft=is_redraft)

    needed = detect_needed_positions(
        pos_ranks, viewer_vals, num_teams, thresholds, floors,
    )

    # Depth ranks are a property of the owner's room, so annotate the full
    # candidate pool before we slice it.
    flat: List[Dict[str, Any]] = []
    for pos in POSITIONS:
        for row in candidates_by_pos.get(pos) or []:
            item = dict(row)
            item["position"] = str(item.get("position") or pos).upper()
            flat.append(item)
    annotate_owner_depth(flat)
    pooled: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POSITIONS}
    for row in flat:
        pos = row.get("position")
        if pos in pooled:
            pooled[pos].append(row)

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
        )

    if needed:
        by_position = {pos: rows for pos in needed if (rows := _rank(pos, per_pos_limit))}
        return {
            "by_position": by_position,
            "all_positions": {},
            "needed_positions": needed,
            "window": window,
        }

    all_positions = {pos: rows for pos in POSITIONS if (rows := _rank(pos, balanced_per_pos))}
    return {
        "by_position": {},
        "all_positions": all_positions,
        "needed_positions": [],
        "window": window,
    }
