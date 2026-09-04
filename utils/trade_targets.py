"""Roster-fit ranking for Trade Targets.

Need detection still uses starter-slot-weighted positional strength (same as
the Teams page). Candidate *selection* used to sort the other teams' players
by raw value, so every QB-needy roster saw the same elites. This module ranks
candidates by how well they fill THIS roster's gap at a price the viewer can
actually pay, prefers owners who need the viewer's surplus, and returns a
mixed list (not four elites per weak position).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.roster_strength import weighted_pos_strength

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
        ordered = sorted(
            rows,
            key=lambda r: (-float(r.get("fit_score") or 0.0), -float(r.get("value") or 0.0)),
        )
        for row in ordered:
            pos = str(row.get("position") or "")
            cap = pos_caps.get(pos, MAX_PER_POS_SOFT)
            if counts.get(pos, 0) >= cap:
                continue
            picked.append(_public(row))
            counts[pos] = counts.get(pos, 0) + 1
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
