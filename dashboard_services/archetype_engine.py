"""
Archetype-based trade suggestion engine.

Computes Contending / Rebuilding / Consolidate / Distribute suggestions
based on roster composition, value trends, and win probability modeling.

No imports from app.py - safe for use from any blueprint or endpoint.
"""
from __future__ import annotations

import logging
import math
import time as _time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from utils.tier_stack import asset_tier
from utils.tier_thresholds import FALLBACK_THRESHOLDS, compute_tier_thresholds

log = logging.getLogger(__name__)

# ── Sim-state cache (keyed by platform:league_id:season, 5-min TTL) ──────────
_SIM_CACHE: Dict[str, Any] = {}
_SIM_CACHE_TTL = 300  # seconds

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

# Consolidate is trading *up* comparable assets, not dumping scrubs: the
# headliner of a send package must sit within this many tiers of the target.
# Two mid-first WRs (T3) for a stud RB (T1) is a real consolidation; a wall of
# T5 depth for that same stud is not, no matter how the raw values add up.
_CONSOLIDATE_MAX_TIER_DROP = 2

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
    use_redraft: bool = False,
) -> float:
    """
    Greedy lineup optimizer. Fill dedicated slots first (QB/RB/WR/TE),
    then fill FLEX with best remaining eligible players.

    use_redraft=True → score by redraft_value (current-season production),
    falling back to dynasty value if redraft is unavailable. Use this for
    win-probability calculations; dynasty value for trade window matching.
    """
    slots = SLOTS_SF if league_type == "sf" else SLOTS_1QB

    by_pos: Dict[str, List[float]] = {}
    for pid in player_ids:
        info = values_by_id.get(pid)
        if not info:
            continue
        pos = str(info.get("position") or "").upper()
        if use_redraft:
            val = _f(info.get("redraft_value")) or _f(info.get("value"))
        else:
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


_ROOKIE_PPG: Dict[str, float] = {"QB": 14.0, "RB": 7.5, "WR": 6.5, "TE": 4.5}
_ROOKIE_PPG_DEFAULT = 6.0
_FLEX_ELIGIBLE = {"RB", "WR", "TE"}
_BENCH_SLOTS   = {"BN", "IR", "TAXI"}


def _ppg_lineup(
    pids: List[str],
    ppg_map: Dict[str, Any],
    pos_map: Dict[str, str],
    roster_positions: List[str],
) -> float:
    """
    Compute projected weekly lineup score using actual PPG data - the same
    logic used by simulate_playoff_odds._position_aware_lineup.

    Falls back to _optimal_lineup_value (dynasty value) when ppg_map is empty.
    """
    if not ppg_map:
        return 0.0

    fixed_slots: Dict[str, int] = {}
    flex_slots = sflex_slots = 0
    for slot in roster_positions:
        s = str(slot).upper()
        if s in _BENCH_SLOTS:
            continue
        if s == "SUPER_FLEX":
            sflex_slots += 1
        elif s in {"FLEX", "WRRB_FLEX", "WRTE_FLEX", "RBWRTE", "RBWR"}:
            flex_slots += 1
        elif s in SKILL_POS:
            fixed_slots[s] = fixed_slots.get(s, 0) + 1

    # Position-average fallback (mirrors simulate_playoff_odds._position_aware_lineup)
    _pos_totals: Dict[str, List] = {}
    for _info in ppg_map.values():
        _p, _g = str(_info.get("pos") or "").upper(), float(_info.get("ppg") or 0)
        if _p and _g > 0:
            _pos_totals.setdefault(_p, [0.0, 0])
            _pos_totals[_p][0] += _g
            _pos_totals[_p][1] += 1
    pos_fallback = {p: v[0] / v[1] for p, v in _pos_totals.items() if v[1] > 0}

    by_pos: Dict[str, List[float]] = {}
    for pid in pids:
        info = ppg_map.get(str(pid))
        if info:
            pos = str(info.get("pos") or "").upper()
            ppg = float(info.get("ppg") or 0) or pos_fallback.get(pos) or _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
        else:
            pos = pos_map.get(str(pid), "")
            ppg = pos_fallback.get(pos) or _ROOKIE_PPG.get(pos, _ROOKIE_PPG_DEFAULT)
        if pos in SKILL_POS:
            by_pos.setdefault(pos, []).append(ppg)

    for pos in by_pos:
        by_pos[pos].sort(reverse=True)

    used: Dict[str, int] = {}
    total = 0.0

    for slot_pos, count in fixed_slots.items():
        pool = by_pos.get(slot_pos, [])
        for _ in range(count):
            i = used.get(slot_pos, 0)
            total += pool[i] if i < len(pool) else 0.0
            used[slot_pos] = i + 1

    flex_pool = sorted(
        [(pos, ppg) for pos in _FLEX_ELIGIBLE for ppg in by_pos.get(pos, [])[used.get(pos, 0):]],
        key=lambda x: x[1], reverse=True,
    )
    for i in range(flex_slots):
        if i < len(flex_pool):
            total += flex_pool[i][1]
    remaining = flex_pool[flex_slots:]

    sflex_pool = sorted(
        [("QB", ppg) for ppg in by_pos.get("QB", [])[used.get("QB", 0):]] + remaining,
        key=lambda x: x[1], reverse=True,
    )
    for i in range(sflex_slots):
        if i < len(sflex_pool):
            total += sflex_pool[i][1]

    return total


# ── Win-probability model ─────────────────────────────────────────────────────

def _win_prob(team_val: float, league_avg: float) -> float:
    """Logistic win probability from relative lineup value.

    k=2 keeps most realistic dynasty rosters in the 0.25–0.75 range so
    that small lineup changes still produce meaningful playoff-odds deltas.
    (k=4 saturated too quickly - a roster 20 % above average hit ~70 % WP,
    pushing playoff-odds deltas to near zero for any further acquisition.)
    """
    ratio = team_val / max(1.0, league_avg)
    return 1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.0)))


def _wp_delta(
    viewer_val: float,
    target_val: float,
    replace_val: float,
    league_avg: float,
) -> float:
    new_val = viewer_val - replace_val + target_val
    return _win_prob(new_val, league_avg) - _win_prob(viewer_val, league_avg)


def _playoff_odds(
    weekly_wp: float,
    num_weeks: int = 14,
    num_teams: int = 10,
    playoff_spots: int = 4,
) -> float:
    """Approximate seasonal playoff probability from weekly win rate.

    Uses a normal approximation of the binomial win-count distribution.
    The cutoff is the win total that ranks a team at the playoff bubble
    in a balanced league.
    """
    exp_wins = weekly_wp * num_weeks
    std_wins = math.sqrt(max(num_weeks * weekly_wp * (1.0 - weekly_wp), 0.01))
    cutoff   = num_weeks * (num_teams - playoff_spots) / max(num_teams, 1)
    z        = (exp_wins - cutoff) / std_wins
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _estimate_acceptance(send_val: float, receive_val: float, is_preferred: bool,
                         fit: int = 0) -> int:
    """Estimate acceptance likelihood (5–90) from the partner team's perspective.

    Partner receives send_val (viewer's assets) and gives up receive_val (their
    player). A higher send/receive ratio means the partner is getting the better
    end. Both values should already be depth-adjusted (effective) so the ratio
    reflects the balance the trade card shows.

    The curve is continuous (a logistic centered at parity) rather than four
    hard buckets, so near-identical deals no longer collapse to the same flat
    number: ratio 1.00 → ~50, 1.10 → ~67, 0.90 → ~33, 1.20 → ~80.

    ``fit`` nudges the estimate for non-value factors the ratio can't see -
    chiefly whether the package addresses a position the partner is thin at
    (positive) or piles onto one they are already deep at (negative).
    """
    ratio = send_val / max(receive_val, 1.0)
    base = 100.0 / (1.0 + math.exp(-7.0 * (ratio - 1.0)))
    score = base + (8 if is_preferred else 0) + fit
    return int(min(90, max(5, round(score))))


def _acceptance_fit(pkg: List[Dict], need: Optional[set], stacked: Optional[set]) -> int:
    """Acceptance nudge for how well a send package fits the partner's roster:
    positive when it sends a position they're thin at, negative when it piles
    onto one they're already deep at. Clamped so it tilts, never dominates, the
    value-based estimate."""
    if not pkg or not (need or stacked):
        return 0
    bonus = 0
    for a in pkg:
        if a.get("is_pick") or a.get("position") == "PICK":
            continue
        pos = a.get("position")
        if need and pos in need:
            bonus += 6
        elif stacked and pos in stacked:
            bonus -= 8
    return max(-12, min(12, bonus))


def _fit_note(pkg: List[Dict], need: Optional[set], target_pos: str,
              partner_counts: Optional[Dict[str, int]]) -> str:
    """Short, human explanation of why a package suits the partner's roster:
    which of their needs the send fills, and whether the target comes from a
    position they're deep at. Empty when there's nothing worth saying."""
    notes: List[str] = []
    send_pos = {a.get("position") for a in (pkg or [])
                if not a.get("is_pick") and a.get("position") != "PICK"}
    matched = [p for p in ("QB", "RB", "WR", "TE")
               if need and p in need and p in send_pos]
    if matched:
        notes.append("fills their " + "/".join(matched) + " need")
    if partner_counts and target_pos and partner_counts.get(target_pos, 0) >= 4:
        notes.append(f"from their {target_pos} surplus")
    text = ", ".join(notes[:2])
    return (text[:1].upper() + text[1:]) if text else ""


def _availability(depth_rank: int, pos_count: int) -> float:
    """How likely a rival is to part with a player, as a ranking multiplier.

    Teams keep their best at a position and move the depth behind it, so a
    surplus frees up a rival's RB3/RB4 - not their RB1. Key off the player's
    depth rank on their OWN roster (1 = their best at that position), so a stud
    stays a keeper even on a stacked team.
    """
    if depth_rank <= 1:
        return 0.75          # their best at the spot - a keeper regardless of depth
    if pos_count >= 4:
        return 1.25          # buried depth on a stacked roster - very movable
    if pos_count >= 3:
        return 1.10
    if pos_count <= 1:
        return 0.85          # their only body there
    return 1.0


def _suggestion_rank(r: Dict[str, Any]) -> float:
    """Composite ordering key for a finished suggestion: weight roster impact
    (net playoff-odds delta) most, then how likely the partner is to accept, so
    the list leads with deals that are both worthwhile and realistic. Negative
    impact is preserved (not floored) so ceiling-trimming deals sink."""
    impact = min(1.0, (r.get("net_playoff_odds_delta") or 0.0) / 0.15)
    accept = (r.get("acceptance_pct") or 0) / 100.0
    return 0.6 * impact + 0.4 * accept


def _depth_penalty(delta: int, sorted_vals: Optional[List[float]], league_size: int = 10) -> float:
    """Roster-depth penalty the side sending MORE assets absorbs in an unequal
    trade. Mirrors utils.tier_stack.apply_tier_stack_adjustment so the value we
    reason about here matches the *effective* value the trade card displays.

    When a package of N assets is acquired for a single target, the acquiring
    side sends ``delta = N - 1`` extra bodies; each extra body is only worth a
    deep-bench replacement, so its contribution to the deal is discounted.
    """
    if delta <= 0:
        return 0.0
    base_rank = int(league_size) * 38
    _PENALTY_FRAC = 0.5
    n = len(sorted_vals) if sorted_vals else 0
    total = 0.0
    for i in range(delta):
        if n:
            rank = min(n, base_rank + i * 10)
            idx = min(rank - 1, n - 1)
            total += sorted_vals[idx] * _PENALTY_FRAC
        else:
            total += max(20.0, (80.0 + i * -5.0) * _PENALTY_FRAC)
    return total


def _acquire_band(target_val: float, archetype: str) -> Tuple[float, float]:
    """Effective-value multiplier band a send package must land in to acquire a
    target. Bands are expressed against the target's value and applied to the
    package's *depth-adjusted* value, so a package that clears the band reads as
    fair (or a slight, deliberate overpay) on the trade card - the balance both
    sides actually see.

    Higher-value targets carry a small "star premium": elite players rarely move
    at par, so the acquiring side is expected to pay a touch over to pry one
    loose. That keeps suggestions realistic (feasible for both sides) instead of
    surfacing lopsided steals the holder would never accept.
    """
    if archetype == "consolidate":
        # Trading up for a star: pay a touch over on effective value. The
        # premium scales with the target so elites cost more to pry loose.
        if target_val >= 900:
            star = 0.05
        elif target_val >= 650:
            star = 0.03
        else:
            star = 0.0
        return (1.00 + star, 1.14 + star)
    # Win-now / lateral swaps: fair effective value, no premium. The depth
    # penalty baked into effective value already keeps multi-asset offers honest.
    return (0.96, 1.08)


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
            # Consolidate = trade several lesser pieces up into one better
            # player, so the send pool wants mid- AND low-tier assets. Studs
            # make poor consolidation chips (you'd be shipping the very
            # concentration you're trying to build), so they score near zero;
            # everyone else scores higher the cheaper they are, keeping the
            # combinatorial search stocked with usable filler.
            if val >= 950:
                sc = 0.05
            else:
                sc = max(0.1, 0.9 - val / 3000.0)
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


def _select_packages(
    sends: List[Dict], target_val: float, archetype: str, max_pkgs: int = 2,
    sorted_vals: Optional[List[float]] = None, league_size: int = 10,
    need_positions: Optional[set] = None, stacked_positions: Optional[set] = None,
    tier_thresholds: Optional[List[float]] = None,
) -> List[List[Dict]]:
    """Return up to max_pkgs distinct send packages whose *depth-adjusted* value
    lands in the acquire band for this target (see _acquire_band).

    Packages are scored on effective (depth-adjusted) value, not raw sum,
    because effective value is what the trade card shows once the side sending
    more bodies absorbs the bench penalty. Matching on effective value is what
    stops a multi-piece offer from reading as a lopsided underpay for a star:
    to clear the band, several lesser pieces must out-total the target in RAW
    value by enough to survive the depth penalty.

    Every candidate structure - consolidate, contending, player+pick, and the
    closest-match fallbacks - is also held to a tier guard: the package's
    headliner must sit within _CONSOLIDATE_MAX_TIER_DROP tiers of the target, so
    no path can quietly assemble a wall of low-tier depth to reach a stud.

    Selection also respects the *partner's* roster: within the value band, a
    package is preferred when it sends a position the partner is thin at
    (need_positions) and de-prioritised when it piles onto one they are already
    deep at (stacked_positions). A fair-value package the partner has no roster
    use for is not actually a likely trade.

    Surfaces different trade structures per target:
      1. Best player-only combo (1-3 players, no picks)
      2. Best player + draft-pick combo (if the viewer has picks)
    Falls back to the closest-effective combo when nothing lands in the window.
    """
    if not sends:
        return []

    lo_mult, hi_mult = _acquire_band(target_val, archetype)
    lo, hi = target_val * lo_mult, target_val * hi_mult

    # Hard floor on effective value: even the fallback paths (which otherwise
    # pick the closest combo with no bounds) must not surface a package the card
    # would show as a meaningful underpay. Showing nothing is better than a
    # lopsided offer the holder would never accept.
    eff_floor = target_val * max(0.90, lo_mult - 0.04)

    _FIT_UNIT = 45.0  # value-points of "closeness" credit per positional-fit signal

    def _eff(*assets) -> float:
        raw = sum(a.get("value", 0) for a in assets)
        return raw - _depth_penalty(max(0, len(assets) - 1), sorted_vals, league_size)

    def _fit_adj(assets) -> float:
        # Reward sending what the partner needs, discourage what they're stacked
        # at. Applied to the closeness metric only (never to the value band), so
        # it breaks ties between otherwise comparable packages rather than
        # overriding value matching.
        if not (need_positions or stacked_positions):
            return 0.0
        adj = 0.0
        for a in assets:
            if a.get("is_pick") or a.get("position") == "PICK":
                continue
            pos = a.get("position")
            if need_positions and pos in need_positions:
                adj -= _FIT_UNIT
            elif stacked_positions and pos in stacked_positions:
                adj += _FIT_UNIT * 1.2
        return adj

    def _dist(*assets) -> float:
        """Closeness metric (lower is better): effective-value gap, nudged by fit."""
        return abs(_eff(*assets) - target_val) + _fit_adj(assets)

    def _in_band(*assets) -> bool:
        e = _eff(*assets)
        return lo <= e <= hi

    # Tier guard, applied to every package: a fair offer trades comparable-quality
    # assets, not a pile of low-tier depth summed up to a stud's price. Require the
    # package's headliner (its best piece) to sit within a bounded number of tiers
    # of the target - "don't go down too many tiers" to reach it. Picks carry no
    # tier and are ignored by the check.
    _thr = tier_thresholds or FALLBACK_THRESHOLDS
    _target_tier = asset_tier(target_val, _thr)

    def _tier_ok(*assets) -> bool:
        tiers = [asset_tier(a.get("value", 0), _thr) for a in assets
                 if not (a.get("is_pick") or a.get("position") == "PICK")]
        if not tiers:
            return True
        return (min(tiers) - _target_tier) <= _CONSOLIDATE_MAX_TIER_DROP

    def _drop_underpays(pkgs: List[List[Dict]]) -> List[List[Dict]]:
        return [p for p in pkgs if _eff(*p) >= eff_floor]

    def _dedup(pkgs: List[List[Dict]]) -> List[List[Dict]]:
        # Drop structurally identical packages (same asset set) so the surfaced
        # options are genuinely different offers, not the same one twice.
        out, seen = [], set()
        for p in pkgs:
            key = frozenset(a.get("player_id") or a.get("name") for a in p)
            if key and key not in seen:
                seen.add(key)
                out.append(p)
        return out

    _players = [s for s in sends
                if not s.get("is_pick") and s.get("position") != "PICK"]
    # Keep the combinatorial search stocked with the best-scored fillers, but
    # always fold in the highest-VALUE assets too. Consolidate deliberately
    # scores a team's own studs as poor everyday chips, which on a deep roster
    # would truncate them out of the pool entirely - yet those are exactly the
    # pieces a loaded team must package to trade up into a bigger stud. Without
    # this, an affordable stud target surfaces but no send package can be built.
    player_pool = _players[:12]
    _seen_pp = {id(s) for s in player_pool}
    for s in sorted(_players, key=lambda x: -x.get("value", 0))[:6]:
        if id(s) not in _seen_pp:
            _seen_pp.add(id(s))
            player_pool.append(s)
    pick_pool   = [s for s in sends
                   if s.get("is_pick") or s.get("position") == "PICK"]

    if archetype == "distribute":
        return [player_pool[:1]] if player_pool else []

    if archetype == "consolidate":
        # Consolidate = trade up: always send 2+ assets, never 1-for-1. Every
        # package is measured on effective value, so bundling three bodies for
        # one star still has to clear the star's price after the depth penalty.
        # Priority: 2 players → 3 players → 2 players + pick
        results_c: List[List[Dict]] = []

        # 1. Best 2-player package
        best2, best2_d = None, float("inf")
        for a, b in combinations(player_pool, 2):
            if a["position"] == "QB" and b["position"] == "QB":
                continue
            if _in_band(a, b) and _tier_ok(a, b) and _dist(a, b) < best2_d:
                best2_d, best2 = _dist(a, b), [a, b]
        if best2:
            results_c.append(best2)

        # 2. Best 3-player package (must differ from pkg 1)
        if len(results_c) < max_pkgs:
            best3, best3_d = None, float("inf")
            used_pids = {p["player_id"] for pkg in results_c for p in pkg}
            for a, b, c in combinations(player_pool[:8], 3):
                if sum(1 for x in (a, b, c) if x["position"] == "QB") > 1:
                    continue
                pkg_pids = {a["player_id"], b["player_id"], c["player_id"]}
                if (_in_band(a, b, c) and _tier_ok(a, b, c)
                        and _dist(a, b, c) < best3_d and not pkg_pids.issubset(used_pids)):
                    best3_d, best3 = _dist(a, b, c), [a, b, c]
            if best3:
                results_c.append(best3)

        # 3. Best 2-player + 1 pick
        if len(results_c) < max_pkgs and pick_pool:
            best_pp, best_pp_d = None, float("inf")
            for pk in pick_pool:
                for a, b in combinations(player_pool[:8], 2):
                    if a["position"] == "QB" and b["position"] == "QB":
                        continue
                    if _in_band(a, b, pk) and _tier_ok(a, b, pk) and _dist(a, b, pk) < best_pp_d:
                        best_pp_d, best_pp = _dist(a, b, pk), [a, b, pk]
            if best_pp:
                results_c.append(best_pp)

        # Fallback: widen window, still require 2+ assets and a tier-comparable
        # headliner, closest on effective value.
        if not results_c:
            fallback2, fallback2_d = None, float("inf")
            for a, b in combinations(player_pool, 2):
                if a["position"] == "QB" and b["position"] == "QB":
                    continue
                if _tier_ok(a, b) and _dist(a, b) < fallback2_d:
                    fallback2_d, fallback2 = _dist(a, b), [a, b]
            if fallback2:
                results_c.append(fallback2)

        return _dedup(_drop_underpays(results_c))[:max_pkgs]

    results: List[List[Dict]] = []

    # ── 1. Player-only: exhaustive 1 / 2 / 3-player search (effective) ───────
    best, best_d = None, float("inf")
    for c in player_pool:
        if _in_band(c) and _tier_ok(c) and _dist(c) < best_d:
            best_d, best = _dist(c), [c]
    for a, b in combinations(player_pool, 2):
        if a["position"] == "QB" and b["position"] == "QB":
            continue
        if _in_band(a, b) and _tier_ok(a, b) and _dist(a, b) < best_d:
            best_d, best = _dist(a, b), [a, b]
    for a, b, c in combinations(player_pool[:7], 3):
        if sum(1 for x in (a, b, c) if x["position"] == "QB") > 1:
            continue
        if _in_band(a, b, c) and _tier_ok(a, b, c) and _dist(a, b, c) < best_d:
            best_d, best = _dist(a, b, c), [a, b, c]
    if best:
        results.append(best)

    # ── 2. Player(s) + 1 pick ────────────────────────────────────────────────
    if pick_pool and len(results) < max_pkgs:
        mp = player_pool[:8]
        best_pp, best_pp_d = None, float("inf")
        for pk in pick_pool:
            # Just the pick alone
            if _in_band(pk) and _dist(pk) < best_pp_d:
                best_pp_d, best_pp = _dist(pk), [pk]
            for c in mp:
                if _in_band(c, pk) and _tier_ok(c, pk) and _dist(c, pk) < best_pp_d:
                    best_pp_d, best_pp = _dist(c, pk), [c, pk]
            for a, b in combinations(mp, 2):
                if a["position"] == "QB" and b["position"] == "QB":
                    continue
                if _in_band(a, b, pk) and _tier_ok(a, b, pk) and _dist(a, b, pk) < best_pp_d:
                    best_pp_d, best_pp = _dist(a, b, pk), [a, b, pk]
        if best_pp:
            results.append(best_pp)

    # ── 3. Player(s) + 2 picks ───────────────────────────────────────────────
    if len(pick_pool) >= 2 and len(results) < max_pkgs:
        mp = player_pool[:6]
        best_2p, best_2p_d = None, float("inf")
        for pk1, pk2 in combinations(pick_pool, 2):
            # Two picks alone
            if _in_band(pk1, pk2) and _dist(pk1, pk2) < best_2p_d:
                best_2p_d, best_2p = _dist(pk1, pk2), [pk1, pk2]
            # One player + two picks
            for c in mp:
                if _in_band(c, pk1, pk2) and _tier_ok(c, pk1, pk2) and _dist(c, pk1, pk2) < best_2p_d:
                    best_2p_d, best_2p = _dist(c, pk1, pk2), [c, pk1, pk2]
        if best_2p:
            results.append(best_2p)

    # ── Fallback: closest-effective across all assets if window missed ────────
    if not results:
        all_pool = (player_pool + pick_pool)[:15]
        fallback, fallback_d = None, float("inf")
        for c in all_pool:
            if _tier_ok(c) and _dist(c) < fallback_d:
                fallback_d, fallback = _dist(c), [c]
        for a, b in combinations(all_pool, 2):
            if a.get("position") == "QB" and b.get("position") == "QB":
                continue
            if _tier_ok(a, b) and _dist(a, b) < fallback_d:
                fallback_d, fallback = _dist(a, b), [a, b]
        for a, b, c in combinations(all_pool[:7], 3):
            if sum(1 for x in (a, b, c) if x["position"] == "QB") > 1:
                continue
            if _tier_ok(a, b, c) and _dist(a, b, c) < fallback_d:
                fallback_d, fallback = _dist(a, b, c), [a, b, c]
        if fallback:
            results.append(fallback)

    return _dedup(_drop_underpays(results))[:max_pkgs]


# ── Pick send candidates ──────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _pick_send_candidates(
    picks: List[Dict],
    num_teams: int,
    slot_map: Optional[Dict[int, int]] = None,
    current_season: int = 0,
) -> List[Dict[str, Any]]:
    """Convert future picks to send-candidate dicts.

    Uses the pre-computed slot_map (roster_id → draft slot) built by the
    existing build_historical_pick_slot_map() so picks display as '2026 1.04'
    when an exact slot is known.
    """
    if not picks:
        return []
    pick_tbl: Dict[str, float] = {}
    try:
        from dashboard_services.picks import load_pick_value_table
        pick_tbl = load_pick_value_table(league_teams=num_teams) or {}
    except Exception:
        pick_tbl = {}

    third = max(1, num_teams // 3)

    out = []
    for pk in picks:
        if not isinstance(pk, dict):
            continue
        season = str(pk.get("season") or pk.get("year") or "")
        rnd    = int(pk.get("round") or 0)
        if not season or rnd <= 0:
            continue

        # Resolve exact slot from pre-computed map - only valid for the current draft year
        slot   = 0
        bucket = "mid"
        pick_yr = int(season) if season else 0
        if slot_map and current_season and pick_yr == current_season:
            try:
                orig_rid = int(
                    pk.get("original_roster_id")
                    or pk.get("original_owner")   # field name used by service.py
                    or pk.get("roster_id")
                    or 0
                )
                slot = slot_map.get(orig_rid, 0)
            except (ValueError, TypeError):
                slot = 0
        if slot > 0:
            if slot <= third:
                bucket = "early"
            elif slot <= 2 * third:
                bucket = "mid"
            else:
                bucket = "late"

        # Look up value: slot-specific key first, then bucket, then fallbacks
        val  = 0.0
        keys: List[str] = []
        if slot:
            keys.append(f"{season}_{rnd}_{slot:02d}")
        keys += [f"{season}_{rnd}_{bucket}", f"{season}_{rnd}_mid",
                 f"{season}_{rnd}", f"{season}_{rnd}_early"]
        for key in keys:
            if key in pick_tbl and float(pick_tbl[key]) > 0:
                val = float(pick_tbl[key])
                break
        if val <= 0:
            val = {1: 650.0, 2: 220.0}.get(rnd, 80.0)

        name = f"{season} {rnd}.{slot:02d}" if slot else f"{season} {_ordinal(rnd)} (Mid)"
        uid  = f"pick_{season}_{rnd}_{slot:02d}" if slot else f"pick_{season}_{rnd}"

        out.append({
            "player_id":   uid,
            "name":        name,
            "position":    "PICK",
            "value":       round(val, 1),
            "is_pick":     True,
            "pick_season": season,
            "pick_round":  rnd,
            "pick_slot":   slot,
            "pick_bucket": bucket,
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

def _compute_lineup_score(
    pids: List[str],
    use_ppg: bool,
    ppg_map,
    pos_map,
    roster_positions,
    values_by_id,
    league_type,
) -> float:
    """Score a lineup either by projected PPG or by optimal trade value.

    Shared dispatch extracted from the identical ``_lineup_score`` closures in
    ``_build_distribute`` and ``_build_rebuilding``; each caller still binds its
    own request-scoped context, so behavior is unchanged.
    """
    if use_ppg:
        return _ppg_lineup(pids, ppg_map, pos_map or {}, roster_positions)  # type: ignore[arg-type]
    return _optimal_lineup_value(pids, values_by_id, league_type, use_redraft=True)


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
    current_wp: float = 0.5,
    num_weeks: int = 14,
    num_teams: int = 10,
    playoff_spots: int = 4,
    viewer_pos_counts: Optional[Dict[str, int]] = None,
    ppg_map: Optional[Dict[str, Any]] = None,
    pos_map: Optional[Dict[str, str]] = None,
    roster_positions: Optional[List[str]] = None,
    sim_state: Optional[Dict] = None,
    current_playoff_pct: float = 0.0,
    viewer_roster_id: Any = None,
) -> List[Dict[str, Any]]:
    """
    Viewer sends one concentrated stud and receives a 2–3 player depth package.
    Each card = one stud → one partner's multi-player return.
    """
    # League-wide value ladder for effective-value (depth-penalty) math. Here the
    # partner is the side sending more bodies (the depth package), so their
    # return is what absorbs the bench penalty.
    _sv = sorted((v for v in (_f(x.get("value")) for x in values_by_id.values()) if v > 0),
                 reverse=True)
    studs = sorted(
        [p for p in viewer_players
         if values_by_id.get(p, {}).get("position") in SKILL_POS
         and _f(values_by_id[p].get("value")) >= 600
         and (not untouchable_ids or p not in untouchable_ids)],
        key=lambda p: _f(values_by_id[p].get("value")),
        reverse=True,
    )[:5]

    results: List[Dict[str, Any]] = []
    used_owners: set = set()

    for stud in studs:
        sval  = _f(values_by_id[stud].get("value"))
        sname = values_by_id[stud].get("name", "")
        spos  = values_by_id[stud].get("position", "")
        # Viewer SENDS the stud and RECEIVES this depth package, so the low end
        # is an underpay against the viewer - keep it tight (≥96%). Receiving a
        # modest depth premium is fine, so the high end stays a touch generous.
        lo, hi = sval * 0.96, sval * 1.18

        # In 1QB leagues a second QB has no FLEX slot and contributes nothing to
        # the lineup if the viewer already has a QB. RB/WR/TE all remain eligible
        # because they can fill FLEX spots.
        saturated: set = set()
        if league_type != "sf" and (viewer_pos_counts or {}).get("QB", 0) >= 1:
            saturated.add("QB")

        # Collect best combo per owner, then take top-3 value matches
        owner_bests: List[Tuple[str, List[Dict], float]] = []
        for owner, pool in targets_by_owner.items():
            if owner in used_owners:
                continue
            cand = sorted(
                [p for p in pool if p["position"] not in saturated],
                key=lambda x: x["value"], reverse=True
            )[:8]
            local_best: Optional[Tuple[str, List[Dict], float]] = None
            for n in (2, 3):
                for combo in combinations(cand, n):
                    s = sum(c["value"] for c in combo)
                    if lo <= s <= hi:
                        diff = abs(s - sval)
                        if local_best is None or diff < local_best[2]:
                            local_best = (owner, list(combo), diff)
            if local_best:
                owner_bests.append(local_best)

        use_ppg = bool(ppg_map and roster_positions)

        def _lineup_score(pids: List[str]) -> float:
            return _compute_lineup_score(
                pids, use_ppg, ppg_map, pos_map, roster_positions, values_by_id, league_type
            )

        # Departure cost: what happens to win% if you just lose this stud
        dep_players = [p for p in viewer_players if p != stud]
        dep_lineup  = _lineup_score(dep_players)
        dep_wpd     = _win_prob(dep_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)
        dep_pod     = (_playoff_odds(current_wp + dep_wpd, num_weeks, num_teams, playoff_spots)
                       - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots))

        # Re-score each owner's best combo by how much it improves the lineup
        scored_bests: List[Tuple[str, List[Dict], float, float]] = []
        for owner, combo, diff in owner_bests:
            recv_ids_trial = [c["player_id"] for c in combo]
            lineup_gain = _lineup_score(dep_players + recv_ids_trial) - dep_lineup
            scored_bests.append((owner, combo, diff, lineup_gain))
        # Primary sort: lineup improvement descending; secondary: value closeness ascending
        scored_bests.sort(key=lambda x: (-x[3], x[2]))
        owner_bests = [(o, c, d) for o, c, d, _ in scored_bests]

        for owner, combo, _ in owner_bests[:3]:
            used_owners.add(owner)

            recv_ids    = [c["player_id"] for c in combo]
            new_players = dep_players + recv_ids
            new_lineup  = _lineup_score(new_players)
            net_wpd     = _win_prob(new_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)

            # Playoff odds: re-run Monte Carlo with swapped roster when sim state available
            if sim_state and viewer_roster_id is not None:
                try:
                    from data_building.simulate_playoff_odds import simulate_with_swap as _sim_swap
                    new_po_pct, _ = _sim_swap(
                        sim_state, int(viewer_roster_id), new_players, n_sims=10_000
                    )
                    net_pod = (new_po_pct - current_playoff_pct) / 100.0
                except Exception:
                    net_pod = _playoff_odds(current_wp + net_wpd, num_weeks, num_teams, playoff_spots) \
                              - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)
            else:
                net_pod = _playoff_odds(current_wp + net_wpd, num_weeks, num_teams, playoff_spots) \
                          - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)

            recv_val = sum(c["value"] for c in combo)
            # Partner ships the multi-player package, so their outgoing value is
            # discounted by the depth penalty - the same effective view the trade
            # card shows. This makes the depth return look (correctly) more
            # attractive for them to give up.
            recv_eff = recv_val - _depth_penalty(max(0, len(combo) - 1), _sv, num_teams)
            acpt    = _estimate_acceptance(sval, recv_eff, is_preferred=True)

            pname  = _roster_name(roster_map, owner)
            p_arch = owner_meta.get(owner, {}).get("arch", "")
            ceiling_note = "lineup ceiling rises" if net_wpd >= 0 else "adds depth but trims your ceiling"

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
                "fit_note":       _fit_note([{"position": spos}],
                                            owner_meta.get(owner, {}).get("need"), None, None),
                "partner_team":   pname,
                "partner_arch":   p_arch,
                # departure cost: impact table (cost of losing this stud alone)
                "win_prob_delta":      round(dep_wpd, 4),
                "playoff_odds_delta":  round(dep_pod, 4),
                # net trade impact: shown on the card (lose stud + gain depth package)
                "net_win_prob_delta":      round(net_wpd, 4),
                "net_playoff_odds_delta":  round(net_pod, 4),
                "acceptance_pct":     acpt,
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
            if len(results) >= 15:
                break
        if len(results) >= 15:
            break

    results.sort(key=_suggestion_rank, reverse=True)
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
    current_wp: float = 0.5,
    num_weeks: int = 14,
    num_teams: int = 10,
    playoff_spots: int = 4,
    picks_by_owner: Optional[Dict[str, List[Dict]]] = None,
    ppg_map: Optional[Dict[str, Any]] = None,
    pos_map: Optional[Dict[str, str]] = None,
    roster_positions: Optional[List[str]] = None,
    sim_state: Optional[Dict] = None,
    current_playoff_pct: float = 0.0,
    viewer_roster_id: Any = None,
    owner_meta: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Rebuild = sell win-now vets for younger assets of similar dynasty value.

    Considers three return types for each vet:
      1. Young player-for-vet swap
      2. Pure draft pick(s) (sell high while the pick cost is real)
      3. Young player + draft pick combo
    Whichever best matches the vet's value wins.
    """
    # League-wide value ladder for effective-value (depth-penalty) math. The
    # partner ships the youth/pick return, so they are the multi-asset side that
    # absorbs the depth penalty.
    _sv = sorted((v for v in (_f(x.get("value")) for x in values_by_id.values()) if v > 0),
                 reverse=True)
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

    # Flatten rival picks: {owner_rid: [pick_dict, ...]}
    pbo = picks_by_owner or {}

    old_pids = [t["player_id"] for t in young]
    old_vals = _get_30d_values(old_pids) if old_pids else {}

    results: List[Dict[str, Any]] = []
    used_targets: set = set()

    for vet in vets:
        vval  = _f(values_by_id[vet].get("value"))
        vname = values_by_id[vet].get("name", "")
        vpos  = values_by_id[vet].get("position", "")
        # Viewer SENDS the vet and RECEIVES youth/picks. The low end is an
        # underpay against the viewer - keep it tight (≥92%); a small discount
        # is acceptable when selling a declining vet for upside. The high end
        # stays generous since receiving more (youth/pick premium) helps the viewer.
        lo, hi = vval * 0.92, vval * 1.22

        # ── Build candidate receive options ──────────────────────────────
        options: List[Dict] = []  # each: {diff, recv_assets, partner_rid, recv_player_ids}

        # Option A: single young player
        for t in young:
            if t["player_id"] in used_targets or not (lo <= t["value"] <= hi):
                continue
            options.append({
                "diff":          abs(t["value"] - vval),
                "recv_assets":   [t],
                "partner_rid":   t["owner_roster_id"],
                "partner_name":  t["partner_name"],
                "partner_arch":  t["partner_arch"],
                "recv_pids":     [t["player_id"]],
                "primary":       t,  # used for impact-table display and why-text
            })

        # Option B: single pick (any rival)
        for owner_rid, owner_picks in pbo.items():
            for pk in owner_picks:
                if not (lo <= pk["value"] <= hi):
                    continue
                p_meta = {}
                # Look up partner meta from all_targets or owner_meta if available
                for t in all_targets:
                    if t.get("owner_roster_id") == owner_rid:
                        p_meta = {"partner_name": t["partner_name"],
                                  "partner_arch": t["partner_arch"]}
                        break
                options.append({
                    "diff":         abs(pk["value"] - vval),
                    "recv_assets":  [pk],
                    "partner_rid":  owner_rid,
                    "partner_name": p_meta.get("partner_name", f"Team {owner_rid}"),
                    "partner_arch": p_meta.get("partner_arch", ""),
                    "recv_pids":    [],
                    "primary":      None,  # pick-only - no young player to display
                })

        # Option C: young player + pick from same owner
        for t in young:
            if t["player_id"] in used_targets:
                continue
            owner_rid = t["owner_roster_id"]
            for pk in pbo.get(owner_rid, []):
                combo_val = t["value"] + pk["value"]
                if not (lo <= combo_val <= hi):
                    continue
                options.append({
                    "diff":         abs(combo_val - vval),
                    "recv_assets":  [t, pk],
                    "partner_rid":  owner_rid,
                    "partner_name": t["partner_name"],
                    "partner_arch": t["partner_arch"],
                    "recv_pids":    [t["player_id"]],
                    "primary":      t,
                })

        if not options:
            continue

        options.sort(key=lambda x: x["diff"])

        use_ppg = bool(ppg_map and roster_positions)

        def _reb_lineup_score(pids: List[str]) -> float:
            return _compute_lineup_score(
                pids, use_ppg, ppg_map, pos_map, roster_positions, values_by_id, league_type
            )

        # ── Departure stats - computed once per vet, shared by all options ─
        dep_players   = [p for p in viewer_players if p != vet]
        dep_lineup    = _reb_lineup_score(dep_players)
        departure_wpd = _win_prob(dep_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)
        dep_wp        = current_wp + departure_wpd
        departure_pod = _playoff_odds(dep_wp, num_weeks, num_teams, playoff_spots) \
                      - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)

        # ── Emit up to 3 suggestions per vet (best options by value diff) ─
        for opt in options[:10]:
            if any(pid in used_targets for pid in opt["recv_pids"]):
                continue

            for pid in opt["recv_pids"]:
                used_targets.add(pid)

            # ── Win-prob for this specific receive package ─────────────────
            if opt["recv_pids"]:
                net_players = dep_players + opt["recv_pids"]
                net_lineup  = _reb_lineup_score(net_players)
                wpd = _win_prob(net_lineup, league_avg) - _win_prob(viewer_lineup_val, league_avg)
            else:
                wpd = departure_wpd  # pick-only: no immediate lineup improvement

            # Playoff odds: re-run Monte Carlo with swapped roster when sim state available
            if sim_state and viewer_roster_id is not None and opt["recv_pids"]:
                try:
                    from data_building.simulate_playoff_odds import simulate_with_swap as _sim_swap
                    new_po_pct, _ = _sim_swap(
                        sim_state, int(viewer_roster_id), net_players, n_sims=10_000
                    )
                    net_pod = (new_po_pct - current_playoff_pct) / 100.0
                except Exception:
                    net_pod = _playoff_odds(current_wp + wpd, num_weeks, num_teams, playoff_spots) \
                              - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)
            else:
                net_pod = _playoff_odds(current_wp + wpd, num_weeks, num_teams, playoff_spots) \
                          - _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)

            recv_total = sum(a.get("value", 0) for a in opt["recv_assets"])
            is_pref    = opt["partner_arch"] == COMPLEMENT.get("rebuilding", "")
            # Partner sends the multi-asset youth/pick return, so discount it by
            # the depth penalty (effective value) for the acceptance estimate.
            recv_eff   = recv_total - _depth_penalty(
                max(0, len(opt["recv_assets"]) - 1), _sv, num_teams)
            acpt       = _estimate_acceptance(vval, recv_eff, is_preferred=is_pref)

            # ── Display / why text ─────────────────────────────────────────
            primary = opt["primary"]
            if primary:
                tp  = _trend_pct(primary["player_id"], primary["value"], old_vals, values_by_id)
                primary["win_prob_delta"] = departure_wpd
                why = _build_why(primary, "rebuilding", tp, wpd)
                display_pid  = primary["player_id"]
                display_name = primary["name"]
                display_pos  = primary["position"]
                display_age  = primary.get("age", 0)
                display_val  = round(primary["value"], 1)
                display_rdft = round(primary.get("redraft_value", 0), 1)
                display_rank = primary.get("pos_rank_label", "")
            else:
                pk = opt["recv_assets"][0]
                why = (f"Sell {vname} while value is high - receive {pk['name']} "
                       f"from {opt['partner_name']}.")
                display_pid  = pk["player_id"]
                display_name = pk["name"]
                display_pos  = "PICK"
                display_age  = 0
                display_val  = round(pk["value"], 1)
                display_rdft = 0.0
                display_rank = ""

            # ── Build suggested_receive ────────────────────────────────────
            suggested_receive = []
            for a in opt["recv_assets"]:
                if a.get("is_pick"):
                    parts = a["player_id"].split("_")  # pick_YYYY_R
                    suggested_receive.append({
                        "player_id":   a["player_id"],
                        "name":        a["name"],
                        "position":    "PICK",
                        "value":       round(a["value"], 1),
                        "is_pick":     True,
                        "pick_season": parts[1] if len(parts) > 1 else "",
                        "pick_round":  int(parts[2]) if len(parts) > 2 else 1,
                    })
                else:
                    suggested_receive.append({
                        "player_id": a["player_id"],
                        "name":      a["name"],
                        "position":  a["position"],
                        "value":     round(a["value"], 1),
                    })

            results.append({
                "player_id":      display_pid,
                "name":           display_name,
                "position":       display_pos,
                "nfl_team":       primary["team"] if primary else "",
                "age":            display_age,
                "value":          display_val,
                "redraft_value":  display_rdft,
                "pos_rank_label": display_rank,
                "why":            why,
                "fit_note":       _fit_note(
                    [{"position": vpos}],
                    (owner_meta or {}).get(str(opt.get("partner_rid")), {}).get("need"),
                    None, None),
                "partner_team":   opt["partner_name"],
                "partner_arch":   opt["partner_arch"],
                "win_prob_delta":         round(departure_wpd, 4),
                "net_win_prob_delta":     round(wpd, 4),
                "playoff_odds_delta":     round(departure_pod, 4),
                "net_playoff_odds_delta": round(net_pod, 4),
                "acceptance_pct":         acpt,
                "direction":              "acquire",
                "suggested_send": [{
                    "player_id": vet, "name": vname,
                    "position": vpos, "value": round(vval, 1),
                }],
                "suggested_receive": suggested_receive,
            })
            if len(results) >= 15:
                break

        if len(results) >= 15:
            break

    results.sort(key=_suggestion_rank, reverse=True)
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

    # Build pick-slot map using the existing app.py logic (roster_id → draft slot)
    slot_map: Dict[int, int] = {}
    try:
        from app import build_historical_pick_slot_map
        slot_map = build_historical_pick_slot_map(
            platform=platform,
            root_league_id=league_id,
            current_season=season,
            source_season=season - 1,
        ) or {}
    except Exception as exc:
        log.debug("[archetype] pick slot map skipped: %s", exc)
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

    # ── Build simulation state (same logic as playoff odds simulator) ─────────
    # Handles preseason (pure FP projections) and in-season (blended) automatically.
    # Sim state + base odds are cached per league for 5 minutes so switching
    # between archetype chips feels instant.
    sim_state: Optional[Dict] = None
    current_playoff_pct: float = 0.0  # viewer's current playoff % (0–100)
    ppg_map:  Dict[str, Any] = {}
    pos_map:  Dict[str, str] = {}
    roster_positions: List[str] = ctx.get("roster_positions") or []
    _cache_key = f"{platform}:{league_id}:{season}"
    try:
        from data_building.simulate_playoff_odds import (
            build_sim_state as _build_sim_state,
            run_base_simulation as _run_base_sim,
            build_ppg_map as _build_ppg_map,
        )
        _cached = _SIM_CACHE.get(_cache_key)
        if _cached and (_time.time() - _cached["ts"]) < _SIM_CACHE_TTL:
            sim_state  = _cached["sim_state"]
            base_odds  = _cached["base_odds"]
            log.debug("[archetype] sim cache hit for %s", _cache_key)
        else:
            sim_state = _build_sim_state(ctx, platform=platform)
            base_odds = _run_base_sim(sim_state, n_sims=10_000) if sim_state else {}
            _SIM_CACHE[_cache_key] = {"sim_state": sim_state, "base_odds": base_odds, "ts": _time.time()}
            log.debug("[archetype] sim cache miss, built fresh for %s", _cache_key)
        if sim_state:
            ppg_map  = sim_state["ppg_map"]
            pos_map  = sim_state["pos_map"]
            roster_positions = sim_state["roster_positions"]
            vid = int(viewer_roster_id) if str(viewer_roster_id).isdigit() else viewer_roster_id
            current_playoff_pct = base_odds.get(vid, 0.0)
            log.debug("[archetype] viewer playoff_pct=%.1f", current_playoff_pct)
        else:
            ppg_map, pos_map = _build_ppg_map(ctx)
    except Exception as exc:
        log.debug("[archetype] sim state unavailable, using analytical model: %s", exc)
        try:
            from data_building.simulate_playoff_odds import build_ppg_map as _build_ppg_map
            ppg_map, pos_map = _build_ppg_map(ctx)
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    def _lval(pids: List[str]) -> float:
        if ppg_map and roster_positions:
            return _ppg_lineup(pids, ppg_map, pos_map, roster_positions)
        return _optimal_lineup_value(pids, values_by_id, league_type, use_redraft=True)

    viewer_lineup_val = _lval(viewer_players)

    def _impact_roster(pid: str, pos: str) -> List[str]:
        """The hypothetical roster if the viewer acquires this target: add it and
        drop the weakest same-position player (by PPG when available, else value)."""
        same = [p for p in viewer_players
                if str(values_by_id.get(p, {}).get("position") or "").upper() == pos]
        if not same:
            return viewer_players + [pid]
        if ppg_map and roster_positions:
            drop = min(same, key=lambda p: (ppg_map.get(str(p)) or {}).get("ppg", 0))
        else:
            drop = min(same, key=lambda p: _f(values_by_id.get(p, {}).get("value")))
        return [p for p in viewer_players if p != drop] + [pid]

    # Count viewer's skill-position players per position (used by distribute)
    viewer_pos_counts: Dict[str, int] = {}
    for pid in viewer_players:
        pos = str(values_by_id.get(pid, {}).get("position") or "").upper()
        if pos in SKILL_POS:
            viewer_pos_counts[pos] = viewer_pos_counts.get(pos, 0) + 1

    # League-average lineup value (PPG-based, matching playoff odds simulator)
    if sim_state:
        # Use the simulator's own team avgs so league_avg matches the simulation baseline
        league_avg = sum(t["avg"] for t in sim_state["teams"]) / max(1, len(sim_state["teams"]))
    else:
        lineup_vals = [_lval([str(p) for p in (r.get("players") or [])]) for r in rosters]
        league_avg = sum(lineup_vals) / max(1, len(lineup_vals)) if lineup_vals else viewer_lineup_val

    settings  = ctx.get("settings") or {}
    num_weeks = max(10, int(settings.get("playoff_week_start", 15)) - 1)

    current_wp = _win_prob(viewer_lineup_val, league_avg)
    current_po = current_playoff_pct / 100.0  # use sim-based odds when available

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
        # Positions the partner is thin at (would welcome) vs deep at (won't want
        # more of). Used to bias which of our players we send and to nudge the
        # acceptance estimate - a fair-value deal a partner has no roster use for
        # is not actually a likely deal.
        p_need   = {pos for pos in SKILL_POS if p_pos_counts.get(pos, 0) < 3}
        p_stacked = {pos for pos in SKILL_POS if p_pos_counts.get(pos, 0) >= 4}
        owner_meta[rid] = {
            "arch": p_arch, "name": p_name, "phrase": p_phrase,
            "pos_counts": p_pos_counts, "need": p_need, "stacked": p_stacked,
        }

        for pid in pids:
            if pid in viewer_players:
                continue  # never suggest the viewer's own players as targets
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
        _sugg = _build_distribute(
            viewer_players, values_by_id, targets_by_owner, owner_meta,
            roster_map, league_type, viewer_lineup_val, league_avg,
            untouchable_ids=untouchable_ids,
            current_wp=current_wp, num_weeks=num_weeks,
            num_teams=num_teams, playoff_spots=playoff_spots,
            viewer_pos_counts=viewer_pos_counts,
            ppg_map=ppg_map, pos_map=pos_map, roster_positions=roster_positions,
            sim_state=sim_state,
            current_playoff_pct=current_playoff_pct,
            viewer_roster_id=viewer_roster_id,
        )
        return {"suggestions": _sugg, "current_playoff_pct": round(current_playoff_pct, 1)}

    # ── Rebuilding: viewer sells a win-now vet for youth / picks ─────────────
    if archetype == "rebuilding":
        # Build rival picks pool for player+pick and pick-only receive options
        picks_by_owner: Dict[str, List[Dict]] = {}
        for rid, pick_list in picks_by_roster.items():
            if str(rid) == str(viewer_roster_id):
                continue
            converted = _pick_send_candidates(pick_list, num_teams, slot_map, current_season=season)
            if converted:
                picks_by_owner[str(rid)] = converted
        _sugg = _build_rebuilding(
            viewer_players, values_by_id, all_targets,
            league_type, viewer_lineup_val, league_avg,
            untouchable_ids=untouchable_ids,
            current_wp=current_wp, num_weeks=num_weeks,
            num_teams=num_teams, playoff_spots=playoff_spots,
            picks_by_owner=picks_by_owner,
            ppg_map=ppg_map, pos_map=pos_map, roster_positions=roster_positions,
            sim_state=sim_state,
            current_playoff_pct=current_playoff_pct,
            viewer_roster_id=viewer_roster_id,
            owner_meta=owner_meta,
        )
        return {"suggestions": _sugg, "current_playoff_pct": round(current_playoff_pct, 1)}

    # ── 30-day trend ──────────────────────────────────────────────────────────
    all_pids = list({t["player_id"] for t in all_targets} | set(viewer_players))
    old_vals = _get_30d_values(all_pids)

    # ── Score & filter targets per archetype ──────────────────────────────────
    scored: List[Tuple[float, Dict]] = []
    peak_pos = PEAK_AGE

    # Owner positional depth rank: 1 = the owner's BEST player at that position.
    # Teams keep their stud at a spot and move the depth behind it, so a surplus
    # makes a rival's RB3/RB4 available - not their RB1.
    owner_depth_rank: Dict[str, int] = {}
    _by_owner_pos: Dict[tuple, list] = {}
    for _t in all_targets:
        _by_owner_pos.setdefault((_t["owner_roster_id"], _t["position"]), []).append(_t)
    for _lst in _by_owner_pos.values():
        for _i, _tt in enumerate(sorted(_lst, key=lambda x: -_f(x.get("value"))), start=1):
            owner_depth_rank[_tt["player_id"]] = _i

    # League value ladder (drives the depth penalty) + the viewer's best sendable
    # assets, used to decide whether they can actually afford a rival's stud.
    sorted_vals = sorted(
        (v for v in (_f(x.get("value")) for x in values_by_id.values()) if v > 0),
        reverse=True,
    )
    _send_vals = sorted(
        (_f(values_by_id.get(p, {}).get("value")) for p in viewer_players
         if str(values_by_id.get(p, {}).get("position") or "").upper() in SKILL_POS
         and not (untouchable_ids and p in untouchable_ids)),
        reverse=True,
    )

    # Value-drop tier boundaries for this league, so consolidate can reason in
    # tiers ("don't trade down too many tiers to reach a stud") rather than raw
    # value alone. values_by_id["value"] is already league-type-adjusted, so let
    # the thresholder key off "value" (its default). Falls back to the static
    # thresholds on a thin value table.
    try:
        tier_thresholds = compute_tier_thresholds(
            list(values_by_id.values())) or FALLBACK_THRESHOLDS
    except Exception:
        tier_thresholds = FALLBACK_THRESHOLDS

    def _can_afford_stud(target_val: float) -> bool:
        """True when the viewer holds enough top-tier value to assemble a package
        whose depth-adjusted total clears the consolidate band for this target -
        i.e. a loaded team really can consolidate into a #1 like Bijan/Gibbs. The
        viewer's headliner must also be tier-comparable to the stud, so a team
        can't reach a #1 by stacking depth that sits too many tiers below it."""
        lo = target_val * _acquire_band(target_val, "consolidate")[0]
        sv = _send_vals
        if not sv:
            return False
        # Headliner has to be within the consolidate tier band of the target.
        if asset_tier(sv[0], tier_thresholds) - asset_tier(target_val, tier_thresholds) > _CONSOLIDATE_MAX_TIER_DROP:
            return False
        if len(sv) >= 2 and (sv[0] + sv[1] - _depth_penalty(1, sorted_vals, num_teams)) >= lo:
            return True
        if len(sv) >= 3 and (sv[0] + sv[1] + sv[2] - _depth_penalty(2, sorted_vals, num_teams)) >= lo:
            return True
        return False

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
            # Consolidating doesn't only mean chasing the very top tier: taking a
            # few lesser pieces up to a solid mid-tier starter is a real upgrade.
            # Allow mid-tier targets in (floor well below the old 600) and soften
            # the raw-value term so a productive ~450-650 player competes with a
            # pure-value stud instead of always being buried behind it.
            if val < 350:
                continue
            rdft_ratio = rdft / max(1, val) if rdft > 0 else 0.6
            score = (
                0.42 * min(1.0, val / 900) +
                0.30 * rdft_ratio +
                0.13 * min(1.0, val / 1400) +
                0.15 * (1.0 if tp >= 0 else 0.7)
            ) * sc * pref_bonus

        if score is None:
            continue

        # Win-prob delta of acquiring this target (add it, drop weakest same-pos).
        # Score the *new lineup* with the same _lval basis as the current lineup
        # so both sides are in the same units. The old _wp_delta added the
        # target's raw dynasty value onto a lineup/PPG-scale total, which mixed
        # scales and saturated the win-prob curve to an absurd ~+50% for every
        # strong target. This is also the fallback the Impact table shows when no
        # sim state is available.
        wpd = _win_prob(_lval(_impact_roster(pid, pos)), league_avg) - current_wp
        t["win_prob_delta"] = wpd

        # Availability: teams keep their best at a position and move the depth
        # behind it, so what a surplus frees up is a rival's RB3/RB4, not their
        # RB1. Key off the player's depth rank on their OWN roster, not just the
        # position count - a stud stays a keeper even on a stacked team.
        _pcounts = owner_meta.get(str(t["owner_roster_id"]), {}).get("pos_counts") or {}
        _drank = owner_depth_rank.get(pid, 1)
        if archetype == "consolidate" and _drank <= 1:
            # A rival's stud (their best at the spot) is only a realistic
            # consolidate target if the viewer is loaded enough to package
            # top-tier assets for it. If they can't afford a fair offer, don't
            # suggest it at all; when they can, it competes on merit.
            if not _can_afford_stud(val):
                continue
            # Prefer the stud over the owner's own depth (which the (owner, pos)
            # dedup would otherwise pick): a loaded team wants Bijan, not the RB4.
            avail = 1.30
        else:
            avail = _availability(_drank, _pcounts.get(pos, 0))

        if archetype == "consolidate":
            # Gear toward the USER's team: how much this target upgrades their
            # lineup (win-prob delta) weighs about as much as the player's raw
            # value/production, so the list isn't just the top-5 overall names -
            # it favors real upgrades at the user's weaker spots. fit is 0..1.
            fit = min(1.0, max(0.0, wpd) / 0.06)
            final = (0.55 * score + 0.45 * fit) * avail
        else:
            final = (score + wpd * 0.25) * avail

        scored.append((final, t))

    # ── Rank: surface a varied slate, not just the top overall names ──────────
    # One player per (team, position), and at most a few per position, so the
    # list spreads across positions/tiers instead of being five RB studs. Falls
    # back to filling remaining slots if the caps leave it thin.
    _MAX_TARGETS = 8
    _MAX_PER_POS = 3
    scored.sort(key=lambda x: x[0], reverse=True)
    seen_owner_pos: set = set()
    seen_players:   set = set()
    pos_in_top:     Dict[str, int] = {}
    top: List[Dict] = []
    for _, t in scored:
        if t["player_id"] in seen_players:
            continue
        key = (t["owner_roster_id"], t["position"])
        if key in seen_owner_pos:
            continue
        if pos_in_top.get(t["position"], 0) >= _MAX_PER_POS:
            continue
        seen_owner_pos.add(key)
        seen_players.add(t["player_id"])
        pos_in_top[t["position"]] = pos_in_top.get(t["position"], 0) + 1
        top.append(t)
        if len(top) >= _MAX_TARGETS:
            break

    # Relax the caps if the slate is thin.
    if len(top) < 4:
        for _, t in scored:
            if t["player_id"] not in seen_players and len(top) < _MAX_TARGETS:
                seen_players.add(t["player_id"])
                top.append(t)

    # ── Build send packages & assemble response ───────────────────────────────
    send_candidates = _score_sends(viewer_players, values_by_id, archetype, untouchable_ids=untouchable_ids)
    # Add viewer's draft picks to the send pool so packages can use them as
    # value-fillers (e.g. two players + a pick for consolidate).
    viewer_picks = picks_by_roster.get(str(viewer_roster_id)) or \
                   picks_by_roster.get(viewer_roster_id) or []
    send_candidates += _pick_send_candidates(viewer_picks, num_teams, slot_map, current_season=season)

    results = []
    new_wp_base = current_wp  # alias for clarity inside loop
    _vid_int: Optional[int] = None
    try:
        _vid_int = int(viewer_roster_id)
    except (TypeError, ValueError):
        logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

    # Per-request memo for the Monte Carlo swap. simulate_with_swap is
    # deterministic (common-random-numbers seed), so identical post-trade rosters
    # return identical odds - different packages that collapse to the same lineup
    # (or the same target reached two ways) then reuse one 10k-sim instead of
    # re-running it. Keyed by the viewer's resulting player set.
    _swap_cache: Dict[frozenset, Tuple[float, float]] = {}

    def _cached_swap(pids_after):
        key = frozenset(str(p) for p in pids_after)
        hit = _swap_cache.get(key)
        if hit is not None:
            return hit
        from data_building.simulate_playoff_odds import simulate_with_swap as _sim_swap
        res = _sim_swap(sim_state, _vid_int, pids_after, n_sims=10_000)
        _swap_cache[key] = res
        return res

    for t in top:
        pid = t["player_id"]
        pos = t["position"]
        tp  = _trend_pct(pid, t["value"], old_vals, values_by_id)

        # Per-target "impact" (win % / playoff odds if acquired) shown in the
        # Impact table: build the hypothetical roster (add target, drop the
        # weakest same-position player) and simulate it. This is the accurate
        # number; the analytical value stored on the target is the fallback when
        # no sim state exists. Routed through _cached_swap so identical rosters
        # reuse a single 10k run.
        new_pids = _impact_roster(pid, pos)

        # Analytical fallback, clamped to a plausible single-acquisition swing so
        # a thin value model can never surface an absurd number (a real trade
        # moves weekly win prob by at most ~20 points). The sim below overrides
        # this with the accurate value whenever sim state is available.
        wpd = max(-0.20, min(0.20, t.get("win_prob_delta", 0.0)))
        pod = _playoff_odds(new_wp_base + wpd, num_weeks, num_teams, playoff_spots) - current_po
        if sim_state is not None and _vid_int is not None:
            try:
                new_po_pct, new_avg = _cached_swap(new_pids)
                pod = (new_po_pct - current_playoff_pct) / 100.0
                wpd = _win_prob(new_avg, league_avg) - current_wp
            except Exception:
                logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

        why = _build_why(t, archetype, tp, wpd)

        # Bias the package toward positions this partner is thin at (and away
        # from ones they're stacked at) so the offer is one they'd actually use.
        _pmeta    = owner_meta.get(str(t["owner_roster_id"]), {})
        _p_need   = _pmeta.get("need")
        _p_stacked = _pmeta.get("stacked")
        pkgs = _select_packages(
            send_candidates, t["value"], archetype, max_pkgs=3,
            sorted_vals=sorted_vals, league_size=num_teams,
            need_positions=_p_need, stacked_positions=_p_stacked,
            tier_thresholds=tier_thresholds,
        )
        # An acquire suggestion with nothing to offer is not a suggestion. If no
        # value/tier-valid send package can be assembled for this target, drop it
        # rather than render a card with an empty "you give" side.
        if not pkgs or not any(pkg for pkg in pkgs):
            continue

        for pkg in pkgs:
            send_val = sum(p.get("value", 0) for p in pkg) if pkg else 0
            # Judge acceptance on the depth-adjusted (effective) value the trade
            # card shows, not the raw sum - a multi-piece package looks generous
            # in raw value but reads as fair once the bench penalty applies.
            eff_send = send_val - _depth_penalty(
                max(0, len(pkg) - 1), sorted_vals, num_teams) if pkg else 0
            recv_val = t["value"]
            acpt     = _estimate_acceptance(
                eff_send, recv_val, is_preferred=t["is_pref"],
                fit=_acceptance_fit(pkg, _p_need, _p_stacked))

            # Per-package net odds: build the exact post-trade roster (drop the
            # sent players, add the target) and re-run the SAME Monte Carlo
            # playoff simulation - same schedule, same opponents, same seed –
            # with only the viewer's lineup changed. This is the accurate path
            # and is consistent with the playoff-odds page. The analytical
            # formula is used only if the simulation is unavailable.
            pkg_player_pids = {str(a.get("player_id", "")) for a in pkg if a.get("player_id") and not a.get("is_pick")}
            net_roster = [p for p in viewer_players if str(p) not in pkg_player_pids] + [pid]

            net_pod_pkg = None
            net_wpd_pkg = None
            if sim_state is not None and _vid_int is not None:
                try:
                    _net_po_pct, _net_avg = _cached_swap(net_roster)
                    net_pod_pkg = (_net_po_pct - current_playoff_pct) / 100.0
                    net_wpd_pkg = _win_prob(_net_avg, league_avg) - current_wp
                except Exception:
                    net_pod_pkg = None

            if net_pod_pkg is None:
                # Analytical fallback (no sim state) - still per-package
                net_lval   = _lval(net_roster)
                net_wp_pkg = _win_prob(net_lval, league_avg)
                current_po_formula = _playoff_odds(current_wp, num_weeks, num_teams, playoff_spots)
                net_pod_pkg = _playoff_odds(net_wp_pkg, num_weeks, num_teams, playoff_spots) - current_po_formula
                net_wpd_pkg = net_wp_pkg - current_wp

            results.append({
                "player_id":      pid,
                "name":           t["name"],
                "position":       pos,
                "nfl_team":       t["team"],
                "age":            t["age"],
                "value":          round(t["value"], 1),
                "redraft_value":  round(t.get("redraft_value", 0), 1),
                "pos_rank_label": t["pos_rank_label"],
                "why":            why,
                "partner_team":   t["partner_name"],
                "partner_arch":   t["partner_arch"],
                "win_prob_delta":          round(wpd, 4),
                "playoff_odds_delta":      round(pod, 4),
                "net_win_prob_delta":      round(net_wpd_pkg, 4),
                "net_playoff_odds_delta":  round(net_pod_pkg, 4),
                "acceptance_pct":          acpt,
                "fit_note":                _fit_note(pkg, _p_need, pos,
                                                     _pmeta.get("pos_counts")),
                "direction":               "acquire",
                "suggested_send":          pkg,
            })
            if len(results) >= 15:
                break
        if len(results) >= 15:
            break

    # Surface the most compelling deals first: ones that both move the needle
    # (net playoff-odds impact) AND the partner would plausibly accept. Value
    # matching already happened per package; this orders across targets so a
    # fair-but-flat deal doesn't sit above a high-impact, high-acceptance one.
    results.sort(key=_suggestion_rank, reverse=True)

    return {"suggestions": results, "current_playoff_pct": round(current_playoff_pct, 1)}
