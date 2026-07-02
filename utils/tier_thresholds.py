"""Pure tier-threshold computation.

Extracted from app.py so the drop-based tiering logic can be unit-tested without
importing the full application (pandas / DB) stack.

Given a table of player value dicts, ``compute_tier_thresholds`` returns the
value boundaries between tiers (T1 elite ... Tn). Boundaries are placed at
natural value drops scored by *local* significance, subject to two hard rules
(max span per tier, min players per tier). See the function docstring for the
full description of the algorithm.
"""
from __future__ import annotations

# Fallback thresholds used when the value distribution is too small or too
# degenerate to derive meaningful tiers from.
FALLBACK_THRESHOLDS = [850.0, 700.0, 550.0, 420.0, 300.0, 200.0, 120.0, 60.0]

# Tier display caps out here (T1 elite ... T12).
MAX_DISPLAY_TIERS = 12


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
