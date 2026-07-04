"""Pure schedule-difficulty presentation helpers.

Extracted from app.py's schedule assistant: NFL team-code normalization across
data sources, and the color/ease mappings for matchup difficulty cells.
"""

# Alternate team codes used by various stat/schedule feeds -> canonical code.
SCHED_TEAM_ALIAS = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR", "OAK": "LV",
                    "SD": "LAC", "STL": "LAR", "ARZ": "ARI", "BLT": "BAL",
                    "CLV": "CLE", "HST": "HOU"}


def norm_sched_team(t) -> str:
    """Canonical NFL team code ('' stays '')."""
    t = (t or "").upper().strip()
    return SCHED_TEAM_ALIAS.get(t, t)


def sched_rank_color(rank, total):
    """(text_color, background) for a matchup by opponent's fpts-allowed rank
    (1 = most points allowed = easiest)."""
    if not rank or not total:
        return "#6b7280", "transparent"
    pct = rank / total
    if pct <= 0.25:
        return "#22c55e", "#22c55e18"   # elite (most pts allowed)
    if pct <= 0.50:
        return "#84cc16", "#84cc1618"
    if pct <= 0.75:
        return "#f59e0b", "#f59e0b18"
    return "#ef4444", "#ef444418"        # brutal (fewest pts allowed)


def matchup_cell_ease(rank, total, info) -> float:
    """Per-cell ease (0-100). Prefer the z-derived ease from the precomputed
    ratings table; fall back to rank percentile."""
    if info and info.get("ease") is not None:
        return float(info["ease"])
    if rank and total and total > 1:
        return round((total - rank) / (total - 1) * 100, 1)
    return 0.0
