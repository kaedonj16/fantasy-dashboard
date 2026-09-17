"""Pure league-aware ranking for unexpected waiver performances.

Kept separate from the Flask blueprint so detector/curation unit tests remain
lightweight and can run in CI without importing the application's data stack.
"""
from __future__ import annotations


def curate_big_game_discoveries(rows, *, superflex=False, qb_need=False, limit=5):
    """Rank and trim discoveries using the league context known by the API."""
    ranked = []
    for original in rows or []:
        d = dict(original)
        pos = str(d.get("position") or "").upper()
        surprise = float(d.get("performance_surprise") or 0)
        sustain = float(d.get("role_sustainability") or 0)
        absolute = float(d.get("absolute_score") or 0)
        value = max(0.0, float(d.get("value") or 0))
        need = 1.0 if (pos == "QB" and qb_need) else float(d.get("viewer_need") or 0)
        caution_count = len(d.get("cautions") or [])
        uncertainty = (0.12 if not d.get("role_confirmed") else 0.0) + 0.05 * caution_count
        scarcity = (0.14 if pos == "QB" and superflex
                    else 0.04 if pos in ("RB", "WR", "TE") else 0)
        ros = min(1.0, value / 5000.0)
        score = (0.34 * surprise + 0.29 * sustain + 0.16 * ros + scarcity
                 + 0.12 * need - uncertainty)
        d["curated_score"] = round(score, 3)

        # In 1QB, a replacement passer needs an actual differentiator. Starting
        # snaps alone cannot supply one because the shared detector ignores them.
        if pos == "QB" and not superflex:
            differentiated = (absolute >= 0.9 or sustain >= 0.65
                              or qb_need or ros >= 0.65)
            if not differentiated:
                continue
        ranked.append(d)

    ranked.sort(
        key=lambda row: (row["curated_score"], row.get("role_sustainability", 0)),
        reverse=True,
    )
    if not superflex:
        result, ordinary_qbs = [], 0
        for d in ranked:
            if (d.get("position") == "QB" and not qb_need
                    and float(d.get("absolute_score") or 0) < 0.9):
                ordinary_qbs += 1
                if ordinary_qbs > 1:
                    continue
            result.append(d)
            if len(result) >= limit:
                break
        return result
    return ranked[:limit]
