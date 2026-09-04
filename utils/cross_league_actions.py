"""Cross-league action digest helpers (roadmap R04).

Ranks per-league to-dos for the My Leagues hub. Pure functions so ranking is
unit-testable without Flask / provider I/O.
"""
from __future__ import annotations

from typing import Any, Optional

# Higher = more urgent. Keep the scale small and documented.
_PRIORITY = {
    "lineup": 100,
    "injury": 70,
    "waiver": 50,
    "trade": 30,
}


def action_priority(kind: str, *, severity: float = 0.0) -> float:
    base = float(_PRIORITY.get(str(kind or "").lower(), 10))
    try:
        sev = max(0.0, min(1.0, float(severity)))
    except (TypeError, ValueError):
        sev = 0.0
    return base + sev * 9.0


def make_action(
    *,
    kind: str,
    platform: str,
    season: int,
    league_id: str,
    league_name: str = "",
    title: str,
    detail: str = "",
    href: str = "",
    severity: float = 0.0,
) -> dict[str, Any]:
    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "").strip()
    path = href or (
        f"/{plat}/{int(season)}/{lid}/waivers?tab=startsit"
        if kind == "lineup"
        else f"/{plat}/{int(season)}/{lid}/waivers"
        if kind == "waiver"
        else f"/{plat}/{int(season)}/{lid}/dashboard"
    )
    return {
        "kind": kind,
        "platform": plat,
        "season": int(season),
        "league_id": lid,
        "league_name": league_name or lid,
        "title": title,
        "detail": detail,
        "href": path,
        "priority": action_priority(kind, severity=severity),
    }


def rank_cross_league_actions(actions: list[dict], *, limit: int = 8) -> list[dict]:
    """Sort by priority desc, then league name; cap to ``limit``."""
    rows = [a for a in (actions or []) if isinstance(a, dict) and a.get("title")]
    rows.sort(key=lambda a: (-float(a.get("priority") or 0), str(a.get("league_name") or "")))
    return rows[: max(0, int(limit or 0))]


def lineup_actions_from_issues(
    issues: list[dict],
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str = "",
) -> list[dict]:
    """Turn ``find_lineup_issues`` rows into digest actions."""
    if not issues:
        return []
    kinds = {str(i.get("kind") or "") for i in issues}
    if "empty" in kinds:
        title = "Empty starting slot"
        sev = 1.0
    elif "injury" in kinds:
        title = "Injured starter needs a swap"
        sev = 0.85
    elif "bye" in kinds:
        title = "Starter on bye"
        sev = 0.7
    else:
        title = "Lineup needs attention"
        sev = 0.5
    detail = "; ".join(
        str(i.get("detail") or i.get("name") or "").strip()
        for i in issues[:3]
        if (i.get("detail") or i.get("name"))
    )
    return [make_action(
        kind="lineup",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=detail,
        severity=sev,
    )]


def injury_stash_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    player_name: str,
    verdict: str,
    weeks_label: str = "",
    already_on_ir: bool = False,
) -> Optional[dict]:
    v = str(verdict or "").strip()
    if v not in ("IR", "Drop candidate", "Stash"):
        return None
    # Already occupying an IR slot — stash/move-to-IR tips are not actionable.
    # Drop candidate can still matter (free the IR slot).
    if already_on_ir and v in ("IR", "Stash"):
        return None
    return make_action(
        kind="injury",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=f"{v}: {player_name}",
        detail=(f"Approx return {weeks_label}" if weeks_label else "Approximate injury guidance"),
        href=f"/{(platform or 'sleeper').strip().lower()}/{int(season)}/{league_id}/waivers?tab=startsit",
        severity=0.8 if v == "Drop candidate" else 0.55,
    )
