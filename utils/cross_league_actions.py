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
    "roster": 60,
    "waiver": 50,
    "calendar": 40,
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


# Waiver pickups are only worth a cross-league nudge when the best available
# free agent clears a bar that reflects the league type. The value scale is the
# shared model value (WEIGHTS.min_value == 25 is the generic waiver floor), so
# these multipliers keep the bar tied to that calibration: a redraft manager
# churns the wire for rest-of-season production (lower bar), while a dynasty
# manager should only be pinged for a genuine long-term asset (higher bar).
WAIVER_MIN_VALUE_MULT = {"redraft": 1.4, "dynasty": 2.0}


def waiver_value_threshold(base_min_value: float, *, is_redraft: bool) -> float:
    """Format-aware minimum value for a waiver pickup to be worth surfacing."""
    try:
        base = float(base_min_value)
    except (TypeError, ValueError):
        base = 25.0
    return base * WAIVER_MIN_VALUE_MULT["redraft" if is_redraft else "dynasty"]


def waiver_pickup_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    player_name: str,
    position: str = "",
    is_redraft: bool = False,
    pos_rank_label: str = "",
    value: float = 0.0,
) -> dict:
    """The single best available free agent for this league's format.

    ``player_name``/``value`` are the top unrostered candidate scored off the
    league's own value column (redraft vs dynasty), so the pickup that surfaces
    is the one worth the most *in that league type*, not a generic best FA.
    """
    fmt = "redraft" if is_redraft else "dynasty"
    pos = str(position or "").upper()
    name = str(player_name or "").strip() or "a free agent"
    title = f"Add {name}" + (f" ({pos})" if pos else "")
    bits = []
    if pos_rank_label:
        bits.append(str(pos_rank_label))
    bits.append(f"Top available by {fmt} value")
    return make_action(
        kind="waiver",
        platform=platform,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=title,
        detail=" · ".join(bits),
        severity=0.5,
    )


# Higher = surfaced first when a roster has more than one wasted-capacity issue.
_ROSTER_ISSUE_RANK = {"ir_activate": 0.9, "ir_stash": 0.8, "taxi_stash": 0.6}
_ROSTER_ISSUE_TITLE = {
    "ir_activate": "Activate or drop a recovered IR player",
    "ir_stash": "Move an injured player to your IR slot",
    "taxi_stash": "Stash a rookie on your open taxi slot",
}


def roster_slot_action(
    issues: list[dict],
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
) -> Optional[dict]:
    """Turn ``roster_compliance_issues`` rows into one digest action per league.

    Surfaces the most actionable wasted-capacity issue (a recovered player stuck
    on IR, an injured player who could free a spot, or an open taxi slot).
    """
    rows = [i for i in (issues or []) if isinstance(i, dict) and i.get("kind")]
    if not rows:
        return None
    top = max(rows, key=lambda i: _ROSTER_ISSUE_RANK.get(str(i.get("kind")), 0.0))
    kind = str(top.get("kind"))
    plat = (platform or "sleeper").strip().lower()
    return make_action(
        kind="roster",
        platform=plat,
        season=season,
        league_id=league_id,
        league_name=league_name,
        title=_ROSTER_ISSUE_TITLE.get(kind, "Free up a roster spot"),
        detail=str(top.get("detail") or ""),
        href=f"/{plat}/{int(season)}/{league_id}/teams",
        severity=_ROSTER_ISSUE_RANK.get(kind, 0.5),
    )


def calendar_action(
    *,
    platform: str,
    season: int,
    league_id: str,
    league_name: str,
    week: int,
    trade_deadline: int = 0,
    playoff_week_start: int = 0,
) -> Optional[dict]:
    """A time-sensitive league-calendar nudge (trade deadline, then playoffs).

    Only fires in-season and within two weeks of an event. The trade deadline
    takes precedence over the playoff countdown when both are near.
    """
    try:
        wk = int(week or 0)
    except (TypeError, ValueError):
        wk = 0
    if wk <= 0:
        return None
    plat = (platform or "sleeper").strip().lower()

    def _weeks(n: int) -> str:
        return f"{n} week" + ("s" if n != 1 else "")

    # Trade deadline: only trust a sane in-season week number (Sleeper stores 0
    # or a large sentinel when there is no deadline).
    try:
        dl = int(trade_deadline or 0)
    except (TypeError, ValueError):
        dl = 0
    if 1 <= dl <= 18:
        gap = dl - wk
        if gap == 0:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title="Trade deadline is this week",
                detail=f"Week {dl} is the last week to make a deal.",
                href=f"/{plat}/{int(season)}/{league_id}/trade",
                severity=0.95,
            )
        if 0 < gap <= 2:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title=f"Trade deadline in {_weeks(gap)}",
                detail=f"Deadline is Week {dl}. Line up any deals before the wire closes.",
                href=f"/{plat}/{int(season)}/{league_id}/trade",
                severity=0.75 if gap == 1 else 0.6,
            )

    # Playoffs approaching: seed-and-lock-your-roster nudge.
    try:
        pw = int(playoff_week_start or 0)
    except (TypeError, ValueError):
        pw = 0
    if 1 <= pw <= 18:
        gap = pw - wk
        if 0 < gap <= 2:
            return make_action(
                kind="calendar", platform=plat, season=season, league_id=league_id,
                league_name=league_name,
                title=f"Playoffs start in {_weeks(gap)}",
                detail=f"Week {pw} begins the playoffs. Lock your roster and seeding now.",
                href=f"/{plat}/{int(season)}/{league_id}/matchups",
                severity=0.55 if gap == 1 else 0.45,
            )
    return None
